# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the final 5 CUDA-C ops → full 46/46 catalog coverage."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arke.backend.cuda_c_backend import CudaCBackend, cuda_c_toolchain_available
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not cuda_c_toolchain_available(),
    reason="CUDA-C toolchain not available",
)


@pytest.fixture
def backend():
    return CudaCBackend(chip="sm_86")


def _run(backend, op, inputs_spec, node_inputs, np_inputs):
    g = IRGraph(name=op)
    for nm, (dt, shp) in inputs_spec.items():
        g.add_input(nm, dtype=dt, shape=shp)
    g.add_node(IRNode(id="n0", op=op, inputs=node_inputs, outputs=["out"]))
    g.set_outputs(["out"])
    art = backend.lower(g)
    ker = backend.compile(art)
    assert ker.success, ker.error
    return backend.run(ker, np_inputs)["out"]


class TestGroupedMatmul:
    def test_grouped_matmul(self, backend):
        B, M, K, N, E = 4, 16, 32, 24, 3
        rng = np.random.default_rng(0)
        X = rng.standard_normal((B, M, K)).astype(np.float32)
        W = rng.standard_normal((E, K, N)).astype(np.float32)
        idx = rng.integers(0, E, size=B).astype(np.int32)
        out = _run(backend, "grouped_matmul",
                   {"X": ("float32", [B, M, K]), "W": ("float32", [E, K, N]),
                    "indices": ("int32", [B])},
                   {"x": "X", "w": "W", "indices": "indices"},
                   {"X": X, "W": W, "indices": idx})
        ref = np.stack([X[b] @ W[idx[b]] for b in range(B)])
        np.testing.assert_allclose(out, ref, atol=1e-3, rtol=1e-3)


class TestQuantizePerToken:
    def test_quantize(self, backend):
        M, N = 32, 128
        rng = np.random.default_rng(1)
        X = rng.standard_normal((M, N)).astype(np.float32)
        out = _run(backend, "quantize_per_token",
                   {"X": ("float32", [M, N])}, {"x": "X"}, {"X": X})
        # Reference per-row symmetric int8
        scale = np.max(np.abs(X), axis=1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        ref = np.clip(np.rint(X / scale), -127, 127).astype(np.int8)
        # Allow off-by-one on rounding ties
        assert np.mean(np.abs(out.astype(np.int32) - ref.astype(np.int32)) <= 1) > 0.99


class TestFusedLinearCrossEntropy:
    def test_flce(self, backend):
        B, D, V = 8, 64, 100
        rng = np.random.default_rng(2)
        X = rng.standard_normal((B, D)).astype(np.float32)
        W = rng.standard_normal((V, D)).astype(np.float32)
        labels = rng.integers(0, V, size=B).astype(np.int32)
        out = _run(backend, "fused_linear_cross_entropy",
                   {"X": ("float32", [B, D]), "W": ("float32", [V, D]),
                    "labels": ("int32", [B])},
                   {"x": "X", "w": "W", "labels": "labels"},
                   {"X": X, "W": W, "labels": labels})
        logits = X @ W.T
        lse = np.log(np.sum(np.exp(logits - logits.max(1, keepdims=True)), axis=1)) + logits.max(1)
        ref = lse - logits[np.arange(B), labels]
        np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)


class TestPagedAttention:
    def test_paged_attention(self, backend):
        B, H, D = 2, 4, 64
        block_size, num_blocks, max_blocks = 16, 8, 4
        rng = np.random.default_rng(3)
        Q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
        Kc = rng.standard_normal((num_blocks, block_size, H, D)).astype(np.float32)
        Vc = rng.standard_normal((num_blocks, block_size, H, D)).astype(np.float32)
        # block_table: each batch uses 2 blocks then -1 sentinel
        bt = np.full((B, max_blocks), -1, dtype=np.int32)
        bt[0, :2] = [0, 1]
        bt[1, :2] = [2, 3]
        out = _run(backend, "paged_attention",
                   {"Q": ("float32", [B, H, 1, D]),
                    "K_cache": ("float32", [num_blocks, block_size, H, D]),
                    "V_cache": ("float32", [num_blocks, block_size, H, D]),
                    "block_table": ("int32", [B, max_blocks])},
                   {"q": "Q", "k_cache": "K_cache", "v_cache": "V_cache", "block_table": "block_table"},
                   {"Q": Q, "K_cache": Kc, "V_cache": Vc, "block_table": bt})
        # Reference: gather 2 blocks (32 keys) per batch, standard attention
        scale = 1.0 / np.sqrt(D)
        ref = np.zeros((B, H, 1, D), dtype=np.float32)
        for b in range(B):
            blocks = [x for x in bt[b] if x >= 0]
            K = np.concatenate([Kc[pb, :, :, :] for pb in blocks], axis=0)  # [nk, H, D]
            V = np.concatenate([Vc[pb, :, :, :] for pb in blocks], axis=0)
            for h in range(H):
                q = Q[b, h, 0]                       # [D]
                s = (K[:, h, :] @ q) * scale         # [nk]
                p = np.exp(s - s.max()); p /= p.sum()
                ref[b, h, 0] = p @ V[:, h, :]
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-6)
        assert rel < 1e-3, f"rel={rel}"


class TestMLA:
    def test_mla(self, backend):
        B, H, S, D, Dc = 1, 2, 16, 32, 24
        rng = np.random.default_rng(4)
        Q = rng.standard_normal((B, H, S, D)).astype(np.float32)
        KVc = rng.standard_normal((B, S, Dc)).astype(np.float32)
        W_uk = rng.standard_normal((Dc, H, D)).astype(np.float32) * 0.1
        W_uv = rng.standard_normal((Dc, H, D)).astype(np.float32) * 0.1
        out = _run(backend, "multi_latent_attention",
                   {"Q": ("float32", [B, H, S, D]), "KV_compressed": ("float32", [B, S, Dc]),
                    "W_uk": ("float32", [Dc, H, D]), "W_uv": ("float32", [Dc, H, D])},
                   {"q": "Q", "kv": "KV_compressed", "w_uk": "W_uk", "w_uv": "W_uv"},
                   {"Q": Q, "KV_compressed": KVc, "W_uk": W_uk, "W_uv": W_uv})
        # Reference: K[b,:,h,:] = KVc[b] @ W_uk[:,h,:]; V similarly; then attention
        scale = 1.0 / np.sqrt(D)
        ref = np.zeros((B, H, S, D), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                K = KVc[b] @ W_uk[:, h, :]   # [S, D]
                V = KVc[b] @ W_uv[:, h, :]
                for sq in range(S):
                    s = (K @ Q[b, h, sq]) * scale
                    p = np.exp(s - s.max()); p /= p.sum()
                    ref[b, h, sq] = p @ V
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-6)
        assert rel < 1e-2, f"rel={rel}"
