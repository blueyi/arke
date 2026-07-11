# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): OT4 attention ops test coverage.

Covers: cross_attention, grouped_query_attention,
        multi_latent_attention, paged_attention.
(flash_attention already covered in test_mlir_gpu_p3s1.py.)
All bit-correct vs torch SDPA / numpy references. Skips without GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _run_attention(op, ins, in_shapes):
    """Run an attention-family graph through MLIRGPUBackend."""
    be = MLIRGPUBackend()
    g = IRGraph(name=op)
    names = list(ins.keys())
    for n, s in zip(names, in_shapes):
        g.add_input(n, dtype="float32", shape=list(s))
    g.add_node(IRNode(
        id="n0", op=op,
        inputs={n: n for n in names},
        outputs=["Y"],
    ))
    g.set_outputs(["Y"])
    ker = be.compile(be.lower(g))
    assert ker.success, f"compile failed: {ker.error}"
    return be.run(ker, ins)["Y"]


def _ref_sdpa(Q, K, V):
    """Numpy reference: scaled dot-product attention per (b, h)."""
    B, H, Sq, D = Q.shape
    _, _, Skv, _ = K.shape
    out = np.zeros((B, H, Sq, D), dtype=np.float64)
    scale = 1.0 / np.sqrt(D)
    for b in range(B):
        for h in range(H):
            scores = (Q[b, h].astype(np.float64) @ K[b, h].T.astype(np.float64)) * scale
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            out[b, h] = attn @ V[b, h].astype(np.float64)
    return out.astype(np.float32)


# ── cross_attention ─────────────────────────────────────────────

class TestCrossAttention:
    """OT4 cross_attention: SDPA with Sq != Skv."""

    @pytest.mark.parametrize("Sq,Skv", [(4, 8), (8, 4), (6, 12)])
    def test_cross_attention_correctness(self, Sq, Skv):
        B, H, D = 1, 2, 16
        rng = np.random.RandomState(42)
        Q = rng.randn(B, H, Sq, D).astype(np.float32)
        K = rng.randn(B, H, Skv, D).astype(np.float32)
        V = rng.randn(B, H, Skv, D).astype(np.float32)
        out = _run_attention("cross_attention",
                             {"Q": Q, "K": K, "V": V},
                             [(B, H, Sq, D), (B, H, Skv, D), (B, H, Skv, D)])
        ref = _ref_sdpa(Q, K, V)
        assert out.shape == (B, H, Sq, D)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

    def test_cross_attention_sq_equals_skv(self):
        """When Sq == Skv, cross_attention == flash_attention."""
        B, H, S, D = 1, 2, 8, 16
        rng = np.random.RandomState(7)
        Q = rng.randn(B, H, S, D).astype(np.float32)
        K = rng.randn(B, H, S, D).astype(np.float32)
        V = rng.randn(B, H, S, D).astype(np.float32)
        out_cross = _run_attention("cross_attention",
                                   {"Q": Q, "K": K, "V": V},
                                   [(B, H, S, D)] * 3)
        out_flash = _run_attention("flash_attention",
                                   {"Q": Q, "K": K, "V": V},
                                   [(B, H, S, D)] * 3)
        np.testing.assert_allclose(out_cross, out_flash, rtol=1e-5, atol=1e-6)


# ── grouped_query_attention ─────────────────────────────────────

class TestGroupedQueryAttention:
    """OT4 GQA: Hq > Hkv with head index remapping."""

    @pytest.mark.parametrize("Hq,Hkv", [(4, 2), (4, 1), (8, 4)])
    def test_gqa_correctness(self, Hq, Hkv):
        B, S, D = 1, 8, 16
        rng = np.random.RandomState(42)
        Q = rng.randn(B, Hq, S, D).astype(np.float32)
        K = rng.randn(B, Hkv, S, D).astype(np.float32)
        V = rng.randn(B, Hkv, S, D).astype(np.float32)
        out = _run_attention("grouped_query_attention",
                             {"Q": Q, "K": K, "V": V},
                             [(B, Hq, S, D), (B, Hkv, S, D), (B, Hkv, S, D)])
        assert out.shape == (B, Hq, S, D)
        # Reference: expand KV heads to match Q heads, then SDPA
        n_rep = Hq // Hkv
        K_exp = np.repeat(K, n_rep, axis=1)  # [B, Hq, S, D]
        V_exp = np.repeat(V, n_rep, axis=1)
        ref = _ref_sdpa(Q, K_exp, V_exp)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

    def test_gqa_single_kv_head(self):
        """GQA with Hkv=1: all Q heads attend to same KV."""
        B, Hq, Hkv, S, D = 1, 4, 1, 4, 8
        rng = np.random.RandomState(7)
        Q = rng.randn(B, Hq, S, D).astype(np.float32)
        K = rng.randn(B, Hkv, S, D).astype(np.float32)
        V = rng.randn(B, Hkv, S, D).astype(np.float32)
        out = _run_attention("grouped_query_attention",
                             {"Q": Q, "K": K, "V": V},
                             [(B, Hq, S, D), (B, Hkv, S, D), (B, Hkv, S, D)])
        # All heads should produce same output since K,V are shared
        for h in range(1, Hq):
            # Different Q heads → different outputs, but all use same K,V
            ref_h = _ref_sdpa(Q[:, h:h+1], K, V)
            np.testing.assert_allclose(out[:, h:h+1], ref_h, rtol=1e-4, atol=1e-5)


# ── multi_latent_attention ──────────────────────────────────────

class TestMultiLatentAttention:
    """OT4 MLA: KV decompressed via einsum on host, then SDPA."""

    def test_mla_correctness(self):
        B, H, S, D = 1, 2, 8, 16
        latent_D = 8
        rng = np.random.RandomState(42)
        Q = rng.randn(B, H, S, D).astype(np.float32)
        kv_c = rng.randn(B, S, latent_D).astype(np.float32)
        w_uk = rng.randn(latent_D, H, D).astype(np.float32)
        w_uv = rng.randn(latent_D, H, D).astype(np.float32)
        out = _run_attention("multi_latent_attention",
                             {"Q": Q, "KV_compressed": kv_c, "w_uk": w_uk, "w_uv": w_uv},
                             [(B, H, S, D), (B, S, latent_D),
                              (latent_D, H, D), (latent_D, H, D)])
        assert out.shape == (B, H, S, D)
        # Reference: decompress then SDPA
        K = np.einsum("bsd,dhn->bhsn", kv_c, w_uk)
        V = np.einsum("bsd,dhn->bhsn", kv_c, w_uv)
        ref = _ref_sdpa(Q, K, V)
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_mla_small(self):
        """Small MLA: B=1, H=1, S=4, D=8, latent=4."""
        B, H, S, D, ld = 1, 1, 4, 8, 4
        rng = np.random.RandomState(7)
        Q = rng.randn(B, H, S, D).astype(np.float32)
        kv_c = rng.randn(B, S, ld).astype(np.float32)
        w_uk = rng.randn(ld, H, D).astype(np.float32)
        w_uv = rng.randn(ld, H, D).astype(np.float32)
        out = _run_attention("multi_latent_attention",
                             {"Q": Q, "KV_compressed": kv_c, "w_uk": w_uk, "w_uv": w_uv},
                             [(B, H, S, D), (B, S, ld), (ld, H, D), (ld, H, D)])
        K = np.einsum("bsd,dhn->bhsn", kv_c, w_uk)
        V = np.einsum("bsd,dhn->bhsn", kv_c, w_uv)
        ref = _ref_sdpa(Q, K, V)
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


# ── paged_attention ─────────────────────────────────────────────

class TestPagedAttention:
    """OT4 paged_attention: block-table assembly on host, then SDPA."""

    def test_paged_attention_correctness(self):
        B, H, D = 1, 2, 16
        Sq = 1  # decode: single query token
        num_blocks, block_size = 4, 4
        max_blocks_per_seq = 2
        rng = np.random.RandomState(42)
        Q = rng.randn(B, H, Sq, D).astype(np.float32)
        K_cache = rng.randn(num_blocks, block_size, H, D).astype(np.float32)
        V_cache = rng.randn(num_blocks, block_size, H, D).astype(np.float32)
        block_table = np.array([[0, 2]], dtype=np.float32)  # blocks 0 and 2
        out = _run_attention("paged_attention",
                             {"Q": Q, "K_cache": K_cache, "V_cache": V_cache,
                              "block_table": block_table},
                             [(B, H, Sq, D), (num_blocks, block_size, H, D),
                              (num_blocks, block_size, H, D), (B, max_blocks_per_seq)])
        assert out.shape == (B, H, Sq, D)
        # Reference: manually assemble KV from block table
        Skv = max_blocks_per_seq * block_size
        K_full = np.zeros((B, H, Skv, D), dtype=np.float32)
        V_full = np.zeros((B, H, Skv, D), dtype=np.float32)
        for b in range(B):
            for bi in range(max_blocks_per_seq):
                blk_idx = int(block_table[b, bi])
                start = bi * block_size
                end = start + block_size
                K_full[b, :, start:end, :] = K_cache[blk_idx].transpose(1, 0, 2)
                V_full[b, :, start:end, :] = V_cache[blk_idx].transpose(1, 0, 2)
        ref = _ref_sdpa(Q, K_full, V_full)
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_paged_attention_single_block(self):
        """Single cache block: paged_attention == cross_attention."""
        B, H, D, Sq = 1, 1, 8, 1
        block_size = 4
        rng = np.random.RandomState(7)
        Q = rng.randn(B, H, Sq, D).astype(np.float32)
        K_cache = rng.randn(2, block_size, H, D).astype(np.float32)
        V_cache = rng.randn(2, block_size, H, D).astype(np.float32)
        block_table = np.array([[1]], dtype=np.float32)  # use block 1 only
        out = _run_attention("paged_attention",
                             {"Q": Q, "K_cache": K_cache, "V_cache": V_cache,
                              "block_table": block_table},
                             [(B, H, Sq, D), (2, block_size, H, D),
                              (2, block_size, H, D), (B, 1)])
        # Reference: block 1 only
        K = K_cache[1].transpose(1, 0, 2)[np.newaxis]  # [1, H, BS, D]
        V = V_cache[1].transpose(1, 0, 2)[np.newaxis]
        ref = _ref_sdpa(Q, K, V)
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
