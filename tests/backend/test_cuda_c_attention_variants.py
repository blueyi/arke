# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CUDA-C attention variants: GQA + cross-attention (exotic ops)."""

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


class TestGQA:
    @pytest.mark.parametrize("B,Hq,Hkv,S,D", [(1, 8, 2, 128, 64), (2, 4, 2, 64, 32)])
    def test_gqa_vs_torch(self, backend, B, Hq, Hkv, S, D):
        g = IRGraph(name="gqa")
        g.add_input("Q", dtype="float32", shape=[B, Hq, S, D])
        g.add_input("K", dtype="float32", shape=[B, Hkv, S, D])
        g.add_input("V", dtype="float32", shape=[B, Hkv, S, D])
        g.add_node(IRNode(id="n0", op="grouped_query_attention",
                          inputs={"Q": "Q", "K": "K", "V": "V"}, outputs=["out"]))
        g.set_outputs(["out"])
        art = backend.lower(g)
        ker = backend.compile(art)
        assert ker.success, ker.error

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((B, Hq, S, D)).astype(np.float32)
        K = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
        V = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
        out = backend.run(ker, {"Q": Q, "K": K, "V": V})["out"]

        # torch reference: repeat kv heads to match query heads
        group = Hq // Hkv
        Kt = torch.tensor(K).repeat_interleave(group, dim=1)
        Vt = torch.tensor(V).repeat_interleave(group, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(Q), Kt, Vt).numpy()
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-6)
        assert rel < 1e-3, f"rel={rel}"


class TestCrossAttention:
    @pytest.mark.parametrize("B,H,Sq,Skv,D", [(1, 4, 64, 128, 64), (2, 2, 32, 48, 32)])
    def test_cross_attn_vs_torch(self, backend, B, H, Sq, Skv, D):
        g = IRGraph(name="cross")
        g.add_input("Q", dtype="float32", shape=[B, H, Sq, D])
        g.add_input("K", dtype="float32", shape=[B, H, Skv, D])
        g.add_input("V", dtype="float32", shape=[B, H, Skv, D])
        g.add_node(IRNode(id="n0", op="cross_attention",
                          inputs={"Q": "Q", "K": "K", "V": "V"}, outputs=["out"]))
        g.set_outputs(["out"])
        art = backend.lower(g)
        ker = backend.compile(art)
        assert ker.success, ker.error

        rng = np.random.default_rng(1)
        Q = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
        K = rng.standard_normal((B, H, Skv, D)).astype(np.float32)
        V = rng.standard_normal((B, H, Skv, D)).astype(np.float32)
        out = backend.run(ker, {"Q": Q, "K": K, "V": V})["out"]

        ref = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(Q), torch.tensor(K), torch.tensor(V)).numpy()
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-6)
        assert rel < 1e-3, f"rel={rel}"
