# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CUDA-C flash_attention (OT4, Phase 4 C-line)."""

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


def _make_graph(B, H, S, D):
    g = IRGraph(name=f"flash_{B}x{H}x{S}x{D}")
    g.add_input("Q", dtype="float32", shape=[B, H, S, D])
    g.add_input("K", dtype="float32", shape=[B, H, S, D])
    g.add_input("V", dtype="float32", shape=[B, H, S, D])
    g.add_node(IRNode(id="n0", op="flash_attention",
                      inputs={"Q": "Q", "K": "K", "V": "V"}, outputs=["out"]))
    g.set_outputs(["out"])
    return g


class TestCudaCFlashAttention:
    @pytest.mark.parametrize("B,H,S,D", [
        (1, 4, 128, 64),
        (2, 2, 64, 32),
        (1, 8, 256, 64),
        (1, 1, 100, 48),   # non-power-of-2 S, D
    ])
    def test_correctness_vs_sdpa(self, backend, B, H, S, D):
        graph = _make_graph(B, H, S, D)
        art = backend.lower(graph)
        ker = backend.compile(art)
        assert ker.success, ker.error

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((B, H, S, D)).astype(np.float32)
        K = rng.standard_normal((B, H, S, D)).astype(np.float32)
        V = rng.standard_normal((B, H, S, D)).astype(np.float32)

        result = backend.run(ker, {"Q": Q, "K": K, "V": V})
        out = result["out"]

        ref = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(Q), torch.tensor(K), torch.tensor(V)
        ).numpy()

        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-6)
        assert rel < 1e-3, f"rel_err={rel}"

    def test_registered(self, backend):
        assert backend.supports_op("flash_attention")
