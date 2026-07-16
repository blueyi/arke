# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Tensor-Core (wmma fp16->fp32) flash_attention variant.

The CudaCBackend routes fp16 flash_attention with head dim in {64,128} to the
TC fused kernel (emit_cuda_c_flash_attention_tc); everything else uses the
correctness-first fp32 warp-per-row kernel. Provenance: T1 /
docs/phase5/c2-tensorcore-attention-*.md, scratch/tc_attn (v7/v8).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from arke.backend.cuda_c_attention import (
    emit_cuda_c_flash_attention_tc,
)
from arke.backend.cuda_c_backend import CudaCBackend, cuda_c_toolchain_available
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not cuda_c_toolchain_available(),
    reason="CUDA-C toolchain not available",
)


@pytest.fixture
def backend():
    return CudaCBackend(chip="sm_86")


def _graph(B, H, S, D, dtype="float16", causal=False):
    g = IRGraph(name=f"fa_{B}x{H}x{S}x{D}")
    g.add_input("Q", dtype=dtype, shape=[B, H, S, D])
    g.add_input("K", dtype=dtype, shape=[B, H, S, D])
    g.add_input("V", dtype=dtype, shape=[B, H, S, D])
    g.add_node(IRNode(id="n0", op="flash_attention",
                      inputs={"Q": "Q", "K": "K", "V": "V"}, outputs=["out"],
                      attrs={"causal": causal}))
    g.set_outputs(["out"])
    return g


class TestTCDispatch:
    """fp16 + D in {64,128} routes to the TC kernel; fp32 stays on fallback."""

    @pytest.mark.parametrize("D", [64, 128])
    def test_fp16_routes_to_tc(self, backend, D):
        emitted = backend.lower(_graph(1, 4, 128, D)).metadata["emitted"]
        assert emitted.kernel_name.startswith("arke_tc_flash_attn")

    def test_fp32_uses_fallback(self, backend):
        emitted = backend.lower(_graph(1, 4, 128, 64, dtype="float32")).metadata["emitted"]
        assert emitted.kernel_name.startswith("arke_flash_attn")
        assert not emitted.kernel_name.startswith("arke_tc")

    def test_fp16_odd_headdim_uses_fallback(self, backend):
        # D=48 is not a TC-supported head dim -> correctness-first fp32 path.
        emitted = backend.lower(_graph(1, 1, 64, 48)).metadata["emitted"]
        assert not emitted.kernel_name.startswith("arke_tc")

    def test_causal_name_tag(self):
        g = _graph(1, 4, 128, 64, causal=True)
        k = emit_cuda_c_flash_attention_tc(g)
        assert k.kernel_name.endswith("_causal")
        assert "#define CAUSAL   1" in k.source

    def test_emitter_rejects_fp32(self):
        with pytest.raises(ValueError, match="float16"):
            emit_cuda_c_flash_attention_tc(_graph(1, 4, 128, 64, dtype="float32"))

    def test_emitter_rejects_bad_headdim(self):
        with pytest.raises(ValueError, match="D in"):
            emit_cuda_c_flash_attention_tc(_graph(1, 1, 64, 32))


class TestTCCorrectness:
    """Compiled + executed vs torch SDPA fp16 through the real backend path."""

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("B,H,S", [(1, 4, 128), (1, 8, 512)])
    def test_vs_sdpa(self, backend, B, H, S, D, causal):
        g = _graph(B, H, S, D, causal=causal)
        ker = backend.compile(backend.lower(g))
        assert ker.success, ker.error

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((B, H, S, D)).astype(np.float16)
        K = rng.standard_normal((B, H, S, D)).astype(np.float16)
        V = rng.standard_normal((B, H, S, D)).astype(np.float16)

        out = backend.run(ker, {"Q": Q, "K": K, "V": V})["out"]
        ref = F.scaled_dot_product_attention(
            torch.tensor(Q).cuda(), torch.tensor(K).cuda(), torch.tensor(V).cuda(),
            is_causal=causal,
        ).cpu().numpy()

        max_err = np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32)))
        assert max_err < 1e-2, f"max_err={max_err}"
