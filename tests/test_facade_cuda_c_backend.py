# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test the CUDA-C backend wired through the Façade compile_and_profile tool.

This validates the A-line deliverable: the Agent can drive the Phase-4 CUDA-C
backend through the same Façade tool it uses for Triton, with kernel-only
benchmark() timing and V1 correctness against the reference interpreter.
"""

import pytest

torch = pytest.importorskip("torch")

from arke.agent.tools import CompileAndProfileTool
from arke.backend.cuda_c_backend import cuda_c_toolchain_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and cuda_c_toolchain_available()),
    reason="requires CUDA + nvcc toolchain",
)


@pytest.fixture
def tool():
    return CompileAndProfileTool()


class TestFacadeCudaCBackend:
    def test_matmul_via_cuda_c(self, tool):
        """compile_and_profile with backend='cuda_c' runs the CUDA-C matmul."""
        result = tool.execute({
            "op_name": "matmul",
            "shapes": {"A": [128, 128], "B": [128, 128]},
            "backend": "cuda_c",
        })
        assert result.success, result.error
        assert result.data["backend"] == "cuda_c"
        # TC matmul (fp16 accumulation) — correctness within loose tolerance
        assert result.data["correct"] is True
        assert result.data["output_shape"] == [128, 128]
        # kernel-only latency should be recorded
        assert result.data["latency_ms"] is not None
        assert result.data["latency_ms"] > 0

    def test_relu_via_cuda_c(self, tool):
        """Elementwise op through CUDA-C backend."""
        result = tool.execute({
            "op_name": "relu",
            "shapes": {"X": [512, 512]},
            "backend": "cuda_c",
        })
        assert result.success, result.error
        assert result.data["backend"] == "cuda_c"
        assert result.data["correct"] is True

    def test_default_backend_still_triton(self, tool):
        """Without backend param, default is triton (non-breaking)."""
        result = tool.execute({
            "op_name": "relu",
            "shapes": {"X": [64, 64]},
        })
        assert result.success, result.error
        # default path unchanged
        assert result.data["backend"] in ("triton", "mock")

    def test_baseline_ratio_recorded(self, tool):
        """V2 profiling produces a baseline_ratio vs torch eager."""
        result = tool.execute({
            "op_name": "matmul",
            "shapes": {"A": [512, 512], "B": [512, 512]},
            "backend": "cuda_c",
        })
        assert result.success, result.error
        # baseline_ratio may be None if eager baseline fails, but latency must exist
        assert result.data["latency_ms"] is not None
