# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLVMBackend (Phase 5, P5-S1): matmul LLVM-IR end-to-end."""

import numpy as np
import pytest

from arke.backend.llvm_backend import LLVMBackend, llvm_toolchain_available
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not llvm_toolchain_available(),
    reason="LLVM toolchain (llc + ptxas + GPU driver) not available",
)


@pytest.fixture
def backend():
    return LLVMBackend(chip="sm_86")


def _make_matmul_graph(M: int, K: int, N: int) -> IRGraph:
    graph = IRGraph(name=f"matmul_{M}x{K}x{N}")
    graph.add_input("A", dtype="float32", shape=[M, K])
    graph.add_input("B", dtype="float32", shape=[K, N])
    graph.add_node(IRNode(
        id="n0", op="matmul",
        inputs={"A": "A", "B": "B"}, outputs=["out"],
    ))
    graph.set_outputs(["out"])
    return graph


class TestLLVMBackendProtocol:
    """Verify LLVMBackend conforms to the ArkeBackend protocol."""

    def test_supports_matmul(self, backend):
        assert backend.supports_op("matmul")

    def test_does_not_support_unknown_op(self, backend):
        assert not backend.supports_op("nonexistent_op")

    def test_name(self, backend):
        assert backend.name == "llvm"


class TestLLVMMatmulLower:
    """Test the lower() step produces valid LLVM IR."""

    def test_lower_matmul(self, backend):
        graph = _make_matmul_graph(64, 64, 64)
        artifact = backend.lower(graph)
        assert "nvptx64-nvidia-cuda" in artifact.source_code
        assert "addrspace(1)" in artifact.source_code
        assert "addrspace(3)" in artifact.source_code
        assert "@llvm.nvvm.barrier0" in artifact.source_code
        assert "!nvvm.annotations" in artifact.source_code
        assert artifact.backend_name == "llvm"
        assert artifact.op_name == "matmul"


class TestLLVMMatmulCompile:
    """Test the compile() step produces a cubin."""

    def test_compile_matmul(self, backend):
        graph = _make_matmul_graph(64, 64, 64)
        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, f"Compile failed: {kernel.error}"
        assert kernel.backend_name == "llvm"
        assert kernel.metadata["cubin"] is not None
        assert len(kernel.metadata["cubin"]) > 0


class TestLLVMMatmulCorrectness:
    """End-to-end correctness tests for matmul via LLVM backend."""

    def test_run_matmul_correctness(self, backend):
        """64x64 matmul, compare output vs numpy reference."""
        M, K, N = 64, 64, 64
        graph = _make_matmul_graph(M, K, N)
        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, f"Compile failed: {kernel.error}"

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = backend.run(kernel, {"A": A, "B": B})
        out = result["out"]
        ref = A @ B

        np.testing.assert_allclose(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("M,K,N", [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ])
    def test_run_matmul_shapes(self, backend, M, K, N):
        """Parametrized shape correctness."""
        graph = _make_matmul_graph(M, K, N)
        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, f"Compile failed: {kernel.error}"

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = backend.run(kernel, {"A": A, "B": B})
        out = result["out"]
        ref = A @ B

        # Tolerance scales with K (accumulation depth)
        atol = max(1e-3, K * 2e-6)
        np.testing.assert_allclose(out, ref, atol=atol, rtol=1e-3)
