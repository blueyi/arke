# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness test for LLVM IR Tensor Core (wmma) matmul.

Validates that the LLVM backend's wmma kernel produces numerically correct
results vs an fp16 reference (same precision path as CUDA-C TC kernel).
"""

import numpy as np
import pytest

from arke.backend.llvm_backend import LLVMBackend
from arke.ir.graph import IRGraph, IRNode, IRValue


def _build_matmul_graph(M: int, K: int, N: int) -> IRGraph:
    """Build a matmul IRGraph: C[M,N] = A[M,K] @ B[K,N]."""
    graph = IRGraph(name=f"matmul_{M}x{K}x{N}")
    graph.add_input("A", dtype="float32", shape=[M, K])
    graph.add_input("B", dtype="float32", shape=[K, N])
    node = IRNode(id="n0", op="matmul", inputs={"a": "A", "b": "B"}, outputs=["C"])
    graph.nodes.append(node)
    graph.values["C"] = IRValue(name="C", dtype="float32", shape=[M, N])
    graph.graph_outputs.append("C")
    return graph


def _fp16_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute matmul via fp16 (same precision as TC kernel)."""
    Ah = A.astype(np.float16)
    Bh = B.astype(np.float16)
    # Accumulate in fp32 for accuracy (same as wmma fp16->fp32)
    return (Ah.astype(np.float32) @ Bh.astype(np.float32)).astype(np.float32)


@pytest.mark.skipif(
    not LLVMBackend(chip="sm_86").llc,
    reason="llc not available"
)
class TestLLVMWmmaCorrectness:
    """Correctness tests for LLVM TC (wmma) matmul."""

    def setup_method(self):
        self.backend = LLVMBackend(chip="sm_86")

    @pytest.mark.parametrize("size", [1024, 2048])
    def test_square_matmul_correctness(self, size):
        """Test square matmul TC kernel vs fp16 reference."""
        M = K = N = size
        graph = _build_matmul_graph(M, K, N)

        # Compile
        art = self.backend.lower(graph)
        ck = self.backend.compile(art)
        assert ck.success, f"Compile failed: {ck.error}"

        # Random inputs (small magnitude to avoid fp16 overflow)
        rng = np.random.default_rng(42)
        A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
        B = rng.uniform(-1, 1, (K, N)).astype(np.float32)

        # Run on GPU
        result = self.backend.run(ck, {"A": A, "B": B})
        C_gpu = result["C"]

        # fp16 reference
        C_ref = _fp16_reference(A, B)

        # TC has fp16 precision: relative tolerance ~1e-2
        # (K=1024 accumulation in fp32 but inputs are fp16)
        atol = 1e-1  # absolute tolerance for near-zero elements
        rtol = 5e-2  # relative tolerance
        max_err = np.max(np.abs(C_gpu - C_ref))
        rel_err = np.max(np.abs(C_gpu - C_ref) / (np.abs(C_ref) + 1e-8))
        print(f"\n  {M}x{K}x{N}: max_abs_err={max_err:.4e}, max_rel_err={rel_err:.4e}")
        assert np.allclose(C_gpu, C_ref, atol=atol, rtol=rtol), (
            f"Correctness FAIL at {M}x{K}x{N}: "
            f"max_abs_err={max_err:.4e}, max_rel_err={rel_err:.4e}"
        )

    def test_rectangular_matmul(self):
        """Test non-square matmul that still qualifies for TC path."""
        M, K, N = 1024, 2048, 1024
        graph = _build_matmul_graph(M, K, N)

        art = self.backend.lower(graph)
        ck = self.backend.compile(art)
        assert ck.success, f"Compile failed: {ck.error}"

        rng = np.random.default_rng(123)
        A = rng.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
        B = rng.uniform(-0.5, 0.5, (K, N)).astype(np.float32)

        result = self.backend.run(ck, {"A": A, "B": B})
        C_gpu = result["C"]
        C_ref = _fp16_reference(A, B)

        max_err = np.max(np.abs(C_gpu - C_ref))
        rel_err = np.max(np.abs(C_gpu - C_ref) / (np.abs(C_ref) + 1e-8))
        print(f"\n  {M}x{K}x{N}: max_abs_err={max_err:.4e}, max_rel_err={rel_err:.4e}")
        assert np.allclose(C_gpu, C_ref, atol=0.1, rtol=0.05)
