# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CudaCBackend (Phase 4, P4-S1): matmul CUDA-C end-to-end."""

import numpy as np
import pytest

from arke.backend.cuda_c_backend import CudaCBackend, cuda_c_toolchain_available
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not cuda_c_toolchain_available(),
    reason="CUDA-C toolchain (nvcc + GPU driver) not available",
)


@pytest.fixture
def backend():
    return CudaCBackend(chip="sm_86")


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


class TestCudaCBackendProtocol:
    """Verify CudaCBackend conforms to the ArkeBackend protocol."""

    def test_supports_matmul(self, backend):
        assert backend.supports_op("matmul")

    def test_does_not_support_unknown_op(self, backend):
        assert not backend.supports_op("nonexistent_op")

    def test_name(self, backend):
        assert backend.name == "cuda-c"


class TestCudaCMatmulCorrectness:
    """End-to-end correctness tests for matmul via CUDA-C backend."""

    @pytest.mark.parametrize("M,K,N", [
        (16, 16, 16),
        (32, 64, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (33, 47, 61),   # non-power-of-2
        (1, 64, 1),     # degenerate
        (128, 1, 128),  # K=1
    ])
    def test_matmul_vs_numpy(self, backend, M, K, N):
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
        atol = max(1e-4, K * 2e-6)
        np.testing.assert_allclose(out, ref, atol=atol, rtol=1e-4)

    def test_matmul_vs_torch(self, backend):
        """Verify exact match against torch CUDA matmul."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("torch CUDA not available")

        M, K, N = 128, 64, 128
        graph = _make_matmul_graph(M, K, N)
        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success

        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = backend.run(kernel, {"A": A, "B": B})
        out = result["out"]

        A_t = torch.from_numpy(A).cuda()
        B_t = torch.from_numpy(B).cuda()
        ref_t = (A_t @ B_t).cpu().numpy()

        np.testing.assert_allclose(out, ref_t, atol=1e-5, rtol=1e-5)

    def test_matmul_large_512(self, backend):
        """512x512 — validates performance at moderate scale."""
        M, K, N = 512, 512, 512
        graph = _make_matmul_graph(M, K, N)
        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success

        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = backend.run(kernel, {"A": A, "B": B})
        out = result["out"]
        ref = A @ B

        np.testing.assert_allclose(out, ref, atol=5e-4, rtol=1e-3)


class TestCudaCCompilation:
    """Test compilation pipeline specifics."""

    def test_lower_produces_cuda_c_source(self, backend):
        graph = _make_matmul_graph(64, 64, 64)
        artifact = backend.lower(graph)
        assert "extern \"C\"" in artifact.source_code
        assert "__global__" in artifact.source_code
        assert "__shared__" in artifact.source_code
        assert artifact.backend_name == "cuda-c"
        assert artifact.op_name == "matmul"

    def test_compile_caches_cubin(self, backend):
        graph = _make_matmul_graph(64, 64, 64)
        artifact = backend.lower(graph)
        k1 = backend.compile(artifact)
        k2 = backend.compile(artifact)
        # Both should succeed; second uses cache
        assert k1.success and k2.success
        assert k1.metadata["cubin"] == k2.metadata["cubin"]

    def test_unsupported_op_raises(self, backend):
        graph = IRGraph(name="test")
        graph.add_input("x", dtype="float32", shape=[64])
        graph.add_node(IRNode(id="n0", op="relu", inputs={"x": "x"}, outputs=["out"]))
        graph.set_outputs(["out"])
        with pytest.raises(ValueError, match="does not support"):
            backend.lower(graph)
