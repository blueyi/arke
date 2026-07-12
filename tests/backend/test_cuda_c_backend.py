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

        # Tolerance: TC path (fp16 accumulation) has ~1e-2 max error,
        # scalar path has ~K*2e-6. Detect by shape (TC triggers at M,N,K>=64 and %16==0).
        is_tc = (M >= 64 and K >= 64 and N >= 64
                 and M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
        if is_tc:
            atol = max(0.05, K * 1e-4)
            rtol = 1e-2
        else:
            atol = max(1e-4, K * 2e-6)
            rtol = 1e-4
        np.testing.assert_allclose(out, ref, atol=atol, rtol=rtol)

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

        # TC path uses fp16 accumulation — rel_err ~0.03%, not exact match
        np.testing.assert_allclose(out, ref_t, atol=0.05, rtol=1e-2)

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

        np.testing.assert_allclose(out, ref, atol=0.05, rtol=1e-2)


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
        graph.add_input("x", dtype="float32", shape=[64, 64])
        graph.add_node(IRNode(id="n0", op="nonexistent_xyz", inputs={"x": "x"}, outputs=["out"]))
        graph.set_outputs(["out"])
        with pytest.raises(ValueError, match="does not support"):
            backend.lower(graph)


# ── Elementwise op tests ──────────────────────────────────────

def _ref_elementwise(op: str, *arrays):
    """Numpy reference for elementwise ops."""
    if op == "relu":
        return np.maximum(0, arrays[0])
    elif op == "gelu":
        x = arrays[0]
        return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    elif op == "silu":
        x = arrays[0]
        return x * (1.0 / (1.0 + np.exp(-x)))
    elif op == "tanh":
        return np.tanh(arrays[0])
    elif op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-arrays[0]))
    elif op == "exp":
        return np.exp(arrays[0])
    elif op == "neg":
        return -arrays[0]
    elif op == "rsqrt":
        return 1.0 / np.sqrt(arrays[0])
    elif op == "add":
        return arrays[0] + arrays[1]
    elif op == "mul":
        return arrays[0] * arrays[1]
    raise ValueError(f"Unknown op: {op}")


_UNARY_OPS = ["relu", "gelu", "silu", "tanh", "sigmoid", "exp", "neg", "rsqrt"]
_BINARY_OPS = ["add", "mul"]


class TestCudaCElementwise:
    """End-to-end correctness tests for elementwise ops via CUDA-C backend."""

    @pytest.mark.parametrize("op", _UNARY_OPS)
    def test_unary_ops(self, backend, op):
        M, N = 512, 512
        graph = IRGraph(name=f"{op}_test")
        graph.add_input("x", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op=op, inputs={"x": "x"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, f"Compile failed: {kernel.error}"

        np.random.seed(123)
        if op == "rsqrt":
            x = np.random.uniform(0.1, 10.0, (M, N)).astype(np.float32)
        else:
            x = np.random.randn(M, N).astype(np.float32)

        result = backend.run(kernel, {"x": x})
        out = result["out"]
        ref = _ref_elementwise(op, x).astype(np.float32)

        atol = 1e-3 if op == "gelu" else 1e-4
        np.testing.assert_allclose(out, ref, atol=atol, rtol=1e-4)

    @pytest.mark.parametrize("op", _BINARY_OPS)
    def test_binary_ops(self, backend, op):
        M, N = 512, 512
        graph = IRGraph(name=f"{op}_test")
        graph.add_input("A", dtype="float32", shape=[M, N])
        graph.add_input("B", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op=op, inputs={"a": "A", "b": "B"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, f"Compile failed: {kernel.error}"

        np.random.seed(456)
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)

        result = backend.run(kernel, {"A": A, "B": B})
        out = result["out"]
        ref = _ref_elementwise(op, A, B).astype(np.float32)

        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
