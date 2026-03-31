# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests: IR → Strategy → Codegen → Compile → Run on GPU.

All GPU tests are skipped if CUDA is not available.
"""

from __future__ import annotations

import ast
import time

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

try:
    from arke.backend.compiler import TritonCompiler
    from arke.backend.triton_backend import TritonBackend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

from arke.backend.triton_template_engine import TritonTemplateEngine
from arke.ir.semantic import (
    Edge,
    FusionGroup,
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    TensorDesc,
)
from arke.ir.strategy import StrategyIR

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
skip_no_gpu = pytest.mark.skipif(not HAS_CUDA, reason="No GPU or torch available")


# ============================================================
# Helpers
# ============================================================


def _build_matmul_ir(
    M: int = 512, N: int = 512, K: int = 512, dtype: str = "float32",
) -> SemanticIR:
    ir = SemanticIR(kernel_id="e2e_matmul")
    ir.add_param(Param(name="A", shape=[M, K], dtype=dtype))
    ir.add_param(Param(name="B", shape=[K, N], dtype=dtype))
    ir.return_type = TensorDesc(shape=[M, N], dtype=dtype)
    ir.add_node(Node(
        id="matmul_0",
        op="matmul",
        inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
        output=TensorDesc(shape=[M, N], dtype=dtype),
        semantics=Semantics(
            computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
            index_vars=["i", "j", "k"],
            reduction_axes=["k"],
        ),
    ))
    ir.return_node = "matmul_0"
    return ir


def _build_matmul_relu_ir(
    M: int = 512, N: int = 512, K: int = 512,
) -> SemanticIR:
    ir = _build_matmul_ir(M, N, K)
    ir.kernel_id = "e2e_matmul_relu"
    ir.add_node(Node(
        id="relu_0",
        op="relu",
        inputs={"X": NodeRef(id="matmul_0")},
        output=TensorDesc(shape=[M, N], dtype="float32"),
        semantics=Semantics(computation="Y = max(X, 0)", properties=["elementwise"]),
    ))
    ir.add_edge(Edge(from_node="matmul_0", to_node="relu_0", tensor_name="C"))
    ir.return_node = "relu_0"
    ir.add_fusion_group(FusionGroup(
        id="fg_0",
        nodes=["matmul_0", "relu_0"],
        fusion_type="epilogue",
        reason="fuse relu into matmul",
    ))
    return ir


def _build_softmax_ir(M: int = 128, N: int = 1024) -> SemanticIR:
    ir = SemanticIR(kernel_id="e2e_softmax")
    ir.add_param(Param(name="X", shape=[M, N], dtype="float32"))
    ir.return_type = TensorDesc(shape=[M, N], dtype="float32")
    ir.add_node(Node(
        id="softmax_0",
        op="softmax",
        inputs={"X": ParamRef(name="X")},
        output=TensorDesc(shape=[M, N], dtype="float32"),
        semantics=Semantics(
            computation="Y[i,j] = exp(X[i,j]) / sum(exp(X[i,:]), axis=j)",
            index_vars=["i", "j"],
            reduction_axes=["j"],
        ),
    ))
    ir.return_node = "softmax_0"
    return ir


def _apply_tiles(
    kernel_id: str, block_m: int = 128, block_n: int = 128, block_k: int = 32,
) -> StrategyIR:
    s = StrategyIR(kernel_id=kernel_id, target_hw="nvidia_ampere")
    s.tile("i", [block_m])
    s.tile("j", [block_n])
    s.tile("k", [block_k])
    return s


# ============================================================
# E2E Matmul Tests (GPU)
# ============================================================


@skip_no_gpu
class TestE2EMatmul:
    """End-to-end: build IR → codegen → compile → run → verify."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()
        self.compiler = TritonCompiler()

    def test_matmul_correctness(self) -> None:
        """Generated matmul kernel matches torch.matmul."""
        M, N, K = 512, 512, 512
        ir = _build_matmul_ir(M, N, K)
        strategy = _apply_tiles("e2e_matmul")

        code = self.engine.translate(ir, strategy)
        # Verify it's valid Python
        ast.parse(code)

        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        result = self.compiler.compile_and_run(code, {"A": A, "B": B})
        expected = torch.matmul(A, B)

        torch.testing.assert_close(result, expected, rtol=5e-2, atol=5e-2)

    def test_matmul_different_tile_sizes(self) -> None:
        """Test with various tile configurations."""
        M, N, K = 256, 256, 256
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        expected = torch.matmul(A, B)

        tile_configs = [
            (64, 64, 32),
            (128, 64, 32),
            (64, 128, 64),
            (32, 32, 32),
        ]

        for bm, bn, bk in tile_configs:
            ir = _build_matmul_ir(M, N, K)
            strategy = _apply_tiles("e2e_matmul", bm, bn, bk)
            code = self.engine.translate(ir, strategy)
            result = self.compiler.compile_and_run(code, {"A": A, "B": B})
            torch.testing.assert_close(
                result, expected, rtol=5e-2, atol=5e-2,
                msg=f"Failed with tiles ({bm}, {bn}, {bk})",
            )

    def test_matmul_non_square(self) -> None:
        """Non-square matrix multiplication."""
        M, N, K = 384, 768, 256
        ir = _build_matmul_ir(M, N, K)
        strategy = _apply_tiles("e2e_matmul", 128, 128, 32)

        code = self.engine.translate(ir, strategy)
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        result = self.compiler.compile_and_run(code, {"A": A, "B": B})
        expected = torch.matmul(A, B)
        torch.testing.assert_close(result, expected, rtol=5e-2, atol=5e-2)

    def test_fused_matmul_relu(self) -> None:
        """Fused matmul + relu should match torch equivalent."""
        M, N, K = 512, 512, 512
        ir = _build_matmul_relu_ir(M, N, K)
        strategy = _apply_tiles("e2e_matmul_relu")
        strategy.fuse(["matmul_0", "relu_0"], fusion_type="epilogue")

        code = self.engine.translate(ir, strategy)
        assert "tl.maximum(acc, 0.0)" in code

        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        result = self.compiler.compile_and_run(code, {"A": A, "B": B})
        expected = torch.relu(torch.matmul(A, B))
        torch.testing.assert_close(result, expected, rtol=5e-2, atol=5e-2)


# ============================================================
# E2E Softmax Tests (GPU)
# ============================================================


@skip_no_gpu
class TestE2ESoftmax:
    """End-to-end softmax tests."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()
        self.compiler = TritonCompiler()

    def test_softmax_correctness(self) -> None:
        """Generated softmax matches torch.softmax."""
        M, N = 128, 1024
        ir = _build_softmax_ir(M, N)
        strategy = StrategyIR(kernel_id="e2e_softmax")

        code = self.engine.translate(ir, strategy)
        X = torch.randn(M, N, device="cuda", dtype=torch.float32)

        result = self.compiler.compile_and_run(code, {"X": X})
        expected = torch.softmax(X, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_softmax_small(self) -> None:
        """Softmax with small N."""
        M, N = 32, 64
        ir = _build_softmax_ir(M, N)
        strategy = StrategyIR(kernel_id="e2e_softmax")

        code = self.engine.translate(ir, strategy)
        X = torch.randn(M, N, device="cuda", dtype=torch.float32)

        result = self.compiler.compile_and_run(code, {"X": X})
        expected = torch.softmax(X, dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


# ============================================================
# Performance Profiling (GPU)
# ============================================================


@skip_no_gpu
class TestPerformanceProfiling:
    """Profile generated kernels and compare to cuBLAS (torch.matmul)."""

    def setup_method(self) -> None:
        self.engine = TritonTemplateEngine()
        self.compiler = TritonCompiler()

    def test_profile_matmul(self) -> None:
        """Profile matmul kernel and report performance vs cuBLAS."""
        M, N, K = 1024, 1024, 1024
        ir = _build_matmul_ir(M, N, K)
        strategy = _apply_tiles("e2e_matmul", 128, 128, 32)
        code = self.engine.translate(ir, strategy)

        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Profile Triton kernel
        triton_result = self.compiler.profile(
            code, {"A": A, "B": B}, warmup=10, runs=50,
        )

        # Profile cuBLAS baseline
        for _ in range(10):
            torch.matmul(A, B)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        for i in range(50):
            start_events[i].record()
            torch.matmul(A, B)
            end_events[i].record()
        torch.cuda.synchronize()
        cublas_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        cublas_avg_ms = sum(cublas_times) / len(cublas_times)
        cublas_latency_us = cublas_avg_ms * 1000.0

        flops = 2.0 * M * N * K
        cublas_tflops = flops / (cublas_avg_ms * 1e-3) / 1e12

        # Report
        print("\n" + "=" * 60)
        print(f"  Matmul Performance ({M}x{N}x{K}, float32)")
        print("=" * 60)
        print(f"  Triton (Arke):  {triton_result.latency_us:.1f} us  |  {triton_result.tflops:.2f} TFLOPS")
        print(f"  cuBLAS:         {cublas_latency_us:.1f} us  |  {cublas_tflops:.2f} TFLOPS")
        if cublas_tflops > 0:
            ratio = triton_result.tflops / cublas_tflops * 100
            print(f"  Ratio:          {ratio:.1f}% of cuBLAS")
        print("=" * 60)

        # Sanity check: kernel should at least run
        assert triton_result.latency_us > 0
        assert triton_result.tflops > 0

    def test_profile_matmul_large(self) -> None:
        """Profile with larger matrices (2048x2048)."""
        M, N, K = 2048, 2048, 2048
        ir = _build_matmul_ir(M, N, K)
        strategy = _apply_tiles("e2e_matmul", 128, 128, 32)
        code = self.engine.translate(ir, strategy)

        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        triton_result = self.compiler.profile(
            code, {"A": A, "B": B}, warmup=10, runs=30,
        )

        # cuBLAS baseline
        for _ in range(10):
            torch.matmul(A, B)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        for i in range(30):
            start_events[i].record()
            torch.matmul(A, B)
            end_events[i].record()
        torch.cuda.synchronize()
        cublas_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        cublas_avg_ms = sum(cublas_times) / len(cublas_times)

        flops = 2.0 * M * N * K
        cublas_tflops = flops / (cublas_avg_ms * 1e-3) / 1e12

        print("\n" + "=" * 60)
        print(f"  Matmul Performance ({M}x{N}x{K}, float32)")
        print("=" * 60)
        print(f"  Triton (Arke):  {triton_result.latency_us:.1f} us  |  {triton_result.tflops:.2f} TFLOPS")
        print(f"  cuBLAS:         {cublas_avg_ms * 1000:.1f} us  |  {cublas_tflops:.2f} TFLOPS")
        if cublas_tflops > 0:
            ratio = triton_result.tflops / cublas_tflops * 100
            print(f"  Ratio:          {ratio:.1f}% of cuBLAS")
        print("=" * 60)

        assert triton_result.latency_us > 0


# ============================================================
# TritonBackend full pipeline (GPU)
# ============================================================


@skip_no_gpu
class TestTritonBackendPipeline:
    """Test the full TritonBackend class pipeline."""

    def test_full_pipeline(self) -> None:
        """translate -> compile -> run through TritonBackend."""
        backend = TritonBackend()
        ir = _build_matmul_ir(256, 256, 256)
        strategy = _apply_tiles("e2e_matmul", 64, 64, 32)

        code = backend.translate(ir, strategy)
        compiled = backend.compile(code)
        assert compiled.success

        A = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        B = torch.randn(256, 256, device="cuda", dtype=torch.float32)

        outputs = backend.run(compiled, {"A": A, "B": B})
        result = outputs["output"]
        expected = torch.matmul(A, B)
        torch.testing.assert_close(result, expected, rtol=5e-2, atol=5e-2)
