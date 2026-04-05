# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5: Arke-generated Triton kernels via KernelCache."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    from arke.integration.kernel_cache import KernelCache  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


@register_baseline
class ArkeRunner(BaselineRunner):
    """P5: Arke KernelCache — LLM-generated Triton kernels.

    Bypasses KernelCache's cuBLAS/PyTorch fallbacks so we always
    measure the actual Arke-generated Triton kernel, even for shapes
    where KernelCache would normally fall back.
    """

    def __init__(self) -> None:
        self._cache: KernelCache | None = None

    def _ensure_cache(self) -> KernelCache:
        if self._cache is None:
            from arke.integration.kernel_cache import KernelCache

            self._cache = KernelCache()
        return self._cache

    @property
    def name(self) -> str:
        return "Arke"

    @property
    def priority(self) -> int:
        return 5

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            import arke

            v = getattr(arke, "__version__", "unknown")
        except Exception:
            pass
        return (
            f"Arke {v} (KernelCache Triton codegen) | "
            "https://github.com/arke-ai/arke | License: Apache-2.0"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    # Ops with specialized compile paths (original high-perf path)
    _SPECIALIZED_OPS = frozenset({"matmul", "softmax", "relu", "gelu", "silu", "layernorm", "rmsnorm"})
    # All ops that compile_op can handle via generic path
    _GENERIC_OPS = frozenset({
        "tanh", "sigmoid", "neg", "exp", "rsqrt", "cast", "copy_",
        "add", "mul", "where_",
        "rmsnorm_residual", "reduce_sum", "reduce_max", "reduce_mean",
        "argmax", "topk", "cumsum",
        "batch_matmul", "transpose",
        "swiglu", "geglu",
        "flash_attention", "grouped_query_attention", "cross_attention",
        # These may fail at compile time, but supports() still returns True
        "concat", "split", "gather", "scatter", "embedding", "permute",
        "rope", "cross_entropy", "fused_linear_cross_entropy",
        "quantize_per_token", "dequantize_per_channel",
        "grouped_matmul", "multi_latent_attention", "paged_attention",
    })

    def supports(self, op: str) -> bool:
        return op in self._SPECIALIZED_OPS or op in self._GENERIC_OPS

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        cache = self._ensure_cache()

        # Use specialized paths for original ops (better performance)
        if op == "matmul":
            return self._get_matmul_fn(cache, M, N, K, dtype)
        elif op == "softmax":
            return self._get_softmax_fn(cache, M, N, dtype)
        elif op in ("relu", "gelu", "silu"):
            return self._get_elementwise_fn(cache, op, M, N, dtype)
        elif op in ("layernorm", "rmsnorm"):
            return self._get_layernorm_fn(cache, op, M, N, dtype)

        # Generic path via compile_op/run_op for all other ops
        return self._get_generic_fn(cache, op, M, N, K, dtype)

    def _get_matmul_fn(
        self,
        cache: KernelCache,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> Callable[[], torch.Tensor] | None:
        """Get a matmul callable that uses the Arke Triton kernel.

        Compiles the kernel for the exact shape and returns the raw function
        for benchmarking with minimal dispatch overhead.
        """
        # Force-compile the Triton kernel for this shape (bypasses threshold)
        cache.precompile_matmul([(M, N, K)])

        # Grab the raw compiled function — this IS the Triton kernel
        raw_fn = cache._matmul_cache.get((M, N, K))
        if raw_fn is None:
            logger.warning(f"Arke matmul compile failed for ({M}, {N}, {K})")
            return None

        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)

        # Warmup the raw Triton kernel
        for _ in range(3):
            raw_fn(A, B)
        torch.cuda.synchronize()

        return lambda: raw_fn(A, B)

    def _get_softmax_fn(
        self,
        cache: KernelCache,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Callable[[], torch.Tensor] | None:
        """Get a softmax callable that uses the Arke Triton kernel.

        Compiles the kernel for the exact shape and returns the raw function
        for benchmarking with minimal dispatch overhead.
        """
        # Force-compile the Triton kernel for this shape
        cache.precompile_softmax([(M, N)])

        raw_fn = cache._softmax_cache.get((M, N))
        if raw_fn is None:
            logger.warning(f"Arke softmax compile failed for ({M}, {N})")
            return None

        X = torch.randn(M, N, device="cuda", dtype=dtype)

        # Warmup the raw Triton kernel
        for _ in range(3):
            raw_fn(X)
        torch.cuda.synchronize()

        return lambda: raw_fn(X)

    def _get_elementwise_fn(
        self,
        cache: KernelCache,
        op: str,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Callable[[], torch.Tensor] | None:
        """Get an elementwise callable that uses the Arke Triton kernel.

        Compiles the kernel for the exact shape and returns the raw function
        for benchmarking with minimal dispatch overhead.
        """
        cache.precompile_elementwise([(op, M, N)])

        X = torch.randn(M, N, device="cuda", dtype=dtype)

        # Warmup
        for _ in range(3):
            cache.elementwise(X, op)
        torch.cuda.synchronize()

        return lambda: cache.elementwise(X, op)

    def _get_layernorm_fn(
        self,
        cache: KernelCache,
        op: str,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Callable[[], torch.Tensor] | None:
        """Get a layernorm/rmsnorm callable that uses the Arke Triton kernel.

        Compiles the kernel for the exact shape and returns the raw function
        for benchmarking with minimal dispatch overhead.
        """
        cache.precompile_layernorm([(M, N)])

        X = torch.randn(M, N, device="cuda", dtype=dtype)
        W = torch.ones(N, device="cuda", dtype=dtype)

        if op == "layernorm":
            B = torch.zeros(N, device="cuda", dtype=dtype)
            # Warmup
            for _ in range(3):
                cache.layernorm(X, W, B)
            torch.cuda.synchronize()
            return lambda: cache.layernorm(X, W, B)
        else:
            # Warmup
            for _ in range(3):
                cache.rmsnorm(X, W)
            torch.cuda.synchronize()
            return lambda: cache.rmsnorm(X, W)

    def _get_generic_fn(
        self,
        cache: KernelCache,
        op: str,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> Callable[[], torch.Tensor] | None:
        """Generic path using KernelCache.run_op for ops without specialized paths."""
        try:
            # Build test inputs
            tensors = self._build_test_inputs(op, M, N, K, dtype)
            if tensors is None:
                return None

            # Try compile + warmup
            result = cache.run_op(op, *tensors)
            if result is None:
                logger.debug(f"Arke generic compile failed for {op}")
                return None

            # Warmup
            for _ in range(3):
                cache.run_op(op, *tensors)
            torch.cuda.synchronize()

            return lambda: cache.run_op(op, *tensors)
        except Exception as e:
            logger.debug(f"Arke generic {op}: {e}")
            return None

    @staticmethod
    def _build_test_inputs(
        op: str, M: int, N: int, K: int, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...] | None:
        """Build test input tensors for a given op."""
        device = "cuda"
        # Unary elementwise
        if op in ("tanh", "sigmoid", "neg", "exp", "rsqrt", "cast", "copy_"):
            X = torch.randn(M, N, device=device, dtype=dtype)
            if op == "rsqrt":
                X = X.abs() + 0.01  # positive for rsqrt
            return (X,)
        # Binary
        if op in ("add", "mul"):
            A = torch.randn(M, N, device=device, dtype=dtype)
            B = torch.randn(M, N, device=device, dtype=dtype)
            return (A, B)
        # Ternary
        if op == "where_":
            cond = torch.randn(M, N, device=device) > 0
            A = torch.randn(M, N, device=device, dtype=dtype)
            B = torch.randn(M, N, device=device, dtype=dtype)
            return (cond, A, B)
        # Reductions
        if op in ("reduce_sum", "reduce_max", "reduce_mean", "argmax", "topk", "cumsum"):
            X = torch.randn(M, N, device=device, dtype=dtype)
            return (X,)
        # Normalization
        if op == "rmsnorm_residual":
            X = torch.randn(M, N, device=device, dtype=dtype)
            residual = torch.randn(M, N, device=device, dtype=dtype)
            W = torch.ones(N, device=device, dtype=dtype)
            return (X, residual, W)
        # Matmul variants
        if op == "batch_matmul":
            B = max(K, 4)  # use K as batch dim
            A = torch.randn(B, M, N, device=device, dtype=dtype)
            Bt = torch.randn(B, N, N, device=device, dtype=dtype)
            return (A, Bt)
        # Data movement
        if op == "transpose":
            X = torch.randn(M, N, device=device, dtype=dtype)
            return (X,)
        # Gated
        if op in ("swiglu", "geglu"):
            X = torch.randn(M, N * 2, device=device, dtype=dtype)  # 2N input
            return (X,)
        # Attention
        if op in ("flash_attention", "grouped_query_attention", "cross_attention"):
            # M=B*H, N=S, K=D from bench_l1 shape mapping
            B_dim = max(1, M // max(N, 1))  # approximate B
            H_dim = max(1, M // max(B_dim, 1))
            S = N
            D = max(K, 64)
            Q = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            Kk = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            V = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            return (Q, Kk, V)
        # Other ops — not yet supported in generic path
        return None
