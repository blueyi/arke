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

    def supports(self, op: str) -> bool:
        return op in ("matmul", "softmax", "relu", "gelu", "silu", "layernorm", "rmsnorm")

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        cache = self._ensure_cache()

        if op == "matmul":
            return self._get_matmul_fn(cache, M, N, K, dtype)
        elif op == "softmax":
            return self._get_softmax_fn(cache, M, N, dtype)
        elif op in ("relu", "gelu", "silu"):
            return self._get_elementwise_fn(cache, op, M, N, dtype)
        elif op in ("layernorm", "rmsnorm"):
            return self._get_layernorm_fn(cache, op, M, N, dtype)
        return None

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
