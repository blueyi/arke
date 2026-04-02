# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fast kernel cache — pre-compile Arke kernels for zero-overhead dispatch.

Eliminates per-call overhead by:
1. Pre-compiling all unique shapes at init
2. Caching the Python function object (not re-importing each time)
3. Direct function call (no dict wrapping, no backend.run())
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from arke.backend.compiler import TritonCompiler
from arke.backend.triton_backend import TritonBackend
from arke.ir.builder import KernelBuilder
from arke.ir.strategy import StrategyIR


class KernelCache:
    """Pre-compiled kernel cache with direct dispatch."""

    def __init__(self):
        self._backend = TritonBackend()
        self._compiler = TritonCompiler()
        self._matmul_cache: dict[tuple[int, int, int], Callable] = {}
        self._softmax_cache: dict[tuple[int, int], Callable] = {}

    def precompile_matmul(self, shapes: list[tuple[int, int, int]]) -> None:
        """Pre-compile matmul kernels for all given (M, N, K) shapes.

        Includes warmup to trigger Triton autotuning.
        """
        for m, n, k in shapes:
            if (m, n, k) not in self._matmul_cache:
                func = self._compile_matmul(m, n, k)
                # Warmup to trigger autotune
                a = torch.randn(m, k, device="cuda", dtype=torch.float16)
                b = torch.randn(k, n, device="cuda", dtype=torch.float16)
                for _ in range(3):
                    func(a, b)
                torch.cuda.synchronize()
                del a, b
                self._matmul_cache[(m, n, k)] = func

    def precompile_softmax(self, shapes: list[tuple[int, int]]) -> None:
        """Pre-compile softmax kernels for all given (M, N) shapes.

        Includes warmup.
        """
        for m, n in shapes:
            if (m, n) not in self._softmax_cache:
                func = self._compile_softmax(m, n)
                x = torch.randn(m, n, device="cuda", dtype=torch.float16)
                for _ in range(3):
                    func(x)
                torch.cuda.synchronize()
                del x
                self._softmax_cache[(m, n)] = func

    # Threshold: use cuBLAS for matmuls where M is small (Triton launch overhead
    # dominates). Empirically on Ampere: Triton wins when M ≥ 512.
    TRITON_M_THRESHOLD = 384  # Use Triton only when M ≥ this

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Direct matmul dispatch — cuBLAS for small M, Triton for large M."""
        orig_shape = a.shape
        k = a.shape[-1]
        m = 1
        for d in orig_shape[:-1]:
            m *= d
        n = b.shape[-1]

        # Small M fast path: cuBLAS is faster due to lower launch overhead
        if m < self.TRITON_M_THRESHOLD:
            return torch.matmul(a, b)

        a_2d = a.reshape(m, k).contiguous()
        b_2d = b.contiguous()

        key = (m, n, k)
        func = self._matmul_cache.get(key)
        if func is None:
            func = self._compile_matmul(m, n, k)
            self._matmul_cache[key] = func

        out = func(a_2d, b_2d)
        out_shape = list(orig_shape[:-1]) + [n]
        return out.reshape(out_shape)

    # Softmax threshold: PyTorch is faster for small M*N
    SOFTMAX_THRESHOLD = 64 * 1024  # ~64K elements

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Direct softmax dispatch — falls back to PyTorch for small shapes."""
        orig_shape = x.shape
        m = 1
        for d in orig_shape[:-1]:
            m *= d
        n = orig_shape[-1]

        # Small shape fast path
        if m * n < self.SOFTMAX_THRESHOLD:
            return torch.nn.functional.softmax(x, dim=-1)

        key = (m, n)
        func = self._softmax_cache.get(key)
        if func is None:
            func = self._compile_softmax(m, n)
            self._softmax_cache[key] = func
        x_2d = x.reshape(m, n).contiguous()
        out = func(x_2d)
        return out.reshape(orig_shape)

    def _compile_matmul(self, m: int, n: int, k: int) -> Callable:
        """Compile and cache the raw function for a matmul shape."""
        b = KernelBuilder(f"matmul_{m}_{n}_{k}")
        b.param("A", [m, k], "f16")
        b.param("B", [k, n], "f16")
        node = b.op("matmul", A="A", B="B")
        b.returns(node, [m, n], "f16")
        ir = b.build()

        strategy = StrategyIR()
        source = self._backend.translate(ir, strategy)
        compiled = self._compiler.compile(source)
        if not compiled.success:
            raise RuntimeError(f"matmul compile failed: {compiled.error}")

        # Extract the raw function — avoid re-import overhead
        module = self._compiler._import_module(compiled.binary_path)
        func = self._compiler._find_entry_function(module)
        return func

    def _compile_softmax(self, m: int, n: int) -> Callable:
        """Compile and cache the raw function for a softmax shape."""
        b = KernelBuilder(f"softmax_{m}_{n}")
        b.param("X", [m, n], "f16")
        node = b.op("softmax", X="X")
        b.returns(node, [m, n], "f16")
        ir = b.build()

        strategy = StrategyIR()
        source = self._backend.translate(ir, strategy)
        compiled = self._compiler.compile(source)
        if not compiled.success:
            raise RuntimeError(f"softmax compile failed: {compiled.error}")

        module = self._compiler._import_module(compiled.binary_path)
        func = self._compiler._find_entry_function(module)
        return func

    @property
    def stats(self) -> dict:
        return {
            "matmul_shapes": len(self._matmul_cache),
            "softmax_shapes": len(self._softmax_cache),
        }
