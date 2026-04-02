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
        """Initialize the kernel cache with empty matmul and softmax caches."""
        self._backend = TritonBackend()
        self._compiler = TritonCompiler()
        self._matmul_cache: dict[tuple[int, int, int], Callable] = {}
        self._softmax_cache: dict[tuple[int, int], Callable] = {}
        self._layernorm_cache: dict[tuple[str, int, int], Callable] = {}
        self._elementwise_cache: dict[tuple[str, int], Callable] = {}

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

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Direct matmul dispatch — always uses Arke Triton kernel."""
        orig_shape = a.shape
        k = a.shape[-1]
        m = 1
        for d in orig_shape[:-1]:
            m *= d
        n = b.shape[-1]

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

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Direct softmax dispatch — always uses Arke Triton kernel."""
        orig_shape = x.shape
        m = 1
        for d in orig_shape[:-1]:
            m *= d
        n = orig_shape[-1]

        key = (m, n)
        func = self._softmax_cache.get(key)
        if func is None:
            func = self._compile_softmax(m, n)
            self._softmax_cache[key] = func
        x_2d = x.reshape(m, n).contiguous()
        out = func(x_2d)
        return out.reshape(orig_shape)

    def precompile_layernorm(self, shapes: list[tuple[int, int]]) -> None:
        """Pre-compile layernorm and rmsnorm kernels for all given (M, N) shapes.

        Includes warmup to trigger Triton autotuning.
        """
        for m, n in shapes:
            for norm_type in ("layernorm", "rmsnorm"):
                key = (norm_type, m, n)
                if key not in self._layernorm_cache:
                    func = self._compile_layernorm(norm_type, m, n)
                    x = torch.randn(m, n, device="cuda", dtype=torch.float16)
                    w = torch.ones(n, device="cuda", dtype=torch.float16)
                    for _ in range(3):
                        func(x, w)
                    torch.cuda.synchronize()
                    del x, w
                    self._layernorm_cache[key] = func

    def layernorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Direct layernorm dispatch — always uses Arke Triton kernel."""
        orig_shape = x.shape
        m = 1
        for d in orig_shape[:-1]:
            m *= d
        n = orig_shape[-1]

        key = ("layernorm", m, n)
        func = self._layernorm_cache.get(key)
        if func is None:
            func = self._compile_layernorm("layernorm", m, n)
            self._layernorm_cache[key] = func
        x_2d = x.reshape(m, n).contiguous()
        out = func(x_2d, weight, bias, eps)
        return out.reshape(orig_shape)

    def rmsnorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Direct rmsnorm dispatch — always uses Arke Triton kernel."""
        orig_shape = x.shape
        m = 1
        for d in orig_shape[:-1]:
            m *= d
        n = orig_shape[-1]

        key = ("rmsnorm", m, n)
        func = self._layernorm_cache.get(key)
        if func is None:
            func = self._compile_layernorm("rmsnorm", m, n)
            self._layernorm_cache[key] = func
        x_2d = x.reshape(m, n).contiguous()
        out = func(x_2d, weight, None, eps)
        return out.reshape(orig_shape)

    def elementwise(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        """Direct elementwise dispatch — always uses Arke Triton kernel."""
        n_elements = x.numel()
        # Round up to next power of 2 for cache key (reduces unique compilations)
        n_rounded = 1
        while n_rounded < n_elements:
            n_rounded *= 2

        key = (activation, n_rounded)
        func = self._elementwise_cache.get(key)
        if func is None:
            func = self._compile_elementwise(activation, n_rounded)
            self._elementwise_cache[key] = func

        return func(x.contiguous())

    def relu(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: elementwise ReLU via Arke Triton kernel."""
        return self.elementwise(x, "relu")

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: elementwise GELU via Arke Triton kernel."""
        return self.elementwise(x, "gelu")

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: elementwise SiLU via Arke Triton kernel."""
        return self.elementwise(x, "silu")

    def precompile_elementwise(
        self, shapes: list[tuple[str, int, int]]
    ) -> None:
        """Pre-compile elementwise kernels for given (activation, M, N) shapes.

        Includes warmup.
        """
        for activation, m, n in shapes:
            n_elements = m * n
            n_rounded = 1
            while n_rounded < n_elements:
                n_rounded *= 2
            key = (activation, n_rounded)
            if key not in self._elementwise_cache:
                func = self._compile_elementwise(activation, n_rounded)
                x = torch.randn(m, n, device="cuda", dtype=torch.float16)
                for _ in range(3):
                    func(x)
                torch.cuda.synchronize()
                del x
                self._elementwise_cache[key] = func

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

    def _compile_elementwise(self, activation: str, n_rounded: int) -> Callable:
        """Compile and cache the raw function for an elementwise op."""
        b = KernelBuilder(f"{activation}_{n_rounded}")
        b.param("X", [n_rounded], "f16")
        node = b.op(activation, X="X")
        b.returns(node, [n_rounded], "f16")
        ir = b.build()

        strategy = StrategyIR()
        source = self._backend.translate(ir, strategy)
        compiled = self._compiler.compile(source)
        if not compiled.success:
            raise RuntimeError(
                f"elementwise {activation} compile failed: {compiled.error}"
            )

        module = self._compiler._import_module(compiled.binary_path)
        func = self._compiler._find_entry_function(module)
        return func

    def _compile_layernorm(self, norm_type: str, m: int, n: int) -> Callable:
        """Compile and cache the raw function for a layernorm/rmsnorm shape."""
        b = KernelBuilder(f"{norm_type}_{m}_{n}")
        b.param("X", [m, n], "f16")
        b.param("W", [n], "f16")
        if norm_type == "layernorm":
            b.param("B", [n], "f16")
            node = b.op(norm_type, X="X", W="W", B="B")
        else:
            node = b.op(norm_type, X="X", W="W")
        b.returns(node, [m, n], "f16")
        ir = b.build()

        strategy = StrategyIR()
        source = self._backend.translate(ir, strategy)
        compiled = self._compiler.compile(source)
        if not compiled.success:
            raise RuntimeError(f"{norm_type} compile failed: {compiled.error}")

        module = self._compiler._import_module(compiled.binary_path)
        func = self._compiler._find_entry_function(module)
        return func

    @property
    def stats(self) -> dict:
        """Return cache statistics with counts of compiled shapes."""
        return {
            "matmul_shapes": len(self._matmul_cache),
            "softmax_shapes": len(self._softmax_cache),
            "layernorm_shapes": len(self._layernorm_cache),
            "elementwise_shapes": len(self._elementwise_cache),
        }
