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
        self._generic_cache: dict[tuple, Callable] = {}  # (op, *shape_key) → fn

    # ============================================================
    # Generic compile/run — covers all 45 ops
    # ============================================================

    # Ops where KernelBuilder can build IR directly (single-input elementwise)
    _UNARY_ELEMENTWISE = frozenset({
        "relu", "gelu", "silu", "tanh", "sigmoid", "neg", "exp", "rsqrt", "cast", "copy_",
    })
    _BINARY_ELEMENTWISE = frozenset({"add", "mul"})

    def compile_op(self, op: str, **shape_params) -> Callable | None:
        """Generic compile: build SemanticIR → TritonBackend → compile → cached fn.

        Returns None if compilation fails (unsupported op, template error, etc.).
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            ir, strategy = self._build_ir(op, **shape_params)
            source = self._backend.translate(ir, strategy)
            compiled = self._compiler.compile(source)
            if not compiled.success:
                logger.warning(f"compile_op({op}): compile failed: {compiled.error}")
                return None
            module = self._compiler._import_module(compiled.binary_path)
            func = self._compiler._find_entry_function(module)
            return func
        except Exception as e:
            logger.warning(f"compile_op({op}): {e}")
            return None

    def run_op(self, op: str, *tensors: torch.Tensor, **kwargs) -> torch.Tensor | None:
        """Generic dispatch: compile if needed, then execute.

        Returns None if compilation fails or op is unsupported.
        """
        key = self._cache_key(op, *tensors)
        if key not in self._generic_cache:
            shape_params = self._shape_params(op, *tensors)
            fn = self.compile_op(op, **shape_params)
            if fn is None:
                return None
            self._generic_cache[key] = fn
        return self._generic_cache[key](*tensors)

    def _cache_key(self, op: str, *tensors: torch.Tensor) -> tuple:
        """Build a cache key from op name + tensor shapes."""
        return (op,) + tuple(t.shape for t in tensors)

    def _shape_params(self, op: str, *tensors: torch.Tensor) -> dict:
        """Extract shape parameters for IR building from tensors."""
        if op in self._UNARY_ELEMENTWISE | {"softmax", "cumsum"}:
            x = tensors[0]
            return {"M": x.shape[0] if x.ndim >= 2 else 1, "N": x.shape[-1], "n_elements": x.numel()}
        if op in self._BINARY_ELEMENTWISE:
            a = tensors[0]
            return {"M": a.shape[0] if a.ndim >= 2 else 1, "N": a.shape[-1], "n_elements": a.numel()}
        if op == "where_":
            a = tensors[1]  # cond, A, B
            return {"M": a.shape[0] if a.ndim >= 2 else 1, "N": a.shape[-1], "n_elements": a.numel()}
        if op == "matmul":
            a, b = tensors[0], tensors[1]
            return {"M": a.shape[0], "K": a.shape[1], "N": b.shape[1]}
        if op == "batch_matmul":
            a, b = tensors[0], tensors[1]
            return {"B": a.shape[0], "M": a.shape[1], "K": a.shape[2], "N": b.shape[2]}
        if op in ("layernorm", "rmsnorm", "rmsnorm_residual"):
            x = tensors[0]
            return {"M": x.shape[0] if x.ndim >= 2 else 1, "N": x.shape[-1]}
        if op in ("reduce_sum", "reduce_max", "reduce_mean", "argmax", "topk"):
            x = tensors[0]
            return {"M": x.shape[0] if x.ndim >= 2 else 1, "N": x.shape[-1]}
        if op in ("swiglu", "geglu"):
            x = tensors[0]
            return {"M": x.shape[0] if x.ndim >= 2 else 1, "N": x.shape[-1]}
        if op in ("flash_attention", "grouped_query_attention", "cross_attention", "multi_latent_attention", "paged_attention"):
            q = tensors[0]
            return {"B": q.shape[0], "H": q.shape[1], "S": q.shape[2], "D": q.shape[3]} if q.ndim == 4 else {"M": q.shape[0], "N": q.shape[-1]}
        # Default: use first tensor
        t = tensors[0]
        return {"M": t.shape[0] if t.ndim >= 2 else 1, "N": t.shape[-1], "n_elements": t.numel()}

    def _build_ir(self, op: str, **sp):
        """Build SemanticIR + StrategyIR for any op."""
        strategy = StrategyIR()

        if op in self._UNARY_ELEMENTWISE:
            n = sp.get("n_elements", sp.get("M", 1) * sp.get("N", 1))
            b = KernelBuilder(f"{op}_{n}")
            b.param("X", [n], "f16")
            node = b.op(op, X="X")
            b.returns(node, [n], "f16")
            return b.build(), strategy

        if op in self._BINARY_ELEMENTWISE:
            n = sp.get("n_elements", sp.get("M", 1) * sp.get("N", 1))
            b = KernelBuilder(f"{op}_{n}")
            b.param("A", [n], "f16")
            b.param("B", [n], "f16")
            node = b.op(op, A="A", B="B")
            b.returns(node, [n], "f16")
            return b.build(), strategy

        if op == "where_":
            n = sp.get("n_elements", sp.get("M", 1) * sp.get("N", 1))
            b = KernelBuilder(f"where_{n}")
            b.param("cond", [n], "bool")
            b.param("A", [n], "f16")
            b.param("B", [n], "f16")
            node = b.op("where_", cond="cond", A="A", B="B")
            b.returns(node, [n], "f16")
            return b.build(), strategy

        if op == "matmul":
            m, n, k = sp["M"], sp["N"], sp["K"]
            b = KernelBuilder(f"matmul_{m}_{n}_{k}")
            b.param("A", [m, k], "f16")
            b.param("B", [k, n], "f16")
            node = b.op("matmul", A="A", B="B")
            b.returns(node, [m, n], "f16")
            return b.build(), strategy

        if op == "batch_matmul":
            B, m, n, k = sp["B"], sp["M"], sp["N"], sp["K"]
            b = KernelBuilder(f"batch_matmul_{B}_{m}_{n}_{k}")
            b.param("A", [B, m, k], "f16")
            b.param("B", [B, k, n], "f16")
            node = b.op("batch_matmul", A="A", B="B")
            b.returns(node, [B, m, n], "f16")
            return b.build(), strategy

        if op == "softmax":
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"softmax_{m}_{n}")
            b.param("X", [m, n], "f16")
            node = b.op("softmax", X="X")
            b.returns(node, [m, n], "f16")
            return b.build(), strategy

        if op in ("layernorm", "rmsnorm"):
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"{op}_{m}_{n}")
            b.param("X", [m, n], "f16")
            b.param("W", [n], "f16")
            if op == "layernorm":
                b.param("B", [n], "f16")
                node = b.op(op, X="X", W="W", B="B")
            else:
                node = b.op(op, X="X", W="W")
            b.returns(node, [m, n], "f16")
            return b.build(), strategy

        if op == "rmsnorm_residual":
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"rmsnorm_residual_{m}_{n}")
            b.param("X", [m, n], "f16")
            b.param("residual", [m, n], "f16")
            b.param("W", [n], "f16")
            node = b.op("rmsnorm_residual", X="X", residual="residual", W="W")
            b.returns(node, [m, n], "f16")
            return b.build(), strategy

        if op in ("reduce_sum", "reduce_max", "reduce_mean", "argmax"):
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"{op}_{m}_{n}")
            b.param("X", [m, n], "f16")
            node = b.op(op, X="X")
            b.returns(node, [m], "f16")
            return b.build(), strategy

        if op in ("topk", "cumsum"):
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"{op}_{m}_{n}")
            b.param("X", [m, n], "f16")
            node = b.op(op, X="X")
            out_shape = [m, n] if op == "cumsum" else [m, n]  # topk: simplified
            b.returns(node, out_shape, "f16")
            return b.build(), strategy

        if op == "transpose":
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"transpose_{m}_{n}")
            b.param("X", [m, n], "f16")
            node = b.op("transpose", X="X")
            b.returns(node, [n, m], "f16")
            return b.build(), strategy

        if op in ("swiglu", "geglu"):
            m, n = sp["M"], sp["N"]
            b = KernelBuilder(f"{op}_{m}_{n}")
            b.param("X", [m, n], "f16")
            node = b.op(op, X="X")
            b.returns(node, [m, n // 2], "f16")
            return b.build(), strategy

        if op in ("flash_attention", "grouped_query_attention", "cross_attention"):
            B, H, S, D = sp.get("B", 1), sp.get("H", 1), sp.get("S", 64), sp.get("D", 64)
            b = KernelBuilder(f"{op}_{B}_{H}_{S}_{D}")
            b.param("Q", [B, H, S, D], "f16")
            b.param("K", [B, H, S, D], "f16")
            b.param("V", [B, H, S, D], "f16")
            node = b.op(op, Q="Q", K="K", V="V")
            b.returns(node, [B, H, S, D], "f16")
            return b.build(), strategy

        # Default fallback: try as unary elementwise
        n = sp.get("n_elements", sp.get("M", 1) * sp.get("N", 1))
        b = KernelBuilder(f"{op}_{n}")
        b.param("X", [n], "f16")
        node = b.op(op, X="X")
        b.returns(node, [n], "f16")
        return b.build(), strategy

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

    # Maximum N for single-block softmax (Triton constexpr limit).
    # Beyond this, compilation is extremely slow or OOM.
    SOFTMAX_MAX_N = 131072  # 128K elements

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Direct softmax dispatch — always uses Arke Triton kernel.

        Raises ValueError for N > SOFTMAX_MAX_N (single-block limitation).
        """
        # Fast path: 2D contiguous input
        if x.ndim == 2 and x.is_contiguous():
            m, n = x.shape
            if n > self.SOFTMAX_MAX_N:
                raise ValueError(
                    f"Arke softmax: N={n} exceeds single-block limit "
                    f"({self.SOFTMAX_MAX_N})."
                )
            func = self._softmax_cache.get((m, n))
            if func is None:
                func = self._compile_softmax(m, n)
                self._softmax_cache[(m, n)] = func
            return func(x)

        # General path: reshape to 2D
        orig_shape = x.shape
        n = orig_shape[-1]
        if n > self.SOFTMAX_MAX_N:
            raise ValueError(
                f"Arke softmax: N={n} exceeds single-block limit "
                f"({self.SOFTMAX_MAX_N})."
            )
        m = x.numel() // n
        func = self._softmax_cache.get((m, n))
        if func is None:
            func = self._compile_softmax(m, n)
            self._softmax_cache[(m, n)] = func
        x_2d = x.reshape(m, n).contiguous()
        return func(x_2d).reshape(orig_shape)

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
        # Fast path: 2D contiguous input
        if x.ndim == 2 and x.is_contiguous():
            m, n = x.shape
            func = self._layernorm_cache.get(("layernorm", m, n))
            if func is None:
                func = self._compile_layernorm("layernorm", m, n)
                self._layernorm_cache[("layernorm", m, n)] = func
            return func(x, weight, bias, eps)

        # General path
        orig_shape = x.shape
        n = orig_shape[-1]
        m = x.numel() // n
        func = self._layernorm_cache.get(("layernorm", m, n))
        if func is None:
            func = self._compile_layernorm("layernorm", m, n)
            self._layernorm_cache[("layernorm", m, n)] = func
        x_2d = x.reshape(m, n).contiguous()
        return func(x_2d, weight, bias, eps).reshape(orig_shape)

    def rmsnorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Direct rmsnorm dispatch — always uses Arke Triton kernel."""
        # Fast path: 2D contiguous input
        if x.ndim == 2 and x.is_contiguous():
            m, n = x.shape
            func = self._layernorm_cache.get(("rmsnorm", m, n))
            if func is None:
                func = self._compile_layernorm("rmsnorm", m, n)
                self._layernorm_cache[("rmsnorm", m, n)] = func
            return func(x, weight, None, eps)

        # General path
        orig_shape = x.shape
        n = orig_shape[-1]
        m = x.numel() // n
        func = self._layernorm_cache.get(("rmsnorm", m, n))
        if func is None:
            func = self._compile_layernorm("rmsnorm", m, n)
            self._layernorm_cache[("rmsnorm", m, n)] = func
        x_2d = x.reshape(m, n).contiguous()
        return func(x_2d, weight, None, eps).reshape(orig_shape)

    def elementwise(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        """Direct elementwise dispatch — always uses Arke Triton kernel."""
        n_elements = x.numel()
        key = (activation, n_elements)
        func = self._elementwise_cache.get(key)
        if func is None:
            func = self._compile_elementwise(activation, n_elements)
            self._elementwise_cache[key] = func
        # Skip .contiguous() if already contiguous (hot path)
        return func(x if x.is_contiguous() else x.contiguous())

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
            key = (activation, n_elements)
            if key not in self._elementwise_cache:
                func = self._compile_elementwise(activation, n_elements)
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

    def _compile_elementwise(self, activation: str, n_elements: int) -> Callable:
        """Compile and cache the raw function for an elementwise op."""
        b = KernelBuilder(f"{activation}_{n_elements}")
        b.param("X", [n_elements], "f16")
        node = b.op(activation, X="X")
        b.returns(node, [n_elements], "f16")
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
            "generic_shapes": len(self._generic_cache),
            "total_compiled": (
                len(self._matmul_cache) + len(self._softmax_cache)
                + len(self._layernorm_cache) + len(self._elementwise_cache)
                + len(self._generic_cache)
            ),
        }
