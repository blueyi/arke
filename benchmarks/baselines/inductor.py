# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P4: torch.compile (Inductor) baselines."""

from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline


@register_baseline
class InductorRunner(BaselineRunner):
    """P4: torch.compile-generated kernels."""

    @property
    def name(self) -> str:
        return "torch.compile"

    @property
    def priority(self) -> int:
        return 4

    @property
    def available(self) -> bool:
        return torch.cuda.is_available() and hasattr(torch, "compile")

    def supports(self, op: str) -> bool:
        return op in ("matmul", "softmax", "layernorm", "relu", "gelu", "silu")

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        if op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)

            @torch.compile(mode="reduce-overhead")
            def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.matmul(a, b)

            # Trigger compilation
            fn(A, B)
            torch.cuda.synchronize()
            return lambda: fn(A, B)

        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            @torch.compile(mode="reduce-overhead")
            def fn(x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.softmax(x, dim=-1)

            fn(X)
            torch.cuda.synchronize()
            return lambda: fn(X)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            b = torch.zeros(N, device="cuda", dtype=dtype)

            @torch.compile(mode="reduce-overhead")
            def fn(
                x: torch.Tensor, ww: torch.Tensor, bb: torch.Tensor
            ) -> torch.Tensor:
                return torch.nn.functional.layer_norm(x, [ww.shape[0]], ww, bb)

            fn(X, w, b)
            torch.cuda.synchronize()
            return lambda: fn(X, w, b)

        elif op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            @torch.compile(mode="reduce-overhead")
            def fn(x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.relu(x)

            fn(X)
            torch.cuda.synchronize()
            return lambda: fn(X)

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            @torch.compile(mode="reduce-overhead")
            def fn(x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x)

            fn(X)
            torch.cuda.synchronize()
            return lambda: fn(X)

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            @torch.compile(mode="reduce-overhead")
            def fn(x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.silu(x)

            fn(X)
            torch.cuda.synchronize()
            return lambda: fn(X)

        return None
