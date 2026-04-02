# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P3: PyTorch eager baselines.

Identical to cuBLAS runner but explicitly named for clarity
in benchmark reports when comparing Arke vs "PyTorch default".
This runner represents what a user gets out of the box.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline


@register_baseline
class PyTorchEagerRunner(BaselineRunner):
    """P3: PyTorch eager mode (user's default)."""

    @property
    def name(self) -> str:
        return "PyTorch-eager"

    @property
    def priority(self) -> int:
        return 3

    @property
    def available(self) -> bool:
        return torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in (
            "matmul", "softmax", "layernorm", "relu", "gelu", "silu",
        )

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
            return lambda: torch.matmul(A, B)
        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.softmax(X, dim=-1)
        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            b = torch.zeros(N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.layer_norm(X, [N], w, b)
        elif op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.relu(X)
        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.gelu(X)
        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.silu(X)
        return None
