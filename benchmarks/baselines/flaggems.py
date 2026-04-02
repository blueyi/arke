# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P1: FlagGems baselines — 200+ Triton operators."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    import flag_gems  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


@register_baseline
class FlagGemsRunner(BaselineRunner):
    """P1: FlagGems Triton operator library."""

    @property
    def name(self) -> str:
        return "FlagGems"

    @property
    def priority(self) -> int:
        return 1

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in (
            "matmul", "softmax", "layernorm", "rmsnorm",
            "relu", "gelu", "silu", "dropout",
        )

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        import flag_gems

        if op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.matmul(A, B)

            return fn

        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.nn.functional.softmax(X, dim=-1)

            return fn

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.ones(N, device="cuda", dtype=dtype)
            bias = torch.zeros(N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.nn.functional.layer_norm(X, [N], weight, bias)

            return fn

        elif op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.relu(X)

            return fn

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.nn.functional.gelu(X)

            return fn

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.nn.functional.silu(X)

            return fn

        elif op == "dropout":
            X = torch.randn(M, N, device="cuda", dtype=dtype)

            def fn() -> torch.Tensor:
                with flag_gems.use_gems():
                    return torch.nn.functional.dropout(X, p=0.1, training=True)

            return fn

        return None
