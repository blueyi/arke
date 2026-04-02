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
_ENABLED = False
try:
    import flag_gems  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


def _ensure_enabled() -> None:
    """Enable FlagGems globally (once). Persistent, no cleanup needed."""
    global _ENABLED
    if _ENABLED:
        return
    import flag_gems

    flag_gems.enable()
    _ENABLED = True


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
    def source(self) -> str:
        v = "unknown"
        try:
            import flag_gems
            v = getattr(flag_gems, "__version__", "unknown")
        except Exception:
            pass
        return (
            f"FlagGems {v} (BAAI/FlagOS) | "
            "https://github.com/flagos-ai/FlagGems | License: Apache-2.0"
        )

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
        # FlagGems registers as ATen backend globally. Once enabled,
        # all torch ops dispatch through FlagGems Triton kernels.
        _ensure_enabled()

        if op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)
            # Pre-warm to trigger Triton compilation
            torch.matmul(A, B)
            torch.cuda.synchronize()
            return lambda: torch.matmul(A, B)

        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.softmax(X, dim=-1)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.softmax(X, dim=-1)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.ones(N, device="cuda", dtype=dtype)
            bias = torch.zeros(N, device="cuda", dtype=dtype)
            torch.nn.functional.layer_norm(X, [N], weight, bias)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.layer_norm(X, [N], weight, bias)

        elif op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.relu(X)
            torch.cuda.synchronize()
            return lambda: torch.relu(X)

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.gelu(X)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.gelu(X)

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.silu(X)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.silu(X)

        elif op == "dropout":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.dropout(X, p=0.1, training=True)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.dropout(X, p=0.1, training=True)

        return None
