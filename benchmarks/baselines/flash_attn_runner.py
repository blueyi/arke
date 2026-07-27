# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P2: flash-attn baseline (Dao-AILab).

Covers ``flash_attention`` and ``grouped_query_attention``. The package is
optional; ``available`` flips False on ImportError so the Golden Kernel
ladder falls through to cuDNN SDPA / PyTorch-eager.

Upstream: https://github.com/Dao-AILab/flash-attention
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    import flash_attn  # noqa: F401

    _AVAILABLE = True
except Exception:
    pass


_SUPPORTED_OPS = frozenset({"flash_attention", "grouped_query_attention"})


@register_baseline
class FlashAttnRunner(BaselineRunner):
    """P2: flash-attn — fused FlashAttention kernels."""

    @property
    def name(self) -> str:
        return "flash-attn"

    @property
    def priority(self) -> int:
        return 2

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            from importlib.metadata import version
            v = version("flash-attn")
        except Exception:
            pass
        return (
            f"flash-attn {v} (Dao-AILab) | "
            "https://github.com/Dao-AILab/flash-attention | License: BSD-3-Clause"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in _SUPPORTED_OPS

    def run_for_output(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if not self.available or not self.supports(op):
            return None

        from flash_attn import flash_attn_func  # type: ignore[import-not-found]

        # flash_attn_func expects (B, S, H, D) layout with q, k, v same shape
        # (or different num heads for GQA — flash_attn_func handles GQA when
        # num_heads_q != num_heads_kv as long as both share head_dim).
        if op in ("flash_attention", "grouped_query_attention"):
            if len(inputs) < 3:
                return None
            q, k, v = inputs[0], inputs[1], inputs[2]
            # Common Arke shape is (B, H, S, D); flash_attn wants (B, S, H, D).
            # Heuristic: if dim 1 looks like heads (small) and dim 2 looks like
            # seq (larger), transpose. Otherwise pass through.
            if q.dim() == 4 and q.shape[1] < q.shape[2]:
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
            out = flash_attn_func(q, k, v, causal=bool(kwargs.get("causal", False)))
            return out
        return None

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        if not self.available or not self.supports(op):
            return None
        from flash_attn import flash_attn_func  # type: ignore[import-not-found]
        from benchmarks.baselines._runtime_ctx import get_current_shape

        shape = get_current_shape()
        if shape is not None and hasattr(shape, "B"):
            B, H, S, D = shape.B, shape.H, shape.S, shape.D
            Hkv = getattr(shape, "Hkv", None) or H
        else:
            B, H, S, D, Hkv = 1, 12, M, N, 12

        # flash_attn_func uses (B, S, H, D). causal=True matches the benchmark
        # semantics used by every other attention baseline (pytorch_eager /
        # flaggems both call SDPA with is_causal=True) — was causal=False,
        # which measured a different computation and could never pass a
        # correctness probe against the causal golden (fixed 2026-07-27).
        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, S, Hkv, D, device="cuda", dtype=dtype)
        v = torch.randn(B, S, Hkv, D, device="cuda", dtype=dtype)
        # Pre-warm once so first-call overhead doesn't pollute measurement.
        flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        return lambda: flash_attn_func(q, k, v, causal=True)
