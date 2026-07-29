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


_SUPPORTED_OPS = frozenset(
    {"flash_attention", "grouped_query_attention", "cross_attention"}
)

# cross_attention is encoder-decoder attention: every query position may attend
# to every key position, so it is NON-causal (unlike self-attention flash /
# GQA benchmarks which run causal). It also has Sq != Skv.
_NONCAUSAL_OPS = frozenset({"cross_attention"})


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

        # flash_attn_func expects (B, S, H, D) layout. For self-attention
        # (flash/GQA) q, k, v share seq length; for cross_attention q has
        # Sq while k/v have Skv — flash_attn_func handles Sq != Skv natively
        # (verified K-XATT 2026-07-29, max_abs_diff 1.2e-4 vs SDPA).
        if op in _SUPPORTED_OPS:
            if len(inputs) < 3:
                return None
            q, k, v = inputs[0], inputs[1], inputs[2]
            # Common Arke shape is (B, H, S, D); flash_attn wants (B, S, H, D).
            # Heuristic: if dim 1 looks like heads (small) and dim 2 looks like
            # seq (larger), transpose. Otherwise pass through. Applied per
            # tensor so cross_attention's Sq != Skv is preserved.
            if q.dim() == 4 and q.shape[1] < q.shape[2]:
                q = q.transpose(1, 2).contiguous()
            if k.dim() == 4 and k.shape[1] < k.shape[2]:
                k = k.transpose(1, 2).contiguous()
            if v.dim() == 4 and v.shape[1] < v.shape[2]:
                v = v.transpose(1, 2).contiguous()
            causal = bool(kwargs.get("causal", op not in _NONCAUSAL_OPS))
            if op in _NONCAUSAL_OPS:
                causal = False
            out = flash_attn_func(q, k, v, causal=causal)
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
            Skv = getattr(shape, "Skv", None) or S
        else:
            B, H, S, D, Hkv = 1, 12, M, N, 12
            Skv = S

        # cross_attention: encoder-decoder, non-causal, Sq (=S) != Skv.
        # Self-attention (flash/GQA): causal, Sq == Skv.
        is_cross = op in _NONCAUSAL_OPS
        causal = not is_cross
        Sq = S
        Skv_eff = Skv if is_cross else S

        # flash_attn_func uses (B, S, H, D). causal matches the benchmark
        # semantics of the matching golden: self-attention baselines call SDPA
        # is_causal=True; cross_attention is non-causal (fixed 2026-07-27 for
        # self-attn; cross_attention added K-XATT 2026-07-29).
        q = torch.randn(B, Sq, H, D, device="cuda", dtype=dtype)
        k = torch.randn(B, Skv_eff, Hkv, D, device="cuda", dtype=dtype)
        v = torch.randn(B, Skv_eff, Hkv, D, device="cuda", dtype=dtype)
        # Pre-warm once so first-call overhead doesn't pollute measurement.
        flash_attn_func(q, k, v, causal=causal)
        torch.cuda.synchronize()
        return lambda: flash_attn_func(q, k, v, causal=causal)
