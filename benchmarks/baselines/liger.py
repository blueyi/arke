# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P1: Liger-Kernel baselines — LLM training Triton kernels."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    import liger_kernel  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


@register_baseline
class LigerRunner(BaselineRunner):
    """P1: Liger-Kernel Triton operators."""

    @property
    def name(self) -> str:
        return "Liger-Kernel"

    @property
    def priority(self) -> int:
        return 1

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            from importlib.metadata import version
            v = version("liger-kernel")
        except Exception:
            pass
        return (
            f"Liger-Kernel {v} (LinkedIn) | "
            "https://github.com/linkedin/Liger-Kernel | License: BSD-2-Clause"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in ("rmsnorm", "gelu", "silu", "rope")

    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if op == "rmsnorm" and len(inputs) == 1:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction

            x = inputs[0]
            weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            return LigerRMSNormFunction.apply(x, weight, 1e-6)

        if op == "gelu" and len(inputs) == 1:
            from liger_kernel.ops.geglu import LigerGELUMulFunction

            x = inputs[0]
            gate = torch.ones_like(x)
            return LigerGELUMulFunction.apply(x, gate)

        if op == "silu" and len(inputs) == 1:
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction

            x = inputs[0]
            gate = torch.ones_like(x)
            return LigerSiLUMulFunction.apply(x, gate)

        if op == "rope" and len(inputs) == 1:
            return None

        return None

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        if op == "rmsnorm":
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction

            X = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.ones(N, device="cuda", dtype=dtype)
            eps = 1e-6
            return lambda: LigerRMSNormFunction.apply(X, weight, eps)

        elif op == "gelu":
            from liger_kernel.ops.geglu import LigerGELUMulFunction

            # Liger's GELU is a fused GELU*gate, so we use a simple wrapper
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            gate = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: LigerGELUMulFunction.apply(X, gate)

        elif op == "rope":
            from liger_kernel.ops.rope import LigerRopeFunction

            # RoPE expects [batch, seq, heads, head_dim]
            batch = 1
            seq = M
            heads = 12
            head_dim = N
            X = torch.randn(
                batch, seq, heads, head_dim, device="cuda", dtype=dtype
            )
            cos = torch.randn(seq, head_dim, device="cuda", dtype=dtype)
            sin = torch.randn(seq, head_dim, device="cuda", dtype=dtype)
            return lambda: LigerRopeFunction.apply(X, cos, sin)

        return None
