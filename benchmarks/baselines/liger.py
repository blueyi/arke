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
        return op in (
            "rmsnorm", "rmsnorm_residual",
            "gelu", "silu", "rope",
            "silu_and_mul", "gelu_and_mul",
            "cross_entropy", "fused_linear_cross_entropy",
        )

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
            # Liger's GELU/SiLU Triton kernels block-tile the last
            # dimension; the recommended block size is 65536. Shapes with
            # last_dim > 65536 (e.g. extreme-flat 1×1048576, extreme-wide
            # softmax) exceed the limit and raise at launch. Return None
            # so the harness marks 'unsupported' with a typed reason; the
            # higher-priority Golden (cuBLAS/cuDNN gelu) remains
            # authoritative for those rows.
            if x.shape[-1] > 65536:
                return None
            try:
                gate = torch.ones_like(x)
                return LigerGELUMulFunction.apply(x, gate)
            except (RuntimeError, ValueError):
                return None

        if op == "silu" and len(inputs) == 1:
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction

            x = inputs[0]
            gate = torch.ones_like(x)
            return LigerSiLUMulFunction.apply(x, gate)

        if op == "rope":
            # rope is a known Golden-ladder gap: Liger is the P1 perf
            # winner, but its (Q_rot, K_rot) tuple output and internal
            # rotate convention diverge from the PyTorch-eager reference
            # (which only rotates Q and uses cat([-x2, x1])). Producing
            # a tensor here would cause a confusing 'mismatch' against
            # the reference. We deliberately return None so the harness
            # marks 'unsupported' with a typed reason and the gate's
            # Golden-protocol exemption can audit it correctly.
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

            # Liger's GELU is a fused GELU*gate, so we use a simple wrapper.
            # Mirror the run_with_inputs guard: Liger's Triton kernel caps
            # block size at 65536, so last_dim > 65536 (extreme-flat
            # 1×1048576) raises at launch. Decline here so the harness
            # records 'unsupported' instead of crashing in perf measure.
            if N > 65536:
                return None
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            gate = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: LigerGELUMulFunction.apply(X, gate)

        elif op == "rope":
            from liger_kernel.ops.rope import LigerRopeFunction

            # Liger 0.7.0 signature:
            #   forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
            # with q/k shape (bsz, n_head, seq_len, head_dim) and
            # cos/sin shape (1, seq_len, head_dim).
            from benchmarks.baselines._runtime_ctx import get_current_shape
            shape = get_current_shape()
            if shape is not None and hasattr(shape, "B"):
                batch = shape.B
                heads = shape.H
                seq = shape.S
                head_dim = shape.D
            else:
                batch = 1
                heads = 12
                seq = M
                head_dim = N
            Q = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype)
            K_ = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype)
            cos = torch.randn(1, seq, head_dim, device="cuda", dtype=dtype)
            sin = torch.randn(1, seq, head_dim, device="cuda", dtype=dtype)
            return lambda: LigerRopeFunction.apply(Q, K_, cos, sin)

        return None
