# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P1: Liger-Kernel *Triton fused* baselines for the OT3 fusion family.

This runner is the Same-Backend Triton denominator for the four L2 fusions
named in ``docs/benchmark/golden-kernel-ladder.md`` (Liger = P1):

    silu_and_mul                -> LigerSiLUMulFunction.apply(a, b)
    gelu_and_mul                -> LigerGELUMulFunction.apply(a, b)
    cross_entropy               -> LigerCrossEntropyFunction.apply(logits, target, weight)[0]
    fused_linear_cross_entropy  -> LigerFusedLinearCrossEntropyFunction.apply(X, W, target)[0]

Unlike ``benchmarks/baselines/liger.py`` (which exposes Liger's *unary*
activation / norm kernels), this module deliberately wires the genuine
*fused* Triton kernels so the L2 harness has a Triton-only fused denominator
(RFC §4, ``docs/benchmark/l2-fusion-measurement-protocol-rfc.md``).

All kernels are pure Triton (no cuBLAS / ATen dispatch), so every row this
runner emits is tagged ``backend="triton"``.

Import-guarded: if ``liger_kernel`` is absent the runner reports
``available == False`` and the harness skips it cleanly. ``GEMS_VENDOR`` and
FlagGems global dispatch state are left untouched.
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
    import liger_kernel  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


# Ops for which Liger ships a genuine Triton *fused* kernel.
_FUSED_OPS: frozenset[str] = frozenset(
    {
        "silu_and_mul",
        "gelu_and_mul",
        "cross_entropy",
        "fused_linear_cross_entropy",
    }
)


def _liger_version() -> str:
    try:
        from importlib.metadata import version

        return version("liger-kernel")
    except Exception:
        return "unknown"


@register_baseline
class LigerFusedRunner(BaselineRunner):
    """P1: Liger-Kernel *Triton fused* kernels for the OT3 fusion family."""

    @property
    def name(self) -> str:
        return "Liger-Kernel"

    @property
    def priority(self) -> int:
        return 1

    @property
    def backend(self) -> str:
        """Same-backend tag — every kernel here is pure Triton."""
        return "triton"

    @property
    def source(self) -> str:
        v = _liger_version()
        return (
            f"Liger {v} Triton fused | "
            "https://github.com/linkedin/Liger-Kernel | License: Apache-2.0"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in _FUSED_OPS

    # ── correctness-oriented execution ────────────────────────────────────
    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Run the Liger Triton fused kernel on caller-supplied inputs.

        Output conventions are reconciled to the SemanticInterpreter
        reference (``arke.ir.ops.interpreter.INTERPRETER.execute``):

          * silu_and_mul / gelu_and_mul: packed ``X[..., 2N]`` is split with
            ``chunk(2, dim=-1)`` (== reference ``[:half] / [half:]`` for even
            widths) and the gated product ``act(a) * b`` is returned (shape
            ``[..., N]``).
          * cross_entropy: ``(logits, labels)`` -> scalar mean loss (Liger
            returns a tuple; we take element [0]).
          * fused_linear_cross_entropy: ``(X, W, labels)`` -> scalar mean loss
            (tuple element [0]).
        """
        if not self.available:
            return None
        try:
            if op == "silu_and_mul" and len(inputs) >= 1:
                from liger_kernel.ops.swiglu import LigerSiLUMulFunction

                x = inputs[0]
                if x.shape[-1] % 2 != 0:
                    return None
                a, b = x.chunk(2, dim=-1)
                return LigerSiLUMulFunction.apply(a, b)

            if op == "gelu_and_mul" and len(inputs) >= 1:
                from liger_kernel.ops.geglu import LigerGELUMulFunction

                x = inputs[0]
                if x.shape[-1] % 2 != 0:
                    return None
                a, b = x.chunk(2, dim=-1)
                return LigerGELUMulFunction.apply(a, b)

            if op == "cross_entropy" and len(inputs) >= 2:
                from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

                logits, labels = inputs[0], inputs[1]
                # Liger 0.7.0 forward(ctx, _input, target, weight, ...).
                # weight=None == no class weighting == F.cross_entropy default,
                # matching ref_cross_entropy. reduction defaults to 'mean'.
                out = LigerCrossEntropyFunction.apply(logits, labels.long(), None)
                return out[0] if isinstance(out, tuple) else out

            if op == "fused_linear_cross_entropy" and len(inputs) >= 3:
                from liger_kernel.ops.fused_linear_cross_entropy import (
                    LigerFusedLinearCrossEntropyFunction,
                )

                x, w, labels = inputs[0], inputs[1], inputs[2]
                out = LigerFusedLinearCrossEntropyFunction.apply(x, w, labels.long())
                return out[0] if isinstance(out, tuple) else out
        except (RuntimeError, ValueError) as exc:
            logger.debug("LigerFused run_with_inputs %s failed: %s", op, exc)
            return None
        return None

    # ── zero-arg callable for perf timing ─────────────────────────────────
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

        try:
            if op in ("silu_and_mul", "gelu_and_mul"):
                if op == "silu_and_mul":
                    from liger_kernel.ops.swiglu import LigerSiLUMulFunction as _Fn
                else:
                    from liger_kernel.ops.geglu import LigerGELUMulFunction as _Fn
                # bench_l2 passes N = output feature width (ffn). Packed input
                # is X[seq, 2*ffn]; split into two ffn-wide halves.
                a = torch.randn(M, N, device="cuda", dtype=dtype)
                b = torch.randn(M, N, device="cuda", dtype=dtype)
                fn = lambda: _Fn.apply(a, b)  # noqa: E731

            elif op == "cross_entropy":
                from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

                V = max(N, 16)
                logits = torch.randn(M, V, device="cuda", dtype=dtype)
                labels = torch.randint(0, V, (M,), device="cuda", dtype=torch.int64)

                def fn() -> torch.Tensor:
                    out = LigerCrossEntropyFunction.apply(logits, labels, None)
                    return out[0] if isinstance(out, tuple) else out

            elif op == "fused_linear_cross_entropy":
                from liger_kernel.ops.fused_linear_cross_entropy import (
                    LigerFusedLinearCrossEntropyFunction,
                )

                V = max(N, 16)
                D = max(K, 16)
                X = torch.randn(M, D, device="cuda", dtype=dtype)
                W = torch.randn(V, D, device="cuda", dtype=dtype)
                labels = torch.randint(0, V, (M,), device="cuda", dtype=torch.int64)

                def fn() -> torch.Tensor:
                    out = LigerFusedLinearCrossEntropyFunction.apply(X, W, labels)
                    return out[0] if isinstance(out, tuple) else out

            else:
                return None

            # Pre-warm to trigger Triton compilation outside the measured loop.
            fn()
            torch.cuda.synchronize()
            return fn
        except (RuntimeError, ValueError) as exc:
            logger.debug("LigerFused get_fn %s (M=%s,N=%s,K=%s) failed: %s", op, M, N, K, exc)
            return None
