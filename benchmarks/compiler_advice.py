# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bridge benchmark-derived memory advice into compiler/baseline execution.

This is intentionally minimal for Stage 7: it converts benchmark-visible shape
pressure into an execution decision that compiler/baseline code can consume
without changing benchmark goals.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.hardware import HardwareInfo
from benchmarks.memory_policy import maybe_attention_preflight


@dataclass(frozen=True)
class CompileAdvice:
    allow_compile: bool
    reason: str = ""
    strategy_hint: str = ""


def compile_advice_for_op(hw: HardwareInfo, op: str, shape) -> CompileAdvice:
    preflight = maybe_attention_preflight(hw, op, shape)
    if preflight is not None and preflight.status != "ok":
        return CompileAdvice(
            allow_compile=False,
            reason=preflight.reason,
            strategy_hint="prefer memory-aware dispatch: smaller tiles / paged-kv / chunked attention",
        )
    return CompileAdvice(allow_compile=True)
