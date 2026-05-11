# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark status helpers for Stage 7 gate-readable artifacts.

These helpers standardize how benchmark runs record non-performance outcomes
such as OOM, skipped execution, or environment limitations. Stage 7 needs a
consistent artifact-level story for 6GB VRAM constraints instead of ad hoc
missing rows.
"""

from __future__ import annotations

from dataclasses import dataclass


TERMINAL_STATUSES = {
    "ok",
    "oom",
    "skipped",
    "unsupported",
    "error",
}


@dataclass(frozen=True)
class BenchmarkStatus:
    status: str
    reason: str = ""
    retryable: bool = False

    def to_csv_fields(self) -> dict[str, str]:
        return {
            "status": self.status,
            "reason": self.reason,
            "retryable": "true" if self.retryable else "false",
        }


def classify_exception(exc: Exception) -> BenchmarkStatus:
    message = str(exc).lower()
    if "out of memory" in message or "cuda oom" in message or "cuda out of memory" in message:
        return BenchmarkStatus(
            status="oom",
            reason=str(exc),
            retryable=True,
        )
    if isinstance(exc, NotImplementedError):
        # Typed declines from the runner / reference (e.g. RoPE odd-D
        # guard) — record as 'unsupported' so the gate audit can read
        # the typed reason rather than treating it as a generic crash.
        return BenchmarkStatus(
            status="unsupported",
            reason=str(exc),
            retryable=False,
        )
    return BenchmarkStatus(
        status="error",
        reason=str(exc),
        retryable=False,
    )


def normalize_status(status: str, reason: str = "", retryable: bool = False) -> BenchmarkStatus:
    status = status.strip().lower()
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Unsupported benchmark status: {status}")
    return BenchmarkStatus(status=status, reason=reason, retryable=retryable)
