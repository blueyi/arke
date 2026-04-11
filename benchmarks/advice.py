# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Translate benchmark artifacts into strategy/agent-readable advice.

Stage 7 requires benchmark results to push back on Lang/IR/compiler/agent.
This module provides a small structured bridge from raw status rows to
recommended next actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Advice:
    kind: str
    severity: str
    message: str


def summarize_status_rows(rows: list[dict], gpu_memory_mb: int) -> list[Advice]:
    advice: list[Advice] = []
    skipped = [r for r in rows if r.get("status") == "skipped"]
    oom = [r for r in rows if r.get("status") == "oom"]
    long_ctx = [
        r for r in rows
        if any(token in str(r.get("shape_tag", "")) for token in ("8k", "16k", "32k", "long", "st4"))
    ]

    if skipped:
        advice.append(Advice(
            kind="memory-policy",
            severity="medium",
            message=(
                f"{len(skipped)} benchmark rows were proactively skipped on {gpu_memory_mb}MB GPU; "
                "consider conditional strategy, smaller tiles, paged KV, or chunked attention lowering."
            ),
        ))
    if oom:
        advice.append(Advice(
            kind="runtime-oom",
            severity="high",
            message=(
                f"{len(oom)} benchmark rows hit runtime OOM on {gpu_memory_mb}MB GPU; "
                "compiler/runtime should prefer memory-aware dispatch before execution."
            ),
        ))
    if long_ctx and (skipped or oom):
        advice.append(Advice(
            kind="long-context",
            severity="high",
            message=(
                "Long-context attention pressure is visible in benchmark artifacts; "
                "prioritize OT4 strategy branches keyed by sequence length and memory budget."
            ),
        ))
    return advice
