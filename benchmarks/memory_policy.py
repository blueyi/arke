# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Memory-aware benchmark policy helpers for Stage 7.

These helpers do not weaken benchmark targets. They provide a consistent,
transparent preflight policy for shapes that are likely to exceed the current
GPU memory budget, so Track 6 artifacts can distinguish:
- executed cases
- proactively skipped cases
- true runtime OOM cases
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.hardware import HardwareInfo
from benchmarks.status import BenchmarkStatus


@dataclass(frozen=True)
class MemoryEstimate:
    bytes_required: int
    bytes_budget: int
    ratio: float


def estimate_attention_bytes(
    *,
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    dtype_bytes: int = 2,
    qkv_factor: int = 3,
    score_factor: int = 1,
) -> int:
    qkv = batch * heads * seq * head_dim * dtype_bytes * qkv_factor
    scores = batch * heads * seq * seq * dtype_bytes * score_factor
    output = batch * heads * seq * head_dim * dtype_bytes
    return qkv + scores + output


def attention_preflight(
    hw: HardwareInfo,
    *,
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    safety_ratio: float = 0.55,
) -> tuple[BenchmarkStatus, MemoryEstimate]:
    total_bytes = max(hw.gpu_memory_mb, 1) * 1024 * 1024
    budget = int(total_bytes * safety_ratio)
    required = estimate_attention_bytes(
        batch=batch,
        heads=heads,
        seq=seq,
        head_dim=head_dim,
    )
    ratio = required / budget if budget > 0 else float("inf")
    estimate = MemoryEstimate(
        bytes_required=required,
        bytes_budget=budget,
        ratio=ratio,
    )
    if required > budget:
        return (
            BenchmarkStatus(
                status="skipped",
                reason=(
                    f"memory preflight: estimated attention footprint {required} bytes "
                    f"> budget {budget} bytes for {hw.gpu_memory_mb}MB GPU"
                ),
                retryable=True,
            ),
            estimate,
        )
    return BenchmarkStatus(status="ok"), estimate
