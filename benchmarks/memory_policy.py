# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Memory-aware benchmark policy helpers for Stage 7.

These helpers do not weaken benchmark targets. They provide a consistent,
transparent preflight policy for shapes that are likely to exceed the current
GPU memory budget, so Track 6 artifacts can distinguish:
- executed cases
- proactively skipped cases
- true runtime OOM cases

The policy intentionally stays conservative. It records gate-readable evidence
for memory pressure while keeping the BL5 target matrix intact: a skipped row is
still a row for the required shape, with an explicit retryable reason and byte
estimate rather than a silent missing datapoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.hardware import HardwareInfo
from benchmarks.status import BenchmarkStatus


ATTENTION_FAMILY_OPS = {
    "flash_attention",
    "grouped_query_attention",
    "multi_latent_attention",
    "cross_attention",
    "paged_attention",
    "qkv_fa",
    "rope",
}

DENSE_FAMILY_OPS = {
    "matmul",
    "batch_matmul",
    "grouped_matmul",
    "matmul_relu",
    "matmul_gelu",
}

LINEAR_CE_OPS = {
    "linear_ce",
    "fused_linear_cross_entropy",
}


@dataclass(frozen=True)
class MemoryEstimate:
    bytes_required: int
    bytes_budget: int
    ratio: float
    category: str = "unknown"


@dataclass(frozen=True)
class MemoryPreflight:
    status: BenchmarkStatus
    estimate: MemoryEstimate

    def to_csv_fields(self) -> dict[str, str]:
        return {
            **self.status.to_csv_fields(),
            "memory_bytes_required": str(self.estimate.bytes_required),
            "memory_bytes_budget": str(self.estimate.bytes_budget),
            "memory_ratio": f"{self.estimate.ratio:.4f}",
            "memory_policy": self.estimate.category,
        }


def _memory_budget(hw: HardwareInfo, safety_ratio: float) -> int:
    total_bytes = max(hw.gpu_memory_mb, 1) * 1024 * 1024
    return int(total_bytes * safety_ratio)


def _status_for_estimate(estimate: MemoryEstimate, *, label: str) -> BenchmarkStatus:
    if estimate.bytes_required > estimate.bytes_budget:
        return BenchmarkStatus(
            status="skipped",
            reason=(
                f"memory preflight: estimated {label} footprint "
                f"{estimate.bytes_required} bytes > budget {estimate.bytes_budget} bytes"
            ),
            retryable=True,
        )
    return BenchmarkStatus(status="ok")


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


def estimate_dense_matmul_bytes(
    *,
    m: int,
    n: int,
    k: int,
    batch: int = 1,
    weight_copies: int = 1,
    dtype_bytes: int = 2,
    output_dtype_bytes: int | None = None,
    workspace_bytes: int = 0,
) -> int:
    """Estimate dense GEMM-family tensor footprint.

    This is a resident-tensor estimate, not a FLOP model. It covers inputs,
    output, and optional workspace. ``weight_copies`` lets grouped matmul
    account for multiple expert weight matrices while keeping activation tensors
    batched by request/token count.
    """
    output_bytes = dtype_bytes if output_dtype_bytes is None else output_dtype_bytes
    lhs = batch * m * k * dtype_bytes
    rhs = max(weight_copies, 1) * k * n * dtype_bytes
    out = batch * m * n * output_bytes
    return lhs + rhs + out + max(workspace_bytes, 0)


def estimate_linear_ce_bytes(
    *,
    tokens: int,
    hidden: int,
    vocab: int,
    dtype_bytes: int = 2,
    logits_dtype_bytes: int = 4,
) -> int:
    activations = tokens * hidden * dtype_bytes
    weights = vocab * hidden * dtype_bytes
    logits = tokens * vocab * logits_dtype_bytes
    labels = tokens * 8
    return activations + weights + logits + labels


def attention_preflight(
    hw: HardwareInfo,
    *,
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    safety_ratio: float = 0.55,
) -> tuple[BenchmarkStatus, MemoryEstimate]:
    budget = _memory_budget(hw, safety_ratio)
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
        category="attention",
    )
    return _status_for_estimate(estimate, label="attention"), estimate


def dense_matmul_preflight(
    hw: HardwareInfo,
    *,
    m: int,
    n: int,
    k: int,
    batch: int = 1,
    weight_copies: int = 1,
    safety_ratio: float = 0.55,
) -> tuple[BenchmarkStatus, MemoryEstimate]:
    budget = _memory_budget(hw, safety_ratio)
    required = estimate_dense_matmul_bytes(
        m=m,
        n=n,
        k=k,
        batch=batch,
        weight_copies=weight_copies,
    )
    ratio = required / budget if budget > 0 else float("inf")
    estimate = MemoryEstimate(
        bytes_required=required,
        bytes_budget=budget,
        ratio=ratio,
        category="dense_matmul",
    )
    return _status_for_estimate(estimate, label="dense matmul"), estimate


def linear_ce_preflight(
    hw: HardwareInfo,
    *,
    tokens: int,
    hidden: int,
    vocab: int,
    safety_ratio: float = 0.55,
) -> tuple[BenchmarkStatus, MemoryEstimate]:
    budget = _memory_budget(hw, safety_ratio)
    required = estimate_linear_ce_bytes(tokens=tokens, hidden=hidden, vocab=vocab)
    ratio = required / budget if budget > 0 else float("inf")
    estimate = MemoryEstimate(
        bytes_required=required,
        bytes_budget=budget,
        ratio=ratio,
        category="linear_ce",
    )
    return _status_for_estimate(estimate, label="linear+cross_entropy"), estimate


def maybe_memory_preflight(hw: HardwareInfo, op: str, shape) -> MemoryPreflight | None:
    op = op.strip().lower()

    if op in ATTENTION_FAMILY_OPS:
        if not (hasattr(shape, "B") and hasattr(shape, "H") and hasattr(shape, "S")):
            return None
        batch = int(getattr(shape, "B"))
        heads = int(getattr(shape, "H"))
        seq = int(getattr(shape, "S"))
        head_dim = int(getattr(shape, "D", 64))
        if batch <= 0 or heads <= 0 or seq <= 0 or head_dim <= 0:
            return None
        status, estimate = attention_preflight(
            hw,
            batch=batch,
            heads=heads,
            seq=seq,
            head_dim=head_dim,
        )
        return MemoryPreflight(status=status, estimate=estimate)

    if op in DENSE_FAMILY_OPS:
        if hasattr(shape, "E"):
            status, estimate = dense_matmul_preflight(
                hw,
                batch=int(getattr(shape, "B", 1)),
                weight_copies=int(getattr(shape, "E", 1)),
                m=int(getattr(shape, "M", 0)),
                n=int(getattr(shape, "N", 0)),
                k=int(getattr(shape, "K", 0)),
            )
            return MemoryPreflight(status=status, estimate=estimate)
        if hasattr(shape, "B") and not hasattr(shape, "H"):
            status, estimate = dense_matmul_preflight(
                hw,
                batch=int(getattr(shape, "B", 1)),
                m=int(getattr(shape, "M", 0)),
                n=int(getattr(shape, "N", 0)),
                k=int(getattr(shape, "K", 0)),
            )
            return MemoryPreflight(status=status, estimate=estimate)
        if all(hasattr(shape, attr) for attr in ("M", "N", "K")):
            status, estimate = dense_matmul_preflight(
                hw,
                m=int(getattr(shape, "M")),
                n=int(getattr(shape, "N")),
                k=int(getattr(shape, "K")),
            )
            return MemoryPreflight(status=status, estimate=estimate)

    if op in LINEAR_CE_OPS and hasattr(shape, "M") and hasattr(shape, "N"):
        tag = str(getattr(shape, "tag", ""))
        vocab = 50257 if "gpt2" in tag else 128256 if "llama3" in tag else 32000
        status, estimate = linear_ce_preflight(
            hw,
            tokens=int(getattr(shape, "M")),
            hidden=int(getattr(shape, "N")),
            vocab=vocab,
        )
        return MemoryPreflight(status=status, estimate=estimate)

    return None


def maybe_attention_preflight(hw: HardwareInfo, op: str, shape) -> BenchmarkStatus | None:
    preflight = maybe_memory_preflight(hw, op, shape)
    if preflight is None or preflight.estimate.category != "attention":
        return None
    return preflight.status
