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

# Baselines whose attention implementation materializes the full BxHxSxS
# score buffer (vs fused kernels that stream tile-wise softmax). Names match
# `BaselineRunner.name` strings as returned by get_all_runners(); kept
# case-insensitively for robustness. Anything not in this set is assumed
# fused — score buffer excluded from the peak estimate.
#
# Current Arke baseline roster (P0..P5):
#   P0 cuBLAS/cuDNN          — vendor; for attention uses cuDNN fused SDPA -> NOT materialized
#   P1 Liger-Kernel          — fused triton kernels                       -> NOT materialized
#   P1 FlagGems              — fused triton kernels                       -> NOT materialized
#   P2 Triton-Tutorial       — tutorial fused softmax+matmul              -> NOT materialized
#   P3 PyTorch-eager         — naive eager [B,H,S,S] scores               -> MATERIALIZED
#   P4 torch.compile         — Inductor fuses → flash-style               -> NOT materialized
#   P5 (flash_attn, FlashMLA, vLLM) — always fused                         -> NOT materialized
#   _torch_reference         — bench_l1 fallback reference                -> MATERIALIZED
SCORE_MATERIALIZING_BASELINES = {
    "pytorch-eager",
    "_torch_reference",
    "torch_reference",
    "torch_eager",
    "pt-eager",
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

# OT1 reduction ops whose preflight is dominated by the [M,N] input plus a
# workspace allocation. topk in particular materializes an fp32 copy of the
# input inside torch.topk's radix sort path; argmax/cumsum likewise spill
# fp32 workspace on large N. k_max here mirrors `_topk_k` in bench_l1 (min(4, N)).
TOPK_LIKE_OPS = {
    "topk",
    "argmax",
    "cumsum",
}
TOPK_K_MAX = 4


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
    materialize_scores: bool = True,
) -> int:
    """Estimate attention resident-tensor footprint.

    ``materialize_scores`` distinguishes baselines that allocate the full
    [B,H,S,S] score matrix (torch eager reference, naive implementations)
    from fused kernels (FlashAttention, Liger, FA3) that stream softmax
    tile-wise and never instantiate the score buffer. When False, the
    S*S term drops out entirely. ``score_factor`` is preserved for cases
    that want partial accounting (e.g. logsumexp scratch on flash paths,
    set to a small fraction).
    """
    qkv = batch * heads * seq * head_dim * dtype_bytes * qkv_factor
    scores = batch * heads * seq * seq * dtype_bytes * score_factor if materialize_scores else 0
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


def estimate_topk_like_bytes(
    *,
    m: int,
    n: int,
    k: int = TOPK_K_MAX,
    dtype_bytes: int = 2,
    workspace_dtype_bytes: int = 4,
) -> int:
    """Estimate topk/argmax/cumsum [M,N] resident + sort/scan workspace.

    Torch.topk on CUDA dispatches to a radix sort that allocates a workspace
    roughly the size of the input promoted to fp32. cumsum/argmax follow
    the same pattern at large N. This estimate captures:
      - input tensor (M*N*dtype_bytes)
      - output values+indices (M*k*(dtype_bytes+8))
      - sort/scan workspace (M*N*workspace_dtype_bytes)
    """
    inp = m * n * dtype_bytes
    out_vals = m * max(k, 1) * dtype_bytes
    out_idx = m * max(k, 1) * 8  # int64 indices
    workspace = m * n * workspace_dtype_bytes
    return inp + out_vals + out_idx + workspace


def attention_preflight(
    hw: HardwareInfo,
    *,
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    safety_ratio: float = 0.55,
    materialize_scores: bool = True,
) -> tuple[BenchmarkStatus, MemoryEstimate]:
    budget = _memory_budget(hw, safety_ratio)
    required = estimate_attention_bytes(
        batch=batch,
        heads=heads,
        seq=seq,
        head_dim=head_dim,
        materialize_scores=materialize_scores,
    )
    ratio = required / budget if budget > 0 else float("inf")
    estimate = MemoryEstimate(
        bytes_required=required,
        bytes_budget=budget,
        ratio=ratio,
        category="attention" if materialize_scores else "attention_fused",
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


def topk_like_preflight(
    hw: HardwareInfo,
    *,
    m: int,
    n: int,
    k: int = TOPK_K_MAX,
    safety_ratio: float = 0.55,
) -> tuple[BenchmarkStatus, MemoryEstimate]:
    budget = _memory_budget(hw, safety_ratio)
    required = estimate_topk_like_bytes(m=m, n=n, k=k)
    ratio = required / budget if budget > 0 else float("inf")
    estimate = MemoryEstimate(
        bytes_required=required,
        bytes_budget=budget,
        ratio=ratio,
        category="topk_like",
    )
    return _status_for_estimate(estimate, label="topk/argmax/cumsum"), estimate


def _baseline_materializes_scores(baseline: str | None) -> bool:
    if baseline is None:
        return True  # conservative default
    return baseline.strip().lower() in SCORE_MATERIALIZING_BASELINES


def maybe_memory_preflight(
    hw: HardwareInfo,
    op: str,
    shape,
    *,
    baseline: str | None = None,
) -> MemoryPreflight | None:
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
        materialize = _baseline_materializes_scores(baseline)
        status, estimate = attention_preflight(
            hw,
            batch=batch,
            heads=heads,
            seq=seq,
            head_dim=head_dim,
            materialize_scores=materialize,
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

    if op in TOPK_LIKE_OPS and hasattr(shape, "M") and hasattr(shape, "N"):
        m = int(getattr(shape, "M"))
        n = int(getattr(shape, "N"))
        if m <= 0 or n <= 0:
            return None
        status, estimate = topk_like_preflight(hw, m=m, n=n)
        return MemoryPreflight(status=status, estimate=estimate)

    return None


def maybe_attention_preflight(
    hw: HardwareInfo,
    op: str,
    shape,
    *,
    baseline: str | None = None,
) -> BenchmarkStatus | None:
    preflight = maybe_memory_preflight(hw, op, shape, baseline=baseline)
    if preflight is None or not preflight.estimate.category.startswith("attention"):
        return None
    return preflight.status
