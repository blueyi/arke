# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""L1 Single Operator Benchmark Runner.

Runs each operator across all shapes against all available baselines.

Usage:
    python -m benchmarks.bench_l1 --op matmul
    python -m benchmarks.bench_l1 --op matmul,softmax
    python -m benchmarks.bench_l1 --all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import benchmarks.baselines.arke_runner  # noqa: F401
import benchmarks.baselines.cublas  # noqa: F401
import benchmarks.baselines.flash_attn_runner  # noqa: F401
import benchmarks.baselines.flash_mla_runner  # noqa: F401
import benchmarks.baselines.inductor  # noqa: F401
import benchmarks.baselines.liger  # noqa: F401
import benchmarks.baselines.pytorch_eager  # noqa: F401
import benchmarks.baselines.triton_tutorial  # noqa: F401
import benchmarks.baselines.vllm_paged_runner  # noqa: F401
from benchmarks.baselines.base import get_all_runners, get_runners_for_op
from benchmarks.golden_ladder import (
    GoldenUnavailable,
    golden_runner_for,
    parse_inline_overrides,
    parse_overrides_file,
)
from benchmarks.baselines._runtime_ctx import (
    clear_current_shape,
    set_current_shape,
)
from benchmarks.artifacts import merge_perf_all, write_perf_csv_from_l1, write_summary
from benchmarks.hardware import collect_hardware_info
from benchmarks.measure import BenchResult, bench_fn, compute_matmul_tflops
from benchmarks.memory_policy import maybe_memory_preflight
from benchmarks.status import classify_exception
from benchmarks.shapes import (
    MatmulShape,
    Shape2D,
    get_shapes,
)
from benchmarks import progress as _progress

logger = logging.getLogger(__name__)


def _maybe_register_optional_baselines() -> None:
    try:
        import benchmarks.baselines.flaggems  # noqa: F401
    except Exception as e:
        logger.info(f"Optional baseline FlagGems unavailable during import: {e}")


_maybe_register_optional_baselines()

from benchmarks.op_registry import ALL_OPS as _REGISTRY_OPS

# Use op_registry as the canonical source; fallback to hardcoded if unavailable
try:
    ALL_OPS = list(_REGISTRY_OPS)
except Exception:
    ALL_OPS = ["matmul", "softmax", "layernorm", "gelu", "relu", "silu"]


@dataclass
class OpResult:
    """Result of one op × one shape × one baseline."""

    op: str
    shape_tag: str
    M: int
    N: int
    K: int
    baseline: str
    priority: int
    source: str
    latency_us: float
    latency_min_us: float
    tflops: float | None = None
    status: str = "ok"
    reason: str = ""
    retryable: bool = False
    allclose: bool | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    rtol: float | None = None
    atol: float | None = None
    correctness_status: str = "unknown"
    correctness_reason: str = ""
    golden_runner: str = ""
    golden_priority: int | None = None
    memory_bytes_required: int | None = None
    memory_bytes_budget: int | None = None
    memory_ratio: float | None = None
    memory_policy: str = ""


def _get_shapes(
    op: str, tier: int | None = None
) -> list:
    try:
        shapes = get_shapes(op, tier=tier)
        if not shapes and tier is not None:
            # Fallback: try all tiers if requested tier has no shapes
            shapes = get_shapes(op)
            if shapes:
                logger.info(f"  No tier ≤{tier} shapes for {op}, using all {len(shapes)} shapes")
        return shapes
    except ValueError:
        return []


def _correctness_tolerances(op: str, dtype: torch.dtype = torch.float16) -> tuple[float, float]:
    if dtype == torch.float16:
        if op in {"softmax", "layernorm", "rmsnorm", "rmsnorm_residual"}:
            return 5e-3, 5e-3
        if op in {"matmul", "batch_matmul", "grouped_matmul", "cross_entropy", "fused_linear_cross_entropy", "swiglu_packed"}:
            return 1e-2, 1e-2
        return 1e-3, 1e-3
    return 1e-5, 1e-6


def _positive_dim(value: int, fallback: int = 1) -> int:
    return value if value > 0 else fallback


def _topk_k(inputs: tuple[torch.Tensor, ...]) -> int:
    return min(4, inputs[0].shape[-1])


def _make_l1_correctness_inputs(op: str, M: int, N: int, K: int, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, ...]:
    M = _positive_dim(M)
    N = _positive_dim(N)
    K = max(K, 0)
    if op == "matmul":
        K = _positive_dim(K)
        return (
            torch.randn(M, K, device="cuda", dtype=dtype),
            torch.randn(K, N, device="cuda", dtype=dtype),
        )
    if op == "batch_matmul":
        batch = max(K, 4)
        return (
            torch.randn(batch, M, N, device="cuda", dtype=dtype),
            torch.randn(batch, N, M, device="cuda", dtype=dtype),
        )
    if op == "grouped_matmul":
        num_groups = max(M // 4, 1)
        group_size = max(M // num_groups, 1)
        a_groups = torch.randn(num_groups, group_size, N, device="cuda", dtype=dtype)
        b_groups = torch.randn(num_groups, N, max(K, 1), device="cuda", dtype=dtype)
        return (a_groups, b_groups)
    if op in {"add", "mul", "rmsnorm_residual", "concat"}:
        return (
            torch.randn(M, N, device="cuda", dtype=dtype),
            torch.randn(M, N, device="cuda", dtype=dtype),
        )
    if op == "where_":
        mask = torch.randn(M, N, device="cuda", dtype=torch.float32) > 0
        return (
            mask,
            torch.randn(M, N, device="cuda", dtype=dtype),
            torch.randn(M, N, device="cuda", dtype=dtype),
        )
    if op == "gather":
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        idx = torch.randint(0, N, (M, N), device="cuda")
        return (x, idx)
    if op == "scatter":
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        idx = torch.stack([torch.randperm(N, device="cuda") for _ in range(M)], dim=0)
        src = torch.randn(M, N, device="cuda", dtype=dtype)
        return (x, idx, src)
    if op == "embedding":
        vocab_size = max(N, 16)
        seq_len = max(M, 1)
        emb_dim = max(K, 128)
        indices = torch.randint(0, vocab_size, (seq_len,), device="cuda")
        weight = torch.randn(vocab_size, emb_dim, device="cuda", dtype=dtype)
        return (indices, weight)
    if op in {"silu_and_mul", "gelu_and_mul"}:
        return (torch.randn(M, 2 * N, device="cuda", dtype=dtype),)
    if op == "swiglu_packed":
        K_eff = _positive_dim(K)
        return (
            torch.randn(M, 2 * K_eff, device="cuda", dtype=dtype),
            torch.randn(K_eff, N, device="cuda", dtype=dtype),
        )
    if op == "cross_entropy":
        logits = torch.randn(M, N, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, N, (M,), device="cuda")
        return (logits, labels)
    if op == "fused_linear_cross_entropy":
        hidden = max(N, 128)
        x = torch.randn(M, hidden, device="cuda", dtype=dtype)
        w = torch.randn(N, hidden, device="cuda", dtype=dtype)
        labels = torch.randint(0, N, (M,), device="cuda")
        return (x, w, labels)
    if op == "permute":
        dim2 = max(K, 64)
        return (torch.randn(M, N, dim2, device="cuda", dtype=dtype),)
    if op == "flash_attention":
        return (
            torch.randn(M, N, max(K, 64), device="cuda", dtype=dtype),
            torch.randn(M, N, max(K, 64), device="cuda", dtype=dtype),
            torch.randn(M, N, max(K, 64), device="cuda", dtype=dtype),
        )
    if op == "grouped_query_attention":
        head_dim = max(K, 64)
        num_kv_groups = max(M // 4, 1)
        return (
            torch.randn(M, N, head_dim, device="cuda", dtype=dtype),
            torch.randn(num_kv_groups, N, head_dim, device="cuda", dtype=dtype),
            torch.randn(num_kv_groups, N, head_dim, device="cuda", dtype=dtype),
        )
    if op == "rope":
        return (torch.randn(M, N, max(K, 64), device="cuda", dtype=dtype),)
    if op == "cross_attention":
        # cross_attention: Sq != Skv; pull both from the active shape if set.
        from benchmarks.baselines._runtime_ctx import get_current_shape
        shape = get_current_shape()
        if shape is not None and getattr(shape, "Skv", None) is not None:
            q_len = shape.S
            kv_len = shape.Skv
            head_dim = shape.D
            batch_heads = shape.B * shape.H
        else:
            batch_heads = M
            q_len = max(N // 2, 1)
            kv_len = N
            head_dim = max(K, 64)
        return (
            torch.randn(batch_heads, q_len, head_dim, device="cuda", dtype=dtype),
            torch.randn(batch_heads, kv_len, head_dim, device="cuda", dtype=dtype),
            torch.randn(batch_heads, kv_len, head_dim, device="cuda", dtype=dtype),
        )
    if op == "quantize_per_token":
        return (torch.randn(M, N, device="cuda", dtype=dtype),)
    if op == "dequantize_per_channel":
        x_q = torch.randint(-128, 128, (M, N), device="cuda", dtype=torch.int8)
        scale = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.01
        return (x_q, scale)
    if op == "rsqrt":
        return (torch.randn(M, N, device="cuda", dtype=dtype).abs() + 1e-6,)
    return (torch.randn(M, N, device="cuda", dtype=dtype),)


def _torch_reference(op: str, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """PyTorch-eager reference path (legacy, kept as final fall-back).

    Used both by the PyTorch-eager runner (which is P3 in the Golden Kernel
    ladder for ops with no production P0/P1/P2 kernel) and as the last-
    resort safety net for :func:`_resolve_golden_for_correctness` when the
    ladder's designated golden returns ``None``.
    """
    if op == "relu":
        return torch.nn.functional.relu(inputs[0])
    if op == "gelu":
        return torch.nn.functional.gelu(inputs[0])
    if op == "silu":
        return torch.nn.functional.silu(inputs[0])
    if op == "tanh":
        return torch.tanh(inputs[0])
    if op == "sigmoid":
        return torch.sigmoid(inputs[0])
    if op == "neg":
        return -inputs[0]
    if op == "exp":
        return torch.exp(inputs[0])
    if op == "rsqrt":
        return torch.rsqrt(inputs[0])
    if op == "softmax":
        return torch.nn.functional.softmax(inputs[0], dim=-1)
    if op == "layernorm":
        x = inputs[0]
        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        b = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b)
    if op == "rmsnorm":
        x = inputs[0]
        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        eps = 1e-6
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
    if op == "rmsnorm_residual":
        x, residual = inputs
        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        eps = 1e-6
        y = x + residual
        return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)) * w
    if op == "add" and len(inputs) == 2:
        return inputs[0] + inputs[1]
    if op == "mul" and len(inputs) == 2:
        return inputs[0] * inputs[1]
    if op == "where_" and len(inputs) == 3:
        return torch.where(inputs[0].bool(), inputs[1], inputs[2])
    if op == "cast":
        return inputs[0].to(torch.float32)
    if op == "reduce_sum":
        return torch.sum(inputs[0], dim=-1)
    if op == "reduce_max":
        return torch.max(inputs[0], dim=-1).values
    if op == "reduce_mean":
        return torch.mean(inputs[0], dim=-1)
    if op == "argmax":
        return torch.argmax(inputs[0], dim=-1)
    if op == "cumsum":
        return torch.cumsum(inputs[0], dim=-1)
    if op == "topk":
        return torch.topk(inputs[0], k=_topk_k(inputs), dim=-1).values
    if op == "matmul" and len(inputs) == 2:
        return torch.matmul(inputs[0], inputs[1])
    if op == "batch_matmul" and len(inputs) == 2:
        return torch.bmm(inputs[0], inputs[1])
    if op == "grouped_matmul" and len(inputs) == 2:
        a_groups, b_groups = inputs
        return torch.cat([a @ b for a, b in zip(a_groups, b_groups, strict=False)], dim=0)
    if op == "transpose":
        return inputs[0].T
    if op == "concat" and len(inputs) == 2:
        return torch.cat([inputs[0], inputs[1]], dim=-1)
    if op == "split":
        return torch.chunk(inputs[0], 2, dim=-1)
    if op == "gather" and len(inputs) == 2:
        return torch.gather(inputs[0], 1, inputs[1].long())
    if op == "scatter" and len(inputs) == 3:
        return torch.zeros_like(inputs[0]).scatter_(1, inputs[1].long(), inputs[2])
    if op == "embedding" and len(inputs) == 2:
        return torch.nn.functional.embedding(inputs[0].long(), inputs[1])
    if op == "permute":
        return inputs[0].permute(0, 2, 1)
    if op == "copy_":
        return inputs[0].clone()
    if op == "silu_and_mul":
        x1, x2 = inputs[0].chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2
    if op == "swiglu_packed" and len(inputs) == 2:
        x1, x2 = inputs[0].chunk(2, dim=-1)
        return (torch.nn.functional.silu(x1) * x2) @ inputs[1]
    if op == "gelu_and_mul":
        x1, x2 = inputs[0].chunk(2, dim=-1)
        return torch.nn.functional.gelu(x1) * x2
    if op == "cross_entropy" and len(inputs) == 2:
        return torch.nn.functional.cross_entropy(inputs[0].to(torch.float32), inputs[1].long())
    if op == "fused_linear_cross_entropy" and len(inputs) == 3:
        x, w, labels = inputs
        logits = x.to(torch.float32) @ w.to(torch.float32).T
        return torch.nn.functional.cross_entropy(logits, labels.long())
    if op == "quantize_per_token":
        x = inputs[0]
        scales = torch.amax(torch.abs(x), dim=1, keepdim=True)
        scales = torch.clamp(scales, min=1e-8)
        x_q = torch.round(x / scales * 127).to(torch.int8)
        return x_q, scales.squeeze(1)
    if op == "dequantize_per_channel" and len(inputs) == 2:
        return inputs[0].to(inputs[1].dtype) * inputs[1].unsqueeze(0)
    if op == "flash_attention" and len(inputs) == 3:
        return torch.nn.functional.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2], is_causal=True)
    if op == "grouped_query_attention" and len(inputs) == 3:
        q, k, v = inputs
        repeats = q.shape[0] // max(k.shape[0], 1)
        k_exp = k.repeat_interleave(repeats, dim=0)
        v_exp = v.repeat_interleave(repeats, dim=0)
        return torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
    if op == "rope" and len(inputs) == 1:
        x = inputs[0]
        head_dim = x.shape[-1]
        if head_dim % 2 != 0:
            # RoPE rotates pairs of channels — head_dim must be even.
            # Odd-D shapes (e.g. non-align-1 D=65, non-align-2 D=127)
            # are mathematically ill-defined for RoPE. Raise so the
            # caller treats this row as 'unsupported' rather than
            # crashing in cat([-x2, x1]) with a confusing message.
            # Mirror of the same guard in baselines/pytorch_eager.py.
            raise NotImplementedError(
                f"RoPE requires even head_dim; got {head_dim} (odd) "
                f"for shape {tuple(x.shape)} — mathematically ill-defined"
            )
        seq_len = x.shape[1]
        # Numerical-stability fix (2026-05-14, Q5a): compute sin/cos in fp32
        # then cast back. With fp16 + seq_len >= 2048, torch.arange in fp16
        # loses integer precision (max exactly-representable int in fp16 is
        # 2048; fp16_max ≈ 65504), so at seq=65536 freqs decays to NaN/inf
        # and the entire rope output becomes NaN. This mirrors how HF
        # Transformers / Liger-Kernel / FlashInfer all compute the rotary
        # frequencies (fp32 trig, fp16 hadamard product).
        freqs = torch.einsum(
            "i,j->ij",
            torch.arange(seq_len, device=x.device, dtype=torch.float32),
            1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)),
        )
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = torch.cos(emb).unsqueeze(0).to(x.dtype)
        sin_emb = torch.sin(emb).unsqueeze(0).to(x.dtype)
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_emb + rotated * sin_emb
    if op == "cross_attention" and len(inputs) == 3:
        return torch.nn.functional.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2])
    raise NotImplementedError(f"No correctness reference for L1 op: {op}")


# Module-level overrides set by main() / run_l1() so that the inner
# correctness loop doesn't need an extra argument threaded through every
# call site. Test harnesses can set this directly.
_GOLDEN_OVERRIDES: dict[str, str] = {}


def _eval_l1_reference(
    op: str,
    inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Correctness reference for op `op` on `inputs`.

    Consults the Golden Kernel ladder (:func:`golden_runner_for`) and asks
    the designated runner to produce the expected output. If the runner
    declines (returns ``None``) or no runner is available, falls through to
    the PyTorch-eager path (:func:`_torch_reference`) — that is the P3
    baseline in the ladder for ops without a production kernel.

    Returns the reference tensor (or tuple). Tests can monkey-patch this
    function to inject a synthetic oracle.
    """
    try:
        golden = golden_runner_for(op, overrides=_GOLDEN_OVERRIDES)
    except GoldenUnavailable:
        return _torch_reference(op, inputs)
    kwargs = {"k": _topk_k(inputs)} if op == "topk" else {}
    try:
        out = golden.run_for_output(op, *inputs, **kwargs)
    except Exception:
        out = None
    if out is None:
        return _torch_reference(op, inputs)
    return out


def _resolve_golden_for_correctness(
    op: str,
    inputs: tuple[torch.Tensor, ...],
    *,
    overrides: dict[str, str] | None = None,
    golden_preflight_failed: bool = False,
    golden_preflight_reason: str = "",
) -> tuple[torch.Tensor | tuple[torch.Tensor, ...] | None, str, int | None, str, str]:
    """Compute correctness reference + Golden Kernel metadata.

    Returns ``(output, golden_runner_name, golden_priority, audit_status,
    audit_reason)``.

    * The ``output`` tensor comes from :func:`_eval_l1_reference`, which
      itself consults the ladder. Patching ``_eval_l1_reference`` in tests
      thus controls the oracle source while still letting metadata flow.
    * ``audit_status`` is ``""`` when the golden ran cleanly, or
      ``"golden_unavailable_pending_baseline"`` when no ladder runner is
      available for ``op`` (output then comes from the legacy PyTorch
      reference so a row can still be emitted).
    * Per-op audit reasons (e.g. ``mla_golden_degraded=true``) are encoded
      inline in ``audit_reason``.
    """
    ov = overrides if overrides is not None else _GOLDEN_OVERRIDES
    audit_status = ""
    audit_reason = ""

    try:
        golden = golden_runner_for(op, overrides=ov)
        name, prio = golden.name, golden.priority
    except GoldenUnavailable as e:
        name, prio = "", None
        audit_status = "golden_unavailable_pending_baseline"
        audit_reason = e.reason
        golden = None

    # Q1c: if the caller pre-computed a golden-preflight verdict and it
    # failed (e.g. golden=PyTorch-eager rope @ extreme-long would need 277GB
    # of attention scratch on a 6GB GPU), drop the oracle and emit an
    # `golden_unavailable_pending_baseline` audit row. Computing the
    # preflight here would require reconstructing a Shape object from the
    # raw tensors; the harness main loop already has the right Shape in
    # hand, so we accept the verdict as a parameter instead.
    if golden is not None and golden_preflight_failed:
        audit_status = "golden_unavailable_pending_baseline"
        audit_reason = (
            f"golden runner {name!r} skipped by memory preflight"
            + (f": {golden_preflight_reason}" if golden_preflight_reason else "")
        )
        golden = None

    # Probe whether the golden actually services this shape — if it returns
    # None we annotate the row as a degraded fall-through.
    if golden is not None:
        kwargs = {"k": _topk_k(inputs)} if op == "topk" else {}
        try:
            probe = golden.run_for_output(op, *inputs, **kwargs)
        except Exception as e:
            probe = None
            audit_reason = f"golden runner {name!r} raised: {e!r}"
        if probe is None:
            if op == "multi_latent_attention" and name == "FlashMLA":
                audit_reason = "mla_golden_degraded=true"
            elif not audit_reason:
                audit_reason = (
                    f"golden_runner={name!r} returned None; "
                    f"used PyTorch-eager reference fallback"
                )

    try:
        out = _eval_l1_reference(op, inputs)
    except NotImplementedError as e:
        msg = str(e)
        # Distinguish two NotImplementedError sources:
        #   (a) "No correctness reference for L1 op: <op>" — true
        #       golden gap, the op has no reference impl at all → mark
        #       golden_unavailable so the audit picks it up.
        #   (b) Shape-specific opt-out (e.g. RoPE odd-D guard) — the op
        #       has a reference, but this catalog shape is mathematically
        #       incompatible. Mark unsupported and let the row carry the
        #       typed reason; counted as a real fail until a baseline-
        #       provided alternative exists.
        if msg.startswith("No correctness reference for L1 op"):
            return None, name, prio, (
                audit_status or "golden_unavailable_pending_baseline"
            ), (audit_reason or msg)
        return None, name, prio, "unsupported", msg

    return out, name, prio, audit_status, audit_reason


def _compare_l1_outputs(
    runner_name: str,
    op: str,
    ref: torch.Tensor | tuple[torch.Tensor, ...],
    cand: torch.Tensor | tuple[torch.Tensor, ...],
    rtol: float,
    atol: float,
) -> dict[str, object]:
    if isinstance(ref, tuple) and isinstance(cand, tuple):
        if len(ref) != len(cand):
            return {
                "allclose": False,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "rtol": rtol,
                "atol": atol,
                "correctness_status": "mismatch",
                "correctness_reason": f"tuple length mismatch: ref={len(ref)} cand={len(cand)}",
            }
        max_diff = 0.0
        mean_diffs: list[float] = []
        ok = True
        for ref_i, cand_i in zip(ref, cand, strict=False):
            if not isinstance(ref_i, torch.Tensor) or not isinstance(cand_i, torch.Tensor):
                return {
                    "allclose": None,
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "rtol": rtol,
                    "atol": atol,
                    "correctness_status": "unsupported",
                    "correctness_reason": f"runner {runner_name} returned non-tensor tuple correctness output for {op}",
                }
            ref32 = ref_i.detach().to(torch.float32)
            cand32 = cand_i.detach().to(torch.float32)
            if ref32.shape != cand32.shape:
                return {
                    "allclose": False,
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "rtol": rtol,
                    "atol": atol,
                    "correctness_status": "mismatch",
                    "correctness_reason": f"shape mismatch in tuple element: ref={tuple(ref32.shape)} cand={tuple(cand32.shape)}",
                }
            diff = (cand32 - ref32).abs()
            max_diff = max(max_diff, float(diff.max().item()) if diff.numel() else 0.0)
            mean_diffs.append(float(diff.mean().item()) if diff.numel() else 0.0)
            ok = ok and torch.allclose(cand32, ref32, rtol=rtol, atol=atol)
        return {
            "allclose": bool(ok),
            "max_abs_diff": max_diff,
            "mean_abs_diff": (sum(mean_diffs) / len(mean_diffs)) if mean_diffs else 0.0,
            "rtol": rtol,
            "atol": atol,
            "correctness_status": "ok" if ok else "mismatch",
            "correctness_reason": "",
        }
    if not isinstance(cand, torch.Tensor) or not isinstance(ref, torch.Tensor):
        return {
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": rtol,
            "atol": atol,
            "correctness_status": "unsupported",
            "correctness_reason": f"runner {runner_name} returned non-tensor correctness output for {op}",
        }
    ref32 = ref.detach().to(torch.float32)
    cand32 = cand.detach().to(torch.float32)
    if ref32.shape != cand32.shape:
        return {
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": rtol,
            "atol": atol,
            "correctness_status": "mismatch",
            "correctness_reason": f"shape mismatch: ref={tuple(ref32.shape)} cand={tuple(cand32.shape)}",
        }
    diff = (cand32 - ref32).abs()
    ok = torch.allclose(cand32, ref32, rtol=rtol, atol=atol)
    return {
        "allclose": bool(ok),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "rtol": rtol,
        "atol": atol,
        "correctness_status": "ok" if ok else "mismatch",
        "correctness_reason": "",
    }


def _measure_l1_correctness(
    runner,
    op: str,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
    *,
    golden_preflight_failed: bool = False,
    golden_preflight_reason: str = "",
) -> dict[str, object]:
    rtol, atol = _correctness_tolerances(op, dtype)
    try:
        inputs = _make_l1_correctness_inputs(op, M, N, K, dtype)
        ref, golden_name, golden_prio, audit_status, audit_reason = (
            _resolve_golden_for_correctness(
                op, inputs,
                golden_preflight_failed=golden_preflight_failed,
                golden_preflight_reason=golden_preflight_reason,
            )
        )
        if ref is None:
            return {
                "allclose": None,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "rtol": rtol,
                "atol": atol,
                "correctness_status": audit_status or "golden_unavailable_pending_baseline",
                "correctness_reason": audit_reason,
                "golden_runner": golden_name,
                "golden_priority": golden_prio,
            }
        kwargs = {"k": _topk_k(inputs)} if op == "topk" else {}
        cand = runner.run_with_inputs(op, *inputs, **kwargs)
        if cand is None:
            return {
                "allclose": None,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "rtol": rtol,
                "atol": atol,
                "correctness_status": "unsupported",
                "correctness_reason": f"runner {runner.name} does not implement run_with_inputs for {op}",
                "golden_runner": golden_name,
                "golden_priority": golden_prio,
            }
        cmp_out = _compare_l1_outputs(runner.name, op, ref, cand, rtol, atol)
        cmp_out["golden_runner"] = golden_name
        cmp_out["golden_priority"] = golden_prio
        if audit_status:
            cmp_out["correctness_status"] = audit_status
        if audit_reason and not cmp_out.get("correctness_reason"):
            cmp_out["correctness_reason"] = audit_reason
        return cmp_out
    except NotImplementedError as e:
        return {
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": rtol,
            "atol": atol,
            "correctness_status": "unsupported",
            "correctness_reason": str(e),
            "golden_runner": "",
            "golden_priority": None,
        }
    except Exception as e:
        return {
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": rtol,
            "atol": atol,
            "correctness_status": "error",
            "correctness_reason": str(e),
            "golden_runner": "",
            "golden_priority": None,
        }

def run_op(
    op: str,
    shapes: list[MatmulShape] | list[Shape2D] | None = None,
    warmup: int = 200,
    reps: int = 500,
    tier: int | None = None,
    shape_tags: list[str] | None = None,
) -> list[OpResult]:
    """Benchmark one operator across shapes and baselines."""
    if shapes is None:
        shapes = _get_shapes(op, tier=tier)

    if shape_tags:
        allowed = set(shape_tags)
        shapes = [s for s in shapes if getattr(s, "tag", None) in allowed]

    runners = get_runners_for_op(op)
    hw = collect_hardware_info()
    if not runners:
        logger.warning(f"No baselines available for op '{op}'")
        return []

    logger.info(
        f"Benchmarking {op}: {len(shapes)} shapes × "
        f"{len(runners)} baselines ({', '.join(r.name for r in runners)})"
    )

    results: list[OpResult] = []

    def _record(r: OpResult) -> None:
        results.append(r)
        if _RESULT_EMITTER is not None:
            _RESULT_EMITTER(r)

    # IMPORTANT: FlagGems.enable() globally replaces ATen dispatch.
    # Run ALL non-FlagGems baselines first (for all shapes),
    # then FlagGems last. This prevents FlagGems from polluting
    # cuBLAS/PyTorch measurements.
    non_fg_runners = [r for r in runners if r.name != "FlagGems"]
    fg_runners = [r for r in runners if r.name == "FlagGems"]

    for runner_group in [non_fg_runners, fg_runners]:
        for shape in shapes:
            # Generic shape extraction — works for all shape types
            tag = shape.tag
            M = getattr(shape, 'M', 0)
            N = getattr(shape, 'N', 0)
            K = getattr(shape, 'K', 0)
            # Attention shapes: map B*H→M, S→N, D→K
            if hasattr(shape, 'B') and hasattr(shape, 'H') and hasattr(shape, 'S'):
                M = shape.B * shape.H
                N = shape.S
                K = getattr(shape, 'D', 64)
                # cross_attention: K/V seq length (Skv) differs from query Sq (=S)
                # — allocations need the larger dim, so expose Skv via N
                skv = getattr(shape, 'Skv', None)
                if skv is not None and op == "cross_attention":
                    N = skv
            # BatchMatmul: use B as M
            elif hasattr(shape, 'B') and not hasattr(shape, 'H'):
                M = getattr(shape, 'B', M)
            # GatedShape: use M, N from shape
            elif hasattr(shape, 'M') and hasattr(shape, 'N'):
                pass  # M, N already set

            # Q1c: pre-compute golden preflight verdict for this (op, shape).
            # If the ladder-designated golden cannot run this shape on the host
            # memory budget (e.g. PyTorch-eager rope @ extreme-long → 277GB on
            # 6GB GPU), every Arke row for this shape must be marked
            # `golden_unavailable_pending_baseline` — no oracle = no allclose
            # check possible. Compute once here (golden choice is per-op,
            # independent of which runner is benchmarked).
            golden_preflight_failed = False
            golden_preflight_reason = ""
            try:
                _golden = golden_runner_for(op, overrides=_GOLDEN_OVERRIDES)
                _gpf = maybe_memory_preflight(hw, op, shape, baseline=_golden.name)
                if _gpf is not None and _gpf.status.status != "ok":
                    golden_preflight_failed = True
                    golden_preflight_reason = _gpf.status.reason
            except GoldenUnavailable:
                # No ladder runner bound — _resolve_golden_for_correctness
                # already marks audit_status accordingly via its own path.
                pass

            for runner in runner_group:
                # Expose the full shape to runners whose get_fn signature
                # is fixed at (op, M, N, K) but need extra dims (e.g.
                # cross_attention needs both Sq and Skv).
                set_current_shape(shape)
                # Resume short-circuit: replay cached row + skip work.
                key = (op, tag, runner.name)
                if key in _SKIP_KEYS:
                    cached = _SKIPPED_ROWS.get(key)
                    if cached is not None:
                        results.append(_l1_row_to_result(cached))
                        logger.info(
                            f"  {tag:15s} {runner.name:15s} "
                            f"[resume] cached status={cached.get('status', 'ok')}"
                        )
                        continue

                preflight = maybe_memory_preflight(hw, op, shape, baseline=runner.name)
                if preflight is not None and preflight.status.status != "ok":
                    _record(OpResult(
                        op=op,
                        shape_tag=tag,
                        M=M,
                        N=N,
                        K=K,
                        baseline=runner.name,
                        priority=runner.priority,
                        source=runner.source,
                        latency_us=float("inf"),
                        latency_min_us=float("inf"),
                        tflops=None,
                        status=preflight.status.status,
                        reason=preflight.status.reason,
                        retryable=preflight.status.retryable,
                        correctness_status="skipped",
                        correctness_reason=preflight.status.reason,
                        memory_bytes_required=preflight.estimate.bytes_required,
                        memory_bytes_budget=preflight.estimate.bytes_budget,
                        memory_ratio=preflight.estimate.ratio,
                        memory_policy=preflight.estimate.category,
                    ))
                    continue

                fn = runner.get_fn(op, M, N, K)
                if fn is None:
                    # Runner declined this (op, shape) — almost always a
                    # shape-specific opt-out (e.g. RoPE odd-D guard);
                    # the broad 'runner doesn't implement op at all' case
                    # is filtered upstream by runner.supports(op). Record
                    # an 'unsupported' row so resume can mark the cell as
                    # permanently declined (PERMANENT_FAILURE_STATUSES)
                    # and the audit reads a typed correctness_reason.
                    logger.info(
                        f"  {tag:15s} {runner.name:15s} unsupported (get_fn returned None)"
                    )
                    correctness = _measure_l1_correctness(
                        runner, op, M, N, K,
                        golden_preflight_failed=golden_preflight_failed,
                        golden_preflight_reason=golden_preflight_reason,
                    )
                    _record(OpResult(
                        op=op,
                        shape_tag=tag,
                        M=M,
                        N=N,
                        K=K,
                        baseline=runner.name,
                        priority=runner.priority,
                        source=runner.source,
                        latency_us=float("inf"),
                        latency_min_us=float("inf"),
                        tflops=None,
                        status="unsupported",
                        reason=f"{runner.name}.get_fn declined {op}@{tag}",
                        retryable=False,
                        allclose=correctness["allclose"],
                        max_abs_diff=correctness["max_abs_diff"],
                        mean_abs_diff=correctness["mean_abs_diff"],
                        rtol=correctness["rtol"],
                        atol=correctness["atol"],
                        correctness_status=correctness["correctness_status"] or "unsupported",
                        correctness_reason=correctness["correctness_reason"] or f"{runner.name}.get_fn declined {op}@{tag}",
                        golden_runner=correctness.get("golden_runner", "") or "",
                        golden_priority=correctness.get("golden_priority"),
                        memory_bytes_required=(preflight.estimate.bytes_required if preflight else None),
                        memory_bytes_budget=(preflight.estimate.bytes_budget if preflight else None),
                        memory_ratio=(preflight.estimate.ratio if preflight else None),
                        memory_policy=(preflight.estimate.category if preflight else ""),
                    ))
                    continue

                try:
                    bench_result: BenchResult = bench_fn(fn, warmup=warmup, reps=reps)

                    tflops = None
                    if op in ("matmul", "batch_matmul") and K > 0:
                        tflops = compute_matmul_tflops(
                            M, N, K, bench_result.latency_us
                        )

                    correctness = _measure_l1_correctness(
                        runner, op, M, N, K,
                        golden_preflight_failed=golden_preflight_failed,
                        golden_preflight_reason=golden_preflight_reason,
                    )
                    result = OpResult(
                        op=op,
                        shape_tag=tag,
                        M=M,
                        N=N,
                        K=K,
                        baseline=runner.name,
                        priority=runner.priority,
                        source=runner.source,
                        latency_us=bench_result.latency_us,
                        latency_min_us=bench_result.latency_min_us,
                        tflops=tflops,
                        allclose=correctness["allclose"],
                        max_abs_diff=correctness["max_abs_diff"],
                        mean_abs_diff=correctness["mean_abs_diff"],
                        rtol=correctness["rtol"],
                        atol=correctness["atol"],
                        correctness_status=correctness["correctness_status"],
                        correctness_reason=correctness["correctness_reason"],
                        golden_runner=correctness.get("golden_runner", "") or "",
                        golden_priority=correctness.get("golden_priority"),
                        memory_bytes_required=(preflight.estimate.bytes_required if preflight else None),
                        memory_bytes_budget=(preflight.estimate.bytes_budget if preflight else None),
                        memory_ratio=(preflight.estimate.ratio if preflight else None),
                        memory_policy=(preflight.estimate.category if preflight else ""),
                    )
                    _record(result)
                    tflops_str = f" {tflops:.2f} TFLOPS" if tflops else ""
                    logger.info(
                        f"  {tag:15s} {runner.name:15s} "
                        f"{bench_result.latency_us:8.1f} μs{tflops_str}"
                    )
                except Exception as e:
                    status = classify_exception(e)
                    logger.warning(f"  {tag} {runner.name}: FAILED ({e})")
                    # Typed unsupported (e.g. RoPE odd-D NotImplementedError)
                    # propagates to correctness_status so the row's
                    # gate-audit reason is consistent across perf+probe.
                    cstatus = "unsupported" if status.status == "unsupported" else "error"
                    _record(OpResult(
                        op=op,
                        shape_tag=tag,
                        M=M,
                        N=N,
                        K=K,
                        baseline=runner.name,
                        priority=runner.priority,
                        source=runner.source,
                        latency_us=float("inf"),
                        latency_min_us=float("inf"),
                        tflops=None,
                        status=status.status,
                        reason=status.reason,
                        retryable=status.retryable,
                        correctness_status=cstatus,
                        correctness_reason=status.reason,
                        memory_bytes_required=(preflight.estimate.bytes_required if preflight else None),
                        memory_bytes_budget=(preflight.estimate.bytes_budget if preflight else None),
                        memory_ratio=(preflight.estimate.ratio if preflight else None),
                        memory_policy=(preflight.estimate.category if preflight else ""),
                    ))

    return results


L1_FIELDNAMES = [
    "op", "shape_tag", "M", "N", "K", "baseline", "priority", "source",
    "latency_us", "latency_min_us", "tflops", "status", "reason", "retryable",
    "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
    "correctness_status", "correctness_reason",
    "golden_runner", "golden_priority",
    "memory_bytes_required", "memory_bytes_budget", "memory_ratio", "memory_policy",
]
L1_KEY_FIELDS = ("op", "shape_tag", "baseline")

# Resume hooks (installed by run_l1).
_SKIP_KEYS: set[tuple[str, str, str]] = set()
_SKIPPED_ROWS: dict[tuple[str, str, str], dict[str, str]] = {}
_RESULT_EMITTER = None  # type: ignore[assignment]


def _l1_row_from_result(r: OpResult) -> dict[str, str]:
    return {
        "op": r.op,
        "shape_tag": r.shape_tag,
        "M": r.M,
        "N": r.N,
        "K": r.K,
        "baseline": r.baseline,
        "priority": r.priority,
        "source": r.source,
        "latency_us": f"{r.latency_us:.1f}",
        "latency_min_us": f"{r.latency_min_us:.1f}",
        "tflops": f"{r.tflops:.3f}" if r.tflops else "",
        "status": r.status,
        "reason": r.reason,
        "retryable": "true" if r.retryable else "false",
        "allclose": "" if r.allclose is None else ("true" if r.allclose else "false"),
        "max_abs_diff": "" if r.max_abs_diff is None else f"{r.max_abs_diff:.6g}",
        "mean_abs_diff": "" if r.mean_abs_diff is None else f"{r.mean_abs_diff:.6g}",
        "rtol": "" if r.rtol is None else f"{r.rtol:.6g}",
        "atol": "" if r.atol is None else f"{r.atol:.6g}",
        "correctness_status": r.correctness_status,
        "correctness_reason": r.correctness_reason,
        "golden_runner": r.golden_runner,
        "golden_priority": "" if r.golden_priority is None else str(r.golden_priority),
        "memory_bytes_required": "" if r.memory_bytes_required is None else str(r.memory_bytes_required),
        "memory_bytes_budget": "" if r.memory_bytes_budget is None else str(r.memory_bytes_budget),
        "memory_ratio": "" if r.memory_ratio is None else f"{r.memory_ratio:.4f}",
        "memory_policy": r.memory_policy,
    }


def _l1_row_to_result(row: dict[str, str]) -> OpResult:
    """Reconstruct an OpResult from a previously persisted CSV row."""
    def _f(key: str, default: float = 0.0) -> float:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _opt_f(key: str) -> float | None:
        v = row.get(key, "")
        if v == "" or v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _opt_bool(key: str) -> bool | None:
        v = (row.get(key, "") or "").strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        return None

    def _opt_int(key: str) -> int | None:
        v = row.get(key, "")
        if v == "" or v is None:
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    return OpResult(
        op=row.get("op", ""),
        shape_tag=row.get("shape_tag", ""),
        M=_opt_int("M") or 0,
        N=_opt_int("N") or 0,
        K=_opt_int("K") or 0,
        baseline=row.get("baseline", ""),
        priority=_opt_int("priority") or 0,
        source=row.get("source", ""),
        latency_us=_f("latency_us", float("inf")),
        latency_min_us=_f("latency_min_us", float("inf")),
        tflops=_opt_f("tflops"),
        status=row.get("status", "") or "ok",
        reason=row.get("reason", ""),
        retryable=(row.get("retryable", "").strip().lower() == "true"),
        allclose=_opt_bool("allclose"),
        max_abs_diff=_opt_f("max_abs_diff"),
        mean_abs_diff=_opt_f("mean_abs_diff"),
        rtol=_opt_f("rtol"),
        atol=_opt_f("atol"),
        correctness_status=row.get("correctness_status", "unknown"),
        correctness_reason=row.get("correctness_reason", ""),
        golden_runner=row.get("golden_runner", ""),
        golden_priority=_opt_int("golden_priority"),
        memory_bytes_required=_opt_int("memory_bytes_required"),
        memory_bytes_budget=_opt_int("memory_bytes_budget"),
        memory_ratio=_opt_f("memory_ratio"),
        memory_policy=row.get("memory_policy", ""),
    )


def save_results(
    results: list[OpResult],
    output_dir: Path,
    op: str,
) -> Path:
    """Save results as CSV (legacy whole-batch writer; resume path uses
    incremental append in run_l1 instead)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{op}_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=L1_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow(_l1_row_from_result(r))

    return csv_path


def print_comparison_table(results: list[OpResult], op: str) -> None:
    """Print a comparison table across baselines for each shape."""
    # Group by shape_tag
    shapes_seen: dict[str, dict[str, float]] = {}
    for r in results:
        if r.shape_tag not in shapes_seen:
            shapes_seen[r.shape_tag] = {}
        shapes_seen[r.shape_tag][r.baseline] = r.latency_us

    # Get all baseline names in priority order
    baselines = sorted(
        {r.baseline for r in results},
        key=lambda b: next(r.priority for r in results if r.baseline == b),
    )

    # Print header
    header = f"{'Shape':15s}"
    for b in baselines:
        header += f" {b:>15s}"
    print(f"\n{'=' * len(header)}")
    print(f"{op.upper()} Comparison (μs)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for tag, baseline_times in shapes_seen.items():
        row = f"{tag:15s}"
        # Find P0 time for ratio calculation
        p0_time = None
        for b in baselines:
            if b in baseline_times:
                t = baseline_times[b]
                if p0_time is None:
                    p0_time = t
                break

        for b in baselines:
            if b in baseline_times:
                t = baseline_times[b]
                if p0_time and p0_time > 0 and b != baselines[0]:
                    ratio = p0_time / t
                    row += f" {t:9.1f}({ratio:4.0%})"
                else:
                    row += f" {t:15.1f}"
            else:
                row += f" {'N/A':>15s}"
        print(row)

    print(f"{'=' * len(header)}")


def run_l1(
    ops: list[str],
    output_dir: str = "benchmarks/results",
    warmup: int = 200,
    reps: int = 500,
    tier: int | None = None,
    phase: int = 1,
    stage: int = 6,
    track: int = 1,
    shape_tags: list[str] | None = None,
    *,
    resume: bool = True,
    retry_policy: str = _progress.RETRY_POLICY_AUTO,
    force_restart: bool = False,
) -> dict[str, list[OpResult]]:
    """Run L1 benchmark suite with incremental persistence + resume."""
    global _SKIP_KEYS, _SKIPPED_ROWS, _RESULT_EMITTER

    # Strip duplicate phase/stage/track tail from output_dir.
    canonical_root = _progress.normalize_output_root(
        output_dir, phase=phase, stage=stage, track=track, layer="l1"
    )
    base_dir = canonical_root / f"phase{phase}" / f"stage{stage}" / f"track{track}" / "l1"
    base_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "ops": ops,
        "warmup": warmup,
        "reps": reps,
        "tier": tier,
        "shape_tags": shape_tags,
        "phase": phase,
        "stage": stage,
        "track": track,
        "layer": "L1",
    }
    config_check = _progress.validate_config(
        base_dir, config, force=force_restart
    )
    if not config_check.compatible:
        raise RuntimeError(
            f"L1 resume aborted at {base_dir}: {config_check.reason} "
            f"(stored={config_check.stored_fingerprint}, "
            f"current={config_check.current_fingerprint})"
        )

    _progress.acquire_lock(base_dir, layer="l1", force=force_restart)
    tracker = _progress.ProgressTracker(
        base_dir=base_dir,
        layer="l1",
        config_fingerprint=config_check.current_fingerprint,
    )

    try:
        # Save hardware info
        hw = collect_hardware_info()
        hw.save(str(base_dir / "hardware.json"))

        # Save baseline sources manifest
        all_runners = get_all_runners()
        sources_manifest = {
            r.name: {
                "priority": f"P{r.priority}",
                "source": r.source,
            }
            for r in all_runners
        }
        with open(base_dir / "sources.json", "w") as f:
            json.dump(sources_manifest, f, indent=2)

        with open(base_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        tracker.emit(
            "run_start", ops=list(ops), tier=tier,
            resume=resume, retry_policy=retry_policy,
        )

        all_results: dict[str, list[OpResult]] = {}

        # Print baseline sources
        logger.info("Baseline Sources:")
        for r in get_all_runners():
            logger.info(f"  P{r.priority} {r.name}: {r.source}")
        logger.info("")

        for op in ops:
            logger.info(f"\n{'='*60}")
            logger.info(f"L1 Benchmark: {op}")
            logger.info(f"{'='*60}")

            csv_path = base_dir / f"{op}_results.csv"
            existing_rows = (
                _progress.load_existing_rows(csv_path) if resume else []
            )
            existing_index = _progress.index_rows(existing_rows, L1_KEY_FIELDS)

            skip_keys: set[tuple[str, str, str]] = set()
            cached_rows: dict[tuple[str, str, str], dict[str, str]] = {}
            for key, row in existing_index.items():
                if _progress.should_skip(row, retry_policy):
                    skip_keys.add(key)
                    cached_rows[key] = row

            if resume and skip_keys:
                logger.info(
                    f"  resume: skipping {len(skip_keys)} already-recorded test points "
                    f"(policy={retry_policy})"
                )
                tracker.emit(
                    "resume_skip",
                    op=op,
                    skipped=len(skip_keys),
                    total_existing=len(existing_index),
                )

            kept_rows = [
                row for key, row in existing_index.items() if key in skip_keys
            ]
            # Bug fix (2026-05-16): when resume=False, we must TRUNCATE the
            # op-specific CSV to its header only — otherwise the prior run's
            # rows survive and the next run appends fresh rows on top, silently
            # double-counting (or worse, mixing stale + fresh measurements for
            # the same key when the new run's row count differs).
            #
            # Note: PERF_ALL.csv is intentionally NOT truncated here. PERF_ALL
            # is built by aggregating perf_<op>.csv across all ops (see
            # bench_l1.py:write_perf_all near end of run); per-op CSV truncation
            # propagates correctly through that pipeline.
            if kept_rows or not csv_path.exists() or not resume:
                tmp = csv_path.with_suffix(".csv.tmp")
                with tmp.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=L1_FIELDNAMES, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for row in kept_rows:
                        writer.writerow(
                            {k: row.get(k, "") for k in L1_FIELDNAMES}
                        )
                tmp.replace(csv_path)
            else:
                _progress.ensure_header(csv_path, L1_FIELDNAMES)

            _SKIP_KEYS = skip_keys
            _SKIPPED_ROWS = cached_rows
            new_count = {"value": 0}

            def _emit(result: OpResult, _csv=csv_path) -> None:
                key = (result.op, result.shape_tag, result.baseline)
                if key in _SKIP_KEYS:
                    return
                row = _l1_row_from_result(result)
                _progress.append_row(_csv, L1_FIELDNAMES, row)
                new_count["value"] += 1
                tracker.emit(
                    "measurement",
                    op=result.op,
                    shape_tag=result.shape_tag,
                    baseline=result.baseline,
                    status=result.status,
                    latency_us=result.latency_us,
                )

            _RESULT_EMITTER = _emit
            try:
                results = run_op(
                    op, warmup=warmup, reps=reps, tier=tier, shape_tags=shape_tags
                )
            finally:
                _RESULT_EMITTER = None
                _SKIP_KEYS = set()
                _SKIPPED_ROWS = {}

            all_results[op] = results

            perf_path = write_perf_csv_from_l1(
                csv_path, base_dir / f"perf_{op}.csv"
            )
            logger.info(f"  Saved: {csv_path}")
            logger.info(f"  Perf : {perf_path}")
            logger.info(
                f"  resume summary: {len(skip_keys)} skipped, "
                f"{new_count['value']} newly written, {len(results)} total in memory"
            )

            print_comparison_table(results, op)

            tracker.emit(
                "op_done",
                op=op,
                skipped=len(skip_keys),
                new=new_count["value"],
                total=len(results),
            )

        merge_perf_all(base_dir)
        write_summary(base_dir)

        per_op_summary = {
            op: _progress.summarize_csv(
                base_dir / f"{op}_results.csv", L1_KEY_FIELDS
            )
            for op in ops
        }
        tracker.snapshot({"per_op": per_op_summary})
        tracker.emit("run_done", per_op=per_op_summary)

        print(f"\nResults saved to: {base_dir}")
        return all_results
    finally:
        _progress.release_lock(base_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1 Single Operator Benchmark"
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help="Comma-separated operator names (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all operators",
    )
    parser.add_argument(
        "--warmup", type=int, default=200,
    )
    parser.add_argument(
        "--reps", type=int, default=500,
    )
    parser.add_argument(
        "--output", default="benchmarks/results",
    )
    parser.add_argument(
        "--shapes", type=str, default=None,
        help="Comma-separated shape tags to run",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Shape tier (1=fast, 2=standard, 3=full). Default: all shapes.",
    )
    parser.add_argument(
        "--phase", type=int, default=1,
        help="Phase number (default: 1)",
    )
    parser.add_argument(
        "--stage", type=int, default=6,
        help="Stage number (default: 6)",
    )
    parser.add_argument(
        "--track", type=str, default="g6",
        help="Gate name (default: g6)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resume. Truncates the per-op CSV "
            "(<op>_results.csv) to header before measuring so the new run's "
            "rows do not stack on top of stale ones. PERF_ALL.csv is rebuilt "
            "from per-op CSVs at end of run and is unaffected by this flag."
        ),
    )
    parser.add_argument(
        "--retry-policy",
        choices=list(_progress.RETRY_POLICIES),
        default=_progress.RETRY_POLICY_AUTO,
        help="Resume retry policy (default: auto = retry transient errors only)",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force resume despite config fingerprint changes / live lock.",
    )
    parser.add_argument(
        "--golden",
        action="append",
        default=None,
        metavar="op=runner_name",
        help=(
            "Pin a specific runner as Golden Kernel for an op (repeatable). "
            "E.g. --golden softmax=FlagGems --golden matmul=cuBLAS"
        ),
    )
    parser.add_argument(
        "--golden-file",
        type=str,
        default=None,
        help="YAML file mapping op→runner_name for Golden Kernel overrides.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.all:
        ops = ALL_OPS
    elif args.op:
        ops = [o.strip() for o in args.op.split(",")]
    else:
        ops = ["matmul", "softmax"]

    shape_tags = [s.strip() for s in args.shapes.split(",")] if args.shapes else None

    # Wire Golden Kernel overrides into the module-level dict consulted by
    # _eval_l1_reference. File entries are loaded first, then inline --golden
    # specs win on collision.
    global _GOLDEN_OVERRIDES
    overrides: dict[str, str] = {}
    overrides.update(parse_overrides_file(args.golden_file))
    overrides.update(parse_inline_overrides(args.golden))
    _GOLDEN_OVERRIDES = overrides
    if overrides:
        logger.info(f"Golden Kernel overrides: {overrides}")

    run_l1(
        ops=ops, output_dir=args.output, warmup=args.warmup, reps=args.reps,
        tier=args.tier, phase=args.phase, stage=args.stage, track=args.track,
        shape_tags=shape_tags,
        resume=not args.no_resume,
        retry_policy=args.retry_policy,
        force_restart=args.force_restart,
    )


if __name__ == "__main__":
    main()
