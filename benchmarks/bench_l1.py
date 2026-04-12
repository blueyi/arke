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
import benchmarks.baselines.inductor  # noqa: F401
import benchmarks.baselines.liger  # noqa: F401
import benchmarks.baselines.pytorch_eager  # noqa: F401
import benchmarks.baselines.triton_tutorial  # noqa: F401
from benchmarks.baselines.base import get_all_runners, get_runners_for_op
from benchmarks.artifacts import merge_perf_all, write_perf_csv_from_l1, write_summary
from benchmarks.hardware import collect_hardware_info
from benchmarks.measure import BenchResult, bench_fn, compute_matmul_tflops
from benchmarks.memory_policy import maybe_attention_preflight
from benchmarks.status import classify_exception
from benchmarks.shapes import (
    MatmulShape,
    Shape2D,
    get_shapes,
)

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
        if op in {"matmul", "batch_matmul", "grouped_matmul", "cross_entropy", "fused_linear_cross_entropy"}:
            return 1e-2, 1e-2
        return 1e-3, 1e-3
    return 1e-5, 1e-6


def _make_l1_correctness_inputs(op: str, M: int, N: int, K: int, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, ...]:
    if op == "matmul" and K > 0:
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
    if op in {"swiglu", "geglu"}:
        return (torch.randn(M, 2 * N, device="cuda", dtype=dtype),)
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
        q_len = max(N // 2, 1)
        head_dim = max(K, 64)
        return (
            torch.randn(M, q_len, head_dim, device="cuda", dtype=dtype),
            torch.randn(M, N, head_dim, device="cuda", dtype=dtype),
            torch.randn(M, N, head_dim, device="cuda", dtype=dtype),
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


def _eval_l1_reference(op: str, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
        k = min(4, inputs[0].shape[-1])
        return torch.topk(inputs[0], k=k, dim=-1).values
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
        split_size = max(inputs[0].shape[-1] // 2, 1)
        return torch.split(inputs[0], split_size, dim=-1)
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
    if op == "swiglu":
        x1, x2 = inputs[0].chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2
    if op == "geglu":
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
        seq_len = x.shape[1]
        freqs = torch.einsum(
            "i,j->ij",
            torch.arange(seq_len, device=x.device, dtype=x.dtype),
            1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)),
        )
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = torch.cos(emb).unsqueeze(0)
        sin_emb = torch.sin(emb).unsqueeze(0)
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_emb + rotated * sin_emb
    if op == "cross_attention" and len(inputs) == 3:
        return torch.nn.functional.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2])
    raise NotImplementedError(f"No correctness reference for L1 op: {op}")


def _measure_l1_correctness(runner, op: str, M: int, N: int, K: int, dtype: torch.dtype = torch.float16) -> dict[str, object]:
    rtol, atol = _correctness_tolerances(op, dtype)
    try:
        inputs = _make_l1_correctness_inputs(op, M, N, K, dtype)
        ref = _eval_l1_reference(op, inputs)
        cand = runner.run_with_inputs(op, *inputs)
        if cand is None:
            return {
                "allclose": None,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "rtol": rtol,
                "atol": atol,
                "correctness_status": "unsupported",
                "correctness_reason": f"runner {runner.name} does not implement run_with_inputs for {op}",
            }
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
                        "correctness_reason": f"runner {runner.name} returned non-tensor tuple correctness output for {op}",
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
                "correctness_reason": f"runner {runner.name} returned non-tensor correctness output for {op}",
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
    except NotImplementedError as e:
        return {
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": rtol,
            "atol": atol,
            "correctness_status": "unsupported",
            "correctness_reason": str(e),
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
            # BatchMatmul: use B as M
            elif hasattr(shape, 'B') and not hasattr(shape, 'H'):
                M = getattr(shape, 'B', M)
            # GatedShape: use M, N from shape
            elif hasattr(shape, 'M') and hasattr(shape, 'N'):
                pass  # M, N already set

            for runner in runner_group:
                preflight = maybe_attention_preflight(hw, op, shape)
                if preflight is not None and preflight.status != "ok":
                    results.append(OpResult(
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
                        status=preflight.status,
                        reason=preflight.reason,
                        retryable=preflight.retryable,
                    ))
                    continue

                fn = runner.get_fn(op, M, N, K)
                if fn is None:
                    logger.debug(
                        f"  {runner.name} does not support {op}@{tag}, skipping"
                    )
                    continue

                try:
                    bench_result: BenchResult = bench_fn(fn, warmup=warmup, reps=reps)

                    tflops = None
                    if op in ("matmul", "batch_matmul") and K > 0:
                        tflops = compute_matmul_tflops(
                            M, N, K, bench_result.latency_us
                        )

                    correctness = _measure_l1_correctness(runner, op, M, N, K)
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
                    )
                    results.append(result)
                    tflops_str = f" {tflops:.2f} TFLOPS" if tflops else ""
                    logger.info(
                        f"  {tag:15s} {runner.name:15s} "
                        f"{bench_result.latency_us:8.1f} μs{tflops_str}"
                    )
                except Exception as e:
                    status = classify_exception(e)
                    logger.warning(f"  {tag} {runner.name}: FAILED ({e})")
                    results.append(OpResult(
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
                    ))

    return results


def save_results(
    results: list[OpResult],
    output_dir: Path,
    op: str,
) -> Path:
    """Save results as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{op}_results.csv"

    fieldnames = [
        "op", "shape_tag", "M", "N", "K", "baseline", "priority", "source",
        "latency_us", "latency_min_us", "tflops", "status", "reason", "retryable",
        "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
        "correctness_status", "correctness_reason",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
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
            })

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
) -> dict[str, list[OpResult]]:
    """Run L1 benchmark suite."""
    # Use phase/stage/gate structure instead of timestamp
    base_dir = Path(output_dir) / f"phase{phase}" / f"stage{stage}" / f"track{track}" / "l1"
    base_dir.mkdir(parents=True, exist_ok=True)

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

    # Save config
    config = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "ops": ops,
        "warmup": warmup,
        "reps": reps,
        "tier": tier,
        "shape_tags": shape_tags,
    }
    with open(base_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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

        results = run_op(op, warmup=warmup, reps=reps, tier=tier, shape_tags=shape_tags)
        all_results[op] = results

        csv_path = save_results(results, base_dir, op)
        perf_path = write_perf_csv_from_l1(csv_path, base_dir / f"perf_{op}.csv")
        logger.info(f"  Saved: {csv_path}")
        logger.info(f"  Perf : {perf_path}")

        print_comparison_table(results, op)

    merge_perf_all(base_dir)
    write_summary(base_dir)

    # Summary
    print(f"\nResults saved to: {base_dir}")
    return all_results


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

    run_l1(
        ops=ops, output_dir=args.output, warmup=args.warmup, reps=args.reps,
        tier=args.tier, phase=args.phase, stage=args.stage, track=args.track,
        shape_tags=shape_tags,
    )


if __name__ == "__main__":
    main()
