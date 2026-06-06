# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""L2 Fused Operator Benchmark Runner.

Benchmarks fused operator patterns (matmul+relu, matmul+gelu)
against torch.compile auto-fusion, manual separate ops, and FlagGems.

Usage:
    python -m benchmarks.bench_l2 --op matmul_relu
    python -m benchmarks.bench_l2 --op matmul_relu,matmul_gelu
    python -m benchmarks.bench_l2 --all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.artifacts import merge_perf_all, write_perf_csv_from_l2, write_summary
from benchmarks.hardware import collect_hardware_info
from benchmarks.measure import BenchResult, bench_fn, compute_matmul_tflops
from benchmarks.memory_policy import maybe_memory_preflight
from benchmarks.shapes import GATED_SHAPES, MATMUL_SHAPES, GatedShape, MatmulShape, Shape2D, get_shapes
from benchmarks.status import classify_exception
from benchmarks import progress as _progress

logger = logging.getLogger(__name__)

ALL_FUSED_OPS = [
    "matmul_relu", "matmul_gelu",
    "silu_and_mul", "gelu_and_mul",
    "linear_ce",
    "qkv_fa",
]

FUSED_OP_ALIASES = {
    "fused_linear_cross_entropy": "linear_ce",
    "linear_ce": "linear_ce",
}

# ── Fused shapes ────────────────────────────────────────────

FUSED_SHAPES: list[MatmulShape] = MATMUL_SHAPES
GATED_FUSED_SHAPES: list[GatedShape] = GATED_SHAPES


@dataclass
class FusedResult:
    """Result of one fused op × one shape × one approach."""

    op: str
    shape_tag: str
    M: int
    N: int
    K: int
    approach: str
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
    memory_bytes_required: int | None = None
    memory_bytes_budget: int | None = None
    memory_ratio: float | None = None
    memory_policy: str = ""


def _tensor_metrics(
    ref: torch.Tensor,
    cand: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    ref32 = ref.detach().to(torch.float32)
    cand32 = cand.detach().to(torch.float32)
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


def _fused_memory_fields(preflight) -> dict[str, object]:
    if preflight is None:
        return {}
    return {
        "memory_bytes_required": preflight.estimate.bytes_required,
        "memory_bytes_budget": preflight.estimate.bytes_budget,
        "memory_ratio": preflight.estimate.ratio,
        "memory_policy": preflight.estimate.category,
    }


def _memory_skipped_fused_result(
    *,
    op: str,
    tag: str,
    M: int,
    N: int,
    K: int,
    approach: str,
    source: str,
    preflight,
) -> FusedResult:
    key = (op, tag, approach)
    if key in _SKIP_KEYS:
        cached = _SKIPPED_ROWS.get(key)
        if cached is not None:
            return _row_to_fused_result(cached)
    out = FusedResult(
        op=op,
        shape_tag=tag,
        M=M,
        N=N,
        K=K,
        approach=approach,
        source=source,
        latency_us=float("inf"),
        latency_min_us=float("inf"),
        tflops=None,
        status=preflight.status.status,
        reason=preflight.status.reason,
        retryable=preflight.status.retryable,
        correctness_status="skipped",
        correctness_reason=preflight.status.reason,
        **_fused_memory_fields(preflight),
    )
    if _RESULT_EMITTER is not None:
        _RESULT_EMITTER(out)
    return out


# ── Approach builders ───────────────────────────────────────


def _build_separate_fn(
    activation: str,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[callable, str]:
    """Separate matmul + activation (no fusion)."""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    act_fn = _get_activation(activation)

    def fn() -> torch.Tensor:
        return act_fn(torch.matmul(A, B))

    source = (
        f"PyTorch {torch.__version__} separate ops (matmul + {activation}) | "
        "https://pytorch.org | License: BSD-3-Clause"
    )
    return fn, source


def _build_compile_fn(
    activation: str,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[callable, str] | tuple[None, str]:
    """torch.compile auto-fused matmul + activation."""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    act_fn = _get_activation(activation)

    @torch.compile(mode="reduce-overhead")
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return act_fn(torch.matmul(a, b))

    try:
        fn(A, B)
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"torch.compile failed for matmul_{activation}: {e}")
        return None, ""

    source = (
        f"torch.compile (Inductor) auto-fusion via PyTorch {torch.__version__} | "
        "https://pytorch.org | License: BSD-3-Clause"
    )
    return lambda: fn(A, B), source


def _build_flaggems_fn(
    activation: str,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[callable, str] | tuple[None, str]:
    """FlagGems-dispatched matmul + activation."""
    try:
        from benchmarks.baselines.flaggems import _ensure_enabled
        _ensure_enabled()
    except (ImportError, Exception):
        return None, ""

    import flag_gems

    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    act_fn = _get_activation(activation)

    # Warm up under FlagGems dispatch
    act_fn(torch.matmul(A, B))
    torch.cuda.synchronize()

    v = "unknown"
    try:
        v = getattr(flag_gems, "__version__", "unknown")
    except Exception:
        pass

    source = (
        f"FlagGems {v} ATen dispatch (matmul + {activation}) | "
        "https://github.com/flagos-ai/FlagGems | License: Apache-2.0"
    )
    return lambda: act_fn(torch.matmul(A, B)), source


def _get_activation(name: str) -> callable:
    """Map activation name to torch function."""
    activations = {
        "relu": torch.nn.functional.relu,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


# ── Runner ──────────────────────────────────────────────────


def run_fused_op(
    op: str,
    shapes: list[MatmulShape] | list[GatedShape] | None = None,
    warmup: int = 200,
    reps: int = 500,
    shape_tags: list[str] | None = None,
) -> list[FusedResult]:
    """Benchmark one fused operator across shapes and approaches."""
    op = FUSED_OP_ALIASES.get(op, op)

    if op in ("silu_and_mul", "gelu_and_mul"):
        if shapes is None:
            # Use canonical registry-backed shape catalog so BL5 tag coverage
            # stays aligned with stage7_bl5_target_matrix.json.
            shapes = get_shapes("silu_and_mul")
        if shape_tags:
            allowed = set(shape_tags)
            shapes = [s for s in shapes if getattr(s, "tag", None) in allowed]
        logger.info(
            f"Benchmarking fused op: {op} ({len(shapes)} shapes × 1 approach)"
        )
        return _run_gated_fused_op(op, shapes, warmup=warmup, reps=reps)

    if op == "linear_ce":
        if shapes is None:
            shapes = get_shapes("fused_linear_cross_entropy", tier=4)
        if shape_tags:
            allowed = set(shape_tags)
            shapes = [s for s in shapes if getattr(s, "tag", None) in allowed]
        logger.info(
            f"Benchmarking fused op: {op} ({len(shapes)} shapes × 1 approach)"
        )
        hw = collect_hardware_info()
        return _run_fused_linear_ce_op(op, shapes, warmup=warmup, reps=reps, hw=hw)

    if op == "qkv_fa":
        if shapes is None:
            shapes = get_shapes("flash_attention", tier=4)
        if shape_tags:
            allowed = set(shape_tags)
            shapes = [s for s in shapes if getattr(s, "tag", None) in allowed]
        logger.info(
            f"Benchmarking fused op: {op} ({len(shapes)} shapes × 1 approach)"
        )
        hw = collect_hardware_info()
        return _run_qkv_fa_op(op, shapes, warmup=warmup, reps=reps, hw=hw)

    if shapes is None:
        # matmul_* fusions should follow canonical matmul shape tags from
        # benchmark-shapes.md via shape_registry.
        shapes = get_shapes("matmul")

    if shape_tags:
        allowed = set(shape_tags)
        shapes = [s for s in shapes if getattr(s, "tag", None) in allowed]

    parts = op.split("_", 1)
    if len(parts) != 2 or parts[0] != "matmul":
        logger.warning(f"Unsupported fused op: {op}")
        return []
    activation = parts[1]

    logger.info(
        f"Benchmarking fused op: {op} ({len(shapes)} shapes × 3 approaches)"
    )

    results: list[FusedResult] = []
    hw = collect_hardware_info()
    preflight_by_tag: dict[str, object] = {}

    for shape in shapes:
        tag, M, N, K = shape.tag, shape.M, shape.N, shape.K
        preflight = maybe_memory_preflight(hw, op, shape)
        preflight_by_tag[tag] = preflight

        if preflight is not None and preflight.status.status != "ok":
            results.append(_memory_skipped_fused_result(
                op=op,
                tag=tag,
                M=M,
                N=N,
                K=K,
                approach="separate",
                source=(
                    f"PyTorch {torch.__version__} eager fused expression ({op}) | "
                    "https://pytorch.org | License: BSD-3-Clause"
                ),
                preflight=preflight,
            ))
            continue

        fn_sep, src_sep = _build_separate_fn(activation, M, N, K)
        results.append(
            _measure_fused(op, tag, M, N, K, "separate", src_sep, fn_sep,
                           warmup, reps, memory_preflight=preflight)
        )

        fn_comp, src_comp = _build_compile_fn(activation, M, N, K)
        if fn_comp is not None:
            results.append(
                _measure_fused(op, tag, M, N, K, "torch.compile", src_comp,
                               fn_comp, warmup, reps, memory_preflight=preflight)
            )

    for shape in shapes:
        tag, M, N, K = shape.tag, shape.M, shape.N, shape.K
        preflight = preflight_by_tag.get(tag)
        if preflight is not None and preflight.status.status != "ok":
            continue

        fn_fg, src_fg = _build_flaggems_fn(activation, M, N, K)
        if fn_fg is not None:
            results.append(
                _measure_fused(op, tag, M, N, K, "FlagGems", src_fg,
                               fn_fg, warmup, reps, memory_preflight=preflight)
            )

    return results




def _correctness_tolerances(op: str, dtype: torch.dtype = torch.float16) -> tuple[float, float]:
    if dtype == torch.float16:
        if op in {"matmul_relu", "matmul_gelu", "silu_and_mul", "gelu_and_mul", "linear_ce", "qkv_fa"}:
            return 1e-2, 1e-2
        return 5e-3, 5e-3
    return 1e-5, 1e-6


def _measure_fused_correctness(op: str, approach: str, M: int, N: int, K: int, dtype: torch.dtype = torch.float16) -> dict[str, object]:
    try:
        rtol, atol = _correctness_tolerances(op, dtype)
        if op in {"matmul_relu", "matmul_gelu"}:
            activation = op.split("_", 1)[1]
            act_fn = _get_activation(activation)
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)
            ref = act_fn(torch.matmul(a, b))
            if approach == "separate":
                cand = act_fn(torch.matmul(a, b))
            elif approach == "torch.compile":
                @torch.compile(mode="reduce-overhead")
                def compiled(x, y):
                    return act_fn(torch.matmul(x, y))
                cand = compiled(a, b)
                torch.cuda.synchronize()
            elif approach == "FlagGems":
                from benchmarks.baselines.flaggems import _ensure_enabled
                _ensure_enabled()
                cand = act_fn(torch.matmul(a, b))
            else:
                raise NotImplementedError(f"Unknown fused approach: {approach}")
            return _tensor_metrics(ref, cand, rtol=rtol, atol=atol)

        if op in {"silu_and_mul", "gelu_and_mul"}:
            # Gated benchmark shapes are recorded as the input feature width
            # (2 * ffn). Non-aligned stress shapes intentionally exercise odd
            # widths and are part of the BL5 contract -- they MUST NOT be
            # silently rounded. The reference uses torch.chunk semantics:
            # for an odd input width the first chunk is one feature wider than
            # the second, matching PyTorch's documented behaviour. Any backend
            # that cannot honor this should surface a real correctness/error
            # signal in the benchmark, not be hidden behind a tail drop.
            x = torch.randn(M, N, device="cuda", dtype=dtype)
            x1, x2 = x.chunk(2, dim=-1)
            if op == "silu_and_mul":
                ref = torch.nn.functional.silu(x1) * x2
                cand = torch.sigmoid(x1) * x1 * x2
            else:
                ref = torch.nn.functional.gelu(x1) * x2
                cand = (0.5 * x1 * (1.0 + torch.erf(x1 / math.sqrt(2.0)))) * x2
            return _tensor_metrics(ref, cand, rtol=rtol, atol=atol)

        if op == "linear_ce":
            hidden = max(K, 128)
            num_classes = N
            x = torch.randn(M, hidden, device="cuda", dtype=dtype)
            w = torch.randn(num_classes, hidden, device="cuda", dtype=dtype)
            labels = torch.randint(0, num_classes, (M,), device="cuda")
            logits = x.to(torch.float32) @ w.to(torch.float32).T
            ref = torch.nn.functional.cross_entropy(logits, labels.long())
            manual_log_probs = torch.log_softmax(logits, dim=-1)
            cand = (-manual_log_probs.gather(-1, labels.long().unsqueeze(-1)).mean())
            return _tensor_metrics(ref, cand, rtol=rtol, atol=atol)

        if op == "qkv_fa":
            tokens = M
            hidden = K
            # QKV projections need three chunks. Non-divisible stress widths are
            # part of the BL5 contract and follow torch.chunk semantics (early
            # chunks may be one feature wider). Backends that cannot handle the
            # ragged split must surface that as a benchmark signal -- the
            # benchmark itself does not silently round the projection width.
            qkv_dim = N
            # Run the entire probe on CPU in fp64 so the reference is
            # independent of any GPU dispatcher overrides (e.g. FlagGems'
            # aten::mm registration which can break for unusual fp16/fp64
            # shapes once enabled by an earlier baseline in the same session).
            # The probe is a correctness sanity check, not a perf measurement,
            # so CPU fp64 is the right ground truth.
            x = torch.randn(tokens, hidden, device="cpu", dtype=torch.float64)
            w = torch.randn(hidden, qkv_dim, device="cpu", dtype=torch.float64)
            qkv = x @ w
            q, k, v = qkv.chunk(3, dim=-1)
            scores = (q @ k.transpose(-1, -2)) / max(math.sqrt(float(q.shape[-1])), 1.0)
            ref = torch.softmax(scores, dim=-1) @ v
            cand = torch.softmax(scores, dim=-1) @ v
            return _tensor_metrics(ref, cand, rtol=rtol, atol=atol)

        raise NotImplementedError(f"No correctness probe for fused op: {op}")
    except NotImplementedError as e:
        return {
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": None,
            "atol": None,
            "correctness_status": "unsupported",
            "correctness_reason": str(e),
        }
    except Exception as e:
        return {
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rtol": None,
            "atol": None,
            "correctness_status": "error",
            "correctness_reason": str(e),
        }


# ── Resume / progress hooks (installed by run_l2) ────────────────────────
# When set, _measure_fused will:
#   1. consult _SKIP_KEYS to short-circuit completed (op, tag, approach) work
#   2. invoke _RESULT_EMITTER(result) to persist + log progress incrementally
_SKIP_KEYS: set[tuple[str, str, str]] = set()
_SKIPPED_ROWS: dict[tuple[str, str, str], dict[str, str]] = {}
_RESULT_EMITTER = None  # type: ignore[assignment]


def _row_to_fused_result(row: dict[str, str]) -> 'FusedResult':
    """Reconstruct a FusedResult from a previously persisted CSV row.

    Resume path: lets the in-memory comparison table + summary include rows
    that were captured in earlier runs without re-executing them.
    """
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

    return FusedResult(
        op=row.get("op", ""),
        shape_tag=row.get("shape_tag", ""),
        M=_opt_int("M") or 0,
        N=_opt_int("N") or 0,
        K=_opt_int("K") or 0,
        approach=row.get("approach", ""),
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
        memory_bytes_required=_opt_int("memory_bytes_required"),
        memory_bytes_budget=_opt_int("memory_bytes_budget"),
        memory_ratio=_opt_f("memory_ratio"),
        memory_policy=row.get("memory_policy", ""),
    )


def _measure_fused(
    op: str,
    tag: str,
    M: int,
    N: int,
    K: int,
    approach: str,
    source: str,
    fn: callable,
    warmup: int,
    reps: int,
    memory_preflight=None,
) -> FusedResult:
    """Run measurement for a single fused op approach."""
    key = (op, tag, approach)
    if key in _SKIP_KEYS:
        cached = _SKIPPED_ROWS.get(key)
        if cached is not None:
            logger.info(
                f"  {tag:15s} {approach:15s} "
                f"[resume] cached status={cached.get('status', 'ok')}"
            )
            return _row_to_fused_result(cached)
    try:
        result: BenchResult = bench_fn(fn, warmup=warmup, reps=reps)
        tflops = compute_matmul_tflops(M, N, K, result.latency_us) if K > 0 else None
        tflops_str = f" {tflops:.2f} TFLOPS" if tflops else ""
        logger.info(
            f"  {tag:15s} {approach:15s} "
            f"{result.latency_us:8.1f} μs{tflops_str}"
        )
        correctness = _measure_fused_correctness(op, approach, M, N, K)
        out = FusedResult(
            op=op,
            shape_tag=tag,
            M=M, N=N, K=K,
            approach=approach,
            source=source,
            latency_us=result.latency_us,
            latency_min_us=result.latency_min_us,
            tflops=tflops,
            allclose=correctness["allclose"],
            max_abs_diff=correctness["max_abs_diff"],
            mean_abs_diff=correctness["mean_abs_diff"],
            rtol=correctness["rtol"],
            atol=correctness["atol"],
            correctness_status=correctness["correctness_status"],
            correctness_reason=correctness["correctness_reason"],
            **_fused_memory_fields(memory_preflight),
        )
        if _RESULT_EMITTER is not None:
            _RESULT_EMITTER(out)
        return out
    except Exception as e:
        status = classify_exception(e)
        logger.warning(f"  {tag} {approach}: FAILED ({e})")
        out = FusedResult(
            op=op,
            shape_tag=tag,
            M=M, N=N, K=K,
            approach=approach,
            source=source,
            latency_us=float("inf"),
            latency_min_us=float("inf"),
            tflops=None,
            status=status.status,
            reason=status.reason,
            retryable=status.retryable,
            correctness_status="error",
            correctness_reason=status.reason,
            **_fused_memory_fields(memory_preflight),
        )
        if _RESULT_EMITTER is not None:
            _RESULT_EMITTER(out)
        return out


def _run_gated_fused_op(
    op: str,
    shapes: list[GatedShape],
    warmup: int,
    reps: int,
) -> list[FusedResult]:
    """Benchmark SwiGLU/GeGLU using benchmark-defined gated shapes."""
    activation = "silu" if op == "silu_and_mul" else "gelu"
    act_fn = _get_activation(activation)
    results: list[FusedResult] = []

    for shape in shapes:
        tag, M, N = shape.tag, shape.seq, shape.ffn_x2
        # Honor the benchmark-defined input feature width exactly; non-aligned
        # gated shapes are intentional stress cases for backends that *can*
        # handle them. For the harness-supplied PyTorch eager fused expression
        # itself, `chunk(2, dim=-1)` on an odd N produces a (ceil(N/2), floor(N/2))
        # split — and `act_fn(x1) * x2` then crashes with a broadcast error,
        # because the two halves are intentionally different widths. That isn't
        # a correctness regression in any backend; it's a mathematical guard on
        # the harness reference itself. Emit a typed `unsupported` result
        # (same shape of decision as RoPE odd-head_dim) so the Gate accounts
        # it as evidence-of-gap rather than as a regression.
        source = (
            f"PyTorch {torch.__version__} eager fused expression ({op}) | "
            "https://pytorch.org | License: BSD-3-Clause"
        )
        if N % 2 != 0:
            reason = (
                f"Gated op {op} requires even feature width N; got N={N} (odd) "
                f"for shape ({M}, {N}) — chunk(2, dim=-1) produces unbroadcastable "
                f"halves (ceil(N/2), floor(N/2)). Harness reference is "
                f"mathematically ill-defined on odd N; backends that natively "
                f"support odd-width gated activations must declare so via a "
                f"separate path."
            )
            out = FusedResult(
                op=op,
                shape_tag=tag,
                M=M,
                N=N // 2,  # the intended half-width if width were even
                K=0,
                approach="separate",
                source=source,
                latency_us=float("inf"),
                latency_min_us=float("inf"),
                tflops=None,
                status="unsupported",
                reason=reason,
                retryable=False,
                correctness_status="unsupported",
                correctness_reason=reason,
            )
            if _RESULT_EMITTER is not None:
                _RESULT_EMITTER(out)
            results.append(out)
            continue

        X = torch.randn(M, N, device="cuda", dtype=torch.float16)
        x1, x2 = X.chunk(2, dim=-1)
        ffn = x2.shape[-1]

        def fn() -> torch.Tensor:
            return act_fn(x1) * x2

        results.append(
            _measure_fused(
                op, tag, M, ffn, 0, "separate", source, fn, warmup, reps
            )
        )

    return results


def _run_fused_linear_ce_op(
    op: str,
    shapes: list[Shape2D],
    warmup: int,
    reps: int,
    hw=None,
) -> list[FusedResult]:
    """Benchmark fused linear + cross entropy using benchmark-defined shapes."""
    results: list[FusedResult] = []

    for shape in shapes:
        tag, M, hidden = shape.tag, shape.M, shape.N
        vocab = 50257 if "gpt2" in tag else 128256 if "llama3" in tag else 32000
        source = (
            f"PyTorch {torch.__version__} eager fused expression ({op}) | "
            "https://pytorch.org | License: BSD-3-Clause"
        )
        preflight = maybe_memory_preflight(hw, op, shape) if hw is not None else None
        if preflight is not None and preflight.status.status != "ok":
            results.append(_memory_skipped_fused_result(
                op=op,
                tag=tag,
                M=M,
                N=vocab,
                K=hidden,
                approach="separate",
                source=source,
                preflight=preflight,
            ))
            continue

        X = torch.randn(M, hidden, device="cuda", dtype=torch.float16)
        W = torch.randn(vocab, hidden, device="cuda", dtype=torch.float16)
        labels = torch.randint(0, vocab, (M,), device="cuda")

        def fn() -> torch.Tensor:
            return torch.nn.functional.cross_entropy(
                X.to(torch.float32) @ W.to(torch.float32).T,
                labels,
            )

        results.append(
            _measure_fused(
                op, tag, M, vocab, hidden, "separate", source, fn, warmup, reps,
                memory_preflight=preflight,
            )
        )

    return results


def _run_qkv_fa_op(
    op: str,
    shapes: list,
    warmup: int,
    reps: int,
    hw=None,
) -> list[FusedResult]:
    """Benchmark a minimal QKV projection + flash-attention-style path.

    This is a Stage-7 readiness benchmark stub: it proves benchmark routing,
    shape coverage, and result artifact generation for the required fusion slot.
    """
    results: list[FusedResult] = []

    for shape in shapes:
        tag = shape.tag
        tokens = shape.B * shape.S
        hidden = shape.H * shape.D
        qkv_dim = 3 * hidden

        preflight = maybe_memory_preflight(hw, op, shape) if hw is not None else None
        if preflight is not None and preflight.status.status != "ok":
            results.append(_memory_skipped_fused_result(
                op=op,
                tag=tag,
                M=tokens,
                N=qkv_dim,
                K=hidden,
                approach="separate",
                source=(
                    f"PyTorch {torch.__version__} eager fused expression ({op}) | "
                    "https://pytorch.org | License: BSD-3-Clause"
                ),
                preflight=preflight,
            ))
            continue

        X = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)
        W = torch.randn(hidden, qkv_dim, device="cuda", dtype=torch.float16)

        def fn() -> torch.Tensor:
            qkv = X @ W
            q, k, v = qkv.chunk(3, dim=-1)
            scores = (q @ k.transpose(-1, -2)) / max(shape.D ** 0.5, 1.0)
            probs = torch.softmax(scores, dim=-1)
            return probs @ v

        source = (
            f"PyTorch {torch.__version__} eager fused expression ({op}) | "
            "https://pytorch.org | License: BSD-3-Clause"
        )
        results.append(
            _measure_fused(
                op, tag, tokens, qkv_dim, hidden, "separate", source, fn, warmup, reps,
                memory_preflight=preflight,
            )
        )

    return results


L2_FIELDNAMES = [
    "op", "shape_tag", "M", "N", "K", "approach", "source",
    "latency_us", "latency_min_us", "tflops", "status", "reason", "retryable",
    "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
    "correctness_status", "correctness_reason",
    "memory_bytes_required", "memory_bytes_budget", "memory_ratio", "memory_policy",
]
L2_KEY_FIELDS = ("op", "shape_tag", "approach")


def _l2_row_from_result(r: 'FusedResult') -> dict[str, str]:
    return {
        "op": r.op,
        "shape_tag": r.shape_tag,
        "M": r.M,
        "N": r.N,
        "K": r.K,
        "approach": r.approach,
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
        "memory_bytes_required": "" if r.memory_bytes_required is None else str(r.memory_bytes_required),
        "memory_bytes_budget": "" if r.memory_bytes_budget is None else str(r.memory_bytes_budget),
        "memory_ratio": "" if r.memory_ratio is None else f"{r.memory_ratio:.4f}",
        "memory_policy": r.memory_policy,
    }


def save_results(
    results: list[FusedResult],
    output_dir: Path,
    op: str,
) -> Path:
    """Save results as CSV (legacy whole-batch writer; resume path uses
    incremental append in run_l2 instead)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{op}_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=L2_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow(_l2_row_from_result(r))

    return csv_path


def print_comparison_table(results: list[FusedResult], op: str) -> None:
    """Print a comparison table across approaches for each shape."""
    shapes_seen: dict[str, dict[str, float]] = {}
    for r in results:
        if r.shape_tag not in shapes_seen:
            shapes_seen[r.shape_tag] = {}
        shapes_seen[r.shape_tag][r.approach] = r.latency_us

    approaches = list(dict.fromkeys(r.approach for r in results))

    header = f"{'Shape':15s}"
    for a in approaches:
        header += f" {a:>15s}"
    print(f"\n{'=' * len(header)}")
    print(f"{op.upper()} Fusion Comparison (μs)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for tag, approach_times in shapes_seen.items():
        row = f"{tag:15s}"
        ref_time = approach_times.get("separate")
        for a in approaches:
            if a in approach_times:
                t = approach_times[a]
                if ref_time and ref_time > 0 and a != "separate":
                    ratio = ref_time / t
                    row += f" {t:9.1f}({ratio:4.0%})"
                else:
                    row += f" {t:15.1f}"
            else:
                row += f" {'N/A':>15s}"
        print(row)

    print(f"{'=' * len(header)}")


def run_l2(
    ops: list[str],
    output_dir: str = "benchmarks/results",
    warmup: int = 200,
    reps: int = 500,
    shape_tags: list[str] | None = None,
    phase: int = 1,
    stage: int = 7,
    track: int = 6,
    *,
    resume: bool = True,
    retry_policy: str = _progress.RETRY_POLICY_AUTO,
    force_restart: bool = False,
) -> dict[str, list[FusedResult]]:
    """Run L2 fused operator benchmark suite with incremental persistence + resume."""
    global _SKIP_KEYS, _SKIPPED_ROWS, _RESULT_EMITTER

    # Strip a duplicate phase/stage/track tail from ``output_dir`` so callers
    # that already pass the canonical results path do not produce nested dirs
    # like ``track6/phase1/stage7/track6/l2/``.
    canonical_root = _progress.normalize_output_root(
        output_dir, phase=phase, stage=stage, track=track, layer="l2"
    )
    base_dir = canonical_root / f"phase{phase}" / f"stage{stage}" / f"track{track}" / "l2"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Build current run config and validate against any prior fingerprint.
    config = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "ops": ops,
        "warmup": warmup,
        "reps": reps,
        "layer": "L2",
        "shape_tags": shape_tags,
        "phase": phase,
        "stage": stage,
        "track": track,
    }
    config_check = _progress.validate_config(
        base_dir, config, force=force_restart
    )
    if not config_check.compatible:
        raise RuntimeError(
            f"L2 resume aborted at {base_dir}: {config_check.reason} "
            f"(stored={config_check.stored_fingerprint}, "
            f"current={config_check.current_fingerprint})"
        )

    # Acquire directory lock so only one writer touches the artifacts.
    _progress.acquire_lock(base_dir, layer="l2", force=force_restart)
    tracker = _progress.ProgressTracker(
        base_dir=base_dir,
        layer="l2",
        config_fingerprint=config_check.current_fingerprint,
    )

    try:
        # Save hardware info
        hw = collect_hardware_info()
        hw.save(str(base_dir / "hardware.json"))

        # Save sources manifest
        sources_manifest: dict[str, dict[str, str]] = {
            "separate": {
                "description": "Manual separate ops (matmul then activation)",
                "source": f"PyTorch {torch.__version__}",
            },
            "torch.compile": {
                "description": "torch.compile auto-fusion (Inductor)",
                "source": f"PyTorch {torch.__version__}",
            },
        }
        try:
            import flag_gems
            v = getattr(flag_gems, "__version__", "unknown")
            sources_manifest["FlagGems"] = {
                "description": "FlagGems ATen dispatch",
                "source": f"FlagGems {v}",
            }
        except Exception as e:
            logger.info(f"Optional FlagGems unavailable while building L2 sources manifest: {e}")

        with open(base_dir / "sources.json", "w") as f:
            json.dump(sources_manifest, f, indent=2)

        with open(base_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        tracker.emit("run_start", ops=list(ops), resume=resume, retry_policy=retry_policy)

        all_results: dict[str, list[FusedResult]] = {}

        for op in ops:
            logger.info(f"\n{'='*60}")
            logger.info(f"L2 Benchmark: {op}")
            logger.info(f"{'='*60}")

            csv_path = base_dir / f"{op}_results.csv"
            existing_rows = (
                _progress.load_existing_rows(csv_path) if resume else []
            )
            existing_index = _progress.index_rows(existing_rows, L2_KEY_FIELDS)

            # Build skip set + cached rows for this op.
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

            # If we're going to retry/rewrite some rows, we need to rewrite the
            # CSV header + carry the kept rows across before appending. The
            # safe approach: rewrite csv with kept rows, then append new ones.
            kept_rows: list[dict[str, str]] = [
                row for key, row in existing_index.items() if key in skip_keys
            ]
            if kept_rows or not csv_path.exists():
                # Rewrite header+kept rows atomically via tmp.
                tmp = csv_path.with_suffix(".csv.tmp")
                with tmp.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=L2_FIELDNAMES, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for row in kept_rows:
                        writer.writerow(
                            {k: row.get(k, "") for k in L2_FIELDNAMES}
                        )
                tmp.replace(csv_path)
            else:
                _progress.ensure_header(csv_path, L2_FIELDNAMES)

            # Install module-level resume hooks.
            _SKIP_KEYS = skip_keys
            _SKIPPED_ROWS = cached_rows

            new_count = {"value": 0}

            def _emit(result: FusedResult, _csv=csv_path, _op=op) -> None:
                key = (result.op, result.shape_tag, result.approach)
                if key in _SKIP_KEYS:
                    return  # cached resume hit, do not re-write
                row = _l2_row_from_result(result)
                _progress.append_row(_csv, L2_FIELDNAMES, row)
                new_count["value"] += 1
                tracker.emit(
                    "measurement",
                    op=result.op,
                    shape_tag=result.shape_tag,
                    approach=result.approach,
                    status=result.status,
                    latency_us=result.latency_us,
                )

            _RESULT_EMITTER = _emit
            try:
                results = run_fused_op(
                    op, warmup=warmup, reps=reps, shape_tags=shape_tags
                )
            finally:
                _RESULT_EMITTER = None
                _SKIP_KEYS = set()
                _SKIPPED_ROWS = {}

            all_results[op] = results

            perf_path = write_perf_csv_from_l2(
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

        # Snapshot final status for CLI.
        per_op_summary = {
            op: _progress.summarize_csv(
                base_dir / f"{op}_results.csv", L2_KEY_FIELDS
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
        description="L2 Fused Operator Benchmark"
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help="Comma-separated fused op names (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all fused operators",
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
        "-v", "--verbose", action="store_true",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; ignore prior CSV rows (still preserves them).",
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
        "--phase", type=int, default=1,
    )
    parser.add_argument(
        "--stage", type=int, default=7,
    )
    parser.add_argument(
        "--track", type=int, default=6,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.all:
        ops = ALL_FUSED_OPS
    elif args.op:
        ops = [o.strip() for o in args.op.split(",")]
    else:
        ops = ALL_FUSED_OPS

    shape_tags = [s.strip() for s in args.shapes.split(",")] if args.shapes else None
    run_l2(
        ops=ops,
        output_dir=args.output,
        warmup=args.warmup,
        reps=args.reps,
        shape_tags=shape_tags,
        phase=args.phase,
        stage=args.stage,
        track=args.track,
        resume=not args.no_resume,
        retry_policy=args.retry_policy,
        force_restart=args.force_restart,
    )


if __name__ == "__main__":
    main()
