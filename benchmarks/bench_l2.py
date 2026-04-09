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
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.hardware import collect_hardware_info
from benchmarks.measure import BenchResult, bench_fn, compute_matmul_tflops
from benchmarks.shapes import GATED_SHAPES, MATMUL_SHAPES, GatedShape, MatmulShape

logger = logging.getLogger(__name__)

ALL_FUSED_OPS = [
    "matmul_relu", "matmul_gelu",
    "swiglu", "geglu",
    # Future: rmsnorm_residual, fused_linear_cross_entropy
]

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
    if op in ("swiglu", "geglu"):
        if shapes is None:
            shapes = GATED_FUSED_SHAPES
        if shape_tags:
            allowed = set(shape_tags)
            shapes = [s for s in shapes if getattr(s, "tag", None) in allowed]
        logger.info(
            f"Benchmarking fused op: {op} ({len(shapes)} shapes × 1 approach)"
        )
        return _run_gated_fused_op(op, shapes, warmup=warmup, reps=reps)

    if shapes is None:
        shapes = FUSED_SHAPES

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

    for shape in shapes:
        tag, M, N, K = shape.tag, shape.M, shape.N, shape.K

        fn_sep, src_sep = _build_separate_fn(activation, M, N, K)
        results.append(
            _measure_fused(op, tag, M, N, K, "separate", src_sep, fn_sep,
                           warmup, reps)
        )

        fn_comp, src_comp = _build_compile_fn(activation, M, N, K)
        if fn_comp is not None:
            results.append(
                _measure_fused(op, tag, M, N, K, "torch.compile", src_comp,
                               fn_comp, warmup, reps)
            )

    for shape in shapes:
        tag, M, N, K = shape.tag, shape.M, shape.N, shape.K
        fn_fg, src_fg = _build_flaggems_fn(activation, M, N, K)
        if fn_fg is not None:
            results.append(
                _measure_fused(op, tag, M, N, K, "FlagGems", src_fg,
                               fn_fg, warmup, reps)
            )

    return results


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
) -> FusedResult:
    """Run measurement for a single fused op approach."""
    try:
        result: BenchResult = bench_fn(fn, warmup=warmup, reps=reps)
        tflops = compute_matmul_tflops(M, N, K, result.latency_us) if K > 0 else None
        tflops_str = f" {tflops:.2f} TFLOPS" if tflops else ""
        logger.info(
            f"  {tag:15s} {approach:15s} "
            f"{result.latency_us:8.1f} μs{tflops_str}"
        )
        return FusedResult(
            op=op,
            shape_tag=tag,
            M=M, N=N, K=K,
            approach=approach,
            source=source,
            latency_us=result.latency_us,
            latency_min_us=result.latency_min_us,
            tflops=tflops,
        )
    except Exception as e:
        logger.warning(f"  {tag} {approach}: FAILED ({e})")
        return FusedResult(
            op=op,
            shape_tag=tag,
            M=M, N=N, K=K,
            approach=approach,
            source=source,
            latency_us=float("inf"),
            latency_min_us=float("inf"),
            tflops=None,
        )


def _run_gated_fused_op(
    op: str,
    shapes: list[GatedShape],
    warmup: int,
    reps: int,
) -> list[FusedResult]:
    """Benchmark SwiGLU/GeGLU using benchmark-defined gated shapes."""
    activation = "silu" if op == "swiglu" else "gelu"
    act_fn = _get_activation(activation)
    results: list[FusedResult] = []

    for shape in shapes:
        tag, M, N = shape.tag, shape.seq, shape.ffn_x2
        X = torch.randn(M, N, device="cuda", dtype=torch.float16)
        x1, x2 = X.chunk(2, dim=-1)

        def fn() -> torch.Tensor:
            return act_fn(x1) * x2

        source = (
            f"PyTorch {torch.__version__} eager fused expression ({op}) | "
            "https://pytorch.org | License: BSD-3-Clause"
        )
        results.append(
            _measure_fused(
                op, tag, M, N // 2, 0, "separate", source, fn, warmup, reps
            )
        )

    return results


def save_results(
    results: list[FusedResult],
    output_dir: Path,
    op: str,
) -> Path:
    """Save results as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{op}_results.csv"

    fieldnames = [
        "op", "shape_tag", "M", "N", "K", "approach", "source",
        "latency_us", "latency_min_us", "tflops",
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
                "approach": r.approach,
                "source": r.source,
                "latency_us": f"{r.latency_us:.1f}",
                "latency_min_us": f"{r.latency_min_us:.1f}",
                "tflops": f"{r.tflops:.3f}" if r.tflops else "",
            })

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
) -> dict[str, list[FusedResult]]:
    """Run L2 fused operator benchmark suite."""
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    base_dir = Path(output_dir) / "L2" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

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
    except ImportError:
        pass

    with open(base_dir / "sources.json", "w") as f:
        json.dump(sources_manifest, f, indent=2)

    # Save config
    config = {
        "timestamp": timestamp,
        "ops": ops,
        "warmup": warmup,
        "reps": reps,
        "layer": "L2",
        "shape_tags": shape_tags,
    }
    with open(base_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_results: dict[str, list[FusedResult]] = {}

    for op in ops:
        logger.info(f"\n{'='*60}")
        logger.info(f"L2 Benchmark: {op}")
        logger.info(f"{'='*60}")

        results = run_fused_op(op, warmup=warmup, reps=reps, shape_tags=shape_tags)
        all_results[op] = results

        csv_path = save_results(results, base_dir, op)
        logger.info(f"  Saved: {csv_path}")

        print_comparison_table(results, op)

    print(f"\nResults saved to: {base_dir}")
    return all_results


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
        "-v", "--verbose", action="store_true",
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

    run_l2(ops=ops, output_dir=args.output, warmup=args.warmup, reps=args.reps)


if __name__ == "__main__":
    main()
