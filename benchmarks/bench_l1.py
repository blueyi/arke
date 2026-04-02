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

import benchmarks.baselines.arke_runner  # noqa: F401
import benchmarks.baselines.cublas  # noqa: F401
import benchmarks.baselines.flaggems  # noqa: F401
import benchmarks.baselines.inductor  # noqa: F401
import benchmarks.baselines.liger  # noqa: F401
import benchmarks.baselines.pytorch_eager  # noqa: F401
import benchmarks.baselines.triton_tutorial  # noqa: F401
from benchmarks.baselines.base import get_all_runners, get_runners_for_op
from benchmarks.hardware import collect_hardware_info
from benchmarks.measure import BenchResult, bench_fn, compute_matmul_tflops
from benchmarks.shapes import (
    MatmulShape,
    Shape2D,
    get_shapes,
)

logger = logging.getLogger(__name__)

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


def _get_shapes(
    op: str, tier: int | None = None
) -> list[MatmulShape] | list[Shape2D]:
    try:
        return get_shapes(op, tier=tier)
    except ValueError:
        return []


def run_op(
    op: str,
    shapes: list[MatmulShape] | list[Shape2D] | None = None,
    warmup: int = 200,
    reps: int = 500,
    tier: int | None = None,
) -> list[OpResult]:
    """Benchmark one operator across shapes and baselines."""
    if shapes is None:
        shapes = _get_shapes(op, tier=tier)

    runners = get_runners_for_op(op)
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
            if isinstance(shape, MatmulShape):
                tag, M, N, K = shape.tag, shape.M, shape.N, shape.K
            else:
                tag, M, N, K = shape.tag, shape.M, shape.N, 0

            for runner in runner_group:
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
                    )
                    results.append(result)
                    tflops_str = f" {tflops:.2f} TFLOPS" if tflops else ""
                    logger.info(
                        f"  {tag:15s} {runner.name:15s} "
                        f"{bench_result.latency_us:8.1f} μs{tflops_str}"
                    )
                except Exception as e:
                    logger.warning(f"  {tag} {runner.name}: FAILED ({e})")

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
                "baseline": r.baseline,
                "priority": r.priority,
                "source": r.source,
                "latency_us": f"{r.latency_us:.1f}",
                "latency_min_us": f"{r.latency_min_us:.1f}",
                "tflops": f"{r.tflops:.3f}" if r.tflops else "",
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
) -> dict[str, list[OpResult]]:
    """Run L1 benchmark suite."""
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    base_dir = Path(output_dir) / "L1" / timestamp
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
        "timestamp": timestamp,
        "ops": ops,
        "warmup": warmup,
        "reps": reps,
        "tier": tier,
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

        results = run_op(op, warmup=warmup, reps=reps, tier=tier)
        all_results[op] = results

        csv_path = save_results(results, base_dir, op)
        logger.info(f"  Saved: {csv_path}")

        print_comparison_table(results, op)

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
        "--tier", type=int, default=None,
        help="Shape tier (1=fast, 2=standard, 3=full). Default: all shapes.",
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

    run_l1(
        ops=ops, output_dir=args.output, warmup=args.warmup, reps=args.reps,
        tier=args.tier,
    )


if __name__ == "__main__":
    main()
