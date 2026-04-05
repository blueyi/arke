# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI entry point for the Arke Benchmark System.

Usage:
    python -m benchmarks                    # BL2 (default: OT0-2 × ST1-2, L1)
    python -m benchmarks --bl 1             # Quick smoke test
    python -m benchmarks --bl 5             # Complete suite (all ops × all shapes)
    python -m benchmarks --bl 6             # E2E model validation
    python -m benchmarks --ot 0             # Elementwise only
    python -m benchmarks --st 4             # Production shapes only
    python -m benchmarks --layer L1         # Single ops only
    python -m benchmarks --op matmul        # Specific operator
    python -m benchmarks --report           # Generate report

See docs/design/benchmark/benchmark-protocol.md for full CLI specification.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

# ── BL → Default expansion table ────────────────────────────────────────

BL_DEFAULTS: dict[int, dict] = {
    1: {"ot_range": (0, 2), "st_range": (1, 1), "layers": ["L1"]},
    2: {"ot_range": (0, 2), "st_range": (1, 2), "layers": ["L1"]},
    3: {"ot_range": (0, 2), "st_range": (1, 3), "layers": ["L1"]},
    4: {"ot_range": (0, 4), "st_range": (1, 2), "layers": ["L1", "L2"]},
    5: {"ot_range": (0, 4), "st_range": (1, 4), "layers": ["L1", "L2"]},
    6: {"ot_range": (0, 4), "st_range": (1, 4), "layers": ["L1", "L2", "L3"]},
}

# ── OT → Operators ──────────────────────────────────────────────────────

OT_OPS: dict[int, list[str]] = {
    0: ["relu", "gelu", "silu", "add", "mul"],
    1: ["softmax", "layernorm", "rmsnorm", "rmsnorm_residual", "reduce_sum", "reduce_max"],
    2: ["matmul", "batch_matmul", "grouped_matmul", "transpose"],
    3: ["swiglu", "geglu"],
    4: ["flash_attention", "grouped_query_attention", "multi_latent_attention"],
}


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arke bench",
        description="Arke Benchmark System — unified CLI",
    )

    # Primary control
    parser.add_argument(
        "--bl", type=int, default=None,
        help="Benchmark Level (1-6). Default: 2. Controls default OT/ST/Layer.",
    )

    # Overrides
    parser.add_argument(
        "--ot", type=str, default=None,
        help="Operator Tier filter (comma-separated, 0-4). Overrides BL default.",
    )
    parser.add_argument(
        "--st", type=str, default=None,
        help="Shape Tier filter (comma-separated, 1-4). Overrides BL default.",
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        help="Evaluation Layer filter (L1, L2, L3). Overrides BL default.",
    )
    parser.add_argument(
        "--op", type=str, default=None,
        help="Specific operator(s) (comma-separated, e.g. matmul,softmax).",
    )

    # Legacy compat
    parser.add_argument(
        "--all", action="store_true",
        help="Run all layers (legacy, equivalent to --bl 6).",
    )

    # Config
    parser.add_argument(
        "--report", action="store_true",
        help="Generate report.md from existing results.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="benchmarks/results",
        help="Results directory (default: benchmarks/results).",
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--warmup", type=int, default=200,
        help="Warmup iterations (default: 200).",
    )
    parser.add_argument(
        "--reps", type=int, default=500,
        help="Measurement repetitions (default: 500).",
    )
    parser.add_argument(
        "--seq-len", type=str, default=None,
        help="Seq lengths for L3 (comma-separated, default: 128,256,512).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model for L3/BL6 (e.g. gpt2, llama2-7b).",
    )
    parser.add_argument(
        "--baselines", type=str, default=None,
        help="Baseline methods (comma-separated, e.g. cublas,flaggems,arke).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Handle report-only mode
    if args.report:
        _run_report(args)
        return

    # Resolve BL
    if args.all:
        bl = 6
    elif args.bl is not None:
        bl = args.bl
    elif any([args.ot, args.st, args.layer, args.op]):
        bl = None  # User specified overrides, don't apply BL defaults
    else:
        bl = 2  # Default

    # Resolve layers, OT range, ST range from BL + overrides
    if bl is not None:
        defaults = BL_DEFAULTS.get(bl, BL_DEFAULTS[2])
        ot_min, ot_max = defaults["ot_range"]
        st_min, st_max = defaults["st_range"]
        layers = defaults["layers"]
    else:
        ot_min, ot_max = 0, 4
        st_min, st_max = 1, 4
        layers = ["L1"]

    # Apply overrides
    if args.layer:
        layer_str = args.layer.upper()
        layers = [layer_str]
        # Validation: L3 implies BL6
        if layer_str == "L3" and bl is not None and bl < 6:
            logger.info("L3 implies BL6, expanding scope")
        # L2 requires OT3+
        if layer_str == "L2" and ot_max < 3:
            ot_max = 4

    if args.ot:
        ot_tiers = _parse_int_list(args.ot)
    else:
        ot_tiers = list(range(ot_min, ot_max + 1))

    if args.st:
        st_tiers = _parse_int_list(args.st)
    else:
        st_tiers = list(range(st_min, st_max + 1))

    # Resolve operators
    if args.op:
        ops = [o.strip() for o in args.op.split(",")]
    else:
        ops = []
        for ot in ot_tiers:
            ops.extend(OT_OPS.get(ot, []))

    max_st = max(st_tiers) if st_tiers else 2

    # Log resolved config
    logger.info(f"Benchmark config: BL={bl} OT={ot_tiers} ST={st_tiers} "
                f"Layers={layers} Ops={ops}")

    # Execute
    for layer in layers:
        if layer == "L1":
            _run_l1(args, ops, max_st)
        elif layer == "L2":
            _run_l2(args)
        elif layer == "L3":
            _run_l3(args)
        else:
            logger.error(f"Unknown layer: {layer}")
            sys.exit(1)

    # Generate summary report
    if layers:
        _run_report(args, silent=True)


def _run_l1(args: argparse.Namespace, ops: list[str], max_st: int) -> None:
    """Run L1 single operator benchmarks."""
    from benchmarks.bench_l1 import ALL_OPS, run_l1

    # Filter to ops that bench_l1 actually supports
    supported = set(ALL_OPS)
    runnable = [op for op in ops if op in supported]
    skipped = [op for op in ops if op not in supported]

    if skipped:
        logger.warning(f"L1: skipping unsupported ops: {skipped}")
    if not runnable:
        logger.warning("L1: no runnable ops for current config")
        return

    logger.info(f"Running L1: ops={runnable}, max_tier={max_st}")
    run_l1(ops=runnable, output_dir=args.output, warmup=args.warmup, reps=args.reps)


def _run_l2(args: argparse.Namespace) -> None:
    """Run L2 fused operator benchmarks."""
    from benchmarks.bench_l2 import ALL_FUSED_OPS, run_l2

    logger.info(f"Running L2: {ALL_FUSED_OPS}")
    run_l2(ops=ALL_FUSED_OPS, output_dir=args.output,
           warmup=args.warmup, reps=args.reps)


def _run_l3(args: argparse.Namespace) -> None:
    """Run L3 E2E model benchmarks."""
    from benchmarks.bench_l3 import DEFAULT_SEQ_LENS, run_l3

    if args.seq_len:
        seq_lens = [int(s.strip()) for s in args.seq_len.split(",")]
    else:
        seq_lens = DEFAULT_SEQ_LENS

    logger.info(f"Running L3: seq_lens={seq_lens}")
    run_l3(seq_lens=seq_lens, output_dir=args.output)


def _run_report(args: argparse.Namespace, silent: bool = False) -> None:
    """Generate report from existing results."""
    from pathlib import Path
    from benchmarks.report import generate_report

    report = generate_report(
        results_dir=Path(args.results_dir),
        output_path=Path(args.results_dir) / "report.md",
    )
    if not silent:
        print(report)
    else:
        logger.info(f"Report saved to: {args.results_dir}/report.md")


if __name__ == "__main__":
    main()
