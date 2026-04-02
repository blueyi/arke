# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI entry point for the Arke Benchmark System.

Usage:
    python -m benchmarks --all              # Run L1 + L2 + L3
    python -m benchmarks --layer L1         # Run L1 only
    python -m benchmarks --layer L2         # Run L2 only
    python -m benchmarks --layer L3         # Run L3 only
    python -m benchmarks --op matmul        # Run L1 for matmul only
    python -m benchmarks --report           # Generate report from results
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Arke Benchmark System — unified CLI",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark layers (L1 + L2 + L3)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Run a specific layer: L1, L2, or L3",
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help="Run L1 for specific ops (comma-separated, e.g. matmul,softmax)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report.md from existing results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Results directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--warmup", type=int, default=200,
        help="Warmup iterations for L1/L2 (default: 200)",
    )
    parser.add_argument(
        "--reps", type=int, default=500,
        help="Measurement repetitions for L1/L2 (default: 500)",
    )
    parser.add_argument(
        "--seq-len",
        type=str,
        default=None,
        help="Seq lengths for L3 (comma-separated, default: 128,256,512)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not any([args.all, args.layer, args.op, args.report]):
        parser.print_help()
        sys.exit(1)

    # Generate report only
    if args.report:
        from pathlib import Path

        from benchmarks.report import generate_report

        report = generate_report(
            results_dir=Path(args.results_dir),
            output_path=Path(args.results_dir) / "report.md",
        )
        print(report)
        return

    # Determine which layers to run
    layers: list[str] = []
    if args.all:
        layers = ["L1", "L2", "L3"]
    elif args.layer:
        layers = [args.layer.upper()]
    elif args.op:
        layers = ["L1"]  # --op implies L1

    # Run layers
    for layer in layers:
        if layer == "L1":
            _run_l1(args)
        elif layer == "L2":
            _run_l2(args)
        elif layer == "L3":
            _run_l3(args)
        else:
            logger.error(f"Unknown layer: {layer}")
            sys.exit(1)

    # Generate summary report after running
    if layers:
        from pathlib import Path

        from benchmarks.report import generate_report

        logger.info("\nGenerating summary report...")
        generate_report(
            results_dir=Path(args.output),
            output_path=Path(args.output) / "report.md",
        )
        logger.info(f"Report saved to: {args.output}/report.md")


def _run_l1(args: argparse.Namespace) -> None:
    """Run L1 single operator benchmarks."""
    from benchmarks.bench_l1 import ALL_OPS, run_l1

    if args.op:
        ops = [o.strip() for o in args.op.split(",")]
    else:
        ops = ALL_OPS

    logger.info(f"Running L1 benchmarks: {ops}")
    run_l1(ops=ops, output_dir=args.output, warmup=args.warmup, reps=args.reps)


def _run_l2(args: argparse.Namespace) -> None:
    """Run L2 fused operator benchmarks."""
    from benchmarks.bench_l2 import ALL_FUSED_OPS, run_l2

    logger.info(f"Running L2 benchmarks: {ALL_FUSED_OPS}")
    run_l2(ops=ALL_FUSED_OPS, output_dir=args.output,
           warmup=args.warmup, reps=args.reps)


def _run_l3(args: argparse.Namespace) -> None:
    """Run L3 E2E model benchmarks."""
    from benchmarks.bench_l3 import DEFAULT_SEQ_LENS, run_l3

    if args.seq_len:
        seq_lens = [int(s.strip()) for s in args.seq_len.split(",")]
    else:
        seq_lens = DEFAULT_SEQ_LENS

    logger.info(f"Running L3 benchmarks: seq_lens={seq_lens}")
    run_l3(seq_lens=seq_lens, output_dir=args.output)


if __name__ == "__main__":
    main()
