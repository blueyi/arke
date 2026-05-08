# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke CLI.

Subcommands:
    compile   Compile .ak file to .akir JSON
    optimize  Stage 8 MVP autonomous strategy generation flow
"""

from __future__ import annotations

import argparse
import json
import sys

from arke.agent.optimize import optimize
from arke.compiler.pipeline import ArkePipeline
from arke.ir.akir import akir_to_dict


def _cmd_compile(args: argparse.Namespace) -> int:
    """Handle 'arke compile' subcommand."""
    pipeline = ArkePipeline()
    result = pipeline.compile_file(args.input)

    if not result.success:
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if args.output:
        result.save_akir(args.output)
        print(f"Compiled {args.input} -> {args.output}", file=sys.stderr)
    else:
        combined = akir_to_dict(
            result.semantic_ir,
            result.strategy_ir,
            schedule_ir=result.schedule_ir,
            instruction_ir=result.instruction_ir,
        )
        print(json.dumps(combined, indent=2))

    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    """Handle 'arke optimize' subcommand."""
    result = optimize(
        args.input,
        kernel=args.kernel,
        shape=args.shape,
        dtype=args.dtype,
        output_dir=args.output,
        cycles=args.cycles,
        dry_run=args.dry_run,
        target_hw=args.target,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "SUCCESS" if result.success else "FAILED"
        print(f"arke optimize {status}: {result.kernel_id}")
        print(f"  cycles: {result.cycles_completed}/{args.cycles}")
        print(f"  decisions: {result.decision_count}")
        print(f"  summary: {result.summary_path}")
        print(f"  trajectory: {result.trajectory_path}")
        if result.errors:
            for err in result.errors:
                print(f"ERROR: {err}", file=sys.stderr)
    return 0 if result.success else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arke",
        description="Arke — AI-First operator description language toolchain",
    )
    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile .ak file to .akir (JSON) format",
    )
    compile_parser.add_argument("input", help="Path to .ak source file")
    compile_parser.add_argument(
        "-o", "--output",
        help="Output .akir file path (default: print to stdout)",
        default=None,
    )

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Generate a bounded StrategyIR and trajectory for .ak, natural language, code, or structured input",
    )
    optimize_parser.add_argument(
        "input",
        nargs="?",
        help="Path to .ak source file, inline .ak source, natural-language request, or code snippet",
    )
    optimize_parser.add_argument(
        "-o", "--output",
        default="benchmarks/results/phase1/stage8/track1/optimize",
        help="Output directory for strategy/result/trajectory artifacts",
    )
    optimize_parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of compile->profile->adjust cycles to record",
    )
    optimize_parser.add_argument(
        "--kernel",
        default=None,
        help="Structured operator name (e.g. matmul, relu, softmax); requires --shape",
    )
    optimize_parser.add_argument(
        "--shape",
        default=None,
        help="Structured shape, comma- or x-separated (e.g. 1024,2048,512)",
    )
    optimize_parser.add_argument(
        "--dtype",
        default="f16",
        help="Input dtype for routed natural-language/code/structured kernels",
    )
    optimize_parser.add_argument(
        "--target",
        default="nvidia_ampere",
        help="Target hardware label for generated strategy",
    )
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate/lower without GPU execution (default for S8 MVP)",
    )
    optimize_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary JSON",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compile":
        sys.exit(_cmd_compile(args))
    if args.command == "optimize":
        sys.exit(_cmd_optimize(args))

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
