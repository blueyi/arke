# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke CLI — Command-line interface (S6+ refactor).

Subcommands:
    compile  — Compile .ak file to .akir (JSON) format
"""

import argparse
import json
import sys

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
        # Print JSON to stdout
        combined = akir_to_dict(
            result.semantic_ir,
            result.strategy_ir,
            schedule_ir=result.schedule_ir,
            instruction_ir=result.instruction_ir,
        )
        print(json.dumps(combined, indent=2))

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arke",
        description="Arke — AI-First operator description language toolchain",
    )
    subparsers = parser.add_subparsers(dest="command")

    # compile subcommand
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile .ak file to .akir (JSON) format",
    )
    compile_parser.add_argument(
        "input",
        help="Path to .ak source file",
    )
    compile_parser.add_argument(
        "-o", "--output",
        help="Output .akir file path (default: print to stdout)",
        default=None,
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compile":
        sys.exit(_cmd_compile(args))


if __name__ == "__main__":
    main()
