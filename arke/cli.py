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

    # ── run (unified Harness entry: pick an agent backend) ────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run the Harness on an op with a chosen agent backend "
             "(builtin live-LLM / heuristic / hermes / openclaw / mcp)",
    )
    run_parser.add_argument("--kernel", required=True,
                            help="Operator to optimize (e.g. matmul, softmax)")
    run_parser.add_argument("--shape", default=None,
                            help="Shape, comma-separated (op-specific, e.g. 512,512,512)")
    run_parser.add_argument("--backend", default="builtin",
                            help="Agent backend: builtin | heuristic | hermes | openclaw | mcp")
    run_parser.add_argument("--model", default=None,
                            help="model_spec for builtin backend, e.g. yunwu/claude-sonnet-4-6")
    run_parser.add_argument("--max-turns", type=int, default=15)
    run_parser.add_argument("--timeout", type=float, default=180.0)
    run_parser.add_argument("--target", default="nvidia_ampere",
                            help="Target hardware label")
    run_parser.add_argument("-o", "--output", default=None,
                            help="Output dir for state/trajectory artifacts")
    run_parser.add_argument("--json", action="store_true",
                            help="Print machine-readable result JSON")

    # ── mcp serve (Mode C, N3) ────────────────────────────────────────
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run Arke as an MCP server (Mode C) — expose the 8 Façade tools over JSON-RPC/stdio",
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    serve_parser = mcp_sub.add_parser("serve", help="Serve the 8-tool Façade over stdio")
    serve_parser.add_argument("--kernel", required=True,
                              help="Operator to expose an optimization env for (e.g. matmul)")
    serve_parser.add_argument("--shape", default=None,
                              help="Shape, comma-separated (op-specific, e.g. 512,512,512)")
    serve_parser.add_argument("--target", default="nvidia_ampere",
                              help="Target hardware label")

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
    if args.command == "run":
        sys.exit(_cmd_run(args))
    if args.command == "mcp":
        sys.exit(_cmd_mcp(args))

    parser.print_help()
    sys.exit(1)


def _cmd_run(args) -> int:
    """Unified Harness entry — dispatch to the selected agent backend."""
    from arke.agent.backends import run_backend
    from benchmarks.live.run_live_optimize import _shapes_for

    dims = [int(x) for x in args.shape.split(",") if x.strip()] if args.shape else []
    shapes = _shapes_for(args.kernel, dims)
    result = run_backend(
        args.backend,
        op_name=args.kernel, shapes=shapes, target_hw=args.target,
        max_turns=args.max_turns, model_spec=args.model,
        output_dir=args.output, timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        status = "OK" if result.success else "FAILED"
        print(f"arke run [{result.backend}/{result.mode}] {status}: {result.op_name}")
        print(f"  {result.message}")
        if result.mode == "mcp-server":
            print(f"  server: {result.detail.get('server_command')}")
    return 0 if result.success else 1


def _cmd_mcp(args) -> int:
    """Run the Arke MCP server (Mode C)."""
    if getattr(args, "mcp_command", None) != "serve":
        print("usage: arke mcp serve --kernel <op> [--shape d,d,...] [--target hw]", file=sys.stderr)
        return 1
    from arke.agent.mcp_server import serve
    from benchmarks.live.run_live_optimize import _shapes_for

    dims = [int(x) for x in args.shape.split(",") if x.strip()] if args.shape else []
    shapes = _shapes_for(args.kernel, dims)
    serve(args.kernel, shapes, args.target)
    return 0


if __name__ == "__main__":
    main()
