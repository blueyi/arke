# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke CLI — parse, optimize, and inspect .ak kernel files.

Usage:
    arke parse kernel.ak [-o kernel.json]
    arke inspect kernel.json
    arke optimize kernel.json --target ampere
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse .ak file → SemanticIR JSON."""
    from arke.parser.converter import ast_to_ir
    from arke.parser.parser import parse_file

    path = Path(args.input)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        program = parse_file(path)
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    if not program.kernels:
        print("Error: no kernel definitions found", file=sys.stderr)
        return 1

    # Convert all kernels
    results = []
    for kernel_ast in program.kernels:
        try:
            ir = ast_to_ir(kernel_ast)
            results.append(ir)
        except Exception as e:
            print(
                f"Conversion error in kernel '{kernel_ast.name}': {e}",
                file=sys.stderr,
            )
            return 1

    # Output
    if len(results) == 1:
        output = results[0].to_json(indent=2)
    else:
        output = json.dumps(
            [json.loads(r.to_json()) for r in results],
            indent=2,
        )

    if args.output:
        Path(args.output).write_text(output)
        print(f"Written: {args.output}")
    else:
        print(output)

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a SemanticIR in human-readable format."""
    from arke.ir.semantic import SemanticIR

    path = Path(args.input)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    # Load from .ak or .json
    if path.suffix == ".ak":
        from arke.parser.converter import ast_to_ir
        from arke.parser.parser import parse_file

        program = parse_file(path)
        if not program.kernels:
            print("Error: no kernel definitions found", file=sys.stderr)
            return 1
        ir = ast_to_ir(program.kernels[0])
    else:
        try:
            ir = SemanticIR.from_json(path.read_text())
        except Exception as e:
            print(f"Error loading IR: {e}", file=sys.stderr)
            return 1

    # Pretty print
    print(f"Kernel: {ir.kernel_id}")
    print(f"Version: {ir.version}")
    print()

    print("Parameters:")
    for p in ir.params:
        layout = f", {p.layout}" if p.layout != "row_major" else ""
        print(f"  {p.name}: Tensor<{p.shape}, {p.dtype}{layout}>")
    print()

    if ir.return_type:
        rt = ir.return_type
        print(f"Returns: Tensor<{rt.shape}, {rt.dtype}>")
        print()

    print("Computation Graph:")
    for node in ir.nodes:
        inputs_parts = []
        for k, v in node.inputs.items():
            if hasattr(v, "name"):
                inputs_parts.append(f"{k}={v.name}")
            elif hasattr(v, "id"):
                inputs_parts.append(f"{k}={v.id}")
            else:
                inputs_parts.append(f"{k}={v}")
        inputs_str = ", ".join(inputs_parts)
        out_shape = f" → {node.output.shape}" if node.output else ""
        print(f"  {node.id} = {node.op}({inputs_str}){out_shape}")

    print()
    print(f"Return Node: {ir.return_node}")

    if ir.fusion_groups:
        print()
        print("Fusion Groups:")
        for fg in ir.fusion_groups:
            print(f"  {fg}")

    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run LLM optimization on a SemanticIR."""
    import os

    from arke.ir.semantic import SemanticIR

    path = Path(args.input)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    # Load IR (from .json or .ak)
    if path.suffix == ".ak":
        from arke.parser.converter import ast_to_ir
        from arke.parser.parser import parse_file

        program = parse_file(path)
        if not program.kernels:
            print("Error: no kernel definitions found", file=sys.stderr)
            return 1
        ir = ast_to_ir(program.kernels[0])
    else:
        ir = SemanticIR.from_json(path.read_text())

    print(f"Kernel: {ir.kernel_id}")
    print(f"Params: {[p.name for p in ir.params]}")
    print(f"Ops: {[n.op for n in ir.nodes]}")
    print()

    # Load LLM config
    from arke.agent.llm_config import load_from_openclaw
    from arke.agent.runner import LLMRunner

    openclaw_dir = os.environ.get(
        "OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw")
    )
    config = load_from_openclaw(openclaw_dir)
    runner = LLMRunner(config, timeout=300.0)

    print(f"LLM: {config.primary}")
    print("Starting optimization...")
    print()

    try:
        result = runner.optimize(ir)
    finally:
        runner.close()

    # Summary
    print()
    print("=" * 50)
    print(f"Decisions: {result.decisions}")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Tokens: {result.tokens_in} in / {result.tokens_out} out")
    print(f"Duration: {result.duration_seconds:.1f}s")

    # Output generated code
    if result.generated_code:
        if args.output:
            Path(args.output).write_text(result.generated_code)
            print(f"\nTriton kernel: {args.output}")
        else:
            print("\n--- Generated Triton Kernel ---")
            print(result.generated_code)

    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="arke",
        description="Arke — AI-First Kernel Optimization Language",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse
    p_parse = subparsers.add_parser(
        "parse",
        help="Parse .ak file to SemanticIR JSON",
    )
    p_parse.add_argument("input", help="Input .ak file")
    p_parse.add_argument(
        "-o", "--output",
        help="Output JSON file (default: stdout)",
    )

    # inspect
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Inspect SemanticIR in human-readable format",
    )
    p_inspect.add_argument(
        "input",
        help="Input .json or .ak file",
    )

    # optimize
    p_optimize = subparsers.add_parser(
        "optimize",
        help="Run LLM optimization on a kernel",
    )
    p_optimize.add_argument(
        "input",
        help="Input .json or .ak file",
    )
    p_optimize.add_argument(
        "-o", "--output",
        help="Output Triton .py file",
    )
    p_optimize.add_argument(
        "--target", default="ampere",
        help="Target hardware (default: ampere)",
    )

    args = parser.parse_args()

    commands = {
        "parse": cmd_parse,
        "inspect": cmd_inspect,
        "optimize": cmd_optimize,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
