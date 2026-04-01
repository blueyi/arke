#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run Arke benchmark at a specified tier.

Wraps benchmarks.run with tier-based task selection.

Usage:
    python scripts/run_tier.py --tier 1 --phase phase1.5 --method both --trials 1
    python scripts/run_tier.py --tier 2 --phase phase2 --method arke --trials 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from gen_tasks import TASK_DEFS, get_tasks  # noqa: E402


def build_benchmark_tasks(tier: int):
    """Build BenchmarkTask objects for the given tier."""
    from arke.ir.builder import KernelBuilder
    from benchmarks.tasks import BenchmarkTask

    task_defs = get_tasks(tier)
    tasks = []

    for td in task_defs:
        op = td["op"]
        name = td["name"]
        M, N = td["M"], td["N"]
        K = td.get("K")

        try:
            ir = _build_ir(name, op, M, N, K)
            tasks.append(BenchmarkTask(
                name=name,
                description=f"{op} {M}×{N}" + (f"×{K}" if K else ""),
                semantic_ir=ir,
                tags=[td["cat"], op],
            ))
        except Exception as e:
            logging.warning(f"Skip {name}: {e}")

    return tasks


def _build_ir(name, op, M, N, K=None, dtype="f16"):
    """Build SemanticIR for a task."""
    from arke.ir.builder import KernelBuilder

    b = KernelBuilder(name)

    if op == "matmul":
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        m = b.op("matmul", A="A", B="B")
        b.returns(m, [M, N], dtype)

    elif op == "matmul_relu":
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        m = b.op("matmul", A="A", B="B")
        r = b.op("relu", X=m)
        b.returns(r, [M, N], dtype)

    elif op == "matmul_gelu":
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        m = b.op("matmul", A="A", B="B")
        g = b.op("gelu", X=m)
        b.returns(g, [M, N], dtype)

    elif op == "matmul_add":
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        b.param("bias", [M, N], dtype)
        m = b.op("matmul", A="A", B="B")
        a = b.op("add", A=m, B="bias")
        b.returns(a, [M, N], dtype)

    elif op == "matmul_add_relu":
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        b.param("bias", [M, N], dtype)
        m = b.op("matmul", A="A", B="B")
        a = b.op("add", A=m, B="bias")
        r = b.op("relu", X=a)
        b.returns(r, [M, N], dtype)

    elif op == "matmul_mul":
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        b.param("scale", [M, N], dtype)
        m = b.op("matmul", A="A", B="B")
        r = b.op("mul", A=m, B="scale")
        b.returns(r, [M, N], dtype)

    elif op in ("relu", "gelu"):
        b.param("X", [M, N], dtype)
        o = b.op(op, X="X")
        b.returns(o, [M, N], dtype)

    elif op in ("add", "mul"):
        b.param("A", [M, N], dtype)
        b.param("B", [M, N], dtype)
        o = b.op(op, A="A", B="B")
        b.returns(o, [M, N], dtype)

    elif op == "softmax":
        b.param("X", [M, N], dtype)
        s = b.op("softmax", X="X")
        b.returns(s, [M, N], dtype)

    elif op == "reduce_sum":
        b.param("X", [M, N], dtype)
        s = b.op("reduce_sum", X="X")
        b.returns(s, [M], dtype)

    elif op == "reduce_max":
        b.param("X", [M, N], dtype)
        s = b.op("reduce_max", X="X")
        b.returns(s, [M], dtype)

    elif op == "softmax_mul":
        b.param("X", [M, N], dtype)
        b.param("V", [M, N], dtype)
        s = b.op("softmax", X="X")
        r = b.op("mul", A=s, B="V")
        b.returns(r, [M, N], dtype)

    else:
        raise ValueError(f"Unknown op: {op}")

    return b.build()


def main():
    parser = argparse.ArgumentParser(description="Run Arke benchmark by tier")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1,
                        help="Tier level (1=core, 2=extended, 3=full)")
    parser.add_argument("--phase", required=True, help="Phase name for archival")
    parser.add_argument("--method", choices=["arke", "direct", "both"], default="both")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show tasks without running")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
    )

    tasks = build_benchmark_tasks(args.tier)
    print(f"Tier {args.tier}: {len(tasks)} tasks")
    for t in tasks:
        print(f"  {t}")

    if args.dry_run:
        return

    methods = ["arke", "direct"] if args.method == "both" else [args.method]

    from benchmarks.run import run_benchmark
    run_benchmark(
        methods=methods,
        trials=args.trials,
        tasks=tasks,
        output_dir="benchmarks/results",
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
