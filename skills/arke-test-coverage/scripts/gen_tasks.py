#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark task definitions for a given tier.

Usage:
    python scripts/gen_tasks.py --tier 1     # Print Tier 1 task names
    python scripts/gen_tasks.py --tier 2     # Print Tier 2 task names
    python scripts/gen_tasks.py --tier 3     # Print all tasks
    python scripts/gen_tasks.py --list       # Show all tasks with tiers
"""

from __future__ import annotations

import argparse
import json
import sys

# Task definitions: (name, tier, category, builder, kwargs)
# builder: "matmul" | "softmax" | "relu" | "gelu" | "add" | "mul" | "reduce_sum" | "reduce_max"
#          | "matmul_relu" | "matmul_gelu" | "matmul_add" | "matmul_add_relu" | "matmul_mul"

TASK_DEFS = [
    # ═══ CUBE Class ═══
    # Tier 1
    {"name": "matmul_1024",       "tier": 1, "cat": "cube",   "op": "matmul", "M": 1024, "N": 1024, "K": 1024},
    {"name": "matmul_2048",       "tier": 1, "cat": "cube",   "op": "matmul", "M": 2048, "N": 2048, "K": 2048},
    {"name": "matmul_rect",       "tier": 1, "cat": "cube",   "op": "matmul", "M": 1024, "N": 2048, "K": 512},
    {"name": "matmul_unaligned",  "tier": 1, "cat": "cube",   "op": "matmul", "M": 997,  "N": 1009, "K": 1013},
    {"name": "matmul_tall",       "tier": 1, "cat": "cube",   "op": "matmul", "M": 4096, "N": 256,  "K": 1024},
    # Tier 2
    {"name": "matmul_small",      "tier": 2, "cat": "cube",   "op": "matmul", "M": 256,  "N": 256,  "K": 256},
    {"name": "matmul_xlarge",     "tier": 2, "cat": "cube",   "op": "matmul", "M": 4096, "N": 4096, "K": 4096},
    {"name": "matmul_wide",       "tier": 2, "cat": "cube",   "op": "matmul", "M": 256,  "N": 4096, "K": 1024},
    {"name": "matmul_deep_k",     "tier": 2, "cat": "cube",   "op": "matmul", "M": 1024, "N": 1024, "K": 4096},
    {"name": "matmul_shallow_k",  "tier": 2, "cat": "cube",   "op": "matmul", "M": 1024, "N": 1024, "K": 64},
    # Tier 3
    {"name": "matmul_attn_qk",    "tier": 3, "cat": "cube",   "op": "matmul", "M": 1024, "N": 1024, "K": 64},
    {"name": "matmul_ffn_up",     "tier": 3, "cat": "cube",   "op": "matmul", "M": 1024, "N": 4096, "K": 1024},
    {"name": "matmul_ffn_down",   "tier": 3, "cat": "cube",   "op": "matmul", "M": 1024, "N": 1024, "K": 4096},
    {"name": "matmul_round",      "tier": 3, "cat": "cube",   "op": "matmul", "M": 1000, "N": 1000, "K": 1000},
    {"name": "matmul_tiny",       "tier": 3, "cat": "cube",   "op": "matmul", "M": 16,   "N": 16,   "K": 16},
    {"name": "matmul_extreme",    "tier": 3, "cat": "cube",   "op": "matmul", "M": 8192, "N": 32,   "K": 1024},

    # ═══ Vector Class — Elementwise ═══
    # Tier 1
    {"name": "relu_medium",       "tier": 1, "cat": "vector", "op": "relu",  "M": 1024, "N": 1024},
    {"name": "add_large",         "tier": 1, "cat": "vector", "op": "add",   "M": 4096, "N": 4096},
    # Tier 2
    {"name": "gelu_tall",         "tier": 2, "cat": "vector", "op": "gelu",  "M": 8192, "N": 128},
    {"name": "mul_wide",          "tier": 2, "cat": "vector", "op": "mul",   "M": 128,  "N": 8192},
    {"name": "add_unaligned",     "tier": 2, "cat": "vector", "op": "add",   "M": 1000, "N": 1000},
    # Tier 3
    {"name": "relu_small",        "tier": 3, "cat": "vector", "op": "relu",  "M": 256,  "N": 256},
    {"name": "relu_1row",         "tier": 3, "cat": "vector", "op": "relu",  "M": 1,    "N": 65536},
    {"name": "gelu_1col",         "tier": 3, "cat": "vector", "op": "gelu",  "M": 65536,"N": 1},
    {"name": "relu_prime",        "tier": 3, "cat": "vector", "op": "relu",  "M": 997,  "N": 1009},
    {"name": "mul_1d",            "tier": 3, "cat": "vector", "op": "mul",   "M": 1,    "N": 1048576},

    # ═══ Vector Class — Reduce ═══
    # Tier 1
    {"name": "softmax_short",     "tier": 1, "cat": "reduce", "op": "softmax",    "M": 4096, "N": 64},
    {"name": "softmax_4096",      "tier": 1, "cat": "reduce", "op": "softmax",    "M": 4096, "N": 4096},
    # Tier 2
    {"name": "reduce_sum_medium", "tier": 2, "cat": "reduce", "op": "reduce_sum", "M": 4096, "N": 1024},
    {"name": "reduce_sum_long",   "tier": 2, "cat": "reduce", "op": "reduce_sum", "M": 1024, "N": 16384},
    # Tier 3
    {"name": "reduce_max_1row",   "tier": 3, "cat": "reduce", "op": "reduce_max", "M": 1,    "N": 65536},
    {"name": "softmax_many_short","tier": 3, "cat": "reduce", "op": "softmax",    "M": 16384,"N": 32},
    {"name": "reduce_sum_unaligned","tier":3,"cat": "reduce", "op": "reduce_sum", "M": 1000, "N": 1000},

    # ═══ Fusion ═══
    # Tier 1
    {"name": "fused_matmul_relu", "tier": 1, "cat": "fusion", "op": "matmul_relu","M": 1024, "N": 1024, "K": 1024},
    {"name": "fused_matmul_gelu", "tier": 1, "cat": "fusion", "op": "matmul_gelu","M": 1024, "N": 2048, "K": 1024},
    {"name": "fused_matmul_add",  "tier": 1, "cat": "fusion", "op": "matmul_add", "M": 1024, "N": 1024, "K": 1024},
    # Tier 2
    {"name": "fused_matmul_add_relu","tier":2,"cat":"fusion", "op": "matmul_add_relu","M":1024,"N":1024,"K":1024},
    {"name": "fused_matmul_mul",  "tier": 2, "cat": "fusion", "op": "matmul_mul", "M": 1024, "N": 1024, "K": 1024},
    # Tier 3
    {"name": "fused_softmax_mul", "tier": 3, "cat": "fusion", "op": "softmax_mul","M": 4096, "N": 4096},
]


def get_tasks(tier: int) -> list[dict]:
    """Get tasks up to and including the given tier."""
    return [t for t in TASK_DEFS if t["tier"] <= tier]


def main():
    parser = argparse.ArgumentParser(description="List benchmark tasks by tier")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--list", action="store_true", help="Show all tasks with tiers")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.list:
        tasks = TASK_DEFS
    else:
        tasks = get_tasks(args.tier)

    if args.json:
        json.dump(tasks, sys.stdout, indent=2)
        print()
    else:
        fmt = "{:<28} {:>4}  {:<8} {:<16} {}"
        print(fmt.format("Name", "Tier", "Cat", "Op", "Shape"))
        print("-" * 80)
        for t in tasks:
            shape_parts = [str(t.get("M", "")), str(t.get("N", ""))]
            if "K" in t:
                shape_parts.append(str(t["K"]))
            shape = "×".join(p for p in shape_parts if p)
            print(fmt.format(t["name"], t["tier"], t["cat"], t["op"], shape))
        print(f"\nTotal: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
