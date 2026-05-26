#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check token efficiency: .ak lines vs estimated Triton equivalent.

Gate criterion G6-LI.4: .ak files should have fewer non-blank, non-comment
lines than equivalent Triton kernel implementations.

Reference Triton line counts are based on:
- Actual Triton kernels in benchmarks/results/ for ops we have
- Community references (Triton tutorials, FlagGems, Liger kernels)
- Conservative estimates for remaining ops

Output: table of op | .ak lines | Triton est. lines | ratio
Assert: all ratios < 1.0
"""

from __future__ import annotations

import sys
from pathlib import Path

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
TRITON_SOURCES = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

# Reference Triton line counts (non-blank, non-comment).
# Sources:
#   - Measured: from benchmarks/results/phase1/gates/G*/sources/triton/*.py
#   - Community: Triton tutorials, FlagGems, Liger-Kernel
#   - Estimated: based on op complexity category
#
# Categories for estimation:
#   D (elementwise): 30-45 lines (kernel + wrapper + pointer arithmetic)
#   C (reduction):   45-65 lines (reduction loop + warp shuffle)
#   A (matmul/dense): 70-120 lines (tiled loops + shared memory)
#   B (attention):   100-200 lines (multi-pass, online softmax)
#   E (data movement): 35-55 lines (strided access patterns)
TRITON_REFERENCE_LINES: dict[str, int] = {
    # ── OT0: Elementwise (12 ops) ──
    "00_relu": 35,
    "03_gelu": 38,        # measured: gelu_128_768.py = 41 lines total
    "07_silu": 35,
    "09_add": 32,
    "10_mul": 32,
    "21_tanh": 35,
    "22_sigmoid": 35,
    "23_where_": 38,
    "24_cast": 35,
    "25_neg": 32,
    "26_exp": 35,
    "27_rsqrt": 35,

    # ── OT1: Reduction (10 ops) ──
    "02_softmax": 50,     # measured: softmax_1024_1024.py = 57 lines
    "04_layernorm": 65,   # measured: layernorm_128_768.py = 73 lines
    "06_rmsnorm": 55,
    "11_reduce_sum": 45,
    "12_reduce_max": 45,
    "28_reduce_mean": 48,
    "29_argmax": 50,
    "30_topk": 60,
    "31_cumsum": 55,
    "40_cross_entropy": 55,

    # ── OT2: Compute-Dense (11 ops) ──
    "01_matmul": 80,      # measured: matmul_1024_1024_1024.py = 85 lines
    "05_matmul_gelu": 95,
    "08_batch_matmul": 90,
    "13_transpose": 40,
    "14_grouped_matmul": 100,
    "32_concat": 40,
    "33_split": 40,
    "34_gather": 45,
    "35_scatter": 50,
    "36_embedding": 45,
    "37_permute": 45,

    # ── OT3: Gated Activation / Specialized (7 ops) ──
    "18_rmsnorm_residual": 65,
    "19_silu_and_mul": 45,
    "20_gelu_and_mul": 45,
    "38_copy_": 35,
    "39_rope": 60,
    "41_fused_linear_cross_entropy": 100,
    "42_quantize_per_token": 55,
    "43_dequantize_per_channel": 55,

    # ── OT4: Attention (5 ops) ──
    "15_flash_attention": 150,
    "16_grouped_query_attention": 160,
    "17_multi_latent_attention": 180,
    "44_cross_attention": 150,
    "45_paged_attention": 170,

    # ── BL5 L2 fused examples (see `examples/operators/l2/*.ak`) ──
    # Triton references are conservative lower bounds for hand-written
    # fused kernels covering the same composition.
    "linear_ce": 110,        # fused Linear + CE, cf. 41_fused_linear_cross_entropy
    "matmul_relu": 90,       # tiled matmul with epilogue fusion
    "qkv_fa": 200,           # QKV projection + flash-attention in one fused kernel
}


def count_code_lines(path: Path) -> int:
    """Count non-blank, non-comment lines in a file."""
    count = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        count += 1
    return count


def main() -> int:
    ak_files = sorted(OPERATORS_DIR.glob("*.ak")) + sorted(
        OPERATORS_DIR.glob("l2/*.ak")
    )
    if not ak_files:
        print("ERROR: No .ak files found in", OPERATORS_DIR)
        return 1

    print(f"{'Op':<45} {'AK Lines':>10} {'Triton Est.':>12} {'Ratio':>8}")
    print("-" * 78)

    all_pass = True
    total_ak = 0
    total_triton = 0

    for ak_file in ak_files:
        stem = ak_file.stem
        ak_lines = count_code_lines(ak_file)
        triton_lines = TRITON_REFERENCE_LINES.get(stem)

        if triton_lines is None:
            print(f"  WARNING: No Triton reference for {stem}, skipping")
            continue

        ratio = ak_lines / triton_lines
        total_ak += ak_lines
        total_triton += triton_lines

        status = "OK" if ratio < 1.0 else "FAIL"
        if ratio >= 1.0:
            all_pass = False

        print(f"  {stem:<43} {ak_lines:>8} {triton_lines:>12} {ratio:>8.3f}  {status}")

    print("-" * 78)
    overall_ratio = total_ak / total_triton if total_triton > 0 else 0
    print(f"  {'TOTAL':<43} {total_ak:>8} {total_triton:>12} {overall_ratio:>8.3f}")
    print()

    if all_pass:
        print("PASS: All .ak files have fewer lines than Triton equivalents")
        return 0
    else:
        print("FAIL: Some .ak files exceed Triton line count")
        return 1


if __name__ == "__main__":
    sys.exit(main())
