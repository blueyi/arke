# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for token efficiency: .ak files are concise vs Triton equivalents.

Gate criterion G6-LI.4: .ak lines <= Triton lines for equivalent ops.

Strategy: Each .ak kernel+strategy is typically 10-30 lines of actual code,
while a Triton kernel for the same operation is 30-200 lines. We verify:
1. Each .ak file has <= MAX_AK_LINES non-blank, non-comment lines
2. The ratio .ak / Triton_estimate < 1.0 for all ops
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))

# Maximum non-blank, non-comment lines allowed per .ak file.
# Triton kernels for equivalent ops are typically 30-200 lines,
# so 50 lines is a generous upper bound for .ak that still guarantees
# token efficiency vs Triton.
MAX_AK_LINES = 50

# Minimum Triton lines for any op category (conservative lower bound):
# Even the simplest Triton kernel (e.g., vector add) is ~30 lines with
# kernel signature, pointer math, mask, load, compute, store, wrapper.
MIN_TRITON_LINES_BY_CATEGORY = {
    "elementwise": 30,    # OT0: relu, gelu, sigmoid, etc.
    "reduction": 45,      # OT1: softmax, layernorm, reduce_*
    "compute_dense": 70,  # OT2: matmul, batch_matmul
    "gated": 40,          # OT3: silu_and_mul, geglu, rope
    "attention": 100,     # OT4: flash_attention, GQA, MLA
    "data_movement": 35,  # transpose, gather, scatter, etc.
}

# Map each .ak file stem to its Triton category
OP_CATEGORY: dict[str, str] = {
    "00_relu": "elementwise",
    "01_matmul": "compute_dense",
    "02_softmax": "reduction",
    "03_gelu": "elementwise",
    "04_layernorm": "reduction",
    "05_matmul_gelu": "compute_dense",
    "06_rmsnorm": "reduction",
    "07_silu": "elementwise",
    "08_batch_matmul": "compute_dense",
    "09_add": "elementwise",
    "10_mul": "elementwise",
    "11_reduce_sum": "reduction",
    "12_reduce_max": "reduction",
    "13_transpose": "data_movement",
    "14_grouped_matmul": "compute_dense",
    "15_flash_attention": "attention",
    "16_grouped_query_attention": "attention",
    "17_multi_latent_attention": "attention",
    "18_rmsnorm_residual": "reduction",
    "19_silu_and_mul": "gated",
    "20_geglu": "gated",
    "21_tanh": "elementwise",
    "22_sigmoid": "elementwise",
    "23_where_": "elementwise",
    "24_cast": "elementwise",
    "25_neg": "elementwise",
    "26_exp": "elementwise",
    "27_rsqrt": "elementwise",
    "28_reduce_mean": "reduction",
    "29_argmax": "reduction",
    "30_topk": "reduction",
    "31_cumsum": "reduction",
    "32_concat": "data_movement",
    "33_split": "data_movement",
    "34_gather": "data_movement",
    "35_scatter": "data_movement",
    "36_embedding": "data_movement",
    "37_permute": "data_movement",
    "38_copy_": "elementwise",
    "39_rope": "gated",
    "40_cross_entropy": "reduction",
    "41_fused_linear_cross_entropy": "compute_dense",
    "42_quantize_per_token": "gated",
    "43_dequantize_per_channel": "gated",
    "44_cross_attention": "attention",
    "45_paged_attention": "attention",
}


def _count_code_lines(path: Path) -> int:
    """Count non-blank, non-comment lines."""
    count = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        count += 1
    return count


class TestTokenEfficiency:
    """Verify .ak files are more concise than Triton equivalents."""

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_ak_under_line_threshold(self, ak_file: Path):
        """Each .ak file has <= MAX_AK_LINES non-blank, non-comment lines."""
        lines = _count_code_lines(ak_file)
        assert lines <= MAX_AK_LINES, (
            f"{ak_file.name}: {lines} code lines exceeds threshold of {MAX_AK_LINES}"
        )

    @pytest.mark.parametrize("ak_file", ALL_AK_FILES, ids=lambda p: p.name)
    def test_ak_fewer_than_triton_estimate(self, ak_file: Path):
        """Each .ak file has fewer lines than the minimum Triton equivalent."""
        stem = ak_file.stem
        category = OP_CATEGORY.get(stem)
        if category is None:
            pytest.skip(f"No category mapping for {stem}")

        ak_lines = _count_code_lines(ak_file)
        triton_min = MIN_TRITON_LINES_BY_CATEGORY[category]

        assert ak_lines < triton_min, (
            f"{ak_file.name}: {ak_lines} .ak lines >= {triton_min} Triton min "
            f"(category: {category})"
        )

    def test_all_ops_have_category(self):
        """Every .ak file is mapped to a Triton category."""
        for ak_file in ALL_AK_FILES:
            assert ak_file.stem in OP_CATEGORY, (
                f"{ak_file.name} missing from OP_CATEGORY mapping"
            )

    def test_aggregate_ratio(self):
        """Overall .ak / Triton ratio is well below 1.0."""
        total_ak = 0
        total_triton = 0
        for ak_file in ALL_AK_FILES:
            stem = ak_file.stem
            category = OP_CATEGORY.get(stem)
            if category is None:
                continue
            total_ak += _count_code_lines(ak_file)
            total_triton += MIN_TRITON_LINES_BY_CATEGORY[category]

        ratio = total_ak / total_triton if total_triton > 0 else 999
        assert ratio < 1.0, (
            f"Aggregate ratio {ratio:.3f} >= 1.0 "
            f"(total .ak={total_ak}, total Triton est.={total_triton})"
        )
