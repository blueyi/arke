# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Guard the BL5-full-coverage semantics of `get_shapes(tier=3)`.

Per docs/phase1/stage7-plan.md L62-64 and docs/benchmark/benchmark-shapes.md L11-21:

* BL5 (the Stage 7 / G7.8b coverage target) requires OT0–OT4 × ST1–ST4 *full*
  coverage — ST4 is part of BL5, not a Phase 2-only tier.
* The benchmark CLI exposes `--tier {1,2,3}` mapping to ST1/ST2/ST3-as-stress-set.
  Therefore `--tier 3` (the documented "Gate validation" level) MUST include
  ST4 production-shape rows; otherwise G7.8b coverage is structurally unreachable.

This test pins that semantic so any future regression to the old
`shape.tier <= tier` filter (which excluded ST4 at tier=3) trips here long
before it silently re-opens the G7.8b gap.
"""

from __future__ import annotations

import pytest

from benchmarks.shapes import get_shapes


# Representative ST4 shape tags that MUST appear at tier=3 for BL5 coverage.
# Pulled from benchmarks/stage7_bl5_target_matrix.json shape_tags_required
# and confirmed present in docs/benchmark/benchmark-shapes.md ST4 sub-tables.
_BL5_REQUIRED_ST4_TAGS: dict[str, list[str]] = {
    "softmax": ["ds-v2-attn-16k", "ds-v3-attn-32k", "llama3-attn-8k", "qwen25-attn-32k"],
    "layernorm": ["llama3-8b-norm", "qwen25-7b-norm", "ds-v2-long", "ds-v3-long"],
    "rmsnorm": ["llama3-8b-norm", "qwen25-7b-norm", "ds-v2-long", "ds-v3-long"],
    "rmsnorm_residual": ["llama3-8b-norm", "qwen25-7b-norm"],
    "matmul": ["ds-v2-attn", "ds-v2-ffn-up", "ds-v2-ffn-down", "ds-v2-lmhead", "ds-v2-long-8k"],
    "grouped_matmul": ["ds-moe-2k", "ds-moe-512", "ds-v3-moe"],
    "silu_and_mul": ["ds-v2-2k", "ds-v3-2k", "qwen25-7b-2k"],
    "gelu_and_mul": ["ds-v2-2k", "ds-v3-2k", "qwen25-7b-2k"],
    "rope": ["ds-v3-2k", "llama3-8k", "qwen25-32k"],
    "fused_linear_cross_entropy": ["llama3-seq2k", "qwen25-seq2k"],
    "flash_attention": ["llama2-7b-2k", "ds-v2-2k", "ds-v3-32k"],
}


@pytest.mark.parametrize("op,required_tags", list(_BL5_REQUIRED_ST4_TAGS.items()))
def test_tier3_includes_st4_bl5_required_shapes(op: str, required_tags: list[str]) -> None:
    """tier=3 must surface every BL5-required ST4 tag for the op."""
    shapes = get_shapes(op, tier=3)
    available = {s.tag for s in shapes}
    missing = [tag for tag in required_tags if tag not in available]
    assert not missing, (
        f"{op} @ tier=3 is missing BL5-required ST4 tags: {missing}\n"
        f"This indicates the BL5 coverage filter regressed — `--tier 3` "
        f"(Gate validation) MUST include ST4 production shapes."
    )


def test_tier3_equals_all_for_bl5_ops() -> None:
    """For BL5-relevant ops, tier=3 returns the same set as tier=None."""
    bl5_ops = list(_BL5_REQUIRED_ST4_TAGS.keys())
    diffs: dict[str, set[str]] = {}
    for op in bl5_ops:
        t3 = {s.tag for s in get_shapes(op, tier=3)}
        all_ = {s.tag for s in get_shapes(op, tier=None)}
        only_in_all = all_ - t3
        if only_in_all:
            diffs[op] = only_in_all
    assert not diffs, (
        f"tier=3 != tier=None for BL5 ops: {diffs}\n"
        f"BL5 full coverage requires tier=3 to be a complete superset."
    )


def test_tier1_and_tier2_still_filter_strictly() -> None:
    """tier=1 and tier=2 must still filter (no accidental BL5-leak into smoke runs)."""
    # softmax has ST1=3 / ST2=10 (cumulative) — picking it as a representative.
    s1 = {s.tag for s in get_shapes("softmax", tier=1)}
    s2 = {s.tag for s in get_shapes("softmax", tier=2)}
    s3 = {s.tag for s in get_shapes("softmax", tier=3)}
    assert s1 < s2 < s3, "tier filtering should be strictly nested for tier 1/2/3"
    # ST4 production tags MUST NOT leak into tier=1 or tier=2
    assert "ds-v2-attn-16k" not in s1
    assert "ds-v2-attn-16k" not in s2
    assert "ds-v2-attn-16k" in s3


def test_attention_ops_only_have_st4_shapes() -> None:
    """flash_attention / GQA / MLA / paged_attention only have ST4 shapes;
    tier=1/2 returns 0 (correct — they are production-only ops), tier=3 returns full set.
    """
    for op in ("flash_attention", "multi_latent_attention",
               "grouped_query_attention", "paged_attention"):
        assert get_shapes(op, tier=1) == [], f"{op} should have 0 ST1 shapes"
        assert get_shapes(op, tier=2) == [], f"{op} should have 0 ST1+ST2 shapes"
        assert len(get_shapes(op, tier=3)) > 0, (
            f"{op} should expose its full ST4 set at tier=3 for BL5 coverage"
        )
