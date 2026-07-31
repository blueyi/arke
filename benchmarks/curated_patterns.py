#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Curated lead-engineer patterns for the @rationale KB.

This module is the *human-experience* write half of the @rationale loop
(SOUL.md / G9[3]). The auto-miners (``build_rationale_kb`` /
``mine_p5s5t_rationale``) can only distill what an agent/heuristic already
tried inside a single trajectory. They structurally cannot synthesize a
**cross-op generalization** a developer discovered by hand across several
independent optimization efforts. This module is where those hand-verified
rules enter the KB, with honest ``curated/<slug>`` provenance, so the read
side (``RationaleKB.recall`` → ``list_legal_actions`` ``rationale_priors``)
surfaces them to the Agent alongside auto-mined priors.

Rule of the house: every ``baseline_ratio`` here is a REAL measured number
from the author's own median-of-N interleaved A/B — never a guess. If a rule
is qualitative (no clean single ratio), leave ``baseline_ratio=None`` and let
the rationale text carry the value.

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    python -m benchmarks.curated_patterns            # seed into data/rationale_kb.jsonl
    python -m benchmarks.curated_patterns --dry-run  # report, don't write
"""

from __future__ import annotations

import argparse
import json
import sys

from arke.learn.rationale_kb import (
    DEFAULT_KB_PATH,
    RationaleEntry,
    curated_pattern,
    mine_curated,
)

# --------------------------------------------------------------------------- #
# sm_86 occupancy rule: "8 warps + wide tile over-subscribes registers and     #
# drops occupancy; 4 warps + a narrower tile wins." Discovered by median-of-N  #
# interleaved A/B across three independent tile-tuning efforts (2026-07-30):   #
#   FA-v5   long-S D=128 attention  128/32/8w → 128/16/4w                       #
#   MM-v2   mid-M matmul            64/128/64 s3 w8 → 64/64/32 s3 w4            #
#   GM-v2   grouped_matmul big>256  128/128 w8 → 64/64 w4 s3                    #
# Same hardware mechanism (Ampere sm_86, RTX 3060 Laptop) in all three; a      #
# per-trajectory miner sees only one op at a time and cannot state the rule.   #
# baseline_ratios below are the measured POST-improvement vs-flash-attn / vs-  #
# baseline geomeans / single-shape wins from that session's A/B logs.          #
# --------------------------------------------------------------------------- #

SM86_OCCUPANCY_SLUG = "sm86-occupancy-8w-to-4w"

_SM86_NOTE = (
    "sm_86 (Ampere, RTX 3060 Laptop) occupancy rule, verified by median-of-N "
    "interleaved A/B across FA-v5/MM-v2/GM-v2 (2026-07-30): for compute-bound "
    "ops, num_warps=8 with a wide tile over-subscribes the register file and "
    "drops occupancy; dropping to num_warps=4 with a narrower tile lifts "
    "occupancy and is a strict win. Trust only median-of-N interleaved A/B — "
    "single-shot sweeps take min-of-noise and fabricate gaps."
)

CURATED_PATTERNS: list[RationaleEntry] = [
    # FA-v5: long-sequence D=128 attention.
    curated_pattern(
        op="flash_attention",
        decision_kind="resource",
        params={"BLOCK_M": 128, "BLOCK_N": 16, "num_warps": 4, "regime": "long_S_D128"},
        rationale=(
            "Long-S D=128 branch: FA-v1's 128/32/8w is register-bound. "
            "128/16/4w is the single MHA+GQA winner (no is_mha split needed): "
            "MHA llama2-7b-4k 0.812->0.977, ds-v2-2k 0.801->0.962; "
            "GQA llama3-8b-2k 0.734->0.879, qwen-2k 0.736->0.885. " + _SM86_NOTE
        ),
        baseline_ratio=0.977,  # measured MHA llama2-7b-4k post-improvement
        correct=True,
        backend="triton",
        slug=SM86_OCCUPANCY_SLUG,
    ),
    # MM-v2: mid-M matmul.
    curated_pattern(
        op="matmul",
        decision_kind="resource",
        params={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_stages": 3, "num_warps": 4, "regime": "mid_M_le512"},
        rationale=(
            "mid-M branch (Mb<=512): 64/128/64 s3 w8 over-subscribes sm_86. "
            "64/64/32 s3 w4 is the unified branch winner, no regression: "
            "seq512 +7.2% (cv1.8%), rect-wide +3.4%, rect-tall +1.2%. " + _SM86_NOTE
        ),
        baseline_ratio=1.072,  # measured seq512 +7.2% post-improvement
        correct=True,
        backend="triton",
        slug=SM86_OCCUPANCY_SLUG,
    ),
    # GM-v2: grouped_matmul big group.
    curated_pattern(
        op="grouped_matmul",
        decision_kind="resource",
        params={"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 3, "num_warps": 4, "regime": "big_gt256"},
        rationale=(
            "big>256 branch: the 128/128 w8 placeholder was never tuned. "
            "64/64 w4 s3 is the unified winner: moe-medium(N=3072) +22.4% "
            "(median-of-7), moe-mid(256,2048) +9.2%, moe-mid(512,4096) +6.7%. "
            + _SM86_NOTE
        ),
        baseline_ratio=1.224,  # measured moe-medium +22.4% post-improvement
        correct=True,
        backend="triton",
        slug=SM86_OCCUPANCY_SLUG,
    ),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed curated lead-engineer patterns into the @rationale KB")
    ap.add_argument("--dry-run", action="store_true", help="report counts, don't write")
    ap.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    args = ap.parse_args(argv)

    for p in CURATED_PATTERNS:
        print(f"  {p.source}  {p.op:16s} {p.decision_kind:10s} ratio={p.baseline_ratio}", file=sys.stderr)
    print(f"{len(CURATED_PATTERNS)} curated patterns", file=sys.stderr)

    if args.dry_run:
        return 0

    result = mine_curated(CURATED_PATTERNS, kb_path=args.kb)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
