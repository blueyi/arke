# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the @rationale KB (G9[3], D8-A4)."""

from __future__ import annotations

import json
from pathlib import Path

from arke.learn.rationale_kb import (
    RationaleEntry,
    RationaleKB,
    RationalePrior,
    _normalize_op,
    curated_pattern,
    mine_curated,
    mine_trajectory,
)


def test_entry_key_stable_and_dedupes(tmp_path):
    e1 = RationaleEntry(op="matmul", decision_kind="tile",
                        params={"loop": "i", "factors": [64]}, rationale="L2 reuse")
    e2 = RationaleEntry(op="matmul", decision_kind="tile",
                        params={"loop": "i", "factors": [64]}, rationale="L2 reuse")
    assert e1.key() == e2.key()
    kb = RationaleKB(tmp_path / "kb.jsonl")
    assert kb.add_entries([e1, e2]) == 1  # dedup
    assert kb.count() == 1


def test_add_entries_append_only(tmp_path):
    kb = RationaleKB(tmp_path / "kb.jsonl")
    kb.add_entries([RationaleEntry(op="a", decision_kind="tile", params={}, rationale="r1")])
    kb.add_entries([RationaleEntry(op="b", decision_kind="unroll", params={}, rationale="r2")])
    assert kb.count() == 2


def test_mine_trajectory_pairs_decision_with_outcome(tmp_path):
    traj = tmp_path / "t.jsonl"
    lines = [
        {"kind": "header", "data": {"kernel_id": "matmul"}},
        {"kind": "decision", "data": {"decision": {"kind": "tile", "params": {"loop": "i"}},
                                       "rationale": "tile M for L2 reuse"}},
        {"kind": "profile", "data": {"vs_baseline": 1.12, "correct": True, "backend": "triton"}},
    ]
    traj.write_text("\n".join(json.dumps(x) for x in lines))
    entries = mine_trajectory(traj)
    assert len(entries) == 1
    e = entries[0]
    assert e.op == "matmul"
    assert e.decision_kind == "tile"
    assert e.rationale == "tile M for L2 reuse"
    assert e.baseline_ratio == 1.12
    assert e.correct is True
    assert e.backend == "triton"


def test_mine_skips_rationale_less_decisions(tmp_path):
    traj = tmp_path / "t.jsonl"
    lines = [
        {"kind": "header", "data": {"kernel_id": "relu"}},
        {"kind": "decision", "data": {"decision": {"kind": "tile", "params": {}}}},  # no rationale
    ]
    traj.write_text("\n".join(json.dumps(x) for x in lines))
    assert mine_trajectory(traj) == []


def test_to_dict_includes_key():
    e = RationaleEntry(op="silu", decision_kind="vectorize", params={"width": 4}, rationale="lanes")
    d = e.to_dict()
    assert d["key"] == e.key()
    assert d["op"] == "silu" and d["rationale"] == "lanes"


def test_mine_strategy_json_normalizes_rationale_dict(tmp_path):
    from arke.learn.rationale_kb import mine_strategy_json

    (tmp_path / "strategy.json").write_text(json.dumps({
        "kernel_id": "matmul",
        "decisions": [
            {"kind": "tile", "params": {"loop": "M", "factors": [128]},
             "rationale": {"text": "heuristic matmul tile for M", "lang": "en"}},
            {"kind": "reorder", "params": {"order": ["M", "N", "K"]},
             "rationale": "plain string rationale"},
            {"kind": "noop", "params": {}},  # no rationale → skipped
        ],
    }))
    (tmp_path / "trajectory.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"kind": "profile", "data": {"vs_baseline": 0.86}},
        {"kind": "profile", "data": {"vs_baseline": 1.04}},
    ]))
    entries = mine_strategy_json(tmp_path / "strategy.json")
    assert len(entries) == 2  # noop skipped
    assert entries[0].op == "matmul"
    assert entries[0].rationale == "heuristic matmul tile for M"  # dict → text
    assert entries[1].rationale == "plain string rationale"
    # best (max) ratio paired
    assert entries[0].baseline_ratio == 1.04


# ── Read side of the @rationale loop (LT-7): recall ──────────────────────────


def test_normalize_op_strips_kernel_suffix():
    # KB↔registry op-name drift: KB stores relu_kernel, registry uses relu.
    assert _normalize_op("relu_kernel") == "relu"
    assert _normalize_op("grouped_matmul_kernel") == "grouped_matmul"
    assert _normalize_op("matmul") == "matmul"  # no suffix → unchanged
    assert _normalize_op("  add_kernel ") == "add"  # whitespace tolerant


def _seed_kb(tmp_path):
    kb = RationaleKB(tmp_path / "kb.jsonl")
    kb.add_entries([
        RationaleEntry(op="matmul", decision_kind="tile",
                       params={"loop": "M", "factors": [128]},
                       rationale="tile M for L2 reuse", baseline_ratio=1.30, correct=True),
        RationaleEntry(op="matmul", decision_kind="tile",
                       params={"loop": "N", "factors": [64]},
                       rationale="tile N", baseline_ratio=0.90, correct=True),
        RationaleEntry(op="matmul", decision_kind="vectorize",
                       params={"width": 4}, rationale="vec lanes", baseline_ratio=1.10),
        # unmeasured — should sink below measured ones
        RationaleEntry(op="matmul", decision_kind="unroll",
                       params={"loop": "K"}, rationale="unroll K", baseline_ratio=None),
        # different op, stored with _kernel suffix
        RationaleEntry(op="grouped_matmul_kernel", decision_kind="tile",
                       params={"loop": "B", "factors": [128]},
                       rationale="grouped tile", baseline_ratio=1.04),
    ])
    return kb


def test_recall_ranks_by_measured_outcome(tmp_path):
    kb = _seed_kb(tmp_path)
    priors = kb.recall("matmul", top_k=10)
    # highest baseline_ratio first; unmeasured (unroll) last.
    assert priors[0].decision_kind == "tile" and priors[0].baseline_ratio == 1.30
    assert priors[-1].decision_kind == "unroll" and priors[-1].baseline_ratio is None
    assert all(isinstance(p, RationalePrior) for p in priors)


def test_recall_filters_by_decision_kind(tmp_path):
    kb = _seed_kb(tmp_path)
    tiles = kb.recall("matmul", decision_kind="tile", top_k=10)
    assert {p.decision_kind for p in tiles} == {"tile"}
    assert len(tiles) == 2


def test_recall_normalizes_op_name_drift(tmp_path):
    kb = _seed_kb(tmp_path)
    # registry name 'grouped_matmul' must match KB's 'grouped_matmul_kernel'
    priors = kb.recall("grouped_matmul", top_k=5)
    assert len(priors) == 1
    assert priors[0].rationale == "grouped tile"


def test_recall_min_ratio_filters_losers(tmp_path):
    kb = _seed_kb(tmp_path)
    winners = kb.recall("matmul", top_k=10, min_ratio=1.0)
    assert all(p.baseline_ratio is not None and p.baseline_ratio >= 1.0 for p in winners)
    # 0.90 tile-N and unmeasured unroll excluded.
    assert len(winners) == 2


def test_recall_respects_top_k(tmp_path):
    kb = _seed_kb(tmp_path)
    assert len(kb.recall("matmul", top_k=1)) == 1
    assert len(kb.recall("matmul", top_k=2)) == 2


def test_recall_empty_on_missing_op_or_kb(tmp_path):
    kb = _seed_kb(tmp_path)
    assert kb.recall("nonexistent_op") == []
    empty = RationaleKB(tmp_path / "does_not_exist.jsonl")
    assert empty.recall("matmul") == []


def test_recall_dedupes_across_op_name_drift(tmp_path):
    kb = RationaleKB(tmp_path / "kb.jsonl")
    # 'silu' and 'silu_kernel' normalize to the same op but have DISTINCT
    # write-keys (op differs), so both are stored. They share the recall
    # signature (kind, params, rationale) → recall must show them once,
    # keeping the best-ranked copy (write-level dedup can't catch this).
    kb.add_entries([
        RationaleEntry(op="silu", decision_kind="vectorize", params={"width": 4},
                       rationale="lanes", baseline_ratio=1.2),
        RationaleEntry(op="silu_kernel", decision_kind="vectorize", params={"width": 4},
                       rationale="lanes", baseline_ratio=1.5),
    ])
    assert kb.count() == 2  # both written (distinct keys)
    priors = kb.recall("silu", top_k=10)
    assert len(priors) == 1  # collapsed by recall dedupe
    assert priors[0].baseline_ratio == 1.5  # keeps best-ranked copy


def test_list_legal_actions_surfaces_priors_and_never_raises():
    """Tool wiring: priors appear in data; a broken KB never breaks the tool."""
    from arke.agent.env import ArkeEnv
    from arke.agent.tools import ToolRegistry

    reg = ToolRegistry.with_env(ArkeEnv.from_op("matmul"))
    tool = reg.get("list_legal_actions")
    result = tool.execute({"top_n": 5, "filter_kind": "tile"})
    assert result.success
    # candidate generator (legality surface) is always present
    assert "candidates" in result.data
    # priors are advisory: present when the shipped KB has matmul entries
    if "rationale_priors" in result.data:
        for p in result.data["rationale_priors"]:
            assert "decision_kind" in p and "rationale" in p

    # never-raise guarantee: _recall_priors swallows any KB failure
    import arke.learn.rationale_kb as kbmod

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("kb exploded")

    orig = kbmod.RationaleKB
    try:
        kbmod.RationaleKB = _Boom  # type: ignore[assignment]
        r2 = tool.execute({"top_n": 5})
        assert r2.success  # tool still works
        assert "rationale_priors" not in r2.data  # gracefully absent
    finally:
        kbmod.RationaleKB = orig  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Curated write channel — the human-experience half of the loop.               #
# --------------------------------------------------------------------------- #

def test_curated_pattern_stamps_honest_provenance():
    """A hand-authored prior is tagged curated/<slug>, never disguised as mined."""
    e = curated_pattern(
        op="matmul", decision_kind="resource",
        params={"num_warps": 4}, rationale="sm_86 4w wins",
        baseline_ratio=1.072, correct=True, backend="triton",
        slug="sm86-occupancy-8w-to-4w",
    )
    assert e.source == "curated/sm86-occupancy-8w-to-4w"
    assert e.baseline_ratio == 1.072
    assert e.backend == "triton"
    # distinguishable from an auto-mined entry by source prefix
    assert e.source.startswith("curated/")


def test_mine_curated_appends_and_is_idempotent(tmp_path):
    kb_path = tmp_path / "kb.jsonl"
    pats = [
        curated_pattern(op="matmul", decision_kind="resource",
                        params={"num_warps": 4}, rationale="4w wins",
                        baseline_ratio=1.072, slug="sm86"),
        curated_pattern(op="grouped_matmul", decision_kind="resource",
                        params={"num_warps": 4}, rationale="4w wins big",
                        baseline_ratio=1.224, slug="sm86"),
    ]
    r1 = mine_curated(pats, kb_path=kb_path)
    assert r1["entries_written"] == 2 and r1["kb_total"] == 2
    # re-seed writes nothing (dedupe on entry key)
    r2 = mine_curated(pats, kb_path=kb_path)
    assert r2["entries_written"] == 0 and r2["kb_total"] == 2


def test_curated_prior_recalled_and_outranks_unmeasured(tmp_path):
    """Read side surfaces a curated prior and ranks it above unmeasured mined ones."""
    kb_path = tmp_path / "kb.jsonl"
    kb = RationaleKB(kb_path)
    # an auto-mined resource entry with no measured ratio
    kb.add_entries([RationaleEntry(op="matmul", decision_kind="resource",
                                   params={"num_warps": 8}, rationale="seed 8w",
                                   source="01_matmul")])
    # a curated, measured prior for the same op/kind
    mine_curated([curated_pattern(
        op="matmul", decision_kind="resource", params={"num_warps": 4},
        rationale="sm_86: 4w beats 8w (measured +7.2%)",
        baseline_ratio=1.072, correct=True, slug="sm86",
    )], kb_path=kb_path)

    priors = kb.recall("matmul", decision_kind="resource", top_k=3)
    assert priors, "curated prior must be recalled"
    # measured curated prior ranks first; unmeasured mined entry sinks below
    assert priors[0].baseline_ratio == 1.072
    assert priors[0].params.get("num_warps") == 4


def test_curated_pattern_default_no_fabricated_outcome():
    """A qualitative curated pattern leaves baseline_ratio None — no fabrication."""
    e = curated_pattern(op="softmax", decision_kind="tile",
                        params={"BLOCK_N": 256}, rationale="row-scan wide tile")
    assert e.baseline_ratio is None
    assert e.source == "curated/pattern"
