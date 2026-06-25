# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the @rationale KB (G9[3], D8-A4)."""

from __future__ import annotations

import json
from pathlib import Path

from arke.learn.rationale_kb import (
    RationaleEntry,
    RationaleKB,
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
