# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for A5 — @rationale trajectory contract audit.

`audit_decision_rationales` is the gate-style complement to S4: it scans a
trajectory JSONL and flags any `decision` record without a non-empty
rationale. Does NOT touch the frozen events.validate_payload contract.
"""

from __future__ import annotations

import json

from arke.learn.trajectory import audit_decision_rationales


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_clean_trajectory_has_no_violations(tmp_path):
    p = tmp_path / "traj.jsonl"
    _write_jsonl(p, [
        {"kind": "header", "data": {"contract_id": "x"}},
        {"kind": "decision", "data": {"decision": {}, "rationale": "warp-friendly tile", "step": 1}},
        {"kind": "compile", "data": {"success": True}},
        {"kind": "decision", "data": {"decision": {}, "rationale": "fuse epilogue", "step": 2}},
        {"kind": "done", "data": {"final_score": 1.0}},
    ])
    assert audit_decision_rationales(p) == []


def test_missing_rationale_flagged(tmp_path):
    p = tmp_path / "traj.jsonl"
    _write_jsonl(p, [
        {"kind": "decision", "data": {"decision": {}, "step": 1}},  # no rationale
    ])
    v = audit_decision_rationales(p)
    assert len(v) == 1 and "step 1" in v[0]


def test_empty_rationale_flagged(tmp_path):
    p = tmp_path / "traj.jsonl"
    _write_jsonl(p, [
        {"kind": "decision", "data": {"decision": {}, "rationale": "   ", "step": 3}},
    ])
    v = audit_decision_rationales(p)
    assert len(v) == 1 and "step 3" in v[0]


def test_non_decision_records_ignored(tmp_path):
    p = tmp_path / "traj.jsonl"
    _write_jsonl(p, [
        {"kind": "compile", "data": {"success": True}},     # no rationale needed
        {"kind": "profile", "data": {"latency_ms": 1.0}},
        {"kind": "fallback", "data": {"layer": "provider"}},
    ])
    assert audit_decision_rationales(p) == []


def test_missing_file_reported(tmp_path):
    v = audit_decision_rationales(tmp_path / "nope.jsonl")
    assert len(v) == 1 and "not found" in v[0]
