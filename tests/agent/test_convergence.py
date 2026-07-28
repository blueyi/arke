# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for arke.agent.convergence (KESTREL K-H5.2).

These tests validate the *pure* convergence-curve extraction — no GPU, no
LLM, no live run. They feed synthetic trajectories mirroring the exact
shape ``LLMRunner`` produces and assert the projection contract:

  1. Only ``compile_and_profile`` events become rows.
  2. Iteration index is 1-based and dense across surviving events.
  3. ``best_so_far_ratio`` is monotone non-decreasing.
  4. Failed / incorrect / unverified events do NOT advance the running best.
  5. ``vs_default`` (inverted) is preferred over ``baseline_ratio`` when both
     are present — vs_default is the gate criterion.
  6. CSV round-trip matches :data:`CONVERGENCE_COLUMNS` exactly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from arke.agent.convergence import (
    CONVERGENCE_COLUMNS,
    build_convergence_rows,
    emit_convergence_csv,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _profile_event(
    step: int,
    *,
    success: bool = True,
    correct: bool | None = True,
    latency_ms: float | None = 1.0,
    baseline_ratio: float | None = None,
    vs_default: float | None = None,
    backend: str = "triton",
    max_diff: float | None = 1e-4,
    meas_spread: float | None = 0.02,
) -> dict:
    """Build a synthetic compile_and_profile trajectory event.

    Payload layout mirrors ArkeTool.to_json + CompileAndProfileTool.execute
    output (see arke/agent/tools.py).
    """
    data: dict = {
        "backend": backend,
        "correct": correct,
        "max_diff": max_diff,
        "latency_ms": latency_ms,
        "baseline_ratio": baseline_ratio,
        "vs_default": vs_default,
        "meas_spread": meas_spread,
    }
    return {
        "type": "action",
        "step": step,
        "tool": "compile_and_profile",
        "params": {"op_name": "matmul"},
        "result": {"success": success, "data": data},
    }


def _noop_event(step: int, tool: str) -> dict:
    """A non-profile trajectory event (should be ignored)."""
    return {
        "type": "action",
        "step": step,
        "tool": tool,
        "params": {},
        "result": {"success": True, "data": {"ok": True}},
    }


# ── Filter contract ────────────────────────────────────────────────────────

def test_only_compile_and_profile_events_become_rows() -> None:
    traj = [
        _noop_event(1, "list_legal_actions"),
        _profile_event(2, baseline_ratio=1.2),
        _noop_event(3, "apply_decision"),
        _profile_event(4, baseline_ratio=1.5),
        _noop_event(5, "checkpoint"),
    ]
    rows = build_convergence_rows(traj)
    assert [r["iteration"] for r in rows] == [1, 2]
    assert all(r["tool"] == "compile_and_profile" for r in rows)


def test_empty_trajectory_returns_empty() -> None:
    assert build_convergence_rows([]) == []


def test_no_profile_events_returns_empty() -> None:
    traj = [_noop_event(1, "list_legal_actions"), _noop_event(2, "apply_decision")]
    assert build_convergence_rows(traj) == []


def test_iteration_dense_and_one_based() -> None:
    traj = [
        _profile_event(10, baseline_ratio=1.0),
        _profile_event(20, baseline_ratio=1.1),
        _profile_event(30, baseline_ratio=0.9),
    ]
    rows = build_convergence_rows(traj)
    assert [r["iteration"] for r in rows] == [1, 2, 3]
    assert [r["step"] for r in rows] == [10, 20, 30]


# ── Best-so-far monotonicity ───────────────────────────────────────────────

def test_best_so_far_monotone_non_decreasing_with_baseline_ratio() -> None:
    traj = [
        _profile_event(1, baseline_ratio=1.0),
        _profile_event(2, baseline_ratio=1.5),
        _profile_event(3, baseline_ratio=1.2),  # regression
        _profile_event(4, baseline_ratio=2.0),  # new best
        _profile_event(5, baseline_ratio=1.8),
    ]
    best = [r["best_so_far_ratio"] for r in build_convergence_rows(traj)]
    assert best == [1.0, 1.5, 1.5, 2.0, 2.0]
    # Explicit monotone check for future-proofing
    for prev, curr in zip(best, best[1:]):
        assert curr >= prev


def test_failed_event_does_not_advance_best() -> None:
    traj = [
        _profile_event(1, baseline_ratio=1.2),
        _profile_event(2, success=False, baseline_ratio=999.0),  # compile error
        _profile_event(3, baseline_ratio=1.3),
    ]
    rows = build_convergence_rows(traj)
    assert [r["best_so_far_ratio"] for r in rows] == [1.2, 1.2, 1.3]
    # Failed row is still emitted (audit trail)
    assert rows[1]["success"] is False


def test_incorrect_event_does_not_advance_best() -> None:
    traj = [
        _profile_event(1, baseline_ratio=1.2),
        _profile_event(2, correct=False, baseline_ratio=5.0),  # numerically wrong
        _profile_event(3, baseline_ratio=1.4),
    ]
    rows = build_convergence_rows(traj)
    assert [r["best_so_far_ratio"] for r in rows] == [1.2, 1.2, 1.4]


def test_unverified_event_does_not_advance_best() -> None:
    """correct=None (V1 skipped) is not eligible — we require verified-correct."""
    traj = [
        _profile_event(1, baseline_ratio=1.0),
        _profile_event(2, correct=None, baseline_ratio=5.0),
        _profile_event(3, baseline_ratio=1.1),
    ]
    rows = build_convergence_rows(traj)
    assert [r["best_so_far_ratio"] for r in rows] == [1.0, 1.0, 1.1]


def test_first_row_with_missing_metric_leaves_best_none() -> None:
    """Mock backend / verify-only: no ratio available → best_so_far stays None."""
    traj = [
        _profile_event(1, latency_ms=None, baseline_ratio=None, vs_default=None),
        _profile_event(2, baseline_ratio=1.5),
    ]
    rows = build_convergence_rows(traj)
    assert rows[0]["best_so_far_ratio"] is None
    assert rows[0]["current_ratio"] is None
    assert rows[1]["best_so_far_ratio"] == pytest.approx(1.5)


# ── Metric selection: vs_default preferred ─────────────────────────────────

def test_vs_default_preferred_over_baseline_ratio() -> None:
    """When both are present, current_ratio comes from inverted vs_default.

    vs_default = 0.5 means agent is 2x faster than the default kernel →
    ratio-space metric = 1/0.5 = 2.0. baseline_ratio (vs eager) is ignored.
    """
    traj = [
        _profile_event(1, baseline_ratio=10.0, vs_default=0.5),
    ]
    rows = build_convergence_rows(traj)
    assert rows[0]["current_ratio"] == pytest.approx(2.0)
    assert rows[0]["best_so_far_ratio"] == pytest.approx(2.0)


def test_baseline_ratio_fallback_when_vs_default_missing() -> None:
    traj = [_profile_event(1, baseline_ratio=1.7, vs_default=None)]
    rows = build_convergence_rows(traj)
    assert rows[0]["current_ratio"] == pytest.approx(1.7)


def test_zero_or_negative_vs_default_falls_back() -> None:
    """Pathological vs_default values fall back to baseline_ratio."""
    traj = [
        _profile_event(1, baseline_ratio=1.3, vs_default=0.0),
        _profile_event(2, baseline_ratio=1.5, vs_default=-0.1),
    ]
    rows = build_convergence_rows(traj)
    assert rows[0]["current_ratio"] == pytest.approx(1.3)
    assert rows[1]["current_ratio"] == pytest.approx(1.5)


# ── CSV round-trip ─────────────────────────────────────────────────────────

def test_emit_csv_columns_match_public_contract(tmp_path: Path) -> None:
    traj = [_profile_event(1, baseline_ratio=1.2)]
    out = tmp_path / "convergence.csv"
    written = emit_convergence_csv(traj, out)
    assert written == 1
    assert out.exists()
    with out.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert tuple(header) == CONVERGENCE_COLUMNS


def test_emit_csv_empty_trajectory_writes_header_only(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    written = emit_convergence_csv([], out)
    assert written == 0
    assert out.exists()
    with out.open() as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 1  # header only


def test_emit_csv_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "path" / "curve.csv"
    written = emit_convergence_csv([_profile_event(1, baseline_ratio=1.0)], out)
    assert written == 1
    assert out.exists()


def test_emit_csv_row_content_matches_build(tmp_path: Path) -> None:
    """The CSV must equal build_convergence_rows() output row-for-row."""
    traj = [
        _profile_event(1, baseline_ratio=1.2, vs_default=0.9, latency_ms=2.5),
        _profile_event(2, success=False, baseline_ratio=None),
        _profile_event(3, baseline_ratio=1.5, correct=True),
    ]
    expected = build_convergence_rows(traj)
    out = tmp_path / "curve.csv"
    emit_convergence_csv(traj, out)
    with out.open() as fh:
        reader = csv.DictReader(fh)
        got = list(reader)
    assert len(got) == len(expected)
    for exp, actual in zip(expected, got):
        # CSV always stringifies; compare via string coercion
        assert actual["iteration"] == str(exp["iteration"])
        assert actual["success"] == str(exp["success"])
        # None → empty string in csv
        if exp["latency_ms"] is None:
            assert actual["latency_ms"] == ""
        else:
            assert float(actual["latency_ms"]) == pytest.approx(exp["latency_ms"])
