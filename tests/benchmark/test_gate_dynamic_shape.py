# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the D2 dynamic-shape soft gate (benchmarks/gate_dynamic_shape.py).

D2 approved by Leon 2026-07-30 ("D推进D2并完成依赖"): same_spec_geomean <= 5x
AND new-spec prediction consistency. The threshold lives in the GATE module,
never in the measurement track (test_no_gate_threshold_in_module guards that).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.gate_dynamic_shape import (
    SAME_SPEC_GEOMEAN_LIMIT,
    evaluate_op,
    evaluate_run,
    main,
)


def _row(op: str, tag: str, spec: str, new: bool, cliff: float,
         status: str = "ok") -> dict:
    return {
        "op": op, "shape_tag": tag, "M": "1", "N": "1", "K": "0",
        "spec_key": spec, "new_spec": str(new), "first_call_ms": "1",
        "steady_ms": "1", "cliff_ratio": str(cliff), "status": status,
        "reason": "",
    }


# ── frozen parameter ─────────────────────────────────────────────────────────


def test_frozen_limit_value() -> None:
    """5x is the Leon-approved (2026-07-30) frozen D2 parameter."""
    assert SAME_SPEC_GEOMEAN_LIMIT == 5.0


def test_track_module_still_threshold_free() -> None:
    """The gate must live OUTSIDE the measurement track: dynamic_shape.py
    stays threshold-free (its own guard also enforces this)."""
    src = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "dynamic_shape.py"
    ).read_text(encoding="utf-8")
    assert "SAME_SPEC_GEOMEAN_LIMIT" not in src
    assert "gate_dynamic_shape" not in src


# ── evaluate_op: the three D2 checks ─────────────────────────────────────────


def test_pass_when_warm_shapes_are_warm() -> None:
    rows = [
        _row("softmax", "n256", "s1", True, 120.0),   # new spec: cliff OK
        _row("softmax", "n300", "s1", False, 1.2),    # warm shape, no cliff
        _row("softmax", "n512", "s2", True, 90.0),
        _row("softmax", "n480", "s2", False, 2.0),
    ]
    v = evaluate_op(rows)
    assert v.passed, v.reasons
    assert v.n_new_spec == 2 and v.n_same_spec == 2
    assert v.same_spec_geomean == pytest.approx((1.2 * 2.0) ** 0.5)


def test_fail_when_predicted_warm_pays_compile() -> None:
    """Check 1: same-spec geomean over the frozen 5x limit -> FAIL
    (accidental despecialization symptom (a))."""
    rows = [
        _row("softmax", "n256", "s1", True, 120.0),
        _row("softmax", "n300", "s1", False, 130.0),  # predicted warm but compiled!
        _row("softmax", "n480", "s1", False, 110.0),
    ]
    v = evaluate_op(rows)
    assert not v.passed
    assert any("same_spec_geomean" in r for r in v.reasons)


def test_fail_on_new_spec_flag_mismatch() -> None:
    """Check 2a: recorded new_spec disagrees with spec_key first-occurrence
    recount -> FAIL (spec_key regression symptom (b))."""
    rows = [
        _row("matmul", "a", "k1", True, 50.0),
        _row("matmul", "b", "k1", True, 1.0),  # k1 already seen: flag lies
    ]
    v = evaluate_op(rows)
    assert not v.passed
    assert any("recount" in r for r in v.reasons)


def test_fail_on_summary_count_mismatch() -> None:
    """Check 2b: summary.json counts must match the rows."""
    rows = [
        _row("rmsnorm", "a", "k1", True, 50.0),
        _row("rmsnorm", "b", "k1", False, 1.0),
    ]
    v = evaluate_op(rows, summary={"n_new_spec": 2, "n_same_spec": 0})
    assert not v.passed
    assert any("n_new_spec" in r or "n_same_spec" in r for r in v.reasons)


def test_non_ok_rows_excluded_from_geomean() -> None:
    """OOM/error rows carry NaN ratios and must not poison the verdict."""
    rows = [
        _row("matmul", "a", "k1", True, 60.0),
        _row("matmul", "b", "k1", False, 1.5),
        _row("matmul", "c", "k2", True, float("nan"), status="oom"),
    ]
    v = evaluate_op(rows)
    assert v.passed, v.reasons
    assert v.n_same_spec == 1


# ── evaluate_run + CLI on a synthetic run dir ────────────────────────────────


def _write_run(tmp_path: Path, rows: list[dict], summary_ops: dict) -> Path:
    import csv as _csv
    run = tmp_path / "run1"
    run.mkdir()
    with (run / "softmax_cliff.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (run / "summary.json").write_text(
        json.dumps({"dtype": "float16", "warm_reps": 50, "ops": summary_ops}),
        encoding="utf-8",
    )
    return run


def test_evaluate_run_and_cli_pass(tmp_path: Path) -> None:
    rows = [
        _row("softmax", "n256", "s1", True, 100.0),
        _row("softmax", "n300", "s1", False, 1.1),
    ]
    run = _write_run(tmp_path, rows, {"softmax": {"n_new_spec": 1, "n_same_spec": 1}})
    verdicts = evaluate_run(run)
    assert len(verdicts) == 1 and verdicts[0].passed
    assert main([str(run)]) == 0


def test_cli_fails_on_bad_run(tmp_path: Path) -> None:
    rows = [
        _row("softmax", "n256", "s1", True, 100.0),
        _row("softmax", "n300", "s1", False, 90.0),  # warm shape compiled
    ]
    run = _write_run(tmp_path, rows, {"softmax": {"n_new_spec": 1, "n_same_spec": 1}})
    assert main([str(run)]) == 1


def test_cli_fails_on_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 1
