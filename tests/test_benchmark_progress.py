# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks.progress (resume / incremental persistence)."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from benchmarks import progress as p


def test_compute_fingerprint_stable_on_reorder():
    a = {"ops": ["matmul", "softmax"], "warmup": 200, "reps": 500}
    b = {"ops": ["softmax", "matmul"], "warmup": 200, "reps": 500}
    assert p.compute_fingerprint(a) == p.compute_fingerprint(b)


def test_compute_fingerprint_changes_on_value():
    a = {"ops": ["matmul"], "warmup": 200, "reps": 500}
    b = {"ops": ["matmul"], "warmup": 100, "reps": 500}
    assert p.compute_fingerprint(a) != p.compute_fingerprint(b)


def test_validate_config_creates_when_missing(tmp_path: Path):
    cfg = {"ops": ["matmul"], "warmup": 200}
    check = p.validate_config(tmp_path, cfg)
    assert check.compatible is True
    assert check.stored_fingerprint == ""
    assert check.current_fingerprint != ""


def test_validate_config_detects_drift(tmp_path: Path):
    cfg = {"ops": ["matmul"], "warmup": 200}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    check_same = p.validate_config(tmp_path, cfg)
    assert check_same.compatible is True
    cfg2 = dict(cfg, warmup=100)
    check_diff = p.validate_config(tmp_path, cfg2)
    assert check_diff.compatible is False
    assert "fingerprint" in check_diff.reason
    check_force = p.validate_config(tmp_path, cfg2, force=True)
    assert check_force.compatible is True


def test_acquire_release_lock(tmp_path: Path):
    p.acquire_lock(tmp_path, layer="l2")
    info = p.lock_status(tmp_path)
    assert info is not None
    assert info["pid"] == os.getpid()
    assert info["alive"] is True
    p.release_lock(tmp_path)
    assert p.lock_status(tmp_path) is None


def test_acquire_lock_rejects_live_holder(tmp_path: Path):
    # Forge a lock owned by current process to simulate a live holder.
    info = p.LockInfo(pid=os.getpid(), started_at=0.0, host="x", layer="l2")
    (tmp_path / p.LOCK_NAME).write_text(info.to_json())
    # acquire_lock should treat self-PID as ok (re-acquire), so we test by
    # forging a parent-process pid (1) which is alive on POSIX.
    info_other = p.LockInfo(pid=1, started_at=0.0, host="x", layer="l2")
    (tmp_path / p.LOCK_NAME).write_text(info_other.to_json())
    with pytest.raises(RuntimeError):
        p.acquire_lock(tmp_path, layer="l2")
    # force=True overrides
    p.acquire_lock(tmp_path, layer="l2", force=True)


def test_acquire_lock_ignores_dead_pid(tmp_path: Path):
    info = p.LockInfo(pid=99999999, started_at=0.0, host="x", layer="l2")
    (tmp_path / p.LOCK_NAME).write_text(info.to_json())
    p.acquire_lock(tmp_path, layer="l2")  # should succeed silently
    p.release_lock(tmp_path)


def test_append_row_creates_header_and_appends(tmp_path: Path):
    csv_path = tmp_path / "x_results.csv"
    fields = ["op", "shape_tag", "baseline", "status"]
    p.append_row(csv_path, fields, {"op": "matmul", "shape_tag": "tiny", "baseline": "P0", "status": "ok"})
    p.append_row(csv_path, fields, {"op": "matmul", "shape_tag": "big",  "baseline": "P0", "status": "oom"})
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "oom"


def test_should_skip_policies():
    ok = {"status": "ok"}
    oom = {"status": "oom"}
    err = {"status": "error"}
    # auto: success+permanent skipped, retryable retried
    assert p.should_skip(ok, p.RETRY_POLICY_AUTO) is True
    assert p.should_skip(oom, p.RETRY_POLICY_AUTO) is True
    assert p.should_skip(err, p.RETRY_POLICY_AUTO) is False
    # none: skip everything that has a row
    assert p.should_skip(err, p.RETRY_POLICY_NONE) is True
    # all: retry everything except already-success
    assert p.should_skip(ok, p.RETRY_POLICY_ALL) is True
    assert p.should_skip(oom, p.RETRY_POLICY_ALL) is False


def test_index_rows_dedupes_on_key(tmp_path: Path):
    rows = [
        {"op": "x", "shape_tag": "t", "baseline": "P0", "status": "error"},
        {"op": "x", "shape_tag": "t", "baseline": "P0", "status": "ok"},  # later wins
    ]
    indexed = p.index_rows(rows, ("op", "shape_tag", "baseline"))
    assert len(indexed) == 1
    assert indexed[("x", "t", "P0")]["status"] == "ok"


def test_summarize_csv(tmp_path: Path):
    csv_path = tmp_path / "x_results.csv"
    fields = ["op", "shape_tag", "baseline", "status"]
    p.append_row(csv_path, fields, {"op": "x", "shape_tag": "a", "baseline": "P0", "status": "ok"})
    p.append_row(csv_path, fields, {"op": "x", "shape_tag": "b", "baseline": "P0", "status": "oom"})
    p.append_row(csv_path, fields, {"op": "x", "shape_tag": "c", "baseline": "P0", "status": "error"})
    summary = p.summarize_csv(csv_path, ("op", "shape_tag", "baseline"))
    assert summary["rows"] == 3
    assert summary["ok"] == 1
    assert summary["permanent_failure"] == 1
    assert summary["retryable_failure"] == 1


@pytest.mark.parametrize(
    "raw,expected_tail",
    [
        ("benchmarks/results", ("benchmarks", "results")),
        ("benchmarks/results/phase1/stage7/track6", ("benchmarks", "results")),
        ("benchmarks/results/phase1/stage7/track6/l2", ("benchmarks", "results")),
        ("results/phase1/stage7", ("results", "phase1", "stage7")),  # only suffix-track stripped
    ],
)
def test_normalize_output_root_strips_phase_stage_track(raw, expected_tail):
    out = p.normalize_output_root(raw, phase=1, stage=7, track=6, layer="l2")
    assert out.parts == expected_tail


def test_progress_tracker_emits_jsonl(tmp_path: Path):
    tr = p.ProgressTracker(base_dir=tmp_path, layer="l2", config_fingerprint="abc")
    tr.emit("measurement", op="matmul", shape_tag="tiny", status="ok")
    tr.emit("op_done", op="matmul", new=1)
    log = (tmp_path / p.PROGRESS_LOG_NAME).read_text().strip().splitlines()
    assert len(log) == 2
    assert json.loads(log[0])["event"] == "measurement"


def test_progress_tracker_snapshot(tmp_path: Path):
    tr = p.ProgressTracker(base_dir=tmp_path, layer="l1", config_fingerprint="xyz")
    tr.snapshot({"per_op": {"matmul": {"rows": 10}}})
    snap = json.loads((tmp_path / p.STATUS_SNAPSHOT_NAME).read_text())
    assert snap["layer"] == "l1"
    assert snap["per_op"]["matmul"]["rows"] == 10
