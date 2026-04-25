# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from benchmarks.gate_g7 import check_stage7_track6_artifacts, run_g7


REQUIRED_ROOT_ARTIFACTS = (
    "coverage_gap.json",
    "audit_report.json",
    "stage7_operator_shape_stats.json",
    "dashboard.json",
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _build_track6_tree(root: Path) -> None:
    for layer in ("l1", "l2"):
        layer_dir = root / layer
        _touch(layer_dir / "config.json")
        _touch(layer_dir / "hardware.json")
        _touch(layer_dir / "sources.json")
        _touch(layer_dir / "summary.json")
        _touch(layer_dir / "PERF_ALL.csv")
    for artifact in REQUIRED_ROOT_ARTIFACTS:
        _touch(root / artifact)


def test_check_stage7_track6_artifacts_accepts_complete_tree(tmp_path: Path):
    _build_track6_tree(tmp_path)

    ok, detail = check_stage7_track6_artifacts(tmp_path)

    assert ok is True
    assert "dashboard.json" in detail
    assert "coverage_gap.json" in detail


def test_check_stage7_track6_artifacts_requires_root_dashboard_artifacts(tmp_path: Path):
    _build_track6_tree(tmp_path)
    (tmp_path / "dashboard.json").unlink()

    ok, detail = check_stage7_track6_artifacts(tmp_path)

    assert ok is False
    assert "dashboard.json" in detail


def test_run_g7_returns_standard_gate_summary(monkeypatch):
    """Guard the G7 runner against drifting from benchmarks.gate.GateSummary."""
    import benchmarks.gate_g7 as gate_g7

    monkeypatch.setattr(gate_g7, "_check_spec_docs", lambda: (True, "docs ok"))
    monkeypatch.setattr(gate_g7, "_run_pytest", lambda _args: (True, "tests ok"))
    monkeypatch.setattr(gate_g7, "_check_all_examples_compile", lambda: (True, "examples ok"))
    monkeypatch.setattr(gate_g7, "_check_benchmark_artifacts", lambda: (True, "artifacts ok"))

    summary = run_g7(tier=1)

    assert summary.gate == "G7"
    assert summary.tier == 1
    assert summary.total == len(summary.results)
    assert summary.passed == summary.total
    assert summary.failed == 0
    assert summary.to_dict()["pass_rate"] == "100.0%"
