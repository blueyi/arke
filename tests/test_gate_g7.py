# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import benchmarks.gate_g7 as gate_g7
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


def test_check_bl5_coverage_evidence_accepts_complete_synthetic_report(monkeypatch):
    coverage = {
        "stage": "S7",
        "gate": "G7",
        "l1": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 1,
            "shapes_required_total": 1,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 1.0,
            "per_op": [
                {
                    "op": "relu",
                    "missing_shape_tags": [],
                    "missing_perf_target_fields": [],
                }
            ],
        },
        "l2": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 1,
            "shapes_required_total": 1,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 1.0,
            "per_op": [
                {
                    "op": "matmul_relu",
                    "missing_shape_tags": [],
                    "missing_perf_target_fields": [],
                }
            ],
        },
    }
    audit = {
        "summary": {
            "l1": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": [],
                "unsupported_surface_cases": [],
            },
            "l2": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": [],
                "unsupported_surface_cases": [],
            },
        }
    }

    def fake_load_json_artifact(path: Path):
        if path.name == "coverage_gap.json":
            return coverage, None
        if path.name == "audit_report.json":
            return audit, None
        return None, f"unexpected artifact: {path}"

    monkeypatch.setattr(gate_g7, "_load_json_artifact", fake_load_json_artifact)

    ok, detail = gate_g7._check_bl5_coverage_evidence(Path("/tmp/unused"))

    assert ok is True
    assert "complete" in detail


def test_check_bl5_coverage_evidence_rejects_gap(monkeypatch):
    coverage = {
        "stage": "S7",
        "gate": "G7",
        "l1": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 0,
            "shapes_required_total": 2,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 0.5,
            "per_op": [
                {
                    "op": "relu",
                    "missing_shape_tags": ["shape-b"],
                    "missing_perf_target_fields": [],
                }
            ],
        },
        "l2": {
            "ops_total": 0,
            "ops_with_any_evidence": 0,
            "ops_fully_covered": 0,
            "shapes_required_total": 0,
            "shapes_observed_total": 0,
            "shape_coverage_ratio": 0.0,
            "per_op": [],
        },
    }
    audit = {
        "summary": {
            "l1": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": ["relu"],
                "unsupported_surface_cases": [],
            },
            "l2": {
                "missing_examples": [],
                "missing_strategy_examples": [],
                "missing_benchmark_evidence": [],
                "missing_full_shape_evidence": [],
                "unsupported_surface_cases": [],
            },
        }
    }

    def fake_load_json_artifact(path: Path):
        if path.name == "coverage_gap.json":
            return coverage, None
        if path.name == "audit_report.json":
            return audit, None
        return None, f"unexpected artifact: {path}"

    monkeypatch.setattr(gate_g7, "_load_json_artifact", fake_load_json_artifact)

    ok, detail = gate_g7._check_bl5_coverage_evidence(Path("/tmp/unused"))

    assert ok is False
    assert "full-shape coverage" in detail or "missing_full_shape_evidence" in detail


def test_check_bl5_correctness_evidence_respects_memory_exclusions(monkeypatch):
    rows = [
        (
            "l1",
            {
                "operator": "relu",
                "shape_tag": "square-1k",
                "status": "ok",
                "correctness_status": "ok",
                "allclose": "true",
                "memory_policy": "",
                "reason": "",
                "correctness_reason": "",
            },
        ),
        (
            "l1",
            {
                "operator": "flash_attention",
                "shape_tag": "tiny",
                "status": "skipped",
                "correctness_status": "skipped",
                "allclose": "",
                "memory_policy": "attention",
                "memory_bytes_required": "2048",
                "memory_bytes_budget": "1024",
                "memory_ratio": "2.0",
                "reason": "OOM preflight",
                "correctness_reason": "OOM preflight",
            },
        ),
        (
            "l2",
            {
                "operator": "matmul_relu",
                "shape_tag": "gpt2-sm",
                "status": "ok",
                "correctness_status": "ok",
                "allclose": "true",
                "memory_policy": "",
                "reason": "",
                "correctness_reason": "",
            },
        ),
    ]
    monkeypatch.setattr(gate_g7, "_iter_perf_rows", lambda _root: (rows, []))

    ok, detail = gate_g7._check_bl5_correctness_evidence(Path("/tmp/unused"))

    assert ok is True
    assert "memory_excluded=1" in detail


def test_check_bl5_performance_evidence_enforces_group_targets(monkeypatch):
    rows = [
        ("l1", {"operator": "relu", "shape_tag": "square-1k", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "gelu", "shape_tag": "square-1k", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "matmul", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "softmax", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "rope", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l1", {"operator": "flash_attention", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "matmul_relu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "matmul_gelu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "swiglu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "geglu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "linear_ce", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "qkv_fa", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "true", "memory_policy": "", "reason": "", "correctness_reason": ""}),
    ]
    monkeypatch.setattr(gate_g7, "_iter_perf_rows", lambda _root: (rows, []))
    monkeypatch.setattr(
        gate_g7,
        "_load_l1_ot_map",
        lambda _matrix: {
            "relu": 0,
            "gelu": 0,
            "matmul": 2,
            "softmax": 1,
            "rope": 3,
            "flash_attention": 4,
        },
    )

    ok, detail = gate_g7._check_bl5_performance_evidence(Path("/tmp/unused"))

    assert ok is True
    assert "weighted_score=1.0000" in detail
    assert "L2 fusions=6" in detail


def test_check_bl5_performance_evidence_rejects_gap(monkeypatch):
    rows = [
        ("l1", {"operator": "relu", "shape_tag": "square-1k", "status": "ok", "correctness_status": "ok", "perf_pass": "false", "memory_policy": "", "reason": "", "correctness_reason": ""}),
        ("l2", {"operator": "matmul_relu", "shape_tag": "gpt2-sm", "status": "ok", "correctness_status": "ok", "perf_pass": "false", "memory_policy": "", "reason": "", "correctness_reason": ""}),
    ]
    monkeypatch.setattr(gate_g7, "_iter_perf_rows", lambda _root: (rows, []))
    monkeypatch.setattr(gate_g7, "_load_l1_ot_map", lambda _matrix: {"relu": 0})

    ok, detail = gate_g7._check_bl5_performance_evidence(Path("/tmp/unused"))

    assert ok is False
    assert "weighted_score" in detail


def test_run_g7_returns_standard_gate_summary(monkeypatch):
    """Guard the G7 runner against drifting from benchmarks.gate.GateSummary."""
    import benchmarks.gate_g7 as gate_g7

    monkeypatch.setattr(gate_g7, "_check_spec_docs", lambda: (True, "docs ok"))
    monkeypatch.setattr(gate_g7, "_run_pytest", lambda _args: (True, "tests ok"))
    monkeypatch.setattr(gate_g7, "_check_all_examples_compile", lambda: (True, "examples ok"))
    monkeypatch.setattr(gate_g7, "_check_benchmark_artifacts", lambda: (True, "artifacts ok"))
    monkeypatch.setattr(gate_g7, "_check_bl5_coverage_evidence", lambda: (True, "coverage ok"))
    monkeypatch.setattr(gate_g7, "_check_bl5_correctness_evidence", lambda: (True, "correctness ok"))
    monkeypatch.setattr(gate_g7, "_check_bl5_performance_evidence", lambda: (True, "performance ok"))

    summary = run_g7(tier=1)

    assert summary.gate == "G7"
    assert summary.tier == 1
    assert summary.total == len(summary.results)
    assert summary.passed == summary.total
    assert summary.failed == 0
    assert summary.to_dict()["pass_rate"] == "100.0%"
    criteria = {result.criterion for result in summary.results}
    assert {"G7.8b", "G7.8c", "G7.8d"}.issubset(criteria)
