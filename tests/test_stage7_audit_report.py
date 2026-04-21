# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks.stage7_audit_report import build_stage7_audit_report


def test_build_stage7_audit_report_summarizes_surface_and_benchmark_gaps(tmp_path: Path):
    ledger = {
        "stage": "S7",
        "gate": "G7",
        "summary": {
            "l1": {"entries": 2, "with_examples": 2, "with_strategy_examples": 1, "with_benchmark_evidence": 1, "with_full_required_shape_evidence": 0},
            "l2": {"entries": 2, "with_examples": 1, "with_strategy_examples": 1, "with_benchmark_evidence": 0, "with_full_required_shape_evidence": 0},
        },
        "l1": [
            {
                "op": "relu",
                "example": {"found": True, "relative_path": "operators/00_relu.ak"},
                "pipeline": {"strategy_ok": True, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": ["square-1k"], "missing_shape_tags": ["gpt2-hidden"], "correctness_evidence_present": True, "performance_evidence_present": True},
            },
            {
                "op": "gelu",
                "example": {"found": True, "relative_path": "operators/03_gelu.ak"},
                "pipeline": {"strategy_ok": False, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": [], "missing_shape_tags": ["square-1k"], "correctness_evidence_present": False, "performance_evidence_present": False},
            },
        ],
        "l2": [
            {
                "op": "matmul_gelu",
                "example": {"found": True, "relative_path": "operators/05_matmul_gelu.ak"},
                "pipeline": {"strategy_ok": True, "has_fusion_decision": True},
                "evidence": {"observed_shape_tags": [], "missing_shape_tags": ["gpt2-ffn"], "correctness_evidence_present": False, "performance_evidence_present": False},
            },
            {
                "op": "qkv_fa",
                "example": {"found": False, "relative_path": None},
                "pipeline": {"strategy_ok": False, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": [], "missing_shape_tags": ["llama-long"], "correctness_evidence_present": False, "performance_evidence_present": False},
            },
        ],
    }
    ledger_path = tmp_path / "coverage_ledger.json"
    ledger_path.write_text(json.dumps(ledger))

    report = build_stage7_audit_report(ledger_path)

    assert report["summary"]["l1"]["entries"] == 2
    assert report["summary"]["l2"]["missing_examples"] == ["qkv_fa"]
    assert report["summary"]["l1"]["missing_strategy_examples"] == ["gelu"]
    assert report["summary"]["l2"]["missing_benchmark_evidence"] == ["matmul_gelu", "qkv_fa"]
    assert report["summary"]["l2"]["missing_full_shape_evidence"] == ["matmul_gelu", "qkv_fa"]

    assert report["priority_actions"][0]["category"] == "missing_l2_examples"
    assert report["priority_actions"][0]["ops"] == ["qkv_fa"]


def test_build_stage7_audit_report_cli_summary_text(tmp_path: Path):
    ledger = {
        "stage": "S7",
        "gate": "G7",
        "summary": {
            "l1": {"entries": 1, "with_examples": 1, "with_strategy_examples": 1, "with_benchmark_evidence": 0, "with_full_required_shape_evidence": 0},
            "l2": {"entries": 1, "with_examples": 0, "with_strategy_examples": 0, "with_benchmark_evidence": 0, "with_full_required_shape_evidence": 0},
        },
        "l1": [
            {
                "op": "relu",
                "example": {"found": True, "relative_path": "operators/00_relu.ak"},
                "pipeline": {"strategy_ok": True, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": [], "missing_shape_tags": ["square-1k"], "correctness_evidence_present": False, "performance_evidence_present": False},
            }
        ],
        "l2": [
            {
                "op": "linear_ce",
                "example": {"found": False, "relative_path": None},
                "pipeline": {"strategy_ok": False, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": [], "missing_shape_tags": ["gpt2-ffn"], "correctness_evidence_present": False, "performance_evidence_present": False},
            }
        ],
    }
    ledger_path = tmp_path / "coverage_ledger.json"
    ledger_path.write_text(json.dumps(ledger))

    report = build_stage7_audit_report(ledger_path)
    text = report["text_summary"]

    assert "missing L2 examples" in text
    assert "linear_ce" in text


def test_stage7_audit_report_cli_writes_output(tmp_path: Path):
    ledger = {
        "stage": "S7",
        "gate": "G7",
        "summary": {
            "l1": {"entries": 1, "with_examples": 1, "with_strategy_examples": 1, "with_benchmark_evidence": 1, "with_full_required_shape_evidence": 1},
            "l2": {"entries": 1, "with_examples": 0, "with_strategy_examples": 0, "with_benchmark_evidence": 0, "with_full_required_shape_evidence": 0},
        },
        "l1": [
            {
                "op": "relu",
                "layer": "l1",
                "example": {"found": True, "relative_path": "operators/00_relu.ak"},
                "pipeline": {"strategy_ok": True, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": ["square-1k"], "missing_shape_tags": [], "correctness_evidence_present": True, "performance_evidence_present": True},
            }
        ],
        "l2": [
            {
                "op": "qkv_fa",
                "layer": "l2",
                "example": {"found": False, "relative_path": None},
                "pipeline": {"strategy_ok": False, "has_fusion_decision": False},
                "evidence": {"observed_shape_tags": [], "missing_shape_tags": ["llama-long"], "correctness_evidence_present": False, "performance_evidence_present": False},
            }
        ],
    }
    ledger_path = tmp_path / "coverage_ledger.json"
    ledger_path.write_text(json.dumps(ledger))

    output_path = tmp_path / "audit_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.stage7_audit_report",
            "--ledger",
            str(ledger_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert proc.returncode == 0, proc.stderr
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["summary"]["l2"]["missing_examples"] == ["qkv_fa"]
    assert "audit report" in proc.stdout.lower()
