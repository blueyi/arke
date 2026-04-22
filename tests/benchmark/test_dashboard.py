# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from benchmarks.dashboard import build_benchmark_dashboard


def _touch_json(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def test_build_benchmark_dashboard_combines_gap_audit_and_status_slices(tmp_path: Path):
    coverage_gap_path = tmp_path / "coverage_gap.json"
    coverage_gap_path.write_text(
        """
        {
          "stage": "S7",
          "gate": "G7",
          "l1": {
            "ops_total": 2,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 1,
            "op_coverage_ratio": 0.5,
            "shapes_required_total": 3,
            "shapes_observed_total": 2,
            "shape_coverage_ratio": 0.6667,
            "per_op": [
              {
                "op": "relu",
                "required_shape_count": 2,
                "observed_shape_count": 2,
                "observed_shape_tags": ["square-1k", "square-2k"],
                "missing_shape_tags": [],
                "missing_perf_target_fields": []
              },
              {
                "op": "gelu",
                "required_shape_count": 1,
                "observed_shape_count": 0,
                "observed_shape_tags": [],
                "missing_shape_tags": ["gpt2-hidden"],
                "missing_perf_target_fields": ["perf_actual"]
              }
            ]
          },
          "l2": {
            "ops_total": 1,
            "ops_with_any_evidence": 1,
            "ops_fully_covered": 0,
            "op_coverage_ratio": 1.0,
            "shapes_required_total": 2,
            "shapes_observed_total": 1,
            "shape_coverage_ratio": 0.5,
            "per_op": [
              {
                "op": "qkv_fa",
                "required_shape_count": 2,
                "observed_shape_count": 1,
                "observed_shape_tags": ["st4-long-32k"],
                "missing_shape_tags": ["llama-long"],
                "missing_perf_target_fields": ["perf_gap"]
              }
            ]
          },
          "combined": {
            "op_coverage_ratio": 0.6667,
            "shape_coverage_ratio": 0.6
          }
        }
        """.strip()
    )

    audit_report_path = tmp_path / "audit_report.json"
    audit_report_path.write_text(
        """
        {
          "stage": "S7",
          "gate": "G7",
          "summary": {
            "l1": {
              "entries": 2,
              "missing_examples": [],
              "missing_strategy_examples": ["gelu"],
              "missing_benchmark_evidence": ["gelu"],
              "missing_full_shape_evidence": ["gelu"],
              "unsupported_surface_cases": []
            },
            "l2": {
              "entries": 1,
              "missing_examples": ["qkv_fa"],
              "missing_strategy_examples": [],
              "missing_benchmark_evidence": ["qkv_fa"],
              "missing_full_shape_evidence": ["qkv_fa"],
              "unsupported_surface_cases": [
                {
                  "op": "qkv_fa",
                  "reason": "missing_example",
                  "missing_shape_tags": ["llama-long"]
                }
              ]
            }
          },
          "priority_actions": [
            {
              "category": "missing_l2_examples",
              "layer": "l2",
              "ops": ["qkv_fa"]
            }
          ]
        }
        """.strip()
    )

    stats_path = tmp_path / "stage7_operator_shape_stats.json"
    stats_path.write_text(
        """
        {
          "l1": {
            "relu": {
              "shape_count": 2,
              "shape_tags": ["square-1k", "square-2k"],
              "rows": 2,
              "status_counts": {"ok": 2}
            },
            "gelu": {
              "shape_count": 1,
              "shape_tags": ["gpt2-hidden"],
              "rows": 1,
              "status_counts": {"skipped": 1}
            }
          },
          "l2": {
            "qkv_fa": {
              "shape_count": 1,
              "shape_tags": ["st4-long-32k"],
              "rows": 1,
              "status_counts": {"oom": 1}
            }
          }
        }
        """.strip()
    )

    dashboard = build_benchmark_dashboard(
        coverage_gap_path=coverage_gap_path,
        audit_report_path=audit_report_path,
        operator_shape_stats_path=stats_path,
        title="Standard benchmark dashboard",
        text_summary_label="Standard benchmark dashboard",
    )

    assert dashboard["title"] == "Standard benchmark dashboard"
    assert dashboard["summary"]["combined"]["shape_coverage_ratio"] == 0.6
    assert dashboard["focus"]["priority_actions"][0]["category"] == "missing_l2_examples"
    assert dashboard["focus"]["l1"]["ops_without_evidence"] == ["gelu"]
    assert dashboard["focus"]["l1"]["memory_pressure_ops"] == [
        {"op": "gelu", "status_counts": {"skipped": 1}}
    ]
    assert dashboard["focus"]["l2"]["partial_coverage_ops"] == [
        {
            "op": "qkv_fa",
            "observed_shape_count": 1,
            "required_shape_count": 2,
            "missing_shape_tags": ["llama-long"],
        }
    ]
    assert dashboard["focus"]["l2"]["unsupported_surface_cases"] == [
        {
            "op": "qkv_fa",
            "reason": "missing_example",
            "missing_shape_tags": ["llama-long"],
        }
    ]
    assert dashboard["text_summary"].startswith("Standard benchmark dashboard")


def test_benchmark_dashboard_cli_writes_output(tmp_path: Path):
    coverage_gap_path = tmp_path / "coverage_gap.json"
    audit_report_path = tmp_path / "audit_report.json"
    stats_path = tmp_path / "stage7_operator_shape_stats.json"

    _touch_json(
        coverage_gap_path,
        '{"stage":"S7","gate":"G7","l1":{"ops_total":1,"ops_with_any_evidence":1,"ops_fully_covered":1,"op_coverage_ratio":1.0,"shapes_required_total":1,"shapes_observed_total":1,"shape_coverage_ratio":1.0,"per_op":[{"op":"relu","required_shape_count":1,"observed_shape_count":1,"observed_shape_tags":["square-1k"],"missing_shape_tags":[],"missing_perf_target_fields":[]}]},"l2":{"ops_total":0,"ops_with_any_evidence":0,"ops_fully_covered":0,"op_coverage_ratio":0.0,"shapes_required_total":0,"shapes_observed_total":0,"shape_coverage_ratio":0.0,"per_op":[]},"combined":{"op_coverage_ratio":1.0,"shape_coverage_ratio":1.0}}',
    )
    _touch_json(
        audit_report_path,
        '{"stage":"S7","gate":"G7","summary":{"l1":{"entries":1,"missing_examples":[],"missing_strategy_examples":[],"missing_benchmark_evidence":[],"missing_full_shape_evidence":[],"unsupported_surface_cases":[]},"l2":{"entries":0,"missing_examples":[],"missing_strategy_examples":[],"missing_benchmark_evidence":[],"missing_full_shape_evidence":[],"unsupported_surface_cases":[]}},"priority_actions":[]}',
    )
    _touch_json(
        stats_path,
        '{"l1":{"relu":{"shape_count":1,"shape_tags":["square-1k"],"rows":1,"status_counts":{"ok":1}}},"l2":{}}',
    )

    output_path = tmp_path / "dashboard.json"
    import subprocess, sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.dashboard",
            "--coverage-gap",
            str(coverage_gap_path),
            "--audit",
            str(audit_report_path),
            "--stats",
            str(stats_path),
            "--output",
            str(output_path),
            "--title",
            "CLI benchmark dashboard",
            "--summary-label",
            "CLI benchmark dashboard",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert output_path.exists()
    assert "CLI benchmark dashboard" in proc.stdout
