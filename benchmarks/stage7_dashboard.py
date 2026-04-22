# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 7 Track 6 dashboard synthesis.

This module merges the machine-readable Track 6 artifacts into a single dashboard
payload that is easy for Gate checks, CI, and humans to consume.

Inputs:
- coverage_gap.json
- audit_report.json
- stage7_operator_shape_stats.json

Output:
- dashboard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6"
)
DEFAULT_COVERAGE_GAP = DEFAULT_RESULTS_ROOT / "coverage_gap.json"
DEFAULT_AUDIT_REPORT = DEFAULT_RESULTS_ROOT / "audit_report.json"
DEFAULT_OPERATOR_SHAPE_STATS = DEFAULT_RESULTS_ROOT / "stage7_operator_shape_stats.json"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "dashboard.json"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _build_layer_focus(
    gap_layer: dict[str, Any],
    audit_layer: dict[str, Any],
    stats_layer: dict[str, Any],
) -> dict[str, Any]:
    per_op = gap_layer.get("per_op", [])

    ops_without_evidence = [
        entry["op"] for entry in per_op if entry.get("observed_shape_count", 0) == 0
    ]
    partial_coverage_ops = [
        {
            "op": entry["op"],
            "observed_shape_count": entry.get("observed_shape_count", 0),
            "required_shape_count": entry.get("required_shape_count", 0),
            "missing_shape_tags": entry.get("missing_shape_tags", []),
        }
        for entry in per_op
        if 0 < entry.get("observed_shape_count", 0) < entry.get("required_shape_count", 0)
    ]
    perf_field_gaps = [
        {
            "op": entry["op"],
            "missing_perf_target_fields": entry.get("missing_perf_target_fields", []),
        }
        for entry in per_op
        if entry.get("missing_perf_target_fields")
    ]
    memory_pressure_ops = [
        {"op": op, "status_counts": stat.get("status_counts", {})}
        for op, stat in sorted(stats_layer.items())
        if any(status != "ok" and count > 0 for status, count in stat.get("status_counts", {}).items())
    ]

    return {
        "ops_without_evidence": ops_without_evidence,
        "partial_coverage_ops": partial_coverage_ops,
        "missing_examples": audit_layer.get("missing_examples", []),
        "missing_strategy_examples": audit_layer.get("missing_strategy_examples", []),
        "missing_benchmark_evidence": audit_layer.get("missing_benchmark_evidence", []),
        "missing_full_shape_evidence": audit_layer.get("missing_full_shape_evidence", []),
        "unsupported_surface_cases": audit_layer.get("unsupported_surface_cases", []),
        "perf_field_gaps": perf_field_gaps,
        "memory_pressure_ops": memory_pressure_ops,
    }


def format_text_summary(report: dict[str, Any]) -> str:
    lines = [f"Stage 7 dashboard — {report['stage']} / {report['gate']}", "=" * 60]
    combined = report["summary"]["combined"]
    lines.append(
        f"Combined coverage: ops={combined['op_coverage_ratio'] * 100:.1f}% "
        f"shapes={combined['shape_coverage_ratio'] * 100:.2f}%"
    )
    for layer in ("l1", "l2"):
        layer_summary = report["summary"][layer]
        layer_focus = report["focus"][layer]
        lines.append(
            f"[{layer.upper()}] ops={layer_summary['ops_with_any_evidence']}/{layer_summary['ops_total']} "
            f"fully_covered={layer_summary['ops_fully_covered']} "
            f"missing_examples={len(layer_focus['missing_examples'])} "
            f"unsupported={len(layer_focus['unsupported_surface_cases'])}"
        )
        if layer_focus["ops_without_evidence"]:
            lines.append(
                f"  no evidence: {', '.join(layer_focus['ops_without_evidence'])}"
            )
        if layer_focus["partial_coverage_ops"]:
            partial = ", ".join(
                f"{item['op']} ({item['observed_shape_count']}/{item['required_shape_count']})"
                for item in layer_focus["partial_coverage_ops"][:5]
            )
            lines.append(f"  partial coverage: {partial}")
        if layer_focus["memory_pressure_ops"]:
            pressure = ", ".join(item["op"] for item in layer_focus["memory_pressure_ops"])
            lines.append(f"  memory pressure / skipped: {pressure}")
    if report["focus"]["priority_actions"]:
        lines.append("-" * 60)
        lines.append("Priority actions:")
        for item in report["focus"]["priority_actions"]:
            lines.append(f"- {item['category']}: {', '.join(item['ops'])}")
    return "\n".join(lines)


def build_stage7_dashboard(
    coverage_gap_path: Path = DEFAULT_COVERAGE_GAP,
    audit_report_path: Path = DEFAULT_AUDIT_REPORT,
    operator_shape_stats_path: Path = DEFAULT_OPERATOR_SHAPE_STATS,
) -> dict[str, Any]:
    coverage_gap = _load_json(coverage_gap_path)
    audit_report = _load_json(audit_report_path)
    operator_shape_stats = _load_json(operator_shape_stats_path)

    summary = {
        "l1": coverage_gap.get("l1", {}),
        "l2": coverage_gap.get("l2", {}),
        "combined": coverage_gap.get("combined", {}),
    }
    focus = {
        "l1": _build_layer_focus(
            summary["l1"],
            audit_report.get("summary", {}).get("l1", {}),
            operator_shape_stats.get("l1", {}),
        ),
        "l2": _build_layer_focus(
            summary["l2"],
            audit_report.get("summary", {}).get("l2", {}),
            operator_shape_stats.get("l2", {}),
        ),
        "priority_actions": audit_report.get("priority_actions", []),
    }

    report = {
        "stage": coverage_gap.get("stage", audit_report.get("stage", "S7")),
        "gate": coverage_gap.get("gate", audit_report.get("gate", "G7")),
        "sources": {
            "coverage_gap": _repo_rel(coverage_gap_path),
            "audit_report": _repo_rel(audit_report_path),
            "operator_shape_stats": _repo_rel(operator_shape_stats_path),
        },
        "summary": summary,
        "focus": focus,
    }
    report["text_summary"] = format_text_summary(report)
    return report


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage 7 Track 6 dashboard")
    parser.add_argument("--coverage-gap", type=Path, default=DEFAULT_COVERAGE_GAP)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_OPERATOR_SHAPE_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_stage7_dashboard(
        coverage_gap_path=args.coverage_gap,
        audit_report_path=args.audit,
        operator_shape_stats_path=args.stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["text_summary"])
    print(f"\nwrote dashboard: {args.output}")
    return 0


def main() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":
    main()
