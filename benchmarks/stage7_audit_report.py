"""Stage 7 BL5 audit report generation (Track 5.6).

Builds a machine-readable audit over the Stage 7 coverage ledger to highlight:
- BL5 operators missing `.ak` examples
- operators whose examples do not yet compile to StrategyIR
- operators/fusions still lacking benchmark evidence for required shape tags
- candidate unsupported cases where Stage 7 Lang surface coverage is still incomplete

The audit is intentionally offline and consumes the Track 5.8 coverage ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = (
    REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6" / "coverage_ledger.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6" / "audit_report.json"
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _layer_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    missing_examples = [entry["op"] for entry in entries if not entry["example"]["found"]]
    missing_strategy_examples = [
        entry["op"] for entry in entries if not entry["pipeline"].get("strategy_ok", False)
    ]
    missing_benchmark_evidence = [
        entry["op"]
        for entry in entries
        if not entry["evidence"].get("observed_shape_tags")
    ]
    missing_full_shape_evidence = [
        entry["op"]
        for entry in entries
        if entry["evidence"].get("missing_shape_tags")
    ]
    unsupported_surface_cases = [
        {
            "op": entry["op"],
            "reason": "missing_example",
            "missing_shape_tags": entry["evidence"].get("missing_shape_tags", []),
        }
        for entry in entries
        if not entry["example"]["found"]
    ]
    unsupported_surface_cases.extend(
        {
            "op": entry["op"],
            "reason": "missing_strategy_surface",
            "missing_shape_tags": entry["evidence"].get("missing_shape_tags", []),
        }
        for entry in entries
        if entry["example"]["found"] and not entry["pipeline"].get("strategy_ok", False)
    )
    unsupported_surface_cases.extend(
        {
            "op": entry["op"],
            "reason": "missing_fusion_strategy",
            "missing_shape_tags": entry["evidence"].get("missing_shape_tags", []),
        }
        for entry in entries
        if entry.get("layer") == "l2"
        and entry["example"]["found"]
        and not entry["pipeline"].get("has_fusion_decision", False)
    )
    unsupported_surface_cases.extend(
        {
            "op": entry["op"],
            "reason": "missing_full_shape_evidence",
            "missing_shape_tags": entry["evidence"].get("missing_shape_tags", []),
        }
        for entry in entries
        if entry.get("layer") == "l2"
        and entry["evidence"].get("missing_shape_tags")
        and entry["pipeline"].get("strategy_ok", False)
        and entry["pipeline"].get("has_fusion_decision", False)
    )

    return {
        "entries": len(entries),
        "missing_examples": missing_examples,
        "missing_strategy_examples": missing_strategy_examples,
        "missing_benchmark_evidence": missing_benchmark_evidence,
        "missing_full_shape_evidence": missing_full_shape_evidence,
        "unsupported_surface_cases": unsupported_surface_cases,
    }


def _build_priority_actions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    ordered = [
        ("l2", "missing_examples", "missing_l2_examples"),
        ("l2", "missing_strategy_examples", "missing_l2_strategy_examples"),
        ("l2", "missing_full_shape_evidence", "missing_l2_shape_evidence"),
        ("l1", "missing_examples", "missing_l1_examples"),
        ("l1", "missing_strategy_examples", "missing_l1_strategy_examples"),
        ("l1", "missing_full_shape_evidence", "missing_l1_shape_evidence"),
    ]
    for layer, key, category in ordered:
        ops = summary[layer][key]
        if ops:
            actions.append({"category": category, "layer": layer, "ops": ops})
    return actions


def format_text_summary(report: dict[str, Any]) -> str:
    lines = [f"Stage 7 audit report — {report['stage']} / {report['gate']}", "=" * 60]
    for layer in ("l1", "l2"):
        summary = report["summary"][layer]
        lines.append(
            f"[{layer.upper()}] entries={summary['entries']}, missing examples={len(summary['missing_examples'])}, "
            f"missing strategy examples={len(summary['missing_strategy_examples'])}, "
            f"missing benchmark evidence={len(summary['missing_benchmark_evidence'])}, "
            f"missing full shape evidence={len(summary['missing_full_shape_evidence'])}"
        )
        if summary["missing_examples"]:
            lines.append(f"  missing {layer.upper()} examples: {', '.join(summary['missing_examples'])}")
        if summary["missing_strategy_examples"]:
            lines.append(
                f"  missing {layer.upper()} strategy examples: {', '.join(summary['missing_strategy_examples'])}"
            )
    if report["priority_actions"]:
        lines.append("-" * 60)
        lines.append("Priority actions:")
        for item in report["priority_actions"]:
            lines.append(f"- {item['category']}: {', '.join(item['ops'])}")
    return "\n".join(lines)


def build_stage7_audit_report(ledger_path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    ledger = _load_json(ledger_path)
    summary = {
        "l1": _layer_summary(ledger.get("l1", [])),
        "l2": _layer_summary(ledger.get("l2", [])),
    }
    report = {
        "stage": ledger.get("stage", "S7"),
        "gate": ledger.get("gate", "G7"),
        "source": _repo_rel(ledger_path),
        "summary": summary,
        "priority_actions": _build_priority_actions(summary),
    }
    report["text_summary"] = format_text_summary(report)
    return report


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage 7 BL5 audit report")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_stage7_audit_report(args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(report["text_summary"])
    print(f"\nwrote audit report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
