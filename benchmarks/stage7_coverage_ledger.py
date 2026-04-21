"""Stage 7 BL5 coverage ledger generation (Track 5.8).

Links BL5 target-matrix entries to:
- `.ak` example artifacts
- ArkePipeline dry-run evidence (SemanticIR / StrategyIR / fusion presence)
- Track 6 benchmark evidence observed in PERF_ALL.csv

The ledger is intended to be the machine-readable bridge between Track 5
(surface / example completeness) and Track 6 (benchmark evidence and gap-driven
closure).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from arke.compiler.pipeline import ArkePipeline

CORRECTNESS_FIELDS = (
    "allclose",
    "max_abs_diff",
    "mean_abs_diff",
    "rtol",
    "atol",
    "correctness_status",
    "correctness_reason",
)
PERF_TARGET_FIELDS = (
    "perf_target",
    "perf_actual",
    "perf_pass",
    "perf_gap",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO_ROOT / "benchmarks" / "stage7_bl5_target_matrix.json"
DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "examples"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "coverage_ledger.json"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _index_examples(examples_root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted((examples_root / "operators").rglob("*.ak")):
        stem = path.stem
        if stem[:2].isdigit() and "_" in stem:
            _, op_name = stem.split("_", 1)
        else:
            op_name = stem
        indexed.setdefault(op_name, path)
    return indexed


def _scan_perf_all(path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return evidence
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            op = (row.get("operator") or row.get("op") or "").strip()
            if not op:
                continue
            bucket = evidence.setdefault(
                op,
                {
                    "observed_shape_tags": set(),
                    "correctness_evidence_present": False,
                    "performance_evidence_present": False,
                },
            )
            shape_tag = (row.get("shape_tag") or "").strip()
            if shape_tag:
                bucket["observed_shape_tags"].add(shape_tag)
            if any((row.get(field) or "").strip() for field in CORRECTNESS_FIELDS):
                bucket["correctness_evidence_present"] = True
            if any((row.get(field) or "").strip() for field in PERF_TARGET_FIELDS):
                bucket["performance_evidence_present"] = True
    for bucket in evidence.values():
        bucket["observed_shape_tags"] = sorted(bucket["observed_shape_tags"])
    return evidence


def _compile_example(path: Path) -> dict[str, Any]:
    result = ArkePipeline().compile_file(str(path))
    fusion_groups = []
    if result.schedule_ir is not None:
        fusion_groups = [group.to_dict() for group in result.schedule_ir.fusion_groups]
    return {
        "compile_success": result.success,
        "errors": result.errors,
        "semantic_ok": result.semantic_ir is not None,
        "strategy_ok": result.strategy_ir is not None,
        "schedule_ok": result.schedule_ir is not None,
        "instruction_ok": result.instruction_ir is not None,
        "has_fusion_decision": any(getattr(decision, "kind", None) == "fuse" for decision in (result.strategy_ir.decisions if result.strategy_ir else [])),
        "fusion_groups": fusion_groups,
    }


def _build_entry(entry: dict[str, Any], layer: str, example_index: dict[str, Path], perf_index: dict[str, dict[str, Any]], examples_root: Path) -> dict[str, Any]:
    op = entry["op"]
    example_path = example_index.get(op)
    pipeline = {
        "compile_success": False,
        "errors": ["example not found"],
        "semantic_ok": False,
        "strategy_ok": False,
        "schedule_ok": False,
        "instruction_ok": False,
        "has_fusion_decision": False,
        "fusion_groups": [],
    }
    example_info = {
        "found": example_path is not None,
        "relative_path": str(example_path.relative_to(examples_root)) if example_path else None,
    }
    if example_path is not None:
        pipeline = _compile_example(example_path)

    evidence = perf_index.get(
        op,
        {
            "observed_shape_tags": [],
            "correctness_evidence_present": False,
            "performance_evidence_present": False,
        },
    )
    required = list(entry.get("shape_tags_required", []))
    observed_required = sorted(set(required) & set(evidence["observed_shape_tags"]))
    missing = sorted(set(required) - set(observed_required))

    return {
        "op": op,
        "layer": layer,
        "ot_tier": entry.get("ot_tier"),
        "required_shape_count": entry.get("shape_count_required", len(required)),
        "required_shape_tags": required,
        "example": example_info,
        "pipeline": pipeline,
        "evidence": {
            "observed_shape_tags": evidence["observed_shape_tags"],
            "observed_required_shape_tags": observed_required,
            "missing_shape_tags": missing,
            "correctness_evidence_present": evidence["correctness_evidence_present"],
            "performance_evidence_present": evidence["performance_evidence_present"],
        },
    }


def build_stage7_coverage_ledger(matrix_path: Path = DEFAULT_MATRIX, examples_root: Path = DEFAULT_EXAMPLES_ROOT, results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    matrix = _load_json(matrix_path)
    example_index = _index_examples(examples_root)
    l1_perf = _scan_perf_all(results_root / "l1" / "PERF_ALL.csv")
    l2_perf = _scan_perf_all(results_root / "l2" / "PERF_ALL.csv")

    l1_entries = [
        _build_entry(entry, "l1", example_index, l1_perf, examples_root)
        for entry in matrix.get("l1", [])
    ]
    l2_entries = [
        _build_entry(entry, "l2", example_index, l2_perf, examples_root)
        for entry in matrix.get("l2", [])
    ]

    def summarize(entries: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "entries": len(entries),
            "with_examples": sum(1 for entry in entries if entry["example"]["found"]),
            "with_strategy_examples": sum(1 for entry in entries if entry["pipeline"]["strategy_ok"]),
            "with_benchmark_evidence": sum(1 for entry in entries if entry["evidence"]["observed_shape_tags"]),
            "with_full_required_shape_evidence": sum(1 for entry in entries if not entry["evidence"]["missing_shape_tags"]),
        }

    return {
        "stage": matrix.get("stage", "S7"),
        "gate": matrix.get("gate", "G7"),
        "sources": {
            "matrix": _repo_rel(matrix_path),
            "examples_root": _repo_rel(examples_root),
            "results_root": _repo_rel(results_root),
        },
        "summary": {
            "l1": summarize(l1_entries),
            "l2": summarize(l2_entries),
        },
        "l1": l1_entries,
        "l2": l2_entries,
    }


def format_text_summary(report: dict[str, Any]) -> str:
    lines = [f"Stage 7 coverage ledger — {report['stage']} / {report['gate']}", "=" * 60]
    for layer in ("l1", "l2"):
        summary = report["summary"][layer]
        lines.append(
            f"[{layer.upper()}] entries={summary['entries']}, examples={summary['with_examples']}, "
            f"strategy_examples={summary['with_strategy_examples']}, benchmark_evidence={summary['with_benchmark_evidence']}, "
            f"full_shape_evidence={summary['with_full_required_shape_evidence']}"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage 7 BL5 coverage ledger")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--examples-root", type=Path, default=DEFAULT_EXAMPLES_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_stage7_coverage_ledger(
        matrix_path=args.matrix,
        examples_root=args.examples_root,
        results_root=args.results_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(format_text_summary(report))
    print(f"\nwrote coverage ledger: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())