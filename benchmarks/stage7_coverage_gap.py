"""Stage 7 BL5 coverage gap computation (Track 6.10).

This module reads the BL5 required-surface declaration from
``benchmarks/stage7_bl5_target_matrix.json`` and the currently produced
benchmark evidence from ``benchmarks/results/phase1/stage7/track6/{l1,l2}/
PERF_ALL.csv``. It produces a machine-readable gap report that lists, per
operator / fusion:

- required shape tags
- observed shape tags (from PERF_ALL.csv)
- missing shape tags
- whether correctness / performance target fields are persisted

The functions are pure-offline and have no GPU dependency. They are the
canonical feed for Track 6.11 (dashboards) and Track 6.12 (gate
integration).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO_ROOT / "benchmarks" / "stage7_bl5_target_matrix.json"
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "benchmarks" / "results" / "phase1" / "stage7" / "track6"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "coverage_gap.json"

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


def _load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _scan_perf_all(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, bool]]]:
    """Return ``(op -> observed shape tags, op -> field-presence flags)``.

    Field presence is reported per operator: a field is considered present if at
    least one row under that operator populates it with a non-empty value.
    """

    observed: dict[str, set[str]] = defaultdict(set)
    field_presence: dict[str, dict[str, bool]] = defaultdict(
        lambda: {f: False for f in (*CORRECTNESS_FIELDS, *PERF_TARGET_FIELDS)}
    )

    if not path.exists():
        return observed, field_presence

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            op = (row.get("operator") or "").strip()
            if not op:
                continue
            shape_tag = (row.get("shape_tag") or "").strip()
            if shape_tag:
                observed[op].add(shape_tag)
            for field in (*CORRECTNESS_FIELDS, *PERF_TARGET_FIELDS):
                val = row.get(field)
                if val not in (None, "", "NA", "nan"):
                    field_presence[op][field] = True
    return observed, field_presence


def _compute_layer_gap(
    entries: list[dict[str, Any]],
    observed: dict[str, set[str]],
    field_presence: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    per_op: list[dict[str, Any]] = []
    total_required_shapes = 0
    total_observed_shapes = 0
    ops_with_any_evidence = 0

    for entry in entries:
        op = entry.get("op")
        required_tags = list(entry.get("shape_tags_required", []))
        required_count = entry.get("shape_count_required", len(required_tags))
        obs_tags = sorted(observed.get(op, set()) & set(required_tags))
        missing_tags = sorted(set(required_tags) - set(obs_tags))

        total_required_shapes += required_count
        total_observed_shapes += len(obs_tags)
        if obs_tags:
            ops_with_any_evidence += 1

        op_fields = field_presence.get(op, {})
        correctness_present = any(op_fields.get(f) for f in CORRECTNESS_FIELDS)
        perf_target_present = any(op_fields.get(f) for f in PERF_TARGET_FIELDS)

        per_op.append(
            {
                "op": op,
                "required_shape_count": required_count,
                "observed_shape_count": len(obs_tags),
                "coverage_ratio": (len(obs_tags) / required_count)
                if required_count
                else 0.0,
                "observed_shape_tags": obs_tags,
                "missing_shape_tags": missing_tags,
                "correctness_fields_present": correctness_present,
                "perf_target_fields_present": perf_target_present,
                "missing_perf_target_fields": [
                    field
                    for field in PERF_TARGET_FIELDS
                    if not op_fields.get(field)
                ],
            }
        )

    total_ops = len(entries)
    op_coverage = ops_with_any_evidence / total_ops if total_ops else 0.0
    shape_coverage = (
        total_observed_shapes / total_required_shapes
        if total_required_shapes
        else 0.0
    )

    return {
        "ops_total": total_ops,
        "ops_with_any_evidence": ops_with_any_evidence,
        "ops_fully_covered": sum(
            1 for entry in per_op if not entry["missing_shape_tags"]
        ),
        "op_coverage_ratio": round(op_coverage, 4),
        "shapes_required_total": total_required_shapes,
        "shapes_observed_total": total_observed_shapes,
        "shape_coverage_ratio": round(shape_coverage, 4),
        "per_op": per_op,
    }


def compute_gap(matrix_path: Path, results_root: Path) -> dict[str, Any]:
    matrix = _load_matrix(matrix_path)
    l1_perf = results_root / "l1" / "PERF_ALL.csv"
    l2_perf = results_root / "l2" / "PERF_ALL.csv"

    l1_observed, l1_fields = _scan_perf_all(l1_perf)
    l2_observed, l2_fields = _scan_perf_all(l2_perf)

    l1_report = _compute_layer_gap(matrix.get("l1", []), l1_observed, l1_fields)
    l2_report = _compute_layer_gap(matrix.get("l2", []), l2_observed, l2_fields)

    combined = {
        "op_coverage_ratio": round(
            (
                l1_report["ops_with_any_evidence"]
                + l2_report["ops_with_any_evidence"]
            )
            / max(1, l1_report["ops_total"] + l2_report["ops_total"]),
            4,
        ),
        "shape_coverage_ratio": round(
            (
                l1_report["shapes_observed_total"]
                + l2_report["shapes_observed_total"]
            )
            / max(
                1,
                l1_report["shapes_required_total"]
                + l2_report["shapes_required_total"],
            ),
            4,
        ),
    }

    return {
        "stage": matrix.get("stage", "S7"),
        "gate": matrix.get("gate", "G7"),
        "sources": {
            "matrix": str(matrix_path),
            "l1_perf_all": str(l1_perf),
            "l1_perf_all_exists": l1_perf.exists(),
            "l2_perf_all": str(l2_perf),
            "l2_perf_all_exists": l2_perf.exists(),
        },
        "l1": l1_report,
        "l2": l2_report,
        "combined": combined,
    }


def format_text_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Stage 7 BL5 Coverage Gap — {report['stage']} / {report['gate']}")
    lines.append("=" * 60)
    for layer in ("l1", "l2"):
        layer_report = report[layer]
        lines.append(
            f"[{layer.upper()}] ops {layer_report['ops_with_any_evidence']}/"
            f"{layer_report['ops_total']} "
            f"({layer_report['op_coverage_ratio'] * 100:.1f}%), "
            f"shapes {layer_report['shapes_observed_total']}/"
            f"{layer_report['shapes_required_total']} "
            f"({layer_report['shape_coverage_ratio'] * 100:.2f}%), "
            f"fully_covered={layer_report['ops_fully_covered']}"
        )
        missing_ops = [
            entry["op"]
            for entry in layer_report["per_op"]
            if entry["observed_shape_count"] == 0
        ]
        if missing_ops:
            lines.append(f"  ops with NO evidence: {missing_ops}")
        partial_ops = [
            (
                entry["op"],
                entry["observed_shape_count"],
                entry["required_shape_count"],
            )
            for entry in layer_report["per_op"]
            if 0 < entry["observed_shape_count"]
            < entry["required_shape_count"]
        ]
        if partial_ops:
            lines.append(
                f"  ops with partial evidence ({len(partial_ops)}):"
            )
            for op, observed_count, required_count in partial_ops[:10]:
                lines.append(f"    {op}: {observed_count}/{required_count}")
            if len(partial_ops) > 10:
                lines.append(f"    ... +{len(partial_ops) - 10} more")
    combined = report["combined"]
    lines.append("-" * 60)
    lines.append(
        f"[COMBINED] op_coverage={combined['op_coverage_ratio'] * 100:.1f}%, "
        f"shape_coverage={combined['shape_coverage_ratio'] * 100:.2f}%"
    )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Stage 7 BL5 coverage gaps from benchmark artifacts"
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Exit with status 2 if combined shape coverage < 1.0",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress text summary on stdout",
    )
    args = parser.parse_args(argv)

    try:
        report = compute_gap(args.matrix, args.results_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except json.JSONDecodeError as exc:
        parser.error(f"failed to parse matrix JSON: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    if not args.quiet:
        print(format_text_summary(report))
        print(f"\nwrote: {args.output}")

    if args.fail_on_gap and report["combined"]["shape_coverage_ratio"] < 1.0:
        return 2
    return 0


def main() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":
    main()
