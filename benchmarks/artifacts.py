# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

MEMORY_FIELDS = (
    "memory_bytes_required",
    "memory_bytes_budget",
    "memory_ratio",
    "memory_policy",
)


def _safe_float(value: str | float | int | None) -> float | None:
    if value in (None, "", "N/A", "None", "inf"):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(val) or math.isnan(val):
        return None
    return val


def _copy_memory_fields(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in MEMORY_FIELDS}


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _merge_fieldnames(*field_lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for field_list in field_lists:
        for field in field_list:
            if field and field not in seen:
                seen.add(field)
                merged.append(field)
    return merged


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _perf_row_key(row: dict[str, str], fieldnames: list[str]) -> tuple[str, str, str]:
    runner_field = "baseline" if "baseline" in fieldnames else "approach" if "approach" in fieldnames else ""
    key = (
        (row.get("operator") or "").strip(),
        (row.get("shape_tag") or "").strip(),
        (row.get(runner_field) or "").strip() if runner_field else "",
    )
    if not all(key):
        raise ValueError(
            "perf rows must include operator, shape_tag, and baseline/approach "
            f"identity fields; got {key!r}"
        )
    return key


def _merge_perf_rows(
    existing_rows: list[dict[str, str]],
    incoming_rows: list[dict[str, str]],
    fieldnames: list[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    merged: list[dict[str, str]] = []
    index_by_key: dict[tuple[str, str, str], int] = {}
    updated = 0
    inserted = 0

    for row in existing_rows:
        key = _perf_row_key(row, fieldnames)
        if key in index_by_key:
            merged[index_by_key[key]] = row
        else:
            index_by_key[key] = len(merged)
            merged.append(row)

    for row in incoming_rows:
        key = _perf_row_key(row, fieldnames)
        if key in index_by_key:
            merged[index_by_key[key]] = row
            updated += 1
        else:
            index_by_key[key] = len(merged)
            merged.append(row)
            inserted += 1

    return merged, {
        "preserved_existing_rows": len(existing_rows) - updated,
        "updated_rows": updated,
        "inserted_rows": inserted,
        "total_rows": len(merged),
    }


def merge_perf_evidence(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """Incrementally merge perf_*.csv evidence from a partial run into a target run.

    Unlike :func:`merge_perf_all`, this function is safe for targeted benchmark runs:
    it updates only matching ``(operator, shape_tag, baseline|approach)`` rows and
    appends new evidence while preserving unrelated per-op rows already present in
    the canonical result directory.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    perf_files = sorted(source_dir.glob("perf_*.csv"))
    if not perf_files:
        raise FileNotFoundError(f"no perf_*.csv files found in {source_dir}")

    totals = {
        "perf_files": 0,
        "preserved_existing_rows": 0,
        "updated_rows": 0,
        "inserted_rows": 0,
        "total_rows": 0,
    }
    merged_files: list[str] = []

    for source_csv in perf_files:
        target_csv = target_dir / source_csv.name
        existing_fields, existing_rows = _read_csv_rows(target_csv)
        incoming_fields, incoming_rows = _read_csv_rows(source_csv)
        fieldnames = _merge_fieldnames(existing_fields, incoming_fields)
        merged_rows, stats = _merge_perf_rows(existing_rows, incoming_rows, fieldnames)
        _write_csv_rows(target_csv, fieldnames, merged_rows)

        totals["perf_files"] += 1
        for key in ("preserved_existing_rows", "updated_rows", "inserted_rows"):
            totals[key] += stats[key]
        totals["total_rows"] += stats["total_rows"]
        merged_files.append(str(target_csv))

    perf_all = merge_perf_all(target_dir)
    summary = write_summary(target_dir)
    return {
        **totals,
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "merged_files": merged_files,
        "perf_all": str(perf_all) if perf_all else None,
        "summary": str(summary) if summary else None,
    }


def _resolve_perf_fields(row: dict[str, str], ratio: float | None) -> dict[str, str]:
    perf_target = (row.get("perf_target") or "").strip()
    perf_actual = (row.get("perf_actual") or "").strip()
    perf_pass = (row.get("perf_pass") or "").strip().lower()
    perf_gap = (row.get("perf_gap") or "").strip()
    status = (row.get("status") or "").strip().lower()

    if not perf_target:
        perf_target = "1.0"

    target = _safe_float(perf_target)
    if target is None:
        target = 1.0

    if not perf_actual:
        if ratio is not None:
            perf_actual = f"{ratio:.4f}"
        elif status in {"skipped", "oom", "error"}:
            perf_actual = "N/A"

    if not perf_pass:
        if ratio is not None:
            perf_pass = "true" if ratio >= target else "false"
        elif perf_actual in {"N/A", "NA"}:
            perf_pass = "false"

    if not perf_gap:
        if ratio is not None:
            perf_gap = f"{(ratio - target):.4f}"
        elif perf_actual in {"N/A", "NA"}:
            perf_gap = "N/A"

    return {
        "perf_target": perf_target,
        "perf_actual": perf_actual,
        "perf_pass": perf_pass,
        "perf_gap": perf_gap,
    }


def _infer_op_from_path(raw_csv: Path) -> str:
    stem = raw_csv.stem
    return stem[:-8] if stem.endswith("_results") else stem


def write_perf_csv_from_l1(raw_csv: Path, out_csv: Path) -> Path:
    rows = list(csv.DictReader(raw_csv.open()))
    op_from_path = _infer_op_from_path(raw_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "operator", "shape_tag", "baseline", "latency_us",
        "latency_min_us", "tflops", "ratio_vs_baseline", "status", "reason", "retryable",
        "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
        "correctness_status", "correctness_reason",
        *MEMORY_FIELDS,
        "perf_target", "perf_actual", "perf_pass", "perf_gap",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["shape_tag"], []).append(row)
        for shape_tag, shape_rows in grouped.items():
            baseline_row = None
            for row in shape_rows:
                if row.get("baseline") == "PyTorch-eager":
                    baseline_row = row
                    break
            if baseline_row is None and shape_rows:
                baseline_row = shape_rows[0]
            baseline_lat = _safe_float(baseline_row.get("latency_us") if baseline_row else None)
            for row in shape_rows:
                lat = _safe_float(row.get("latency_us"))
                ratio = baseline_lat / lat if baseline_lat and lat else None
                perf_fields = _resolve_perf_fields(row, ratio)
                writer.writerow({
                    "operator": row.get("op") or op_from_path,
                    "shape_tag": shape_tag,
                    "baseline": row.get("baseline", "unknown"),
                    "latency_us": row.get("latency_us", ""),
                    "latency_min_us": row.get("latency_min_us", ""),
                    "tflops": row.get("tflops", ""),
                    "ratio_vs_baseline": f"{ratio:.4f}" if ratio is not None else "",
                    "status": row.get("status", "ok"),
                    "reason": row.get("reason", ""),
                    "retryable": row.get("retryable", "false"),
                    "allclose": row.get("allclose", ""),
                    "max_abs_diff": row.get("max_abs_diff", ""),
                    "mean_abs_diff": row.get("mean_abs_diff", ""),
                    "rtol": row.get("rtol", ""),
                    "atol": row.get("atol", ""),
                    "correctness_status": row.get("correctness_status", "unknown"),
                    "correctness_reason": row.get("correctness_reason", ""),
                    **_copy_memory_fields(row),
                    **perf_fields,
                })
    return out_csv


def write_perf_csv_from_l2(raw_csv: Path, out_csv: Path) -> Path:
    rows = list(csv.DictReader(raw_csv.open()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "operator", "shape_tag", "approach", "latency_us",
        "latency_min_us", "tflops", "ratio_vs_baseline", "status", "reason", "retryable",
        "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
        "correctness_status", "correctness_reason",
        *MEMORY_FIELDS,
        "perf_target", "perf_actual", "perf_pass", "perf_gap",
        # ── L2 fusion measurement protocol (RFC §4) — additive provenance ──
        "golden_runner", "golden_priority", "backend", "perf_oracle_unavailable_triton",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["shape_tag"], []).append(row)
        for shape_tag, shape_rows in grouped.items():
            baseline_row = None
            for row in shape_rows:
                if row.get("approach") == "separate":
                    baseline_row = row
                    break
            if baseline_row is None and shape_rows:
                baseline_row = shape_rows[0]
            baseline_lat = _safe_float(baseline_row.get("latency_us") if baseline_row else None)
            for row in shape_rows:
                lat = _safe_float(row.get("latency_us"))
                ratio = baseline_lat / lat if baseline_lat and lat else None
                perf_fields = _resolve_perf_fields(row, ratio)
                writer.writerow({
                    "operator": row.get("op", "unknown"),
                    "shape_tag": shape_tag,
                    "approach": row.get("approach", "unknown"),
                    "latency_us": row.get("latency_us", ""),
                    "latency_min_us": row.get("latency_min_us", ""),
                    "tflops": row.get("tflops", ""),
                    "ratio_vs_baseline": f"{ratio:.4f}" if ratio is not None else "",
                    "status": row.get("status", "ok"),
                    "reason": row.get("reason", ""),
                    "retryable": row.get("retryable", "false"),
                    "allclose": row.get("allclose", ""),
                    "max_abs_diff": row.get("max_abs_diff", ""),
                    "mean_abs_diff": row.get("mean_abs_diff", ""),
                    "rtol": row.get("rtol", ""),
                    "atol": row.get("atol", ""),
                    "correctness_status": row.get("correctness_status", "unknown"),
                    "correctness_reason": row.get("correctness_reason", ""),
                    **_copy_memory_fields(row),
                    **perf_fields,
                    "golden_runner": row.get("golden_runner", ""),
                    "golden_priority": row.get("golden_priority", ""),
                    "backend": row.get("backend", ""),
                    "perf_oracle_unavailable_triton": row.get("perf_oracle_unavailable_triton", ""),
                })
    return out_csv


def merge_perf_all(run_dir: Path) -> Path | None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()

    for csv_file in sorted(run_dir.glob("perf_*.csv")):
        with csv_file.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for name in reader.fieldnames or []:
                if name and name not in seen_fields:
                    seen_fields.add(name)
                    fieldnames.append(name)
            for row in reader:
                rows.append(row)
                for name in row.keys():
                    if name and name not in seen_fields:
                        seen_fields.add(name)
                        fieldnames.append(name)

    if not rows:
        return None

    out = run_dir / "PERF_ALL.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out


def write_summary(run_dir: Path) -> Path | None:
    perf_all = run_dir / "PERF_ALL.csv"
    if not perf_all.exists():
        return None
    perf_rows = list(csv.DictReader(perf_all.open()))
    ratios_by_operator: dict[str, list[float]] = {}
    for row in perf_rows:
        operator = row.get("operator", "unknown")
        ratio = _safe_float(row.get("ratio_vs_baseline"))
        if ratio is not None and ratio > 0:
            ratios_by_operator.setdefault(operator, []).append(ratio)

    # Honesty guard (2026-07-27, see docs/audit/): an op whose Arke rows are
    # all declined/unsupported still accumulates ratio rows from the baseline
    # runners' self-ratios (e.g. PyTorch-eager vs itself = 1.0), which used to
    # yield a phantom op_score of 1.0 with zero Arke evidence. Score such ops
    # as None and surface them in `no_data_ops` so coverage narratives can't
    # silently count them as measured passes.
    # Evidence definition covers both artifact shapes:
    #   * L1 rows: baseline == 'Arke' with status ok
    #   * L2 rows: no baseline column populated (approach-keyed fusion rows
    #     are Arke's own measurements by construction)
    ops_with_arke_evidence: set[str] = set()
    for row in perf_rows:
        if (row.get("status", "") or "").strip() != "ok":
            continue
        baseline = (row.get("baseline", "") or "").strip()
        approach = (row.get("approach", "") or "").strip()
        if baseline == "Arke" or (not baseline and approach):
            ops_with_arke_evidence.add(row.get("operator", "unknown"))
    no_data_ops = sorted(
        op for op in ratios_by_operator if op not in ops_with_arke_evidence
    )

    def geomean(vals: list[float]) -> float:
        return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0

    status_counts: dict[str, int] = {}
    correctness_counts: dict[str, int] = {}
    memory_policy_counts: dict[str, int] = {}
    memory_pressure_rows = 0
    for row in perf_rows:
        status = row.get("status", "ok")
        status_counts[status] = status_counts.get(status, 0) + 1
        correctness = row.get("correctness_status", "unknown")
        correctness_counts[correctness] = correctness_counts.get(correctness, 0) + 1
        policy = (row.get("memory_policy", "") or "none").strip() or "none"
        memory_policy_counts[policy] = memory_policy_counts.get(policy, 0) + 1
        memory_ratio = _safe_float(row.get("memory_ratio"))
        if status in {"skipped", "oom"} or (memory_ratio is not None and memory_ratio > 1.0):
            memory_pressure_rows += 1

    perf_target_counts: dict[str, int] = {}
    perf_pass_counts: dict[str, int] = {}
    perf_targets_by_operator: dict[str, list[float]] = {}
    perf_actuals_by_operator: dict[str, list[float]] = {}
    perf_gaps_by_operator: dict[str, list[float]] = {}
    for row in perf_rows:
        operator = row.get("operator", "unknown")
        perf_target = _safe_float(row.get("perf_target"))
        perf_actual = _safe_float(row.get("perf_actual"))
        perf_gap = _safe_float(row.get("perf_gap"))
        perf_pass = (row.get("perf_pass", "") or "").strip().lower()
        if perf_target is not None:
            perf_targets_by_operator.setdefault(operator, []).append(perf_target)
        if perf_actual is not None:
            perf_actuals_by_operator.setdefault(operator, []).append(perf_actual)
        if perf_gap is not None:
            perf_gaps_by_operator.setdefault(operator, []).append(perf_gap)
        if perf_pass:
            perf_pass_counts[perf_pass] = perf_pass_counts.get(perf_pass, 0) + 1
        else:
            perf_pass_counts["unknown"] = perf_pass_counts.get("unknown", 0) + 1

    summary = {
        "overall_geomean": round(geomean([v for vals in ratios_by_operator.values() for v in vals]), 4)
        if ratios_by_operator else 0.0,
        "op_scores": {
            op: (round(geomean(vals), 4) if op in ops_with_arke_evidence else None)
            for op, vals in ratios_by_operator.items()
        },
        "no_data_ops": no_data_ops,
        "total_shapes": sum(len(vals) for vals in ratios_by_operator.values()),
        "operators": sorted(ratios_by_operator.keys()),
        "status_counts": status_counts,
        "correctness_counts": correctness_counts,
        "memory_policy_counts": memory_policy_counts,
        "memory_pressure_rows": memory_pressure_rows,
        "perf_target_counts": {
            "with_target": sum(len(vals) for vals in perf_targets_by_operator.values()),
            "without_target": sum(1 for row in perf_rows if _safe_float(row.get("perf_target")) is None),
        },
        "perf_pass_counts": perf_pass_counts,
        "perf_targets": {op: round(geomean(vals), 4) for op, vals in perf_targets_by_operator.items()},
        "perf_actuals": {op: round(geomean(vals), 4) for op, vals in perf_actuals_by_operator.items()},
        "perf_gaps": {op: round(sum(vals) / len(vals), 4) for op, vals in perf_gaps_by_operator.items()},
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark artifact utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    merge_parser = sub.add_parser(
        "merge-evidence",
        help="Safely merge partial perf_*.csv evidence into a canonical run directory",
    )
    merge_parser.add_argument("--source", required=True, type=Path)
    merge_parser.add_argument("--target", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "merge-evidence":
        result = merge_perf_evidence(args.source, args.target)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
