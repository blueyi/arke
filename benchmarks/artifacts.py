# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

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


def write_perf_csv_from_l1(raw_csv: Path, out_csv: Path) -> Path:
    rows = list(csv.DictReader(raw_csv.open()))
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
                writer.writerow({
                    "operator": row.get("op", "unknown"),
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
                    "perf_target": row.get("perf_target", ""),
                    "perf_actual": row.get("perf_actual", row.get("ratio_vs_baseline", "")),
                    "perf_pass": row.get("perf_pass", ""),
                    "perf_gap": row.get("perf_gap", ""),
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
                    "perf_target": row.get("perf_target", ""),
                    "perf_actual": row.get("perf_actual", row.get("ratio_vs_baseline", "")),
                    "perf_pass": row.get("perf_pass", ""),
                    "perf_gap": row.get("perf_gap", ""),
                })
    return out_csv


def merge_perf_all(run_dir: Path) -> Path | None:
    rows: list[dict[str, str]] = []
    for csv_file in sorted(run_dir.glob("perf_*.csv")):
        rows.extend(list(csv.DictReader(csv_file.open())))
    if not rows:
        return None
    out = run_dir / "PERF_ALL.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
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
        "op_scores": {op: round(geomean(vals), 4) for op, vals in ratios_by_operator.items()},
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
