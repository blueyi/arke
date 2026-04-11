# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


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


def write_perf_csv_from_l1(raw_csv: Path, out_csv: Path) -> Path:
    rows = list(csv.DictReader(raw_csv.open()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "operator", "shape_tag", "baseline", "latency_us",
        "latency_min_us", "tflops", "ratio_vs_baseline", "status", "reason", "retryable",
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
                })
    return out_csv


def write_perf_csv_from_l2(raw_csv: Path, out_csv: Path) -> Path:
    rows = list(csv.DictReader(raw_csv.open()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "operator", "shape_tag", "approach", "latency_us",
        "latency_min_us", "tflops", "ratio_vs_baseline", "status", "reason", "retryable",
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
    ratios_by_operator: dict[str, list[float]] = {}
    for row in csv.DictReader(perf_all.open()):
        operator = row.get("operator", "unknown")
        ratio = _safe_float(row.get("ratio_vs_baseline"))
        if ratio is not None and ratio > 0:
            ratios_by_operator.setdefault(operator, []).append(ratio)
    def geomean(vals: list[float]) -> float:
        return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
    status_counts: dict[str, int] = {}
    for row in csv.DictReader(perf_all.open()):
        status = row.get("status", "ok")
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "overall_geomean": round(geomean([v for vals in ratios_by_operator.values() for v in vals]), 4)
        if ratios_by_operator else 0.0,
        "op_scores": {op: round(geomean(vals), 4) for op, vals in ratios_by_operator.items()},
        "total_shapes": sum(len(vals) for vals in ratios_by_operator.values()),
        "operators": sorted(ratios_by_operator.keys()),
        "status_counts": status_counts,
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return out
