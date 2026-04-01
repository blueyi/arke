# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Export benchmark results to CSV and save kernel artifacts.

Produces:
- benchmarks/results/benchmark_results.csv   — flat table of all trials
- benchmarks/results/kernels/                 — generated Triton kernel files
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.tasks import BENCHMARK_TASKS, BenchmarkTask


def _get_task_metadata(task_name: str) -> dict:
    """Extract shape/param/op metadata from a task definition."""
    for t in BENCHMARK_TASKS:
        if t.name == task_name:
            return _extract_metadata(t)
    return {}


def _extract_metadata(task: BenchmarkTask) -> dict:
    """Extract structured metadata from a BenchmarkTask."""
    ir = task.semantic_ir
    params = []
    for p in ir.params:
        params.append({
            "name": p.name,
            "shape": p.shape,
            "dtype": p.dtype,
        })

    ops = []
    for node in ir.nodes:
        # Convert inputs to plain strings (ParamRef/NodeRef → name)
        inputs = {}
        for k, v in node.inputs.items():
            if hasattr(v, "name"):
                inputs[k] = v.name
            elif hasattr(v, "id"):
                inputs[k] = v.id
            else:
                inputs[k] = str(v)
        ops.append({
            "id": node.id,
            "op": node.op,
            "inputs": inputs,
        })

    out_shape = ir.return_type.shape if ir.return_type else None
    out_dtype = ir.return_type.dtype if ir.return_type else None

    return {
        "kernel_id": ir.kernel_id,
        "target_hw": task.target_hw,
        "dtype": task.dtype,
        "tags": task.tags,
        "params": params,
        "ops": ops,
        "output_shape": out_shape,
        "output_dtype": out_dtype,
    }


def _shape_str(shape: list) -> str:
    """Convert shape list to string like '1024x1024'."""
    return "x".join(str(s) for s in shape)


def export_csv(report_path: str, output_path: str | None = None) -> str:
    """Export benchmark report JSON to a detailed CSV file.

    Args:
        report_path: Path to benchmark_report.json
        output_path: Output CSV path (default: same dir as report)

    Returns:
        Path to the generated CSV file.
    """
    report_path = Path(report_path)
    if output_path is None:
        output_path = report_path.parent / "benchmark_results.csv"
    else:
        output_path = Path(output_path)

    with open(report_path) as f:
        report = json.load(f)

    rows = []
    for method_key in ["arke", "direct"]:
        method_data = report.get(method_key, {})
        for task_name, task_data in method_data.items():
            meta = _get_task_metadata(task_name)
            for trial_data in task_data.get("trials", []):
                row = _build_csv_row(task_name, method_key, meta, trial_data)
                rows.append(row)

    # Sort: task_name, method, trial
    rows.sort(key=lambda r: (r["task_name"], r["method"], r["trial"]))

    # Write CSV
    if not rows:
        return str(output_path)

    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(output_path)


def _build_csv_row(
    task_name: str,
    method: str,
    meta: dict,
    trial: dict,
) -> dict:
    """Build one CSV row from trial data + task metadata."""
    # Extract input shapes as readable strings
    params = meta.get("params", [])
    input_shapes = {}
    for p in params:
        input_shapes[p["name"]] = _shape_str(p["shape"])

    # Build operation description
    ops = meta.get("ops", [])
    ops_desc = " → ".join(n["op"] for n in ops)

    # Compute dimensions for matmul-like tasks
    m_dim = n_dim = k_dim = ""
    if params:
        shapes = [p["shape"] for p in params]
        if len(params) == 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
            # Matmul: A[M,K] × B[K,N]
            m_dim = str(shapes[0][0])
            k_dim = str(shapes[0][1])
            n_dim = str(shapes[1][1])
        elif len(params) == 1 and len(shapes[0]) == 2:
            # Element-wise: X[M,N]
            m_dim = str(shapes[0][0])
            n_dim = str(shapes[0][1])

    return {
        "task_name": task_name,
        "method": method,
        "trial": trial.get("trial", 0),
        "correct": trial.get("correct", False),
        "vs_baseline": trial.get("vs_baseline"),
        "latency_us": trial.get("latency_us"),
        "tflops": trial.get("tflops"),
        "kernel_id": meta.get("kernel_id", ""),
        "dtype": meta.get("dtype", ""),
        "target_hw": meta.get("target_hw", ""),
        "operations": ops_desc,
        "tags": ",".join(meta.get("tags", [])),
        "M": m_dim,
        "N": n_dim,
        "K": k_dim,
        "input_A_shape": input_shapes.get("A", input_shapes.get("X", "")),
        "input_B_shape": input_shapes.get("B", ""),
        "output_shape": _shape_str(meta["output_shape"]) if meta.get("output_shape") else "",
        "output_dtype": meta.get("output_dtype", ""),
        "decisions": trial.get("decisions", 0),
        "tool_calls": trial.get("tool_calls", 0),
        "tokens_in": trial.get("tokens_in", 0),
        "tokens_out": trial.get("tokens_out", 0),
        "total_tokens": trial.get("tokens_in", 0) + trial.get("tokens_out", 0),
        "duration_s": round(trial.get("duration_s", 0), 2),
        "error": trial.get("error", ""),
    }


def export_task_catalog(output_path: str | None = None) -> str:
    """Export a catalog of all benchmark tasks with full metadata.

    Args:
        output_path: Output CSV path

    Returns:
        Path to the generated CSV file.
    """
    if output_path is None:
        output_path = "benchmarks/results/task_catalog.csv"
    output_path = Path(output_path)

    rows = []
    for task in BENCHMARK_TASKS:
        meta = _extract_metadata(task)
        params = meta.get("params", [])
        input_shapes = {p["name"]: _shape_str(p["shape"]) for p in params}

        # Dimensions
        m_dim = n_dim = k_dim = ""
        shapes = [p["shape"] for p in params]
        if len(params) == 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
            m_dim = str(shapes[0][0])
            k_dim = str(shapes[0][1])
            n_dim = str(shapes[1][1])
        elif len(params) == 1 and len(shapes[0]) == 2:
            m_dim = str(shapes[0][0])
            n_dim = str(shapes[0][1])

        ops_desc = " → ".join(n["op"] for n in meta.get("ops", []))

        rows.append({
            "task_name": task.name,
            "description": task.description,
            "kernel_id": meta["kernel_id"],
            "dtype": task.dtype,
            "target_hw": task.target_hw,
            "operations": ops_desc,
            "tags": ",".join(task.tags),
            "M": m_dim,
            "N": n_dim,
            "K": k_dim,
            "input_A_shape": input_shapes.get("A", input_shapes.get("X", "")),
            "input_B_shape": input_shapes.get("B", ""),
            "output_shape": _shape_str(meta["output_shape"]) if meta.get("output_shape") else "",
            "output_dtype": meta["output_dtype"],
            "num_params": len(params),
            "num_ops": len(meta.get("ops", [])),
            "param_details": json.dumps(params),
            "op_details": json.dumps(meta.get("ops", [])),
        })

    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return str(output_path)
