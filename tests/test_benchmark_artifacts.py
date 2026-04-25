# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.artifacts import (
    merge_perf_all,
    write_perf_csv_from_l1,
    write_perf_csv_from_l2,
    write_summary,
)


class TestBenchmarkArtifacts:
    def test_write_perf_csv_from_l1_and_summary(self, tmp_path: Path):
        raw = tmp_path / "relu_results.csv"
        with raw.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "op", "shape_tag", "M", "N", "K", "baseline", "priority",
                    "source", "latency_us", "latency_min_us", "tflops", "status", "reason", "retryable",
                    "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
                    "correctness_status", "correctness_reason",
                    "memory_bytes_required", "memory_bytes_budget", "memory_ratio", "memory_policy",
                    "perf_target", "perf_actual", "perf_pass", "perf_gap",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "op": "relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 0,
                "baseline": "PyTorch-eager", "priority": 3, "source": "x",
                "latency_us": "10.0", "latency_min_us": "9.0", "tflops": "", "status": "ok", "reason": "", "retryable": "false",
                "allclose": "true", "max_abs_diff": "0.0", "mean_abs_diff": "0.0", "rtol": "1e-5", "atol": "1e-6",
                "correctness_status": "pass", "correctness_reason": "",
                "memory_bytes_required": "0", "memory_bytes_budget": "1000", "memory_ratio": "0.0", "memory_policy": "none",
                "perf_target": "1.0", "perf_actual": "1.0", "perf_pass": "true", "perf_gap": "0.0",
            })
            writer.writerow({
                "op": "relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 0,
                "baseline": "FlagGems", "priority": 1, "source": "x",
                "latency_us": "8.0", "latency_min_us": "7.5", "tflops": "", "status": "ok", "reason": "", "retryable": "false",
                "allclose": "true", "max_abs_diff": "0.0", "mean_abs_diff": "0.0", "rtol": "1e-5", "atol": "1e-6",
                "correctness_status": "pass", "correctness_reason": "",
                "memory_bytes_required": "2048", "memory_bytes_budget": "1024", "memory_ratio": "1.25", "memory_policy": "dense_matmul",
                "perf_target": "1.0", "perf_actual": "1.25", "perf_pass": "true", "perf_gap": "0.25",
            })
        perf = write_perf_csv_from_l1(raw, tmp_path / "perf_relu.csv")
        assert perf.exists()
        merge_perf_all(tmp_path)
        summary = write_summary(tmp_path)
        assert summary is not None and summary.exists()
        data = json.loads(summary.read_text())
        assert data["operators"] == ["relu"]
        assert data["overall_geomean"] > 0
        assert data["status_counts"]["ok"] == 2
        assert data["correctness_counts"]["pass"] == 2
        assert data["memory_policy_counts"]["dense_matmul"] == 1
        assert data["memory_policy_counts"]["none"] == 1
        assert data["memory_pressure_rows"] == 1
        assert data["perf_target_counts"] == {"with_target": 2, "without_target": 0}
        assert data["perf_pass_counts"]["true"] == 2
        assert data["perf_targets"]["relu"] == 1.0
        assert data["perf_actuals"]["relu"] > 1.0
        assert data["perf_gaps"]["relu"] == 0.125

    def test_write_perf_csv_from_l2_and_summary(self, tmp_path: Path):
        raw = tmp_path / "matmul_relu_results.csv"
        with raw.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "op", "shape_tag", "M", "N", "K", "approach",
                    "source", "latency_us", "latency_min_us", "tflops", "status", "reason", "retryable",
                    "allclose", "max_abs_diff", "mean_abs_diff", "rtol", "atol",
                    "correctness_status", "correctness_reason",
                    "memory_bytes_required", "memory_bytes_budget", "memory_ratio", "memory_policy",
                    "perf_target", "perf_actual", "perf_pass", "perf_gap",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "op": "matmul_relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 1024,
                "approach": "separate", "source": "x",
                "latency_us": "12.0", "latency_min_us": "11.0", "tflops": "1.0", "status": "ok", "reason": "", "retryable": "false",
                "allclose": "true", "max_abs_diff": "0.0", "mean_abs_diff": "0.0", "rtol": "1e-5", "atol": "1e-6",
                "correctness_status": "pass", "correctness_reason": "",
                "memory_bytes_required": "0", "memory_bytes_budget": "1000", "memory_ratio": "0.0", "memory_policy": "none",
                "perf_target": "1.0", "perf_actual": "1.0", "perf_pass": "true", "perf_gap": "0.0",
            })
            writer.writerow({
                "op": "matmul_relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 1024,
                "approach": "torch.compile", "source": "x",
                "latency_us": "9.0", "latency_min_us": "8.5", "tflops": "1.2", "status": "oom", "reason": "CUDA out of memory", "retryable": "true",
                "allclose": "", "max_abs_diff": "", "mean_abs_diff": "", "rtol": "", "atol": "",
                "correctness_status": "skipped", "correctness_reason": "oom",
                "memory_bytes_required": "2048", "memory_bytes_budget": "1024", "memory_ratio": "1.5", "memory_policy": "dense_matmul",
                "perf_target": "1.0", "perf_actual": "1.3333", "perf_pass": "true", "perf_gap": "0.3333",
            })
        perf = write_perf_csv_from_l2(raw, tmp_path / "perf_matmul_relu.csv")
        assert perf.exists()
        merge_perf_all(tmp_path)
        summary = write_summary(tmp_path)
        assert summary is not None and summary.exists()
        data = json.loads(summary.read_text())
        assert data["operators"] == ["matmul_relu"]
        assert data["op_scores"]["matmul_relu"] > 0
        assert data["status_counts"]["ok"] == 1
        assert data["status_counts"]["oom"] == 1
        assert data["correctness_counts"]["pass"] == 1
        assert data["correctness_counts"]["skipped"] == 1
        assert data["memory_policy_counts"]["dense_matmul"] == 1
        assert data["memory_policy_counts"]["none"] == 1
        assert data["memory_pressure_rows"] == 1
        assert data["perf_target_counts"] == {"with_target": 2, "without_target": 0}
        assert data["perf_pass_counts"]["true"] == 2
        assert data["perf_targets"]["matmul_relu"] == 1.0
        assert data["perf_actuals"]["matmul_relu"] > 1.0
        assert data["perf_gaps"]["matmul_relu"] == 0.1666
