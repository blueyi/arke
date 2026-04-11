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
                ],
            )
            writer.writeheader()
            writer.writerow({
                "op": "relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 0,
                "baseline": "PyTorch-eager", "priority": 3, "source": "x",
                "latency_us": "10.0", "latency_min_us": "9.0", "tflops": "", "status": "ok", "reason": "", "retryable": "false",
            })
            writer.writerow({
                "op": "relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 0,
                "baseline": "FlagGems", "priority": 1, "source": "x",
                "latency_us": "8.0", "latency_min_us": "7.5", "tflops": "", "status": "ok", "reason": "", "retryable": "false",
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

    def test_write_perf_csv_from_l2_and_summary(self, tmp_path: Path):
        raw = tmp_path / "matmul_relu_results.csv"
        with raw.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "op", "shape_tag", "M", "N", "K", "approach",
                    "source", "latency_us", "latency_min_us", "tflops", "status", "reason", "retryable",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "op": "matmul_relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 1024,
                "approach": "separate", "source": "x",
                "latency_us": "12.0", "latency_min_us": "11.0", "tflops": "1.0", "status": "ok", "reason": "", "retryable": "false",
            })
            writer.writerow({
                "op": "matmul_relu", "shape_tag": "square-1k", "M": 1024, "N": 1024, "K": 1024,
                "approach": "torch.compile", "source": "x",
                "latency_us": "9.0", "latency_min_us": "8.5", "tflops": "1.2", "status": "oom", "reason": "CUDA out of memory", "retryable": "true",
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
