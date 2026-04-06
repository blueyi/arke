# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified performance CSV writer.

Usage:
    from benchmarks.perf_csv import PerfCSVWriter, PerfRow

    writer = PerfCSVWriter("benchmarks/results/stage1/gates/G2/performance/perf_matmul.csv")
    writer.write(PerfRow(
        stage="stage1", gate="G2", run_id="2026-04-05_012345",
        operator="matmul", op_tier=2, category="A",
        shape_tag="square-1k", shape_tier=2,
        benchmark_level=2, eval_layer="L1",
        M=1024, N=1024, K=1024, dtype="f16", backend="nvidia",
        method="arke", latency_us=39.8, correct=True,
        baseline_method="cublas", baseline_latency_us=44.2,
    ))
    writer.close()

See docs/design/stage1/benchmark/benchmark-csv-spec.md for full column specification.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class PerfRow:
    """One row in the unified performance CSV."""

    # === Required ===
    stage: str = ""
    gate: str = ""
    run_id: str = ""
    operator: str = ""
    op_tier: Optional[int] = None
    category: str = ""
    shape_tag: str = ""
    shape_tier: int = 0
    benchmark_level: Optional[int] = None
    eval_layer: Optional[str] = None
    dtype: str = "f16"
    backend: str = "nvidia"
    method: str = ""
    baseline_tier: Optional[str] = None
    latency_us: float = 0.0
    correct: bool = True

    # === Shape dimensions (optional, depends on operator) ===
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    batch: Optional[int] = None
    seq_len: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # === Performance details ===
    latency_min_us: Optional[float] = None
    latency_max_us: Optional[float] = None
    latency_std_us: Optional[float] = None
    tflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None

    # === Correctness details ===
    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None

    # === Baseline comparison ===
    baseline_method: Optional[str] = None
    baseline_latency_us: Optional[float] = None
    ratio_vs_baseline: Optional[float] = None

    # === Benchmark config ===
    warmup_iters: Optional[int] = None
    bench_iters: Optional[int] = None

    # === Hardware context ===
    gpu_name: Optional[str] = None
    gpu_mem_mb: Optional[int] = None
    cuda_version: Optional[str] = None
    triton_version: Optional[str] = None
    pytorch_version: Optional[str] = None

    # === Notes ===
    notes: Optional[str] = None

    def compute_ratio(self) -> None:
        """Auto-compute ratio_vs_baseline if both latencies are set."""
        if (
            self.baseline_latency_us
            and self.latency_us
            and self.latency_us > 0
        ):
            self.ratio_vs_baseline = round(
                self.baseline_latency_us / self.latency_us, 4
            )


# Canonical column order (matches PERF_CSV_SPEC.md)
COLUMNS = [f.name for f in fields(PerfRow)]


class PerfCSVWriter:
    """Write performance CSV files with the unified schema.

    Handles UTF-8 BOM for Excel compatibility.
    """

    def __init__(self, path: str | Path, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._append = append
        self._file = None
        self._writer = None

    def _open(self) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        if self._append and exists:
            self._file = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file, fieldnames=COLUMNS, extrasaction="ignore"
            )
        else:
            self._file = open(self.path, "w", newline="", encoding="utf-8-sig")
            self._writer = csv.DictWriter(
                self._file, fieldnames=COLUMNS, extrasaction="ignore"
            )
            self._writer.writeheader()

    def write(self, row: PerfRow) -> None:
        """Write a single row."""
        if self._writer is None:
            self._open()
        row.compute_ratio()
        d = asdict(row)
        # Convert booleans to Excel-friendly TRUE/FALSE
        for k, v in d.items():
            if isinstance(v, bool):
                d[k] = "TRUE" if v else "FALSE"
            elif v is None:
                d[k] = ""
        self._writer.writerow(d)

    def write_many(self, rows: list[PerfRow]) -> None:
        """Write multiple rows."""
        for row in rows:
            self.write(row)

    def close(self) -> None:
        """Flush and close the file."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def merge_stage_csvs(stage_dir: str | Path, output: str = "STAGE_PERF_ALL.csv") -> Path:
    """Merge all perf_*.csv files under a stage directory into one consolidated CSV."""
    stage_path = Path(stage_dir)
    out_path = stage_path / output
    all_rows = []

    for csv_file in sorted(stage_path.rglob("perf_*.csv")):
        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    with PerfCSVWriter(out_path) as writer:
        for row_dict in all_rows:
            # Convert back to PerfRow
            pr = PerfRow()
            for k, v in row_dict.items():
                if hasattr(pr, k) and v != "":
                    field_type = type(getattr(pr, k))
                    if field_type == bool or v in ("TRUE", "FALSE"):
                        setattr(pr, k, v == "TRUE")
                    elif field_type == int or (field_type == type(None) and k.endswith(("_mb", "_iters", "tier", "op_tier", "benchmark_level"))):
                        try:
                            setattr(pr, k, int(v))
                        except ValueError:
                            setattr(pr, k, v)
                    elif field_type == float or field_type == type(None):
                        try:
                            setattr(pr, k, float(v))
                        except ValueError:
                            setattr(pr, k, v)
                    else:
                        setattr(pr, k, v)
            writer.write(pr)

    return out_path
