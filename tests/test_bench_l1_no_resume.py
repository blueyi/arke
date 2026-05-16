# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Regression test for bench_l1 --no-resume CSV truncation bug.

Bug history (2026-05-16): When `bench_l1.run` was invoked with `resume=False`
on an already-populated per-op CSV (e.g. `matmul_results.csv`), the old rows
were not cleared — only `ensure_header` was called, which is a no-op when the
file already has a header. Subsequent measurement emissions then appended on
top of stale rows, silently double-counting or mixing stale+fresh data for the
same (op, shape_tag, baseline) key.

The fix changes the conditional at bench_l1.py:1268 from

    if kept_rows or not csv_path.exists():
        # truncate to header (+ kept_rows)
    else:
        ensure_header(...)  # ← bug: keeps stale rows when resume=False

to

    if kept_rows or not csv_path.exists() or not resume:
        # truncate to header (+ kept_rows)
    else:
        ensure_header(...)

This test exercises the truncation path directly: it stages an existing CSV
with stale rows, runs the per-op CSV-prep block with `resume=False`, and
asserts the CSV is left with only the header.

PERF_ALL.csv is intentionally NOT touched by --no-resume — it is rebuilt by
aggregating all `perf_<op>.csv` files at the end of the run. This test only
covers the per-op CSV path.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# Pull the constants & helper from the module under test
from benchmarks.bench_l1 import L1_FIELDNAMES, L1_KEY_FIELDS
from benchmarks import progress as _progress


def _stage_existing_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Write a CSV with header + rows, simulating a prior bench run."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=L1_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in L1_FIELDNAMES})


def _run_csv_prep_block(csv_path: Path, resume: bool, retry_policy: str = "auto") -> None:
    """
    Replicate the per-op CSV-prep block from bench_l1.run() (lines ~1240-1281).

    This is the block under test. If `resume=False`, the CSV must end up with
    only its header row. If `resume=True`, the CSV must keep existing rows that
    pass `should_skip`.
    """
    existing_rows = _progress.load_existing_rows(csv_path) if resume else []
    existing_index = _progress.index_rows(existing_rows, L1_KEY_FIELDS)

    skip_keys: set[tuple[str, str, str]] = set()
    cached_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for key, row in existing_index.items():
        if _progress.should_skip(row, retry_policy):
            skip_keys.add(key)
            cached_rows[key] = row

    kept_rows = [
        row for key, row in existing_index.items() if key in skip_keys
    ]
    # The fix: include `or not resume` in the truncate condition
    if kept_rows or not csv_path.exists() or not resume:
        tmp = csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=L1_FIELDNAMES, extrasaction="ignore"
            )
            writer.writeheader()
            for row in kept_rows:
                writer.writerow({k: row.get(k, "") for k in L1_FIELDNAMES})
        tmp.replace(csv_path)
    else:
        _progress.ensure_header(csv_path, L1_FIELDNAMES)


# A trio of "stale" rows simulating a prior matmul run
_STALE_ROWS = [
    {
        "op": "matmul",
        "shape_tag": "tiny",
        "baseline": "Arke",
        "status": "ok",
        "latency_us": "100.0",
    },
    {
        "op": "matmul",
        "shape_tag": "tiny",
        "baseline": "FlagGems",
        "status": "ok",
        "latency_us": "95.0",
    },
    {
        "op": "matmul",
        "shape_tag": "tiny",
        "baseline": "Triton-Tutorial",
        "status": "ok",
        "latency_us": "98.0",
    },
]


class TestNoResumeCsvTruncation:
    """Verify --no-resume (resume=False) truncates per-op CSV to header only."""

    def test_no_resume_clears_stale_op_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "matmul_results.csv"
        _stage_existing_csv(csv_path, _STALE_ROWS)

        # Sanity: the staged CSV has 3 stale rows + header
        with csv_path.open() as f:
            lines_before = f.readlines()
        assert len(lines_before) == 4, "staged CSV should have header + 3 rows"

        # Run the CSV-prep block with resume=False (the buggy path)
        _run_csv_prep_block(csv_path, resume=False)

        # After --no-resume, the CSV must contain ONLY the header line
        with csv_path.open() as f:
            lines_after = f.readlines()
        assert len(lines_after) == 1, (
            f"resume=False must truncate stale rows; "
            f"got {len(lines_after)} lines (expected 1 header-only)"
        )
        # And the header is intact and well-formed
        assert lines_after[0].strip().startswith("op,"), (
            f"truncated CSV must keep the L1 header row; "
            f"got first line: {lines_after[0]!r}"
        )

    def test_resume_keeps_stale_op_csv(self, tmp_path: Path) -> None:
        """Counter-test: resume=True must NOT truncate the CSV."""
        csv_path = tmp_path / "matmul_results.csv"
        _stage_existing_csv(csv_path, _STALE_ROWS)

        _run_csv_prep_block(csv_path, resume=True)

        # With resume=True, all 3 ok-status stale rows should be kept (skip-policy)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            kept = list(reader)
        assert len(kept) == 3, (
            f"resume=True must keep ok-status rows; got {len(kept)}"
        )
        # And the keys match what we staged
        kept_keys = {
            (r["op"], r["shape_tag"], r["baseline"]) for r in kept
        }
        staged_keys = {
            (r["op"], r["shape_tag"], r["baseline"]) for r in _STALE_ROWS
        }
        assert kept_keys == staged_keys

    def test_no_resume_on_missing_csv_creates_header(self, tmp_path: Path) -> None:
        """resume=False on a non-existent CSV should create a header-only file."""
        csv_path = tmp_path / "matmul_results.csv"
        assert not csv_path.exists()

        _run_csv_prep_block(csv_path, resume=False)

        assert csv_path.exists()
        with csv_path.open() as f:
            lines = f.readlines()
        assert len(lines) == 1, "new CSV should have only header"
        assert lines[0].strip().startswith("op,"), "header must be present"
