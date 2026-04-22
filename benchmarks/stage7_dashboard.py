# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 7 Track 6 dashboard synthesis compatibility wrapper.

The benchmark dashboard implementation is stage-agnostic and lives in
``benchmarks.dashboard``. This module preserves the existing Stage 7 entrypoint
and defaults while delegating to the generic implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.dashboard import (
    DEFAULT_AUDIT_REPORT,
    DEFAULT_COVERAGE_GAP,
    DEFAULT_OPERATOR_SHAPE_STATS,
    DEFAULT_OUTPUT,
    _cli as _generic_cli,
    build_benchmark_dashboard,
    format_text_summary,
)


def build_stage7_dashboard(
    coverage_gap_path: Path = DEFAULT_COVERAGE_GAP,
    audit_report_path: Path = DEFAULT_AUDIT_REPORT,
    operator_shape_stats_path: Path = DEFAULT_OPERATOR_SHAPE_STATS,
) -> dict[str, Any]:
    return build_benchmark_dashboard(
        coverage_gap_path=coverage_gap_path,
        audit_report_path=audit_report_path,
        operator_shape_stats_path=operator_shape_stats_path,
        default_stage="S7",
        default_gate="G7",
        title="Stage 7 benchmark dashboard",
        text_summary_label="Stage 7 dashboard",
    )


def main() -> None:
    import sys

    argv = sys.argv[1:]
    has_label_override = any(arg.startswith("--summary-label") for arg in argv)
    has_title_override = any(arg.startswith("--title") for arg in argv)
    if not has_label_override:
        argv = [*argv, "--summary-label", "Stage 7 dashboard"]
    if not has_title_override:
        argv = [*argv, "--title", "Stage 7 benchmark dashboard"]
    _generic_cli(argv)


__all__ = ["build_stage7_dashboard", "format_text_summary", "main"]


if __name__ == "__main__":
    main()
