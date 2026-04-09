# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 7 Gate G7 verification (implementation-oriented partial closure).

This gate runner focuses on criteria that are currently machine-verifiable from
code/tests in the repo. Performance/benchmark closure remains delegated to the
benchmark runners and result artifacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.gate import GateResult, GateSummary

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_pytest(args: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", *args]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    detail = (proc.stdout + "\n" + proc.stderr).strip()
    return ok, detail[-2000:]


def run_g7(tier: int = 2) -> GateSummary:
    results: list[GateResult] = []

    checks = [
        (
            "G7.1",
            "Lang/IR v2 spec-facing parser + symbolic shape + roundtrip tests",
            "function",
            [
                "tests/test_symbolic_shape.py",
                "tests/test_stage7_roundtrip.py",
                "tests/test_backend_agnostic.py",
            ],
        ),
        (
            "G7.2",
            "MLIR skeleton path verified on BL1 matmul",
            "function",
            [
                "tests/test_mlir_backend.py",
                "tests/test_pipeline.py::TestCompileAll::test_compile_matmul_emits_mlir_skeleton",
            ],
        ),
        (
            "G7.3",
            "Track 4 multi-layer lowering path stays green",
            "correctness",
            [
                "tests/test_track4_lowering.py",
                "tests/test_akir.py",
                "tests/test_rationale_e2e.py",
            ],
        ),
        (
            "G7.4",
            "L2 fusion benchmark harness basic coverage exists",
            "function",
            [
                "tests/test_benchmark_l2.py",
                "tests/test_benchmark_l2_fused_ce.py",
                "tests/test_benchmark_cli.py",
            ],
        ),
    ]

    for criterion, description, category, pytest_args in checks:
        passed, details = _run_pytest(pytest_args)
        results.append(
            GateResult(
                gate="G7",
                criterion=criterion,
                description=description,
                category=category,
                passed=passed,
                details=details,
            )
        )

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    return GateSummary(
        gate="G7",
        tier=tier,
        total=len(results),
        passed=passed,
        failed=failed,
        results=results,
    )
