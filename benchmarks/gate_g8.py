# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 8 Gate G8 MVP verification.

This runner covers the machine-checkable MVP contract introduced at the start of
Stage 8.  It is not the full final G8 acceptance suite; it verifies that the two
critical entry points now exist and emit stable artifacts for future real-GPU and
real-model measurements.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.gate import GateResult, GateSummary

REPO_ROOT = Path(__file__).resolve().parent.parent
MATMUL_AK = REPO_ROOT / "examples" / "operators" / "01_matmul.ak"


def _run_cmd(args: list[str], timeout: int = 180) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    detail = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, detail[-2500:]


def _run_pytest(args: list[str]) -> tuple[bool, str]:
    return _run_cmd([sys.executable, "-m", "pytest", "-q", *args], timeout=300)


def _check_optimize_cli_contract() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="arke-g8-opt-") as tmp:
        ok, detail = _run_cmd([
            sys.executable,
            "-m",
            "arke.cli",
            "optimize",
            str(MATMUL_AK),
            "--output",
            tmp,
            "--cycles",
            "3",
            "--json",
        ])
        if not ok:
            return False, detail
        summary = json.loads((Path(tmp) / "summary.json").read_text())
        trajectory = (Path(tmp) / "trajectory.jsonl").read_text().splitlines()
        strategy = json.loads((Path(tmp) / "strategy.json").read_text())
        if not summary.get("success"):
            return False, json.dumps(summary, indent=2)
        if summary.get("cycles_completed") != 3:
            return False, f"cycles_completed={summary.get('cycles_completed')}"
        if len(strategy.get("decisions", [])) < 3:
            return False, "generated strategy has fewer than 3 decisions"
        actions = [json.loads(line) for line in trajectory if '"event_type": "action"' in line]
        tools = [entry.get("tool") for entry in actions]
        required = ["compile", "profile", "adjust"] * 3
        if tools != required:
            return False, f"unexpected trajectory tool order: {tools}"
        return True, (
            f"cycles=3 decisions={summary['decision_count']} "
            f"trajectory_events={len(trajectory)}"
        )


def _check_bench_l3_mock_contract() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="arke-g8-l3-") as tmp:
        ok, detail = _run_cmd([
            sys.executable,
            "-m",
            "benchmarks.bench_l3",
            "--mock",
            "--device",
            "cpu",
            "--seq-len",
            "8",
            "--warmup",
            "1",
            "--runs",
            "2",
            "--output",
            tmp,
        ], timeout=240)
        if not ok:
            return False, detail
        run_dirs = sorted(Path(tmp).glob("*"))
        if not run_dirs:
            return False, "bench_l3 produced no run directory"
        run_dir = run_dirs[-1]
        summary = json.loads((run_dir / "summary.json").read_text())
        required_files = [
            "config.json",
            "hardware.json",
            "sources.json",
            "gpt2_results.csv",
            "results.json",
            "summary.json",
        ]
        missing = [name for name in required_files if not (run_dir / name).exists()]
        if missing:
            return False, f"missing artifacts: {missing}"
        if summary.get("compile_rows") != 1:
            return False, json.dumps(summary, indent=2)
        return True, (
            f"rows={summary['rows']} compile_rows={summary['compile_rows']} "
            f"g8_gpt2_pass={summary['g8_gpt2_pass']}"
        )


def run_g8(tier: int = 2) -> GateSummary:
    results: list[GateResult] = []

    ok, detail = _check_optimize_cli_contract()
    results.append(GateResult(
        "G8",
        "G8.MVP.1",
        "arke optimize emits bounded StrategyIR and 3 compile->profile->adjust cycles",
        "function",
        ok,
        detail,
    ))

    ok, detail = _check_bench_l3_mock_contract()
    results.append(GateResult(
        "G8",
        "G8.MVP.2",
        "bench_l3 emits GPT-2 eager vs torch.compile CSV/JSON artifacts",
        "function",
        ok,
        detail,
    ))

    ok, detail = _run_pytest([
        "tests/test_agent_optimize.py",
        "tests/test_bench_l3.py",
    ])
    results.append(GateResult(
        "G8",
        "G8.MVP.3",
        "Stage 8 MVP regression tests pass",
        "regression",
        ok,
        detail,
    ))

    passed = sum(1 for result in results if result.passed)
    return GateSummary(
        gate="G8",
        tier=tier,
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )
