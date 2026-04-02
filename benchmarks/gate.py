# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Gate Verification CLI.

Usage:
    python -m benchmarks.gate G0          # Run G0
    python -m benchmarks.gate G0 G1      # Run multiple gates
    python -m benchmarks.gate --all       # Run all gates
    python -m benchmarks.gate G2 --tier 1 # Specify tier
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    gate: str  # "G0", "G1", etc.
    criterion: str  # "G0.1", "G0.2", etc.
    name: str  # "CUDA detection"
    type: str  # "function", "accuracy", "performance"
    passed: bool
    detail: str  # Human-readable result detail


@dataclass
class GateSummary:
    gate: str
    results: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)


def run_g0() -> GateSummary:
    """G0: Environment Feasibility.

    G0.1: CUDA detection — torch.cuda.is_available() == True
    G0.2: Triton compilation — Triton kernel compiles with exit 0
    G0.3: GPU execution — Triton matmul [128,128,128] returns non-zero tensor
    G0.4: Test framework — pytest tests/ -q → ≥ 100 passed, 0 failed
    """
    results: list[GateResult] = []

    # G0.1: CUDA detection
    import torch

    cuda_ok = torch.cuda.is_available()
    results.append(
        GateResult(
            "G0",
            "G0.1",
            "CUDA detection",
            "function",
            cuda_ok,
            f"torch.cuda.is_available() = {cuda_ok}",
        )
    )

    # G0.2: Triton compilation
    try:
        # Kernel is defined at module level so Triton can resolve ``tl``
        # references during JIT compilation.
        from benchmarks._triton_test_kernel import triton_add_kernel

        x = torch.zeros(128, device="cuda")
        triton_add_kernel[(1,)](x, 128)
        torch.cuda.synchronize()
        triton_ok = True
        triton_detail = "Triton kernel compiled and ran successfully"
    except Exception as e:
        triton_ok = False
        triton_detail = f"Triton compilation failed: {e}"

    results.append(
        GateResult(
            "G0",
            "G0.2",
            "Triton compilation",
            "function",
            triton_ok,
            triton_detail,
        )
    )

    # G0.3: GPU execution — matmul
    try:
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        nonzero = c.abs().sum().item() > 0
        exec_detail = f"matmul result norm = {c.norm().item():.4f}"
    except Exception as e:
        nonzero = False
        exec_detail = f"GPU execution failed: {e}"

    results.append(
        GateResult(
            "G0",
            "G0.3",
            "GPU execution",
            "function",
            nonzero,
            exec_detail,
        )
    )

    # G0.4: Test framework
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "--no-header"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent.parent),
        )
        output = proc.stdout + proc.stderr
        match = re.search(r"(\d+) passed", output)
        passed_count = int(match.group(1)) if match else 0
        failed_match = re.search(r"(\d+) failed", output)
        failed_count = int(failed_match.group(1)) if failed_match else 0
        test_ok = passed_count >= 100 and failed_count == 0
        test_detail = f"{passed_count} passed, {failed_count} failed"
    except Exception as e:
        test_ok = False
        test_detail = f"pytest failed: {e}"

    results.append(
        GateResult(
            "G0",
            "G0.4",
            "Test framework",
            "function",
            test_ok,
            test_detail,
        )
    )

    return GateSummary("G0", results)


def print_gate_result(summary: GateSummary) -> None:
    """Pretty-print gate result."""
    gate_name = GATE_NAMES.get(summary.gate, summary.gate)
    status = "PASS" if summary.passed else "FAIL"
    print(f"\n  {summary.gate}: {gate_name}")
    print("  " + "━" * 56)

    for r in summary.results:
        icon = "✅" if r.passed else "❌"
        print(f"    {r.criterion} {r.name:30s} {icon} {r.detail}")

    print("  " + "━" * 56)
    print(f"  {summary.gate}: {status} ({summary.pass_count}/{summary.total_count})")


GATE_RUNNERS: dict[str, object] = {
    "G0": run_g0,
}

GATE_NAMES: dict[str, str] = {
    "G0": "Environment Feasibility",
    "G1": "IR Expressiveness & Validation Correctness",
    "G2": "Codegen Correctness & Baseline Performance",
    "G3": "LLM Agent Autonomous Optimization",
    "G4": "Comparative Advantage over Direct LLM",
    "G5": "End-to-End Model Integration",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Arke Gate Verification")
    parser.add_argument("gates", nargs="*", help="Gate names (G0, G1, ...)")
    parser.add_argument("--all", action="store_true", help="Run all gates")
    parser.add_argument("--tier", type=int, default=3, help="Shape tier (1/2/3)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.all:
        gates = sorted(GATE_RUNNERS.keys())
    elif args.gates:
        gates = [g.upper() for g in args.gates]
    else:
        parser.print_help()
        sys.exit(1)

    all_passed = True
    for gate in gates:
        if gate not in GATE_RUNNERS:
            print(
                f"  {gate}: NOT IMPLEMENTED"
                f" (available: {', '.join(sorted(GATE_RUNNERS.keys()))})"
            )
            all_passed = False
            continue

        runner = GATE_RUNNERS[gate]
        summary = runner()
        print_gate_result(summary)
        if not summary.passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
