# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gate verification framework for Arke Phase 1.

Gates are the contract between design and development. Each Gate defines
verifiable acceptance standards for a Stage. Once finalized, Gates are locked.

Entry point:
    python -m benchmarks gate G0       # Run Gate G0
    python -m benchmarks gate G6 --tier 2  # Run Gate G6 with tier 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass
class GateResult:
    """Single criterion result within a Gate."""
    gate: str  # e.g., "G6"
    criterion: str  # e.g., "G6.1"
    description: str  # Human-readable description
    category: Literal["function", "performance", "correctness", "regression"]
    passed: bool
    details: str  # Additional info (counts, ratios, error messages)


@dataclass
class GateSummary:
    """Summary of all criteria for a Gate."""
    gate: str
    tier: int
    total: int
    passed: int
    failed: int
    results: list[GateResult]

    @property
    def pass_rate(self) -> float:
        """Percentage of criteria passed."""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gate": self.gate,
            "tier": self.tier,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.pass_rate:.1f}%",
            "results": [asdict(r) for r in self.results],
        }


def run_gate(gate_id: str, tier: int = 2) -> GateSummary:
    """Run a specific Gate and return summary.

    Args:
        gate_id: Gate identifier (e.g., "G0", "G6")
        tier: Benchmark tier (1-3, higher = more comprehensive)

    Returns:
        GateSummary with all criterion results
    """
    gate_id = gate_id.upper()

    if gate_id == "G6":
        from benchmarks.gate_g6 import run_g6
        return run_g6(tier=tier)
    elif gate_id == "G7":
        from benchmarks.gate_g7 import run_g7
        return run_g7(tier=tier)
    elif gate_id == "G8":
        from benchmarks.gate_g8 import run_g8
        return run_g8(tier=tier)
    else:
        # Placeholder for other gates
        return GateSummary(
            gate=gate_id,
            tier=tier,
            total=1,
            passed=0,
            failed=1,
            results=[
                GateResult(
                    gate=gate_id,
                    criterion=f"{gate_id}.1",
                    description=f"Gate {gate_id} not yet implemented",
                    category="function",
                    passed=False,
                    details=f"Gate {gate_id} runner not found",
                )
            ],
        )


def main() -> None:
    """CLI entry point for gate verification."""
    parser = argparse.ArgumentParser(
        description="Arke Phase 1 Gate Verification",
        prog="python -m benchmarks gate",
    )
    parser.add_argument("gate", help="Gate ID (e.g., G0, G6)")
    parser.add_argument(
        "--tier",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Benchmark tier (1=quick, 2=standard, 3=comprehensive)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to file",
    )

    args = parser.parse_args()

    # Run the gate
    summary = run_gate(args.gate, tier=args.tier)

    # Format output
    if args.json:
        output = json.dumps(summary.to_dict(), indent=2)
    else:
        # Human-readable format
        lines = [
            f"{'='*80}",
            f"Gate {summary.gate} Verification (Tier {summary.tier})",
            f"{'='*80}",
            "",
            f"Results: {summary.passed}/{summary.total} criteria passed ({summary.pass_rate:.1f}%)",
            "",
        ]

        # Group by category
        by_category = {}
        for result in summary.results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)

        for category in ["function", "correctness", "performance", "regression"]:
            if category not in by_category:
                continue

            lines.append(f"[{category.upper()}]")
            for result in by_category[category]:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                lines.append(f"  {result.criterion}: {status}")
                lines.append(f"    {result.description}")
                if result.details:
                    lines.append(f"    {result.details}")
            lines.append("")

        lines.append(f"{'='*80}")
        if summary.failed == 0:
            lines.append(f"✅ Gate {summary.gate} PASSED")
        else:
            lines.append(f"❌ Gate {summary.gate} FAILED ({summary.failed} criteria)")
        lines.append(f"{'='*80}")

        output = "\n".join(lines)

    # Output
    print(output)

    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"\nResults saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if summary.failed == 0 else 1)


if __name__ == "__main__":
    main()
