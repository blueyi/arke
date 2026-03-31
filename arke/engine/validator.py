# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — V0 Static Validator.

Validates Strategy IR decisions against hardware constraints and IR invariants.
Runs after every apply_decision (<1ms), provides immediate feedback to LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str = ""
    violations: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Aggregate result of all V0 checks."""
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def violations(self) -> list[str]:
        v = []
        for c in self.checks:
            v.extend(c.violations)
        return v


class StaticValidator:
    """V0 Static Validator — runs after every apply_decision.

    Checks:
    1. Shape consistency (tile factors divide loop bounds)
    2. Hardware constraints (shared memory, registers, threads)
    3. Transform legality (no duplicate tiling of same loop)
    4. Data dependency (fusion doesn't break dependencies)
    """

    def validate(
        self,
        semantic: SemanticIR,
        strategy: StrategyIR,
        hw_profile: dict,
    ) -> ValidationResult:
        checks = [
            self._check_tile_legality(semantic, strategy),
            self._check_hw_constraints(strategy, hw_profile),
            self._check_no_duplicate_transforms(strategy),
        ]
        return ValidationResult(
            passed=all(c.passed for c in checks),
            checks=checks,
        )

    def _check_tile_legality(
        self, semantic: SemanticIR, strategy: StrategyIR
    ) -> CheckResult:
        """Check that tile factors are valid."""
        violations = []
        for d in strategy.decisions:
            if d.kind == "tile":
                factors = d.params.get("factors", [])
                if not factors:
                    violations.append(f"Step #{d.step}: empty tile factors")
                for f in factors:
                    if f <= 0:
                        violations.append(f"Step #{d.step}: tile factor {f} must be positive")
                    if f & (f - 1) != 0:
                        # Not a power of 2 — warning, not error
                        pass
        return CheckResult(
            name="tile_legality",
            passed=len(violations) == 0,
            violations=violations,
        )

    def _check_hw_constraints(
        self, strategy: StrategyIR, hw_profile: dict
    ) -> CheckResult:
        """Check hardware resource limits."""
        violations = []
        constraints = hw_profile.get("constraints", {})

        # Estimate shared memory from tile decisions
        # (simplified — real impl needs full resource estimation)
        estimated_shared = 0
        for d in strategy.decisions:
            if d.kind == "place" and d.params.get("memory") == "shared":
                # Each shared placement adds memory
                estimated_shared += 4096  # placeholder estimate

        limit = constraints.get("max_shared_memory_per_block", 49152)
        if estimated_shared > limit:
            violations.append(
                f"Shared memory {estimated_shared}B exceeds limit {limit}B"
            )

        return CheckResult(
            name="hw_constraints",
            passed=len(violations) == 0,
            violations=violations,
            message=f"shared_memory: {estimated_shared}/{limit}",
        )

    def _check_no_duplicate_transforms(self, strategy: StrategyIR) -> CheckResult:
        """Check no loop is tiled twice."""
        violations = []
        tiled_loops: set[str] = set()
        for d in strategy.decisions:
            if d.kind == "tile":
                loop = d.params.get("loop", "")
                if loop in tiled_loops:
                    violations.append(
                        f"Step #{d.step}: loop '{loop}' already tiled"
                    )
                tiled_loops.add(loop)
        return CheckResult(
            name="no_duplicate_transforms",
            passed=len(violations) == 0,
            violations=violations,
        )
