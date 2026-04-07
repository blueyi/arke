#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check backend-agnostic strategy: no Triton-specific fields in StrategyIR L1.

Gate criterion G6-LI.8: StrategyIR L1 decisions must be backend-agnostic.
L2 decisions (level=2) are allowed to have backend-specific content.

Checks:
- No Triton-specific field names in L1 decision params:
    num_warps, num_stages, BLOCK_SIZE_*, tl.load, tl.store, etc.
- No Triton-specific strings in L1 decision param values
- L1 decision kinds are from the allowed set
"""

from __future__ import annotations

import sys
from pathlib import Path

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"

# Triton-specific field names that MUST NOT appear in L1 decisions
TRITON_SPECIFIC_FIELDS = {
    "num_warps",
    "num_stages",
    "num_ctas",
    "block_size",
    "BLOCK_SIZE",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "grid",
}

# Triton-specific strings that MUST NOT appear in L1 param values
TRITON_SPECIFIC_STRINGS = [
    "tl.load",
    "tl.store",
    "tl.dot",
    "tl.program_id",
    "triton.jit",
    "@triton.jit",
    "triton.language",
    "num_warps",
    "num_stages",
]

# Allowed L1 decision kinds
L1_ALLOWED_KINDS = {
    "tile",
    "reorder",
    "fuse",
    "parallel",
    "place",
    "vectorize",
    "unroll",
    "algorithm",
}


def check_file(ak_path: Path) -> list[str]:
    """Check a single .ak file for backend-agnostic compliance.

    Returns list of violation messages (empty = pass).
    """
    from arke.ir.converters import ast_to_strategy
    from arke.ir.strategy import ConditionalDecision, Decision
    from arke.lang.grammar import parse_file

    violations: list[str] = []

    try:
        program = parse_file(str(ak_path))
    except Exception as e:
        violations.append(f"Parse error: {e}")
        return violations

    if not program.strategies:
        return violations  # No strategy = no violations

    for strat_def in program.strategies:
        ir = ast_to_strategy(strat_def)

        for d in ir.decisions:
            if isinstance(d, ConditionalDecision):
                # Check decisions inside conditional arms
                all_inner = d.true_decisions + d.false_decisions
                for inner in all_inner:
                    violations.extend(_check_decision(inner, ak_path.name))
            elif isinstance(d, Decision):
                violations.extend(_check_decision(d, ak_path.name))

    return violations


def _check_decision(d, filename: str) -> list[str]:
    """Check a single Decision for L1 backend-agnostic compliance."""
    from arke.ir.strategy import Decision

    violations: list[str] = []

    # Only check L1 decisions
    if d.level != 1:
        return violations

    # Check kind is in allowed set
    if d.kind not in L1_ALLOWED_KINDS:
        violations.append(
            f"{filename}: L1 decision has non-standard kind '{d.kind}'"
        )

    # Check param keys for Triton-specific names
    for key in d.params:
        if key in TRITON_SPECIFIC_FIELDS:
            violations.append(
                f"{filename}: L1 decision '{d.kind}' has Triton-specific "
                f"param key '{key}'"
            )
        # Also check BLOCK_SIZE_* pattern
        if key.startswith("BLOCK_SIZE"):
            violations.append(
                f"{filename}: L1 decision '{d.kind}' has Triton-specific "
                f"param key '{key}'"
            )

    # Check param values for Triton-specific strings
    for key, val in d.params.items():
        val_str = str(val)
        for ts in TRITON_SPECIFIC_STRINGS:
            if ts in val_str:
                violations.append(
                    f"{filename}: L1 decision '{d.kind}' param '{key}' "
                    f"contains Triton-specific string '{ts}'"
                )

    return violations


def main() -> int:
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    ak_files = sorted(OPERATORS_DIR.glob("*.ak"))
    if not ak_files:
        print("ERROR: No .ak files found in", OPERATORS_DIR)
        return 1

    print(f"Checking {len(ak_files)} .ak files for backend-agnostic compliance...\n")

    all_violations: list[str] = []
    for ak_file in ak_files:
        violations = check_file(ak_file)
        if violations:
            for v in violations:
                print(f"  VIOLATION: {v}")
            all_violations.extend(violations)
        else:
            print(f"  OK: {ak_file.name}")

    print()
    if all_violations:
        print(f"FAIL: {len(all_violations)} violations found")
        return 1
    else:
        print(f"PASS: All {len(ak_files)} .ak files are backend-agnostic at L1")
        return 0


if __name__ == "__main__":
    sys.exit(main())
