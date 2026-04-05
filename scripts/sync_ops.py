#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Sync operator catalog from benchmark-ops.md into code and test files.

Run manually:
    python scripts/sync_ops.py

Or automatically via tests/conftest.py on every pytest run when
benchmark-ops.md has changed.

What it validates/updates
--------------------------
- benchmarks/cli.py      : imports OT_OPS from op_registry (live, no sync needed)
- benchmarks/shapes.py   : imports OP_TIER from op_registry (live, no sync needed)
- tests/test_benchmark_protocol.py : imports ALL_OPS from op_registry (live, no sync needed)

Since all three files now import from op_registry at runtime, this script's
primary job is:
  1. Validate that the live op_registry parses cleanly
  2. Verify shapes.py routes every operator from the catalog
  3. Report any new ops that lack a shape set (warning, not error)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmarks.op_registry import ALL_OPS, OT_OPS, TOTAL_OPS, parse_ops_md  # noqa: E402


def validate_registry() -> list[str]:
    """Re-parse benchmark-ops.md and verify it matches the loaded module state."""
    fresh = parse_ops_md()
    errors = []
    for tier in sorted(OT_OPS):
        if set(OT_OPS[tier]) != set(fresh[tier]):
            errors.append(
                f"OT{tier}: module cache {set(OT_OPS[tier])} != fresh parse {set(fresh[tier])}"
            )
    return errors


def check_shapes_coverage() -> tuple[list[str], list[str]]:
    """Check which ops have shape coverage and which don't."""
    from benchmarks.shapes import get_shapes

    covered, missing = [], []
    for op in ALL_OPS:
        try:
            shapes = get_shapes(op)
            if shapes:
                covered.append(op)
            else:
                missing.append(f"{op} (empty)")
        except (ValueError, KeyError):
            missing.append(op)
    return covered, missing


def main() -> None:
    print(
        f"[sync_ops] benchmark-ops.md → {TOTAL_OPS} ops "
        f"({', '.join(f'OT{t}:{len(v)}' for t, v in sorted(OT_OPS.items()))})"
    )

    # 1. Validate registry self-consistency
    errs = validate_registry()
    if errs:
        print("[sync_ops] ERROR: registry inconsistency:")
        for e in errs:
            print(f"  {e}")
        sys.exit(1)
    print("[sync_ops] registry OK — op_registry matches benchmark-ops.md")

    # 2. Check shapes coverage
    covered, missing = check_shapes_coverage()
    print(f"[sync_ops] shapes coverage: {len(covered)}/{TOTAL_OPS} ops have shapes")
    if missing:
        print(f"[sync_ops] WARNING: {len(missing)} ops have no shape set (will use fallback):")
        for op in missing:
            print(f"  - {op}")

    print("[sync_ops] done.")


if __name__ == "__main__":
    main()
