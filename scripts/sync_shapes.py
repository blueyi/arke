#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Sync shape catalog from benchmark-shapes.md into code and snapshot file.

Run manually:
    python scripts/sync_shapes.py

Or automatically via tests/conftest.py on every pytest run when
benchmark-shapes.md has changed.

What it validates / updates
----------------------------
1. Validates that shape_registry parses cleanly and is self-consistent
   (re-parse vs module state).
2. Checks that every op in the benchmark catalog has shape coverage
   (warns on gaps, does not error).
3. Reports added / removed / modified shapes vs the previous snapshot.
4. Updates ``.benchmark_shapes_snapshot.json`` with current state.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmarks.shape_registry import (  # noqa: E402
    SHAPE_TABLES,
    TOTAL_SHAPES,
    ALL_SHAPE_TAGS,
    parse_shapes_md,
)

_SNAPSHOT_PATH = _REPO / ".benchmark_shapes_snapshot.json"


# ── Snapshot helpers ──────────────────────────────────────────────────────

def _compute_tags_hash(tags: list[str]) -> str:
    """SHA-256 of the sorted tag list (order-independent change detection)."""
    joined = "\n".join(sorted(tags))
    return hashlib.sha256(joined.encode()).hexdigest()


def build_snapshot(tables: dict, tags: list[str], total: int) -> dict:
    return {
        "total_shapes": total,
        "tables": {k: len(v) for k, v in sorted(tables.items())},
        "tags_hash": _compute_tags_hash(tags),
    }


def load_snapshot() -> dict | None:
    if _SNAPSHOT_PATH.exists():
        return json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return None


def save_snapshot(snap: dict) -> None:
    _SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")


# ── Diff helpers ─────────────────────────────────────────────────────────

def diff_snapshots(old: dict, new: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (added_tables, removed_tables, changed_tables) by row count."""
    old_tables: dict = old.get("tables", {})
    new_tables: dict = new.get("tables", {})

    added = [k for k in new_tables if k not in old_tables]
    removed = [k for k in old_tables if k not in new_tables]
    changed = [
        f"{k}({old_tables[k]}→{new_tables[k]})"
        for k in old_tables
        if k in new_tables and old_tables[k] != new_tables[k]
    ]
    return added, removed, changed


# ── Validation ────────────────────────────────────────────────────────────

def validate_registry() -> list[str]:
    """Re-parse benchmark-shapes.md and verify it matches the loaded module."""
    fresh = parse_shapes_md()
    errors = []
    for table_name, rows in SHAPE_TABLES.items():
        if table_name not in fresh:
            errors.append(f"Table '{table_name}' in module but not in fresh parse")
            continue
        fresh_rows = fresh[table_name]
        if len(rows) != len(fresh_rows):
            errors.append(
                f"Table '{table_name}': module has {len(rows)} rows, "
                f"fresh parse has {len(fresh_rows)} rows"
            )
    for table_name in fresh:
        if table_name not in SHAPE_TABLES:
            errors.append(f"Table '{table_name}' in fresh parse but not in module")
    return errors


def check_shapes_coverage() -> tuple[list[str], list[str]]:
    """Check which ops have shape coverage and which don't."""
    from benchmarks.shapes import get_shapes
    from benchmarks.op_registry import ALL_OPS

    covered, missing = [], []
    for op in ALL_OPS:
        try:
            shapes = get_shapes(op)
            if shapes:
                covered.append(op)
            else:
                missing.append(f"{op} (empty)")
        except (ValueError, KeyError) as exc:
            missing.append(f"{op} ({exc})")
    return covered, missing


def check_registry_coverage() -> tuple[list[str], list[str]]:
    """Check which ops have registry (markdown-sourced) shapes vs fallback."""
    from benchmarks.shape_registry import get_registry_shapes_for_op
    from benchmarks.op_registry import ALL_OPS

    registry_covered, fallback_only = [], []
    for op in ALL_OPS:
        rows = get_registry_shapes_for_op(op)
        if rows:
            registry_covered.append(op)
        else:
            fallback_only.append(op)
    return registry_covered, fallback_only


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print(
        f"[sync_shapes] benchmark-shapes.md → {TOTAL_SHAPES} shapes "
        f"across {len(SHAPE_TABLES)} tables"
    )

    # 1. Validate registry self-consistency
    errs = validate_registry()
    if errs:
        print("[sync_shapes] ERROR: registry inconsistency:")
        for e in errs:
            print(f"  {e}")
        sys.exit(1)
    print("[sync_shapes] registry OK — shape_registry matches benchmark-shapes.md")

    # 2. Check shapes coverage (all ops)
    covered, missing = check_shapes_coverage()
    print(f"[sync_shapes] shapes coverage: {len(covered)}/{len(covered)+len(missing)} ops have shapes")
    if missing:
        print(f"[sync_shapes] WARNING: {len(missing)} ops have no shape set (will use fallback):")
        for op in missing:
            print(f"  - {op}")

    # 3. Check which ops have registry (markdown) vs fallback shapes
    reg_covered, fallback_only = check_registry_coverage()
    print(f"[sync_shapes] registry coverage: {len(reg_covered)} ops served from markdown, "
          f"{len(fallback_only)} ops use hardcoded fallback")
    if fallback_only:
        print(f"[sync_shapes] ops using hardcoded fallback: {fallback_only}")

    # 4. Compare with snapshot and report diff
    current_snap = build_snapshot(SHAPE_TABLES, ALL_SHAPE_TAGS, TOTAL_SHAPES)
    saved_snap = load_snapshot()

    if saved_snap is None:
        print("[sync_shapes] No previous snapshot found — creating initial snapshot.")
    elif current_snap["tags_hash"] == saved_snap.get("tags_hash"):
        print("[sync_shapes] No shape changes detected (tags_hash matches).")
    else:
        added, removed, changed = diff_snapshots(saved_snap, current_snap)
        old_total = saved_snap.get("total_shapes", "?")
        print(
            f"[sync_shapes] Shape catalog changed: {old_total} → {TOTAL_SHAPES} shapes\n"
            f"  Added tables  : {added or 'none'}\n"
            f"  Removed tables: {removed or 'none'}\n"
            f"  Changed tables: {changed or 'none'}"
        )

    # 5. Write updated snapshot
    save_snapshot(current_snap)
    print(f"[sync_shapes] Snapshot written to {_SNAPSHOT_PATH.name}")

    # 6. Summary stats
    print(f"\n[sync_shapes] Shape statistics:")
    print(f"  Total shapes : {TOTAL_SHAPES}")
    print(f"  Total tables : {len(SHAPE_TABLES)}")
    for tname, rows in sorted(SHAPE_TABLES.items()):
        print(f"    {tname:40s}: {len(rows):3d} rows")

    print("\n[sync_shapes] done.")


if __name__ == "__main__":
    main()
