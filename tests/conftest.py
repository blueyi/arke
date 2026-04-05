# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pytest conftest: auto-sync op catalog from benchmark-ops.md before tests.

Every pytest run automatically:
1. Parses benchmark-ops.md via op_registry
2. Compares against the last-known snapshot (.benchmark_ops_snapshot.json)
3. If unchanged  → proceeds immediately (zero overhead)
4. If changed    → runs ``scripts/sync_ops.py`` to update cli.py / shapes.py,
                   writes a new snapshot, and prints a summary

The sync is intentionally non-interactive: it mutates files and lets pytest
continue.  CI will catch any resulting failures if code is out of sync.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOT = _REPO_ROOT / ".benchmark_ops_snapshot.json"
_SYNC_SCRIPT = _REPO_ROOT / "scripts" / "sync_ops.py"


def _current_snapshot() -> dict:
    """Return the current OT catalog as a JSON-serialisable dict."""
    from benchmarks.op_registry import OT_OPS
    return {str(k): v for k, v in OT_OPS.items()}


def _load_snapshot() -> dict | None:
    if _SNAPSHOT.exists():
        return json.loads(_SNAPSHOT.read_text())
    return None


def _save_snapshot(snap: dict) -> None:
    _SNAPSHOT.write_text(json.dumps(snap, indent=2))


def pytest_configure(config):  # noqa: ARG001
    """Called once at the very start of every pytest session."""
    # ── Op catalog change detection ──────────────────────────────────────
    current = _current_snapshot()
    saved = _load_snapshot()

    if current != saved:
        # Catalog changed (or first run)
        if saved is None:
            print("\n[op_registry] First run — creating snapshot and syncing code.")
        else:
            added, removed, moved = _diff_snapshots(saved, current)
            print(
                f"\n[op_registry] benchmark-ops.md changed — auto-syncing code.\n"
                f"  Added  : {added or 'none'}\n"
                f"  Removed: {removed or 'none'}\n"
                f"  Moved  : {moved or 'none'}"
            )

        # Run the sync script
        result = subprocess.run(
            [sys.executable, str(_SYNC_SCRIPT)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.exit(
                f"[op_registry] sync_ops.py failed:\n{result.stdout}\n{result.stderr}",
                returncode=1,
            )

        print(result.stdout.strip() or "[op_registry] sync complete.")
        _save_snapshot(current)

    # ── Shape catalog change detection ───────────────────────────────────
    _check_shapes_snapshot()


def _diff_snapshots(old: dict, new: dict) -> tuple[list, list, list]:
    """Return (added, removed, moved) operator lists."""
    old_map = {op: int(t) for t, ops in old.items() for op in ops}
    new_map = {op: int(t) for t, ops in new.items() for op in ops}

    added = [op for op in new_map if op not in old_map]
    removed = [op for op in old_map if op not in new_map]
    moved = [
        f"{op}(OT{old_map[op]}\u2192OT{new_map[op]})"
        for op in old_map
        if op in new_map and old_map[op] != new_map[op]
    ]
    return added, removed, moved


# ── Shape snapshot helpers ────────────────────────────────────────────

_SHAPES_SNAPSHOT = _REPO_ROOT / ".benchmark_shapes_snapshot.json"
_SYNC_SHAPES_SCRIPT = _REPO_ROOT / "scripts" / "sync_shapes.py"


def _current_shapes_snapshot() -> dict:
    """Return the current shape catalog as a JSON-serialisable snapshot."""
    import hashlib
    from benchmarks.shape_registry import SHAPE_TABLES, TOTAL_SHAPES, ALL_SHAPE_TAGS
    tags_hash = hashlib.sha256("\n".join(sorted(ALL_SHAPE_TAGS)).encode()).hexdigest()
    return {
        "total_shapes": TOTAL_SHAPES,
        "tables": {k: len(v) for k, v in sorted(SHAPE_TABLES.items())},
        "tags_hash": tags_hash,
    }


def _load_shapes_snapshot() -> dict | None:
    if _SHAPES_SNAPSHOT.exists():
        return json.loads(_SHAPES_SNAPSHOT.read_text(encoding="utf-8"))
    return None


def _save_shapes_snapshot(snap: dict) -> None:
    _SHAPES_SNAPSHOT.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")


def _diff_shapes_snapshots(old: dict, new: dict) -> tuple[list, list, list]:
    """Return (added_tables, removed_tables, changed_tables) from shape snapshots."""
    old_tables: dict = old.get("tables", {})
    new_tables: dict = new.get("tables", {})
    added = [k for k in new_tables if k not in old_tables]
    removed = [k for k in old_tables if k not in new_tables]
    changed = [
        f"{k}({old_tables[k]}\u2192{new_tables[k]} rows)"
        for k in old_tables
        if k in new_tables and old_tables[k] != new_tables[k]
    ]
    return added, removed, changed


def _check_shapes_snapshot() -> None:
    """Compare current shape catalog against snapshot; auto-sync if changed."""
    try:
        current = _current_shapes_snapshot()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[shape_registry] WARNING: could not read shape catalog: {exc}")
        return

    saved = _load_shapes_snapshot()

    if saved is not None and current.get("tags_hash") == saved.get("tags_hash"):
        return  # nothing changed

    if saved is None:
        print("\n[shape_registry] First run \u2014 creating shapes snapshot and syncing.")
    else:
        added, removed, changed = _diff_shapes_snapshots(saved, current)
        old_total = saved.get("total_shapes", "?")
        print(
            f"\n[shape_registry] benchmark-shapes.md changed \u2014 auto-syncing.\n"
            f"  Total: {old_total} \u2192 {current['total_shapes']}\n"
            f"  Added tables  : {added or 'none'}\n"
            f"  Removed tables: {removed or 'none'}\n"
            f"  Changed tables: {changed or 'none'}"
        )

    result = subprocess.run(
        [sys.executable, str(_SYNC_SHAPES_SCRIPT)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Non-fatal: warn but don't abort test run
        print(
            f"[shape_registry] WARNING: sync_shapes.py failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        return

    print(result.stdout.strip() or "[shape_registry] sync complete.")
    _save_shapes_snapshot(current)
