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
    current = _current_snapshot()
    saved = _load_snapshot()

    if current == saved:
        return  # nothing changed, proceed silently

    # ── Catalog changed (or first run) ──────────────────────────────────
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


def _diff_snapshots(old: dict, new: dict) -> tuple[list, list, list]:
    """Return (added, removed, moved) operator lists."""
    old_map = {op: int(t) for t, ops in old.items() for op in ops}
    new_map = {op: int(t) for t, ops in new.items() for op in ops}

    added = [op for op in new_map if op not in old_map]
    removed = [op for op in old_map if op not in new_map]
    moved = [
        f"{op}(OT{old_map[op]}→OT{new_map[op]})"
        for op in old_map
        if op in new_map and old_map[op] != new_map[op]
    ]
    return added, removed, moved
