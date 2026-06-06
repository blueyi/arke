#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the frozen Façade v1.0 schema snapshot.

Usage:
    python scripts/regen_facade_v1_schema.py [--check]

Without ``--check`` the script overwrites ``arke/agent/facade_v1_schema.json``.

With ``--check`` it exits non-zero if the on-disk snapshot differs from
the freshly-generated one — used by CI to flag accidental Façade drift.

Deterministic ordering guarantees
---------------------------------
* Tools are emitted in the canonical ``FACADE_V1_TOOLS`` order.
* Each tool entry has fixed key order: name, description, meta, parameters_schema.
* Each ``meta`` block has fixed key order matching ``ToolMeta.to_dict()``.
* JSON is written with ``indent=2``, ``ensure_ascii=False`` and a trailing newline.

Design ref: docs/architecture/arke-harness.md §6.1
Stage tracker: docs/phase1/stage8-plan.md D8-F1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# Ensure project root is importable when invoked as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arke.agent.env import ArkeEnv  # noqa: E402
from arke.agent.facade import (  # noqa: E402
    FACADE_CONTRACT_ID,
    FACADE_LOCKED_ON,
    FACADE_V1_SCHEMA_PATH,
    FACADE_V1_TOOLS,
    FACADE_VERSION,
)
from arke.agent.tools import ToolRegistry  # noqa: E402


def build_snapshot() -> str:
    """Build the deterministic JSON snapshot string."""
    env = ArkeEnv.from_op("matmul")
    reg = ToolRegistry.with_env(env)

    got = set(reg.names())
    expected = set(FACADE_V1_TOOLS)
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise SystemExit(
            f"Registry mismatch with FACADE_V1_TOOLS:\n"
            f"  missing: {sorted(missing)}\n"
            f"  extra:   {sorted(extra)}\n"
            "Fix arke/agent/tools.py or arke/agent/facade.py before re-locking."
        )

    tools_block = []
    for name in FACADE_V1_TOOLS:
        t = reg.get(name)
        tools_block.append(OrderedDict([
            ("name", name),
            ("description", t.description),
            ("meta", t.meta.to_dict()),
            ("parameters_schema", t.parameters_schema()),
        ]))

    doc = OrderedDict([
        ("facade_version", FACADE_VERSION),
        ("contract_id", FACADE_CONTRACT_ID),
        ("design_ref", "docs/architecture/arke-harness.md §6.1"),
        ("locked_on", FACADE_LOCKED_ON),
        ("tool_count", 8),
        ("tools", tools_block),
    ])
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if on-disk snapshot differs (CI mode)",
    )
    args = p.parse_args()

    fresh = build_snapshot()

    if args.check:
        if not FACADE_V1_SCHEMA_PATH.exists():
            print(f"ERROR: {FACADE_V1_SCHEMA_PATH} does not exist", file=sys.stderr)
            return 2
        on_disk = FACADE_V1_SCHEMA_PATH.read_text(encoding="utf-8")
        if on_disk != fresh:
            print(
                f"ERROR: Façade v1.0 snapshot drift detected.\n"
                f"  File: {FACADE_V1_SCHEMA_PATH}\n"
                f"  Either an intended Façade change happened (re-run without --check\n"
                f"  and review the diff carefully), or the Façade was modified by\n"
                f"  accident (revert the offending source change).",
                file=sys.stderr,
            )
            return 1
        print("OK: Façade v1.0 snapshot in sync.")
        return 0

    FACADE_V1_SCHEMA_PATH.write_text(fresh, encoding="utf-8")
    print(f"wrote {FACADE_V1_SCHEMA_PATH} ({len(fresh)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
