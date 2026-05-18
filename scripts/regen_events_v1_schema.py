#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the frozen OptimizationEvent v1.0 schema snapshot.

Usage:
    python scripts/regen_events_v1_schema.py [--check]

Without ``--check`` the script overwrites ``arke/agent/events_v1_schema.json``.

With ``--check`` it exits non-zero if the on-disk snapshot differs from
the freshly-generated one — used by CI to flag accidental event-stream drift.

Deterministic ordering guarantees
---------------------------------
* Event kinds are emitted in the canonical ``EVENT_KINDS_V1`` order.
* Each event entry has fixed key order: kind, description, payload_fields.
* Each payload field has fixed key order: name, type, required, description.
* JSON is written with ``indent=2``, ``ensure_ascii=False`` and a trailing newline.

Design ref: docs/architecture/arke-harness.md §4 + §15
Stage tracker: docs/phase1/stage8-plan.md D8-F2
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

from arke.agent.events import (  # noqa: E402
    EVENT_KINDS_V1,
    EVENT_PAYLOADS_V1,
    EVENTS_CONTRACT_ID,
    EVENTS_LOCKED_ON,
    EVENTS_V1_SCHEMA_PATH,
    EVENTS_VERSION,
)

#: One-line per-kind descriptions baked into the snapshot for downstream
#: consumers (CLI/REST/MCP/Jupyter) so they don't need to crack the
#: Python module to render help text. Kept terse — full prose lives in
#: docs/architecture/arke-harness.md §4 + §15.
KIND_DESCRIPTIONS: dict[str, str] = {
    "decision":   "apply_decision applied a legal mutation to StrategyIR",
    "compile":    "compile_and_profile produced (or failed to produce) a build artifact",
    "profile":    "V2 GPU microbench measurement vs baseline",
    "verify":     "V0 static or V1 numeric correctness check against reference",
    "checkpoint": "Labelled snapshot of OptimizationState taken (recoverable via rollback)",
    "rollback":   "Restored a previous checkpoint; state/budget rewound, audit trail preserved",
    "compact":    "Message log compacted; OptimizationState (ground truth) untouched",
    "fallback":   "Strategy / provider / tier fallback engaged (see §16)",
    "done":       "Loop terminated; final result, totals, and termination reason attached",
}
assert set(KIND_DESCRIPTIONS.keys()) == set(EVENT_KINDS_V1), (
    "KIND_DESCRIPTIONS must cover every EVENT_KINDS_V1 entry exactly"
)


def build_snapshot() -> str:
    """Build the deterministic JSON snapshot string."""
    events_block = []
    for kind in EVENT_KINDS_V1:
        fields_block = [
            OrderedDict([
                ("name", f.name),
                ("type", f.type),
                ("required", f.required),
                ("description", f.description),
            ])
            for f in EVENT_PAYLOADS_V1[kind]
        ]
        events_block.append(OrderedDict([
            ("kind", kind),
            ("description", KIND_DESCRIPTIONS[kind]),
            ("payload_fields", fields_block),
        ]))

    doc = OrderedDict([
        ("events_version", EVENTS_VERSION),
        ("contract_id", EVENTS_CONTRACT_ID),
        ("design_ref", "docs/architecture/arke-harness.md §4 + §15"),
        ("locked_on", EVENTS_LOCKED_ON),
        ("kind_count", len(EVENT_KINDS_V1)),
        ("wire_format", OrderedDict([
            ("envelope", '{"t": <float>, "kind": <string>, "data": <object>}'),
            ("t_unit", "seconds since session start (monotonic)"),
            ("kind_enum", list(EVENT_KINDS_V1)),
        ])),
        ("events", events_block),
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
        if not EVENTS_V1_SCHEMA_PATH.exists():
            print(f"ERROR: {EVENTS_V1_SCHEMA_PATH} does not exist", file=sys.stderr)
            return 2
        on_disk = EVENTS_V1_SCHEMA_PATH.read_text(encoding="utf-8")
        if on_disk != fresh:
            print(
                f"ERROR: OptimizationEvent v1.0 snapshot drift detected.\n"
                f"  File: {EVENTS_V1_SCHEMA_PATH}\n"
                f"  Either an intended event-stream change happened (re-run without\n"
                f"  --check and review the diff carefully), or the contract was\n"
                f"  modified by accident (revert the offending source change).",
                file=sys.stderr,
            )
            return 1
        print("OK: OptimizationEvent v1.0 snapshot in sync.")
        return 0

    EVENTS_V1_SCHEMA_PATH.write_text(fresh, encoding="utf-8")
    print(f"wrote {EVENTS_V1_SCHEMA_PATH} ({len(fresh)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
