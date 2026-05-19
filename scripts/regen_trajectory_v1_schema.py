#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the frozen trajectory v1.0 record-level schema snapshot.

Usage:
    python scripts/regen_trajectory_v1_schema.py [--check]

Without ``--check`` the script overwrites
``arke/learn/trajectory_v1_schema.json``.

With ``--check`` it exits non-zero if the on-disk snapshot differs from
the freshly-generated one — used by CI to flag accidental trajectory
record contract drift.

Deterministic ordering guarantees
---------------------------------
* Record kinds are emitted in the canonical ``RECORD_KINDS_V1`` order
  (``header`` first, then the 9 stream kinds, then ``adjust``).
* Each kind entry has fixed key order: kind, description, payload_fields.
* Each payload field has fixed key order: name, type, required, description.
* JSON is written with ``indent=2``, ``ensure_ascii=False`` and a trailing newline.

Design ref: docs/architecture/arke-harness.md §15
Stage tracker: docs/phase1/stage8-plan.md D8-F3
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

from arke.learn.trajectory_schema import (  # noqa: E402
    LEGACY_SCHEMA,
    RECORD_KINDS_V1,
    RECORD_PAYLOADS_V1,
    TRAJECTORY_CONTRACT_ID,
    TRAJECTORY_LOCKED_ON,
    TRAJECTORY_V1_SCHEMA_PATH,
    TRAJECTORY_VERSION,
)

#: One-line per-kind descriptions baked into the snapshot. Stream-kind
#: descriptions intentionally mirror the D8-F2 events snapshot so the
#: two contracts agree on terminology. The 2 record-only kinds get
#: their own definitions here.
KIND_DESCRIPTIONS: dict[str, str] = {
    "header":     "Session metadata + initial SemanticIR snapshot (first line, exactly once)",
    "decision":   "apply_decision applied a legal mutation to StrategyIR",
    "compile":    "compile_and_profile produced (or failed to produce) a build artifact",
    "profile":    "V2 GPU microbench measurement vs baseline",
    "verify":     "V0 static or V1 numeric correctness check against reference",
    "checkpoint": "Labelled snapshot of OptimizationState taken (recoverable via rollback)",
    "rollback":   "Restored a previous checkpoint; state/budget rewound, audit trail preserved",
    "compact":    "Message log compacted; OptimizationState (ground truth) untouched",
    "fallback":   "Strategy / provider / tier fallback engaged (see §16)",
    "done":       "Loop terminated; final result, totals, and termination reason attached",
    "adjust":     "End-of-cycle StrategyIR refinement marker (record-only, between cycles)",
}
assert set(KIND_DESCRIPTIONS.keys()) == set(RECORD_KINDS_V1), (
    "KIND_DESCRIPTIONS must cover every RECORD_KINDS_V1 entry exactly"
)


def build_snapshot() -> str:
    """Build the deterministic JSON snapshot string."""
    records_block = []
    for kind in RECORD_KINDS_V1:
        fields_block = [
            OrderedDict([
                ("name", f.name),
                ("type", f.type),
                ("required", f.required),
                ("description", f.description),
            ])
            for f in RECORD_PAYLOADS_V1[kind]
        ]
        records_block.append(OrderedDict([
            ("kind", kind),
            ("description", KIND_DESCRIPTIONS[kind]),
            ("payload_fields", fields_block),
        ]))

    doc = OrderedDict([
        ("trajectory_version", TRAJECTORY_VERSION),
        ("contract_id", TRAJECTORY_CONTRACT_ID),
        ("legacy_schema", LEGACY_SCHEMA),
        ("design_ref", "docs/architecture/arke-harness.md §15"),
        ("locked_on", TRAJECTORY_LOCKED_ON),
        ("kind_count", len(RECORD_KINDS_V1)),
        ("layering", OrderedDict([
            ("stream_contract", "arke-harness-events-v1.0.0 (D8-F2)"),
            ("relationship", "strict superset: record-level adds 'header' and 'adjust'"),
            ("shared_envelope", '{"t": <float>, "kind": <string>, "data": <object>}'),
        ])),
        ("wire_format", OrderedDict([
            ("envelope", '{"t": <float>, "kind": <string>, "data": <object>}'),
            ("t_unit", "seconds since session start (monotonic)"),
            ("kind_enum", list(RECORD_KINDS_V1)),
            ("first_line_kind", "header"),
            ("first_line_count", "exactly 1"),
        ])),
        ("records", records_block),
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
        if not TRAJECTORY_V1_SCHEMA_PATH.exists():
            print(f"ERROR: {TRAJECTORY_V1_SCHEMA_PATH} does not exist", file=sys.stderr)
            return 2
        on_disk = TRAJECTORY_V1_SCHEMA_PATH.read_text(encoding="utf-8")
        if on_disk != fresh:
            print(
                f"ERROR: Trajectory v1.0 snapshot drift detected.\n"
                f"  File: {TRAJECTORY_V1_SCHEMA_PATH}\n"
                f"  Either an intended trajectory contract change happened\n"
                f"  (re-run without --check and review the diff carefully),\n"
                f"  or the contract was modified by accident (revert the\n"
                f"  offending source change).",
                file=sys.stderr,
            )
            return 1
        print("OK: Trajectory v1.0 snapshot in sync.")
        return 0

    TRAJECTORY_V1_SCHEMA_PATH.write_text(fresh, encoding="utf-8")
    print(f"wrote {TRAJECTORY_V1_SCHEMA_PATH} ({len(fresh)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
