# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Harness Façade v1.0 — locked public contract.

This module defines the version constants and frozen tool list for the
public Façade contract per ``docs/architecture/arke-harness.md`` §6.1.

The Façade is the **vendor-agnostic, agent-runtime-agnostic** surface of
Arke: it exposes exactly 8 tools that any MCP-compatible agent (Claude
Code, OpenClaw, Hermes, Cline, Continue, …) can drive without depending
on any Substrate-internal types.

Versioning policy
-----------------
* ``FACADE_VERSION`` is locked at ``1.0.0`` from 2026-05-18.
* Within the same MAJOR version (``1.y.z``), all changes are
  backward-compatible: new tools / new event kinds / new hook points
  MAY be added; existing tool signatures, names, ``ToolMeta`` flags,
  and parameter schemas MUST NOT change.
* Any breaking change (renamed tool, removed tool, signature change,
  ``ToolMeta`` flag change that breaks orchestrator assumptions, etc.)
  bumps MAJOR.
* The frozen contract snapshot lives in
  ``arke/agent/facade_v1_schema.json`` and is enforced by
  ``tests/test_facade_contract_v1.py``.

Re-locking procedure
--------------------
If a Façade change is necessary, regenerate the frozen snapshot via
``scripts/regen_facade_v1_schema.py`` (deterministic; same ordering
guarantees as initial lock), then update the contract test if the
change is intentional. Any failure of the contract test is a signal
that the Façade was modified — review carefully before re-locking.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Version constants ─────────────────────────────────────────────
FACADE_VERSION: str = "1.0.0"
FACADE_CONTRACT_ID: str = "arke-harness-facade-v1.0.0"
FACADE_LOCKED_ON: str = "2026-05-18"

# ── The 8 locked tools (arke-harness.md §6.1, ordered) ────────────
FACADE_V1_TOOLS: tuple[str, ...] = (
    "get_hw_profile",
    "analyze_compute",
    "list_legal_actions",
    "apply_decision",
    "verify_correctness",
    "compile_and_profile",
    "checkpoint",
    "rollback",
)
assert len(FACADE_V1_TOOLS) == 8, "Façade v1.0 must have exactly 8 tools"

# ── Frozen schema snapshot path ───────────────────────────────────
FACADE_V1_SCHEMA_PATH: Path = Path(__file__).parent / "facade_v1_schema.json"


def load_facade_v1_schema() -> dict:
    """Load the frozen Façade v1.0 schema snapshot.

    Returns the parsed JSON document with keys:
      facade_version, contract_id, design_ref, locked_on, tool_count, tools

    Raises FileNotFoundError if the snapshot is missing — this should
    never happen in a checked-out source tree; it indicates a packaging
    bug.
    """
    if not FACADE_V1_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Façade v1.0 frozen schema missing at {FACADE_V1_SCHEMA_PATH}; "
            "re-run scripts/regen_facade_v1_schema.py"
        )
    return json.loads(FACADE_V1_SCHEMA_PATH.read_text(encoding="utf-8"))


__all__ = [
    "FACADE_VERSION",
    "FACADE_CONTRACT_ID",
    "FACADE_LOCKED_ON",
    "FACADE_V1_TOOLS",
    "FACADE_V1_SCHEMA_PATH",
    "load_facade_v1_schema",
]
