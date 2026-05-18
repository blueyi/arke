# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Façade v1.0 Contract Test — D8-F1 Lock.

This test is the **immutability gate** for the Arke Harness public Façade
contract per ``docs/architecture/arke-harness.md`` §6.1 + §3.0.2.

It enforces three layers of immutability:

1. **Tool set immutability** — exactly 8 tools, named per FACADE_V1_TOOLS.
2. **ToolMeta immutability** — every tool's declarative flags
   (concurrent_safe, idempotent, requires_compile, mutates_strategy,
   budget_type, cost) match the frozen snapshot byte-for-byte.
3. **Schema immutability** — every tool's parameters_schema and
   description match the frozen snapshot byte-for-byte.

Versioning policy
-----------------
* MINOR/PATCH bumps (1.y.z) MAY add new tools, event kinds, hook points.
  Existing entries MUST NOT change. The frozen snapshot is the canonical
  source of "what existed at 1.0.0".
* MAJOR bumps (2.0.0+) require Leon-approved Gate/Stage doc updates +
  a new ``facade_v2_schema.json`` snapshot + a new contract test file.

If this test fails
------------------
Either:
  (a) Façade was modified intentionally — re-run
      ``python scripts/regen_facade_v1_schema.py``, review diff carefully,
      ensure version bump is correct, update this test if needed.
  (b) Façade was modified accidentally — revert the offending change.

Stage tracker: docs/phase1/stage8-plan.md D8-F1
"""

from __future__ import annotations

import json

import pytest

from arke.agent.env import ArkeEnv
from arke.agent.facade import (
    FACADE_CONTRACT_ID,
    FACADE_LOCKED_ON,
    FACADE_V1_SCHEMA_PATH,
    FACADE_V1_TOOLS,
    FACADE_VERSION,
    load_facade_v1_schema,
)
from arke.agent.tools import ArkeTool, ToolRegistry


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env() -> ArkeEnv:
    """Stable env for tool wiring (matmul is the canonical reference op)."""
    return ArkeEnv.from_op("matmul")


@pytest.fixture(scope="module")
def reg(env: ArkeEnv) -> ToolRegistry:
    return ToolRegistry.with_env(env)


@pytest.fixture(scope="module")
def frozen() -> dict:
    return load_facade_v1_schema()


# ── 1. Version constants ─────────────────────────────────────────


class TestFacadeVersion:

    def test_version_constant(self):
        assert FACADE_VERSION == "1.0.0"

    def test_contract_id(self):
        assert FACADE_CONTRACT_ID == "arke-harness-facade-v1.0.0"

    def test_locked_date(self):
        assert FACADE_LOCKED_ON == "2026-05-18"

    def test_frozen_snapshot_exists(self):
        assert FACADE_V1_SCHEMA_PATH.exists(), (
            f"Frozen schema missing at {FACADE_V1_SCHEMA_PATH}; "
            "run scripts/regen_facade_v1_schema.py"
        )

    def test_frozen_snapshot_metadata(self, frozen: dict):
        assert frozen["facade_version"] == FACADE_VERSION
        assert frozen["contract_id"] == FACADE_CONTRACT_ID
        assert frozen["locked_on"] == FACADE_LOCKED_ON
        assert frozen["tool_count"] == 8
        assert frozen["design_ref"] == "docs/architecture/arke-harness.md §6.1"


# ── 2. Tool set immutability ─────────────────────────────────────


class TestToolSet:

    def test_facade_v1_tools_constant(self):
        assert len(FACADE_V1_TOOLS) == 8
        assert FACADE_V1_TOOLS == (
            "get_hw_profile",
            "analyze_compute",
            "list_legal_actions",
            "apply_decision",
            "verify_correctness",
            "compile_and_profile",
            "checkpoint",
            "rollback",
        )

    def test_registry_has_exactly_8(self, reg: ToolRegistry):
        names = set(reg.names())
        assert names == set(FACADE_V1_TOOLS), (
            f"missing: {set(FACADE_V1_TOOLS) - names}, "
            f"extra:   {names - set(FACADE_V1_TOOLS)}"
        )

    def test_no_unexpected_tools(self, reg: ToolRegistry):
        """No tools beyond the locked 8 — guards against accidental registration."""
        forbidden = {"benchmark_advice_summary"}
        names = set(reg.names())
        assert names.isdisjoint(forbidden), (
            f"Forbidden tools registered in Façade: {names & forbidden}. "
            "These are Phase-1 internal helpers, not part of the locked contract."
        )

    def test_frozen_tool_order(self, frozen: dict):
        """Frozen snapshot preserves canonical ordering."""
        names_in_order = [t["name"] for t in frozen["tools"]]
        assert tuple(names_in_order) == FACADE_V1_TOOLS

    def test_each_tool_implements_abc(self, reg: ToolRegistry):
        for name in FACADE_V1_TOOLS:
            tool = reg.get(name)
            assert isinstance(tool, ArkeTool), f"{name} not an ArkeTool"


# ── 3. ToolMeta immutability ─────────────────────────────────────


class TestToolMetaImmutability:
    """Every tool's ToolMeta flags must match the frozen snapshot."""

    @pytest.mark.parametrize("tool_name", FACADE_V1_TOOLS)
    def test_meta_matches_frozen(
        self, tool_name: str, reg: ToolRegistry, frozen: dict
    ):
        live_meta = reg.get(tool_name).meta.to_dict()
        frozen_entry = next(t for t in frozen["tools"] if t["name"] == tool_name)
        frozen_meta = frozen_entry["meta"]
        assert live_meta == frozen_meta, (
            f"{tool_name} ToolMeta drift:\n"
            f"  live:   {live_meta}\n"
            f"  frozen: {frozen_meta}\n"
            "Either re-run scripts/regen_facade_v1_schema.py (intentional) "
            "or revert the source change (accidental)."
        )


# ── 4. parameters_schema + description immutability ──────────────


class TestSchemaImmutability:

    @pytest.mark.parametrize("tool_name", FACADE_V1_TOOLS)
    def test_description_matches_frozen(
        self, tool_name: str, reg: ToolRegistry, frozen: dict
    ):
        live = reg.get(tool_name).description
        frozen_desc = next(
            t["description"] for t in frozen["tools"] if t["name"] == tool_name
        )
        assert live == frozen_desc, (
            f"{tool_name} description drift:\n"
            f"  live:   {live!r}\n"
            f"  frozen: {frozen_desc!r}"
        )

    @pytest.mark.parametrize("tool_name", FACADE_V1_TOOLS)
    def test_parameters_schema_matches_frozen(
        self, tool_name: str, reg: ToolRegistry, frozen: dict
    ):
        live = reg.get(tool_name).parameters_schema()
        frozen_schema = next(
            t["parameters_schema"] for t in frozen["tools"] if t["name"] == tool_name
        )
        assert live == frozen_schema, (
            f"{tool_name} parameters_schema drift.\n"
            f"  live:   {json.dumps(live, indent=2)}\n"
            f"  frozen: {json.dumps(frozen_schema, indent=2)}"
        )


# ── 5. Byte-for-byte snapshot regeneration determinism ───────────


class TestSnapshotDeterminism:
    """Ensure the snapshot is byte-stable when regenerated.

    Catches accidental dict-iteration-order changes (Python 3.7+ guarantees
    insertion order, but explicit OrderedDict in the regen script makes
    this robust against future refactors).
    """

    def test_regen_is_byte_stable(self):
        # Import the regen function directly so we don't shell out.
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        scripts = root / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        # Import the regen module fresh
        import importlib
        regen = importlib.import_module("regen_facade_v1_schema")

        fresh = regen.build_snapshot()
        on_disk = FACADE_V1_SCHEMA_PATH.read_text(encoding="utf-8")
        assert fresh == on_disk, (
            "Snapshot drift detected — Façade source changed without "
            "regenerating the frozen schema. "
            "Run: python scripts/regen_facade_v1_schema.py"
        )


# ── 6. Façade ↔ Substrate boundary smoke ─────────────────────────


class TestFacadeSurface:
    """Light surface checks — Façade tools never leak Substrate types as
    opaque blobs in their declared schemas. (per arke-harness.md §3.0.1)"""

    @pytest.mark.parametrize("tool_name", FACADE_V1_TOOLS)
    def test_schema_is_pure_json(
        self, tool_name: str, reg: ToolRegistry
    ):
        schema = reg.get(tool_name).parameters_schema()
        # Round-trip through JSON — fails if schema contains non-JSON types.
        try:
            json.loads(json.dumps(schema))
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{tool_name} schema not JSON-serializable: {exc}")

    @pytest.mark.parametrize("tool_name", FACADE_V1_TOOLS)
    def test_function_calling_schema_is_valid(
        self, tool_name: str, reg: ToolRegistry
    ):
        fn = reg.get(tool_name).to_function_schema()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == tool_name
        assert "parameters" in fn["function"]
        assert "description" in fn["function"]
