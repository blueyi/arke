# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""OptimizationEvent v1.0 Contract Test — D8-F2 Lock.

This test is the **immutability gate** for the Arke Harness public
event-stream contract per ``docs/architecture/arke-harness.md``
§4 (The Loop) + §15 (Trajectory & Learning) + §3.0.2 (Versioning).

It enforces four layers of immutability:

1. **Version constants immutability** — VERSION / CONTRACT_ID /
   LOCKED_ON pinned at 1.0.0 / arke-harness-events-v1.0.0 / 2026-05-18.
2. **Kind set immutability** — exactly 9 kinds, named + ordered per
   ``EVENT_KINDS_V1``; EventKind enum agrees with the tuple.
3. **Payload schema immutability** — every kind's payload field set
   (names + types + required flags + descriptions) matches the
   frozen snapshot byte-for-byte.
4. **Wire format immutability** — envelope shape
   ``{"t": <float>, "kind": <string>, "data": <object>}`` is round-trip
   safe through ``OptimizationEvent.to_dict / from_dict`` and matches
   the golden trajectory fixture.

Versioning policy
-----------------
* MINOR/PATCH bumps (1.y.z) MAY add new kinds OR new OPTIONAL payload
  fields. Existing kinds + required fields MUST NOT change.
* MAJOR bumps (2.0.0+) require Leon-approved doc updates +
  a new ``events_v2_schema.json`` snapshot + a new contract test file.

If this test fails
------------------
Either:
  (a) Event stream was modified intentionally — re-run
      ``python scripts/regen_events_v1_schema.py``, review diff carefully,
      ensure version bump is correct, update this test if needed.
  (b) Event stream was modified accidentally — revert the offending change.

Stage tracker: docs/phase1/stage8-plan.md D8-F2
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arke.agent.events import (
    EVENT_KINDS_V1,
    EVENT_PAYLOADS_V1,
    EVENTS_CONTRACT_ID,
    EVENTS_LOCKED_ON,
    EVENTS_V1_SCHEMA_PATH,
    EVENTS_VERSION,
    EventKind,
    OptimizationEvent,
    PayloadField,
    load_events_v1_schema,
    make_event,
    validate_payload,
)


# ── Constants ────────────────────────────────────────────────────

EXPECTED_KINDS: tuple[str, ...] = (
    "decision",
    "compile",
    "profile",
    "verify",
    "checkpoint",
    "rollback",
    "compact",
    "fallback",
    "done",
)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def snapshot() -> dict:
    """Frozen events_v1_schema.json snapshot."""
    return load_events_v1_schema()


@pytest.fixture(scope="module")
def golden_trajectory_path() -> Path:
    """Path to the locked golden trajectory fixture."""
    p = Path(__file__).parent / "fixtures" / "events_v1_golden_trajectory.jsonl"
    assert p.exists(), f"missing golden fixture at {p}"
    return p


# ── Layer 1: Version constants immutability ─────────────────────

class TestEventsVersion:
    """The frozen version identifiers must not drift."""

    def test_events_version_is_1_0_0(self) -> None:
        assert EVENTS_VERSION == "1.0.0"

    def test_contract_id_matches(self) -> None:
        assert EVENTS_CONTRACT_ID == "arke-harness-events-v1.0.0"

    def test_locked_on_date(self) -> None:
        assert EVENTS_LOCKED_ON == "2026-05-18"

    def test_snapshot_header_matches_constants(self, snapshot: dict) -> None:
        assert snapshot["events_version"] == EVENTS_VERSION
        assert snapshot["contract_id"] == EVENTS_CONTRACT_ID
        assert snapshot["locked_on"] == EVENTS_LOCKED_ON
        assert snapshot["kind_count"] == 9

    def test_design_ref_points_to_harness_doc(self, snapshot: dict) -> None:
        assert snapshot["design_ref"] == "docs/architecture/arke-harness.md §4 + §15"


# ── Layer 2: Kind set immutability ──────────────────────────────

class TestKindSet:
    """The 9-kind set is frozen — additions OK on MINOR, removals/renames forbidden."""

    def test_kind_count_is_9(self) -> None:
        assert len(EVENT_KINDS_V1) == 9

    def test_kinds_exact_set(self) -> None:
        assert set(EVENT_KINDS_V1) == set(EXPECTED_KINDS)

    def test_kinds_canonical_order(self) -> None:
        """Order is part of the contract for deterministic snapshots."""
        assert EVENT_KINDS_V1 == EXPECTED_KINDS

    def test_event_kind_enum_agrees(self) -> None:
        """EventKind enum and EVENT_KINDS_V1 tuple must not drift."""
        assert tuple(k.value for k in EventKind) == EVENT_KINDS_V1

    def test_snapshot_kinds_match_constants(self, snapshot: dict) -> None:
        snap_kinds = tuple(e["kind"] for e in snapshot["events"])
        assert snap_kinds == EVENT_KINDS_V1

    def test_wire_format_kind_enum_matches(self, snapshot: dict) -> None:
        assert tuple(snapshot["wire_format"]["kind_enum"]) == EVENT_KINDS_V1


# ── Layer 3: Payload schema immutability ────────────────────────

class TestPayloadSchemaImmutability:
    """Every kind's payload schema must match the frozen snapshot."""

    @pytest.mark.parametrize("kind", EXPECTED_KINDS)
    def test_payload_present_for_kind(self, kind: str) -> None:
        assert kind in EVENT_PAYLOADS_V1, f"missing payload schema for {kind!r}"

    @pytest.mark.parametrize("kind", EXPECTED_KINDS)
    def test_payload_field_set_matches_snapshot(self, snapshot: dict, kind: str) -> None:
        """Field names + order + types + required flags + descriptions all frozen."""
        snap_entry = next(e for e in snapshot["events"] if e["kind"] == kind)
        snap_fields = [
            (f["name"], f["type"], f["required"], f["description"])
            for f in snap_entry["payload_fields"]
        ]
        code_fields = [
            (f.name, f.type, f.required, f.description)
            for f in EVENT_PAYLOADS_V1[kind]
        ]
        assert code_fields == snap_fields, (
            f"Payload drift for {kind!r}:\n"
            f"  code: {code_fields}\n"
            f"  snap: {snap_fields}"
        )

    @pytest.mark.parametrize("kind", EXPECTED_KINDS)
    def test_required_fields_have_string_descriptions(self, kind: str) -> None:
        """Required fields must document themselves (no empty descriptions)."""
        for f in EVENT_PAYLOADS_V1[kind]:
            if f.required:
                assert f.description, f"{kind}.{f.name}: required field needs description"


# ── Layer 4: Wire format + round-trip immutability ──────────────

class TestWireFormat:
    """Wire-format envelope and round-trip semantics must not regress."""

    def test_envelope_shape_documented(self, snapshot: dict) -> None:
        wf = snapshot["wire_format"]
        assert "t" in wf["envelope"] and "kind" in wf["envelope"] and "data" in wf["envelope"]
        assert wf["t_unit"] == "seconds since session start (monotonic)"

    def test_to_dict_returns_t_kind_data(self) -> None:
        ev = OptimizationEvent(t=1.5, kind="profile", data={"latency_ms": 0.4, "vs_baseline": 1.1})
        assert ev.to_dict() == {"t": 1.5, "kind": "profile",
                                "data": {"latency_ms": 0.4, "vs_baseline": 1.1}}

    def test_to_jsonl_is_single_line(self) -> None:
        ev = make_event("done", {"final_score": 1.2, "decisions": 3, "compiles": 1,
                                  "termination": "budget_exhausted"}, t=2.0)
        line = ev.to_jsonl()
        assert "\n" not in line
        round_trip = OptimizationEvent.from_dict(json.loads(line))
        assert round_trip == ev

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown event kind"):
            OptimizationEvent(t=0.0, kind="bogus", data={})

    def test_from_dict_missing_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required keys"):
            OptimizationEvent.from_dict({"kind": "done"})  # missing t

    def test_from_dict_tolerates_extra_keys(self) -> None:
        """Forward-compat: unknown top-level keys are ignored, not errored."""
        ev = OptimizationEvent.from_dict({
            "t": 1.0, "kind": "done",
            "data": {"final_score": 1.0, "decisions": 1, "compiles": 1,
                     "termination": "llm_no_more_tool_use"},
            "future_key": "ignored",
        })
        assert ev.kind == "done"


# ── Layer 5: Payload validator round-trip ───────────────────────

class TestPayloadValidator:
    """validate_payload must agree with the schema declarations."""

    @pytest.mark.parametrize("kind", EXPECTED_KINDS)
    def test_minimal_required_payload_validates(self, kind: str) -> None:
        """Build a minimal payload with only required fields — must pass."""
        minimal = {f.name: _sample_for(f.type) for f in EVENT_PAYLOADS_V1[kind] if f.required}
        errs = validate_payload(kind, minimal)
        assert not errs, f"minimal payload for {kind} should validate: {errs}"

    @pytest.mark.parametrize("kind", EXPECTED_KINDS)
    def test_missing_required_field_fails(self, kind: str) -> None:
        """Dropping any required field must produce at least one error."""
        required = [f for f in EVENT_PAYLOADS_V1[kind] if f.required]
        if not required:
            pytest.skip(f"{kind} has no required fields")
        full = {f.name: _sample_for(f.type) for f in required}
        # Drop the first required field
        full.pop(required[0].name)
        errs = validate_payload(kind, full)
        assert any(required[0].name in e for e in errs), (
            f"validator should flag missing {required[0].name!r} for {kind}: {errs}"
        )

    def test_type_mismatch_fails(self) -> None:
        errs = validate_payload("profile", {"latency_ms": "fast", "vs_baseline": 1.0})
        assert any("latency_ms" in e and "number" in e for e in errs)

    def test_unknown_kind_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            validate_payload("bogus", {})


# ── Layer 6: Golden trajectory fixture immutability ─────────────

class TestGoldenTrajectory:
    """The locked golden fixture must round-trip + validate against the schema."""

    def test_fixture_exists_and_nonempty(self, golden_trajectory_path: Path) -> None:
        text = golden_trajectory_path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) >= 9, "golden trajectory must cover all 9 kinds"

    def test_fixture_covers_all_9_kinds(self, golden_trajectory_path: Path) -> None:
        text = golden_trajectory_path.read_text(encoding="utf-8")
        kinds_seen: set[str] = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            kinds_seen.add(obj["kind"])
        assert kinds_seen == set(EVENT_KINDS_V1), (
            f"golden fixture must exercise every kind; missing: {set(EVENT_KINDS_V1) - kinds_seen}"
        )

    def test_every_fixture_event_round_trips(self, golden_trajectory_path: Path) -> None:
        text = golden_trajectory_path.read_text(encoding="utf-8")
        for ln_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            ev = OptimizationEvent.from_dict(obj)
            assert ev.to_dict() == obj, f"line {ln_no}: round-trip drift"

    def test_every_fixture_payload_validates(self, golden_trajectory_path: Path) -> None:
        text = golden_trajectory_path.read_text(encoding="utf-8")
        for ln_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            errs = validate_payload(obj["kind"], obj["data"])
            assert not errs, f"line {ln_no} ({obj['kind']}): validation errors {errs}"


# ── Helpers ──────────────────────────────────────────────────────

def _sample_for(type_tag: str) -> object:
    """Return a minimal value of the right shape for the given type tag."""
    return {
        "string": "x",
        "int": 0,
        "number": 0.0,
        "bool": True,
        "object": {},
        "array": [],
        "null": None,
    }[type_tag]
