# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for Arke trajectory v1.0 record-level schema (D8-F3).

Locks the on-disk snapshot, the enum/payload contract, the wire format,
and the layering relationship to the D8-F2 stream contract. Every test
here is a sentinel — a failing test means the trajectory record contract
has drifted, which is a versioning-bump-worthy event.

Mirrors the structure of ``tests/test_facade_events_contract_v1.py``
(D8-F2 stream contract tests) so producers can read one pattern and
understand both contracts.

Design ref: docs/architecture/arke-harness.md §15
Stage tracker: docs/phase1/stage8-plan.md D8-F3
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from arke.agent.events import EVENT_KINDS_V1
from arke.learn.trajectory_schema import (
    ADJUST_PAYLOAD_V1,
    HEADER_PAYLOAD_V1,
    LEGACY_SCHEMA,
    RECORD_KINDS_V1,
    RECORD_PAYLOADS_V1,
    TRAJECTORY_CONTRACT_ID,
    TRAJECTORY_LOCKED_ON,
    TRAJECTORY_V1_SCHEMA_PATH,
    TRAJECTORY_VERSION,
    RecordKind,
    TrajectoryRecord,
    build_header_data,
    load_trajectory_v1_schema,
    make_record,
    validate_payload,
)
from arke.learn.trajectory import TrajectoryWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "trajectory_v1_golden.jsonl"


# ── Version pins ──────────────────────────────────────────────────
def test_trajectory_version_locked_at_1_0_0() -> None:
    assert TRAJECTORY_VERSION == "1.0.0", (
        "Trajectory record contract is locked at 1.0.0; bump only with explicit "
        "Leon approval — see docs/architecture/arke-harness.md §15"
    )


def test_trajectory_contract_id_canonical() -> None:
    assert TRAJECTORY_CONTRACT_ID == "arke-trajectory-v1.0.0"


def test_legacy_schema_pin_preserved() -> None:
    # Old parsers grep'd for this exact string in the header; keep it stable.
    assert LEGACY_SCHEMA == "s8-compile-profile-adjust-v1"


def test_locked_on_date_is_set() -> None:
    assert TRAJECTORY_LOCKED_ON == "2026-05-19"


# ── Record kind enum ──────────────────────────────────────────────
def test_record_kinds_v1_has_exactly_11_kinds() -> None:
    assert len(RECORD_KINDS_V1) == 11


def test_record_kinds_v1_ordering_canonical() -> None:
    # header first, then the 9 stream kinds in stream order, then adjust.
    assert RECORD_KINDS_V1 == ("header", *EVENT_KINDS_V1, "adjust")


def test_record_kind_enum_matches_tuple() -> None:
    assert tuple(k.value for k in RecordKind) == RECORD_KINDS_V1


def test_record_kinds_is_strict_superset_of_stream_kinds() -> None:
    """D8-F3 contract: every D8-F2 stream kind is a valid record kind."""
    assert set(EVENT_KINDS_V1).issubset(set(RECORD_KINDS_V1))
    extra = set(RECORD_KINDS_V1) - set(EVENT_KINDS_V1)
    assert extra == {"header", "adjust"}


# ── Payload schema ────────────────────────────────────────────────
def test_record_payloads_covers_every_kind() -> None:
    assert set(RECORD_PAYLOADS_V1.keys()) == set(RECORD_KINDS_V1)


def test_record_payloads_for_stream_kinds_match_d8_f2() -> None:
    """Record-level reuse must not silently shadow stream-level fields."""
    from arke.agent.events import EVENT_PAYLOADS_V1

    for kind in EVENT_KINDS_V1:
        assert RECORD_PAYLOADS_V1[kind] is EVENT_PAYLOADS_V1[kind], (
            f"record payload for stream kind {kind!r} diverged from D8-F2 — "
            f"contracts must share the same PayloadField objects"
        )


def test_header_payload_has_required_version_pins() -> None:
    required_fields = {f.name for f in HEADER_PAYLOAD_V1 if f.required}
    # These three are the version pins consumers grep for; they MUST stay required.
    assert {"schema", "trajectory_version", "contract_id"}.issubset(required_fields)
    # Session identity must also be required so trajectories are self-describing.
    assert {"kernel_id", "target_hw", "mode"}.issubset(required_fields)


def test_adjust_payload_has_required_cycle_deltas() -> None:
    required_fields = {f.name for f in ADJUST_PAYLOAD_V1 if f.required}
    assert {"cycle", "decisions_before", "decisions_after", "changed"} == required_fields


# ── TrajectoryRecord round-trip ───────────────────────────────────
def test_trajectory_record_envelope_keys() -> None:
    rec = make_record("header", build_header_data(
        kernel_id="matmul", target_hw="nvidia-sm86", mode="compile"))
    d = rec.to_dict()
    assert set(d.keys()) == {"t", "kind", "data"}
    assert d["kind"] == "header"


def test_trajectory_record_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown trajectory record kind"):
        TrajectoryRecord(t=0.0, kind="not_a_real_kind", data={})


def test_trajectory_record_jsonl_round_trip() -> None:
    rec = make_record("adjust", {
        "cycle": 1, "decisions_before": 0, "decisions_after": 1, "changed": True,
    })
    line = rec.to_jsonl()
    parsed = TrajectoryRecord.from_dict(json.loads(line))
    assert parsed.kind == rec.kind
    assert parsed.data == rec.data


# ── validate_payload ──────────────────────────────────────────────
def test_validate_payload_header_accepts_canonical() -> None:
    data = build_header_data(
        kernel_id="matmul", target_hw="nvidia-sm86", mode="compile",
        semantic_ir={"kernel_id": "matmul", "node_count": 7},
    )
    assert validate_payload("header", data) == []


def test_validate_payload_header_flags_missing_required() -> None:
    errs = validate_payload("header", {"kernel_id": "matmul"})
    assert any("trajectory_version" in e for e in errs)
    assert any("contract_id" in e for e in errs)


def test_validate_payload_unknown_kind_raises() -> None:
    with pytest.raises(KeyError):
        validate_payload("not_a_real_kind", {})


def test_validate_payload_adjust_accepts_canonical() -> None:
    payload = {
        "cycle": 1, "decisions_before": 2, "decisions_after": 3, "changed": True,
        "bottleneck": "memory_bandwidth",
    }
    assert validate_payload("adjust", payload) == []


def test_validate_payload_adjust_flags_wrong_type() -> None:
    payload = {
        "cycle": "one", "decisions_before": 2, "decisions_after": 3, "changed": True,
    }
    errs = validate_payload("adjust", payload)
    assert any("cycle" in e and "int" in e for e in errs)


# ── Frozen snapshot ───────────────────────────────────────────────
def test_frozen_snapshot_exists() -> None:
    assert TRAJECTORY_V1_SCHEMA_PATH.exists(), (
        f"missing {TRAJECTORY_V1_SCHEMA_PATH}; "
        "re-run scripts/regen_trajectory_v1_schema.py"
    )


def test_frozen_snapshot_loads_and_has_expected_top_keys() -> None:
    doc = load_trajectory_v1_schema()
    assert doc["trajectory_version"] == "1.0.0"
    assert doc["contract_id"] == "arke-trajectory-v1.0.0"
    assert doc["legacy_schema"] == "s8-compile-profile-adjust-v1"
    assert doc["kind_count"] == 11
    assert doc["wire_format"]["kind_enum"] == list(RECORD_KINDS_V1)
    assert doc["wire_format"]["first_line_kind"] == "header"
    assert doc["layering"]["stream_contract"].startswith("arke-harness-events-v1.0.0")


def test_frozen_snapshot_records_cover_every_kind() -> None:
    doc = load_trajectory_v1_schema()
    snapshot_kinds = [r["kind"] for r in doc["records"]]
    assert snapshot_kinds == list(RECORD_KINDS_V1)
    for record in doc["records"]:
        live_schema = RECORD_PAYLOADS_V1[record["kind"]]
        snapshot_fields = record["payload_fields"]
        assert len(snapshot_fields) == len(live_schema), (
            f"field count drift in {record['kind']!r}: "
            f"live={len(live_schema)} snapshot={len(snapshot_fields)}"
        )
        for f_live, f_snap in zip(live_schema, snapshot_fields):
            assert f_snap["name"] == f_live.name
            assert f_snap["type"] == f_live.type
            assert f_snap["required"] == f_live.required


def test_regen_script_check_mode_passes_on_current_snapshot() -> None:
    """CI guard: ``scripts/regen_trajectory_v1_schema.py --check`` must exit 0."""
    proc = subprocess.run(
        [sys.executable, "scripts/regen_trajectory_v1_schema.py", "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"regen --check failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )


# ── Golden fixture round-trip ─────────────────────────────────────
def test_golden_fixture_parses_with_v1_codec() -> None:
    assert GOLDEN_FIXTURE.exists(), (
        f"missing golden fixture {GOLDEN_FIXTURE}; this fixture lives in-repo "
        "as the canonical wire-format exemplar"
    )
    lines = GOLDEN_FIXTURE.read_text().splitlines()
    assert len(lines) >= 5
    records = [TrajectoryRecord.from_dict(json.loads(line)) for line in lines]
    # First line must be header (record-level contract invariant).
    assert records[0].kind == "header"
    # Every kind in the fixture is valid v1.0.
    for rec in records:
        assert rec.kind in RECORD_KINDS_V1


def test_golden_fixture_payloads_validate() -> None:
    """Every record in the golden fixture passes :func:`validate_payload`."""
    lines = GOLDEN_FIXTURE.read_text().splitlines()
    for i, line in enumerate(lines):
        d = json.loads(line)
        errs = validate_payload(d["kind"], d.get("data", {}))
        assert errs == [], (
            f"golden fixture line {i+1} (kind={d['kind']!r}) failed validation: {errs}"
        )


# ── Writer integration ────────────────────────────────────────────
@pytest.mark.parametrize("kind", list(RECORD_KINDS_V1))
def test_writer_emits_every_kind(tmp_path: Path, kind: str) -> None:
    """The :class:`TrajectoryWriter` accepts every locked record kind."""
    out = tmp_path / "trajectory.jsonl"
    with TrajectoryWriter(out) as writer:
        if kind == "header":
            writer.write_header({
                "kernel_id": "k", "target_hw": "hw", "mode": "compile",
            })
        else:
            # Header must come first; write a minimal header then the kind.
            writer.write_header({
                "kernel_id": "k", "target_hw": "hw", "mode": "compile",
            })
            # Build a minimally-valid payload for the kind by filling
            # required fields with type-appropriate placeholders.
            payload = _minimal_payload(kind)
            writer.write_record(kind, payload)

    lines = out.read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    kinds = [p["kind"] for p in parsed]
    assert kinds[0] == "header"
    if kind != "header":
        assert kind in kinds


def test_writer_rejects_second_header(tmp_path: Path) -> None:
    out = tmp_path / "trajectory.jsonl"
    with TrajectoryWriter(out) as writer:
        writer.write_header({"kernel_id": "k", "target_hw": "hw", "mode": "compile"})
        with pytest.raises(RuntimeError, match="header already written"):
            writer.write_header({"kernel_id": "k", "target_hw": "hw", "mode": "compile"})


def test_writer_rejects_unknown_kind(tmp_path: Path) -> None:
    out = tmp_path / "trajectory.jsonl"
    with TrajectoryWriter(out) as writer:
        writer.write_header({"kernel_id": "k", "target_hw": "hw", "mode": "compile"})
        with pytest.raises(ValueError, match="Unknown trajectory record kind"):
            writer.write_record("not_a_real_kind", {})


# ── Helpers ───────────────────────────────────────────────────────
def _minimal_payload(kind: str) -> dict:
    """Synthesize a payload satisfying the required fields of ``kind``."""
    schema = RECORD_PAYLOADS_V1[kind]
    payload: dict = {}
    for f in schema:
        if not f.required:
            continue
        payload[f.name] = _placeholder_for(f.type)
    return payload


def _placeholder_for(tag: str):
    if tag == "string":
        return "x"
    if tag == "int":
        return 1
    if tag == "number":
        return 1.0
    if tag == "bool":
        return True
    if tag == "object":
        return {}
    if tag == "array":
        return []
    if tag == "null":
        return None
    raise ValueError(f"no placeholder for tag {tag!r}")
