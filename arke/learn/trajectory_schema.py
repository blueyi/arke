# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Harness trajectory.jsonl v1.0 — locked record-level contract.

This module defines the **record-level** JSONL contract for the trajectory
file produced by every optimize run, per ``docs/architecture/arke-harness.md``
§15 (Trajectory & Learning).

Two layers — don't confuse them
--------------------------------
* **Stream-level** (:mod:`arke.agent.events`, D8-F2): wire format for live
  progress events emitted as an ``AsyncGenerator[OptimizationEvent, None]``.
  9 kinds. Consumers: CLI/REST/MCP/Jupyter.
* **Record-level** (this module + :mod:`arke.learn.trajectory`, D8-F3):
  JSONL file persisted to disk for post-hoc analysis & SFT/RL. **Strict
  superset** of the stream contract — every stream event is also a valid
  record, plus 2 record-only kinds:

  - ``header``  (exactly 1 line, must be the first line) carries session
    metadata + initial SemanticIR snapshot
  - ``adjust``  marks the end of a ``compile → profile → adjust`` cycle
    (the StrategyIR refinement step). Not a stream event because in the
    streaming view it is folded into the next ``decision`` event burst.

The codec (this module) lives under :mod:`arke.learn` because the
trajectory file is the **learning artifact** — SFT corpora, RL signal
extraction, and ``@rationale`` mining all read from here. The Façade
agent (:mod:`arke.agent`) does not depend on this module; conversely
this module imports the D8-F2 stream contract from
:mod:`arke.agent.events` because the record format is a strict
superset of the stream format.

Wire format
-----------
Identical envelope to :class:`arke.agent.events.OptimizationEvent`::

    {"t": <float>, "kind": <string>, "data": <object>}

This means a producer can emit a stream event straight to the JSONL file
with no shape adapter — D8-F3 deliberately keeps the same shape so the
SFT/RL extraction code has exactly one schema to parse.

Versioning policy (mirrors :mod:`arke.agent.events`)
----------------------------------------------------
* ``TRAJECTORY_VERSION`` is locked at ``1.0.0`` from 2026-05-19.
* The legacy ``schema`` string ``"s8-compile-profile-adjust-v1"`` is
  pinned in the header for backward-compat parsers. The new authoritative
  identifier is ``contract_id = "arke-trajectory-v1.0.0"``.
* Within MAJOR ``1.y.z``: new record kinds / new optional payload fields
  MAY be added; existing kind names + required payload fields MUST NOT
  change.
* Breaking changes bump MAJOR.
* The frozen snapshot lives at ``arke/learn/trajectory_v1_schema.json``
  and is enforced by ``tests/test_facade_trajectory_contract_v1.py``.

Relationship to D8-F2
---------------------
``RECORD_KINDS_V1`` = ``("header",) + EVENT_KINDS_V1 + ("adjust",)``.
Any change to ``EVENT_KINDS_V1`` propagates automatically to the record
kinds via the contract test, so the two contracts can never silently drift.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from arke.agent.events import (
    EVENT_KINDS_V1,
    EVENT_PAYLOADS_V1,
    PayloadField,
)

# ── Version constants ─────────────────────────────────────────────
TRAJECTORY_VERSION: str = "1.0.0"
TRAJECTORY_CONTRACT_ID: str = "arke-trajectory-v1.0.0"
TRAJECTORY_LOCKED_ON: str = "2026-05-19"

#: Legacy schema string pinned in the header for backward-compat parsers
#: that grep'd for it (e.g. ``benchmarks/gate_g8.py`` historically). Newer
#: consumers should key off ``contract_id`` instead.
LEGACY_SCHEMA: str = "s8-compile-profile-adjust-v1"


# ── RecordKind enum ───────────────────────────────────────────────
class RecordKind(str, Enum):
    """The locked v1.0 record kinds for ``trajectory.jsonl``.

    Strict superset of :class:`arke.agent.events.EventKind`: the 9 stream
    kinds plus ``header`` (envelope/session metadata) and ``adjust``
    (record-only cycle boundary).

    Inherits from ``str`` so wire format is exactly the kind name —
    JSONL stays human-readable.
    """

    # Record-only envelope (first line, exactly once)
    HEADER = "header"

    # The 9 stream kinds (must mirror EventKind 1:1)
    DECISION = "decision"
    COMPILE = "compile"
    PROFILE = "profile"
    VERIFY = "verify"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    COMPACT = "compact"
    FALLBACK = "fallback"
    DONE = "done"

    # Record-only cycle boundary
    ADJUST = "adjust"


#: Canonical ordering of v1.0 record kinds. ``header`` first, then the
#: full :data:`EVENT_KINDS_V1` in stream order, then ``adjust``.
RECORD_KINDS_V1: tuple[str, ...] = (
    "header",
    *EVENT_KINDS_V1,
    "adjust",
)
assert len(RECORD_KINDS_V1) == 11, "Trajectory v1.0 must have exactly 11 record kinds"
assert tuple(k.value for k in RecordKind) == RECORD_KINDS_V1, (
    "RecordKind enum drift vs RECORD_KINDS_V1 — both must agree"
)


# ── Record-only payload schemas ───────────────────────────────────
# Record-only kinds (``header`` and ``adjust``) get their payload schemas
# defined here. The 9 stream kinds reuse EVENT_PAYLOADS_V1 verbatim — no
# duplication, no drift surface.

#: Payload contract for the trajectory header envelope. Required fields
#: must be present in every header; optional fields are best-effort
#: metadata captured by the producer.
HEADER_PAYLOAD_V1: tuple[PayloadField, ...] = (
    PayloadField("schema", "string", True,
        "Legacy schema string (pinned for backward compat)"),
    PayloadField("trajectory_version", "string", True,
        "Trajectory contract semver (e.g. '1.0.0')"),
    PayloadField("contract_id", "string", True,
        "Frozen contract identifier (e.g. 'arke-trajectory-v1.0.0')"),
    PayloadField("kernel_id", "string", True,
        "Identifier of the kernel being optimized"),
    PayloadField("target_hw", "string", True,
        "Target hardware tag (e.g. 'nvidia-sm86', 'cpu', 'mock')"),
    PayloadField("mode", "string", True,
        "Run mode: 'compile' | 'dry-run'"),
    PayloadField("input_kind", "string", False,
        "Routed input kind (e.g. 'ak', 'natural_language', 'code')"),
    PayloadField("input_path", "string", False,
        "Display path of the input source"),
    PayloadField("normalized_source_path", "string", False,
        "Absolute path to the normalized .ak source"),
    PayloadField("source_text_path", "string", False,
        "Absolute path to the raw input text, when applicable"),
    PayloadField("required_cycle_order", "array", False,
        "Producer-asserted cycle action order, e.g. ['compile','profile','adjust']"),
    PayloadField("semantic_ir", "object", False,
        "Snapshot of the SemanticIR at session start "
        "(kernel_id, node_count, param_count, symbolic_dims)"),
)

#: Payload contract for the record-only ``adjust`` boundary marker.
#: Emitted at the end of each compile→profile→adjust cycle to report
#: StrategyIR refinement deltas.
ADJUST_PAYLOAD_V1: tuple[PayloadField, ...] = (
    PayloadField("cycle", "int", True,
        "1-indexed cycle number within the session"),
    PayloadField("decisions_before", "int", True,
        "Count of strategy decisions before this cycle's refinement"),
    PayloadField("decisions_after", "int", True,
        "Count of strategy decisions after this cycle's refinement"),
    PayloadField("changed", "bool", True,
        "Whether refinement modified the strategy decision list"),
    PayloadField("bottleneck", "string", False,
        "Heuristic bottleneck label that drove the refinement, if any"),
)


#: Full v1.0 record payload contract — record-only kinds plus the
#: 9 stream kinds via reference to :data:`EVENT_PAYLOADS_V1`. Field
#: order per kind is the canonical wire order.
RECORD_PAYLOADS_V1: dict[str, tuple[PayloadField, ...]] = {
    "header": HEADER_PAYLOAD_V1,
    **{k: EVENT_PAYLOADS_V1[k] for k in EVENT_KINDS_V1},
    "adjust": ADJUST_PAYLOAD_V1,
}

assert set(RECORD_PAYLOADS_V1.keys()) == set(RECORD_KINDS_V1), (
    "RECORD_PAYLOADS_V1 must cover every RECORD_KINDS_V1 entry exactly"
)


# ── TrajectoryRecord dataclass ────────────────────────────────────
@dataclass(frozen=True)
class TrajectoryRecord:
    """A single line in ``trajectory.jsonl``.

    Wire format (JSON)::

        {"t": <float>, "kind": <string>, "data": <object>}

    Identical envelope to :class:`arke.agent.events.OptimizationEvent`,
    by design — D8-F3 deliberately reuses the D8-F2 shape so SFT/RL
    extraction code parses one schema, not two.

    ``t`` is monotonic seconds since session start. ``kind`` must be one
    of :data:`RECORD_KINDS_V1`. ``data`` is the kind-specific payload.
    """

    t: float
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in RECORD_KINDS_V1:
            raise ValueError(
                f"Unknown trajectory record kind {self.kind!r}; "
                f"must be one of {RECORD_KINDS_V1}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to wire format (deterministic key order)."""
        return {"t": self.t, "kind": self.kind, "data": dict(self.data)}

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (no trailing newline)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryRecord:
        """Parse from wire format. Validates kind; tolerates extra keys."""
        if "kind" not in d or "t" not in d:
            raise ValueError(
                f"TrajectoryRecord dict missing required keys 't' or 'kind': {d!r}"
            )
        return cls(t=float(d["t"]), kind=str(d["kind"]), data=dict(d.get("data", {})))


def make_record(
    kind: str,
    data: dict[str, Any],
    *,
    t: float | None = None,
) -> TrajectoryRecord:
    """Convenience constructor that auto-stamps ``t`` if not given.

    The harness should pass ``t = elapsed_since_session_start`` for a
    monotonic timeline; default falls back to wall-clock seconds.
    """
    return TrajectoryRecord(t=t if t is not None else time.time(), kind=kind, data=data)


def validate_payload(kind: str, payload: dict[str, Any]) -> list[str]:
    """Check a payload against the v1.0 schema for ``kind``.

    Returns a list of validation errors. Empty list = valid.

    Validation rules (mirror :func:`arke.agent.events.validate_payload`):

      - All required fields present
      - Each present field's type matches the declared type tag
        (with permissive ``number`` = int|float, ``object`` = dict, etc.)
      - Extra unknown fields are tolerated (forward-compat)

    Raises ``KeyError`` if ``kind`` is not in :data:`RECORD_KINDS_V1`.
    """
    if kind not in RECORD_PAYLOADS_V1:
        raise KeyError(f"Unknown record kind: {kind!r}")
    errors: list[str] = []
    schema = RECORD_PAYLOADS_V1[kind]
    for f_spec in schema:
        if f_spec.required and f_spec.name not in payload:
            errors.append(f"missing required field {f_spec.name!r}")
            continue
        if f_spec.name in payload:
            value = payload[f_spec.name]
            if not _matches_type(value, f_spec.type):
                errors.append(
                    f"field {f_spec.name!r}: expected {f_spec.type}, "
                    f"got {type(value).__name__}"
                )
    return errors


def _matches_type(value: Any, tag: str) -> bool:
    """Check ``value`` against a payload type tag."""
    if tag == "string":
        return isinstance(value, str)
    if tag == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if tag == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tag == "bool":
        return isinstance(value, bool)
    if tag == "object":
        return isinstance(value, dict)
    if tag == "array":
        return isinstance(value, list)
    if tag == "null":
        return value is None
    return False


def build_header_data(
    *,
    kernel_id: str,
    target_hw: str,
    mode: str,
    input_kind: str | None = None,
    input_path: str | None = None,
    normalized_source_path: str | None = None,
    source_text_path: str | None = None,
    required_cycle_order: list[str] | None = None,
    semantic_ir: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a canonical header ``data`` payload.

    Always emits the three required version pins (``schema``,
    ``trajectory_version``, ``contract_id``) plus the caller-provided
    session metadata. Optional fields are dropped when ``None`` so the
    header stays compact.
    """
    data: dict[str, Any] = {
        "schema": LEGACY_SCHEMA,
        "trajectory_version": TRAJECTORY_VERSION,
        "contract_id": TRAJECTORY_CONTRACT_ID,
        "kernel_id": kernel_id,
        "target_hw": target_hw,
        "mode": mode,
    }
    if input_kind is not None:
        data["input_kind"] = input_kind
    if input_path is not None:
        data["input_path"] = input_path
    if normalized_source_path is not None:
        data["normalized_source_path"] = normalized_source_path
    if source_text_path is not None:
        data["source_text_path"] = source_text_path
    if required_cycle_order is not None:
        data["required_cycle_order"] = list(required_cycle_order)
    if semantic_ir is not None:
        data["semantic_ir"] = dict(semantic_ir)
    return data


# ── Frozen schema snapshot ────────────────────────────────────────
TRAJECTORY_V1_SCHEMA_PATH: Path = Path(__file__).parent / "trajectory_v1_schema.json"


def load_trajectory_v1_schema() -> dict[str, Any]:
    """Load the frozen trajectory v1.0 record-level schema snapshot.

    Returns the parsed JSON document. Raises FileNotFoundError if the
    snapshot is missing — re-run ``scripts/regen_trajectory_v1_schema.py``.
    """
    if not TRAJECTORY_V1_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Trajectory v1.0 frozen schema missing at {TRAJECTORY_V1_SCHEMA_PATH}; "
            "re-run scripts/regen_trajectory_v1_schema.py"
        )
    return json.loads(TRAJECTORY_V1_SCHEMA_PATH.read_text(encoding="utf-8"))


__all__ = [
    "TRAJECTORY_VERSION",
    "TRAJECTORY_CONTRACT_ID",
    "TRAJECTORY_LOCKED_ON",
    "LEGACY_SCHEMA",
    "RecordKind",
    "RECORD_KINDS_V1",
    "RECORD_PAYLOADS_V1",
    "HEADER_PAYLOAD_V1",
    "ADJUST_PAYLOAD_V1",
    "TrajectoryRecord",
    "make_record",
    "validate_payload",
    "build_header_data",
    "TRAJECTORY_V1_SCHEMA_PATH",
    "load_trajectory_v1_schema",
]
