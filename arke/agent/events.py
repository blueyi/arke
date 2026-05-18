# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Harness OptimizationEvent stream v1.0 — locked public contract.

This module defines the public **stream-level event contract** for the
Façade per ``docs/architecture/arke-harness.md`` §4 (The Loop) +
§15 (Trajectory & Learning).

The harness exposes optimization progress as
``AsyncGenerator[OptimizationEvent, None]`` — a single primitive consumed
identically by the CLI, the Python API, the REST surface, the Jupyter
renderer, and the MCP server. **Any** Façade-compatible agent must be
able to decode every event kind in :data:`EVENT_KINDS_V1` without
peeking into Substrate-internal types.

Two layers — don't confuse them
--------------------------------
* **Stream-level** (this module): wire format for live progress events.
  Producers: the harness loop. Consumers: CLI/REST/MCP/Jupyter.
* **Record-level** (``arke/learn/trajectory.py``): JSONL log of
  ``(step, action, result)`` triplets for post-hoc analysis & SFT/RL.
  D8-F3 will lock its schema (``s8-compile-profile-adjust-v1``) on top
  of this stream contract.

The 9 locked event kinds (§4 ∪ §15, with `checkpoint` reconciled in)
-------------------------------------------------------------------
1. ``decision``    — apply_decision applied a legal mutation
2. ``compile``     — compile_and_profile completed a build attempt
3. ``profile``     — V2 GPU microbench measurement
4. ``verify``      — V1 numeric correctness check vs reference
5. ``checkpoint``  — labelled snapshot taken (§15 example, recovered into §4)
6. ``rollback``    — restored a previous checkpoint
7. ``compact``     — message log was compacted (state survives — §8)
8. ``fallback``    — provider/strategy fallback engaged (§16)
9. ``done``        — loop terminated; final result attached

Versioning policy (mirrors :mod:`arke.agent.facade`)
----------------------------------------------------
* ``EVENTS_VERSION`` is locked at ``1.0.0`` from 2026-05-18.
* Within MAJOR ``1.y.z``: new kinds / new optional payload fields MAY
  be added; existing kind names + required payload fields MUST NOT
  change.
* Breaking changes (renaming a kind, removing a kind, changing a
  required field's type) bump MAJOR.
* The frozen snapshot lives at
  ``arke/agent/events_v1_schema.json`` and is enforced by
  ``tests/test_facade_events_contract_v1.py``.

Doc-bug reconciliation (D8-F2, 2026-05-18)
------------------------------------------
Pre-D8-F2, ``arke-harness.md`` §4 listed 8 kinds (omitting
``checkpoint``) while §15's JSONL example used ``checkpoint``. The
``checkpoint`` event is a real observable produced by tool 7
(``checkpoint``) on :class:`arke.agent.state.OptimizationState`.
D8-F2 reconciles to the 9-kind union; §4 is corrected accordingly.
This is a doc-bug fix, not a contract widening — the 9th kind always
existed in the Substrate; only §4's table was incomplete.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ── Version constants ─────────────────────────────────────────────
EVENTS_VERSION: str = "1.0.0"
EVENTS_CONTRACT_ID: str = "arke-harness-events-v1.0.0"
EVENTS_LOCKED_ON: str = "2026-05-18"


# ── EventKind enum ────────────────────────────────────────────────
class EventKind(str, Enum):
    """The 9 locked OptimizationEvent kinds.

    Inherits from ``str`` so the wire format is exactly the kind name
    (e.g. ``"decision"``), keeping JSONL output human-readable and
    backward-compatible with raw-string consumers.
    """

    DECISION = "decision"
    COMPILE = "compile"
    PROFILE = "profile"
    VERIFY = "verify"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    COMPACT = "compact"
    FALLBACK = "fallback"
    DONE = "done"


#: Canonical ordering of the 9 v1.0 event kinds. Used for schema
#: enumeration, contract tests, and documentation. Order is the §4
#: listing extended with ``checkpoint`` re-inserted before ``rollback``
#: (its natural lifecycle neighbour).
EVENT_KINDS_V1: tuple[str, ...] = (
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
assert len(EVENT_KINDS_V1) == 9, "OptimizationEvent v1.0 must have exactly 9 kinds"
assert tuple(k.value for k in EventKind) == EVENT_KINDS_V1, (
    "EventKind enum drift vs EVENT_KINDS_V1 — both must agree"
)


# ── Payload field schemas ─────────────────────────────────────────
# Each kind declares its payload contract: required + optional fields,
# each with a JSON-compatible type tag. The contract is intentionally
# permissive — payloads MAY carry additional keys (for forward compat),
# but the required keys MUST be present with the declared shape.

#: Wire-format JSON type tags. Kept minimal on purpose — Façade agents
#: should not need a full JSON-Schema validator to consume the stream.
_TYPE_TAGS = frozenset({
    "string",
    "int",
    "number",   # int or float
    "bool",
    "object",   # arbitrary nested dict (Substrate-internal blobs go here)
    "array",
    "null",
})


@dataclass(frozen=True)
class PayloadField:
    """One typed field within an event payload."""

    name: str
    type: str         # one of _TYPE_TAGS
    required: bool
    description: str

    def __post_init__(self) -> None:
        if self.type not in _TYPE_TAGS:
            raise ValueError(
                f"PayloadField {self.name!r}: unknown type tag {self.type!r}; "
                f"must be one of {sorted(_TYPE_TAGS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }


# Payload contracts per kind. Field order is the canonical wire order
# (used by the regen script to produce deterministic snapshots).
EVENT_PAYLOADS_V1: dict[str, tuple[PayloadField, ...]] = {
    "decision": (
        PayloadField("decision", "object", True,
            "Serialized Decision (kind + params + rationale)"),
        PayloadField("rationale", "string", True,
            "Human-readable WHY for the decision"),
        PayloadField("step", "int", False,
            "1-indexed decision step within the session"),
        PayloadField("decisions_used", "int", False,
            "Budget counter snapshot after this decision"),
    ),
    "compile": (
        PayloadField("backend", "string", True,
            "Compile backend: 'triton' | 'cuda' | 'mock'"),
        PayloadField("success", "bool", True,
            "Whether the compile step produced a runnable artifact"),
        PayloadField("build_ms", "number", False,
            "Wall-clock build time in milliseconds"),
        PayloadField("error", "string", False,
            "Compiler error string when success=false"),
        PayloadField("artifact_path", "string", False,
            "Absolute path to the produced binary/PTX when applicable"),
    ),
    "profile": (
        PayloadField("latency_ms", "number", True,
            "Median wall-clock latency from microbench"),
        PayloadField("vs_baseline", "number", True,
            "latency_baseline / latency_arke (>1 = Arke faster)"),
        PayloadField("baseline_name", "string", False,
            "Baseline runner identifier (e.g. 'pytorch', 'flagems')"),
        PayloadField("samples", "int", False,
            "Number of microbench iterations sampled"),
        PayloadField("bottleneck", "string", False,
            "Heuristic bottleneck label (e.g. 'memory_bandwidth')"),
    ),
    "verify": (
        PayloadField("tier", "string", True,
            "Verification tier: 'v0' (static) | 'v1' (numeric)"),
        PayloadField("pass", "bool", True,
            "Whether the check passed"),
        PayloadField("max_diff", "number", False,
            "V1: max abs delta vs NumPy reference"),
        PayloadField("tolerance", "number", False,
            "V1: tolerance threshold used for pass/fail"),
        PayloadField("error", "string", False,
            "Error string when pass=false"),
    ),
    "checkpoint": (
        PayloadField("label", "string", True,
            "Checkpoint label (used for rollback target)"),
        PayloadField("score", "number", False,
            "Best-known vs_baseline at checkpoint time"),
        PayloadField("decision_count_at", "int", False,
            "Budget snapshot — decisions_used at checkpoint"),
        PayloadField("compile_count_at", "int", False,
            "Budget snapshot — compiles_used at checkpoint"),
    ),
    "rollback": (
        PayloadField("label", "string", True,
            "Checkpoint label that was restored"),
        PayloadField("discarded_decisions", "int", False,
            "Count of decisions popped from log on restore"),
        PayloadField("reason", "string", False,
            "Why rollback was triggered (e.g. 'regression detected')"),
    ),
    "compact": (
        PayloadField("removed", "int", True,
            "Number of messages removed by compaction"),
        PayloadField("kept", "int", True,
            "Number of messages retained after compaction"),
        PayloadField("tokens_before", "int", False,
            "Estimated token count before compact"),
        PayloadField("tokens_after", "int", False,
            "Estimated token count after compact"),
    ),
    "fallback": (
        PayloadField("layer", "string", True,
            "Fallback layer: 'strategy' | 'provider' | 'tier'"),
        PayloadField("from", "string", True,
            "Source provider/strategy/tier identifier"),
        PayloadField("to", "string", True,
            "Destination provider/strategy/tier identifier"),
        PayloadField("reason", "string", False,
            "Why fallback engaged (e.g. 'provider_unreachable')"),
    ),
    "done": (
        PayloadField("final_score", "number", True,
            "Final vs_baseline of the chosen strategy"),
        PayloadField("decisions", "int", True,
            "Total decisions applied in the session"),
        PayloadField("compiles", "int", True,
            "Total compile_and_profile calls in the session"),
        PayloadField("termination", "string", True,
            "Why loop ended: 'llm_no_more_tool_use' | "
            "'budget_exhausted' | 'caller_aclose' | 'hard_error'"),
        PayloadField("chosen", "string", False,
            "Which strategy won: 'llm' | 'heuristic_floor'"),
    ),
}

assert set(EVENT_PAYLOADS_V1.keys()) == set(EVENT_KINDS_V1), (
    "EVENT_PAYLOADS_V1 must cover every EVENT_KINDS_V1 entry exactly"
)


# ── OptimizationEvent dataclass ───────────────────────────────────
@dataclass(frozen=True)
class OptimizationEvent:
    """A single event emitted by the Façade loop.

    Wire format (JSON):
        {"t": <float>, "kind": <string>, "data": <object>}

    where ``t`` is monotonic seconds since session start, ``kind`` is
    one of :data:`EVENT_KINDS_V1`, and ``data`` is the kind-specific
    payload (see :data:`EVENT_PAYLOADS_V1`).

    The dataclass is frozen because events are immutable values once
    emitted — downstream consumers may cache or replay them.
    """

    t: float
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS_V1:
            raise ValueError(
                f"Unknown event kind {self.kind!r}; "
                f"must be one of {EVENT_KINDS_V1}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to wire format (deterministic key order)."""
        return {"t": self.t, "kind": self.kind, "data": dict(self.data)}

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (no trailing newline)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OptimizationEvent:
        """Parse from wire format. Validates kind; tolerates extra keys."""
        if "kind" not in d or "t" not in d:
            raise ValueError(
                f"OptimizationEvent dict missing required keys 't' or 'kind': {d!r}"
            )
        return cls(t=float(d["t"]), kind=str(d["kind"]), data=dict(d.get("data", {})))


def make_event(kind: str, data: dict[str, Any], *, t: float | None = None) -> OptimizationEvent:
    """Convenience constructor that auto-stamps ``t`` if not given.

    The default stamps wall-clock seconds; the harness loop should pass
    ``t = elapsed_since_session_start`` for monotonic stream timing.
    """
    return OptimizationEvent(t=t if t is not None else time.time(), kind=kind, data=data)


def validate_payload(kind: str, payload: dict[str, Any]) -> list[str]:
    """Check a payload against the v1.0 schema for ``kind``.

    Returns a list of validation errors. Empty list = valid.

    Validation rules:
      - All required fields present
      - Each present field's type matches the declared type tag
        (with permissive ``number`` = int|float, ``object`` = dict, etc.)
      - Extra unknown fields are tolerated (forward-compat)

    Raises ``KeyError`` if ``kind`` is not one of EVENT_KINDS_V1.
    """
    if kind not in EVENT_PAYLOADS_V1:
        raise KeyError(f"Unknown event kind: {kind!r}")
    errors: list[str] = []
    schema = EVENT_PAYLOADS_V1[kind]
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


# ── Frozen schema snapshot ────────────────────────────────────────
EVENTS_V1_SCHEMA_PATH: Path = Path(__file__).parent / "events_v1_schema.json"


def load_events_v1_schema() -> dict[str, Any]:
    """Load the frozen OptimizationEvent v1.0 schema snapshot.

    Returns the parsed JSON document. Raises FileNotFoundError if the
    snapshot is missing — re-run ``scripts/regen_events_v1_schema.py``.
    """
    if not EVENTS_V1_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"OptimizationEvent v1.0 frozen schema missing at {EVENTS_V1_SCHEMA_PATH}; "
            "re-run scripts/regen_events_v1_schema.py"
        )
    return json.loads(EVENTS_V1_SCHEMA_PATH.read_text(encoding="utf-8"))


__all__ = [
    "EVENTS_VERSION",
    "EVENTS_CONTRACT_ID",
    "EVENTS_LOCKED_ON",
    "EventKind",
    "EVENT_KINDS_V1",
    "PayloadField",
    "EVENT_PAYLOADS_V1",
    "OptimizationEvent",
    "make_event",
    "validate_payload",
    "EVENTS_V1_SCHEMA_PATH",
    "load_events_v1_schema",
]
