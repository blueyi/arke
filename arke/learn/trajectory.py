# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Learn — Trajectory recording and export.

Persists optimization trajectories to ``trajectory.jsonl`` for post-hoc
analysis, SFT/RL extraction, and ``@rationale`` knowledge mining.

Contract: D8-F3 trajectory v1.0 — see :mod:`arke.agent.trajectory` for
the locked record schema. Wire format is the same envelope as the
D8-F2 stream contract::

    {"t": <float>, "kind": <string>, "data": <object>}

This module is the **record-level writer** — the persistence sink. The
stream-level wire format lives in :mod:`arke.agent.events`. Both layers
share the same envelope by design; the trajectory format is a strict
superset (adds ``header`` and ``adjust`` record-only kinds).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from arke.learn.trajectory_schema import (
    RECORD_KINDS_V1,
    TrajectoryRecord,
    build_header_data,
)


class TrajectoryWriter:
    """Writes optimization trajectory to JSONL file (v1.0 record contract).

    Every emitted line is a :class:`arke.agent.trajectory.TrajectoryRecord`
    serialized as a single JSONL row with the canonical envelope
    ``{"t": <float>, "kind": <string>, "data": <object>}``.

    The first line MUST be a ``header`` record; subsequent lines are
    stream-level events (``decision``, ``compile``, ``profile``,
    ``verify``, ``checkpoint``, ``rollback``, ``compact``, ``fallback``,
    ``done``) and the record-only ``adjust`` cycle marker.

    Typical usage::

        with TrajectoryWriter(path) as writer:
            writer.write_header({
                "kernel_id": ..., "target_hw": ..., "mode": "compile",
                "semantic_ir": {...},
            })
            writer.write_record("compile", {"backend": "triton", "success": True, ...})
            writer.write_record("profile", {"latency_ms": 0.4, "vs_baseline": 1.2})
            writer.write_record("adjust", {"cycle": 1, ...})
            writer.write_record("done", {"final_score": 1.2, ...})

    Convenience helpers (``write_compile``, ``write_profile``, etc.)
    wrap ``write_record`` with the corresponding kind hard-coded so
    call sites stay self-documenting.
    """

    def __init__(self, path: str | Path):
        """Initialize the trajectory writer for the given file path."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        self._t0 = time.monotonic()
        self._header_written = False

    # ── low-level write ───────────────────────────────────────────
    def _elapsed(self) -> float:
        """Monotonic seconds since the writer was opened."""
        return time.monotonic() - self._t0

    def write_record(self, kind: str, data: dict[str, Any]) -> None:
        """Emit a single trajectory record.

        Stamps ``t`` with monotonic seconds since session start.
        Validates ``kind`` against the v1.0 record contract.
        """
        if kind not in RECORD_KINDS_V1:
            raise ValueError(
                f"Unknown trajectory record kind {kind!r}; "
                f"must be one of {RECORD_KINDS_V1}"
            )
        rec = TrajectoryRecord(t=self._elapsed(), kind=kind, data=dict(data))
        self._file.write(json.dumps(rec.to_dict(), ensure_ascii=False, default=str) + "\n")

    # ── header (record-only) ──────────────────────────────────────
    def write_header(self, metadata: dict[str, Any]) -> None:
        """Write the session header line. Must be the first line in the file.

        ``metadata`` should carry the caller-supplied session fields
        (``kernel_id``, ``target_hw``, ``mode``, ``semantic_ir``, etc.).
        The writer auto-injects the three required version pins
        (``schema``, ``trajectory_version``, ``contract_id``) via
        :func:`arke.agent.trajectory.build_header_data`.
        """
        if self._header_written:
            raise RuntimeError(
                "trajectory.jsonl header already written; v1.0 contract "
                "requires exactly one header record at file start"
            )
        data = build_header_data(
            kernel_id=metadata.get("kernel_id", ""),
            target_hw=metadata.get("target_hw", ""),
            mode=metadata.get("mode", "compile"),
            input_kind=metadata.get("input_kind"),
            input_path=metadata.get("input_path"),
            normalized_source_path=metadata.get("normalized_source_path"),
            source_text_path=metadata.get("source_text_path"),
            required_cycle_order=metadata.get("required_cycle_order"),
            semantic_ir=metadata.get("semantic_ir"),
        )
        self.write_record("header", data)
        self._header_written = True

    # ── stream-kind convenience wrappers ──────────────────────────
    def write_decision(self, payload: dict[str, Any]) -> None:
        """Emit a ``decision`` event (D8-F2 stream kind)."""
        self.write_record("decision", payload)

    def write_compile(self, payload: dict[str, Any]) -> None:
        """Emit a ``compile`` event (D8-F2 stream kind)."""
        self.write_record("compile", payload)

    def write_profile(self, payload: dict[str, Any]) -> None:
        """Emit a ``profile`` event (D8-F2 stream kind)."""
        self.write_record("profile", payload)

    def write_verify(self, payload: dict[str, Any]) -> None:
        """Emit a ``verify`` event (D8-F2 stream kind)."""
        self.write_record("verify", payload)

    def write_checkpoint(self, payload: dict[str, Any]) -> None:
        """Emit a ``checkpoint`` event (D8-F2 stream kind)."""
        self.write_record("checkpoint", payload)

    def write_rollback(self, payload: dict[str, Any]) -> None:
        """Emit a ``rollback`` event (D8-F2 stream kind)."""
        self.write_record("rollback", payload)

    def write_compact(self, payload: dict[str, Any]) -> None:
        """Emit a ``compact`` event (D8-F2 stream kind)."""
        self.write_record("compact", payload)

    def write_fallback(self, payload: dict[str, Any]) -> None:
        """Emit a ``fallback`` event (D8-F2 stream kind)."""
        self.write_record("fallback", payload)

    def write_done(self, payload: dict[str, Any]) -> None:
        """Emit a ``done`` event (D8-F2 stream kind, terminal)."""
        self.write_record("done", payload)

    # ── record-only kind ──────────────────────────────────────────
    def write_adjust(self, payload: dict[str, Any]) -> None:
        """Emit an ``adjust`` cycle-boundary record (record-only)."""
        self.write_record("adjust", payload)

    # ── lifecycle ─────────────────────────────────────────────────
    def flush(self) -> None:
        """Flush buffered trajectory data to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the trajectory file."""
        self._file.close()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def export_session_trajectory(
    trajectory: list[dict[str, Any]],
    session_summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export a completed session's trajectory to JSONL (v1.0 contract).

    Args:
        trajectory: List of trajectory entries from
            ``OptimizationSession.export_trajectory()``.
        session_summary: Session summary dict (carries ``kernel_id``,
            ``target_hw``, ``decisions``, ``best_performance``, etc.).
        output_path: Path to write JSONL file.

    Translates the v0 ``{type, step, tool, params, result}`` entries
    used by :mod:`arke.agent.session` into v1.0 stream events. Each
    ``action`` entry maps to the matching stream kind (e.g. tool
    ``compile_and_profile`` → ``compile``) and is paired with the next
    ``result`` entry's success flag + metrics. Entries that don't map
    cleanly fall through as ``decision`` events carrying the raw tool
    call payload so no signal is lost.
    """
    with TrajectoryWriter(output_path) as writer:
        # Header
        writer.write_header({
            "kernel_id": session_summary.get("kernel_id", ""),
            "target_hw": session_summary.get("target_hw", ""),
            "mode": "compile",
        })

        # Pair up (action, result) tuples and emit as stream events
        pending_action: dict[str, Any] | None = None
        for entry in trajectory:
            etype = entry.get("type")
            if etype == "action":
                pending_action = entry
                continue
            if etype == "result" and pending_action is not None:
                kind = _action_tool_to_kind(pending_action.get("tool", ""))
                payload = _build_paired_payload(pending_action, entry, kind)
                writer.write_record(kind, payload)
                pending_action = None

        # Final done event
        writer.write_done({
            "final_score": float(session_summary.get("best_performance") or 0.0),
            "decisions": int(session_summary.get("decisions", 0) or 0),
            "compiles": int(session_summary.get("compiles", 0) or 0),
            "termination": str(session_summary.get("termination", "llm_no_more_tool_use")),
        })


_TOOL_TO_KIND: dict[str, str] = {
    "apply_decision": "decision",
    "compile_and_profile": "compile",
    "verify_correctness": "verify",
    "checkpoint": "checkpoint",
    "rollback": "rollback",
}


def _action_tool_to_kind(tool: str) -> str:
    """Map a session tool name to a v1.0 record kind. Falls back to ``decision``."""
    return _TOOL_TO_KIND.get(tool, "decision")


def _build_paired_payload(
    action: dict[str, Any],
    result: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    """Merge an (action, result) pair into a single payload for ``kind``.

    Best-effort: copies action params into the payload, then overlays
    any matching result fields. The contract permits extra keys, so we
    keep raw data accessible for downstream tooling.
    """
    payload: dict[str, Any] = dict(action.get("params", {}))
    res = result.get("result", {}) or {}
    if isinstance(res, dict):
        payload.update(res)
    success = res.get("success") if isinstance(res, dict) else None
    if success is not None and kind == "compile" and "success" not in payload:
        payload["success"] = bool(success)
    return payload


# ── A5 (2026-06-26): @rationale trajectory contract assertion ─────────────


def audit_decision_rationales(trajectory_path: str | Path) -> list[str]:
    """Audit a trajectory JSONL: every `decision` record must carry a
    non-empty rationale.

    This is the gate-style enforcement complement to S4 (which enforces
    rationale at the `apply_decision` tool boundary). It catches any
    `decision` record — from any producer path — that reached the trajectory
    without a WHY. Returns a list of violation strings (empty = clean).

    Deliberately *not* implemented by tightening `events.validate_payload`
    (that would touch the frozen events contract); this is an additive audit
    over the record stream, leaving Façade/events v1.0 untouched.
    """
    import json

    violations: list[str] = []
    path = Path(trajectory_path)
    if not path.is_file():
        return [f"trajectory file not found: {path}"]

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                violations.append(f"line {lineno}: invalid JSON")
                continue
            if rec.get("kind") != "decision":
                continue
            data = rec.get("data", {}) or {}
            rationale = data.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                step = data.get("step", "?")
                violations.append(
                    f"line {lineno} (decision step {step}): missing/empty rationale"
                )
    return violations
