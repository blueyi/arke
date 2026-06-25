# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""@rationale Knowledge Base — distill decision rationales + measured outcomes.

The @rationale KB (G9[3], D8-A4) is Phase 1's accumulated optimization
experience: each entry pairs an optimization **decision** (op, kind, params,
the LLM/heuristic's @rationale for WHY) with the **measured outcome** that
decision produced (correctness, latency, baseline_ratio). This is the
human-experience → LLM-optimization feedback loop the project thesis calls for
(see SOUL.md "@rationale").

Sources:
  - live + heuristic optimization trajectories (``*.jsonl`` with ``decision``
    + ``profile`` records, written by ``arke.learn.trajectory.TrajectoryWriter``)

The KB is an append-only JSONL at ``data/rationale_kb.jsonl``. Each line is one
``RationaleEntry``. The miner is idempotent on (op, kind, params_hash,
rationale) — re-mining the same trajectories does not duplicate entries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_KB_PATH = "data/rationale_kb.jsonl"


@dataclass
class RationaleEntry:
    """One distilled (decision → rationale → outcome) record."""

    op: str
    decision_kind: str
    params: dict[str, Any]
    rationale: str
    # Measured outcome (best-effort; None when not paired with a profile).
    correct: bool | None = None
    baseline_ratio: float | None = None
    latency_ms: float | None = None
    backend: str | None = None
    source: str = ""           # trajectory file / provenance
    phase: int = 1

    def key(self) -> str:
        """Stable dedupe key over (op, kind, params, rationale)."""
        blob = json.dumps(
            {"op": self.op, "k": self.decision_kind, "p": self.params, "r": self.rationale},
            sort_keys=True, default=str,
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key()
        return d


@dataclass
class RationaleKB:
    """Append-only @rationale knowledge base."""

    path: Path = field(default_factory=lambda: Path(DEFAULT_KB_PATH))

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def _existing_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        keys: set[str] = set()
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line).get("key", ""))
            except json.JSONDecodeError:
                continue
        return keys

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for ln in self.path.read_text().splitlines() if ln.strip())

    def add_entries(self, entries: list[RationaleEntry]) -> int:
        """Append non-duplicate entries; return number actually written."""
        existing = self._existing_keys()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.path.open("a") as f:
            for e in entries:
                if e.key() in existing:
                    continue
                f.write(json.dumps(e.to_dict(), default=str) + "\n")
                existing.add(e.key())
                written += 1
        return written


def mine_trajectory(path: str | Path) -> list[RationaleEntry]:
    """Extract RationaleEntry rows from one trajectory JSONL file.

    Pairs each ``decision`` record carrying a non-empty rationale with the
    next ``profile``/``compile`` record's measured outcome (op-scoped). When a
    decision has no following measurement, the entry is still emitted with
    outcome fields left None (the rationale itself is the value).
    """
    path = Path(path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Header gives op context if present.
    op_ctx = ""
    for r in records:
        if r.get("kind") == "header":
            op_ctx = (r.get("data", {}) or {}).get("op") or (r.get("data", {}) or {}).get("kernel_id") or ""
            break

    entries: list[RationaleEntry] = []
    pending: RationaleEntry | None = None
    for r in records:
        kind = r.get("kind")
        data = r.get("data", {}) or {}
        if kind == "decision":
            rationale = (
                data.get("rationale")
                or (data.get("decision", {}) or {}).get("rationale")
                or ""
            )
            if not rationale:
                continue
            dkind = data.get("kind") or (data.get("decision", {}) or {}).get("kind") or "decision"
            params = data.get("params") or (data.get("decision", {}) or {}).get("params") or {}
            op = data.get("op") or op_ctx or "unknown"
            pending = RationaleEntry(
                op=op, decision_kind=str(dkind), params=dict(params),
                rationale=str(rationale), source=str(path.name),
            )
            entries.append(pending)
        elif kind in ("profile", "compile") and pending is not None:
            # Overlay measured outcome onto the most recent pending entry.
            if "correct" in data and pending.correct is None:
                pending.correct = bool(data["correct"])
            # Trajectory v1 profile records use ``vs_baseline``; the live
            # measurement path uses ``baseline_ratio`` — accept either.
            ratio = data.get("baseline_ratio")
            if ratio is None:
                ratio = data.get("vs_baseline")
            if ratio is not None and pending.baseline_ratio is None:
                pending.baseline_ratio = float(ratio)
            if data.get("latency_ms") is not None and pending.latency_ms is None:
                pending.latency_ms = float(data["latency_ms"])
            if data.get("backend") and pending.backend is None:
                pending.backend = str(data["backend"])
            if kind == "profile":
                pending = None  # close the pairing window after a profile

    return entries


def mine_strategy_json(
    strategy_path: str | Path,
    trajectory_path: str | Path | None = None,
) -> list[RationaleEntry]:
    """Extract RationaleEntry rows from an ``arke optimize`` strategy.json.

    The heuristic/auto optimizer writes its decisions (each with a
    ``@rationale``) to ``strategy.json``; the measured outcome lands in the
    sibling ``trajectory.jsonl`` profile records. This pairs every decision
    with the run's best (max) ``vs_baseline`` ratio so the KB records which
    rationale-backed strategies produced wins.

    Rationale may be a plain string or a ``{"text": ..., "lang": ...}`` dict
    (the StrategyIR Rationale shape) — both are normalized to text.
    """
    strategy_path = Path(strategy_path)
    if not strategy_path.exists():
        return []
    try:
        strat = json.loads(strategy_path.read_text())
    except json.JSONDecodeError:
        return []

    op = strat.get("kernel_id") or "unknown"

    # Best measured ratio from the trajectory (if present).
    best_ratio: float | None = None
    if trajectory_path is None:
        trajectory_path = strategy_path.parent / "trajectory.jsonl"
    trajectory_path = Path(trajectory_path)
    if trajectory_path.exists():
        for line in trajectory_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "profile":
                d = rec.get("data", {}) or {}
                ratio = d.get("baseline_ratio")
                if ratio is None:
                    ratio = d.get("vs_baseline")
                if ratio is not None:
                    best_ratio = max(best_ratio or 0.0, float(ratio))

    def _rationale_text(r: Any) -> str:
        if isinstance(r, dict):
            return str(r.get("text", "")).strip()
        return str(r or "").strip()

    entries: list[RationaleEntry] = []
    for d in strat.get("decisions", []):
        rationale = _rationale_text(d.get("rationale"))
        if not rationale:
            continue
        entries.append(RationaleEntry(
            op=op,
            decision_kind=str(d.get("kind", "decision")),
            params=dict(d.get("params", {})),
            rationale=rationale,
            baseline_ratio=best_ratio,
            source=str(strategy_path.parent.name),
        ))
    return entries


def mine_directory(root: str | Path, kb_path: str | Path = DEFAULT_KB_PATH) -> dict[str, int]:
    """Mine all ``*.jsonl`` trajectories under ``root`` into the KB.

    Returns {"trajectories": N, "entries_found": M, "entries_written": W,
             "kb_total": T}.
    """
    root = Path(root)
    kb = RationaleKB(Path(kb_path))
    files = sorted(root.rglob("*.jsonl"))
    found: list[RationaleEntry] = []
    for f in files:
        found.extend(mine_trajectory(f))
    written = kb.add_entries(found)
    return {
        "trajectories": len(files),
        "entries_found": len(found),
        "entries_written": written,
        "kb_total": kb.count(),
    }
