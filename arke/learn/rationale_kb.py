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
  - **curated** lead-engineer patterns (``curated/<slug>`` provenance): the
    *human-experience* write half of the loop. Cross-op generalizations a dev
    discovered by hand (e.g. an occupancy rule holding across several ops) that
    no per-trajectory miner can synthesize — see ``curated_pattern`` /
    ``mine_curated`` below and ``benchmarks/curated_patterns.py``.

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

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def recall(
        self,
        op: str,
        *,
        decision_kind: str | None = None,
        top_k: int = 3,
        min_ratio: float | None = None,
    ) -> list[RationalePrior]:
        """Read side of the @rationale loop: recall priors for an op.

        Returns up to ``top_k`` distilled priors matching ``op`` (op-name
        drift normalized), optionally filtered to one ``decision_kind`` and to
        entries whose measured ``baseline_ratio >= min_ratio``. Priors are
        ranked best-outcome-first: entries with a measured ``baseline_ratio``
        outrank unmeasured ones (ratio treated as -inf when absent), so the
        Agent sees the decisions that actually *worked* on this op first.

        This is what makes the 390-entry KB load-bearing instead of
        write-only — it feeds accumulated optimization experience back into
        the Agent's next decision, closing the human/heuristic-experience →
        LLM-optimization feedback loop the @rationale system exists for
        (LT-7 / SOUL.md "@rationale").
        """
        want_op = _normalize_op(op)
        want_kind = decision_kind
        matched: list[RationalePrior] = []
        for e in self._load_entries():
            if _normalize_op(str(e.get("op", ""))) != want_op:
                continue
            if want_kind is not None and e.get("decision_kind") != want_kind:
                continue
            ratio = e.get("baseline_ratio")
            if min_ratio is not None and (ratio is None or float(ratio) < min_ratio):
                continue
            matched.append(RationalePrior(
                op=str(e.get("op", want_op)),
                decision_kind=str(e.get("decision_kind", "")),
                params=dict(e.get("params", {}) or {}),
                rationale=str(e.get("rationale", "")),
                baseline_ratio=(float(ratio) if ratio is not None else None),
                correct=e.get("correct"),
                backend=e.get("backend"),
            ))
        # Rank best-measured-outcome first; unmeasured sink to the bottom.
        matched.sort(
            key=lambda p: (p.baseline_ratio is not None, p.baseline_ratio or 0.0),
            reverse=True,
        )
        # Dedupe identical (kind, params, rationale) priors, keep best-ranked.
        seen: set[str] = set()
        out: list[RationalePrior] = []
        for p in matched:
            sig = json.dumps(
                {"k": p.decision_kind, "p": p.params, "r": p.rationale},
                sort_keys=True, default=str,
            )
            if sig in seen:
                continue
            seen.add(sig)
            out.append(p)
            if len(out) >= top_k:
                break
        return out


def _normalize_op(op: str) -> str:
    """Bridge KB↔registry op-name drift.

    The KB stores some ops with a ``_kernel`` suffix (``relu_kernel``,
    ``grouped_matmul_kernel``, ``add_kernel``) while the runtime registry
    (``benchmarks.op_registry`` / ``arke.ir.ops.registry``) uses the bare
    name (``relu`` / ``grouped_matmul`` / ``add``). Recall must match across
    that gap or ~half the corpus is invisible to the Agent.
    """
    op = (op or "").strip()
    if op.endswith("_kernel"):
        op = op[: -len("_kernel")]
    return op


@dataclass
class RationalePrior:
    """A compact recalled prior surfaced to the Agent at decision time.

    Distilled from a ``RationaleEntry`` for the read side of the @rationale
    loop: what a prior decision of this (op, kind) did and what it measured.
    """

    op: str
    decision_kind: str
    params: dict[str, Any]
    rationale: str
    baseline_ratio: float | None = None
    correct: bool | None = None
    backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


# --------------------------------------------------------------------------- #
# Curated write channel — the human-experience half of the @rationale loop.    #
# --------------------------------------------------------------------------- #

def curated_pattern(
    op: str,
    decision_kind: str,
    params: dict[str, Any],
    rationale: str,
    *,
    baseline_ratio: float | None = None,
    correct: bool | None = None,
    backend: str | None = None,
    slug: str = "pattern",
    phase: int = 4,
) -> RationaleEntry:
    """Author one lead-engineer-discovered prior as a ``RationaleEntry``.

    This is the write side the auto-miners cannot cover: a cross-op
    generalization a human found by hand (e.g. a hardware occupancy rule that
    holds across several ops), distilled into the SAME entry shape the miners
    produce so ``recall`` surfaces it to the Agent identically.

    Provenance is honest and auditable: ``source`` is stamped
    ``curated/<slug>`` so a hand-authored prior is never mistaken for a
    measured trajectory. When a ``baseline_ratio`` is supplied it MUST be a
    real measured number (the dev's own median-of-N A/B result) — the KB does
    not fabricate outcomes.
    """
    return RationaleEntry(
        op=op,
        decision_kind=decision_kind,
        params=dict(params),
        rationale=rationale,
        correct=correct,
        baseline_ratio=baseline_ratio,
        backend=backend,
        source=f"curated/{slug}",
        phase=phase,
    )


def mine_curated(
    patterns: list[RationaleEntry],
    kb_path: str | Path = DEFAULT_KB_PATH,
) -> dict[str, int]:
    """Append curated patterns into the KB (idempotent, deduped).

    Returns {"entries_found": M, "entries_written": W, "kb_total": T}. Re-
    seeding the same patterns writes nothing new (dedupe on entry ``key()``).
    """
    kb = RationaleKB(Path(kb_path))
    written = kb.add_entries(patterns)
    return {
        "entries_found": len(patterns),
        "entries_written": written,
        "kb_total": kb.count(),
    }
