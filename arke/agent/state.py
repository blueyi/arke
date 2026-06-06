# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — OptimizationState (D8-F1.1).

Façade-side ground-truth container. The authoritative record of an
optimization session — strategy mutation log, V1/V2 results, checkpoints,
and budget. Lives outside the message log so it survives context
compression (arke-harness.md §8).

This module is part of the public Façade contract (arke-harness.md §3).
Substrate types (Decision, ScheduleIR) are borrowed read-only; Façade
holds the orchestration shell.

Design ref: docs/architecture/arke-harness.md §3 §8
Stage tracker: docs/phase1/stage8-plan.md D8-F1.1
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from arke.ir.schedule import ScheduleIR
from arke.ir.strategy import Decision


# ─── Result + Snapshot types ──────────────────────────────────────────────

@dataclass
class CompileResult:
    """Outcome of a compile_and_profile / verify_correctness call.

    All fields optional except `success` + `backend` — populated based on
    which validation tier ran (V0 static / V1 numeric / V2 GPU profile).
    """
    success: bool
    backend: str                        # "mock" | "triton" | "cuda" | ...
    correct: bool | None = None         # V1 numeric check vs reference
    max_diff: float | None = None       # V1: max abs delta vs reference
    latency_ms: float | None = None     # V2: median wall-clock (None on Mock)
    baseline_ratio: float | None = None # V2: latency_baseline / latency_arke
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"success": self.success, "backend": self.backend}
        for k in ("correct", "max_diff", "latency_ms", "baseline_ratio", "error"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


@dataclass
class Checkpoint:
    """Snapshot of state at a labelled moment."""
    label: str
    timestamp: float
    strategy_snapshot: dict[str, Any]   # ScheduleIR.to_dict()
    decision_log_snapshot: list[dict[str, Any]]
    best_result_snapshot: dict[str, Any] | None
    decision_count_at: int
    compile_count_at: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "timestamp": self.timestamp,
            "strategy": self.strategy_snapshot,
            "decision_log": self.decision_log_snapshot,
            "best_result": self.best_result_snapshot,
            "decision_count_at": self.decision_count_at,
            "compile_count_at": self.compile_count_at,
        }


# ─── Budget ────────────────────────────────────────────────────────────────

@dataclass
class OptimizationBudget:
    """Tracks decision + compile usage against caps.

    Defaults chosen per arke-harness.md §6 (decision = strategy mutation;
    compile = GPU call). Tunable per session via `OptimizationState`.
    """
    decision_max: int = 50
    compile_max: int = 30
    decisions_used: int = 0
    compiles_used: int = 0

    @property
    def decisions_remaining(self) -> int:
        return max(0, self.decision_max - self.decisions_used)

    @property
    def compiles_remaining(self) -> int:
        return max(0, self.compile_max - self.compiles_used)

    @property
    def exhausted(self) -> bool:
        return self.decisions_remaining == 0 and self.compiles_remaining == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_max": self.decision_max,
            "compile_max": self.compile_max,
            "decisions_used": self.decisions_used,
            "compiles_used": self.compiles_used,
        }


# ─── OptimizationState ─────────────────────────────────────────────────────

class OptimizationState:
    """Ground truth outside the message log (arke-harness.md §8).

    Owns:
      - `strategy`     : current ScheduleIR (mutated by apply_decision)
      - `decision_log` : full history of decisions applied
      - `compile_results` : V1/V2 result history
      - `best_result`  : best compile result tracked separately for finalization
      - `checkpoints`  : labelled snapshots (deep-copied) for rollback
      - `budget`       : decision + compile usage caps

    Façade tools 4 (`apply_decision`), 7 (`checkpoint`), 8 (`rollback`)
    are thin wrappers around methods on this class.
    """

    def __init__(
        self,
        strategy: ScheduleIR | None = None,
        budget: OptimizationBudget | None = None,
    ):
        self.strategy: ScheduleIR = strategy if strategy is not None else ScheduleIR()
        self.decision_log: list[Decision] = []
        self.compile_results: list[CompileResult] = []
        self.best_result: CompileResult | None = None
        self.checkpoints: dict[str, Checkpoint] = {}
        self.budget: OptimizationBudget = budget if budget is not None else OptimizationBudget()

    # ── Mutation methods (Façade tools 4/7/8) ────────────────────────────

    def apply_decision(self, decision: Decision) -> None:
        """Apply a decision: mutates strategy, appends log, advances budget.

        Raises:
            RuntimeError: if decision budget exhausted.
        """
        if self.budget.decisions_remaining <= 0:
            raise RuntimeError(
                f"Decision budget exhausted ({self.budget.decisions_used}/{self.budget.decision_max})"
            )
        # auto-assign step
        if decision.step == 0:
            decision.step = len(self.decision_log) + 1
        self.strategy.apply_decision(decision)
        self.decision_log.append(decision)
        self.budget.decisions_used += 1

    def record_compile(self, result: CompileResult) -> None:
        """Record a compile/profile/verify result + update best.

        Advances compile budget. Updates `best_result` if this result is
        better (lower latency_ms, ties broken by correctness then order).
        """
        if self.budget.compiles_remaining <= 0:
            raise RuntimeError(
                f"Compile budget exhausted ({self.budget.compiles_used}/{self.budget.compile_max})"
            )
        self.compile_results.append(result)
        self.budget.compiles_used += 1
        if result.success and result.correct is not False:
            if self.best_result is None:
                self.best_result = result
            elif (
                result.latency_ms is not None
                and self.best_result.latency_ms is not None
                and result.latency_ms < self.best_result.latency_ms
            ):
                self.best_result = result

    def checkpoint(self, label: str) -> Checkpoint:
        """Snapshot current state under `label`. Overwrites if exists.

        Free operation (no budget cost).
        """
        from arke.ir.strategy import _decision_to_dict  # local import to avoid cycle
        snap = Checkpoint(
            label=label,
            timestamp=time.time(),
            strategy_snapshot=self.strategy.to_dict(),
            decision_log_snapshot=[_decision_to_dict(d) for d in self.decision_log],
            best_result_snapshot=self.best_result.to_dict() if self.best_result else None,
            decision_count_at=self.budget.decisions_used,
            compile_count_at=self.budget.compiles_used,
        )
        self.checkpoints[label] = snap
        return snap

    def rollback(self, label: str) -> None:
        """Restore state from a previous checkpoint.

        Mutates `strategy`, `decision_log`, `best_result`, and budget
        counters. Compile_results history is preserved (audit trail).

        Raises:
            KeyError: if label not in checkpoints.
        """
        if label not in self.checkpoints:
            raise KeyError(
                f"Unknown checkpoint: {label!r}. Available: {list(self.checkpoints.keys())}"
            )
        snap = self.checkpoints[label]
        # Restore strategy via from_dict (deepcopy via roundtrip)
        self.strategy = ScheduleIR.from_dict(copy.deepcopy(snap.strategy_snapshot))
        # Restore decision log
        from arke.ir.strategy import _parse_decision
        self.decision_log = [_parse_decision(d) for d in snap.decision_log_snapshot]
        # Restore best_result
        if snap.best_result_snapshot is not None:
            br = snap.best_result_snapshot
            self.best_result = CompileResult(
                success=br.get("success", False),
                backend=br.get("backend", ""),
                correct=br.get("correct"),
                max_diff=br.get("max_diff"),
                latency_ms=br.get("latency_ms"),
                baseline_ratio=br.get("baseline_ratio"),
                error=br.get("error"),
                metadata=dict(br.get("metadata", {})),
            )
        else:
            self.best_result = None
        # Restore budget counters
        self.budget.decisions_used = snap.decision_count_at
        self.budget.compiles_used = snap.compile_count_at

    # ── Inspection ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for trajectory snapshot or debug dump."""
        from arke.ir.strategy import _decision_to_dict
        return {
            "strategy": self.strategy.to_dict(),
            "decision_log": [_decision_to_dict(d) for d in self.decision_log],
            "compile_results": [r.to_dict() for r in self.compile_results],
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "checkpoints": {k: v.to_dict() for k, v in self.checkpoints.items()},
            "budget": self.budget.to_dict(),
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"OptimizationState("
            f"decisions={len(self.decision_log)}/{self.budget.decision_max}, "
            f"compiles={len(self.compile_results)}/{self.budget.compile_max}, "
            f"checkpoints={len(self.checkpoints)}, "
            f"best_lat={self.best_result.latency_ms if self.best_result else None})"
        )
