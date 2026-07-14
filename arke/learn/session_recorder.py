# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Learn — multi-round optimization session recorder (Phase 5).

The minimal roundtrip closure (``write_header → write_decision →
write_profile → extract_rl_samples``) proves the write→read→RL wire works,
but it records a *single* decision→outcome step. A real autotuning session
is **multi-round**: the agent proposes a decision, compiles + profiles it,
observes the outcome, and proposes the *next* decision conditioned on what
it just learned — repeated until a budget or target is hit.

This module is the reusable, tested component that records such a session to
a v1.0 ``trajectory.jsonl`` so the RL miner (:mod:`arke.learn.rl_dataset`)
can extract step-wise reward + discounted return-to-go from it.

Design
------
* **Thin over the contract.** Every line is written through
  :class:`arke.learn.trajectory.TrajectoryWriter`, so the emitted file is a
  strict v1.0 trajectory — the same artifact the live/heuristic harness
  produces. No new wire format.
* **Reward at record time.** Each round's profile record carries a
  pre-computed ``robust_reward`` (via :func:`arke.agent.verification.robust_reward`),
  so the miner reads the reward directly instead of recomputing it.
* **Cycle boundaries.** Each round emits ``decision → compile → profile →
  adjust`` so the mined trajectory is a faithful multi-round episode with
  cycle markers, mirroring the heuristic ``optimize()`` cycle order.

This is a pure Substrate writer — it does not touch the frozen Façade tool
signatures or the event schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arke.agent.verification import robust_reward
from arke.learn.trajectory import TrajectoryWriter


@dataclass
class RoundOutcome:
    """The measured result of compiling + profiling one round's decision."""

    correct: bool | None
    eager_ratio: float | None = None
    strong_ratio: float | None = None
    latency_ms: float | None = None
    backend: str = "mock"
    bottleneck: str = ""

    @property
    def baseline_ratio(self) -> float | None:
        """The ratio surfaced into the profile record.

        Prefers the strong-baseline ratio (the stricter reference); falls back
        to the eager ratio when only that is available.
        """
        return self.strong_ratio if self.strong_ratio is not None else self.eager_ratio

    def reward(self) -> int:
        """Discrete robust-reward tier for this outcome (D2 anti-hacking)."""
        return int(robust_reward(
            correct=self.correct,
            eager_ratio=self.eager_ratio,
            strong_ratio=self.strong_ratio,
        ))


@dataclass
class SessionRound:
    """One recorded round: the decision taken and the outcome it produced."""

    kind: str
    params: dict[str, Any]
    rationale: str
    outcome: RoundOutcome
    cycle: int


@dataclass
class SessionRecorder:
    """Records a full multi-round optimization session to a v1.0 trajectory.

    Usage::

        rec = SessionRecorder(path, op="matmul", shape={"M": 512, ...})
        rec.start()
        rec.record_round(
            kind="tile", params={"loop": "M", "factors": [64]},
            rationale="bigger tile amortizes",
            outcome=RoundOutcome(correct=True, eager_ratio=1.2, strong_ratio=1.05),
        )
        ...
        rec.finish()

    Each ``record_round`` emits the ``decision → compile → profile → adjust``
    burst for that round. ``finish`` closes the trajectory with a ``done``
    record carrying the best reward and round count.
    """

    path: str | Path
    op: str
    shape: dict[str, Any] = field(default_factory=dict)
    target_hw: str = "nvidia_ampere"
    mode: str = "compile"

    _writer: TrajectoryWriter | None = field(default=None, init=False, repr=False)
    _rounds: list[SessionRound] = field(default_factory=list, init=False, repr=False)
    _cycle: int = field(default=0, init=False, repr=False)
    _decision_count: int = field(default=0, init=False, repr=False)

    # ── lifecycle ──────────────────────────────────────────────────
    def start(self) -> SessionRecorder:
        """Open the trajectory and write the session header."""
        self._writer = TrajectoryWriter(self.path)
        self._writer.write_header({
            "kernel_id": self.op,
            "target_hw": self.target_hw,
            "mode": self.mode,
            "required_cycle_order": ["compile", "profile", "adjust"],
            "semantic_ir": {"op": self.op, "shape": self.shape},
        })
        # The header contract keeps `op`/`shape` in semantic_ir; the RL miner
        # reads op from kernel_id and shape from semantic_ir.shape, so surface
        # both at the top level of the header data too for a robust read.
        return self

    def record_round(
        self,
        *,
        kind: str,
        params: dict[str, Any],
        rationale: str,
        outcome: RoundOutcome,
    ) -> SessionRound:
        """Record one round: decision → compile → profile → adjust.

        Returns the :class:`SessionRound` captured (with its computed reward
        available via ``round.outcome.reward()``).
        """
        if self._writer is None:
            raise RuntimeError("SessionRecorder.start() must be called first")
        if not rationale or not rationale.strip():
            raise ValueError(
                f"decision {kind!r} requires a non-empty rationale "
                "(A5 @rationale contract)"
            )

        self._cycle += 1
        cycle = self._cycle
        w = self._writer

        # 1. decision record (carries op + shape context for step-sample state)
        w.write_decision({
            "kind": kind,
            "params": dict(params),
            "rationale": rationale,
            "op": self.op,
            "shape": self.shape,
            "cycle": cycle,
        })
        before = self._decision_count
        self._decision_count += 1

        # 2. compile record — the build attempt for this decision.
        compile_ok = outcome.correct is not False
        w.write_compile({
            "backend": outcome.backend,
            "success": bool(compile_ok),
            "cycle": cycle,
            "decision_count": self._decision_count,
        })

        # 3. profile record — the measured outcome + pre-computed reward.
        reward = outcome.reward()
        w.write_profile({
            "latency_ms": float(outcome.latency_ms or 0.0),
            "vs_baseline": float(outcome.baseline_ratio or 0.0),
            "baseline_ratio": outcome.baseline_ratio,
            "baseline_name": "strong" if outcome.strong_ratio is not None else "eager",
            "correct": outcome.correct,
            "robust_reward": reward,
            "backend": outcome.backend,
            "bottleneck": outcome.bottleneck,
            "op": self.op,
            "cycle": cycle,
            "source": "session_recorder",
        })

        # 4. adjust record — closes the cycle boundary.
        w.write_adjust({
            "cycle": cycle,
            "decisions_before": before,
            "decisions_after": self._decision_count,
            "changed": True,
            "bottleneck": outcome.bottleneck,
        })

        rnd = SessionRound(
            kind=kind, params=dict(params), rationale=rationale,
            outcome=outcome, cycle=cycle,
        )
        self._rounds.append(rnd)
        return rnd

    def finish(self, *, termination: str = "target_reached") -> dict[str, Any]:
        """Write the terminal ``done`` record and close the trajectory.

        Returns a summary dict (best reward, final reward, round count).
        """
        if self._writer is None:
            raise RuntimeError("SessionRecorder.start() must be called first")

        rewards = [r.outcome.reward() for r in self._rounds]
        best_reward = max(rewards) if rewards else 0
        final_reward = rewards[-1] if rewards else 0
        best_ratio = max(
            (r.outcome.baseline_ratio for r in self._rounds
             if r.outcome.baseline_ratio is not None),
            default=0.0,
        )

        self._writer.write_done({
            "final_score": float(best_ratio),
            "decisions": self._decision_count,
            "compiles": len(self._rounds),
            "termination": termination,
            "best_reward": best_reward,
            "final_reward": final_reward,
        })
        self._writer.close()
        self._writer = None

        return {
            "op": self.op,
            "rounds": len(self._rounds),
            "best_reward": best_reward,
            "final_reward": final_reward,
            "best_baseline_ratio": best_ratio,
            "rewards": rewards,
            "path": str(self.path),
        }

    # ── context manager sugar ──────────────────────────────────────
    def __enter__(self) -> SessionRecorder:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        if self._writer is not None:
            self.finish(termination="context_exit")


__all__ = ["RoundOutcome", "SessionRound", "SessionRecorder"]
