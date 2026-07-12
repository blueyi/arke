# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Learn — D3 RL dataset extraction from optimization trajectories.

Leon-approved 2026-07-12 (D3=yes): mine Arke optimization trajectories into an
agentic-RL training corpus, in the style of CUDA Agent (ByteDance/Tsinghua,
arXiv 2602.24286) which trains a policy on discrete robust-reward-labeled
trajectories.

This is the structural advantage Arke has over test-time-only kernel-gen
systems: every optimization run already emits a frozen v1.0 trajectory
(decision → verify → profile records). This module turns those into
(prompt, action, reward) samples the way an RL/SFT pipeline consumes them.

Two sample shapes are produced:
  1. **Step samples** — one per decision: the state (op + shape + prior
     decisions), the action taken (decision kind + params + rationale), and the
     robust-reward earned by the resulting kernel. Suitable for step-level RL
     (PPO/GRPO advantage estimation).
  2. **Trajectory samples** — the full decision sequence + final robust-reward,
     suitable for trajectory-level ranking / preference pairs (best-of-N,
     least-to-most ordering à la Sakana).

Design principle (from harness-build-vs-reuse-2026-07.md §3): the reward is the
discrete robust_reward tier (anti-reward-hacking), NOT raw continuous speedup.
Correctness dominates — an incorrect kernel earns -1 regardless of speed.

This module is a pure Substrate reader over trajectory JSONL — it does not
touch the frozen Façade or event schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arke.agent.verification import robust_reward


@dataclass
class RLStepSample:
    """One (state, action, reward) RL training sample from a decision step."""
    op: str
    shape: dict[str, Any]
    prior_decisions: list[dict[str, Any]]   # decisions taken before this one
    action: dict[str, Any]                   # {kind, params, rationale}
    reward: int                              # robust_reward tier at this step
    correct: bool | None
    baseline_ratio: float | None
    latency_ms: float | None
    backend: str | None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "step",
            "op": self.op,
            "shape": self.shape,
            "prior_decisions": self.prior_decisions,
            "action": self.action,
            "reward": self.reward,
            "correct": self.correct,
            "baseline_ratio": self.baseline_ratio,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "source": self.source,
        }


@dataclass
class RLTrajectorySample:
    """A full trajectory + its final robust-reward, for trajectory-level RL."""
    op: str
    shape: dict[str, Any]
    decisions: list[dict[str, Any]]
    final_reward: int
    final_correct: bool | None
    final_baseline_ratio: float | None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "trajectory",
            "op": self.op,
            "shape": self.shape,
            "decisions": self.decisions,
            "final_reward": self.final_reward,
            "final_correct": self.final_correct,
            "final_baseline_ratio": self.final_baseline_ratio,
            "source": self.source,
        }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _reward_from_profile(data: dict[str, Any]) -> int:
    """Compute robust reward from a profile record's outcome fields.

    Prefers a pre-recorded 'robust_reward' (compile_and_profile now emits it);
    falls back to recomputing from correct + baseline_ratio.
    """
    if "robust_reward" in data and data["robust_reward"] is not None:
        return int(data["robust_reward"])
    correct = data.get("correct")
    ratio = data.get("baseline_ratio")
    return int(robust_reward(correct=correct, eager_ratio=ratio, strong_ratio=ratio))


def extract_rl_samples(
    path: str | Path,
) -> tuple[list[RLStepSample], RLTrajectorySample | None]:
    """Extract step + trajectory RL samples from one trajectory JSONL file.

    Walks the record stream, tracking the running decision list. Each decision
    is paired with the NEXT profile/verify outcome to assign its reward (the
    kernel state after applying that decision). The final profile determines
    the trajectory-level reward.
    """
    path = Path(path)
    records = _load_records(path)
    if not records:
        return [], None

    # Header → op + shape context.
    op_ctx = ""
    shape_ctx: dict[str, Any] = {}
    for r in records:
        if r.get("kind") == "header":
            d = r.get("data", {}) or {}
            op_ctx = d.get("op") or d.get("kernel_id") or ""
            shape_ctx = d.get("shape") or d.get("shapes") or {}
            break

    steps: list[RLStepSample] = []
    prior: list[dict[str, Any]] = []
    pending_action: dict[str, Any] | None = None
    last_reward = int(robust_reward(correct=None, eager_ratio=None, strong_ratio=None))
    last_correct: bool | None = None
    last_ratio: float | None = None
    all_decisions: list[dict[str, Any]] = []

    for r in records:
        kind = r.get("kind")
        data = r.get("data", {}) or {}

        if kind == "decision":
            dkind = data.get("kind") or (data.get("decision", {}) or {}).get("kind") or "decision"
            params = data.get("params") or (data.get("decision", {}) or {}).get("params") or {}
            rationale = (
                data.get("rationale")
                or (data.get("decision", {}) or {}).get("rationale")
                or ""
            )
            action = {"kind": dkind, "params": params, "rationale": rationale}
            pending_action = action
            all_decisions.append(action)

        elif kind in ("profile", "verify"):
            reward = _reward_from_profile(data)
            correct = data.get("correct")
            ratio = data.get("baseline_ratio")
            backend = data.get("backend")
            latency = data.get("latency_ms")
            last_reward, last_correct, last_ratio = reward, correct, ratio
            if pending_action is not None:
                steps.append(RLStepSample(
                    op=data.get("op") or op_ctx or "unknown",
                    shape=shape_ctx,
                    prior_decisions=list(prior),
                    action=pending_action,
                    reward=reward,
                    correct=correct,
                    baseline_ratio=ratio,
                    latency_ms=latency,
                    backend=backend,
                    source=str(path),
                ))
                prior.append(pending_action)
                pending_action = None

    traj = RLTrajectorySample(
        op=op_ctx or "unknown",
        shape=shape_ctx,
        decisions=all_decisions,
        final_reward=last_reward,
        final_correct=last_correct,
        final_baseline_ratio=last_ratio,
        source=str(path),
    )
    return steps, traj


def build_rl_dataset(
    trajectory_paths: list[str | Path],
    out_path: str | Path,
) -> dict[str, int]:
    """Mine multiple trajectories into an RL dataset JSONL.

    Writes step + trajectory samples to ``out_path`` (JSONL). Returns counts.
    Idempotent-append is NOT enforced here (RL datasets are typically rebuilt
    from scratch per training run); the caller controls overwrite vs append.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = 0
    n_traj = 0
    with out_path.open("w") as f:
        for p in trajectory_paths:
            steps, traj = extract_rl_samples(p)
            for s in steps:
                f.write(json.dumps(s.to_dict(), default=str) + "\n")
                n_steps += 1
            if traj is not None:
                f.write(json.dumps(traj.to_dict(), default=str) + "\n")
                n_traj += 1
    return {"steps": n_steps, "trajectories": n_traj,
            "files": len(trajectory_paths)}


def reward_histogram(samples: list[RLStepSample]) -> dict[int, int]:
    """Distribution of robust-reward tiers across step samples (for RL diagnostics)."""
    hist: dict[int, int] = {}
    for s in samples:
        hist[s.reward] = hist.get(s.reward, 0) + 1
    return hist


__all__ = [
    "RLStepSample",
    "RLTrajectorySample",
    "extract_rl_samples",
    "build_rl_dataset",
    "reward_histogram",
]
