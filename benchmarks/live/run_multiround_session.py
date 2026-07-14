#!/usr/bin/env python
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end multi-round optimization session → mineable RL corpus.

Drives a real *multi-round* optimization session (a decision policy proposes
one decision per round, each is "compiled+profiled" to an outcome, and the
next decision is conditioned on the running state), records it to a v1.0
``trajectory.jsonl`` via :class:`arke.learn.session_recorder.SessionRecorder`,
then mines the trajectory into an RL corpus with step-wise reward and
discounted return-to-go.

By default it uses a deterministic, monotonically-improving mock policy so the
script is reproducible and requires no LLM or GPU — this is the closed-loop
proof that the multi-round write→read→RL pipeline works. A live-LLM/GPU policy
plugs into the same recorder by swapping ``mock_policy`` for a real one.

Run::

    python -m benchmarks.live.run_multiround_session --out /tmp/mr_session

Outputs, under ``--out``:
    matmul.trajectory.jsonl / softmax.trajectory.jsonl   (per-op episodes)
    rl_corpus.jsonl                                       (mined step+traj samples)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arke.learn.rl_dataset import (
    build_rl_dataset,
    extract_rl_samples,
    reward_histogram,
)
from arke.learn.rl_quality import quality_gate
from arke.learn.session_recorder import RoundOutcome, SessionRecorder


# ── deterministic multi-round policy ────────────────────────────────
# A scripted episode: a sequence of (decision, outcome) rounds that mimics a
# real autotuning trajectory — start correct-but-slow, tighten tiling, then
# beat eager and finally the strong baseline. Reproducible; no LLM/GPU.
def mock_episode(op: str) -> list[dict[str, Any]]:
    """Return a scripted improving multi-round episode for ``op``."""
    if op == "matmul":
        return [
            {"kind": "tile", "params": {"loop": "M", "factors": [32]},
             "rationale": "seed with a small output tile to establish correctness",
             "outcome": RoundOutcome(correct=True, eager_ratio=0.9, strong_ratio=0.6,
                                     latency_ms=0.30, bottleneck="memory_bandwidth")},
            {"kind": "tile", "params": {"loop": "M", "factors": [64]},
             "rationale": "larger tile amortizes shared-memory traffic, beats eager",
             "outcome": RoundOutcome(correct=True, eager_ratio=1.25, strong_ratio=0.95,
                                     latency_ms=0.20, bottleneck="shared_memory_pressure")},
            {"kind": "compute", "params": {"warps": 8, "num_stages": 3},
             "rationale": "deeper pipeline + more warps saturates tensor cores, beats cuBLAS",
             "outcome": RoundOutcome(correct=True, eager_ratio=1.6, strong_ratio=1.12,
                                     latency_ms=0.15, bottleneck="none")},
        ]
    if op == "softmax":
        return [
            {"kind": "tile", "params": {"loop": "N", "factors": [128]},
             "rationale": "row-block tile for coalesced reduction loads",
             "outcome": RoundOutcome(correct=True, eager_ratio=0.95, strong_ratio=0.8,
                                     latency_ms=0.12, bottleneck="memory_bandwidth")},
            {"kind": "vectorize", "params": {"loop": "N", "width": 4},
             "rationale": "vectorized loads lift memory throughput above eager",
             "outcome": RoundOutcome(correct=True, eager_ratio=1.3, strong_ratio=1.05,
                                     latency_ms=0.08, bottleneck="none")},
        ]
    raise ValueError(f"no scripted episode for op {op!r}")


_SHAPES: dict[str, dict[str, Any]] = {
    "matmul": {"M": 512, "N": 512, "K": 512},
    "softmax": {"rows": 64, "cols": 4096},
}


def run_session(op: str, out_dir: Path) -> dict[str, Any]:
    """Record one multi-round session for ``op`` and return its summary."""
    traj_path = out_dir / f"{op}.trajectory.jsonl"
    rec = SessionRecorder(traj_path, op=op, shape=_SHAPES.get(op, {}))
    rec.start()
    for round_spec in mock_episode(op):
        rec.record_round(
            kind=round_spec["kind"],
            params=round_spec["params"],
            rationale=round_spec["rationale"],
            outcome=round_spec["outcome"],
        )
    summary = rec.finish()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmarks/results/phase5/multiround",
                    help="output directory for trajectories + corpus")
    ap.add_argument("--ops", nargs="+", default=["matmul", "softmax"])
    ap.add_argument("--discount", type=float, default=0.95)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== recording {len(args.ops)} multi-round session(s) ===")
    traj_paths: list[str | Path] = []
    for op in args.ops:
        summary = run_session(op, out_dir)
        traj_paths.append(Path(summary["path"]))
        print(f"  {op}: {summary['rounds']} rounds, "
              f"rewards={summary['rewards']}, best={summary['best_reward']}")

    # Mine the corpus with step-wise reward + discounted return-to-go.
    corpus = out_dir / "rl_corpus.jsonl"
    counts = build_rl_dataset(traj_paths, corpus, discount=args.discount)
    print(f"=== mined RL corpus (γ={args.discount}): {counts} ===")

    # Diagnostics: reward histogram + per-op returns.
    all_steps = []
    for p in traj_paths:
        steps, traj = extract_rl_samples(p, discount=args.discount)
        all_steps += steps
        if traj is None:
            continue
        print(f"  {traj.op}: step_rewards={traj.step_rewards} "
              f"discounted_return(G0)={traj.discounted_return} "
              f"num_steps={traj.num_steps}")
        for s in steps:
            print(f"      step[{s.step_index}] {s.action['kind']:9s} "
                  f"reward={s.reward:+d} delta={s.reward_delta:+d} "
                  f"return_to_go={s.return_to_go}")
    print(f"  reward histogram (all steps): {reward_histogram(all_steps)}")

    # Quality gate over the mined corpus.
    report = quality_gate(corpus, min_beat=1)
    print("=== corpus quality gate ===")
    print(report.summary())

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
