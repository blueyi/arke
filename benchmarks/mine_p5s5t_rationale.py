#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mine the P5-S5-T live-run trajectories into the @rationale KB.

The P5-S5-T explore runs (benchmarks/results/phase5/s5/t/*/trajectory.json)
are the project's first corpus of LIVE-LLM, LLVM-backend, L3 (instruction-
level) optimization rationales measured against a real GPU with the
vs_default signal. Unlike the Phase-1 heuristic corpus (which dominates the
existing KB), these carry:

  - backend = "llvm" provenance,
  - real per-decision @rationale authored by the live agent,
  - the measured vs_default / latency outcome from compile_and_profile,
  - L3 decision kinds (block_threads / wmma_tile / pipeline_stages) plus the
    L1 kinds the agent tried.

The trajectory.json written by arke.agent.backends is a dict (OptimizeResult
.to_dict()), NOT the JSONL trajectory schema that mine_trajectory expects, so
this dedicated miner walks the `trajectory` list of tool records, pairs each
apply_decision (carrying rationale) with the NEXT compile_and_profile record's
measured outcome, and appends deduped RationaleEntry rows.

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    python -m benchmarks.mine_p5s5t_rationale [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arke.learn.rationale_kb import DEFAULT_KB_PATH, RationaleEntry, RationaleKB

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "benchmarks" / "results" / "phase5" / "s5" / "t"


def _rationale_text(r) -> str:
    if isinstance(r, dict):
        return str(r.get("text", "") or "")
    return str(r or "")


def mine_run(trajectory_json: Path) -> list[RationaleEntry]:
    """Mine one P5-S5-T run's trajectory.json (dict form) -> entries."""
    obj = json.loads(trajectory_json.read_text())
    traj = obj.get("trajectory") or []
    # op context: the run dir is {op}_{shape}
    op = trajectory_json.parent.name.split("_")[0]

    entries: list[RationaleEntry] = []
    pending: RationaleEntry | None = None
    for rec in traj:
        tool = rec.get("tool")
        if tool == "apply_decision":
            p = rec.get("params", {}) or {}
            rat = _rationale_text(p.get("rationale"))
            if not rat:
                pending = None
                continue
            pending = RationaleEntry(
                op=op,
                decision_kind=str(p.get("kind") or "decision"),
                params=dict(p.get("params") or {}),
                rationale=rat,
                source=f"p5s5t/{trajectory_json.parent.name}",
                backend="llvm",
                phase=5,
            )
            entries.append(pending)
        elif tool == "compile_and_profile" and pending is not None:
            data = (rec.get("result") or {}).get("data", {}) or {}
            if pending.correct is None and "correct" in data:
                pending.correct = bool(data["correct"])
            # Prefer vs_default (the P5-S5-T signal); fall back to baseline_ratio.
            ratio = data.get("vs_default")
            if ratio is None:
                ratio = data.get("baseline_ratio")
            if ratio is not None and pending.baseline_ratio is None:
                pending.baseline_ratio = float(ratio)
            if data.get("latency_ms") is not None and pending.latency_ms is None:
                pending.latency_ms = float(data["latency_ms"])
            pending = None  # close the pairing window
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mine P5-S5-T live trajectories into the @rationale KB")
    ap.add_argument("--dry-run", action="store_true", help="report counts, don't write")
    ap.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    args = ap.parse_args(argv)

    all_entries: list[RationaleEntry] = []
    runs = sorted(p for p in RUNS_DIR.iterdir() if (p / "trajectory.json").exists()) if RUNS_DIR.exists() else []
    for run in runs:
        ents = mine_run(run / "trajectory.json")
        all_entries.extend(ents)
        print(f"  {run.name}: {len(ents)} rationale entries", file=sys.stderr)

    print(f"mined {len(all_entries)} entries from {len(runs)} P5-S5-T runs", file=sys.stderr)
    if args.dry_run:
        return 0

    kb = RationaleKB(args.kb)
    added = kb.add_entries(all_entries)
    print(f"KB: +{added} new (deduped), total now {kb.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
