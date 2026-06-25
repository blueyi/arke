#!/usr/bin/env python
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the @rationale KB (G9[3]/D8-A4) by optimizing all .ak examples.

Runs ``arke optimize --cycles 3`` over every ``examples/operators/*.ak``,
writing strategies + trajectories, then mines them into
``data/rationale_kb.jsonl`` via ``arke.learn.rationale_kb.mine_strategy_json``.

The heuristic optimizer path is deterministic + fast (no live-LLM cost) and
emits a @rationale on every decision; pairing with the run's best measured
``vs_baseline`` records which rationale-backed strategies produced wins.

Idempotent: re-running dedupes on (op, kind, params, rationale). Usage:

    python -m benchmarks.build_rationale_kb            # → data/rationale_kb.jsonl
    python -m benchmarks.build_rationale_kb --kb other.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

from arke.learn.rationale_kb import RationaleKB, mine_strategy_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the @rationale knowledge base")
    ap.add_argument("--kb", default="data/rationale_kb.jsonl", help="KB output path")
    ap.add_argument("--examples", default="examples/operators/*.ak", help="glob of .ak files")
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("GEMS_VENDOR", "nvidia")
    examples = sorted(glob.glob(args.examples))
    kb = RationaleKB(args.kb)

    all_entries = []
    ok_runs = 0
    with tempfile.TemporaryDirectory(prefix="arke-kb-") as tmp:
        for ak in examples:
            outdir = os.path.join(tmp, os.path.basename(ak).replace(".ak", ""))
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "arke.cli", "optimize", ak,
                     "--output", outdir, "--cycles", str(args.cycles), "--json"],
                    capture_output=True, text=True, timeout=120,
                )
            except Exception as e:
                print(f"  {os.path.basename(ak)}: run error {e}")
                continue
            strat = os.path.join(outdir, "strategy.json")
            if os.path.exists(strat):
                ents = mine_strategy_json(strat)
                all_entries.extend(ents)
                ok_runs += 1
                print(f"  {os.path.basename(ak):32s} decisions={len(ents)}")
            else:
                print(f"  {os.path.basename(ak):32s} NO strategy.json (rc={r.returncode})")

    written = kb.add_entries(all_entries)
    total = kb.count()
    print(json.dumps({
        "examples_run_ok": ok_runs,
        "entries_found": len(all_entries),
        "entries_written": written,
        "kb_total": total,
        "meets_g9_3": total >= 50,
    }, indent=2))
    return 0 if total >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
