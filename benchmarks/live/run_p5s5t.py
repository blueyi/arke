#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5-S5-T explore driver: multi-op live-LLM L3 optimization over the
tightened-gate shape matrix, plus final-strategy extraction.

For every (op, dims) in EXPLORE_MATRIX this script:

  1. drives the frozen Facade with the builtin live-LLM backend
     (``arke.agent.backends.run_backend("builtin", ...)``), writing
     ``state.json`` / ``trajectory.json`` under
     ``benchmarks/results/phase5/s5/t/{op}_{shape_label}/``;
  2. extracts the agent's FINAL strategy from ``state.json`` (see
     :func:`extract_strategy`) and writes
     ``benchmarks/results/phase5/s5/strategies/{op}_{shape_label}.json``.

Strategy file schema::

    {"op": ..., "shape": ..., "role": "explore",
     "decisions": [{"kind", "params", "level", "rationale"}, ...],
     "source": {"state_json", "best_latency_ms", "baseline_ratio",
                "extracted_from"}}

``decisions == []`` means "keep the backend default" — a legitimate and
important outcome (e.g. layernorm, where the L3 sweep showed no headroom).

Extraction rule (criteria-4 honest: only the agent's own run artifacts are
consulted, never the sweep):

  - every checkpoint whose ``best_result`` is ``correct=True`` is a candidate
    (latency = ``best_result.latency_ms``, decisions = its ``decision_log``);
  - the top-level ``best_result`` (if correct) is also a candidate; its
    reproducing decision set is ``decision_log[:strategy_decisions]`` (the
    top-level log may contain post-best experiments that were never profiled
    or made things worse — ``best_result.metadata.strategy_decisions`` says
    how many decisions the best kernel actually consumed);
  - the minimum-latency candidate wins; ties prefer checkpoints (their
    decision_log is an exact snapshot).

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    python -m benchmarks.live.run_p5s5t [--only KEY_SUBSTR]
        [--max-turns 40] [--timeout 900] [--force]

Resume: cases whose strategies/*.json already exists are skipped
(delete the file or pass --force to redo).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.live.run_live_optimize import _shapes_for

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
S5_DIR = REPO_ROOT / "benchmarks" / "results" / "phase5" / "s5"
RUNS_DIR = S5_DIR / "t"
STRATEGIES_DIR = S5_DIR / "strategies"

# Tightened-gate explore matrix (P5-S5-T): gate shapes + large shapes.
# rowwise dims = [M(rows), N(cols)]; matmul dims = [M, K, N].
EXPLORE_MATRIX: list[tuple[str, list[int]]] = [
    ("rmsnorm", [32, 4096]),
    ("rmsnorm", [1024, 4096]),
    ("softmax", [32, 4096]),
    ("softmax", [1024, 4096]),
    ("layernorm", [32, 4096]),
    ("layernorm", [1024, 4096]),
    ("matmul", [1024, 1024, 1024]),
    ("matmul", [2048, 2048, 2048]),
]

DEFAULT_MAX_TURNS = 40
DEFAULT_TIMEOUT = 900.0


def shape_label(dims: list[int]) -> str:
    return "x".join(str(d) for d in dims)


def case_key(op: str, dims: list[int]) -> str:
    return f"{op}@{shape_label(dims)}"


def _rationale_text(r: Any) -> str:
    """Normalize a rationale (Rationale dict / plain string / None) to text."""
    if isinstance(r, dict):
        return str(r.get("text", "") or "")
    return str(r or "")


def _normalize_decisions(decision_log: list[dict]) -> list[dict]:
    out = []
    for d in decision_log or []:
        out.append({
            "kind": d.get("kind"),
            "params": d.get("params", {}),
            "level": d.get("level", 3),
            "rationale": _rationale_text(d.get("rationale")),
        })
    return out


def extract_strategy(state: dict, state_json_path: str) -> tuple[list[dict], dict]:
    """Extract the agent's final strategy from a live-run state.json dict.

    Returns ``(decisions, source)`` where ``decisions`` is the normalized
    decision list ([] = keep default) and ``source`` is the provenance dict.
    """
    # (latency_ms, decision_log, provenance_label, best_result)
    candidates: list[tuple[float, list[dict], str, dict]] = []

    for label, ck in (state.get("checkpoints") or {}).items():
        br = ck.get("best_result")
        if br and br.get("correct") and br.get("latency_ms") is not None:
            candidates.append((
                float(br["latency_ms"]),
                ck.get("decision_log") or [],
                f"checkpoint:{label}",
                br,
            ))

    top_br = state.get("best_result")
    if top_br and top_br.get("correct") and top_br.get("latency_ms") is not None:
        dl = list(state.get("decision_log") or [])
        # The top-level decision_log may include decisions made AFTER the
        # best profile (never profiled / regressions). metadata.strategy_decisions
        # records how many decisions the best kernel actually consumed —
        # that prefix is the reproducing decision set.
        nd = (top_br.get("metadata") or {}).get("strategy_decisions")
        if isinstance(nd, int) and 0 <= nd <= len(dl):
            dl = dl[:nd]
        candidates.append((float(top_br["latency_ms"]), dl, "top_level", top_br))

    if not candidates:
        # No correct measured result at all -> keep default.
        return [], {
            "state_json": state_json_path,
            "best_latency_ms": None,
            "baseline_ratio": None,
            "extracted_from": state_json_path + "::none",
        }

    # min() keeps the FIRST minimal element -> ties prefer checkpoints
    # (appended before the top-level candidate).
    latency, decision_log, label, br = min(candidates, key=lambda c: c[0])
    return _normalize_decisions(decision_log), {
        "state_json": state_json_path,
        "best_latency_ms": latency,
        "baseline_ratio": br.get("baseline_ratio"),
        "extracted_from": f"{state_json_path}::{label}",
    }


def write_strategy_file(op: str, dims: list[int], decisions: list[dict],
                        source: dict, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = STRATEGIES_DIR  # resolved at call time (testable)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "op": op,
        "shape": shape_label(dims),
        "role": "explore",
        "decisions": decisions,
        "source": source,
    }
    path = out_dir / f"{op}_{shape_label(dims)}.json"
    path.write_text(json.dumps(rec, indent=2))
    return path


def run_case(op: str, dims: list[int], *, max_turns: int, timeout: float,
             model_spec: str | None = None) -> Path:
    """Run one live case end-to-end and write its strategies file."""
    from arke.agent.backends import run_backend  # lazy: needs BYOK env

    label = shape_label(dims)
    output_dir = RUNS_DIR / f"{op}_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = _shapes_for(op, dims)
    result = run_backend(
        "builtin",
        op_name=op,
        shapes=shapes,
        target_hw="nvidia_ampere",
        max_turns=max_turns,
        model_spec=model_spec,
        output_dir=str(output_dir),
        timeout=timeout,
    )
    if not result.success:
        raise RuntimeError(
            f"live run failed for {case_key(op, dims)}: {result.message}")

    state_path = output_dir / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"live run wrote no state.json: {state_path}")
    state = json.loads(state_path.read_text())
    decisions, source = extract_strategy(state, str(state_path))
    return write_strategy_file(op, dims, decisions, source)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P5-S5-T explore driver (multi-op live L3 optimization)")
    ap.add_argument("--only", default=None,
                    help="only run cases whose key contains this substring")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--model", default=None, help="BYOK model spec")
    ap.add_argument("--force", action="store_true",
                    help="re-run cases even if strategies file exists")
    args = ap.parse_args(argv)

    failures = 0
    for op, dims in EXPLORE_MATRIX:
        key = case_key(op, dims)
        if args.only and args.only not in key:
            continue
        strat_path = STRATEGIES_DIR / f"{op}_{shape_label(dims)}.json"
        if strat_path.is_file() and not args.force:
            print(f"skip {key} (strategies file exists: {strat_path})")
            continue
        print(f"=== {key}: live run (max_turns={args.max_turns}, "
              f"timeout={args.timeout}s) ===", flush=True)
        try:
            path = run_case(op, dims, max_turns=args.max_turns,
                            timeout=args.timeout, model_spec=args.model)
            rec = json.loads(path.read_text())
            nd = len(rec["decisions"])
            print(f"[{key}] strategy extracted -> {path} "
                  f"({nd} decisions{' = keep default' if nd == 0 else ''})",
                  flush=True)
        except Exception as e:
            failures += 1
            print(f"[{key}] FAILED: {e}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
