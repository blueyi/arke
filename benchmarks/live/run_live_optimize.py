#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""Arke Harness v2 — live LLM autotuning driver + evidence-card writer.

Drives a real LLM (yunwu /v1 by default) through the 8-tool Façade to
generate + tune a real Triton GPU kernel, then writes an evidence card with
the quantitative result (decisions, tool calls, tokens, real baseline_ratio,
@rationale audit, S2 resume provenance).

Usage:
    source ~/.venvs/arke/bin/activate && source ~/.env.rc
    python -m benchmarks.live.run_live_optimize --op matmul --shape 512,512,512 \\
        --model yunwu/claude-sonnet-4-6 --max-turns 25 \\
        --out benchmarks/results/phase1/harness_v2/live/matmul

Writes under --out:
    state.json        (S2 resumable snapshot)
    result.json       (full OptimizeResult.to_dict)
    trajectory.jsonl  (replayable action log)
    evidence.md       (human-readable evidence card)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("arke.live")


# op → default input-shape builder from a comma list of dims
def _shapes_for(op: str, dims: list[int]) -> dict[str, list[int]]:
    if op in ("matmul", "batch_matmul"):
        # M,K,N → A[M,K] B[K,N]
        if len(dims) >= 3:
            m, k, n = dims[0], dims[1], dims[2]
            return {"A": [m, k], "B": [k, n]}
        return {"A": [512, 512], "B": [512, 512]}
    if op == "rmsnorm":
        m, n = (dims[0], dims[1]) if len(dims) >= 2 else (4096, 4096)
        return {"X": [m, n], "W": [n]}
    if op == "layernorm":
        m, n = (dims[0], dims[1]) if len(dims) >= 2 else (4096, 4096)
        return {"X": [m, n], "W": [n], "B": [n]}
    if op in ("softmax", "relu", "gelu", "silu"):
        if len(dims) >= 2:
            return {"X": [dims[0], dims[1]]}
        return {"X": [4096, 4096]}
    if op in ("flash_attention", "grouped_query_attention"):
        # B,H,S,D
        if len(dims) >= 4:
            b, h, s, d = dims[:4]
            return {"Q": [b, h, s, d], "K": [b, h, s, d], "V": [b, h, s, d]}
        return {"Q": [1, 8, 256, 64], "K": [1, 8, 256, 64], "V": [1, 8, 256, 64]}
    # fallback: single tensor of the given dims
    return {"X": dims or [512, 512]}


def _write_live_trajectory(path, op: str, result) -> None:
    """Emit a mineable trajectory.jsonl from the LLMRunner action log.

    Maps each apply_decision action → a `decision` record (with rationale,
    kind, params) and each compile_and_profile action → a `profile` record
    (latency_ms, baseline_ratio, correct). Header carries the op for context.
    This is the shape arke.learn.rationale_kb.mine_trajectory consumes.
    """
    import json as _json

    lines = [{"kind": "header", "data": {"op": op, "contract_id": "arke-live-trajectory"}}]
    for e in result.trajectory:
        if e.get("type") != "action":
            continue
        tool = e.get("tool")
        params = e.get("params", {}) or {}
        data = (e.get("result") or {}).get("data") or {}
        if tool == "apply_decision":
            lines.append({"kind": "decision", "data": {
                "kind": params.get("kind"),
                "params": params.get("params", {}),
                "rationale": params.get("rationale", ""),
                "op": op,
            }})
        elif tool == "compile_and_profile":
            lines.append({"kind": "profile", "data": {
                "latency_ms": data.get("latency_ms"),
                "baseline_ratio": data.get("baseline_ratio"),
                "correct": data.get("correct"),
                "backend": data.get("backend"),
            }})
        elif tool == "verify_correctness":
            lines.append({"kind": "compile", "data": {"correct": data.get("correct")}})
    Path(path).write_text("\n".join(_json.dumps(x) for x in lines) + "\n", encoding="utf-8")


def _write_evidence_card(out_dir: Path, op: str, shapes: dict, result) -> Path:
    from arke.learn.trajectory import audit_decision_rationales

    summary = result.session_summary
    traj_path = out_dir / "trajectory.jsonl"

    # A5 audit on the emitted trajectory (if present).
    audit = audit_decision_rationales(traj_path) if traj_path.is_file() else ["(no trajectory.jsonl)"]
    rationale_clean = (audit == [])

    # Count LLM-authored decisions with rationale.
    dlog = summary.get("decision_log", [])
    with_rationale = sum(1 for d in dlog if (d.get("rationale") or "").strip())

    # Extract the BEST real GPU profile straight from the action trajectory
    # (the source of truth — best_performance in session_summary can be empty
    # if the runner's best-tracking path differs). best = lowest latency among
    # correct compile_and_profile results.
    profiles = []
    for e in result.trajectory:
        if e.get("type") == "action" and e.get("tool") == "compile_and_profile":
            d = (e.get("result") or {}).get("data") or {}
            if d.get("correct") and d.get("latency_ms") is not None:
                profiles.append((d.get("latency_ms"), d.get("baseline_ratio"), d.get("backend")))
    profiles.sort(key=lambda x: x[0])  # lowest latency = best
    best_lat, best_ratio, best_backend = profiles[0] if profiles else (None, None, None)
    real_profile = bool(profiles)  # any real correct GPU profile recorded

    card = f"""# Live evidence — {op}

**Date:** generated by run_live_optimize
**Op / shapes:** `{op}` {json.dumps(shapes)}
**Model:** {result.model_used}
**Target HW:** nvidia_ampere (RTX 3060 Laptop, SM 8.6, 6 GB)

## Result

| Metric | Value |
|:--|:--|
| Stop reason | {result.stop_reason} |
| Decisions | {result.decisions} |
| Decisions with @rationale | {with_rationale}/{len(dlog)} |
| Tool calls | {result.tool_calls} |
| compile_and_profile calls (correct) | {len(profiles)} |
| Tokens (in/out) | {result.tokens_in} / {result.tokens_out} |
| Duration | {result.duration_seconds}s |
| **Best latency (real GPU)** | {f"{best_lat:.4f} ms" if best_lat is not None else "n/a"} |
| **Best baseline_ratio** | {f"{best_ratio:.4f}" if best_ratio is not None else "n/a"} |
| Backend | {best_backend or "?"} |
| Real GPU profile? | {"✅ yes — real Triton latency measured" if real_profile else "⚠️ none"} |
| Errors | {result.errors or "none"} |

## @rationale trajectory audit (A5)

{"✅ clean — every decision carries a non-empty rationale" if rationale_clean else "⚠️ violations:"}
{chr(10).join("- " + v for v in audit) if not rationale_clean else ""}

## Real GPU profiles (compile_and_profile results, by latency)

{chr(10).join(f"- latency={lat:.4f} ms, baseline_ratio={ratio}, backend={bk}" for lat, ratio, bk in profiles) or "(none)"}

## S2 resume provenance

```json
{json.dumps(summary.get("resume", {}), indent=2)}
```

## Decisions (with @rationale)

{chr(10).join(f"- `{d['kind']}({json.dumps(d['params'], ensure_ascii=False)})` :: {(d.get('rationale') or '')[:120]}" for d in dlog) or "(none)"}

## Provider fallback events (S3)

```json
{json.dumps(summary.get("fallback_events", []), indent=2)}
```
"""
    card_path = out_dir / "evidence.md"
    card_path.write_text(card, encoding="utf-8")
    return card_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, help="operator name (matmul/rmsnorm/flash_attention/...)")
    ap.add_argument("--shape", default="", help="comma-separated dims, op-specific (e.g. 512,512,512)")
    ap.add_argument("--model", default=None, help="model_spec, e.g. yunwu/claude-sonnet-4-6")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", required=True, help="output dir for state/result/trajectory/evidence")
    ap.add_argument("--resume-from", default=None, help="resume from a prior run's dir/state.json")
    args = ap.parse_args()

    from arke.agent.llm_config import LLMConfigError, load_from_env
    from arke.agent.runner import LLMRunner

    try:
        config = load_from_env()
    except LLMConfigError as e:
        logger.error("No LLM provider configured: %s", e)
        return 1
    logger.info("primary=%s providers=%s fallback=%s",
                config.primary, list(config.providers.keys()), config.fallback)

    dims = [int(x) for x in args.shape.split(",") if x.strip()] if args.shape else []
    shapes = _shapes_for(args.op, dims)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with LLMRunner(config, timeout=args.timeout) as runner:
        result = runner.optimize(
            op_name=args.op, shapes=shapes, target_hw="nvidia_ampere",
            max_turns=args.max_turns, model_spec=args.model,
            resume_from=args.resume_from, state_out=str(out_dir),
        )

    # Persist artifacts.
    (out_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    # Write a trajectory.jsonl directly from the action log. The LLMRunner
    # trajectory is [{type:action, tool, params, result}] with the result
    # already attached per action (no separate 'result' entries), so we map
    # each apply_decision action to a `decision` record carrying its rationale
    # (mineable by arke.learn.rationale_kb.mine_trajectory), and other tools to
    # their matching record kind.
    _write_live_trajectory(out_dir / "trajectory.jsonl", args.op, result)

    card = _write_evidence_card(out_dir, args.op, shapes, result)
    logger.info("evidence card: %s", card)
    print(card.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
