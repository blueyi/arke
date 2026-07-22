#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5-S5-T generalization: synthesize per-op shape->strategy rule tables
from the agent's OWN exploration artifacts (strategies/*.json).

One LLM call per op (single-turn chat completion, no tool-use). The prompt
contains ONLY agent-produced exploration data — decisions, @rationale, and
the run's own best latency / baseline_ratio. It NEVER contains any data from
the L3 sweep (criteria-4 iron rule: no seeded answers).

The model must answer with a JSON rule table::

    {"op": "rmsnorm",
     "rules": [{"when": {"var": "M", "cmp": "<=", "value": 64},
                "decisions": [{"kind": ..., "params": ..., "level": 3,
                               "rationale": ...}]},
               ...],
     "fallback_decisions": [...],
     "rationale": "..."}

which is written to ``benchmarks/results/phase5/s5/strategies/{op}_rule.json``.

Variable vocabulary: rowwise ops {M=rows, N=cols}; matmul {M, K, N}.
``cmp`` supports <=, <, >=, >, ==. Rules are evaluated in order; the first
match wins; no match -> fallback_decisions.

:func:`apply_rule` is the pure evaluator reused by the gate (C5 held-out).

Usage:
    source ~/.venvs/arke/bin/activate  # BYOK: ~/.arke/llm.yaml or env
    python -m benchmarks.live.generalize_p5s5t [--only OP] [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGIES_DIR = REPO_ROOT / "benchmarks" / "results" / "phase5" / "s5" / "strategies"

OPS = ("rmsnorm", "softmax", "layernorm", "matmul")

# Shape-variable vocabulary per op family (documented contract for the LLM
# and for apply_rule).
ROWWISE_VARS = ("M", "N")        # M=rows, N=cols
MATMUL_VARS = ("M", "K", "N")    # A[M,K] @ B[K,N]

_CMPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


# ── pure rule evaluator (gate reuse) ────────────────────────────────────────

def _shape_vars(op: str, shape_dims: list[int]) -> dict[str, int]:
    if op == "matmul":
        if len(shape_dims) != 3:
            raise ValueError(f"matmul dims must be [M,K,N], got {shape_dims}")
        return dict(zip(MATMUL_VARS, shape_dims))
    if len(shape_dims) != 2:
        raise ValueError(f"rowwise dims must be [M,N], got {shape_dims}")
    return dict(zip(ROWWISE_VARS, shape_dims))


def _cond_matches(cond: dict, varmap: dict[str, int]) -> bool:
    var = cond.get("var")
    cmp = cond.get("cmp")
    value = cond.get("value")
    if var not in varmap:
        raise ValueError(f"unknown shape var {var!r} (have {sorted(varmap)})")
    if cmp not in _CMPS:
        raise ValueError(f"unsupported cmp {cmp!r} (supported {sorted(_CMPS)})")
    return _CMPS[cmp](varmap[var], value)


def apply_rule(rule_dict: dict, shape_dims: list[int]) -> list[dict]:
    """Evaluate a rule table against a concrete shape -> decisions list.

    ``rule_dict`` is the {op, rules, fallback_decisions, ...} table.
    ``shape_dims`` is [M, N] for rowwise ops or [M, K, N] for matmul.
    Rules are checked in order; a rule's ``when`` may be a single condition
    dict or a list of conditions (AND). First matching rule wins; if none
    match, ``fallback_decisions`` (default []) is returned.
    """
    varmap = _shape_vars(rule_dict.get("op", ""), shape_dims)
    for rule in rule_dict.get("rules") or []:
        when = rule.get("when")
        conds = when if isinstance(when, list) else [when]
        if all(_cond_matches(c, varmap) for c in conds if c is not None):
            return list(rule.get("decisions") or [])
    return list(rule_dict.get("fallback_decisions") or [])


# ── prompt construction (agent-artifacts ONLY — no sweep data) ──────────────

def load_explore_strategies(op: str,
                            strategies_dir: Path = STRATEGIES_DIR) -> list[dict]:
    """Load the agent's own exploration strategies for one op."""
    out = []
    for path in sorted(strategies_dir.glob(f"{op}_*.json")):
        if path.name == f"{op}_rule.json":
            continue
        rec = json.loads(path.read_text())
        if rec.get("role") == "explore":
            out.append(rec)
    return out


def build_prompt(op: str, strategies: list[dict]) -> str:
    """Build the single-turn generalization prompt for one op.

    Input data is exclusively the agent's own exploration products:
    per-shape decisions (+rationale) and that run's best latency /
    baseline_ratio. Nothing from the sweep is included.
    """
    if op == "matmul":
        vocab = ("Shape variables: M, K, N where the problem is "
                 "A[M,K] @ B[K,N].")
    else:
        vocab = f"Shape variables: M = number of rows, N = row width (cols), for {op} over [M,N]."

    blocks = []
    for rec in strategies:
        src = rec.get("source") or {}
        blocks.append(json.dumps({
            "shape": rec.get("shape"),
            "decisions": rec.get("decisions"),
            "best_latency_ms": src.get("best_latency_ms"),
            "baseline_ratio": src.get("baseline_ratio"),
        }, indent=1))

    exploration = "\n".join(blocks) if blocks else "(no exploration data)"

    return f"""You are an optimization-strategy generalizer for the Arke kernel compiler.

An autonomous agent explored L3 backend-tuning decisions for the op `{op}` on
a small set of shapes. Below are the agent's OWN per-shape results: the final
decision list it settled on (with its @rationale) and that run's measured best
latency and baseline_ratio (agent kernel vs default kernel; higher = better).
An empty decisions list means the agent decided the backend default was best.

{vocab}

Exploration results:
{exploration}

Task: synthesize a generalization rule table that predicts, for an UNSEEN
shape of this op, which decisions to apply. Interpolate/extrapolate from the
agent's rationale (e.g. row count vs occupancy, tile size vs reuse) — do not
just memorize the explored shapes.

Answer with ONLY a JSON object, no prose, in this exact schema:
{{"op": "{op}",
 "rules": [{{"when": {{"var": "M", "cmp": "<=", "value": 64}},
            "decisions": [{{"kind": "...", "params": {{...}}, "level": 3,
                           "rationale": "..."}}]}}],
 "fallback_decisions": [],
 "rationale": "one-paragraph explanation of the rule structure"}}

Constraints:
- `when` conditions may only use the shape variables listed above and
  cmp in {{<=, <, >=, >, ==}}. A `when` may be a list of conditions (AND).
- Rules are evaluated in order; first match wins; no match -> fallback_decisions.
- Decision kinds/params must come from the exploration data above (do not
  invent new kinds).
- An empty decisions list ("keep default") is a valid and often correct rule.
- BE CONSERVATIVE — the rule table is scored on UNSEEN shapes where every
  case must be at least as fast as the backend default; a rule that fires a
  wrong config on an unseen shape is a scored regression, while "keep
  default" is always safe. Therefore: only emit a non-empty rule for a
  shape region where the exploration shows a DECISIVE, well-understood win
  (e.g. clearly better latency with a rationale that names the mechanism).
  If the win is small, shape-specific, or the mechanism doesn't obviously
  extend beyond the explored point, the rule for that region — and the
  fallback — should be [] (keep default). Extrapolate a config to a shape
  region ONLY when the rationale's mechanism (occupancy, reuse, register
  pressure) clearly applies there too.
"""


# ── LLM call (same client pattern as arke/agent/runner.py, openai protocol) ─

def _parse_json_reply(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates fences)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        if start > 0:
            text = text[start:]
    return json.loads(text)


def synthesize_rule(op: str, strategies: list[dict], *,
                    model_spec: str | None = None,
                    timeout: float = 120.0) -> dict:
    """One single-turn LLM call -> validated rule table dict."""
    from arke.agent.llm_config import load_config

    config = load_config()
    provider, model = config.resolve(model_spec)
    if provider.protocol != "openai":
        # Same-protocol discipline as LLMRunner; generalization only needs
        # a plain chat completion, so require an OpenAI-protocol provider.
        chain = [p for p in config.provider_chain() if p.protocol == "openai"]
        if not chain:
            raise RuntimeError("No OpenAI-protocol provider configured (BYOK)")
        provider = chain[0]
        model = provider.default_model

    import openai
    client = openai.OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=timeout,
    )
    prompt = build_prompt(op, strategies)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = resp.choices[0].message.content or ""
    rule = _parse_json_reply(reply)

    # sanity: schema + evaluability on a probe shape
    rule.setdefault("op", op)
    if rule["op"] != op:
        raise ValueError(f"rule op mismatch: {rule['op']!r} != {op!r}")
    rule.setdefault("rules", [])
    rule.setdefault("fallback_decisions", [])
    probe = [1536, 1536, 1536] if op == "matmul" else [256, 4096]
    apply_rule(rule, probe)  # raises if the table is malformed
    return rule


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P5-S5-T rule synthesis (one LLM call per op)")
    ap.add_argument("--only", default=None, help="only this op")
    ap.add_argument("--model", default=None, help="BYOK model spec")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing *_rule.json")
    args = ap.parse_args(argv)

    failures = 0
    for op in OPS:
        if args.only and args.only != op:
            continue
        out_path = STRATEGIES_DIR / f"{op}_rule.json"
        if out_path.is_file() and not args.force:
            print(f"skip {op} (rule exists: {out_path})")
            continue
        strategies = load_explore_strategies(op)
        if not strategies:
            print(f"[{op}] no exploration strategies found — run "
                  f"benchmarks.live.run_p5s5t first", file=sys.stderr)
            failures += 1
            continue
        try:
            rule = synthesize_rule(op, strategies, model_spec=args.model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rule, indent=2))
            print(f"[{op}] rule table -> {out_path} "
                  f"({len(rule['rules'])} rules)")
        except Exception as e:
            failures += 1
            print(f"[{op}] FAILED: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
