# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""Arke Harness — extension runtimes (D1 Skills, D2 Hooks).

Two Claude-Code-style extension surfaces, wired to the Façade-driven loop:

* **Skills (D1)** — `SKILL.md` autotune recipes (YAML frontmatter + markdown
  body). `load_skill` / `load_skills_dir` parse them; `skills_prompt_block`
  renders selected recipes into a system-prompt addendum so the LLM can follow
  a proven optimization procedure (e.g. "sweep one op across all shape tiers").
  Same on-disk format as the repo's `skills/*/SKILL.md`.

* **Hooks (D2)** — lifecycle callbacks at `PreDecision` / `PostCompile` /
  `PostProfile` / `OnRollback`. A `PreDecision` hook MAY veto a decision
  (return False) — e.g. reject a `place(shared)` that would blow a register
  budget. Hook exceptions are isolated (never break the loop). This is the
  Arke analog of Claude Code's `PreToolUse` / `PostToolUse`.

Both are **Substrate** — additive, not part of the frozen Façade contract.
Design ref: docs/architecture/arke-harness.md §11 (Skills) §12 (Hooks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── D1: Skills ────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """A parsed SKILL.md autotune recipe."""
    name: str
    description: str
    body: str
    meta: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a `---\\n...\\n---\\n` YAML frontmatter from the markdown body.

    Uses a tiny dependency-free key: value / list parser (the SKILL.md
    frontmatter we author is shallow). Falls back to {} on any oddity.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw_fm, body = parts[1], parts[2]
    meta: dict[str, Any] = {}
    cur_key: str | None = None
    for line in raw_fm.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and cur_key:
            meta.setdefault(cur_key, [])
            if isinstance(meta[cur_key], list):
                meta[cur_key].append(line.lstrip()[2:].split("#", 1)[0].strip())
            continue
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            cur_key = k.strip()
            v = v.split("#", 1)[0].strip() if "#" not in v[:1] else v.strip()
            meta[cur_key] = v if v else []
    return meta, body.lstrip("\n")


def load_skill(path: str | Path) -> Skill:
    """Load and parse one SKILL.md."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    return Skill(
        name=str(meta.get("name") or p.parent.name),
        description=str(meta.get("description") or ""),
        body=body, meta=meta, path=str(p),
    )


def load_skills_dir(skills_dir: str | Path) -> dict[str, Skill]:
    """Load every `*/SKILL.md` under a directory. Bad files are skipped."""
    out: dict[str, Skill] = {}
    base = Path(skills_dir)
    if not base.is_dir():
        return out
    for sk in sorted(base.glob("*/SKILL.md")):
        try:
            s = load_skill(sk)
            out[s.name] = s
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping unparseable skill %s: %s", sk, e)
    return out


def skills_prompt_block(skills: list[Skill], *, max_body_chars: int = 1500) -> str:
    """Render selected skills into a system-prompt addendum.

    Each skill contributes its name + description + a (truncated) body so the
    model can follow the recipe. Returns "" for an empty selection.
    """
    if not skills:
        return ""
    blocks = ["\n\n# Loaded optimization skills (procedural recipes)\n"]
    for s in skills:
        body = s.body.strip()
        if len(body) > max_body_chars:
            body = body[:max_body_chars] + "\n…(truncated)"
        blocks.append(f"\n## SKILL: {s.name}\n{s.description}\n\n{body}\n")
    return "".join(blocks)


# ── D2: Hooks ───────────────────────────────────────────────────────────────

# Lifecycle hook points (Claude-Code analog).
HOOK_POINTS = ("PreDecision", "PostCompile", "PostProfile", "OnRollback")

# A hook is callable(context: dict) -> bool | None.
#   PreDecision: returning False VETOES the decision.
#   Others: return value ignored (observation only).
Hook = Callable[[dict[str, Any]], "bool | None"]


@dataclass
class HookRegistry:
    """Registry of lifecycle hooks. Exceptions in a hook are isolated."""
    hooks: dict[str, list[Hook]] = field(default_factory=lambda: {p: [] for p in HOOK_POINTS})

    def register(self, point: str, fn: Hook) -> None:
        if point not in HOOK_POINTS:
            raise ValueError(f"unknown hook point {point!r}; valid: {HOOK_POINTS}")
        self.hooks.setdefault(point, []).append(fn)

    def fire(self, point: str, context: dict[str, Any]) -> bool:
        """Fire all hooks at ``point``. Returns False iff a PreDecision hook
        vetoed (returned False). Observation hooks never veto."""
        allow = True
        for fn in self.hooks.get(point, []):
            try:
                rv = fn(context)
            except Exception as e:  # noqa: BLE001 — hook errors must not break the loop
                logger.warning("hook %s at %s raised: %s", getattr(fn, "__name__", fn), point, e)
                continue
            if point == "PreDecision" and rv is False:
                allow = False
        return allow

    def __bool__(self) -> bool:
        return any(self.hooks.get(p) for p in HOOK_POINTS)


# ── D3: Subagent design-space sweep ─────────────────────────────────────────

@dataclass
class SweepVariant:
    """One explored point in the design space."""
    label: str
    decisions: list[dict[str, Any]]   # [{kind, params, rationale}]
    latency_ms: float | None = None
    correct: bool | None = None
    error: str | None = None


def sweep_design_space(
    op_name: str,
    shapes: dict[str, list[int]],
    variants: list[tuple[str, list[dict[str, Any]]]],
    *,
    target_hw: str = "nvidia_ampere",
    max_workers: int = 3,
) -> tuple[SweepVariant | None, list[SweepVariant]]:
    """Explore N strategy variants in **isolated forked states**, in parallel.

    Each ``variant`` is ``(label, decisions)`` where ``decisions`` is a list of
    ``{kind, params, rationale}`` dicts. Each variant runs on its OWN
    ``ArkeEnv`` (isolated budget + state) so a failing fork can't corrupt a
    sibling or the parent. Returns ``(best, all_variants)`` where ``best`` is
    the lowest-latency correct variant (or None if none succeeded).

    This is the Arke analog of Claude Code's subagent fanout: parallel
    exploration of the bounded action space, only the winner reported up.
    The parent caller can then replay ``best.decisions`` into its own state.
    """
    from concurrent.futures import ThreadPoolExecutor

    from arke.agent.env import ArkeEnv
    from arke.agent.tools import ToolRegistry

    def _run_variant(label: str, decisions: list[dict[str, Any]]) -> SweepVariant:
        v = SweepVariant(label=label, decisions=decisions)
        try:
            env = ArkeEnv.from_op(op_name, shapes)
            reg = ToolRegistry.with_env(env)
            for d in decisions:
                params = {"kind": d["kind"], "params": d.get("params", {}),
                          "rationale": d.get("rationale", f"sweep variant {label}")}
                reg.get("apply_decision").execute(params)
            # verify + profile the assembled strategy. These stateless tools
            # take op_name + shapes explicitly (same as the live LLM passes).
            import json as _json
            ctx = {"op_name": op_name, "shapes": shapes}
            reg.get("verify_correctness").execute(ctx)
            prof = _json.loads(reg.get("compile_and_profile").execute(ctx).to_json())
            data = prof.get("data", {}) if isinstance(prof, dict) else {}
            v.latency_ms = data.get("latency_ms")
            v.correct = data.get("correct")
        except Exception as e:  # noqa: BLE001 — isolate fork failure
            v.error = f"{type(e).__name__}: {e}"
        return v

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda lv: _run_variant(lv[0], lv[1]), variants))

    correct = [v for v in results if v.correct and v.latency_ms is not None]
    correct.sort(key=lambda v: float(v.latency_ms))  # type: ignore[arg-type]
    best = correct[0] if correct else None
    return best, results
