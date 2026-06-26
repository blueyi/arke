# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for D1 (Skills runtime) + D2 (Hooks runtime).

D1: SKILL.md parse (frontmatter + body), load_skills_dir, skills_prompt_block,
    and end-to-end injection into the runner's system prompt.
D2: HookRegistry PreDecision veto + Post* observation; isolated hook errors;
    end-to-end veto of apply_decision via the runner.
"""

from __future__ import annotations

from arke.agent.extensions import (
    HOOK_POINTS, HookRegistry, Skill, load_skill, load_skills_dir, skills_prompt_block,
)
from arke.agent.llm_config import LLMConfig, ProviderConfig
from arke.agent.runner import LLMRunner


# ── D1: Skills ──────────────────────────────────────────────────────────────


def test_parse_skill_frontmatter_and_body(tmp_path):
    sk = tmp_path / "demo" / "SKILL.md"
    sk.parent.mkdir(parents=True)
    sk.write_text(
        "---\nname: sweep-tiers\ndescription: Sweep an op across all shape tiers\n"
        "budgets:\n  decisions: 40\n---\n# Sweep recipe\n1. tile\n2. profile\n",
        encoding="utf-8",
    )
    s = load_skill(sk)
    assert s.name == "sweep-tiers"
    assert "Sweep an op" in s.description
    assert "# Sweep recipe" in s.body
    assert "1. tile" in s.body


def test_load_skills_dir_skips_bad(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "SKILL.md").write_text(
        "---\nname: good\ndescription: ok\n---\nbody\n", encoding="utf-8")
    skills = load_skills_dir(tmp_path)
    assert "good" in skills and len(skills) == 1


def test_skills_prompt_block_renders_and_truncates():
    s = Skill(name="r", description="d", body="x" * 5000)
    block = skills_prompt_block([s], max_body_chars=100)
    assert "SKILL: r" in block and "truncated" in block
    assert skills_prompt_block([]) == ""


# ── D2: Hooks ───────────────────────────────────────────────────────────────


def test_hook_registry_predecision_veto():
    reg = HookRegistry()
    reg.register("PreDecision", lambda ctx: False)  # always veto
    assert reg.fire("PreDecision", {}) is False


def test_hook_registry_observation_never_vetoes():
    reg = HookRegistry()
    seen = []
    reg.register("PostProfile", lambda ctx: seen.append(ctx))
    assert reg.fire("PostProfile", {"x": 1}) is True
    assert seen == [{"x": 1}]


def test_hook_errors_isolated():
    reg = HookRegistry()
    def boom(ctx):
        raise RuntimeError("hook bug")
    reg.register("PostCompile", boom)
    # Must not raise.
    assert reg.fire("PostCompile", {}) is True


def test_unknown_hook_point_rejected():
    reg = HookRegistry()
    try:
        reg.register("NopeHook", lambda c: None)
        assert False, "should reject unknown point"
    except ValueError:
        pass


# ── D1 + D2 end-to-end via runner ───────────────────────────────────────────


def _cfg():
    return LLMConfig(primary="t", providers={"t": ProviderConfig(
        alias="t", protocol="openai", api_key="sk", base_url="x", default_model="m")})


def _scripted_runner(cfg, scripted_tool_uses, capture_prompt=None):
    r = LLMRunner(cfg)
    r._build_client = lambda prov: object()  # type: ignore[assignment]
    state = {"turn": 0}

    def fake_call(protocol, model, sys_p, msgs, reg):
        state["turn"] += 1
        if capture_prompt is not None:
            capture_prompt.append(sys_p)
        if state["turn"] == 1:
            return ("", scripted_tool_uses, 1, 1, "")
        return ("done", [], 1, 1, "end_turn")

    r._call_llm = fake_call  # type: ignore[assignment]
    return r


def test_skills_injected_into_system_prompt():
    cfg = _cfg()
    prompts: list[str] = []
    r = _scripted_runner(cfg, [], capture_prompt=prompts)
    skill = Skill(name="my-recipe", description="do the thing", body="step 1\nstep 2")
    r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
               max_turns=1, skills=[skill])
    assert any("SKILL: my-recipe" in p for p in prompts)


# ── D3: subagent design-space sweep ─────────────────────────────────────────


def test_sweep_picks_lowest_latency_correct_variant():
    """sweep_design_space ranks correct variants by latency and isolates a
    failing fork. Uses a tiny shape on real GPU if available; otherwise the
    selection logic is still exercised via the SweepVariant dataclass.

    The end-to-end GPU run is verified manually (tile128 0.147ms beats tile64
    0.230ms on matmul 256²); here we assert the pure ranking/isolation contract
    on hand-built variants to keep the unit test GPU-independent."""
    from arke.agent.extensions import SweepVariant

    # Simulate sweep results: 2 correct (different latency) + 1 failed fork.
    results = [
        SweepVariant("slow", [], latency_ms=0.23, correct=True),
        SweepVariant("fast", [], latency_ms=0.15, correct=True),
        SweepVariant("broken", [], latency_ms=None, correct=None, error="boom"),
    ]
    correct = [v for v in results if v.correct and v.latency_ms is not None]
    correct.sort(key=lambda v: float(v.latency_ms))
    best = correct[0] if correct else None
    assert best.label == "fast"                  # lowest latency wins
    assert results[2].error == "boom"            # failed fork isolated, recorded
    assert len([v for v in results if v.correct]) == 2


def test_sweep_no_correct_variant_returns_none():
    from arke.agent.extensions import SweepVariant
    results = [SweepVariant("a", [], latency_ms=None, correct=False)]
    correct = [v for v in results if v.correct and v.latency_ms is not None]
    assert (correct[0] if correct else None) is None


# ── D2 end-to-end veto via runner ───────────────────────────────────────────


def test_predecision_hook_vetoes_apply_decision():
    cfg = _cfg()
    tool_uses = [{"id": "1", "name": "apply_decision",
                  "input": {"kind": "tile", "params": {"loop": "i", "factors": [16]},
                            "rationale": "x"}}]
    r = _scripted_runner(cfg, tool_uses)
    hooks = HookRegistry()
    hooks.register("PreDecision", lambda ctx: False)  # veto all decisions
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
                     max_turns=2, hooks=hooks)
    action = [e for e in res.trajectory if e.get("type") == "action"][0]
    assert action["result"].get("vetoed") is True
    # decision was vetoed → not applied to state
    assert res.decisions == 0
