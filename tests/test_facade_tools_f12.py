# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Façade tools 3/4/5/7/8 + input generator (D8-F1.2).

Design ref: docs/architecture/arke-harness.md §6
Stage tracker: docs/phase1/stage8-plan.md D8-F1.2
"""

from __future__ import annotations

import pytest
import torch

from arke.agent.env import ArkeEnv
from arke.agent.inputs import generate_inputs
from arke.agent.state import OptimizationBudget
from arke.agent.tools import ToolRegistry


# ─── generate_inputs ───────────────────────────────────────────────────────


def test_generate_inputs_shapes_and_dtype():
    out = generate_inputs("rmsnorm", {"X": [4, 8], "W": [8]}, seed=1)
    assert set(out.keys()) == {"X", "W"}
    assert tuple(out["X"].shape) == (4, 8)
    assert tuple(out["W"].shape) == (8,)
    assert out["X"].dtype == torch.float32


def test_generate_inputs_deterministic():
    a = generate_inputs("rmsnorm", {"X": [4, 8], "W": [8]}, seed=42)
    b = generate_inputs("rmsnorm", {"X": [4, 8], "W": [8]}, seed=42)
    assert torch.equal(a["X"], b["X"])
    assert torch.equal(a["W"], b["W"])


def test_generate_inputs_seed_sensitivity():
    a = generate_inputs("rmsnorm", {"X": [4, 8], "W": [8]}, seed=1)
    b = generate_inputs("rmsnorm", {"X": [4, 8], "W": [8]}, seed=2)
    assert not torch.equal(a["X"], b["X"])


def test_generate_inputs_uniform_range_respected():
    # exp has input_gen distributions={X: uniform}, ranges={X: (-5.0, 5.0)}
    out = generate_inputs("exp", {"X": [128]}, seed=1)
    assert out["X"].min().item() >= -5.0
    assert out["X"].max().item() <= 5.0


def test_generate_inputs_randint_dtype():
    # grouped_matmul has indices=randint
    out = generate_inputs(
        "grouped_matmul",
        {"X": [2, 4, 8], "W": [3, 8, 16], "indices": [2]},
        seed=1,
    )
    assert out["indices"].dtype == torch.int64


def test_generate_inputs_unknown_op_raises():
    with pytest.raises(KeyError, match="Unknown op"):
        generate_inputs("not_a_real_op_xyz", {})


def test_generate_inputs_missing_shape_raises():
    with pytest.raises(KeyError, match="shape missing"):
        generate_inputs("rmsnorm", {"X": [4, 8]})  # W missing


# ─── Registry & wiring ─────────────────────────────────────────────────────


@pytest.fixture
def env() -> ArkeEnv:
    return ArkeEnv.from_op("rmsnorm", {"X": [4, 32], "W": [32]})


@pytest.fixture
def reg(env: ArkeEnv) -> ToolRegistry:
    return ToolRegistry.with_env(env)


def test_registry_with_env_has_all_8_tools(reg: ToolRegistry):
    names = set(reg.names())
    # The 8 façade tools + benchmark_advice_summary carry-over
    expected_8 = {
        "get_hw_profile", "analyze_compute",
        "list_legal_actions", "apply_decision", "verify_correctness",
        "compile_and_profile", "checkpoint", "rollback",
    }
    assert expected_8.issubset(names), f"missing: {expected_8 - names}"


def test_registry_default_omits_env_tools():
    reg = ToolRegistry.default()
    names = set(reg.names())
    assert "apply_decision" not in names
    assert "checkpoint" not in names


def test_envbound_tool_rejects_non_env():
    from arke.agent.tools import ApplyDecisionTool
    with pytest.raises(TypeError, match="requires ArkeEnv"):
        ApplyDecisionTool("not an env")


# ─── list_legal_actions tool ───────────────────────────────────────────────


def test_list_legal_actions_default(reg: ToolRegistry):
    r = reg.get("list_legal_actions").execute({})
    assert r.success
    assert r.data["count"] > 0
    assert "candidates" in r.data
    for c in r.data["candidates"]:
        assert {"kind", "params", "level"}.issubset(c.keys())


def test_list_legal_actions_filter_kind(reg: ToolRegistry):
    r = reg.get("list_legal_actions").execute({"top_n": 100, "filter_kind": "tile"})
    assert r.success
    assert all(c["kind"] == "tile" for c in r.data["candidates"])
    assert r.data["count"] > 0


def test_list_legal_actions_top_n(reg: ToolRegistry):
    r = reg.get("list_legal_actions").execute({"top_n": 3})
    assert r.success
    assert r.data["count"] == 3


# ─── apply_decision tool ───────────────────────────────────────────────────


def test_apply_decision_happy_path(env: ArkeEnv, reg: ToolRegistry):
    r = reg.get("apply_decision").execute({
        "kind": "tile",
        "params": {"loop": "i", "factors": [32]},
        "rationale": "Pick a warp-friendly factor",
    })
    assert r.success
    assert r.data["decisions_used"] == 1
    assert r.data["applied"]["kind"] == "tile"
    assert r.data["applied"]["step"] == 1
    # Strategy actually mutated
    assert [ln.loop for ln in env.state.strategy.loop_nests] == ["i"]
    # Decision log has the rationale captured
    assert env.state.decision_log[0].rationale is not None
    assert env.state.decision_log[0].rationale.text == "Pick a warp-friendly factor"


def test_apply_decision_missing_kind(reg: ToolRegistry):
    r = reg.get("apply_decision").execute({"params": {}})
    assert not r.success
    assert "kind" in r.error


def test_apply_decision_rejects_non_dict_params(reg: ToolRegistry):
    r = reg.get("apply_decision").execute({"kind": "tile", "params": "not-a-dict"})
    assert not r.success
    assert "params" in r.error


def test_apply_decision_budget_exhaustion():
    env = ArkeEnv.from_op("rmsnorm", {"X": [4, 8], "W": [8]},
                          budget=OptimizationBudget(decision_max=1))
    reg = ToolRegistry.with_env(env)
    r1 = reg.get("apply_decision").execute({"kind": "tile", "params": {"loop": "i", "factors": [16]}})
    assert r1.success
    r2 = reg.get("apply_decision").execute({"kind": "tile", "params": {"loop": "j", "factors": [32]}})
    assert not r2.success
    assert "budget exhausted" in r2.error.lower()


# ─── verify_correctness tool ───────────────────────────────────────────────


def test_verify_correctness_no_trial(env: ArkeEnv, reg: ToolRegistry):
    r = reg.get("verify_correctness").execute({})
    assert r.success
    assert r.data["correct"] is True
    assert r.data["max_diff"] == 0.0
    assert r.data["validation_tier"] == "V0_mock"
    assert env.state.budget.compiles_used == 1


def test_verify_correctness_with_trial_does_not_consume_decision_budget(env: ArkeEnv, reg: ToolRegistry):
    decisions_before = env.state.budget.decisions_used
    decision_log_len_before = len(env.state.decision_log)
    r = reg.get("verify_correctness").execute({
        "decision": {"kind": "unroll", "params": {"loop": "j", "factor": 4}},
    })
    assert r.success
    # Trial was rolled back — decision counters unchanged
    assert env.state.budget.decisions_used == decisions_before
    assert len(env.state.decision_log) == decision_log_len_before
    # But compile budget did advance
    assert env.state.budget.compiles_used == 1


def test_verify_correctness_default_tolerance_per_dtype(reg: ToolRegistry):
    r = reg.get("verify_correctness").execute({})
    assert r.success
    # f32 default
    assert r.data["rtol"] == pytest.approx(1e-3)
    assert r.data["atol"] == pytest.approx(1e-5)


def test_verify_correctness_tolerance_override(reg: ToolRegistry):
    r = reg.get("verify_correctness").execute({"rtol": 1e-6, "atol": 1e-9})
    assert r.success
    assert r.data["rtol"] == pytest.approx(1e-6)
    assert r.data["atol"] == pytest.approx(1e-9)


def test_verify_correctness_no_trial_cleanup_when_no_checkpoint():
    """Probe without trial must not leave temp checkpoint behind."""
    env = ArkeEnv.from_op("rmsnorm", {"X": [4, 8], "W": [8]})
    reg = ToolRegistry.with_env(env)
    reg.get("verify_correctness").execute({})
    assert "__verify_tmp__" not in env.state.checkpoints


def test_verify_correctness_trial_cleanup_after_rollback():
    """After trial-balloon, temp checkpoint must be gone."""
    env = ArkeEnv.from_op("rmsnorm", {"X": [4, 8], "W": [8]})
    reg = ToolRegistry.with_env(env)
    reg.get("verify_correctness").execute({
        "decision": {"kind": "tile", "params": {"loop": "i", "factors": [32]}},
    })
    assert "__verify_tmp__" not in env.state.checkpoints


def test_verify_correctness_invalid_trial_payload(reg: ToolRegistry):
    r = reg.get("verify_correctness").execute({"decision": "not-a-dict"})
    assert not r.success
    assert "decision" in r.error.lower()


def test_verify_correctness_compile_budget_exhaustion():
    env = ArkeEnv.from_op("rmsnorm", {"X": [4, 8], "W": [8]},
                          budget=OptimizationBudget(compile_max=1))
    reg = ToolRegistry.with_env(env)
    r1 = reg.get("verify_correctness").execute({})
    assert r1.success
    r2 = reg.get("verify_correctness").execute({})
    assert not r2.success
    assert "exhausted" in r2.error.lower()


# ─── checkpoint tool ───────────────────────────────────────────────────────


def test_checkpoint_happy_path(env: ArkeEnv, reg: ToolRegistry):
    reg.get("apply_decision").execute({"kind": "tile", "params": {"loop": "i", "factors": [32]}})
    r = reg.get("checkpoint").execute({"label": "alpha"})
    assert r.success
    assert r.data["label"] == "alpha"
    assert r.data["decision_count_at"] == 1
    assert r.data["total_checkpoints"] == 1
    assert "alpha" in env.state.checkpoints


def test_checkpoint_rejects_empty_label(reg: ToolRegistry):
    r = reg.get("checkpoint").execute({"label": ""})
    assert not r.success


def test_checkpoint_rejects_missing_label(reg: ToolRegistry):
    r = reg.get("checkpoint").execute({})
    assert not r.success


# ─── rollback tool ─────────────────────────────────────────────────────────


def test_rollback_happy_path(env: ArkeEnv, reg: ToolRegistry):
    reg.get("apply_decision").execute({"kind": "tile", "params": {"loop": "i", "factors": [32]}})
    reg.get("checkpoint").execute({"label": "snap"})
    reg.get("apply_decision").execute({"kind": "tile", "params": {"loop": "j", "factors": [64]}})
    reg.get("apply_decision").execute({"kind": "unroll", "params": {"loop": "i", "factor": 4}})
    assert len(env.state.decision_log) == 3

    r = reg.get("rollback").execute({"label": "snap"})
    assert r.success
    assert r.data["restored_to"] == "snap"
    assert len(env.state.decision_log) == 1
    assert env.state.budget.decisions_used == 1


def test_rollback_unknown_label(reg: ToolRegistry):
    r = reg.get("rollback").execute({"label": "nope"})
    assert not r.success
    assert "Unknown" in r.error or "unknown" in r.error.lower()


def test_rollback_rejects_empty_label(reg: ToolRegistry):
    r = reg.get("rollback").execute({"label": ""})
    assert not r.success


# ─── End-to-end Façade workflow ────────────────────────────────────────────


def test_facade_workflow_explore_then_rollback():
    """An LLM-like flow: list → apply → verify → checkpoint → explore → rollback."""
    env = ArkeEnv.from_op("rmsnorm", {"X": [4, 16], "W": [16]})
    reg = ToolRegistry.with_env(env)

    r = reg.get("list_legal_actions").execute({"filter_kind": "tile", "top_n": 1})
    assert r.success and r.data["count"] == 1
    chosen = r.data["candidates"][0]

    r = reg.get("apply_decision").execute({"kind": chosen["kind"], "params": chosen["params"]})
    assert r.success
    assert env.state.budget.decisions_used == 1

    r = reg.get("verify_correctness").execute({})
    assert r.success and r.data["correct"]

    r = reg.get("checkpoint").execute({"label": "baseline"})
    assert r.success

    # Explore: trial another decision via verify (does NOT mutate)
    r = reg.get("verify_correctness").execute({
        "decision": {"kind": "unroll", "params": {"loop": "i", "factor": 4}},
    })
    assert r.success
    assert env.state.budget.decisions_used == 1  # unchanged

    # Then really apply it
    r = reg.get("apply_decision").execute({"kind": "unroll", "params": {"loop": "i", "factor": 4}})
    assert r.success and env.state.budget.decisions_used == 2

    # Roll back
    r = reg.get("rollback").execute({"label": "baseline"})
    assert r.success
    assert env.state.budget.decisions_used == 1
    assert len(env.state.decision_log) == 1
