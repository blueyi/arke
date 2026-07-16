# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OptimizationState + ArkeEnv (D8-F1.1).

Design ref: docs/architecture/arke-harness.md §3 §8
Stage tracker: docs/phase1/stage8-plan.md D8-F1.1
"""

from __future__ import annotations

import pytest

from arke.agent.env import ArkeEnv
from arke.agent.state import (
    Checkpoint,
    CompileResult,
    OptimizationBudget,
    OptimizationState,
)
from arke.ir.strategy import Decision, Rationale


# ─── OptimizationBudget ────────────────────────────────────────────────────

def test_budget_defaults():
    b = OptimizationBudget()
    assert b.decision_max == 50
    assert b.compile_max == 30
    assert b.decisions_used == 0
    assert b.compiles_used == 0
    assert b.decisions_remaining == 50
    assert b.compiles_remaining == 30
    assert not b.exhausted


def test_budget_exhaustion():
    b = OptimizationBudget(decision_max=2, compile_max=1)
    b.decisions_used = 2
    b.compiles_used = 1
    assert b.decisions_remaining == 0
    assert b.compiles_remaining == 0
    assert b.exhausted


def test_budget_serialization():
    b = OptimizationBudget(decision_max=10, decisions_used=3)
    d = b.to_dict()
    assert d == {"decision_max": 10, "compile_max": 30, "decisions_used": 3, "compiles_used": 0}


# ─── CompileResult ─────────────────────────────────────────────────────────

def test_compile_result_minimal():
    r = CompileResult(success=True, backend="mock")
    d = r.to_dict()
    assert d == {"success": True, "backend": "mock"}


def test_compile_result_full():
    r = CompileResult(
        success=True, backend="triton", correct=True,
        max_diff=1e-5, latency_ms=1.2, baseline_ratio=1.5,
    )
    d = r.to_dict()
    assert d["latency_ms"] == 1.2
    assert d["baseline_ratio"] == 1.5
    assert "error" not in d


# ─── OptimizationState ─────────────────────────────────────────────────────

def test_state_defaults():
    s = OptimizationState()
    assert s.strategy is not None
    assert s.decision_log == []
    assert s.compile_results == []
    assert s.best_result is None
    assert s.checkpoints == {}
    assert s.budget.decisions_used == 0


def test_apply_decision_mutates_strategy_and_log():
    s = OptimizationState()
    d = Decision(kind="tile", params={"loop": "i", "factors": [32]})
    s.apply_decision(d)
    assert len(s.decision_log) == 1
    assert s.decision_log[0].step == 1
    assert s.budget.decisions_used == 1
    assert [ln.loop for ln in s.strategy.loop_nests] == ["i"]
    assert s.strategy.loop_nests[0].tile_factors == [32]


def test_apply_decision_budget_exhausted():
    s = OptimizationState(budget=OptimizationBudget(decision_max=1))
    s.apply_decision(Decision(kind="tile", params={"loop": "i", "factors": [16]}))
    with pytest.raises(RuntimeError, match="Decision budget exhausted"):
        s.apply_decision(Decision(kind="tile", params={"loop": "j", "factors": [32]}))


def test_record_compile_tracks_best():
    s = OptimizationState()
    s.record_compile(CompileResult(success=True, backend="triton", correct=True, latency_ms=2.0))
    s.record_compile(CompileResult(success=True, backend="triton", correct=True, latency_ms=1.0))
    s.record_compile(CompileResult(success=True, backend="triton", correct=True, latency_ms=3.0))
    assert s.best_result is not None
    assert s.best_result.latency_ms == 1.0
    assert s.budget.compiles_used == 3


def test_record_compile_ignores_failed():
    s = OptimizationState()
    s.record_compile(CompileResult(success=False, backend="mock", error="oops"))
    assert s.best_result is None


def test_record_compile_budget_exhausted():
    s = OptimizationState(budget=OptimizationBudget(compile_max=1))
    s.record_compile(CompileResult(success=True, backend="mock"))
    with pytest.raises(RuntimeError, match="Compile budget exhausted"):
        s.record_compile(CompileResult(success=True, backend="mock"))


# ─── Checkpoint + Rollback ─────────────────────────────────────────────────

def test_checkpoint_snapshot():
    s = OptimizationState()
    s.apply_decision(Decision(kind="tile", params={"loop": "i", "factors": [32]}))
    cp = s.checkpoint("alpha")
    assert isinstance(cp, Checkpoint)
    assert cp.label == "alpha"
    assert cp.decision_count_at == 1
    assert "alpha" in s.checkpoints


def test_rollback_restores_state():
    s = OptimizationState()
    s.apply_decision(Decision(kind="tile", params={"loop": "i", "factors": [32]}))
    s.checkpoint("snap1")
    s.apply_decision(Decision(kind="tile", params={"loop": "j", "factors": [64]}))
    s.apply_decision(Decision(kind="unroll", params={"loop": "i", "factor": 4}))
    assert len(s.decision_log) == 3

    s.rollback("snap1")
    assert len(s.decision_log) == 1
    assert s.budget.decisions_used == 1
    # Strategy restored: only "i" loop_nest with tile [32]
    loops = {ln.loop: ln.tile_factors for ln in s.strategy.loop_nests}
    assert loops == {"i": [32]}


def test_rollback_unknown_label():
    s = OptimizationState()
    with pytest.raises(KeyError, match="Unknown checkpoint"):
        s.rollback("nope")


def test_checkpoint_preserves_best_result():
    s = OptimizationState()
    s.record_compile(CompileResult(success=True, backend="triton", correct=True, latency_ms=2.0))
    s.checkpoint("c1")
    s.record_compile(CompileResult(success=True, backend="triton", correct=True, latency_ms=0.5))
    assert s.best_result.latency_ms == 0.5
    s.rollback("c1")
    assert s.best_result is not None
    assert s.best_result.latency_ms == 2.0


# ─── ArkeEnv ───────────────────────────────────────────────────────────────

def test_arkeenv_from_op():
    env = ArkeEnv.from_op("rmsnorm", {"X": [4, 128], "W": [128]})
    assert env.op_name == "rmsnorm"
    assert env.op_inputs == {"X": [4, 128], "W": [128]}
    assert env.state.strategy.kernel_id == "rmsnorm"


def test_arkeenv_unknown_op_raises():
    with pytest.raises(KeyError, match="Unknown op"):
        ArkeEnv.from_op("nonexistent_op_xyz")


def test_arkeenv_fills_default_shapes():
    env = ArkeEnv.from_op("rmsnorm")  # no shapes provided
    # Should default to [4,8] for each input
    assert "X" in env.op_inputs
    assert "W" in env.op_inputs


def test_list_legal_actions_default():
    env = ArkeEnv.from_op("rmsnorm")
    actions = env.list_legal_actions(top_n=20)
    assert len(actions) > 0
    assert len(actions) <= 20
    kinds = {a.kind for a in actions}
    # Default generator includes tile / unroll / vectorize / parallel / place
    assert "tile" in kinds


def test_list_legal_actions_filter_kind():
    env = ArkeEnv.from_op("rmsnorm")
    tile_only = env.list_legal_actions(top_n=100, filter_kind="tile")
    assert all(a.kind == "tile" for a in tile_only)
    assert len(tile_only) > 0

    place_only = env.list_legal_actions(top_n=100, filter_kind="place")
    assert all(a.kind == "place" for a in place_only)


def test_list_legal_actions_filters_redundant():
    env = ArkeEnv.from_op("rmsnorm")
    # Apply a tile decision
    env.state.apply_decision(Decision(kind="tile", params={"loop": "i", "factors": [32]}))
    # Listing tile candidates should no longer include the same (loop=i, factors=[32])
    tiles = env.list_legal_actions(top_n=100, filter_kind="tile")
    found_dup = any(
        a.params.get("loop") == "i" and a.params.get("factors") == [32]
        for a in tiles
    )
    assert not found_dup, "Redundant tile candidate not filtered"


def test_list_legal_actions_cumulative_tile_shrink():
    """After N consecutive tile applies on the same loop, the tile candidate
    count for that loop must monotonically decrease (A5 cumulative filter).

    Since default small shapes may produce few candidates per loop,
    we test directly via _filter_redundant with synthetic candidates.
    """
    env = ArkeEnv.from_op("rmsnorm")

    # Build synthetic candidates: 5 tile decisions on loop "i" with different factors
    synthetic_candidates = [
        Decision(kind="tile", params={"loop": "i", "factors": [f]}, level=1)
        for f in [16, 32, 64, 128, 256]
    ]

    # Before any applies, all 5 should survive
    filtered = env._filter_redundant(synthetic_candidates)
    prev_count = len(filtered)
    assert prev_count == 5

    # Apply tiles one by one; each apply should shrink the filtered set by 1
    for step, factor in enumerate([16, 32, 64, 128], start=1):
        env.state.apply_decision(
            Decision(kind="tile", params={"loop": "i", "factors": [factor]})
        )
        filtered = env._filter_redundant(synthetic_candidates)
        cur_count = len(filtered)
        assert cur_count == prev_count - 1, (
            f"Step {step}: applied factors=[{factor}] on loop 'i'. "
            f"Expected {prev_count - 1} candidates, got {cur_count}."
        )
        prev_count = cur_count

    # After 4 applies, only 1 candidate (factors=[256]) should remain
    assert prev_count == 1


def test_arkeenv_summary_contains_hints():
    env = ArkeEnv.from_op("rmsnorm")
    s = env.summary()
    assert "rmsnorm" in s
    assert "nvidia_ampere" in s or "hw=" in s
