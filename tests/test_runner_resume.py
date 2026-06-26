# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for S2 — resumable trajectory / OptimizationState (de)serialization.

Covers OptimizationState.to_dict <-> from_dict round-trip, LLMRunner state
snapshot dump (state_out) + rehydrate (resume_from), and graceful degradation
on a missing/partial snapshot. No network: the LLM turn is stubbed so the loop
ends immediately (model "stops calling tools").
"""

from __future__ import annotations

import json
import os

from arke.agent.env import ArkeEnv
from arke.agent.llm_config import LLMConfig, ProviderConfig
from arke.agent.runner import LLMRunner
from arke.agent.state import OptimizationState
from arke.ir.strategy import Decision, Rationale


def _prov():
    return ProviderConfig(
        alias="t", protocol="openai", api_key="sk", base_url="x", default_model="m",
    )


def _cfg():
    return LLMConfig(primary="t", providers={"t": _prov()})


# ── OptimizationState round-trip ───────────────────────────────────────────


def test_state_to_from_dict_round_trip():
    env = ArkeEnv.from_op("matmul", {"A": [64, 32], "B": [32, 64]})
    st = env.state
    st.apply_decision(Decision(kind="tile", params={"loop": "i", "factors": [32]},
                               rationale=Rationale(text="warp-friendly"), level=1))
    st.apply_decision(Decision(kind="unroll", params={"loop": "i", "factor": 4},
                               rationale=Rationale(text="reduce overhead"), level=1))

    d = st.to_dict()
    st2 = OptimizationState.from_dict(d)

    assert len(st2.decision_log) == 2
    assert st2.decision_log[0].kind == "tile"
    assert st2.decision_log[0].rationale.text == "warp-friendly"
    assert st2.decision_log[1].kind == "unroll"
    assert st2.budget.decisions_used == 2
    assert st2.budget.decision_max == st.budget.decision_max


def test_state_from_partial_dict_uses_defaults():
    """A truncated/crashed snapshot still yields a usable state."""
    st = OptimizationState.from_dict({})  # nothing
    assert st.decision_log == []
    assert st.budget.decisions_used == 0
    assert st.best_result is None


# ── F1 (2026-06-26): best_result prefers a real profile over verify-only ──


def _cr(latency=None, correct=True):
    from arke.agent.state import CompileResult
    return CompileResult(success=True, backend="triton", correct=correct,
                         latency_ms=latency, baseline_ratio=(1.0 if latency else None))


def test_best_result_profile_displaces_verify_only():
    """A real profile (has latency) must replace a verify-only incumbent."""
    env = ArkeEnv.from_op("matmul", {"A": [64, 32], "B": [32, 64]})
    st = env.state
    st.record_compile(_cr(latency=None))      # verify-only lands first
    assert st.best_result.latency_ms is None
    st.record_compile(_cr(latency=0.08))      # real profile arrives
    assert st.best_result.latency_ms == 0.08  # F1: profile wins


def test_best_result_keeps_lower_latency():
    env = ArkeEnv.from_op("matmul", {"A": [64, 32], "B": [32, 64]})
    st = env.state
    st.record_compile(_cr(latency=0.20))
    st.record_compile(_cr(latency=0.08))   # faster
    st.record_compile(_cr(latency=0.15))   # slower — must not displace
    assert st.best_result.latency_ms == 0.08


def test_best_result_verify_only_does_not_displace_profile():
    env = ArkeEnv.from_op("matmul", {"A": [64, 32], "B": [32, 64]})
    st = env.state
    st.record_compile(_cr(latency=0.08))   # profile best
    st.record_compile(_cr(latency=None))   # verify-only later — must NOT win
    assert st.best_result.latency_ms == 0.08


# ── LLMRunner resume ───────────────────────────────────────────────────────


def _make_runner_that_stops_immediately(cfg):
    """A runner whose LLM turn returns no tool calls → loop ends turn 1."""
    r = LLMRunner(cfg)
    r._build_client = lambda prov: object()  # type: ignore[assignment]
    r._call_llm = lambda *a, **k: ("done", [], 1, 1, "end_turn")  # type: ignore[assignment]
    return r


def test_optimize_writes_state_out(tmp_path):
    cfg = _cfg()
    r = _make_runner_that_stops_immediately(cfg)
    out = tmp_path / "run1"
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
                     max_turns=1, state_out=str(out))
    state_file = out / "state.json"
    assert state_file.is_file()
    assert res.session_summary["resume"]["state_out"] == str(state_file)


def test_optimize_resume_rehydrates_spent_budget(tmp_path):
    cfg = _cfg()
    # Hand-craft a prior state with 5 decisions + 3 compiles already spent.
    env = ArkeEnv.from_op("matmul", {"A": [64, 32], "B": [32, 64]})
    for i in range(5):
        env.state.apply_decision(Decision(kind="tile", params={"loop": "i", "factors": [16]},
                                          rationale=Rationale(text=f"d{i}"), level=1))
    env.state.budget.compiles_used = 3
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(env.state.to_dict(), default=str), encoding="utf-8")

    r = _make_runner_that_stops_immediately(cfg)
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
                     max_turns=1, resume_from=str(state_file))

    # The spent budget was carried over, not reset.
    assert res.session_summary["resume"]["replayed_decisions"] == 5
    assert res.session_summary["resume"]["replayed_compiles"] == 3
    assert res.session_summary["budget"]["decisions_used"] == 5
    assert res.session_summary["budget"]["compiles_used"] == 3
    assert res.decisions == 5  # decision_log carried


def test_optimize_resume_missing_file_starts_fresh(tmp_path):
    cfg = _cfg()
    r = _make_runner_that_stops_immediately(cfg)
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
                     max_turns=1, resume_from=str(tmp_path / "nonexistent"))
    # No crash; starts fresh (0 replayed).
    assert res.session_summary["resume"]["replayed_decisions"] == 0
    assert res.decisions == 0
