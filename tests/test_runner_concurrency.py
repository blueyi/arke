# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for C1 (on_event streaming) + C2 (concurrent read-tool partitioning).

C1: optimize(on_event=...) calls back with each action as it lands.
C2: tools partition by ToolMeta.concurrent_safe; concurrent_safe batches run
together (thread pool), mutating tools force their own serial batch; trajectory
ordering is preserved.

No network: the LLM turn is stubbed to emit a scripted set of tool calls.
"""

from __future__ import annotations

from arke.agent.llm_config import LLMConfig, ProviderConfig
from arke.agent.runner import LLMRunner
from arke.agent.tools import ToolRegistry


def _cfg():
    return LLMConfig(primary="t", providers={"t": ProviderConfig(
        alias="t", protocol="openai", api_key="sk", base_url="x", default_model="m")})


# ── C2: partition_for_execution ────────────────────────────────────────────


def test_partition_groups_concurrent_then_serial():
    from arke.agent.env import ArkeEnv
    reg = ToolRegistry.with_env(ArkeEnv.from_op("matmul", {"A": [64, 32], "B": [32, 64]}))
    # 2 read-only (concurrent_safe) then 1 mutating (apply_decision).
    calls = [("get_hw_profile", {}), ("analyze_compute", {}),
             ("apply_decision", {"kind": "tile", "params": {"loop": "i", "factors": [16]},
                                 "rationale": "x"})]
    batches = reg.partition_for_execution(calls)
    # First batch = the 2 concurrent_safe reads; apply_decision breaks into its own.
    assert batches[0][0][2] is True and batches[0][1][2] is True  # concurrent flag
    assert len(batches[0]) == 2
    assert batches[-1][-1][0] == "apply_decision" and batches[-1][-1][2] is False


# ── C1 + C2 via the runner ─────────────────────────────────────────────────


def _scripted_runner(cfg, scripted_tool_uses):
    """Runner whose first LLM turn emits scripted tool_uses, then stops."""
    r = LLMRunner(cfg)
    r._build_client = lambda prov: object()  # type: ignore[assignment]
    state = {"turn": 0}

    def fake_call(protocol, model, sys_p, msgs, reg):
        state["turn"] += 1
        if state["turn"] == 1:
            return ("", scripted_tool_uses, 1, 1, "")
        return ("done", [], 1, 1, "end_turn")  # second turn: stop

    r._call_llm = fake_call  # type: ignore[assignment]
    return r


def test_on_event_callback_receives_each_action():
    cfg = _cfg()
    tool_uses = [
        {"id": "1", "name": "get_hw_profile", "input": {}},
        {"id": "2", "name": "analyze_compute", "input": {}},
        {"id": "3", "name": "list_legal_actions", "input": {"top_n": 3}},
    ]
    r = _scripted_runner(cfg, tool_uses)
    seen = []
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
                     max_turns=2, on_event=lambda a: seen.append(a["tool"]))
    # All 3 read tools streamed via on_event, order preserved.
    assert seen == ["get_hw_profile", "analyze_compute", "list_legal_actions"]
    assert res.tool_calls == 3


def test_trajectory_order_preserved_with_concurrency():
    cfg = _cfg()
    tool_uses = [
        {"id": "1", "name": "get_hw_profile", "input": {}},        # concurrent
        {"id": "2", "name": "analyze_compute", "input": {}},       # concurrent
        {"id": "3", "name": "apply_decision",                       # mutating → serial
         "input": {"kind": "tile", "params": {"loop": "i", "factors": [16]}, "rationale": "x"}},
    ]
    r = _scripted_runner(cfg, tool_uses)
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]}, max_turns=2)
    steps = [(e["step"], e["tool"]) for e in res.trajectory if e.get("type") == "action"]
    assert steps == [(1, "get_hw_profile"), (2, "analyze_compute"), (3, "apply_decision")]


def test_concurrent_disabled_still_correct():
    cfg = _cfg()
    tool_uses = [
        {"id": "1", "name": "get_hw_profile", "input": {}},
        {"id": "2", "name": "analyze_compute", "input": {}},
    ]
    r = _scripted_runner(cfg, tool_uses)
    res = r.optimize(op_name="matmul", shapes={"A": [64, 32], "B": [32, 64]},
                     max_turns=2, concurrent_tools=False)
    assert res.tool_calls == 2
    steps = [e["tool"] for e in res.trajectory if e.get("type") == "action"]
    assert steps == ["get_hw_profile", "analyze_compute"]
