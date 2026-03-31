# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool orchestrator (concurrency partitioning)."""

from arke.agent.tools.orchestrator import ToolCall, ToolBatch, partition_tool_calls
from arke.agent.tools.base import ArkeTool, ToolMeta, ToolResult


class MockTool(ArkeTool):
    """Mock tool for testing."""
    def __init__(self, meta: ToolMeta):
        self.meta = meta

    def schema(self) -> dict:
        return {}

    async def execute(self, params, env) -> ToolResult:
        return ToolResult(success=True, data={})


def _make_registry() -> dict[str, ArkeTool]:
    return {
        "analyze_compute": MockTool(ToolMeta(
            name="analyze_compute", description="", concurrent_safe=True)),
        "get_hw_profile": MockTool(ToolMeta(
            name="get_hw_profile", description="", concurrent_safe=True)),
        "list_legal_actions": MockTool(ToolMeta(
            name="list_legal_actions", description="", concurrent_safe=True)),
        "apply_decision": MockTool(ToolMeta(
            name="apply_decision", description="", concurrent_safe=False,
            mutates_strategy=True, budget_type="decision")),
        "compile_and_profile": MockTool(ToolMeta(
            name="compile_and_profile", description="", concurrent_safe=False,
            requires_compile=True, budget_type="compile")),
    }


def test_partition_all_concurrent():
    """All safe tools → one concurrent batch."""
    reg = _make_registry()
    calls = [
        ToolCall(name="analyze_compute", params={}),
        ToolCall(name="get_hw_profile", params={}),
        ToolCall(name="list_legal_actions", params={}),
    ]
    batches = partition_tool_calls(calls, reg)
    assert len(batches) == 1
    assert batches[0].concurrent is True
    assert len(batches[0].calls) == 3


def test_partition_all_sequential():
    """All non-safe tools → N singleton batches."""
    reg = _make_registry()
    calls = [
        ToolCall(name="apply_decision", params={}),
        ToolCall(name="compile_and_profile", params={}),
    ]
    batches = partition_tool_calls(calls, reg)
    assert len(batches) == 1  # Both non-safe, one batch
    assert batches[0].concurrent is False


def test_partition_mixed():
    """Mixed safe/non-safe → alternating batches."""
    reg = _make_registry()
    calls = [
        ToolCall(name="analyze_compute", params={}),
        ToolCall(name="get_hw_profile", params={}),
        ToolCall(name="apply_decision", params={}),
        ToolCall(name="list_legal_actions", params={}),
    ]
    batches = partition_tool_calls(calls, reg)
    assert len(batches) == 3
    assert batches[0].concurrent is True
    assert len(batches[0].calls) == 2
    assert batches[1].concurrent is False
    assert len(batches[1].calls) == 1
    assert batches[2].concurrent is True
    assert len(batches[2].calls) == 1


def test_partition_empty():
    """Empty call list → empty batches."""
    batches = partition_tool_calls([], {})
    assert batches == []
