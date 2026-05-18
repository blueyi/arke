# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agent Tool Infrastructure (S6 Track 4).

Agent-G6-M2: ToolMeta + ArkeTool ABC + ToolRegistry
Agent-G6-CLI: compile_and_profile structured JSON output
"""

import json
import pytest

from arke.agent.tools import (
    ArkeTool, ToolMeta, ToolResult, ToolRegistry,
    BudgetType, CostLevel,
    GetHWProfileTool, AnalyzeComputeTool, BenchmarkAdviceSummaryTool,
    CompileAndProfileTool, TOOL_REGISTRY,
)


# ── ToolMeta ───────────────────────────────────────────────

class TestToolMeta:

    def test_defaults(self):
        m = ToolMeta()
        assert m.concurrent_safe is True
        assert m.idempotent is True
        assert m.requires_compile is False
        assert m.mutates_strategy is False
        assert m.budget_type == BudgetType.FREE
        assert m.cost == CostLevel.CHEAP

    def test_to_dict(self):
        m = ToolMeta(concurrent_safe=False, budget_type=BudgetType.COMPILE, cost=CostLevel.EXPENSIVE)
        d = m.to_dict()
        assert d["concurrent_safe"] is False
        assert d["budget_type"] == "compile"
        assert d["cost"] == "expensive"

    def test_frozen(self):
        m = ToolMeta()
        with pytest.raises(AttributeError):
            m.concurrent_safe = False


# ── ToolResult ─────────────────────────────────────────────

class TestToolResult:

    def test_success(self):
        r = ToolResult(success=True, data={"key": "value"})
        assert r.success
        j = json.loads(r.to_json())
        assert j["success"] is True
        assert j["data"]["key"] == "value"

    def test_error(self):
        r = ToolResult(success=False, error="something broke")
        j = json.loads(r.to_json())
        assert j["success"] is False
        assert j["error"] == "something broke"

    def test_json_roundtrip(self):
        r = ToolResult(success=True, data={"ops": 45}, warnings=["low mem"])
        j = json.loads(r.to_json())
        assert j["warnings"] == ["low mem"]


# ── Built-in Tools ─────────────────────────────────────────

class TestGetHWProfile:

    def test_execute(self):
        tool = GetHWProfileTool()
        result = tool.execute({})
        assert result.success
        assert result.data["name"] == "nvidia_ampere"
        assert result.data["compute_capability"] == [8, 6]
        assert result.data["warp_size"] == 32

    def test_meta(self):
        tool = GetHWProfileTool()
        assert tool.meta.concurrent_safe is True
        assert tool.meta.cost == CostLevel.CHEAP

    def test_schema(self):
        tool = GetHWProfileTool()
        schema = tool.to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_hw_profile"


class TestAnalyzeCompute:

    def test_matmul(self):
        tool = AnalyzeComputeTool()
        result = tool.execute({"op_name": "matmul"})
        assert result.success
        assert result.data["name"] == "matmul"
        assert result.data["category"] == "compute"
        assert "shape_rule" in result.data
        assert result.data["shape_rule"]["kind"] == "matmul_rule"
        assert result.data["template_hint"]["template_name"] == "matmul"

    def test_relu(self):
        tool = AnalyzeComputeTool()
        result = tool.execute({"op_name": "relu"})
        assert result.success
        assert "elementwise" in result.data["properties"]

    def test_unknown_op(self):
        tool = AnalyzeComputeTool()
        result = tool.execute({"op_name": "NONEXISTENT"})
        assert not result.success
        assert "Unknown op" in result.error



class TestBenchmarkAdviceSummary:

    def test_meta(self):
        tool = BenchmarkAdviceSummaryTool()
        assert tool.meta.concurrent_safe is True
        assert tool.meta.idempotent is True
        assert tool.meta.budget_type == BudgetType.FREE
        assert tool.meta.cost == CostLevel.CHEAP

    def test_schema(self):
        schema = BenchmarkAdviceSummaryTool().parameters_schema()
        assert schema["required"] == ["csv_path", "gpu_memory_mb"]
        assert schema["properties"]["csv_path"]["type"] == "string"
        assert schema["properties"]["gpu_memory_mb"]["type"] == "integer"


class TestCompileAndProfile:

    def test_matmul_structured_json(self):
        """C3.2/Agent-G6-M2: compile_and_profile on matmul returns structured JSON."""
        tool = CompileAndProfileTool()
        result = tool.execute({"op_name": "matmul"})
        assert result.success
        j = json.loads(result.to_json())
        assert j["success"] is True
        assert j["data"]["op_name"] == "matmul"
        assert j["data"]["correct"] is True
        assert isinstance(j["data"]["output_shape"], list)
        assert j["data"]["backend"] == "mock"
        assert "SSAValidation" in j["data"]["pipeline_passes"]
        assert "ShapeInference" in j["data"]["pipeline_passes"]

    def test_relu(self):
        tool = CompileAndProfileTool()
        result = tool.execute({"op_name": "relu"})
        assert result.success
        assert result.data["correct"] is True

    def test_softmax(self):
        tool = CompileAndProfileTool()
        result = tool.execute({"op_name": "softmax"})
        assert result.success
        assert result.data["correct"] is True

    def test_layernorm(self):
        tool = CompileAndProfileTool()
        result = tool.execute({"op_name": "layernorm"})
        assert result.success
        assert result.data["correct"] is True

    def test_unknown_op_fails(self):
        tool = CompileAndProfileTool()
        result = tool.execute({"op_name": "FAKE_OP"})
        assert not result.success

    def test_meta_is_expensive_serial(self):
        tool = CompileAndProfileTool()
        assert tool.meta.concurrent_safe is False
        assert tool.meta.requires_compile is True
        assert tool.meta.cost == CostLevel.EXPENSIVE
        assert tool.meta.budget_type == BudgetType.COMPILE


# ── Tool Registry ──────────────────────────────────────────

class TestToolRegistry:

    def test_default_has_3_tools(self):
        # Façade v1.0 stateless tools only — benchmark_advice_summary
        # is intentionally not in TOOL_REGISTRY (see ToolRegistry.default docstring).
        assert len(TOOL_REGISTRY) == 3
        assert "get_hw_profile" in TOOL_REGISTRY
        assert "analyze_compute" in TOOL_REGISTRY
        assert "compile_and_profile" in TOOL_REGISTRY
        assert "benchmark_advice_summary" not in TOOL_REGISTRY

    def test_get_tool(self):
        tool = TOOL_REGISTRY.get("get_hw_profile")
        assert isinstance(tool, ArkeTool)

    def test_unknown_tool(self):
        with pytest.raises(KeyError, match="Unknown tool"):
            TOOL_REGISTRY.get("nonexistent")

    def test_all_schemas(self):
        schemas = TOOL_REGISTRY.all_schemas()
        assert len(schemas) == 3
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "parameters" in s["function"]

    def test_partition_concurrent(self):
        """Concurrent partitioning: read-only tools batch together."""
        calls = [
            ("analyze_compute", {"op_name": "matmul"}),
            ("get_hw_profile", {}),
            ("compile_and_profile", {"op_name": "matmul"}),
            ("analyze_compute", {"op_name": "relu"}),
        ]
        batches = TOOL_REGISTRY.partition_for_execution(calls)
        assert len(batches) == 3
        # First batch: 2 concurrent reads
        assert len(batches[0]) == 2
        assert all(b[2] for b in batches[0])  # concurrent=True
        # Second batch: 1 serial compile
        assert len(batches[1]) == 1
        assert not batches[1][0][2]  # concurrent=False
        # Third batch: 1 concurrent read
        assert len(batches[2]) == 1
        assert batches[2][0][2]  # concurrent=True

    def test_partition_empty(self):
        assert TOOL_REGISTRY.partition_for_execution([]) == []

    def test_names(self):
        names = TOOL_REGISTRY.names()
        assert sorted(names) == [
            "analyze_compute",
            "compile_and_profile",
            "get_hw_profile",
        ]
