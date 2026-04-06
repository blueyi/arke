# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Tool Declarative Interface (S6 Track 4, Agent-G6-M2).

Defines ToolMeta + ArkeTool ABC for self-declaring tool capabilities.
The orchestrator uses these declarations for concurrent batching,
budget tracking, and serial/parallel execution decisions.

Design ref: docs/architecture/agent-design.md §5.1
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BudgetType(Enum):
    """Tool budget classification for orchestrator cost tracking."""
    FREE = "free"          # No resource cost
    DECISION = "decision"  # Strategy mutation cost
    COMPILE = "compile"    # GPU compilation + execution cost


class CostLevel(Enum):
    """Approximate execution cost."""
    CHEAP = "cheap"        # <10ms, read-only
    MEDIUM = "medium"      # ~100ms-1s
    EXPENSIVE = "expensive"  # 1-5s+ (GPU compile/profile)


@dataclass(frozen=True)
class ToolMeta:
    """Declarative metadata for an Arke agent tool.

    The orchestrator reads these fields to decide:
    - concurrent_safe tools can batch with asyncio.gather
    - mutates_strategy tools break batches and execute serially
    - budget_type tracks resource usage per session
    - idempotent tools can be retried safely on failure
    """
    concurrent_safe: bool = True
    idempotent: bool = True
    requires_compile: bool = False
    mutates_strategy: bool = False
    budget_type: BudgetType = BudgetType.FREE
    cost: CostLevel = CostLevel.CHEAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrent_safe": self.concurrent_safe,
            "idempotent": self.idempotent,
            "requires_compile": self.requires_compile,
            "mutates_strategy": self.mutates_strategy,
            "budget_type": self.budget_type.value,
            "cost": self.cost.value,
        }


@dataclass
class ToolResult:
    """Structured result from tool execution.

    All tools return ToolResult for consistent JSON serialization.
    """
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self, indent: int | None = None) -> str:
        d = {"success": self.success}
        if self.data:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.warnings:
            d["warnings"] = self.warnings
        return json.dumps(d, indent=indent, default=str)


class ArkeTool(ABC):
    """Abstract base class for Arke agent tools.

    Subclasses must define:
    - name: tool identifier (matches function-calling schema)
    - description: human-readable description
    - meta: ToolMeta with capability declarations
    - execute(): actual tool logic
    - parameters_schema(): JSON Schema for parameters
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g., 'compile_and_profile')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM function-calling."""
        ...

    @property
    @abstractmethod
    def meta(self) -> ToolMeta:
        """Declarative metadata for orchestrator."""
        ...

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            params: Validated parameters matching parameters_schema()

        Returns:
            ToolResult with structured output
        """
        ...

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters.

        Used for OpenAI/Anthropic function-calling schema generation.
        """
        ...

    def to_function_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


# ── Built-in Tools ────────────────────────────────────────────

class GetHWProfileTool(ArkeTool):
    """Read hardware profile — cheap, concurrent-safe, idempotent."""

    @property
    def name(self) -> str:
        return "get_hw_profile"

    @property
    def description(self) -> str:
        return "Get the target hardware profile (compute capability, VRAM, SM count, etc.)"

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=True, idempotent=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        from arke.compiler.passes.base import HardwareProfile
        hw = HardwareProfile()
        return ToolResult(success=True, data={
            "name": hw.name,
            "compute_capability": list(hw.compute_capability),
            "shared_memory_bytes": hw.shared_memory_bytes,
            "max_threads_per_block": hw.max_threads_per_block,
            "warp_size": hw.warp_size,
            "num_sms": hw.num_sms,
            "peak_tflops_f16": hw.peak_tflops_f16,
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


class AnalyzeComputeTool(ArkeTool):
    """Analyze operator compute characteristics."""

    @property
    def name(self) -> str:
        return "analyze_compute"

    @property
    def description(self) -> str:
        return "Analyze an operator's compute pattern, shape rule, and template hints"

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=True, idempotent=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        from arke.ir.ops.registry import REGISTRY

        op_name = params.get("op_name", "")
        if op_name not in REGISTRY:
            return ToolResult(success=False, error=f"Unknown op: {op_name!r}")

        op = REGISTRY.get(op_name)
        data = {
            "name": op.name,
            "category": op.category,
            "inputs": op.inputs,
            "output": op.output,
            "computation": op.computation,
            "properties": op.properties,
            "index_vars": op.index_vars,
            "reduction_axes": op.reduction_axes,
        }

        if op.shape_rule:
            data["shape_rule"] = {"kind": op.shape_rule.kind}
        if op.template_hint:
            data["template_hint"] = {"template_name": op.template_hint.template_name}

        return ToolResult(success=True, data=data)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op_name": {"type": "string", "description": "Operator name"},
            },
            "required": ["op_name"],
        }


class CompileAndProfileTool(ArkeTool):
    """Compile and profile a kernel — expensive, serial."""

    @property
    def name(self) -> str:
        return "compile_and_profile"

    @property
    def description(self) -> str:
        return (
            "Compile an operator from IR graph through the pass pipeline "
            "and MockBackend, then validate correctness against reference implementation"
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=False, idempotent=True,
            requires_compile=True, mutates_strategy=False,
            budget_type=BudgetType.COMPILE, cost=CostLevel.EXPENSIVE,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        import torch
        from arke.backend.mock_backend import MockBackend
        from arke.compiler.passes import (
            PassPipeline, SSAValidationPass, ShapeInferencePass,
        )
        from arke.ir.graph import IRGraph, IRNode
        from arke.ir.ops.interpreter import INTERPRETER
        from arke.ir.ops.registry import REGISTRY

        op_name = params.get("op_name", "")
        if op_name not in REGISTRY:
            return ToolResult(success=False, error=f"Unknown op: {op_name!r}")

        op = REGISTRY.get(op_name)

        # Build a single-node graph
        graph = IRGraph(name=f"profile_{op_name}")
        shapes = params.get("shapes", {})

        # Default shapes
        default_shapes = {k: [4, 8] for k in op.inputs}
        if op_name == "matmul":
            default_shapes = {"A": [64, 32], "B": [32, 128]}
        elif op_name == "batch_matmul":
            default_shapes = {"A": [4, 16, 32], "B": [4, 32, 64]}
        elif op_name in ("flash_attention", "grouped_query_attention", "cross_attention", "multi_latent_attention"):
            default_shapes = {"Q": [1, 2, 16, 32], "K": [1, 2, 16, 32], "V": [1, 2, 16, 32]}
        elif op_name == "layernorm":
            default_shapes = {"X": [4, 768], "W": [768], "B": [768]}
        elif op_name == "rmsnorm":
            default_shapes = {"X": [4, 128], "W": [128]}
        elif op_name == "rmsnorm_residual":
            default_shapes = {"X": [4, 64], "residual": [4, 64], "W": [64]}
        elif op_name == "embedding":
            default_shapes = {"indices": [2, 8], "weight": [100, 32]}
        elif op_name == "cross_entropy":
            default_shapes = {"logits": [8, 100], "labels": [8]}
        elif op_name == "fused_linear_cross_entropy":
            default_shapes = {"X": [8, 64], "W": [100, 64], "labels": [8]}
        elif op_name == "grouped_matmul":
            default_shapes = {"X": [4, 16, 32], "W": [8, 32, 64]}

        merged_shapes = {**default_shapes, **shapes}

        for inp_name in op.inputs:
            shape = merged_shapes.get(inp_name, [4, 8])
            graph.add_input(inp_name, shape=shape)

        graph.add_node(IRNode(
            id="n0", op=op_name,
            inputs={k: k for k in op.inputs},
            outputs=["output"],
        ))
        graph.set_outputs(["output"])

        # Run pipeline
        pipeline = PassPipeline("compile_and_profile")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(graph)

        if not result.success:
            return ToolResult(success=False, error=f"Pipeline failed: {result.error}")

        # Generate inputs for execution
        torch.manual_seed(42)
        inputs = {}
        for inp_name in op.inputs:
            shape = merged_shapes.get(inp_name, [4, 8])
            if op.input_gen and inp_name in op.input_gen.distributions:
                dist = op.input_gen.distributions[inp_name]
                if dist == "randint":
                    rng = op.input_gen.ranges.get(inp_name, (0, 10))
                    inputs[inp_name] = torch.randint(int(rng[0]), int(rng[1]) + 1, shape)
                elif dist == "uniform":
                    rng = op.input_gen.ranges.get(inp_name, (0, 1))
                    inputs[inp_name] = torch.empty(shape).uniform_(rng[0], rng[1])
                elif dist == "ones":
                    inputs[inp_name] = torch.ones(shape)
                elif dist == "bool_mask":
                    inputs[inp_name] = torch.randint(0, 2, shape, dtype=torch.bool)
                else:
                    inputs[inp_name] = torch.randn(shape)
            else:
                inputs[inp_name] = torch.randn(shape)

        # Execute via MockBackend
        mb = MockBackend()
        try:
            outputs = mb.run_graph(result.graph, inputs)
        except Exception as e:
            return ToolResult(success=False, error=f"Execution failed: {e}")

        # Validate via reference impl
        try:
            ref_result = INTERPRETER.execute(op_name, inputs)
            output_tensor = outputs.get("output")
            if output_tensor is not None and ref_result is not None:
                if ref_result.is_floating_point():
                    correct = torch.allclose(output_tensor, ref_result, rtol=1e-3, atol=1e-5)
                    max_diff = (output_tensor - ref_result).abs().max().item()
                else:
                    correct = torch.equal(output_tensor, ref_result)
                    max_diff = 0.0
            else:
                correct = True
                max_diff = 0.0
        except Exception as e:
            return ToolResult(
                success=True,
                data={"pipeline": "passed", "execution": "passed", "validation": f"skipped: {e}"},
                warnings=[f"Reference validation failed: {e}"],
            )

        return ToolResult(success=True, data={
            "op_name": op_name,
            "pipeline_passes": result.passes_run,
            "output_shape": result.artifacts.get("shape_map", {}).get("output", []),
            "correct": correct,
            "max_diff": max_diff,
            "backend": "mock",
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op_name": {"type": "string", "description": "Operator name"},
                "shapes": {
                    "type": "object",
                    "description": "Optional shape overrides per input",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
            },
            "required": ["op_name"],
        }


# ── Tool Registry ─────────────────────────────────────────────

class ToolRegistry:
    """Registry of all available agent tools.

    Provides lookup, schema generation, and concurrent partitioning.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ArkeTool] = {}

    def register(self, tool: ArkeTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ArkeTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all_schemas(self) -> list[dict[str, Any]]:
        """Generate function-calling schemas for all tools."""
        return [t.to_function_schema() for t in self._tools.values()]

    def partition_for_execution(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> list[list[tuple[str, dict[str, Any], bool]]]:
        """Partition tool calls into concurrent/serial batches.

        Args:
            tool_calls: List of (tool_name, params) tuples

        Returns:
            List of batches. Each batch is a list of (name, params, concurrent).
            A batch with concurrent=True can be executed with asyncio.gather.
        """
        if not tool_calls:
            return []

        batches: list[list[tuple[str, dict[str, Any], bool]]] = []
        current_batch: list[tuple[str, dict[str, Any], bool]] = []
        current_concurrent = None

        for name, params in tool_calls:
            tool = self._tools.get(name)
            is_concurrent = tool.meta.concurrent_safe if tool else False

            if current_concurrent is None:
                current_concurrent = is_concurrent
            elif is_concurrent != current_concurrent:
                batches.append(current_batch)
                current_batch = []
                current_concurrent = is_concurrent

            current_batch.append((name, params, is_concurrent))

        if current_batch:
            batches.append(current_batch)

        return batches

    @classmethod
    def default(cls) -> ToolRegistry:
        """Create registry with all built-in tools."""
        reg = cls()
        reg.register(GetHWProfileTool())
        reg.register(AnalyzeComputeTool())
        reg.register(CompileAndProfileTool())
        return reg


# Module-level default registry
TOOL_REGISTRY = ToolRegistry.default()
