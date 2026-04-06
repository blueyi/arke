# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Tool-use Schema definitions.

Defines all tools available to the LLM agent, in OpenAI function calling format.
Compatible with Anthropic tool_use and other providers.

See docs/spec/arke-ir-spec-v1.md §9 and docs/deprecated/detailed-design-v2.1.md §1.2.
"""

from __future__ import annotations

from typing import Any

# ============================================================
# Tool Schemas (OpenAI function calling format)
# ============================================================

TOOLS: list[dict[str, Any]] = [
    # ── Tool 1: create_kernel ──
    {
        "type": "function",
        "function": {
            "name": "create_kernel",
            "description": (
                "Create a new kernel by defining its computation semantics. "
                "Describes WHAT to compute (operators, shapes, data flow), not HOW. "
                "Returns the Semantic IR and automatic analysis (flops, bottleneck, fusion opportunities)."
            ),
            "parameters": {
                "type": "object",
                "required": ["name", "params", "return_type", "computations"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Kernel name (e.g., 'fused_matmul_relu')",
                    },
                    "params": {
                        "type": "array",
                        "description": "Input tensor parameters",
                        "items": {
                            "type": "object",
                            "required": ["name", "shape", "dtype"],
                            "properties": {
                                "name": {"type": "string"},
                                "shape": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 1},
                                },
                                "dtype": {
                                    "type": "string",
                                    "enum": ["f16", "f32", "f64", "bf16", "i8", "i16", "i32", "i64"],
                                },
                                "layout": {
                                    "type": "string",
                                    "enum": ["row_major", "col_major"],
                                    "default": "row_major",
                                },
                            },
                        },
                    },
                    "return_type": {
                        "type": "object",
                        "required": ["shape", "dtype"],
                        "properties": {
                            "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                            "dtype": {"type": "string"},
                        },
                    },
                    "computations": {
                        "type": "array",
                        "description": "Computation nodes. Inputs reference param names or '@node_id' for prior nodes.",
                        "items": {
                            "type": "object",
                            "required": ["id", "op", "inputs"],
                            "properties": {
                                "id": {"type": "string"},
                                "op": {"type": "string"},
                                "inputs": {
                                    "type": "object",
                                    "description": "Named inputs: param name or '@node_id' reference",
                                    "additionalProperties": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },

    # ── Tool 2: get_hw_profile ──
    {
        "type": "function",
        "function": {
            "name": "get_hw_profile",
            "description": (
                "Get the target hardware's complete parameters: compute units, memory hierarchy, "
                "constraints, peak TFLOPS, tensor core shapes. Essential for making informed "
                "optimization decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Hardware target (e.g., 'nvidia_ampere', 'ascend_a3'). Defaults to session target.",
                    },
                },
            },
        },
    },

    # ── Tool 3: analyze_compute ──
    {
        "type": "function",
        "function": {
            "name": "analyze_compute",
            "description": (
                "Analyze the kernel's computation characteristics: total FLOPs, arithmetic intensity, "
                "bottleneck type (compute/memory bound), per-operator analysis, fusion opportunities, "
                "and suggested optimization priority order."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ── Tool 4: list_legal_actions ──
    {
        "type": "function",
        "function": {
            "name": "list_legal_actions",
            "description": (
                "List all legal optimization actions from the current state. "
                "Each action includes estimated impact (shared memory delta, parallelism, data reuse). "
                "Optionally filter by kind. Returns top candidates + total search space size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["tile", "fuse", "reorder", "parallel", "place", "vectorize", "unroll", "algorithm"],
                        "description": "Filter actions by kind. Omit to get all kinds.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max actions to return (default: 10, to save context)",
                        "default": 10,
                    },
                },
            },
        },
    },

    # ── Tool 5: apply_decision ──
    {
        "type": "function",
        "function": {
            "name": "apply_decision",
            "description": (
                "Apply one optimization decision. Auto-validated (V0) immediately. "
                "If validation fails, auto-rollbacks and returns error with guidance. "
                "Always provide a rationale explaining your reasoning."
            ),
            "parameters": {
                "type": "object",
                "required": ["kind", "params", "rationale"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["tile", "reorder", "fuse", "parallel", "place",
                                 "vectorize", "unroll", "algorithm"],
                    },
                    "params": {
                        "type": "object",
                        "description": "Decision-specific parameters (see list_legal_actions for format)",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Natural language explanation for this decision. Required.",
                    },
                },
            },
        },
    },

    # ── Tool 6: verify_correctness ──
    {
        "type": "function",
        "function": {
            "name": "verify_correctness",
            "description": (
                "Compile the current strategy and verify numerical correctness against NumPy reference. "
                "Runs 3 trials with random inputs. Reports max error and pass/fail. "
                "Cost: ~100ms-1s (moderate). Use after major strategy changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trials": {
                        "type": "integer",
                        "description": "Number of random input trials (default: 3)",
                        "default": 3,
                    },
                },
            },
        },
    },

    # ── Tool 7: compile_and_profile ──
    {
        "type": "function",
        "function": {
            "name": "compile_and_profile",
            "description": (
                "Compile and profile the current strategy on actual GPU hardware. "
                "Returns latency, TFLOPS, roofline efficiency, and comparison vs vendor baseline (cuBLAS). "
                "EXPENSIVE: ~1-5s, counts against compile budget. Use sparingly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "warmup": {
                        "type": "integer",
                        "description": "Warmup iterations (default: 5)",
                        "default": 5,
                    },
                    "runs": {
                        "type": "integer",
                        "description": "Profiling iterations (default: 20)",
                        "default": 20,
                    },
                },
            },
        },
    },

    # ── Tool 8: rollback ──
    {
        "type": "function",
        "function": {
            "name": "rollback",
            "description": "Undo the last N optimization decisions. Use when a strategy direction isn't working.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of decisions to undo (default: 1)",
                        "default": 1,
                    },
                },
            },
        },
    },

    # ── Tool 9: checkpoint ──
    {
        "type": "function",
        "function": {
            "name": "checkpoint",
            "description": "Save the current strategy state. Use before trying risky optimizations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Checkpoint name (auto-generated if omitted)",
                    },
                },
            },
        },
    },

    # ── Tool 10: restore ──
    {
        "type": "function",
        "function": {
            "name": "restore",
            "description": "Restore a previously saved checkpoint.",
            "parameters": {
                "type": "object",
                "required": ["checkpoint_id"],
                "properties": {
                    "checkpoint_id": {
                        "type": "string",
                        "description": "ID of the checkpoint to restore",
                    },
                },
            },
        },
    },
]


# ============================================================
# Tool metadata (for orchestrator)
# ============================================================

TOOL_METADATA: dict[str, dict[str, Any]] = {
    "create_kernel":       {"concurrent_safe": False, "budget_type": "free",     "cost": "cheap"},
    "get_hw_profile":      {"concurrent_safe": True,  "budget_type": "free",     "cost": "cheap"},
    "analyze_compute":     {"concurrent_safe": True,  "budget_type": "free",     "cost": "cheap"},
    "list_legal_actions":  {"concurrent_safe": True,  "budget_type": "free",     "cost": "cheap"},
    "apply_decision":      {"concurrent_safe": False, "budget_type": "decision", "cost": "cheap"},
    "verify_correctness":  {"concurrent_safe": False, "budget_type": "compile",  "cost": "medium"},
    "compile_and_profile": {"concurrent_safe": False, "budget_type": "compile",  "cost": "expensive"},
    "rollback":            {"concurrent_safe": False, "budget_type": "free",     "cost": "cheap"},
    "checkpoint":          {"concurrent_safe": False, "budget_type": "free",     "cost": "cheap"},
    "restore":             {"concurrent_safe": False, "budget_type": "free",     "cost": "cheap"},
}


# ============================================================
# Helpers
# ============================================================

def get_tool_schemas() -> list[dict[str, Any]]:
    """Get all tool schemas for LLM provider."""
    return TOOLS


def get_tool_names() -> list[str]:
    """Get all tool names."""
    return [t["function"]["name"] for t in TOOLS]


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """Get a specific tool schema by name."""
    for t in TOOLS:
        if t["function"]["name"] == name:
            return t
    return None


def export_json(path: str, indent: int = 2) -> None:
    """Export tool schemas to a JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(TOOLS, f, indent=indent)
