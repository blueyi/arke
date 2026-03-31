# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Tool concurrency orchestrator.

Borrows the partition algorithm from Claude Code's toolOrchestration.ts:
consecutive concurrent-safe tools batch together; non-safe tools serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncGenerator

from arke.agent.tools.base import ArkeTool, ToolResult


@dataclass
class ToolCall:
    """A pending tool invocation from the LLM."""
    name: str
    params: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolBatch:
    """A batch of tool calls to execute together."""
    calls: list[ToolCall]
    concurrent: bool


def partition_tool_calls(
    calls: list[ToolCall],
    tool_registry: dict[str, ArkeTool],
) -> list[ToolBatch]:
    """Partition tool calls into concurrent/sequential batches.

    Algorithm (from Claude Code toolOrchestration.ts):
      Consecutive concurrent_safe tools → one concurrent batch
      Non-safe tool → singleton sequential batch
      Repeat

    Example:
      [analyze, get_hw, apply_decision, list_legal]
      → Batch([analyze, get_hw], concurrent=True)
      → Batch([apply_decision], concurrent=False)
      → Batch([list_legal], concurrent=True)
    """
    if not calls:
        return []

    batches: list[ToolBatch] = []
    current: list[ToolCall] = []
    current_safe = True

    for call in calls:
        tool = tool_registry.get(call.name)
        is_safe = tool.meta.concurrent_safe if tool else False

        if not current:
            current = [call]
            current_safe = is_safe
        elif is_safe == current_safe:
            current.append(call)
        else:
            batches.append(ToolBatch(calls=current, concurrent=current_safe))
            current = [call]
            current_safe = is_safe

    if current:
        batches.append(ToolBatch(calls=current, concurrent=current_safe))

    return batches
