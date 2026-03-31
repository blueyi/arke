# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Tool base class and ToolMeta.

Inspired by Claude Code's Tool.ts — each tool self-describes its
concurrency, safety, and cost properties. The orchestrator uses these
declarations to auto-decide execution strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolMeta:
    """Declarative metadata for an ArkeEnv tool.

    The orchestrator uses these properties to decide:
    - concurrent_safe → can run in parallel with other safe tools
    - idempotent → safe to retry on failure
    - requires_compile → triggers GPU compilation (expensive)
    - mutates_strategy → modifies Strategy IR (must be serialized)
    - budget_type → which budget counter to decrement
    """
    name: str
    description: str
    concurrent_safe: bool = False
    idempotent: bool = False
    requires_compile: bool = False
    mutates_strategy: bool = False
    budget_type: str = "free"          # "free" | "decision" | "compile"
    estimated_cost: str = "cheap"      # "cheap" | "medium" | "expensive"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: dict[str, Any]
    error: str | None = None


class ArkeTool(ABC):
    """Abstract base class for all ArkeEnv tools."""

    meta: ToolMeta

    @abstractmethod
    def schema(self) -> dict:
        """Return JSON Schema for LLM tool-use (OpenAI function calling format)."""
        ...

    @abstractmethod
    async def execute(self, params: dict, env: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    def validate_params(self, params: dict, env: Any) -> ToolResult:
        """Optional: pre-execution parameter validation (V0 level)."""
        return ToolResult(success=True, data={})
