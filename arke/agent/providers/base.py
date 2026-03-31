# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — LLM Provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | list[dict] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request."""
        ...
