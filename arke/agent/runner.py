# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — LLM Runner.

Executes optimization sessions by calling LLMs with tool-use.
Supports Anthropic Messages API and OpenAI Chat Completions API.

Usage:
    config = load_from_openclaw()
    runner = LLMRunner(config)
    result = runner.optimize(semantic_ir, target_hw="nvidia_ampere")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from arke.agent.llm_config import LLMConfig, ModelConfig, ProviderConfig
from arke.agent.session import OptimizationSession
from arke.agent.tools_schema import get_tool_schemas
from arke.ir.semantic import SemanticIR

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a complete LLM optimization run."""
    model_used: str
    decisions: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    duration_seconds: float
    session_summary: dict[str, Any]
    trajectory: list[dict[str, Any]]
    conversation: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)


class LLMRunner:
    """Drives LLM optimization sessions via tool-use."""

    def __init__(self, config: LLMConfig, timeout: float = 300.0):
        self.config = config
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def optimize(
        self,
        semantic_ir: SemanticIR,
        target_hw: str = "nvidia_ampere",
        max_turns: int = 30,
        model_spec: str | None = None,
    ) -> RunResult:
        """Run a complete optimization session.

        Args:
            semantic_ir: The kernel to optimize
            target_hw: Hardware target
            max_turns: Max LLM conversation turns
            model_spec: Override model (default: use primary)
        """
        start = time.time()
        spec = model_spec or self.config.primary
        provider, model = self.config.get_provider_and_model(spec)

        # Create session
        session = OptimizationSession(
            semantic_ir=semantic_ir,
            target_hw=target_hw,
        )

        # Conversation state
        messages = list(session.messages)  # Start with system prompt
        # Add initial user message
        initial_msg = self._build_initial_message(session)
        messages.append({"role": "user", "content": initial_msg})

        tool_calls_total = 0
        tokens_in = 0
        tokens_out = 0
        errors: list[str] = []

        for turn in range(max_turns):
            dec_count = session.env.strategy.decision_count
            logger.info(f"Turn {turn + 1}/{max_turns}, decisions: {dec_count}")

            response = None
            for attempt in range(3):
                try:
                    response = self._call_llm(provider, model, messages)
                    break
                except httpx.ReadTimeout:
                    logger.warning(f"Timeout on attempt {attempt + 1}/3, retrying...")
                    if attempt == 2:
                        errors.append("LLM call timed out after 3 attempts")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait = min(2 ** attempt * 5, 30)
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        errors.append(f"LLM call failed: {e}")
                        break
                except Exception as e:
                    error_msg = f"LLM call failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    break

            if response is None:
                # Try fallback
                fallback_result = self._try_fallback(messages, errors)
                if fallback_result is None:
                    break
                response = fallback_result

            # Track tokens
            tokens_in += response.get("usage", {}).get("input_tokens", 0)
            tokens_out += response.get("usage", {}).get("output_tokens", 0)

            # Extract assistant message and tool calls
            assistant_msg, tool_uses = self._parse_response(response, model.api)
            messages.append(assistant_msg)

            if not tool_uses:
                # LLM stopped calling tools — optimization complete
                logger.info("LLM finished optimization (no more tool calls)")
                break

            # Execute each tool call
            tool_results = []
            for tool_use in tool_uses:
                tool_calls_total += 1
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_id = tool_use.get("id", f"call_{tool_calls_total}")

                input_str = json.dumps(
                    tool_input, ensure_ascii=False
                )[:100]
                logger.info(f"  Tool: {tool_name}({input_str})")

                result = session.run_tool(tool_name, tool_input)

                tool_results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "result": result,
                })

            # Add tool results to conversation
            tool_result_msg = self._format_tool_results(tool_results, model.api)
            if isinstance(tool_result_msg, list):
                messages.extend(tool_result_msg)
            else:
                messages.append(tool_result_msg)

            # Check if budget exhausted
            if session.budget.exhausted:
                messages.append({
                    "role": "user",
                    "content": (
                        "Budget exhausted. Please summarize your"
                        " optimization strategy and results."
                    ),
                })

            # Nudge LLM toward verify+compile after enough decisions
            decisions = session.env.strategy.decision_count
            has_verified = any(
                e.tool == "verify_correctness" for e in session.trajectory
            )
            has_compiled = any(
                e.tool == "compile_and_profile" for e in session.trajectory
            )
            if decisions >= 4 and not has_verified and turn >= 8:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have {decisions} decisions applied. "
                        "Before adding more, call"
                        " `verify_correctness()` to check accuracy, "
                        "then `compile_and_profile()` to measure GPU performance."
                    ),
                })
            elif has_verified and not has_compiled and turn >= 10:
                messages.append({
                    "role": "user",
                    "content": (
                        "Verification passed. Now call `compile_and_profile()` to measure "
                        "actual GPU performance against the cuBLAS baseline."
                    ),
                })

        duration = time.time() - start

        return RunResult(
            model_used=spec,
            decisions=session.env.strategy.decision_count,
            tool_calls=tool_calls_total,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_seconds=round(duration, 2),
            session_summary=session.summary(),
            trajectory=session.export_trajectory(),
            conversation=messages,
            errors=errors,
        )

    # ─── LLM API calls ───

    def _call_llm(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict],
        tools: bool = True,
    ) -> dict:
        """Call the LLM API.

        Args:
            tools: If False, omit tools (plain text generation).
        """
        if model.api == "anthropic-messages":
            return self._call_anthropic(
                provider, model, messages, tools=tools
            )
        elif model.api == "openai-completions":
            return self._call_openai(
                provider, model, messages, tools=tools
            )
        else:
            raise ValueError(f"Unsupported API type: {model.api}")

    def _call_anthropic(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict],
        tools: bool = True,
    ) -> dict:
        """Call Anthropic Messages API."""
        # Separate system message
        system = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                api_messages.append(msg)

        body: dict[str, Any] = {
            "model": model.id,
            "max_tokens": model.max_tokens,
            "messages": api_messages,
        }
        if system:
            body["system"] = system

        # Only include tools if requested
        if tools:
            body["tools"] = self._tools_to_anthropic()

        headers = {
            "x-api-key": provider.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = self.client.post(
            f"{provider.base_url}/messages",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def _call_openai(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict],
        tools: bool = True,
    ) -> dict:
        """Call OpenAI-compatible Chat Completions API."""
        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "max_tokens": model.max_tokens,
        }

        if tools:
            body["tools"] = get_tool_schemas()

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        response = self.client.post(
            f"{provider.base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # Normalize to Anthropic-like usage format
        usage = data.get("usage", {})
        data["usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        return data

    # ─── Response parsing ───

    def _parse_response(
        self, response: dict, api: str
    ) -> tuple[dict, list[dict]]:
        """Parse LLM response into assistant message + tool uses."""
        if api == "anthropic-messages":
            return self._parse_anthropic_response(response)
        elif api == "openai-completions":
            return self._parse_openai_response(response)
        else:
            raise ValueError(f"Unsupported API: {api}")

    def _parse_anthropic_response(
        self, response: dict
    ) -> tuple[dict, list[dict]]:
        """Parse Anthropic Messages API response."""
        content = response.get("content", [])
        tool_uses = []
        text_parts = []

        for block in content:
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_uses.append({
                    "id": block["id"],
                    "name": block["name"],
                    "input": block["input"],
                })

        # Build assistant message in Anthropic format
        assistant_msg = {"role": "assistant", "content": content}

        return assistant_msg, tool_uses

    def _parse_openai_response(
        self, response: dict
    ) -> tuple[dict, list[dict]]:
        """Parse OpenAI Chat Completions response."""
        choice = response["choices"][0]
        message = choice["message"]
        tool_uses = []

        for tc in message.get("tool_calls", []):
            tool_uses.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })

        assistant_msg = {"role": "assistant", "content": message.get("content", "")}
        if message.get("tool_calls"):
            assistant_msg["tool_calls"] = message["tool_calls"]

        return assistant_msg, tool_uses

    # ─── Tool formatting ───

    def _tools_to_anthropic(self) -> list[dict]:
        """Convert tool schemas to Anthropic format."""
        tools = []
        for t in get_tool_schemas():
            fn = t["function"]
            tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            })
        return tools

    def _format_tool_results(
        self, results: list[dict], api: str
    ) -> dict | list[dict]:
        """Format tool results for the next LLM turn.

        For Anthropic: returns a single user message with tool_result blocks.
        For OpenAI: returns a list of tool messages (one per result).
        """
        if api == "anthropic-messages":
            content = []
            for r in results:
                content.append({
                    "type": "tool_result",
                    "tool_use_id": r["id"],
                    "content": json.dumps(r["result"], ensure_ascii=False),
                })
            return {"role": "user", "content": content}
        else:
            # OpenAI format — each tool result is a separate message
            msgs = []
            for r in results:
                msgs.append({
                    "role": "tool",
                    "tool_call_id": r["id"],
                    "content": json.dumps(r["result"], ensure_ascii=False),
                })
            return msgs if msgs else [{"role": "user", "content": "No tool results."}]

    def _build_initial_message(self, session: OptimizationSession) -> str:
        """Build the initial user message."""
        from arke.agent.prompts import build_initial_user_message

        # Build summary of semantic IR
        sir = session.env.get_semantic_ir()
        nodes_str = ", ".join(
            f"{n['op']}({n['id']})" for n in sir.get("nodes", [])
        )
        params_str = ", ".join(
            f"{p['name']}[{'×'.join(str(d) for d in p['shape'])}]"
            for p in sir.get("params", [])
        )
        summary = f"Params: {params_str}\nNodes: {nodes_str}"

        analysis = session.env.analyze_compute()

        return build_initial_user_message(
            kernel_name=sir.get("kernel_id", "unknown"),
            semantic_ir_summary=summary,
            auto_analysis=analysis,
        )

    def _try_fallback(
        self, messages: list[dict], errors: list[str]
    ) -> dict | None:
        """Try fallback models when primary fails."""
        for fallback_spec in self.config.fallbacks:
            try:
                provider, model = self.config.get_provider_and_model(fallback_spec)
                logger.info(f"Trying fallback: {fallback_spec}")
                return self._call_llm(provider, model, messages)
            except Exception as e:
                errors.append(f"Fallback {fallback_spec} failed: {e}")
                continue
        return None

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> LLMRunner:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
