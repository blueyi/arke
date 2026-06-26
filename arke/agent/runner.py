# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — LLMRunner: live LLM tool-use orchestrator (D8-F1.3, P0-A).

This is the module that finally **puts an LLM in the driver's seat** of
the Arke optimization loop. Until now ``arke optimize`` ran a
deterministic heuristic with mock profiles; the LLM-as-decision-maker
pillar of the LLM-Native thesis had never been exercised with a real
model. ``LLMRunner`` closes that gap.

What it does
------------
Given a ``SemanticIR`` (or an op name + shapes), it:

1. Builds an :class:`arke.agent.env.ArkeEnv` for the kernel.
2. Wires the locked **Façade v1.0** 8-tool registry to that env
   (``ToolRegistry.with_env``).
3. Opens a tool-use conversation with the configured provider
   (Anthropic Messages API, e.g. the yunwu.ai relay) exposing those 8
   tools as function-calling schemas.
4. Runs the agentic loop: the model thinks → calls tools → reads
   structured ``ToolResult`` JSON → iterates, until it stops calling
   tools, hits ``max_turns``, or exhausts the optimization budget.
5. Records a **real trajectory** of ``(step, tool, params, result)``
   triples and returns an :class:`OptimizeResult`.

It is **Substrate** — external agents never import this; they drive the
8 tools directly. ``LLMRunner`` is Arke's *own* in-tree agent used for
dogfooding, benchmarking (G8 Tier-2), and trajectory generation.

Design refs: docs/architecture/arke-harness.md §3 §4 §6
Stage tracker: docs/phase1/stage8-plan.md (G8 Tier-2 live LLM path)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from arke.agent.env import ArkeEnv
from arke.agent.llm_config import LLMConfig, ProviderConfig
from arke.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ── S3: transient-error classification + retry policy ────────────────────
#
# A transient error (timeout / rate-limit / 5xx / connection reset) is worth
# retrying on the same provider with exponential backoff, and — if retries are
# exhausted — failing over to the next provider in the chain. A non-transient
# error (auth, bad request, model-not-found) aborts immediately: retrying or
# failing over won't help.
_RETRYABLE_SUBSTRINGS: tuple[str, ...] = (
    "timeout", "timed out", "rate limit", "rate_limit", "429",
    "overloaded", "503", "502", "500", "connection", "reset by peer",
    "temporarily unavailable", "econnreset", "read timed out",
)
_MAX_RETRIES_PER_PROVIDER = 2      # → up to 3 attempts per provider
_BACKOFF_BASE_SECONDS = 1.5        # exp backoff: 1.5, 3.0, 6.0 …


def _is_transient(exc: Exception) -> bool:
    """Heuristic: is this exception worth a retry / provider failover?"""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


# ── System prompt: frames the LLM as Arke's optimization decision-maker ──

_SYSTEM_PROMPT = """\
You are an expert GPU-kernel optimization agent driving the Arke compiler \
toolchain. Your job is to find a high-performance, *correct* optimization \
strategy for a single AI operator on the target hardware.

You operate through a bounded tool-use protocol. You are a DECISION-MAKER, \
not a code generator: you never write kernel code. You choose among the \
legal optimization moves the compiler surfaces, and the compiler does the \
codegen + measurement.

Workflow (repeat the compile→profile→adjust cycle):
  1. get_hw_profile      — learn the hardware constraints.
  2. analyze_compute     — understand the operator (op_name given below).
  3. list_legal_actions  — see what optimization moves are legal NOW.
  4. apply_decision      — apply one move. ALWAYS include a `rationale`
                           explaining WHY (this is a hard contract).
  5. verify_correctness  — confirm the strategy still matches the reference.
  6. compile_and_profile — measure real latency on the GPU.
  7. checkpoint / rollback — explore safely; roll back regressions.

Rules:
  - Bounded action space: only apply decisions whose `kind`/`params` come
    from list_legal_actions results.
  - @rationale is mandatory on every apply_decision.
  - Budget-aware: compile/profile is expensive, but it is the ONLY way to
    know whether a strategy actually helped. Do NOT stack many
    apply_decision calls before measuring — apply 1–3 related decisions,
    then immediately verify_correctness + compile_and_profile to close the
    loop and learn from a real latency/baseline_ratio number.
  - Iterate: you MUST complete at least 3 full compile→profile→adjust
    cycles (apply → verify → profile → read the result → adjust). Each
    cycle ends with a compile_and_profile call; keep the best-performing
    correct strategy and roll back regressions.
  - When you have run ≥3 profiled cycles and further moves don't help, STOP
    calling tools and write a short final summary of the strategy you
    landed on and why.
"""


@dataclass
class OptimizeResult:
    """Outcome of an :meth:`LLMRunner.optimize` run.

    Field names match the expectations of
    ``examples/agents/agent_matmul.py``.
    """

    model_used: str
    decisions: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    session_summary: dict[str, Any] = field(default_factory=dict)
    final_message: str = ""
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_used": self.model_used,
            "decisions": self.decisions,
            "tool_calls": self.tool_calls,
            "tokens": {"in": self.tokens_in, "out": self.tokens_out},
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "trajectory": self.trajectory,
            "session_summary": self.session_summary,
            "final_message": self.final_message,
            "stop_reason": self.stop_reason,
        }


class LLMRunner:
    """Drive the Arke optimization loop with a live LLM via tool-use.

    Usage::

        with LLMRunner(config, timeout=300.0) as runner:
            result = runner.optimize(
                semantic_ir=ir, target_hw="nvidia_ampere",
                max_turns=25, model_spec="anthropic/claude-sonnet-4-20250514",
            )

    Currently supports the **Anthropic Messages API** protocol (covers the
    yunwu.ai relay). OpenAI-protocol providers raise NotImplementedError
    for now (the function-calling translation is a small future add).
    """

    def __init__(self, config: LLMConfig, *, timeout: float = 300.0) -> None:
        self.config = config
        self.timeout = timeout
        self._client: Any = None
        self._provider: ProviderConfig | None = None

    # ── context manager ──────────────────────────────────────────────
    def __enter__(self) -> LLMRunner:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client = None

    # ── client construction ──────────────────────────────────────────
    def _build_client(self, provider: ProviderConfig) -> Any:
        if provider.protocol == "anthropic":
            import anthropic
            return anthropic.Anthropic(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=self.timeout,
            )
        if provider.protocol == "openai":
            import openai
            return openai.OpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=self.timeout,
            )
        raise NotImplementedError(
            f"Protocol {provider.protocol!r} not supported by LLMRunner."
        )

    # ── op resolution from SemanticIR ─────────────────────────────────
    @staticmethod
    def _op_and_shapes_from_ir(semantic_ir: Any) -> tuple[str, dict[str, list[int]]]:
        """Extract a primary op name + input shapes from a SemanticIR/IRGraph.

        The ArkeEnv is single-op in Phase 1; for a multi-node graph we
        pick the most compute-heavy node's op (matmul/attention-like) if
        present, else the last node. Shapes come from graph inputs.
        """
        # IRGraph-like: has .nodes (list of IRNode with .op) + inputs
        nodes = getattr(semantic_ir, "nodes", None)
        shapes: dict[str, list[int]] = {}
        # input shapes
        inputs = getattr(semantic_ir, "inputs", None) or getattr(semantic_ir, "params", None)
        if isinstance(inputs, dict):
            for name, spec in inputs.items():
                shp = getattr(spec, "shape", None) or (spec.get("shape") if isinstance(spec, dict) else None)
                if shp:
                    shapes[name] = list(shp)
        elif isinstance(inputs, (list, tuple)):
            for spec in inputs:
                name = getattr(spec, "name", None) or (spec.get("name") if isinstance(spec, dict) else None)
                shp = getattr(spec, "shape", None) or (spec.get("shape") if isinstance(spec, dict) else None)
                if name and shp:
                    shapes[name] = list(shp)

        op_name = ""
        if nodes:
            # prefer a heavyweight op if present
            _HEAVY = ("matmul", "batch_matmul", "grouped_matmul", "flash_attention")
            heavy = [getattr(n, "op", "") for n in nodes if getattr(n, "op", "") in _HEAVY]
            op_name = heavy[0] if heavy else getattr(nodes[-1], "op", "")
        if not op_name:
            op_name = getattr(semantic_ir, "kernel_id", "") or getattr(semantic_ir, "name", "")
        return op_name, shapes

    # ── the loop ──────────────────────────────────────────────────────
    def optimize(
        self,
        *,
        semantic_ir: Any = None,
        op_name: str | None = None,
        shapes: dict[str, list[int]] | None = None,
        target_hw: str = "nvidia_ampere",
        max_turns: int = 30,
        model_spec: str | None = None,
    ) -> OptimizeResult:
        """Run the live-LLM optimization loop.

        Provide either ``semantic_ir`` (an IRGraph/SemanticIR) or an
        explicit ``op_name`` (+ optional ``shapes``).
        """
        provider, model = self.config.resolve(model_spec)
        self._provider = provider
        self._client = self._build_client(provider)

        if op_name is None:
            if semantic_ir is None:
                raise ValueError("Provide either semantic_ir or op_name.")
            op_name, ir_shapes = self._op_and_shapes_from_ir(semantic_ir)
            shapes = shapes or ir_shapes
        if not op_name:
            raise ValueError("Could not determine op_name to optimize.")

        env = ArkeEnv.from_op(op_name, shapes or {})
        registry = ToolRegistry.with_env(env)
        protocol = provider.protocol

        # S3: provider chain (primary first, then same-protocol fallbacks).
        # Cross-protocol failover is intentionally skipped — the message log
        # is kept in the active protocol's native shape, so switching protocol
        # mid-run would require rebuilding messages. Same-protocol siblings
        # (e.g. two OpenAI-compatible relays) fail over cleanly.
        full_chain = self.config.provider_chain(first=provider.alias)
        provider_chain = [p for p in full_chain if p.protocol == protocol]
        if not provider_chain:
            provider_chain = [provider]
        fallback_events: list[dict[str, Any]] = []

        trajectory: list[dict[str, Any]] = []
        errors: list[str] = []
        tool_calls = 0
        tokens_in = 0
        tokens_out = 0
        step = 0
        stop_reason = "max_turns"
        final_message = ""

        system_prompt = _SYSTEM_PROMPT + f"\n\nOperator under optimization: {op_name}"
        user_intro = (
            f"Optimize the `{op_name}` operator for target hardware "
            f"`{target_hw}`. Input shapes: {json.dumps(shapes or env.op_inputs)}. "
            f"Decision budget: {env.state.budget.decision_max}, "
            f"compile budget: {env.state.budget.compile_max}. "
            "Begin by inspecting the hardware and the op, then iterate."
        )

        # ``messages`` is kept in the active protocol's native shape.
        if protocol == "anthropic":
            messages: list[dict[str, Any]] = [{"role": "user", "content": user_intro}]
        else:  # openai
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_intro},
            ]

        t0 = time.time()
        for _turn in range(max_turns):
            try:
                text, tool_uses, ti, to, raw_stop = self._call_llm_resilient(
                    provider_chain, model, system_prompt, messages, registry,
                    fallback_events,
                )
            except Exception as e:  # all providers + retries exhausted
                errors.append(f"llm_call_failed: {e}")
                stop_reason = "llm_error"
                break

            tokens_in += ti
            tokens_out += to
            if text:
                final_message = text

            if not tool_uses:
                stop_reason = raw_stop or "end_turn"
                break

            # Execute each requested tool, feed results back.
            results_for_model: list[tuple[str, str, dict[str, Any]]] = []
            for tu in tool_uses:
                tool_calls += 1
                step += 1
                name = tu["name"]
                params = tu["input"] if isinstance(tu["input"], dict) else {}
                try:
                    tool = registry.get(name)
                    result = tool.execute(params)
                    result_payload = json.loads(result.to_json())
                except Exception as e:
                    result_payload = {"success": False, "error": f"{type(e).__name__}: {e}"}
                    errors.append(f"tool_{name}_failed: {e}")

                trajectory.append({
                    "type": "action", "step": step,
                    "tool": name, "params": params,
                    "result": result_payload,
                })
                results_for_model.append((tu["id"], name, result_payload))

            # Append assistant turn + tool results in the protocol's shape.
            self._append_turn(protocol, messages, text, tool_uses, results_for_model)

            # Stop early if budget exhausted.
            if env.state.budget.exhausted:
                stop_reason = "budget_exhausted"
                break

        duration = round(time.time() - t0, 2)

        session_summary = {
            "state": env.state.summary(),
            "budget": env.state.budget.to_dict(),
            "best_performance": (
                env.state.best_result.to_dict() if env.state.best_result else None
            ),
            "decision_log": [
                {"kind": d.kind, "params": d.params,
                 "rationale": (d.rationale.text if d.rationale else None)}
                for d in env.state.decision_log
            ],
            "fallback_events": fallback_events,  # S3: provider failovers, if any
        }

        return OptimizeResult(
            model_used=f"{provider.alias}/{model}",
            decisions=len(env.state.decision_log),
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_seconds=duration,
            errors=errors,
            trajectory=trajectory,
            session_summary=session_summary,
            final_message=final_message,
            stop_reason=stop_reason,
        )

    # ── S3: resilient call (retry + provider failover) ───────────────
    def _call_llm_resilient(
        self,
        provider_chain: list[ProviderConfig],
        model: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
        fallback_events: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], int, int, str]:
        """One LLM turn with same-provider retry + provider failover.

        Walks ``provider_chain`` (all same protocol). For each provider, makes
        up to ``_MAX_RETRIES_PER_PROVIDER + 1`` attempts with exponential
        backoff on transient errors. A non-transient error aborts immediately.
        When a provider is abandoned for the next one, records a structured
        ``fallback{layer:"provider"}`` entry. Raises the last exception only if
        every provider in the chain is exhausted.
        """
        last_exc: Exception | None = None
        for pi, prov in enumerate(provider_chain):
            # (Re)build the client when switching providers.
            if self._provider is not prov or self._client is None:
                self._provider = prov
                self._client = self._build_client(prov)
            for attempt in range(_MAX_RETRIES_PER_PROVIDER + 1):
                try:
                    return self._call_llm(prov.protocol, model, system_prompt, messages, registry)
                except Exception as e:  # noqa: BLE001 — classify below
                    last_exc = e
                    if not _is_transient(e):
                        raise  # auth / bad-request / model-not-found → no point retrying
                    if attempt < _MAX_RETRIES_PER_PROVIDER:
                        delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
                        logger.warning(
                            "transient LLM error on %s (attempt %d/%d): %s — backing off %.1fs",
                            prov.alias, attempt + 1, _MAX_RETRIES_PER_PROVIDER + 1, e, delay,
                        )
                        time.sleep(delay)
                    # else: retries exhausted for this provider → fall through to next
            # Provider exhausted; record failover if there's a next one.
            if pi + 1 < len(provider_chain):
                nxt = provider_chain[pi + 1]
                fallback_events.append({
                    "layer": "provider", "from": prov.alias, "to": nxt.alias,
                    "reason": f"{type(last_exc).__name__}: {last_exc}",
                })
                logger.warning("failing over provider %s → %s", prov.alias, nxt.alias)
        # Whole chain exhausted.
        raise last_exc if last_exc else RuntimeError("all providers exhausted")

    # ── protocol adapters ────────────────────────────────────────────
    #
    # Both adapters return / consume a normalized tool_use shape:
    #   tool_use = {"id": str, "name": str, "input": dict}
    # so the optimize() loop stays protocol-agnostic.

    def _call_llm(
        self,
        protocol: str,
        model: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> tuple[str, list[dict[str, Any]], int, int, str]:
        """One LLM turn. Returns (text, tool_uses, tokens_in, tokens_out, stop)."""
        if protocol == "anthropic":
            resp = self._client.messages.create(
                model=model, max_tokens=1500, system=system_prompt,
                tools=self._anthropic_tool_schemas(registry), messages=messages,
            )
            text = ""
            tool_uses: list[dict[str, Any]] = []
            for block in resp.content:
                if block.type == "text":
                    text = block.text
                elif block.type == "tool_use":
                    tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
            return (
                text, tool_uses,
                getattr(resp.usage, "input_tokens", 0),
                getattr(resp.usage, "output_tokens", 0),
                resp.stop_reason or "",
            )

        # openai protocol
        resp = self._client.chat.completions.create(
            model=model, max_tokens=1500,
            tools=self._openai_tool_schemas(registry),
            messages=messages,
        )
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_uses = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_uses.append({"id": tc.id, "name": tc.function.name, "input": args})
        usage = resp.usage
        return (
            text, tool_uses,
            getattr(usage, "prompt_tokens", 0) if usage else 0,
            getattr(usage, "completion_tokens", 0) if usage else 0,
            choice.finish_reason or "",
        )

    @staticmethod
    def _append_turn(
        protocol: str,
        messages: list[dict[str, Any]],
        text: str,
        tool_uses: list[dict[str, Any]],
        results: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Append the assistant turn + tool results in the protocol's shape."""
        if protocol == "anthropic":
            assistant_blocks: list[dict[str, Any]] = []
            if text:
                assistant_blocks.append({"type": "text", "text": text})
            for tu in tool_uses:
                assistant_blocks.append({
                    "type": "tool_use", "id": tu["id"],
                    "name": tu["name"], "input": tu["input"],
                })
            messages.append({"role": "assistant", "content": assistant_blocks})
            tool_result_content = [
                {"type": "tool_result", "tool_use_id": tid,
                 "content": json.dumps(payload, default=str)}
                for tid, _name, payload in results
            ]
            messages.append({"role": "user", "content": tool_result_content})
            return

        # openai protocol
        messages.append({
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {"id": tu["id"], "type": "function",
                 "function": {"name": tu["name"], "arguments": json.dumps(tu["input"])}}
                for tu in tool_uses
            ],
        })
        for tid, _name, payload in results:
            messages.append({
                "role": "tool", "tool_call_id": tid,
                "content": json.dumps(payload, default=str),
            })

    # ── tool schema translation ──────────────────────────────────────
    @staticmethod
    def _anthropic_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
        """Translate the 8 Façade tools into Anthropic tool-use schemas."""
        schemas = []
        for name in registry.names():
            tool = registry.get(name)
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters_schema(),
            })
        return schemas

    @staticmethod
    def _openai_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
        """Translate the 8 Façade tools into OpenAI function-calling schemas."""
        return [registry.get(name).to_function_schema() for name in registry.names()]
