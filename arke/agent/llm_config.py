# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — LLM provider configuration (D8-F1.3, P0-A).

Resolves *which model, via which endpoint, with which credentials* the
:class:`arke.agent.runner.LLMRunner` should drive the optimization loop
with. This is the thin credential/endpoint layer beneath the Façade —
it is **Substrate**, never exposed across the 8-tool public contract.

Design goals (AI-Native lens)
-----------------------------
* **Zero hardcoded secrets.** Keys come from the environment only.
* **Provider-agnostic resolution.** A ``model_spec`` string of the form
  ``"<provider-alias>/<model-name>"`` selects a provider + model; a bare
  ``"<model-name>"`` falls back to the primary provider.
* **Anthropic-compatible first.** Phase-1 validation runs against an
  Anthropic Messages-API-compatible endpoint (the user's yunwu.ai relay
  via ``ANTHROPIC_API_KEY`` + ``ANTHROPIC_BASE_URL``). OpenAI-compatible
  endpoints are representable too (``protocol="openai"``) for future use.

Env precedence (highest first)
-------------------------------
1. Explicit kwargs to :func:`load_from_env`.
2. ``ARKE_LLM_*`` overrides (``ARKE_LLM_API_KEY`` / ``ARKE_LLM_BASE_URL``
   / ``ARKE_LLM_MODEL`` / ``ARKE_LLM_PROTOCOL``).
3. ``ANTHROPIC_API_KEY`` + ``ANTHROPIC_BASE_URL`` (the user's shell setup
   in ``~/.env.rc``; auto-loaded by non-interactive bash via ``BASH_ENV``).
4. ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL``.

``load_from_openclaw`` is kept as a backward-compatible alias (the demo
``examples/agents/agent_matmul.py`` imports that name) — it delegates to
:func:`load_from_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Protocol = Literal["anthropic", "openai"]


class LLMConfigError(RuntimeError):
    """Raised when no usable LLM provider can be resolved from the env."""


@dataclass(frozen=True)
class ProviderConfig:
    """One resolved LLM endpoint + credential + protocol."""

    alias: str
    protocol: Protocol
    api_key: str
    base_url: str | None
    default_model: str

    def redacted(self) -> dict[str, str | None]:
        """Safe-to-log view (api_key masked)."""
        return {
            "alias": self.alias,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "api_key": ("***" + self.api_key[-4:]) if self.api_key else None,
        }


@dataclass
class LLMConfig:
    """Top-level LLM configuration consumed by :class:`LLMRunner`.

    Attributes
    ----------
    primary:
        Alias of the default provider (key into ``providers``).
    providers:
        Map of provider-alias → :class:`ProviderConfig`.
    """

    primary: str
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # ── model_spec resolution ────────────────────────────────────────
    def resolve(self, model_spec: str | None = None) -> tuple[ProviderConfig, str]:
        """Resolve a ``model_spec`` into (ProviderConfig, model_name).

        ``model_spec`` forms:
          * ``"<alias>/<model>"`` → that provider + that model
          * ``"<model>"``         → primary provider + that model
          * ``None``              → primary provider + its default_model

        Raises:
            LLMConfigError: if the alias or primary provider is unknown.
        """
        if not self.providers:
            raise LLMConfigError("No LLM providers configured (check env vars).")

        if model_spec is None:
            prov = self._provider(self.primary)
            return prov, prov.default_model

        if "/" in model_spec:
            alias, model = model_spec.split("/", 1)
            # Allow the demo's "api-proxy-claude/..." alias to map onto the
            # primary anthropic provider when no exact alias match exists.
            if alias in self.providers:
                return self._provider(alias), model
            prov = self._provider(self.primary)
            return prov, model

        # bare model name → primary provider
        return self._provider(self.primary), model_spec

    def _provider(self, alias: str) -> ProviderConfig:
        if alias not in self.providers:
            raise LLMConfigError(
                f"Unknown provider alias {alias!r}. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[alias]

    def redacted(self) -> dict:
        return {
            "primary": self.primary,
            "providers": {k: v.redacted() for k, v in self.providers.items()},
        }


# ── Loaders ──────────────────────────────────────────────────────────

# Default model names per protocol. Overridable via ARKE_LLM_MODEL.
# Sonnet is the Phase-1 workhorse (fast enough for long tool-use loops;
# opus is too slow for 25-turn conversations — see agent_matmul.py note).
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_OPENAI_MODEL = "gpt-4o"


def load_from_env(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    protocol: Protocol | None = None,
) -> LLMConfig:
    """Build an :class:`LLMConfig` from environment + optional overrides.

    Resolves at least one provider, or raises :class:`LLMConfigError`.
    See module docstring for env precedence.
    """
    providers: dict[str, ProviderConfig] = {}
    primary: str | None = None

    # ARKE_LLM_* explicit overrides win — treat as the primary provider.
    ov_key = api_key or os.environ.get("ARKE_LLM_API_KEY")
    ov_base = base_url or os.environ.get("ARKE_LLM_BASE_URL")
    ov_model = model or os.environ.get("ARKE_LLM_MODEL")
    ov_proto = protocol or os.environ.get("ARKE_LLM_PROTOCOL")  # type: ignore[assignment]
    if ov_key:
        proto: Protocol = "openai" if ov_proto == "openai" else "anthropic"
        default_model = ov_model or (
            _DEFAULT_OPENAI_MODEL if proto == "openai" else _DEFAULT_ANTHROPIC_MODEL
        )
        providers["arke-llm"] = ProviderConfig(
            alias="arke-llm", protocol=proto, api_key=ov_key,
            base_url=ov_base, default_model=default_model,
        )
        primary = "arke-llm"

    # Anthropic-compatible (the user's yunwu.ai relay).
    #
    # IMPORTANT (P0-A finding, 2026-06-24): the bare ``https://yunwu.ai``
    # *Anthropic* endpoint injects a ~30K-token Claude-Code system context
    # (Skill/Glob/Agent/Bash tools) into every request, which hijacks our
    # tool-use loop. The ``/v1`` *OpenAI-compatible* endpoint is clean
    # (input_tokens ~500 vs ~30600). So when ANTHROPIC_API_KEY points at
    # yunwu.ai, we prefer routing through the clean OpenAI ``/v1`` surface
    # using the openai protocol, and register the raw anthropic endpoint
    # only as a secondary alias.
    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    yunwu_key = os.environ.get("YUNWU_API_KEY")
    anth_base = os.environ.get("ANTHROPIC_BASE_URL")
    if (anth_key or yunwu_key) and (yunwu_key or (anth_base and "yunwu" in anth_base)):
        clean_key = yunwu_key or anth_key
        providers["yunwu"] = ProviderConfig(
            alias="yunwu", protocol="openai", api_key=clean_key,  # type: ignore[arg-type]
            base_url="https://yunwu.ai/v1",
            default_model=ov_model or "claude-sonnet-4-6",
        )
        primary = primary or "yunwu"
    if anth_key:
        providers["anthropic"] = ProviderConfig(
            alias="anthropic", protocol="anthropic", api_key=anth_key,
            base_url=anth_base,
            default_model=ov_model or _DEFAULT_ANTHROPIC_MODEL,
        )
        primary = primary or "anthropic"

    # OpenAI-compatible.
    oai_key = os.environ.get("OPENAI_API_KEY")
    if oai_key:
        providers["openai"] = ProviderConfig(
            alias="openai", protocol="openai", api_key=oai_key,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            default_model=ov_model or _DEFAULT_OPENAI_MODEL,
        )
        primary = primary or "openai"

    if not providers or primary is None:
        raise LLMConfigError(
            "No LLM provider credentials found. Set one of: ARKE_LLM_API_KEY, "
            "ANTHROPIC_API_KEY (+ ANTHROPIC_BASE_URL), or OPENAI_API_KEY. "
            "On this machine the yunwu.ai relay lives in ~/.env.rc — "
            "ensure it is sourced (BASH_ENV) before running."
        )

    return LLMConfig(primary=primary, providers=providers)


def load_from_openclaw(**kwargs) -> LLMConfig:
    """Backward-compatible alias for :func:`load_from_env`.

    The original Stage-8 demo imported ``load_from_openclaw``; the loader
    is now environment-driven (provider-agnostic), so this simply
    delegates. Kept to avoid breaking ``examples/agents/agent_matmul.py``.
    """
    return load_from_env(**kwargs)
