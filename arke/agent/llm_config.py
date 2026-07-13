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
from pathlib import Path
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
    models: tuple[str, ...] = ()   # BYOK allow-list; empty = any model permitted

    def allows(self, model: str) -> bool:
        """Whether ``model`` is permitted by this provider's allow-list.

        An empty ``models`` tuple means no restriction (any model allowed) —
        preserves backward-compatible behavior for env-derived providers.
        """
        return not self.models or model in self.models

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
    fallback:
        Ordered list of provider aliases to try, in order, when the
        active provider raises a *transient* error (timeout / 429 / 5xx /
        connection). S3 (2026-06-26): empty by default (no behavior change);
        when populated, :class:`LLMRunner` walks this chain before giving up
        and degrading to the heuristic floor. The primary provider is always
        tried first regardless of whether it appears in this list.
    """

    primary: str
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    fallback: list[str] = field(default_factory=list)

    def provider_chain(self, first: str | None = None) -> list[ProviderConfig]:
        """Return the ordered provider chain to try (S3).

        Starts with ``first`` (or ``primary``), then appends each alias in
        ``fallback`` that exists and isn't already in the chain. Unknown
        aliases are skipped silently (robust to stale config).
        """
        chain: list[ProviderConfig] = []
        seen: set[str] = set()
        head = first or self.primary
        for alias in [head, *self.fallback]:
            if alias in self.providers and alias not in seen:
                chain.append(self.providers[alias])
                seen.add(alias)
        return chain

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

    # S3 (2026-06-26): auto-populate the fallback chain from every other
    # resolved provider (deterministic order: anthropic-clean yunwu, raw
    # anthropic, openai), so a transient failure of the primary degrades to a
    # sibling provider before hitting the heuristic floor. Primary excluded
    # (provider_chain always tries it first).
    fallback = [a for a in providers if a != primary]

    return LLMConfig(primary=primary, providers=providers, fallback=fallback)


def _api_mode_to_protocol(api_mode: str | None) -> Protocol:
    """Map a Hermes-style ``api_mode`` to an Arke protocol.

    ``chat_completions`` / ``responses`` → ``openai``; ``anthropic`` /
    ``messages`` → ``anthropic``. Defaults to ``openai`` (the clean surface).
    """
    m = (api_mode or "").lower()
    if m in ("anthropic", "messages"):
        return "anthropic"
    return "openai"


def load_from_yaml(path: str | os.PathLike | None = None) -> LLMConfig:
    """Build an :class:`LLMConfig` from a Hermes-compatible BYOK YAML file.

    This implements the same model-configuration scheme Hermes uses, so a user
    can Bring Your Own Key via a config file instead of hard-wired env vars.

    Schema (Hermes-compatible)::

        model:
          default: claude-sonnet-4-6      # default model name
          provider: yunwu-claude          # default provider alias (key into providers)
        providers:
          yunwu-claude:
            base_url: https://yunwu.ai/v1
            key_env: ARKE_YUNWU_CLAUDE_API_KEY   # BYOK: env var holding the key
            api_mode: chat_completions           # → openai protocol
            default_model: claude-sonnet-4-6
            models: [claude-sonnet-4-6, claude-opus-4-8]   # allow-list (optional)
            fallback: [openai, yunwu-all]        # optional per-config fallback order

    Key resolution (BYOK): each provider names ``key_env``; the actual secret is
    read from that environment variable at load time. A literal ``api_key`` field
    is also honored (discouraged — keeps secrets out of the file by default).

    Search order when ``path`` is None:
      1. ``$ARKE_LLM_CONFIG``
      2. ``./arke_llm.yaml`` (cwd)
      3. ``~/.arke/llm.yaml``

    Raises :class:`LLMConfigError` if no file is found or no provider resolves.
    """
    import yaml  # local import — keeps PyYAML optional for env-only users

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("ARKE_LLM_CONFIG")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path.cwd() / "arke_llm.yaml")
        candidates.append(Path.home() / ".arke" / "llm.yaml")

    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        raise LLMConfigError(
            "No Arke LLM YAML config found. Looked in: "
            + ", ".join(str(p) for p in candidates)
        )

    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    model_sec = doc.get("model", {}) or {}
    providers_sec = doc.get("providers", {}) or {}
    if not providers_sec:
        raise LLMConfigError(f"{cfg_path}: no 'providers' section.")

    providers: dict[str, ProviderConfig] = {}
    for alias, pc in providers_sec.items():
        pc = pc or {}
        key_env = pc.get("key_env")
        api_key = pc.get("api_key") or (os.environ.get(key_env) if key_env else None)
        if not api_key:
            # Skip providers whose key isn't present in the env — a user may
            # declare several and only have keys for some (Hermes behavior).
            continue
        default_model = pc.get("default_model") or pc.get("model") or ""
        models = tuple(pc.get("models", []) or ())
        if not default_model and models:
            default_model = models[0]
        providers[alias] = ProviderConfig(
            alias=alias,
            protocol=_api_mode_to_protocol(pc.get("api_mode") or pc.get("protocol")),
            api_key=api_key,
            base_url=pc.get("base_url"),
            default_model=default_model,
            models=models,
        )

    if not providers:
        raise LLMConfigError(
            f"{cfg_path}: no provider had a resolvable key. Set the key_env "
            "environment variables (BYOK)."
        )

    primary = model_sec.get("provider")
    if primary and primary not in providers:
        # Named default provider has no key — fall back to any resolved one.
        primary = None
    primary = primary or next(iter(providers))

    # Optional: a top-level or per-primary fallback list; else all-others.
    fallback = model_sec.get("fallback") or providers_sec.get(primary, {}).get("fallback")
    if fallback:
        fallback = [a for a in fallback if a in providers and a != primary]
    else:
        fallback = [a for a in providers if a != primary]

    # A top-level default model overrides the primary provider's default.
    default_model = model_sec.get("default")
    if default_model and providers[primary].default_model != default_model:
        p = providers[primary]
        providers[primary] = ProviderConfig(
            alias=p.alias, protocol=p.protocol, api_key=p.api_key,
            base_url=p.base_url, default_model=default_model, models=p.models,
        )

    return LLMConfig(primary=primary, providers=providers, fallback=fallback)


def load_config(path: str | os.PathLike | None = None) -> LLMConfig:
    """Unified loader: prefer a BYOK YAML config, fall back to env vars.

    Resolution order:
      1. If a YAML config is found (explicit ``path``, ``$ARKE_LLM_CONFIG``,
         ``./arke_llm.yaml``, or ``~/.arke/llm.yaml``) → :func:`load_from_yaml`.
      2. Otherwise → :func:`load_from_env`.

    This is the recommended entry point; it gives users Hermes-style BYOK via a
    config file while preserving the zero-config env-var path.
    """
    try:
        return load_from_yaml(path)
    except (LLMConfigError, ImportError):
        return load_from_env()


def load_from_openclaw(**kwargs) -> LLMConfig:
    """Backward-compatible alias for :func:`load_from_env`.

    The original Stage-8 demo imported ``load_from_openclaw``; the loader
    is now environment-driven (provider-agnostic), so this simply
    delegates. Kept to avoid breaking ``examples/agents/agent_matmul.py``.
    """
    return load_from_env(**kwargs)
