# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — LLM Configuration.

Reads LLM provider configuration from arke.config.yaml or OpenClaw's
agent config files. Supports Anthropic Messages API and OpenAI Completions API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str
    api: str  # "anthropic-messages" | "openai-completions" | "google-generative-ai"
    reasoning: bool = False
    context_window: int = 200000
    max_tokens: int = 8192


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    base_url: str
    api_key: str
    api: str  # "anthropic-messages" | "openai-completions" | "google-generative-ai"
    models: list[ModelConfig] = field(default_factory=list)


@dataclass
class LLMConfig:
    """Complete LLM configuration with primary + fallbacks."""
    primary: str  # "provider/model_id"
    fallbacks: list[str] = field(default_factory=list)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    def get_provider_and_model(self, spec: str) -> tuple[ProviderConfig, ModelConfig]:
        """Resolve a 'provider/model_id' spec to provider + model config."""
        parts = spec.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model spec '{spec}', expected 'provider/model_id'")
        prov_name, model_id = parts

        provider = self.providers.get(prov_name)
        if not provider:
            raise ValueError(f"Unknown provider '{prov_name}'")

        model = next((m for m in provider.models if m.id == model_id), None)
        if not model:
            raise ValueError(f"Unknown model '{model_id}' in provider '{prov_name}'")

        return provider, model

    def get_primary(self) -> tuple[ProviderConfig, ModelConfig]:
        return self.get_provider_and_model(self.primary)

    def get_all_models(self) -> list[str]:
        """Return all model specs in priority order."""
        return [self.primary] + self.fallbacks


def load_from_openclaw(
    openclaw_dir: str | Path | None = None,
    agent_id: str = "main",
) -> LLMConfig:
    """Load LLM config from OpenClaw's agent config files.

    Reads:
    - ~/.openclaw/agents/<agent_id>/agent/models.json (providers + keys)
    - ~/.openclaw/openclaw.json (primary model + fallbacks)
    """
    if openclaw_dir is None:
        openclaw_dir = Path(os.environ.get("OPENCLAW_STATE_DIR", "~/.openclaw")).expanduser()
    else:
        openclaw_dir = Path(openclaw_dir)

    # Load provider models + keys
    models_path = openclaw_dir / "agents" / agent_id / "agent" / "models.json"
    if not models_path.exists():
        raise FileNotFoundError(f"Models config not found: {models_path}")

    with open(models_path) as f:
        models_data = json.load(f)

    providers: dict[str, ProviderConfig] = {}
    for prov_name, prov_data in models_data.get("providers", {}).items():
        if not prov_data.get("apiKey"):
            continue

        models = []
        for m in prov_data.get("models", []):
            models.append(ModelConfig(
                id=m["id"],
                name=m.get("name", m["id"]),
                api=prov_data.get("api", "openai-completions"),
                reasoning=m.get("reasoning", False),
                context_window=m.get("contextWindow", 200000),
                max_tokens=m.get("maxTokens", 8192),
            ))

        providers[prov_name] = ProviderConfig(
            name=prov_name,
            base_url=prov_data.get("baseUrl", ""),
            api_key=prov_data["apiKey"],
            api=prov_data.get("api", "openai-completions"),
            models=models,
        )

    # Load primary + fallbacks from openclaw.json
    config_path = openclaw_dir / "openclaw.json"
    primary = "api-proxy-claude/claude-opus-4-6"
    fallbacks: list[str] = []

    if config_path.exists():
        # Parse JSONC (json5-like) — openclaw.json uses unquoted keys
        # For simplicity, just use the models from the agent config
        # and default to claude opus as primary
        pass

    # Try to find fallback order from the parsed config
    for agent in _load_agents_list(openclaw_dir):
        if agent.get("id") == agent_id or agent.get("default"):
            model_cfg = agent.get("model", {})
            if model_cfg.get("primary"):
                primary = model_cfg["primary"]
            if model_cfg.get("fallbacks"):
                fallbacks = model_cfg["fallbacks"]
            break

    return LLMConfig(
        primary=primary,
        fallbacks=fallbacks,
        providers=providers,
    )


def _load_agents_list(openclaw_dir: Path) -> list[dict]:
    """Load agents list from openclaw.json (with JSONC tolerance)."""
    config_path = openclaw_dir / "openclaw.json"
    if not config_path.exists():
        return []

    try:
        import json5  # type: ignore
        with open(config_path) as f:
            data = json5.load(f)
        return data.get("agents", {}).get("list", [])
    except ImportError:
        # Fallback: try parsing as regular JSON
        try:
            with open(config_path) as f:
                data = json.load(f)
            return data.get("agents", {}).get("list", [])
        except json.JSONDecodeError:
            return []


def load_from_yaml(path: str | Path) -> LLMConfig:
    """Load LLM config from arke.config.yaml."""
    import yaml  # type: ignore
    with open(path) as f:
        data = yaml.safe_load(f)

    providers = {}
    for prov_name, prov_data in data.get("providers", {}).items():
        models = []
        for m in prov_data.get("models", []):
            models.append(ModelConfig(
                id=m["id"],
                name=m.get("name", m["id"]),
                api=prov_data.get("api", "openai-completions"),
            ))
        providers[prov_name] = ProviderConfig(
            name=prov_name,
            base_url=prov_data["base_url"],
            api_key=prov_data.get("api_key", os.environ.get(f"ARKE_{prov_name.upper()}_KEY", "")),
            api=prov_data.get("api", "openai-completions"),
            models=models,
        )

    return LLMConfig(
        primary=data.get("primary", ""),
        fallbacks=data.get("fallbacks", []),
        providers=providers,
    )
