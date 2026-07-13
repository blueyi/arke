# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BYOK YAML LLM config (Hermes-compatible scheme)."""

import pytest

from arke.agent.llm_config import (
    LLMConfig, LLMConfigError, ProviderConfig,
    load_from_yaml, load_config, _api_mode_to_protocol,
)

yaml = pytest.importorskip("yaml")


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


BASIC = """
model:
  default: claude-sonnet-4-6
  provider: yunwu-claude
providers:
  yunwu-claude:
    base_url: https://yunwu.ai/v1
    key_env: TEST_YUNWU_CLAUDE_KEY
    api_mode: chat_completions
    default_model: claude-opus-4-8
    models: [claude-sonnet-4-6, claude-opus-4-8]
  openai:
    base_url: https://api.openai.com/v1
    key_env: TEST_OPENAI_KEY
    api_mode: chat_completions
    default_model: gpt-4o
"""


class TestApiModeMapping:
    def test_chat_completions_is_openai(self):
        assert _api_mode_to_protocol("chat_completions") == "openai"
        assert _api_mode_to_protocol("responses") == "openai"

    def test_anthropic_modes(self):
        assert _api_mode_to_protocol("anthropic") == "anthropic"
        assert _api_mode_to_protocol("messages") == "anthropic"

    def test_default_openai(self):
        assert _api_mode_to_protocol(None) == "openai"


class TestLoadFromYaml:
    def test_byok_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "sk-yunwu-xyz")
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-openai-abc")
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        cfg = load_from_yaml(p)
        assert cfg.primary == "yunwu-claude"
        assert set(cfg.providers) == {"yunwu-claude", "openai"}
        assert cfg.providers["yunwu-claude"].api_key == "sk-yunwu-xyz"
        assert cfg.providers["yunwu-claude"].protocol == "openai"

    def test_top_level_default_model_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.setenv("TEST_OPENAI_KEY", "k2")
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        cfg = load_from_yaml(p)
        # model.default = claude-sonnet-4-6 overrides provider default_model
        assert cfg.providers["yunwu-claude"].default_model == "claude-sonnet-4-6"

    def test_models_allowlist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.setenv("TEST_OPENAI_KEY", "k2")
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        cfg = load_from_yaml(p)
        prov = cfg.providers["yunwu-claude"]
        assert prov.allows("claude-sonnet-4-6")
        assert not prov.allows("some-unlisted-model")
        # openai provider has no models list → allows anything
        assert cfg.providers["openai"].allows("anything")

    def test_provider_without_key_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        cfg = load_from_yaml(p)
        assert set(cfg.providers) == {"yunwu-claude"}  # openai skipped (no key)

    def test_fallback_auto_populated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.setenv("TEST_OPENAI_KEY", "k2")
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        cfg = load_from_yaml(p)
        assert cfg.fallback == ["openai"]

    def test_no_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_YUNWU_CLAUDE_KEY", raising=False)
        monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        with pytest.raises(LLMConfigError, match="resolvable key"):
            load_from_yaml(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(LLMConfigError, match="No Arke LLM YAML config"):
            load_from_yaml(tmp_path / "does_not_exist.yaml")

    def test_env_var_config_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.setenv("TEST_OPENAI_KEY", "k2")
        p = _write(tmp_path / "custom.yaml", BASIC)
        monkeypatch.setenv("ARKE_LLM_CONFIG", p)
        cfg = load_from_yaml()  # no explicit path → uses ARKE_LLM_CONFIG
        assert cfg.primary == "yunwu-claude"

    def test_resolve_model_spec(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.setenv("TEST_OPENAI_KEY", "k2")
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        cfg = load_from_yaml(p)
        prov, model = cfg.resolve("openai/gpt-4o")
        assert prov.alias == "openai" and model == "gpt-4o"


class TestLoadConfig:
    def test_prefers_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_YUNWU_CLAUDE_KEY", "k")
        monkeypatch.setenv("TEST_OPENAI_KEY", "k2")
        p = _write(tmp_path / "arke_llm.yaml", BASIC)
        monkeypatch.setenv("ARKE_LLM_CONFIG", p)
        cfg = load_config()
        assert cfg.primary == "yunwu-claude"

    def test_falls_back_to_env(self, tmp_path, monkeypatch):
        # No YAML anywhere → env path. Point config search at empty dir.
        monkeypatch.delenv("ARKE_LLM_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ARKE_LLM_API_KEY", "sk-env")
        monkeypatch.setenv("ARKE_LLM_BASE_URL", "https://yunwu.ai/v1")
        monkeypatch.setenv("ARKE_LLM_PROTOCOL", "openai")
        # Ensure ~/.arke/llm.yaml isn't present in the sandbox HOME
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = load_config()
        assert "arke-llm" in cfg.providers
