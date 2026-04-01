# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM config loading and runner basics."""

import os

import pytest

from arke.agent.llm_config import LLMConfig, ModelConfig, ProviderConfig, load_from_openclaw

# ============================================================
# Config Loading Tests
# ============================================================

def test_load_from_openclaw():
    """Load config from OpenClaw directory."""
    openclaw_dir = os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw"))
    config = load_from_openclaw(openclaw_dir)

    assert config.primary != ""
    assert len(config.providers) > 0


def test_config_has_claude_provider():
    """Config should have api-proxy-claude provider."""
    openclaw_dir = os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw"))
    config = load_from_openclaw(openclaw_dir)

    assert "api-proxy-claude" in config.providers
    claude = config.providers["api-proxy-claude"]
    assert claude.base_url == "https://yunwu.ai/v1"
    assert claude.api == "anthropic-messages"
    assert len(claude.models) > 0


def test_config_resolve_model():
    """Test resolving provider/model spec."""
    config = LLMConfig(
        primary="test-provider/test-model",
        providers={
            "test-provider": ProviderConfig(
                name="test-provider",
                base_url="https://example.com",
                api_key="test-key",
                api="anthropic-messages",
                models=[ModelConfig(id="test-model", name="Test", api="anthropic-messages")],
            ),
        },
    )

    provider, model = config.get_provider_and_model("test-provider/test-model")
    assert provider.name == "test-provider"
    assert model.id == "test-model"


def test_config_resolve_invalid():
    """Test resolving invalid model spec."""
    config = LLMConfig(primary="", providers={})
    with pytest.raises(ValueError):
        config.get_provider_and_model("nonexistent/model")


def test_config_get_all_models():
    """Test getting all model specs in order."""
    config = LLMConfig(
        primary="p1/m1",
        fallbacks=["p2/m2", "p3/m3"],
    )
    assert config.get_all_models() == ["p1/m1", "p2/m2", "p3/m3"]


# ============================================================
# LLM Runner Unit Tests (no API calls)
# ============================================================

def test_runner_creation():
    """Test creating a runner."""
    from arke.agent.runner import LLMRunner
    config = LLMConfig(
        primary="test/model",
        providers={
            "test": ProviderConfig(
                name="test",
                base_url="https://example.com",
                api_key="test-key",
                api="anthropic-messages",
                models=[ModelConfig(id="model", name="Test", api="anthropic-messages")],
            ),
        },
    )
    runner = LLMRunner(config)
    assert runner is not None
    runner.close()


def test_runner_tools_anthropic_format():
    """Test tool schema conversion to Anthropic format."""
    from arke.agent.runner import LLMRunner
    config = LLMConfig(primary="", providers={})
    runner = LLMRunner(config)

    tools = runner._tools_to_anthropic()
    assert len(tools) == 10
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t
    runner.close()


def test_runner_build_initial_message():
    """Test building initial optimization message."""
    from arke.agent.runner import LLMRunner
    from arke.agent.session import OptimizationSession
    from arke.ir.builder import KernelBuilder

    config = LLMConfig(primary="", providers={})
    runner = LLMRunner(config)

    b = KernelBuilder("test_mm")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    b.returns(m, [1024, 2048], "f16")
    ir = b.build()

    session = OptimizationSession(semantic_ir=ir, target_hw="nvidia_ampere")
    msg = runner._build_initial_message(session)

    assert "test_mm" in msg
    assert "1024" in msg
    runner.close()
