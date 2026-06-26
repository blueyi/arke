# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for S3 — LLMRunner provider fallback chain + retry/backoff.

Covers `LLMConfig.provider_chain`, transient-error classification, same-
provider exponential-backoff retry, and same-protocol provider failover.
No network: the LLM turn (`_call_llm`) is monkeypatched. Backoff sleep is
neutralized so the suite stays fast.
"""

from __future__ import annotations

import pytest

from arke.agent import runner as runner_mod
from arke.agent.llm_config import LLMConfig, ProviderConfig
from arke.agent.runner import LLMRunner, _is_transient


def _prov(alias: str, protocol: str = "openai") -> ProviderConfig:
    return ProviderConfig(
        alias=alias, protocol=protocol, api_key="sk-test",
        base_url="https://example/v1", default_model="m",
    )


# ── transient classification ──────────────────────────────────────────────


@pytest.mark.parametrize("msg", [
    "Request timed out", "rate limit exceeded", "HTTP 429 Too Many Requests",
    "503 Service Unavailable", "Connection reset by peer", "overloaded_error",
])
def test_transient_errors_classified(msg):
    assert _is_transient(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "401 Unauthorized", "invalid api key", "model not found", "bad request 400",
])
def test_nontransient_errors_classified(msg):
    assert _is_transient(RuntimeError(msg)) is False


# ── provider_chain ─────────────────────────────────────────────────────────


def test_provider_chain_primary_first_then_fallback():
    cfg = LLMConfig(
        primary="a",
        providers={"a": _prov("a"), "b": _prov("b"), "c": _prov("c")},
        fallback=["b", "c"],
    )
    chain = [p.alias for p in cfg.provider_chain()]
    assert chain == ["a", "b", "c"]


def test_provider_chain_dedups_and_skips_unknown():
    cfg = LLMConfig(
        primary="a",
        providers={"a": _prov("a"), "b": _prov("b")},
        fallback=["a", "b", "ghost"],  # 'a' dup, 'ghost' unknown
    )
    chain = [p.alias for p in cfg.provider_chain()]
    assert chain == ["a", "b"]


# ── retry + failover behavior ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(runner_mod.time, "sleep", lambda *_: None)


def _make_runner(cfg):
    r = LLMRunner(cfg)
    # neutralize real client construction
    r._build_client = lambda prov: object()  # type: ignore[assignment]
    return r


def test_retry_then_success_on_same_provider():
    """First two attempts raise transient; third succeeds → no failover."""
    cfg = LLMConfig(primary="a", providers={"a": _prov("a")})
    r = _make_runner(cfg)
    calls = {"n": 0}

    def fake_call(protocol, model, sys_p, msgs, reg):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("timed out")
        return ("ok", [], 10, 5, "end_turn")

    r._call_llm = fake_call  # type: ignore[assignment]
    fe: list[dict] = []
    text, tool_uses, ti, to, stop = r._call_llm_resilient(
        cfg.provider_chain(), "m", "sys", [], None, fe,
    )
    assert text == "ok" and calls["n"] == 3
    assert fe == []  # no provider failover, just retries


def test_failover_to_next_provider_records_event():
    """Provider 'a' always transient-fails → fail over to 'b' which succeeds."""
    cfg = LLMConfig(
        primary="a", providers={"a": _prov("a"), "b": _prov("b")}, fallback=["b"],
    )
    r = _make_runner(cfg)
    seen_clients = {"n": 0}

    def fake_call(protocol, model, sys_p, msgs, reg):
        # provider 'a' client built first; we detect provider via self._provider
        if r._provider.alias == "a":
            raise RuntimeError("503 overloaded")
        return ("from-b", [], 1, 1, "end_turn")

    r._call_llm = fake_call  # type: ignore[assignment]
    fe: list[dict] = []
    text, *_ = r._call_llm_resilient(cfg.provider_chain(), "m", "sys", [], None, fe)
    assert text == "from-b"
    assert len(fe) == 1
    assert fe[0]["layer"] == "provider" and fe[0]["from"] == "a" and fe[0]["to"] == "b"


def test_nontransient_aborts_immediately_no_failover():
    """A 401 on the primary aborts — no retry, no failover."""
    cfg = LLMConfig(
        primary="a", providers={"a": _prov("a"), "b": _prov("b")}, fallback=["b"],
    )
    r = _make_runner(cfg)
    calls = {"n": 0}

    def fake_call(protocol, model, sys_p, msgs, reg):
        calls["n"] += 1
        raise RuntimeError("401 Unauthorized")

    r._call_llm = fake_call  # type: ignore[assignment]
    fe: list[dict] = []
    with pytest.raises(RuntimeError, match="401"):
        r._call_llm_resilient(cfg.provider_chain(), "m", "sys", [], None, fe)
    assert calls["n"] == 1  # no retry
    assert fe == []         # no failover


def test_whole_chain_exhausted_raises():
    """Every provider transient-fails → raises last exception."""
    cfg = LLMConfig(
        primary="a", providers={"a": _prov("a"), "b": _prov("b")}, fallback=["b"],
    )
    r = _make_runner(cfg)

    def fake_call(protocol, model, sys_p, msgs, reg):
        raise RuntimeError("connection reset")

    r._call_llm = fake_call  # type: ignore[assignment]
    fe: list[dict] = []
    with pytest.raises(RuntimeError, match="connection reset"):
        r._call_llm_resilient(cfg.provider_chain(), "m", "sys", [], None, fe)
    # one failover event recorded (a → b) before final exhaustion
    assert len(fe) == 1 and fe[0]["from"] == "a" and fe[0]["to"] == "b"
