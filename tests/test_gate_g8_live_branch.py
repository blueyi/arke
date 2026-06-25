# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the G8 Tier-2 live-LLM liveness gate branch (D1=c).

These tests verify the SAFETY property: the live branch skips gracefully
(returns PASS) when prerequisites (GPU / LLM credentials) are absent, so the
gate stays green on CI/CPU. The substantive live-loop assertion is exercised
by `python -m benchmarks.gate G8` on a GPU host with credentials.
"""

from __future__ import annotations

import benchmarks.gate_g8 as g8


def test_live_branch_skips_without_gpu(monkeypatch):
    """No CUDA → PASS with a 'skipped' detail (never fails the gate)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ok, detail = g8._check_live_llm_loop_contract()
    assert ok is True
    assert "skipped" in detail.lower()


def test_live_branch_skips_without_credentials(monkeypatch):
    """CUDA present but no LLM credentials → PASS with a 'skipped' detail."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _raise():
        raise RuntimeError("No LLM provider credentials found.")

    # load_from_env raising → graceful skip
    monkeypatch.setattr(
        "arke.agent.llm_config.load_from_env", _raise, raising=True
    )
    ok, detail = g8._check_live_llm_loop_contract()
    assert ok is True
    assert "skipped" in detail.lower()


def test_live_branch_function_registered_in_gate():
    """The gate's run_g8 must include the G8.LIVE.1 criterion."""
    import inspect

    src = inspect.getsource(g8.run_g8)
    assert "G8.LIVE.1" in src
    assert "_check_live_llm_loop_contract" in src
