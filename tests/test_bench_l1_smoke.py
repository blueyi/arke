# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Golden Kernel ladder integration in bench_l1.

Verifies:
  * ``_resolve_golden_for_correctness`` consults the ladder and returns
    the chosen golden's name + priority.
  * ``GoldenUnavailable`` produces a row with status
    ``golden_unavailable_pending_baseline`` rather than crashing.
  * ``--golden`` / ``--golden-file`` overrides parse correctly and pin
    the chosen runner.
"""

from __future__ import annotations

import textwrap

import pytest
import torch

import benchmarks.bench_l1 as bench_l1
from benchmarks.bench_l1 import (
    _measure_l1_correctness,
    _resolve_golden_for_correctness,
)
from benchmarks.golden_ladder import (
    GoldenUnavailable,
    LADDER_PREFERENCES,
    golden_runner_for,
    parse_inline_overrides,
    parse_overrides_file,
)


# ── ladder helper ───────────────────────────────────────────────────


def test_inline_override_parser():
    out = parse_inline_overrides(["matmul=cuBLAS", "softmax=FlagGems"])
    assert out == {"matmul": "cuBLAS", "softmax": "FlagGems"}


def test_ladder_preferences_pin_rope_to_pytorch_eager():
    """G7.8c locked: rope must resolve to PyTorch-eager regardless of P-order."""
    assert LADDER_PREFERENCES.get("rope") == "PyTorch-eager"
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for live ladder resolution")
    runner = golden_runner_for("rope")
    assert runner.name == "PyTorch-eager"
    assert runner.priority == 3


def test_caller_overrides_take_precedence_over_ladder_preferences():
    """Caller --golden rope=Liger-Kernel must override the locked default."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for live ladder resolution")
    runner = golden_runner_for("rope", overrides={"rope": "Liger-Kernel"})
    assert runner.name == "Liger-Kernel"


def test_inline_override_parser_rejects_malformed():
    with pytest.raises(ValueError):
        parse_inline_overrides(["matmul-cuBLAS"])


def test_overrides_file_parser_yaml(tmp_path):
    p = tmp_path / "ov.yaml"
    p.write_text("matmul: cuBLAS\nsoftmax: Liger-Kernel\n")
    out = parse_overrides_file(str(p))
    assert out == {"matmul": "cuBLAS", "softmax": "Liger-Kernel"}


def test_overrides_file_parser_returns_empty_when_path_none():
    assert parse_overrides_file(None) == {}


# ── ladder resolution ───────────────────────────────────────────────


def test_golden_unavailable_for_unknown_op():
    """An op no runner declares supports() for raises GoldenUnavailable."""
    with pytest.raises(GoldenUnavailable):
        golden_runner_for("definitely_not_a_real_op_xyz_42")


def test_override_pins_runner(monkeypatch):
    """An override forces a specific runner regardless of priority order."""
    # PyTorch-eager is P3 — it would normally lose to P0/P1. Pinning it
    # for an op it supports must succeed.
    import benchmarks.baselines.pytorch_eager  # noqa: F401  (registers)

    if not torch.cuda.is_available():
        pytest.skip("PyTorch-eager runner requires CUDA")

    runner = golden_runner_for("relu", overrides={"relu": "PyTorch-eager"})
    assert runner.name == "PyTorch-eager"


def test_override_for_unavailable_runner_raises():
    with pytest.raises(GoldenUnavailable):
        golden_runner_for("matmul", overrides={"matmul": "NotARealRunner"})


# ── correctness flow ────────────────────────────────────────────────


class _DummyRunner:
    """Minimal stand-in to feed into _measure_l1_correctness."""

    def __init__(self, output, name: str = "DummyRunner"):
        self.output = output
        self.name = name
        self.available = True
        self.priority = 9

    def run_with_inputs(self, op, *inputs, **kwargs):
        return self.output


def test_resolve_returns_metadata_about_golden(monkeypatch):
    """The resolver carries golden_runner / golden_priority back to caller."""
    # Force the ladder via overrides so the test is deterministic regardless
    # of which runners are installed.
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA-backed runners")

    monkeypatch.setattr(
        bench_l1, "_GOLDEN_OVERRIDES", {"relu": "PyTorch-eager"}, raising=False
    )

    inputs = (torch.zeros((4, 4), dtype=torch.float16, device="cuda"),)
    out, name, prio, status, reason = _resolve_golden_for_correctness("relu", inputs)
    assert name == "PyTorch-eager"
    assert prio == 3
    assert status == ""  # clean run
    assert isinstance(out, torch.Tensor)


def test_measure_l1_correctness_threads_golden_metadata(monkeypatch):
    """OpResult dict picks up golden_runner / golden_priority."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA-backed runners")

    monkeypatch.setattr(
        bench_l1, "_GOLDEN_OVERRIDES", {"relu": "PyTorch-eager"}, raising=False
    )

    runner = _DummyRunner(torch.zeros((128, 128), dtype=torch.float16, device="cuda"))
    result = _measure_l1_correctness(
        runner, "relu", 128, 128, 0, dtype=torch.float16
    )
    assert result.get("golden_runner") == "PyTorch-eager"
    assert result.get("golden_priority") == 3


def test_measure_l1_correctness_emits_audit_when_golden_unavailable():
    """Unknown op → audit row with golden_unavailable_pending_baseline."""
    runner = _DummyRunner(torch.zeros((1, 1), dtype=torch.float16, device="cpu"))
    result = _measure_l1_correctness(
        runner, "definitely_not_a_real_op_xyz_42", 1, 1, 0, dtype=torch.float16
    )
    assert result["correctness_status"] == "golden_unavailable_pending_baseline"
    assert "definitely_not_a_real_op_xyz_42" in result["correctness_reason"]


# ── CSV schema ──────────────────────────────────────────────────────


def test_l1_csv_schema_includes_golden_columns():
    from benchmarks.bench_l1 import L1_FIELDNAMES

    assert "golden_runner" in L1_FIELDNAMES
    assert "golden_priority" in L1_FIELDNAMES
