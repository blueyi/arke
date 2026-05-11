# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.liger import LigerRunner
from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("runner", "expected_status"),
    [
        (PyTorchEagerRunner(), "ok"),
        # Liger rope is the P1 perf winner but its tuple output and rotate
        # convention diverge from the PyTorch-eager reference, so
        # run_with_inputs returns None (status='unsupported') by design.
        # Per docs/benchmark/golden-kernel-ladder.md this is a known
        # ladder gap audited via golden_unavailable_pending_baseline at
        # the gate layer rather than fixed at the runner.
        (LigerRunner(), "unsupported"),
    ],
)
def test_l1_correctness_probe_linea12_rope(runner, expected_status):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not runner.available:
        pytest.skip(f"{runner.name} unavailable")

    result = _measure_l1_correctness(runner, "rope", 1, 128, 64)

    assert result["correctness_status"] == expected_status
    if expected_status == "ok":
        assert result["allclose"] is True
        assert result["max_abs_diff"] is not None
        assert result["mean_abs_diff"] is not None
    else:
        assert result["allclose"] is None


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("runner",),
    [(LigerRunner(),), (PyTorchEagerRunner(),)],
)
def test_rope_odd_head_dim_returns_none(runner):
    """RoPE rotates pairs of channels — odd head_dim is mathematically
    ill-defined. Both Liger and PyTorch-eager reference must return None
    so the harness records 'unsupported' instead of crashing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not runner.available:
        pytest.skip(f"{runner.name} unavailable")

    # Mimic non-align-1: B=1, H=13, S=127, D=65 (odd)
    Q = torch.randn(1, 13, 127, 65, device="cuda", dtype=torch.float16)
    out = runner.run_with_inputs("rope", Q)
    assert out is None, (
        f"{runner.name} must return None for odd head_dim, got {type(out)}"
    )


@pytest.mark.cuda
def test_rope_odd_head_dim_reference_marks_unsupported():
    """Reference (_eval_l1_reference) for RoPE on odd head_dim must
    raise NotImplementedError so _measure_l1_correctness marks the row
    as 'unsupported' (not 'error', not 'golden_unavailable')."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    runner = PyTorchEagerRunner()
    if not runner.available:
        pytest.skip("PyTorch-eager unavailable")

    # non-align-1 catalog shape: H=13, S=127, D=65 (odd)
    result = _measure_l1_correctness(runner, "rope", 13, 127, 65, torch.float16)
    assert result["correctness_status"] == "unsupported", (
        f"Expected 'unsupported' for odd head_dim, got {result['correctness_status']!r}: "
        f"{result.get('correctness_reason')!r}"
    )
    assert "even head_dim" in (result["correctness_reason"] or "").lower() \
        or "odd" in (result["correctness_reason"] or "").lower()
