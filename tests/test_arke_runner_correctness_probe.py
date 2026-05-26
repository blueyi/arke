# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.arke_runner import ArkeRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
def test_arke_runner_reports_unavailable_when_kernel_cache_is_missing():
    runner = ArkeRunner()
    if runner.available:
        pytest.skip("ArkeRunner is available in this environment; missing-kernel-cache path not applicable")

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0)

    assert result["correctness_status"] == "unsupported"
    assert "does not implement run_with_inputs" in result["correctness_reason"]
    assert result["allclose"] is None


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("op", "shape"),
    [
        ("relu", (128, 128, 0)),
        ("matmul", (64, 64, 64)),
        ("softmax", (128, 128, 0)),
        ("layernorm", (128, 128, 0)),
        ("rmsnorm", (128, 128, 0)),
        ("add", (128, 128, 0)),
        ("reduce_sum", (128, 128, 0)),
        ("batch_matmul", (8, 64, 32)),
        ("silu_and_mul", (64, 64, 0)),
        ("flash_attention", (8, 128, 64)),
        ("cross_attention", (8, 128, 64)),
        ("rope", (1, 128, 64)),
    ],
)
def test_arke_runner_correctness_probe_supported_ops(op, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    runner = ArkeRunner()
    if not runner.available:
        pytest.skip("ArkeRunner unavailable")

    M, N, K = shape
    result = _measure_l1_correctness(runner, op, M, N, K)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] is not None
    assert result["mean_abs_diff"] is not None
