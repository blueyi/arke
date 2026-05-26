# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("op", "shape"),
    [
        ("silu_and_mul", (128, 3072, 0)),
        ("gelu_and_mul", (128, 3072, 0)),
    ],
)
def test_l1_correctness_probe_linea9_fused_gated_ops(op, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = PyTorchEagerRunner()
    if not runner.available:
        pytest.skip(f"{runner.name} unavailable")

    M, N, K = shape
    result = _measure_l1_correctness(runner, op, M, N, K)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] is not None
    assert result["mean_abs_diff"] is not None
