# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.cublas import CuBLASRunner
from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("runner_cls", "op", "shape"),
    [
        (PyTorchEagerRunner, "where_", (64, 64, 0)),
        (PyTorchEagerRunner, "cast", (64, 64, 0)),
        (CuBLASRunner, "reduce_sum", (64, 64, 0)),
        (PyTorchEagerRunner, "reduce_max", (64, 64, 0)),
        (CuBLASRunner, "reduce_mean", (64, 64, 0)),
        (PyTorchEagerRunner, "argmax", (64, 64, 0)),
        (PyTorchEagerRunner, "cumsum", (64, 64, 0)),
        (PyTorchEagerRunner, "topk", (64, 64, 0)),
    ],
)
def test_l1_correctness_probe_linea_ops_ok(runner_cls, op, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = runner_cls()
    if not runner.available:
        pytest.skip(f"{runner.name} unavailable")

    M, N, K = shape
    result = _measure_l1_correctness(runner, op, M, N, K)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["mean_abs_diff"] == 0.0
