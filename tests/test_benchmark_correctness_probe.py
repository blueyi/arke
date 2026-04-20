# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.cublas import CuBLASRunner
from benchmarks.baselines.inductor import InductorRunner
from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("runner_cls", "op", "shape"),
    [
        (CuBLASRunner, "relu", (128, 128, 0)),
        (PyTorchEagerRunner, "relu", (128, 128, 0)),
        (InductorRunner, "relu", (128, 128, 0)),
        (CuBLASRunner, "matmul", (64, 64, 64)),
        (PyTorchEagerRunner, "matmul", (64, 64, 64)),
        (CuBLASRunner, "tanh", (128, 128, 0)),
        (PyTorchEagerRunner, "sigmoid", (128, 128, 0)),
        (CuBLASRunner, "add", (128, 128, 0)),
        (PyTorchEagerRunner, "mul", (128, 128, 0)),
        (CuBLASRunner, "neg", (128, 128, 0)),
        (PyTorchEagerRunner, "exp", (128, 128, 0)),
        (CuBLASRunner, "rsqrt", (128, 128, 0)),
    ],
)
def test_l1_correctness_probe_ok_for_supported_runners(runner_cls, op, shape):
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
