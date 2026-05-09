# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.cublas import CuBLASRunner
from benchmarks.baselines.inductor import InductorRunner
from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.baselines.triton_tutorial import TritonTutorialRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("runner_cls", "op", "shape", "expected_status"),
    [
        (CuBLASRunner, "matmul", (128, 128, 128), "ok"),
        (PyTorchEagerRunner, "matmul", (128, 128, 128), "ok"),
        (InductorRunner, "matmul", (128, 128, 128), "ok"),
        (TritonTutorialRunner, "matmul", (128, 128, 128), "ok"),
    ],
)
def test_l1_correctness_probe_linea8_matmul(runner_cls, op, shape, expected_status):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = runner_cls()
    if not runner.available:
        pytest.skip(f"{runner.name} unavailable")

    M, N, K = shape
    result = _measure_l1_correctness(runner, op, M, N, K)

    assert result["correctness_status"] == expected_status
    if expected_status == "unsupported":
        assert result["allclose"] is None
        return
    assert result["allclose"] is True
    assert result["max_abs_diff"] is not None
    assert result["mean_abs_diff"] is not None
