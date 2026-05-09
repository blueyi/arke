# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.cublas import CuBLASRunner
from benchmarks.baselines.inductor import InductorRunner
from benchmarks.baselines.liger import LigerRunner
from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("runner_cls", "op", "shape", "expected_status"),
    [
        (CuBLASRunner, "gelu", (128, 3072, 0), "ok"),
        (PyTorchEagerRunner, "gelu", (128, 3072, 0), "ok"),
        (InductorRunner, "gelu", (128, 3072, 0), "ok"),
        (LigerRunner, "gelu", (128, 3072, 0), "ok"),
        (CuBLASRunner, "silu", (128, 3072, 0), "ok"),
        (PyTorchEagerRunner, "silu", (128, 3072, 0), "ok"),
        (InductorRunner, "silu", (128, 3072, 0), "ok"),
        (LigerRunner, "silu", (128, 3072, 0), "ok"),
    ],
)
def test_l1_correctness_probe_linea7_ops_ok(runner_cls, op, shape, expected_status):
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
