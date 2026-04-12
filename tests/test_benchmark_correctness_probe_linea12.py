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
