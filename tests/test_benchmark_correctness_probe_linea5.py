# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
from benchmarks.bench_l1 import _measure_l1_correctness


@pytest.mark.cuda
def test_l1_correctness_probe_linea5_grouped_query_attention_ok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = PyTorchEagerRunner()
    if not runner.available:
        pytest.skip(f"{runner.name} unavailable")

    result = _measure_l1_correctness(runner, "grouped_query_attention", 8, 128, 64)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["mean_abs_diff"] == 0.0
