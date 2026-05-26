# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.bench_l2 import _measure_fused_correctness


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("op", "shape"),
    [
        ("silu_and_mul", (16, 128, 0)),
        ("gelu_and_mul", (16, 128, 0)),
        ("linear_ce", (8, 64, 128)),
        ("qkv_fa", (16, 192, 64)),
    ],
)
def test_l2_fused_correctness_probe_ok(op, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    M, N, K = shape
    result = _measure_fused_correctness(op, "separate", M, N, K, dtype=torch.float16)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] is not None
    assert result["mean_abs_diff"] is not None
