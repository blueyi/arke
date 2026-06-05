# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""D8-X1 extensibility demo: `swiglu_packed` onboarding contract.

`swiglu_packed` is intentionally distinct from `silu_and_mul`:

    X:[M, 2K] -> split into x1/x2:[M, K]
    H = silu(x1) * x2
    W:[K, N]
    Y = H @ W -> [M, N]

This locks the true fused FFN down-projection semantics before the implementation
is wired through the benchmark catalog, IR schema, and L1 runner stack.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _manual_swiglu_packed(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return (F.silu(x1) * x2) @ w


def test_swiglu_packed_catalog_schema_and_independent_baseline():
    from benchmarks.op_registry import ALL_OPS, OP_TIER
    from arke.ir.ops.registry import REGISTRY
    from tests.independent_baseline import baseline_swiglu_packed

    assert "swiglu_packed" in ALL_OPS
    assert OP_TIER["swiglu_packed"] == 3

    schema = REGISTRY.get("swiglu_packed")
    assert schema.inputs == {"X": "Tensor[M,2K]", "W": "Tensor[K,N]"}
    assert schema.output == "Tensor[M,N]"

    x = torch.tensor(
        [[-1.0, 0.5, 2.0, -3.0], [0.25, -0.75, 1.5, 0.125]],
        dtype=torch.float32,
    )
    w = torch.tensor([[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]], dtype=torch.float32)

    expected = _manual_swiglu_packed(x, w)
    assert torch.allclose(schema.reference_impl.fn({"X": x, "W": w}, {}), expected)
    assert torch.allclose(baseline_swiglu_packed({"X": x, "W": w}, {}), expected)


def test_swiglu_packed_shape_mapping_uses_matmul_semantics():
    from benchmarks.shapes import get_shapes

    shapes = get_shapes("swiglu_packed", tier=1)
    assert shapes
    assert all(hasattr(s, "K") and s.K > 0 for s in shapes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bench_l1 input builder")
def test_swiglu_packed_bench_l1_inputs_and_pytorch_eager_runner():
    from benchmarks.baselines.pytorch_eager import PyTorchEagerRunner
    from benchmarks.bench_l1 import _make_l1_correctness_inputs, _torch_reference

    m, n, k = 4, 3, 2
    x, w = _make_l1_correctness_inputs("swiglu_packed", m, n, k, torch.float32)
    assert tuple(x.shape) == (m, 2 * k)
    assert tuple(w.shape) == (k, n)

    expected = _manual_swiglu_packed(x, w)
    assert torch.allclose(_torch_reference("swiglu_packed", (x, w)), expected)

    runner = PyTorchEagerRunner()
    assert runner.supports("swiglu_packed")
    actual = runner.run_with_inputs("swiglu_packed", x, w)
    assert isinstance(actual, torch.Tensor)
    assert torch.allclose(actual, expected)
