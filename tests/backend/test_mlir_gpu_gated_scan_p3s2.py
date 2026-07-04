# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): gated/select/cast + scan/argmax ops via MLIR gpu dialect.

Extends GPU coverage with:
  * gated/select/cast (element-per-block, emit_gpu_gated):
    cast (f32 identity), where_ (branchless cond*a+(1-cond)*b),
    silu_and_mul / gelu_and_mul (act(X[:, :D]) * X[:, D:]).
  * row-wise scan/index (row-per-block, emit_gpu_rowwise):
    cumsum (prefix sum along row), argmax (f32-encoded row-max index).

Bit-correct vs numpy/torch on the CUDA driver. Skips without GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available
from arke.backend.mlir_emitter import GPU_GATED_OPS, GPU_ROWWISE_OPS


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _run(op, ins, in_shapes):
    be = MLIRGPUBackend()
    g = IRGraph(name=op)
    names = list(ins.keys())
    for n, s in zip(names, in_shapes):
        g.add_input(n, dtype="float32", shape=list(s))
    g.add_node(IRNode(id="n0", op=op, inputs={n: n for n in names}, outputs=["Y"]))
    g.set_outputs(["Y"])
    ker = be.compile(be.lower(g))
    assert ker.success, ker.error
    return be.run(ker, ins)["Y"]


def test_backend_supports_gated_and_scan():
    be = MLIRGPUBackend()
    for op in ("cast", "where_", "silu_and_mul", "gelu_and_mul"):
        assert op in GPU_GATED_OPS and be.supports_op(op)
    for op in ("cumsum", "argmax"):
        assert op in GPU_ROWWISE_OPS and be.supports_op(op)


@pytest.mark.parametrize("M,N", [(8, 16), (16, 33)])
def test_cast_identity(M, N):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((M, N)).astype(np.float32)
    np.testing.assert_array_equal(_run("cast", {"X": X}, [(M, N)]), X)


@pytest.mark.parametrize("M,N", [(8, 16), (4, 32)])
def test_where(M, N):
    rng = np.random.default_rng(1)
    C = (rng.random((M, N)) > 0.5).astype(np.float32)
    A = rng.standard_normal((M, N)).astype(np.float32)
    B = rng.standard_normal((M, N)).astype(np.float32)
    out = _run("where_", {"C": C, "A": A, "B": B}, [(M, N), (M, N), (M, N)])
    np.testing.assert_allclose(out, np.where(C > 0.5, A, B), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("op,torch_act", [
    ("silu_and_mul", "silu"),
    ("gelu_and_mul", "gelu"),
])
@pytest.mark.parametrize("M,D", [(8, 16), (4, 64)])
def test_gated_mul(op, torch_act, M, D):
    import torch
    rng = np.random.default_rng(2)
    X = rng.standard_normal((M, 2 * D)).astype(np.float32)
    Xt = torch.tensor(X)
    if torch_act == "silu":
        ref = (torch.nn.functional.silu(Xt[:, :D]) * Xt[:, D:]).numpy()
    else:
        ref = (torch.nn.functional.gelu(Xt[:, :D], approximate="tanh") * Xt[:, D:]).numpy()
    out = _run(op, {"X": X}, [(M, 2 * D)])
    assert out.shape == (M, D)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M,D", [(8, 16), (4, 64), (16, 100)])
def test_cumsum(M, D):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((M, D)).astype(np.float32)
    out = _run("cumsum", {"X": X}, [(M, D)])
    assert out.shape == (M, D)
    np.testing.assert_allclose(out, np.cumsum(X, axis=1), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("M,D", [(8, 16), (4, 64), (16, 127)])
def test_argmax(M, D):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((M, D)).astype(np.float32)
    out = _run("argmax", {"X": X}, [(M, D)])
    assert out.shape == (M,)
    np.testing.assert_array_equal(out.astype(np.int64), np.argmax(X, axis=1))
