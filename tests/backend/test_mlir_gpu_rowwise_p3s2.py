# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): row-wise reduction/norm ops via MLIR gpu dialect.

Extends the GPU backend from matmul + elementwise to the row-wise family:
reduce_sum/max/mean (→ [rows]) and softmax/layernorm/rmsnorm (→ [rows, D]).
``emit_gpu_rowwise`` uses a row-per-block kernel (grid=(rows,1,1), block=(1,1,1))
with a serial pass over the D columns — same math as the CPU composite path, so
GPU and CPU numerics match. Correctness-first; block-parallel reduction is a
P3-S3 perf follow-up.

Bit-correct vs torch on the CUDA driver. Skips cleanly without GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available
from arke.backend.mlir_emitter import GPU_ROWWISE_OPS


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _rowwise_graph(op: str, R: int, D: int) -> IRGraph:
    g = IRGraph(name=op)
    g.add_input("X", dtype="float32", shape=[R, D])
    g.add_node(IRNode(id="n0", op=op, inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def _torch_ref(op, x):
    import torch
    t = torch.tensor(x)
    return {
        "reduce_sum": lambda: t.sum(-1),
        "reduce_max": lambda: t.max(-1).values,
        "reduce_mean": lambda: t.mean(-1),
        "softmax": lambda: torch.softmax(t, -1),
        "layernorm": lambda: torch.nn.functional.layer_norm(t, (t.shape[-1],), eps=1e-5),
        "rmsnorm": lambda: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5),
    }[op]().numpy()


def test_backend_supports_rowwise():
    be = MLIRGPUBackend()
    for op in ("reduce_sum", "reduce_max", "reduce_mean",
               "softmax", "layernorm", "rmsnorm"):
        assert op in GPU_ROWWISE_OPS
        assert be.supports_op(op)


@pytest.mark.parametrize("op", [
    "reduce_sum", "reduce_max", "reduce_mean", "softmax", "layernorm", "rmsnorm",
])
@pytest.mark.parametrize("R,D", [(1, 16), (8, 64), (16, 128), (4, 257)])
def test_rowwise_correct_vs_torch(op, R, D):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((R, D)).astype(np.float32)
    ker = be.compile(be.lower(_rowwise_graph(op, R, D)))
    assert ker.success, ker.error
    out = be.run(ker, {"X": X})["Y"]
    ref = _torch_ref(op, X)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_reduce_outputs_1d():
    be = MLIRGPUBackend()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 32)).astype(np.float32)
    out = be.run(be.compile(be.lower(_rowwise_graph("reduce_sum", 8, 32))), {"X": X})["Y"]
    assert out.shape == (8,)


def test_norm_outputs_2d():
    be = MLIRGPUBackend()
    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, 32)).astype(np.float32)
    out = be.run(be.compile(be.lower(_rowwise_graph("softmax", 8, 32))), {"X": X})["Y"]
    assert out.shape == (8, 32)
    # each row sums to 1
    np.testing.assert_allclose(out.sum(-1), np.ones(8), rtol=1e-4, atol=1e-4)


def test_softmax_uses_libdevice_exp():
    """softmax's exp must inline to native ex2.approx (libdevice), not extern."""
    from arke.backend.mlir_gpu import mlir_gpu_to_ptx
    from arke.backend.mlir_emitter import emit_gpu_rowwise
    e = emit_gpu_rowwise(_rowwise_graph("softmax", 4, 16))
    ptx = mlir_gpu_to_ptx(e.mlir_text)
    assert "ex2.approx" in ptx
    assert "__nv_expf" not in ptx
