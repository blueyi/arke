# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): rope + 2-input row-wise ops (rmsnorm_residual, embedding).

  * rope (single input, emit_gpu_rowwise): rotary position embedding, pos=row,
    theta=10000^(-2k/D); uses libdevice cos/sin/log/exp.
  * rmsnorm_residual (emit_gpu_rowwise2): rmsnorm(x + residual).
  * embedding (emit_gpu_rowwise2): out[i,:] = table[int(idx[i]), :] (1D f32 idx).

Bit-correct vs numpy/torch on the CUDA driver. Skips without GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available
from arke.backend.mlir_emitter import GPU_ROWWISE_OPS, GPU_ROWWISE2_OPS


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


def _np_rope(x):
    R, D = x.shape
    half = D // 2
    out = np.empty_like(x)
    for r in range(R):
        for k in range(half):
            theta = 10000.0 ** (-2 * k / D)
            ang = r * theta
            c, s = np.cos(ang), np.sin(ang)
            x1, x2 = x[r, k], x[r, k + half]
            out[r, k] = x1 * c - x2 * s
            out[r, k + half] = x2 * c + x1 * s
    return out


def test_backend_supports_new_ops():
    be = MLIRGPUBackend()
    assert "rope" in GPU_ROWWISE_OPS and be.supports_op("rope")
    for op in ("rmsnorm_residual", "embedding"):
        assert op in GPU_ROWWISE2_OPS and be.supports_op(op)


@pytest.mark.parametrize("R,D", [(1, 4), (8, 16), (4, 64)])
def test_rope_correct(R, D):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((R, D)).astype(np.float32)
    out = _run("rope", {"X": X}, [(R, D)])
    assert out.shape == (R, D)
    np.testing.assert_allclose(out, _np_rope(X), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("R,D", [(8, 64), (4, 128), (16, 33)])
def test_rmsnorm_residual_correct(R, D):
    import torch
    rng = np.random.default_rng(1)
    X = rng.standard_normal((R, D)).astype(np.float32)
    Res = rng.standard_normal((R, D)).astype(np.float32)
    s = torch.tensor(X) + torch.tensor(Res)
    ref = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5)).numpy()
    out = _run("rmsnorm_residual", {"X": X, "R": Res}, [(R, D), (R, D)])
    assert out.shape == (R, D)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("nidx,vocab,dim", [(8, 100, 32), (4, 50, 16), (16, 200, 64)])
def test_embedding_correct(nidx, vocab, dim):
    rng = np.random.default_rng(2)
    idx = rng.integers(0, vocab, size=nidx).astype(np.float32)
    tbl = rng.standard_normal((vocab, dim)).astype(np.float32)
    out = _run("embedding", {"I": idx, "T": tbl}, [(nidx,), (vocab, dim)])
    assert out.shape == (nidx, dim)
    np.testing.assert_array_equal(out, tbl[idx.astype(int)])
