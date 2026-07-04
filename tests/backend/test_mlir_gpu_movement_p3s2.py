# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): 2D data-movement ops via MLIR gpu dialect.

Extends the GPU backend with pure address-remapping ops (no math):
transpose, copy_, split (first-half chunk), concat (last-axis). ``emit_gpu_movement``
uses an element-per-block kernel (grid = output shape, block=(1,1,1)); same index
math as the CPU composite path. concat uses scf.if to pick source A vs B.

Bit-correct vs numpy on the CUDA driver. Skips cleanly without GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available
from arke.backend.mlir_emitter import GPU_MOVEMENT_OPS


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def test_backend_supports_movement():
    be = MLIRGPUBackend()
    for op in ("transpose", "copy_", "split", "concat"):
        assert op in GPU_MOVEMENT_OPS
        assert be.supports_op(op)


@pytest.mark.parametrize("M,N", [(1, 1), (8, 16), (16, 8), (33, 17)])
def test_transpose_correct(M, N):
    be = MLIRGPUBackend()
    g = IRGraph(name="transpose")
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op="transpose", inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((M, N)).astype(np.float32)
    out = be.run(be.compile(be.lower(g)), {"X": X})["Y"]
    assert out.shape == (N, M)
    np.testing.assert_allclose(out, X.T, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("M,D", [(8, 16), (16, 32)])
def test_copy_correct(M, D):
    be = MLIRGPUBackend()
    g = IRGraph(name="copy_")
    g.add_input("X", dtype="float32", shape=[M, D])
    g.add_node(IRNode(id="n0", op="copy_", inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    rng = np.random.default_rng(1)
    X = rng.standard_normal((M, D)).astype(np.float32)
    out = be.run(be.compile(be.lower(g)), {"X": X})["Y"]
    np.testing.assert_allclose(out, X, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("M,D", [(8, 16), (4, 64)])
def test_split_first_half(M, D):
    be = MLIRGPUBackend()
    g = IRGraph(name="split")
    g.add_input("X", dtype="float32", shape=[M, D])
    g.add_node(IRNode(id="n0", op="split", inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    rng = np.random.default_rng(2)
    X = rng.standard_normal((M, D)).astype(np.float32)
    out = be.run(be.compile(be.lower(g)), {"X": X})["Y"]
    assert out.shape == (M, D // 2)
    np.testing.assert_allclose(out, X[:, : D // 2], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("M,Da,Db", [(8, 16, 4), (4, 8, 8), (16, 32, 16)])
def test_concat_last_axis(M, Da, Db):
    be = MLIRGPUBackend()
    g = IRGraph(name="concat")
    g.add_input("A", dtype="float32", shape=[M, Da])
    g.add_input("B", dtype="float32", shape=[M, Db])
    g.add_node(IRNode(id="n0", op="concat", inputs={"A": "A", "B": "B"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    rng = np.random.default_rng(3)
    A = rng.standard_normal((M, Da)).astype(np.float32)
    B = rng.standard_normal((M, Db)).astype(np.float32)
    out = be.run(be.compile(be.lower(g)), {"A": A, "B": B})["Y"]
    assert out.shape == (M, Da + Db)
    np.testing.assert_allclose(out, np.concatenate([A, B], axis=1), rtol=1e-6, atol=1e-6)
