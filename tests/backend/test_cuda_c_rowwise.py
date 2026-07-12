# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for rowwise reduction CUDA-C kernels (softmax, layernorm, rmsnorm, reduce_*)."""

import numpy as np
import pytest

from arke.backend.cuda_c_backend import CudaCBackend, cuda_c_toolchain_available
from arke.backend.cuda_c_rowwise import ROWWISE_EMITTERS
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not cuda_c_toolchain_available(),
    reason="CUDA-C toolchain (nvcc + GPU driver) not available",
)

M, N = 64, 512


@pytest.fixture
def backend():
    # Register rowwise emitters
    b = CudaCBackend(chip="sm_86")
    b._EMITTERS = {**b._EMITTERS, **ROWWISE_EMITTERS}
    return b


class TestSoftmax:
    def test_correctness(self, backend):
        torch = pytest.importorskip("torch")
        graph = IRGraph(name="softmax_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="softmax", inputs={"x": "X"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, kernel.error

        np.random.seed(0)
        X = np.random.randn(M, N).astype(np.float32)
        result = backend.run(kernel, {"X": X})
        ref = torch.nn.functional.softmax(torch.from_numpy(X), dim=-1).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4, rtol=1e-4)


class TestLayerNorm:
    def test_correctness(self, backend):
        torch = pytest.importorskip("torch")
        graph = IRGraph(name="layernorm_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_input("W", dtype="float32", shape=[N])
        graph.add_input("B", dtype="float32", shape=[N])
        graph.add_node(IRNode(id="n0", op="layernorm", inputs={"x": "X", "W": "W", "B": "B"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, kernel.error

        np.random.seed(1)
        X = np.random.randn(M, N).astype(np.float32)
        W = np.random.randn(N).astype(np.float32)
        B = np.random.randn(N).astype(np.float32)
        result = backend.run(kernel, {"X": X, "W": W, "B": B})
        ref = torch.nn.functional.layer_norm(
            torch.from_numpy(X), [N],
            torch.from_numpy(W), torch.from_numpy(B)
        ).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-3, rtol=1e-3)


class TestRMSNorm:
    def test_correctness(self, backend):
        torch = pytest.importorskip("torch")
        graph = IRGraph(name="rmsnorm_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_input("W", dtype="float32", shape=[N])
        graph.add_node(IRNode(id="n0", op="rmsnorm", inputs={"x": "X", "W": "W"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, kernel.error

        np.random.seed(2)
        X = np.random.randn(M, N).astype(np.float32)
        W = np.random.randn(N).astype(np.float32)
        result = backend.run(kernel, {"X": X, "W": W})
        X_t = torch.from_numpy(X)
        rms = torch.sqrt(X_t.pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = (X_t / rms * torch.from_numpy(W)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-3, rtol=1e-3)


class TestReduceSum:
    def test_correctness(self, backend):
        graph = IRGraph(name="reduce_sum_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="reduce_sum", inputs={"x": "X"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, kernel.error

        np.random.seed(3)
        X = np.random.randn(M, N).astype(np.float32)
        result = backend.run(kernel, {"X": X})
        ref = X.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(result["out"], ref, atol=1e-4, rtol=1e-4)


class TestReduceMax:
    def test_correctness(self, backend):
        graph = IRGraph(name="reduce_max_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="reduce_max", inputs={"x": "X"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, kernel.error

        np.random.seed(4)
        X = np.random.randn(M, N).astype(np.float32)
        result = backend.run(kernel, {"X": X})
        ref = X.max(axis=-1, keepdims=True)
        np.testing.assert_allclose(result["out"], ref, atol=1e-4, rtol=1e-4)


class TestReduceMean:
    def test_correctness(self, backend):
        graph = IRGraph(name="reduce_mean_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="reduce_mean", inputs={"x": "X"}, outputs=["out"]))
        graph.set_outputs(["out"])

        artifact = backend.lower(graph)
        kernel = backend.compile(artifact)
        assert kernel.success, kernel.error

        np.random.seed(5)
        X = np.random.randn(M, N).astype(np.float32)
        result = backend.run(kernel, {"X": X})
        ref = X.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(result["out"], ref, atol=1e-4, rtol=1e-4)
