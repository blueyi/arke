# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CUDA-C movement + gated ops (Phase 4, P4-S2)."""

import numpy as np
import pytest
import torch

from arke.backend.cuda_c_backend import CudaCBackend, cuda_c_toolchain_available
from arke.ir.graph import IRGraph, IRNode

pytestmark = pytest.mark.skipif(
    not cuda_c_toolchain_available(),
    reason="CUDA-C toolchain not available",
)


@pytest.fixture
def backend():
    return CudaCBackend(chip="sm_86")


class TestCudaCTranspose:
    @pytest.mark.parametrize("M,N", [(64, 128), (128, 64), (33, 47)])
    def test_transpose(self, backend, M, N):
        graph = IRGraph(name=f"transpose_{M}x{N}")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="transpose",
                              inputs={"x": "X"}, outputs=["out"]))
        graph.set_outputs(["out"])

        art = backend.lower(graph)
        ker = backend.compile(art)
        assert ker.success, ker.error

        X = np.random.randn(M, N).astype(np.float32)
        result = backend.run(ker, {"X": X})
        np.testing.assert_allclose(result["out"], X.T, atol=0, rtol=0)


class TestCudaCCopy:
    def test_copy(self, backend):
        M, N = 64, 128
        graph = IRGraph(name="copy_test")
        graph.add_input("X", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="copy_",
                              inputs={"x": "X"}, outputs=["out"]))
        graph.set_outputs(["out"])

        art = backend.lower(graph)
        ker = backend.compile(art)
        assert ker.success, ker.error

        X = np.random.randn(M, N).astype(np.float32)
        result = backend.run(ker, {"X": X})
        np.testing.assert_allclose(result["out"], X, atol=0, rtol=0)


class TestCudaCGatedOps:
    def test_silu_and_mul(self, backend):
        M, N = 64, 128
        graph = IRGraph(name="silu_and_mul_test")
        graph.add_input("A", dtype="float32", shape=[M, N])
        graph.add_input("B", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="silu_and_mul",
                              inputs={"a": "A", "b": "B"}, outputs=["out"]))
        graph.set_outputs(["out"])

        art = backend.lower(graph)
        ker = backend.compile(art)
        assert ker.success, ker.error

        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = backend.run(ker, {"A": A, "B": B})

        At = torch.from_numpy(A)
        ref = (torch.nn.functional.silu(At) * torch.from_numpy(B)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4, rtol=1e-4)

    def test_gelu_and_mul(self, backend):
        M, N = 64, 128
        graph = IRGraph(name="gelu_and_mul_test")
        graph.add_input("A", dtype="float32", shape=[M, N])
        graph.add_input("B", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="gelu_and_mul",
                              inputs={"a": "A", "b": "B"}, outputs=["out"]))
        graph.set_outputs(["out"])

        art = backend.lower(graph)
        ker = backend.compile(art)
        assert ker.success, ker.error

        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = backend.run(ker, {"A": A, "B": B})

        At = torch.from_numpy(A)
        # Use tanh approximation to match our CUDA kernel
        ref = (torch.nn.functional.gelu(At, approximate="tanh") * torch.from_numpy(B)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-3, rtol=1e-3)

    def test_where(self, backend):
        M, N = 64, 128
        graph = IRGraph(name="where_test")
        graph.add_input("C", dtype="float32", shape=[M, N])
        graph.add_input("A", dtype="float32", shape=[M, N])
        graph.add_input("B", dtype="float32", shape=[M, N])
        graph.add_node(IRNode(id="n0", op="where_",
                              inputs={"cond": "C", "a": "A", "b": "B"},
                              outputs=["out"]))
        graph.set_outputs(["out"])

        art = backend.lower(graph)
        ker = backend.compile(art)
        assert ker.success, ker.error

        C = (np.random.randn(M, N) > 0).astype(np.float32)
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        result = backend.run(ker, {"C": C, "A": A, "B": B})

        ref = np.where(C != 0, A, B)
        np.testing.assert_allclose(result["out"], ref, atol=0, rtol=0)
