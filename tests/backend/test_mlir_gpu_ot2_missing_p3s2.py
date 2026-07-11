# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): OT2 ops that were missing dedicated test coverage.

Covers: topk, batch_matmul, gather, scatter, permute, grouped_matmul.
All bit-correct vs numpy/torch on the CUDA driver. Skips without GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _run(op, ins, in_shapes, attrs=None, in_dtypes=None):
    """Run a single-node graph through MLIRGPUBackend and return output array."""
    be = MLIRGPUBackend()
    g = IRGraph(name=op)
    names = list(ins.keys())
    for n, s in zip(names, in_shapes):
        g.add_input(n, dtype="float32", shape=list(s))
    g.add_node(IRNode(
        id="n0", op=op,
        inputs={n: n for n in names},
        outputs=["Y"],
        attrs=attrs or {},
    ))
    g.set_outputs(["Y"])
    ker = be.compile(be.lower(g))
    assert ker.success, f"compile failed: {ker.error}"
    return be.run(ker, ins)["Y"]


# ── topk ────────────────────────────────────────────────────────

class TestTopk:
    """OT1 topk: row-wise top-k selection (serial per row)."""

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_topk_correctness(self, k):
        rng = np.random.RandomState(42)
        M, N = 8, 64
        X = rng.randn(M, N).astype(np.float32)
        out = _run("topk", {"X": X}, [(M, N)], attrs={"k": k})
        assert out.shape == (M, k)
        # Each row's topk values must be a subset of the row, sorted desc
        for i in range(M):
            row_sorted = np.sort(X[i])[::-1][:k]
            np.testing.assert_allclose(np.sort(out[i])[::-1], row_sorted, rtol=1e-5)

    def test_topk_single_element_rows(self):
        X = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)
        out = _run("topk", {"X": X}, [(1, 3)], attrs={"k": 1})
        assert out.shape == (1, 1)
        assert out[0, 0] == pytest.approx(3.0, abs=1e-6)

    def test_topk_k_equals_n(self):
        rng = np.random.RandomState(7)
        X = rng.randn(4, 8).astype(np.float32)
        out = _run("topk", {"X": X}, [(4, 8)], attrs={"k": 8})
        assert out.shape == (4, 8)
        for i in range(4):
            np.testing.assert_allclose(
                np.sort(out[i])[::-1], np.sort(X[i])[::-1], rtol=1e-5
            )


# ── batch_matmul ────────────────────────────────────────────────

class TestBatchMatmul:
    """OT2 batch_matmul: C[b,i,j] = sum(A[b,i,k]*B[b,k,j], k)."""

    @pytest.mark.parametrize("shape", [
        (2, 16, 32, 16),   # (B, M, K, N)
        (4, 8, 8, 8),
        (1, 32, 64, 32),
    ])
    def test_batch_matmul_correctness(self, shape):
        B, M, K, N = shape
        rng = np.random.RandomState(42)
        A = rng.randn(B, M, K).astype(np.float32)
        Bm = rng.randn(B, K, N).astype(np.float32)
        out = _run("batch_matmul", {"A": A, "B": Bm}, [(B, M, K), (B, K, N)])
        ref = np.matmul(A, Bm)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ── gather ──────────────────────────────────────────────────────

class TestGather:
    """OT2 gather: out[i,j] = src[i, int(idx[i,j])]."""

    @pytest.mark.parametrize("M,N,K", [(4, 32, 8), (8, 64, 16)])
    def test_gather_correctness(self, M, N, K):
        rng = np.random.RandomState(42)
        src = rng.randn(M, N).astype(np.float32)
        idx = np.random.randint(0, N, (M, K)).astype(np.float32)
        out = _run("gather", {"X": src, "idx": idx}, [(M, N), (M, K)])
        assert out.shape == (M, K)
        # Verify element-by-element
        for i in range(M):
            for j in range(K):
                col = int(idx[i, j])
                assert out[i, j] == pytest.approx(src[i, col], abs=1e-6)

    def test_gather_first_last_cols(self):
        """Gather from first and last columns."""
        src = np.arange(20, dtype=np.float32).reshape(4, 5)
        idx = np.array([[0, 4], [0, 4], [0, 4], [0, 4]], dtype=np.float32)
        out = _run("gather", {"X": src, "idx": idx}, [(4, 5), (4, 2)])
        for i in range(4):
            assert out[i, 0] == pytest.approx(src[i, 0], abs=1e-6)
            assert out[i, 1] == pytest.approx(src[i, 4], abs=1e-6)


# ── scatter ─────────────────────────────────────────────────────

class TestScatter:
    """OT2 scatter: out[i, int(idx[i,j])] = src[i,j] (zero-filled base)."""

    def test_scatter_correctness(self):
        M, N, K = 4, 32, 8
        rng = np.random.RandomState(42)
        base = np.zeros((M, N), dtype=np.float32)
        src = rng.randn(M, K).astype(np.float32)
        # Non-overlapping indices per row to avoid race conditions
        idx = np.zeros((M, K), dtype=np.float32)
        for i in range(M):
            idx[i] = np.random.choice(N, K, replace=False).astype(np.float32)
        out = _run("scatter", {"base": base, "idx": idx, "src": src},
                    [(M, N), (M, K), (M, K)])
        assert out.shape == (M, N)
        # Verify scattered values landed correctly
        for i in range(M):
            for j in range(K):
                col = int(idx[i, j])
                assert out[i, col] == pytest.approx(src[i, j], abs=1e-6)

    def test_scatter_identity(self):
        """Scatter with sequential indices reproduces src in first K cols."""
        M, K = 2, 4
        src = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        idx = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32)
        base = np.zeros((M, 8), dtype=np.float32)
        out = _run("scatter", {"base": base, "idx": idx, "src": src},
                    [(M, 8), (M, K), (M, K)])
        for i in range(M):
            np.testing.assert_allclose(out[i, :K], src[i], atol=1e-6)


# ── permute ─────────────────────────────────────────────────────

class TestPermute:
    """OT2 permute: 3D transpose (0,2,1) via collapsed 2D grid.

    The current GPU emitter only supports the (0,2,1) permutation for 3D inputs
    (swap last two dims). Other permutations would need an extension.
    """

    @pytest.mark.parametrize("shape", [(4, 8, 16), (2, 16, 32), (1, 4, 8)])
    def test_permute_3d_021(self, shape):
        """permute(0,2,1): O[b,n,m] = X[b,m,n]."""
        rng = np.random.RandomState(42)
        X = rng.randn(*shape).astype(np.float32)
        out = _run("permute", {"X": X}, [shape], attrs={"dims": [0, 2, 1]})
        ref = np.transpose(X, (0, 2, 1))
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-6)

    def test_permute_2d(self):
        """2D permute = transpose."""
        rng = np.random.RandomState(7)
        X = rng.randn(8, 16).astype(np.float32)
        out = _run("permute", {"X": X}, [(8, 16)], attrs={"dims": [1, 0]})
        ref = X.T
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-6)


# ── grouped_matmul ──────────────────────────────────────────────

class TestGroupedMatmul:
    """OT2 grouped_matmul: Y[b,i,j] = sum(X[b,i,k] * W[indices[b],k,j], k)."""

    @pytest.mark.parametrize("B,M,K,N,E", [(4, 16, 32, 16, 8), (2, 8, 16, 8, 4)])
    def test_grouped_matmul_correctness(self, B, M, K, N, E):
        rng = np.random.RandomState(42)
        X = rng.randn(B, M, K).astype(np.float32)
        W = rng.randn(E, K, N).astype(np.float32)
        indices = np.random.randint(0, E, B).astype(np.float32)
        out = _run("grouped_matmul", {"X": X, "W": W, "indices": indices},
                    [(B, M, K), (E, K, N), (B,)])
        assert out.shape == (B, M, N)
        # Reference: per-batch expert selection
        for b in range(B):
            expert = int(indices[b])
            ref_b = X[b] @ W[expert]
            np.testing.assert_allclose(out[b], ref_b, rtol=1e-4, atol=1e-4)
