# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): OT3 fused ops that were missing dedicated test coverage.

Covers: cross_entropy, quantize_per_token, dequantize_per_channel,
        swiglu_packed, fused_linear_cross_entropy.
All verified vs numpy/torch references. Skips without GPU toolchain.
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


def _run(op, ins, in_shapes, attrs=None):
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


def _ref_cross_entropy(X, labels):
    """Numpy reference: per-row cross-entropy loss."""
    B, V = X.shape
    losses = np.zeros(B, dtype=np.float32)
    for i in range(B):
        row = X[i].astype(np.float64)
        mx = np.max(row)
        log_sum_exp = mx + np.log(np.sum(np.exp(row - mx)))
        losses[i] = log_sum_exp - row[int(labels[i])]
    return losses.astype(np.float32)


def _ref_quantize_per_token(X):
    """Numpy reference: per-row symmetric quantization to int8 range."""
    M, N = X.shape
    out = np.zeros_like(X)
    for i in range(M):
        abs_max = np.max(np.abs(X[i]))
        inv_scale = 127.0 / max(abs_max, 1e-10)
        out[i] = np.clip(np.round(X[i] * inv_scale), -128, 127)
    return out


def _ref_dequantize_per_channel(X_int8, scale, zero_point):
    """Numpy reference: per-channel dequantization."""
    return (X_int8 - zero_point[np.newaxis, :]) * scale[np.newaxis, :]


def _ref_swiglu_packed(X, W):
    """Numpy reference: SwiGLU packed — silu(X[:,:D]) * X[:,D:] @ W."""
    D = X.shape[1] // 2
    gate = X[:, :D]
    up = X[:, D:]
    silu_gate = gate / (1.0 + np.exp(-gate.astype(np.float64))).astype(np.float32)
    hidden = silu_gate * up
    return hidden @ W


def _ref_fused_linear_cross_entropy(X, W, labels):
    """Numpy reference: fused linear projection + cross-entropy."""
    logits = X @ W.T  # [B, V]
    return _ref_cross_entropy(logits, labels)


# ── cross_entropy ───────────────────────────────────────────────

class TestCrossEntropy:
    """OT3 cross_entropy: X[B,V], labels[B] -> loss[B]."""

    @pytest.mark.parametrize("B,V", [(4, 64), (8, 128), (2, 32)])
    def test_cross_entropy_correctness(self, B, V):
        rng = np.random.RandomState(42)
        X = rng.randn(B, V).astype(np.float32)
        labels = np.random.randint(0, V, B).astype(np.float32)
        out = _run("cross_entropy", {"X": X, "labels": labels}, [(B, V), (B,)])
        ref = _ref_cross_entropy(X, labels)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_cross_entropy_uniform(self):
        """Uniform logits → loss ≈ log(V)."""
        B, V = 2, 16
        X = np.ones((B, V), dtype=np.float32)
        labels = np.array([0, V - 1], dtype=np.float32)
        out = _run("cross_entropy", {"X": X, "labels": labels}, [(B, V), (B,)])
        expected = np.log(V)
        np.testing.assert_allclose(out, expected, rtol=1e-4)


# ── quantize_per_token ──────────────────────────────────────────

class TestQuantizePerToken:
    """OT3 quantize_per_token: row-wise symmetric int8 quantization (f32-encoded)."""

    @pytest.mark.parametrize("M,N", [(4, 32), (8, 64)])
    def test_quantize_per_token_correctness(self, M, N):
        rng = np.random.RandomState(42)
        X = rng.randn(M, N).astype(np.float32) * 3.0
        out = _run("quantize_per_token", {"X": X}, [(M, N)])
        ref = _ref_quantize_per_token(X)
        assert out.shape == (M, N)
        np.testing.assert_allclose(out, ref, atol=1.0)  # int8 quantized

    def test_quantize_per_token_range(self):
        """Output values must be in [-128, 127]."""
        rng = np.random.RandomState(7)
        X = rng.randn(4, 16).astype(np.float32) * 10.0
        out = _run("quantize_per_token", {"X": X}, [(4, 16)])
        assert np.all(out >= -128) and np.all(out <= 127)


# ── dequantize_per_channel ──────────────────────────────────────

class TestDequantizePerChannel:
    """OT3 dequantize_per_channel: (X - zp) * scale, per-channel."""

    @pytest.mark.parametrize("M,N", [(4, 32), (8, 64)])
    def test_dequantize_correctness(self, M, N):
        rng = np.random.RandomState(42)
        X_int8 = np.round(rng.randn(M, N) * 50).clip(-128, 127).astype(np.float32)
        scale = (rng.rand(N) * 0.1 + 0.01).astype(np.float32)
        zero_point = np.round(rng.randn(N) * 10).clip(-128, 127).astype(np.float32)
        out = _run("dequantize_per_channel",
                    {"X_int8": X_int8, "scale": scale, "zero_point": zero_point},
                    [(M, N), (N,), (N,)])
        ref = _ref_dequantize_per_channel(X_int8, scale, zero_point)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_dequantize_zero_zp(self):
        """With zero_point=0, output = X * scale."""
        M, N = 2, 8
        X = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 2, dtype=np.float32)
        scale = np.ones(N, dtype=np.float32) * 0.5
        zp = np.zeros(N, dtype=np.float32)
        out = _run("dequantize_per_channel",
                    {"X_int8": X, "scale": scale, "zero_point": zp},
                    [(M, N), (N,), (N,)])
        np.testing.assert_allclose(out, X * 0.5, atol=1e-6)


# ── swiglu_packed ───────────────────────────────────────────────

class TestSwigluPacked:
    """OT3 swiglu_packed: silu(X[:,:D]) * X[:,D:] @ W."""

    @pytest.mark.parametrize("M,D,N", [(4, 32, 16), (2, 16, 8)])
    def test_swiglu_packed_correctness(self, M, D, N):
        rng = np.random.RandomState(42)
        X = rng.randn(M, 2 * D).astype(np.float32)
        W = rng.randn(D, N).astype(np.float32)
        out = _run("swiglu_packed", {"X": X, "W": W}, [(M, 2 * D), (D, N)])
        ref = _ref_swiglu_packed(X, W)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


# ── fused_linear_cross_entropy ──────────────────────────────────

class TestFusedLinearCrossEntropy:
    """OT3 fused_linear_cross_entropy: X[B,D]@W[V,D].T -> logits -> CE loss."""

    @pytest.mark.parametrize("B,D,V", [(4, 32, 64), (2, 16, 32)])
    def test_fused_lce_correctness(self, B, D, V):
        rng = np.random.RandomState(42)
        X = rng.randn(B, D).astype(np.float32) * 0.1
        W = rng.randn(V, D).astype(np.float32) * 0.1
        labels = np.random.randint(0, V, B).astype(np.float32)
        out = _run("fused_linear_cross_entropy",
                    {"X": X, "W": W, "labels": labels},
                    [(B, D), (V, D), (B,)])
        ref = _ref_fused_linear_cross_entropy(X, W, labels)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-2)

    def test_fused_lce_matches_unfused(self):
        """fused_linear_cross_entropy == linear + cross_entropy."""
        rng = np.random.RandomState(7)
        B, D, V = 4, 16, 32
        X = rng.randn(B, D).astype(np.float32) * 0.1
        W = rng.randn(V, D).astype(np.float32) * 0.1
        labels = np.random.randint(0, V, B).astype(np.float32)
        # fused
        out_fused = _run("fused_linear_cross_entropy",
                         {"X": X, "W": W, "labels": labels},
                         [(B, D), (V, D), (B,)])
        # unfused: matmul + cross_entropy
        logits = X @ W.T
        out_unfused = _run("cross_entropy",
                           {"X": logits, "labels": labels},
                           [(B, V), (B,)])
        np.testing.assert_allclose(out_fused, out_unfused, rtol=1e-3, atol=1e-2)
