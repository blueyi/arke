# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for s7.followup.1 — bench_l1 shape encoding.

Background: the bench harness extracts canonical shape attrs into (M, N, K)
via a lossy squash at bench_l1.py:766-785. For ops whose shape class has
extra dims beyond M/N/K (BatchMatmulShape, GroupedMatmulShape,
AttentionShape with Hkv, GatedShape) the squash produced WRONG workloads:

  * batch_matmul llama-attn-2k: bench measured 0.5 GFLOP (vs canonical 34 GFLOP)
  * grouped_matmul: dropped expert count E entirely
  * GQA qwen25 shapes: num_kv_groups = (B*H)//4 = 7 instead of canonical Hkv=4
  * silu_and_mul / gelu_and_mul: M=N=K=0 → randn(1, 2), 6μs of useless data

Fix: each op's _make_l1_correctness_inputs branch now reads
_runtime_ctx.get_current_shape() first (same pattern as cross_attention)
and only falls back to (M, N, K) when no runtime context is set.

This test pins the post-fix behaviour: when runtime ctx holds the canonical
shape, _make_l1_correctness_inputs must produce tensors with the canonical
dims. If anyone reintroduces the squash, this test catches it immediately.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for bench_l1 input construction", allow_module_level=True)

from benchmarks.bench_l1 import _make_l1_correctness_inputs
from benchmarks.baselines._runtime_ctx import set_current_shape, clear_current_shape
from benchmarks.shapes import (
    BATCH_MATMUL_SHAPES,
    GROUPED_MATMUL_SHAPES,
    GQA_SHAPES,
    GATED_SHAPES,
)


@pytest.fixture(autouse=True)
def _cleanup_runtime_ctx():
    yield
    clear_current_shape()


def _replay_bench_squash(shape):
    """Reproduce bench_l1.py:766-785 shape→(M,N,K) extraction."""
    M = getattr(shape, "M", 0)
    N = getattr(shape, "N", 0)
    K = getattr(shape, "K", 0)
    if hasattr(shape, "B") and hasattr(shape, "H") and hasattr(shape, "S"):
        M = shape.B * shape.H
        N = shape.S
        K = getattr(shape, "D", 64)
    elif hasattr(shape, "B") and not hasattr(shape, "H"):
        M = getattr(shape, "B", M)
    return M, N, K


@pytest.mark.parametrize("shape", BATCH_MATMUL_SHAPES)
def test_batch_matmul_uses_canonical_shape(shape):
    """A=(B,M,K), B=(B,K,N) — not the legacy (max(K,4), M, N) bug shape."""
    M, N, K = _replay_bench_squash(shape)
    set_current_shape(shape)
    A, Bt = _make_l1_correctness_inputs("batch_matmul", M, N, K)
    assert tuple(A.shape) == (shape.B, shape.M, shape.K), \
        f"{shape.tag}: A shape mismatch, got {tuple(A.shape)} want ({shape.B},{shape.M},{shape.K})"
    assert tuple(Bt.shape) == (shape.B, shape.K, shape.N), \
        f"{shape.tag}: B shape mismatch, got {tuple(Bt.shape)} want ({shape.B},{shape.K},{shape.N})"


@pytest.mark.parametrize("shape", GROUPED_MATMUL_SHAPES)
def test_grouped_matmul_uses_canonical_shape(shape):
    """A=(B,M,K), B=(E,K,N) — preserves expert count E."""
    M, N, K = _replay_bench_squash(shape)
    set_current_shape(shape)
    A, Bt = _make_l1_correctness_inputs("grouped_matmul", M, N, K)
    assert tuple(A.shape) == (shape.B, shape.M, shape.K), \
        f"{shape.tag}: A shape mismatch, got {tuple(A.shape)} want ({shape.B},{shape.M},{shape.K})"
    assert tuple(Bt.shape) == (shape.E, shape.K, shape.N), \
        f"{shape.tag}: B shape mismatch, got {tuple(Bt.shape)} want ({shape.E},{shape.K},{shape.N})"


@pytest.mark.parametrize("shape", GQA_SHAPES)
def test_grouped_query_attention_uses_canonical_Hkv(shape):
    """Q=(B*H,S,D), K=V=(B*Hkv,S,D) — Hkv honoured, not M//4 heuristic."""
    if shape.Hkv is None:
        pytest.skip(f"{shape.tag}: no Hkv on this shape")
    M, N, K = _replay_bench_squash(shape)
    set_current_shape(shape)
    Q, K_, V = _make_l1_correctness_inputs("grouped_query_attention", M, N, K)
    BH = shape.B * shape.H
    BHkv = shape.B * shape.Hkv
    assert tuple(Q.shape) == (BH, shape.S, shape.D), \
        f"{shape.tag}: Q shape mismatch, got {tuple(Q.shape)} want ({BH},{shape.S},{shape.D})"
    assert tuple(K_.shape) == (BHkv, shape.S, shape.D), \
        f"{shape.tag}: K shape mismatch, got {tuple(K_.shape)} want ({BHkv},{shape.S},{shape.D})"
    assert tuple(V.shape) == (BHkv, shape.S, shape.D), \
        f"{shape.tag}: V shape mismatch, got {tuple(V.shape)} want ({BHkv},{shape.S},{shape.D})"


@pytest.mark.parametrize("op", ["silu_and_mul", "gelu_and_mul"])
@pytest.mark.parametrize("shape", GATED_SHAPES)
def test_gated_activation_uses_canonical_shape(op, shape):
    """x=(seq, ffn_x2) — not the degenerate (1, 2) bug shape."""
    M, N, K = _replay_bench_squash(shape)
    set_current_shape(shape)
    (x,) = _make_l1_correctness_inputs(op, M, N, K)
    assert tuple(x.shape) == (shape.seq, shape.ffn_x2), \
        f"{op}/{shape.tag}: x shape mismatch, got {tuple(x.shape)} want ({shape.seq},{shape.ffn_x2})"


def test_bench_falls_back_when_no_runtime_ctx():
    """When runtime ctx is empty (e.g. unit tests calling directly), the
    legacy (M, N, K) path still works — does not crash, returns sensible
    tensors. This protects callers that don't set_current_shape."""
    clear_current_shape()
    # batch_matmul fallback
    A, B = _make_l1_correctness_inputs("batch_matmul", 32, 64, 16)
    assert A.dim() == 3 and B.dim() == 3
    # grouped_matmul fallback
    A, B = _make_l1_correctness_inputs("grouped_matmul", 16, 64, 32)
    assert A.dim() == 3 and B.dim() == 3
    # GQA fallback
    Q, K_, V = _make_l1_correctness_inputs("grouped_query_attention", 16, 128, 64)
    assert Q.dim() == 3 and K_.dim() == 3 and V.dim() == 3
    # silu_and_mul fallback
    (x,) = _make_l1_correctness_inputs("silu_and_mul", 32, 128, 0)
    assert x.dim() == 2
