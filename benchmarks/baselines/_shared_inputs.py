# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared perf-input builder for baseline runners.

Single source of truth for the input tensors used by ``Runner.get_fn`` during
perf timing. Historically every runner re-implemented input construction in its
own ``get_fn`` using the squashed ``(M, N, K)`` convention. For ops whose
benchmark shape dataclass is NOT a flat ``(M, N, K)`` — ``batch_matmul``
(B,M,K,N), ``grouped_matmul`` (B,E,M,K,N), ``silu_and_mul`` / ``gelu_and_mul``
(seq, ffn_x2) — that squash built the WRONG-shaped workload, so PERF_ALL latency
for those ops measured something unrelated to its shape tag. The correctness
path already consulted ``get_current_shape()``; the perf path diverged.

This module centralizes the canonical-shape construction so every runner stays
in sync. See docs/benchmark/harness-perf-shape-encoding-bug.md.

Design:
  * ``build_gemm_perf_inputs`` returns ``(A, B[, indices])`` for the dense /
    batched / grouped matmul family, consulting the runtime shape context.
  * ``build_gated_perf_inputs`` returns the single packed ``X[seq, 2*ffn]``.
  * Each returns ``None`` when it has no opinion (caller keeps its legacy path).
  * A no-context fallback (unit tests, ad-hoc calls) preserves the old squash
    behaviour so nothing that relied on it breaks.
"""

from __future__ import annotations

import torch

from benchmarks.baselines._runtime_ctx import get_current_shape


def build_batch_matmul_inputs(
    M: int, N: int, K: int, dtype: torch.dtype, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """A[B, M, K] @ B[B, K, N] -> [B, M, N].

    Canonical (B, M, K, N) from the runtime shape ctx; legacy squash (K as
    batch) only as the no-context fallback.
    """
    shape = get_current_shape()
    if (
        shape is not None
        and all(hasattr(shape, a) for a in ("B", "M", "K", "N"))
        and not hasattr(shape, "E")  # disambiguate from GroupedMatmulShape
    ):
        A = torch.randn(shape.B, shape.M, shape.K, device=device, dtype=dtype)
        Bt = torch.randn(shape.B, shape.K, shape.N, device=device, dtype=dtype)
        return A, Bt
    # Fallback: legacy squash convention.
    B_dim = max(K, 4)
    A = torch.randn(B_dim, M, N, device=device, dtype=dtype)
    Bt = torch.randn(B_dim, N, N, device=device, dtype=dtype)
    return A, Bt


def build_grouped_matmul_inputs(
    M: int, N: int, K: int, dtype: torch.dtype, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X[B, M, K] @ W[indices[b], K, N] -> [B, M, N] (returns X, W, indices)."""
    shape = get_current_shape()
    if shape is not None and all(hasattr(shape, a) for a in ("B", "E", "M", "K", "N")):
        A = torch.randn(shape.B, shape.M, shape.K, device=device, dtype=dtype)
        W = torch.randn(shape.E, shape.K, shape.N, device=device, dtype=dtype)
        idx = torch.randint(0, shape.E, (shape.B,), device=device, dtype=torch.int64)
        return A, W, idx
    # Fallback: legacy squash convention.
    E = 4
    B_dim = max(M, 1)
    A = torch.randn(B_dim, N, max(K, 1), device=device, dtype=dtype)
    W = torch.randn(E, max(K, 1), N, device=device, dtype=dtype)
    idx = torch.randint(0, E, (B_dim,), device=device, dtype=torch.int64)
    return A, W, idx


def build_gated_perf_inputs(
    M: int, N: int, dtype: torch.dtype, device: str = "cuda"
) -> torch.Tensor:
    """Gated activation packed input X[seq, 2*ffn] (gate | value)."""
    shape = get_current_shape()
    if shape is not None and hasattr(shape, "seq") and hasattr(shape, "ffn_x2"):
        return torch.randn(shape.seq, shape.ffn_x2, device=device, dtype=dtype)
    # Fallback: legacy squash convention.
    return torch.randn(M, N * 2, device=device, dtype=dtype)
