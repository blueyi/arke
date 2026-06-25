# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Arke→torch.compile bridge (D7-E1.4, G8[4b]).

CPU-safe: verifies op registration, register_fake tracing, and eager-fallback
correctness. The real-Triton GPU path + dynamo-invocation evidence are covered
by benchmarks/results/phase1/stage8/track4/bridge_2026-*/ (GPU host).
"""

from __future__ import annotations

import torch

from arke.integration.torch_bridge import (
    arke_matmul,
    arke_rmsnorm,
    register_arke_ops,
)

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_DT = torch.float16 if _DEV == "cuda" else torch.float32
# Looser tol on CUDA fp16 (real Triton numerics); tight on CPU fp32 eager.
_RTOL = 1e-2 if _DEV == "cuda" else 1e-4
_ATOL = 1e-2 if _DEV == "cuda" else 1e-4


def test_register_is_idempotent():
    register_arke_ops()
    register_arke_ops()  # second call must not raise
    assert hasattr(torch.ops.arke, "rmsnorm")
    assert hasattr(torch.ops.arke, "matmul")


def test_matmul_correct():
    register_arke_ops()
    a = torch.randn(64, 64, device=_DEV, dtype=_DT)
    b = torch.randn(64, 64, device=_DEV, dtype=_DT)
    out = arke_matmul(a, b)
    assert out.shape == (64, 64)
    assert torch.allclose(out.float(), torch.matmul(a, b).float(), rtol=_RTOL, atol=0.5)


def test_rmsnorm_correct():
    register_arke_ops()
    x = torch.randn(8, 64, device=_DEV, dtype=_DT)
    w = torch.randn(64, device=_DEV, dtype=_DT)
    out = arke_rmsnorm(x, w, 1e-6)
    assert out.shape == (8, 64)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    assert torch.allclose(out.float(), ref, rtol=_RTOL, atol=_ATOL)


def test_register_fake_enables_compile_trace():
    """torch.compile must trace through the custom ops (no graph break crash)."""
    register_arke_ops()

    def model(a, b):
        return arke_matmul(a, b)

    a = torch.randn(16, 16, device=_DEV, dtype=_DT)
    b = torch.randn(16, 16, device=_DEV, dtype=_DT)
    compiled = torch.compile(model, dynamic=False)
    out = compiled(a, b)
    assert out.shape == (16, 16)
    assert torch.allclose(out.float(), torch.matmul(a, b).float(), rtol=_RTOL, atol=0.5)


def test_not_exported_from_arke_init():
    """Scope guardrail: bridge must NOT be re-exported from arke.__init__."""
    import arke

    assert not hasattr(arke, "register_arke_ops")
    assert not hasattr(arke, "torch_bridge")
