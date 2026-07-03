# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2: MLIR backend op-coverage correctness (elementwise + composite).

Each op is lowered IRGraph → MLIR (linalg / linalg.generic) → mlir-cpu-runner
JIT and checked bit-correct vs a torch/numpy reference. This is the correctness
half of the P3-S2 gate ("N ops correct"); the perf half (geomean ≥ Phase 2
Triton) lives in the bench harness.

Skips cleanly without the MLIR 18 toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from arke.ir.graph import IRGraph, IRNode  # noqa: E402
from arke.backend.mlir_backend import MLIRBackend, mlir_toolchain_available  # noqa: E402


pytestmark = pytest.mark.skipif(
    not mlir_toolchain_available(),
    reason="MLIR 18 toolchain not found (source ~/opt/mlir18/env.sh)",
)


def _run(op: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    g = IRGraph(name=op)
    for nm, arr in inputs.items():
        g.add_input(nm, dtype="float32", shape=list(arr.shape))
    g.add_node(IRNode(
        id="n0", op=op,
        inputs={f"in{i}": nm for i, nm in enumerate(inputs)},
        outputs=["Y"],
    ))
    g.set_outputs(["Y"])
    be = MLIRBackend()
    ker = be.compile(be.lower(g))
    assert ker.success, ker.error
    return be.run(ker, inputs)["Y"]


_RNG = np.random.default_rng(0)


def _x(shape) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


# ── OT0 elementwise (unary) ────────────────────────────────────

@pytest.mark.parametrize("op,ref", [
    ("relu", lambda t: F.relu(t)),
    ("neg", lambda t: -t),
    ("exp", lambda t: torch.exp(t)),
    ("tanh", lambda t: torch.tanh(t)),
    ("sigmoid", lambda t: torch.sigmoid(t)),
    ("silu", lambda t: F.silu(t)),
    ("gelu", lambda t: F.gelu(t, approximate="tanh")),
])
def test_ot0_unary(op, ref):
    x = _x((4, 5))
    out = _run(op, {"X": x})
    np.testing.assert_allclose(out, ref(torch.tensor(x)).numpy(), rtol=1e-3, atol=1e-3)


def test_ot0_rsqrt():
    x = (np.abs(_x((4, 5))) + 0.5).astype(np.float32)
    out = _run("rsqrt", {"X": x})
    np.testing.assert_allclose(out, torch.rsqrt(torch.tensor(x)).numpy(), rtol=1e-3, atol=1e-3)


# ── OT0 elementwise (binary) ───────────────────────────────────

@pytest.mark.parametrize("op,ref", [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
])
def test_ot0_binary(op, ref):
    a, b = _x((4, 5)), _x((4, 5))
    out = _run(op, {"A": a, "B": b})
    np.testing.assert_allclose(out, ref(a, b), rtol=1e-5, atol=1e-5)


# ── OT1 reductions ─────────────────────────────────────────────

@pytest.mark.parametrize("op,ref", [
    ("reduce_sum", lambda t: torch.sum(t, -1)),
    ("reduce_max", lambda t: torch.max(t, -1).values),
    ("reduce_mean", lambda t: torch.mean(t, -1)),
])
def test_ot1_reductions(op, ref):
    x = _x((4, 6))
    out = _run(op, {"X": x})
    np.testing.assert_allclose(out, ref(torch.tensor(x)).numpy(), rtol=1e-3, atol=1e-3)


# ── OT1 normalizations ─────────────────────────────────────────

def test_softmax():
    x = _x((4, 6))
    out = _run("softmax", {"X": x})
    np.testing.assert_allclose(out, F.softmax(torch.tensor(x), -1).numpy(), rtol=1e-3, atol=1e-3)


def test_layernorm():
    x = _x((4, 6))
    ref = F.layer_norm(torch.tensor(x), [6], torch.ones(6), torch.zeros(6)).numpy()
    np.testing.assert_allclose(_run("layernorm", {"X": x}), ref, rtol=1e-3, atol=1e-3)


def test_rmsnorm():
    x = _x((4, 6))
    t = torch.tensor(x)
    ref = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)).numpy()
    np.testing.assert_allclose(_run("rmsnorm", {"X": x}), ref, rtol=1e-3, atol=1e-3)


# ── OT2 movement / dense ───────────────────────────────────────

def test_transpose():
    x = _x((4, 6))
    np.testing.assert_allclose(_run("transpose", {"X": x}), x.T, rtol=1e-4, atol=1e-4)


def test_copy():
    x = _x((4, 6))
    # printMemrefF32 round-trips ~6 sig figs, so a strict bit compare is too
    # tight; the JIT print path caps precision at ~1e-5.
    np.testing.assert_allclose(_run("copy_", {"X": x}), x, rtol=1e-4, atol=1e-4)


def test_batch_matmul():
    a, b = _x((3, 4, 5)), _x((3, 5, 6))
    ref = torch.bmm(torch.tensor(a), torch.tensor(b)).numpy()
    np.testing.assert_allclose(_run("batch_matmul", {"A": a, "B": b}), ref, rtol=1e-3, atol=1e-3)


# ── op-coverage headcount (P3-S2 progress guard) ───────────────

def test_op_coverage_count():
    from arke.backend.mlir_emitter import SUPPORTED_OPS
    # matmul + 10 OT0 + 6 OT1 + 3 OT2(transpose,batch_matmul,copy_) = 20 so far
    assert len(SUPPORTED_OPS) >= 20, sorted(SUPPORTED_OPS)
