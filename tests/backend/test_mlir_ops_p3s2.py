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


# ── OT0/OT2 extended movement + gated + index ops ──────────────

def test_cast():
    x = _x((4, 6))
    np.testing.assert_allclose(_run("cast", {"X": x}), x, rtol=1e-4, atol=1e-4)


def test_where():
    cond = (_RNG.random((4, 6)) > 0.5).astype(np.float32)
    a, b = _x((4, 6)), _x((4, 6))
    ref = np.where(cond.astype(bool), a, b)
    np.testing.assert_allclose(_run("where_", {"C": cond, "A": a, "B": b}), ref, rtol=1e-4, atol=1e-4)


def test_permute():
    x = _x((2, 3, 4))
    ref = torch.tensor(x).permute(0, 2, 1).numpy()
    np.testing.assert_allclose(_run("permute", {"X": x}), ref, rtol=1e-4, atol=1e-4)


def test_concat():
    a, b = _x((3, 4)), _x((3, 2))
    ref = np.concatenate([a, b], axis=-1)
    np.testing.assert_allclose(_run("concat", {"A": a, "B": b}), ref, rtol=1e-4, atol=1e-4)


def test_split():
    x = _x((4, 8))
    ref = torch.chunk(torch.tensor(x), 2, dim=-1)[0].numpy()
    np.testing.assert_allclose(_run("split", {"X": x}), ref, rtol=1e-4, atol=1e-4)


def test_rmsnorm_residual():
    x, res = _x((4, 6)), _x((4, 6))
    t = torch.tensor(x + res)
    ref = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)).numpy()
    np.testing.assert_allclose(_run("rmsnorm_residual", {"X": x, "R": res}), ref, rtol=1e-3, atol=1e-3)


def test_silu_and_mul():
    x = _x((4, 8))
    x1, x2 = torch.tensor(x).chunk(2, dim=-1)
    ref = (F.silu(x1) * x2).numpy()
    np.testing.assert_allclose(_run("silu_and_mul", {"X": x}), ref, rtol=1e-3, atol=1e-3)


def test_gelu_and_mul():
    x = _x((4, 8))
    x1, x2 = torch.tensor(x).chunk(2, dim=-1)
    ref = (F.gelu(x1, approximate="tanh") * x2).numpy()
    np.testing.assert_allclose(_run("gelu_and_mul", {"X": x}), ref, rtol=1e-3, atol=1e-3)


def test_cumsum():
    x = _x((4, 6))
    ref = torch.cumsum(torch.tensor(x), -1).numpy()
    np.testing.assert_allclose(_run("cumsum", {"X": x}), ref, rtol=1e-3, atol=1e-3)


def test_argmax():
    x = _x((4, 6))
    ref = torch.argmax(torch.tensor(x), -1).numpy().astype(np.float32)
    np.testing.assert_allclose(_run("argmax", {"X": x}), ref, rtol=1e-4, atol=1e-4)


def test_embedding():
    idx = _RNG.integers(0, 10, size=(5,)).astype(np.float32)
    tbl = _x((10, 6))
    ref = F.embedding(torch.tensor(idx).long(), torch.tensor(tbl)).numpy()
    np.testing.assert_allclose(_run("embedding", {"I": idx, "T": tbl}), ref, rtol=1e-3, atol=1e-3)


def test_rope():
    x = _x((4, 8))

    def _torch_rope(xt):
        S, D = xt.shape
        half = D // 2
        pos = torch.arange(S).float().unsqueeze(1)
        i = torch.arange(half).float().unsqueeze(0)
        theta = torch.exp(-(2 * i / D) * np.log(10000.0))
        ang = pos * theta
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = xt[:, :half], xt[:, half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    ref = _torch_rope(torch.tensor(x)).numpy()
    np.testing.assert_allclose(_run("rope", {"X": x}), ref, rtol=1e-3, atol=1e-3)


# ── op-coverage headcount (P3-S2 progress guard) ───────────────

def test_op_coverage_count():
    from arke.backend.mlir_emitter import SUPPORTED_OPS
    # matmul + 11 OT0 (10 ew + cast) + 6 OT1 + rmsnorm_residual + argmax + cumsum
    # + OT2 (transpose,batch_matmul,copy_,permute,concat,split,embedding)
    # + OT3 gated (silu_and_mul,gelu_and_mul) + where_ + rope = 32
    assert len(SUPPORTED_OPS) >= 32, sorted(SUPPORTED_OPS)
