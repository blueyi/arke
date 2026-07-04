# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU): elementwise + transcendental ops on NVIDIA via MLIR.

Validates the Arke → MLIR **gpu.module** path for elementwise ops, split into
two classes that both JIT-execute bit-correct against torch on the CUDA driver:

  * pure-arith (relu/neg/add/mul) — no external symbols.
  * transcendental (exp/tanh/sigmoid/silu/gelu/rsqrt) — the ``math.*`` ops emit
    ``__nv_*`` libdevice calls, RESOLVED by linking ``libdevice.bc`` into the gpu
    binary (``gpu-module-to-binary=... l=<libdevice.10.bc>`` in
    ``mlir_gpu._ptx_passes``). libdevice inlines them to native PTX (exp →
    ``ex2.approx`` etc.), so the driver-only load succeeds. This is the correct
    libdevice-linking path — deliberately NOT a pure-arith subset.

Skips cleanly when the GPU toolchain (mlir-opt+NVPTX, cuda-python, CUDA device,
libdevice.bc) is unavailable, keeping CPU-only CI green.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import (
    MLIRGPUBackend,
    gpu_toolchain_available,
    _find_libdevice,
)
from arke.backend.mlir_emitter import GPU_ELEMENTWISE_OPS


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _unary_graph(op: str, M: int, N: int) -> IRGraph:
    g = IRGraph(name=op)
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op=op, inputs={"X": "X"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def _binary_graph(op: str, M: int, N: int) -> IRGraph:
    g = IRGraph(name=op)
    g.add_input("X", dtype="float32", shape=[M, N])
    g.add_input("W", dtype="float32", shape=[M, N])
    g.add_node(IRNode(id="n0", op=op, inputs={"X": "X", "W": "W"}, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def _torch_ref(op, x):
    import torch
    t = torch.tensor(x)
    return {
        "relu": lambda: torch.relu(t),
        "neg": lambda: -t,
        "exp": lambda: torch.exp(t),
        "tanh": lambda: torch.tanh(t),
        "sigmoid": lambda: torch.sigmoid(t),
        "silu": lambda: torch.nn.functional.silu(t),
        # emitter uses the tanh approximation for gelu
        "gelu": lambda: torch.nn.functional.gelu(t, approximate="tanh"),
        "rsqrt": lambda: torch.rsqrt(t),
    }[op]().numpy()


# ── the GPU set includes both classes ──────────────────────────

def test_gpu_set_includes_transcendentals():
    for op in ("relu", "neg", "add", "mul", "exp", "tanh",
               "sigmoid", "silu", "gelu", "rsqrt"):
        assert op in GPU_ELEMENTWISE_OPS
    be = MLIRGPUBackend()
    assert be.supports_op("exp")
    assert be.supports_op("gelu")


def test_libdevice_present_for_transcendentals():
    """Transcendental GPU ops need libdevice.bc; the test env must provide it."""
    assert _find_libdevice() is not None, (
        "libdevice.bc not found — transcendental GPU ops can't lower"
    )


# ── pure-arith unary: bit-correct ──────────────────────────────

@pytest.mark.parametrize("op", ["relu", "neg"])
@pytest.mark.parametrize("M,N", [(1, 1), (4, 4), (16, 16), (32, 8)])
def test_gpu_arith_unary_correct(op, M, N):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((M, N)).astype(np.float32)
    ker = be.compile(be.lower(_unary_graph(op, M, N)))
    assert ker.success, ker.error
    out = be.run(ker, {"X": x})["Y"]
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, _torch_ref(op, x), rtol=1e-4, atol=1e-4)


# ── transcendental unary via libdevice: correct to fp32 tol ────

@pytest.mark.parametrize("op", ["exp", "tanh", "sigmoid", "silu", "gelu", "rsqrt"])
@pytest.mark.parametrize("M,N", [(4, 4), (16, 16), (32, 16)])
def test_gpu_transcendental_correct(op, M, N):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((M, N)).astype(np.float32)
    if op == "rsqrt":  # domain > 0
        x = np.abs(x) + 0.1
    ker = be.compile(be.lower(_unary_graph(op, M, N)))
    assert ker.success, ker.error
    out = be.run(ker, {"X": x})["Y"]
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, _torch_ref(op, x), rtol=1e-3, atol=1e-3)


def test_gpu_exp_lowers_via_libdevice_native_ptx():
    """exp must inline to native ex2.approx (libdevice), not leave __nv_expf."""
    from arke.backend.mlir_gpu import mlir_gpu_to_ptx
    from arke.backend.mlir_emitter import emit_gpu_elementwise
    e = emit_gpu_elementwise(_unary_graph("exp", 4, 4))
    ptx = mlir_gpu_to_ptx(e.mlir_text)
    assert "ex2.approx" in ptx
    assert "__nv_expf" not in ptx  # fully inlined, no dangling extern


# ── binary via libdevice-free arith: bit-correct ───────────────

@pytest.mark.parametrize("op", ["add", "mul"])
@pytest.mark.parametrize("M,N", [(4, 4), (16, 16)])
def test_gpu_binary_correct(op, M, N):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((M, N)).astype(np.float32)
    w = rng.standard_normal((M, N)).astype(np.float32)
    ker = be.compile(be.lower(_binary_graph(op, M, N)))
    assert ker.success, ker.error
    out = be.run(ker, {"X": x, "W": w})["Y"]
    ref = x + w if op == "add" else x * w
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
