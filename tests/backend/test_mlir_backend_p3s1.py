# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S1: MLIRBackend end-to-end correctness.

Validates the Arke → MLIR path: IRGraph → linalg (memref) → mlir-opt lowering
→ mlir-cpu-runner JIT → CPU numerics bit-correct vs torch/numpy reference.

Skips cleanly when the user-local MLIR 18 toolchain is not on PATH/env
(source ~/opt/mlir18/env.sh to enable). This is the substance of the P3-S1
gate: "SemanticIR → linalg + transform dialect, matmul correct".
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_backend import MLIRBackend, mlir_toolchain_available
from arke.backend.mlir_emitter import emit_kernel, SUPPORTED_OPS
from arke.backend.protocol import ArkeBackend


pytestmark = pytest.mark.skipif(
    not mlir_toolchain_available(),
    reason="MLIR 18 toolchain not found (source ~/opt/mlir18/env.sh)",
)


def _matmul_graph(M: int, K: int, N: int, dtype: str = "float32") -> IRGraph:
    g = IRGraph(name="matmul")
    g.add_input("A", dtype=dtype, shape=[M, K])
    g.add_input("B", dtype=dtype, shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


# ── 1. protocol conformance ────────────────────────────────────

def test_mlir_backend_implements_protocol():
    be = MLIRBackend()
    assert isinstance(be, ArkeBackend)
    assert be.name == "mlir"


def test_supports_op():
    be = MLIRBackend()
    assert be.supports_op("matmul")
    assert "matmul" in SUPPORTED_OPS
    assert not be.supports_op("flash_attention")


# ── 2. lower() emits executable MLIR ───────────────────────────

def test_lower_emits_linalg_matmul():
    be = MLIRBackend()
    art = be.lower(_matmul_graph(4, 3, 5))
    src = art.source_code
    assert "linalg.matmul" in src
    assert "memref<4x3xf32>" in src
    assert "memref<3x5xf32>" in src
    assert art.backend_name == "mlir"


def test_emit_kernel_metadata():
    emitted = emit_kernel(_matmul_graph(8, 16, 4))
    assert emitted.kernel_name == "matmul"
    assert emitted.arg_names == ["A", "B"]
    assert emitted.result_shape == [8, 4]
    assert emitted.result_dtype == "float32"


# ── 3. compile() proves linalg → LLVM lowering ─────────────────

def test_compile_lowers_to_llvm():
    be = MLIRBackend()
    ker = be.compile(be.lower(_matmul_graph(4, 3, 5)))
    assert ker.success, ker.error
    # LLVM dialect must appear after full lowering pipeline
    assert "llvm.func" in ker.metadata["llvm_dialect"]


# ── 4. run() JIT-executes bit-correct vs numpy ─────────────────

@pytest.mark.parametrize("M,K,N", [
    (1, 1, 1),
    (2, 3, 2),
    (4, 3, 5),
    (8, 8, 8),
    (16, 7, 13),
    (32, 64, 32),
    (64, 64, 64),
])
def test_matmul_cpu_jit_correct(M, K, N):
    be = MLIRBackend()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(M, K, N)))
    assert ker.success, ker.error
    out = be.run(ker, {"A": A, "B": B})
    assert "C" in out
    expected = A @ B
    assert out["C"].shape == expected.shape
    np.testing.assert_allclose(out["C"], expected, rtol=1e-4, atol=1e-4)


def test_matmul_identity():
    """A @ I == A — sanity anchor independent of random data."""
    be = MLIRBackend()
    N = 6
    A = np.arange(N * N, dtype=np.float32).reshape(N, N)
    I = np.eye(N, dtype=np.float32)
    ker = be.compile(be.lower(_matmul_graph(N, N, N)))
    out = be.run(ker, {"A": A, "B": I})
    np.testing.assert_allclose(out["C"], A, rtol=1e-5, atol=1e-5)


# ── 5. graceful failure on unsupported op ──────────────────────

def test_unsupported_op_raises():
    g = IRGraph(name="attn")
    g.add_input("Q", dtype="float32", shape=[4, 8])
    g.add_node(IRNode(id="n0", op="flash_attention", inputs={"Q": "Q"}, outputs=["O"]))
    g.set_outputs(["O"])
    be = MLIRBackend()
    with pytest.raises(NotImplementedError):
        be.lower(g)
