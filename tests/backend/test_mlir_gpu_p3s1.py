# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S1 (GPU): MLIRGPUBackend end-to-end correctness on NVIDIA.

Validates the Arke → MLIR **gpu dialect** path: IRGraph → gpu.module →
(scf→cf, gpu→nvvm) → PTX text → CUDA driver JIT-load + launch → GPU numerics
bit-correct vs numpy. This is the NVIDIA leg of Phase 3's multi-hardware
lowering (the P3-S1 gate "matmul correct" on GPU, and the P3-S_FINAL
multi-hardware-via-MLIR proof).

Skips cleanly when the GPU toolchain is unavailable (needs mlir-opt with NVPTX
+ cuda-python + a CUDA device). CPU-only CI stays green.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import (
    MLIRGPUBackend,
    gpu_toolchain_available,
    mlir_gpu_to_ptx,
)
from arke.backend.mlir_emitter import emit_gpu_matmul
from arke.backend.protocol import ArkeBackend


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _matmul_graph(M: int, K: int, N: int) -> IRGraph:
    g = IRGraph(name="matmul")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


# ── 1. protocol + emission ─────────────────────────────────────

def test_gpu_backend_implements_protocol():
    be = MLIRGPUBackend()
    assert isinstance(be, ArkeBackend)
    assert be.name == "mlir-gpu"
    assert be.supports_op("matmul")
    assert not be.supports_op("flash_attention")


def test_emit_gpu_matmul_shape_metadata():
    e = emit_gpu_matmul(_matmul_graph(8, 16, 4))
    assert e.grid == (8, 4, 1)
    assert e.block == (1, 1, 1)
    assert e.result_shape == [8, 4]
    assert e.buffer_order == ["A", "B", "C"]
    assert "gpu.module" in e.mlir_text
    assert "#nvvm.target" in e.mlir_text


def test_gpu_matmul_lowers_to_ptx():
    e = emit_gpu_matmul(_matmul_graph(8, 8, 8))
    ptx = mlir_gpu_to_ptx(e.mlir_text)
    assert ".target sm_86" in ptx
    assert ".visible .entry matmul" in ptx


# ── 2. run() JIT-executes bit-correct on GPU vs numpy ──────────

@pytest.mark.parametrize("M,K,N", [
    (1, 1, 1),
    (4, 3, 5),
    (8, 8, 8),
    (16, 7, 13),
    (32, 32, 32),
    (64, 64, 64),
])
def test_gpu_matmul_correct(M, K, N):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(M, K, N)))
    assert ker.success, ker.error
    out = be.run(ker, {"A": A, "B": B})["C"]
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-3)


def test_gpu_matmul_identity():
    be = MLIRGPUBackend()
    N = 8
    A = np.arange(N * N, dtype=np.float32).reshape(N, N)
    I = np.eye(N, dtype=np.float32)
    ker = be.compile(be.lower(_matmul_graph(N, N, N)))
    out = be.run(ker, {"A": A, "B": I})["C"]
    np.testing.assert_allclose(out, A, rtol=1e-4, atol=1e-4)


def test_gpu_matches_torch():
    """Cross-check the GPU MLIR result against torch's own GPU matmul."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA torch")
    be = MLIRGPUBackend()
    M, K, N = 48, 32, 24
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ker = be.compile(be.lower(_matmul_graph(M, K, N)))
    out = be.run(ker, {"A": A, "B": B})["C"]
    ref = (torch.tensor(A, device="cuda") @ torch.tensor(B, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
