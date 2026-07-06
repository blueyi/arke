# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU perf): register-blocked (2D thread-tile) matmul.

The shared-memory tiled kernel is 1-output-per-thread; ``emit_gpu_matmul_regblock``
adds CUTLASS-style register blocking — each thread computes a TM x TN micro-tile
of C in private (register) memrefs, reusing each shared-memory fetch TM/TN times.
Block tile BM x BN, K-step BK, threads=(BN/TN)x(BM/TM) (16x16=256 by default).

Checks: (1) conforming shapes emit the regblock kernel (workgroup + private
attributions) and are bit-correct on the CUDA driver vs numpy/torch;
(2) MLIRGPUBackend.lower picks regblock at the top of the perf ladder for
BM/BN/BK-conforming shapes; (3) regblock is faster than the plain tiled kernel.

Skips cleanly without the GPU toolchain.
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
from arke.backend.mlir_emitter import (
    emit_gpu_matmul_regblock,
    emit_gpu_matmul_tiled,
    GPU_MM_BM, GPU_MM_BN, GPU_MM_BK, GPU_MM_TM, GPU_MM_TN,
)
from arke.backend.protocol import CompiledKernel


pytestmark = pytest.mark.skipif(
    not gpu_toolchain_available(),
    reason="GPU toolchain unavailable (needs mlir-opt+NVPTX, cuda-python, CUDA device)",
)


def _mm(M: int, K: int, N: int) -> IRGraph:
    g = IRGraph(name="matmul")
    g.add_input("A", dtype="float32", shape=[M, K])
    g.add_input("B", dtype="float32", shape=[K, N])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.set_outputs(["C"])
    return g


def _compile(emitted):
    ptx = mlir_gpu_to_ptx(emitted.mlir_text)
    return CompiledKernel.ok(fn=None, backend_name="mlir-gpu", emitted=emitted, ptx=ptx)


# ── emission + grid/block metadata ─────────────────────────────

def test_regblock_grid_block_metadata():
    e = emit_gpu_matmul_regblock(_mm(128, 64, 256))
    # block = (BN/TN, BM/TM); grid = (N/BN, M/BM)
    assert e.block == (GPU_MM_BN // GPU_MM_TN, GPU_MM_BM // GPU_MM_TM, 1)
    assert e.grid == (256 // GPU_MM_BN, 128 // GPU_MM_BM, 1)
    assert "workgroup(" in e.mlir_text
    assert "private(" in e.mlir_text
    assert "#gpu.address_space<private>" in e.mlir_text
    assert "gpu.barrier" in e.mlir_text


def test_regblock_rejects_non_conforming():
    # M not %BM
    with pytest.raises(NotImplementedError):
        emit_gpu_matmul_regblock(_mm(96, 64, 64))
    # K not %BK
    with pytest.raises(NotImplementedError):
        emit_gpu_matmul_regblock(_mm(64, 24, 64))


def test_regblock_ptx_uses_shared_and_regs():
    e = emit_gpu_matmul_regblock(_mm(64, 64, 64))
    ptx = mlir_gpu_to_ptx(e.mlir_text)
    assert ".shared" in ptx
    assert "malloc" not in ptx


# ── correctness on the CUDA driver ─────────────────────────────

@pytest.mark.parametrize("M,K,N", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (64, 128, 192),   # rectangular, BM/BN/BK-conforming
    (128, 64, 64),
])
def test_regblock_correct_vs_numpy(M, K, N):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    out = be.run(_compile(emit_gpu_matmul_regblock(_mm(M, K, N))), {"A": A, "B": B})["C"]
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-3 * max(1, K / 64))


def test_backend_picks_regblock_for_conforming():
    """lower() must select regblock (private attribution) for a 64-aligned shape."""
    be = MLIRGPUBackend()
    art = be.lower(_mm(128, 128, 128))
    assert "private(" in art.source_code   # regblock signature
    rng = np.random.default_rng(3)
    A = rng.standard_normal((128, 128)).astype(np.float32)
    B = rng.standard_normal((128, 128)).astype(np.float32)
    out = be.run(be.compile(art), {"A": A, "B": B})["C"]
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-2)


def test_backend_ladder_falls_to_tiled_then_correctness():
    be = MLIRGPUBackend()
    # 32-aligned: regblock now picks BM=32 TM=2 for small shapes (private used)
    art = be.lower(_mm(32, 32, 32))
    assert "private(" in art.source_code  # small-shape regblock
    assert "workgroup(" in art.source_code
    # fully non-aligned → correctness kernel (neither)
    art2 = be.lower(_mm(17, 13, 19))
    assert "private(" not in art2.source_code
    assert "workgroup(" not in art2.source_code


def test_regblock_matches_torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA torch")
    be = MLIRGPUBackend()
    M, K, N = 256, 256, 256
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    out = be.run(_compile(emit_gpu_matmul_regblock(_mm(M, K, N))), {"A": A, "B": B})["C"]
    ref = (torch.tensor(A, device="cuda") @ torch.tensor(B, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-1)


# ── perf: regblock must beat the plain tiled kernel ────────────

def test_regblock_faster_than_tiled():
    be = MLIRGPUBackend()
    M = K = N = 512
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    rb_ms = be.benchmark(_compile(emit_gpu_matmul_regblock(_mm(M, K, N))),
                         {"A": A, "B": B}, iters=20, warmup=8)
    t_ms = be.benchmark(_compile(emit_gpu_matmul_tiled(_mm(M, K, N))),
                        {"A": A, "B": B}, iters=20, warmup=8)
    assert rb_ms < t_ms, f"regblock {rb_ms:.3f}ms not faster than tiled {t_ms:.3f}ms"
