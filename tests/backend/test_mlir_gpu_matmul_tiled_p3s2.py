# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU perf): shared-memory tiled matmul.

The P3-S1 correctness matmul is one-thread-per-output with a global-memory
K-loop (block=(1,1,1)) — orders of magnitude off cuBLAS. This suite validates
the P3-S2 perf kernel ``emit_gpu_matmul_tiled``: grid=(N/T,M/T), block=(T,T),
each block computes one TxT output tile, cooperatively staging A/B tiles into
**workgroup (shared) memory** (via gpu.func workgroup attributions → real
``.shared`` PTX, NOT malloc+addrspacecast), gpu.barrier, shared-mem inner
product, accumulate.

Checks: (1) tile-aligned shapes emit the tiled kernel + are bit-correct on the
CUDA driver vs numpy/torch; (2) non-aligned shapes fall back to the correctness
kernel through MLIRGPUBackend.lower; (3) the tiled PTX actually uses .shared
memory (regression guard against the malloc+addrspacecast illegal-address bug).

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
    emit_gpu_matmul_tiled,
    GPU_MM_TILE,
)


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


# ── emission + grid/block metadata ─────────────────────────────

def test_tiled_grid_block_metadata():
    T = GPU_MM_TILE
    e = emit_gpu_matmul_tiled(_mm(64, 32, 48))
    assert e.block == (T, T, 1)
    assert e.grid == (48 // T, 64 // T, 1)
    assert "workgroup(" in e.mlir_text
    assert "gpu.barrier" in e.mlir_text
    assert "#gpu.address_space<workgroup>" in e.mlir_text


def test_tiled_rejects_non_aligned():
    with pytest.raises(NotImplementedError):
        emit_gpu_matmul_tiled(_mm(17, 16, 16))
    with pytest.raises(NotImplementedError):
        emit_gpu_matmul_tiled(_mm(16, 15, 16))


def test_tiled_ptx_uses_shared_memory():
    """Regression guard: tiled kernel must lower to real .shared PTX.

    A plain memref.alloc in workgroup space lowers to malloc+addrspacecast →
    illegal .shared address (CUDA_ERROR_ILLEGAL_ADDRESS at runtime). The
    workgroup-attribution form must instead emit a .shared global.
    """
    e = emit_gpu_matmul_tiled(_mm(32, 32, 32))
    ptx = mlir_gpu_to_ptx(e.mlir_text)
    assert ".shared" in ptx
    assert "malloc" not in ptx


# ── correctness on the CUDA driver ─────────────────────────────

@pytest.mark.parametrize("M,K,N", [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (32, 64, 48),   # rectangular, all tile-aligned
])
def test_tiled_correct_vs_numpy(M, K, N):
    be = MLIRGPUBackend()
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    # lower() should pick the tiled kernel for these aligned shapes
    art = be.lower(_mm(M, K, N))
    assert "workgroup(" in art.source_code
    ker = be.compile(art)
    assert ker.success, ker.error
    out = be.run(ker, {"A": A, "B": B})["C"]
    assert out.shape == (M, N)
    # f32 tiled accumulation: looser tol on larger K
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-3 * max(1, K / 64))


def test_tiled_matches_torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA torch")
    be = MLIRGPUBackend()
    M, K, N = 128, 128, 128
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    out = be.run(be.compile(be.lower(_mm(M, K, N))), {"A": A, "B": B})["C"]
    ref = (torch.tensor(A, device="cuda") @ torch.tensor(B, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-2)


# ── backend fallback path for non-aligned shapes ───────────────

def test_backend_falls_back_for_non_aligned():
    be = MLIRGPUBackend()
    art = be.lower(_mm(17, 13, 19))  # non-aligned → correctness kernel
    assert "workgroup(" not in art.source_code  # not the tiled kernel
    rng = np.random.default_rng(1)
    A = rng.standard_normal((17, 13)).astype(np.float32)
    B = rng.standard_normal((13, 19)).astype(np.float32)
    out = be.run(be.compile(art), {"A": A, "B": B})["C"]
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-3)


# ── perf: tiled must beat the correctness kernel ───────────────

def test_tiled_faster_than_correctness_kernel():
    """Sanity: shared-mem tiling must speed up a mid-size matmul, not slow it."""
    from arke.backend.mlir_emitter import emit_gpu_matmul
    from arke.backend.protocol import CompiledKernel
    be = MLIRGPUBackend()
    M = K = N = 256
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    tiled_ms = be.benchmark(be.compile(be.lower(_mm(M, K, N))), {"A": A, "B": B},
                            iters=20, warmup=5)
    # force the correctness kernel
    e_corr = emit_gpu_matmul(_mm(M, K, N))
    ptx = mlir_gpu_to_ptx(e_corr.mlir_text)
    ker_corr = CompiledKernel.ok(fn=None, backend_name="mlir-gpu", emitted=e_corr, ptx=ptx)
    corr_ms = be.benchmark(ker_corr, {"A": A, "B": B}, iters=10, warmup=3)
    assert tiled_ms < corr_ms, f"tiled {tiled_ms:.3f}ms not faster than corr {corr_ms:.3f}ms"
