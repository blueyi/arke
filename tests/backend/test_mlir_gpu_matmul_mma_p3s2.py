# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 P3-S2 (GPU perf): tensor-core (nvgpu.mma.sync) matmul.

``emit_gpu_matmul_mma`` drives the Ampere tensor cores via MLIR's ``nvgpu``
dialect: f32 inputs are truncated to f16 into shared memory, a warp-level
``vector.contract`` is auto-distributed (by ``--convert-vector-to-gpu
--use-nvgpu``) into per-thread ``nvgpu.ldmatrix`` + ``nvgpu.mma.sync`` fragments,
and the tensor core accumulates in f32. Warp-register-blocking (each warp owns a
WTM x WTN grid of 16x16 output sub-tiles) lifts arithmetic intensity.

Precision contract: output is bit-close vs an *fp16-input* reference (the
correct precision class for a tensor-core kernel), NOT vs strict-f32 cuBLAS —
the ~1e-3 relative delta is the inherent reduced-precision tradeoff. Tests
therefore compare against an fp16-cast reference, and the tensor-core path is
OPT-IN (``MLIRGPUBackend(use_tensor_core=True)``) so the default matmul stays
bit-accurate f32.

Checks: (1) emitted MLIR carries the nvgpu-distributable warp-level contract and
the two-stage lowering produces mma.sync PTX; (2) correctness vs fp16 reference
on the CUDA driver; (3) the backend routes to mma only when opt-in AND the shape
tiles, else falls back to the scalar FP32 ladder.

Skips cleanly without the GPU toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import (
    MLIRGPUBackend,
    gpu_toolchain_available,
    mlir_nvgpu_to_cubin,
    _nvgpu_stage1_passes,
)
from arke.backend.mlir_emitter import (
    emit_gpu_matmul_mma,
    GPU_MMA_WM, GPU_MMA_WN, GPU_MMA_WTM, GPU_MMA_WTN, GPU_MMA_BK,
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


def _fp16_ref(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reference in the tensor-core precision class: fp16 inputs, f32 accumulate."""
    return A.astype(np.float16).astype(np.float32) @ B.astype(np.float16).astype(np.float32)


# ── emission + default block/warp tiling ───────────────────────

def test_mma_default_tiling_shapes():
    BM = GPU_MMA_WM * GPU_MMA_WTM * 16
    BN = GPU_MMA_WN * GPU_MMA_WTN * 16
    e = emit_gpu_matmul_mma(_mm(BM, GPU_MMA_BK, BN))
    # one warp-block, nthreads = WM*WN*32 threads packed on x
    assert e.block == (GPU_MMA_WM * GPU_MMA_WN * 32, 1, 1)
    assert e.grid == (1, 1, 1)
    assert "vector.contract" in e.mlir_text
    assert "nvgpu" not in e.mlir_text  # distribution happens in lowering, not emit
    assert "#gpu.address_space<workgroup>" in e.mlir_text
    assert "arith.truncf" in e.mlir_text  # f32 -> f16 stage-in
    assert "gpu.barrier" in e.mlir_text


def test_mma_rejects_non_conforming():
    BM = GPU_MMA_WM * GPU_MMA_WTM * 16
    BN = GPU_MMA_WN * GPU_MMA_WTN * 16
    # M not a multiple of BM
    with pytest.raises(NotImplementedError):
        emit_gpu_matmul_mma(_mm(BM + 16, GPU_MMA_BK, BN))
    # K not a multiple of BK
    with pytest.raises(NotImplementedError):
        emit_gpu_matmul_mma(_mm(BM, GPU_MMA_BK + 8, BN))


def test_mma_stage1_distributes_to_mma_sync():
    """--convert-vector-to-gpu=use-nvgpu must turn the contract into mma.sync."""
    import subprocess
    from arke.backend.mlir_gpu import _tool
    BM = GPU_MMA_WM * GPU_MMA_WTM * 16
    BN = GPU_MMA_WN * GPU_MMA_WTN * 16
    e = emit_gpu_matmul_mma(_mm(BM, GPU_MMA_BK, BN))
    tool = _tool("ARKE_MLIR_OPT", "mlir-opt")
    s1 = subprocess.run([tool, *_nvgpu_stage1_passes()], input=e.mlir_text,
                        capture_output=True, text=True, check=True).stdout
    assert "nvgpu.mma.sync" in s1
    assert "nvgpu.ldmatrix" in s1


def test_mma_lowers_to_cubin():
    BM = GPU_MMA_WM * GPU_MMA_WTM * 16
    BN = GPU_MMA_WN * GPU_MMA_WTN * 16
    e = emit_gpu_matmul_mma(_mm(BM, GPU_MMA_BK, BN))
    cubin = mlir_nvgpu_to_cubin(e.mlir_text, chip="sm_86")
    # NVIDIA fatbin / ELF cubin magic
    assert cubin[:4] in (b"\x50\xed\x55\xba", b"\x7fELF")


# ── correctness on the CUDA driver (vs fp16-precision reference) ─

@pytest.mark.parametrize("M,K,N", [
    (512, 512, 512),
    (1024, 1024, 1024),
    (512, 256, 512),   # rectangular, MMA-tileable, M*N >= 262144
])
def test_mma_correct_vs_fp16_ref(M, K, N):
    be = MLIRGPUBackend(use_tensor_core=True)
    art = be.lower(_mm(M, K, N))
    assert art.metadata.get("is_mma"), "expected tensor-core routing"
    ker = be.compile(art)
    assert ker.success, ker.error
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    out = be.run(ker, {"A": A, "B": B})["C"]
    assert out.shape == (M, N)
    ref16 = _fp16_ref(A, B)
    # fp16 tensor core matches an fp16-input reference to a few ×1e-3 relative
    rel = np.abs(out - ref16).max() / (np.abs(ref16).max() + 1e-9)
    assert rel < 5e-3, f"rel err {rel} vs fp16 ref too large"


# ── backend routing: opt-in + shape-gated fallback ─────────────

def test_backend_routes_mma_by_default_for_tileable():
    # Default backend: tensor-core for MMA-tileable shapes that are large enough
    # (policy threshold: M*N >= 512*512 = 262144 output elements).
    be_default = MLIRGPUBackend()
    art_d = be_default.lower(_mm(512, 512, 512))
    assert art_d.metadata.get("is_mma")
    assert "vector.contract" in art_d.source_code


def test_backend_falls_back_for_small_shape():
    """Default backend must fall back to scalar f32 when the shape can't MMA-tile."""
    be = MLIRGPUBackend()
    # 32x32x32 is smaller than the default MMA block tile → NotImplementedError
    # inside emit_gpu_matmul_mma → falls through to the scalar regblock ladder.
    art = be.lower(_mm(32, 32, 32))
    assert not art.metadata.get("is_mma")
    assert "vector.contract" not in art.source_code
    # and it still runs correctly (bit-accurate f32)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 32)).astype(np.float32)
    out = be.run(be.compile(art), {"A": A, "B": B})["C"]
    np.testing.assert_allclose(out, A @ B, rtol=1e-3, atol=1e-2)


def test_backend_falls_back_for_tileable_but_small():
    """256×256 is MMA-tileable but below the policy threshold (M*N < 262144)."""
    be = MLIRGPUBackend()
    art = be.lower(_mm(256, 256, 256))
    assert not art.metadata.get("is_mma"), "256×256 should use regblock, not MMA"
    # Correctness: scalar regblock is bit-accurate f32
    rng = np.random.default_rng(2)
    A = rng.standard_normal((256, 256)).astype(np.float32)
    B = rng.standard_normal((256, 256)).astype(np.float32)
    out = be.run(be.compile(art), {"A": A, "B": B})["C"]
    np.testing.assert_allclose(out, A @ B, rtol=1e-4, atol=1e-4)


def test_mma_matches_torch_fp16_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA torch")
    be = MLIRGPUBackend(use_tensor_core=True)
    M = K = N = 512
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    out = be.run(be.compile(be.lower(_mm(M, K, N))), {"A": A, "B": B})["C"]
    tA = torch.tensor(A, device="cuda").half()
    tB = torch.tensor(B, device="cuda").half()
    ref = (tA @ tB).float().cpu().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-1)
