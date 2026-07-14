# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Parameterized CUDA-C matmul templates driven by StrategyIR decisions.

This module implements the "reverse enhancement" from perf-gap-analysis:
instead of one hardcoded matmul kernel, we provide parameterized templates
that the Agent can configure through StrategyIR decisions.

Templates:
  - matmul_scalar_tiled: configurable tile + unroll FP32 kernel
  - matmul_tensor_core: (future) wmma/mma-based kernel

Usage via StrategyIR:
  Decision(kind="tile", params={"BM": 64, "BN": 64, "BK": 16})
  Decision(kind="unroll", params={"loop": "K_tile", "factor": 4})
  Decision(kind="algorithm", params={"name": "tensor_core"})  # future
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_c
from arke.ir.graph import IRGraph


@dataclass
class MatmulConfig:
    """Matmul kernel configuration derived from StrategyIR decisions."""
    BM: int = 32       # tile size in M dimension
    BN: int = 32       # tile size in N dimension
    BK: int = 16       # tile size in K dimension
    UNROLL_K: int = 0  # K-loop unroll factor (0 = no pragma unroll)
    THREADS_M: int = 16  # threads per block in M
    THREADS_N: int = 16  # threads per block in N
    VEC_LOAD: int = 1  # vectorized load width (1 = scalar, 4 = float4)
    algorithm: str = "scalar_tiled"  # "scalar_tiled" or "tensor_core"

    @classmethod
    def from_strategy(cls, decisions: list[Any]) -> "MatmulConfig":
        """Extract matmul config from StrategyIR decisions."""
        cfg = cls()
        for d in decisions:
            if hasattr(d, 'kind') and hasattr(d, 'params'):
                if d.kind == "tile":
                    p = d.params
                    if "BM" in p: cfg.BM = p["BM"]
                    if "BN" in p: cfg.BN = p["BN"]
                    if "BK" in p: cfg.BK = p["BK"]
                elif d.kind == "unroll":
                    if d.params.get("loop") in ("K_tile", "K"):
                        cfg.UNROLL_K = d.params.get("factor", 4)
                elif d.kind == "vectorize":
                    cfg.VEC_LOAD = d.params.get("width", 1)
                elif d.kind == "algorithm":
                    cfg.algorithm = d.params.get("name", "scalar_tiled")
                elif d.kind == "compute":
                    # Derive thread config from warp count
                    warps = d.params.get("warps", 4)
                    total_threads = warps * 32
                    # Square-ish thread block
                    import math
                    side = int(math.sqrt(total_threads))
                    cfg.THREADS_M = side
                    cfg.THREADS_N = total_threads // side
        # Ensure thread block covers tile
        cfg.THREADS_M = min(cfg.THREADS_M, cfg.BM)
        cfg.THREADS_N = min(cfg.THREADS_N, cfg.BN)
        return cfg

    @classmethod
    def default_for_shape(cls, M: int, N: int, K: int) -> "MatmulConfig":
        """Auto-select config based on shape (no StrategyIR).

        Data-driven crossover (RTX 3060, measured 2026-07-12):
          - Small shapes (max dim < 1024): the scalar register-blocked kernel
            WINS — TC's fp16-convert + shmem-staging overhead isn't amortized,
            and at N=256 the scalar kernel hits 0.92× vs TC's 0.28×. Scalar is
            also bit-exact (fp32) vs TC's fp16 accumulation.
          - Large shapes (max dim >= 1024): the WMMA tensor-core kernel WINS
            (N=2048: 1.06× vs scalar 0.50×) — TC throughput dominates once the
            staging cost is amortized over enough work.

        C3 (2026-07-13): MEASURED that an occupancy-maximizing tile search
        (favoring many small BM=BN=16 blocks for wave-fill) was 2-4.5× SLOWER
        than fixed BM=BN=64 at N=256/512 (kernel-only CUDA events):
          N=256: 0.44× of old,  N=512: 0.22× of old.
        Reason: tiny tiles destroy arithmetic intensity — poor register/shmem
        data reuse dominates over any wave-fill benefit. Reverted to the
        measured-optimal fixed BM=BN=64 register-blocked kernel.
        """
        max_dim = max(M, N, K)
        tc_eligible = (M % 16 == 0 and N % 16 == 0 and K % 16 == 0
                       and M >= 64 and N >= 64 and K >= 64)
        # Large + TC-eligible → tensor core.
        if max_dim >= 1024 and tc_eligible:
            return cls(BM=64, BN=64, BK=16, UNROLL_K=0, THREADS_M=16, THREADS_N=16,
                       algorithm="tensor_core")
        # Small/medium → scalar register-blocked (measured-optimal, bit-exact).
        if M >= 64 and N >= 64:
            return cls(BM=64, BN=64, BK=16, UNROLL_K=4, THREADS_M=16, THREADS_N=16,
                       algorithm="scalar_tiled")
        if M >= 128 and N >= 128:
            return cls(BM=32, BN=32, BK=16, UNROLL_K=0, THREADS_M=16, THREADS_N=16)
        return cls(BM=16, BN=16, BK=16, UNROLL_K=0, THREADS_M=16, THREADS_N=16)


def emit_cuda_c_matmul_parameterized(
    graph: IRGraph,
    chip: str = "sm_86",
    config: MatmulConfig | None = None,
) -> CudaCKernel:
    """Emit a parameterized CUDA-C matmul kernel.

    If config is None, auto-selects based on shape.
    Routes to tensor_core template when algorithm="tensor_core".
    """
    node = graph.nodes[0]
    assert node.op == "matmul", f"Expected matmul, got {node.op}"

    input_names = list(node.inputs.values())
    a_name, b_name = input_names[0], input_names[1]
    out_name = node.outputs[0]

    a_val = graph.get_value(a_name)
    b_val = graph.get_value(b_name)
    a_shape = list(a_val.shape) if a_val.shape else [64, 64]
    b_shape = list(b_val.shape) if b_val.shape else [64, 64]

    M, K = a_shape[0], a_shape[1]
    K2, N = b_shape[0], b_shape[1]
    assert K == K2, f"K mismatch: {K} vs {K2}"

    if config is None:
        config = MatmulConfig.default_for_shape(M, N, K)

    dtype = a_val.dtype or "float32"

    if config.algorithm == "tensor_core":
        return _emit_matmul_tensor_core(a_name, b_name, out_name, M, K, N, dtype, config)
    else:
        return _emit_matmul_scalar_tiled(a_name, b_name, out_name, M, K, N, dtype, config)


def _emit_matmul_tensor_core(
    a_name: str, b_name: str, out_name: str,
    M: int, K: int, N: int, dtype: str, config: MatmulConfig,
) -> CudaCKernel:
    """Emit WMMA tensor-core matmul: fp16 inputs, fp32 accumulation."""
    c_type = _ir_dtype_to_c(dtype)
    BM, BN, BK = 64, 64, 16  # Fixed for wmma 16x16x16
    WARPS = 4
    THREADS = WARPS * 32  # 128

    kernel_name = f"arke_matmul_tc_{M}x{N}x{K}"

    source = f"""\
// Auto-generated by Arke CudaCBackend (Phase 4, double-buffered tensor core matmul)
// Shape: [{M},{K}] @ [{K},{N}] -> [{M},{N}], wmma fp16→fp32
// Double-buffered: next K-tile is prefetched while the current tile computes,
// hiding global-load latency. Near-parity with cuBLAS at large shapes (2048: 0.96×).
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define BM {BM}
#define BN {BN}
#define BK {BK}

extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ A,
    const {c_type}* __restrict__ B,
    {c_type}* __restrict__ C,
    int M_dim, int N_dim, int K_dim)
{{
    int bm = blockIdx.y * BM;
    int bn = blockIdx.x * BN;
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id * 16;

    // Double-buffered shared memory (2 pipeline stages).
    __shared__ half sA[2][BM][BK];
    __shared__ half sB[2][BK][BN];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(acc[i], 0.0f);

    int nkt = (K_dim + BK - 1) / BK;

    // Load a K-tile [bk] into shared buffer `buf`.
    #define LOAD_TILE(buf, bk) do {{ \\
        for (int t = threadIdx.x; t < BM * BK; t += blockDim.x) {{ \\
            int r = t / BK, c = t % BK; \\
            int gr = bm + r, gc = (bk) + c; \\
            sA[buf][r][c] = (gr < M_dim && gc < K_dim) \\
                ? __float2half(A[gr * K_dim + gc]) : __float2half(0.0f); \\
        }} \\
        for (int t = threadIdx.x; t < BK * BN; t += blockDim.x) {{ \\
            int r = t / BN, c = t % BN; \\
            int gr = (bk) + r, gc = bn + c; \\
            sB[buf][r][c] = (gr < K_dim && gc < N_dim) \\
                ? __float2half(B[gr * N_dim + gc]) : __float2half(0.0f); \\
        }} \\
    }} while (0)

    LOAD_TILE(0, 0);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    for (int kt = 0; kt < nkt; kt++) {{
        int cur = kt & 1;
        int nxt = (kt + 1) & 1;
        // Prefetch the next K-tile while we compute on the current one.
        if (kt + 1 < nkt) {{
            LOAD_TILE(nxt, (kt + 1) * BK);
        }}
        wmma::load_matrix_sync(a_frag, &sA[cur][warp_row][0], BK);
        for (int col_tile = 0; col_tile < 4; col_tile++) {{
            wmma::load_matrix_sync(b_frag, &sB[cur][0][col_tile * 16], BN);
            wmma::mma_sync(acc[col_tile], a_frag, b_frag, acc[col_tile]);
        }}
        __syncthreads();
    }}

    // Store
    for (int col_tile = 0; col_tile < 4; col_tile++) {{
        int out_row = bm + warp_row;
        int out_col = bn + col_tile * 16;
        if (out_row < M_dim && out_col < N_dim) {{
            wmma::store_matrix_sync(&C[out_row * N_dim + out_col], acc[col_tile],
                                    N_dim, wmma::mem_row_major);
        }}
    }}
}}
"""

    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (THREADS, 1, 1)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="matmul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, K], b_name: [K, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
            ("int", M), ("int", N), ("int", K),
        ],
    )


def _emit_matmul_f4_doublebuf(
    a_name: str, b_name: str, out_name: str,
    M: int, K: int, N: int, dtype: str, config: MatmulConfig,
) -> CudaCKernel:
    """float4-vectorized + double-buffered register-blocked FP32 matmul.

    Two optimizations over the scalar base kernel:
      1. Double-buffered shared memory: the next K-tile is prefetched into a
         second shared buffer while the current tile computes, hiding global
         load latency (same technique as the tensor-core kernel).
      2. float4 vectorized global loads: each thread loads 4 contiguous floats
         per instruction, quadrupling per-instruction global bandwidth.

    Requires float4 alignment: K%4==0, N%4==0, BK%4==0, and the per-tile
    float4 counts must divide the thread count. Caller (_emit_matmul_scalar_tiled)
    guards these. MEASURED 1.19-1.35× faster than the scalar base at
    N=256/512/1024 (RTX 3060, kernel-only CUDA events): ~0.5×→0.66× cuBLAS.
    """
    c_type = _ir_dtype_to_c(dtype)
    BM, BN, BK = config.BM, config.BN, config.BK
    TM, TN = config.THREADS_M, config.THREADS_N
    EPT_M, EPT_N = BM // TM, BN // TN
    kernel_name = f"arke_matmul_f4db_{M}x{N}x{K}_bm{BM}_bn{BN}_bk{BK}"

    source = f"""\
// Auto-generated by Arke CudaCBackend (Phase 4, float4 double-buffered matmul)
// Shape: [{M},{K}] @ [{K},{N}] -> [{M},{N}] | BM={BM} BN={BN} BK={BK} EPT={EPT_M}x{EPT_N}
// float4 vectorized loads + 2-stage double buffering (C3, measured 1.2-1.35x
// over scalar base at N=256/512/1024).
#include <cuda_runtime.h>
#define BM {BM}
#define BN {BN}
#define BK {BK}

extern "C"
__global__ void {kernel_name}(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M_dim, int N_dim, int K_dim)
{{
    __shared__ float sA[2][BK][BM];  // transposed for compute-side coalescing
    __shared__ float sB[2][BK][BN];
    int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    int tid = threadIdx.y * {TN} + threadIdx.x;
    int total = {TM} * {TN};
    float acc[{EPT_M}][{EPT_N}];
    #pragma unroll
    for (int i = 0; i < {EPT_M}; i++)
        for (int j = 0; j < {EPT_N}; j++) acc[i][j] = 0.0f;

    // Load one K-tile into shared buffer `buf` using float4 loads.
    #define LOAD(buf, bk) do {{ \\
        for (int t = tid; t < BM * BK / 4; t += total) {{ \\
            int idx = t * 4, r = idx / BK, c = idx % BK; \\
            int gr = bm + r, gc = (bk) + c; \\
            if (gr < M_dim && gc + 3 < K_dim) {{ \\
                float4 v = *reinterpret_cast<const float4*>(&A[gr * K_dim + gc]); \\
                sA[buf][c][r] = v.x; sA[buf][c+1][r] = v.y; \\
                sA[buf][c+2][r] = v.z; sA[buf][c+3][r] = v.w; \\
            }} else {{ \\
                for (int e = 0; e < 4; e++) \\
                    sA[buf][c+e][r] = (gr < M_dim && gc+e < K_dim) ? A[gr*K_dim+gc+e] : 0.0f; \\
            }} \\
        }} \\
        for (int t = tid; t < BK * BN / 4; t += total) {{ \\
            int idx = t * 4, r = idx / BN, c = idx % BN; \\
            int gr = (bk) + r, gc = bn + c; \\
            if (gr < K_dim && gc + 3 < N_dim) {{ \\
                float4 v = *reinterpret_cast<const float4*>(&B[gr * N_dim + gc]); \\
                sB[buf][r][c] = v.x; sB[buf][r][c+1] = v.y; \\
                sB[buf][r][c+2] = v.z; sB[buf][r][c+3] = v.w; \\
            }} else {{ \\
                for (int e = 0; e < 4; e++) \\
                    sB[buf][r][c+e] = (gr < K_dim && gc+e < N_dim) ? B[gr*N_dim+gc+e] : 0.0f; \\
            }} \\
        }} \\
    }} while (0)

    int nkt = (K_dim + BK - 1) / BK;
    LOAD(0, 0);
    __syncthreads();
    for (int kt = 0; kt < nkt; kt++) {{
        int cur = kt & 1, nxt = (kt + 1) & 1;
        if (kt + 1 < nkt) LOAD(nxt, (kt + 1) * BK);
        #pragma unroll
        for (int k = 0; k < BK; k++) {{
            for (int i = 0; i < {EPT_M}; i++) {{
                float a = sA[cur][k][threadIdx.y * {EPT_M} + i];
                for (int j = 0; j < {EPT_N}; j++)
                    acc[i][j] += a * sB[cur][k][threadIdx.x * {EPT_N} + j];
            }}
        }}
        __syncthreads();
    }}
    for (int i = 0; i < {EPT_M}; i++)
        for (int j = 0; j < {EPT_N}; j++) {{
            int gr = bm + threadIdx.y * {EPT_M} + i;
            int gc = bn + threadIdx.x * {EPT_N} + j;
            if (gr < M_dim && gc < N_dim) C[gr * N_dim + gc] = acc[i][j];
        }}
}}
"""
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (TN, TM, 1)
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="matmul",
        param_names=[a_name, b_name, out_name], output_name=out_name,
        shapes={a_name: [M, K], b_name: [K, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid, block=block,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
            ("int", M), ("int", N), ("int", K),
        ],
    )


def _emit_matmul_scalar_tiled(
    a_name: str, b_name: str, out_name: str,
    M: int, K: int, N: int, dtype: str, config: MatmulConfig,
) -> CudaCKernel:
    """Emit scalar FP32 register-blocked tiled matmul.

    C3 (2026-07-13): routes to a float4-vectorized + double-buffered variant
    when the shape is float4-aligned (K%4==0 and N%4==0 and BK%4==0). That
    variant is MEASURED 1.19-1.35× faster than this base at N=256/512/1024
    (kernel-only CUDA events, RTX 3060): 0.5×→0.66× cuBLAS. Non-aligned shapes
    fall back to this scalar loader (correctness over speed).
    """
    c_type = _ir_dtype_to_c(dtype)

    BM, BN, BK = config.BM, config.BN, config.BK
    TM, TN = config.THREADS_M, config.THREADS_N
    # Elements per thread
    EPT_M = BM // TM  # rows per thread
    EPT_N = BN // TN  # cols per thread

    # Route to the faster float4 + double-buffered kernel when alignment allows
    # AND shape is large enough to amortize the double-buffering overhead.
    # N<256: float4 overhead dominates (measured worse at N=128).
    can_f4 = (dtype in ("float32", "f32") and K % 4 == 0 and N % 4 == 0
              and BK % 4 == 0 and (BM * BK) % (4 * TM * TN) == 0
              and (BK * BN) % (4 * TM * TN) == 0 and min(M, N) >= 256)
    if can_f4:
        return _emit_matmul_f4_doublebuf(
            a_name, b_name, out_name, M, K, N, dtype, config)

    kernel_name = f"arke_matmul_{M}x{N}x{K}_bm{BM}_bn{BN}_bk{BK}"
    unroll_pragma = "#pragma unroll" if config.UNROLL_K > 0 else ""

    source = f"""\
// Auto-generated by Arke CudaCBackend (Phase 4, parameterized matmul)
// Shape: [{M},{K}] @ [{K},{N}] -> [{M},{N}]
// Config: BM={BM} BN={BN} BK={BK} TM={TM} TN={TN} EPT={EPT_M}x{EPT_N}
#include <cuda_runtime.h>

#define BM {BM}
#define BN {BN}
#define BK {BK}
#define TM {TM}
#define TN {TN}

extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ A,
    const {c_type}* __restrict__ B,
    {c_type}* __restrict__ C,
    int M_dim, int N_dim, int K_dim)
{{
    // Block computes a BM x BN output tile
    __shared__ {c_type} sA[BK][BM];  // transposed for coalesced writes
    __shared__ {c_type} sB[BK][BN];

    int bm = blockIdx.y * BM;
    int bn = blockIdx.x * BN;
    int tid = threadIdx.y * TN + threadIdx.x;
    int total_threads = TM * TN;

    // Each thread accumulates a EPT_M x EPT_N register tile
    {c_type} acc[{EPT_M}][{EPT_N}];
    for (int i = 0; i < {EPT_M}; i++)
        for (int j = 0; j < {EPT_N}; j++)
            acc[i][j] = ({c_type})0;

    // Iterate over K dimension in BK-sized tiles
    for (int bk = 0; bk < K_dim; bk += BK) {{
        // Cooperative load: all threads load sA and sB
        for (int t = tid; t < BM * BK; t += total_threads) {{
            int sm_row = t / BK;
            int sm_col = t % BK;
            int gm_row = bm + sm_row;
            int gm_col = bk + sm_col;
            sA[sm_col][sm_row] = (gm_row < M_dim && gm_col < K_dim)
                ? A[gm_row * K_dim + gm_col] : ({c_type})0;
        }}
        for (int t = tid; t < BK * BN; t += total_threads) {{
            int sm_row = t / BN;
            int sm_col = t % BN;
            int gm_row = bk + sm_row;
            int gm_col = bn + sm_col;
            sB[sm_row][sm_col] = (gm_row < K_dim && gm_col < N_dim)
                ? B[gm_row * N_dim + gm_col] : ({c_type})0;
        }}
        __syncthreads();

        // Compute: each thread processes its EPT_M x EPT_N tile
        {unroll_pragma}
        for (int k = 0; k < BK; k++) {{
            for (int i = 0; i < {EPT_M}; i++) {{
                {c_type} a_val = sA[k][threadIdx.y * {EPT_M} + i];
                for (int j = 0; j < {EPT_N}; j++) {{
                    acc[i][j] += a_val * sB[k][threadIdx.x * {EPT_N} + j];
                }}
            }}
        }}
        __syncthreads();
    }}

    // Write results
    for (int i = 0; i < {EPT_M}; i++) {{
        for (int j = 0; j < {EPT_N}; j++) {{
            int gm_row = bm + threadIdx.y * {EPT_M} + i;
            int gm_col = bn + threadIdx.x * {EPT_N} + j;
            if (gm_row < M_dim && gm_col < N_dim) {{
                C[gm_row * N_dim + gm_col] = acc[i][j];
            }}
        }}
    }}
}}
"""

    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (TN, TM, 1)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="matmul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, K], b_name: [K, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
            ("int", M), ("int", N), ("int", K),
        ],
    )
