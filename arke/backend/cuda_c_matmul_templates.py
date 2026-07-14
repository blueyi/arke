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

        C3 (2026-07-13): Added occupancy-aware tile search for small shapes.
        Instead of fixed BM=BN=64 for all small shapes, we pick tiles that
        maximize occupancy (fill SM waves) on the target GPU:
          - RTX 3060 has 30 SMs × 1024 max_threads = 30720 max concurrent threads
          - Blocks = ceil(M/BM) × ceil(N/BN); each block uses THREADS_M × THREADS_N threads
          - We want blocks ≥ SM_count to avoid idle SMs (wave-quantization)
        """
        max_dim = max(M, N, K)
        tc_eligible = (M % 16 == 0 and N % 16 == 0 and K % 16 == 0
                       and M >= 64 and N >= 64 and K >= 64)
        # Large + TC-eligible → tensor core.
        if max_dim >= 1024 and tc_eligible:
            return cls(BM=64, BN=64, BK=16, UNROLL_K=0, THREADS_M=16, THREADS_N=16,
                       algorithm="tensor_core")
        # Small/medium → occupancy-aware scalar tile search.
        return cls._occupancy_search(M, N, K)

    @classmethod
    def _occupancy_search(cls, M: int, N: int, K: int,
                          sm_count: int = 30,
                          max_threads_per_block: int = 1024) -> "MatmulConfig":
        """Pick scalar tile config that maximizes SM occupancy for small shapes.

        Searches over candidate (BM, BN) tile sizes, scoring each by:
        1. Blocks launched = ceil(M/BM) × ceil(N/BN) — more blocks = better wave fill
        2. Block size = min(BM,TM) × min(BN,TN) threads — must not exceed limit
        3. Work per thread = BM×BN×K / threads — prefer larger (more ILP)
        4. Penalty if blocks < sm_count (under-occupancy)

        Picks the config with the best score.
        """
        import math
        candidates = [16, 32, 64, 128]
        best_score = -1.0
        best_cfg = cls(BM=16, BN=16, BK=16, UNROLL_K=0, THREADS_M=16, THREADS_N=16)

        for bm in candidates:
            if bm > max(M, 16):
                continue
            for bn in candidates:
                if bn > max(N, 16):
                    continue
                tm = min(bm, 16)
                tn = min(bn, 16)
                block_threads = tm * tn
                if block_threads > max_threads_per_block:
                    continue
                blocks_m = math.ceil(M / bm)
                blocks_n = math.ceil(N / bn)
                total_blocks = blocks_m * blocks_n
                # Score: maximize blocks (wave fill), penalize under-occupancy
                waves = math.ceil(total_blocks / sm_count)
                wave_fill = total_blocks / (waves * sm_count)  # 0..1
                work_per_thread = (bm * bn * K) / block_threads
                # Normalize work to [0,1] range (16*16*K/256 to 128*128*K/256)
                work_norm = min(work_per_thread / (128 * 128 * K / 256), 1.0)
                # Combined score: occupancy dominates, work is secondary
                score = wave_fill * 0.7 + work_norm * 0.3
                if total_blocks < sm_count:
                    score *= 0.5  # heavy penalty for under-occupancy
                if score > best_score:
                    best_score = score
                    unroll = 4 if K >= 64 else 0
                    best_cfg = cls(BM=bm, BN=bn, BK=16, UNROLL_K=unroll,
                                   THREADS_M=tm, THREADS_N=tn,
                                   algorithm="scalar_tiled")
        return best_cfg


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


def _emit_matmul_scalar_tiled(
    a_name: str, b_name: str, out_name: str,
    M: int, K: int, N: int, dtype: str, config: MatmulConfig,
) -> CudaCKernel:
    """Emit scalar FP32 register-blocked tiled matmul."""
    c_type = _ir_dtype_to_c(dtype)

    BM, BN, BK = config.BM, config.BN, config.BK
    TM, TN = config.THREADS_M, config.THREADS_N
    # Elements per thread
    EPT_M = BM // TM  # rows per thread
    EPT_N = BN // TN  # cols per thread

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
