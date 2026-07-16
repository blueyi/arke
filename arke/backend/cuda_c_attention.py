# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA-C emitters for OT4 attention ops (Phase 4, C-line).

Ops: flash_attention (online-softmax, correctness-first FP32).

Design: one block per (batch*head, query-tile), one thread per query row.
Online softmax accumulation over BC-sized K/V tiles staged in shared memory.
This is the correctness-first base — a TC/tiled high-performance variant is a
follow-up (tracked in docs/phase4/stage-progress.md). Verified exact match
(rel_err=0.0 fp32) vs torch scaled_dot_product_attention.
"""

from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_c
from arke.ir.graph import IRGraph

# Query rows per block / key cols per K-V tile.
# BR=8 warps (256 threads) balances occupancy vs per-block work: 1024 threads
# (BR=32) over-subscribes and drops to 1-2 blocks/SM. BC=32 keeps the K/V
# shared tile small enough for good occupancy on Ampere (48KB smem).
#
# C2 (2026-07-13): MEASURED that adaptive BR=4/BC=64 for large-seq made
# S=512/1024/2048 ~17% SLOWER (kernel-only CUDA-events), so it was reverted.
# BR=8/BC=32 remains the measured-optimal for this warp-per-row kernel.
# The large-seq performance gap (0.15× vs SDPA at S=2048) is inherent to the
# O(S) serial K-scan per query warp — closing it needs a true FlashAttention-2
# rewrite (cross-block K-split + two-pass/atomic reduction), tracked as a
# week-level follow-up. See docs/phase4/audit-2026-07-13.md C2.
_BR_DEFAULT = 8
_BC_DEFAULT = 32


def _select_br_bc(S: int, D: int) -> tuple[int, int]:
    """Select BR/BC. BR=8/BC=32 is measured-optimal across all S for this
    warp-per-row kernel (adaptive large-seq variant was measured slower)."""
    return _BR_DEFAULT, _BC_DEFAULT


# Tensor-Core (wmma m16n16k16 fp16->fp32) fused flash-attention kernel body.
# Verified in scratch/tc_attn (v7 3-stage pipeline + v8 D/causal generalization):
# ~5-6x over the fp32 warp-per-row kernel, ~0.35-0.42x torch SDPA at large S,
# max_err <=4.9e-4 vs torch (incl. is_causal). Tokens substituted at emit time:
#   __KERNEL_NAME__, __HEAD_D__ (64|128), __CAUSAL__ (0|1).
# Fragment layout (sm_86): warp owns rows [warp*16..+16); per-thread element
# row_lo=warp*16+lane/4, row_hi=+8; base_col=nc*16+(lane%4)*2. Elements
# {0,1,4,5}=row_lo, {2,3,6,7}=row_hi. Causal masks each element by (qrow,kcol).
_TC_ATTN_KERNEL = r"""
#include <cstdint>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>
using namespace nvcuda;

#define D        __HEAD_D__
#define CAUSAL   __CAUSAL__
#define BR       64
#define BC       32
#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define NCOL     (BC / WMMA_N)
#define NDCOL    (D  / WMMA_N)
#define NSTAGE   3

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

extern "C" __global__ void __KERNEL_NAME__(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O,
    int B, int H, int S, int Dd, float scale)
{
    const int bh=blockIdx.y, qtile=blockIdx.x, warp=threadIdx.x>>5, lane=threadIdx.x&31, tid=threadIdx.x;
    const int q0=qtile*BR;
    const half* Qbh=Q+(long)bh*S*D; const half* Kbh=K+(long)bh*S*D;
    const half* Vbh=V+(long)bh*S*D; half* Obh=O+(long)bh*S*D;

    extern __shared__ char smem[];
    half* Qsh_=(half*)smem;
    half* KV_base=Qsh_+BR*D;
    half* KTsh[NSTAGE]; half* Vsh[NSTAGE];
    #pragma unroll
    for (int s=0;s<NSTAGE;s++){ KTsh[s]=KV_base+s*(2*BC*D); Vsh[s]=KTsh[s]+BC*D; }
    #define Qsh(r,c) Qsh_[(r)*D+(c)]

    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> o_frag[NDCOL];
    #pragma unroll
    for (int nc=0;nc<NDCOL;nc++) wmma::fill_fragment(o_frag[nc],0.0f);
    float m_lo=-FLT_MAX,m_hi=-FLT_MAX,l_lo=0.0f,l_hi=0.0f;
    const int ntiles=(S+BC-1)/BC;
    const int qrow_lo=q0+warp*WMMA_M+lane/4;
    const int qrow_hi=qrow_lo+8;

    for (int idx=tid; idx<BR*D; idx+=blockDim.x){
        int r=idx/D,c=idx%D,qr=q0+r;
        Qsh(r,c)=(qr<S)?Qbh[qr*D+c]:__float2half(0.0f);
    }

    #define PREFETCH_TILE(tt,slot) do {                                     \
        int _kt=(tt)*BC; int _vr=min(BC,S-_kt); int _ve=_vr*D;              \
        const half* _ks=Kbh+(long)_kt*D; const half* _vs=Vbh+(long)_kt*D;  \
        int _pt=(BC*D)/blockDim.x; int _b=tid*_pt;                         \
        for(int i=0;i<_pt;i+=8){int p=_b+i;                                \
            if(p<_ve)cp_async_cg(&KTsh[slot][p],&_ks[p]);                  \
            else *((uint4*)&KTsh[slot][p])=make_uint4(0,0,0,0);}           \
        for(int i=0;i<_pt;i+=8){int p=_b+i;                                \
            if(p<_ve)cp_async_cg(&Vsh[slot][p],&_vs[p]);                   \
            else *((uint4*)&Vsh[slot][p])=make_uint4(0,0,0,0);}            \
        cp_async_commit();                                                 \
    } while(0)

    #pragma unroll
    for (int s=0;s<NSTAGE-1;s++){ if(s<ntiles) PREFETCH_TILE(s,s); }

    for (int t=0;t<ntiles;t++){
        int cur=t%NSTAGE, kt=t*BC, krem=min(S-kt,BC);
        int pf=t+(NSTAGE-1);
        if(pf<ntiles) PREFETCH_TILE(pf,pf%NSTAGE);
        cp_async_wait_group<NSTAGE-1>();
        __syncthreads();

        wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> s_frag[NCOL];
        #pragma unroll
        for (int nc=0;nc<NCOL;nc++){
            wmma::fill_fragment(s_frag[nc],0.0f);
            #pragma unroll
            for (int k=0;k<D;k+=WMMA_K){
                wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag,&Qsh(warp*WMMA_M,k),D);
                wmma::load_matrix_sync(b_frag,&KTsh[cur][nc*WMMA_N*D+k],D);
                wmma::mma_sync(s_frag[nc],a_frag,b_frag,s_frag[nc]);
            }
        }

        wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> p_frag[NCOL];
        {
            float pmax_lo=-FLT_MAX,pmax_hi=-FLT_MAX;
            #pragma unroll
            for (int nc=0;nc<NCOL;nc++){
                int base_col=nc*16+(lane%4)*2;
                int kc0=kt+base_col,kc1=kt+base_col+1,kc8=kt+base_col+8,kc9=kt+base_col+9;
#if CAUSAL
                bool m0=(base_col<krem)&&(kc0<=qrow_lo), m1=(base_col+1<krem)&&(kc1<=qrow_lo);
                bool m4=(base_col+8<krem)&&(kc8<=qrow_lo), m5=(base_col+9<krem)&&(kc9<=qrow_lo);
                bool n2=(base_col<krem)&&(kc0<=qrow_hi), n3=(base_col+1<krem)&&(kc1<=qrow_hi);
                bool n6=(base_col+8<krem)&&(kc8<=qrow_hi), n7=(base_col+9<krem)&&(kc9<=qrow_hi);
#else
                bool m0=(base_col<krem),m1=(base_col+1<krem),m4=(base_col+8<krem),m5=(base_col+9<krem);
                bool n2=m0,n3=m1,n6=m4,n7=m5;
#endif
                float v0=m0?s_frag[nc].x[0]*scale:-FLT_MAX, v1=m1?s_frag[nc].x[1]*scale:-FLT_MAX;
                float v4=m4?s_frag[nc].x[4]*scale:-FLT_MAX, v5=m5?s_frag[nc].x[5]*scale:-FLT_MAX;
                pmax_lo=fmaxf(pmax_lo,fmaxf(fmaxf(v0,v1),fmaxf(v4,v5)));
                float v2=n2?s_frag[nc].x[2]*scale:-FLT_MAX, v3=n3?s_frag[nc].x[3]*scale:-FLT_MAX;
                float v6=n6?s_frag[nc].x[6]*scale:-FLT_MAX, v7=n7?s_frag[nc].x[7]*scale:-FLT_MAX;
                pmax_hi=fmaxf(pmax_hi,fmaxf(fmaxf(v2,v3),fmaxf(v6,v7)));
            }
            #pragma unroll
            for (int d=1;d<=2;d<<=1){
                pmax_lo=fmaxf(pmax_lo,__shfl_xor_sync(0xffffffff,pmax_lo,d));
                pmax_hi=fmaxf(pmax_hi,__shfl_xor_sync(0xffffffff,pmax_hi,d));
            }
            float mp_lo=m_lo,mp_hi=m_hi;
            m_lo=fmaxf(m_lo,pmax_lo); m_hi=fmaxf(m_hi,pmax_hi);
            float corr_lo=(mp_lo==-FLT_MAX)?0.0f:expf(mp_lo-m_lo);
            float corr_hi=(mp_hi==-FLT_MAX)?0.0f:expf(mp_hi-m_hi);
            #pragma unroll
            for (int nc=0;nc<NDCOL;nc++){
                o_frag[nc].x[0]*=corr_lo;o_frag[nc].x[1]*=corr_lo;o_frag[nc].x[4]*=corr_lo;o_frag[nc].x[5]*=corr_lo;
                o_frag[nc].x[2]*=corr_hi;o_frag[nc].x[3]*=corr_hi;o_frag[nc].x[6]*=corr_hi;o_frag[nc].x[7]*=corr_hi;
            }
            float psum_lo=0.0f,psum_hi=0.0f;
            #pragma unroll
            for (int nc=0;nc<NCOL;nc++){
                int base_col=nc*16+(lane%4)*2;
                int kc0=kt+base_col,kc1=kt+base_col+1,kc8=kt+base_col+8,kc9=kt+base_col+9;
#if CAUSAL
                bool m0=(base_col<krem)&&(kc0<=qrow_lo), m1=(base_col+1<krem)&&(kc1<=qrow_lo);
                bool m4=(base_col+8<krem)&&(kc8<=qrow_lo), m5=(base_col+9<krem)&&(kc9<=qrow_lo);
                bool n2=(base_col<krem)&&(kc0<=qrow_hi), n3=(base_col+1<krem)&&(kc1<=qrow_hi);
                bool n6=(base_col+8<krem)&&(kc8<=qrow_hi), n7=(base_col+9<krem)&&(kc9<=qrow_hi);
#else
                bool m0=(base_col<krem),m1=(base_col+1<krem),m4=(base_col+8<krem),m5=(base_col+9<krem);
                bool n2=m0,n3=m1,n6=m4,n7=m5;
#endif
                float p0=m0?expf(s_frag[nc].x[0]*scale-m_lo):0.0f, p1=m1?expf(s_frag[nc].x[1]*scale-m_lo):0.0f;
                float p4=m4?expf(s_frag[nc].x[4]*scale-m_lo):0.0f, p5=m5?expf(s_frag[nc].x[5]*scale-m_lo):0.0f;
                psum_lo+=p0+p1+p4+p5;
                float p2=n2?expf(s_frag[nc].x[2]*scale-m_hi):0.0f, p3=n3?expf(s_frag[nc].x[3]*scale-m_hi):0.0f;
                float p6=n6?expf(s_frag[nc].x[6]*scale-m_hi):0.0f, p7=n7?expf(s_frag[nc].x[7]*scale-m_hi):0.0f;
                psum_hi+=p2+p3+p6+p7;
                p_frag[nc].x[0]=__float2half(p0);p_frag[nc].x[1]=__float2half(p1);
                p_frag[nc].x[2]=__float2half(p2);p_frag[nc].x[3]=__float2half(p3);
                p_frag[nc].x[4]=__float2half(p4);p_frag[nc].x[5]=__float2half(p5);
                p_frag[nc].x[6]=__float2half(p6);p_frag[nc].x[7]=__float2half(p7);
                p_frag[nc].x[8]=__float2half(p0);p_frag[nc].x[9]=__float2half(p1);
                p_frag[nc].x[10]=__float2half(p2);p_frag[nc].x[11]=__float2half(p3);
                p_frag[nc].x[12]=__float2half(p4);p_frag[nc].x[13]=__float2half(p5);
                p_frag[nc].x[14]=__float2half(p6);p_frag[nc].x[15]=__float2half(p7);
            }
            #pragma unroll
            for (int d=1;d<=2;d<<=1){
                psum_lo+=__shfl_xor_sync(0xffffffff,psum_lo,d);
                psum_hi+=__shfl_xor_sync(0xffffffff,psum_hi,d);
            }
            l_lo=l_lo*corr_lo+psum_lo; l_hi=l_hi*corr_hi+psum_hi;
        }

        #pragma unroll
        for (int nc=0;nc<NDCOL;nc++){
            #pragma unroll
            for (int kk=0;kk<NCOL;kk++){
                wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag,&Vsh[cur][kk*WMMA_K*D+nc*WMMA_N],D);
                wmma::mma_sync(o_frag[nc],p_frag[kk],b_frag,o_frag[nc]);
            }
        }
        __syncthreads();
    }

    {
        float inv_lo=(l_lo>0.0f)?1.0f/l_lo:0.0f, inv_hi=(l_hi>0.0f)?1.0f/l_hi:0.0f;
        #pragma unroll
        for (int nc=0;nc<NDCOL;nc++){
            o_frag[nc].x[0]*=inv_lo;o_frag[nc].x[1]*=inv_lo;o_frag[nc].x[4]*=inv_lo;o_frag[nc].x[5]*=inv_lo;
            o_frag[nc].x[2]*=inv_hi;o_frag[nc].x[3]*=inv_hi;o_frag[nc].x[6]*=inv_hi;o_frag[nc].x[7]*=inv_hi;
        }
        int row_lo=warp*WMMA_M+lane/4, row_hi=row_lo+8, grow_lo=q0+row_lo, grow_hi=q0+row_hi;
        #pragma unroll
        for (int nc=0;nc<NDCOL;nc++){
            int cb=nc*WMMA_N+(lane%4)*2;
            if(grow_lo<S){
                Obh[grow_lo*D+cb]=__float2half(o_frag[nc].x[0]);Obh[grow_lo*D+cb+1]=__float2half(o_frag[nc].x[1]);
                Obh[grow_lo*D+cb+8]=__float2half(o_frag[nc].x[4]);Obh[grow_lo*D+cb+9]=__float2half(o_frag[nc].x[5]);
            }
            if(grow_hi<S){
                Obh[grow_hi*D+cb]=__float2half(o_frag[nc].x[2]);Obh[grow_hi*D+cb+1]=__float2half(o_frag[nc].x[3]);
                Obh[grow_hi*D+cb+8]=__float2half(o_frag[nc].x[6]);Obh[grow_hi*D+cb+9]=__float2half(o_frag[nc].x[7]);
            }
        }
    }
}
"""


def emit_cuda_c_flash_attention_tc(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit the Tensor-Core (wmma fp16->fp32) fused flash-attention kernel.

    Requires dtype float16 and head dim D in {64,128}. BR=64/BC=32, 4 warps,
    register-resident O+P, 3-stage cp.async pipeline. Supports optional causal
    mask via node.attrs["causal"]. ~5-6x faster than the fp32 warp-per-row
    kernel at large S. Provenance: docs/phase5/c2-tensorcore-attention-*.md,
    scratch/tc_attn (v7/v8, verified max_err <=4.9e-4 vs torch SDPA).
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    q_name, k_name, v_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    B, H, S, D = list(q_val.shape)
    dtype = q_val.dtype or "float16"
    if dtype != "float16":
        raise ValueError(f"TC flash_attention requires float16, got {dtype}")
    if D not in (64, 128):
        raise ValueError(f"TC flash_attention requires D in {{64,128}}, got {D}")

    causal = bool(node.attrs.get("causal", False))

    import math
    scale = 1.0 / math.sqrt(D)

    BR, BC, NSTAGE = 64, 32, 3
    THREADS = 128
    kernel_name = f"arke_tc_flash_attn_{B}x{H}x{S}x{D}{'_causal' if causal else ''}"

    # Dynamic shared memory: Qsh(BR*D*2) + NSTAGE*(K BC*D*2 + V BC*D*2).
    shared_mem = (BR * D * 2) + NSTAGE * (2 * BC * D * 2)

    source = (
        _TC_ATTN_KERNEL
        .replace("__KERNEL_NAME__", kernel_name)
        .replace("__HEAD_D__", str(D))
        .replace("__CAUSAL__", "1" if causal else "0")
    )

    grid = ((S + BR - 1) // BR, B * H, 1)
    block = (THREADS, 1, 1)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="flash_attention",
        param_names=[q_name, k_name, v_name, out_name],
        output_name=out_name,
        shapes={q_name: [B, H, S, D], k_name: [B, H, S, D],
                v_name: [B, H, S, D], out_name: [B, H, S, D]},
        dtypes={q_name: "float16", k_name: "float16",
                v_name: "float16", out_name: "float16"},
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        kernel_args=[
            ("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
            ("int", B), ("int", H), ("int", S), ("int", D), ("float", scale),
        ],
    )


def emit_cuda_c_flash_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit online-softmax flash-attention: O = softmax(Q@K^T/sqrt(D)) @ V.

    Q,K,V,O are [B,H,S,D] row-major. Head dim D is baked into the kernel as a
    compile-time bound for the per-thread register arrays and shared tiles.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    q_name, k_name, v_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    shape = list(q_val.shape)
    if len(shape) != 4:
        raise NotImplementedError(
            f"flash_attention: expected [B,H,S,D], got {shape}")
    B, H, S, D = shape
    dtype = q_val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)

    # High-performance Tensor-Core path: fp16 + head dim in {64,128}. The
    # wmma fp16->fp32 fused kernel (T1, docs/phase5/c2-tensorcore-*.md) is
    # ~5-6x faster than this warp-per-row fp32 kernel at large S. Anything
    # else (fp32, other D) uses the correctness-first path below.
    if dtype == "float16" and D in (64, 128):
        return emit_cuda_c_flash_attention_tc(graph, chip=chip)

    import math
    scale = 1.0 / math.sqrt(D)

    BR, BC = _select_br_bc(S, D)

    kernel_name = f"arke_flash_attn_{B}x{H}x{S}x{D}"

    # Warp-per-query-row parallelization: each of WARPS warps in a block owns
    # one query row; its 32 lanes split the D dimension. This gives 32× the
    # parallelism of the naive 1-thread-per-row kernel on both the Q·K dot
    # product (warp-shuffle reduction) and the P·V accumulation.
    WARPS = BR  # BR query rows per block → BR warps
    THREADS = WARPS * 32
    # Per-lane register slice of the head dim: ceil(D/32).
    DPL = (D + 31) // 32

    source = f"""\
// Auto-generated by Arke CudaCBackend — flash-attention (online softmax, warp-per-row)
// Q,K,V,O: [{B},{H},{S},{D}] row-major. One block per (bh, query-tile).
// Each warp owns one query row; 32 lanes split the head dim.
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BR {BR}
#define BC {BC}
#define HEAD_DIM {D}
#define DPL {DPL}

extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ Q,
    const {c_type}* __restrict__ K,
    const {c_type}* __restrict__ V,
    {c_type}* __restrict__ O,
    int B, int H, int S, int D, float scale)
{{
    int bh = blockIdx.y;
    int qtile = blockIdx.x;
    int warp = threadIdx.x >> 5;      // which query row within the tile
    int lane = threadIdx.x & 31;      // lane within the warp

    int q_row = qtile * BR + warp;
    int valid = q_row < S;

    const {c_type}* Qbh = Q + (long)bh * S * D;
    const {c_type}* Kbh = K + (long)bh * S * D;
    const {c_type}* Vbh = V + (long)bh * S * D;
    {c_type}* Obh = O + (long)bh * S * D;

    // Each lane holds a strided slice of the head dim: d = lane + 32*t.
    {c_type} q_reg[DPL];
    {c_type} acc[DPL];
    #pragma unroll
    for (int t = 0; t < DPL; t++) {{
        int d = lane + 32 * t;
        q_reg[t] = (valid && d < D) ? Qbh[q_row * D + d] : 0.0f;
        acc[t] = 0.0f;
    }}
    {c_type} m_i = -FLT_MAX;
    {c_type} l_i = 0.0f;

    __shared__ {c_type} sK[BC][HEAD_DIM];
    __shared__ {c_type} sV[BC][HEAD_DIM];

    for (int kt = 0; kt < S; kt += BC) {{
        // Cooperative load of the K/V tile by the whole block.
        for (int idx = threadIdx.x; idx < BC * D; idx += {THREADS}) {{
            int r = idx / D, c = idx % D;
            int k_row = kt + r;
            if (k_row < S) {{
                sK[r][c] = Kbh[k_row * D + c];
                sV[r][c] = Vbh[k_row * D + c];
            }} else {{
                sK[r][c] = 0.0f;
                sV[r][c] = 0.0f;
            }}
        }}
        __syncthreads();

        for (int j = 0; j < BC; j++) {{
            int k_row = kt + j;
            if (k_row >= S) break;
            // Partial dot product over this lane's slice, then warp-reduce.
            {c_type} s_partial = 0.0f;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{
                int d = lane + 32 * t;
                if (d < D) s_partial += q_reg[t] * sK[j][d];
            }}
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s_partial += __shfl_down_sync(0xffffffff, s_partial, off);
            // Broadcast the full score to all lanes.
            {c_type} s = __shfl_sync(0xffffffff, s_partial, 0) * scale;
            {c_type} m_new = fmaxf(m_i, s);
            {c_type} corr = expf(m_i - m_new);
            {c_type} p = expf(s - m_new);
            l_i = l_i * corr + p;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{
                int d = lane + 32 * t;
                if (d < D) acc[t] = acc[t] * corr + p * sV[j][d];
            }}
            m_i = m_new;
        }}
        __syncthreads();
    }}

    if (valid) {{
        {c_type} inv_l = 1.0f / l_i;
        #pragma unroll
        for (int t = 0; t < DPL; t++) {{
            int d = lane + 32 * t;
            if (d < D) Obh[q_row * D + d] = acc[t] * inv_l;
        }}
    }}
}}
"""

    grid = ((S + BR - 1) // BR, B * H, 1)
    block = (WARPS * 32, 1, 1)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="flash_attention",
        param_names=[q_name, k_name, v_name, out_name],
        output_name=out_name,
        shapes={q_name: [B, H, S, D], k_name: [B, H, S, D],
                v_name: [B, H, S, D], out_name: [B, H, S, D]},
        dtypes={q_name: dtype, k_name: dtype, v_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        kernel_args=[
            ("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
            ("int", B), ("int", H), ("int", S), ("int", D), ("float", scale),
        ],
    )


def emit_cuda_c_gqa(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Grouped-query attention: H_q query heads share H_kv key/value heads.

    Q [B,H_q,S,D], K/V [B,H_kv,S,D] -> O [B,H_q,S,D]. Query head hq maps to
    kv head hq // (H_q // H_kv). Same online-softmax kernel as flash_attention,
    with distinct Q-head and KV-head base pointers.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    q_name, k_name, v_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]

    q_shape = list(graph.get_value(q_name).shape)
    k_shape = list(graph.get_value(k_name).shape)
    B, Hq, S, D = q_shape
    Hkv = k_shape[1]
    dtype = graph.get_value(q_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)

    import math
    scale = 1.0 / math.sqrt(D)
    group = Hq // Hkv

    BR, BC = _BR_DEFAULT, _BC_DEFAULT
    WARPS = BR
    THREADS = WARPS * 32
    DPL = (D + 31) // 32
    kernel_name = f"arke_gqa_{B}x{Hq}x{Hkv}x{S}x{D}"

    source = f"""\
// Auto-generated by Arke CudaCBackend — grouped-query attention (online softmax)
// Q:[{B},{Hq},{S},{D}] K/V:[{B},{Hkv},{S},{D}] -> O:[{B},{Hq},{S},{D}]
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define BR {BR}
#define BC {BC}
#define HEAD_DIM {D}
#define DPL {DPL}
#define GROUP {group}
#define HKV {Hkv}

extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ Q,
    const {c_type}* __restrict__ K,
    const {c_type}* __restrict__ V,
    {c_type}* __restrict__ O,
    int B, int Hq, int S, int D, float scale)
{{
    int bh = blockIdx.y;              // batch*query-head index
    int b = bh / Hq;
    int hq = bh % Hq;
    int hkv = hq / GROUP;            // shared kv head
    int qtile = blockIdx.x;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int q_row = qtile * BR + warp;
    int valid = q_row < S;

    const {c_type}* Qbh = Q + ((long)(b * Hq + hq)) * S * D;
    const {c_type}* Kbh = K + ((long)(b * HKV + hkv)) * S * D;
    const {c_type}* Vbh = V + ((long)(b * HKV + hkv)) * S * D;
    {c_type}* Obh = O + ((long)(b * Hq + hq)) * S * D;

    {c_type} q_reg[DPL];
    {c_type} acc[DPL];
    #pragma unroll
    for (int t = 0; t < DPL; t++) {{
        int d = lane + 32 * t;
        q_reg[t] = (valid && d < D) ? Qbh[q_row * D + d] : 0.0f;
        acc[t] = 0.0f;
    }}
    {c_type} m_i = -FLT_MAX;
    {c_type} l_i = 0.0f;

    __shared__ {c_type} sK[BC][HEAD_DIM];
    __shared__ {c_type} sV[BC][HEAD_DIM];

    for (int kt = 0; kt < S; kt += BC) {{
        for (int idx = threadIdx.x; idx < BC * D; idx += {THREADS}) {{
            int r = idx / D, c = idx % D;
            int k_row = kt + r;
            if (k_row < S) {{ sK[r][c] = Kbh[k_row * D + c]; sV[r][c] = Vbh[k_row * D + c]; }}
            else {{ sK[r][c] = 0.0f; sV[r][c] = 0.0f; }}
        }}
        __syncthreads();
        for (int j = 0; j < BC; j++) {{
            int k_row = kt + j;
            if (k_row >= S) break;
            {c_type} s_partial = 0.0f;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) s_partial += q_reg[t] * sK[j][d]; }}
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) s_partial += __shfl_down_sync(0xffffffff, s_partial, off);
            {c_type} s = __shfl_sync(0xffffffff, s_partial, 0) * scale;
            {c_type} m_new = fmaxf(m_i, s);
            {c_type} corr = expf(m_i - m_new);
            {c_type} p = expf(s - m_new);
            l_i = l_i * corr + p;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) acc[t] = acc[t] * corr + p * sV[j][d]; }}
            m_i = m_new;
        }}
        __syncthreads();
    }}
    if (valid) {{
        {c_type} inv_l = 1.0f / l_i;
        #pragma unroll
        for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) Obh[q_row * D + d] = acc[t] * inv_l; }}
    }}
}}
"""
    grid = ((S + BR - 1) // BR, B * Hq, 1)
    block = (THREADS, 1, 1)
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="grouped_query_attention",
        param_names=[q_name, k_name, v_name, out_name], output_name=out_name,
        shapes={q_name: [B, Hq, S, D], k_name: [B, Hkv, S, D],
                v_name: [B, Hkv, S, D], out_name: [B, Hq, S, D]},
        dtypes={q_name: dtype, k_name: dtype, v_name: dtype, out_name: dtype},
        grid=grid, block=block,
        kernel_args=[("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
                     ("int", B), ("int", Hq), ("int", S), ("int", D), ("float", scale)],
    )


def emit_cuda_c_cross_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Cross attention: query length Sq differs from key/value length Skv.

    Q [B,H,Sq,D], K/V [B,H,Skv,D] -> O [B,H,Sq,D]. Same online-softmax kernel,
    with the query loop bounded by Sq and the K/V loop by Skv.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    q_name, k_name, v_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]

    q_shape = list(graph.get_value(q_name).shape)
    k_shape = list(graph.get_value(k_name).shape)
    B, H, Sq, D = q_shape
    Skv = k_shape[2]
    dtype = graph.get_value(q_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)

    import math
    scale = 1.0 / math.sqrt(D)

    BR, BC = _BR_DEFAULT, _BC_DEFAULT
    WARPS = BR
    THREADS = WARPS * 32
    DPL = (D + 31) // 32
    kernel_name = f"arke_cross_attn_{B}x{H}x{Sq}x{Skv}x{D}"

    source = f"""\
// Auto-generated by Arke CudaCBackend — cross attention (online softmax)
// Q:[{B},{H},{Sq},{D}] K/V:[{B},{H},{Skv},{D}] -> O:[{B},{H},{Sq},{D}]
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define BR {BR}
#define BC {BC}
#define HEAD_DIM {D}
#define DPL {DPL}

extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ Q,
    const {c_type}* __restrict__ K,
    const {c_type}* __restrict__ V,
    {c_type}* __restrict__ O,
    int B, int H, int Sq, int Skv, int D, float scale)
{{
    int bh = blockIdx.y;
    int qtile = blockIdx.x;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int q_row = qtile * BR + warp;
    int valid = q_row < Sq;

    const {c_type}* Qbh = Q + (long)bh * Sq * D;
    const {c_type}* Kbh = K + (long)bh * Skv * D;
    const {c_type}* Vbh = V + (long)bh * Skv * D;
    {c_type}* Obh = O + (long)bh * Sq * D;

    {c_type} q_reg[DPL];
    {c_type} acc[DPL];
    #pragma unroll
    for (int t = 0; t < DPL; t++) {{
        int d = lane + 32 * t;
        q_reg[t] = (valid && d < D) ? Qbh[q_row * D + d] : 0.0f;
        acc[t] = 0.0f;
    }}
    {c_type} m_i = -FLT_MAX;
    {c_type} l_i = 0.0f;

    __shared__ {c_type} sK[BC][HEAD_DIM];
    __shared__ {c_type} sV[BC][HEAD_DIM];

    for (int kt = 0; kt < Skv; kt += BC) {{
        for (int idx = threadIdx.x; idx < BC * D; idx += {THREADS}) {{
            int r = idx / D, c = idx % D;
            int k_row = kt + r;
            if (k_row < Skv) {{ sK[r][c] = Kbh[k_row * D + c]; sV[r][c] = Vbh[k_row * D + c]; }}
            else {{ sK[r][c] = 0.0f; sV[r][c] = 0.0f; }}
        }}
        __syncthreads();
        for (int j = 0; j < BC; j++) {{
            int k_row = kt + j;
            if (k_row >= Skv) break;
            {c_type} s_partial = 0.0f;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) s_partial += q_reg[t] * sK[j][d]; }}
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) s_partial += __shfl_down_sync(0xffffffff, s_partial, off);
            {c_type} s = __shfl_sync(0xffffffff, s_partial, 0) * scale;
            {c_type} m_new = fmaxf(m_i, s);
            {c_type} corr = expf(m_i - m_new);
            {c_type} p = expf(s - m_new);
            l_i = l_i * corr + p;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) acc[t] = acc[t] * corr + p * sV[j][d]; }}
            m_i = m_new;
        }}
        __syncthreads();
    }}
    if (valid) {{
        {c_type} inv_l = 1.0f / l_i;
        #pragma unroll
        for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) Obh[q_row * D + d] = acc[t] * inv_l; }}
    }}
}}
"""
    grid = ((Sq + BR - 1) // BR, B * H, 1)
    block = (THREADS, 1, 1)
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="cross_attention",
        param_names=[q_name, k_name, v_name, out_name], output_name=out_name,
        shapes={q_name: [B, H, Sq, D], k_name: [B, H, Skv, D],
                v_name: [B, H, Skv, D], out_name: [B, H, Sq, D]},
        dtypes={q_name: dtype, k_name: dtype, v_name: dtype, out_name: dtype},
        grid=grid, block=block,
        kernel_args=[("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
                     ("int", B), ("int", H), ("int", Sq), ("int", Skv), ("int", D), ("float", scale)],
    )
