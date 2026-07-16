#include <cstdint>
// TC flash-attention v8 — v7 generalized to D in {64,128} + optional causal mask.
//
// Config via preprocessor:
//   HEAD_D : head dim, 64 or 128 (default 64)
//   CAUSAL : 0 (full attention) or 1 (causal mask) (default 0)
//
// Derived from v7 (BR=64, BC=32, 3-stage cp.async pipeline, register-resident
// O + P). Structure is identical; only NDCOL (=HEAD_D/16) and the per-element
// masking change with the config.
//
// Smem = Qsh(BR*D*2) + NSTAGE*(K BC*D*2 + V BC*D*2):
//   D=64  -> 8K  + 3*(4K+4K) = 32 KB  (3 blocks/SM)
//   D=128 -> 16K + 3*(8K+8K) = 64 KB  (1 block/SM; needs >48K dynamic smem opt-in)
//
// Fragment layout (sm_86, m16n16k16), warp owns rows [warp*16 .. +16):
//   row_lo = warp*16 + lane/4 ; row_hi = row_lo + 8
//   base_col = nc*16 + (lane%4)*2
//   .x[0]->(row_lo,base_col+0) .x[1]->(+1) .x[4]->(+8) .x[5]->(+9)
//   .x[2]->(row_hi,base_col+0) .x[3]->(+1) .x[6]->(+8) .x[7]->(+9)
// -> elements {0,1,4,5} are row_lo, {2,3,6,7} are row_hi. Causal masks each of
// the 8 elements independently by comparing its global (qrow,kcol).

#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#ifndef HEAD_D
#define HEAD_D 64
#endif
#ifndef CAUSAL
#define CAUSAL 0
#endif

#define D        HEAD_D
#define BR       64
#define BC       32
#define NWARPS   4
#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define NCOL     (BC / WMMA_N)   // 2
#define NDCOL    (D  / WMMA_N)   // 4 (D=64) or 8 (D=128)
#define NSTAGE   3

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template<int N> __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

extern "C" __global__ void KERNEL_NAME(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int S, float scale)
{
    const int bh    = blockIdx.y;
    const int qtile = blockIdx.x;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int tid   = threadIdx.x;
    const int q0    = qtile * BR;

    const half* Qbh = Q + (long)bh * S * D;
    const half* Kbh = K + (long)bh * S * D;
    const half* Vbh = V + (long)bh * S * D;
    half*       Obh = O + (long)bh * S * D;

    extern __shared__ char smem[];
    half* Qsh_    = (half*) smem;                        // BR*D*2
    half* KV_base = Qsh_ + BR * D;
    half* KTsh[NSTAGE];
    half* Vsh[NSTAGE];
    #pragma unroll
    for (int s = 0; s < NSTAGE; s++) {
        KTsh[s] = KV_base + s * (2 * BC * D);
        Vsh[s]  = KTsh[s] + BC * D;
    }

    #define Qsh(r,c) Qsh_[(r)*D + (c)]

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[NDCOL];
    #pragma unroll
    for (int nc = 0; nc < NDCOL; nc++)
        wmma::fill_fragment(o_frag[nc], 0.0f);
    float m_lo = -FLT_MAX, m_hi = -FLT_MAX;
    float l_lo = 0.0f, l_hi = 0.0f;

    const int ntiles = (S + BC - 1) / BC;

    // Global query rows this thread's fragment elements map to (constant across loop).
    const int qrow_lo = q0 + warp * WMMA_M + lane / 4;
    const int qrow_hi = qrow_lo + 8;

    // Load Q
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh(r, c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
    }

    #define PREFETCH_TILE(tt, slot) do {                                        \
        int _kt = (tt) * BC;                                                    \
        int _vr = min(BC, S - _kt);                                             \
        int _ve = _vr * D;                                                      \
        const half* _ks = Kbh + (long)_kt * D;                                  \
        const half* _vs = Vbh + (long)_kt * D;                                  \
        int _pt = (BC * D) / blockDim.x;                                        \
        int _b = tid * _pt;                                                     \
        for (int i = 0; i < _pt; i += 8) { int p=_b+i;                          \
            if (p < _ve) cp_async_cg(&KTsh[slot][p], &_ks[p]);                  \
            else *((uint4*)&KTsh[slot][p]) = make_uint4(0,0,0,0); }             \
        for (int i = 0; i < _pt; i += 8) { int p=_b+i;                          \
            if (p < _ve) cp_async_cg(&Vsh[slot][p], &_vs[p]);                   \
            else *((uint4*)&Vsh[slot][p]) = make_uint4(0,0,0,0); }             \
        cp_async_commit();                                                      \
    } while(0)

    #pragma unroll
    for (int s = 0; s < NSTAGE - 1; s++) {
        if (s < ntiles) PREFETCH_TILE(s, s);
    }

    for (int t = 0; t < ntiles; t++) {
        int cur = t % NSTAGE;
        int kt = t * BC;
        int krem = min(S - kt, BC);

        int pf = t + (NSTAGE - 1);
        if (pf < ntiles) PREFETCH_TILE(pf, pf % NSTAGE);

        cp_async_wait_group<NSTAGE - 1>();
        __syncthreads();

        // ==== QK^T → s_frag ====
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[NCOL];
        {
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                wmma::fill_fragment(s_frag[nc], 0.0f);
                #pragma unroll
                for (int k = 0; k < D; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Qsh(warp * WMMA_M, k), D);
                    wmma::load_matrix_sync(b_frag, &KTsh[cur][nc * WMMA_N * D + k], D);
                    wmma::mma_sync(s_frag[nc], a_frag, b_frag, s_frag[nc]);
                }
            }
        }

        // ==== Softmax + O rescale + construct P fragments ====
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag[NCOL];
        {
            // Per-element validity: within key range AND (non-causal OR kcol<=qrow).
            // kcol for element group: kt + base_col + {0,1,8,9}.
            // {0,1} share col offset {0,1} on row_lo/hi; {4,5}/{6,7} at +8/+9.
            float pmax_lo = -FLT_MAX, pmax_hi = -FLT_MAX;
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                int base_col = nc * 16 + (lane % 4) * 2;
                int kc0 = kt + base_col;         // for .x[0],.x[2]
                int kc1 = kt + base_col + 1;     // for .x[1],.x[3]
                int kc8 = kt + base_col + 8;     // for .x[4],.x[6]
                int kc9 = kt + base_col + 9;     // for .x[5],.x[7]
#if CAUSAL
                bool m0_lo = (base_col     < krem) && (kc0 <= qrow_lo);
                bool m1_lo = (base_col + 1 < krem) && (kc1 <= qrow_lo);
                bool m4_lo = (base_col + 8 < krem) && (kc8 <= qrow_lo);
                bool m5_lo = (base_col + 9 < krem) && (kc9 <= qrow_lo);
                bool m2_hi = (base_col     < krem) && (kc0 <= qrow_hi);
                bool m3_hi = (base_col + 1 < krem) && (kc1 <= qrow_hi);
                bool m6_hi = (base_col + 8 < krem) && (kc8 <= qrow_hi);
                bool m7_hi = (base_col + 9 < krem) && (kc9 <= qrow_hi);
#else
                bool m0_lo = (base_col     < krem);
                bool m1_lo = (base_col + 1 < krem);
                bool m4_lo = (base_col + 8 < krem);
                bool m5_lo = (base_col + 9 < krem);
                bool m2_hi = m0_lo, m3_hi = m1_lo, m6_hi = m4_lo, m7_hi = m5_lo;
#endif
                float v0 = m0_lo ? s_frag[nc].x[0] * scale : -FLT_MAX;
                float v1 = m1_lo ? s_frag[nc].x[1] * scale : -FLT_MAX;
                float v4 = m4_lo ? s_frag[nc].x[4] * scale : -FLT_MAX;
                float v5 = m5_lo ? s_frag[nc].x[5] * scale : -FLT_MAX;
                pmax_lo = fmaxf(pmax_lo, fmaxf(fmaxf(v0, v1), fmaxf(v4, v5)));
                float v2 = m2_hi ? s_frag[nc].x[2] * scale : -FLT_MAX;
                float v3 = m3_hi ? s_frag[nc].x[3] * scale : -FLT_MAX;
                float v6 = m6_hi ? s_frag[nc].x[6] * scale : -FLT_MAX;
                float v7 = m7_hi ? s_frag[nc].x[7] * scale : -FLT_MAX;
                pmax_hi = fmaxf(pmax_hi, fmaxf(fmaxf(v2, v3), fmaxf(v6, v7)));
            }
            #pragma unroll
            for (int delta = 1; delta <= 2; delta <<= 1) {
                pmax_lo = fmaxf(pmax_lo, __shfl_xor_sync(0xffffffff, pmax_lo, delta));
                pmax_hi = fmaxf(pmax_hi, __shfl_xor_sync(0xffffffff, pmax_hi, delta));
            }
            float m_prev_lo = m_lo, m_prev_hi = m_hi;
            m_lo = fmaxf(m_lo, pmax_lo);
            m_hi = fmaxf(m_hi, pmax_hi);
            // Guard against all-masked row (m stays -FLT_MAX): corr = exp(-inf - -inf) = nan.
            float corr_lo = (m_prev_lo == -FLT_MAX) ? 0.0f : expf(m_prev_lo - m_lo);
            float corr_hi = (m_prev_hi == -FLT_MAX) ? 0.0f : expf(m_prev_hi - m_hi);
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                o_frag[nc].x[0] *= corr_lo; o_frag[nc].x[1] *= corr_lo;
                o_frag[nc].x[4] *= corr_lo; o_frag[nc].x[5] *= corr_lo;
                o_frag[nc].x[2] *= corr_hi; o_frag[nc].x[3] *= corr_hi;
                o_frag[nc].x[6] *= corr_hi; o_frag[nc].x[7] *= corr_hi;
            }
            float psum_lo = 0.0f, psum_hi = 0.0f;
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                int base_col = nc * 16 + (lane % 4) * 2;
                int kc0 = kt + base_col, kc1 = kt + base_col + 1;
                int kc8 = kt + base_col + 8, kc9 = kt + base_col + 9;
#if CAUSAL
                bool m0_lo = (base_col     < krem) && (kc0 <= qrow_lo);
                bool m1_lo = (base_col + 1 < krem) && (kc1 <= qrow_lo);
                bool m4_lo = (base_col + 8 < krem) && (kc8 <= qrow_lo);
                bool m5_lo = (base_col + 9 < krem) && (kc9 <= qrow_lo);
                bool m2_hi = (base_col     < krem) && (kc0 <= qrow_hi);
                bool m3_hi = (base_col + 1 < krem) && (kc1 <= qrow_hi);
                bool m6_hi = (base_col + 8 < krem) && (kc8 <= qrow_hi);
                bool m7_hi = (base_col + 9 < krem) && (kc9 <= qrow_hi);
#else
                bool m0_lo = (base_col     < krem);
                bool m1_lo = (base_col + 1 < krem);
                bool m4_lo = (base_col + 8 < krem);
                bool m5_lo = (base_col + 9 < krem);
                bool m2_hi = m0_lo, m3_hi = m1_lo, m6_hi = m4_lo, m7_hi = m5_lo;
#endif
                float p0 = m0_lo ? expf(s_frag[nc].x[0] * scale - m_lo) : 0.0f;
                float p1 = m1_lo ? expf(s_frag[nc].x[1] * scale - m_lo) : 0.0f;
                float p4 = m4_lo ? expf(s_frag[nc].x[4] * scale - m_lo) : 0.0f;
                float p5 = m5_lo ? expf(s_frag[nc].x[5] * scale - m_lo) : 0.0f;
                psum_lo += p0 + p1 + p4 + p5;
                float p2 = m2_hi ? expf(s_frag[nc].x[2] * scale - m_hi) : 0.0f;
                float p3 = m3_hi ? expf(s_frag[nc].x[3] * scale - m_hi) : 0.0f;
                float p6 = m6_hi ? expf(s_frag[nc].x[6] * scale - m_hi) : 0.0f;
                float p7 = m7_hi ? expf(s_frag[nc].x[7] * scale - m_hi) : 0.0f;
                psum_hi += p2 + p3 + p6 + p7;

                p_frag[nc].x[0]  = __float2half(p0);
                p_frag[nc].x[1]  = __float2half(p1);
                p_frag[nc].x[2]  = __float2half(p2);
                p_frag[nc].x[3]  = __float2half(p3);
                p_frag[nc].x[4]  = __float2half(p4);
                p_frag[nc].x[5]  = __float2half(p5);
                p_frag[nc].x[6]  = __float2half(p6);
                p_frag[nc].x[7]  = __float2half(p7);
                p_frag[nc].x[8]  = __float2half(p0);
                p_frag[nc].x[9]  = __float2half(p1);
                p_frag[nc].x[10] = __float2half(p2);
                p_frag[nc].x[11] = __float2half(p3);
                p_frag[nc].x[12] = __float2half(p4);
                p_frag[nc].x[13] = __float2half(p5);
                p_frag[nc].x[14] = __float2half(p6);
                p_frag[nc].x[15] = __float2half(p7);
            }
            #pragma unroll
            for (int delta = 1; delta <= 2; delta <<= 1) {
                psum_lo += __shfl_xor_sync(0xffffffff, psum_lo, delta);
                psum_hi += __shfl_xor_sync(0xffffffff, psum_hi, delta);
            }
            l_lo = l_lo * corr_lo + psum_lo;
            l_hi = l_hi * corr_hi + psum_hi;
        }

        // ==== PV: O += P . V ====
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                #pragma unroll
                for (int kk = 0; kk < NCOL; kk++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &Vsh[cur][kk * WMMA_K * D + nc * WMMA_N], D);
                    wmma::mma_sync(o_frag[nc], p_frag[kk], b_frag, o_frag[nc]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue
    {
        float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
        float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;
        #pragma unroll
        for (int nc = 0; nc < NDCOL; nc++) {
            o_frag[nc].x[0] *= inv_lo; o_frag[nc].x[1] *= inv_lo;
            o_frag[nc].x[4] *= inv_lo; o_frag[nc].x[5] *= inv_lo;
            o_frag[nc].x[2] *= inv_hi; o_frag[nc].x[3] *= inv_hi;
            o_frag[nc].x[6] *= inv_hi; o_frag[nc].x[7] *= inv_hi;
        }
        int row_lo = warp * WMMA_M + lane / 4;
        int row_hi = row_lo + 8;
        int grow_lo = q0 + row_lo;
        int grow_hi = q0 + row_hi;
        #pragma unroll
        for (int nc = 0; nc < NDCOL; nc++) {
            int col_base = nc * WMMA_N + (lane % 4) * 2;
            if (grow_lo < S) {
                Obh[grow_lo * D + col_base]     = __float2half(o_frag[nc].x[0]);
                Obh[grow_lo * D + col_base + 1] = __float2half(o_frag[nc].x[1]);
                Obh[grow_lo * D + col_base + 8] = __float2half(o_frag[nc].x[4]);
                Obh[grow_lo * D + col_base + 9] = __float2half(o_frag[nc].x[5]);
            }
            if (grow_hi < S) {
                Obh[grow_hi * D + col_base]     = __float2half(o_frag[nc].x[2]);
                Obh[grow_hi * D + col_base + 1] = __float2half(o_frag[nc].x[3]);
                Obh[grow_hi * D + col_base + 8] = __float2half(o_frag[nc].x[6]);
                Obh[grow_hi * D + col_base + 9] = __float2half(o_frag[nc].x[7]);
            }
        }
    }
}
