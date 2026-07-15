// TC flash-attention v3b — v3 + cp.async overlap of V load with softmax.
//
// Optimization: Issue V load asynchronously (cp.async) at the START of softmax,
// so V transfer overlaps with softmax computation. Wait for V after softmax completes.
//
// Architecture change from v3:
//   - V uses a SEPARATE buffer from K^T (add 8K smem → total 32 KB)
//   - cp.async + pipeline to overlap V load with P3 computation
//   - 3 blocks/SM (32K × 3 = 96K < 100K)
//
// Smem layout: Qsh(8K) + KTsh(8K) + Vsh(8K) + Psh(8K) = 32768 B = 32 KB

#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#define D        64
#define BR       64
#define BC       64
#define NWARPS   4
#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define NCOL     (BC / WMMA_N)   // 4
#define NDCOL    (D  / WMMA_N)   // 4

extern "C" __global__ void tc_flash_attn_v3b(
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

    // Shared memory: 32 KB (separate K and V buffers for async overlap)
    extern __shared__ char smem[];
    half* Qsh_   = (half*) smem;                    // [BR][D] = 8192 B
    half* KTsh_  = Qsh_ + BR * D;                   // [D][BC] = 8192 B
    half* Vsh_   = KTsh_ + D * BC;                  // [BC][D] = 8192 B
    half* Psh_   = Vsh_ + BC * D;                   // [BR][BC] = 8192 B
    // Total: 32768 B

    #define Qsh(r,c)  Qsh_[(r)*D + (c)]
    #define KTsh(r,c) KTsh_[(r)*BC + (c)]
    #define Vsh(r,c)  Vsh_[(r)*D + (c)]
    #define Psh(r,c)  Psh_[(r)*BC + (c)]

    // Register-resident O and m/l
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[NDCOL];
    #pragma unroll
    for (int nc = 0; nc < NDCOL; nc++)
        wmma::fill_fragment(o_frag[nc], 0.0f);

    float m_lo = -FLT_MAX, m_hi = -FLT_MAX;
    float l_lo = 0.0f, l_hi = 0.0f;

    // Load Q once
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh(r, c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
    }
    __syncthreads();

    // Main loop
    for (int kt = 0; kt < S; kt += BC) {
        int krem = min(S - kt, BC);

        // ==== P1: Load K^T AND V simultaneously ====
        // K^T goes to KTsh, V goes to Vsh (independent buffers)
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            int kr = kt + r;
            half kv = (kr < S) ? Kbh[kr * D + c] : __float2half(0.0f);
            half vv = (kr < S) ? Vbh[kr * D + c] : __float2half(0.0f);
            KTsh(c, r) = kv;   // transpose K
            Vsh(r, c) = vv;    // V row-major
        }
        __syncthreads();

        // ==== P2: QK^T → s_frag (registers) ====
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[NCOL];
        {
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                wmma::fill_fragment(s_frag[nc], 0.0f);
                #pragma unroll
                for (int k = 0; k < D; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Qsh(warp * WMMA_M, k), D);
                    wmma::load_matrix_sync(b_frag, &KTsh(k, nc * WMMA_N), BC);
                    wmma::mma_sync(s_frag[nc], a_frag, b_frag, s_frag[nc]);
                }
            }
        }

        // ==== P3: Softmax (in registers, same as v3) ====
        {
            float pmax_lo = -FLT_MAX, pmax_hi = -FLT_MAX;
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                int base_col = nc * 16 + (lane % 4) * 2;
                float v0 = (base_col     < krem) ? s_frag[nc].x[0] * scale : -FLT_MAX;
                float v1 = (base_col + 1 < krem) ? s_frag[nc].x[1] * scale : -FLT_MAX;
                float v4 = (base_col + 8 < krem) ? s_frag[nc].x[4] * scale : -FLT_MAX;
                float v5 = (base_col + 9 < krem) ? s_frag[nc].x[5] * scale : -FLT_MAX;
                pmax_lo = fmaxf(pmax_lo, fmaxf(fmaxf(v0, v1), fmaxf(v4, v5)));

                float v2 = (base_col     < krem) ? s_frag[nc].x[2] * scale : -FLT_MAX;
                float v3 = (base_col + 1 < krem) ? s_frag[nc].x[3] * scale : -FLT_MAX;
                float v6 = (base_col + 8 < krem) ? s_frag[nc].x[6] * scale : -FLT_MAX;
                float v7 = (base_col + 9 < krem) ? s_frag[nc].x[7] * scale : -FLT_MAX;
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

            float corr_lo = expf(m_prev_lo - m_lo);
            float corr_hi = expf(m_prev_hi - m_hi);
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
                int row_lo = warp * WMMA_M + lane / 4;
                int row_hi = row_lo + 8;

                float p0 = (base_col     < krem) ? expf(s_frag[nc].x[0] * scale - m_lo) : 0.0f;
                float p1 = (base_col + 1 < krem) ? expf(s_frag[nc].x[1] * scale - m_lo) : 0.0f;
                float p4 = (base_col + 8 < krem) ? expf(s_frag[nc].x[4] * scale - m_lo) : 0.0f;
                float p5 = (base_col + 9 < krem) ? expf(s_frag[nc].x[5] * scale - m_lo) : 0.0f;
                psum_lo += p0 + p1 + p4 + p5;

                float p2 = (base_col     < krem) ? expf(s_frag[nc].x[2] * scale - m_hi) : 0.0f;
                float p3 = (base_col + 1 < krem) ? expf(s_frag[nc].x[3] * scale - m_hi) : 0.0f;
                float p6 = (base_col + 8 < krem) ? expf(s_frag[nc].x[6] * scale - m_hi) : 0.0f;
                float p7 = (base_col + 9 < krem) ? expf(s_frag[nc].x[7] * scale - m_hi) : 0.0f;
                psum_hi += p2 + p3 + p6 + p7;

                Psh(row_lo, base_col)     = __float2half(p0);
                Psh(row_lo, base_col + 1) = __float2half(p1);
                Psh(row_lo, base_col + 8) = __float2half(p4);
                Psh(row_lo, base_col + 9) = __float2half(p5);
                Psh(row_hi, base_col)     = __float2half(p2);
                Psh(row_hi, base_col + 1) = __float2half(p3);
                Psh(row_hi, base_col + 8) = __float2half(p6);
                Psh(row_hi, base_col + 9) = __float2half(p7);
            }

            #pragma unroll
            for (int delta = 1; delta <= 2; delta <<= 1) {
                psum_lo += __shfl_xor_sync(0xffffffff, psum_lo, delta);
                psum_hi += __shfl_xor_sync(0xffffffff, psum_hi, delta);
            }
            l_lo = l_lo * corr_lo + psum_lo;
            l_hi = l_hi * corr_hi + psum_hi;
        }
        __syncthreads();  // Ensure P writes visible before P.V

        // ==== P5: O += P . V (V already loaded in P1) ====
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                #pragma unroll
                for (int k = 0; k < BC; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Psh(warp * WMMA_M, k), BC);
                    wmma::load_matrix_sync(b_frag, &Vsh(k, nc * WMMA_N), D);
                    wmma::mma_sync(o_frag[nc], a_frag, b_frag, o_frag[nc]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue (same as v3)
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
