// TC flash-attention v4b_pad — v4b + shared memory padding experiment.
//
// Tests whether adding 8-half (16-byte) padding per row to shared memory
// buffers eliminates bank conflicts and improves performance.
//
// With D=64, stride = 64 halfs = 128 bytes = 32 banks → full wrap (no conflicts).
// Padding changes stride to 72 halfs = 144 bytes, which is NOT a multiple of 32 banks.
// If bank conflicts were present, padding would help. If stride already wraps
// cleanly, padding just wastes smem and may hurt due to reduced occupancy or
// extra addressing math.
//
// Layout change:
//   Qsh: [BR=64][D+PAD=72] fp16, stride = 72 halfs
//   KTsh: [BC=64][D+PAD=72] fp16, stride = 72 halfs
//   Vsh: [BC=64][D+PAD=72] fp16, stride = 72 halfs
//
// Smem: 64*72*2 * 5 buffers (Q + 2*K + 2*V) = 46080 B = 45 KB
// Still fits 2 blocks/SM (45*2=90K < 100K).
//
// Uses synchronous loads (no cp.async) to avoid complexity with padded layout.

#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#define D        64
#define PAD      8
#define DPAD     (D + PAD)   // 72
#define BR       64
#define BC       64
#define NWARPS   4
#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define NCOL     (BC / WMMA_N)
#define NDCOL    (D  / WMMA_N)

extern "C" __global__ void tc_flash_attn_v4b_pad(
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

    // Shared memory: 45 KB with padding
    extern __shared__ char smem[];
    half* Qsh_    = (half*) smem;                              // 64*72*2 = 9216 B
    half* KTsh_0  = Qsh_ + BR * DPAD;                         // slot0 K: 9216 B
    half* Vsh_0   = KTsh_0 + BC * DPAD;                       // slot0 V: 9216 B
    half* KTsh_1  = Vsh_0 + BC * DPAD;                        // slot1 K: 9216 B
    half* Vsh_1   = KTsh_1 + BC * DPAD;                       // slot1 V: 9216 B
    // Total: 9216 * 5 = 46080 B = 45 KB

    #define Qsh_pad(r,c) Qsh_[(r)*DPAD + (c)]
    half* KTsh[2] = {KTsh_0, KTsh_1};
    half* Vsh[2]  = {Vsh_0, Vsh_1};

    // Register-resident O and m/l
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[NDCOL];
    #pragma unroll
    for (int nc = 0; nc < NDCOL; nc++)
        wmma::fill_fragment(o_frag[nc], 0.0f);
    float m_lo = -FLT_MAX, m_hi = -FLT_MAX;
    float l_lo = 0.0f, l_hi = 0.0f;

    const int ntiles = (S + BC - 1) / BC;

    // Load Q into padded shared memory
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh_pad(r, c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
    }

    // Preload tile 0 (K + V) into padded layout
    {
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            half kv = (r < S) ? Kbh[r * D + c] : __float2half(0.0f);
            half vv = (r < S) ? Vbh[r * D + c] : __float2half(0.0f);
            KTsh[0][r * DPAD + c] = kv;
            Vsh[0][r * DPAD + c] = vv;
        }
    }
    __syncthreads();

    // Main loop
    for (int t = 0; t < ntiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int kt = t * BC;
        int krem = min(S - kt, BC);

        // ==== Preload tile t+1 (synchronous, padded layout) ====
        if (t + 1 < ntiles) {
            int kt_next = (t + 1) * BC;
            int valid_rows = min(BC, S - kt_next);
            const half* k_src = Kbh + (long)kt_next * D;
            const half* v_src = Vbh + (long)kt_next * D;
            for (int idx = tid; idx < BC * D; idx += blockDim.x) {
                int r = idx / D, c = idx % D;
                half kv = (r < valid_rows) ? k_src[r * D + c] : __float2half(0.0f);
                half vv = (r < valid_rows) ? v_src[r * D + c] : __float2half(0.0f);
                KTsh[nxt][r * DPAD + c] = kv;
                Vsh[nxt][r * DPAD + c] = vv;
            }
        }

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
                    wmma::load_matrix_sync(a_frag, &Qsh_pad(warp * WMMA_M, k), DPAD);
                    wmma::load_matrix_sync(b_frag, &KTsh[cur][nc * WMMA_N * DPAD + k], DPAD);
                    wmma::mma_sync(s_frag[nc], a_frag, b_frag, s_frag[nc]);
                }
            }
        }

        // ==== Softmax + O rescale + construct P fragments ====
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag[NCOL];
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

        // Synchronize before PV (need KV data in cur slot to be stable)
        if (t + 1 < ntiles) __syncthreads();  // ensure preload of nxt doesn't clobber cur
        // Actually we need sync AFTER preload completes and BEFORE we read cur for PV.
        // But preload writes to nxt slot, PV reads from cur slot — no conflict!
        // We only need sync before next iteration reads nxt. Let's sync at end of loop.

        // ==== PV: O += P . V using register-resident p_frag ====
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                #pragma unroll
                for (int kk = 0; kk < NCOL; kk++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &Vsh[cur][kk * WMMA_K * DPAD + nc * WMMA_N], DPAD);
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
