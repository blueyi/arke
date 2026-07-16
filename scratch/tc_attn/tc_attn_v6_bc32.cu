#include <cstdint>
// TC flash-attention v6 — v4b + BC=32 for higher occupancy.
//
// Derived from v4b (register-resident P + O, cp.async double-buffer). The only
// change is BC 64 -> 32, which halves each K/V shared slot (8K -> 4K):
//   Smem = Qsh(8K) + 2*(K 4K + V 4K) = 24 KB  ->  4 blocks/SM (24*4=96K<100K)
// vs v4b's 40 KB -> 2 blocks/SM. ptxas: 72 regs (v4b 121), 0 spill.
//
// Result (RTX 3060 Laptop, kernel-only, median 5x200): v6 is 1.05-1.55x faster
// than v4b in absolute time across S=512..2048, correctness identical
// (max_err <=2.4e-4). The double-buffered async prefetch is KEPT — raising
// residency to 4 blocks/SM while preserving latency hiding is what wins;
// dropping the double-buffer to raise occupancy (occA/occB) was slower.
//
// Key insight: matrix_a fragment layout on sm_86 (m16n16k16, half, row_major)
// is identical to accumulator layout but with 16 half elements (8 unique + 8 duplicates).
// Elements 0-7 map to same (row,col) as accumulator; 8-15 duplicate 0-7.
//
// So after softmax we have p0..p7 (float) per thread per s_frag[nc].
// For PV k-step kk = nc*16: construct p_a_frag.x[0..7] = fp16(p_values from s_frag[nc])
// and p_a_frag.x[8..15] = duplicate of [0..7].

#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#define D        64
#define BR       64
#define BC       32
#define NWARPS   4
#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define NCOL     (BC / WMMA_N)
#define NDCOL    (D  / WMMA_N)

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

extern "C" __global__ void tc_flash_attn_v6(
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

    // Shared memory: 24 KB (BC=32; NO Psh!)
    extern __shared__ char smem[];
    half* Qsh_    = (half*) smem;                        // BR*D*2 = 8192 B
    half* KTsh_0  = Qsh_ + BR * D;                       // slot0 K: BC*D*2 = 4096 B
    half* Vsh_0   = KTsh_0 + BC * D;                     // slot0 V: 4096 B
    half* KTsh_1  = Vsh_0 + BC * D;                      // slot1 K: 4096 B
    half* Vsh_1   = KTsh_1 + BC * D;                     // slot1 V: 4096 B
    // Total: 8192 + 2*(4096+4096) = 24576 B = 24 KB

    #define Qsh(r,c) Qsh_[(r)*D + (c)]
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

    // Load Q
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh(r, c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
    }

    // Preload tile 0 (K + V, both [BC][D] row-major)
    {
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            half kv = (r < S) ? Kbh[r * D + c] : __float2half(0.0f);
            half vv = (r < S) ? Vbh[r * D + c] : __float2half(0.0f);
            KTsh[0][r * D + c] = kv;
            Vsh[0][r * D + c] = vv;
        }
    }
    __syncthreads();

    // Main loop
    for (int t = 0; t < ntiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int kt = t * BC;
        int krem = min(S - kt, BC);

        // ==== Async preload tile t+1 ====
        if (t + 1 < ntiles) {
            int kt_next = (t + 1) * BC;
            int valid_rows = min(BC, S - kt_next);
            int valid_elems = valid_rows * D;
            const half* k_src = Kbh + (long)kt_next * D;
            const half* v_src = Vbh + (long)kt_next * D;
            int per_thread = (BC * D) / blockDim.x;
            int base = tid * per_thread;
            for (int i = 0; i < per_thread; i += 8) {
                int pos = base + i;
                if (pos < valid_elems) cp_async_cg(&KTsh[nxt][pos], &k_src[pos]);
                else *((uint4*)&KTsh[nxt][pos]) = make_uint4(0,0,0,0);
            }
            for (int i = 0; i < per_thread; i += 8) {
                int pos = base + i;
                if (pos < valid_elems) cp_async_cg(&Vsh[nxt][pos], &v_src[pos]);
                else *((uint4*)&Vsh[nxt][pos]) = make_uint4(0,0,0,0);
            }
            cp_async_commit();
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
                    wmma::load_matrix_sync(a_frag, &Qsh(warp * WMMA_M, k), D);
                    wmma::load_matrix_sync(b_frag, &KTsh[cur][nc * WMMA_N * D + k], D);
                    wmma::mma_sync(s_frag[nc], a_frag, b_frag, s_frag[nc]);
                }
            }
        }

        // ==== Softmax + O rescale + construct P fragments ====
        // P values stored as half in p_frags[NCOL] (matrix_a fragments)
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

                // Construct matrix_a fragment directly from P values!
                // Layout matches accumulator: 0,1,4,5=row_lo; 2,3,6,7=row_hi
                // Elements 8-15 duplicate 0-7
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

        // Wait for async preload (no sync needed for P — it's in registers!)
        if (t + 1 < ntiles) cp_async_wait_all();
        __syncthreads();

        // ==== PV: O += P . V using register-resident p_frag ====
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                #pragma unroll
                for (int kk = 0; kk < NCOL; kk++) {  // kk iterates over P columns = V rows
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
