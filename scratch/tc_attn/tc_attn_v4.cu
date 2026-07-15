// TC flash-attention v4 — DOUBLE-BUFFERED K+V with software pipelining.
//
// Core idea: Two KV buffer slots (ping-pong). While computing QK^T+softmax+PV
// on the current tile (from slot A), simultaneously load the NEXT tile's K+V
// into slot B. This completely overlaps memory latency with computation.
//
// Smem layout: 48 KB
//   Qsh:      [BR=64][D=64]  fp16 = 8192 B  (persistent)
//   KV_slot0: [D][BC] + [BC][D] fp16 = 16384 B (KT + V interleaved)
//   KV_slot1: [D][BC] + [BC][D] fp16 = 16384 B (KT + V interleaved)
//   Psh:      [BR=64][BC=64] fp16 = 8192 B  (P for P.V)
//   Total: 49152 B = 48 KB → 2 blocks/SM (48*2=96K < 100K)
//
// Pipeline structure:
//   Preload tile 0 → slot0
//   sync
//   for t = 0 to N-1:
//     cur_slot = t % 2
//     next_slot = (t+1) % 2
//     if t < N-1: start loading tile t+1 → next_slot (async via cooperative threads)
//     QK^T from cur_slot.KT
//     Softmax → P
//     sync (ensure preload of next_slot is done + P writes visible)
//     PV from cur_slot.V + Psh
//     sync (ensure PV reads done before next iteration overwrites Psh)
//
// Challenge: "async" load means some threads load while others compute.
// With 4 warps, we can't truly specialize (wmma needs all warp threads).
// Instead: just restructure the loop so loads are BEFORE the sync,
// maximizing the overlap window between load→sync→compute.
//
// SIMPLER approach that still benefits:
//   Iteration t:
//     1. Load tile t+1 into next_slot (if t < N-1)  ← START immediately
//     2. Compute QK^T from cur_slot.KT              ← overlaps with load! (different smem)
//     3. Softmax → P
//     4. __syncthreads() (ensures load done + P visible)
//     5. PV from cur_slot.V + Psh
//     6. __syncthreads() (ensures PV done before next iter overwrites)
//
// But wait — steps 1 and 2 can't truly execute in parallel because each thread
// does BOTH load and compute sequentially. True overlap requires:
//   - cp.async (hardware DMA, doesn't use thread cycles)
//   - OR warp specialization (some warps load, others compute)
//
// For cp.async: each thread issues a 16-byte async copy instruction, then the
// memory controller handles it in background while the thread does compute.
// This IS true overlap!
//
// Let me implement with cp.async (requires sm_80+, we have sm_86):
//   1. Issue cp.async for tile t+1 → next_slot
//   2. QK^T on cur_slot (TC compute — no memory unit needed)
//   3. Softmax in registers (ALU only)
//   4. __pipeline_wait_prior(0) + __syncthreads()
//   5. PV from cur_slot
//   6. __syncthreads()
//
// cp.async details: use cp.async.cg.shared.global (4/8/16 bytes per thread).
// For our load: BC*D = 4096 halfs = 8192 bytes per matrix (K or V).
// 128 threads → 64 bytes per thread per matrix = 4 × uint4 (16B) loads.
// Total: 128B per thread for K+V = 8 cp.async ops of 16B each.

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
#define NCOL     (BC / WMMA_N)
#define NDCOL    (D  / WMMA_N)

// cp.async intrinsics (sm_80+)
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    // Copy 16 bytes from global to shared asynchronously
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

extern "C" __global__ void tc_flash_attn_v4(
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

    // Shared memory: 48 KB
    extern __shared__ char smem[];
    half* Qsh_    = (half*) smem;                        // 8192 B
    half* KTsh_0  = Qsh_ + BR * D;                       // slot0 K^T: 8192 B
    half* Vsh_0   = KTsh_0 + D * BC;                     // slot0 V: 8192 B
    half* KTsh_1  = Vsh_0 + BC * D;                      // slot1 K^T: 8192 B
    half* Vsh_1   = KTsh_1 + D * BC;                     // slot1 V: 8192 B
    half* Psh_    = Vsh_1 + BC * D;                      // 8192 B
    // Total: 8192 + 16384 + 16384 + 8192 = 49152 B = 48 KB

    #define Qsh(r,c)    Qsh_[(r)*D + (c)]
    #define Psh(r,c)    Psh_[(r)*BC + (c)]

    // Slot-indexed access
    half* KTsh[2] = {KTsh_0, KTsh_1};
    half* Vsh[2]  = {Vsh_0, Vsh_1};
    #define KTslot(s,r,c) KTsh[s][(r)*BC + (c)]
    #define Vslot(s,r,c)  Vsh[s][(r)*D + (c)]

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

    // Preload tile 0 into slot 0 (synchronous for first tile)
    // K stored as [BC][D] row-major (NO TRANSPOSE); use col_major load in wmma
    // V stored as [BC][D] row-major
    {
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            half kv = (r < S) ? Kbh[r * D + c] : __float2half(0.0f);
            half vv = (r < S) ? Vbh[r * D + c] : __float2half(0.0f);
            KTsh[0][r * D + c] = kv;  // K stored [BC][D] row-major (NOT transposed!)
            Vsh[0][r * D + c] = vv;
        }
    }
    __syncthreads();

    // Main loop with double-buffering
    for (int t = 0; t < ntiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int kt = t * BC;
        int krem = min(S - kt, BC);

        // ==== Issue async preload of tile t+1 → next_slot ====
        if (t + 1 < ntiles) {
            int kt_next = (t + 1) * BC;
            int valid_rows = min(BC, S - kt_next);
            int valid_elems = valid_rows * D;  // contiguous halfs
            
            // Both K and V are now [BC][D] row-major = contiguous in global memory!
            // Use cp.async for both (16 bytes = 8 halfs per op)
            const half* k_src = Kbh + (long)kt_next * D;
            const half* v_src = Vbh + (long)kt_next * D;
            int total_elems = BC * D;  // 4096 halfs
            int per_thread = total_elems / blockDim.x;  // 32 halfs = 64 bytes = 4 × 16B
            int base = tid * per_thread;
            
            // K: cp.async
            for (int i = 0; i < per_thread; i += 8) {
                int pos = base + i;
                if (pos < valid_elems) {
                    cp_async_cg(&KTsh[nxt][pos], &k_src[pos]);
                } else {
                    *((uint4*)&KTsh[nxt][pos]) = make_uint4(0,0,0,0);
                }
            }
            // V: cp.async
            for (int i = 0; i < per_thread; i += 8) {
                int pos = base + i;
                if (pos < valid_elems) {
                    cp_async_cg(&Vsh[nxt][pos], &v_src[pos]);
                } else {
                    *((uint4*)&Vsh[nxt][pos]) = make_uint4(0,0,0,0);
                }
            }
            cp_async_commit();
        }

        // ==== P2: QK^T → s_frag (registers) from cur_slot ====
        // K stored as [BC][D] row-major. Load as col_major to get K^T effect.
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
                    // K[BC][D] row-major; col_major load from &K[nc*16, k] with stride D gives K^T
                    wmma::load_matrix_sync(b_frag, &KTsh[cur][nc * WMMA_N * D + k], D);
                    wmma::mma_sync(s_frag[nc], a_frag, b_frag, s_frag[nc]);
                }
            }
        }

        // ==== P3: Softmax + write P ====
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

        // Wait for async preload AND ensure P writes visible
        if (t + 1 < ntiles) cp_async_wait_all();
        __syncthreads();

        // ==== P5: O += P . V from cur_slot ====
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                #pragma unroll
                for (int k = 0; k < BC; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Psh(warp * WMMA_M, k), BC);
                    wmma::load_matrix_sync(b_frag, &Vsh[cur][k * D + nc * WMMA_N], D);
                    wmma::mma_sync(o_frag[nc], a_frag, b_frag, o_frag[nc]);
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
