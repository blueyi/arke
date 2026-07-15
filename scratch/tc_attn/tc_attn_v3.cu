// TC flash-attention v3 — REGISTER-RESIDENT O + fragment-level softmax.
//
// Key breakthrough: O accumulator stays in wmma fragments (registers) throughout.
// Softmax operates directly on S fragment values using the known sm_86 layout.
// This eliminates:
//   - Osh (16K shared memory)
//   - All load/store_matrix_sync for O
//   - Ssh store + reload for softmax
//
// sm_86 wmma m16n16k16 fp32 accumulator layout:
//   Thread t (lane 0-31) holds 8 floats. Each thread owns rows [t/4] and [t/4+8].
//   Elements 0,1,4,5 → row (t/4);  elements 2,3,6,7 → row (t/4)+8.
//   Column mapping: elements 0,1 → cols (t%4)*2, (t%4)*2+1
//                   elements 4,5 → cols (t%4)*2+8, (t%4)*2+9
//                   (same for elem 2,3 and 6,7 with row+8)
//
// Shared memory layout (~33 KB):
//   Qsh:   [BR=64][D=64]  fp16 = 8192 B  (persistent)
//   KSsh:  16384 B (K^T as [D][BC] fp16 first 8K, then S [BR][BC] fp32 full 16K)
//   Psh:   [BR=64][BC=64] fp16 = 8192 B  (for P.V, separate from S)
//   Vsh:   [BC=64][D=64]  fp16 = 8192 B  (V data for P.V)  — OR reuse KSsh after S consumed
//   m_sh:  [BR] fp32 = 256 B
//   l_sh:  [BR] fp32 = 256 B
//
// Wait — without Osh we save 16K! New layout:
//   Qsh:  8K + KSsh: 16K + Psh: 8K + m/l: 512B = 32.5 KB → MUCH better!
//   But we still need V somewhere. After softmax, KSsh is consumed, reuse for V (8K).
//   Psh (8K) = P for P.V. KSsh first 8K = V. That's what v2 did.
//   Total: Qsh(8K) + KSsh(16K) + Psh(8K) + m_sh(256) + l_sh(256) = 32768 B = 32 KB
//   2 blocks × 32K = 64K < 100K → could even fit 3 blocks! (3×32K = 96K < 100K)
//   3 blocks/SM = 384 threads = 12 warps → much better latency hiding!
//
// But wait — we need V. After softmax writes P to Psh and S (in KSsh) is consumed,
// V goes into KSsh (reuse). Layout:
//   Phase 1: K^T in KSsh first 8K
//   Phase 2: S (fp32) in KSsh full 16K (overwrites K)
//   Phase 3: Softmax from S (fragment-resident, but we need full S row for max/sum
//            which is distributed across NCOL=4 fragments)
//            → actually we DO need S in shared for the cross-column-tile reduction!
//            → OR we can do the reduction across fragments in registers using shuffle
//   Phase 4: V in KSsh first 8K (S consumed), P in Psh (8K)
//   Phase 5: P.V → accumulate into O fragments (register-resident!)
//
// The softmax challenge: each warp has NCOL=4 fragments, each 16×16.
// For a given row r, the full S row has 64 values spread across 4 fragments:
//   frag[0]: cols 0-15, frag[1]: cols 16-31, frag[2]: cols 32-47, frag[3]: cols 48-63
// Each thread holds 2 values per fragment for its two rows (elem 0,1 for row_lo cols 0-1 of that tile).
// Actually per fragment per thread: 2 values for row_lo (elem 0,1 or 4,5) and 2 for row_hi.
// For row_lo across 4 frags: 4×2 = 8 values per half-tile, but BC=64 needs 64 values/row.
// Wait — let me recount. Per fragment, thread t has:
//   For row_lo: elem 0 → col (t%4)*2, elem 1 → col (t%4)*2+1, elem 4 → col (t%4)*2+8, elem 5 → col (t%4)*2+9
//   That's 4 values of row_lo per fragment.
//   Across NCOL=4 fragments, each covering 16 columns: 4 frags × 4 values = 16 values per row per thread? No...
//   Wait: col offset within the 16-col tile is (t%4)*2 and (t%4)*2+1 for elem 0,1
//   and (t%4)*2+8, (t%4)*2+9 for elem 4,5.
//   So per fragment tile nc, the global columns are nc*16 + local_col.
//   Thread t owns cols: nc*16 + (t%4)*2, nc*16 + (t%4)*2+1, nc*16 + (t%4)*2+8, nc*16 + (t%4)*2+9
//   That's 4 columns per fragment. Across 4 frags: 16 columns.
//   But each row has 64 columns (BC=64). Thread t only owns 16 of the 64 columns per row.
//   The other 48 columns are owned by other threads in the warp.
//   
//   So for softmax (row max, exp, sum): we need cross-thread reduction within the warp!
//   Each thread computes partial max/sum over its 16 columns, then warp-reduce.
//   This is perfectly doable with __shfl_xor!
//
// But there's a subtlety: 4 threads share the same row assignments. 
// Threads with same (t/4) value own the same rows but different columns.
// t/4 gives a "thread-group" of 4 threads (lanes 0-3, 4-7, ..., 28-31) → 8 groups.
// Within each group of 4 threads, all own the same 2 rows (lo and hi).
// Each thread in the group owns different columns (t%4 determines col offset).
// So we have 4 threads × 4 values/frag × 4 frags = 64 values/row covered by 4 threads!
// Wait: 4 threads × (4 cols per frag × 4 frags) = 4 threads × 16 cols = 64 cols ✓
//
// For softmax reduction: only 4 threads need to cooperate per row!
// That's a 4-thread reduction, not a 32-thread reduction!
// Much cheaper: just 2 rounds of __shfl_xor with mask 1,2 among the 4 threads in a group.
//
// This is MUCH more efficient than the current shared-memory softmax!
//
// HOWEVER: we still need to write P to shared memory for P.V (wmma load_matrix_sync needs
// data in shared/global memory). So after computing P values in registers, we store them to Psh.
// The S → softmax → P path is:
//   1. S values in fragments (from QK^T)
//   2. Scale + row max (4-thread warp-group reduce)
//   3. exp + row sum (4-thread reduce)
//   4. P = exp_val (fp16) → store to Psh
//   5. Rescale O fragments: frag_O.x[i] *= corr for appropriate corr
//
// This eliminates Ssh entirely (S stays in registers) and Osh entirely (O stays in registers).
// Only Psh needed for P.V input.
//
// FINAL SMEM LAYOUT:
//   Qsh:  [64][64] fp16 = 8192 B
//   KVsh: [64][64] fp16 = 8192 B (K^T then V, time-shared — no S in smem!)
//   Psh:  [64][64] fp16 = 8192 B (P for P.V)
//   m_sh: [64] fp32 = 256 B (row-level m, needed for cross-tile rescale)
//   l_sh: [64] fp32 = 256 B (row-level l)
//   Total: 24832 B + 512 B = 25088 B ≈ 24.5 KB
//   → 4 blocks/SM! (4 × 25K = 100K ≤ 100K)... tight. Let's use 3 blocks safely.
//   Actually 4 × 25088 = 100352 > 102400 (100 KB). Need to check exact SM limit.
//   sm_86 configurable shared = 100 KB = 102400 B. 4 × 25088 = 100352 < 102400 ✓ barely!
//
// Wait — we don't even need m_sh/l_sh in shared if we keep them in registers!
// Each thread owns 2 rows. It can keep m and l for those rows in 4 floats (m_lo, m_hi, l_lo, l_hi).
// Then we don't need m_sh/l_sh at all!
// But at epilogue (O/l divide), we'd need l per row... each thread knows l for its rows.
// Wait — the epilogue writes O back to global. O is in fragments. Each element just needs
// to be divided by l of its row. Thread t knows l_lo (for row t/4) and l_hi (for row t/4+8).
// Perfect — no shared memory needed for m/l either!
//
// ULTRA-MINIMAL SMEM:
//   Qsh: 8K + KVsh: 8K + Psh: 8K = 24576 B = 24 KB
//   4 blocks × 24K = 96K < 102.4K ✓ → 4 blocks/SM!
//   4 blocks × 128 threads = 512 threads/SM (33% occupancy on 1536 max — much better than 8.3%)
//
// Let me implement this!

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

// sm_86 fragment layout helpers:
// Thread t's row_lo = t/4, row_hi = t/4 + 8 (within the 16x16 tile)
// Elements 0,1,4,5 belong to row_lo; elements 2,3,6,7 belong to row_hi.
// For 4-thread group reduction: threads with same (t/4) own same rows, different cols.
// Group-local reduction uses mask = lane & 3 neighbors.

extern "C" __global__ void tc_flash_attn_v3(
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

    // Shared memory: 24 KB
    extern __shared__ char smem[];
    half* Qsh_  = (half*) smem;                     // [BR][D] = 8192 B
    half* KVsh_ = Qsh_ + BR * D;                    // [max(D,BC)][max(D,BC)] = 8192 B
    half* Psh_  = KVsh_ + BC * D;                   // [BR][BC] = 8192 B
    // Total: 24576 B

    #define Qsh(r,c)  Qsh_[(r)*D + (c)]
    #define KTsh(r,c) KVsh_[(r)*BC + (c)]     // K^T: [D][BC]
    #define Vsh(r,c)  KVsh_[(r)*D + (c)]      // V: [BC][D]
    #define Psh(r,c)  Psh_[(r)*BC + (c)]

    // Register-resident O accumulators: NDCOL=4 fragments per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[NDCOL];
    #pragma unroll
    for (int nc = 0; nc < NDCOL; nc++)
        wmma::fill_fragment(o_frag[nc], 0.0f);

    // Register-resident m and l (per-thread, for its 2 rows)
    float m_lo = -FLT_MAX, m_hi = -FLT_MAX;  // row max
    float l_lo = 0.0f, l_hi = 0.0f;          // row sum (denominator)

    // Load Q tile once
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh(r, c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
    }
    __syncthreads();

    // ---- Main loop over key tiles ----
    for (int kt = 0; kt < S; kt += BC) {
        int krem = min(S - kt, BC);

        // ==== P1: Load K^T into KVsh ====
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            int kr = kt + r;
            half kv = (kr < S) ? Kbh[kr * D + c] : __float2half(0.0f);
            KTsh(c, r) = kv;  // [D][BC]
        }
        __syncthreads();

        // ==== P2: S = Q . K^T (TC) — result stays in s_frag registers ====
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
        // No sync needed here — s_frag is register-local!

        // ==== P3: Online softmax entirely in registers ====
        // Each thread owns elements for row_lo (lane/4) and row_hi (lane/4 + 8).
        // Across NCOL=4 fragments, each thread owns 4 values per row per fragment = 16 per row total.
        // A "thread group" of 4 threads (same lane/4) collectively owns all 64 cols of their 2 rows.
        // Reduction needs only 4-thread groups (lanes sharing same lane/4 → different lane%4).
        {
            // Step 1: Compute per-thread partial max for row_lo and row_hi
            float pmax_lo = -FLT_MAX, pmax_hi = -FLT_MAX;
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                // For each fragment nc, this thread's elements for row_lo: indices 0,1,4,5
                // Global column for elem 0: nc*16 + (lane%4)*2
                // Global column for elem 1: nc*16 + (lane%4)*2 + 1
                // Global column for elem 4: nc*16 + (lane%4)*2 + 8
                // Global column for elem 5: nc*16 + (lane%4)*2 + 9
                int base_col = nc * 16 + (lane % 4) * 2;
                float v0 = (base_col     < krem) ? s_frag[nc].x[0] * scale : -FLT_MAX;
                float v1 = (base_col + 1 < krem) ? s_frag[nc].x[1] * scale : -FLT_MAX;
                float v4 = (base_col + 8 < krem) ? s_frag[nc].x[4] * scale : -FLT_MAX;
                float v5 = (base_col + 9 < krem) ? s_frag[nc].x[5] * scale : -FLT_MAX;
                pmax_lo = fmaxf(pmax_lo, fmaxf(fmaxf(v0, v1), fmaxf(v4, v5)));

                // For row_hi: indices 2,3,6,7
                float v2 = (base_col     < krem) ? s_frag[nc].x[2] * scale : -FLT_MAX;
                float v3 = (base_col + 1 < krem) ? s_frag[nc].x[3] * scale : -FLT_MAX;
                float v6 = (base_col + 8 < krem) ? s_frag[nc].x[6] * scale : -FLT_MAX;
                float v7 = (base_col + 9 < krem) ? s_frag[nc].x[7] * scale : -FLT_MAX;
                pmax_hi = fmaxf(pmax_hi, fmaxf(fmaxf(v2, v3), fmaxf(v6, v7)));
            }

            // 4-thread group reduction for max (threads with same lane/4)
            // Group members are at lanes: (lane/4)*4 + 0, +1, +2, +3
            // Use __shfl_xor with delta 1 and 2 (only within the 4-lane group)
            #pragma unroll
            for (int delta = 1; delta <= 2; delta <<= 1) {
                pmax_lo = fmaxf(pmax_lo, __shfl_xor_sync(0xffffffff, pmax_lo, delta));
                pmax_hi = fmaxf(pmax_hi, __shfl_xor_sync(0xffffffff, pmax_hi, delta));
            }
            // Now all 4 threads in a group have the same pmax_lo/pmax_hi

            // Update running m
            float m_prev_lo = m_lo, m_prev_hi = m_hi;
            m_lo = fmaxf(m_lo, pmax_lo);
            m_hi = fmaxf(m_hi, pmax_hi);

            // Rescale O fragments
            float corr_lo = expf(m_prev_lo - m_lo);
            float corr_hi = expf(m_prev_hi - m_hi);
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                o_frag[nc].x[0] *= corr_lo;
                o_frag[nc].x[1] *= corr_lo;
                o_frag[nc].x[4] *= corr_lo;
                o_frag[nc].x[5] *= corr_lo;
                o_frag[nc].x[2] *= corr_hi;
                o_frag[nc].x[3] *= corr_hi;
                o_frag[nc].x[6] *= corr_hi;
                o_frag[nc].x[7] *= corr_hi;
            }

            // Step 2: Compute exp and sum, write P to Psh
            float psum_lo = 0.0f, psum_hi = 0.0f;
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                int base_col = nc * 16 + (lane % 4) * 2;

                // row_lo elements
                float p0 = (base_col     < krem) ? expf(s_frag[nc].x[0] * scale - m_lo) : 0.0f;
                float p1 = (base_col + 1 < krem) ? expf(s_frag[nc].x[1] * scale - m_lo) : 0.0f;
                float p4 = (base_col + 8 < krem) ? expf(s_frag[nc].x[4] * scale - m_lo) : 0.0f;
                float p5 = (base_col + 9 < krem) ? expf(s_frag[nc].x[5] * scale - m_lo) : 0.0f;
                psum_lo += p0 + p1 + p4 + p5;

                // row_hi elements
                float p2 = (base_col     < krem) ? expf(s_frag[nc].x[2] * scale - m_hi) : 0.0f;
                float p3 = (base_col + 1 < krem) ? expf(s_frag[nc].x[3] * scale - m_hi) : 0.0f;
                float p6 = (base_col + 8 < krem) ? expf(s_frag[nc].x[6] * scale - m_hi) : 0.0f;
                float p7 = (base_col + 9 < krem) ? expf(s_frag[nc].x[7] * scale - m_hi) : 0.0f;
                psum_hi += p2 + p3 + p6 + p7;

                // Write P to shared (need proper row/col mapping)
                int row_lo = warp * WMMA_M + lane / 4;
                int row_hi = row_lo + 8;
                Psh(row_lo, base_col)     = __float2half(p0);
                Psh(row_lo, base_col + 1) = __float2half(p1);
                Psh(row_lo, base_col + 8) = __float2half(p4);
                Psh(row_lo, base_col + 9) = __float2half(p5);
                Psh(row_hi, base_col)     = __float2half(p2);
                Psh(row_hi, base_col + 1) = __float2half(p3);
                Psh(row_hi, base_col + 8) = __float2half(p6);
                Psh(row_hi, base_col + 9) = __float2half(p7);
            }

            // 4-thread group reduction for sum
            #pragma unroll
            for (int delta = 1; delta <= 2; delta <<= 1) {
                psum_lo += __shfl_xor_sync(0xffffffff, psum_lo, delta);
                psum_hi += __shfl_xor_sync(0xffffffff, psum_hi, delta);
            }

            // Update running l
            l_lo = l_lo * corr_lo + psum_lo;
            l_hi = l_hi * corr_hi + psum_hi;
        }
        // P3 writes to Psh, P4 writes to KVsh — different buffers, no sync needed between them!

        // ==== P4: Load V into KVsh (K^T consumed, safe to overwrite) ====
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            int kr = kt + r;
            half vv = (kr < S) ? Vbh[kr * D + c] : __float2half(0.0f);
            Vsh(r, c) = vv;
        }
        __syncthreads();  // Ensure BOTH P (from P3) and V (from P4) are ready for P5

        // ==== P5: O += P . V (TC, accumulate into register-resident o_frag) ====
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
        __syncthreads();  // Ensure all P/V reads done before next tile overwrites
    }

    // ---- Epilogue: O = o_frag / l, write fp16 to global ----
    // Each thread divides its elements by l of the corresponding row, then stores.
    // Use store_matrix_sync to write o_frag to a temp shared area, then copy to global.
    // Reuse Psh as temp (8K, we need BR*D*4 = 16K for fp32... too big).
    // Alternative: divide in registers, cast to fp16, then write using fragment knowledge.
    {
        float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
        float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;

        #pragma unroll
        for (int nc = 0; nc < NDCOL; nc++) {
            o_frag[nc].x[0] *= inv_lo;
            o_frag[nc].x[1] *= inv_lo;
            o_frag[nc].x[4] *= inv_lo;
            o_frag[nc].x[5] *= inv_lo;
            o_frag[nc].x[2] *= inv_hi;
            o_frag[nc].x[3] *= inv_hi;
            o_frag[nc].x[6] *= inv_hi;
            o_frag[nc].x[7] *= inv_hi;
        }

        // Write O to global via shared memory staging (reuse KVsh + Psh = 16K for [16][D] fp32)
        // Each warp writes its 16×64 tile. Use KVsh_ area (8K) which can hold 16*16*4 = 1024 B per nc tile.
        // Actually 16*64*4 = 4096 B. KVsh_ is 8K so we can fit one warp's full row at a time.
        // Better: use store_matrix_sync to write each 16x16 tile to Psh area, then copy to global.
        float* Otemp = (float*) KVsh_;  // reuse KVsh as float[16][D] (4096 B, fits in 8K)

        #pragma unroll
        for (int nc = 0; nc < NDCOL; nc++) {
            wmma::store_matrix_sync(&Otemp[nc * WMMA_N], o_frag[nc], D, wmma::mem_row_major);
        }
        __syncthreads();

        // Copy from Otemp to global (each thread handles some elements)
        // Otemp is [WMMA_M][D] per warp, but we stored all warps' fragments there...
        // Wait — all 4 warps store to the SAME Otemp area! That's a race!
        // Fix: each warp uses a different offset in Otemp, or serialize, or use separate areas.
        // KVsh_ is 8K = 2048 floats. Each warp needs 16*64 = 1024 floats = 4096 B.
        // 4 warps × 4K = 16K > 8K. Doesn't fit.
        //
        // SOLUTION: Do epilogue per-warp, one warp at a time? Too slow.
        // BETTER: Write O directly from fragments to global using the known layout.
        // Thread t writes its 8 elements directly to the correct positions in global O.
        // Position: row = warp*16 + row_within_tile, col = nc*16 + col_within_tile
        // row_lo = warp*16 + lane/4, row_hi = warp*16 + lane/4 + 8
        // For fragment nc: cols are nc*16 + (lane%4)*2, +1, +8, +9

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
