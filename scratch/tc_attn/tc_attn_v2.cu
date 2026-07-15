// TC flash-attention v2 — LOW-SMEM, safe aliasing.
//
// Smem budget: ~48.6 KB → 2 blocks × 48.6 KB = 97.2 KB < 100 KB/SM ✓
//
// Layout (no unsafe overlaps):
//   Qsh:   [BR=64][D=64]   fp16 = 8192 B  (persistent)
//   KSsh:  [BR=64][BC=64]  fp32 = 16384 B (K^T uses first 8K as fp16, then S uses full 16K as fp32)
//   Psh:   [BR=64][BC=64]  fp16 = 8192 B  (independent — no aliasing with S!)
//   Osh:   [BR=64][D=64]   fp32 = 16384 B (running O accumulator)
//   m_sh:  [BR]            fp32 = 256 B
//   l_sh:  [BR]            fp32 = 256 B
//   Total: 49664 B = 48.5 KB
//
// Phases per tile:
//   P1: Load K^T into KSsh (first 8K as [D][BC] fp16)
//   P2: QK^T → S [BR][BC] fp32 into KSsh (full 16K, overwrites K)
//   P3: Softmax: reads S from KSsh, writes P fp16 to Psh (SAFE — separate buffers!)
//   P4: Load V into KSsh (reuse as [BC][D] fp16, only first 8K needed)
//   P5: P.V TC: reads P from Psh, V from KSsh, accumulates into Osh
//
// P5 uses load-mma-store trick: load Osh tile into accumulator, mma P.V into it, store back.
// Eliminates Otile buffer entirely.

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

extern "C" __global__ void tc_flash_attn_v2(
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

    // Shared memory allocation
    extern __shared__ char smem[];
    half*  Qsh_   = (half*)  smem;                          // [BR][D] fp16, 8192 B
    float* KSsh_  = (float*)(Qsh_ + BR * D);                // [BR][BC] fp32, 16384 B (multi-use)
    half*  Psh_   = (half*) (KSsh_ + BR * BC);              // [BR][BC] fp16, 8192 B
    float* Osh_   = (float*)(Psh_ + BR * BC);               // [BR][D] fp32, 16384 B
    float* m_sh   = Osh_ + BR * D;                          // [BR] fp32, 256 B
    float* l_sh   = m_sh + BR;                              // [BR] fp32, 256 B

    // KSsh_ viewed as fp16 for K^T and V loading (first 8K = 4096 halfs)
    half*  KTsh_  = (half*) KSsh_;                          // [D][BC] fp16 (D*BC=4096 halfs = 8192B)
    half*  Vsh_   = (half*) KSsh_;                          // [BC][D] fp16 (BC*D=4096 halfs = 8192B)

    #define Qsh(r,c)   Qsh_[(r)*D + (c)]
    #define KTsh(r,c)  KTsh_[(r)*BC + (c)]      // K transposed: [D][BC]
    #define Ssh(r,c)   KSsh_[(r)*BC + (c)]      // S: [BR][BC] fp32
    #define Psh(r,c)   Psh_[(r)*BC + (c)]       // P: [BR][BC] fp16
    #define Vsh(r,c)   Vsh_[(r)*D + (c)]        // V: [BC][D] fp16
    #define Osh(r,c)   Osh_[(r)*D + (c)]

    // ---- Init: Load Q, zero Osh/m/l ----
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh(r, c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
        Osh(r, c) = 0.0f;
    }
    for (int r = tid; r < BR; r += blockDim.x) {
        m_sh[r] = -FLT_MAX;
        l_sh[r] = 0.0f;
    }
    __syncthreads();

    // ---- Main loop over key tiles ----
    for (int kt = 0; kt < S; kt += BC) {
        int krem = min(S - kt, BC);

        // ======== P1: Load K^T into KTsh (first 8K of KSsh area) ========
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            int kr = kt + r;
            half kv = (kr < S) ? Kbh[kr * D + c] : __float2half(0.0f);
            KTsh(c, r) = kv;  // transpose: [D][BC]
        }
        __syncthreads();

        // ======== P2: S = Q . K^T via Tensor Cores ========
        // IMPORTANT: Store S into KSsh which aliases KTsh. All warps must finish
        // reading KTsh BEFORE any warp stores to KSsh. Solution: accumulate in
        // register fragments first, sync, then store.
        {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[NCOL];
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
            // All warps done reading KTsh — safe to overwrite with S
            __syncthreads();
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                wmma::store_matrix_sync(&Ssh(warp * WMMA_M, nc * WMMA_N), s_frag[nc], BC,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ======== P3: Online softmax — reads S (KSsh), writes P (Psh) ========
        // Psh is a SEPARATE buffer — no aliasing hazard!
        {
            for (int r = warp * WMMA_M; r < warp * WMMA_M + WMMA_M; r++) {
                float m_prev = m_sh[r];

                // Row max
                float lmax = -FLT_MAX;
                for (int j = lane; j < BC; j += 32) {
                    float s = (j < krem) ? Ssh(r, j) * scale : -FLT_MAX;
                    if (s > lmax) lmax = s;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));

                float m_cur = fmaxf(m_prev, lmax);

                // Exp + sum + write P
                float lsum = 0.0f;
                for (int j = lane; j < BC; j += 32) {
                    float p = (j < krem) ? expf(Ssh(r, j) * scale - m_cur) : 0.0f;
                    Psh(r, j) = __float2half(p);  // safe: Psh is separate from Ssh!
                    lsum += p;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    lsum += __shfl_xor_sync(0xffffffff, lsum, off);

                // Rescale running O (skip if m unchanged for perf)
                float corr = expf(m_prev - m_cur);
                if (corr != 1.0f) {
                    for (int c = lane; c < D; c += 32)
                        Osh(r, c) *= corr;
                }

                if (lane == 0) {
                    m_sh[r] = m_cur;
                    l_sh[r] = l_sh[r] * corr + lsum;
                }
            }
        }
        __syncthreads();

        // ======== P4: Load V into Vsh (reuse KSsh area as fp16) ========
        // S is fully consumed, KSsh is free for V
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;
            int kr = kt + r;
            half vv = (kr < S) ? Vbh[kr * D + c] : __float2half(0.0f);
            Vsh(r, c) = vv;   // [BC][D] fp16
        }
        __syncthreads();

        // ======== P5: O += P . V via Tensor Cores ========
        // Load current Osh tile into accumulator, mma P.V into it, store back.
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                // Load current Osh[warp*16, nc*16] into accumulator
                wmma::load_matrix_sync(c_frag, &Osh(warp * WMMA_M, nc * WMMA_N), D,
                                       wmma::mem_row_major);
                // Accumulate P.V into it
                #pragma unroll
                for (int k = 0; k < BC; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Psh(warp * WMMA_M, k), BC);
                    wmma::load_matrix_sync(b_frag, &Vsh(k, nc * WMMA_N), D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                // Store back to Osh
                wmma::store_matrix_sync(&Osh(warp * WMMA_M, nc * WMMA_N), c_frag, D,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: O = Osh / l ----
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        if (qr < S) {
            Obh[qr * D + c] = __float2half(Osh(r, c) / l_sh[r]);
        }
    }
}
