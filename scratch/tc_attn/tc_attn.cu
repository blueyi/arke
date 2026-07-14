// TC flash-attention prototype (Arke C2, Phase 5).
// fp16 -> fp32 fused attention using Tensor Cores (wmma 16x16x16) for both
// Q.K^T and P.V matmuls, with fp32 accumulation + fp32 online-softmax between.
//
// Config: head_dim D = 64, one head per grid.y. Query tile BR = 64 rows,
// key tile BC = 64. Block = 4 warps (128 threads); warp w owns query
// row-tile w (rows 16w..16w+16) throughout — Q.K^T, softmax, and P.V.
// Larger BR amortizes each K/V tile load across 64 queries (vs 16 before).
//
// Layout: Q,K,V,O are [B,H,S,D] row-major fp16. Softmax scale passed in.
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#define D        64
#define BR       64
#define BC       64
#define NWARPS   4      // BR / 16 : one row-tile per warp
#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define NCOL     (BC / WMMA_N)   // 4 column tiles of S
#define NDCOL    (D  / WMMA_N)   // 4 column tiles of D

extern "C" __global__ void tc_flash_attn(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int S, float scale)
{
    const int bh    = blockIdx.y;              // batch*head
    const int qtile = blockIdx.x;              // which BR-row query tile
    const int warp  = threadIdx.x >> 5;        // 0..3, == row-tile index
    const int lane  = threadIdx.x & 31;
    const int tid   = threadIdx.x;             // 0..127
    const int q0    = qtile * BR;              // first query row (global)

    const half* Qbh = Q + (long)bh * S * D;
    const half* Kbh = K + (long)bh * S * D;
    const half* Vbh = V + (long)bh * S * D;
    half*       Obh = O + (long)bh * S * D;

    // Dynamic shared memory (needs > 48KB, opt-in via cudaFuncAttribute).
    // Layout (byte offsets), fp16 blocks first then fp32 (16B aligned):
    extern __shared__ char smem[];
    half*  Qsh_  = (half*)  smem;                          // [BR][D]
    half*  Ktsh_ = Qsh_  + BR * D;                         // [D][BC]
    half*  Vsh_  = Ktsh_ + D  * BC;                        // [BC][D]
    half*  Psh_  = Vsh_  + BC * D;                         // [BR][BC]
    float* Ssh_  = (float*)(Psh_ + BR * BC);               // [BR][BC]
    float* Osh_  = Ssh_  + BR * BC;                        // [BR][D]
    float* Otile_= Osh_  + BR * D;                         // [BR][D]
    float* m_sh  = Otile_+ BR * D;                         // [BR]
    float* l_sh  = m_sh  + BR;                             // [BR]
    #define Qsh(r,c)   Qsh_[(r)*D + (c)]
    #define Ktsh(r,c)  Ktsh_[(r)*BC + (c)]
    #define Vsh(r,c)   Vsh_[(r)*D + (c)]
    #define Psh(r,c)   Psh_[(r)*BC + (c)]
    #define Ssh(r,c)   Ssh_[(r)*BC + (c)]
    #define Osh(r,c)   Osh_[(r)*D + (c)]
    #define Otile(r,c) Otile_[(r)*D + (c)]

    // ---- load Q tile once, init accumulators ----
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        Qsh(r,c) = (qr < S) ? Qbh[qr * D + c] : __float2half(0.0f);
        Osh(r,c) = 0.0f;
    }
    for (int r = tid; r < BR; r += blockDim.x) { m_sh[r] = -FLT_MAX; l_sh[r] = 0.0f; }
    __syncthreads();

    // ---- loop over key tiles (FA-2 online softmax) ----
    for (int kt = 0; kt < S; kt += BC) {
        // load K (transposed) and V tiles into shared
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D, c = idx % D;      // r=key-within-tile, c=head-dim
            int kr = kt + r;
            half kv, vv;
            if (kr < S) { kv = Kbh[kr * D + c]; vv = Vbh[kr * D + c]; }
            else        { kv = __float2half(0.0f); vv = __float2half(0.0f); }
            Ktsh(c,r) = kv;                   // transpose on store
            Vsh(r,c)  = vv;
        }
        __syncthreads();

        // ---- S = Q . K^T  (Tensor Core, fp32 accum) ----
        // warp owns row-tile `warp`; iterate the NCOL column tiles.
        {
            #pragma unroll
            for (int nc = 0; nc < NCOL; nc++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int k = 0; k < D; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Qsh(warp * WMMA_M, k), D);
                    wmma::load_matrix_sync(b_frag, &Ktsh(k, nc * WMMA_N), BC);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(&Ssh(warp * WMMA_M, nc * WMMA_N), c_frag, BC, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- online softmax over this tile + rescale O accumulator ----
        // each warp handles its 16 rows; 32 lanes split BC/D.
        {
            int krem = S - kt;
            if (krem > BC) krem = BC;
            for (int r = warp * WMMA_M; r < warp * WMMA_M + WMMA_M; r++) {
                float m_prev = m_sh[r];
                float lmax = -FLT_MAX;
                for (int j = lane; j < BC; j += 32) {
                    float s = (j < krem) ? Ssh(r,j) * scale : -FLT_MAX;
                    Ssh(r,j) = s;
                    if (s > lmax) lmax = s;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));
                float m_cur = fmaxf(m_prev, lmax);
                float lsum = 0.0f;
                for (int j = lane; j < BC; j += 32) {
                    float p = (j < krem) ? expf(Ssh(r,j) - m_cur) : 0.0f;
                    Psh(r,j) = __float2half(p);
                    lsum += p;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    lsum += __shfl_xor_sync(0xffffffff, lsum, off);
                float corr = expf(m_prev - m_cur);
                for (int c = lane; c < D; c += 32) Osh(r,c) *= corr;
                if (lane == 0) { m_sh[r] = m_cur; l_sh[r] = l_sh[r] * corr + lsum; }
            }
        }
        __syncthreads();

        // ---- O += P . V  (Tensor Core, fp32 accum) ----
        // warp owns row-tile `warp`; iterate NDCOL D-column tiles.
        {
            #pragma unroll
            for (int nc = 0; nc < NDCOL; nc++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int k = 0; k < BC; k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, &Psh(warp * WMMA_M, k), BC);
                    wmma::load_matrix_sync(b_frag, &Vsh(k, nc * WMMA_N), D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(&Otile(warp * WMMA_M, nc * WMMA_N), c_frag, D, wmma::mem_row_major);
            }
            __syncthreads();
            for (int idx = tid; idx < BR * D; idx += blockDim.x) {
                int r = idx / D, c = idx % D;
                Osh(r,c) += Otile(r,c);
            }
        }
        __syncthreads();
    }

    // ---- epilogue: O = Osh / l, write out ----
    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D, c = idx % D;
        int qr = q0 + r;
        if (qr < S) {
            float inv = 1.0f / l_sh[r];
            Obh[qr * D + c] = __float2half(Osh(r,c) * inv);
        }
    }
}
