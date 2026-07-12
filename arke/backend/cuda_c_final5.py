# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA-C emitters for the final 5 ops → full 46/46 catalog coverage.

Ops: grouped_matmul, fused_linear_cross_entropy, quantize_per_token,
multi_latent_attention, paged_attention. All FP32 correctness-first.
"""

from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_c
from arke.ir.graph import IRGraph


def emit_cuda_c_grouped_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Grouped/batched matmul with per-batch expert selection.

    X[B,M,K], W[E,K,N], indices[B] (int32) → out[B,M,N].
    out[b] = X[b] @ W[indices[b]]. One thread per (b, m, n) output element.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    x_name, w_name, idx_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]

    x_shape = list(graph.get_value(x_name).shape)
    w_shape = list(graph.get_value(w_name).shape)
    B, M, K = x_shape
    E, K2, N = w_shape
    dtype = graph.get_value(x_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    BLOCK = 256
    total = B * M * N
    kernel_name = f"arke_grouped_matmul_{B}x{M}x{K}x{N}"

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    const {c_type}* __restrict__ W,
    const int* __restrict__ indices,
    {c_type}* __restrict__ Out,
    int B, int M, int K, int N)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= B * M * N) return;
    int b = gid / (M * N);
    int rem = gid % (M * N);
    int m = rem / N;
    int n = rem % N;
    int e = indices[b];
    const {c_type}* Xb = X + (long)b * M * K;
    const {c_type}* We = W + (long)e * K * N;
    {c_type} acc = 0.0f;
    for (int k = 0; k < K; k++) acc += Xb[m * K + k] * We[k * N + n];
    Out[(long)b * M * N + m * N + n] = acc;
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="grouped_matmul",
        param_names=[x_name, w_name, idx_name, out_name], output_name=out_name,
        shapes={x_name: [B, M, K], w_name: [E, K, N], idx_name: [B], out_name: [B, M, N]},
        dtypes={x_name: dtype, w_name: dtype, idx_name: "int32", out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", x_name), ("ptr", w_name), ("ptr", idx_name), ("ptr", out_name),
                     ("int", B), ("int", M), ("int", K), ("int", N)],
    )


def emit_cuda_c_quantize_per_token(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Per-token (per-row) symmetric int8 quantization.

    X[M,N] → Xq[M,N] (int8). scale[m] = max(|X[m,:]|)/127; Xq = round(X/scale).
    NOTE: single-output emitter — writes only the quantized int8 tensor (the
    per-row scale is recomputable / a secondary output not modeled here).
    One block per row, block=256, shared-mem max-abs reduce.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    x_name = in_names[0]
    out_name = node.outputs[0]
    shape = list(graph.get_value(x_name).shape)
    M, N = shape
    dtype = graph.get_value(x_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    BLOCK = 256
    kernel_name = f"arke_quantize_per_token_{M}x{N}"

    source = f"""\
#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_SIZE {BLOCK}
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    signed char* __restrict__ Xq,
    int M, int N)
{{
    __shared__ {c_type} sdata[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    {c_type} maxabs = 0.0f;
    for (int j = tid; j < N; j += BLOCK_SIZE) {{
        {c_type} a = fabsf(X[row * N + j]);
        if (a > maxabs) maxabs = a;
    }}
    sdata[tid] = maxabs;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {{
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }}
    {c_type} scale = sdata[0] / 127.0f;
    if (scale == 0.0f) scale = 1.0f;
    for (int j = tid; j < N; j += BLOCK_SIZE) {{
        {c_type} q = rintf(X[row * N + j] / scale);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        Xq[row * N + j] = (signed char)q;
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="quantize_per_token",
        param_names=[x_name, out_name], output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: "int8"},
        grid=(M, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_cuda_c_fused_linear_cross_entropy(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Fused linear + cross-entropy: logits = X@W^T, loss = CE(logits, labels).

    X[B,D], W[V,D], labels[B] (int32) → loss[B] (per-sample, mean externally).
    One block per sample (row). Computes logits on the fly (no V-sized buffer),
    online logsumexp, picks label logit. block=256.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    x_name, w_name, lbl_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]

    x_shape = list(graph.get_value(x_name).shape)
    w_shape = list(graph.get_value(w_name).shape)
    B, D = x_shape
    V, D2 = w_shape
    dtype = graph.get_value(x_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    BLOCK = 256
    kernel_name = f"arke_flce_{B}x{D}x{V}"

    source = f"""\
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define BLOCK_SIZE {BLOCK}
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    const {c_type}* __restrict__ W,
    const int* __restrict__ labels,
    {c_type}* __restrict__ loss,
    int B, int D, int V)
{{
    __shared__ {c_type} sdata[BLOCK_SIZE];
    int b = blockIdx.x;
    int tid = threadIdx.x;
    const {c_type}* Xb = X + (long)b * D;

    // Pass 1: max logit over vocab (each thread strides vocab rows).
    {c_type} mx = -FLT_MAX;
    for (int v = tid; v < V; v += BLOCK_SIZE) {{
        const {c_type}* Wv = W + (long)v * D;
        {c_type} logit = 0.0f;
        for (int d = 0; d < D; d++) logit += Xb[d] * Wv[d];
        if (logit > mx) mx = logit;
    }}
    sdata[tid] = mx; __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {{
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }}
    {c_type} row_max = sdata[0]; __syncthreads();

    // Pass 2: sum exp(logit - max).
    {c_type} sum_exp = 0.0f;
    for (int v = tid; v < V; v += BLOCK_SIZE) {{
        const {c_type}* Wv = W + (long)v * D;
        {c_type} logit = 0.0f;
        for (int d = 0; d < D; d++) logit += Xb[d] * Wv[d];
        sum_exp += expf(logit - row_max);
    }}
    sdata[tid] = sum_exp; __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {{
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }}
    {c_type} lse = logf(sdata[0]) + row_max;

    if (tid == 0) {{
        int lbl = labels[b];
        const {c_type}* Wl = W + (long)lbl * D;
        {c_type} label_logit = 0.0f;
        for (int d = 0; d < D; d++) label_logit += Xb[d] * Wl[d];
        loss[b] = lse - label_logit;
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="fused_linear_cross_entropy",
        param_names=[x_name, w_name, lbl_name, out_name], output_name=out_name,
        shapes={x_name: [B, D], w_name: [V, D], lbl_name: [B], out_name: [B]},
        dtypes={x_name: dtype, w_name: dtype, lbl_name: "int32", out_name: dtype},
        grid=(B, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", x_name), ("ptr", w_name), ("ptr", lbl_name), ("ptr", out_name),
                     ("int", B), ("int", D), ("int", V)],
    )


def emit_cuda_c_paged_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Paged attention (decode, single query step) with a block table.

    Q[B,H,1,D], K_cache/V_cache[num_blocks,block_size,H,D], block_table[B,max_blocks]
    → O[B,H,1,D]. One warp per (b,h): online softmax over gathered KV pages.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    q_name, kc_name, vc_name, bt_name = in_names[0], in_names[1], in_names[2], in_names[3]
    out_name = node.outputs[0]

    q_shape = list(graph.get_value(q_name).shape)
    kc_shape = list(graph.get_value(kc_name).shape)
    bt_shape = list(graph.get_value(bt_name).shape)
    B, H, _one, D = q_shape
    num_blocks, block_size, Hc, Dc = kc_shape
    max_blocks = bt_shape[1]
    dtype = graph.get_value(q_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)

    import math
    scale = 1.0 / math.sqrt(D)
    DPL = (D + 31) // 32
    kernel_name = f"arke_paged_attn_{B}x{H}x{D}_bs{block_size}"

    source = f"""\
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define HEAD_DIM {D}
#define DPL {DPL}
#define BLOCK_SIZE {block_size}
#define MAX_BLOCKS {max_blocks}

extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ Q,
    const {c_type}* __restrict__ Kc,
    const {c_type}* __restrict__ Vc,
    const int* __restrict__ block_table,
    {c_type}* __restrict__ O,
    int B, int H, int D, int num_blocks, int block_size, int max_blocks, float scale)
{{
    // One warp per (b, h). Grid.x = B*H, block = 32 threads.
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int lane = threadIdx.x & 31;

    const {c_type}* Qbh = Q + ((long)(b * H + h)) * D;   // Q[b,h,0,:]
    {c_type}* Obh = O + ((long)(b * H + h)) * D;
    const int* bt = block_table + (long)b * max_blocks;

    {c_type} q_reg[DPL], acc[DPL];
    #pragma unroll
    for (int t = 0; t < DPL; t++) {{
        int d = lane + 32 * t;
        q_reg[t] = (d < D) ? Qbh[d] : 0.0f;
        acc[t] = 0.0f;
    }}
    {c_type} m_i = -FLT_MAX, l_i = 0.0f;

    for (int blk = 0; blk < max_blocks; blk++) {{
        int phys = bt[blk];
        if (phys < 0) break;   // -1 sentinel = end of sequence
        for (int s = 0; s < block_size; s++) {{
            // K_cache[phys, s, h, :]
            const {c_type}* Kp = Kc + (((long)phys * block_size + s) * H + h) * D;
            const {c_type}* Vp = Vc + (((long)phys * block_size + s) * H + h) * D;
            {c_type} sp = 0.0f;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) sp += q_reg[t] * Kp[d]; }}
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) sp += __shfl_down_sync(0xffffffff, sp, off);
            {c_type} score = __shfl_sync(0xffffffff, sp, 0) * scale;
            {c_type} m_new = fmaxf(m_i, score);
            {c_type} corr = expf(m_i - m_new);
            {c_type} p = expf(score - m_new);
            l_i = l_i * corr + p;
            #pragma unroll
            for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) acc[t] = acc[t] * corr + p * Vp[d]; }}
            m_i = m_new;
        }}
    }}
    {c_type} inv_l = 1.0f / l_i;
    #pragma unroll
    for (int t = 0; t < DPL; t++) {{ int d = lane + 32 * t; if (d < D) Obh[d] = acc[t] * inv_l; }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="paged_attention",
        param_names=[q_name, kc_name, vc_name, bt_name, out_name], output_name=out_name,
        shapes={q_name: [B, H, 1, D], kc_name: [num_blocks, block_size, H, D],
                vc_name: [num_blocks, block_size, H, D], bt_name: [B, max_blocks],
                out_name: [B, H, 1, D]},
        dtypes={q_name: dtype, kc_name: dtype, vc_name: dtype,
                bt_name: "int32", out_name: dtype},
        grid=(B * H, 1, 1), block=(32, 1, 1),
        kernel_args=[("ptr", q_name), ("ptr", kc_name), ("ptr", vc_name),
                     ("ptr", bt_name), ("ptr", out_name),
                     ("int", B), ("int", H), ("int", D), ("int", num_blocks),
                     ("int", block_size), ("int", max_blocks), ("float", scale)],
    )


def emit_cuda_c_mla(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Multi-latent attention (DeepSeek-V2 style), correctness-first.

    Q[B,H,S,D], KV_compressed[B,S,Dc], W_uk[Dc,H,D], W_uv[Dc,H,D] → O[B,H,S,D].
    Up-projects the compressed latent to per-head K,V then does standard
    attention. One block per (b,h,query-tile); K,V reconstructed on the fly.
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    q_name, kv_name, wuk_name, wuv_name = in_names[0], in_names[1], in_names[2], in_names[3]
    out_name = node.outputs[0]

    q_shape = list(graph.get_value(q_name).shape)
    kv_shape = list(graph.get_value(kv_name).shape)
    B, H, S, D = q_shape
    Dc = kv_shape[2]
    dtype = graph.get_value(q_name).dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)

    import math
    scale = 1.0 / math.sqrt(D)
    kernel_name = f"arke_mla_{B}x{H}x{S}x{D}x{Dc}"

    # One thread per query row (simple, correctness-first — MLA up-projection is
    # the novel part; attention is standard). Thread reconstructs K_j, V_j from
    # the compressed latent for each key j via W_uk/W_uv.
    source = f"""\
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define HEAD_DIM {D}
#define DC {Dc}
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ Q,
    const {c_type}* __restrict__ KVc,
    const {c_type}* __restrict__ W_uk,
    const {c_type}* __restrict__ W_uv,
    {c_type}* __restrict__ O,
    int B, int H, int S, int D, int Dc, float scale)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * S) return;
    int s_q = idx % S;
    int h = (idx / S) % H;
    int b = idx / (S * H);

    const {c_type}* Qr = Q + (((long)(b * H + h)) * S + s_q) * D;
    const {c_type}* KVb = KVc + ((long)b * S) * Dc;
    {c_type}* Or = O + (((long)(b * H + h)) * S + s_q) * D;

    {c_type} q_reg[HEAD_DIM];
    for (int d = 0; d < D; d++) q_reg[d] = Qr[d];

    {c_type} acc[HEAD_DIM];
    for (int d = 0; d < D; d++) acc[d] = 0.0f;
    {c_type} m_i = -FLT_MAX, l_i = 0.0f;

    for (int j = 0; j < S; j++) {{
        const {c_type}* lat = KVb + (long)j * Dc;
        // Reconstruct K_j[d] = sum_c lat[c] * W_uk[c,h,d]; V_j similarly.
        {c_type} score = 0.0f;
        for (int d = 0; d < D; d++) {{
            {c_type} kjd = 0.0f;
            for (int c = 0; c < Dc; c++)
                kjd += lat[c] * W_uk[((long)c * H + h) * D + d];
            score += q_reg[d] * kjd;
        }}
        score *= scale;
        {c_type} m_new = fmaxf(m_i, score);
        {c_type} corr = expf(m_i - m_new);
        {c_type} p = expf(score - m_new);
        l_i = l_i * corr + p;
        for (int d = 0; d < D; d++) {{
            {c_type} vjd = 0.0f;
            for (int c = 0; c < Dc; c++)
                vjd += lat[c] * W_uv[((long)c * H + h) * D + d];
            acc[d] = acc[d] * corr + p * vjd;
        }}
        m_i = m_new;
    }}
    {c_type} inv_l = 1.0f / l_i;
    for (int d = 0; d < D; d++) Or[d] = acc[d] * inv_l;
}}
"""
    BLOCK = 128
    total = B * H * S
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="multi_latent_attention",
        param_names=[q_name, kv_name, wuk_name, wuv_name, out_name], output_name=out_name,
        shapes={q_name: [B, H, S, D], kv_name: [B, S, Dc],
                wuk_name: [Dc, H, D], wuv_name: [Dc, H, D], out_name: [B, H, S, D]},
        dtypes={q_name: dtype, kv_name: dtype, wuk_name: dtype,
                wuv_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", q_name), ("ptr", kv_name), ("ptr", wuk_name),
                     ("ptr", wuv_name), ("ptr", out_name),
                     ("int", B), ("int", H), ("int", S), ("int", D), ("int", Dc),
                     ("float", scale)],
    )
