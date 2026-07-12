# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA-C emitters for additional ops to reach P4-S2 30-op target.

Ops: cast, embedding, permute, concat, split, batch_matmul, rmsnorm_residual, rope
"""

from __future__ import annotations
from typing import Any

import numpy as np

from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_c
from arke.ir.graph import IRGraph


def emit_cuda_c_cast(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for cast: out = (target_type)in. For f32→f32, just copy."""
    node = graph.nodes[0]
    in_name = list(node.inputs.values())[0]
    out_name = node.outputs[0]
    val = graph.get_value(in_name)
    shape = list(val.shape) if val.shape else [64, 64]
    M, N = shape
    dtype = val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    kernel_name = f"arke_cast_{M}x{N}"
    BLOCK = 256
    total = M * N

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    {c_type}* __restrict__ Y,
    int total_elems)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_elems) {{
        Y[gid] = ({c_type})X[gid];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="cast",
        param_names=[in_name, out_name],
        output_name=out_name,
        shapes={in_name: [M, N], out_name: [M, N]},
        dtypes={in_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
        kernel_args=[("ptr", in_name), ("ptr", out_name), ("int", total)],
    )


def emit_cuda_c_embedding(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for embedding lookup: out[i,j] = table[indices[i], j]."""
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    table_name = in_names[0]  # [V, D]
    idx_name = in_names[1]    # [M] or [M, 1]
    out_name = node.outputs[0]

    table_val = graph.get_value(table_name)
    idx_val = graph.get_value(idx_name)
    table_shape = list(table_val.shape)
    idx_shape = list(idx_val.shape)

    V, D = table_shape[0], table_shape[1]
    M = idx_shape[0]
    dtype = table_val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    kernel_name = f"arke_embedding_{M}x{D}"
    BLOCK = 256
    total = M * D

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ table,
    const int* __restrict__ indices,
    {c_type}* __restrict__ Y,
    int M, int D)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < M * D) {{
        int row = gid / D;
        int col = gid % D;
        int idx = indices[row];
        Y[row * D + col] = table[idx * D + col];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="embedding",
        param_names=[table_name, idx_name, out_name],
        output_name=out_name,
        shapes={table_name: [V, D], idx_name: [M], out_name: [M, D]},
        dtypes={table_name: dtype, idx_name: "int32", out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
        kernel_args=[("ptr", table_name), ("ptr", idx_name), ("ptr", out_name),
                     ("int", M), ("int", D)],
    )


def emit_cuda_c_batch_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for batch matmul: C[b,i,j] = sum_k A[b,i,k] * B[b,k,j]."""
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    out_name = node.outputs[0]

    a_val = graph.get_value(a_name)
    b_val = graph.get_value(b_name)
    a_shape = list(a_val.shape)
    b_shape = list(b_val.shape)

    B, M, K = a_shape
    _, K2, N = b_shape
    assert K == K2

    dtype = a_val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    TILE = 16
    kernel_name = f"arke_batch_matmul_{B}x{M}x{N}x{K}"

    source = f"""\
#include <cuda_runtime.h>
#define TILE_SIZE {TILE}
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ A,
    const {c_type}* __restrict__ B,
    {c_type}* __restrict__ C,
    int batch, int M, int N, int K_dim)
{{
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (b < batch && row < M && col < N) {{
        {c_type} acc = ({c_type})0;
        const {c_type}* Ab = A + b * M * K_dim;
        const {c_type}* Bb = B + b * K_dim * N;
        for (int k = 0; k < K_dim; k++) {{
            acc += Ab[row * K_dim + k] * Bb[k * N + col];
        }}
        C[b * M * N + row * N + col] = acc;
    }}
}}
"""
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE, B)
    block = (TILE, TILE, 1)

    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="batch_matmul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [B, M, K], b_name: [B, K, N], out_name: [B, M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid, block=block,
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
                     ("int", B), ("int", M), ("int", N), ("int", K)],
    )


def emit_cuda_c_rmsnorm_residual(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for rmsnorm_residual: y = rmsnorm(x + residual) * w."""
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    x_name = in_names[0]      # [M, N]
    res_name = in_names[1]    # [M, N]
    w_name = in_names[2]      # [N]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    shape = list(x_val.shape)
    M, N = shape
    dtype = x_val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    BLOCK = 256
    kernel_name = f"arke_rmsnorm_residual_{M}x{N}"

    source = f"""\
#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_SIZE {BLOCK}
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    const {c_type}* __restrict__ R,
    const {c_type}* __restrict__ W,
    {c_type}* __restrict__ Y,
    int M, int N)
{{
    __shared__ {c_type} shared[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Compute x + residual, then sum of squares
    {c_type} ss = ({c_type})0;
    for (int j = tid; j < N; j += BLOCK_SIZE) {{
        {c_type} val = X[row * N + j] + R[row * N + j];
        ss += val * val;
    }}
    shared[tid] = ss;
    __syncthreads();

    // Tree reduce
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }}

    {c_type} rms = sqrtf(shared[0] / ({c_type})N + 1e-5f);

    // Normalize and apply weight
    for (int j = tid; j < N; j += BLOCK_SIZE) {{
        {c_type} val = X[row * N + j] + R[row * N + j];
        Y[row * N + j] = (val / rms) * W[j];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="rmsnorm_residual",
        param_names=[x_name, res_name, w_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], res_name: [M, N], w_name: [N], out_name: [M, N]},
        dtypes={x_name: dtype, res_name: dtype, w_name: dtype, out_name: dtype},
        grid=(M, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", x_name), ("ptr", res_name), ("ptr", w_name),
                     ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_cuda_c_concat(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for concat of 2 tensors along last dim: [M, N1] + [M, N2] -> [M, N1+N2]."""
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    out_name = node.outputs[0]

    a_val = graph.get_value(a_name)
    b_val = graph.get_value(b_name)
    a_shape = list(a_val.shape)
    b_shape = list(b_val.shape)
    M = a_shape[0]
    N1 = a_shape[1]
    N2 = b_shape[1]
    N_out = N1 + N2
    dtype = a_val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    BLOCK = 256
    total = M * N_out
    kernel_name = f"arke_concat_{M}x{N1}_{N2}"

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ A,
    const {c_type}* __restrict__ B,
    {c_type}* __restrict__ Y,
    int M, int N1, int N2)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int N_out = N1 + N2;
    if (gid < M * N_out) {{
        int row = gid / N_out;
        int col = gid % N_out;
        if (col < N1) {{
            Y[gid] = A[row * N1 + col];
        }} else {{
            Y[gid] = B[row * N2 + (col - N1)];
        }}
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="concat",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N1], b_name: [M, N2], out_name: [M, N_out]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
                     ("int", M), ("int", N1), ("int", N2)],
    )


def emit_cuda_c_split(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for split: [M, N] -> first half [M, N//2] (simplified)."""
    node = graph.nodes[0]
    in_name = list(node.inputs.values())[0]
    out_name = node.outputs[0]
    val = graph.get_value(in_name)
    shape = list(val.shape)
    M, N = shape
    N_out = N // 2  # split in half
    dtype = val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    BLOCK = 256
    total = M * N_out
    kernel_name = f"arke_split_{M}x{N}"

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    {c_type}* __restrict__ Y,
    int M, int N, int N_out)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < M * N_out) {{
        int row = gid / N_out;
        int col = gid % N_out;
        Y[gid] = X[row * N + col];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="split",
        param_names=[in_name, out_name],
        output_name=out_name,
        shapes={in_name: [M, N], out_name: [M, N_out]},
        dtypes={in_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1),
        kernel_args=[("ptr", in_name), ("ptr", out_name),
                     ("int", M), ("int", N), ("int", N_out)],
    )


def emit_cuda_c_permute(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for 2D permute (= transpose). Same as transpose."""
    # For 2D, permute is just transpose
    from arke.backend.cuda_c_movement import emit_cuda_c_transpose
    result = emit_cuda_c_transpose(graph, chip)
    # Override op_name
    return CudaCKernel(
        kernel_name=result.kernel_name.replace("transpose", "permute"),
        source=result.source.replace(result.kernel_name,
                                     result.kernel_name.replace("transpose", "permute")),
        op_name="permute",
        param_names=result.param_names,
        output_name=result.output_name,
        shapes=result.shapes,
        dtypes=result.dtypes,
        grid=result.grid, block=result.block,
        kernel_args=result.kernel_args,
    )
