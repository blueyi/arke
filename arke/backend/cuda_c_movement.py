# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA-C emitters for data-movement ops (Phase 4, P4-S2).

Ops: transpose, copy_, concat, split, permute, embedding
"""

from __future__ import annotations

from typing import Any

from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_c
from arke.ir.graph import IRGraph


def emit_cuda_c_transpose(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for 2D transpose: out[j,i] = in[i,j]."""
    node = graph.nodes[0]
    in_name = list(node.inputs.values())[0]
    out_name = node.outputs[0]
    val = graph.get_value(in_name)
    shape = list(val.shape) if val.shape else [64, 64]
    M, N = shape
    dtype = val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    kernel_name = f"arke_transpose_{M}x{N}"
    BLOCK = 256
    total = M * N

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ X,
    {c_type}* __restrict__ Y,
    int M, int N)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < M * N) {{
        int i = gid / N;
        int j = gid % N;
        Y[j * M + i] = X[i * N + j];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="transpose",
        param_names=[in_name, out_name],
        output_name=out_name,
        shapes={in_name: [M, N], out_name: [N, M]},
        dtypes={in_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
        kernel_args=[("ptr", in_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_cuda_c_copy(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for copy: out = in (element-wise copy)."""
    node = graph.nodes[0]
    in_name = list(node.inputs.values())[0]
    out_name = node.outputs[0]
    val = graph.get_value(in_name)
    shape = list(val.shape) if val.shape else [64, 64]
    M, N = shape
    dtype = val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    kernel_name = f"arke_copy_{M}x{N}"
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
        Y[gid] = X[gid];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="copy_",
        param_names=[in_name, out_name],
        output_name=out_name,
        shapes={in_name: [M, N], out_name: [M, N]},
        dtypes={in_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
        kernel_args=[("ptr", in_name), ("ptr", out_name), ("int", total)],
    )
