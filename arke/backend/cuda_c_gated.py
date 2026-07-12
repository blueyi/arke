# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA-C emitters for gated/fused elementwise ops (Phase 4, P4-S2).

Ops: silu_and_mul, gelu_and_mul, cast, where_
"""

from __future__ import annotations

from typing import Any

from arke.backend.cuda_c_backend import CudaCKernel, _ir_dtype_to_c
from arke.ir.graph import IRGraph


def _emit_gated(graph: IRGraph, chip: str, op_name: str,
                gate_expr: str) -> CudaCKernel:
    """Generic emitter for gated ops: out = gate(a) * b.

    gate_expr is the C expression for gate(a), e.g.:
      silu_and_mul: "a * (1.0f / (1.0f + expf(-a)))"
      gelu_and_mul: "a * 0.5f * (1.0f + tanhf(0.7978845608f * (a + 0.044715f * a*a*a)))"
    """
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    out_name = node.outputs[0]
    val = graph.get_value(a_name)
    shape = list(val.shape) if val.shape else [64, 64]
    M, N = shape
    dtype = val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    kernel_name = f"arke_{op_name}_{M}x{N}"
    BLOCK = 256
    total = M * N

    source = f"""\
#include <cuda_runtime.h>
#include <math.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ A,
    const {c_type}* __restrict__ B,
    {c_type}* __restrict__ Y,
    int total_elems)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_elems) {{
        {c_type} a = A[gid];
        {c_type} b = B[gid];
        {c_type} gate = {gate_expr};
        Y[gid] = gate * b;
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name=op_name,
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
                     ("int", total)],
    )


def emit_cuda_c_silu_and_mul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    return _emit_gated(graph, chip, "silu_and_mul",
                       "a * (1.0f / (1.0f + expf(-a)))")


def emit_cuda_c_gelu_and_mul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    return _emit_gated(graph, chip, "gelu_and_mul",
                       "a * 0.5f * (1.0f + tanhf(0.7978845608f * (a + 0.044715f * a*a*a)))")


def emit_cuda_c_where(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit CUDA C for where: out = cond ? a : b."""
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    cond_name, a_name, b_name = in_names[0], in_names[1], in_names[2]
    out_name = node.outputs[0]
    val = graph.get_value(a_name)
    shape = list(val.shape) if val.shape else [64, 64]
    M, N = shape
    dtype = val.dtype or "float32"
    c_type = _ir_dtype_to_c(dtype)
    kernel_name = f"arke_where_{M}x{N}"
    BLOCK = 256
    total = M * N

    source = f"""\
#include <cuda_runtime.h>
extern "C"
__global__ void {kernel_name}(
    const {c_type}* __restrict__ cond,
    const {c_type}* __restrict__ A,
    const {c_type}* __restrict__ B,
    {c_type}* __restrict__ Y,
    int total_elems)
{{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total_elems) {{
        Y[gid] = (cond[gid] != ({c_type})0) ? A[gid] : B[gid];
    }}
}}
"""
    return CudaCKernel(
        kernel_name=kernel_name, source=source, op_name="where_",
        param_names=[cond_name, a_name, b_name, out_name],
        output_name=out_name,
        shapes={cond_name: [M, N], a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={cond_name: dtype, a_name: dtype, b_name: dtype, out_name: dtype},
        grid=((total + BLOCK - 1) // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
        kernel_args=[("ptr", cond_name), ("ptr", a_name), ("ptr", b_name),
                     ("ptr", out_name), ("int", total)],
    )
