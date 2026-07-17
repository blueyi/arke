# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR emitters for dense/data-movement ops (Phase 5, P5-S2).

Generates LLVM IR text targeting nvptx64-nvidia-cuda for 10 dense and
data-movement operations:
  batch_matmul, grouped_matmul, concat, copy_, embedding,
  gather, scatter, permute, split, transpose

Each emitter produces a CudaCKernel with LLVM IR source text, grid/block
dimensions, and kernel argument metadata.
"""

from __future__ import annotations

import math

from arke.backend.cuda_c_backend import CudaCKernel
from arke.ir.graph import IRGraph


# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────

_LLVM_HEADER = """\
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
"""

_NVVM_INTRINSICS = """\
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare void @llvm.nvvm.barrier0()
"""


def _annotation(func_sig: str, kernel_name: str) -> str:
    """Generate !nvvm.annotations metadata."""
    return (
        f"!nvvm.annotations = !{{!0}}\n"
        f"!0 = !{{{func_sig}* @{kernel_name}, !\"kernel\", i32 1}}\n"
    )


def _flat_grid(total: int, block_size: int = 256):
    """Compute flat 1D grid for total elements."""
    grid_x = (total + block_size - 1) // block_size
    return (grid_x, 1, 1), (block_size, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. batch_matmul: A[B,M,K] @ B_mat[B,K,N] -> C[B,M,N]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_batch_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit batch_matmul kernel: C[B,M,N] = A[B,M,K] @ B_mat[B,K,N].

    Flat 1D grid: 1 thread per output element in B*M*N space.
    Each thread computes one output via inner loop over K.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    a_name, b_name = input_names[0], input_names[1]
    out_name = node.outputs[0]

    a_val = graph.get_value(a_name)
    b_val = graph.get_value(b_name)
    a_shape = list(a_val.shape) if a_val.shape else [2, 64, 64]
    b_shape = list(b_val.shape) if b_val.shape else [2, 64, 64]

    B, M, K = a_shape[0], a_shape[1], a_shape[2]
    N = b_shape[2]
    dtype = a_val.dtype or "float32"
    total = B * M * N

    kernel_name = f"arke_batch_matmul_{B}x{M}x{K}x{N}"
    grid, block = _flat_grid(total)

    source = _gen_batch_matmul_ir(kernel_name, B, M, K, N)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="batch_matmul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: a_shape, b_name: b_shape, out_name: [B, M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
            ("int", B), ("int", M), ("int", K), ("int", N),
        ],
    )


def _gen_batch_matmul_ir(kernel_name: str, B: int, M: int, K: int, N: int) -> str:
    total = B * M * N
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B_mat, float addrspace(1)* %C, i32 %B_size, i32 %M, i32 %K, i32 %N) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid -> (b, m, n): gid = b*M*N + m*N + n
    ln(f"  %mn = add i32 0, {M * N}")
    ln("  %b_idx = sdiv i32 %gid, %mn")
    ln("  %rem_mn = srem i32 %gid, %mn")
    ln(f"  %n_size = add i32 0, {N}")
    ln("  %m_idx = sdiv i32 %rem_mn, %n_size")
    ln("  %n_idx = srem i32 %rem_mn, %n_size")
    # Base offsets: A[b,m,:] at b*M*K + m*K, B_mat[b,:,n] at b*K*N + k*N + n
    ln(f"  %mk = add i32 0, {M * K}")
    ln("  %a_batch_off = mul i32 %b_idx, %mk")
    ln(f"  %k_val = add i32 0, {K}")
    ln("  %a_row_off = mul i32 %m_idx, %k_val")
    ln("  %a_base = add i32 %a_batch_off, %a_row_off")
    ln(f"  %kn = add i32 0, {K * N}")
    ln("  %b_batch_off = mul i32 %b_idx, %kn")
    ln("  br label %k_loop_header")
    ln("")
    # K-loop
    ln("k_loop_header:")
    ln("  %k = phi i32 [0, %compute], [%k_next, %k_loop_body]")
    ln("  %acc = phi float [0.0, %compute], [%acc_new, %k_loop_body]")
    ln(f"  %k_done = icmp sge i32 %k, {K}")
    ln("  br i1 %k_done, label %k_loop_exit, label %k_loop_body")
    ln("")
    ln("k_loop_body:")
    # Load A[b,m,k]
    ln("  %a_off = add i32 %a_base, %k")
    ln("  %a_ptr = getelementptr float, float addrspace(1)* %A, i32 %a_off")
    ln("  %a_val = load float, float addrspace(1)* %a_ptr")
    # Load B_mat[b,k,n] = b_batch_off + k*N + n
    ln("  %b_k_off = mul i32 %k, %n_size")
    ln("  %b_off_1 = add i32 %b_batch_off, %b_k_off")
    ln("  %b_off = add i32 %b_off_1, %n_idx")
    ln("  %b_ptr = getelementptr float, float addrspace(1)* %B_mat, i32 %b_off")
    ln("  %b_val = load float, float addrspace(1)* %b_ptr")
    ln("  %prod = fmul float %a_val, %b_val")
    ln("  %acc_new = fadd float %acc, %prod")
    ln("  %k_next = add i32 %k, 1")
    ln("  br label %k_loop_header")
    ln("")
    ln("k_loop_exit:")
    # Store C[b,m,n] = b*M*N + m*N + n = gid
    ln("  %c_ptr = getelementptr float, float addrspace(1)* %C, i32 %gid")
    ln("  store float %acc, float addrspace(1)* %c_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. grouped_matmul: X[B,M,K], W[E,K,N], indices[B] -> out[B,M,N]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_grouped_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit grouped_matmul: out[b] = X[b] @ W[indices[b]].

    X[B,M,K], W[E,K,N], indices[B] (int32) -> out[B,M,N].
    Flat 1D grid, 1 thread per output element.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name, w_name, idx_name = input_names[0], input_names[1], input_names[2]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    w_val = graph.get_value(w_name)
    x_shape = list(x_val.shape) if x_val.shape else [4, 64, 64]
    w_shape = list(w_val.shape) if w_val.shape else [8, 64, 64]

    B, M, K = x_shape[0], x_shape[1], x_shape[2]
    E, _, N = w_shape[0], w_shape[1], w_shape[2]
    dtype = x_val.dtype or "float32"
    total = B * M * N

    kernel_name = f"arke_grouped_matmul_{B}x{M}x{K}x{N}"
    grid, block = _flat_grid(total)

    source = _gen_grouped_matmul_ir(kernel_name, B, M, K, N, E)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="grouped_matmul",
        param_names=[x_name, w_name, idx_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, w_name: w_shape, idx_name: [B], out_name: [B, M, N]},
        dtypes={x_name: dtype, w_name: dtype, idx_name: "int32", out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", idx_name), ("ptr", out_name),
            ("int", B), ("int", M), ("int", K), ("int", N),
        ],
    )


def _gen_grouped_matmul_ir(kernel_name: str, B: int, M: int, K: int, N: int, E: int) -> str:
    total = B * M * N
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, i32 addrspace(1)* %Idx, float addrspace(1)* %Out, i32 %B_size, i32 %M, i32 %K, i32 %N) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid -> (b, m, n)
    ln(f"  %mn = add i32 0, {M * N}")
    ln("  %b_idx = sdiv i32 %gid, %mn")
    ln("  %rem_mn = srem i32 %gid, %mn")
    ln(f"  %n_size = add i32 0, {N}")
    ln("  %m_idx = sdiv i32 %rem_mn, %n_size")
    ln("  %n_idx = srem i32 %rem_mn, %n_size")
    # Load expert index: e = Idx[b]
    ln("  %idx_ptr = getelementptr i32, i32 addrspace(1)* %Idx, i32 %b_idx")
    ln("  %e_idx = load i32, i32 addrspace(1)* %idx_ptr")
    # X base: X[b,m,:] at b*M*K + m*K
    ln(f"  %mk = add i32 0, {M * K}")
    ln("  %x_batch_off = mul i32 %b_idx, %mk")
    ln(f"  %k_val = add i32 0, {K}")
    ln("  %x_row_off = mul i32 %m_idx, %k_val")
    ln("  %x_base = add i32 %x_batch_off, %x_row_off")
    # W base: W[e,:,n] at e*K*N + k*N + n
    ln(f"  %kn = add i32 0, {K * N}")
    ln("  %w_expert_off = mul i32 %e_idx, %kn")
    ln("  br label %k_loop_header")
    ln("")
    ln("k_loop_header:")
    ln("  %k = phi i32 [0, %compute], [%k_next, %k_loop_body]")
    ln("  %acc = phi float [0.0, %compute], [%acc_new, %k_loop_body]")
    ln(f"  %k_done = icmp sge i32 %k, {K}")
    ln("  br i1 %k_done, label %k_loop_exit, label %k_loop_body")
    ln("")
    ln("k_loop_body:")
    # Load X[b,m,k]
    ln("  %x_off = add i32 %x_base, %k")
    ln("  %x_ptr = getelementptr float, float addrspace(1)* %X, i32 %x_off")
    ln("  %x_val = load float, float addrspace(1)* %x_ptr")
    # Load W[e,k,n] = w_expert_off + k*N + n
    ln("  %w_k_off = mul i32 %k, %n_size")
    ln("  %w_off_1 = add i32 %w_expert_off, %w_k_off")
    ln("  %w_off = add i32 %w_off_1, %n_idx")
    ln("  %w_ptr = getelementptr float, float addrspace(1)* %W, i32 %w_off")
    ln("  %w_val = load float, float addrspace(1)* %w_ptr")
    ln("  %prod = fmul float %x_val, %w_val")
    ln("  %acc_new = fadd float %acc, %prod")
    ln("  %k_next = add i32 %k, 1")
    ln("  br label %k_loop_header")
    ln("")
    ln("k_loop_exit:")
    # Store Out[gid]
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid")
    ln("  store float %acc, float addrspace(1)* %out_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. concat: A[M,N1], B[M,N2] -> out[M, N1+N2]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_concat(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit concat kernel: out[M, N1+N2] from A[M,N1] and B[M,N2].

    Flat 1D grid. Each thread handles one output element.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    a_name, b_name = input_names[0], input_names[1]
    out_name = node.outputs[0]

    a_val = graph.get_value(a_name)
    b_val = graph.get_value(b_name)
    a_shape = list(a_val.shape) if a_val.shape else [64, 32]
    b_shape = list(b_val.shape) if b_val.shape else [64, 32]

    M = a_shape[0]
    N1 = a_shape[1]
    N2 = b_shape[1]
    N_out = N1 + N2
    dtype = a_val.dtype or "float32"
    total = M * N_out

    kernel_name = f"arke_concat_{M}x{N1}x{N2}"
    grid, block = _flat_grid(total)

    source = _gen_concat_ir(kernel_name, M, N1, N2, N_out)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="concat",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: a_shape, b_name: b_shape, out_name: [M, N_out]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
            ("int", M), ("int", N1), ("int", N2),
        ],
    )


def _gen_concat_ir(kernel_name: str, M: int, N1: int, N2: int, N_out: int) -> str:
    total = M * N_out
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %Out, i32 %M, i32 %N1, i32 %N2) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid -> (row, col) in output [M, N_out]
    ln(f"  %n_out = add i32 0, {N_out}")
    ln("  %row = sdiv i32 %gid, %n_out")
    ln("  %col = srem i32 %gid, %n_out")
    # If col < N1, read from A[row, col], else from B[row, col - N1]
    ln(f"  %n1_val = add i32 0, {N1}")
    ln("  %from_a = icmp slt i32 %col, %n1_val")
    ln("  br i1 %from_a, label %load_a, label %load_b")
    ln("")
    ln("load_a:")
    ln(f"  %a_row_off = mul i32 %row, {N1}")
    ln("  %a_idx = add i32 %a_row_off, %col")
    ln("  %a_ptr = getelementptr float, float addrspace(1)* %A, i32 %a_idx")
    ln("  %a_val = load float, float addrspace(1)* %a_ptr")
    ln("  br label %store")
    ln("")
    ln("load_b:")
    ln("  %b_col = sub i32 %col, %n1_val")
    ln(f"  %b_row_off = mul i32 %row, {N2}")
    ln("  %b_idx = add i32 %b_row_off, %b_col")
    ln("  %b_ptr = getelementptr float, float addrspace(1)* %B, i32 %b_idx")
    ln("  %b_val = load float, float addrspace(1)* %b_ptr")
    ln("  br label %store")
    ln("")
    ln("store:")
    ln("  %val = phi float [%a_val, %load_a], [%b_val, %load_b]")
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid")
    ln("  store float %val, float addrspace(1)* %out_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. copy_: X[M,N] -> out[M,N]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_copy_(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit copy_ kernel: simple element-wise copy. Flat 1D."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name = input_names[0]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 64]

    M, N = x_shape[0], x_shape[1] if len(x_shape) > 1 else 1
    dtype = x_val.dtype or "float32"
    total = M * N

    kernel_name = f"arke_copy_{M}x{N}"
    grid, block = _flat_grid(total)

    source = _gen_copy_ir(kernel_name, total)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="copy_",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, out_name: x_shape},
        dtypes={x_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def _gen_copy_ir(kernel_name: str, total: int) -> str:
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %total) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %do_copy, label %done")
    ln("")
    ln("do_copy:")
    ln("  %src_ptr = getelementptr float, float addrspace(1)* %X, i32 %gid")
    ln("  %val = load float, float addrspace(1)* %src_ptr")
    ln("  %dst_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid")
    ln("  store float %val, float addrspace(1)* %dst_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. embedding: table[V,D], indices[M] -> out[M,D]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_embedding(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit embedding lookup: out[m,d] = table[indices[m], d].

    Flat 1D grid over M*D elements.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    table_name, idx_name = input_names[0], input_names[1]
    out_name = node.outputs[0]

    table_val = graph.get_value(table_name)
    idx_val = graph.get_value(idx_name)
    table_shape = list(table_val.shape) if table_val.shape else [1000, 64]
    idx_shape = list(idx_val.shape) if idx_val.shape else [64]

    V, D = table_shape[0], table_shape[1]
    M = idx_shape[0]
    dtype = table_val.dtype or "float32"
    total = M * D

    kernel_name = f"arke_embedding_{V}x{D}_M{M}"
    grid, block = _flat_grid(total)

    source = _gen_embedding_ir(kernel_name, M, D, V)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="embedding",
        param_names=[table_name, idx_name, out_name],
        output_name=out_name,
        shapes={table_name: table_shape, idx_name: idx_shape, out_name: [M, D]},
        dtypes={table_name: dtype, idx_name: "int32", out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", table_name), ("ptr", idx_name), ("ptr", out_name),
            ("int", M), ("int", D),
        ],
    )


def _gen_embedding_ir(kernel_name: str, M: int, D: int, V: int) -> str:
    total = M * D
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %Table, i32 addrspace(1)* %Indices, float addrspace(1)* %Out, i32 %M, i32 %D) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid -> (row, col): row = gid / D, col = gid % D
    ln(f"  %d_val = add i32 0, {D}")
    ln("  %row = sdiv i32 %gid, %d_val")
    ln("  %col = srem i32 %gid, %d_val")
    # Load index: idx = Indices[row]
    ln("  %idx_ptr = getelementptr i32, i32 addrspace(1)* %Indices, i32 %row")
    ln("  %idx = load i32, i32 addrspace(1)* %idx_ptr")
    # Load table[idx, col] = Table[idx * D + col]
    ln("  %table_row_off = mul i32 %idx, %d_val")
    ln("  %table_off = add i32 %table_row_off, %col")
    ln("  %table_ptr = getelementptr float, float addrspace(1)* %Table, i32 %table_off")
    ln("  %val = load float, float addrspace(1)* %table_ptr")
    # Store out[gid]
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid")
    ln("  store float %val, float addrspace(1)* %out_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. gather: X[M,N], idx[M,K] -> out[M,K]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_gather(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit gather: out[i,j] = X[i, idx[i,j]].

    Flat 1D grid over M*K elements.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name, idx_name = input_names[0], input_names[1]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    idx_val = graph.get_value(idx_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 64]
    idx_shape = list(idx_val.shape) if idx_val.shape else [64, 16]

    M, N = x_shape[0], x_shape[1]
    K_out = idx_shape[1]
    dtype = x_val.dtype or "float32"
    total = M * K_out

    kernel_name = f"arke_gather_{M}x{N}_K{K_out}"
    grid, block = _flat_grid(total)

    source = _gen_gather_ir(kernel_name, M, N, K_out)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="gather",
        param_names=[x_name, idx_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, idx_name: idx_shape, out_name: [M, K_out]},
        dtypes={x_name: dtype, idx_name: "int32", out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", idx_name), ("ptr", out_name),
            ("int", M), ("int", N), ("int", K_out),
        ],
    )


def _gen_gather_ir(kernel_name: str, M: int, N: int, K: int) -> str:
    total = M * K
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %X, i32 addrspace(1)* %Idx, float addrspace(1)* %Out, i32 %M, i32 %N, i32 %K) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid -> (row, j): row = gid / K, j = gid % K
    ln(f"  %k_val = add i32 0, {K}")
    ln("  %row = sdiv i32 %gid, %k_val")
    ln("  %j = srem i32 %gid, %k_val")
    # Load col index: col = Idx[row * K + j] = Idx[gid]
    ln("  %idx_ptr = getelementptr i32, i32 addrspace(1)* %Idx, i32 %gid")
    ln("  %col = load i32, i32 addrspace(1)* %idx_ptr")
    # Load X[row, col] = X[row*N + col]
    ln(f"  %n_val = add i32 0, {N}")
    ln("  %x_row_off = mul i32 %row, %n_val")
    ln("  %x_off = add i32 %x_row_off, %col")
    ln("  %x_ptr = getelementptr float, float addrspace(1)* %X, i32 %x_off")
    ln("  %val = load float, float addrspace(1)* %x_ptr")
    # Store out[gid]
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid")
    ln("  store float %val, float addrspace(1)* %out_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. scatter: X[M,N], idx[M,K], src[M,K] -> out[M,N]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_scatter(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit scatter: copy X to out, then out[i, idx[i,j]] = src[i,j].

    Per-row single block: grid=(M,1,1), block=(256,1,1).
    Phase 1: copy row from X. Barrier. Phase 2: scatter src values.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name, idx_name, src_name = input_names[0], input_names[1], input_names[2]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    idx_val = graph.get_value(idx_name)
    src_val = graph.get_value(src_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 64]
    idx_shape = list(idx_val.shape) if idx_val.shape else [64, 16]

    M, N = x_shape[0], x_shape[1]
    K = idx_shape[1]
    dtype = x_val.dtype or "float32"

    kernel_name = f"arke_scatter_{M}x{N}_K{K}"
    grid = (M, 1, 1)
    block = (256, 1, 1)

    source = _gen_scatter_ir(kernel_name, M, N, K)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="scatter",
        param_names=[x_name, idx_name, src_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, idx_name: idx_shape, src_name: [M, K], out_name: x_shape},
        dtypes={x_name: dtype, idx_name: "int32", src_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", idx_name), ("ptr", src_name), ("ptr", out_name),
            ("int", M), ("int", N), ("int", K),
        ],
    )


def _gen_scatter_ir(kernel_name: str, M: int, N: int, K: int) -> str:
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %X, i32 addrspace(1)* %Idx, float addrspace(1)* %Src, float addrspace(1)* %Out, i32 %M, i32 %N, i32 %K) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    # Phase 1: copy X[row, :] -> Out[row, :]
    # Thread loops: j = tid, tid+256, ... < N
    ln("  br label %copy_header")
    ln("")
    ln("copy_header:")
    ln("  %cj = phi i32 [%tid, %entry], [%cj_next, %copy_body]")
    ln(f"  %cj_done = icmp sge i32 %cj, {N}")
    ln("  br i1 %cj_done, label %barrier1, label %copy_body")
    ln("")
    ln("copy_body:")
    # X[row*N + cj]
    ln(f"  %x_row_off = mul i32 %row, {N}")
    ln("  %x_off = add i32 %x_row_off, %cj")
    ln("  %x_ptr = getelementptr float, float addrspace(1)* %X, i32 %x_off")
    ln("  %x_val = load float, float addrspace(1)* %x_ptr")
    ln("  %out_off = add i32 %x_row_off, %cj")
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %out_off")
    ln("  store float %x_val, float addrspace(1)* %out_ptr")
    ln("  %cj_next = add i32 %cj, 256")
    ln("  br label %copy_header")
    ln("")
    ln("barrier1:")
    ln("  call void @llvm.nvvm.barrier0()")
    # Phase 2: scatter src[row, j] -> Out[row, Idx[row,j]]
    ln("  br label %scatter_header")
    ln("")
    ln("scatter_header:")
    ln("  %sj = phi i32 [%tid, %barrier1], [%sj_next, %scatter_body]")
    ln(f"  %sj_done = icmp sge i32 %sj, {K}")
    ln("  br i1 %sj_done, label %done, label %scatter_body")
    ln("")
    ln("scatter_body:")
    # Load Idx[row*K + sj]
    ln(f"  %idx_row_off = mul i32 %row, {K}")
    ln("  %idx_off = add i32 %idx_row_off, %sj")
    ln("  %idx_ptr = getelementptr i32, i32 addrspace(1)* %Idx, i32 %idx_off")
    ln("  %dest_col = load i32, i32 addrspace(1)* %idx_ptr")
    # Load Src[row*K + sj]
    ln("  %src_off = add i32 %idx_row_off, %sj")
    ln("  %src_ptr = getelementptr float, float addrspace(1)* %Src, i32 %src_off")
    ln("  %src_val = load float, float addrspace(1)* %src_ptr")
    # Store Out[row*N + dest_col]
    ln(f"  %out_row_off2 = mul i32 %row, {N}")
    ln("  %out_off2 = add i32 %out_row_off2, %dest_col")
    ln("  %out_ptr2 = getelementptr float, float addrspace(1)* %Out, i32 %out_off2")
    ln("  store float %src_val, float addrspace(1)* %out_ptr2")
    ln("  %sj_next = add i32 %sj, 256")
    ln("  br label %scatter_header")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 8. permute (2D transpose): out[j,i] = in[i,j]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_permute(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit permute (2D transpose): out[j,i] = in[i,j]. Flat 1D grid."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name = input_names[0]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 64]

    M, N = x_shape[0], x_shape[1]
    dtype = x_val.dtype or "float32"
    total = M * N

    kernel_name = f"arke_permute_{M}x{N}"
    grid, block = _flat_grid(total)

    source = _gen_transpose_ir(kernel_name, M, N, total)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="permute",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, out_name: [N, M]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. split: X[M,N] -> out[M, N//2] (first half of each row)
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_split(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit split: out[m,n] = X[m,n] for n < N//2. Flat 1D grid."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name = input_names[0]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 64]

    M, N = x_shape[0], x_shape[1]
    N_half = N // 2
    dtype = x_val.dtype or "float32"
    total = M * N_half

    kernel_name = f"arke_split_{M}x{N}"
    grid, block = _flat_grid(total)

    source = _gen_split_ir(kernel_name, M, N, N_half)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="split",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, out_name: [M, N_half]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def _gen_split_ir(kernel_name: str, M: int, N: int, N_half: int) -> str:
    total = M * N_half
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid -> (row, col) in output [M, N_half]
    ln(f"  %n_half = add i32 0, {N_half}")
    ln("  %row = sdiv i32 %gid, %n_half")
    ln("  %col = srem i32 %gid, %n_half")
    # src offset: X[row*N + col]
    ln(f"  %n_full = add i32 0, {N}")
    ln("  %x_row_off = mul i32 %row, %n_full")
    ln("  %x_off = add i32 %x_row_off, %col")
    ln("  %x_ptr = getelementptr float, float addrspace(1)* %X, i32 %x_off")
    ln("  %val = load float, float addrspace(1)* %x_ptr")
    # Store out[gid]
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid")
    ln("  store float %val, float addrspace(1)* %out_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 10. transpose: out[j,i] = in[i,j]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_transpose(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit transpose: out[j,i] = in[i,j]. Flat 1D grid over M*N."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    x_name = input_names[0]
    out_name = node.outputs[0]

    x_val = graph.get_value(x_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 64]

    M, N = x_shape[0], x_shape[1]
    dtype = x_val.dtype or "float32"
    total = M * N

    kernel_name = f"arke_transpose_{M}x{N}"
    grid, block = _flat_grid(total)

    source = _gen_transpose_ir(kernel_name, M, N, total)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="transpose",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: x_shape, out_name: [N, M]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def _gen_transpose_ir(kernel_name: str, M: int, N: int, total: int) -> str:
    """Generate transpose LLVM IR. Used by both transpose and permute."""
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{")
    ln("entry:")
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %bsize = mul i32 %bid, 256")
    ln("  %gid = add i32 %bsize, %tid")
    ln(f"  %total = add i32 0, {total}")
    ln("  %in_bounds = icmp slt i32 %gid, %total")
    ln("  br i1 %in_bounds, label %compute, label %done")
    ln("")
    ln("compute:")
    # gid indexes input in row-major: gid = i*N + j
    ln(f"  %n_val = add i32 0, {N}")
    ln(f"  %m_val = add i32 0, {M}")
    ln("  %i = sdiv i32 %gid, %n_val")
    ln("  %j = srem i32 %gid, %n_val")
    # Load in[i,j] = in[gid]
    ln("  %in_ptr = getelementptr float, float addrspace(1)* %X, i32 %gid")
    ln("  %val = load float, float addrspace(1)* %in_ptr")
    # Store out[j,i] = out[j*M + i]
    ln("  %out_off = mul i32 %j, %m_val")
    ln("  %out_idx = add i32 %out_off, %i")
    ln("  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %out_idx")
    ln("  store float %val, float addrspace(1)* %out_ptr")
    ln("  br label %done")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)
