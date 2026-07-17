# Copyright 2025 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""LLVM IR emitters for rowwise/reduction operations.

Implements 10 rowwise ops as NVPTX LLVM IR kernels:
  softmax, layernorm, rmsnorm, reduce_sum, reduce_max, reduce_mean,
  argmax, cumsum, topk, rmsnorm_residual

All reduction ops use a simplified 1-thread-per-row approach:
  Grid=(M,1,1), Block=(1,1,1) — thread 0 loops sequentially over N.
"""
from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel
from arke.ir.graph import IRGraph

# ---------------------------------------------------------------------------
# Common LLVM IR header / footer helpers
# ---------------------------------------------------------------------------

_DATALAYOUT = (
    'target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32'
    "-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32"
    '-v64:64:64-v128:128:128-n16:32:64"'
)
_TRIPLE = 'target triple = "nvptx64-nvidia-cuda"'

_INTRINSICS = """\
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare float @llvm.nvvm.ex2.approx.f(float)
declare float @llvm.nvvm.sqrt.rn.f(float)
declare float @llvm.nvvm.lg2.approx.f(float)
declare float @llvm.fabs.f32(float)
declare float @llvm.maxnum.f32(float, float)
"""


def _module_header() -> str:
    return f"{_DATALAYOUT}\n{_TRIPLE}\n\n{_INTRINSICS}\n"


def _nvvm_annotation(kernel_name: str) -> str:
    return (
        f"!nvvm.annotations = !{{!0}}\n"
        f'!0 = !{{void (float addrspace(1)*, float addrspace(1)*, i32, i32)* @{kernel_name}, !"kernel", i32 1}}\n'
    )


def _nvvm_annotation_custom(kernel_name: str, signature: str) -> str:
    return (
        f"!nvvm.annotations = !{{!0}}\n"
        f'!0 = !{{{signature}* @{kernel_name}, !"kernel", i32 1}}\n'
    )


def _extract_2d(graph: IRGraph, input_idx: int = 0):
    """Extract M, N, dtype, and input/output names from graph."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_name = input_names[input_idx]
    x_val = graph.get_value(x_name)
    shape = list(x_val.shape) if x_val.shape else [64, 64]
    M, N = shape[0], shape[1]
    dtype = x_val.dtype or "float32"
    return node, input_names, out_name, x_name, M, N, dtype


# ---------------------------------------------------------------------------
# 1. softmax — 3-pass: max, exp+sum, normalize. 1 thread per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_softmax(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row softmax. Grid=(M,1,1), Block=(1,1,1)."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_softmax"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; Pass 1: find max
  br label %max_loop

max_loop:
  %j1 = phi i32 [0, %entry], [%j1_next, %max_body]
  %cur_max = phi float [0xFFF0000000000000, %entry], [%new_max, %max_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %max_body, label %exp_sum_init

max_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %new_max = call float @llvm.maxnum.f32(float %cur_max, float %val1)
  %j1_next = add i32 %j1, 1
  br label %max_loop

exp_sum_init:
  ; Pass 2: compute exp(x-max) and sum
  br label %exp_loop

exp_loop:
  %j2 = phi i32 [0, %exp_sum_init], [%j2_next, %exp_body]
  %sum_acc = phi float [0.0, %exp_sum_init], [%new_sum, %exp_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %exp_body, label %norm_init

exp_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %shifted = fsub float %val2, %cur_max
  %shifted_lg2e = fmul float %shifted, 0x3FF7154760000000
  %exp_val = call float @llvm.nvvm.ex2.approx.f(float %shifted_lg2e)
  ; Store exp(x-max) temporarily in output
  %optr2 = getelementptr float, float addrspace(1)* %Out, i64 %idx2
  store float %exp_val, float addrspace(1)* %optr2
  %new_sum = fadd float %sum_acc, %exp_val
  %j2_next = add i32 %j2, 1
  br label %exp_loop

norm_init:
  ; Pass 3: normalize by sum
  br label %norm_loop

norm_loop:
  %j3 = phi i32 [0, %norm_init], [%j3_next, %norm_body]
  %cmp3 = icmp slt i32 %j3, %N
  br i1 %cmp3, label %norm_body, label %done

norm_body:
  %j3_64 = sext i32 %j3 to i64
  %idx3 = add i64 %base, %j3_64
  %optr3 = getelementptr float, float addrspace(1)* %Out, i64 %idx3
  %exp_stored = load float, float addrspace(1)* %optr3
  %normed = fdiv float %exp_stored, %sum_acc
  store float %normed, float addrspace(1)* %optr3
  %j3_next = add i32 %j3, 1
  br label %norm_loop

done:
  ret void
}}

"""
    source += _nvvm_annotation(kernel_name)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="softmax",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 2. layernorm — inputs: X[M,N], W[N], B[N]. 1 thread per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_layernorm(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row layer normalization."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    w_name = input_names[1]
    b_name = input_names[2]
    kernel_name = "arke_layernorm"
    eps = 1e-5

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, float addrspace(1)* %B, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; Pass 1: compute mean
  br label %mean_loop

mean_loop:
  %j1 = phi i32 [0, %entry], [%j1_next, %mean_body]
  %sum1 = phi float [0.0, %entry], [%new_sum1, %mean_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %mean_body, label %mean_done

mean_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %new_sum1 = fadd float %sum1, %val1
  %j1_next = add i32 %j1, 1
  br label %mean_loop

mean_done:
  %Nf = sitofp i32 %N to float
  %mean = fdiv float %sum1, %Nf
  ; Pass 2: compute variance
  br label %var_loop

var_loop:
  %j2 = phi i32 [0, %mean_done], [%j2_next, %var_body]
  %sum2 = phi float [0.0, %mean_done], [%new_sum2, %var_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %var_body, label %var_done

var_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %diff = fsub float %val2, %mean
  %sq = fmul float %diff, %diff
  %new_sum2 = fadd float %sum2, %sq
  %j2_next = add i32 %j2, 1
  br label %var_loop

var_done:
  %var = fdiv float %sum2, %Nf
  %eps_val = fadd float %var, 0x3EE4F8B580000000
  %rstd = call float @llvm.nvvm.sqrt.rn.f(float %eps_val)
  ; Pass 3: normalize and apply affine
  br label %norm_loop

norm_loop:
  %j3 = phi i32 [0, %var_done], [%j3_next, %norm_body]
  %cmp3 = icmp slt i32 %j3, %N
  br i1 %cmp3, label %norm_body, label %done

norm_body:
  %j3_64 = sext i32 %j3 to i64
  %idx3 = add i64 %base, %j3_64
  %ptr3 = getelementptr float, float addrspace(1)* %X, i64 %idx3
  %val3 = load float, float addrspace(1)* %ptr3
  %diff3 = fsub float %val3, %mean
  %normed = fdiv float %diff3, %rstd
  ; Load weight and bias
  %wptr = getelementptr float, float addrspace(1)* %W, i64 %j3_64
  %w = load float, float addrspace(1)* %wptr
  %bptr = getelementptr float, float addrspace(1)* %B, i64 %j3_64
  %b = load float, float addrspace(1)* %bptr
  ; affine: out = normed * w + b
  %scaled = fmul float %normed, %w
  %out_val = fadd float %scaled, %b
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %idx3
  store float %out_val, float addrspace(1)* %optr
  %j3_next = add i32 %j3, 1
  br label %norm_loop

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="layernorm",
        param_names=[x_name, w_name, b_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], w_name: [N], b_name: [N], out_name: [M, N]},
        dtypes={x_name: dtype, w_name: dtype, b_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", b_name),
            ("ptr", out_name), ("int", M), ("int", N),
        ],
    )


# ---------------------------------------------------------------------------
# 3. rmsnorm — inputs: X[M,N], W[N]. 1 thread per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_rmsnorm(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row RMS normalization."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    w_name = input_names[1]
    kernel_name = "arke_rmsnorm"
    eps = 1e-5

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; Pass 1: sum of squares
  br label %ss_loop

ss_loop:
  %j1 = phi i32 [0, %entry], [%j1_next, %ss_body]
  %ss = phi float [0.0, %entry], [%new_ss, %ss_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %ss_body, label %ss_done

ss_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %sq = fmul float %val1, %val1
  %new_ss = fadd float %ss, %sq
  %j1_next = add i32 %j1, 1
  br label %ss_loop

ss_done:
  %Nf = sitofp i32 %N to float
  %mean_ss = fdiv float %ss, %Nf
  %eps_val = fadd float %mean_ss, 0x3EE4F8B580000000
  %rms = call float @llvm.nvvm.sqrt.rn.f(float %eps_val)
  ; Pass 2: normalize and scale
  br label %norm_loop

norm_loop:
  %j2 = phi i32 [0, %ss_done], [%j2_next, %norm_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %norm_body, label %done

norm_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %normed = fdiv float %val2, %rms
  %wptr = getelementptr float, float addrspace(1)* %W, i64 %j2_64
  %w = load float, float addrspace(1)* %wptr
  %out_val = fmul float %normed, %w
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %idx2
  store float %out_val, float addrspace(1)* %optr
  %j2_next = add i32 %j2, 1
  br label %norm_loop

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="rmsnorm",
        param_names=[x_name, w_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], w_name: [N], out_name: [M, N]},
        dtypes={x_name: dtype, w_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", out_name),
            ("int", M), ("int", N),
        ],
    )


# ---------------------------------------------------------------------------
# 4. reduce_sum — 1 thread per row, sequential sum.
# ---------------------------------------------------------------------------


def emit_llvm_ir_reduce_sum(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row sum reduction. Output [M,1]."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_reduce_sum"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  br label %loop

loop:
  %j = phi i32 [0, %entry], [%j_next, %loop_body]
  %acc = phi float [0.0, %entry], [%new_acc, %loop_body]
  %cmp = icmp slt i32 %j, %N
  br i1 %cmp, label %loop_body, label %loop_exit

loop_body:
  %j64 = sext i32 %j to i64
  %idx = add i64 %base, %j64
  %ptr = getelementptr float, float addrspace(1)* %X, i64 %idx
  %val = load float, float addrspace(1)* %ptr
  %new_acc = fadd float %acc, %val
  %j_next = add i32 %j, 1
  br label %loop

loop_exit:
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %row64
  store float %acc, float addrspace(1)* %optr
  ret void
}}

"""
    source += _nvvm_annotation(kernel_name)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="reduce_sum",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, 1]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 5. reduce_max — 1 thread per row, sequential max.
# ---------------------------------------------------------------------------


def emit_llvm_ir_reduce_max(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row max reduction. Output [M,1]."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_reduce_max"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  br label %loop

loop:
  %j = phi i32 [0, %entry], [%j_next, %loop_body]
  %acc = phi float [0xFFF0000000000000, %entry], [%new_acc, %loop_body]
  %cmp = icmp slt i32 %j, %N
  br i1 %cmp, label %loop_body, label %loop_exit

loop_body:
  %j64 = sext i32 %j to i64
  %idx = add i64 %base, %j64
  %ptr = getelementptr float, float addrspace(1)* %X, i64 %idx
  %val = load float, float addrspace(1)* %ptr
  %new_acc = call float @llvm.maxnum.f32(float %acc, float %val)
  %j_next = add i32 %j, 1
  br label %loop

loop_exit:
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %row64
  store float %acc, float addrspace(1)* %optr
  ret void
}}

"""
    source += _nvvm_annotation(kernel_name)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="reduce_max",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, 1]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 6. reduce_mean — 1 thread per row, sum/N.
# ---------------------------------------------------------------------------


def emit_llvm_ir_reduce_mean(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row mean reduction. Output [M,1]."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_reduce_mean"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  br label %loop

loop:
  %j = phi i32 [0, %entry], [%j_next, %loop_body]
  %acc = phi float [0.0, %entry], [%new_acc, %loop_body]
  %cmp = icmp slt i32 %j, %N
  br i1 %cmp, label %loop_body, label %loop_exit

loop_body:
  %j64 = sext i32 %j to i64
  %idx = add i64 %base, %j64
  %ptr = getelementptr float, float addrspace(1)* %X, i64 %idx
  %val = load float, float addrspace(1)* %ptr
  %new_acc = fadd float %acc, %val
  %j_next = add i32 %j, 1
  br label %loop

loop_exit:
  %Nf = sitofp i32 %N to float
  %mean = fdiv float %acc, %Nf
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %row64
  store float %mean, float addrspace(1)* %optr
  ret void
}}

"""
    source += _nvvm_annotation(kernel_name)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="reduce_mean",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, 1]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 7. argmax — per-row argmax. X[M,N] -> out[M] (int32). 1 thread per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_argmax(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row argmax. Output is int32."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_argmax"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, i32 addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  br label %loop

loop:
  %j = phi i32 [0, %entry], [%j_next, %loop_body]
  %best_val = phi float [0xFFF0000000000000, %entry], [%sel_val, %loop_body]
  %best_idx = phi i32 [0, %entry], [%sel_idx, %loop_body]
  %cmp = icmp slt i32 %j, %N
  br i1 %cmp, label %loop_body, label %loop_exit

loop_body:
  %j64 = sext i32 %j to i64
  %idx = add i64 %base, %j64
  %ptr = getelementptr float, float addrspace(1)* %X, i64 %idx
  %val = load float, float addrspace(1)* %ptr
  %is_greater = fcmp ogt float %val, %best_val
  %sel_val = select i1 %is_greater, float %val, float %best_val
  %sel_idx = select i1 %is_greater, i32 %j, i32 %best_idx
  %j_next = add i32 %j, 1
  br label %loop

loop_exit:
  %optr = getelementptr i32, i32 addrspace(1)* %Out, i64 %row64
  store i32 %best_idx, i32 addrspace(1)* %optr
  ret void
}}

"""
    sig = "void (float addrspace(1)*, i32 addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="argmax",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M]},
        dtypes={x_name: dtype, out_name: "int32"},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 8. cumsum — per-row inclusive prefix sum. 1 thread per row (sequential).
# ---------------------------------------------------------------------------


def emit_llvm_ir_cumsum(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row inclusive prefix sum."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_cumsum"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  br label %loop

loop:
  %j = phi i32 [0, %entry], [%j_next, %loop_body]
  %running = phi float [0.0, %entry], [%new_running, %loop_body]
  %cmp = icmp slt i32 %j, %N
  br i1 %cmp, label %loop_body, label %done

loop_body:
  %j64 = sext i32 %j to i64
  %idx = add i64 %base, %j64
  %ptr = getelementptr float, float addrspace(1)* %X, i64 %idx
  %val = load float, float addrspace(1)* %ptr
  %new_running = fadd float %running, %val
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %idx
  store float %new_running, float addrspace(1)* %optr
  %j_next = add i32 %j, 1
  br label %loop

done:
  ret void
}}

"""
    source += _nvvm_annotation(kernel_name)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="cumsum",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 9. topk — per-row top-K selection. X[M,N] -> out[M,K]. 1 thread per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_topk(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row top-K. Uses K sequential max-selects."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_topk"

    # Determine K from output shape or node attrs
    out_val = graph.get_value(out_name)
    if out_val and out_val.shape and len(out_val.shape) >= 2:
        K = out_val.shape[1]
    elif hasattr(node, "attrs") and "k" in node.attrs:
        K = node.attrs["k"]
    else:
        K = 8

    # TopK: For each of K iterations, find max excluding previously selected.
    # We use a simple approach: mark selected positions with -inf.
    # Since we can't easily allocate a mask in pure LLVM IR, we write the
    # selected values to output and use a nested loop approach.
    # Actually simplest: copy row to output area as scratch, then K times
    # find-max-and-set-to-neginf.

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, float addrspace(1)* %Scratch, i32 %M, i32 %N, i32 %K) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %K64 = sext i32 %K to i64
  %base_in = mul i64 %row64, %N64
  %base_scratch = mul i64 %row64, %N64
  %base_out = mul i64 %row64, %K64
  ; Copy input row to scratch
  br label %copy_loop

copy_loop:
  %c = phi i32 [0, %entry], [%c_next, %copy_body]
  %cmp_c = icmp slt i32 %c, %N
  br i1 %cmp_c, label %copy_body, label %topk_init

copy_body:
  %c64 = sext i32 %c to i64
  %src_idx = add i64 %base_in, %c64
  %src_ptr = getelementptr float, float addrspace(1)* %X, i64 %src_idx
  %src_val = load float, float addrspace(1)* %src_ptr
  %dst_idx = add i64 %base_scratch, %c64
  %dst_ptr = getelementptr float, float addrspace(1)* %Scratch, i64 %dst_idx
  store float %src_val, float addrspace(1)* %dst_ptr
  %c_next = add i32 %c, 1
  br label %copy_loop

topk_init:
  ; K iterations: find max, store, set to -inf
  br label %k_loop

k_loop:
  %ki = phi i32 [0, %topk_init], [%ki_next, %k_store]
  %cmp_k = icmp slt i32 %ki, %K
  br i1 %cmp_k, label %find_max_init, label %done

find_max_init:
  br label %find_loop

find_loop:
  %fj = phi i32 [0, %find_max_init], [%fj_next, %find_body]
  %fmax = phi float [0xFFF0000000000000, %find_max_init], [%sel_fmax, %find_body]
  %fmax_idx = phi i32 [0, %find_max_init], [%sel_fmax_idx, %find_body]
  %cmp_f = icmp slt i32 %fj, %N
  br i1 %cmp_f, label %find_body, label %k_store

find_body:
  %fj64 = sext i32 %fj to i64
  %s_idx = add i64 %base_scratch, %fj64
  %s_ptr = getelementptr float, float addrspace(1)* %Scratch, i64 %s_idx
  %s_val = load float, float addrspace(1)* %s_ptr
  %is_gt = fcmp ogt float %s_val, %fmax
  %sel_fmax = select i1 %is_gt, float %s_val, float %fmax
  %sel_fmax_idx = select i1 %is_gt, i32 %fj, i32 %fmax_idx
  %fj_next = add i32 %fj, 1
  br label %find_loop

k_store:
  ; Store the found max value in output
  %ki64 = sext i32 %ki to i64
  %out_idx = add i64 %base_out, %ki64
  %out_ptr = getelementptr float, float addrspace(1)* %Out, i64 %out_idx
  store float %fmax, float addrspace(1)* %out_ptr
  ; Set found position to -inf in scratch
  %fmax_idx64 = sext i32 %fmax_idx to i64
  %neginf_idx = add i64 %base_scratch, %fmax_idx64
  %neginf_ptr = getelementptr float, float addrspace(1)* %Scratch, i64 %neginf_idx
  store float 0xFFF0000000000000, float addrspace(1)* %neginf_ptr
  %ki_next = add i32 %ki, 1
  br label %k_loop

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    # Scratch buffer is same size as input (M*N)
    scratch_name = f"{x_name}_scratch"
    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="topk",
        param_names=[x_name, out_name, scratch_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, K], scratch_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype, scratch_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", out_name), ("ptr", scratch_name),
            ("int", M), ("int", N), ("int", K),
        ],
    )


# ---------------------------------------------------------------------------
# 10. rmsnorm_residual — inputs: X[M,N], R[M,N], W[N]. y = rmsnorm(x+r)*w.
# ---------------------------------------------------------------------------


def emit_llvm_ir_rmsnorm_residual(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for fused rmsnorm with residual addition."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    r_name = input_names[1]
    w_name = input_names[2]
    kernel_name = "arke_rmsnorm_residual"
    eps = 1e-5

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %R, float addrspace(1)* %W, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; Pass 1: compute sum of squares of (x + r)
  br label %ss_loop

ss_loop:
  %j1 = phi i32 [0, %entry], [%j1_next, %ss_body]
  %ss = phi float [0.0, %entry], [%new_ss, %ss_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %ss_body, label %ss_done

ss_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %xptr = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %xval = load float, float addrspace(1)* %xptr
  %rptr = getelementptr float, float addrspace(1)* %R, i64 %idx1
  %rval = load float, float addrspace(1)* %rptr
  %xr = fadd float %xval, %rval
  %sq = fmul float %xr, %xr
  %new_ss = fadd float %ss, %sq
  %j1_next = add i32 %j1, 1
  br label %ss_loop

ss_done:
  %Nf = sitofp i32 %N to float
  %mean_ss = fdiv float %ss, %Nf
  %eps_val = fadd float %mean_ss, 0x3EE4F8B580000000
  %rms = call float @llvm.nvvm.sqrt.rn.f(float %eps_val)
  ; Pass 2: normalize and scale by W
  br label %norm_loop

norm_loop:
  %j2 = phi i32 [0, %ss_done], [%j2_next, %norm_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %norm_body, label %done

norm_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %xptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %xval2 = load float, float addrspace(1)* %xptr2
  %rptr2 = getelementptr float, float addrspace(1)* %R, i64 %idx2
  %rval2 = load float, float addrspace(1)* %rptr2
  %xr2 = fadd float %xval2, %rval2
  %normed = fdiv float %xr2, %rms
  %wptr = getelementptr float, float addrspace(1)* %W, i64 %j2_64
  %w = load float, float addrspace(1)* %wptr
  %out_val = fmul float %normed, %w
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %idx2
  store float %out_val, float addrspace(1)* %optr
  %j2_next = add i32 %j2, 1
  br label %norm_loop

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="rmsnorm_residual",
        param_names=[x_name, r_name, w_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], r_name: [M, N], w_name: [N], out_name: [M, N]},
        dtypes={x_name: dtype, r_name: dtype, w_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", r_name), ("ptr", w_name),
            ("ptr", out_name), ("int", M), ("int", N),
        ],
    )
