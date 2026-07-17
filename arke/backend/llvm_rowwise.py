# Copyright 2025 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""LLVM IR emitters for rowwise/reduction operations.

Implements 10 rowwise ops as NVPTX LLVM IR kernels:
  softmax, layernorm, rmsnorm, reduce_sum, reduce_max, reduce_mean,
  argmax, cumsum, topk, rmsnorm_residual

softmax, layernorm, rmsnorm use warp-level parallelism:
  Grid=(M,1,1), Block=(32,1,1) -- 1 warp per row, 32 threads cooperate.
  Each thread handles ceil(N/32) elements via strided access.
  Warp reductions via inline PTX shfl.sync.down.b32.
  Broadcast from lane 0 via shfl.sync.idx.b32.

All other reduction ops use 1-thread-per-row:
  Grid=(M,1,1), Block=(1,1,1) -- thread 0 loops sequentially over N.
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

_INTRINSICS_WARP = """\
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare float @llvm.nvvm.ex2.approx.f(float)
declare float @llvm.nvvm.sqrt.rn.f(float)
declare float @llvm.nvvm.lg2.approx.f(float)
declare float @llvm.nvvm.rcp.approx.ftz.f(float)
declare float @llvm.nvvm.rsqrt.approx.f(float)
declare float @llvm.fabs.f32(float)
declare float @llvm.maxnum.f32(float, float)
"""


def _module_header() -> str:
    return f"{_DATALAYOUT}\n{_TRIPLE}\n\n{_INTRINSICS}\n"


def _module_header_warp() -> str:
    return f"{_DATALAYOUT}\n{_TRIPLE}\n\n{_INTRINSICS_WARP}\n"


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
# Warp reduce helpers (inline PTX asm patterns as IR strings)
# ---------------------------------------------------------------------------

def _warp_reduce_max_ir(acc_name: str, result_name: str, prefix: str = "") -> str:
    """Generate LLVM IR for 5-step warp max reduction + broadcast.

    Takes accumulator %{acc_name}, produces %{result_name} (broadcast to all lanes).
    """
    p = prefix  # unique prefix to avoid SSA conflicts
    return f"""\
  %{p}s1 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{acc_name}, i32 16, i32 31, i32 -1)
  %{p}m1 = call float @llvm.maxnum.f32(float %{acc_name}, float %{p}s1)
  %{p}s2 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m1, i32 8, i32 31, i32 -1)
  %{p}m2 = call float @llvm.maxnum.f32(float %{p}m1, float %{p}s2)
  %{p}s3 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m2, i32 4, i32 31, i32 -1)
  %{p}m3 = call float @llvm.maxnum.f32(float %{p}m2, float %{p}s3)
  %{p}s4 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m3, i32 2, i32 31, i32 -1)
  %{p}m4 = call float @llvm.maxnum.f32(float %{p}m3, float %{p}s4)
  %{p}s5 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m4, i32 1, i32 31, i32 -1)
  %{p}m5 = call float @llvm.maxnum.f32(float %{p}m4, float %{p}s5)
  %{result_name} = call float asm "shfl.sync.idx.b32 $0, $1, 0, 31, $2;", "=f,f,r"(float %{p}m5, i32 -1)"""


def _warp_reduce_sum_ir(acc_name: str, result_name: str, prefix: str = "") -> str:
    """Generate LLVM IR for 5-step warp sum reduction + broadcast.

    Takes accumulator %{acc_name}, produces %{result_name} (broadcast to all lanes).
    """
    p = prefix
    return f"""\
  %{p}s1 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{acc_name}, i32 16, i32 31, i32 -1)
  %{p}m1 = fadd float %{acc_name}, %{p}s1
  %{p}s2 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m1, i32 8, i32 31, i32 -1)
  %{p}m2 = fadd float %{p}m1, %{p}s2
  %{p}s3 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m2, i32 4, i32 31, i32 -1)
  %{p}m3 = fadd float %{p}m2, %{p}s3
  %{p}s4 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m3, i32 2, i32 31, i32 -1)
  %{p}m4 = fadd float %{p}m3, %{p}s4
  %{p}s5 = call float asm "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=f,f,r,r,r"(float %{p}m4, i32 1, i32 31, i32 -1)
  %{p}m5 = fadd float %{p}m4, %{p}s5
  %{result_name} = call float asm "shfl.sync.idx.b32 $0, $1, 0, 31, $2;", "=f,f,r"(float %{p}m5, i32 -1)"""


# ---------------------------------------------------------------------------
# 1. softmax -- 2-pass online softmax (Milakov & Gimelshein).
#    Pass 1: fused running max + running sum (online update).
#    Pass 2: exp(x - global_max) * inv_sum  (normalize in one pass).
#    Reduces memory reads from 3N to 2N per row.
#    Grid=(M,1,1), Block=(32,1,1) -- 1 warp per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_softmax(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row softmax. Grid=(M,1,1), Block=(32,1,1).

    Uses 2-pass online softmax algorithm:
      Pass 1: Each thread tracks running_max and running_sum simultaneously.
              For each element x:
                new_max = max(running_max, x)
                correction = exp2((old_max - new_max) * log2e)
                running_sum = running_sum * correction + exp2((x - new_max) * log2e)
                running_max = new_max
              Then warp-reduce max, correct local sums, warp-reduce sum.
      Pass 2: Read X again, compute exp(x - global_max) * inv_sum, store.
    """
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_softmax"

    source = _module_header_warp()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; ---- Pass 1: online softmax -- fused running max + running sum ----
  br label %online_loop

online_loop:
  %j1 = phi i32 [%tid, %entry], [%j1_next, %online_body]
  %run_max = phi float [0xFFF0000000000000, %entry], [%updated_max, %online_body]
  %run_sum = phi float [0.0, %entry], [%updated_sum, %online_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %online_body, label %warp_reduce

online_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  ; new_max = max(running_max, x)
  %updated_max = call float @llvm.maxnum.f32(float %run_max, float %val1)
  ; correction = exp2((old_max - new_max) * log2e)
  ; Note: log2e factor is needed for natural exp via exp2. But since
  ; (old_max - new_max) is already the natural-log-scale difference and
  ; we use exp2(z * log2e) = exp(z), we apply log2e here.
  %corr_diff = fsub float %run_max, %updated_max
  %corr_scaled = fmul float %corr_diff, 0x3FF7154760000000
  %correction = call float @llvm.nvvm.ex2.approx.f(float %corr_scaled)
  ; exp_x = exp2((x - new_max) * log2e) = exp(x - new_max)
  %x_diff = fsub float %val1, %updated_max
  %x_scaled = fmul float %x_diff, 0x3FF7154760000000
  %exp_x = call float @llvm.nvvm.ex2.approx.f(float %x_scaled)
  ; running_sum = running_sum * correction + exp_x
  %corrected_sum = fmul float %run_sum, %correction
  %updated_sum = fadd float %corrected_sum, %exp_x
  %j1_next = add i32 %j1, 32
  br label %online_loop

warp_reduce:
  ; Step 1: Warp reduce max + broadcast to get global_max
{_warp_reduce_max_ir("run_max", "global_max", "mx_")}
  ; Step 2: Each thread corrects its local sum to account for global_max
  ;         local_sum *= exp2((local_max - global_max) * log2e)
  %lm_diff = fsub float %run_max, %global_max
  %lm_scaled = fmul float %lm_diff, 0x3FF7154760000000
  %lm_corr = call float @llvm.nvvm.ex2.approx.f(float %lm_scaled)
  %corrected_local_sum = fmul float %run_sum, %lm_corr
  ; Step 3: Warp reduce sum + broadcast
{_warp_reduce_sum_ir("corrected_local_sum", "global_sum", "sm_")}
  ; Step 4: Compute 1/sum via rcp.approx.ftz.f
  %inv_sum = call float @llvm.nvvm.rcp.approx.ftz.f(float %global_sum)
  ; ---- Pass 2: normalize -- exp(x - global_max) * inv_sum ----
  br label %norm_loop

norm_loop:
  %j2 = phi i32 [%tid, %warp_reduce], [%j2_next, %norm_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %norm_body, label %done

norm_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %shifted = fsub float %val2, %global_max
  %shifted_lg2e = fmul float %shifted, 0x3FF7154760000000
  %exp_val = call float @llvm.nvvm.ex2.approx.f(float %shifted_lg2e)
  %normed = fmul float %exp_val, %inv_sum
  %optr2 = getelementptr float, float addrspace(1)* %Out, i64 %idx2
  store float %normed, float addrspace(1)* %optr2
  %j2_next = add i32 %j2, 32
  br label %norm_loop

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="softmax",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(M, 1, 1),
        block=(32, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 2. layernorm -- 3-pass warp-parallel + affine: mean, variance, normalize.
#    Grid=(M,1,1), Block=(32,1,1) -- 1 warp per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_layernorm(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row layer normalization. Grid=(M,1,1), Block=(32,1,1)."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    w_name = input_names[1]
    b_name = input_names[2]
    kernel_name = "arke_layernorm"

    source = _module_header_warp()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, float addrspace(1)* %B, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; ---- Pass 1: partial sum for mean ----
  br label %mean_loop

mean_loop:
  %j1 = phi i32 [%tid, %entry], [%j1_next, %mean_body]
  %psum = phi float [0.0, %entry], [%new_psum, %mean_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %mean_body, label %mean_reduce

mean_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %new_psum = fadd float %psum, %val1
  %j1_next = add i32 %j1, 32
  br label %mean_loop

mean_reduce:
  ; Warp reduce sum for mean + broadcast
{_warp_reduce_sum_ir("psum", "total_sum", "mn_")}
  %Nf = sitofp i32 %N to float
  %inv_N = call float @llvm.nvvm.rcp.approx.ftz.f(float %Nf)
  %mean = fmul float %total_sum, %inv_N
  ; ---- Pass 2: partial variance ----
  br label %var_loop

var_loop:
  %j2 = phi i32 [%tid, %mean_reduce], [%j2_next, %var_body]
  %pvar = phi float [0.0, %mean_reduce], [%new_pvar, %var_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %var_body, label %var_reduce

var_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %diff = fsub float %val2, %mean
  %sq = fmul float %diff, %diff
  %new_pvar = fadd float %pvar, %sq
  %j2_next = add i32 %j2, 32
  br label %var_loop

var_reduce:
  ; Warp reduce sum for variance + broadcast
{_warp_reduce_sum_ir("pvar", "total_var", "vr_")}
  %var = fmul float %total_var, %inv_N
  %var_eps = fadd float %var, 0x3EE4F8B580000000
  ; rsqrt(var + eps) for normalization
  %rstd = call float @llvm.nvvm.rsqrt.approx.f(float %var_eps)
  ; ---- Pass 3: normalize + affine ----
  br label %norm_loop

norm_loop:
  %j3 = phi i32 [%tid, %var_reduce], [%j3_next, %norm_body]
  %cmp3 = icmp slt i32 %j3, %N
  br i1 %cmp3, label %norm_body, label %done

norm_body:
  %j3_64 = sext i32 %j3 to i64
  %idx3 = add i64 %base, %j3_64
  %ptr3 = getelementptr float, float addrspace(1)* %X, i64 %idx3
  %val3 = load float, float addrspace(1)* %ptr3
  %diff3 = fsub float %val3, %mean
  %normed = fmul float %diff3, %rstd
  ; Load weight and bias (1D indexing by column)
  %wptr = getelementptr float, float addrspace(1)* %W, i64 %j3_64
  %w = load float, float addrspace(1)* %wptr
  %bptr = getelementptr float, float addrspace(1)* %B, i64 %j3_64
  %b = load float, float addrspace(1)* %bptr
  ; affine: out = normed * w + b
  %scaled = fmul float %normed, %w
  %out_val = fadd float %scaled, %b
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %idx3
  store float %out_val, float addrspace(1)* %optr
  %j3_next = add i32 %j3, 32
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
        block=(32, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", b_name),
            ("ptr", out_name), ("int", M), ("int", N),
        ],
    )


# ---------------------------------------------------------------------------
# 3. rmsnorm -- 2-pass warp-parallel + affine: sum-of-squares, normalize.
#    Grid=(M,1,1), Block=(32,1,1) -- 1 warp per row.
# ---------------------------------------------------------------------------


def emit_llvm_ir_rmsnorm(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row RMS normalization. Grid=(M,1,1), Block=(32,1,1)."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    w_name = input_names[1]
    kernel_name = "arke_rmsnorm"

    source = _module_header_warp()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, float addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; ---- Pass 1: partial sum of squares ----
  br label %ss_loop

ss_loop:
  %j1 = phi i32 [%tid, %entry], [%j1_next, %ss_body]
  %pss = phi float [0.0, %entry], [%new_pss, %ss_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %ss_body, label %ss_reduce

ss_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %sq = fmul float %val1, %val1
  %new_pss = fadd float %pss, %sq
  %j1_next = add i32 %j1, 32
  br label %ss_loop

ss_reduce:
  ; Warp reduce sum for sum-of-squares + broadcast
{_warp_reduce_sum_ir("pss", "total_ss", "ss_")}
  %Nf = sitofp i32 %N to float
  %inv_N = call float @llvm.nvvm.rcp.approx.ftz.f(float %Nf)
  %mean_ss = fmul float %total_ss, %inv_N
  %eps_val = fadd float %mean_ss, 0x3EE4F8B580000000
  ; rsqrt(mean_ss + eps) = 1/rms
  %inv_rms = call float @llvm.nvvm.rsqrt.approx.f(float %eps_val)
  ; ---- Pass 2: normalize and scale by weight ----
  br label %norm_loop

norm_loop:
  %j2 = phi i32 [%tid, %ss_reduce], [%j2_next, %norm_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %norm_body, label %done

norm_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %normed = fmul float %val2, %inv_rms
  %wptr = getelementptr float, float addrspace(1)* %W, i64 %j2_64
  %w = load float, float addrspace(1)* %wptr
  %out_val = fmul float %normed, %w
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %idx2
  store float %out_val, float addrspace(1)* %optr
  %j2_next = add i32 %j2, 32
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
        block=(32, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", out_name),
            ("int", M), ("int", N),
        ],
    )


# ---------------------------------------------------------------------------
# 4. reduce_sum -- 1 thread per row, sequential sum.
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
# 5. reduce_max -- 1 thread per row, sequential max.
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
# 6. reduce_mean -- 1 thread per row, sum/N.
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
# 7. argmax -- per-row argmax. X[M,N] -> out[M] (int32). 1 thread per row.
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
# 8. cumsum -- per-row inclusive prefix sum. 1 thread per row (sequential).
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
# 9. topk -- per-row top-K selection. X[M,N] -> out[M,K]. 1 thread per row.
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
# 10. rmsnorm_residual -- inputs: X[M,N], R[M,N], W[N]. y = rmsnorm(x+r)*w.
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
