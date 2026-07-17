# Copyright 2025 Arke Contributors
# SPDX-License-Identifier: Apache-2.0
"""LLVM IR emitters for fused operations.

Implements 8 fused ops as NVPTX LLVM IR kernels:
  cross_entropy, dequantize_per_channel, fused_linear_cross_entropy,
  gelu_and_mul, quantize_per_token, rope, silu_and_mul, swiglu_packed
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
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare float @llvm.nvvm.ex2.approx.f(float)
declare float @llvm.nvvm.sqrt.rn.f(float)
declare float @llvm.nvvm.lg2.approx.f(float)
declare float @llvm.fabs.f32(float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.nvvm.rcp.approx.ftz.f(float)
"""


def _module_header() -> str:
    return f"{_DATALAYOUT}\n{_TRIPLE}\n\n{_INTRINSICS}\n"


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
    M = shape[0]
    N = shape[1] if len(shape) > 1 else shape[0]
    dtype = x_val.dtype or "float32"
    return node, input_names, out_name, x_name, M, N, dtype


# ---------------------------------------------------------------------------
# 1. cross_entropy — logits[M,N], targets[M] (int32) -> loss[M,1].
#    1 thread per row: max, logsumexp, pick label logit.
# ---------------------------------------------------------------------------


def emit_llvm_ir_cross_entropy(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-row cross entropy loss."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    targets_name = input_names[1]
    kernel_name = "arke_cross_entropy"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %Logits, i32 addrspace(1)* %Targets, float addrspace(1)* %Loss, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; Load target label for this row
  %tgt_ptr = getelementptr i32, i32 addrspace(1)* %Targets, i64 %row64
  %label = load i32, i32 addrspace(1)* %tgt_ptr
  ; Pass 1: find max logit
  br label %max_loop

max_loop:
  %j1 = phi i32 [0, %entry], [%j1_next, %max_body]
  %cur_max = phi float [0xFFF0000000000000, %entry], [%new_max, %max_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %max_body, label %sum_init

max_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %Logits, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %new_max = call float @llvm.maxnum.f32(float %cur_max, float %val1)
  %j1_next = add i32 %j1, 1
  br label %max_loop

sum_init:
  ; Pass 2: compute sum of exp(logit - max)
  br label %sum_loop

sum_loop:
  %j2 = phi i32 [0, %sum_init], [%j2_next, %sum_body]
  %sum_exp = phi float [0.0, %sum_init], [%new_sum, %sum_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %sum_body, label %compute_loss

sum_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %Logits, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  %shifted = fsub float %val2, %cur_max
  %shifted_lg2e = fmul float %shifted, 0x3FF7154760000000
  %exp_val = call float @llvm.nvvm.ex2.approx.f(float %shifted_lg2e)
  %new_sum = fadd float %sum_exp, %exp_val
  %j2_next = add i32 %j2, 1
  br label %sum_loop

compute_loss:
  ; logsumexp = max + log(sum_exp)
  %sum_exp_lg2 = call float @llvm.nvvm.lg2.approx.f(float %sum_exp)
  %log_sum = fmul float %sum_exp_lg2, 0x3FE62E4300000000
  %logsumexp = fadd float %cur_max, %log_sum
  ; Get logit at label index
  %label64 = sext i32 %label to i64
  %label_idx = add i64 %base, %label64
  %label_ptr = getelementptr float, float addrspace(1)* %Logits, i64 %label_idx
  %label_logit = load float, float addrspace(1)* %label_ptr
  ; loss = logsumexp - label_logit
  %loss = fsub float %logsumexp, %label_logit
  %out_ptr = getelementptr float, float addrspace(1)* %Loss, i64 %row64
  store float %loss, float addrspace(1)* %out_ptr
  ret void
}}

"""
    sig = "void (float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="cross_entropy",
        param_names=[x_name, targets_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], targets_name: [M], out_name: [M, 1]},
        dtypes={x_name: dtype, targets_name: "int32", out_name: dtype},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", targets_name), ("ptr", out_name),
            ("int", M), ("int", N),
        ],
    )


# ---------------------------------------------------------------------------
# 2. dequantize_per_channel — X_int8[M,N], scale[N], zero_point[N] -> out[M,N]
#    Flat 1D: out[gid] = (float(X[gid]) - float(Zp[j])) * Scale[j]
# ---------------------------------------------------------------------------


def emit_llvm_ir_dequantize_per_channel(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-channel dequantization (int8 -> float32)."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    scale_name = input_names[1]
    zp_name = input_names[2]
    kernel_name = "arke_dequantize_per_channel"
    total = M * N
    block = 256
    grid = (total + block - 1) // block

    source = _module_header()
    source += f"""\
define void @{kernel_name}(i8 addrspace(1)* %X, float addrspace(1)* %Scale, i8 addrspace(1)* %Zp, float addrspace(1)* %Out, i32 %M, i32 %N, i32 %Total) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bsz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %boff = mul i32 %bid, %bsz
  %gid = add i32 %boff, %tid
  ; Bounds check
  %in_bounds = icmp slt i32 %gid, %Total
  br i1 %in_bounds, label %compute, label %done

compute:
  %gid64 = sext i32 %gid to i64
  ; j = gid % N (channel index)
  %j = srem i32 %gid, %N
  %j64 = sext i32 %j to i64
  ; Load int8 value
  %xptr = getelementptr i8, i8 addrspace(1)* %X, i64 %gid64
  %x_i8 = load i8, i8 addrspace(1)* %xptr
  %x_f = sitofp i8 %x_i8 to float
  ; Load zero_point
  %zpptr = getelementptr i8, i8 addrspace(1)* %Zp, i64 %j64
  %zp_i8 = load i8, i8 addrspace(1)* %zpptr
  %zp_f = sitofp i8 %zp_i8 to float
  ; Load scale
  %sptr = getelementptr float, float addrspace(1)* %Scale, i64 %j64
  %scale = load float, float addrspace(1)* %sptr
  ; out = (x - zp) * scale
  %diff = fsub float %x_f, %zp_f
  %out_val = fmul float %diff, %scale
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %gid64
  store float %out_val, float addrspace(1)* %optr
  br label %done

done:
  ret void
}}

"""
    sig = "void (i8 addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, float addrspace(1)*, i32, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="dequantize_per_channel",
        param_names=[x_name, scale_name, zp_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], scale_name: [N], zp_name: [N], out_name: [M, N]},
        dtypes={x_name: "int8", scale_name: "float32", zp_name: "int8", out_name: "float32"},
        grid=(grid, 1, 1),
        block=(block, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", scale_name), ("ptr", zp_name),
            ("ptr", out_name), ("int", M), ("int", N), ("int", total),
        ],
    )


# ---------------------------------------------------------------------------
# 3. fused_linear_cross_entropy — X[B,D], W[V,D], labels[B] -> loss[B]
#    1 thread per sample. Compute logits = X@W^T on the fly, then CE.
# ---------------------------------------------------------------------------


def emit_llvm_ir_fused_linear_cross_entropy(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for fused linear + cross entropy (1 thread per sample)."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_name = input_names[0]
    w_name = input_names[1]
    labels_name = input_names[2]

    x_val = graph.get_value(x_name)
    w_val = graph.get_value(w_name)
    x_shape = list(x_val.shape) if x_val.shape else [32, 128]
    w_shape = list(w_val.shape) if w_val.shape else [1024, 128]
    B, D = x_shape[0], x_shape[1]
    V = w_shape[0]
    dtype = x_val.dtype or "float32"
    kernel_name = "arke_fused_linear_cross_entropy"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, i32 addrspace(1)* %Labels, float addrspace(1)* %Loss, i32 %B, i32 %D, i32 %V) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %D64 = sext i32 %D to i64
  %V64 = sext i32 %V to i64
  %x_base = mul i64 %row64, %D64
  ; Load label
  %lbl_ptr = getelementptr i32, i32 addrspace(1)* %Labels, i64 %row64
  %label = load i32, i32 addrspace(1)* %lbl_ptr
  ; We compute logits on-the-fly, finding max and label_logit
  ; Pass 1: compute max logit and label logit
  br label %logit_loop1

logit_loop1:
  %v1 = phi i32 [0, %entry], [%v1_next, %logit1_done]
  %max_logit = phi float [0xFFF0000000000000, %entry], [%new_max, %logit1_done]
  %label_logit = phi float [0.0, %entry], [%new_label_logit, %logit1_done]
  %cmp_v1 = icmp slt i32 %v1, %V
  br i1 %cmp_v1, label %dot1_init, label %pass2_init

dot1_init:
  ; Compute dot(X[row], W[v1])
  %v1_64 = sext i32 %v1 to i64
  %w_base1 = mul i64 %v1_64, %D64
  br label %dot1_loop

dot1_loop:
  %d1 = phi i32 [0, %dot1_init], [%d1_next, %dot1_body]
  %acc1 = phi float [0.0, %dot1_init], [%new_acc1, %dot1_body]
  %cmp_d1 = icmp slt i32 %d1, %D
  br i1 %cmp_d1, label %dot1_body, label %logit1_done

dot1_body:
  %d1_64 = sext i32 %d1 to i64
  %x_idx1 = add i64 %x_base, %d1_64
  %xptr1 = getelementptr float, float addrspace(1)* %X, i64 %x_idx1
  %xv1 = load float, float addrspace(1)* %xptr1
  %w_idx1 = add i64 %w_base1, %d1_64
  %wptr1 = getelementptr float, float addrspace(1)* %W, i64 %w_idx1
  %wv1 = load float, float addrspace(1)* %wptr1
  %prod1 = fmul float %xv1, %wv1
  %new_acc1 = fadd float %acc1, %prod1
  %d1_next = add i32 %d1, 1
  br label %dot1_loop

logit1_done:
  ; acc1 is the logit for vocab v1
  %new_max = call float @llvm.maxnum.f32(float %max_logit, float %acc1)
  %is_label = icmp eq i32 %v1, %label
  %new_label_logit = select i1 %is_label, float %acc1, float %label_logit
  %v1_next = add i32 %v1, 1
  br label %logit_loop1

pass2_init:
  ; Pass 2: compute sum of exp(logit - max)
  br label %logit_loop2

logit_loop2:
  %v2 = phi i32 [0, %pass2_init], [%v2_next, %logit2_done]
  %sum_exp = phi float [0.0, %pass2_init], [%new_sum, %logit2_done]
  %cmp_v2 = icmp slt i32 %v2, %V
  br i1 %cmp_v2, label %dot2_init, label %final

dot2_init:
  %v2_64 = sext i32 %v2 to i64
  %w_base2 = mul i64 %v2_64, %D64
  br label %dot2_loop

dot2_loop:
  %d2 = phi i32 [0, %dot2_init], [%d2_next, %dot2_body]
  %acc2 = phi float [0.0, %dot2_init], [%new_acc2, %dot2_body]
  %cmp_d2 = icmp slt i32 %d2, %D
  br i1 %cmp_d2, label %dot2_body, label %logit2_done

dot2_body:
  %d2_64 = sext i32 %d2 to i64
  %x_idx2 = add i64 %x_base, %d2_64
  %xptr2 = getelementptr float, float addrspace(1)* %X, i64 %x_idx2
  %xv2 = load float, float addrspace(1)* %xptr2
  %w_idx2 = add i64 %w_base2, %d2_64
  %wptr2 = getelementptr float, float addrspace(1)* %W, i64 %w_idx2
  %wv2 = load float, float addrspace(1)* %wptr2
  %prod2 = fmul float %xv2, %wv2
  %new_acc2 = fadd float %acc2, %prod2
  %d2_next = add i32 %d2, 1
  br label %dot2_loop

logit2_done:
  ; acc2 is the logit for vocab v2
  %shifted = fsub float %acc2, %max_logit
  %shifted_lg2e = fmul float %shifted, 0x3FF7154760000000
  %exp_val = call float @llvm.nvvm.ex2.approx.f(float %shifted_lg2e)
  %new_sum = fadd float %sum_exp, %exp_val
  %v2_next = add i32 %v2, 1
  br label %logit_loop2

final:
  ; loss = log(sum_exp) + max - label_logit
  %sum_exp_lg2 = call float @llvm.nvvm.lg2.approx.f(float %sum_exp)
  %log_sum = fmul float %sum_exp_lg2, 0x3FE62E4300000000
  %logsumexp = fadd float %max_logit, %log_sum
  %loss = fsub float %logsumexp, %label_logit
  %out_ptr = getelementptr float, float addrspace(1)* %Loss, i64 %row64
  store float %loss, float addrspace(1)* %out_ptr
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, i32, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="fused_linear_cross_entropy",
        param_names=[x_name, w_name, labels_name, out_name],
        output_name=out_name,
        shapes={x_name: [B, D], w_name: [V, D], labels_name: [B], out_name: [B]},
        dtypes={x_name: dtype, w_name: dtype, labels_name: "int32", out_name: dtype},
        grid=(B, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", labels_name),
            ("ptr", out_name), ("int", B), ("int", D), ("int", V),
        ],
    )


# ---------------------------------------------------------------------------
# 4. gelu_and_mul — A[M,N], B[M,N] -> out = gelu(A) * B. Flat 1D.
#    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
#    Simplified approx: gelu(x) ≈ x * sigmoid(1.702 * x) for fast version
#    We use: gelu(x) = 0.5*x*(1 + erf(x/sqrt(2))) ≈ x*sigmoid(1.702*x)
# ---------------------------------------------------------------------------


def emit_llvm_ir_gelu_and_mul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for gelu(A) * B, flat 1D grid."""
    node, input_names, out_name, a_name, M, N, dtype = _extract_2d(graph)
    b_name = input_names[1]
    kernel_name = "arke_gelu_and_mul"
    total = M * N
    block = 256
    grid = (total + block - 1) // block

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %Out, i32 %Total) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bsz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %boff = mul i32 %bid, %bsz
  %gid = add i32 %boff, %tid
  %in_bounds = icmp slt i32 %gid, %Total
  br i1 %in_bounds, label %compute, label %done

compute:
  %gid64 = sext i32 %gid to i64
  ; Load A and B
  %aptr = getelementptr float, float addrspace(1)* %A, i64 %gid64
  %a = load float, float addrspace(1)* %aptr
  %bptr = getelementptr float, float addrspace(1)* %B, i64 %gid64
  %b = load float, float addrspace(1)* %bptr
  ; gelu(a) ≈ a * sigmoid(1.702 * a)
  ; sigmoid(x) = 1 / (1 + exp(-x))
  %scaled = fmul float %a, 0x3FFB3B6460000000
  %neg_scaled = fneg float %scaled
  %neg_scaled_lg2e = fmul float %neg_scaled, 0x3FF7154760000000
  %exp_neg = call float @llvm.nvvm.ex2.approx.f(float %neg_scaled_lg2e)
  %denom = fadd float 1.0, %exp_neg
  %sig = call float @llvm.nvvm.rcp.approx.ftz.f(float %denom)
  %gelu = fmul float %a, %sig
  ; out = gelu(a) * b
  %out_val = fmul float %gelu, %b
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %gid64
  store float %out_val, float addrspace(1)* %optr
  br label %done

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="gelu_and_mul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(grid, 1, 1),
        block=(block, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name), ("int", total),
        ],
    )


# ---------------------------------------------------------------------------
# 5. quantize_per_token — X[M,N] -> Xq[M,N] (int8). 1 thread per row.
#    Find max abs, scale = maxabs/127, quantize each element.
# ---------------------------------------------------------------------------


def emit_llvm_ir_quantize_per_token(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for per-token (per-row) quantization to int8."""
    node, input_names, out_name, x_name, M, N, dtype = _extract_2d(graph)
    kernel_name = "arke_quantize_per_token"

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, i8 addrspace(1)* %Out, i32 %M, i32 %N) {{
entry:
  %row = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %row64 = sext i32 %row to i64
  %N64 = sext i32 %N to i64
  %base = mul i64 %row64, %N64
  ; Pass 1: find max absolute value
  br label %abs_loop

abs_loop:
  %j1 = phi i32 [0, %entry], [%j1_next, %abs_body]
  %max_abs = phi float [0.0, %entry], [%new_max_abs, %abs_body]
  %cmp1 = icmp slt i32 %j1, %N
  br i1 %cmp1, label %abs_body, label %abs_done

abs_body:
  %j1_64 = sext i32 %j1 to i64
  %idx1 = add i64 %base, %j1_64
  %ptr1 = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %val1 = load float, float addrspace(1)* %ptr1
  %abs_val = call float @llvm.fabs.f32(float %val1)
  %new_max_abs = call float @llvm.maxnum.f32(float %max_abs, float %abs_val)
  %j1_next = add i32 %j1, 1
  br label %abs_loop

abs_done:
  ; scale = max_abs / 127.0 (avoid div by zero with max(max_abs, 1e-10))
  %safe_max = call float @llvm.maxnum.f32(float %max_abs, float 0x3DDB7CDFE0000000)
  %scale = fdiv float %safe_max, 1.270000e+02
  %inv_scale = fdiv float 0x405FC00000000000, %safe_max
  ; Pass 2: quantize
  br label %quant_loop

quant_loop:
  %j2 = phi i32 [0, %abs_done], [%j2_next, %quant_body]
  %cmp2 = icmp slt i32 %j2, %N
  br i1 %cmp2, label %quant_body, label %done

quant_body:
  %j2_64 = sext i32 %j2 to i64
  %idx2 = add i64 %base, %j2_64
  %ptr2 = getelementptr float, float addrspace(1)* %X, i64 %idx2
  %val2 = load float, float addrspace(1)* %ptr2
  ; quantized = round(val / scale) = round(val * inv_scale)
  %scaled_val = fmul float %val2, %inv_scale
  ; Clamp to [-127, 127] and convert to i8
  %clamped_lo = call float @llvm.maxnum.f32(float %scaled_val, float 0xC05FC00000000000)
  %clamped = call float @llvm.maxnum.f32(float %clamped_lo, float %clamped_lo)
  ; Note: maxnum with +127 would be minnum; use comparison instead
  %over127 = fcmp ogt float %clamped_lo, 1.270000e+02
  %final_f = select i1 %over127, float 0x405FC00000000000, float %clamped_lo
  %q_i8 = fptosi float %final_f to i8
  %optr = getelementptr i8, i8 addrspace(1)* %Out, i64 %idx2
  store i8 %q_i8, i8 addrspace(1)* %optr
  %j2_next = add i32 %j2, 1
  br label %quant_loop

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, i8 addrspace(1)*, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="quantize_per_token",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: "float32", out_name: "int8"},
        grid=(M, 1, 1),
        block=(1, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


# ---------------------------------------------------------------------------
# 6. rope — X[B,H,S,D], cos[S,D/2], sin[S,D/2] -> out[B,H,S,D]
#    Flat 1D over B*H*S*D/2 threads.
# ---------------------------------------------------------------------------


def emit_llvm_ir_rope(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for rotary position embeddings (RoPE)."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_name = input_names[0]
    cos_name = input_names[1]
    sin_name = input_names[2]

    x_val = graph.get_value(x_name)
    shape = list(x_val.shape) if x_val.shape else [2, 8, 64, 128]
    B, H, S, D = shape[0], shape[1], shape[2], shape[3]
    Dh = D // 2
    dtype = x_val.dtype or "float32"
    kernel_name = "arke_rope"

    total = B * H * S * Dh
    block = 256
    grid_size = (total + block - 1) // block

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Cos, float addrspace(1)* %Sin, float addrspace(1)* %Out, i32 %B, i32 %H, i32 %S, i32 %D, i32 %Dh, i32 %Total) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bsz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %boff = mul i32 %bid, %bsz
  %gid = add i32 %boff, %tid
  %in_bounds = icmp slt i32 %gid, %Total
  br i1 %in_bounds, label %compute, label %done

compute:
  ; gid indexes into flattened B*H*S*Dh space
  ; d = gid % Dh
  %d = srem i32 %gid, %Dh
  ; bhs = gid / Dh
  %bhs = sdiv i32 %gid, %Dh
  ; s = bhs % S
  %s = srem i32 %bhs, %S
  ; base offset in X (4D -> linear): bhs * D
  %bhs64 = sext i32 %bhs to i64
  %D64 = sext i32 %D to i64
  %Dh64 = sext i32 %Dh to i64
  %base = mul i64 %bhs64, %D64
  %d64 = sext i32 %d to i64
  ; Load x0 = X[base + d], x1 = X[base + d + Dh]
  %idx0 = add i64 %base, %d64
  %x0ptr = getelementptr float, float addrspace(1)* %X, i64 %idx0
  %x0 = load float, float addrspace(1)* %x0ptr
  %dh_off = add i64 %d64, %Dh64
  %idx1 = add i64 %base, %dh_off
  %x1ptr = getelementptr float, float addrspace(1)* %X, i64 %idx1
  %x1 = load float, float addrspace(1)* %x1ptr
  ; Load cos[s*Dh + d] and sin[s*Dh + d]
  %s64 = sext i32 %s to i64
  %cs_base = mul i64 %s64, %Dh64
  %cs_idx = add i64 %cs_base, %d64
  %cos_ptr = getelementptr float, float addrspace(1)* %Cos, i64 %cs_idx
  %c = load float, float addrspace(1)* %cos_ptr
  %sin_ptr = getelementptr float, float addrspace(1)* %Sin, i64 %cs_idx
  %sn = load float, float addrspace(1)* %sin_ptr
  ; out[base+d] = x0*c - x1*sn
  %x0c = fmul float %x0, %c
  %x1sn = fmul float %x1, %sn
  %out0 = fsub float %x0c, %x1sn
  %out0ptr = getelementptr float, float addrspace(1)* %Out, i64 %idx0
  store float %out0, float addrspace(1)* %out0ptr
  ; out[base+d+Dh] = x1*c + x0*sn
  %x1c = fmul float %x1, %c
  %x0sn = fmul float %x0, %sn
  %out1 = fadd float %x1c, %x0sn
  %out1ptr = getelementptr float, float addrspace(1)* %Out, i64 %idx1
  store float %out1, float addrspace(1)* %out1ptr
  br label %done

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="rope",
        param_names=[x_name, cos_name, sin_name, out_name],
        output_name=out_name,
        shapes={
            x_name: [B, H, S, D],
            cos_name: [S, Dh],
            sin_name: [S, Dh],
            out_name: [B, H, S, D],
        },
        dtypes={x_name: dtype, cos_name: dtype, sin_name: dtype, out_name: dtype},
        grid=(grid_size, 1, 1),
        block=(block, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", cos_name), ("ptr", sin_name),
            ("ptr", out_name), ("int", B), ("int", H), ("int", S),
            ("int", D), ("int", Dh), ("int", total),
        ],
    )


# ---------------------------------------------------------------------------
# 7. silu_and_mul — A[M,N], B[M,N] -> out = silu(A) * B. Flat 1D.
#    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
# ---------------------------------------------------------------------------


def emit_llvm_ir_silu_and_mul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for silu(A) * B, flat 1D grid."""
    node, input_names, out_name, a_name, M, N, dtype = _extract_2d(graph)
    b_name = input_names[1]
    kernel_name = "arke_silu_and_mul"
    total = M * N
    block = 256
    grid = (total + block - 1) // block

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %Out, i32 %Total) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bsz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %boff = mul i32 %bid, %bsz
  %gid = add i32 %boff, %tid
  %in_bounds = icmp slt i32 %gid, %Total
  br i1 %in_bounds, label %compute, label %done

compute:
  %gid64 = sext i32 %gid to i64
  ; Load A and B
  %aptr = getelementptr float, float addrspace(1)* %A, i64 %gid64
  %a = load float, float addrspace(1)* %aptr
  %bptr = getelementptr float, float addrspace(1)* %B, i64 %gid64
  %b = load float, float addrspace(1)* %bptr
  ; silu(a) = a * sigmoid(a) = a / (1 + exp(-a))
  %neg_a = fneg float %a
  %neg_a_lg2e = fmul float %neg_a, 0x3FF7154760000000
  %exp_neg_a = call float @llvm.nvvm.ex2.approx.f(float %neg_a_lg2e)
  %denom = fadd float 1.0, %exp_neg_a
  %rcp_denom = call float @llvm.nvvm.rcp.approx.ftz.f(float %denom)
  %sig = fmul float %a, %rcp_denom
  ; out = silu(a) * b
  %out_val = fmul float %sig, %b
  %optr = getelementptr float, float addrspace(1)* %Out, i64 %gid64
  store float %out_val, float addrspace(1)* %optr
  br label %done

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="silu_and_mul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(grid, 1, 1),
        block=(block, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name), ("int", total),
        ],
    )


# ---------------------------------------------------------------------------
# 8. swiglu_packed — X[M,2K], W[K,N] -> out[M,N]
#    Split X into gate[M,K] and up[M,K]. out[i,n] = sum_k(silu(gate[i,k]) * up[i,k] * W[k,n])
#    1 thread per output element (i, n).
# ---------------------------------------------------------------------------


def emit_llvm_ir_swiglu_packed(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for SwiGLU with packed input + matmul."""
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_name = input_names[0]
    w_name = input_names[1]

    x_val = graph.get_value(x_name)
    w_val = graph.get_value(w_name)
    x_shape = list(x_val.shape) if x_val.shape else [64, 256]
    w_shape = list(w_val.shape) if w_val.shape else [128, 64]
    M = x_shape[0]
    TwoK = x_shape[1]
    K = TwoK // 2
    N_out = w_shape[1] if len(w_shape) > 1 else w_shape[0]
    dtype = x_val.dtype or "float32"
    kernel_name = "arke_swiglu_packed"

    total = M * N_out
    block = 256
    grid_size = (total + block - 1) // block

    source = _module_header()
    source += f"""\
define void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %W, float addrspace(1)* %Out, i32 %M, i32 %K, i32 %N, i32 %Total) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bsz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %boff = mul i32 %bid, %bsz
  %gid = add i32 %boff, %tid
  %in_bounds = icmp slt i32 %gid, %Total
  br i1 %in_bounds, label %compute, label %done

compute:
  ; gid -> (i, n): i = gid / N, n = gid % N
  %i = sdiv i32 %gid, %N
  %n = srem i32 %gid, %N
  %i64 = sext i32 %i to i64
  %n64 = sext i32 %n to i64
  %K64 = sext i32 %K to i64
  %N64 = sext i32 %N to i64
  %TwoK = mul i32 %K, 2
  %TwoK64 = sext i32 %TwoK to i64
  ; X row base: i * 2K
  %x_base = mul i64 %i64, %TwoK64
  ; Accumulate: sum_k silu(gate[k]) * up[k] * W[k,n]
  br label %k_loop

k_loop:
  %kk = phi i32 [0, %compute], [%kk_next, %k_body]
  %acc = phi float [0.0, %compute], [%new_acc, %k_body]
  %cmp_k = icmp slt i32 %kk, %K
  br i1 %cmp_k, label %k_body, label %store_out

k_body:
  %kk64 = sext i32 %kk to i64
  ; gate = X[i, kk] (first half)
  %gate_idx = add i64 %x_base, %kk64
  %gate_ptr = getelementptr float, float addrspace(1)* %X, i64 %gate_idx
  %gate = load float, float addrspace(1)* %gate_ptr
  ; up = X[i, kk + K] (second half)
  %up_off = add i64 %kk64, %K64
  %up_idx = add i64 %x_base, %up_off
  %up_ptr = getelementptr float, float addrspace(1)* %X, i64 %up_idx
  %up = load float, float addrspace(1)* %up_ptr
  ; silu(gate) = gate / (1 + exp(-gate))
  %neg_gate = fneg float %gate
  %neg_gate_lg2e = fmul float %neg_gate, 0x3FF7154760000000
  %exp_ng = call float @llvm.nvvm.ex2.approx.f(float %neg_gate_lg2e)
  %denom = fadd float 1.0, %exp_ng
  %rcp_d = call float @llvm.nvvm.rcp.approx.ftz.f(float %denom)
  %silu_gate = fmul float %gate, %rcp_d
  ; activated = silu(gate) * up
  %activated = fmul float %silu_gate, %up
  ; W[kk, n]: row-major index = kk * N + n
  %w_row_off = mul i64 %kk64, %N64
  %w_idx = add i64 %w_row_off, %n64
  %w_ptr = getelementptr float, float addrspace(1)* %W, i64 %w_idx
  %w_val = load float, float addrspace(1)* %w_ptr
  ; Accumulate
  %prod = fmul float %activated, %w_val
  %new_acc = fadd float %acc, %prod
  %kk_next = add i32 %kk, 1
  br label %k_loop

store_out:
  ; Store result at Out[i*N + n]
  %out_row = mul i64 %i64, %N64
  %out_idx = add i64 %out_row, %n64
  %out_ptr = getelementptr float, float addrspace(1)* %Out, i64 %out_idx
  store float %acc, float addrspace(1)* %out_ptr
  br label %done

done:
  ret void
}}

"""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32)"
    source += _nvvm_annotation_custom(kernel_name, sig)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="swiglu_packed",
        param_names=[x_name, w_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, TwoK], w_name: [K, N_out], out_name: [M, N_out]},
        dtypes={x_name: dtype, w_name: dtype, out_name: dtype},
        grid=(grid_size, 1, 1),
        block=(block, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", x_name), ("ptr", w_name), ("ptr", out_name),
            ("int", M), ("int", K), ("int", N_out), ("int", total),
        ],
    )
