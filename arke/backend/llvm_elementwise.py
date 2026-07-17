# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR emitters for elementwise ops (Phase 5, P5-S3: vectorized).

Generates LLVM IR targeting nvptx64-nvidia-cuda for all 12 elementwise
operations: relu, gelu, silu, tanh, sigmoid, exp, neg, rsqrt, add, mul, cast,
where_.

P5-S3 optimizations (over P5-S2):
  - float4 vectorized load/store (128-bit coalesced transactions)
  - Grid-stride loop (each thread processes ELEMENTS_PER_THREAD elements)
  - Block size 256 (occupancy optimal for SM 8.6)
  - rcp.approx for divisions (replaces fdiv)
  - Tail handling for non-multiple-of-4 sizes
"""

from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel
from arke.ir.graph import IRGraph


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256
ELEMENTS_PER_THREAD = 4  # float4 vectorization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llvm_ir_header(extra_declares: str = "") -> str:
    """Return the common LLVM IR module header + intrinsic declarations."""
    header = (
        'target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32'
        '-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32'
        '-v64:64:64-v128:128:128-n16:32:64"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n"
    )
    if extra_declares:
        header += extra_declares + "\n"
    return header


def _llvm_ir_footer(kernel_name: str, signature: str) -> str:
    """Return the nvvm.annotations metadata block."""
    return (
        "\n"
        "!nvvm.annotations = !{!0}\n"
        f"!0 = !{{{signature}* @{kernel_name}, !\"kernel\", i32 1}}\n"
    )


def _unary_kernel_ir_vectorized(
    kernel_name: str,
    compute_body_scalar: str,
    compute_body_vec: str | None = None,
    extra_declares: str = "",
) -> str:
    """Build full LLVM IR for a vectorized unary elementwise kernel.

    Strategy:
      1. Process bulk of elements using float4 (128-bit) loads/stores
         with grid-stride loop
      2. Handle tail elements (when total % 4 != 0) with scalar ops

    Parameters
    ----------
    kernel_name : str
        The @kernel_name in LLVM IR.
    compute_body_scalar : str
        IR for scalar computation: takes %x_val, produces %result.
    compute_body_vec : str | None
        IR for vectorized computation on 4 floats: takes %v0..%v3,
        produces %r0..%r3. If None, uses scalar compute on each element.
    extra_declares : str
        Additional function declarations.
    """
    sig = "void (float addrspace(1)*, float addrspace(1)*, i32)"

    # We pass total_elements as a single i32 (M*N pre-computed by emitter)
    ir = _llvm_ir_header(extra_declares)
    ir += f"\ndefine void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %total) {{\n"
    ir += "entry:\n"
    ir += "  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
    ir += "  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
    ir += "  %num_blocks = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n"
    ir += f"  %bx_scaled = mul i32 %bx, {BLOCK_SIZE}\n"
    ir += "  %thread_id = add i32 %bx_scaled, %tx\n"
    ir += f"  %grid_stride = mul i32 %num_blocks, {BLOCK_SIZE}\n"
    ir += "\n"

    # --- Vectorized path: process 4 floats at a time ---
    # total_vec = total / 4  (number of float4 groups)
    ir += "  %total_vec = lshr i32 %total, 2\n"

    # Cast pointers to <4 x float>* for vectorized access
    ir += "  %X_v = bitcast float addrspace(1)* %X to <4 x float> addrspace(1)*\n"
    ir += "  %Out_v = bitcast float addrspace(1)* %Out to <4 x float> addrspace(1)*\n"

    # Check if we have any vectorized work
    ir += "  %has_vec_work = icmp slt i32 %thread_id, %total_vec\n"
    ir += "  br i1 %has_vec_work, label %vec_loop, label %tail_check\n"

    # --- Vectorized loop ---
    ir += "\nvec_loop:\n"
    ir += "  %vid = phi i32 [%thread_id, %entry], [%vid_next, %vec_continue]\n"

    # Load <4 x float>
    ir += "  %xv_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %X_v, i32 %vid\n"
    ir += "  %xv = load <4 x float>, <4 x float> addrspace(1)* %xv_ptr, align 16\n"

    # Extract elements
    ir += "  %v0 = extractelement <4 x float> %xv, i32 0\n"
    ir += "  %v1 = extractelement <4 x float> %xv, i32 1\n"
    ir += "  %v2 = extractelement <4 x float> %xv, i32 2\n"
    ir += "  %v3 = extractelement <4 x float> %xv, i32 3\n"

    # Apply computation to each element
    if compute_body_vec:
        ir += compute_body_vec
    else:
        # Auto-expand scalar compute to 4 elements
        ir += _expand_scalar_to_vec4(compute_body_scalar)

    # Build result vector
    ir += "  %rv0 = insertelement <4 x float> undef, float %r0, i32 0\n"
    ir += "  %rv1 = insertelement <4 x float> %rv0, float %r1, i32 1\n"
    ir += "  %rv2 = insertelement <4 x float> %rv1, float %r2, i32 2\n"
    ir += "  %rv3 = insertelement <4 x float> %rv2, float %r3, i32 3\n"

    # Store <4 x float>
    ir += "  %ov_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %Out_v, i32 %vid\n"
    ir += "  store <4 x float> %rv3, <4 x float> addrspace(1)* %ov_ptr, align 16\n"

    # Grid-stride increment
    ir += "  %vid_next = add i32 %vid, %grid_stride\n"
    ir += "  %vec_more = icmp slt i32 %vid_next, %total_vec\n"
    ir += "  br i1 %vec_more, label %vec_continue, label %tail_check\n"

    ir += "\nvec_continue:\n"
    ir += "  br label %vec_loop\n"

    # --- Tail handling (last 0-3 elements) ---
    ir += "\ntail_check:\n"
    ir += "  %tail_start = shl i32 %total_vec, 2\n"  # total_vec * 4
    ir += "  %tail_idx = add i32 %tail_start, %tx\n"  # only first few threads handle tail
    ir += "  %in_tail = icmp slt i32 %tail_idx, %total\n"
    ir += "  br i1 %in_tail, label %tail_compute, label %exit\n"

    ir += "\ntail_compute:\n"
    ir += "  %tx_ptr = getelementptr float, float addrspace(1)* %X, i32 %tail_idx\n"
    ir += "  %x_val = load float, float addrspace(1)* %tx_ptr\n"
    ir += compute_body_scalar
    ir += "  %to_ptr = getelementptr float, float addrspace(1)* %Out, i32 %tail_idx\n"
    ir += "  store float %result, float addrspace(1)* %to_ptr\n"
    ir += "  br label %exit\n"

    ir += "\nexit:\n"
    ir += "  ret void\n"
    ir += "}\n"
    ir += _llvm_ir_footer(kernel_name, sig)
    return ir


def _expand_scalar_to_vec4(compute_body: str) -> str:
    """Expand a scalar compute body (using %x_val → %result) into 4 copies
    for %v0→%r0, %v1→%r1, %v2→%r2, %v3→%r3."""
    result = ""
    for i in range(4):
        # Replace %x_val with %vi, all SSA names get _i suffix, %result → %ri
        body = compute_body
        # First replace %result (must be before %r prefix matches)
        body = body.replace("%result", f"%r{i}")
        body = body.replace("%x_val", f"%v{i}")
        # Rename all other SSA values by appending _i
        # Find all %name patterns that aren't %v0-3 or %r0-3
        import re
        def _rename(m):
            name = m.group(1)
            if name in (f"v{i}", f"r{i}"):
                return f"%{name}"
            return f"%{name}_{i}"
        body = re.sub(r'%([a-zA-Z_][a-zA-Z0-9_]*)', _rename, body)
        result += f"  ; --- element {i} ---\n"
        result += body
    return result


def _binary_kernel_ir_vectorized(
    kernel_name: str,
    compute_body_scalar: str,
    compute_body_vec: str | None = None,
    extra_declares: str = "",
) -> str:
    """Build full LLVM IR for a vectorized binary elementwise kernel."""
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)"

    ir = _llvm_ir_header(extra_declares)
    ir += (
        f"\ndefine void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, "
        f"float addrspace(1)* %Out, i32 %total) {{\n"
    )
    ir += "entry:\n"
    ir += "  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
    ir += "  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
    ir += "  %num_blocks = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n"
    ir += f"  %bx_scaled = mul i32 %bx, {BLOCK_SIZE}\n"
    ir += "  %thread_id = add i32 %bx_scaled, %tx\n"
    ir += f"  %grid_stride = mul i32 %num_blocks, {BLOCK_SIZE}\n"
    ir += "\n"

    ir += "  %total_vec = lshr i32 %total, 2\n"
    ir += "  %A_v = bitcast float addrspace(1)* %A to <4 x float> addrspace(1)*\n"
    ir += "  %B_v = bitcast float addrspace(1)* %B to <4 x float> addrspace(1)*\n"
    ir += "  %Out_v = bitcast float addrspace(1)* %Out to <4 x float> addrspace(1)*\n"

    ir += "  %has_vec_work = icmp slt i32 %thread_id, %total_vec\n"
    ir += "  br i1 %has_vec_work, label %vec_loop, label %tail_check\n"

    # Vectorized loop
    ir += "\nvec_loop:\n"
    ir += "  %vid = phi i32 [%thread_id, %entry], [%vid_next, %vec_continue]\n"
    ir += "  %av_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %A_v, i32 %vid\n"
    ir += "  %av = load <4 x float>, <4 x float> addrspace(1)* %av_ptr, align 16\n"
    ir += "  %bv_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %B_v, i32 %vid\n"
    ir += "  %bv = load <4 x float>, <4 x float> addrspace(1)* %bv_ptr, align 16\n"

    ir += "  %a0 = extractelement <4 x float> %av, i32 0\n"
    ir += "  %a1 = extractelement <4 x float> %av, i32 1\n"
    ir += "  %a2 = extractelement <4 x float> %av, i32 2\n"
    ir += "  %a3 = extractelement <4 x float> %av, i32 3\n"
    ir += "  %b0 = extractelement <4 x float> %bv, i32 0\n"
    ir += "  %b1 = extractelement <4 x float> %bv, i32 1\n"
    ir += "  %b2 = extractelement <4 x float> %bv, i32 2\n"
    ir += "  %b3 = extractelement <4 x float> %bv, i32 3\n"

    if compute_body_vec:
        ir += compute_body_vec
    else:
        ir += _expand_binary_scalar_to_vec4(compute_body_scalar)

    ir += "  %rv0 = insertelement <4 x float> undef, float %r0, i32 0\n"
    ir += "  %rv1 = insertelement <4 x float> %rv0, float %r1, i32 1\n"
    ir += "  %rv2 = insertelement <4 x float> %rv1, float %r2, i32 2\n"
    ir += "  %rv3 = insertelement <4 x float> %rv2, float %r3, i32 3\n"
    ir += "  %ov_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %Out_v, i32 %vid\n"
    ir += "  store <4 x float> %rv3, <4 x float> addrspace(1)* %ov_ptr, align 16\n"

    ir += "  %vid_next = add i32 %vid, %grid_stride\n"
    ir += "  %vec_more = icmp slt i32 %vid_next, %total_vec\n"
    ir += "  br i1 %vec_more, label %vec_continue, label %tail_check\n"

    ir += "\nvec_continue:\n"
    ir += "  br label %vec_loop\n"

    # Tail
    ir += "\ntail_check:\n"
    ir += "  %tail_start = shl i32 %total_vec, 2\n"
    ir += "  %tail_idx = add i32 %tail_start, %tx\n"
    ir += "  %in_tail = icmp slt i32 %tail_idx, %total\n"
    ir += "  br i1 %in_tail, label %tail_compute, label %exit\n"

    ir += "\ntail_compute:\n"
    ir += "  %ta_ptr = getelementptr float, float addrspace(1)* %A, i32 %tail_idx\n"
    ir += "  %a_val = load float, float addrspace(1)* %ta_ptr\n"
    ir += "  %tb_ptr = getelementptr float, float addrspace(1)* %B, i32 %tail_idx\n"
    ir += "  %b_val = load float, float addrspace(1)* %tb_ptr\n"
    ir += compute_body_scalar
    ir += "  %to_ptr = getelementptr float, float addrspace(1)* %Out, i32 %tail_idx\n"
    ir += "  store float %result, float addrspace(1)* %to_ptr\n"
    ir += "  br label %exit\n"

    ir += "\nexit:\n"
    ir += "  ret void\n"
    ir += "}\n"
    ir += _llvm_ir_footer(kernel_name, sig)
    return ir


def _expand_binary_scalar_to_vec4(compute_body: str) -> str:
    """Expand binary scalar compute (%a_val, %b_val → %result) into 4 copies."""
    result = ""
    for i in range(4):
        body = compute_body
        body = body.replace("%result", f"%r{i}")
        body = body.replace("%a_val", f"%a{i}")
        body = body.replace("%b_val", f"%b{i}")
        import re
        def _rename(m):
            name = m.group(1)
            if name in (f"a{i}", f"b{i}", f"r{i}"):
                return f"%{name}"
            return f"%{name}_{i}"
        body = re.sub(r'%([a-zA-Z_][a-zA-Z0-9_]*)', _rename, body)
        result += f"  ; --- element {i} ---\n"
        result += body
    return result


# Also keep the legacy non-vectorized helpers for ops that need them
def _ternary_kernel_ir(kernel_name: str, compute_body: str, extra_declares: str = "") -> str:
    """Build full LLVM IR for a ternary elementwise kernel (where_).
    Non-vectorized for simplicity — where_ is rarely perf-critical.
    """
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32)"
    ir = _llvm_ir_header(extra_declares)
    ir += (
        f"\ndefine void @{kernel_name}(float addrspace(1)* %Cond, float addrspace(1)* %A, "
        f"float addrspace(1)* %B, float addrspace(1)* %Out, i32 %M, i32 %N) {{\n"
    )
    ir += "entry:\n"
    ir += "  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
    ir += "  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
    ir += f"  %bx_scaled = mul i32 %bx, {BLOCK_SIZE}\n"
    ir += "  %gid = add i32 %bx_scaled, %tx\n"
    ir += "  %total = mul i32 %M, %N\n"
    ir += "  %in_bounds = icmp slt i32 %gid, %total\n"
    ir += "  br i1 %in_bounds, label %compute, label %exit\n"
    ir += "\n"
    ir += "compute:\n"
    ir += "  %cond_ptr = getelementptr float, float addrspace(1)* %Cond, i32 %gid\n"
    ir += "  %cond_val = load float, float addrspace(1)* %cond_ptr\n"
    ir += "  %a_ptr = getelementptr float, float addrspace(1)* %A, i32 %gid\n"
    ir += "  %a_val = load float, float addrspace(1)* %a_ptr\n"
    ir += "  %b_ptr = getelementptr float, float addrspace(1)* %B, i32 %gid\n"
    ir += "  %b_val = load float, float addrspace(1)* %b_ptr\n"
    ir += compute_body
    ir += "  %out_ptr = getelementptr float, float addrspace(1)* %Out, i32 %gid\n"
    ir += "  store float %result, float addrspace(1)* %out_ptr\n"
    ir += "  br label %exit\n"
    ir += "\n"
    ir += "exit:\n"
    ir += "  ret void\n"
    ir += "}\n"
    ir += _llvm_ir_footer(kernel_name, sig)
    return ir


def _extract_unary_info(graph: IRGraph, op_name: str):
    """Extract input/output names, shape, dtype for a unary op."""
    node = graph.nodes[0]
    assert node.op == op_name, f"Expected {op_name}, got {node.op}"
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_val = graph.get_value(input_names[0])
    shape = list(x_val.shape) if x_val.shape else [64, 64]
    M, N = shape[0], shape[1] if len(shape) > 1 else 1
    dtype = x_val.dtype or "float32"
    return input_names[0], out_name, M, N, dtype


def _extract_binary_info(graph: IRGraph, op_name: str):
    """Extract input/output names, shape, dtype for a binary op."""
    node = graph.nodes[0]
    assert node.op == op_name, f"Expected {op_name}, got {node.op}"
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_val = graph.get_value(input_names[0])
    shape = list(x_val.shape) if x_val.shape else [64, 64]
    M, N = shape[0], shape[1] if len(shape) > 1 else 1
    dtype = x_val.dtype or "float32"
    return input_names[0], input_names[1], out_name, M, N, dtype


def _extract_ternary_info(graph: IRGraph, op_name: str):
    """Extract input/output names, shape, dtype for a ternary op (where_)."""
    node = graph.nodes[0]
    assert node.op == op_name, f"Expected {op_name}, got {node.op}"
    input_names = list(node.inputs.values())
    out_name = node.outputs[0]
    x_val = graph.get_value(input_names[0])
    shape = list(x_val.shape) if x_val.shape else [64, 64]
    M, N = shape[0], shape[1] if len(shape) > 1 else 1
    dtype = x_val.dtype or "float32"
    return input_names[0], input_names[1], input_names[2], out_name, M, N, dtype


def _grid_for(M: int, N: int, block_size: int = BLOCK_SIZE) -> int:
    """Compute grid_x for a flat 1D launch over M*N elements."""
    total = M * N
    return (total + block_size - 1) // block_size


def _vec_grid_for(M: int, N: int, block_size: int = BLOCK_SIZE) -> int:
    """Compute grid_x for vectorized kernel: over total/4 float4 groups."""
    total_vec = (M * N) // 4
    return (total_vec + block_size - 1) // block_size


# ---------------------------------------------------------------------------
# Individual op emitters — vectorized (P5-S3)
# ---------------------------------------------------------------------------

def emit_llvm_ir_relu(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for relu: max(x, 0)."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "relu")
    total = M * N
    kernel_name = f"arke_relu_{M}x{N}"

    compute_scalar = (
        "  %cmp = fcmp ogt float %x_val, 0.0\n"
        "  %result = select i1 %cmp, float %x_val, float 0.0\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="relu",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_gelu(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for gelu (tanh approximation).

    gelu(x) = x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    Uses rcp.approx for division instead of fdiv.
    """
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "gelu")
    total = M * N
    kernel_name = f"arke_gelu_{M}x{N}"

    extra_declares = (
        "declare float @llvm.nvvm.ex2.approx.f(float)\n"
        "declare float @llvm.nvvm.rcp.approx.ftz.f(float)"
    )

    compute_scalar = (
        "  ; x^2\n"
        "  %x2 = fmul float %x_val, %x_val\n"
        "  ; x^3\n"
        "  %x3 = fmul float %x2, %x_val\n"
        "  ; 0.044715 * x^3\n"
        "  %t1 = fmul float %x3, 0x3FA6BA3A00000000\n"
        "  ; x + 0.044715 * x^3\n"
        "  %t2 = fadd float %x_val, %t1\n"
        "  ; 0.7978845608 * (...)\n"
        "  %t3 = fmul float %t2, 0x3FE9884540000000\n"
        "  ; tanh(t3) via exp: (exp(2*t3) - 1) / (exp(2*t3) + 1)\n"
        "  %tanh_2x = fmul float %t3, 2.0\n"
        "  %tanh_2x_lg2e = fmul float %tanh_2x, 0x3FF7154760000000\n"
        "  %tanh_exp = call float @llvm.nvvm.ex2.approx.f(float %tanh_2x_lg2e)\n"
        "  %tanh_num = fsub float %tanh_exp, 1.0\n"
        "  %tanh_den = fadd float %tanh_exp, 1.0\n"
        "  %tanh_den_rcp = call float @llvm.nvvm.rcp.approx.ftz.f(float %tanh_den)\n"
        "  %t4 = fmul float %tanh_num, %tanh_den_rcp\n"
        "  ; 1.0 + tanh(...)\n"
        "  %t5 = fadd float %t4, 1.0\n"
        "  ; x * 0.5\n"
        "  %t6 = fmul float %x_val, 0.5\n"
        "  ; x * 0.5 * (1.0 + tanh(...))\n"
        "  %result = fmul float %t6, %t5\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar, extra_declares=extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="gelu",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_silu(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for silu: x * sigmoid(x) = x / (1 + exp(-x)).
    Uses rcp.approx for division.
    """
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "silu")
    total = M * N
    kernel_name = f"arke_silu_{M}x{N}"

    extra_declares = (
        "declare float @llvm.nvvm.ex2.approx.f(float)\n"
        "declare float @llvm.nvvm.rcp.approx.ftz.f(float)"
    )

    compute_scalar = (
        "  ; -x\n"
        "  %neg_x = fsub float 0.0, %x_val\n"
        "  ; exp(-x) = ex2(-x * log2(e))\n"
        "  %neg_x_lg2e = fmul float %neg_x, 0x3FF7154760000000\n"
        "  %exp_neg_x = call float @llvm.nvvm.ex2.approx.f(float %neg_x_lg2e)\n"
        "  ; 1 + exp(-x)\n"
        "  %denom = fadd float 1.0, %exp_neg_x\n"
        "  ; sigmoid(x) = 1 / (1 + exp(-x)) via rcp.approx\n"
        "  %sigmoid = call float @llvm.nvvm.rcp.approx.ftz.f(float %denom)\n"
        "  ; x * sigmoid(x)\n"
        "  %result = fmul float %x_val, %sigmoid\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar, extra_declares=extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="silu",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_tanh(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for tanh: (exp(2x)-1)/(exp(2x)+1)."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "tanh")
    total = M * N
    kernel_name = f"arke_tanh_{M}x{N}"

    extra_declares = (
        "declare float @llvm.nvvm.ex2.approx.f(float)\n"
        "declare float @llvm.nvvm.rcp.approx.ftz.f(float)"
    )

    compute_scalar = (
        "  %tanh_2x = fmul float %x_val, 2.0\n"
        "  %tanh_2x_lg2e = fmul float %tanh_2x, 0x3FF7154760000000\n"
        "  %tanh_exp = call float @llvm.nvvm.ex2.approx.f(float %tanh_2x_lg2e)\n"
        "  %tanh_num = fsub float %tanh_exp, 1.0\n"
        "  %tanh_den = fadd float %tanh_exp, 1.0\n"
        "  %tanh_rcp = call float @llvm.nvvm.rcp.approx.ftz.f(float %tanh_den)\n"
        "  %result = fmul float %tanh_num, %tanh_rcp\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar, extra_declares=extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="tanh",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_sigmoid(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for sigmoid: 1 / (1 + exp(-x)).
    Uses rcp.approx for division.
    """
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "sigmoid")
    total = M * N
    kernel_name = f"arke_sigmoid_{M}x{N}"

    extra_declares = (
        "declare float @llvm.nvvm.ex2.approx.f(float)\n"
        "declare float @llvm.nvvm.rcp.approx.ftz.f(float)"
    )

    compute_scalar = (
        "  ; -x\n"
        "  %neg_x = fsub float 0.0, %x_val\n"
        "  ; exp(-x) = ex2(-x * log2(e))\n"
        "  %neg_x_lg2e = fmul float %neg_x, 0x3FF7154760000000\n"
        "  %exp_neg_x = call float @llvm.nvvm.ex2.approx.f(float %neg_x_lg2e)\n"
        "  ; 1 + exp(-x)\n"
        "  %denom = fadd float 1.0, %exp_neg_x\n"
        "  ; 1 / (1 + exp(-x)) via rcp.approx\n"
        "  %result = call float @llvm.nvvm.rcp.approx.ftz.f(float %denom)\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar, extra_declares=extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="sigmoid",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_exp(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for exp using NVVM ex2.approx intrinsic."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "exp")
    total = M * N
    kernel_name = f"arke_exp_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.ex2.approx.f(float)"

    compute_scalar = (
        "  %x_lg2e = fmul float %x_val, 0x3FF7154760000000\n"
        "  %result = call float @llvm.nvvm.ex2.approx.f(float %x_lg2e)\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar, extra_declares=extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="exp",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_neg(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for neg: -x."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "neg")
    total = M * N
    kernel_name = f"arke_neg_{M}x{N}"

    compute_scalar = (
        "  %result = fsub float 0.0, %x_val\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="neg",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_rsqrt(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for rsqrt: 1/sqrt(x) using NVVM rsqrt.approx."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "rsqrt")
    total = M * N
    kernel_name = f"arke_rsqrt_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.rsqrt.approx.f(float)"

    compute_scalar = (
        "  %result = call float @llvm.nvvm.rsqrt.approx.f(float %x_val)\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar, extra_declares=extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="rsqrt",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_add(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for add: a + b."""
    a_name, b_name, out_name, M, N, dtype = _extract_binary_info(graph, "add")
    total = M * N
    kernel_name = f"arke_add_{M}x{N}"

    compute_scalar = (
        "  %result = fadd float %a_val, %b_val\n"
    )
    ir_source = _binary_kernel_ir_vectorized(kernel_name, compute_scalar)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="add",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_mul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit vectorized LLVM IR for mul: a * b."""
    a_name, b_name, out_name, M, N, dtype = _extract_binary_info(graph, "mul")
    total = M * N
    kernel_name = f"arke_mul_{M}x{N}"

    compute_scalar = (
        "  %result = fmul float %a_val, %b_val\n"
    )
    ir_source = _binary_kernel_ir_vectorized(kernel_name, compute_scalar)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="mul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_cast(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for cast (identity for float32→float32, the common case)."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "cast")
    total = M * N
    kernel_name = f"arke_cast_{M}x{N}"

    compute_scalar = (
        "  %result = fadd float %x_val, 0.0\n"
    )
    ir_source = _unary_kernel_ir_vectorized(kernel_name, compute_scalar)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="cast",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_vec_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", total)],
    )


def emit_llvm_ir_where(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for where: out = cond ? a : b."""
    cond_name, a_name, b_name, out_name, M, N, dtype = _extract_ternary_info(graph, "where_")
    kernel_name = f"arke_where_{M}x{N}"

    compute = (
        "  %is_true = fcmp one float %cond_val, 0.0\n"
        "  %result = select i1 %is_true, float %a_val, float %b_val\n"
    )
    ir_source = _ternary_kernel_ir(kernel_name, compute)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="where_",
        param_names=[cond_name, a_name, b_name, out_name],
        output_name=out_name,
        shapes={cond_name: [M, N], a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={cond_name: dtype, a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", cond_name), ("ptr", a_name), ("ptr", b_name),
            ("ptr", out_name), ("int", M), ("int", N)
        ],
    )
