# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR emitters for elementwise ops (Phase 5, P5-S2).

Generates LLVM IR text targeting nvptx64-nvidia-cuda for all 12 elementwise
operations: relu, gelu, silu, tanh, sigmoid, exp, neg, rsqrt, add, mul, cast,
where_.

Each emitter produces a CudaCKernel with LLVM IR source (instead of CUDA C),
ready for compilation via llc + ptxas in the LLVM backend pipeline.

Design: flat 1D grid, 256 threads/block, one element per thread.
"""

from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel
from arke.ir.graph import IRGraph


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


def _unary_kernel_ir(kernel_name: str, compute_body: str, extra_declares: str = "") -> str:
    """Build full LLVM IR for a unary elementwise kernel.

    Parameters
    ----------
    kernel_name : str
        The @kernel_name in LLVM IR.
    compute_body : str
        IR instructions between loading %x_val and storing the result.
        Must produce %result as the final float value to store.
    extra_declares : str
        Additional function declarations (e.g. libdevice intrinsics).
    """
    sig = "void (float addrspace(1)*, float addrspace(1)*, i32, i32)"
    ir = _llvm_ir_header(extra_declares)
    ir += f"\ndefine void @{kernel_name}(float addrspace(1)* %X, float addrspace(1)* %Out, i32 %M, i32 %N) {{\n"
    ir += "entry:\n"
    ir += "  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
    ir += "  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
    ir += "  %bx_scaled = mul i32 %bx, 256\n"
    ir += "  %gid = add i32 %bx_scaled, %tx\n"
    ir += "  %total = mul i32 %M, %N\n"
    ir += "  %in_bounds = icmp slt i32 %gid, %total\n"
    ir += "  br i1 %in_bounds, label %compute, label %exit\n"
    ir += "\n"
    ir += "compute:\n"
    ir += "  %x_ptr = getelementptr float, float addrspace(1)* %X, i32 %gid\n"
    ir += "  %x_val = load float, float addrspace(1)* %x_ptr\n"
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


def _binary_kernel_ir(kernel_name: str, compute_body: str, extra_declares: str = "") -> str:
    """Build full LLVM IR for a binary elementwise kernel.

    compute_body must produce %result from %a_val and %b_val.
    """
    sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32)"
    ir = _llvm_ir_header(extra_declares)
    ir += (
        f"\ndefine void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, "
        f"float addrspace(1)* %Out, i32 %M, i32 %N) {{\n"
    )
    ir += "entry:\n"
    ir += "  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
    ir += "  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
    ir += "  %bx_scaled = mul i32 %bx, 256\n"
    ir += "  %gid = add i32 %bx_scaled, %tx\n"
    ir += "  %total = mul i32 %M, %N\n"
    ir += "  %in_bounds = icmp slt i32 %gid, %total\n"
    ir += "  br i1 %in_bounds, label %compute, label %exit\n"
    ir += "\n"
    ir += "compute:\n"
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


def _ternary_kernel_ir(kernel_name: str, compute_body: str, extra_declares: str = "") -> str:
    """Build full LLVM IR for a ternary elementwise kernel (where_).

    compute_body must produce %result from %cond_val, %a_val, %b_val.
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
    ir += "  %bx_scaled = mul i32 %bx, 256\n"
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


def _grid_for(M: int, N: int, block_size: int = 256) -> int:
    """Compute grid_x for a flat 1D launch over M*N elements."""
    total = M * N
    return (total + block_size - 1) // block_size


# ---------------------------------------------------------------------------
# Individual op emitters
# ---------------------------------------------------------------------------

def emit_llvm_ir_relu(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for relu: max(x, 0)."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "relu")
    kernel_name = f"arke_relu_{M}x{N}"

    compute = (
        "  %cmp = fcmp ogt float %x_val, 0.0\n"
        "  %result = select i1 %cmp, float %x_val, float 0.0\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="relu",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_gelu(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for gelu (tanh approximation).

    gelu(x) = x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    """
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "gelu")
    kernel_name = f"arke_gelu_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.ex2.approx.f(float)"

    # Compute: gelu(x) = x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    # tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    # exp(x) = ex2.approx(x * log2(e))  where log2(e) = 1.4426950408...
    compute = (
        "  ; x^2\n"
        "  %x2 = fmul float %x_val, %x_val\n"
        "  ; x^3\n"
        "  %x3 = fmul float %x2, %x_val\n"
        "  ; 0.044715 * x^3\n"
        "  %t1 = fmul float %x3, 0x3FA6BA3A00000000\n"
        "  ; x + 0.044715 * x^3\n"
        "  %t2 = fadd float %x_val, %t1\n"
        "  ; 0.7978845608 * (x + 0.044715 * x^3)\n"
        "  %t3 = fmul float %t2, 0x3FE9884540000000\n"
        "  ; tanh(t3) via exp: (exp(2*t3) - 1) / (exp(2*t3) + 1)\n"
        "  %tanh_2x = fmul float %t3, 2.0\n"
        "  %tanh_2x_lg2e = fmul float %tanh_2x, 0x3FF7154760000000\n"
        "  %tanh_exp = call float @llvm.nvvm.ex2.approx.f(float %tanh_2x_lg2e)\n"
        "  %tanh_num = fsub float %tanh_exp, 1.0\n"
        "  %tanh_den = fadd float %tanh_exp, 1.0\n"
        "  %t4 = fdiv float %tanh_num, %tanh_den\n"
        "  ; 1.0 + tanh(...)\n"
        "  %t5 = fadd float %t4, 1.0\n"
        "  ; x * 0.5\n"
        "  %t6 = fmul float %x_val, 0.5\n"
        "  ; x * 0.5 * (1.0 + tanh(...))\n"
        "  %result = fmul float %t6, %t5\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute, extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="gelu",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_silu(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for silu: x * sigmoid(x) = x / (1 + exp(-x))."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "silu")
    kernel_name = f"arke_silu_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.ex2.approx.f(float)"

    # silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    # exp(x) = ex2.approx(x * log2(e))
    compute = (
        "  ; -x\n"
        "  %neg_x = fsub float 0.0, %x_val\n"
        "  ; exp(-x) = ex2(-x * log2(e))\n"
        "  %neg_x_lg2e = fmul float %neg_x, 0x3FF7154760000000\n"
        "  %exp_neg_x = call float @llvm.nvvm.ex2.approx.f(float %neg_x_lg2e)\n"
        "  ; 1 + exp(-x)\n"
        "  %denom = fadd float 1.0, %exp_neg_x\n"
        "  ; sigmoid(x) = 1 / (1 + exp(-x))\n"
        "  %sigmoid = fdiv float 1.0, %denom\n"
        "  ; x * sigmoid(x)\n"
        "  %result = fmul float %x_val, %sigmoid\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute, extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="silu",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_tanh(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for tanh via exp: (exp(2x)-1)/(exp(2x)+1)."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "tanh")
    kernel_name = f"arke_tanh_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.ex2.approx.f(float)"

    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # exp(x) = ex2.approx(x * log2(e))
    compute = (
        "  %tanh_2x = fmul float %x_val, 2.0\n"
        "  %tanh_2x_lg2e = fmul float %tanh_2x, 0x3FF7154760000000\n"
        "  %tanh_exp = call float @llvm.nvvm.ex2.approx.f(float %tanh_2x_lg2e)\n"
        "  %tanh_num = fsub float %tanh_exp, 1.0\n"
        "  %tanh_den = fadd float %tanh_exp, 1.0\n"
        "  %result = fdiv float %tanh_num, %tanh_den\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute, extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="tanh",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_sigmoid(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for sigmoid: 1 / (1 + exp(-x))."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "sigmoid")
    kernel_name = f"arke_sigmoid_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.ex2.approx.f(float)"

    # exp(x) = ex2.approx(x * log2(e))
    compute = (
        "  ; -x\n"
        "  %neg_x = fsub float 0.0, %x_val\n"
        "  ; exp(-x) = ex2(-x * log2(e))\n"
        "  %neg_x_lg2e = fmul float %neg_x, 0x3FF7154760000000\n"
        "  %exp_neg_x = call float @llvm.nvvm.ex2.approx.f(float %neg_x_lg2e)\n"
        "  ; 1 + exp(-x)\n"
        "  %denom = fadd float 1.0, %exp_neg_x\n"
        "  ; 1 / (1 + exp(-x))\n"
        "  %result = fdiv float 1.0, %denom\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute, extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="sigmoid",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_exp(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for exp using NVVM ex2.approx intrinsic."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "exp")
    kernel_name = f"arke_exp_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.ex2.approx.f(float)"

    # exp(x) = ex2.approx(x * log2(e))
    compute = (
        "  %x_lg2e = fmul float %x_val, 0x3FF7154760000000\n"
        "  %result = call float @llvm.nvvm.ex2.approx.f(float %x_lg2e)\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute, extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="exp",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_neg(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for neg: -x."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "neg")
    kernel_name = f"arke_neg_{M}x{N}"

    compute = (
        "  %result = fsub float 0.0, %x_val\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="neg",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_rsqrt(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for rsqrt: 1/sqrt(x) using NVVM rsqrt.approx."""
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "rsqrt")
    kernel_name = f"arke_rsqrt_{M}x{N}"

    extra_declares = "declare float @llvm.nvvm.rsqrt.approx.f(float)"

    compute = (
        "  %result = call float @llvm.nvvm.rsqrt.approx.f(float %x_val)\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute, extra_declares)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="rsqrt",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_add(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for add: a + b."""
    a_name, b_name, out_name, M, N, dtype = _extract_binary_info(graph, "add")
    kernel_name = f"arke_add_{M}x{N}"

    compute = (
        "  %result = fadd float %a_val, %b_val\n"
    )
    ir_source = _binary_kernel_ir(kernel_name, compute)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="add",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_mul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for mul: a * b."""
    a_name, b_name, out_name, M, N, dtype = _extract_binary_info(graph, "mul")
    kernel_name = f"arke_mul_{M}x{N}"

    compute = (
        "  %result = fmul float %a_val, %b_val\n"
    )
    ir_source = _binary_kernel_ir(kernel_name, compute)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="mul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: [M, N], b_name: [M, N], out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", a_name), ("ptr", b_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_cast(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for cast (identity copy for float32->float32).

    For same-type casts, this is a simple element copy. For cross-type casts,
    LLVM IR would use fpext/fptrunc/sitofp etc, but we simplify to f32->f32.
    """
    x_name, out_name, M, N, dtype = _extract_unary_info(graph, "cast")
    kernel_name = f"arke_cast_{M}x{N}"

    # Identity copy — the value passes through unchanged
    compute = (
        "  %result = fadd float %x_val, 0.0\n"
    )
    ir_source = _unary_kernel_ir(kernel_name, compute)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=ir_source,
        op_name="cast",
        param_names=[x_name, out_name],
        output_name=out_name,
        shapes={x_name: [M, N], out_name: [M, N]},
        dtypes={x_name: dtype, out_name: dtype},
        grid=(_grid_for(M, N), 1, 1),
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[("ptr", x_name), ("ptr", out_name), ("int", M), ("int", N)],
    )


def emit_llvm_ir_where(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit LLVM IR for where_: cond ? a : b (element-wise select).

    Condition is stored as float; non-zero => true.
    """
    cond_name, a_name, b_name, out_name, M, N, dtype = _extract_ternary_info(graph, "where_")
    kernel_name = f"arke_where_{M}x{N}"

    # Compare cond != 0.0 (one = ordered and not-equal)
    compute = (
        "  %cmp = fcmp one float %cond_val, 0.0\n"
        "  %result = select i1 %cmp, float %a_val, float %b_val\n"
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
        block=(256, 1, 1),
        shared_mem=0,
        kernel_args=[
            ("ptr", cond_name), ("ptr", a_name), ("ptr", b_name),
            ("ptr", out_name), ("int", M), ("int", N),
        ],
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_EMITTER_TABLE = {
    "relu": emit_llvm_ir_relu,
    "gelu": emit_llvm_ir_gelu,
    "silu": emit_llvm_ir_silu,
    "tanh": emit_llvm_ir_tanh,
    "sigmoid": emit_llvm_ir_sigmoid,
    "exp": emit_llvm_ir_exp,
    "neg": emit_llvm_ir_neg,
    "rsqrt": emit_llvm_ir_rsqrt,
    "add": emit_llvm_ir_add,
    "mul": emit_llvm_ir_mul,
    "cast": emit_llvm_ir_cast,
    "where_": emit_llvm_ir_where,
}


def emit_llvm_ir_elementwise(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Dispatch to the correct LLVM IR emitter based on the graph's op.

    Supports all 12 elementwise ops: relu, gelu, silu, tanh, sigmoid, exp,
    neg, rsqrt, add, mul, cast, where_.
    """
    node = graph.nodes[0]
    op = node.op
    if op not in _EMITTER_TABLE:
        raise ValueError(
            f"Unsupported elementwise op '{op}'. "
            f"Supported: {sorted(_EMITTER_TABLE.keys())}"
        )
    return _EMITTER_TABLE[op](graph, chip)
