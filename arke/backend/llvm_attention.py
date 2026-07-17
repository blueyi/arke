# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR emitters for attention ops (Phase 5, P5-S2).

Generates LLVM IR text targeting nvptx64-nvidia-cuda for 5 attention
operations:
  flash_attention, grouped_query_attention, cross_attention,
  paged_attention, multi_latent_attention

All attention ops use a simple 1-thread-per-query-row implementation
with online softmax. This is a correctness-first approach; performance
optimization via tiling/shared memory is left to future stages.
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
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0()
declare float @llvm.nvvm.ex2.approx.f(float)
declare float @llvm.maxnum.f32(float, float)
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
# 1. flash_attention: Q,K,V [B,H,S,D] -> O [B,H,S,D]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_flash_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit flash_attention kernel with online softmax.

    Q, K, V, O all [B,H,S,D].
    1 thread per (b,h,s) query row. Sequential attention over all key rows.
    Accumulator stored in alloca'd local array of size D.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    q_name, k_name, v_name = input_names[0], input_names[1], input_names[2]
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    q_shape = list(q_val.shape) if q_val.shape else [2, 8, 64, 64]

    B, H, S, D = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
    dtype = q_val.dtype or "float32"
    total = B * H * S
    scale = 1.0 / math.sqrt(D)

    kernel_name = f"arke_flash_attention_{B}x{H}x{S}x{D}"
    grid, block = _flat_grid(total)

    source = _gen_flash_attention_ir(kernel_name, B, H, S, D, scale)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="flash_attention",
        param_names=[q_name, k_name, v_name, out_name],
        output_name=out_name,
        shapes={
            q_name: q_shape, k_name: q_shape,
            v_name: q_shape, out_name: q_shape,
        },
        dtypes={q_name: dtype, k_name: dtype, v_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
            ("int", B), ("int", H), ("int", S), ("int", D),
        ],
    )


def _gen_flash_attention_ir(
    kernel_name: str, B: int, H: int, S: int, D: int, scale: float
) -> str:
    """Generate flash attention LLVM IR with online softmax.

    Each thread handles one query row (b,h,q). It:
    1. Allocates local acc[D] array
    2. Loops over all S key rows, computing dot-product scores
    3. Maintains online softmax (m, l) and accumulates weighted V
    4. Writes final O = acc / l
    """
    total = B * H * S
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %Q, float addrspace(1)* %K, float addrspace(1)* %V, float addrspace(1)* %O, i32 %B_size, i32 %H_size, i32 %S_size, i32 %D_size) {{")
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
    # gid -> (b, h, q): gid = b*H*S + h*S + q
    ln(f"  %hs = add i32 0, {H * S}")
    ln("  %b_idx = sdiv i32 %gid, %hs")
    ln("  %rem_hs = srem i32 %gid, %hs")
    ln(f"  %s_val = add i32 0, {S}")
    ln("  %h_idx = sdiv i32 %rem_hs, %s_val")
    ln("  %q_idx = srem i32 %rem_hs, %s_val")
    # Base offset for Q/K/V: [B,H,S,D] -> offset = ((b*H + h)*S + s)*D
    ln(f"  %d_val = add i32 0, {D}")
    ln(f"  %hsd = add i32 0, {H * S * D}")
    ln(f"  %sd = add i32 0, {S * D}")
    ln("  %base_bh = mul i32 %b_idx, %hsd")
    ln("  %base_h = mul i32 %h_idx, %sd")
    ln("  %base_bh2 = add i32 %base_bh, %base_h")
    ln("  %q_base = mul i32 %q_idx, %d_val")
    ln("  %q_off = add i32 %base_bh2, %q_base")
    # Alloca for accumulator array [D x float]
    ln(f"  %acc_arr = alloca [{D} x float]")
    # Initialize acc to 0
    ln("  br label %init_header")
    ln("")
    ln("init_header:")
    ln("  %init_d = phi i32 [0, %compute], [%init_d_next, %init_body]")
    ln(f"  %init_done = icmp sge i32 %init_d, {D}")
    ln("  br i1 %init_done, label %kv_loop_header, label %init_body")
    ln("")
    ln("init_body:")
    ln(f"  %init_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %init_d")
    ln("  store float 0.0, float* %init_ptr")
    ln("  %init_d_next = add i32 %init_d, 1")
    ln("  br label %init_header")
    ln("")
    # Main KV loop: for each key row j = 0..S-1
    ln("kv_loop_header:")
    ln("  %j = phi i32 [0, %init_header], [%j_next, %update_acc_exit]")
    ln("  %m_val = phi float [0xC6293E5940000000, %init_header], [%m_new, %update_acc_exit]")
    ln("  %l_val = phi float [0.0, %init_header], [%l_new, %update_acc_exit]")
    ln(f"  %j_done = icmp sge i32 %j, {S}")
    ln("  br i1 %j_done, label %write_output, label %dot_product_header")
    ln("")
    # Compute dot product: score = sum_d Q[q_off+d] * K[k_off+d]
    # k_off = base_bh2 + j*D
    ln("dot_product_header:")
    ln("  %k_row_off = mul i32 %j, %d_val")
    ln("  %k_base = add i32 %base_bh2, %k_row_off")
    ln("  br label %dot_loop_header")
    ln("")
    ln("dot_loop_header:")
    ln("  %dd = phi i32 [0, %dot_product_header], [%dd_next, %dot_loop_body]")
    ln("  %dot_acc = phi float [0.0, %dot_product_header], [%dot_acc_new, %dot_loop_body]")
    ln(f"  %dd_done = icmp sge i32 %dd, {D}")
    ln("  br i1 %dd_done, label %softmax_update, label %dot_loop_body")
    ln("")
    ln("dot_loop_body:")
    # Load Q[q_off + dd]
    ln("  %q_d_off = add i32 %q_off, %dd")
    ln("  %q_ptr = getelementptr float, float addrspace(1)* %Q, i32 %q_d_off")
    ln("  %q_elem = load float, float addrspace(1)* %q_ptr")
    # Load K[k_base + dd]
    ln("  %k_d_off = add i32 %k_base, %dd")
    ln("  %k_ptr = getelementptr float, float addrspace(1)* %K, i32 %k_d_off")
    ln("  %k_elem = load float, float addrspace(1)* %k_ptr")
    ln("  %qk_prod = fmul float %q_elem, %k_elem")
    ln("  %dot_acc_new = fadd float %dot_acc, %qk_prod")
    ln("  %dd_next = add i32 %dd, 1")
    ln("  br label %dot_loop_header")
    ln("")
    # Online softmax update
    ln("softmax_update:")
    ln(f"  %score = fmul float %dot_acc, {scale:#.8e}")
    ln("  %m_new = call float @llvm.maxnum.f32(float %m_val, float %score)")
    # corr = exp(m_old - m_new), p = exp(score - m_new)
    ln("  %m_diff = fsub float %m_val, %m_new")
    ln("  %corr_lg2e = fmul float %m_diff, 0x3FF7154760000000")

    ln("  %corr = call float @llvm.nvvm.ex2.approx.f(float %corr_lg2e)")
    ln("  %score_diff = fsub float %score, %m_new")
    ln("  %p_lg2e = fmul float %score_diff, 0x3FF7154760000000")

    ln("  %p = call float @llvm.nvvm.ex2.approx.f(float %p_lg2e)")
    # l_new = l * corr + p
    ln("  %l_corr = fmul float %l_val, %corr")
    ln("  %l_new = fadd float %l_corr, %p")
    # Update acc: acc[d] = acc[d]*corr + p*V[v_base+dd]
    # v_base = base_bh2 + j*D (same layout as K)
    ln("  %v_base = add i32 %base_bh2, %k_row_off")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_header:")
    ln("  %ud = phi i32 [0, %softmax_update], [%ud_next, %update_acc_body]")
    ln(f"  %ud_done = icmp sge i32 %ud, {D}")
    ln("  br i1 %ud_done, label %update_acc_exit, label %update_acc_body")
    ln("")
    ln("update_acc_body:")
    # Load acc[ud]
    ln(f"  %acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %ud")
    ln("  %acc_old = load float, float* %acc_ptr")
    # acc_old * corr
    ln("  %acc_corr = fmul float %acc_old, %corr")
    # Load V[v_base + ud]
    ln("  %v_d_off = add i32 %v_base, %ud")
    ln("  %v_ptr = getelementptr float, float addrspace(1)* %V, i32 %v_d_off")
    ln("  %v_elem = load float, float addrspace(1)* %v_ptr")
    # p * V[d]
    ln("  %pv = fmul float %p, %v_elem")
    # acc_new = acc_corr + pv
    ln("  %acc_updated = fadd float %acc_corr, %pv")
    ln("  store float %acc_updated, float* %acc_ptr")
    ln("  %ud_next = add i32 %ud, 1")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_exit:")
    ln("  %j_next = add i32 %j, 1")
    ln("  br label %kv_loop_header")
    ln("")
    # Write output: O[q_off + d] = acc[d] / l
    ln("write_output:")
    ln("  br label %write_header")
    ln("")
    ln("write_header:")
    ln("  %wd = phi i32 [0, %write_output], [%wd_next, %write_body]")
    ln(f"  %wd_done = icmp sge i32 %wd, {D}")
    ln("  br i1 %wd_done, label %done, label %write_body")
    ln("")
    ln("write_body:")
    ln(f"  %w_acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %wd")
    ln("  %w_acc_val = load float, float* %w_acc_ptr")
    ln("  %o_val = fdiv float %w_acc_val, %l_val")
    ln("  %o_d_off = add i32 %q_off, %wd")
    ln("  %o_ptr = getelementptr float, float addrspace(1)* %O, i32 %o_d_off")
    ln("  store float %o_val, float addrspace(1)* %o_ptr")
    ln("  %wd_next = add i32 %wd, 1")
    ln("  br label %write_header")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. grouped_query_attention: Q[B,Hq,S,D], K/V[B,Hkv,S,D] -> O[B,Hq,S,D]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_grouped_query_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit GQA kernel. Same as flash_attention but K/V head = hq // (Hq//Hkv).

    Q[B,Hq,S,D], K[B,Hkv,S,D], V[B,Hkv,S,D] -> O[B,Hq,S,D].
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    q_name, k_name, v_name = input_names[0], input_names[1], input_names[2]
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    k_val = graph.get_value(k_name)
    q_shape = list(q_val.shape) if q_val.shape else [2, 32, 64, 64]
    k_shape = list(k_val.shape) if k_val.shape else [2, 8, 64, 64]

    B, Hq, S, D = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
    Hkv = k_shape[1]
    dtype = q_val.dtype or "float32"
    total = B * Hq * S
    scale = 1.0 / math.sqrt(D)
    heads_per_group = Hq // Hkv if Hkv > 0 else 1

    kernel_name = f"arke_gqa_{B}x{Hq}x{Hkv}x{S}x{D}"
    grid, block = _flat_grid(total)

    source = _gen_gqa_ir(kernel_name, B, Hq, Hkv, S, D, scale, heads_per_group)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="grouped_query_attention",
        param_names=[q_name, k_name, v_name, out_name],
        output_name=out_name,
        shapes={
            q_name: q_shape, k_name: k_shape,
            v_name: k_shape, out_name: q_shape,
        },
        dtypes={q_name: dtype, k_name: dtype, v_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
            ("int", B), ("int", Hq), ("int", Hkv), ("int", S), ("int", D),
        ],
    )


def _gen_gqa_ir(
    kernel_name: str, B: int, Hq: int, Hkv: int, S: int, D: int,
    scale: float, heads_per_group: int
) -> str:
    """Generate GQA LLVM IR. Same as flash_attention but K/V use kv_head index."""
    total = B * Hq * S
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %Q, float addrspace(1)* %K, float addrspace(1)* %V, float addrspace(1)* %O, i32 %B_size, i32 %Hq, i32 %Hkv, i32 %S_size, i32 %D_size) {{")
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
    # gid -> (b, hq, q): gid = b*Hq*S + hq*S + q
    ln(f"  %hqs = add i32 0, {Hq * S}")
    ln("  %b_idx = sdiv i32 %gid, %hqs")
    ln("  %rem_hqs = srem i32 %gid, %hqs")
    ln(f"  %s_val = add i32 0, {S}")
    ln("  %hq_idx = sdiv i32 %rem_hqs, %s_val")
    ln("  %q_idx = srem i32 %rem_hqs, %s_val")
    # kv_head = hq_idx / heads_per_group
    ln(f"  %kv_head = sdiv i32 %hq_idx, {heads_per_group}")
    # Q base: offset = ((b*Hq + hq)*S + q)*D
    ln(f"  %d_val = add i32 0, {D}")
    ln(f"  %hq_sd = add i32 0, {Hq * S * D}")
    ln(f"  %q_sd = add i32 0, {S * D}")
    ln("  %q_b_off = mul i32 %b_idx, %hq_sd")
    ln("  %q_h_off = mul i32 %hq_idx, %q_sd")
    ln("  %q_bh = add i32 %q_b_off, %q_h_off")
    ln("  %q_s_off = mul i32 %q_idx, %d_val")
    ln("  %q_off = add i32 %q_bh, %q_s_off")
    # K/V base: offset = ((b*Hkv + kv_head)*S)*D
    ln(f"  %kv_hsd = add i32 0, {Hkv * S * D}")
    ln(f"  %kv_sd = add i32 0, {S * D}")
    ln("  %kv_b_off = mul i32 %b_idx, %kv_hsd")
    ln("  %kv_h_off = mul i32 %kv_head, %kv_sd")
    ln("  %kv_bh = add i32 %kv_b_off, %kv_h_off")
    # Alloca for acc
    ln(f"  %acc_arr = alloca [{D} x float]")
    ln("  br label %init_header")
    ln("")
    ln("init_header:")
    ln("  %init_d = phi i32 [0, %compute], [%init_d_next, %init_body]")
    ln(f"  %init_done = icmp sge i32 %init_d, {D}")
    ln("  br i1 %init_done, label %kv_loop_header, label %init_body")
    ln("")
    ln("init_body:")
    ln(f"  %init_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %init_d")
    ln("  store float 0.0, float* %init_ptr")
    ln("  %init_d_next = add i32 %init_d, 1")
    ln("  br label %init_header")
    ln("")
    # KV loop
    ln("kv_loop_header:")
    ln("  %j = phi i32 [0, %init_header], [%j_next, %update_acc_exit]")
    ln("  %m_val2 = phi float [0xC6293E5940000000, %init_header], [%m_new, %update_acc_exit]")
    ln("  %l_val2 = phi float [0.0, %init_header], [%l_new, %update_acc_exit]")
    ln(f"  %j_done = icmp sge i32 %j, {S}")
    ln("  br i1 %j_done, label %write_output, label %dot_product_header")
    ln("")
    # Dot product
    ln("dot_product_header:")
    ln("  %k_row_off = mul i32 %j, %d_val")
    ln("  %k_base = add i32 %kv_bh, %k_row_off")
    ln("  br label %dot_loop_header")
    ln("")
    ln("dot_loop_header:")
    ln("  %dd = phi i32 [0, %dot_product_header], [%dd_next, %dot_loop_body]")
    ln("  %dot_acc = phi float [0.0, %dot_product_header], [%dot_acc_new, %dot_loop_body]")
    ln(f"  %dd_done = icmp sge i32 %dd, {D}")
    ln("  br i1 %dd_done, label %softmax_update, label %dot_loop_body")
    ln("")
    ln("dot_loop_body:")
    ln("  %q_d_off = add i32 %q_off, %dd")
    ln("  %q_ptr = getelementptr float, float addrspace(1)* %Q, i32 %q_d_off")
    ln("  %q_elem = load float, float addrspace(1)* %q_ptr")
    ln("  %k_d_off = add i32 %k_base, %dd")
    ln("  %k_ptr = getelementptr float, float addrspace(1)* %K, i32 %k_d_off")
    ln("  %k_elem = load float, float addrspace(1)* %k_ptr")
    ln("  %qk_prod = fmul float %q_elem, %k_elem")
    ln("  %dot_acc_new = fadd float %dot_acc, %qk_prod")
    ln("  %dd_next = add i32 %dd, 1")
    ln("  br label %dot_loop_header")
    ln("")
    # Softmax update
    ln("softmax_update:")
    ln(f"  %score = fmul float %dot_acc, {scale:#.8e}")
    ln("  %m_new = call float @llvm.maxnum.f32(float %m_val2, float %score)")
    ln("  %m_diff = fsub float %m_val2, %m_new")
    ln("  %corr_lg2e = fmul float %m_diff, 0x3FF7154760000000")

    ln("  %corr = call float @llvm.nvvm.ex2.approx.f(float %corr_lg2e)")
    ln("  %score_diff = fsub float %score, %m_new")
    ln("  %p_lg2e = fmul float %score_diff, 0x3FF7154760000000")

    ln("  %p = call float @llvm.nvvm.ex2.approx.f(float %p_lg2e)")
    ln("  %l_corr = fmul float %l_val2, %corr")
    ln("  %l_new = fadd float %l_corr, %p")
    # V base same as K base
    ln("  %v_base = add i32 %kv_bh, %k_row_off")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_header:")
    ln("  %ud = phi i32 [0, %softmax_update], [%ud_next, %update_acc_body]")
    ln(f"  %ud_done = icmp sge i32 %ud, {D}")
    ln("  br i1 %ud_done, label %update_acc_exit, label %update_acc_body")
    ln("")
    ln("update_acc_body:")
    ln(f"  %acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %ud")
    ln("  %acc_old = load float, float* %acc_ptr")
    ln("  %acc_corr = fmul float %acc_old, %corr")
    ln("  %v_d_off = add i32 %v_base, %ud")
    ln("  %v_ptr = getelementptr float, float addrspace(1)* %V, i32 %v_d_off")
    ln("  %v_elem = load float, float addrspace(1)* %v_ptr")
    ln("  %pv = fmul float %p, %v_elem")
    ln("  %acc_updated = fadd float %acc_corr, %pv")
    ln("  store float %acc_updated, float* %acc_ptr")
    ln("  %ud_next = add i32 %ud, 1")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_exit:")
    ln("  %j_next = add i32 %j, 1")
    ln("  br label %kv_loop_header")
    ln("")
    # Write output
    ln("write_output:")
    ln("  br label %write_header")
    ln("")
    ln("write_header:")
    ln("  %wd = phi i32 [0, %write_output], [%wd_next, %write_body]")
    ln(f"  %wd_done = icmp sge i32 %wd, {D}")
    ln("  br i1 %wd_done, label %done, label %write_body")
    ln("")
    ln("write_body:")
    ln(f"  %w_acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %wd")
    ln("  %w_acc_val = load float, float* %w_acc_ptr")
    ln("  %o_val = fdiv float %w_acc_val, %l_val2")
    ln("  %o_d_off = add i32 %q_off, %wd")
    ln("  %o_ptr = getelementptr float, float addrspace(1)* %O, i32 %o_d_off")
    ln("  store float %o_val, float addrspace(1)* %o_ptr")
    ln("  %wd_next = add i32 %wd, 1")
    ln("  br label %write_header")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. cross_attention: Q[B,H,Sq,D], K/V[B,H,Skv,D] -> O[B,H,Sq,D]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_cross_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit cross_attention: Q[B,H,Sq,D], K/V[B,H,Skv,D] -> O[B,H,Sq,D].

    Same as flash_attention but K/V sequence length (Skv) differs from Q (Sq).
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    q_name, k_name, v_name = input_names[0], input_names[1], input_names[2]
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    k_val = graph.get_value(k_name)
    q_shape = list(q_val.shape) if q_val.shape else [2, 8, 64, 64]
    k_shape = list(k_val.shape) if k_val.shape else [2, 8, 128, 64]

    B, H, Sq, D = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
    Skv = k_shape[2]
    dtype = q_val.dtype or "float32"
    total = B * H * Sq
    scale = 1.0 / math.sqrt(D)

    kernel_name = f"arke_cross_attention_{B}x{H}x{Sq}x{Skv}x{D}"
    grid, block = _flat_grid(total)

    source = _gen_cross_attention_ir(kernel_name, B, H, Sq, Skv, D, scale)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="cross_attention",
        param_names=[q_name, k_name, v_name, out_name],
        output_name=out_name,
        shapes={
            q_name: q_shape, k_name: k_shape,
            v_name: k_shape, out_name: q_shape,
        },
        dtypes={q_name: dtype, k_name: dtype, v_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", q_name), ("ptr", k_name), ("ptr", v_name), ("ptr", out_name),
            ("int", B), ("int", H), ("int", Sq), ("int", Skv), ("int", D),
        ],
    )


def _gen_cross_attention_ir(
    kernel_name: str, B: int, H: int, Sq: int, Skv: int, D: int, scale: float
) -> str:
    """Cross attention: Q loops over Skv key rows instead of Sq."""
    total = B * H * Sq
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %Q, float addrspace(1)* %K, float addrspace(1)* %V, float addrspace(1)* %O, i32 %B_size, i32 %H_size, i32 %Sq, i32 %Skv, i32 %D_size) {{")
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
    # gid -> (b, h, q): gid = b*H*Sq + h*Sq + q
    ln(f"  %h_sq = add i32 0, {H * Sq}")
    ln("  %b_idx = sdiv i32 %gid, %h_sq")
    ln("  %rem_hsq = srem i32 %gid, %h_sq")
    ln(f"  %sq_val = add i32 0, {Sq}")
    ln("  %h_idx = sdiv i32 %rem_hsq, %sq_val")
    ln("  %q_idx = srem i32 %rem_hsq, %sq_val")
    # Q offset: ((b*H + h)*Sq + q)*D
    ln(f"  %d_val = add i32 0, {D}")
    ln(f"  %q_hsd = add i32 0, {H * Sq * D}")
    ln(f"  %q_sd = add i32 0, {Sq * D}")
    ln("  %q_b_off = mul i32 %b_idx, %q_hsd")
    ln("  %q_h_off = mul i32 %h_idx, %q_sd")
    ln("  %q_bh = add i32 %q_b_off, %q_h_off")
    ln("  %q_s_off = mul i32 %q_idx, %d_val")
    ln("  %q_off = add i32 %q_bh, %q_s_off")
    # K/V offset base: ((b*H + h)*Skv)*D -- loop over j for key rows
    ln(f"  %kv_hsd = add i32 0, {H * Skv * D}")
    ln(f"  %kv_sd = add i32 0, {Skv * D}")
    ln("  %kv_b_off = mul i32 %b_idx, %kv_hsd")
    ln("  %kv_h_off = mul i32 %h_idx, %kv_sd")
    ln("  %kv_bh = add i32 %kv_b_off, %kv_h_off")
    # Alloca
    ln(f"  %acc_arr = alloca [{D} x float]")
    ln("  br label %init_header")
    ln("")
    ln("init_header:")
    ln("  %init_d = phi i32 [0, %compute], [%init_d_next, %init_body]")
    ln(f"  %init_done = icmp sge i32 %init_d, {D}")
    ln("  br i1 %init_done, label %kv_loop_header, label %init_body")
    ln("")
    ln("init_body:")
    ln(f"  %init_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %init_d")
    ln("  store float 0.0, float* %init_ptr")
    ln("  %init_d_next = add i32 %init_d, 1")
    ln("  br label %init_header")
    ln("")
    # KV loop over Skv
    ln("kv_loop_header:")
    ln("  %j = phi i32 [0, %init_header], [%j_next, %update_acc_exit]")
    ln("  %m_val2 = phi float [0xC6293E5940000000, %init_header], [%m_new, %update_acc_exit]")
    ln("  %l_val2 = phi float [0.0, %init_header], [%l_new, %update_acc_exit]")
    ln(f"  %j_done = icmp sge i32 %j, {Skv}")
    ln("  br i1 %j_done, label %write_output, label %dot_product_header")
    ln("")
    ln("dot_product_header:")
    ln("  %k_row_off = mul i32 %j, %d_val")
    ln("  %k_base = add i32 %kv_bh, %k_row_off")
    ln("  br label %dot_loop_header")
    ln("")
    ln("dot_loop_header:")
    ln("  %dd = phi i32 [0, %dot_product_header], [%dd_next, %dot_loop_body]")
    ln("  %dot_acc = phi float [0.0, %dot_product_header], [%dot_acc_new, %dot_loop_body]")
    ln(f"  %dd_done = icmp sge i32 %dd, {D}")
    ln("  br i1 %dd_done, label %softmax_update, label %dot_loop_body")
    ln("")
    ln("dot_loop_body:")
    ln("  %q_d_off = add i32 %q_off, %dd")
    ln("  %q_ptr = getelementptr float, float addrspace(1)* %Q, i32 %q_d_off")
    ln("  %q_elem = load float, float addrspace(1)* %q_ptr")
    ln("  %k_d_off = add i32 %k_base, %dd")
    ln("  %k_ptr = getelementptr float, float addrspace(1)* %K, i32 %k_d_off")
    ln("  %k_elem = load float, float addrspace(1)* %k_ptr")
    ln("  %qk_prod = fmul float %q_elem, %k_elem")
    ln("  %dot_acc_new = fadd float %dot_acc, %qk_prod")
    ln("  %dd_next = add i32 %dd, 1")
    ln("  br label %dot_loop_header")
    ln("")
    ln("softmax_update:")
    ln(f"  %score = fmul float %dot_acc, {scale:#.8e}")
    ln("  %m_new = call float @llvm.maxnum.f32(float %m_val2, float %score)")
    ln("  %m_diff = fsub float %m_val2, %m_new")
    ln("  %corr_lg2e = fmul float %m_diff, 0x3FF7154760000000")

    ln("  %corr = call float @llvm.nvvm.ex2.approx.f(float %corr_lg2e)")
    ln("  %score_diff = fsub float %score, %m_new")
    ln("  %p_lg2e = fmul float %score_diff, 0x3FF7154760000000")

    ln("  %p = call float @llvm.nvvm.ex2.approx.f(float %p_lg2e)")
    ln("  %l_corr = fmul float %l_val2, %corr")
    ln("  %l_new = fadd float %l_corr, %p")
    ln("  %v_base = add i32 %kv_bh, %k_row_off")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_header:")
    ln("  %ud = phi i32 [0, %softmax_update], [%ud_next, %update_acc_body]")
    ln(f"  %ud_done = icmp sge i32 %ud, {D}")
    ln("  br i1 %ud_done, label %update_acc_exit, label %update_acc_body")
    ln("")
    ln("update_acc_body:")
    ln(f"  %acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %ud")
    ln("  %acc_old = load float, float* %acc_ptr")
    ln("  %acc_corr = fmul float %acc_old, %corr")
    ln("  %v_d_off = add i32 %v_base, %ud")
    ln("  %v_ptr = getelementptr float, float addrspace(1)* %V, i32 %v_d_off")
    ln("  %v_elem = load float, float addrspace(1)* %v_ptr")
    ln("  %pv = fmul float %p, %v_elem")
    ln("  %acc_updated = fadd float %acc_corr, %pv")
    ln("  store float %acc_updated, float* %acc_ptr")
    ln("  %ud_next = add i32 %ud, 1")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_exit:")
    ln("  %j_next = add i32 %j, 1")
    ln("  br label %kv_loop_header")
    ln("")
    ln("write_output:")
    ln("  br label %write_header")
    ln("")
    ln("write_header:")
    ln("  %wd = phi i32 [0, %write_output], [%wd_next, %write_body]")
    ln(f"  %wd_done = icmp sge i32 %wd, {D}")
    ln("  br i1 %wd_done, label %done, label %write_body")
    ln("")
    ln("write_body:")
    ln(f"  %w_acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %wd")
    ln("  %w_acc_val = load float, float* %w_acc_ptr")
    ln("  %o_val = fdiv float %w_acc_val, %l_val2")
    ln("  %o_d_off = add i32 %q_off, %wd")
    ln("  %o_ptr = getelementptr float, float addrspace(1)* %O, i32 %o_d_off")
    ln("  store float %o_val, float addrspace(1)* %o_ptr")
    ln("  %wd_next = add i32 %wd, 1")
    ln("  br label %write_header")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. paged_attention: Q[B,H,1,D], K_cache/V_cache[num_blocks,block_size,H,D],
#    block_table[B,max_blocks] -> O[B,H,1,D]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_paged_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit paged_attention kernel for KV-cache with block tables.

    Q[B,H,1,D], K_cache[num_blocks,block_size,H,D],
    V_cache[num_blocks,block_size,H,D], block_table[B,max_blocks] (int32).
    1 thread per (b,h). Loops over pages.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    q_name = input_names[0]
    kc_name = input_names[1]
    vc_name = input_names[2]
    bt_name = input_names[3]
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    kc_val = graph.get_value(kc_name)
    bt_val = graph.get_value(bt_name)
    q_shape = list(q_val.shape) if q_val.shape else [2, 8, 1, 64]
    kc_shape = list(kc_val.shape) if kc_val.shape else [32, 16, 8, 64]
    bt_shape = list(bt_val.shape) if bt_val.shape else [2, 4]

    B, H, _, D = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
    num_blocks, block_size = kc_shape[0], kc_shape[1]
    max_blocks = bt_shape[1]
    dtype = q_val.dtype or "float32"
    total = B * H
    scale = 1.0 / math.sqrt(D)

    kernel_name = f"arke_paged_attention_{B}x{H}x{D}_bs{block_size}"
    grid, block_dim = _flat_grid(total)

    source = _gen_paged_attention_ir(
        kernel_name, B, H, D, block_size, max_blocks, scale
    )

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="paged_attention",
        param_names=[q_name, kc_name, vc_name, bt_name, out_name],
        output_name=out_name,
        shapes={
            q_name: q_shape, kc_name: kc_shape,
            vc_name: kc_shape, bt_name: bt_shape, out_name: q_shape,
        },
        dtypes={
            q_name: dtype, kc_name: dtype, vc_name: dtype,
            bt_name: "int32", out_name: dtype,
        },
        grid=grid,
        block=block_dim,
        shared_mem=0,
        kernel_args=[
            ("ptr", q_name), ("ptr", kc_name), ("ptr", vc_name),
            ("ptr", bt_name), ("ptr", out_name),
            ("int", B), ("int", H), ("int", D),
            ("int", block_size), ("int", max_blocks),
        ],
    )


def _gen_paged_attention_ir(
    kernel_name: str, B: int, H: int, D: int,
    block_size: int, max_blocks: int, scale: float
) -> str:
    """Generate paged attention LLVM IR.

    1 thread per (b,h). For each page p in block_table[b, 0..max_blocks-1]:
      page_id = block_table[b, p]
      if page_id < 0: break (sentinel for unused pages)
      For each position pos in 0..block_size-1:
        Compute score = dot(Q[b,h,0,:], K_cache[page_id, pos, h, :]) / sqrt(D)
        Online softmax update
        Update acc with V_cache[page_id, pos, h, :]
    """
    total = B * H
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %Q, float addrspace(1)* %Kc, float addrspace(1)* %Vc, i32 addrspace(1)* %BT, float addrspace(1)* %O, i32 %B_size, i32 %H_size, i32 %D_size, i32 %BlockSize, i32 %MaxBlocks) {{")
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
    # gid -> (b, h)
    ln(f"  %h_val = add i32 0, {H}")
    ln("  %b_idx = sdiv i32 %gid, %h_val")
    ln("  %h_idx = srem i32 %gid, %h_val")
    # Q offset: Q[b,h,0,:] at (b*H*1*D + h*1*D) = (b*H + h)*D
    ln(f"  %d_val = add i32 0, {D}")
    ln(f"  %hd = add i32 0, {H * D}")
    ln("  %q_b_off = mul i32 %b_idx, %hd")
    ln("  %q_h_off = mul i32 %h_idx, %d_val")
    ln("  %q_off = add i32 %q_b_off, %q_h_off")
    # block_table offset: BT[b, :] at b*max_blocks
    ln(f"  %max_blk = add i32 0, {max_blocks}")
    ln("  %bt_base = mul i32 %b_idx, %max_blk")
    # Kc/Vc layout: [num_blocks, block_size, H, D]
    # offset for (page_id, pos, h, :) = ((page_id*block_size + pos)*H + h)*D
    ln(f"  %bs_val = add i32 0, {block_size}")
    ln(f"  %bshd = add i32 0, {block_size * H * D}")
    # Alloca for acc
    ln(f"  %acc_arr = alloca [{D} x float]")
    ln("  br label %init_header")
    ln("")
    ln("init_header:")
    ln("  %init_d = phi i32 [0, %compute], [%init_d_next, %init_body]")
    ln(f"  %init_done = icmp sge i32 %init_d, {D}")
    ln("  br i1 %init_done, label %page_loop_header, label %init_body")
    ln("")
    ln("init_body:")
    ln(f"  %init_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %init_d")
    ln("  store float 0.0, float* %init_ptr")
    ln("  %init_d_next = add i32 %init_d, 1")
    ln("  br label %init_header")
    ln("")
    # Page loop: for p = 0..max_blocks-1
    ln("page_loop_header:")
    ln("  %p = phi i32 [0, %init_header], [%p_next, %pos_loop_exit]")
    ln("  %m_pg = phi float [0xC6293E5940000000, %init_header], [%m_pos_out, %pos_loop_exit]")
    ln("  %l_pg = phi float [0.0, %init_header], [%l_pos_out, %pos_loop_exit]")
    ln(f"  %p_done = icmp sge i32 %p, {max_blocks}")
    ln("  br i1 %p_done, label %write_output, label %load_page_id")
    ln("")
    ln("load_page_id:")
    # page_id = BT[b, p]
    ln("  %bt_off = add i32 %bt_base, %p")
    ln("  %bt_ptr = getelementptr i32, i32 addrspace(1)* %BT, i32 %bt_off")
    ln("  %page_id = load i32, i32 addrspace(1)* %bt_ptr")
    # Sentinel check: if page_id < 0, break
    ln("  %page_valid = icmp sge i32 %page_id, 0")
    ln("  br i1 %page_valid, label %pos_loop_header, label %write_output")
    ln("")
    # Position loop within page
    ln("pos_loop_header:")
    ln("  %pos = phi i32 [0, %load_page_id], [%pos_next, %update_acc_exit]")
    ln("  %m_pos = phi float [%m_pg, %load_page_id], [%m_new, %update_acc_exit]")
    ln("  %l_pos = phi float [%l_pg, %load_page_id], [%l_new, %update_acc_exit]")
    ln(f"  %pos_done = icmp sge i32 %pos, {block_size}")
    ln("  br i1 %pos_done, label %pos_loop_exit, label %dot_header")
    ln("")
    # Dot product Q[b,h,0,:] . K_cache[page_id, pos, h, :]
    ln("dot_header:")
    # K offset: ((page_id*block_size + pos)*H + h)*D
    ln("  %k_page_off = mul i32 %page_id, %bshd")
    ln(f"  %k_pos_hd = mul i32 %pos, {H * D}")
    ln("  %k_off_1 = add i32 %k_page_off, %k_pos_hd")
    ln("  %k_h_off = mul i32 %h_idx, %d_val")
    ln("  %k_base = add i32 %k_off_1, %k_h_off")
    ln("  br label %dot_loop_header")
    ln("")
    ln("dot_loop_header:")
    ln("  %dd = phi i32 [0, %dot_header], [%dd_next, %dot_loop_body]")
    ln("  %dot_acc = phi float [0.0, %dot_header], [%dot_acc_new, %dot_loop_body]")
    ln(f"  %dd_done = icmp sge i32 %dd, {D}")
    ln("  br i1 %dd_done, label %softmax_update, label %dot_loop_body")
    ln("")
    ln("dot_loop_body:")
    ln("  %q_d_off = add i32 %q_off, %dd")
    ln("  %q_ptr = getelementptr float, float addrspace(1)* %Q, i32 %q_d_off")
    ln("  %q_elem = load float, float addrspace(1)* %q_ptr")
    ln("  %k_d_off = add i32 %k_base, %dd")
    ln("  %k_ptr = getelementptr float, float addrspace(1)* %Kc, i32 %k_d_off")
    ln("  %k_elem = load float, float addrspace(1)* %k_ptr")
    ln("  %qk_prod = fmul float %q_elem, %k_elem")
    ln("  %dot_acc_new = fadd float %dot_acc, %qk_prod")
    ln("  %dd_next = add i32 %dd, 1")
    ln("  br label %dot_loop_header")
    ln("")
    ln("softmax_update:")
    ln(f"  %score = fmul float %dot_acc, {scale:#.8e}")
    ln("  %m_new = call float @llvm.maxnum.f32(float %m_pos, float %score)")
    ln("  %m_diff = fsub float %m_pos, %m_new")
    ln("  %corr_lg2e = fmul float %m_diff, 0x3FF7154760000000")

    ln("  %corr = call float @llvm.nvvm.ex2.approx.f(float %corr_lg2e)")
    ln("  %score_diff = fsub float %score, %m_new")
    ln("  %p_exp_lg2e = fmul float %score_diff, 0x3FF7154760000000")

    ln("  %p_exp = call float @llvm.nvvm.ex2.approx.f(float %p_exp_lg2e)")
    ln("  %l_corr = fmul float %l_pos, %corr")
    ln("  %l_new = fadd float %l_corr, %p_exp")
    # V offset same as K
    ln("  %v_base = add i32 %k_off_1, %k_h_off")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_header:")
    ln("  %ud = phi i32 [0, %softmax_update], [%ud_next, %update_acc_body]")
    ln(f"  %ud_done = icmp sge i32 %ud, {D}")
    ln("  br i1 %ud_done, label %update_acc_exit, label %update_acc_body")
    ln("")
    ln("update_acc_body:")
    ln(f"  %acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %ud")
    ln("  %acc_old = load float, float* %acc_ptr")
    ln("  %acc_corr = fmul float %acc_old, %corr")
    ln("  %v_d_off = add i32 %v_base, %ud")
    ln("  %v_ptr = getelementptr float, float addrspace(1)* %Vc, i32 %v_d_off")
    ln("  %v_elem = load float, float addrspace(1)* %v_ptr")
    ln("  %pv = fmul float %p_exp, %v_elem")
    ln("  %acc_updated = fadd float %acc_corr, %pv")
    ln("  store float %acc_updated, float* %acc_ptr")
    ln("  %ud_next = add i32 %ud, 1")
    ln("  br label %update_acc_header")
    ln("")
    ln("update_acc_exit:")
    ln("  %pos_next = add i32 %pos, 1")
    ln("  br label %pos_loop_header")
    ln("")
    ln("pos_loop_exit:")
    ln("  %m_pos_out = phi float [%m_pos, %pos_loop_header]")
    ln("  %l_pos_out = phi float [%l_pos, %pos_loop_header]")
    ln("  %p_next = add i32 %p, 1")
    ln("  br label %page_loop_header")
    ln("")
    # Write output: O[b,h,0,d] = acc[d] / l
    ln("write_output:")
    ln("  %final_l = phi float [%l_pg, %page_loop_header], [%l_pg, %load_page_id]")
    ln("  br label %write_header")
    ln("")
    ln("write_header:")
    ln("  %wd = phi i32 [0, %write_output], [%wd_next, %write_body]")
    ln(f"  %wd_done = icmp sge i32 %wd, {D}")
    ln("  br i1 %wd_done, label %done, label %write_body")
    ln("")
    ln("write_body:")
    ln(f"  %w_acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %wd")
    ln("  %w_acc_val = load float, float* %w_acc_ptr")
    ln("  %o_val = fdiv float %w_acc_val, %final_l")
    # O offset = q_off + wd (same as Q layout)
    ln("  %o_d_off = add i32 %q_off, %wd")
    ln("  %o_ptr = getelementptr float, float addrspace(1)* %O, i32 %o_d_off")
    ln("  store float %o_val, float addrspace(1)* %o_ptr")
    ln("  %wd_next = add i32 %wd, 1")
    ln("  br label %write_header")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. multi_latent_attention (MLA):
#    Q[B,H,S,D], KVc[B,S,Dc], W_uk[Dc,H,D], W_uv[Dc,H,D] -> O[B,H,S,D]
# ─────────────────────────────────────────────────────────────────────────────

def emit_llvm_ir_multi_latent_attention(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit MLA kernel: up-project latent KVc to K,V, then attention.

    Q[B,H,S,D], KVc[B,S,Dc], W_uk[Dc,H,D], W_uv[Dc,H,D] -> O[B,H,S,D].
    1 thread per (b,h,s). For each key row j, reconstruct K/V from KVc via
    matrix-vector product with W_uk/W_uv, then standard attention.
    """
    node = graph.nodes[0]
    input_names = list(node.inputs.values())
    q_name, kvc_name, wuk_name, wuv_name = (
        input_names[0], input_names[1], input_names[2], input_names[3]
    )
    out_name = node.outputs[0]

    q_val = graph.get_value(q_name)
    kvc_val = graph.get_value(kvc_name)
    wuk_val = graph.get_value(wuk_name)
    q_shape = list(q_val.shape) if q_val.shape else [2, 8, 64, 64]
    kvc_shape = list(kvc_val.shape) if kvc_val.shape else [2, 64, 128]
    wuk_shape = list(wuk_val.shape) if wuk_val.shape else [128, 8, 64]

    B, H, S, D = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
    Dc = kvc_shape[2]
    dtype = q_val.dtype or "float32"
    total = B * H * S
    scale = 1.0 / math.sqrt(D)

    kernel_name = f"arke_mla_{B}x{H}x{S}x{D}_Dc{Dc}"
    grid, block = _flat_grid(total)

    source = _gen_mla_ir(kernel_name, B, H, S, D, Dc, scale)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="multi_latent_attention",
        param_names=[q_name, kvc_name, wuk_name, wuv_name, out_name],
        output_name=out_name,
        shapes={
            q_name: q_shape, kvc_name: kvc_shape,
            wuk_name: wuk_shape, wuv_name: wuk_shape, out_name: q_shape,
        },
        dtypes={
            q_name: dtype, kvc_name: dtype,
            wuk_name: dtype, wuv_name: dtype, out_name: dtype,
        },
        grid=grid,
        block=block,
        shared_mem=0,
        kernel_args=[
            ("ptr", q_name), ("ptr", kvc_name), ("ptr", wuk_name),
            ("ptr", wuv_name), ("ptr", out_name),
            ("int", B), ("int", H), ("int", S), ("int", D), ("int", Dc),
        ],
    )


def _gen_mla_ir(
    kernel_name: str, B: int, H: int, S: int, D: int, Dc: int, scale: float
) -> str:
    """Generate MLA LLVM IR.

    For each thread (b,h,q):
      For each key row j = 0..S-1:
        Reconstruct K[d] = sum_c KVc[b,j,c] * W_uk[c,h,d] for all d
        score = dot(Q[b,h,q,:], K[:]) / sqrt(D)
        Online softmax update
        Reconstruct V[d] = sum_c KVc[b,j,c] * W_uv[c,h,d]
        acc[d] += p * V[d]
      O[b,h,q,d] = acc[d] / l

    This uses two alloca'd arrays: acc[D] and kv_tmp[D] for reconstructed K/V.
    """
    total = B * H * S
    lines = []
    ln = lines.append

    ln(_LLVM_HEADER)
    ln(_NVVM_INTRINSICS)

    func_sig = "void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32, i32)"
    ln(f"define void @{kernel_name}(float addrspace(1)* %Q, float addrspace(1)* %KVc, float addrspace(1)* %Wuk, float addrspace(1)* %Wuv, float addrspace(1)* %O, i32 %B_size, i32 %H_size, i32 %S_size, i32 %D_size, i32 %Dc_size) {{")
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
    # gid -> (b, h, q)
    ln(f"  %hs = add i32 0, {H * S}")
    ln("  %b_idx = sdiv i32 %gid, %hs")
    ln("  %rem_hs = srem i32 %gid, %hs")
    ln(f"  %s_val = add i32 0, {S}")
    ln("  %h_idx = sdiv i32 %rem_hs, %s_val")
    ln("  %q_idx = srem i32 %rem_hs, %s_val")
    # Q offset: ((b*H + h)*S + q)*D
    ln(f"  %d_val = add i32 0, {D}")
    ln(f"  %dc_val = add i32 0, {Dc}")
    ln(f"  %hsd = add i32 0, {H * S * D}")
    ln(f"  %sd_q = add i32 0, {S * D}")
    ln("  %q_b_off = mul i32 %b_idx, %hsd")
    ln("  %q_h_off = mul i32 %h_idx, %sd_q")
    ln("  %q_bh = add i32 %q_b_off, %q_h_off")
    ln("  %q_s_off = mul i32 %q_idx, %d_val")
    ln("  %q_off = add i32 %q_bh, %q_s_off")
    # KVc base: KVc[b,:,:] at b*S*Dc
    ln(f"  %s_dc = add i32 0, {S * Dc}")
    ln("  %kvc_b_off = mul i32 %b_idx, %s_dc")
    # W_uk/W_uv layout: [Dc, H, D] -> offset(c, h, d) = (c*H + h)*D + d
    ln(f"  %hd_w = add i32 0, {H * D}")
    ln("  %w_h_off = mul i32 %h_idx, %d_val")
    # Alloca arrays
    ln(f"  %acc_arr = alloca [{D} x float]")
    ln(f"  %kv_tmp = alloca [{D} x float]")
    # Init acc to 0
    ln("  br label %init_header")
    ln("")
    ln("init_header:")
    ln("  %init_d = phi i32 [0, %compute], [%init_d_next, %init_body]")
    ln(f"  %init_done = icmp sge i32 %init_d, {D}")
    ln("  br i1 %init_done, label %kv_loop_header, label %init_body")
    ln("")
    ln("init_body:")
    ln(f"  %init_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %init_d")
    ln("  store float 0.0, float* %init_ptr")
    ln("  %init_d_next = add i32 %init_d, 1")
    ln("  br label %init_header")
    ln("")
    # KV loop: for each key row j = 0..S-1
    ln("kv_loop_header:")
    ln("  %j = phi i32 [0, %init_header], [%j_next, %v_acc_exit]")
    ln("  %m_val2 = phi float [0xC6293E5940000000, %init_header], [%m_new, %v_acc_exit]")
    ln("  %l_val2 = phi float [0.0, %init_header], [%l_new, %v_acc_exit]")
    ln(f"  %j_done = icmp sge i32 %j, {S}")
    ln("  br i1 %j_done, label %write_output, label %reconstruct_k")
    ln("")
    # Reconstruct K: K[d] = sum_c KVc[b,j,c] * W_uk[c,h,d]
    # KVc offset for row j: kvc_b_off + j*Dc
    ln("reconstruct_k:")
    ln("  %kvc_j_off = mul i32 %j, %dc_val")
    ln("  %kvc_base = add i32 %kvc_b_off, %kvc_j_off")
    # For each d, compute K[d] and simultaneously dot with Q
    # We compute score directly: score = sum_d Q[d] * (sum_c KVc[c] * Wuk[c,h,d])
    # Reorganize: score = sum_d Q[d] * K[d] where K[d] = sum_c ...
    # Do it as: for each d, compute K[d], accumulate Q[d]*K[d]
    ln("  br label %score_d_header")
    ln("")
    ln("score_d_header:")
    ln("  %sd = phi i32 [0, %reconstruct_k], [%sd_next, %score_d_body_exit]")
    ln("  %score_acc = phi float [0.0, %reconstruct_k], [%score_acc_new, %score_d_body_exit]")
    ln(f"  %sd_done = icmp sge i32 %sd, {D}")
    ln("  br i1 %sd_done, label %softmax_update, label %k_inner_header")
    ln("")
    # Inner loop: K[sd] = sum_c KVc[b,j,c] * Wuk[c*H*D + h*D + sd]
    ln("k_inner_header:")
    ln("  %kc = phi i32 [0, %score_d_header], [%kc_next, %k_inner_body]")
    ln("  %k_acc = phi float [0.0, %score_d_header], [%k_acc_new, %k_inner_body]")
    ln(f"  %kc_done = icmp sge i32 %kc, {Dc}")
    ln("  br i1 %kc_done, label %score_d_body_exit, label %k_inner_body")
    ln("")
    ln("k_inner_body:")
    # Load KVc[b, j, kc] = KVc[kvc_base + kc]
    ln("  %kvc_off = add i32 %kvc_base, %kc")
    ln("  %kvc_ptr = getelementptr float, float addrspace(1)* %KVc, i32 %kvc_off")
    ln("  %kvc_elem = load float, float addrspace(1)* %kvc_ptr")
    # Load Wuk[kc, h, sd] = Wuk[kc*H*D + h*D + sd]
    ln("  %wuk_c_off = mul i32 %kc, %hd_w")
    ln("  %wuk_ch_off = add i32 %wuk_c_off, %w_h_off")
    ln("  %wuk_off = add i32 %wuk_ch_off, %sd")
    ln("  %wuk_ptr = getelementptr float, float addrspace(1)* %Wuk, i32 %wuk_off")
    ln("  %wuk_elem = load float, float addrspace(1)* %wuk_ptr")
    ln("  %kvc_w = fmul float %kvc_elem, %wuk_elem")
    ln("  %k_acc_new = fadd float %k_acc, %kvc_w")
    ln("  %kc_next = add i32 %kc, 1")
    ln("  br label %k_inner_header")
    ln("")
    # After computing K[sd], multiply by Q[sd] and accumulate score
    ln("score_d_body_exit:")
    # k_acc is now K[sd]. score += Q[q_off+sd] * K[sd]
    ln("  %q_d_off = add i32 %q_off, %sd")
    ln("  %q_ptr = getelementptr float, float addrspace(1)* %Q, i32 %q_d_off")
    ln("  %q_elem = load float, float addrspace(1)* %q_ptr")
    ln("  %qk = fmul float %q_elem, %k_acc")
    ln("  %score_acc_new = fadd float %score_acc, %qk")
    # Also store K[sd] in kv_tmp for later V reconstruction reuse pattern
    ln(f"  %ktmp_ptr = getelementptr [{D} x float], [{D} x float]* %kv_tmp, i32 0, i32 %sd")
    ln("  store float %k_acc, float* %ktmp_ptr")
    ln("  %sd_next = add i32 %sd, 1")
    ln("  br label %score_d_header")
    ln("")
    # Online softmax
    ln("softmax_update:")
    ln(f"  %score = fmul float %score_acc, {scale:#.8e}")
    ln("  %m_new = call float @llvm.maxnum.f32(float %m_val2, float %score)")
    ln("  %m_diff = fsub float %m_val2, %m_new")
    ln("  %corr_lg2e = fmul float %m_diff, 0x3FF7154760000000")

    ln("  %corr = call float @llvm.nvvm.ex2.approx.f(float %corr_lg2e)")
    ln("  %score_diff = fsub float %score, %m_new")
    ln("  %p_exp_lg2e = fmul float %score_diff, 0x3FF7154760000000")

    ln("  %p_exp = call float @llvm.nvvm.ex2.approx.f(float %p_exp_lg2e)")
    ln("  %l_corr = fmul float %l_val2, %corr")
    ln("  %l_new = fadd float %l_corr, %p_exp")
    # Reconstruct V and update acc: for each d:
    #   V[d] = sum_c KVc[b,j,c] * Wuv[c,h,d]
    #   acc[d] = acc[d]*corr + p*V[d]
    ln("  br label %v_acc_header")
    ln("")
    ln("v_acc_header:")
    ln("  %vd = phi i32 [0, %softmax_update], [%vd_next, %v_acc_body_exit]")
    ln(f"  %vd_done = icmp sge i32 %vd, {D}")
    ln("  br i1 %vd_done, label %v_acc_exit, label %v_inner_header")
    ln("")
    # Inner loop for V[vd] = sum_c KVc[b,j,c] * Wuv[c,h,vd]
    ln("v_inner_header:")
    ln("  %vc = phi i32 [0, %v_acc_header], [%vc_next, %v_inner_body]")
    ln("  %v_acc2 = phi float [0.0, %v_acc_header], [%v_acc2_new, %v_inner_body]")
    ln(f"  %vc_done = icmp sge i32 %vc, {Dc}")
    ln("  br i1 %vc_done, label %v_acc_body_exit, label %v_inner_body")
    ln("")
    ln("v_inner_body:")
    ln("  %kvc_off2 = add i32 %kvc_base, %vc")
    ln("  %kvc_ptr2 = getelementptr float, float addrspace(1)* %KVc, i32 %kvc_off2")
    ln("  %kvc_elem2 = load float, float addrspace(1)* %kvc_ptr2")
    ln("  %wuv_c_off = mul i32 %vc, %hd_w")
    ln("  %wuv_ch_off = add i32 %wuv_c_off, %w_h_off")
    ln("  %wuv_off = add i32 %wuv_ch_off, %vd")
    ln("  %wuv_ptr = getelementptr float, float addrspace(1)* %Wuv, i32 %wuv_off")
    ln("  %wuv_elem = load float, float addrspace(1)* %wuv_ptr")
    ln("  %kvc_wuv = fmul float %kvc_elem2, %wuv_elem")
    ln("  %v_acc2_new = fadd float %v_acc2, %kvc_wuv")
    ln("  %vc_next = add i32 %vc, 1")
    ln("  br label %v_inner_header")
    ln("")
    # Update acc[vd] = acc[vd]*corr + p*V[vd]
    ln("v_acc_body_exit:")
    # v_acc2 is V[vd]
    ln(f"  %acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %vd")
    ln("  %acc_old = load float, float* %acc_ptr")
    ln("  %acc_corr = fmul float %acc_old, %corr")
    ln("  %pv = fmul float %p_exp, %v_acc2")
    ln("  %acc_updated = fadd float %acc_corr, %pv")
    ln("  store float %acc_updated, float* %acc_ptr")
    ln("  %vd_next = add i32 %vd, 1")
    ln("  br label %v_acc_header")
    ln("")
    ln("v_acc_exit:")
    ln("  %j_next = add i32 %j, 1")
    ln("  br label %kv_loop_header")
    ln("")
    # Write output
    ln("write_output:")
    ln("  br label %write_header")
    ln("")
    ln("write_header:")
    ln("  %wd = phi i32 [0, %write_output], [%wd_next, %write_body]")
    ln(f"  %wd_done = icmp sge i32 %wd, {D}")
    ln("  br i1 %wd_done, label %done, label %write_body")
    ln("")
    ln("write_body:")
    ln(f"  %w_acc_ptr = getelementptr [{D} x float], [{D} x float]* %acc_arr, i32 0, i32 %wd")
    ln("  %w_acc_val = load float, float* %w_acc_ptr")
    ln("  %o_val = fdiv float %w_acc_val, %l_val2")
    ln("  %o_d_off = add i32 %q_off, %wd")
    ln("  %o_ptr = getelementptr float, float addrspace(1)* %O, i32 %o_d_off")
    ln("  store float %o_val, float addrspace(1)* %o_ptr")
    ln("  %wd_next = add i32 %wd, 1")
    ln("  br label %write_header")
    ln("")
    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")
    ln(_annotation(func_sig, kernel_name))

    return "\n".join(lines)
