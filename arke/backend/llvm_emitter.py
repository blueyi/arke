# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR emitter for Arke (Phase 5, P5-S2).

Generates LLVM IR text targeting nvptx64-nvidia-cuda for compilation via
llc + ptxas.  The emitted IR uses:
  - address space 1 for global memory pointers
  - address space 3 for shared memory (module-level globals)
  - NVVM intrinsics for thread/block IDs and barriers
  - !nvvm.annotations metadata for kernel entry points

This module re-exports all per-category emitters and provides the unified
dispatch table used by LLVMBackend.
"""

from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel
from arke.ir.graph import IRGraph

# ── Elementwise (OT0) ────────────────────────────────────────────────
from arke.backend.llvm_elementwise import (
    emit_llvm_ir_relu,
    emit_llvm_ir_gelu,
    emit_llvm_ir_silu,
    emit_llvm_ir_tanh,
    emit_llvm_ir_sigmoid,
    emit_llvm_ir_exp,
    emit_llvm_ir_neg,
    emit_llvm_ir_rsqrt,
    emit_llvm_ir_add,
    emit_llvm_ir_mul,
    emit_llvm_ir_cast,
    emit_llvm_ir_where,
)

# ── Rowwise / Reduction (OT1) ───────────────────────────────────────
from arke.backend.llvm_rowwise import (
    emit_llvm_ir_softmax,
    emit_llvm_ir_layernorm,
    emit_llvm_ir_rmsnorm,
    emit_llvm_ir_reduce_sum,
    emit_llvm_ir_reduce_max,
    emit_llvm_ir_reduce_mean,
    emit_llvm_ir_argmax,
    emit_llvm_ir_cumsum,
    emit_llvm_ir_topk,
    emit_llvm_ir_rmsnorm_residual,
)

# ── Dense / Data Movement (OT2) ─────────────────────────────────────
from arke.backend.llvm_dense import (
    emit_llvm_ir_batch_matmul,
    emit_llvm_ir_grouped_matmul,
    emit_llvm_ir_concat,
    emit_llvm_ir_copy_,
    emit_llvm_ir_embedding,
    emit_llvm_ir_gather,
    emit_llvm_ir_scatter,
    emit_llvm_ir_permute,
    emit_llvm_ir_split,
    emit_llvm_ir_transpose,
)

# ── Fused Compound (OT3) ────────────────────────────────────────────
from arke.backend.llvm_fused import (
    emit_llvm_ir_cross_entropy,
    emit_llvm_ir_dequantize_per_channel,
    emit_llvm_ir_fused_linear_cross_entropy,
    emit_llvm_ir_gelu_and_mul,
    emit_llvm_ir_quantize_per_token,
    emit_llvm_ir_rope,
    emit_llvm_ir_silu_and_mul,
    emit_llvm_ir_swiglu_packed,
)

# ── Attention (OT4) ─────────────────────────────────────────────────
from arke.backend.llvm_attention import (
    emit_llvm_ir_flash_attention,
    emit_llvm_ir_grouped_query_attention,
    emit_llvm_ir_cross_attention,
    emit_llvm_ir_paged_attention,
    emit_llvm_ir_multi_latent_attention,
)


# ── Matmul (P5-S2: upgraded 64x64 tiling) ────────────────────────────

def emit_llvm_ir_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit a tiled shared-memory matmul kernel as LLVM IR text.

    C[M,N] = A[M,K] @ B[K,N]

    Register-blocked tiling: BM=64, BN=64, BK=16, TM=4, TN=4.
    Each thread computes a 4x4 sub-tile of C (16 accumulators).
    Block: (16, 16) = 256 threads covers the 64x64 output tile.
    Grid: (ceil(N/64), ceil(M/64), 1).
    """
    node = graph.nodes[0]
    assert node.op == "matmul", f"Expected matmul, got {node.op}"

    input_names = list(node.inputs.values())
    a_name, b_name = input_names[0], input_names[1]
    out_name = node.outputs[0]

    a_val = graph.get_value(a_name)
    b_val = graph.get_value(b_name)
    a_shape = list(a_val.shape) if a_val.shape else [64, 64]
    b_shape = list(b_val.shape) if b_val.shape else [64, 64]

    M, K = a_shape[0], a_shape[1]
    K2, N = b_shape[0], b_shape[1]
    assert K == K2, f"K mismatch: {K} vs {K2}"

    dtype = a_val.dtype or "float32"
    BM, BN, BK = 64, 64, 16
    num_tiles = (K + BK - 1) // BK

    kernel_name = f"arke_matmul_{M}x{N}x{K}"

    source = _gen_tiled_matmul_ir(kernel_name, M, K, N, BM, BN, BK, num_tiles)

    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (16, 16, 1)

    return CudaCKernel(
        kernel_name=kernel_name,
        source=source,
        op_name="matmul",
        param_names=[a_name, b_name, out_name],
        output_name=out_name,
        shapes={a_name: a_shape, b_name: b_shape, out_name: [M, N]},
        dtypes={a_name: dtype, b_name: dtype, out_name: dtype},
        grid=grid,
        block=block,
        shared_mem=0,  # static shared memory via module globals
        kernel_args=[
            ("ptr", a_name), ("ptr", b_name), ("ptr", out_name),
            ("int", M), ("int", K), ("int", N),
        ],
    )


def _gen_tiled_matmul_ir(
    kernel_name: str, M: int, K: int, N: int,
    BM: int, BN: int, BK: int, num_tiles: int,
) -> str:
    """Generate LLVM IR for register-blocked tiled matmul.

    BM=64, BN=64, BK=16, TM=4, TN=4.
    Block = (16,16) = 256 threads; each thread computes 4x4 output elements.
    Thread (tx, ty) owns C rows [by*64 + ty*4 + 0..3] x cols [bx*64 + tx*4 + 0..3].
    Shared memory: shmem_A[64][16], shmem_B[16][64].
    Each thread cooperatively loads 4 elements of A and 4 elements of B per tile.
    """
    TM, TN = 4, 4
    BLOCK_X, BLOCK_Y = 16, 16

    lines: list[str] = []
    ln = lines.append

    # ── Module header ──
    ln('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"')
    ln('target triple = "nvptx64-nvidia-cuda"')
    ln("")

    # ── Shared memory: shmem_A[BM][BK] = [64][16], shmem_B[BK][BN] = [16][64] ──
    ln(f"@shmem_A = internal addrspace(3) global [{BM} x [{BK} x float]] undef")
    ln(f"@shmem_B = internal addrspace(3) global [{BK} x [{BN} x float]] undef")
    ln("")

    # ── NVVM intrinsics ──
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")
    ln("declare void @llvm.nvvm.barrier0()")
    ln("")

    # ── Kernel function ──
    ln(f"define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C, i32 %M, i32 %K, i32 %N) {{")
    ln("entry:")

    # Thread/block IDs
    ln("  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()")
    ln("  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")

    # Base row/col for this thread's 4x4 sub-tile
    # row0 = by * 64 + ty * 4, col0 = bx * 64 + tx * 4
    ln(f"  %row_base = mul i32 %by, {BM}")
    ln(f"  %ty_x_tm = mul i32 %ty, {TM}")
    ln("  %row0 = add i32 %row_base, %ty_x_tm")
    ln("  %row1 = add i32 %row0, 1")
    ln("  %row2 = add i32 %row0, 2")
    ln("  %row3 = add i32 %row0, 3")
    ln(f"  %col_base = mul i32 %bx, {BN}")
    ln(f"  %tx_x_tn = mul i32 %tx, {TN}")
    ln("  %col0 = add i32 %col_base, %tx_x_tn")
    ln("  %col1 = add i32 %col0, 1")
    ln("  %col2 = add i32 %col0, 2")
    ln("  %col3 = add i32 %col0, 3")

    # Linear thread id for cooperative loading
    ln(f"  %lin_id = mul i32 %ty, {BLOCK_X}")
    ln("  %lin_id2 = add i32 %lin_id, %tx")

    ln("  br label %tile_loop_header")
    ln("")

    # ── Tile loop header (PHI for tile index + 16 accumulators) ──
    ln("tile_loop_header:")
    ln("  %t = phi i32 [ 0, %entry ], [ %t_next, %tile_loop_latch ]")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %acc_r{r}c{c} = phi float [ 0.0, %entry ], [ %acc_r{r}c{c}_out, %tile_loop_latch ]")
    ln(f"  %tile_done = icmp sge i32 %t, {num_tiles}")
    ln("  br i1 %tile_done, label %store_result, label %load_tile")
    ln("")

    # ── Load tile into shared memory ──
    # shmem_A[64][16] = 1024 floats, 256 threads → 4 per thread
    # shmem_B[16][64] = 1024 floats, 256 threads → 4 per thread
    # lin_id * 4 + i gives flat index
    # For A: flat_idx / 16 = shmem_row, flat_idx % 16 = shmem_col
    # For B: flat_idx / 64 = shmem_row, flat_idx % 64 = shmem_col

    ln("load_tile:")
    ln(f"  %t_times_bk = mul i32 %t, {BK}")

    # Base flat index = lin_id2 * 4
    ln("  %flat_base = mul i32 %lin_id2, 4")

    # Load 4 elements of A into shared memory
    for i in range(4):
        sfx = f"sa{i}"
        ln(f"  %{sfx}_flat = add i32 %flat_base, {i}")
        ln(f"  %{sfx}_srow = lshr i32 %{sfx}_flat, 4")  # / 16
        ln(f"  %{sfx}_scol = and i32 %{sfx}_flat, 15")   # % 16
        ln(f"  %{sfx}_grow = add i32 %row_base, %{sfx}_srow")
        ln(f"  %{sfx}_gcol = add i32 %t_times_bk, %{sfx}_scol")
        ln(f"  %{sfx}_row_ok = icmp slt i32 %{sfx}_grow, %M")
        ln(f"  %{sfx}_col_ok = icmp slt i32 %{sfx}_gcol, %K")
        ln(f"  %{sfx}_valid = and i1 %{sfx}_row_ok, %{sfx}_col_ok")
        ln(f"  br i1 %{sfx}_valid, label %load_{sfx}_valid, label %load_{sfx}_zero")
        ln("")

        ln(f"load_{sfx}_valid:")
        ln(f"  %{sfx}_gidx = mul i32 %{sfx}_grow, %K")
        ln(f"  %{sfx}_gidx2 = add i32 %{sfx}_gidx, %{sfx}_gcol")
        ln(f"  %{sfx}_gep = getelementptr float, float addrspace(1)* %A, i32 %{sfx}_gidx2")
        ln(f"  %{sfx}_val = load float, float addrspace(1)* %{sfx}_gep")
        ln(f"  br label %store_{sfx}")
        ln("")

        ln(f"load_{sfx}_zero:")
        ln(f"  br label %store_{sfx}")
        ln("")

        ln(f"store_{sfx}:")
        prev_label_valid = f"%{sfx}_val"
        ln(f"  %{sfx}_data = phi float [ {prev_label_valid}, %load_{sfx}_valid ], [ 0.0, %load_{sfx}_zero ]")
        ln(f"  %{sfx}_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %{sfx}_srow, i32 %{sfx}_scol")
        ln(f"  store float %{sfx}_data, float addrspace(3)* %{sfx}_ptr")

    # Load 4 elements of B into shared memory
    for i in range(4):
        sfx = f"sb{i}"
        ln(f"  %{sfx}_flat = add i32 %flat_base, {i}")
        ln(f"  %{sfx}_srow = lshr i32 %{sfx}_flat, 6")  # / 64
        ln(f"  %{sfx}_scol = and i32 %{sfx}_flat, 63")   # % 64
        ln(f"  %{sfx}_grow = add i32 %t_times_bk, %{sfx}_srow")
        ln(f"  %{sfx}_gcol = add i32 %col_base, %{sfx}_scol")
        ln(f"  %{sfx}_row_ok = icmp slt i32 %{sfx}_grow, %K")
        ln(f"  %{sfx}_col_ok = icmp slt i32 %{sfx}_gcol, %N")
        ln(f"  %{sfx}_valid = and i1 %{sfx}_row_ok, %{sfx}_col_ok")
        ln(f"  br i1 %{sfx}_valid, label %load_{sfx}_valid, label %load_{sfx}_zero")
        ln("")

        ln(f"load_{sfx}_valid:")
        ln(f"  %{sfx}_gidx = mul i32 %{sfx}_grow, %N")
        ln(f"  %{sfx}_gidx2 = add i32 %{sfx}_gidx, %{sfx}_gcol")
        ln(f"  %{sfx}_gep = getelementptr float, float addrspace(1)* %B, i32 %{sfx}_gidx2")
        ln(f"  %{sfx}_val = load float, float addrspace(1)* %{sfx}_gep")
        ln(f"  br label %store_{sfx}")
        ln("")

        ln(f"load_{sfx}_zero:")
        ln(f"  br label %store_{sfx}")
        ln("")

        ln(f"store_{sfx}:")
        prev_label_valid = f"%{sfx}_val"
        ln(f"  %{sfx}_data = phi float [ {prev_label_valid}, %load_{sfx}_valid ], [ 0.0, %load_{sfx}_zero ]")
        ln(f"  %{sfx}_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %{sfx}_srow, i32 %{sfx}_scol")
        ln(f"  store float %{sfx}_data, float addrspace(3)* %{sfx}_ptr")

    # __syncthreads after loading shared memory
    ln("  call void @llvm.nvvm.barrier0()")
    ln("  br label %k_loop_header")
    ln("")

    # ── k-loop header: iterate over BK=16 dimension ──
    # The last store block is store_sb3
    ln("k_loop_header:")
    ln("  %ki = phi i32 [ 0, %store_sb3 ], [ %ki_next, %k_loop_body ]")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %acc_k_r{r}c{c} = phi float [ %acc_r{r}c{c}, %store_sb3 ], [ %acc_k_r{r}c{c}_next, %k_loop_body ]")
    ln(f"  %k_done = icmp sge i32 %ki, {BK}")
    ln("  br i1 %k_done, label %k_loop_exit, label %k_loop_body")
    ln("")

    # ── k-loop body: register-blocked 4x4 accumulation ──
    ln("k_loop_body:")

    # Load 4 values from shmem_A: shmem_A[ty*4+r][ki] for r=0..3
    for r in range(TM):
        row_reg = f"%ty_x_tm" if r == 0 else f"%ty_tm_p{r}"
        if r > 0:
            ln(f"  %ty_tm_p{r} = add i32 %ty_x_tm, {r}")
        ln(f"  %sa_r{r}_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 {row_reg}, i32 %ki")
        ln(f"  %a_reg{r} = load float, float addrspace(3)* %sa_r{r}_ptr")

    # Load 4 values from shmem_B: shmem_B[ki][tx*4+c] for c=0..3
    for c in range(TN):
        col_reg = f"%tx_x_tn" if c == 0 else f"%tx_tn_p{c}"
        if c > 0:
            ln(f"  %tx_tn_p{c} = add i32 %tx_x_tn, {c}")
        ln(f"  %sb_c{c}_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %ki, i32 {col_reg}")
        ln(f"  %b_reg{c} = load float, float addrspace(3)* %sb_c{c}_ptr")

    # 16 FMAs: acc[r][c] += a_reg[r] * b_reg[c]
    for r in range(TM):
        for c in range(TN):
            ln(f"  %prod_r{r}c{c} = fmul float %a_reg{r}, %b_reg{c}")
            ln(f"  %acc_k_r{r}c{c}_next = fadd float %acc_k_r{r}c{c}, %prod_r{r}c{c}")

    ln("  %ki_next = add i32 %ki, 1")
    ln("  br label %k_loop_header")
    ln("")

    # ── k-loop exit: sync before next tile ──
    ln("k_loop_exit:")
    ln("  call void @llvm.nvvm.barrier0()")
    ln("  br label %tile_loop_latch")
    ln("")

    ln("tile_loop_latch:")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %acc_r{r}c{c}_out = phi float [ %acc_k_r{r}c{c}, %k_loop_exit ]")
    ln("  %t_next = add i32 %t, 1")
    ln("  br label %tile_loop_header")
    ln("")

    # ── Store 4x4 result with bounds checking ──
    ln("store_result:")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %final_r{r}c{c} = phi float [ %acc_r{r}c{c}, %tile_loop_header ]")

    # Precompute row/col bounds
    ln("  %r0_ok = icmp slt i32 %row0, %M")
    ln("  %r1_ok = icmp slt i32 %row1, %M")
    ln("  %r2_ok = icmp slt i32 %row2, %M")
    ln("  %r3_ok = icmp slt i32 %row3, %M")
    ln("  %c0_ok = icmp slt i32 %col0, %N")
    ln("  %c1_ok = icmp slt i32 %col1, %N")
    ln("  %c2_ok = icmp slt i32 %col2, %N")
    ln("  %c3_ok = icmp slt i32 %col3, %N")

    # Generate 16 bounded stores using a chain of blocks
    store_pairs = [(r, c) for r in range(TM) for c in range(TN)]
    first_label = f"check_r{store_pairs[0][0]}c{store_pairs[0][1]}"
    ln(f"  br label %{first_label}")
    ln("")

    for idx, (r, c) in enumerate(store_pairs):
        next_label = f"check_r{store_pairs[idx+1][0]}c{store_pairs[idx+1][1]}" if idx + 1 < len(store_pairs) else "done"

        ln(f"check_r{r}c{c}:")
        ln(f"  %rc{r}{c}_ok = and i1 %r{r}_ok, %c{c}_ok")
        ln(f"  br i1 %rc{r}{c}_ok, label %do_store_r{r}c{c}, label %{next_label}")
        ln("")

        ln(f"do_store_r{r}c{c}:")
        ln(f"  %c_off_r{r}c{c} = mul i32 %row{r}, %N")
        ln(f"  %c_idx_r{r}c{c} = add i32 %c_off_r{r}c{c}, %col{c}")
        ln(f"  %c_gep_r{r}c{c} = getelementptr float, float addrspace(1)* %C, i32 %c_idx_r{r}c{c}")
        ln(f"  store float %final_r{r}c{c}, float addrspace(1)* %c_gep_r{r}c{c}")
        ln(f"  br label %{next_label}")
        ln("")

    ln("done:")
    ln("  ret void")
    ln("}")
    ln("")

    # ── NVVM annotations ──
    ln("!nvvm.annotations = !{!0}")
    ln(f"!0 = !{{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @{kernel_name}, !\"kernel\", i32 1}}")
    ln("")

    return "\n".join(lines)


from typing import Any, Callable

# ── Unified Emitter Dispatch Table ───────────────────────────────────

LLVM_EMITTERS: dict[str, Callable[..., Any]] = {
    # OT0: Elementwise
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
    # OT1: Rowwise / Reduction
    "softmax": emit_llvm_ir_softmax,
    "layernorm": emit_llvm_ir_layernorm,
    "rmsnorm": emit_llvm_ir_rmsnorm,
    "reduce_sum": emit_llvm_ir_reduce_sum,
    "reduce_max": emit_llvm_ir_reduce_max,
    "reduce_mean": emit_llvm_ir_reduce_mean,
    "argmax": emit_llvm_ir_argmax,
    "cumsum": emit_llvm_ir_cumsum,
    "topk": emit_llvm_ir_topk,
    "rmsnorm_residual": emit_llvm_ir_rmsnorm_residual,
    # OT2: Dense / Data Movement
    "matmul": emit_llvm_ir_matmul,
    "batch_matmul": emit_llvm_ir_batch_matmul,
    "grouped_matmul": emit_llvm_ir_grouped_matmul,
    "concat": emit_llvm_ir_concat,
    "copy_": emit_llvm_ir_copy_,
    "embedding": emit_llvm_ir_embedding,
    "gather": emit_llvm_ir_gather,
    "scatter": emit_llvm_ir_scatter,
    "permute": emit_llvm_ir_permute,
    "split": emit_llvm_ir_split,
    "transpose": emit_llvm_ir_transpose,
    # OT3: Fused Compound
    "cross_entropy": emit_llvm_ir_cross_entropy,
    "dequantize_per_channel": emit_llvm_ir_dequantize_per_channel,
    "fused_linear_cross_entropy": emit_llvm_ir_fused_linear_cross_entropy,
    "gelu_and_mul": emit_llvm_ir_gelu_and_mul,
    "quantize_per_token": emit_llvm_ir_quantize_per_token,
    "rope": emit_llvm_ir_rope,
    "silu_and_mul": emit_llvm_ir_silu_and_mul,
    "swiglu_packed": emit_llvm_ir_swiglu_packed,
    # OT4: Attention
    "flash_attention": emit_llvm_ir_flash_attention,
    "grouped_query_attention": emit_llvm_ir_grouped_query_attention,
    "cross_attention": emit_llvm_ir_cross_attention,
    "paged_attention": emit_llvm_ir_paged_attention,
    "multi_latent_attention": emit_llvm_ir_multi_latent_attention,
}
