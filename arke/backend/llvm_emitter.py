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


# ── Matmul (original P5-S1 implementation) ───────────────────────────

def emit_llvm_ir_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit a tiled shared-memory matmul kernel as LLVM IR text.

    C[M,N] = A[M,K] @ B[K,N]

    Register-blocked tiling: BM=32, BN=32, BK=16, TM=2, TN=2.
    Each thread computes a 2x2 sub-tile of C.
    Block: (16, 16) = 256 threads covers the 32x32 output tile.
    Grid: (ceil(N/BN), ceil(M/BM), 1).
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
    BM, BN, BK = 32, 32, 16
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

    BM=32, BN=32, BK=16, TM=2, TN=2.
    Block = (16,16) = 256 threads; each thread computes 2x2 output elements.
    Thread (tx, ty) owns C rows [by*32 + ty*2 + 0..1] x cols [bx*32 + tx*2 + 0..1].
    Shared memory: shmem_A[32][16], shmem_B[16][32].
    Each thread cooperatively loads 2 elements of A and 2 elements of B per tile.
    """
    TM, TN = 2, 2
    BLOCK_X, BLOCK_Y = 16, 16

    lines = []
    ln = lines.append

    # ── Module header ──
    ln('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"')
    ln('target triple = "nvptx64-nvidia-cuda"')
    ln("")

    # ── Shared memory: shmem_A[BM][BK] = [32][16], shmem_B[BK][BN] = [16][32] ──
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

    # Base row/col for this thread's 2x2 sub-tile
    # row0 = by * BM + ty * TM, col0 = bx * BN + tx * TN
    ln(f"  %row_base = mul i32 %by, {BM}")
    ln(f"  %ty_x_tm = mul i32 %ty, {TM}")
    ln("  %row0 = add i32 %row_base, %ty_x_tm")
    ln("  %row1 = add i32 %row0, 1")
    ln(f"  %col_base = mul i32 %bx, {BN}")
    ln(f"  %tx_x_tn = mul i32 %tx, {TN}")
    ln("  %col0 = add i32 %col_base, %tx_x_tn")
    ln("  %col1 = add i32 %col0, 1")

    ln("  br label %tile_loop_header")
    ln("")

    # ── Tile loop header (PHI for tile index + 4 accumulators) ──
    ln("tile_loop_header:")
    ln("  %t = phi i32 [ 0, %entry ], [ %t_next, %tile_loop_latch ]")
    ln("  %acc_r0c0 = phi float [ 0.0, %entry ], [ %acc_r0c0_out, %tile_loop_latch ]")
    ln("  %acc_r0c1 = phi float [ 0.0, %entry ], [ %acc_r0c1_out, %tile_loop_latch ]")
    ln("  %acc_r1c0 = phi float [ 0.0, %entry ], [ %acc_r1c0_out, %tile_loop_latch ]")
    ln("  %acc_r1c1 = phi float [ 0.0, %entry ], [ %acc_r1c1_out, %tile_loop_latch ]")
    ln(f"  %tile_done = icmp sge i32 %t, {num_tiles}")
    ln("  br i1 %tile_done, label %store_result, label %load_tile")
    ln("")

    # ── Load tile into shared memory ──
    # shmem_A[BM][BK] = [32][16] = 512 floats, 256 threads → 2 per thread
    # shmem_B[BK][BN] = [16][32] = 512 floats, 256 threads → 2 per thread
    #
    # For shmem_A: thread lin_id loads elements at flat indices lin_id and lin_id+256
    #   flat index f → shmem_A[f / BK][f % BK] = shmem_A[f / 16][f % 16]
    #   Global A: row = by*BM + f/BK, col = t*BK + f%BK
    #
    # For shmem_B: same pattern
    #   flat index f → shmem_B[f / BN][f % BN] = shmem_B[f / 32][f % 32]
    #   Global B: row = t*BK + f/BN, col = bx*BN + f%BN

    ln("load_tile:")
    ln(f"  %t_times_bk = mul i32 %t, {BK}")

    # Recompute lin_id correctly here: ty * 16 + tx
    ln(f"  %ty_x_blockx = mul i32 %ty, {BLOCK_X}")
    ln("  %lin = add i32 %ty_x_blockx, %tx")

    # ── Load shmem_A element 0: flat index = lin ──
    ln(f"  %sa0_srow = udiv i32 %lin, {BK}")
    ln(f"  %sa0_scol = urem i32 %lin, {BK}")
    ln("  %sa0_grow = add i32 %row_base, %sa0_srow")
    ln("  %sa0_gcol = add i32 %t_times_bk, %sa0_scol")
    ln("  %sa0_row_ok = icmp slt i32 %sa0_grow, %M")
    ln("  %sa0_col_ok = icmp slt i32 %sa0_gcol, %K")
    ln("  %sa0_valid = and i1 %sa0_row_ok, %sa0_col_ok")
    ln("  br i1 %sa0_valid, label %load_sa0_valid, label %load_sa0_zero")
    ln("")

    ln("load_sa0_valid:")
    ln("  %sa0_gidx = mul i32 %sa0_grow, %K")
    ln("  %sa0_gidx2 = add i32 %sa0_gidx, %sa0_gcol")
    ln("  %sa0_gep = getelementptr float, float addrspace(1)* %A, i32 %sa0_gidx2")
    ln("  %sa0_val = load float, float addrspace(1)* %sa0_gep")
    ln("  br label %store_sa0")
    ln("")

    ln("load_sa0_zero:")
    ln("  br label %store_sa0")
    ln("")

    ln("store_sa0:")
    ln("  %sa0_data = phi float [ %sa0_val, %load_sa0_valid ], [ 0.0, %load_sa0_zero ]")
    ln(f"  %sa0_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %sa0_srow, i32 %sa0_scol")
    ln("  store float %sa0_data, float addrspace(3)* %sa0_ptr")

    # ── Load shmem_A element 1: flat index = lin + 256 ──
    ln("  %lin_plus_256 = add i32 %lin, 256")
    ln(f"  %sa1_srow = udiv i32 %lin_plus_256, {BK}")
    ln(f"  %sa1_scol = urem i32 %lin_plus_256, {BK}")
    ln("  %sa1_grow = add i32 %row_base, %sa1_srow")
    ln("  %sa1_gcol = add i32 %t_times_bk, %sa1_scol")
    ln("  %sa1_row_ok = icmp slt i32 %sa1_grow, %M")
    ln("  %sa1_col_ok = icmp slt i32 %sa1_gcol, %K")
    ln("  %sa1_valid = and i1 %sa1_row_ok, %sa1_col_ok")
    ln("  br i1 %sa1_valid, label %load_sa1_valid, label %load_sa1_zero")
    ln("")

    ln("load_sa1_valid:")
    ln("  %sa1_gidx = mul i32 %sa1_grow, %K")
    ln("  %sa1_gidx2 = add i32 %sa1_gidx, %sa1_gcol")
    ln("  %sa1_gep = getelementptr float, float addrspace(1)* %A, i32 %sa1_gidx2")
    ln("  %sa1_val = load float, float addrspace(1)* %sa1_gep")
    ln("  br label %store_sa1")
    ln("")

    ln("load_sa1_zero:")
    ln("  br label %store_sa1")
    ln("")

    ln("store_sa1:")
    ln("  %sa1_data = phi float [ %sa1_val, %load_sa1_valid ], [ 0.0, %load_sa1_zero ]")
    ln(f"  %sa1_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %sa1_srow, i32 %sa1_scol")
    ln("  store float %sa1_data, float addrspace(3)* %sa1_ptr")

    # ── Load shmem_B element 0: flat index = lin ──
    # shmem_B[BK][BN] = [16][32], flat f → row = f/32, col = f%32
    ln(f"  %sb0_srow = udiv i32 %lin, {BN}")
    ln(f"  %sb0_scol = urem i32 %lin, {BN}")
    ln("  %sb0_grow = add i32 %t_times_bk, %sb0_srow")
    ln("  %sb0_gcol = add i32 %col_base, %sb0_scol")
    ln("  %sb0_row_ok = icmp slt i32 %sb0_grow, %K")
    ln("  %sb0_col_ok = icmp slt i32 %sb0_gcol, %N")
    ln("  %sb0_valid = and i1 %sb0_row_ok, %sb0_col_ok")
    ln("  br i1 %sb0_valid, label %load_sb0_valid, label %load_sb0_zero")
    ln("")

    ln("load_sb0_valid:")
    ln("  %sb0_gidx = mul i32 %sb0_grow, %N")
    ln("  %sb0_gidx2 = add i32 %sb0_gidx, %sb0_gcol")
    ln("  %sb0_gep = getelementptr float, float addrspace(1)* %B, i32 %sb0_gidx2")
    ln("  %sb0_val = load float, float addrspace(1)* %sb0_gep")
    ln("  br label %store_sb0")
    ln("")

    ln("load_sb0_zero:")
    ln("  br label %store_sb0")
    ln("")

    ln("store_sb0:")
    ln("  %sb0_data = phi float [ %sb0_val, %load_sb0_valid ], [ 0.0, %load_sb0_zero ]")
    ln(f"  %sb0_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %sb0_srow, i32 %sb0_scol")
    ln("  store float %sb0_data, float addrspace(3)* %sb0_ptr")

    # ── Load shmem_B element 1: flat index = lin + 256 ──
    ln(f"  %sb1_srow = udiv i32 %lin_plus_256, {BN}")
    ln(f"  %sb1_scol = urem i32 %lin_plus_256, {BN}")
    ln("  %sb1_grow = add i32 %t_times_bk, %sb1_srow")
    ln("  %sb1_gcol = add i32 %col_base, %sb1_scol")
    ln("  %sb1_row_ok = icmp slt i32 %sb1_grow, %K")
    ln("  %sb1_col_ok = icmp slt i32 %sb1_gcol, %N")
    ln("  %sb1_valid = and i1 %sb1_row_ok, %sb1_col_ok")
    ln("  br i1 %sb1_valid, label %load_sb1_valid, label %load_sb1_zero")
    ln("")

    ln("load_sb1_valid:")
    ln("  %sb1_gidx = mul i32 %sb1_grow, %N")
    ln("  %sb1_gidx2 = add i32 %sb1_gidx, %sb1_gcol")
    ln("  %sb1_gep = getelementptr float, float addrspace(1)* %B, i32 %sb1_gidx2")
    ln("  %sb1_val = load float, float addrspace(1)* %sb1_gep")
    ln("  br label %store_sb1")
    ln("")

    ln("load_sb1_zero:")
    ln("  br label %store_sb1")
    ln("")

    ln("store_sb1:")
    ln("  %sb1_data = phi float [ %sb1_val, %load_sb1_valid ], [ 0.0, %load_sb1_zero ]")
    ln(f"  %sb1_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %sb1_srow, i32 %sb1_scol")
    ln("  store float %sb1_data, float addrspace(3)* %sb1_ptr")

    # __syncthreads after loading shared memory
    ln("  call void @llvm.nvvm.barrier0()")

    # ── k-loop: iterate over BK dimension ──
    ln("  br label %k_loop_header")
    ln("")

    ln("k_loop_header:")
    ln("  %ki = phi i32 [ 0, %store_sb1 ], [ %ki_next, %k_loop_body ]")
    ln("  %acc_k_r0c0 = phi float [ %acc_r0c0, %store_sb1 ], [ %acc_k_r0c0_next, %k_loop_body ]")
    ln("  %acc_k_r0c1 = phi float [ %acc_r0c1, %store_sb1 ], [ %acc_k_r0c1_next, %k_loop_body ]")
    ln("  %acc_k_r1c0 = phi float [ %acc_r1c0, %store_sb1 ], [ %acc_k_r1c0_next, %k_loop_body ]")
    ln("  %acc_k_r1c1 = phi float [ %acc_r1c1, %store_sb1 ], [ %acc_k_r1c1_next, %k_loop_body ]")
    ln(f"  %k_done = icmp sge i32 %ki, {BK}")
    ln("  br i1 %k_done, label %k_loop_exit, label %k_loop_body")
    ln("")

    # ── k-loop body: register-blocked 2x2 accumulation ──
    # Load 2 values from shmem_A: shmem_A[ty*2 + 0][ki], shmem_A[ty*2 + 1][ki]
    # Load 2 values from shmem_B: shmem_B[ki][tx*2 + 0], shmem_B[ki][tx*2 + 1]
    # Compute 4 products and accumulate
    ln("k_loop_body:")

    # shmem_A row indices: ty*TM+0 = row in shared mem
    # We precomputed ty_x_tm = ty*2, so rows are ty_x_tm and ty_x_tm+1
    # But ty_x_tm was computed in entry block - we can use it here since it dominates
    ln(f"  %sa_r0_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %ty_x_tm, i32 %ki")
    ln("  %a_r0 = load float, float addrspace(3)* %sa_r0_ptr")

    # shmem_A[ty*2 + 1][ki]
    ln("  %ty_tm_p1 = add i32 %ty_x_tm, 1")
    ln(f"  %sa_r1_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %ty_tm_p1, i32 %ki")
    ln("  %a_r1 = load float, float addrspace(3)* %sa_r1_ptr")

    # shmem_B[ki][tx*2 + 0]
    ln(f"  %sb_c0_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %ki, i32 %tx_x_tn")
    ln("  %b_c0 = load float, float addrspace(3)* %sb_c0_ptr")

    # shmem_B[ki][tx*2 + 1]
    ln("  %tx_tn_p1 = add i32 %tx_x_tn, 1")
    ln(f"  %sb_c1_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %ki, i32 %tx_tn_p1")
    ln("  %b_c1 = load float, float addrspace(3)* %sb_c1_ptr")

    # 4 FMAs: acc[r][c] += a_r[r] * b_c[c]
    ln("  %prod_r0c0 = fmul float %a_r0, %b_c0")
    ln("  %acc_k_r0c0_next = fadd float %acc_k_r0c0, %prod_r0c0")

    ln("  %prod_r0c1 = fmul float %a_r0, %b_c1")
    ln("  %acc_k_r0c1_next = fadd float %acc_k_r0c1, %prod_r0c1")

    ln("  %prod_r1c0 = fmul float %a_r1, %b_c0")
    ln("  %acc_k_r1c0_next = fadd float %acc_k_r1c0, %prod_r1c0")

    ln("  %prod_r1c1 = fmul float %a_r1, %b_c1")
    ln("  %acc_k_r1c1_next = fadd float %acc_k_r1c1, %prod_r1c1")

    ln("  %ki_next = add i32 %ki, 1")
    ln("  br label %k_loop_header")
    ln("")

    # ── k-loop exit: sync before next tile ──
    ln("k_loop_exit:")
    ln("  call void @llvm.nvvm.barrier0()")
    ln("  br label %tile_loop_latch")
    ln("")

    ln("tile_loop_latch:")
    ln("  %acc_r0c0_out = phi float [ %acc_k_r0c0, %k_loop_exit ]")
    ln("  %acc_r0c1_out = phi float [ %acc_k_r0c1, %k_loop_exit ]")
    ln("  %acc_r1c0_out = phi float [ %acc_k_r1c0, %k_loop_exit ]")
    ln("  %acc_r1c1_out = phi float [ %acc_k_r1c1, %k_loop_exit ]")
    ln("  %t_next = add i32 %t, 1")
    ln("  br label %tile_loop_header")
    ln("")

    # ── Store 2x2 result ──
    ln("store_result:")
    ln("  %final_r0c0 = phi float [ %acc_r0c0, %tile_loop_header ]")
    ln("  %final_r0c1 = phi float [ %acc_r0c1, %tile_loop_header ]")
    ln("  %final_r1c0 = phi float [ %acc_r1c0, %tile_loop_header ]")
    ln("  %final_r1c1 = phi float [ %acc_r1c1, %tile_loop_header ]")

    # Store C[row0][col0]
    ln("  %r0_ok = icmp slt i32 %row0, %M")
    ln("  %c0_ok = icmp slt i32 %col0, %N")
    ln("  %rc00_ok = and i1 %r0_ok, %c0_ok")
    ln("  br i1 %rc00_ok, label %do_store_r0c0, label %check_r0c1")
    ln("")

    ln("do_store_r0c0:")
    ln("  %c_off_r0 = mul i32 %row0, %N")
    ln("  %c_idx_r0c0 = add i32 %c_off_r0, %col0")
    ln("  %c_gep_r0c0 = getelementptr float, float addrspace(1)* %C, i32 %c_idx_r0c0")
    ln("  store float %final_r0c0, float addrspace(1)* %c_gep_r0c0")
    ln("  br label %check_r0c1")
    ln("")

    # Store C[row0][col1]
    ln("check_r0c1:")
    ln("  %c1_ok = icmp slt i32 %col1, %N")
    ln("  %rc01_ok = and i1 %r0_ok, %c1_ok")
    ln("  br i1 %rc01_ok, label %do_store_r0c1, label %check_r1c0")
    ln("")

    ln("do_store_r0c1:")
    ln("  %c_off_r0b = mul i32 %row0, %N")
    ln("  %c_idx_r0c1 = add i32 %c_off_r0b, %col1")
    ln("  %c_gep_r0c1 = getelementptr float, float addrspace(1)* %C, i32 %c_idx_r0c1")
    ln("  store float %final_r0c1, float addrspace(1)* %c_gep_r0c1")
    ln("  br label %check_r1c0")
    ln("")

    # Store C[row1][col0]
    ln("check_r1c0:")
    ln("  %r1_ok = icmp slt i32 %row1, %M")
    ln("  %rc10_ok = and i1 %r1_ok, %c0_ok")
    ln("  br i1 %rc10_ok, label %do_store_r1c0, label %check_r1c1")
    ln("")

    ln("do_store_r1c0:")
    ln("  %c_off_r1 = mul i32 %row1, %N")
    ln("  %c_idx_r1c0 = add i32 %c_off_r1, %col0")
    ln("  %c_gep_r1c0 = getelementptr float, float addrspace(1)* %C, i32 %c_idx_r1c0")
    ln("  store float %final_r1c0, float addrspace(1)* %c_gep_r1c0")
    ln("  br label %check_r1c1")
    ln("")

    # Store C[row1][col1]
    ln("check_r1c1:")
    ln("  %rc11_ok = and i1 %r1_ok, %c1_ok")
    ln("  br i1 %rc11_ok, label %do_store_r1c1, label %done")
    ln("")

    ln("do_store_r1c1:")
    ln("  %c_off_r1b = mul i32 %row1, %N")
    ln("  %c_idx_r1c1 = add i32 %c_off_r1b, %col1")
    ln("  %c_gep_r1c1 = getelementptr float, float addrspace(1)* %C, i32 %c_idx_r1c1")
    ln("  store float %final_r1c1, float addrspace(1)* %c_gep_r1c1")
    ln("  br label %done")
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
