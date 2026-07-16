# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR emitter for Arke (Phase 5, P5-S1).

Generates LLVM IR text targeting nvptx64-nvidia-cuda for compilation via
llc + ptxas.  The emitted IR uses:
  - address space 1 for global memory pointers
  - address space 3 for shared memory (module-level globals)
  - NVVM intrinsics for thread/block IDs and barriers
  - !nvvm.annotations metadata for kernel entry points
"""

from __future__ import annotations

from arke.backend.cuda_c_backend import CudaCKernel
from arke.ir.graph import IRGraph


def emit_llvm_ir_matmul(graph: IRGraph, chip: str = "sm_86") -> CudaCKernel:
    """Emit a tiled shared-memory matmul kernel as LLVM IR text.

    C[M,N] = A[M,K] @ B[K,N]

    Tile sizes: BM=16, BN=16, BK=16.  Each thread computes one C element.
    Grid: (ceil(N/BN), ceil(M/BM), 1), Block: (BN, BM, 1).
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
    BM, BN, BK = 16, 16, 16
    num_tiles = (K + BK - 1) // BK

    kernel_name = f"arke_matmul_{M}x{N}x{K}"

    source = _gen_tiled_matmul_ir(kernel_name, M, K, N, BM, BN, BK, num_tiles)

    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (BN, BM, 1)

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
    """Generate the complete LLVM IR text for a tiled matmul kernel.

    The kernel implements:
      row = blockIdx.y * BM + threadIdx.y
      col = blockIdx.x * BN + threadIdx.x
      acc = 0.0
      for t in 0..num_tiles:
          load A[row, t*BK+tx] -> shmem_A[ty][tx]  (with bounds check)
          load B[t*BK+ty, col] -> shmem_B[ty][tx]   (with bounds check)
          barrier
          for k in 0..BK:  acc += shmem_A[ty][k] * shmem_B[k][tx]
          barrier
      if row < M and col < N: C[row*N+col] = acc

    The inner k-loop is fully unrolled in the IR for clarity and performance.
    """
    lines = []
    ln = lines.append

    # ── Module header ──
    ln('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"')
    ln('target triple = "nvptx64-nvidia-cuda"')
    ln("")

    # ── Shared memory ──
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

    # row = by * BM + ty, col = bx * BN + tx
    ln(f"  %row_base = mul i32 %by, {BM}")
    ln("  %row = add i32 %row_base, %ty")
    ln(f"  %col_base = mul i32 %bx, {BN}")
    ln("  %col = add i32 %col_base, %tx")

    # acc = 0.0
    ln("  br label %tile_loop_header")
    ln("")

    # ── Tile loop header (PHI) ──
    ln("tile_loop_header:")
    ln(f"  %t = phi i32 [ 0, %entry ], [ %t_next, %tile_loop_latch ]")
    ln(f"  %acc = phi float [ 0.0, %entry ], [ %acc_after_k, %tile_loop_latch ]")
    ln(f"  %tile_done = icmp sge i32 %t, {num_tiles}")
    ln("  br i1 %tile_done, label %store_result, label %load_tile")
    ln("")

    # ── Load tile: A[row, t*BK+tx] -> shmem_A[ty][tx], B[t*BK+ty, col] -> shmem_B[ty][tx] ──
    ln("load_tile:")

    # a_col = t * BK + tx
    ln(f"  %a_col = add i32 %tx, 0")  # placeholder
    ln(f"  %t_times_bk = mul i32 %t, {BK}")
    ln("  %a_col_real = add i32 %t_times_bk, %tx")

    # bounds check for A: row < M && a_col < K
    ln("  %a_row_ok = icmp slt i32 %row, %M")
    ln("  %a_col_ok = icmp slt i32 %a_col_real, %K")
    ln("  %a_valid = and i1 %a_row_ok, %a_col_ok")
    ln("  br i1 %a_valid, label %load_a_valid, label %load_a_zero")
    ln("")

    ln("load_a_valid:")
    ln("  %a_offset = mul i32 %row, %K")
    ln("  %a_idx = add i32 %a_offset, %a_col_real")
    ln("  %a_gep = getelementptr float, float addrspace(1)* %A, i32 %a_idx")
    ln("  %a_val = load float, float addrspace(1)* %a_gep")
    ln("  br label %store_shmem_a")
    ln("")

    ln("load_a_zero:")
    ln("  br label %store_shmem_a")
    ln("")

    ln("store_shmem_a:")
    ln("  %a_data = phi float [ %a_val, %load_a_valid ], [ 0.0, %load_a_zero ]")
    ln(f"  %sa_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %ty, i32 %tx")
    ln("  store float %a_data, float addrspace(3)* %sa_ptr")

    # b_row = t * BK + ty
    ln("  %b_row = add i32 %t_times_bk, %ty")

    # bounds check for B: b_row < K && col < N
    ln("  %b_row_ok = icmp slt i32 %b_row, %K")
    ln("  %b_col_ok = icmp slt i32 %col, %N")
    ln("  %b_valid = and i1 %b_row_ok, %b_col_ok")
    ln("  br i1 %b_valid, label %load_b_valid, label %load_b_zero")
    ln("")

    ln("load_b_valid:")
    ln("  %b_offset = mul i32 %b_row, %N")
    ln("  %b_idx = add i32 %b_offset, %col")
    ln("  %b_gep = getelementptr float, float addrspace(1)* %B, i32 %b_idx")
    ln("  %b_val = load float, float addrspace(1)* %b_gep")
    ln("  br label %store_shmem_b")
    ln("")

    ln("load_b_zero:")
    ln("  br label %store_shmem_b")
    ln("")

    ln("store_shmem_b:")
    ln("  %b_data = phi float [ %b_val, %load_b_valid ], [ 0.0, %load_b_zero ]")
    ln(f"  %sb_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %ty, i32 %tx")
    ln("  store float %b_data, float addrspace(3)* %sb_ptr")

    # __syncthreads
    ln("  call void @llvm.nvvm.barrier0()")

    # ── Unrolled k-loop: acc += shmem_A[ty][k] * shmem_B[k][tx] ──
    # Generate as a loop to keep IR manageable for larger BK
    ln("  br label %k_loop_header")
    ln("")

    ln("k_loop_header:")
    ln(f"  %ki = phi i32 [ 0, %store_shmem_b ], [ %ki_next, %k_loop_body ]")
    ln(f"  %acc_k = phi float [ %acc, %store_shmem_b ], [ %acc_k_next, %k_loop_body ]")
    ln(f"  %k_done = icmp sge i32 %ki, {BK}")
    ln("  br i1 %k_done, label %k_loop_exit, label %k_loop_body")
    ln("")

    ln("k_loop_body:")
    # Load shmem_A[ty][ki]
    ln(f"  %sa_k_ptr = getelementptr [{BM} x [{BK} x float]], [{BM} x [{BK} x float]] addrspace(3)* @shmem_A, i32 0, i32 %ty, i32 %ki")
    ln("  %sa_k_val = load float, float addrspace(3)* %sa_k_ptr")
    # Load shmem_B[ki][tx]
    ln(f"  %sb_k_ptr = getelementptr [{BK} x [{BN} x float]], [{BK} x [{BN} x float]] addrspace(3)* @shmem_B, i32 0, i32 %ki, i32 %tx")
    ln("  %sb_k_val = load float, float addrspace(3)* %sb_k_ptr")
    # acc += a * b
    ln("  %prod = fmul float %sa_k_val, %sb_k_val")
    ln("  %acc_k_next = fadd float %acc_k, %prod")
    ln("  %ki_next = add i32 %ki, 1")
    ln("  br label %k_loop_header")
    ln("")

    ln("k_loop_exit:")
    # __syncthreads
    ln("  call void @llvm.nvvm.barrier0()")

    # Increment tile counter
    ln("  br label %tile_loop_latch")
    ln("")

    ln("tile_loop_latch:")
    ln("  %acc_after_k = phi float [ %acc_k, %k_loop_exit ]")
    ln("  %t_next = add i32 %t, 1")
    ln("  br label %tile_loop_header")
    ln("")

    # ── Store result: C[row*N+col] = acc ──
    ln("store_result:")
    ln("  %final_acc = phi float [ %acc, %tile_loop_header ]")
    ln("  %row_ok = icmp slt i32 %row, %M")
    ln("  %col_ok = icmp slt i32 %col, %N")
    ln("  %both_ok = and i1 %row_ok, %col_ok")
    ln("  br i1 %both_ok, label %do_store, label %done")
    ln("")

    ln("do_store:")
    ln("  %c_offset = mul i32 %row, %N")
    ln("  %c_idx = add i32 %c_offset, %col")
    ln("  %c_gep = getelementptr float, float addrspace(1)* %C, i32 %c_idx")
    ln("  store float %final_acc, float addrspace(1)* %c_gep")
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
