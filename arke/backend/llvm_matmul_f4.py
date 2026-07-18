# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""float4-vectorized double-buffered FP32 matmul emitter (Phase 5).

Matches the CUDA-C _emit_matmul_f4_doublebuf kernel: each thread loads 4
contiguous floats per global access via LLVM <4 x float> vector loads
(which llc lowers to ld.global.v4.f32), quadrupling per-instruction global
bandwidth over the scalar path.

Requires K%4==0 and N%4==0 (16-byte alignment of the 4-element runs), plus
M%BM==0, N%BN==0, K%BK==0 for the bounds-check-free double-buffer path.

Algorithm: BM=64, BN=64, BK=16, TM=TN=4, Block=(16,16)=256 threads.
Each thread computes a 4x4 output sub-tile (16 accumulators).
"""

from __future__ import annotations


def _gen_tiled_matmul_ir_f4_doublebuf(
    kernel_name: str, M: int, K: int, N: int,
    BM: int, BN: int, BK: int, num_tiles: int,
) -> str:
    """float4-vectorized double-buffered tiled matmul as LLVM IR.

    Each thread's 4 cooperative-load elements form a contiguous 16-byte run
    in global memory (same A row / same B row), loaded as <4 x float>.
    The 4 elements also land contiguously in shared memory (col c..c+3 within
    the same shmem row), so the store is a <4 x float> vector store.
    """
    TM, TN = 4, 4
    BLOCK_X, BLOCK_Y = 16, 16

    lines: list[str] = []
    ln = lines.append

    # -- Module header --
    ln('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"')
    ln('target triple = "nvptx64-nvidia-cuda"')
    ln("")

    # -- Double-buffered shared memory --
    ln(f"@shmem_A = internal addrspace(3) global [2 x [{BM} x [{BK} x float]]] undef")
    ln(f"@shmem_B = internal addrspace(3) global [2 x [{BK} x [{BN} x float]]] undef")
    ln("")

    # -- NVVM intrinsics --
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")
    ln("declare void @llvm.nvvm.barrier0()")
    ln("")

    # -- Kernel function --
    ln(f"define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C, i32 %M, i32 %K, i32 %N) {{")
    ln("entry:")

    ln("  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()")
    ln("  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")

    ln(f"  %row_base = mul i32 %by, {BM}")
    ln(f"  %ty_x_tm = mul i32 %ty, {TM}")
    ln("  %row0 = add i32 %row_base, %ty_x_tm")
    ln(f"  %col_base = mul i32 %bx, {BN}")
    ln(f"  %tx_x_tn = mul i32 %tx, {TN}")
    ln("  %col0 = add i32 %col_base, %tx_x_tn")

    ln(f"  %lin_id = mul i32 %ty, {BLOCK_X}")
    ln("  %lin_tid = add i32 %lin_id, %tx")

    # Each thread loads a contiguous run of 4 elements (flat_base .. +3).
    ln("  %flat_base = mul i32 %lin_tid, 4")

    # A cooperative-load indices: 4 contiguous elements within one shmem row.
    # flat_base is a multiple of 4; BK=16 so the run [flat_base, flat_base+3]
    # never crosses a 16-boundary -> srow constant, scol = flat_base%16.
    ln("  %sa_srow = lshr i32 %flat_base, 4")   # / BK(16)
    ln("  %sa_scol = and i32 %flat_base, 15")   # % BK(16)
    ln("  %sa_grow = add i32 %row_base, %sa_srow")

    # B cooperative-load indices: BN=64, run never crosses a 64-boundary.
    ln("  %sb_srow = lshr i32 %flat_base, 6")   # / BN(64)
    ln("  %sb_scol = and i32 %flat_base, 63")   # % BN(64)
    ln("  %sb_gcol = add i32 %col_base, %sb_scol")

    # ---- Preload tile 0 into buffer 0 (float4) ----
    # A: load A[sa_grow, sa_scol .. +3] as <4 x float>
    ln("  %pre_a_gidx = mul i32 %sa_grow, %K")
    ln("  %pre_a_gidx2 = add i32 %pre_a_gidx, %sa_scol")
    ln("  %pre_a_gep = getelementptr float, float addrspace(1)* %A, i32 %pre_a_gidx2")
    ln("  %pre_a_gep4 = bitcast float addrspace(1)* %pre_a_gep to <4 x float> addrspace(1)*")
    ln("  %pre_a_vec = load <4 x float>, <4 x float> addrspace(1)* %pre_a_gep4, align 16")
    ln("  %pre_a_sptr = getelementptr [2 x [{0} x [{1} x float]]], [2 x [{0} x [{1} x float]]] addrspace(3)* @shmem_A, i32 0, i32 0, i32 %sa_srow, i32 %sa_scol".format(BM, BK))
    ln("  %pre_a_sptr4 = bitcast float addrspace(3)* %pre_a_sptr to <4 x float> addrspace(3)*")
    ln("  store <4 x float> %pre_a_vec, <4 x float> addrspace(3)* %pre_a_sptr4, align 16")

    # B: load B[sb_srow, sb_gcol .. +3] as <4 x float>
    ln("  %pre_b_gidx = mul i32 %sb_srow, %N")
    ln("  %pre_b_gidx2 = add i32 %pre_b_gidx, %sb_gcol")
    ln("  %pre_b_gep = getelementptr float, float addrspace(1)* %B, i32 %pre_b_gidx2")
    ln("  %pre_b_gep4 = bitcast float addrspace(1)* %pre_b_gep to <4 x float> addrspace(1)*")
    ln("  %pre_b_vec = load <4 x float>, <4 x float> addrspace(1)* %pre_b_gep4, align 16")
    ln("  %pre_b_sptr = getelementptr [2 x [{0} x [{1} x float]]], [2 x [{0} x [{1} x float]]] addrspace(3)* @shmem_B, i32 0, i32 0, i32 %sb_srow, i32 %sb_scol".format(BK, BN))
    ln("  %pre_b_sptr4 = bitcast float addrspace(3)* %pre_b_sptr to <4 x float> addrspace(3)*")
    ln("  store <4 x float> %pre_b_vec, <4 x float> addrspace(3)* %pre_b_sptr4, align 16")

    ln("  call void @llvm.nvvm.barrier0()")
    ln("  br label %tile_loop_header")
    ln("")

    # ---- Tile loop header ----
    ln("tile_loop_header:")
    ln("  %t = phi i32 [ 0, %entry ], [ %t_next, %tile_loop_latch ]")
    ln("  %buf = phi i32 [ 0, %entry ], [ %buf_next, %tile_loop_latch ]")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %acc_r{r}c{c} = phi float [ 0.0, %entry ], [ %acc_r{r}c{c}_out, %tile_loop_latch ]")
    ln(f"  %tile_done = icmp sge i32 %t, {num_tiles}")
    ln("  br i1 %tile_done, label %store_result, label %tile_body")
    ln("")

    # ---- Tile body ----
    ln("tile_body:")
    ln("  %buf_next = xor i32 %buf, 1")
    ln("  %t_next_tile = add i32 %t, 1")
    ln(f"  %has_next = icmp slt i32 %t_next_tile, {num_tiles}")
    ln("  br i1 %has_next, label %load_next, label %compute")
    ln("")

    # ---- Load next tile into buf_next (float4) ----
    ln("load_next:")
    ln(f"  %next_tile_bk = mul i32 %t_next_tile, {BK}")
    # A
    ln("  %nla_gcol = add i32 %next_tile_bk, %sa_scol")
    ln("  %nla_gidx = mul i32 %sa_grow, %K")
    ln("  %nla_gidx2 = add i32 %nla_gidx, %nla_gcol")
    ln("  %nla_gep = getelementptr float, float addrspace(1)* %A, i32 %nla_gidx2")
    ln("  %nla_gep4 = bitcast float addrspace(1)* %nla_gep to <4 x float> addrspace(1)*")
    ln("  %nla_vec = load <4 x float>, <4 x float> addrspace(1)* %nla_gep4, align 16")
    ln("  %nla_sptr = getelementptr [2 x [{0} x [{1} x float]]], [2 x [{0} x [{1} x float]]] addrspace(3)* @shmem_A, i32 0, i32 %buf_next, i32 %sa_srow, i32 %sa_scol".format(BM, BK))
    ln("  %nla_sptr4 = bitcast float addrspace(3)* %nla_sptr to <4 x float> addrspace(3)*")
    ln("  store <4 x float> %nla_vec, <4 x float> addrspace(3)* %nla_sptr4, align 16")
    # B
    ln("  %nlb_grow = add i32 %next_tile_bk, %sb_srow")
    ln("  %nlb_gidx = mul i32 %nlb_grow, %N")
    ln("  %nlb_gidx2 = add i32 %nlb_gidx, %sb_gcol")
    ln("  %nlb_gep = getelementptr float, float addrspace(1)* %B, i32 %nlb_gidx2")
    ln("  %nlb_gep4 = bitcast float addrspace(1)* %nlb_gep to <4 x float> addrspace(1)*")
    ln("  %nlb_vec = load <4 x float>, <4 x float> addrspace(1)* %nlb_gep4, align 16")
    ln("  %nlb_sptr = getelementptr [2 x [{0} x [{1} x float]]], [2 x [{0} x [{1} x float]]] addrspace(3)* @shmem_B, i32 0, i32 %buf_next, i32 %sb_srow, i32 %sb_scol".format(BK, BN))
    ln("  %nlb_sptr4 = bitcast float addrspace(3)* %nlb_sptr to <4 x float> addrspace(3)*")
    ln("  store <4 x float> %nlb_vec, <4 x float> addrspace(3)* %nlb_sptr4, align 16")
    ln("  br label %compute")
    ln("")

    # ---- Compute on current buffer ----
    ln("compute:")
    ln("  br label %k_loop_header")
    ln("")

    ln("k_loop_header:")
    ln("  %ki = phi i32 [ 0, %compute ], [ %ki_next, %k_loop_body ]")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %acc_k_r{r}c{c} = phi float [ %acc_r{r}c{c}, %compute ], [ %acc_k_r{r}c{c}_next, %k_loop_body ]")
    ln(f"  %k_done = icmp sge i32 %ki, {BK}")
    ln("  br i1 %k_done, label %k_loop_exit, label %k_loop_body")
    ln("")

    ln("k_loop_body:")
    for r in range(TM):
        row_reg = "%ty_x_tm" if r == 0 else f"%ty_tm_p{r}"
        if r > 0:
            ln(f"  %ty_tm_p{r} = add i32 %ty_x_tm, {r}")
        ln(f"  %sa_r{r}_ptr = getelementptr [2 x [{BM} x [{BK} x float]]], [2 x [{BM} x [{BK} x float]]] addrspace(3)* @shmem_A, i32 0, i32 %buf, i32 {row_reg}, i32 %ki")
        ln(f"  %a_reg{r} = load float, float addrspace(3)* %sa_r{r}_ptr")

    for c in range(TN):
        col_reg = "%tx_x_tn" if c == 0 else f"%tx_tn_p{c}"
        if c > 0:
            ln(f"  %tx_tn_p{c} = add i32 %tx_x_tn, {c}")
        ln(f"  %sb_c{c}_ptr = getelementptr [2 x [{BK} x [{BN} x float]]], [2 x [{BK} x [{BN} x float]]] addrspace(3)* @shmem_B, i32 0, i32 %buf, i32 %ki, i32 {col_reg}")
        ln(f"  %b_reg{c} = load float, float addrspace(3)* %sb_c{c}_ptr")

    for r in range(TM):
        for c in range(TN):
            # contract fast-math flag lets llc fuse fmul+fadd into a single
            # fma.rn.f32 (2 PTX instrs -> 1), matching nvcc's default FMA codegen.
            ln(f"  %prod_r{r}c{c} = fmul contract float %a_reg{r}, %b_reg{c}")
            ln(f"  %acc_k_r{r}c{c}_next = fadd contract float %acc_k_r{r}c{c}, %prod_r{r}c{c}")

    ln("  %ki_next = add i32 %ki, 1")
    ln("  br label %k_loop_header")
    ln("")

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

    # ---- Store 4x4 results ----
    ln("store_result:")
    for r in range(TM):
        for c in range(TN):
            ln(f"  %final_r{r}c{c} = phi float [ %acc_r{r}c{c}, %tile_loop_header ]")

    for r in range(TM):
        for c in range(TN):
            if r > 0 and c == 0:
                ln(f"  %row_r{r} = add i32 %row0, {r}")
            if c > 0 and r == 0:
                ln(f"  %col_c{c} = add i32 %col0, {c}")

    for r in range(TM):
        for c in range(TN):
            row_r = "%row0" if r == 0 else f"%row_r{r}"
            col_c = "%col0" if c == 0 else f"%col_c{c}"
            ln(f"  %c_off_r{r}c{c} = mul i32 {row_r}, {N}")
            ln(f"  %c_idx_r{r}c{c} = add i32 %c_off_r{r}c{c}, {col_c}")
            ln(f"  %c_gep_r{r}c{c} = getelementptr float, float addrspace(1)* %C, i32 %c_idx_r{r}c{c}")
            ln(f"  store float %final_r{r}c{c}, float addrspace(1)* %c_gep_r{r}c{c}")

    ln("  ret void")
    ln("}")
    ln("")

    ln("!nvvm.annotations = !{!0}")
    ln(f'!0 = !{{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @{kernel_name}, !"kernel", i32 1}}')
    ln("")

    return "\n".join(lines)


if __name__ == "__main__":
    import subprocess
    import sys

    ir = _gen_tiled_matmul_ir_f4_doublebuf("arke_matmul_f4_512x512x512", 512, 512, 512, 64, 64, 16, 512 // 16)
    with open("/tmp/test_f4_matmul.ll", "w") as f:
        f.write(ir)
    print(f"Generated {len(ir)} bytes")

    llc = "/home/blueyi/opt/llvm20-src/usr/lib/llvm-20/bin/llc"
    r = subprocess.run([llc, "-march=nvptx64", "-mcpu=sm_86", "-O2", "/tmp/test_f4_matmul.ll", "-o", "/tmp/test_f4_matmul.ptx"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"llc FAILED:\n{r.stderr}")
        sys.exit(1)
    print("llc: OK")

    ptxas = "/usr/local/cuda-13.2/bin/ptxas"
    r = subprocess.run([ptxas, "--gpu-name", "sm_86", "-O2", "/tmp/test_f4_matmul.ptx", "-o", "/tmp/test_f4_matmul.cubin"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ptxas FAILED:\n{r.stderr}")
        sys.exit(1)
    print("ptxas: OK")

    # Check for v4 loads
    with open("/tmp/test_f4_matmul.ptx") as f:
        ptx = f.read()
    v4_count = ptx.count("ld.global.v4")
    print(f"ld.global.v4 count: {v4_count}")
    print("SUCCESS")
