# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR Tensor Core (wmma) matmul emitter for Phase 5.

Generates LLVM IR using inline PTX assembly for NVIDIA wmma instructions.
This enables Tensor Core acceleration in the LLVM backend, matching the
CUDA-C backend's TC path performance.

Algorithm:
  - m16n16k16 wmma (fp16 inputs, fp32 accumulation)
  - BM=64, BN=64, BK=16
  - 4 warps (128 threads), each warp handles 16×64 of output
  - Each warp iterates over 4 column-tiles of 16
  - Single-buffered shared memory (simpler, correctness-first)
  - f32 global loads → fptrunc to fp16 → store to shared → wmma

Verified: inline PTX wmma with .f32.f32 syntax compiles through
llc (LLVM 20) + ptxas (CUDA 13.2) targeting sm_86.
"""

from __future__ import annotations


def _gen_tiled_matmul_ir_wmma(
    kernel_name: str, M: int, K: int, N: int,
) -> str:
    """Generate LLVM IR for wmma tensor-core matmul.

    C[M,N] = A[M,K] @ B[K,N], fp16 TC with fp32 accumulation.
    Requires M%64==0, N%64==0, K%16==0.
    """
    BM, BN, BK = 64, 64, 16
    WARPS = 4
    THREADS = WARPS * 32  # 128
    num_tiles = K // BK

    lines: list[str] = []
    ln = lines.append

    # ── Module header ──
    ln('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"')
    ln('target triple = "nvptx64-nvidia-cuda"')
    ln("")

    # ── Shared memory: single-buffered ──
    # shmem_A[BM][BK] = [64][16] as half
    # shmem_B[BK][BN] = [16][64] as half
    ln(f"@shmem_A = internal addrspace(3) global [{BM} x [{BK} x half]] zeroinitializer")
    ln(f"@shmem_B = internal addrspace(3) global [{BK} x [{BN} x half]] zeroinitializer")
    ln("")

    # ── NVVM intrinsics ──
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")
    ln("declare void @llvm.nvvm.barrier0()")
    ln("")

    # ── Kernel function ──
    ln(f"define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C, i32 %M, i32 %K, i32 %N) {{")
    ln("entry:")

    # Thread/block IDs
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")
    ln("")

    # Warp ID and warp row
    ln("  %warp_id = lshr i32 %tid, 5")  # tid / 32
    ln("  %warp_row = mul i32 %warp_id, 16")
    ln("")

    # Block base coordinates
    ln(f"  %bm = mul i32 %by, {BM}")  # base row
    ln(f"  %bn = mul i32 %bx, {BN}")  # base col
    ln("")

    # Jump to tile loop
    ln("  br label %tile_loop")
    ln("")

    # ── Tile loop ──
    ln("tile_loop:")
    ln("  %t = phi i32 [ 0, %entry ], [ %t_next, %tile_loop ]")

    # Phi nodes for 4 col-tiles × 8 accumulator regs = 32 accumulators
    for ct in range(4):
        for r in range(8):
            ln(f"  %acc_c{ct}_r{r} = phi float [ 0.0, %entry ], [ %acc_c{ct}_r{r}_new, %tile_loop ]")
    ln("")

    # ── Load tile into shared memory ──
    # Total elements: A = BM*BK = 1024, B = BK*BN = 1024
    # 128 threads → 8 elements each for A, 8 for B
    # Each thread: for i in 0..7: load A[flat], convert f32→f16, store to shmem
    ln(f"  %tile_k_base = mul i32 %t, {BK}")
    ln("")

    # Load A into shared: shmem_A[row][col] = (half)A[bm+row, tile_k_base+col]
    # flat_idx_base = tid * 8
    ln("  %a_flat_base = mul i32 %tid, 8")
    for i in range(8):
        sfx = f"la{i}"
        ln(f"  %{sfx}_flat = add i32 %a_flat_base, {i}")
        ln(f"  %{sfx}_row = lshr i32 %{sfx}_flat, 4")  # flat / 16 (BK=16)
        ln(f"  %{sfx}_col = and i32 %{sfx}_flat, 15")   # flat % 16
        # Global address: A[(bm + row) * K + (tile_k_base + col)]
        ln(f"  %{sfx}_grow = add i32 %bm, %{sfx}_row")
        ln(f"  %{sfx}_gcol = add i32 %tile_k_base, %{sfx}_col")
        ln(f"  %{sfx}_gidx = mul i32 %{sfx}_grow, %K")
        ln(f"  %{sfx}_gidx2 = add i32 %{sfx}_gidx, %{sfx}_gcol")
        ln(f"  %{sfx}_gep = getelementptr float, float addrspace(1)* %A, i32 %{sfx}_gidx2")
        ln(f"  %{sfx}_f32 = load float, float addrspace(1)* %{sfx}_gep")
        ln(f"  %{sfx}_f16 = fptrunc float %{sfx}_f32 to half")
        # Store to shmem_A[row][col]
        ln(f"  %{sfx}_sptr = getelementptr [{BM} x [{BK} x half]], [{BM} x [{BK} x half]] addrspace(3)* @shmem_A, i32 0, i32 %{sfx}_row, i32 %{sfx}_col")
        ln(f"  store half %{sfx}_f16, half addrspace(3)* %{sfx}_sptr")
    ln("")

    # Load B into shared: shmem_B[row][col] = (half)B[tile_k_base+row, bn+col]
    ln("  %b_flat_base = mul i32 %tid, 8")
    for i in range(8):
        sfx = f"lb{i}"
        ln(f"  %{sfx}_flat = add i32 %b_flat_base, {i}")
        ln(f"  %{sfx}_row = lshr i32 %{sfx}_flat, 6")  # flat / 64 (BN=64)
        ln(f"  %{sfx}_col = and i32 %{sfx}_flat, 63")   # flat % 64
        # Global address: B[(tile_k_base + row) * N + (bn + col)]
        ln(f"  %{sfx}_grow = add i32 %tile_k_base, %{sfx}_row")
        ln(f"  %{sfx}_gcol = add i32 %bn, %{sfx}_col")
        ln(f"  %{sfx}_gidx = mul i32 %{sfx}_grow, %N")
        ln(f"  %{sfx}_gidx2 = add i32 %{sfx}_gidx, %{sfx}_gcol")
        ln(f"  %{sfx}_gep = getelementptr float, float addrspace(1)* %B, i32 %{sfx}_gidx2")
        ln(f"  %{sfx}_f32 = load float, float addrspace(1)* %{sfx}_gep")
        ln(f"  %{sfx}_f16 = fptrunc float %{sfx}_f32 to half")
        # Store to shmem_B[row][col]
        ln(f"  %{sfx}_sptr = getelementptr [{BK} x [{BN} x half]], [{BK} x [{BN} x half]] addrspace(3)* @shmem_B, i32 0, i32 %{sfx}_row, i32 %{sfx}_col")
        ln(f"  store half %{sfx}_f16, half addrspace(3)* %{sfx}_sptr")
    ln("")

    # Barrier after loading shared memory
    ln("  call void @llvm.nvvm.barrier0()")
    ln("")

    # ── Compute: wmma for each of 4 column tiles ──
    # sA ptr for this warp: &shmem_A[warp_row][0]
    ln(f"  %sa_warp_ptr = getelementptr [{BM} x [{BK} x half]], [{BM} x [{BK} x half]] addrspace(3)* @shmem_A, i32 0, i32 %warp_row, i32 0")
    ln("")

    # Load A fragment ONCE (shared across all 4 col-tiles)
    ln(f"  %a_frag = call {{i32, i32, i32, i32, i32, i32, i32, i32}} asm sideeffect \"wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {{$0,$1,$2,$3,$4,$5,$6,$7}}, [$8], $9;\", \"=r,=r,=r,=r,=r,=r,=r,=r,l,r\"(half addrspace(3)* %sa_warp_ptr, i32 {BK})")
    for r in range(8):
        ln(f"  %a_r{r} = extractvalue {{i32, i32, i32, i32, i32, i32, i32, i32}} %a_frag, {r}")
    ln("")

    for ct in range(4):
        # sB ptr for col_tile ct: &shmem_B[0][ct*16]
        col_offset = ct * 16
        ln(f"  %sb_ct{ct}_ptr = getelementptr [{BK} x [{BN} x half]], [{BK} x [{BN} x half]] addrspace(3)* @shmem_B, i32 0, i32 0, i32 {col_offset}")
        ln("")

        # wmma.load.b (row, stride=BN=64 elements)
        ln(f"  %b_frag_ct{ct} = call {{i32, i32, i32, i32, i32, i32, i32, i32}} asm sideeffect \"wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 {{$0,$1,$2,$3,$4,$5,$6,$7}}, [$8], $9;\", \"=r,=r,=r,=r,=r,=r,=r,=r,l,r\"(half addrspace(3)* %sb_ct{ct}_ptr, i32 {BN})")
        for r in range(8):
            ln(f"  %b_ct{ct}_r{r} = extractvalue {{i32, i32, i32, i32, i32, i32, i32, i32}} %b_frag_ct{ct}, {r}")
        ln("")

        # wmma.mma (row.row, f32.f32) — reuse A fragment
        a_args = ", ".join([f"i32 %a_r{r}" for r in range(8)])
        b_args = ", ".join([f"i32 %b_ct{ct}_r{r}" for r in range(8)])
        c_args = ", ".join([f"float %acc_c{ct}_r{r}" for r in range(8)])
        all_args = f"{a_args}, {b_args}, {c_args}"

        ln(f"  %mma_ct{ct} = call {{float, float, float, float, float, float, float, float}} asm sideeffect \"wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {{$0,$1,$2,$3,$4,$5,$6,$7}}, {{$8,$9,$10,$11,$12,$13,$14,$15}}, {{$16,$17,$18,$19,$20,$21,$22,$23}}, {{$24,$25,$26,$27,$28,$29,$30,$31}};\", \"=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f\"({all_args})")
        for r in range(8):
            ln(f"  %acc_c{ct}_r{r}_new = extractvalue {{float, float, float, float, float, float, float, float}} %mma_ct{ct}, {r}")
        ln("")

    # Barrier before next tile load
    ln("  call void @llvm.nvvm.barrier0()")
    ln("")

    # ── Tile loop latch (same block, no new label needed) ──
    ln("  %t_next = add i32 %t, 1")
    ln(f"  %t_cond = icmp slt i32 %t_next, {num_tiles}")
    ln("  br i1 %t_cond, label %tile_loop, label %store_results")
    ln("")

    # ── Store results ──
    ln("store_results:")

    # Global row for this warp: bm + warp_row
    ln("  %out_row = add i32 %bm, %warp_row")
    ln("")

    for ct in range(4):
        col_offset = ct * 16
        # Global ptr: &C[out_row * N + bn + col_offset]
        ln(f"  %out_col_ct{ct} = add i32 %bn, {col_offset}")
        ln(f"  %out_off_ct{ct} = mul i32 %out_row, %N")
        ln(f"  %out_idx_ct{ct} = add i32 %out_off_ct{ct}, %out_col_ct{ct}")
        ln(f"  %out_ptr_ct{ct} = getelementptr float, float addrspace(1)* %C, i32 %out_idx_ct{ct}")
        ln("")

        # wmma.store.d (row, stride=N)
        d_args = ", ".join([f"float %acc_c{ct}_r{r}_new" for r in range(8)])
        ln(f"  call void asm sideeffect \"wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [$0], {{$1,$2,$3,$4,$5,$6,$7,$8}}, $9;\", \"l,f,f,f,f,f,f,f,f,r\"(float addrspace(1)* %out_ptr_ct{ct}, {d_args}, i32 %N)")
        ln("")

    ln("  ret void")
    ln("}")
    ln("")

    # ── NVVM annotations ──
    ln("!nvvm.annotations = !{!0}")
    ln(f"!0 = !{{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @{kernel_name}, !\"kernel\", i32 1}}")
    ln("")

    return "\n".join(lines)


if __name__ == "__main__":
    import subprocess
    import sys

    ir = _gen_tiled_matmul_ir_wmma("arke_matmul_tc_1024x1024x1024", 1024, 1024, 1024)

    ll_path = "/tmp/test_wmma_matmul.ll"
    ptx_path = "/tmp/test_wmma_matmul.ptx"
    cubin_path = "/tmp/test_wmma_matmul.cubin"

    with open(ll_path, "w") as f:
        f.write(ir)

    print(f"Generated {len(ir)} bytes of LLVM IR")

    # Compile: llc
    llc = "/home/blueyi/opt/llvm20-src/usr/lib/llvm-20/bin/llc"
    r = subprocess.run(
        [llc, "-march=nvptx64", "-mcpu=sm_86", "-O2", ll_path, "-o", ptx_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"llc FAILED:\n{r.stderr}")
        sys.exit(1)
    print("llc: OK")

    # Compile: ptxas
    ptxas = "/usr/local/cuda-13.2/bin/ptxas"
    r = subprocess.run(
        [ptxas, "--gpu-name", "sm_86", "-O2", ptx_path, "-o", cubin_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"ptxas FAILED:\n{r.stderr}")
        sys.exit(1)
    print("ptxas: OK")

    import os
    print(f"cubin size: {os.path.getsize(cubin_path)} bytes")
    print("SUCCESS: LLVM IR → PTX → cubin (wmma TC matmul)")
