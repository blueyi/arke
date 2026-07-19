# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVM IR Tensor Core (wmma) matmul emitter for Phase 5.

Generates LLVM IR using inline PTX assembly for NVIDIA wmma instructions.
This enables Tensor Core acceleration in the LLVM backend, matching the
CUDA-C backend's TC path performance.

Algorithm (2x2 warp grid — matches CUDA-C MMAConfig WM=2,WN=2,WTM=2,WTN=4):
  - m16n16k16 wmma (fp16 inputs, fp32 accumulation)
  - BM=64, BN=128, BK=16 block tile
  - 4 warps arranged in a 2x2 grid (WM=2 x WN=2), 128 threads
  - Each warp owns a WTM*16 x WTN*16 = 32x64 sub-tile:
      warp_m = warp_id / WN,  warp_n = warp_id % WN
      rows [warp_m*32, +32), cols [warp_n*64, +64)
  - Per K-tile each warp: loads WTM=2 A-fragments + WTN=4 B-fragments,
    computes WTM*WTN=8 mma. Fragments are reused bidirectionally:
    each A-frag feeds WTN mma, each B-frag feeds WTM mma.
    → 6 fragment loads / 8 mma (vs 9/8 for a 1x4 warp strip).
  - Software-pipelined DOUBLE-BUFFERED shared memory: the NEXT K-tile is
    staged (global load -> fptrunc -> shared store) into the alternate
    buffer WHILE the current tile's wmma fragments are loaded + computed.
    See the "Double buffering in LLVM IR" note below for why this pays off
    here where the earlier fragment-level double-buffer attempt did not.
  - f32 global loads -> fptrunc to fp16 -> store to shared -> wmma

Double buffering in LLVM IR (2026-07-18, replicating CUDA-C strategy):
  An earlier attempt double-buffered the *wmma fragment loads* (inline PTX
  `sideeffect` asm) and was 31% SLOWER — `sideeffect` blocks the compiler
  from overlapping load and compute. THIS design double-buffers the
  *global->shared staging* instead. Those staging ops are ordinary LLVM
  `load`/`fptrunc`/`store` instructions (NOT sideeffect asm), and they write
  to the ALTERNATE shared buffer, so they carry no dependency on the current
  tile's wmma reads. The instruction scheduler + GPU memory pipeline can
  therefore issue the next tile's global loads while the tensor cores consume
  the current buffer — exactly the latency-hiding CUDA-C gets from its
  LOAD_TILE(nxt) / compute(cur) split. Control flow is block-uniform (the
  K-tile index is identical across all threads), so the single per-iteration
  barrier is safe; it is emitted as inline `bar.sync 0` asm to keep the LLVM
  NVPTX structurizer from folding it into a conditional region.

Register note (sm_86: 64K reg/SM, 48KB smem, 1536 threads/SM):
  WTM*WTN*8 = 64 fp32 accumulators/thread. Double buffering doubles shared
  memory (12KB for a 64x128 tile) — well under the 48KB budget.

Key findings:
  - PTX wmma.mma syntax: use '.f32.f32' (NOT '.f32.f16.f16.f32')
  - wmma.load.a.row + wmma.load.b.row + wmma.mma.row.row for C=A*B
  - Constraint 'l' handles addrspace(3) shared mem pointers correctly
  - Double-buffer the STAGING (plain loads), NOT the fragment loads (asm)
"""

from __future__ import annotations

# wmma fragment register/element counts for m16n16k16 f16/f32.
_FRAG_REGS = 8   # a/b fragments: 8 x i32 (each packs 2 halfs)
_ACC_REGS = 8    # c/d fragments: 8 x f32


def _gen_tiled_matmul_ir_wmma(
    kernel_name: str, M: int, K: int, N: int,
    *, WM: int = 2, WN: int = 2, WTM: int = 2, WTN: int = 4,
) -> str:
    """Generate LLVM IR for wmma tensor-core matmul (2x2 warp grid).

    C[M,N] = A[M,K] @ B[K,N], fp16 TC with fp32 accumulation.

    Block tile: BM = WM*WTM*16, BN = WN*WTN*16, BK = 16.
    Threads/block = WM*WN*32. Requires M%BM==0, N%BN==0, K%16==0.
    Defaults (WM=2,WN=2,WTM=2,WTN=4) -> 64x128 tile, 128 threads.

    Software-pipelined double-buffered shared memory: the next K-tile is
    staged into the alternate buffer while the current tile is computed.
    """
    BK = 16
    WARP = 32
    BM = WM * WTM * 16
    BN = WN * WTN * 16
    THREADS = WM * WN * WARP
    num_tiles = K // BK

    a_elems = (BM * BK) // THREADS   # A staging elems/thread
    b_elems = (BK * BN) // THREADS   # B staging elems/thread

    bk_shift = BK.bit_length() - 1   # log2(BK)
    bk_mask = BK - 1
    bn_shift = BN.bit_length() - 1   # log2(BN)
    bn_mask = BN - 1

    lines: list[str] = []
    ln = lines.append

    def frag_ty(n: int, ty: str) -> str:
        return "{" + ", ".join([ty] * n) + "}"

    RTY = frag_ty(_FRAG_REGS, "i32")   # {i32 x8}
    FTY = frag_ty(_ACC_REGS, "float")  # {float x8}

    # Shared-memory array types (double-buffered: leading [2 x ...]).
    ATY = f"[2 x [{BM} x [{BK} x half]]]"
    BTY = f"[2 x [{BK} x [{BN} x half]]]"

    def emit_stage(prefix: str, buf: str, kbase: str) -> None:
        """Emit global->shared staging for one K-tile.

        prefix: unique SSA name prefix (avoid collisions across call sites).
        buf:    SSA/const i32 buffer index ("0" or "%cur"/"%nxt").
        kbase:  SSA value holding the K offset of this tile (elements).
        All staging ops are plain load/fptrunc/store (no sideeffect asm) so
        the scheduler may overlap them with the current tile's wmma compute.
        """
        # Stage A: shmem_A[buf][row][col] = (half)A[bm+row, kbase+col]
        ln(f"  %{prefix}a_flat_base = mul i32 %tid, {a_elems}")
        for i in range(a_elems):
            s = f"{prefix}a{i}"
            ln(f"  %{s}_flat = add i32 %{prefix}a_flat_base, {i}")
            ln(f"  %{s}_row = lshr i32 %{s}_flat, {bk_shift}")
            ln(f"  %{s}_col = and i32 %{s}_flat, {bk_mask}")
            ln(f"  %{s}_grow = add i32 %bm, %{s}_row")
            ln(f"  %{s}_gcol = add i32 {kbase}, %{s}_col")
            ln(f"  %{s}_gidx = mul i32 %{s}_grow, %K")
            ln(f"  %{s}_gidx2 = add i32 %{s}_gidx, %{s}_gcol")
            ln(f"  %{s}_gep = getelementptr float, float addrspace(1)* %A, i32 %{s}_gidx2")
            ln(f"  %{s}_f32 = load float, float addrspace(1)* %{s}_gep")
            ln(f"  %{s}_f16 = fptrunc float %{s}_f32 to half")
            ln(f"  %{s}_sptr = getelementptr {ATY}, {ATY} addrspace(3)* @shmem_A, i32 0, i32 {buf}, i32 %{s}_row, i32 %{s}_col")
            ln(f"  store half %{s}_f16, half addrspace(3)* %{s}_sptr")
        # Stage B: shmem_B[buf][row][col] = (half)B[kbase+row, bn+col]
        ln(f"  %{prefix}b_flat_base = mul i32 %tid, {b_elems}")
        for i in range(b_elems):
            s = f"{prefix}b{i}"
            ln(f"  %{s}_flat = add i32 %{prefix}b_flat_base, {i}")
            ln(f"  %{s}_row = lshr i32 %{s}_flat, {bn_shift}")
            ln(f"  %{s}_col = and i32 %{s}_flat, {bn_mask}")
            ln(f"  %{s}_grow = add i32 {kbase}, %{s}_row")
            ln(f"  %{s}_gcol = add i32 %bn, %{s}_col")
            ln(f"  %{s}_gidx = mul i32 %{s}_grow, %N")
            ln(f"  %{s}_gidx2 = add i32 %{s}_gidx, %{s}_gcol")
            ln(f"  %{s}_gep = getelementptr float, float addrspace(1)* %B, i32 %{s}_gidx2")
            ln(f"  %{s}_f32 = load float, float addrspace(1)* %{s}_gep")
            ln(f"  %{s}_f16 = fptrunc float %{s}_f32 to half")
            ln(f"  %{s}_sptr = getelementptr {BTY}, {BTY} addrspace(3)* @shmem_B, i32 0, i32 {buf}, i32 %{s}_row, i32 %{s}_col")
            ln(f"  store half %{s}_f16, half addrspace(3)* %{s}_sptr")

    # ── Module header ──
    ln('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"')
    ln('target triple = "nvptx64-nvidia-cuda"')
    ln("")

    # ── Shared memory: double-buffered ──
    ln(f"@shmem_A = internal addrspace(3) global {ATY} zeroinitializer")
    ln(f"@shmem_B = internal addrspace(3) global {BTY} zeroinitializer")
    ln("")

    # ── NVVM intrinsics ──
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")
    ln("")

    # ── Kernel function ──
    ln(f"define void @{kernel_name}(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C, i32 %M, i32 %K, i32 %N) {{")
    ln("entry:")

    # Thread/block IDs
    ln("  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")
    ln("  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")
    ln("  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()")
    ln("")

    # Warp grid: warp_id = tid/32; warp_m = warp_id/WN; warp_n = warp_id%WN
    ln("  %warp_id = lshr i32 %tid, 5")
    ln(f"  %warp_m = udiv i32 %warp_id, {WN}")
    ln(f"  %warp_n = urem i32 %warp_id, {WN}")
    # Warp's base row/col within the block tile
    ln(f"  %warp_base_row = mul i32 %warp_m, {WTM * 16}")   # warp_m * 32
    ln(f"  %warp_base_col = mul i32 %warp_n, {WTN * 16}")   # warp_n * 64
    ln("")

    # Block base coordinates
    ln(f"  %bm = mul i32 %by, {BM}")
    ln(f"  %bn = mul i32 %bx, {BN}")
    ln("")

    # ── Prologue: stage tile 0 into buffer 0, then barrier ──
    emit_stage("p", "0", "0")
    ln("")
    ln('  call void asm sideeffect "bar.sync 0;", ""()')
    ln("")

    ln("  br label %tile_loop")
    ln("")

    # ── Tile loop ──
    ln("tile_loop:")
    ln("  %t = phi i32 [ 0, %entry ], [ %t_next, %after_prefetch ]")

    # Accumulator phis: WTM x WTN sub-tiles, each _ACC_REGS fp32 regs
    for tm in range(WTM):
        for tn in range(WTN):
            for r in range(_ACC_REGS):
                ln(f"  %acc_m{tm}_n{tn}_r{r} = phi float [ 0.0, %entry ], [ %acc_m{tm}_n{tn}_r{r}_new, %after_prefetch ]")
    ln("")

    # cur = t & 1, nxt = (t+1) & 1 (block-uniform)
    ln("  %cur = and i32 %t, 1")
    ln("  %tp1 = add i32 %t, 1")
    ln("  %nxt = and i32 %tp1, 1")
    ln(f"  %need_prefetch = icmp slt i32 %tp1, {num_tiles}")
    ln("  br i1 %need_prefetch, label %prefetch, label %after_prefetch")
    ln("")

    # ── Prefetch next K-tile into the alternate buffer (uniform branch) ──
    ln("prefetch:")
    ln(f"  %next_k_base = mul i32 %tp1, {BK}")
    emit_stage("l", "%nxt", "%next_k_base")
    ln("  br label %after_prefetch")
    ln("")

    # ── Compute on the current buffer ──
    ln("after_prefetch:")

    # Load A fragments (WTM), reused across all WTN col-tiles.
    for tm in range(WTM):
        row_off = tm * 16
        ln(f"  %a_row_m{tm} = add i32 %warp_base_row, {row_off}")
        ln(f"  %sa_ptr_m{tm} = getelementptr {ATY}, {ATY} addrspace(3)* @shmem_A, i32 0, i32 %cur, i32 %a_row_m{tm}, i32 0")
        ln(f"  %a_frag_m{tm} = call {RTY} asm sideeffect \"wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {{$0,$1,$2,$3,$4,$5,$6,$7}}, [$8], $9;\", \"=r,=r,=r,=r,=r,=r,=r,=r,l,r\"(half addrspace(3)* %sa_ptr_m{tm}, i32 {BK})")
        for r in range(_FRAG_REGS):
            ln(f"  %a_m{tm}_r{r} = extractvalue {RTY} %a_frag_m{tm}, {r}")
        ln("")

    # Load B fragments (WTN), reused across all WTM row-tiles.
    for tn in range(WTN):
        col_off = tn * 16
        ln(f"  %b_col_n{tn} = add i32 %warp_base_col, {col_off}")
        ln(f"  %sb_ptr_n{tn} = getelementptr {BTY}, {BTY} addrspace(3)* @shmem_B, i32 0, i32 %cur, i32 0, i32 %b_col_n{tn}")
        ln(f"  %b_frag_n{tn} = call {RTY} asm sideeffect \"wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 {{$0,$1,$2,$3,$4,$5,$6,$7}}, [$8], $9;\", \"=r,=r,=r,=r,=r,=r,=r,=r,l,r\"(half addrspace(3)* %sb_ptr_n{tn}, i32 {BN})")
        for r in range(_FRAG_REGS):
            ln(f"  %b_n{tn}_r{r} = extractvalue {RTY} %b_frag_n{tn}, {r}")
        ln("")

    # Compute WTM x WTN mma ops.
    for tm in range(WTM):
        for tn in range(WTN):
            a_args = ", ".join([f"i32 %a_m{tm}_r{r}" for r in range(_FRAG_REGS)])
            b_args = ", ".join([f"i32 %b_n{tn}_r{r}" for r in range(_FRAG_REGS)])
            c_args = ", ".join([f"float %acc_m{tm}_n{tn}_r{r}" for r in range(_ACC_REGS)])
            all_args = f"{a_args}, {b_args}, {c_args}"
            ln(f"  %mma_m{tm}_n{tn} = call {FTY} asm sideeffect \"wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {{$0,$1,$2,$3,$4,$5,$6,$7}}, {{$8,$9,$10,$11,$12,$13,$14,$15}}, {{$16,$17,$18,$19,$20,$21,$22,$23}}, {{$24,$25,$26,$27,$28,$29,$30,$31}};\", \"=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f\"({all_args})")
            for r in range(_ACC_REGS):
                ln(f"  %acc_m{tm}_n{tn}_r{r}_new = extractvalue {FTY} %mma_m{tm}_n{tn}, {r}")
            ln("")

    ln('  call void asm sideeffect "bar.sync 0;", ""()')
    ln("")

    # Tile loop latch
    ln("  %t_next = add i32 %t, 1")
    ln(f"  %t_cond = icmp slt i32 %t_next, {num_tiles}")
    ln("  br i1 %t_cond, label %tile_loop, label %store_results")
    ln("")

    # ── Store results ──
    ln("store_results:")
    for tm in range(WTM):
        row_off = tm * 16
        ln(f"  %out_row_m{tm}_a = add i32 %bm, %warp_base_row")
        ln(f"  %out_row_m{tm} = add i32 %out_row_m{tm}_a, {row_off}")
        for tn in range(WTN):
            col_off = tn * 16
            ln(f"  %out_col_m{tm}_n{tn}_a = add i32 %bn, %warp_base_col")
            ln(f"  %out_col_m{tm}_n{tn} = add i32 %out_col_m{tm}_n{tn}_a, {col_off}")
            ln(f"  %out_off_m{tm}_n{tn} = mul i32 %out_row_m{tm}, %N")
            ln(f"  %out_idx_m{tm}_n{tn} = add i32 %out_off_m{tm}_n{tn}, %out_col_m{tm}_n{tn}")
            ln(f"  %out_ptr_m{tm}_n{tn} = getelementptr float, float addrspace(1)* %C, i32 %out_idx_m{tm}_n{tn}")
            d_args = ", ".join([f"float %acc_m{tm}_n{tn}_r{r}_new" for r in range(_ACC_REGS)])
            ln(f"  call void asm sideeffect \"wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [$0], {{$1,$2,$3,$4,$5,$6,$7,$8}}, $9;\", \"l,f,f,f,f,f,f,f,f,r\"(float addrspace(1)* %out_ptr_m{tm}_n{tn}, {d_args}, i32 %N)")
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

    llc = "/home/blueyi/opt/llvm20-src/usr/lib/llvm-20/bin/llc"
    r = subprocess.run(
        [llc, "-march=nvptx64", "-mcpu=sm_86", "-O2", ll_path, "-o", ptx_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"llc FAILED:\n{r.stderr}")
        sys.exit(1)
    print("llc: OK")

    ptxas = "/usr/local/cuda-13.2/bin/ptxas"
    r = subprocess.run(
        [ptxas, "--gpu-name", "sm_86", "-O2", "-v", ptx_path, "-o", cubin_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"ptxas FAILED:\n{r.stderr}")
        sys.exit(1)
    print("ptxas: OK")
    print(r.stderr)

    import os
    print(f"cubin size: {os.path.getsize(cubin_path)} bytes")
    print("SUCCESS: LLVM IR -> PTX -> cubin (wmma TC matmul, double-buffered)")
