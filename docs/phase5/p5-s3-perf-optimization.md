# P5-S3: Performance Optimization — Design & Results

## Overview

P5-S3 optimizes the LLVM IR backend kernels to approach or exceed PyTorch/cuBLAS
performance on an RTX 3060 Laptop GPU (SM 8.6, 6GB VRAM, CUDA 12.4).

## Architecture

### Execution Model — CachedModule API

```
                    ┌─── prepare(kernel) ──────────────────────┐
                    │  cuModuleLoadData(cubin) → CUmodule      │
                    │  cuMemAlloc × N → GPU buffers            │
                    │  Build kernel arg array (fixed pointers) │
                    └──────────────── returns _CachedModule ───┘
                                      │
            ┌─── run_fast(cached, inputs) ─┐    ┌─── run_fast_no_copy(cached) ─┐
            │  cuMemcpyHtoD (input buffers) │    │  (data already on GPU)        │
            │  cuLaunchKernel + sync        │    │  cuLaunchKernel + sync        │
            │  cuMemcpyDtoH (output)        │    │  (no copy)                    │
            └───────────────────────────────┘    └──────────────────────────────┘
```

**Key insight:** Module load + alloc/free was 40-57× overhead. CachedModule eliminates it.

### Elementwise Vectorization (OT0)

**Pattern:** float4 vectorized load/store + grid-stride loop + rcp.approx.ftz.f

```
Grid = (total/4/BLOCK_SIZE, 1, 1), Block = (256, 1, 1)

vec_loop:
  %xv = load <4 x float>, <4 x float> addrspace(1)* %X_v, align 16
  ; extract → compute on 4 elements → insert
  store <4 x float> %rv, <4 x float> addrspace(1)* %Out_v, align 16
  %vid_next = add i32 %vid, %grid_stride
  br i1 %more, label %vec_loop, label %tail

tail: ; handles remaining 0-3 elements scalar
```

**PTX output:** `ld.global.v4.f32` / `st.global.v4.f32` → 128-bit coalesced transactions.

### Warp-Parallel Reduction (OT1: softmax, layernorm, rmsnorm)

**Pattern:** Grid=(M,1,1), Block=(32,1,1) — 1 warp per row

```
; Each thread processes elements at stride 32
loop:
  %j = phi i32 [%tid, %entry], [%j_next, %body]
  %acc = phi float [INIT, %entry], [%new_acc, %body]
  ; load + accumulate
  %j_next = add i32 %j, 32

; 5-step warp reduction via inline PTX
  %s1 = call float asm "shfl.sync.down.b32 $0,$1,$2,$3,$4;", "=f,f,r,r,r"(...)
  ; ... 16, 8, 4, 2, 1

; Broadcast from lane 0
  %bcast = call float asm "shfl.sync.idx.b32 $0,$1,0,31,$2;", "=f,f,r"(...)
```

**Online softmax (Milakov & Gimelshein):** 2-pass instead of 3-pass.
- Pass 1: Simultaneous running_max + running_sum with online correction
- Pass 2: exp(x - max) * inv_sum → fused normalize

### Matmul Tiling & Register Blocking (OT2)

**Pattern:** Shared memory tiling + register blocking

```
BM=32, BN=32, BK=16, TM=TN=2
Block = (16, 16), Grid = (ceil(N/BN), ceil(M/BM), 1)
Each thread: 4 accumulators (2×2 output submatrix)

@shmem_A = internal addrspace(3) global [BM x [BK x float]] undef
@shmem_B = internal addrspace(3) global [BK x [BN x float]] undef

tile_loop:
  cooperative_load(A_tile → shmem_A)
  cooperative_load(B_tile → shmem_B)
  barrier0
  k_loop (0..BK-1):
    load 2 A regs from shmem_A[ty*2+{0,1}][ki]
    load 2 B regs from shmem_B[ki][tx*2+{0,1}]
    4 FMAs (outer product)
  barrier0
```

### Fast Math Intrinsics

| Operation | Old | New | Speedup |
|---|---|---|---|
| 1/x | `fdiv float 1.0, %x` | `@llvm.nvvm.rcp.approx.ftz.f` | ~10× |
| exp(x) | libdevice call | `@llvm.nvvm.ex2.approx.f(x * log2e)` | inline |
| 1/√x | sqrt + fdiv | `@llvm.nvvm.rsqrt.approx.f` | ~5× |

## Performance Results (kernel-only, RTX 3060 Laptop)

### Ops that beat PyTorch
| Op | Shape | LLVM | PT | Ratio |
|---|---|---|---|---|
| silu | 128×4096 | 39µs | 72µs | **0.55×** |
| rmsnorm | 32×4096 | 54µs | 89µs | **0.60×** |
| rmsnorm | 1024×4096 | 257µs | 394µs | **0.65×** |
| silu_and_mul | 2048×4096 | 384µs | 539µs | **0.71×** |
| gelu_and_mul | 2048×4096 | 416µs | 537µs | **0.78×** |

### Overall progression (P5-S2 → P5-S3)
| Category | Before | After | Speedup |
|---|---|---|---|
| Elementwise (geomean) | 6.28× | 1.08× | 5.8× |
| Softmax | 23.4× | 1.99× | 11.8× |
| Layernorm | 28.9× | 1.66× | 17.4× |
| RMSNorm | 9.7× | 0.65× | 14.9× + beat PT |
| Matmul | 7.1× | 3.22× | 2.2× |

## Why some ops are still slower

- **Matmul 3.22×:** No Tensor Core usage. cuBLAS uses HMMA/IMMA instructions on SM 8.6.
  Pure FP32 scalar FMA can never match TC throughput (theoretical 4× gap).
- **Softmax 1.99×:** Single warp per row limits parallelism. PyTorch uses multi-warp blocks.
- **Elementwise 1.18×:** At memory-bandwidth limit. Gap is kernel launch overhead (~5µs fixed).

## Files Modified
- `arke/backend/llvm_backend.py` — _CachedModule, prepare/run_fast/run_fast_no_copy
- `arke/backend/llvm_elementwise.py` — float4 vectorized template
- `arke/backend/llvm_rowwise.py` — warp-parallel + online softmax
- `arke/backend/llvm_emitter.py` — matmul register blocking
- `arke/backend/llvm_fused.py` — rcp.approx.ftz.f optimization
- `benchmarks/llvm_vs_pytorch.py` — updated with cached timing
- `benchmarks/llvm_quick_bench.py` — multi-shape kernel-only benchmark
