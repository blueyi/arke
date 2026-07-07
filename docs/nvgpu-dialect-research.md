# MLIR 20 `nvgpu` Dialect Research: Tensor Core Matmul

## 1. Complete Op Inventory

The `nvgpu` dialect provides a bridge between higher-level target-agnostic dialects (GPU and Vector) and the lower-level NVVM dialect. All ops and their purposes:

### 1.1 Tensor Core MMA Operations

#### `nvgpu.mma.sync` (MmaSyncOp) — **PRIMARY for Ampere sm_86**
Warp-level matrix-multiply-and-accumulate. Intermediate between `vector.contract` and `nvvm.mma.sync`.

```mlir
// f16 inputs, f16 accumulator — m16n8k16
%d = nvgpu.mma.sync (%matA, %matB, %matC) {mmaShape = [16, 8, 16]}
    : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>

// f16 inputs, f32 accumulator — m16n8k16
%d = nvgpu.mma.sync (%matA, %matB, %matC) {mmaShape = [16, 8, 16]}
    : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>

// f32 inputs via TF32 — m16n8k4
%d = nvgpu.mma.sync (%matA, %matB, %matC) {mmaShape = [16, 8, 4], tf32Enabled}
    : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>

// f32 inputs via TF32 — m16n8k8 (two m16n8k4 ops fused)
%d = nvgpu.mma.sync (%matA, %matB, %matC) {mmaShape = [16, 8, 8], tf32Enabled}
    : (vector<4x1xf32>, vector<2x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
```

Key attributes:
- `mmaShape`: `[M, N, K]` array — the warp-level MMA shape
- `tf32Enabled`: UnitAttr — when present, f32 inputs use TF32 tensor cores (reduced precision, 10-bit mantissa)
- Operand vectors are **thread-level ownership** (each thread holds a fragment of the warp-level tile)

#### `nvgpu.mma.sp.sync` (MmaSparseSyncOp)
Structured-sparse MMA (2:4 sparsity). Operand A is compressed + metadata.

```mlir
nvgpu.mma.sp.sync (%a, %b, %c) metadata (%meta) {mmaShape = [16, 8, 32]}
    : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
```

#### `nvgpu.warpgroup.mma` (WarpgroupMmaOp) — **Hopper sm_90 only**
Uses `wgmma` instructions (warpgroup = 4 warps). NOT available on Ampere.

### 1.2 Async Copy Operations (cp.async)

#### `nvgpu.device_async_copy` (DeviceAsyncCopyOp)
Asynchronous copy from global memory to shared memory without blocking threads.

```mlir
%cp1 = nvgpu.device_async_copy %src[%i, %j], %dst[%i, %j], 4
    : memref<16xf32> to memref<16xf32, 3>
```

- `bypassL1`: hint to bypass L1 cache during copy
- `dstElements`: total elements written to shared memory
- `srcElements`: optional, enables zero-fill for partial copies (predicated loads)

#### `nvgpu.device_async_create_group` (DeviceAsyncCreateGroupOp)
Groups pending async copies into a completion group.

```mlir
%token = nvgpu.device_async_create_group %cp1, %cp2
```

#### `nvgpu.device_async_wait` (DeviceAsyncWaitOp)
Blocks until a copy group completes.

```mlir
nvgpu.device_async_wait %token           // wait for all
nvgpu.device_async_wait %token {numGroups = 1 : i32}  // pipeline: wait until ≤1 groups in flight
```

The `numGroups` attribute is **critical for software pipelining** — it allows overlapping computation with the next tile's data transfer.

### 1.3 Matrix Load Operations

#### `nvgpu.ldmatrix` (LdMatrixOp)
Warp-cooperative matrix load from shared memory to registers (maps to PTX `ldmatrix`).

```mlir
%a = nvgpu.ldmatrix %smem[%r, %c] {numTiles = 4 : i32, transpose = false}
    : memref<128x128xf16, 3> -> vector<4x2xf16>
```

- `numTiles`: number of 8x8 matrix tiles to load (1, 2, or 4)
- `transpose`: load with transpose (only at 16-bit granularity)

### 1.4 Shared Memory Optimization

#### `-nvgpu-optimize-shared-memory` pass
XOR-based shared memory index permutation to eliminate bank conflicts.

```
// Pass available in our mlir-opt:
--nvgpu-optimize-shared-memory  -  Optimizes accesses to shared memory memrefs
                                   in order to reduce bank conflicts.
```

Algorithm: For a `memref<?x?xDT, 3>`, applies `tgt_idx = xor(src_idx >> N, tgt_idx >> V) << V + tgt_idx % V`
where N = log2(128/elementSizeBits), V = vectorSize. This is the standard swizzle pattern.

Constants: `kSharedMemoryLineSizeBytes = 128`, `kDefaultVectorSizeBits = 128`

### 1.5 TensorMap / TMA Operations (Hopper sm_90 only)
- `nvgpu.tma.create.descriptor` — create TMA descriptor on host
- `nvgpu.tma.async.load` / `nvgpu.tma.async.store` — TMA transfers
- `nvgpu.tma.fence.descriptor` / `nvgpu.tma.prefetch.descriptor`
- `nvgpu.warpgroup.generate.descriptor`

### 1.6 MBarrier Operations (Hopper sm_90 only, partially on Ampere)
- `nvgpu.mbarrier.create` / `nvgpu.mbarrier.init`
- `nvgpu.mbarrier.arrive` / `nvgpu.mbarrier.arrive.nocomplete` / `nvgpu.mbarrier.arrive.expect_tx`
- `nvgpu.mbarrier.test.wait` / `nvgpu.mbarrier.try_wait.parity`

### 1.7 Other
- `nvgpu.rcp` — fast reciprocal with rounding mode control

---

## 2. Lowering Pipeline: nvgpu → nvvm → PTX

### 2.1 Available Passes (confirmed in our mlir-opt build)

```
--convert-vector-to-gpu          # vector.contract → nvgpu.mma.sync (with --use-nvgpu flag)
                                 # Also: vector.transfer_read → nvgpu.ldmatrix
--convert-nvgpu-to-nvvm          # nvgpu.mma.sync → nvvm.mma.sync
                                 # nvgpu.device_async_copy → nvvm.cp.async
                                 # nvgpu.ldmatrix → nvvm.ldmatrix
--convert-nvvm-to-llvm           # nvvm.mma.sync → PTX inline assembly in LLVM dialect
--nvgpu-optimize-shared-memory   # XOR-based bank conflict avoidance
--test-nvgpu-mmasync-f32-to-tf32-patterns  # Convert f32 mma.sync to use TF32 tensor cores
--gpu-lower-to-nvvm-pipeline     # One-shot pipeline: all dialects → NVVM (configurable)
--llvm-optimize-for-nvvm-target  # LLVM-level optimizations for NVVM
--nvvm-attach-target             # Attach NVVM target to GPU module
```

### 2.2 Full Lowering Chain

```
linalg.matmul
  ↓ --convert-linalg-to-loops / --convert-linalg-to-affine-loops
  ↓ (or: --linalg-to-vector via tiling + vectorization)
vector.contract  (16x16xf16) × (8x16xf16) → (16x8xf16)
  ↓ --convert-vector-to-gpu --use-nvgpu
  ↓ Rewrites: vector.transfer_read → nvgpu.ldmatrix
  ↓           vector.contract     → nvgpu.mma.sync
  ↓           vector.transfer_write → vector.store (distributed)
nvgpu.mma.sync + nvgpu.ldmatrix + nvgpu.device_async_copy
  ↓ --convert-nvgpu-to-nvvm
  ↓ Rewrites: nvgpu.mma.sync → nvvm.mma.sync
  ↓           nvgpu.ldmatrix → nvvm.ldmatrix
  ↓           nvgpu.device_async_copy → nvvm.cp.async
nvvm.mma.sync + nvvm.ldmatrix + nvvm.cp.async
  ↓ --convert-nvvm-to-llvm
  ↓ Rewrites: NVVM ops → LLVM inline assembly with PTX instructions
llvm.inline_asm "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ..."
  ↓ LLVM NVPTX backend
PTX assembly → CUBIN (via ptxas)
```

### 2.3 One-Shot Pipeline

```bash
mlir-opt example.mlir \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_86 cubin-features=+ptx72 opt-level=3"
```

This handles: arith, memref, scf, vector, gpu, and nvgpu → NVVM → fatbin/cubin.

---

## 3. Supported MMA Shapes for sm_86 (Ampere)

### 3.1 Complete Shape Table

| Shape (M×N×K) | Input Type | Accum Type | A shape (per-thread) | B shape (per-thread) | C/D shape (per-thread) |
|---------------|-----------|------------|---------------------|---------------------|----------------------|
| **m16n8k16**  | **f16**   | **f16**    | vector<4x2xf16>     | vector<2x2xf16>     | vector<2x2xf16>      |
| **m16n8k16**  | **f16**   | **f32**    | vector<4x2xf16>     | vector<2x2xf16>     | vector<2x2xf32>      |
| **m16n8k16**  | **bf16**  | **f32**    | vector<4x2xbf16>    | vector<2x2xbf16>    | vector<2x2xf32>      |
| **m16n8k8**   | **f16**   | **f16**    | vector<2x2xf16>     | vector<1x2xf16>     | vector<2x2xf16>      |
| m16n8k4       | f32(tf32) | f32        | vector<2x1xf32>     | vector<1x1xf32>     | vector<2x2xf32>      |
| **m16n8k8**   | **f32(tf32)** | **f32** | vector<4x1xf32>     | vector<2x1xf32>     | vector<2x2xf32>      |
| m16n8k32      | i8        | i32        | vector<4x4xi8>      | vector<2x4xi8>      | vector<2x2xi32>      |
| m16n8k32      | i4        | i32        | vector<2x8xi4>      | vector<1x8xi4>      | vector<2x2xi32>      |
| m16n8k64      | i4        | i32        | vector<4x8xi4>      | vector<2x8xi4>      | vector<2x2xi32>      |
| m8n8k4        | f64       | f64        | vector<1x1xf64>     | vector<1x1xf64>     | vector<1x2xf64>      |

### 3.2 Recommended Shapes for Our Use Cases (sm_86 RTX 3060 Laptop)

**For f16 matmul (highest throughput):**
- **m16n8k16** — the primary shape. Each warp computes a 16×8 output tile per instruction.
- To fill a 16×16 output tile, issue two m16n8k16 instructions with different B fragments.
- Peak: 256 TFLOPS on sm_86.

**For f32 matmul via TF32 tensor cores:**
- **m16n8k8** with `tf32Enabled` — fastest for f32, but reduced precision (10-bit mantissa → ~3 decimal digits).
- Each m16n8k8 is internally two m16n8k4 ops.
- The `--test-nvgpu-mmasync-f32-to-tf32-patterns --precision=tf32` pass auto-adds `tf32Enabled` to f32 mma.sync ops.

**For f32 matmul with full precision:**
- No direct f32 tensor core op with full precision on Ampere.
- Options: (a) Use f16 with mixed precision, (b) Use TF32 if precision is acceptable, (c) Use scalar FMA (no tensor cores).

### 3.3 Thread-Level Fragment Sizes Explained

For **m16n8k16 f16→f32**:
- Matrix A is 16×16 (distributed across 32 threads in a warp)
  - Each thread holds `vector<4x2xf16>` = 4 pairs of f16 = 8 elements
  - 32 threads × 8 = 256 elements = 16×16 ✓
- Matrix B is 8×16 (note: k=16, n=8; B is k×n)
  - Each thread holds `vector<2x2xf16>` = 4 elements
  - 32 threads × 4 = 128 elements = 8×16 ✓
- Matrix C/D is 16×8
  - Each thread holds `vector<2x2xf32>` = 4 elements
  - 32 threads × 4 = 128 elements = 16×8 ✓

---

## 4. Key Test Files and Examples in LLVM/MLIR Repository

### 4.1 NVGPUToNVVM Conversion Tests
**File:** `mlir/test/Conversion/NVGPUToNVVM/nvgpu-to-nvvm.mlir`

Contains 53+ test cases showing nvgpu → nvvm lowering for every MMA shape:
- `@m16n8k16_fp16` — f16 → f16 mma.sync
- `@m16n8k16_fp16_fp32` — f16 → f32 mma.sync
- `@m16n8k16_bf16_fp32` — bf16 → f32 mma.sync
- `@m16n8k8_fp16` — smaller k dimension
- `@m16n8k4_tf32` — TF32 tensor cores for f32
- `@m16n8k32_int8` — int8 matmul
- `@m8n8k4_f64` — f64 tensor cores
- `@ldmatrix_x4` — ldmatrix lowering
- Async copy / wait / group tests

### 4.2 VectorToGPU Conversion Tests (vector.contract → nvgpu.mma.sync)
**File:** `mlir/test/Conversion/VectorToGPU/vector-to-mma-ops-mma-sync.mlir`

32 test functions showing how `vector.contract` is decomposed into `nvgpu.ldmatrix` + `nvgpu.mma.sync`:

Key example — **f16 m16n8k16 complete matmul lowering**:
```mlir
// Input: warp-level vector.contract
%A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
    : memref<20x20xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
%B = vector.transfer_read %arg1[%c0, %c0], %cst {permutation_map = #transpose}
    : memref<20x20xf16, #gpu.address_space<workgroup>>, vector<8x16xf16>
%C = vector.transfer_read %arg2[%c0, %c0], %cst
    : memref<20x20xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
%D = vector.contract ... %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
vector.transfer_write %D, %arg2[%c0, %c0]

// Output: distributed nvgpu ops
// A loaded via: nvgpu.ldmatrix {numTiles = 4, transpose = false} → vector<4x2xf16>
// B loaded via: nvgpu.ldmatrix {numTiles = 2, transpose = true}  → vector<2x2xf16>
// C loaded via: distributed vector.load → vector<2x2xf16>
// Compute:      nvgpu.mma.sync {mmaShape = [16, 8, 16]} → vector<2x2xf16>
// D stored via: distributed vector.store
```

Key example — **f32 via TF32 m16n8k8**:
```mlir
// Input: warp-level f32 vector.contract
%A = vector.transfer_read %arg0[...] : memref<20x20xf32, #gpu.address_space<workgroup>>, vector<16x8xf32>
%B = vector.transfer_read %arg1[...] {permutation_map = #transpose} : ..., vector<8x8xf32>
%D = vector.contract ... %A, %B, %cst_0 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>

// Output: nvgpu ops
// A loaded via: nvgpu.ldmatrix {numTiles = 4} → vector<4x1xf32>  (reinterpreted as f32)
// B loaded via: scalar memref.load → vector<2x1xf32>
// Compute:      nvgpu.mma.sync {mmaShape = [16, 8, 8]} → vector<2x2xf32>
```

### 4.3 Complete GEMM Examples (Python-driven, Hopper)
**Files:** `mlir/test/Examples/NVGPU/Ch1.py` through `Ch5.py`

These are Hopper (sm_90) examples using `nvgpu.warpgroup.mma` + TMA. While not directly usable on Ampere, they show the overall pattern:
- Ch1: basic TMA load
- Ch2: basic warpgroup MMA
- Ch3: single-stage GEMM
- Ch4: multi-stage pipelined GEMM
- Ch5: multi-stage + epilogue

### 4.4 Shared Memory Optimization Test
**File:** `mlir/test/Dialect/NVGPU/optimize-shared-memory.mlir`

Shows the XOR-based swizzle transformation for bank conflict avoidance.

---

## 5. Architecture Decision: Best Approach for Arke Tensor Core Matmul

### 5.1 Recommended Strategy for sm_86

For our RTX 3060 Laptop (sm_86, Ampere), the available tensor core path is:

1. **Use `nvgpu.mma.sync`** (not `warpgroup.mma` which is Hopper-only)
2. **f16 path (m16n8k16)** for maximum throughput
3. **f32 TF32 path (m16n8k8)** if f32 precision is needed and TF32 precision (~3 decimal digits) is acceptable
4. Use `nvgpu.device_async_copy` for cp.async pipelining
5. Use `nvgpu.ldmatrix` for efficient shared → register loads
6. Apply `--nvgpu-optimize-shared-memory` for bank conflict avoidance

### 5.2 Key Lowering Pipeline for Our Kernel

```bash
mlir-opt kernel.mlir \
  --convert-vector-to-gpu="use-nvgpu=true" \
  --nvgpu-optimize-shared-memory \
  --convert-nvgpu-to-nvvm \
  --convert-gpu-to-nvvm \
  --convert-nvvm-to-llvm \
  --llvm-optimize-for-nvvm-target
```

Or the one-shot pipeline:
```bash
mlir-opt kernel.mlir \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_86 cubin-features=+ptx72 opt-level=3"
```

### 5.3 Performance-Critical Pattern

A high-performance matmul kernel needs:

```
Outer loop (k-tiles, software-pipelined):
  Stage 0: nvgpu.device_async_copy (global → shared, next tile)
  Stage 1: nvgpu.device_async_wait (previous tile ready)
           nvgpu.ldmatrix (shared → registers)
           nvgpu.mma.sync (compute on current tile)

# Double-buffer shared memory for overlap
```

The `numGroups` attribute on `device_async_wait` controls pipeline depth:
- `numGroups = 0`: wait for ALL groups (no pipelining)
- `numGroups = N-1`: allow N-1 groups in flight (N-stage pipeline)

---

## 6. Implementation Status (2026-07-07, commit `0ccbc1b`)

The research above is now **shipped as a production emitter**. Result:
**tensor-core matmul geomean 0.91× cuBLAS**, beating cuBLAS at inference sizes
(1024³ 1.16×, 2048³ 1.41×) vs the scalar-f32 regblock's 0.52×.

### 6.1 What was built

- `arke/backend/mlir_emitter.py::emit_gpu_matmul_mma` — warp-register-blocked
  GEMM. f32 inputs → `arith.truncf` to f16 into shared, warp-level
  `vector.contract`, f32 accumulation. Block `BM×BN`, `WM×WN` warps, each warp
  owns a `WTM×WTN` grid of 16×16 output sub-tiles held as `WTM*WTN*2` f32
  register accumulators. Each A-fragment is reused `WTN×`, each B-fragment `WTM×`
  (the CUTLASS-style arithmetic-intensity multiplier). Defaults `WM=WN=2`,
  `WTM=2 WTN=4` → `BM=64 BN=128 BK=16`, 128 threads/block.
- `arke/backend/mlir_gpu.py::mlir_nvgpu_to_cubin` — the verified two-stage
  lowering (see §6.2).
- `MLIRGPUBackend(use_tensor_core=True)` — **opt-in** flag. Default backend keeps
  the bit-accurate scalar-f32 matmul (see §6.4).
- Tests: `tests/backend/test_mlir_gpu_matmul_mma_p3s2.py` (11).

### 6.2 The lowering path that actually works (two-stage, NOT single-pass)

The manual single-pass list (`-convert-nvgpu-to-nvvm -convert-vector-to-llvm
-convert-arith-to-llvm -convert-scf-to-cf -convert-gpu-to-nvvm
-reconcile-unrealized-casts -gpu-module-to-binary`) **FAILS** on these kernels:
when `gpu.lane_id` + workgroup-address-space memrefs are present (introduced by
`--convert-vector-to-gpu=use-nvgpu`), the memref-space→integer conversion errors.

Verified working path:
```
stage 1: --convert-vector-to-gpu=use-nvgpu
         (distributes warp vector.contract → per-thread nvgpu.ldmatrix + mma.sync)
stage 2: -convert-nvgpu-to-nvvm -gpu-lower-to-nvvm-pipeline=cubin-chip=sm_86
         (one-shot pipeline handles the workgroup memref-space conversion)
```
The cubin is extracted from the emitted `#gpu.object<#nvvm.target<...>, "BLOB">`
attribute (NOT the `bin = "..."` of `-gpu-module-to-binary=format=bin`). Cubin
magic `50ed55ba` (NVIDIA fatbin).

**Two hard guards** (both cost debug time):
- After stage 1, assert `"nvgpu.mma.sync" in output`. If absent, the
  `vector.transfer_read`s aren't sourcing **workgroup (shared) memory** (reading
  from a global memref silently fails to distribute), or the contract shape isn't
  a valid MMA shape.
- `vector.contract` MxNxK must be a hardware MMA shape. **m16n8k16 works;
  m16n16k16 does NOT** ("unimplemented variant"). To compute a 16×16 output tile,
  emit TWO m16n8k16 contracts (each 16×8), never one 16×16. B is read with a
  transpose `permutation_map` as `vector<8x16xf16>` ([n,k]); contract indexing
  maps are A=(m,k), B=(n,k), C=(m,n).

### 6.3 Precision (honest finding)

Reduced-precision throughput path. Output is bit-close vs an **fp16-input**
reference (~1e-6 rel) but ~3e-4 rel vs strict-f32 cuBLAS. This is the inherent
tensor-core tradeoff: cuBLAS itself uses **tf32** tensor cores by default for f32
inputs on Ampere (a comparable reduced-precision class). Both fp16 and tf32
tensor cores differ from a strict-f32 result on ~0.35% of (near-zero) elements
(max abs err ~0.08 on values up to ~100). Tests validate against the fp16
reference, which is the correct precision class for a tensor-core kernel.

### 6.4 Why it is opt-in (not the default matmul path)

Making tensor-core the DEFAULT matmul lowering changes the **precision class of
all matmul gate results** (bit-accurate f32 → tf32/fp16-grade). That is a
benchmark-semantics decision — a hard stop requiring project-lead sign-off — so
the emitter ships behind `use_tensor_core=True`. `lower()` tags
`metadata["is_mma"]`; `compile()` routes tagged artifacts through the nvgpu
pipeline; the emitter raises `NotImplementedError` on shapes that don't MMA-tile,
falling through to the scalar-f32 ladder (verified: 32³/64³ fall back and stay
bit-accurate).

### 6.5 Not yet done (future perf headroom)

- **cp.async double-buffering** (§5.3): the current kernel uses synchronous
  cooperative loads + two barriers per K-tile. Software-pipelined
  `nvgpu.device_async_copy` with `numGroups = N-1` would overlap the next tile's
  global→shared transfer with the current tile's mma.sync — the main remaining
  gap to cuBLAS at 512³ and the driver-overhead-bound small shapes.
- `--nvgpu-optimize-shared-memory` (XOR bank-conflict swizzle) not yet applied.
- tf32 mma path (`mmaShape=[16,8,8] tf32Enabled`) as an exact-cuBLAS-precision
  alternative to the f16 cast.
