# Phase 4 Performance Gap Analysis & Reverse Enhancement Plan

**Date:** 2026-07-12  
**Context:** P4-S2 correctness half complete (30 ops). Perf benchmark reveals matmul 0.23× cuBLAS as primary drag on OVERALL geomean.

---

## 1. Performance Data (kernel-only, CUDA events, vs cuBLAS/cuDNN)

| Category | CUDA-C geomean | Phase 3 MLIR geomean | Notes |
|---|---|---|---|
| **Elementwise** | **1.15-1.26×** | 0.95-1.14× | CUDA-C wins, especially small shapes |
| **Rowwise reductions** | **1.07-1.17×** | 0.88-1.20× | Competitive |
| **Matmul** | **0.23×** | 0.42× | Both far from cuBLAS TC |
| **OVERALL** | **0.91×** | 1.03× | Matmul drag dominates |

**Key insight:** For elementwise/rowwise, our naive CUDA-C kernels already beat cuBLAS at small shapes and match at large shapes. The problem is exclusively matmul — and it's not a CUDA-C codegen problem, it's a **strategy problem**.

---

## 2. Root Cause: Why matmul is 0.23× cuBLAS

Our current matmul is a textbook shared-memory tiled kernel (TILE=16):
```c
__shared__ float sA[16][16], sB[16][16];
// each thread: 1 output element, 16-step K-loop, 16x16 shared tiles
```

cuBLAS matmul at 512+ uses:
1. **Tensor cores** (wmma/mma instructions) — 16× throughput vs FP32 FMA
2. **Double-buffered shared memory** — overlaps compute with next tile load
3. **Software pipelining** — cp.async + multi-stage pipeline hides latency
4. **Cooperative warp scheduling** — 2+ warps cooperative on one output tile
5. **Register blocking** — each thread computes a 8×8 output tile, not 1×1

None of these are accessible from our current codegen because **the StrategyIR → CUDA-C lowering path doesn't exist**. The backend hardcodes a fixed kernel template regardless of any optimization decisions the Agent might make.

---

## 3. Reverse Enhancement: What Arke IR / Harness Needs

### 3.1 StrategyIR Consumption (MISSING — High Priority)

**Current state:** CudaCBackend.lower(graph) ignores StrategyIR entirely.  
**Needed:** `lower(graph, strategy=None)` → if strategy has decisions, apply them.

Required StrategyIR → CUDA-C mappings:

| StrategyIR Decision | CUDA-C Effect |
|---|---|
| `tile(loop="K", factors=[32])` | Change TILE_SIZE from 16 to 32 |
| `tile(loop="M", factors=[64])` + `tile(loop="N", factors=[128])` | Change output tile per block |
| `algorithm(name="tensor_core", params={precision: "tf32"})` | Emit wmma/mma instructions |
| `compute(warps=4, num_stages=3)` | Multi-warp + pipeline stages |
| `vectorize(loop="K_inner", width=4)` | Use float4 vectorized loads |
| `unroll(loop="K_tile", factor=4)` | #pragma unroll on K loop |
| `place(tensor="A_tile", memory="shared")` | Double-buffer shared memory |

### 3.2 Bounded Action Space for CUDA-C Matmul (MISSING)

The Harness's `list_legal_actions()` currently returns decisions for the Triton/MLIR path. For CUDA-C, we need a CUDA-C-specific action space:

```python
# Example legal actions for matmul on SM 8.6:
actions = [
    Decision(kind="tile", params={"loop": "MN", "BM": 64, "BN": 64}),
    Decision(kind="tile", params={"loop": "MN", "BM": 64, "BN": 128}),
    Decision(kind="tile", params={"loop": "MN", "BM": 128, "BN": 128}),
    Decision(kind="algorithm", params={"name": "tensor_core", "precision": "tf32"}),
    Decision(kind="algorithm", params={"name": "scalar_fma"}),
    Decision(kind="compute", params={"warps": 4, "num_stages": 2}),
    Decision(kind="compute", params={"warps": 4, "num_stages": 3}),
    Decision(kind="vectorize", params={"loop": "load", "width": 4}),
    Decision(kind="unroll", params={"loop": "K_tile", "factor": 4}),
]
```

### 3.3 Parameterized Kernel Templates (MISSING)

Instead of one hardcoded matmul kernel, we need parameterized templates:

```
Template: matmul_scalar_tiled
  Parameters: TILE_M, TILE_N, TILE_K, UNROLL_K
  Generates: shared-memory tiled FP32 kernel

Template: matmul_tensor_core  
  Parameters: BM, BN, BK, WARPS, STAGES, PRECISION
  Generates: wmma-based kernel with double-buffered shmem

Template: matmul_cutlass_style
  Parameters: BM, BN, BK, WM, WN, STAGES, SWIZZLE
  Generates: CUTLASS-inspired kernel with software pipelining
```

The Agent (via Harness) picks the template + parameters through StrategyIR decisions. The backend instantiates the template.

---

## 4. Implementation Plan (Reverse Enhancement)

### Phase A: StrategyIR-Aware lower() (THIS SESSION)

1. Extend `CudaCBackend.lower()` to accept optional `StrategyIR`
2. For matmul: read tile/unroll/algorithm decisions → select template + params
3. Implement `matmul_scalar_tiled` with configurable TILE_M/N/K + unroll
4. Verify: Agent Decision(tile MN=32) → different kernel → different perf

### Phase B: Tensor Core Template (NEXT)

1. Add `matmul_tensor_core` template using `__wmma_*` / `mma.sync` PTX
2. StrategyIR `algorithm(name="tensor_core")` → routes to this template
3. Target: 0.5-0.7× cuBLAS (realistic for a first TC implementation)

### Phase C: Harness Integration

1. Register CUDA-C legal actions in `arke/engine/` action space
2. Agent can `list_legal_actions()` → pick tile/algorithm → `apply_decision()`
3. compile_and_profile() uses `CudaCBackend.benchmark()` for perf feedback

---

## 5. Non-Matmul Ops: No Enhancement Needed

Elementwise and rowwise ops are **already competitive or winning** vs cuBLAS/cuDNN. These don't need StrategyIR enhancement — the fixed templates are sufficient because:
- Elementwise: bandwidth-bound, kernel body is trivial, no strategic decisions
- Rowwise: block-per-row pattern is near-optimal for reductions

The StrategyIR enhancement is specifically for **compute-bound ops** (matmul, attention) where algorithm choice and tiling strategy have 10-100× impact on performance.

---

*This analysis drives the "reverse enhancement" direction: backend perf gaps feed back into Arke IR / Harness design requirements.*
