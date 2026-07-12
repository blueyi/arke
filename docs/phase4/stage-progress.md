# Phase 4: Arke → CUDA C Backend — Stage Progress

**Status:** 🚧 In Progress (P4-S1 matmul ✅)  
**Started:** 2026-07-12  
**Backend:** Arke IR → CUDA C source → nvcc → cubin → CUDA driver API  
**Hardware:** RTX 3060 Laptop 6GB (SM 8.6), CUDA 13.2, nvcc V13.2.51

---

## Goal

Generate vendor-supplied C-like kernel languages directly from Arke IR. Phase 4 fills the gap between MLIR (Phase 3) and bare LLVM IR (Phase 5): CUDA C is the vendor-stable kernel surface with the largest body of hand-tuned reference kernels.

---

## Stage Structure

| Stage | Milestone | Exit Criteria | Status |
|---|---|---|---|
| **P4-S1** | CUDA-C matmul E2E | SemanticIR → CUDA C → nvcc → correct (vs torch) | ✅ (commit 6bd6eb6) |
| **P4-S2** | Cat A+B+C via CUDA-C | 30 ops correct + geomean ≥ Phase 3 MLIR | ✅ 30 ops correct, **OVERALL 1.05× cuBLAS** (999b3b5) |
| **P4-S3** | CCE-C / Bang-C cross-vendor | Non-NVIDIA C-like backends | ⬜ (may audit/defer) |
| **P4-S4** | Performance ≥ MLIR | CUDA-C geomean ≥ Phase 3 | ⬜ |
| **P4-S_FINAL** | Multi-vendor C-like + H5 | Vendor-DSL portability via Arke IR | ⬜ |

---

## P4-S1: Complete (2026-07-12)

### Implementation

- **File:** `arke/backend/cuda_c_backend.py`
- **Backend name:** `cuda-c` (registered in protocol.py roadmap)
- **Protocol:** implements `ArkeBackend` (lower/compile/run/supports_op)

### Pipeline

```
IRGraph → emit_cuda_c_matmul() → CUDA C source
  → nvcc --cubin -arch=sm_86 → .cubin file (SHA256-cached)
  → cuda.bindings.driver: cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel
  → cuMemcpyDtoH → numpy result
```

### Matmul Implementation

- Tiled shared-memory kernel (TILE_SIZE=16)
- Handles non-power-of-2 shapes via boundary checks
- Kernel arguments: `(const float* A, const float* B, float* C, int M, int N, int K)`

### Verification

- **16 tests** in `tests/backend/test_cuda_c_backend.py`: all PASS
- Shapes tested: 16² to 1024², non-power-of-2 (33×47×61), degenerate (1×64×1)
- max_err vs numpy: ≤2.3e-4 (FP32 accumulation-order dependent)
- max_err vs torch CUDA: **0.0** (exact match)

### Key Design Decisions

1. **cubin, not .so** — nvcc `--cubin` produces device code loadable via `cuModuleLoadData` (same pattern as Phase 3 MLIR GPU). No host-side wrapper needed.
2. **SHA256 caching** — recompilation only if source changes.
3. **cuda.bindings.driver** — same CUDA driver API as Phase 3, no torch dependency in the critical path (torch used only for input conversion convenience).
4. **Tile=16 baseline** — correct first, fast later. Phase 4-S4 will add larger tiles + tensor-core paths.

---

## Next Steps (P4-S2)

- Add elementwise ops (relu, gelu, silu, exp, add, mul — straightforward 1D grid)
- Add reduction ops (softmax, layernorm, rmsnorm — block-reduce pattern)
- Register `CudaCBackend` in `BackendRegistry` with targets `["cuda-c", "cuda_c"]`
- Expand toward 30-op coverage for P4-S2 gate

---

*Last updated: 2026-07-12*
