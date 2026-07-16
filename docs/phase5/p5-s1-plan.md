# Phase 5 — Arke → LLVM IR (P5-S1 Plan)

> **Goal:** SemanticIR → LLVM IR → PTX → cubin → GPU execution. Matmul correct.
> **Exit criteria (P5-S1):** `arke run --backend llvm --kernel matmul --shape 128,128,128` → success, correctness verified vs torch.mm reference.

## Architecture

```
IRGraph ──► LLVMEmitter ──► .ll text ──► llc (nvptx64) ──► .ptx ──► ptxas ──► .cubin ──► CUDA driver API
             (arke/backend/       (subprocess)         (subprocess)        (reuse from
              llvm_emitter.py)                                              cuda_c_backend.py)
```

### Key files (new)
- `arke/backend/llvm_backend.py` — LLVMBackend (ArkeBackend protocol)
- `arke/backend/llvm_emitter.py` — IRGraph node → LLVM IR text
- `tests/backend/test_llvm_backend.py` — correctness tests

### Key files (reuse)
- `arke/backend/cuda_c_backend.py` — cubin loading + CUDA driver API execution infrastructure
- `arke/backend/protocol.py` — register LLVMBackend in default registry

## P5-S1 Scope (matmul only)

1. **Emit LLVM IR for matmul:** tiled register-blocked FP32 matmul kernel targeting nvptx64
   - Thread/block mapping: blockDim(BN/TN, BM/TM), gridDim(N/BN, M/BM)
   - Shared memory tiling (BM×BK + BK×BN per tile)
   - K-loop with syncthreads
2. **Compile chain:** `.ll` → `llc --march=nvptx64 --mcpu=sm_86` → `.ptx` → `ptxas --gpu-name sm_86` → `.cubin`
3. **Execute:** Load cubin via CUDA driver API, launch kernel, read back results
4. **Verify:** Output matches torch.mm within fp32 tolerance (atol=1e-3, rtol=1e-3)

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| IR generation | Direct `.ll` text emit | Minimal deps (no llvmlite); mirrors CUDA-C backend pattern |
| Target | nvptx64, sm_86 | Our hardware; expandable to sm_XX via config |
| First op | matmul (scalar tiled) | P5-S1 exit criteria; complex enough to validate full chain |
| Execution | Reuse CudaCBackend cubin infra | Proven driver API code; DRY |
| FP precision | float32 | Correctness-first; fp16/TC later |

## Non-goals (P5-S1)
- Performance parity with cuBLAS (not required)
- wmma/Tensor Core intrinsics (P5-S3+)
- Multi-op support (P5-S2)
- Cross-hardware (P5-S4)

---
*Created: 2026-07-16*
