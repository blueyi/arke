# Arke IR Multi-Layer Architecture

> **Version:** active design reference for IR spec v2.0
> **Status:** Design Spec — active architecture reference after Stage 7 V2-only cleanup  
> **Owner:** IR Architecture Team  
> **Created:** 2026-04-06  
> **Applies to:** `arke/ir/` module, all IR-touching subsystems

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Multi-Layer Architecture Overview](#2-multi-layer-architecture-overview)
3. [Layer 4: Semantic IR](#3-layer-4-semantic-ir)
4. [Layer 3: Strategy IR](#4-layer-3-strategy-ir)
5. [Layer 2: Schedule IR](#5-layer-2-schedule-ir)
6. [Layer 1: Instruction IR](#6-layer-1-instruction-ir)
7. [Implementation Notes for Active Mainline](#7-implementation-notes-for-active-mainline)

---

## 1. Executive Summary

### 1.1 What Arke IR Is

Arke IR is the central intermediate representation of the Arke compiler toolchain — the backbone through which an AI accelerator kernel travels from high-level mathematical description down to hardware-specific binary.

**Core positioning:** Arke IR is an **LLM-Native, multi-layer IR** designed to be the core IR of the AI compilation stack. It occupies the same architectural position as MLIR but is designed from the ground up for LLM Agents as the primary decision-maker.

In the traditional AI compilation stack:

```
PyTorch / JAX / TensorFlow    (framework layer)
    ↓
Triton / TVM / XLA              (high-level compilation)
    ↓
MLIR (multi-dialect)            (compiler IR infrastructure)
    ↓
LLVM IR                         (low-level, hardware-agnostic)
    ↓
PTX / ISA                       (hardware instructions)
```

Arke IR’s position:

```
LLM Agent ←→ Arke Lang (.ak)
                ↓ parse
            Arke IR (multi-layer, LLM-Native)
              Layer 4: Semantic    ← "what to compute" (LLM primary interface)
              Layer 3: Strategy    ← "how to optimize" (LLM-driven decisions)
              Layer 2: Schedule   ← "schedule mapping" (mostly automated)
              Layer 1: Instruction ← "near-LLVM" (fully automated)
                ↓ emit
            LLVM IR → PTX / ISA
```

Arke IR can lower through MLIR standard dialects (`linalg`, `transform`, `scf`, `gpu`) when leveraging existing MLIR infrastructure is beneficial. It can also lower directly to LLVM IR when deeper hardware control is needed. If MLIR developments provide LLM-friendly interfaces in the future, Arke can directly reuse them.

#### Why LLM-Native, Not MLIR-Native

MLIR is an excellent compiler infrastructure designed for human compiler engineers:
- Dialect definitions require C++ / TableGen
- Pass implementations require C++ pattern matching
- Debugging requires understanding SSA, dominance, region semantics
- Text format is human-readable but not LLM-structured

If Arke’s vision is **LLM as the compiler’s decision-maker**, the IR must be something the LLM can directly understand and operate on:
1. **LLM-readable representation** — Arke Lang syntax and JSON serialization, not C++ text IR
2. **LLM-participatable lowering** — LLM drives optimization decisions at each layer through structured actions
3. **Clear semantics at every layer** — each IR node/op has well-defined meaning the LLM can reason about
4. **Complete lowering path** — Arke IR can lower all the way to LLVM IR without requiring MLIR as intermediary

### 1.2 Arke IR vs Traditional IRs

| Dimension | MLIR | Arke IR |
|-----------|------|---------|
| Primary author | Human compiler engineer | LLM Agent |
| Representation | C++ objects, text/binary format | Arke Lang syntax (`.ak`), serializable to JSON |
| Extensibility | New dialect in C++ | Declarative op definition in op registry |
| LLM legibility | Poor (verbose, C++ centric) | First-class (structured, minimal-token) |
| LLVM IR path | Through lowering pipelines | Through MLIR or direct emit |
| Adoption barrier | High (MLIR/C++ expertise required) | Low (Python + Arke Lang) |

Arke IR focuses on LLM-optimized ergonomics for AI kernel optimization workloads where:
- Control flow is structured (no arbitrary CFG needed at operator level)
- The decision-maker is an LLM, not a human compiler author
- Structured representation enables agent introspection and learning

Arke IR and MLIR are **complementary**: Arke IR provides the LLM-native interface; MLIR provides battle-tested compiler infrastructure. Arke IR lowers to MLIR standard dialects to leverage existing optimization passes and hardware backends.

### 1.3 Stage Evolution

Arke IR grows incrementally, with progressively deeper MLIR integration:

| Stage | Arke IR Scope | MLIR Integration | Codegen Path |
|-------|-------------|------------------|--------------|
| **Phase 1** | Layer 4 + 3 (L1) | Framework + BL1 basic pathway | Via Triton |
| **Phase 2** | + L2 | Full capability (NVIDIA + Ascend) | Via Triton + MLIR |
| **Phase 3** | + L3 | Complete integration, deeper HW control | MLIR primary |
| **Phase 4** | Full stack | Available as optional target | Direct LLVM IR |

### 1.4 Design Philosophy

**SSA by construction.** SemanticIR (Layer 4) is a DAG where every Node output is written exactly once. SSA is not a constraint to enforce — it is a structural invariant of the representation itself.

**CFG lives downstream.** Arke IR uses structured conditional flow (`ConditionalNode`) at the operator level. Arbitrary control flow graphs appear only in MLIR lowering targets (e.g., `scf.if`, `scf.for`) or LLVM IR, where they belong.

**JSON is a serialization format, not the IR itself.** Arke IR has its own language syntax (`.ak` files, see `arke-lang-spec`). JSON is a lossless serialization used for LLM Agent communication, caching, and debugging. The IR lives as typed data structures in memory; passes operate on these structures, not on JSON text.

**Semantic/Strategy separation.** `SemanticIR` (what to compute) and `StrategyIR` (how to optimize) are distinct objects. The LLM Agent explores `StrategyIR` decisions while `SemanticIR` remains immutable after construction. This separation is the core Arke architectural principle.

### 1.5 Layer-by-Layer Example: `matmul_gelu`

A fused `matmul + gelu` kernel illustrates how the same computation is represented at each Arke IR layer, with progressively more detail.

#### Layer 4: Semantic IR — "What to compute"

Pure math. No tiling, no loops, no hardware. This is the LLM Agent’s primary interface and the single source of truth for correctness verification.

```arke
kernel matmul_gelu(
    A: Tensor<[128, 768], f16>,
    B: Tensor<[768, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let C = matmul(A, B);
    let Y = gelu(C);
    return Y;
}
```

**LLM role:** Defines the computation. Immutable after construction.

#### Layer 3: Strategy IR — "How to optimize"

Optimization decisions that transform the semantic description into a concrete execution plan. The LLM Agent explores this layer — selecting tile sizes, fusion strategies, memory placement, and loop structure.

**L1 — operator-level decisions** (LLM primary):

```arke
strategy matmul_gelu for target("nvidia_ampere") {
    tile(loop="M", factors=[64])
        @rationale("64 rows = L2 cache line aligned, good for 128-thread block");
    tile(loop="N", factors=[128])
        @rationale("128 cols = maximize memory coalescing for f16");
    tile(loop="K", factors=[32])
        @rationale("A+B tiles = 64*32*2 + 32*128*2 = 12KB ≤ smem/2");
    fuse(ops=["matmul", "gelu"], type=epilogue)
        @rationale("eliminate global memory round-trip between matmul and gelu");
    place("A_tile", memory="shared");
    place("B_tile", memory="shared");
    compute(warps=4, num_stages=3);
}
```

**L2 — loop nests + memory hierarchy** (expanded from L1, LLM can refine):

```
func @matmul_gelu(%A: tensor<128x768xf16>, %B: tensor<768x3072xf16>)
    -> tensor<128x3072xf16> {
  %C = alloc tensor<128x3072xf16>
  for %i = 0 to 128 step 64 {           // M tile
    for %j = 0 to 3072 step 128 {       // N tile
      %acc = alloc tensor<64x128xf16> = 0.0
      for %k = 0 to 768 step 32 {       // K tile
        %a_tile = load %A[%i:%i+64, %k:%k+32] -> shared   // A tile in SMEM
        %b_tile = load %B[%k:%k+32, %j:%j+128] -> shared  // B tile in SMEM
        %acc = mac(%a_tile, %b_tile, %acc)                 // accumulate
      }
      %result = gelu(%acc)               // fused epilogue
      store %result -> %C[%i:%i+64, %j:%j+128]
    }
  }
  return %C
}
```

**LLM role:** Drives L1 decisions directly. Can review and refine L2 loop structure.

#### Layer 2: Schedule IR — "Schedule mapping"

Concrete schedule and hardware execution model: loop nests, thread blocks, shared memory allocation, barriers, pipeline stages. Mostly auto-generated from Strategy IR; LLM can intervene for extreme optimization.

```
func @matmul_gelu_kernel()
    grid(2, 24) block(128, 1) {
  %bid_x = blockIdx.x                // M tile index
  %bid_y = blockIdx.y                // N tile index
  %tid   = threadIdx.x

  %smem_a = alloc shared<64x32xf16>  // 4 KB
  %smem_b = alloc shared<32x128xf16> // 8 KB

  // Pipeline stage 0: prefetch first K tile
  %a_global = load global %A[%bid_x*64 : +64, 0:32]
  store %a_global -> %smem_a
  %b_global = load global %B[0:32, %bid_y*128 : +128]
  store %b_global -> %smem_b
  barrier()

  // Accumulation loop with 3-stage software pipeline
  %acc = alloc register<64x128xf16> = 0.0
  for %k_stage = 0 to 24 {           // 768/32 = 24 iterations
    %acc = mma(%smem_a, %smem_b, %acc)  // tensor core MMA
    barrier()
    // ... pipeline: async load next tile while computing ...
  }

  // Fused epilogue
  %result = gelu(%acc)
  store %result -> global %C[%bid_x*64 : +64, %bid_y*128 : +128]
}
```

**LLM role:** Review only. May intervene for extreme optimization (register pressure, barrier placement).

#### Layer 1: Instruction IR — "Near-LLVM"

Direct instruction-level representation. Fully auto-generated. LLM does not participate.

```
// Near-LLVM IR (simplified)
define void @matmul_gelu_kernel(ptr %A, ptr %B, ptr %C) {
entry:
  %bid.x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %tid.x = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %smem  = alloca [12288 x i8], align 128, addrspace(3)
  ; ... load, mma, gelu, store ...
  ret void
}
```

**LLM role:** None. Emitted directly to LLVM IR.

#### JSON Role at Each Layer

| Layer | LLM Involvement | Representation | JSON Role |
|-------|----------------|----------------|----------|
| Layer 4: Semantic | Primary author | Arke Lang (`.ak`) | Serialization, Agent API, caching |
| Layer 3: Strategy | Decision-maker (L1), reviewer (L2) | Arke Lang strategy block + IR structures | Agent API, trajectory logging |
| Layer 2: Schedule | Review only (extreme cases) | IR structures | Debug dump |
| Layer 1: Instruction | None | IR structures | None (emit LLVM IR directly) |

---

## 2. Multi-Layer Architecture Overview

Arke IR in the active mainline is organized as four conceptual layers:

- **Layer 4 — SemanticIR:** immutable operator semantics and symbolic shape information
- **Layer 3 — StrategyIR:** optimization decisions and rationale, expressed in canonical v2 decision kinds
- **Layer 2 — ScheduleIR:** compiler-generated schedule mapping and hardware-near lowering layer
- **Layer 1 — InstructionIR:** compiler-generated low-level representation near the backend boundary

### Active V2-only rules

1. `SemanticIR` is the canonical Layer 4 name. `SemanticGraph` is not part of the active contract.
2. `StrategyIR` uses canonical v2 decision names. `compute(...)` is the active resource decision surface.
3. Legacy decision kinds such as `launch_config` and transitional names such as `compute_resource` are not part of the active Stage 7 mainline.
4. Historical JSON auto-migration behavior is out of scope for the active design reference.
5. Backend-specific lowering details belong below StrategyIR, not in the Layer 3 surface.

## 3. Layer 4: Semantic IR

SemanticIR represents **what to compute**: operator graph structure, typed tensor values, symbolic dimensions, and semantic attributes required for correctness and shape reasoning.

Design constraints:
- immutable after construction for optimization work
- structured refs only; active loaders should not accept legacy unstructured string refs
- symbolic dimensions are first-class and preserved through lowering boundaries until backend-specific materialization

## 4. Layer 3: Strategy IR

StrategyIR represents **how to optimize** through bounded, rationale-carrying decisions.

Canonical active decisions include:
- `tile(...)`
- `compute(...)`
- `fuse(...)`
- `memory_layout(...)`
- conditional `when` / `otherwise` branches

Design constraints:
- target-neutral at the core representation level
- no Triton-specific directive names in the active language/IR surface
- rationale preserved through parse → IR → serialization

## 5. Layer 2: Schedule IR

ScheduleIR captures compiler-generated mapping from StrategyIR into hardware-executable structure: loop nests, resource mapping, synchronization, memory hierarchy placement, and backend-facing launch structure.

This layer is not the user-authored optimization surface. It represents the concrete schedule and hardware mapping decisions that bridge between the abstract StrategyIR and the low-level InstructionIR.

## 6. Layer 1: Instruction IR

Layer 1 is the compiler-generated low-level representation near MLIR / LLVM / backend-specific code emission. It exists to make lowering boundaries explicit, not to expose a human-authored programming model.

## 7. Implementation Notes for Active Mainline

This document is retained as an active architecture note, but its historical migration content has been removed.

Current rules:
- use `SemanticIR`, not `SemanticGraph`
- use canonical v2 StrategyIR decisions only
- remove migration shims from active code/tests/docs rather than preserving them in the mainline
- keep historical compatibility discussion out of the active tree; use git history if old migration context is needed

### Terminology note

Active naming is:
- Layer 4 = `SemanticIR`
- Layer 3 = `StrategyIR`
- Layer 2 = `ScheduleIR`
- Layer 1 = `InstructionIR`

`HardwareIR` is not an active single-layer name. If used at all, it should only appear as an informal umbrella term for the hardware-near backend stack below StrategyIR.

For the normative current-format definition, see `docs/spec/arke-ir-spec.md`.
