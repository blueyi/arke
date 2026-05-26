# Arke IR Specification v0.1.0 — Multi-Layer Architecture

> **Version:** 0.1.0
> **Status:** Final Specification
> **Date:** 2026-04-09
> **Based on:** Arke Lang Spec v0.1.0, E2E Flow Design, Agent Design
> **Philosophy:** LLM-Native, algorithm-agnostic, multi-layer separation, MLIR/LLVM interoperable

---

## Table of Contents

1. [Overview & Design Core](#1-overview--design-core)
2. [Why Arke IR — Comparison with MLIR / LLVM IR](#2-why-arke-ir--comparison-with-mlir--llvm-ir)
3. [LLM-Native Design Principles](#3-llm-native-design-principles)
4. [Multi-Layer Architecture](#4-multi-layer-architecture)
5. [Layer 4 — SemanticIR](#5-layer-4--semanticir)
6. [Layer 3 — StrategyIR](#6-layer-3--strategyir)
7. [Layer 2 — ScheduleIR](#7-layer-2--scheduleir)
8. [Layer 1 — InstructionIR](#8-layer-1--instructionir)
9. [Symbolic Dimension System](#9-symbolic-dimension-system)
10. [Verification & Rollback](#10-verification--rollback)
11. [Op Registry Interface](#11-op-registry-interface)
12. [MLIR / LLVM IR Interoperability](#12-mlir--llvm-ir-interoperability)
13. [JSON Serialization](#13-json-serialization)
14. [Layered Lowering Flow & Terminology](#14-layered-lowering-flow--terminology)
15. [Versioning](#15-versioning)

---

## 1. Overview & Design Core

### 1.1 What Is Arke IR

Arke IR is the **central intermediate representation** of the Arke compiler toolchain. It bridges the gap between high-level operator semantics (expressed in `.ak` language) and low-level hardware execution (GPU/NPU binary).

Arke IR is **not** a replacement for MLIR or LLVM IR. It is a **complementary, LLM-facing layer** that sits above them:

```
.ak (Arke Language v0.1.0)      ← Human / LLM authored
    │
    ▼
Arke IR (4 layers)              ← LLM-Native IR (this spec)
    │
    ▼
MLIR / Triton / LLVM IR         ← Traditional compiler infrastructure
    │
    ▼
GPU / NPU Binary                ← Hardware execution
```

### 1.2 Core Design Decisions

1. **LLM is the decision-maker, not the code generator.** Arke IR provides structured representations that LLMs can read, reason about, and modify through bounded actions — not free-form code.

2. **Separation of concerns across 4 layers.** Each layer has a single responsibility and a clear LLM participation level.

3. **Algorithm-agnostic.** Arke IR defines universal node types, edge semantics, and decision primitives. Specific operators are registered externally via the Op Registry.

4. **Symbolic dimensions are first-class.** Shape constraints propagate through all 4 layers, enabling dynamic shape support without losing static analysis capability.

5. **MLIR/LLVM as backend, not dependency.** Arke IR can lower to MLIR dialects or LLVM IR, but does not require them for its core semantics.

### 1.3 Satisfying README Key Features

| README Feature | Arke IR Realization |
|:---|:---|
| Semantic/Strategy Separation | Layer 4 (what) vs Layer 3 (how), strictly decoupled |
| Minimal-Token End-to-End | JSON-serializable, compact data structures, no boilerplate |
| Bounded Action Space | StrategyIR decisions are finite, compiler-enumerable |
| @rationale Annotations | First-class field on every Decision and Node |
| Compiler-as-Verifier | V0/V1/V2 verification integrated at layer boundaries |
| Structured LLM-Compiler Protocol | Tool API operates directly on IR layers via JSON |
| Safe Exploration | Checkpoint/rollback on StrategyIR, V0 rejects in <1ms |
| Multi-Hardware | Single SemanticIR → multiple StrategyIR per target |

---

## 2. Why Arke IR — Comparison with MLIR / LLVM IR

### 2.1 The Problem with Using MLIR/LLVM IR Directly

MLIR and LLVM IR are powerful compiler infrastructures designed for **human compiler engineers**:

- **Representation**: C++ objects, SSA-based text format, TableGen dialect definitions
- **Extension**: New dialects require C++ development, recompilation
- **LLM legibility**: Poor — verbose, deeply nested, requires compiler domain knowledge
- **Decision model**: Passes are hardcoded transformations, not LLM-driven choices

An LLM trying to optimize a kernel through MLIR would need to:
1. Understand SSA form, dominance, region semantics
2. Read/write C++ text IR format
3. Reason about pass ordering and interactions
4. Generate valid MLIR transformations (extremely error-prone)

This is fundamentally incompatible with the "LLM as decision-maker" paradigm.

### 2.2 Arke IR's Approach

| Dimension | MLIR / LLVM IR | Arke IR |
|:---|:---|:---|
| **Primary user** | Human compiler engineer | LLM Agent |
| **Representation** | C++ objects, SSA text format | Python dataclasses, JSON serialization |
| **Extension** | C++ dialect + TableGen | Python Op Registry (declarative) |
| **LLM legibility** | Poor (verbose, compiler-centric) | Excellent (structured, minimal-token) |
| **Decision model** | Hardcoded passes | LLM-driven bounded actions with @rationale |
| **Verification** | Static analysis passes | Multi-level V0→V1→V2 with rollback |
| **Multi-hardware** | Multiple dialects, multiple paths | Single SemanticIR, per-target StrategyIR |
| **Adoption barrier** | High (C++/MLIR expertise) | Low (Python + JSON) |

### 2.3 Complementary, Not Competing

Arke IR does not replace MLIR/LLVM IR. It **complements** them:

```
Arke IR Layer 4 (SemanticIR)     ← LLM reads/writes this
Arke IR Layer 3 (StrategyIR)     ← LLM makes decisions here
    ↓ (automated lowering)
Arke IR Layer 2 (ScheduleIR)     ← Compiler generates this
Arke IR Layer 1 (InstructionIR)  ← Compiler generates this
    ↓ (emit)
MLIR standard dialects           ← Leverage existing infrastructure
    ↓ (lower)
LLVM IR → PTX / ISA              ← Hardware execution
```

Arke IR provides the **LLM-native interface**; MLIR/LLVM provides the **battle-tested backend**.

---

## 3. LLM-Native Design Principles

### 3.1 Structured, Not Textual

Traditional IRs use text-based SSA format:
```mlir
%0 = arith.mulf %arg0, %arg1 : f32
%1 = arith.addf %0, %arg2 : f32
```

Arke IR uses structured data:
```json
{"id": "n1", "op": "mul", "inputs": ["A", "B"], "attrs": {}}
{"id": "n2", "op": "add", "inputs": ["n1", "bias"], "attrs": {}}
```

LLMs process structured JSON far more reliably than SSA text.

### 3.2 Bounded, Not Open-Ended

MLIR passes can perform arbitrary transformations. Arke IR constrains LLM actions to a **finite decision space**:

```
get_legal_actions(strategy_ir, hw_profile)
→ [
    {"kind": "tile", "dim": "M", "legal_factors": [64, 128, 256]},
    {"kind": "fuse", "candidates": [["n1", "n2"], ["n3", "n4"]]},
    {"kind": "compute", "legal_threads": [128, 256, 512]}
  ]
```

The LLM selects from this set. The compiler validates immediately.

### 3.3 Annotated, Not Opaque

Every decision carries a `@rationale` — a natural language explanation:

```json
{
  "kind": "tile",
  "dim": "M",
  "factors": [128],
  "rationale": "128 aligns with tensor core shape 16x8x16, maximizes occupancy"
}
```

This enables:
- LLM self-reflection and learning
- Cross-kernel knowledge transfer
- Human auditability
- Trajectory-based training data

### 3.4 Layered Participation

Not all layers need LLM involvement:

| Layer | LLM Role | Interaction |
|:---|:---|:---|
| Layer 4 (SemanticIR) | Author / reviewer | Read + write via `.ak` or JSON |
| Layer 3 (StrategyIR) | Decision-maker | Bounded actions via tool API |
| Layer 2 (ScheduleIR) | Observer (optional) | Read-only, review generated schedule |
| Layer 1 (InstructionIR) | None | Fully automated |

This graduated participation means the LLM focuses on high-value decisions (Layer 3) while the compiler handles mechanical lowering (Layers 2→1).

---

## 4. Multi-Layer Architecture

### 4.1 Overview

```
Layer 4: SemanticIR      WHAT to compute
                          Pure operator DAG, symbolic shapes, no optimization
                          ↕ LLM: primary author

Layer 3: StrategyIR      HOW to optimize
                          Optimization decisions, @rationale, conditional dispatch
                          ↕ LLM: decision-maker (bounded actions)

Layer 2: ScheduleIR      WHERE to execute
                          Thread/block/warp mapping, memory hierarchy placement
                          ↕ LLM: observer only (compiler-generated)

Layer 1: InstructionIR   WHAT instructions
                          Near-LLVM instructions, register allocation
                          ↕ LLM: none (fully automated)
```

### 4.2 Lowering Flow

```
.ak parse → Layer 4 (SemanticIR)
                ↓ + LLM decisions
            Layer 3 (StrategyIR)
                ↓ automated scheduling
            Layer 2 (ScheduleIR)
                ↓ instruction selection
            Layer 1 (InstructionIR)
                ↓ emit
            MLIR / Triton / LLVM IR
                ↓ compile
            GPU / NPU Binary
```

Each lowering step is:
- **Deterministic** given the input from the layer above
- **Independently serializable** (each layer can be saved/loaded as JSON)
- **Verifiable** at layer boundaries (V0 static checks)

---

## 5. Layer 4 — SemanticIR

### 5.1 Purpose

**What to compute** — Operator-level DAG representing pure computation without optimization details.

SemanticIR is:
- **Immutable** — represents the mathematical intent, never changes during optimization
- **Algorithm-agnostic** — uses universal node types, not algorithm-specific constructs
- **Symbolic-shape aware** — dimensions can be symbolic (runtime variables)
- **Verifiable** — can be checked against a reference implementation (NumPy, PyTorch eager)

### 5.2 Core Data Structures

```python
@dataclass
class SymbolicDim:
    """Named symbolic dimension (runtime variable or static constant)."""
    name: str                    # e.g., "B", "S", "D", "M", "N"
    is_static: bool = False      # True if compile-time constant
    min_val: int | None = None   # optional lower bound
    max_val: int | None = None   # optional upper bound

@dataclass
class TensorDesc:
    """Tensor descriptor: shape (may be symbolic), dtype, layout."""
    shape: list[int | str]       # int for concrete, str for symbolic dim name
    dtype: str                   # "f16", "f32", "i32", etc.
    layout: str = "row_major"    # "row_major", "col_major", "blocked", etc.

@dataclass
class Node:
    """Computation node (operator invocation)."""
    id: str                      # unique node ID (e.g., "n1", "n2")
    op_name: str                 # operator name (e.g., "matmul", "relu", "softmax")
    inputs: list[str]            # input node IDs or param names
    outputs: list[str]           # output node IDs
    attrs: dict[str, Any]        # operator-specific attributes
    rationale: str = ""          # @rationale annotation (why this op)

@dataclass
class SemanticIR:
    """Layer 4: Semantic IR."""
    kernel_id: str               # unique kernel identifier
    params: list[tuple[str, TensorDesc]]  # (name, descriptor) for inputs
    return_type: TensorDesc | list[TensorDesc]  # output type(s)
    symbolic_dims: list[SymbolicDim]  # all symbolic dimensions used
    nodes: list[Node]            # DAG nodes
    edges: list[tuple[str, str]] # (source, target) data flow edges
```

### 5.3 Example: matmul + relu

```json
{
  "kernel_id": "matmul_relu",
  "params": [
    ["A", {"shape": ["M", "K"], "dtype": "f32", "layout": "row_major"}],
    ["B", {"shape": ["K", "N"], "dtype": "f32", "layout": "col_major"}]
  ],
  "return_type": {"shape": ["M", "N"], "dtype": "f32"},
  "symbolic_dims": [
    {"name": "M", "is_static": false, "max_val": 4096},
    {"name": "K", "is_static": true},
    {"name": "N", "is_static": false, "max_val": 4096}
  ],
  "nodes": [
    {
      "id": "n1",
      "op_name": "matmul",
      "inputs": ["A", "B"],
      "outputs": ["n2"],
      "attrs": {"transpose_a": false, "transpose_b": false},
      "rationale": "Core matrix multiplication"
    },
    {
      "id": "n2",
      "op_name": "relu",
      "inputs": ["n1"],
      "outputs": ["output"],
      "attrs": {},
      "rationale": "Activation function"
    }
  ],
  "edges": [["A", "n1"], ["B", "n1"], ["n1", "n2"], ["n2", "output"]]
}
```

### 5.4 Semantics

- **Nodes** represent operator invocations. Each node has a unique `op_name` resolved via Op Registry.
- **Edges** represent data flow. An edge from `A` to `n1` means `A` is an input to node `n1`.
- **Symbolic dimensions** are declared at kernel level and propagated through node outputs via shape inference.
- **Attributes** encode operator-specific parameters (e.g., `axis` for `reduce_sum`, `transpose_a` for `matmul`).
- **Rationale** preserves LLM reasoning for each node (optional but encouraged).

---

## 6. Layer 3 — StrategyIR

### 6.1 Purpose

**How to optimize** — Optimization decisions (tiling, fusion, parallelization) independent of backend.

StrategyIR is:
- **LLM-driven** — LLM selects decisions from compiler-enumerated legal actions
- **Annotated** — every decision carries `@rationale` for transparency
- **Conditional** — supports shape-dependent dispatch via `when`/`otherwise`
- **Verifiable** — each decision is validated at V0 (static) before compilation

### 6.2 Core Data Structures

```python
@dataclass
class Decision:
    """Single optimization decision."""
    id: str                      # unique decision ID
    kind: str                    # "tile", "fuse", "parallelize", "compute", "memory", etc.
    target_nodes: list[str]      # which nodes this applies to
    level: str                   # "L1" (instruction), "L2" (block), "L3" (warp)
    params: dict[str, Any]       # decision-specific parameters
    rationale: str = ""          # @rationale annotation
    constraints: list[str] = field(default_factory=list)  # validation constraints

@dataclass
class ConditionalDecision:
    """Shape-dependent decision (when/otherwise)."""
    condition: str               # e.g., "M > 512", "S <= 1024"
    true_branch: list[Decision]  # decisions if condition is true
    false_branch: list[Decision] = field(default_factory=list)  # else branch

@dataclass
class StrategyIR:
    """Layer 3: Strategy IR."""
    kernel_id: str
    target_hw: str               # "nvidia_ampere", "ascend_910b", etc.
    decisions: list[Decision | ConditionalDecision]
    symbolic_dims: list[SymbolicDim]  # propagated from SemanticIR
    hw_constraints: dict[str, Any]  # hardware limits (shared_mem, threads, etc.)
```

### 6.3 Universal Decision Types

| Kind | Parameters | Example | Purpose |
|:---|:---|:---|:---|
| `tile` | `dim`, `factors` | `{"dim": "M", "factors": [128, 8]}` | Loop tiling for cache locality |
| `fuse` | `ops`, `type` | `{"ops": ["n1", "n2"], "type": "epilogue"}` | Operator fusion |
| `parallelize` | `dim`, `num_threads` | `{"dim": "N", "num_threads": 256}` | Thread parallelization |
| `compute` | `num_threads`, `num_stages`, `shared_mem` | `{"num_threads": 256, "num_stages": 3}` | Compute resource allocation |
| `memory` | `tensor`, `layout`, `cache_level` | `{"tensor": "A", "layout": "col_major", "cache_level": "L1"}` | Memory layout & caching |
| `compute_order` | `nodes` | `{"nodes": ["load_A", "load_B", "compute"]}` | Execution order |

### 6.3.1 Compact Fused Operator Surfaces

A Stage 7 L2 surface may be represented as either an explicit multi-node graph or as a compact registered fused op. In both cases, StrategyIR keeps the fusion intent explicit through `fuse` decisions:

- Explicit graph: `matmul -> relu` uses `fuse(ops=["matmul", "relu"], type="epilogue")`.
- Compact gated op: `silu_and_mul` carries `fuse(ops=["silu", "mul"], type="epilogue")` to expose the logical inner activation/multiply fusion even when SemanticIR has one canonical `silu_and_mul` node.
- Compact gated op: `gelu_and_mul` carries `fuse(ops=["gelu", "mul"], type="epilogue")`.

This keeps audit/coverage checks tied to real strategy evidence without forcing every registered fused operator to expand into multiple SemanticIR nodes.

### 6.4 Example: matmul_relu strategy for Ampere

```json
{
  "kernel_id": "matmul_relu",
  "target_hw": "nvidia_ampere",
  "decisions": [
    {
      "id": "d1",
      "kind": "tile",
      "target_nodes": ["n1"],
      "level": "L2",
      "params": {"dim": "M", "factors": [128, 8]},
      "rationale": "128 aligns with tensor core, 8 for warp size"
    },
    {
      "id": "d2",
      "kind": "tile",
      "target_nodes": ["n1"],
      "level": "L2",
      "params": {"dim": "N", "factors": [128, 8]},
      "rationale": "Balanced M/N tiling for occupancy"
    },
    {
      "id": "d3",
      "kind": "compute",
      "target_nodes": ["n1"],
      "level": "L2",
      "params": {"num_threads": 256, "num_stages": 3, "shared_memory": 49152},
      "rationale": "3-stage pipeline hides memory latency"
    },
    {
      "id": "d4",
      "kind": "fuse",
      "target_nodes": ["n1", "n2"],
      "level": "L3",
      "params": {"type": "epilogue"},
      "rationale": "Fuse relu as epilogue, eliminate intermediate write"
    }
  ],
  "symbolic_dims": [...],
  "hw_constraints": {
    "shared_memory": 98304,
    "max_threads_per_block": 1024,
    "warp_size": 32,
    "tensor_core_shape": [16, 8, 16]
  }
}
```

### 6.5 Conditional Strategies

```json
{
  "kind": "conditional",
  "condition": "M > 512",
  "true_branch": [
    {"id": "d_large", "kind": "tile", "params": {"dim": "M", "factors": [256]}}
  ],
  "false_branch": [
    {"id": "d_small", "kind": "tile", "params": {"dim": "M", "factors": [128]}}
  ]
}
```

---

## 7. Layer 2 — ScheduleIR

### 7.1 Purpose

**Where to execute** — Thread/block/warp mapping and memory hierarchy placement.

ScheduleIR is:
- **Hardware-specific** — encodes target-specific scheduling decisions
- **Compiler-generated** — derived from StrategyIR + hardware profile
- **LLM-observable** — LLM can review but not directly modify
- **Verifiable** — checked against hardware constraints (V0 validation)

### 7.2 Core Data Structures

```python
@dataclass
class ThreadMapping:
    """Thread/block/warp assignment."""
    loop_dim: str                # which loop dimension
    block_size: int              # threads per block
    grid_size: int               # blocks per grid
    warp_size: int = 32          # typically 32 for NVIDIA

@dataclass
class MemoryPlacement:
    """Tensor placement in memory hierarchy."""
    tensor_id: str               # which tensor
    level: str                   # "registers", "shared", "global"
    layout: str                  # "row_major", "col_major", "blocked"
    bank_conflicts: int = 0      # estimated bank conflicts

@dataclass
class ScheduleIR:
    """Layer 2: Schedule IR."""
    kernel_id: str
    target_hw: str
    thread_mappings: list[ThreadMapping]
    memory_placements: list[MemoryPlacement]
    loop_order: list[str]        # execution order of loops
    synchronization_points: list[str]  # where to insert barriers
```

### 7.3 Example

```json
{
  "kernel_id": "matmul_relu",
  "target_hw": "nvidia_ampere",
  "thread_mappings": [
    {"loop_dim": "M", "block_size": 128, "grid_size": 32, "warp_size": 32},
    {"loop_dim": "N", "block_size": 128, "grid_size": 16, "warp_size": 32}
  ],
  "memory_placements": [
    {"tensor_id": "A", "level": "shared", "layout": "row_major", "bank_conflicts": 0},
    {"tensor_id": "B", "level": "shared", "layout": "col_major", "bank_conflicts": 0},
    {"tensor_id": "C", "level": "registers", "layout": "row_major"}
  ],
  "loop_order": ["block_m", "block_n", "k", "thread_m", "thread_n"],
  "synchronization_points": ["after_load_A", "after_load_B", "after_compute"]
}
```

---

## 8. Layer 1 — InstructionIR

### 8.1 Purpose

**What instructions** — Near-LLVM instructions, register allocation, instruction scheduling.

InstructionIR is:
- **Fully automated** — compiler generates from ScheduleIR
- **Backend-ready** — can be emitted to MLIR, Triton, or LLVM IR
- **LLM-irrelevant** — LLM does not interact with this layer
- **Verifiable** — checked for correctness and resource constraints

### 8.2 Core Data Structures

```python
@dataclass
class Instruction:
    """Low-level instruction."""
    id: str
    opcode: str                  # "load", "store", "compute", "barrier", etc.
    operands: list[str]          # register/memory references
    result: str | None           # destination register
    latency: int = 1             # instruction latency

@dataclass
class InstructionIR:
    """Layer 1: Instruction IR."""
    kernel_id: str
    target_hw: str
    instructions: list[Instruction]
    register_allocation: dict[str, int]  # tensor → register count
    memory_footprint: int        # total memory used
```

### 8.3 Example (Triton-like pseudocode)

```json
{
  "kernel_id": "matmul_relu",
  "target_hw": "nvidia_ampere",
  "instructions": [
    {"id": "i1", "opcode": "load", "operands": ["A_ptr"], "result": "A_reg", "latency": 400},
    {"id": "i2", "opcode": "load", "operands": ["B_ptr"], "result": "B_reg", "latency": 400},
    {"id": "i3", "opcode": "compute", "operands": ["A_reg", "B_reg"], "result": "C_reg", "latency": 8},
    {"id": "i4", "opcode": "compute", "operands": ["C_reg"], "result": "C_relu", "latency": 1},
    {"id": "i5", "opcode": "store", "operands": ["C_relu", "C_ptr"], "latency": 400}
  ],
  "register_allocation": {"A": 256, "B": 256, "C": 128},
  "memory_footprint": 1048576
}
```

---

## 9. Symbolic Dimension System

### 9.1 Propagation Through Layers

Symbolic dimensions declared in SemanticIR propagate through all layers:

```
Layer 4 (SemanticIR):
  symbolic_dims = [M: dynamic(max=4096), K: static, N: dynamic(max=4096)]
  nodes = [matmul(A:[M,K], B:[K,N]) → C:[M,N]]

  ↓ shape inference

Layer 3 (StrategyIR):
  decisions reference M, N for conditional tiling
  e.g., "when M > 512 { tile(M, [256]) }"

  ↓ schedule generation

Layer 2 (ScheduleIR):
  thread_mappings use M, N to compute grid/block sizes
  e.g., grid_size = ceil(M / 128)

  ↓ instruction generation

Layer 1 (InstructionIR):
  concrete values substituted at runtime
```

### 9.2 Constraint Propagation

Constraints on symbolic dimensions are validated at each layer:

```
Layer 4: M ≤ 4096 (declared in where clause)
Layer 3: tile(M, [128]) requires M % 128 == 0 (added constraint)
Layer 2: grid_size = ceil(M / 128) must fit in GPU grid (hardware constraint)
Layer 1: register allocation must fit in 256KB per block (resource constraint)
```

If any constraint is violated, the decision is rejected at V0 validation.

---

## 10. Verification & Rollback

### 10.1 Multi-Level Verification

Each layer boundary has verification gates:

```
Layer 4 → Layer 3:
  V0 Static: Check StrategyIR decisions against SemanticIR
    - All target_nodes exist in SemanticIR
    - Decision parameters are valid (e.g., tile factors > 0)
    - No conflicting decisions

Layer 3 → Layer 2:
  V0 Static: Check ScheduleIR against StrategyIR + hardware profile
    - Thread counts fit in hardware limits
    - Shared memory usage ≤ available
    - Grid dimensions are valid

Layer 2 → Layer 1:
  V0 Static: Check InstructionIR against ScheduleIR
    - Register allocation ≤ available
    - Memory footprint ≤ available
    - Instruction dependencies are valid

V1 Numerical: Execute on reference implementation (NumPy)
  - Output matches expected result
  - Numerical accuracy within tolerance

V2 Performance: Profile on actual hardware
  - Execution time measured
  - Memory bandwidth utilization
  - Occupancy and throughput
```

### 10.2 Rollback Mechanism

If verification fails at any stage:

```
LLM proposes Decision D
  ↓
Compiler validates D
  ├─ V0 fails → reject immediately, suggest alternatives
  ├─ V1 fails → rollback D, suggest numerical fix
  └─ V2 fails → rollback D, suggest performance alternative
  ↓
LLM receives feedback + legal_actions
  ↓
LLM proposes new Decision D'
```

Each rollback is recorded in the trajectory for learning.

---

## 11. Op Registry Interface

### 11.1 Purpose

The Op Registry defines how new operators are registered and extended.

### 11.2 Op Definition

```python
@dataclass
class OpSignature:
    """Operator signature."""
    name: str                    # e.g., "matmul"
    inputs: list[TensorDesc]     # input tensor descriptors
    outputs: list[TensorDesc]    # output tensor descriptors
    attrs: dict[str, type]       # attribute types

@dataclass
class OpDefinition:
    """Complete operator definition."""
    signature: OpSignature
    semantic_rules: list[str]    # shape inference, constraint rules
    strategy_space: list[str]    # legal strategy decisions
    constraints: list[str]       # hardware/correctness constraints
    reference_impl: str          # reference implementation (NumPy)
```

### 11.3 Example: matmul

```python
matmul_def = OpDefinition(
    signature=OpSignature(
        name="matmul",
        inputs=[
            TensorDesc(shape=["M", "K"], dtype="f32"),
            TensorDesc(shape=["K", "N"], dtype="f32")
        ],
        outputs=[TensorDesc(shape=["M", "N"], dtype="f32")],
        attrs={"transpose_a": bool, "transpose_b": bool}
    ),
    semantic_rules=[
        "output_shape = [input[0].shape[0], input[1].shape[1]]",
        "output_dtype = input[0].dtype"
    ],
    strategy_space=[
        "tile(M, factors=[64, 128, 256])",
        "tile(N, factors=[64, 128, 256])",
        "tile(K, factors=[32, 64])",
        "fuse with relu/gelu/softmax"
    ],
    constraints=[
        "M > 0, K > 0, N > 0",
        "input[0].dtype == input[1].dtype"
    ],
    reference_impl="numpy.matmul"
)
```

---

## 12. MLIR / LLVM IR Interoperability

### 12.1 Lowering Path

Arke IR lowers to MLIR standard dialects:

```
Layer 1 (InstructionIR)
    ↓ emit
MLIR standard dialects:
  - linalg (linear algebra ops)
  - scf (structured control flow)
  - gpu (GPU-specific ops)
  - memref (memory references)
  - arith (arithmetic)
    ↓ lower
LLVM IR
    ↓ compile
PTX / ISA
```

### 12.2 Symbolic Shape Handling

Symbolic dimensions are represented in MLIR as dynamic dimensions:

```mlir
// Concrete shape
%A: memref<1024x512xf32>

// Symbolic shape (dynamic)
%A: memref<?x?xf32>

// Bounded symbolic shape (via attributes)
%A: memref<?x?xf32> {"arke.symbolic_dims": [
  {"name": "M", "max": 4096},
  {"name": "K", "static": true}
]}
```

### 12.3 Decision Mapping

StrategyIR decisions map to MLIR transformations:

| StrategyIR Decision | MLIR Transformation |
|:---|:---|
| `tile(M, [128])` | `scf.for` with tiling |
| `fuse(nodes)` | `linalg.fuse_ops` |
| `parallelize(N, 256)` | `gpu.launch_func` with thread mapping |
| `memory(layout="col_major")` | `memref.transpose` or layout attribute |

---

## 13. JSON Serialization

### 13.1 Format

All IR layers are serializable to JSON for:
- LLM interaction (read/write via tool API)
- Checkpointing (save/restore optimization state)
- Trajectory logging (JSONL format for learning)
- Version control (git-friendly text format)

### 13.2 Example: Complete IR Stack

```json
{
  "semantic_ir": { ... },
  "strategy_ir": { ... },
  "schedule_ir": { ... },
  "instruction_ir": { ... },
  "verification": {
    "v0_static": {"passed": true, "checks": [...]},
    "v1_numerical": {"passed": true, "error": 1e-6},
    "v2_performance": {"passed": true, "throughput_gflops": 150}
  },
  "metadata": {
    "kernel_id": "matmul_relu",
    "target_hw": "nvidia_ampere",
    "timestamp": "2026-04-09T00:50:00Z",
    "llm_model": "[REDACTED]"
  }
}
```

---

## 14. Layered Lowering Flow & Terminology

### 14.1 Canonical Layered Flow

The canonical active lowering path is:

```text
Arke Lang (.ak)
    │
    ├─ kernel block
    │      ↓ parse / type resolution
    │   Layer 4: SemanticIR
    │   - operator DAG
    │   - tensor types / symbolic dims
    │   - correctness source of truth
    │
    └─ strategy block
           ↓ decision parsing / validation
        Layer 3: StrategyIR
        - bounded optimization decisions
        - rationale annotations
        - conditional strategy regimes
                ↓ deterministic lowering
        Layer 2: ScheduleIR
        - loop nests and tile structure
        - thread/block/warp placement
        - memory hierarchy + synchronization
                ↓ instruction selection / backend shaping
        Layer 1: InstructionIR
        - backend-near instruction form
        - explicit low-level execution intent
                ↓ emission
        MLIR dialects / Triton / LLVM IR
                ↓ backend compilation
        PTX / ISA / device binary
```

### 14.2 Responsibility Split by Layer

| Layer | Canonical Name | Core Question | Primary Contents | Authorship |
|:---|:---|:---|:---|:---|
| 4 | `SemanticIR` | What to compute? | operator semantics, typed values, symbolic shape facts | human / LLM authored |
| 3 | `StrategyIR` | How to optimize? | bounded decisions, rationale, conditional dispatch | LLM-guided + compiler validated |
| 2 | `ScheduleIR` | How is work scheduled onto execution resources? | loop structure, resource mapping, memory/sync schedule | compiler-generated |
| 1 | `InstructionIR` | What low-level instructions / backend ops are emitted? | backend-near execution form, explicit low-level intent | compiler-generated |

### 14.3 Standard Terminology

The following names are the canonical active terms for Stage 7+:

- **Layer 4:** `SemanticIR`
- **Layer 3:** `StrategyIR`
- **Layer 2:** `ScheduleIR`
- **Layer 1:** `InstructionIR`

These names should be used consistently in:
- active spec documents
- architecture/design docs
- code comments and APIs
- tests and validation output

### 14.4 `ScheduleIR` vs `HardwareIR`

`HardwareIR` is **not** a canonical single-layer name in the active architecture.

Use the terms as follows:

| Term | Status | Meaning | Usage Rule |
|:---|:---|:---|:---|
| `ScheduleIR` | canonical | Layer 2 compiler-generated scheduling / hardware-near mapping layer | use for any reference to the Layer 2 IR |
| `InstructionIR` | canonical | Layer 1 backend-near low-level IR | use for any reference to the Layer 1 IR |
| `HardwareIR` | non-canonical umbrella term | informal shorthand for hardware-near IR below StrategyIR, usually Layer 2 + Layer 1 together | avoid as a layer name in active docs/code |

Recommended wording:
- ✅ "StrategyIR lowers to ScheduleIR, then to InstructionIR."
- ✅ "ScheduleIR and InstructionIR together form Arke's hardware-near backend IR stack."
- ❌ "Layer 2 is HardwareIR."
- ❌ "HardwareIR lowers to InstructionIR" when the real meaning is specifically `ScheduleIR`.

### 14.5 MLIR / C++ Integration Naming Guidance

When discussing integration with a C++-implemented MLIR backend:

- map **Arke Layer 2 (`ScheduleIR`)** to schedule/materialization-oriented MLIR constructs
- map **Arke Layer 1 (`InstructionIR`)** to backend-near MLIR dialect ops or direct LLVM-oriented emission
- avoid collapsing both layers into a single vague `HardwareIR` term unless the discussion is explicitly about the combined backend-facing region as a whole

This naming discipline matters because the engineering boundary is different:
- `StrategyIR → ScheduleIR` is primarily an Arke scheduling/lowering problem
- `ScheduleIR → InstructionIR` is primarily an instruction-selection / backend-shaping problem
- `InstructionIR → MLIR/LLVM` is primarily a backend emission problem

## 15. Versioning

### 15.1 Canonical Version Tag

All IR documents include version metadata:

```json
{
  "version": "0.1.0",
  "created": "2026-04-09",
  "schema": "arke-ir-v0.1.0"
}
```

This `version` is the **IR schema version**. It identifies the active contract used by `SemanticIR`, `StrategyIR`, `ScheduleIR`, `InstructionIR`, and the top-level `.akir` wrapper; it is not the Python package release version.

### 15.2 Package Version vs Schema Version

- Python distribution metadata (for example `arke.__version__` / `pyproject.toml`) tracks the package release line.
- `0.1.0` is the canonical schema version for the active IR surface and serialized `.akir` artifacts.
- The active codebase defines IR semantics directly through the Layer 4/3/2/1 model; non-canonical aliases and auto-translation paths are outside the active tree.
- Supporting docs and roadmap milestones should reference the same active IR schema version unless they are explicitly about another subsystem.

---

## References

- `docs/spec/arke-lang-spec.md` — Arke Language v0.1.0
- `docs/spec/arke-lang-vs-python-triton.md` — Comparative analysis
- `docs/architecture/e2e-flow.md` — End-to-end LLM optimization flow
- `docs/architecture/arke-harness.md` — Arke Harness architecture
- `docs/phase1/dynamic-shape-feasibility.md` — Symbolic shape design

---

**End of Arke IR Specification v0.1.0**
