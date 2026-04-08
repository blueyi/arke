# Arke IR Specification v2.0 — Multi-Layer Architecture

> **Version:** 2.0.0  
> **Status:** Final Specification  
> **Date:** 2026-04-08  
> **Scope:** Layer 4 (Semantic) through Layer 1 (Instruction)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Layer 4 — SemanticIR](#2-layer-4--semanticir)
3. [Layer 3 — StrategyIR](#3-layer-3--strategyir)
4. [Layer 2 — ScheduleIR](#4-layer-2--scheduleir)
5. [Layer 1 — InstructionIR](#5-layer-1--instructionir)
6. [Symbolic Dimension Propagation](#6-symbolic-dimension-propagation)
7. [All 45 Operators](#7-all-45-operators)
8. [MLIR Integration](#8-mlir-integration)

---

## 1. Overview

### 1.1 Multi-Layer Architecture

Arke IR is a **4-layer intermediate representation** that separates concerns across the compilation pipeline:

```
.ak (Arke Language v2.0)
    ↓
Layer 4: SemanticIR      (What to compute — operator graph + symbolic shapes)
    ↓
Layer 3: StrategyIR      (How to optimize — decisions, L1/L2/L3 levels)
    ↓
Layer 2: ScheduleIR      (Thread/block/warp mapping — hardware-specific)
    ↓
Layer 1: InstructionIR   (Near-LLVM IR — backend-ready)
    ↓
LLVM IR / MLIR standard dialects
```

### 1.2 Design Principles

- **Separation of concerns** — Each layer has a single responsibility
- **LLM-Native** — Layers 4 and 3 are LLM-authored; Layers 2 and 1 are compiler-generated
- **Symbolic shapes first-class** — Symbolic dimensions propagate through all layers
- **Backend-agnostic core** — Layer 3 decisions are independent of backend
- **Reversible lowering** — Each layer can be serialized/deserialized independently

---

## 2. Layer 4 — SemanticIR

### 2.1 Purpose

**What to compute** — Operator-level DAG representing the computation without optimization details.

### 2.2 Core Data Structures

```python
@dataclass
class SymbolicDim:
    """Named symbolic dimension (runtime variable)."""
    name: str                    # e.g., "B", "S", "D"
    min: int | None = None       # optional lower bound
    max: int | None = None       # optional upper bound
    is_dynamic: bool = True

@dataclass
class TensorDesc:
    """Tensor descriptor: shape (may be symbolic), dtype, layout."""
    shape: list[int | SymbolicDim]
    dtype: str                   # "f16", "f32", "i32", etc.
    layout: str = "row_major"    # "row_major", "col_major", etc.

@dataclass
class Param:
    """Kernel input parameter."""
    name: str
    shape: list[int | SymbolicDim]
    dtype: str
    layout: str = "row_major"

@dataclass
class Node:
    """Computation node (operator invocation)."""
    id: str                      # unique node ID
    op_name: str                 # operator name (e.g., "matmul", "relu")
    inputs: list[str]            # input node IDs
    outputs: list[str]           # output node IDs
    attrs: dict[str, Any]        # operator-specific attributes
    rationale: str = ""          # @rationale annotation

@dataclass
class SemanticIR:
    """Layer 4: Semantic IR."""
    kernel_name: str
    params: list[Param]
    return_type: TensorDesc
    symbolic_dims: list[SymbolicDim]
    nodes: list[Node]            # DAG of computation
    edges: list[tuple[str, str]] # (source_node_id, target_node_id)
```

### 2.3 Semantics

- **Nodes** represent operator invocations (e.g., `matmul`, `relu`, `softmax`)
- **Edges** represent data flow between nodes
- **Symbolic dimensions** are declared at kernel level and propagated through node outputs
- **Attributes** encode operator-specific parameters (e.g., `axis` for `reduce_sum`)
- **Rationale** preserves LLM reasoning for each node

### 2.4 Example

```python
# Kernel: matmul_relu(A: [M, K], B: [K, N]) -> [M, N]
# where M: dynamic(max=4096), K: static, N: dynamic(max=4096)

semantic_ir = SemanticIR(
    kernel_name="matmul_relu",
    params=[
        Param("A", [SymbolicDim("M", max=4096), SymbolicDim("K")], "f32"),
        Param("B", [SymbolicDim("K"), SymbolicDim("N", max=4096)], "f32"),
    ],
    return_type=TensorDesc([SymbolicDim("M"), SymbolicDim("N")], "f32"),
    symbolic_dims=[
        SymbolicDim("M", max=4096),
        SymbolicDim("K"),
        SymbolicDim("N", max=4096),
    ],
    nodes=[
        Node("n1", "matmul", ["A", "B"], ["n2"], {"transpose_b": False}),
        Node("n2", "relu", ["n1"], ["output"], {}),
    ],
    edges=[("A", "n1"), ("B", "n1"), ("n1", "n2"), ("n2", "output")],
)
```

---

## 3. Layer 3 — StrategyIR

### 3.1 Purpose

**How to optimize** — Optimization decisions (tiling, fusion, parallelization) independent of backend.

### 3.2 Core Data Structures

```python
@dataclass
class Decision:
    """Single optimization decision."""
    id: str
    decision_type: str           # "tile", "fuse", "parallelize", "compute", etc.
    target_node: str             # which node this decision applies to
    level: str                   # "L1" (instruction), "L2" (block), "L3" (warp)
    params: dict[str, Any]       # decision-specific parameters
    rationale: str = ""          # @rationale annotation
    constraints: list[str] = field(default_factory=list)

@dataclass
class ConditionalDecision:
    """Shape-dependent decision (when/otherwise)."""
    condition: str               # e.g., "S > 1024"
    true_decisions: list[Decision]
    false_decisions: list[Decision]

@dataclass
class StrategyIR:
    """Layer 3: Strategy IR."""
    kernel_id: str
    target_hw: str               # "nvidia_ampere", "ascend_910b", etc.
    decisions: list[Decision | ConditionalDecision]
    symbolic_dims: list[SymbolicDim]  # propagated from SemanticIR
    constraints: dict[str, Any]  # hardware constraints
```

### 3.3 Decision Types

| Type | Parameters | Example |
|:-----|:-----------|:--------|
| `tile` | `loop`, `factors` | `tile(loop="m", factors=[128, 8])` |
| `fuse` | `nodes` | `fuse(nodes=["n1", "n2"])` |
| `parallelize` | `loop`, `num_threads` | `parallelize(loop="n", num_threads=256)` |
| `compute` | `num_threads`, `num_stages` | `compute(num_threads=256, num_stages=3)` |
| `memory` | `layout`, `cache_level` | `memory(layout="col_major", cache_level="L1")` |

### 3.4 Example

```python
strategy_ir = StrategyIR(
    kernel_id="matmul_relu",
    target_hw="nvidia_ampere",
    decisions=[
        Decision("d1", "tile", "n1", "L2", {"loop": "m", "factors": [128]}),
        Decision("d2", "tile", "n1", "L2", {"loop": "n", "factors": [128]}),
        Decision("d3", "compute", "n1", "L1", {"num_threads": 256, "num_stages": 3}),
        Decision("d4", "fuse", "n2", "L1", {"nodes": ["n1", "n2"]}),
    ],
    symbolic_dims=[...],  # propagated from SemanticIR
    constraints={"max_threads_per_block": 1024, "max_shared_memory": 96000},
)
```

---

## 4. Layer 2 — ScheduleIR

### 4.1 Purpose

**Thread/block/warp mapping** — Hardware-specific scheduling decisions.

### 4.2 Core Data Structures

```python
@dataclass
class ThreadMapping:
    """Maps loop dimensions to thread hierarchy."""
    loop_name: str
    thread_level: str            # "thread", "warp", "block"
    num_threads: int
    stride: int = 1

@dataclass
class MemoryMapping:
    """Maps tensors to memory hierarchy."""
    tensor_name: str
    memory_level: str            # "register", "shared", "global"
    layout: str                  # "row_major", "col_major"

@dataclass
class ScheduleIR:
    """Layer 2: Schedule IR."""
    kernel_id: str
    target_hw: str
    thread_mappings: list[ThreadMapping]
    memory_mappings: list[MemoryMapping]
    launch_config: dict[str, int]  # grid_size, block_size, etc.
```

---

## 5. Layer 1 — InstructionIR

### 5.1 Purpose

**Near-LLVM IR** — Backend-ready instructions, ready for code generation.

### 5.2 Core Data Structures

```python
@dataclass
class Instruction:
    """Single instruction."""
    id: str
    opcode: str                  # "load", "store", "compute", "sync", etc.
    operands: list[str]          # register/memory references
    result: str | None           # destination register/memory

@dataclass
class InstructionIR:
    """Layer 1: Instruction IR."""
    kernel_id: str
    target_hw: str
    instructions: list[Instruction]
    register_allocation: dict[str, str]  # variable → register mapping
    memory_layout: dict[str, tuple[int, int]]  # variable → (address, size)
```

---

## 6. Symbolic Dimension Propagation

### 6.1 Propagation Rules

Symbolic dimensions declared in `where` clause propagate through all layers:

1. **Layer 4 (SemanticIR)** — Symbolic dims in parameter types and node outputs
2. **Layer 3 (StrategyIR)** — Symbolic dims used in conditional decisions
3. **Layer 2 (ScheduleIR)** — Symbolic dims in thread mapping bounds
4. **Layer 1 (InstructionIR)** — Symbolic dims in loop bounds and memory allocation

### 6.2 Shape Inference Pass

```python
def infer_shapes(semantic_ir: SemanticIR) -> dict[str, TensorDesc]:
    """Infer output shapes for all nodes."""
    shapes = {}
    for param in semantic_ir.params:
        shapes[param.name] = TensorDesc(param.shape, param.dtype)
    
    for node in semantic_ir.nodes:
        # Apply operator-specific shape inference rules
        output_shape = infer_op_shape(node.op_name, node.inputs, shapes, node.attrs)
        for output_id in node.outputs:
            shapes[output_id] = output_shape
    
    return shapes
```

### 6.3 Constraint Propagation

Symbolic dimension constraints (e.g., `S % 128 == 0`) are propagated and validated at each layer.

---

## 7. All 45 Operators

### Operator Catalog

Each operator has:
- **Signature** — Input/output types with symbolic dimensions
- **Semantics** — Computation definition
- **Layer 4 representation** — SemanticIR node attributes
- **Layer 3 decisions** — Typical optimization strategies

#### OT0: Elementwise (12 ops)

| Op | Signature | L4 Attrs | L3 Decisions |
|:---|:----------|:---------|:------------|
| relu | `[...] f → [...] f` | `{}` | `tile(factors=[128])` |
| gelu | `[...] f → [...] f` | `{}` | `tile(factors=[128])` |
| ... | ... | ... | ... |

#### OT1: Reduction (10 ops)

| Op | Signature | L4 Attrs | L3 Decisions |
|:---|:----------|:---------|:------------|
| softmax | `[B,S,D] f → [B,S,D] f` | `axis=2` | `tile(loop="B", factors=[8])` |
| layernorm | `[B,S,D] f → [B,S,D] f` | `eps=1e-5` | `parallelize(loop="B", num_threads=256)` |
| ... | ... | ... | ... |

#### OT2: Compute-Dense (11 ops)

| Op | Signature | L4 Attrs | L3 Decisions |
|:---|:----------|:---------|:------------|
| matmul | `[M,K] f × [K,N] f → [M,N] f` | `{}` | `tile(loop="m", factors=[128]); tile(loop="n", factors=[128])` |
| batch_matmul | `[B,M,K] f × [B,K,N] f → [B,M,N] f` | `{}` | `parallelize(loop="B", num_threads=32)` |
| ... | ... | ... | ... |

#### OT3: Gated Activation (7 ops)

| Op | Signature | L4 Attrs | L3 Decisions |
|:---|:----------|:---------|:------------|
| swiglu | `[B,S,2D] f → [B,S,D] f` | `{}` | `fuse(nodes=["linear", "swiglu"])` |
| geglu | `[B,S,2D] f → [B,S,D] f` | `{}` | `fuse(nodes=["linear", "geglu"])` |
| ... | ... | ... | ... |

#### OT4: Attention (5 ops)

| Op | Signature | L4 Attrs | L3 Decisions |
|:---|:----------|:---------|:------------|
| flash_attention | `[B,H,S,D] f × [B,H,S,D] f × [B,H,S,D] f → [B,H,S,D] f` | `block_size=128` | `tile(loop="S", factors=[128]); compute(num_stages=3)` |
| grouped_query_attention | `[B,H,S,D] f × [B,1,S,D] f × [B,1,S,D] f → [B,H,S,D] f` | `num_kv_heads=1` | `parallelize(loop="H", num_threads=256)` |
| ... | ... | ... | ... |

---

## 8. MLIR Integration

### 8.1 Layer 1 → MLIR Lowering

InstructionIR lowers to MLIR standard dialects:

- **`linalg`** — Linear algebra operations (matmul, reduce, etc.)
- **`scf`** — Structured control flow (loops, conditionals)
- **`gpu`** — GPU-specific operations (thread/block mapping)
- **`memref`** — Memory reference operations
- **`arith`** — Arithmetic operations

### 8.2 Example: matmul

```mlir
// Layer 1 InstructionIR → MLIR linalg

func.func @matmul_kernel(
    %A: memref<?x?xf32>,
    %B: memref<?x?xf32>,
    %C: memref<?x?xf32>
) {
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                  outs(%C : memref<?x?xf32>)
    return
}
```

### 8.3 Symbolic Shape Handling in MLIR

Symbolic dimensions are represented as `?` in memref types:

```mlir
// Dynamic shapes
%A: memref<?x?xf32>  // [B, S] with B, S dynamic

// Bounded dynamic shapes (via attributes)
%A: memref<?x?xf32> {"arke.symbolic_dims": [
    {"name": "B", "max": 64},
    {"name": "S", "max": 8192}
]}
```

---

## References

- `docs/spec/arke-lang-spec-v2.md` — Arke Language v2.0
- `docs/phase1/dynamic-shape-feasibility.md` — Symbolic shape design
- `docs/architecture/e2e-flow.md` — End-to-end LLM optimization
- `docs/spec/ir-mlir-mapping.md` — Detailed IR-to-MLIR mapping

---

**End of Arke IR Specification v2.0**
