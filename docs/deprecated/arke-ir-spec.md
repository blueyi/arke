# Arke IR Specification

> Version: 0.1.0-draft
> Status: 🚧 Draft — schema may change before v0.1.0

---

## 1. Overview

Arke IR is a two-layer intermediate representation:

```
Layer 1: Semantic IR (SIR)  — WHAT to compute
Layer 2: Strategy IR (StrIR) — HOW to optimize
```

Both layers are JSON-serializable. The LLM interacts with them through
the ArkeEnv tool-use protocol.

## 2. Semantic IR

### 2.1 Structure

```json
{
  "version": "0.1.0",
  "graph_id": "fused_matmul_relu",
  "nodes": [...],
  "edges": [...],
  "fusion_groups": [...]
}
```

### 2.2 Node

```json
{
  "id": "matmul_0",
  "op": "matmul",
  "inputs": {
    "A": {"shape": [1024, 512], "dtype": "f16", "layout": "row_major"},
    "B": {"shape": [512, 2048], "dtype": "f16", "layout": "row_major"}
  },
  "output": {"shape": [1024, 2048], "dtype": "f16", "layout": "row_major"},
  "semantics": {
    "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
    "index_vars": ["i", "j", "k"],
    "reduction_axes": ["k"],
    "properties": ["associative", "distributive"]
  }
}
```

### 2.3 Edge

```json
{
  "from_node": "matmul_0",
  "to_node": "relu_0",
  "tensor_name": "intermediate",
  "lifetime": "local"
}
```

### 2.4 Fusion Group

```json
{
  "id": "fg_0",
  "nodes": ["matmul_0", "relu_0"],
  "fusion_type": "epilogue"
}
```

Fusion types: `epilogue` | `prologue` | `horizontal` | `vertical`

### 2.5 Invariants

1. Each node has a unique `id`
2. All `inputs` reference either a TensorDesc or `@node_id` (for data dependencies)
3. Edges form a DAG (no cycles)
4. `fusion_groups` reference existing node IDs
5. Semantic IR is **immutable** after creation — optimization does not modify it

## 3. Strategy IR

### 3.1 Structure

```json
{
  "version": "0.1.0",
  "kernel_id": "fused_matmul_relu",
  "target_hw": "nvidia_ampere",
  "decisions": [...],
  "constraints": {
    "shared_memory_limit": 49152,
    "register_limit": 255,
    "max_threads_per_block": 1024,
    "warp_size": 32
  }
}
```

### 3.2 Decision

```json
{
  "kind": "tile",
  "params": {"loop": "i", "factors": [64, 16]},
  "rationale": {"text": "L2 cache line = 64, warp size = 16", "lang": "en"},
  "step": 1
}
```

### 3.3 Decision Kinds

| Kind | Params | Semantics | Modifies |
|:-----|:-------|:----------|:---------|
| `tile` | `{loop, factors}` | Split loop into tiles | Loop structure |
| `reorder` | `{order}` | Reorder loop nest | Loop order |
| `fuse` | `{ops, type}` | Fuse operators | Operator graph |
| `parallel` | `{loops, mapping}` | Map to GPU threads | Parallelism |
| `place` | `{tensor, memory}` | Memory placement | Data layout |
| `vectorize` | `{loop, width}` | SIMD vectorization | Inner loop |
| `unroll` | `{loop, factor}` | Loop unrolling | Inner loop |
| `algorithm` | `{name, params}` | Algorithm selection | Global strategy |

### 3.4 Invariants

1. `kernel_id` must reference an existing Semantic IR
2. Decisions are ordered — step numbers are monotonically increasing
3. Each decision passes V0 validation before being committed
4. `rationale` is optional but strongly encouraged
5. Strategy IR is **append-only** (except rollback, which pops from end)

## 4. Relationship Between Layers

```
Semantic IR (immutable)          Strategy IR (mutable)
┌──────────────────┐             ┌──────────────────┐
│ graph_id: "mm"   │◄────────────│ kernel_id: "mm"  │
│                  │  references │                  │
│ nodes:           │             │ decisions:       │
│   matmul_0       │             │   #1 tile i      │
│   relu_0         │             │   #2 tile j      │
│                  │             │   #3 fuse         │
│ edges:           │             │   #4 parallel     │
│   mm→relu        │             │                  │
│                  │             │ target_hw:       │
│ fusion_groups:   │             │   nvidia_ampere  │
│   [mm, relu]     │             │                  │
└──────────────────┘             └──────────────────┘
```

One Semantic IR can have **multiple** Strategy IRs (different targets, different strategies).

## 5. Hardware Profile

```json
{
  "name": "nvidia_ampere",
  "compute_capability": "8.6",
  "compute_units": 30,
  "matrix_unit": {
    "name": "tensor_core",
    "shapes": [[16, 8, 16]],
    "dtypes": ["f16", "bf16", "tf32"]
  },
  "memory_hierarchy": [
    {"name": "register", "size_per_cu": 65536, "latency_cycles": 1},
    {"name": "shared",   "size_per_cu": 49152, "bandwidth_gbps": 19000},
    {"name": "l2_cache", "size_total": 3145728, "bandwidth_gbps": 2000},
    {"name": "global",   "bandwidth_gbps": 336, "latency_cycles": 500}
  ],
  "constraints": {
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
    "max_registers_per_thread": 255,
    "warp_size": 32
  },
  "peak_tflops": {"f16": 21.7, "f32": 10.9}
}
```

## 6. JSON Schema Files

| Schema | Path | Validates |
|:-------|:-----|:----------|
| Semantic IR | `arke/ir/schemas/semantic.schema.json` | SemanticGraph JSON |
| Strategy IR | `arke/ir/schemas/strategy.schema.json` | StrategyIR JSON |
| HW Profile | `arke/ir/schemas/hw_profile.schema.json` | Hardware profile JSON |

> JSON Schema files will be generated from the Python dataclass definitions (W1-04/05).

---

*Spec version: 0.1.0-draft | Date: 2026-03-31*
*Implementation: arke/ir/semantic.py, arke/ir/strategy.py, arke/ir/builder.py*
