# Arke IR Specification

> Version: 0.2.0-draft
> Status: 🚧 Draft
> Principle: **AI First, Human Verifiable**

---

## 0. Design Philosophy

Arke IR serves three audiences through one unified representation:

```
┌─────────────────────────────────────────────────────┐
│                    Arke IR (JSON)                    │
│                                                     │
│  ┌─────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │   LLM   │   │  Human   │   │  Tooling/CI     │  │
│  │ Agent   │   │ Developer│   │  Validators     │  │
│  │         │   │          │   │                 │  │
│  │ Reads & │   │ Reads via│   │ Validates via   │  │
│  │ writes  │   │ inspect/ │   │ JSON Schema +   │  │
│  │ via     │   │ .ak view │   │ V0/V1/V2       │  │
│  │ tool-use│   │ Edits via│   │ checks         │  │
│  │ JSON API│   │ .ak or   │   │                 │  │
│  │         │   │ JSON     │   │                 │  │
│  └─────────┘   └──────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**AI First**: The IR is JSON. Every field name, structure, and convention is
optimized for LLM consumption via tool-use. No abbreviations, no magic
numbers — explicit, self-describing, machine-parseable.

**Human Verifiable**: Every IR has a deterministic human-readable rendering
(`arke inspect`), and humans can author equivalent IR via `.ak` syntax.
All invariants are machine-checkable — a human should never need to "trust"
that an IR is valid; the validator tells them.

**Single Source of Truth**: `.ak` files, JSON files, and LLM tool-use
all produce/consume the same IR structure. There is no "LLM IR" vs
"human IR" — they are one thing with multiple views.

---

## 1. Overview

Arke IR is a two-layer intermediate representation:

```
Layer 1: Semantic IR   — WHAT to compute (immutable after creation)
Layer 2: Strategy IR   — HOW to optimize (built incrementally)
```

Both layers are JSON-serializable and self-contained. They are linked by
`kernel_id` but stored and versioned independently.

### 1.1 Why Two Layers?

Separation enables:

| Capability | How |
|:-----------|:----|
| **Multiple strategies per kernel** | Same Semantic IR → different Strategy IRs for different hardware |
| **LLM focuses on decisions** | Strategy IR is append-only decisions; LLM doesn't touch compute semantics |
| **Independent verification** | V0 validates Strategy against Semantic + HW constraints; V1 validates generated code against Semantic |
| **Human review** | Semantic IR is "the math"; Strategy IR is "the engineering decisions with rationale" |

---

## 2. Semantic IR (Layer 1) — "What to Compute"

Semantic IR is a computation graph: nodes are operators, edges are data flow.
It is **immutable after creation** — optimization never modifies the computation
it only changes the strategy for executing it.

### 2.1 Top-Level Structure

```json
{
  "version": "0.2.0",
  "kernel_id": "fused_matmul_relu",
  "params": [
    {
      "name": "A",
      "shape": [1024, 512],
      "dtype": "f16",
      "layout": "row_major"
    },
    {
      "name": "B",
      "shape": [512, 2048],
      "dtype": "f16",
      "layout": "row_major"
    }
  ],
  "return_type": {
    "shape": [1024, 2048],
    "dtype": "f16"
  },
  "nodes": [ ... ],
  "edges": [ ... ],
  "return_node": "relu_0",
  "fusion_groups": [ ... ]
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `version` | string | ✅ | Spec version (semver) |
| `kernel_id` | string | ✅ | Unique identifier for this computation |
| `params` | Param[] | ✅ | Input tensor parameters (ordered) |
| `return_type` | TensorDesc | ✅ | Output tensor description |
| `nodes` | Node[] | ✅ | Computation nodes (operators) |
| `edges` | Edge[] | ✅ | Data flow edges |
| `return_node` | string | ✅ | Node ID whose output is the kernel result |
| `fusion_groups` | FusionGroup[] | ❌ | Pre-analyzed fusion opportunities |

### 2.2 Param

Describes a kernel input parameter. Each param defines a named tensor
that can be referenced by nodes.

```json
{
  "name": "A",
  "shape": [1024, 512],
  "dtype": "f16",
  "layout": "row_major"
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `name` | string | ✅ | Parameter name (unique within kernel) |
| `shape` | int[] | ✅ | Tensor dimensions (static, positive integers) |
| `dtype` | string | ✅ | Scalar type (see §7 Type System) |
| `layout` | string | ❌ | `"row_major"` (default) or `"col_major"` |

### 2.3 Node

A computation node represents one operator application.

```json
{
  "id": "matmul_0",
  "op": "matmul",
  "inputs": {
    "A": {"ref": "param", "name": "A"},
    "B": {"ref": "param", "name": "B"}
  },
  "output": {
    "shape": [1024, 2048],
    "dtype": "f16",
    "layout": "row_major"
  },
  "semantics": {
    "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
    "index_vars": ["i", "j", "k"],
    "reduction_axes": ["k"],
    "properties": ["associative", "distributive"]
  }
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `id` | string | ✅ | Unique node identifier |
| `op` | string | ✅ | Operator name (must exist in op catalog) |
| `inputs` | dict[string, InputRef] | ✅ | Named inputs — each is a reference (see §2.4) |
| `output` | TensorDesc | ✅ | Output tensor description |
| `semantics` | Semantics | ✅ | Mathematical semantics |

### 2.4 Input References

Node inputs can reference two sources:

**Parameter reference** — reads directly from a kernel input:
```json
{"ref": "param", "name": "A"}
```

**Node reference** — reads output of a previous node:
```json
{"ref": "node", "id": "matmul_0"}
```

This two-form reference system is explicit and unambiguous. There is no
magic string parsing — both LLM and tooling can distinguish param refs
from node refs by the `"ref"` field.

> **Rationale**: The deprecated v0.1.0 spec used raw strings like `"A"` (param)
> vs `"@matmul_0"` (node ref). This was ambiguous — what if a param is named
> `"@foo"`? The new format eliminates ambiguity at the cost of verbosity,
> which is acceptable since LLMs handle JSON structure well.

### 2.5 Semantics

Mathematical description of what a node computes.

```json
{
  "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
  "index_vars": ["i", "j", "k"],
  "reduction_axes": ["k"],
  "properties": ["associative", "distributive"]
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `computation` | string | ✅ | Human-readable math formula |
| `index_vars` | string[] | ❌ | Loop variables (for tiling/parallelization) |
| `reduction_axes` | string[] | ❌ | Axes reduced over (determines parallelism constraints) |
| `properties` | string[] | ❌ | Algebraic properties (see below) |

**Standard properties**: `associative`, `commutative`, `distributive`,
`elementwise`, `monotonic`, `idempotent`, `row-wise`

Properties are critical for the LLM and tooling to reason about legal
transformations:
- `elementwise` → can be fused as epilogue
- `associative` + `commutative` → reduction order can change
- `monotonic` → output ordering preserved

### 2.6 Edge

```json
{
  "from_node": "matmul_0",
  "to_node": "relu_0",
  "tensor_name": "matmul_0_out",
  "lifetime": "local"
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `from_node` | string | ✅ | Source node ID |
| `to_node` | string | ✅ | Destination node ID |
| `tensor_name` | string | ✅ | Name for the intermediate tensor |
| `lifetime` | string | ❌ | `"local"` (default, can be eliminated by fusion) or `"persistent"` |

### 2.7 Fusion Group

Pre-analyzed fusion opportunity (auto-detected or user-specified).

```json
{
  "id": "fg_0",
  "nodes": ["matmul_0", "relu_0"],
  "fusion_type": "epilogue",
  "reason": "relu is elementwise; fusing eliminates intermediate write"
}
```

Fusion types: `epilogue` | `prologue` | `horizontal` | `vertical`

### 2.8 Semantic IR Invariants

These invariants are machine-checkable (V0 validation):

1. **Unique IDs**: Each node `id` is unique within the graph
2. **Valid references**: All input refs point to existing params or nodes
3. **DAG property**: Edges form a directed acyclic graph (no cycles)
4. **Valid ops**: All `op` names exist in the operator catalog
5. **Shape consistency**: Output shapes are consistent with op semantics and input shapes
6. **Return node exists**: `return_node` references an existing node ID
7. **All params used**: Every param is referenced by at least one node input
8. **Immutability**: Semantic IR is never modified after creation

---

## 3. Strategy IR (Layer 2) — "How to Optimize"

Strategy IR is a sequence of optimization decisions applied to a
Semantic IR. It is built incrementally — each decision is appended,
validated, and may carry a natural-language rationale.

### 3.1 Top-Level Structure

```json
{
  "version": "0.2.0",
  "kernel_id": "fused_matmul_relu",
  "target_hw": "nvidia_ampere",
  "decisions": [ ... ],
  "constraints": {
    "shared_memory_limit": 49152,
    "register_limit": 255,
    "max_threads_per_block": 1024,
    "warp_size": 32
  },
  "compile_results": [ ... ]
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `version` | string | ✅ | Spec version |
| `kernel_id` | string | ✅ | References Semantic IR's `kernel_id` |
| `target_hw` | string | ✅ | Hardware target (e.g., `"nvidia_ampere"`) |
| `decisions` | Decision[] | ✅ | Ordered optimization decisions |
| `constraints` | HWConstraints | ✅ | Hardware resource limits (copied from HW profile) |
| `compile_results` | CompileResult[] | ❌ | Results from `compile_and_profile` calls |

### 3.2 Decision

A single optimization decision. Decisions are ordered — step numbers
are monotonically increasing.

```json
{
  "step": 1,
  "level": 1,
  "kind": "tile",
  "params": {
    "loop": "i",
    "factors": [64, 16]
  },
  "rationale": {
    "text": "L2 cache line = 64 bytes, warp size = 16 threads. Tiling i by 64 maximizes L2 reuse across j iterations.",
    "lang": "en"
  },
  "validation": {
    "passed": true,
    "resource_delta": {
      "shared_memory": "+8KB"
    }
  },
  "timestamp": "2026-04-01T00:30:00Z"
}
```

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `step` | int | ✅ | 1-indexed, monotonically increasing |
| `level` | int | ❌ | Decision abstraction level (default: 1, see §3.6) |
| `kind` | string | ✅ | Decision type (see §3.3) |
| `params` | dict | ✅ | Decision-specific parameters |
| `rationale` | Rationale | ❌ | Natural language explanation (strongly encouraged) |
| `validation` | ValidationSnapshot | ❌ | V0 result at time of application |
| `timestamp` | string | ❌ | ISO-8601 timestamp |

### 3.3 Decision Kinds

| Kind | Params | What it does | Requires |
|:-----|:-------|:-------------|:---------|
| `tile` | `{loop: str, factors: int[]}` | Split a loop into tiles | Loop exists, factors divide bound |
| `reorder` | `{order: str[]}` | Reorder loop nest | Valid loop names |
| `fuse` | `{nodes: str[], type: str}` | Fuse operators | Nodes connected, type legal |
| `parallel` | `{loops: str[], mapping: dict}` | Map to GPU threads/blocks | Within thread limits |
| `place` | `{tensor: str, memory: str}` | Assign tensor to memory level | Within memory limits |
| `vectorize` | `{loop: str, width: int}` | SIMD vectorization | Innermost loop, width valid |
| `unroll` | `{loop: str, factor: int}` | Loop unrolling | Factor divides bound |
| `algorithm` | `{name: str, params: dict}` | Algorithm selection | Op supports named algorithm |

### 3.4 Rationale

Every decision can (and should) carry a natural language explanation.
This is **not a comment** — it is structured data, preserved in the IR,
and available for human review, LLM learning, and audit trails.

```json
{
  "text": "Tiling i by 64 aligns with L2 cache line and gives 16 blocks in the i dimension, enough for good SM occupancy on Ampere (30 SMs).",
  "lang": "en"
}
```

**Why rationale matters:**
- **For humans**: Makes optimization decisions auditable ("why did the AI do this?")
- **For LLMs**: Training data — rationale + outcome teaches future optimization
- **For tools**: Can flag decisions with missing rationale as "unexplained"

### 3.5 Strategy IR Invariants

1. **References valid Semantic IR**: `kernel_id` must match an existing Semantic IR
2. **Ordered decisions**: Step numbers are monotonically increasing
3. **Validated decisions**: Each decision passes V0 validation before being committed
4. **Append-only**: New decisions append to the end (rollback pops from end)
5. **HW constraints respected**: Resource estimates stay within `constraints` limits
6. **No redundant transforms**: Same loop not tiled twice, same nodes not fused twice

### 3.6 Decision Levels (Extensibility)

Decisions have an abstraction level that maps to the compilation stage:

```
Level 1: Strategy (Phase 1)     — WHAT to optimize
  tile, fuse, parallel, place, reorder, algorithm
  LLM decides, compiler handles loop generation + codegen

Level 2: Loop (Phase 2, future) — HOW to structure loops
  vectorize, unroll, pipeline, prefetch, access_pattern
  LLM decides loop-level details, compiler handles HW mapping

Level 3: Hardware (Phase 3, future) — HOW to map to hardware
  register_hint, barrier, instruction_schedule_hint, bank_conflict_avoid
  LLM decides hardware-level details, LLVM handles final codegen
```

Phase 1 only implements Level 1. Higher levels are forward-compatible:
- Old tools ignore `level` field (default: 1)
- New tools can produce/consume Level 2-3 decisions
- `list_legal_actions()` filters by supported levels

---

## 4. Relationship Between Layers

```
Semantic IR (immutable)          Strategy IR (mutable, append-only)
┌──────────────────────┐         ┌──────────────────────┐
│ kernel_id: "fmr"     │◄────────│ kernel_id: "fmr"     │
│                      │  refs   │                      │
│ params:              │         │ decisions:           │
│   A: [1024,512] f16  │         │   #1 fuse mm+relu    │
│   B: [512,2048] f16  │         │   #2 tile i=[64,16]  │
│                      │         │   #3 place A→shared  │
│ nodes:               │         │   #4 parallel        │
│   matmul_0           │         │                      │
│   relu_0             │         │ target_hw:           │
│                      │         │   nvidia_ampere      │
│ edges:               │         │                      │
│   mm → relu          │         │ constraints:         │
│                      │         │   shared_mem: 48KB   │
│ return_node: relu_0  │         │   ...                │
└──────────────────────┘         └──────────────────────┘

One Semantic IR → multiple Strategy IRs (different targets, different strategies)
```

---

## 5. Hardware Profile

Hardware profiles are JSON files that describe a target's capabilities
and constraints. They are read-only reference data — not part of the IR
itself, but referenced by Strategy IR's `target_hw` and `constraints`.

```json
{
  "name": "nvidia_ampere",
  "display_name": "NVIDIA Ampere (RTX 3060 Laptop)",
  "compute_capability": "8.6",
  "compute_units": 30,
  "matrix_unit": {
    "name": "tensor_core",
    "shapes": [[16, 8, 16]],
    "dtypes": ["f16", "bf16", "tf32"]
  },
  "memory_hierarchy": [
    {"name": "register", "size_per_cu": 65536, "latency_cycles": 1},
    {"name": "shared",   "size_per_cu": 49152, "bandwidth_gbps": 19000, "latency_cycles": 20},
    {"name": "l2_cache", "size_total": 3145728, "bandwidth_gbps": 2000},
    {"name": "global",   "bandwidth_gbps": 336, "latency_cycles": 500}
  ],
  "constraints": {
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
    "max_registers_per_thread": 255,
    "warp_size": 32
  },
  "peak_tflops": {
    "f16": 21.7,
    "f32": 10.9
  }
}
```

HW profiles live at `arke/ir/targets/<name>.json` and are loaded by
`ArkeEnv` based on the `target_hw` field.

---

## 6. Operator Catalog

Each operator in the catalog provides the information needed for all
three audiences:

```python
@dataclass(frozen=True)
class OpDefinition:
    name: str                     # "matmul"
    category: str                 # "compute" | "elementwise" | "reduce" | "move"
    inputs: dict[str, str]        # {"A": "Tensor[M,K]", "B": "Tensor[K,N]"}
    output: str                   # "Tensor[M,N]"
    computation: str              # "C[i,j] = sum(A[i,k] * B[k,j], axis=k)"
    index_vars: list[str]         # ["i", "j", "k"]
    reduction_axes: list[str]     # ["k"]
    properties: list[str]         # ["associative", "distributive"]
    can_fuse_as: str | None       # "epilogue" | "prologue" | None
    numpy_ref: str                # "np.matmul(A, B)"
    shape_inference: str          # Rule for output shape derivation
```

**P0 Operators (10)**:

| Op | Category | Computation | Fusable As | NumPy Ref |
|:---|:---------|:------------|:-----------|:----------|
| `matmul` | compute | C[i,j]=Σ_k A[i,k]*B[k,j] | prologue | `np.matmul` |
| `batch_matmul` | compute | C[b,i,j]=Σ_k A[b,i,k]*B[b,k,j] | prologue | `np.matmul` |
| `relu` | elementwise | Y=max(X,0) | epilogue | `np.maximum(X,0)` |
| `gelu` | elementwise | Y=X·Φ(X) | epilogue | `scipy.special` |
| `add` | elementwise | Y=A+B | epilogue | `A+B` |
| `mul` | elementwise | Y=A*B | epilogue | `A*B` |
| `softmax` | reduce | Y[i,j]=exp(X[i,j])/Σ exp | — | `scipy.special.softmax` |
| `reduce_sum` | reduce | Y[i]=Σ_j X[i,j] | — | `np.sum` |
| `reduce_max` | reduce | Y[i]=max_j X[i,j] | — | `np.max` |
| `transpose` | move | Y[j,i]=X[i,j] | — | `X.T` |

The op catalog is the **bridge between all audiences**:
- LLM reads `computation` and `properties` to reason about optimizations
- Humans read the same in `.ak` `kernel` definitions
- V1 validator uses `numpy_ref` to generate reference implementations

---

## 7. Type System

### 7.1 Scalar Types

16 types in 4 groups:

| Group | Types |
|:------|:------|
| Float | `f16`, `f32`, `f64`, `bf16` |
| Integer | `i8`, `i16`, `i32`, `i64` |
| Unsigned | `u8`, `u16`, `u32`, `u64` |
| Special | `bool`, `index` |

### 7.2 Tensor Types

```
shape: int[]       — e.g., [1024, 512]
dtype: string      — e.g., "f16"
layout: string     — "row_major" | "col_major" (default: "row_major")
```

Shapes are static in v0.2.0 (all dimensions known at IR creation time).
Dynamic shapes are a future extension.

### 7.3 Memory Levels

`"global"` | `"shared"` | `"local"` | `"register"`

Used in `place` decisions to specify where a tensor tile should reside.

---

## 8. JSON Schema Validation

Each IR component has a corresponding JSON Schema file for machine
validation. These are the **formal specification** — if there's a
conflict between this document and the schema, the schema wins.

| Schema | Path | Validates |
|:-------|:-----|:----------|
| Semantic IR | `arke/ir/schemas/semantic.schema.json` | SemanticIR JSON |
| Strategy IR | `arke/ir/schemas/strategy.schema.json` | StrategyIR JSON |
| HW Profile | `arke/ir/schemas/hw_profile.schema.json` | Hardware profile |
| Decision | `arke/ir/schemas/decision.schema.json` | Single decision |

Validation chain:
```
JSON file/string
  → JSON Schema validation (structural correctness)
  → V0 Static Validation (semantic correctness: shapes, constraints, legality)
  → V1 Numerical Validation (runtime correctness: vs NumPy reference)
  → V2 Performance Validation (efficiency: vs vendor baseline)
```

---

## 9. Interaction Patterns

### 9.1 LLM Creates Kernel via Tool-Use

```
LLM calls create_kernel({
  name: "fused_matmul_relu",
  params: [{name:"A", shape:[1024,512], dtype:"f16"}, ...],
  return_type: {shape:[1024,2048], dtype:"f16"},
  computations: [
    {id:"matmul_0", op:"matmul", inputs:{A:"A", B:"B"}},
    {id:"relu_0", op:"relu", inputs:{X:"@matmul_0"}}
  ]
})

→ System builds Semantic IR (resolves refs, infers shapes, detects fusions)
→ Returns: semantic_ir + auto_analysis
```

### 9.2 Human Authors .ak File

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}
```

→ Parser produces identical Semantic IR as §9.1

### 9.3 Tool Validates IR

```bash
arke verify kernel.json          # V0: structural + semantic invariants
arke verify kernel.json --v1     # V1: + numerical correctness
arke verify kernel.json --v2     # V2: + performance profiling
```

### 9.4 Human Inspects IR

```bash
arke inspect kernel.json
```

Output:
```
Kernel: fused_matmul_relu
  Params: A[1024×512, f16], B[512×2048, f16]
  Return: [1024×2048, f16]

  Nodes:
    matmul_0: matmul(A, B) → [1024×2048, f16]
      C[i,j] = Σ_k A[i,k]·B[k,j]  [associative, distributive]
    relu_0: relu(@matmul_0) → [1024×2048, f16]
      Y = max(X, 0)  [elementwise, monotonic]

  Edges: matmul_0 → relu_0
  Fusion: [matmul_0 + relu_0] epilogue

  Strategy: (none — no optimization applied yet)
```

### 9.5 Human Reviews Strategy

```bash
arke inspect strategy.json
```

Output:
```
Strategy for fused_matmul_relu on nvidia_ampere:
  #1 fuse(matmul_0 + relu_0, epilogue)
     → "relu is elementwise; fusing eliminates 4MB intermediate write"
  #2 tile(i=[64, 16])
     → "L2 cache line = 64, warp size = 16"
  #3 place(A_tile → shared, B_tile → shared)
     → "A reused 16× across j, B reused 16× across i"
  #4 parallel(i_outer → block.x, j_outer → block.y)
     → "16×16 = 256 blocks, good SM occupancy"

  Resources: shared_mem=24KB/48KB, est_threads=256
  Compiled: 2 attempts, best=82% cuBLAS (125μs)
```

---

## 10. Versioning and Compatibility

- IR versions follow semver: `MAJOR.MINOR.PATCH`
- **MAJOR** bump = breaking schema changes (old IR won't validate)
- **MINOR** bump = new optional fields (old IR still valid)
- **PATCH** bump = clarifications, no schema changes
- Every IR file carries its `version` — tooling can detect and migrate

---

## 11. File Conventions

| Content | Extension | Location |
|:--------|:----------|:---------|
| Semantic IR | `.semantic.json` | Anywhere |
| Strategy IR | `.strategy.json` | Anywhere |
| HW Profile | `.json` | `arke/ir/targets/` |
| Combined (kernel + strategy) | `.arke.json` | CLI output |
| Human source | `.ak` | User workspace |
| JSON Schema | `.schema.json` | `arke/ir/schemas/` |

---

*Spec version: 0.2.0-draft | Date: 2026-04-01*
*Previous: docs/spec/deprecated/arke-ir-spec.md (v0.1.0)*
*Implementation: arke/ir/semantic.py, arke/ir/strategy.py, arke/ir/builder.py*
