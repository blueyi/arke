# Arke IR Specification v1.0

> **Version:** 1.0  
> **Status:** Frozen (Phase 1 exit)  
> **Compatibility:** Arke compiler ≥ 0.1.0-dev  
> **Implementation:** `arke/ir/semantic.py`, `arke/ir/strategy.py`

---

## 1. Overview

Arke IR is a **two-layer intermediate representation**:

| Layer | Class | Describes | Immutable? |
|:------|:------|:----------|:----------:|
| **SemanticIR** | `arke.ir.semantic.SemanticIR` | *What* to compute — pure math, no hardware assumptions | ✅ Yes |
| **StrategyIR** | `arke.ir.strategy.StrategyIR` | *How* to optimize — tiling, mapping, fusion decisions | ❌ Mutable (LLM Agent iterates) |

This separation is fundamental: the LLM Agent operates only on StrategyIR, never mutating the mathematical semantics.

---

## 2. SemanticIR

### 2.1 Top-level Fields

```python
@dataclass
class SemanticIR:
    version: str           # "0.3.0"
    kernel_id: str         # unique kernel name
    params: list[Param]    # input tensors
    nodes: list[Node]      # operator DAG
    edges: list[Edge]      # data flow edges
    output: TensorDesc     # output tensor descriptor
    fusion_groups: list[FusionGroup]  # optional fusion hints
    metadata: dict         # arbitrary key-value metadata
```

### 2.2 Param

```python
@dataclass
class Param:
    name: str          # parameter name (e.g., "A", "weight")
    shape: list[int]   # e.g., [1024, 1024]
    dtype: str         # e.g., "f16"
    layout: str        # "row_major" | "col_major"
```

### 2.3 Node

```python
@dataclass
class Node:
    id: str            # unique node id, e.g., "matmul_0"
    op: str            # operator name from OP_CATALOG
    inputs: dict[str, InputRef]   # {input_name: InputRef}
    output: TensorDesc             # output tensor shape/dtype
    semantics: Semantics           # mathematical description
    attrs: dict                    # op-specific attributes (e.g., eps, axis)
```

### 2.4 InputRef

```python
@dataclass
class InputRef:
    kind: str   # "param" | "node"
    name: str   # param.name or node.id
```

### 2.5 Edge

```python
@dataclass
class Edge:
    src_node: str   # source node id (or "param:<name>" for inputs)
    dst_node: str   # destination node id
    src_port: str   # output port name on source
    dst_port: str   # input port name on destination
```

### 2.6 TensorDesc

```python
@dataclass
class TensorDesc:
    shape: list[int]
    dtype: str
    layout: str = "row_major"
```

### 2.7 Semantics

Human-readable mathematical description of an operator:

```python
@dataclass
class Semantics:
    description: str   # e.g., "C[i,j] = sum_k(A[i,k] * B[k,j])"
    domain: str        # category: "compute" | "elementwise" | "reduction" | "memory"
    properties: list[str]   # e.g., ["associative", "commutative"]
```

### 2.8 FusionGroup

```python
@dataclass
class FusionGroup:
    id: str
    nodes: list[str]   # node ids to fuse
    kind: str          # "epilogue" | "horizontal" | "vertical"
    rationale: str
```

### 2.9 JSON Serialization

`SemanticIR.to_dict()` → `SemanticIR.from_dict()` is **lossless** for all OP_CATALOG operators (verified in G6.5 tests).

---

## 3. StrategyIR

### 3.1 Top-level Fields

```python
@dataclass
class StrategyIR:
    version: str         # "0.2.0"
    kernel_id: str       # reference to SemanticIR.kernel_id
    target_hw: str       # e.g., "nvidia_ampere"
    decisions: list[Decision]
    constraints: HardwareConstraints
```

### 3.2 Decision

```python
@dataclass
class Decision:
    step: int            # auto-assigned sequential step number
    kind: str            # see §3.3
    params: dict         # kind-specific parameters
    rationale: Rationale | None   # optional @rationale
```

### 3.3 Decision Kinds

| Kind | Required params | Optional params | Description |
|:-----|:----------------|:----------------|:------------|
| `tile` | `loop: str`, `factors: list[int]` | — | Tile a loop |
| `reorder` | `order: list[str]` | — | Reorder loop nest |
| `parallel` | `loops: list[str]`, `mapping: dict[str,str]` | — | Map to GPU threads/blocks |
| `fuse` | `ops: list[str]` | `type: str` | Fuse operator epilogue |
| `vectorize` | `loop: str`, `width: int` | — | Vectorize a loop |
| `place` | `tensor: str`, `memory: str` | — | Memory placement |
| `launch_config` | `num_warps: int` | `num_stages: int`, `block_sizes: dict` | GPU launch parameters |
| `unroll` | `loop: str`, `factor: int` | — | Loop unrolling |
| `autotune` | `configs: list[dict]`, `key: list[str]` | — | Autotuning candidates |
| `algorithm` | `name: str` | — | Algorithm variant selection |

### 3.4 Rationale

```python
@dataclass
class Rationale:
    text: str   # human-readable explanation
    lang: str   # language code, default "en"
```

`@rationale` is **non-semantic** — it does not affect code generation. It serves:
- Human + LLM explainability
- Trajectory logging (Agent JSONL)
- MLIR annotation preservation (Phase 3)

### 3.5 HardwareConstraints

```python
@dataclass
class HardwareConstraints:
    shared_memory_limit: int    # bytes
    register_limit: int         # registers per thread
    max_threads_per_block: int
    warp_size: int              # default 32
```

### 3.6 Mutation API

```python
ir = StrategyIR(kernel_id="matmul_k", target_hw="nvidia_ampere")

# Add decisions
ir.tile("M", [64], "tensor-core aligned")
ir.reorder(["M", "N", "K"])
ir.add_decision(Decision(kind="launch_config", params={"num_warps": 4}))

# Rollback (Agent error recovery)
ir.pop_decisions(n=2)   # remove last 2 decisions
```

### 3.7 JSON Serialization

`StrategyIR.to_dict()` → `StrategyIR.from_dict()` is **lossless** including nested map/array params and `@rationale` text (verified in G6.5 tests).

---

## 4. OP_CATALOG

Defined in `arke/ir/ops/catalog.py`. Each entry is an `OpDef`:

```python
@dataclass
class OpDef:
    name: str
    category: str        # "A" | "B" | "C" | "D" | "E"
    inputs: dict[str, TensorSpec]   # {name: spec}
    output: TensorSpec
    description: str
    properties: list[str]
    fusable_epilogues: list[str]     # ops that can fuse as epilogue
```

Current P0 catalog (13 ops): `matmul`, `batch_matmul`, `softmax`, `layernorm`, `rmsnorm`, `relu`, `gelu`, `silu`, `add`, `mul`, `reduce_sum`, `reduce_max`, `transpose`.

---

## 5. IR Pipeline

```
.ak file
  │  parse_file() → arke.parser.parser
  ▼
Program (AST)
  │  ast_to_ir() → arke.parser.converter      [kernel → SemanticIR]
  │  program_to_strategy() → arke.compiler    [strategy → StrategyIR]
  │  DefaultStrategyGenerator (if no strategy block)
  ▼
SemanticIR + StrategyIR
  │  ArkeEnv.apply_decision() iteratively (LLM Agent)
  ▼
StrategyIR (finalized)
  │  TritonBackend.translate()
  ▼
Triton Python source
  │  TritonCompiler.compile() + .run()
  ▼
GPU execution result
```

---

## 6. Versioning

Both IR classes carry a `version` field:
- **SemanticIR**: `"0.3.0"` — stable since Phase 1 G1
- **StrategyIR**: `"0.2.0"` — stable since Phase 1 G2

Version increments when schema fields are added/changed. The `from_dict()` methods handle forward compatibility by ignoring unknown fields.
