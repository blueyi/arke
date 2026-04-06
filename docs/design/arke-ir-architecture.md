# Arke IR Multi-Layer Architecture

> **Version:** 1.0  
> **Status:** Design Spec — Stage 1 foundation, Stage 2-4 roadmap  
> **Owner:** IR Architecture Team  
> **Created:** 2026-04-06  
> **Applies to:** `arke/ir/` module, all IR-touching subsystems

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Multi-Layer Architecture Overview](#2-multi-layer-architecture-overview)
3. [Layer 4: Semantic IR Spec v1.0](#3-layer-4-semantic-ir-spec-v10)
4. [Layer 3: Strategy IR Spec](#4-layer-3-strategy-ir-spec)
5. [Layer 2: Hardware IR Spec](#5-layer-2-hardware-ir-spec)
6. [Layer 1: Instruction IR](#6-layer-1-instruction-ir)
7. [StrategyIR v1.0 Spec](#7-strategyir-v10-spec)
8. [Pass Infrastructure Design](#8-pass-infrastructure-design)
9. [JSON Schema & Compact Format](#9-json-schema--compact-format)
10. [MLIR Integration Design](#10-mlir-integration-design)
11. [Backward Compatibility](#11-backward-compatibility)
12. [Stage 1 Implementation Scope](#12-stage-1-implementation-scope)

---

## 1. Executive Summary

### 1.1 What Arke IR Is

Arke IR is the central intermediate representation of the Arke compiler toolchain — the backbone through which an AI accelerator kernel travels from high-level mathematical description down to hardware-specific binary.

**Core positioning:** Arke IR is an **LLM-Native, layered IR** with complete expressiveness for AI kernel optimization. It can lower to MLIR (leveraging its standard dialects such as `linalg`, `transform`, `scf`, `gpu`), and can also lower to LLVM IR directly when deeper hardware control is needed. If MLIR fully satisfies the optimization requirements, Arke IR serves as the LLM-friendly frontend to the MLIR ecosystem. Arke IR is designed for:

- **LLM Agents** as the primary decision-making consumer (Layer 4)
- **Structured optimization decisions** that LLMs can understand and manipulate (Layer 3)
- **Flexible lowering targets**: MLIR dialects (Stage 1-3) or direct LLVM IR (Stage 4)

### 1.2 Arke IR vs Traditional IRs

| Dimension | MLIR | Arke IR |
|-----------|------|---------|
| Primary author | Human compiler engineer | LLM Agent |
| Representation | Text/binary (C++ objects) | Python dataclasses + JSON |
| Extensibility | New dialect in C++ | New op in `ops/catalog.py` |
| LLM legibility | Poor (verbose, C++ centric) | First-class (JSON native) |
| LLVM IR path | Through lowering pipelines | Direct emit (Stage 4 goal) |
| Adoption barrier | High (MLIR expertise required) | Low (Python + JSON) |

Arke IR focuses on LLM-optimized ergonomics for AI kernel optimization workloads where:
- Control flow is structured (no arbitrary CFG needed at operator level)
- The decision-maker is an LLM, not a human compiler author
- JSON legibility enables agent introspection and learning

Arke IR and MLIR are **complementary**: Arke IR provides the LLM-native interface; MLIR provides battle-tested compiler infrastructure. Arke IR lowers to MLIR standard dialects to leverage existing optimization passes and hardware backends.

### 1.3 Stage Evolution

Arke IR grows incrementally, with progressively deeper MLIR integration:

| Stage | Arke IR Scope | MLIR Integration | Codegen Path |
|-------|-------------|------------------|--------------|
| **Stage 1** | Layer 4 + 3 (L1) | Framework + BL1 basic pathway | Via Triton |
| **Stage 2** | + L2 | Full capability (NVIDIA + Ascend) | Via Triton + MLIR |
| **Stage 3** | + L3 | Complete integration, deeper HW control | MLIR primary |
| **Stage 4** | Full stack | Available as optional target | Direct LLVM IR |

### 1.4 Design Philosophy

**SSA by construction.** SemanticIR (Layer 4) is a DAG where every Node output is written exactly once. SSA is not a constraint to enforce — it is a structural invariant of the representation itself.

**CFG lives downstream.** Arke IR uses structured conditional flow (`ConditionalNode`) at the operator level. Arbitrary control flow graphs appear only in MLIR lowering targets (e.g., `scf.if`, `scf.for`) or LLVM IR, where they belong.

**JSON is not the IR.** JSON is the serialization format and the LLM Agent's API surface. The IR lives as Python dataclass objects in memory. JSON serialization is lossless but JSON is not where passes operate.

**Semantic/Strategy separation.** `SemanticIR` (what to compute) and `StrategyIR` (how to optimize) are distinct objects. The LLM Agent explores `StrategyIR` decisions while `SemanticIR` remains immutable after construction. This separation is the core Arke architectural principle.

---

## 2. Multi-Layer Architecture Overview

### 2.1 Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 4: Semantic IR                         │
│                                                                 │
│  Operator-level DAG. "What to compute."                        │
│  Primary LLM Agent interface. JSON-first.                      │
│  SymbolicDim, ConditionalNode, MultiOutputNode                 │
│  ─────────────────────────────────────────────────────────     │
│  Python: arke/ir/semantic.py   JSON: SemanticIR v1.0           │
│  Stage 1: IMPLEMENTED                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ Lowering Pass: SemanticToStrategy
                         │ (StrategyIR L1: Stage 1, L2: Stage 2)
┌────────────────────────▼────────────────────────────────────────┐
│                    Layer 3: Strategy IR                         │
│                                                                 │
│  Optimization decisions. "How to optimize."                    │
│  L1: tile/fuse/vectorize/place (operator-level, LLM-driven)   │
│  L2: loop nests + memory hierarchy (ForNode, LoadTile, MAC)   │
│  L3: hardware mapping (thread/block/warp/vector assignment)    │
│  ─────────────────────────────────────────────────────────     │
│  Python: arke/ir/strategy.py   JSON: StrategyIR v1.0          │
│  Stage 1: L1 IMPLEMENTED, L2/L3: SPEC ONLY                    │
└────────────────────────┬────────────────────────────────────────┘
                         │ Lowering Pass: StrategyToHardware
                         │ (Stage 3)
┌────────────────────────▼────────────────────────────────────────┐
│                    Layer 2: Hardware IR                         │
│                                                                 │
│  Hardware mapping. "Thread blocks, warps, shared memory."      │
│  LaunchKernel, ThreadBlock, SharedAlloc, Barrier               │
│  GlobalLoad, SharedLoad, WarpReduce                            │
│  ─────────────────────────────────────────────────────────     │
│  Python: arke/ir/hardware.py   JSON: debug dump only           │
│  Stage 1: SPEC ONLY, Stage 3: IMPLEMENTED                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ Lowering Pass: HardwareToInstruction
                         │ (Stage 4)
┌────────────────────────▼────────────────────────────────────────┐
│                    Layer 1: Instruction IR                      │
│                                                                 │
│  Near-LLVM IR abstraction. "Instructions and values."          │
│  Fully auto-generated. LLM does not participate.               │
│  Direct LLVM IR emission interface.                            │
│  ─────────────────────────────────────────────────────────     │
│  Python: arke/ir/instruction.py  JSON: none                    │
│  Stage 1: INTERFACE ONLY, Stage 4: IMPLEMENTED                 │
└─────────────────────────────────────────────────────────────────┘
```

#### MLIR Lowering Targets

Arke IR lowers to MLIR standard dialects. Each layer maps to specific dialect families:

```
Arke IR Layer              MLIR Standard Dialects          Specialized Targets (examples)
─────────────────────────────────────────────────────────────────────────────────────────
Layer 4: SemanticIR   ──►  linalg / tensor                  (correctness verification)
Layer 3: StrategyIR   ──►  transform / scf / affine          │
         L1 decisions ──►  transform.structured.*             │
         L2 decisions ──►  scf.for / scf.forall / memref      ├──►  Triton TTIR
         L3 decisions ──►  gpu.launch / gpu.thread_id         │     Triton TTGPUIR
Layer 2: HardwareIR   ──►  gpu / nvvm / rocdl                 │     CUDA Tile IR
Layer 1: InstructionIR──►  llvm (LLVM dialect)               (direct LLVM IR)
─────────────────────────────────────────────────────────────────────────────────────────
```

> **Note:** The "Specialized Targets" column shows concrete MLIR-based compilation
> targets that Arke may lower to. They are all MLIR dialects internally.
> See §10.4 for detailed mapping examples.

### 2.2 StrategyIR as Layer 3

`StrategyIR` **is** Layer 3 in the lowering chain. It represents the optimization decisions that the LLM Agent makes — from high-level tiling/fusion (L1) through loop nest structure (L2) to hardware mapping (L3):

```
SemanticIR (Layer 4)  ←── "What to compute" (immutable after construction)
       │
       │ SemanticToStrategy pass
       │ (LLM Agent drives decisions)
       ▼
StrategyIR (Layer 3)  ←── "How to optimize" (LLM explores this)
  L1: tile / fuse / vectorize / place
  L2: loop nests + memory hierarchy
  L3: thread / block / warp mapping
       │
       ▼
HardwareIR (Layer 2) → InstructionIR (Layer 1)
```

### 2.3 JSON Roles by Layer

| Layer | JSON Role | Used For |
|-------|-----------|----------|
| **Layer 4** | **Primary representation** | Agent API, serialization, debug, caching |
| **Layer 3** | **Primary** (L1 decisions) / Optional (L2/L3) | LLM Agent API + debug inspection |
| **Layer 2** | Optional dump | Debug only (hardware mapping details) |
| **Layer 1** | **None** | Directly emit LLVM IR; no JSON intermediary |

### 2.4 Lowering Pipeline Overview

```python
class LoweringPipeline:
    """Full IR lowering chain: Layer 4 → Layer 1."""

    def lower_to_strategy(
        self, sem: SemanticIR, strategy: StrategyIR
    ) -> StrategyIR: ...  # Stage 1 (L1), Stage 2 (L2)

    def lower_to_hardware(
        self, strategy: StrategyIR, target: HWTarget
    ) -> HardwareIR: ...  # Stage 3

    def lower_to_instruction(
        self, hw: HardwareIR
    ) -> InstructionIR: ...  # Stage 4

    def emit_llvm(
        self, instr: InstructionIR
    ) -> str: ...  # Stage 4, returns LLVM IR text
```

---

## 3. Layer 4: Semantic IR Spec v1.0

### 3.1 Overview

SemanticIR v1.0 upgrades the existing v0.2.0 with:
- **SymbolicDim**: shape dimensions that are runtime variables (e.g., `seq_len`)
- **ShapeConstraint**: algebraic constraints between symbolic dims
- **ConditionalNode**: structured conditional computation (shape-regime branching)
- **MultiOutputNode**: nodes producing multiple tensors
- **Enhanced InputRef**: carries type information for reference-time type checking
- **Node.attrs**: op-specific attributes (e.g., `eps`, `axis`, `k`, `groups`)

### 3.2 Complete Python Schema

```python
# arke/ir/semantic.py — SemanticIR v1.0
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Union


# ─── Scalar Types ──────────────────────────────────────────────────────────

VALID_DTYPES = frozenset({
    "f16", "bf16", "f32", "f64",
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "bool",
})


# ─── Symbolic Dimensions ───────────────────────────────────────────────────

@dataclass
class SymbolicDim:
    """A named symbolic dimension (runtime variable).

    Examples:
        SymbolicDim("B")         # batch size
        SymbolicDim("S")         # sequence length
        SymbolicDim("H", min=1, max=128)  # bounded head count
    """
    name: str
    min: int | None = None   # optional lower bound (for compiler hints)
    max: int | None = None   # optional upper bound (for compiler hints)

    def to_dict(self) -> dict:
        d: dict = {"sym": self.name}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SymbolicDim:
        return cls(name=d["sym"], min=d.get("min"), max=d.get("max"))


# A dimension can be a concrete int or a symbolic variable
Dim = Union[int, SymbolicDim]


def dim_to_json(d: Dim) -> int | dict:
    """Serialize a Dim to JSON-compatible form."""
    if isinstance(d, int):
        return d
    return d.to_dict()


def dim_from_json(v: int | dict) -> Dim:
    """Deserialize a Dim from JSON."""
    if isinstance(v, int):
        return v
    return SymbolicDim.from_dict(v)


@dataclass
class ShapeConstraint:
    """An algebraic constraint between symbolic dims.

    Examples:
        ShapeConstraint("S % 128 == 0", "softmax tile alignment")
        ShapeConstraint("H * D == model_dim", "attention head consistency")
    """
    expr: str    # Python-evaluable expression using SymbolicDim names
    reason: str = ""

    def to_dict(self) -> dict:
        d: dict = {"expr": self.expr}
        if self.reason:
            d["reason"] = self.reason
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ShapeConstraint:
        return cls(expr=d["expr"], reason=d.get("reason", ""))


# ─── Tensor Descriptor ─────────────────────────────────────────────────────

@dataclass
class TensorDesc:
    """Describes a tensor: shape (may be symbolic), dtype, layout."""
    shape: list[Dim]
    dtype: str
    layout: str = "row_major"

    def to_dict(self) -> dict:
        d: dict = {
            "shape": [dim_to_json(d) for d in self.shape],
            "dtype": self.dtype,
        }
        if self.layout != "row_major":
            d["layout"] = self.layout
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TensorDesc:
        return cls(
            shape=[dim_from_json(v) for v in d["shape"]],
            dtype=d["dtype"],
            layout=d.get("layout", "row_major"),
        )

    def is_symbolic(self) -> bool:
        return any(isinstance(d, SymbolicDim) for d in self.shape)


# ─── Parameters ────────────────────────────────────────────────────────────

@dataclass
class Param:
    """A kernel input parameter (named tensor)."""
    name: str
    shape: list[Dim]
    dtype: str
    layout: str = "row_major"

    def to_tensor_desc(self) -> TensorDesc:
        return TensorDesc(shape=self.shape, dtype=self.dtype, layout=self.layout)

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "shape": [dim_to_json(d) for d in self.shape],
            "dtype": self.dtype,
        }
        if self.layout != "row_major":
            d["layout"] = self.layout
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Param:
        return cls(
            name=d["name"],
            shape=[dim_from_json(v) for v in d["shape"]],
            dtype=d["dtype"],
            layout=d.get("layout", "row_major"),
        )


# ─── Input References (Enhanced with Type Info) ────────────────────────────

@dataclass
class ParamRef:
    """Reference to a kernel parameter, with resolved type info."""
    name: str
    dtype: str | None = None    # resolved at IR construction time
    shape: list[Dim] | None = None  # resolved at IR construction time

    def to_dict(self) -> dict:
        d: dict = {"ref": "param", "name": self.name}
        # dtype/shape are derived — not serialized (avoid redundancy)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ParamRef:
        return cls(name=d["name"])


@dataclass
class NodeRef:
    """Reference to a previous node's output, with resolved type info."""
    id: str
    dtype: str | None = None    # resolved at IR construction time
    shape: list[Dim] | None = None  # resolved at IR construction time

    def to_dict(self) -> dict:
        d: dict = {"ref": "node", "id": self.id}
        # dtype/shape are derived — not serialized (avoid redundancy)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NodeRef:
        return cls(id=d["id"])


InputRef = Union[ParamRef, NodeRef]


def input_ref_from_dict(d: dict | str) -> InputRef:
    """Parse InputRef from dict or legacy string format."""
    if isinstance(d, str):
        # Legacy v0.2.0: "@node_id" or "param_name"
        if d.startswith("@"):
            return NodeRef(id=d[1:])
        return ParamRef(name=d)
    ref_kind = d.get("ref")
    if ref_kind == "param":
        return ParamRef.from_dict(d)
    if ref_kind == "node":
        return NodeRef.from_dict(d)
    raise ValueError(f"Invalid input reference: {d}")


# ─── Semantics ─────────────────────────────────────────────────────────────

@dataclass
class Semantics:
    """Mathematical description of an operator."""
    computation: str              # e.g., "C[i,j] = sum(A[i,k]*B[k,j], k)"
    index_vars: list[str] = field(default_factory=list)
    reduction_axes: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    # properties examples: "associative", "commutative", "elementwise",
    #                       "monotonic", "idempotent"


# ─── Core Node Types ───────────────────────────────────────────────────────

@dataclass
class Node:
    """A single operator node in the SemanticIR DAG.

    v1.0 additions:
    - attrs: op-specific attributes (eps, axis, k, groups, etc.)
    - output is now a TensorDesc with Dim (may be symbolic)
    - inputs have type info via enhanced InputRef
    """
    id: str
    op: str
    inputs: dict[str, InputRef]
    output: TensorDesc
    semantics: Semantics
    attrs: dict[str, Any] = field(default_factory=dict)
    # attrs examples:
    #   layernorm: {"eps": 1e-5, "normalized_dims": [-1]}
    #   softmax:   {"axis": -1, "temperature": 1.0}
    #   topk:      {"k": 10, "dim": -1, "largest": True}
    #   conv2d:    {"stride": [1,1], "padding": [0,0], "groups": 1}

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "op": self.op,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "output": self.output.to_dict(),
            "semantics": {
                "computation": self.semantics.computation,
            },
        }
        if self.semantics.index_vars:
            d["semantics"]["index_vars"] = self.semantics.index_vars
        if self.semantics.reduction_axes:
            d["semantics"]["reduction_axes"] = self.semantics.reduction_axes
        if self.semantics.properties:
            d["semantics"]["properties"] = self.semantics.properties
        if self.attrs:
            d["attrs"] = self.attrs
        return d


@dataclass
class MultiOutputNode:
    """A node that produces multiple named output tensors.

    Examples: topk (values + indices), split (multiple chunks),
              qkv_proj (Q, K, V projections fused).

    In the DAG, downstream nodes reference outputs via:
        NodeRef(id="split_0", port="chunk_0")
    """
    id: str
    op: str
    inputs: dict[str, InputRef]
    outputs: dict[str, TensorDesc]   # named output ports
    semantics: Semantics
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op": self.op,
            "multi_output": True,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "semantics": {
                "computation": self.semantics.computation,
                "index_vars": self.semantics.index_vars,
                "reduction_axes": self.semantics.reduction_axes,
                "properties": self.semantics.properties,
            },
            "attrs": self.attrs,
        }


@dataclass
class ConditionalNode:
    """Structured conditional computation (shape-regime branching).

    Selects between two sub-DAGs based on a predicate over symbolic dims.
    This is NOT arbitrary CFG — both branches must produce the same output type.

    Example use case:
        if S <= 512:
            use_flash_attention_short(Q, K, V)
        else:
            use_flash_attention_long(Q, K, V)

    Note: The predicate is evaluated at runtime based on symbolic dim values.
    The compiler may specialize for known regimes.
    """
    id: str
    predicate: str           # e.g., 'dim("S") <= 512'
    true_branch: list[str]   # node ids to execute when predicate is True
    false_branch: list[str]  # node ids to execute when predicate is False
    output: TensorDesc       # both branches must produce this type

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op": "__conditional__",
            "predicate": self.predicate,
            "true_branch": self.true_branch,
            "false_branch": self.false_branch,
            "output": self.output.to_dict(),
        }


# AnyNode: a node in the SemanticIR DAG
AnyNode = Union[Node, MultiOutputNode, ConditionalNode]


# ─── Edges & Fusion ────────────────────────────────────────────────────────

@dataclass
class Edge:
    """Data flow edge between nodes (or param → node)."""
    from_node: str      # source node id (or "param:<name>" for param inputs)
    to_node: str        # destination node id
    tensor_name: str    # logical name of the tensor on this edge
    from_port: str = "output"   # output port name (for MultiOutputNode)
    lifetime: str = "local"     # "local" | "persistent"

    def to_dict(self) -> dict:
        d: dict = {
            "from": self.from_node,
            "to": self.to_node,
            "tensor": self.tensor_name,
        }
        if self.from_port != "output":
            d["from_port"] = self.from_port
        if self.lifetime != "local":
            d["lifetime"] = self.lifetime
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(
            from_node=d.get("from_node") or d.get("from", ""),
            to_node=d.get("to_node") or d.get("to", ""),
            tensor_name=d.get("tensor_name") or d.get("tensor", ""),
            from_port=d.get("from_port", "output"),
            lifetime=d.get("lifetime", "local"),
        )


@dataclass
class FusionGroup:
    """Hint that a set of nodes should be fused in codegen."""
    id: str
    nodes: list[str]
    fusion_type: str   # "epilogue" | "prologue" | "horizontal" | "vertical"
    reason: str = ""

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "nodes": self.nodes, "type": self.fusion_type}
        if self.reason:
            d["reason"] = self.reason
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FusionGroup:
        return cls(
            id=d["id"],
            nodes=d["nodes"],
            fusion_type=d.get("fusion_type") or d.get("type", "epilogue"),
            reason=d.get("reason", ""),
        )


# ─── Top-Level SemanticIR v1.0 ─────────────────────────────────────────────

@dataclass
class SemanticIR:
    """Layer 4 of Arke IR — the Semantic IR (v1.0).

    Describes computation at operator level. Immutable after construction.
    The LLM Agent reads this; StrategyIR is what the LLM writes.

    v1.0 changes from v0.2.0:
    - version bumped to "1.0.0"
    - params.shape supports SymbolicDim
    - nodes supports MultiOutputNode and ConditionalNode
    - nodes (Node type) gains .attrs field
    - InputRef gains .dtype and .shape (resolved at construction)
    - edges use compact dict keys ("from", "to", "tensor")
    - shape_constraints added
    - symbolic_dims registry added
    - return_type replaced by return_node + return_ports
    """

    version: str = "1.0.0"
    kernel_id: str = ""
    params: list[Param] = field(default_factory=list)
    symbolic_dims: list[SymbolicDim] = field(default_factory=list)
    shape_constraints: list[ShapeConstraint] = field(default_factory=list)
    nodes: list[AnyNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    return_node: str = ""
    return_ports: list[str] = field(default_factory=list)  # for MultiOutputNode returns
    fusion_groups: list[FusionGroup] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ─── Mutation (construction only) ────────────────────────────────────

    def add_param(self, param: Param) -> None:
        self.params.append(param)

    def add_node(self, node: AnyNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_fusion_group(self, group: FusionGroup) -> None:
        self.fusion_groups.append(group)

    def add_symbolic_dim(self, dim: SymbolicDim) -> None:
        self.symbolic_dims.append(dim)

    def add_shape_constraint(self, constraint: ShapeConstraint) -> None:
        self.shape_constraints.append(constraint)

    # ─── Lookup ──────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> AnyNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_param(self, name: str) -> Param | None:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def is_symbolic(self) -> bool:
        return len(self.symbolic_dims) > 0

    # ─── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "kernel_id": self.kernel_id,
            "params": [p.to_dict() for p in self.params],
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "return_node": self.return_node,
        }
        # Omit defaults for compact output
        if self.symbolic_dims:
            d["symbolic_dims"] = [sd.to_dict() for sd in self.symbolic_dims]
        if self.shape_constraints:
            d["shape_constraints"] = [sc.to_dict() for sc in self.shape_constraints]
        if self.return_ports:
            d["return_ports"] = self.return_ports
        if self.fusion_groups:
            d["fusion_groups"] = [fg.to_dict() for fg in self.fusion_groups]
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> SemanticIR:
        """Deserialize — handles both v0.2.0 and v1.0 JSON."""
        ir = cls(
            version=data.get("version", "0.2.0"),
            kernel_id=data.get("kernel_id", ""),
            return_node=data.get("return_node", ""),
            return_ports=data.get("return_ports", []),
        )

        # Symbolic dims (v1.0 only)
        for sd in data.get("symbolic_dims", []):
            ir.add_symbolic_dim(SymbolicDim.from_dict(sd))

        # Shape constraints (v1.0 only)
        for sc in data.get("shape_constraints", []):
            ir.add_shape_constraint(ShapeConstraint.from_dict(sc))

        # Params
        for p in data.get("params", []):
            ir.add_param(Param(
                name=p["name"],
                shape=[dim_from_json(v) for v in p["shape"]],
                dtype=p["dtype"],
                layout=p.get("layout", "row_major"),
            ))

        # Nodes (dispatch on type)
        for nd in data.get("nodes", []):
            if nd.get("op") == "__conditional__":
                ir.add_node(ConditionalNode(
                    id=nd["id"],
                    predicate=nd["predicate"],
                    true_branch=nd["true_branch"],
                    false_branch=nd["false_branch"],
                    output=TensorDesc.from_dict(nd["output"]),
                ))
            elif nd.get("multi_output"):
                outputs = {k: TensorDesc.from_dict(v) for k, v in nd["outputs"].items()}
                inputs = {k: input_ref_from_dict(v) for k, v in nd.get("inputs", {}).items()}
                sem = nd.get("semantics", {})
                ir.add_node(MultiOutputNode(
                    id=nd["id"],
                    op=nd["op"],
                    inputs=inputs,
                    outputs=outputs,
                    semantics=Semantics(
                        computation=sem.get("computation", ""),
                        index_vars=sem.get("index_vars", []),
                        reduction_axes=sem.get("reduction_axes", []),
                        properties=sem.get("properties", []),
                    ),
                    attrs=nd.get("attrs", {}),
                ))
            else:
                inputs = {k: input_ref_from_dict(v) for k, v in nd.get("inputs", {}).items()}
                sem = nd.get("semantics", {})
                ir.add_node(Node(
                    id=nd["id"],
                    op=nd["op"],
                    inputs=inputs,
                    output=TensorDesc.from_dict(nd["output"]),
                    semantics=Semantics(
                        computation=sem.get("computation", ""),
                        index_vars=sem.get("index_vars", []),
                        reduction_axes=sem.get("reduction_axes", []),
                        properties=sem.get("properties", []),
                    ),
                    attrs=nd.get("attrs", {}),
                ))

        # Edges
        for e in data.get("edges", []):
            ir.add_edge(Edge.from_dict(e))

        # Fusion groups
        for fg in data.get("fusion_groups", []):
            ir.add_fusion_group(FusionGroup.from_dict(fg))

        # Metadata
        ir.metadata = data.get("metadata", {})

        return ir

    @classmethod
    def from_json(cls, json_str: str) -> SemanticIR:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> SemanticIR:
        with open(path) as f:
            return cls.from_json(f.read())


# Backward compatibility alias
SemanticGraph = SemanticIR
```

### 3.3 SSA Rules

SemanticIR is **SSA by construction**:

1. Every `Node.id` is unique within a `SemanticIR`.
2. Every `Node.output` is a fresh tensor (written exactly once).
3. `InputRef` always points backward in topological order (no cycles except through `ConditionalNode` branches which are self-contained).
4. `MultiOutputNode` outputs are identified by `(node_id, port_name)` — still written once.
5. `ConditionalNode` branches are separate sub-DAGs; each is independently SSA.

**SSA Verifier interface:**
```python
class SSAVerifier:
    """Validate SemanticIR SSA invariants."""

    def verify(self, ir: SemanticIR) -> list[str]:
        """Returns list of violation messages (empty = valid)."""
        errors: list[str] = []
        errors.extend(self._check_unique_ids(ir))
        errors.extend(self._check_no_cycles(ir))
        errors.extend(self._check_ref_resolution(ir))
        errors.extend(self._check_type_consistency(ir))
        return errors

    def _check_unique_ids(self, ir: SemanticIR) -> list[str]: ...
    def _check_no_cycles(self, ir: SemanticIR) -> list[str]: ...
    def _check_ref_resolution(self, ir: SemanticIR) -> list[str]: ...
    def _check_type_consistency(self, ir: SemanticIR) -> list[str]: ...
```

### 3.4 JSON Example: Softmax with Symbolic Sequence Length

```json
{
  "version": "1.0.0",
  "kernel_id": "softmax_dynamic",
  "symbolic_dims": [
    {"sym": "S", "min": 1, "max": 65536}
  ],
  "shape_constraints": [
    {"expr": "S % 64 == 0", "reason": "tile alignment"}
  ],
  "params": [
    {"name": "X", "shape": [8, {"sym": "S"}, 512], "dtype": "f16"}
  ],
  "nodes": [
    {
      "id": "softmax_0",
      "op": "softmax",
      "inputs": {"X": {"ref": "param", "name": "X"}},
      "output": {"shape": [8, {"sym": "S"}, 512], "dtype": "f16"},
      "semantics": {
        "computation": "Y[b,s,d] = exp(X[b,s,d]) / sum(exp(X[b,s,:]), axis=-1)",
        "reduction_axes": ["d"],
        "properties": ["row_independent"]
      },
      "attrs": {"axis": -1}
    }
  ],
  "edges": [],
  "return_node": "softmax_0"
}
```

### 3.5 JSON Example: ConditionalNode (Shape Regime)

```json
{
  "version": "1.0.0",
  "kernel_id": "attn_dispatch",
  "symbolic_dims": [{"sym": "S"}],
  "params": [
    {"name": "Q", "shape": [8, {"sym": "S"}, 64], "dtype": "f16"},
    {"name": "K", "shape": [8, {"sym": "S"}, 64], "dtype": "f16"},
    {"name": "V", "shape": [8, {"sym": "S"}, 64], "dtype": "f16"}
  ],
  "nodes": [
    {"id": "attn_short", "op": "flash_attention",
     "inputs": {"Q": {"ref": "param", "name": "Q"}, "K": {"ref": "param", "name": "K"}, "V": {"ref": "param", "name": "V"}},
     "output": {"shape": [8, {"sym": "S"}, 64], "dtype": "f16"},
     "semantics": {"computation": "flash attention (short)"},
     "attrs": {"causal": false, "regime": "short"}
    },
    {"id": "attn_long", "op": "flash_attention",
     "inputs": {"Q": {"ref": "param", "name": "Q"}, "K": {"ref": "param", "name": "K"}, "V": {"ref": "param", "name": "V"}},
     "output": {"shape": [8, {"sym": "S"}, 64], "dtype": "f16"},
     "semantics": {"computation": "flash attention (long)"},
     "attrs": {"causal": false, "regime": "long"}
    },
    {
      "id": "dispatch_0",
      "op": "__conditional__",
      "predicate": "dim(\"S\") <= 512",
      "true_branch": ["attn_short"],
      "false_branch": ["attn_long"],
      "output": {"shape": [8, {"sym": "S"}, 64], "dtype": "f16"}
    }
  ],
  "edges": [],
  "return_node": "dispatch_0"
}
```

---

## 4. Layer 3: Strategy IR Spec

> **Stage 1 status: L1 IMPLEMENTED, L2/L3 SPEC ONLY.**  
> L2 implementation target: Stage 2. L3 implementation target: Stage 3.

### 4.1 Purpose

Layer 3 is the **Strategy IR** — it captures the full spectrum of optimization decisions that transform operator semantics (Layer 4) into hardware-executable code. StrategyIR is where the LLM Agent operates: exploring tiling, fusion, memory placement, loop nest structure, and hardware mapping.

StrategyIR has three levels of depth:

| Level | Scope | LLM Role | Stage |
|-------|-------|----------|-------|
| **L1** | Operator-level decisions: tile, fuse, vectorize, place, launch_config | Primary decision maker | Stage 1 |
| **L2** | Loop nest + memory hierarchy: explicit ForNode, LoadTile, MAC, memory tiers | Guided exploration | Stage 2-3 |
| **L3** | Hardware mapping: thread/block/warp assignment, barrier, register allocation | Expert-level (optional) | Stage 3-4 |

### 4.2 Relationship to SemanticIR

```
SemanticIR (Layer 4) + StrategyIR L1 decisions
              │
              ▼
     SemanticToStrategy pass (L1 → L2 expansion)
              │
      tile(M=64) → ForNode("i_outer", range=M//64)
      tile(K=32) → ForNode("k_inner", range=K//32)
      place(A, shared) → LoadTile(A, dst=shared_A)
      parallel(i→block.x) → ForNode marked grid_dim=blockIdx.x
              │
              ▼
       StrategyIR L2 (explicit loop nests + memory)
```

### 4.3 L2 Compute Schema (Python)

The following schema defines the L2 compute structures within StrategyIR — the explicit loop nests, memory hierarchy, and compute primitives that emerge from expanding L1 decisions.

```python
# arke/ir/strategy_compute.py — StrategyIR L2 structures (Stage 1: spec only)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union


# ─── Memory Hierarchy ──────────────────────────────────────────────────────

class MemorySpace:
    GLOBAL = "global"       # HBM / DRAM — off-chip
    SHARED = "shared"       # GPU shared memory (SMEM) / L1 scratchpad
    REGISTER = "register"   # Register file — per-thread
    L2 = "l2"               # L2 cache (hint only, not explicitly managed)


@dataclass
 class MemoryBuffer:
    """A named buffer at a specific memory tier."""
    name: str
    dtype: str
    shape: list[int | str]     # int or loop variable name
    space: str                 # MemorySpace constant
    align: int = 128           # byte alignment


# ─── Compute Nodes ─────────────────────────────────────────────────────────

@dataclass
class ForNode:
    """A loop over a range, potentially mapped to a hardware dimension."""
    var: str               # loop variable name, e.g. "i", "k_inner"
    start: int | str       # int literal or outer loop var
    stop: int | str        # int literal or outer loop var
    step: int = 1
    body: list[ComputeNode] = field(default_factory=list)
    # Hardware mapping (set by Strategy → Compute lowering)
    grid_dim: str | None = None     # e.g. "blockIdx.x" → maps to GPU block
    thread_dim: str | None = None   # e.g. "threadIdx.x" → maps to GPU thread
    unroll: int | None = None       # unroll factor (None = no unroll)
    vectorize: int | None = None    # SIMD width (None = scalar)


@dataclass
class IfNode:
    """Structured conditional within a compute body."""
    predicate: str            # expression over loop vars or symbolic dims
    true_body: list[ComputeNode] = field(default_factory=list)
    false_body: list[ComputeNode] = field(default_factory=list)


@dataclass
class LoadTile:
    """Load a tile from one memory space into another."""
    src: str          # source buffer name (global or higher tier)
    dst: str          # destination buffer name (shared or register)
    src_indices: list[str]     # index expressions
    dst_indices: list[str]
    mask: str | None = None    # optional mask expression for out-of-bounds
    prefetch: bool = False     # software prefetch hint


@dataclass
class StoreTile:
    """Store a tile from one memory space to another."""
    src: str
    dst: str
    src_indices: list[str]
    dst_indices: list[str]
    mask: str | None = None


@dataclass
class MAC:
    """Multiply-accumulate: acc += A[...] * B[...].

    Represents tensor core or SIMD multiply-accumulate.
    """
    acc: str              # accumulator buffer name
    a: str                # left operand buffer
    b: str                # right operand buffer
    a_indices: list[str]
    b_indices: list[str]
    acc_indices: list[str]
    use_tensor_core: bool = False  # hint to use wmma/mma instructions


@dataclass
class ReduceNode:
    """Reduction over a dimension."""
    src: str
    dst: str
    reduce_op: str    # "sum" | "max" | "min" | "prod"
    axis: int
    src_indices: list[str]
    dst_indices: list[str]


@dataclass
class ComputeOp:
    """An elementwise compute operation."""
    dst: str
    expr: str           # e.g. "max(x, 0.0)" for relu, "x * sigmoid(x)" for silu
    dst_indices: list[str]


ComputeNode = Union[ForNode, IfNode, LoadTile, StoreTile, MAC, ReduceNode, ComputeOp]


# ─── Top-Level StrategyIR L2 Container ─────────────────────────────────────

@dataclass
class StrategyComputeIR:
    """StrategyIR Level 2 — explicit loop nests + memory hierarchy.


    Generated from SemanticIR + StrategyIR L1 by SemanticToStrategy L2 pass.
    """
    version: str = "1.0.0"
    kernel_id: str = ""
    target_hw: str = ""
    buffers: list[MemoryBuffer] = field(default_factory=list)
    body: list[ComputeNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.4 MLIR Mapping (L2 Structures)

| StrategyIR L2 Node | MLIR Standard Dialect | Notes |
|--------------------|----------------------|-------|
| `ForNode` (sequential) | `scf.for` | Loop variable and range map directly |
| `ForNode` (grid_dim set) | `scf.forall` / `gpu.launch` | Parallel loop → GPU dimension |
| `IfNode` | `scf.if` | Structured conditional |
| `LoadTile` | `memref.copy` + `affine.load` | Tiled load with affine maps |
| `StoreTile` | `affine.store` + `memref.copy` | Tiled store |
| `MAC` | `linalg.matmul` (inner tile) | Tensor core variant via target-specific dialect |
| `ReduceNode` | `linalg.reduce` | With appropriate combiner region |
| `MemoryBuffer` (shared) | `memref` with `#gpu.address_space<workgroup>` | Shared SMEM buffer |
| `MemoryBuffer` (register) | `memref` with `#gpu.address_space<private>` | Register buffer |

---

## 5. Layer 2: Hardware IR Spec

> **Stage 1 status: SPEC ONLY — not implemented.**  
> Implementation target: Stage 3.

### 5.1 Purpose

Layer 2 makes GPU execution explicit: thread blocks, warp organization, shared memory allocation, synchronization barriers, and explicit global/shared memory loads. At this layer, the IR is isomorphic to PTX-level abstractions.

### 5.2 Interface Definition

```python
# arke/ir/hardware.py — HardwareIR v1.0 (Stage 1: interface only)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class LaunchKernel:
    """Top-level GPU kernel launch configuration."""
    kernel_id: str
    grid_dim: tuple[int | str, int | str, int | str]   # (x, y, z) blocks
    block_dim: tuple[int | str, int | str, int | str]  # (x, y, z) threads
    shared_mem_bytes: int
    body: list[HWNode] = field(default_factory=list)


@dataclass
class ThreadBlock:
    """Logical thread block region with a defined set of operations."""
    block_id_x: str = "blockIdx.x"
    block_id_y: str = "blockIdx.y"
    block_id_z: str = "blockIdx.z"
    body: list[HWNode] = field(default_factory=list)


@dataclass
class SharedAlloc:
    """Allocate a buffer in GPU shared memory (SMEM)."""
    name: str
    dtype: str
    shape: list[int]
    align: int = 128   # byte alignment
    num_stages: int = 1  # double-buffer → 2


@dataclass
class Barrier:
    """Synchronization barrier."""
    kind: str = "block"   # "block" = __syncthreads(), "warp" = warp barrier


@dataclass
class GlobalLoad:
    """Load from global memory (HBM) into register or shared."""
    src_ptr: str
    dst: str
    indices: list[str]
    dtype: str
    vector_width: int = 1   # vectorized load (e.g., 4 = float4)
    mask: str | None = None
    cache_hint: str = "ca"  # "ca" = cache all, "cg" = cache global


@dataclass
class SharedLoad:
    """Load from shared memory into register."""
    src: str
    dst: str
    indices: list[str]
    dtype: str
    vector_width: int = 1


@dataclass
class WarpReduce:
    """Warp-level reduction using shuffle instructions."""
    src: str
    dst: str
    reduce_op: str    # "sum" | "max" | "min"
    dtype: str


@dataclass
class MMAInstruction:
    """Warp-level matrix multiply-accumulate (tensor core)."""
    a: str; b: str; c: str; d: str   # d = a*b + c
    shape: tuple[int, int, int]      # e.g., (16, 8, 16) for m16n8k16
    dtype_ab: str                    # input dtype
    dtype_cd: str                    # accumulator dtype


HWNode = Union[
    LaunchKernel, ThreadBlock, SharedAlloc, Barrier,
    GlobalLoad, SharedLoad, WarpReduce, MMAInstruction
]


@dataclass
class HardwareIR:
    """Layer 2 of Arke IR — the Hardware IR.

    Maps StrategyIR L2/L3 to concrete GPU execution model.
    Generated from StrategyIR by StrategyToHardwarePass.
    """
    version: str = "1.0.0"
    kernel_id: str = ""
    target: str = ""   # e.g., "nvidia_ampere_sm86"
    launch: LaunchKernel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.3 MLIR Mapping

| HardwareIR Node | MLIR Equivalent |
|-----------------|------------------|
| `LaunchKernel` | `gpu.launch_func` + `gpu.module` |
| `ThreadBlock` | `gpu.launch` body with `blockIdx.*` |
| `SharedAlloc` | `memref.alloc` with `#gpu.address_space<workgroup>` |
| `Barrier` (block) | `gpu.barrier` |
| `Barrier` (warp) | `nvvm.bar.warp.sync` |
| `GlobalLoad` | `gpu.load` / `nvvm.ld.global` |
| `SharedLoad` | `gpu.load` with workgroup address space |
| `WarpReduce` | `gpu.shuffle` + reduce combine |
| `MMAInstruction` | `nvgpu.warpgroup_mma` (Ampere+) or `nvvm.wmma.*` |

---

## 6. Layer 1: Instruction IR

> **Stage 1 status: INTERFACE ONLY.**  
> Implementation target: Stage 4.

### 6.1 Purpose

Layer 1 is the final pre-LLVM IR abstraction. It is fully auto-generated — no LLM participation. Its purpose is to bridge HardwareIR to LLVM IR's SSA/basic-block model.

### 6.2 Interface Definition

```python
# arke/ir/instruction.py — InstructionIR v1.0 (Stage 1: interface only)
from __future__ import annotations
from typing import Protocol, Any


class InstructionIREmitter(Protocol):
    """Protocol for emitting InstructionIR from HardwareIR."""

    def emit_from_hardware(self, hw: HardwareIR) -> InstructionIR:
        """Lower HardwareIR to InstructionIR."""
        ...


class LLVMIREmitter(Protocol):
    """Protocol for emitting LLVM IR text from InstructionIR."""

    def emit_llvm_ir(self, instr: InstructionIR) -> str:
        """Returns LLVM IR module text (.ll format)."""
        ...

    def emit_ptx(self, instr: InstructionIR, sm: str) -> str:
        """Returns PTX assembly for given SM target (e.g., 'sm_86')."""
        ...


class InstructionIR:
    """Layer 1 of Arke IR — near-LLVM IR abstraction.

    Represents SSA values, basic blocks, and instructions
    at a level directly translatable to LLVM IR.

    This class is a placeholder — the full implementation is Stage 4.
    """
    version: str = "1.0.0"
    kernel_id: str = ""
    # Full schema TBD in Stage 4 design
    _raw: Any = None  # Stage 4: will hold LLVM-Python bindings object

    def to_llvm_ir(self) -> str:
        """Emit LLVM IR module as text."""
        raise NotImplementedError("Stage 4 only")

    def to_ptx(self, sm: str = "sm_86") -> str:
        """Emit PTX assembly for target SM."""
        raise NotImplementedError("Stage 4 only")
```

### 6.3 LLVM IR Bridge Design

At Stage 4, the LLVM IR bridge will use `llvmlite` or `llvm-project` Python bindings:

```
HardwareIR
    │
    ▼
HardwareToInstructionPass
    │  - Map each HWNode to LLVM IR instructions
    │  - Thread block → LLVM function with nvptx metadata
    │  - SharedAlloc → alloca in address space 3 (shared)
    │  - GlobalLoad → llvm.nvvm.ld.global.* intrinsics
    │  - MMAInstruction → llvm.nvvm.wmma.* intrinsics
    ▼
InstructionIR (LLVM module)
    │
    ▼
llvm.Target("nvptx64-nvidia-cuda").emit(module, "ptx")
    │
    ▼
PTX assembly → cubin via ptxas
```

---

## 7. StrategyIR v1.0 Spec

### 7.1 Design Goals

StrategyIR v1.0 addresses the following v0.2.0 deficiencies:

| Issue | v0.2.0 | v1.0 Fix |
|-------|--------|----------|
| Backend-specific decisions | `launch_config(num_warps=4)` | Abstracted to `compute_resource(warps=4)` with backend mapping |
| No conditional decisions | — | `ConditionalDecision` with shape predicates |
| No symbolic shape support | — | `ShapeRegime` for regime-specific strategies |
| No multi-output awareness | — | Decision params accept port names |
| Flat decision list | — | Strategy levels L1 (abstract) → L2 (concrete) |

### 7.2 Strategy Levels

```
L1: Backend-agnostic strategy
    tile, reorder, fuse, parallel, place, vectorize, unroll, algorithm
    ↓ Backend-Specialization Pass
L2: Backend-specific refinement
    compute_resource (maps to num_warps etc.), cache_config, pipeline_stages
    ↓ Code Generation
Triton / CUDA C / (Stage 4) LLVM IR
```

### 7.3 Complete Python Schema

```python
# arke/ir/strategy.py — StrategyIR v1.0
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Union


@dataclass
class Rationale:
    """Natural language explanation for an optimization decision."""
    text: str
    lang: str = "en"


# ─── Decision Types ────────────────────────────────────────────────────────

@dataclass
class Decision:
    """A single L1 (backend-agnostic) optimization decision.

    Kinds (L1):
        tile         - Tile a loop: {loop, factors}
        reorder      - Reorder loop nest: {order}
        fuse         - Fuse operators: {ops, type}
        parallel     - Map loops to hardware dims: {loops, mapping}
        place        - Tensor memory placement: {tensor, memory}
        vectorize    - Vectorize a loop: {loop, width}
        unroll       - Unroll a loop: {loop, factor}
        algorithm    - Algorithm variant: {name, params}

    Kinds (L2 — backend-specific, set by backend specialization pass):
        compute_resource  - {warps, stages, pipeline_depth}
        cache_config      - {l1_size, l2_hint}
        memory_fence      - {scope}
    """
    kind: str
    params: dict[str, Any]
    rationale: Rationale | None = None
    step: int = 0          # auto-assigned by StrategyIR
    level: int = 1         # 1 = L1 backend-agnostic, 2 = L2 backend-specific


@dataclass
class ConditionalDecision:
    """A decision that applies only when a shape predicate holds.

    Example:
        when dim("S") <= 512 { tile(K=32) } otherwise { tile(K=64) }
    """
    predicate: str
    true_decisions: list[Decision]
    false_decisions: list[Decision]
    rationale: Rationale | None = None
    step: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "kind": "__conditional__",
            "predicate": self.predicate,
            "true_decisions": [_decision_to_dict(x) for x in self.true_decisions],
            "false_decisions": [_decision_to_dict(x) for x in self.false_decisions],
            "step": self.step,
        }
        if self.rationale:
            d["rationale"] = {"text": self.rationale.text, "lang": self.rationale.lang}
        return d


AnyDecision = Union[Decision, ConditionalDecision]


def _decision_to_dict(d: AnyDecision) -> dict:
    if isinstance(d, ConditionalDecision):
        return d.to_dict()
    result: dict = {"kind": d.kind, "params": d.params, "step": d.step, "level": d.level}
    if d.rationale:
        result["rationale"] = {"text": d.rationale.text, "lang": d.rationale.lang}
    return result


def _parse_decision(d: dict) -> Decision:
    """Parse a single decision dict. Handles v0.2.0 launch_config compat."""
    rat = None
    if d.get("rationale"):
        rat_data = d["rationale"]
        rat = Rationale(
            text=rat_data if isinstance(rat_data, str) else rat_data.get("text", ""),
            lang=rat_data.get("lang", "en") if isinstance(rat_data, dict) else "en",
        )
    kind = d["kind"]
    params = dict(d["params"])
    level = d.get("level", 1)
    # v0.2.0 compat: map launch_config -> compute_resource (L2)
    if kind == "launch_config":
        kind = "compute_resource"
        level = 2
        new_params: dict = {}
        if "num_warps" in params or "warps" in params:
            new_params["warps"] = params.get("num_warps", params.get("warps"))
        if "num_stages" in params or "stages" in params:
            new_params["stages"] = params.get("num_stages", params.get("stages"))
        if "block_sizes" in params:
            new_params["block_sizes"] = params["block_sizes"]
        params = new_params
    return Decision(kind=kind, params=params, rationale=rat,
                    step=d.get("step", 0), level=level)


@dataclass
class ShapeRegime:
    """A named strategy regime for a specific shape range."""
    name: str
    predicate: str
    decisions: list[AnyDecision] = field(default_factory=list)


@dataclass
class HardwareConstraints:
    """Hardware resource constraints."""
    shared_memory_limit: int = 0
    register_limit: int = 0
    max_threads_per_block: int = 0
    warp_size: int = 32


@dataclass
class StrategyIR:
    """Optimization strategy IR v1.0.

    v1.0 changes from v0.2.0:
    - decisions list accepts AnyDecision (Decision | ConditionalDecision)
    - shape_regimes: named profiles for shape-based dispatch
    - level field on each Decision (L1 vs L2)
    - launch_config replaced by compute_resource (L2 decision)
    - Backward compatible: v0.2.0 JSON loads via from_dict()
    """
    version: str = "1.0.0"
    kernel_id: str = ""
    target_hw: str = ""
    decisions: list[AnyDecision] = field(default_factory=list)
    shape_regimes: list[ShapeRegime] = field(default_factory=list)
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    def add_decision(self, decision: AnyDecision) -> AnyDecision:
        decision.step = len(self.decisions) + 1
        self.decisions.append(decision)
        return decision

    def pop_decisions(self, n: int = 1) -> list[AnyDecision]:
        removed = []
        for _ in range(min(n, len(self.decisions))):
            removed.append(self.decisions.pop())
        return removed

    def tile(self, loop: str, factors: list[int],
             rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="tile", params={"loop": loop, "factors": factors},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def reorder(self, order: list[str], rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="reorder", params={"order": order},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def fuse(self, ops: list[str], fusion_type: str = "epilogue",
             rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="fuse", params={"ops": ops, "type": fusion_type},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def parallel(self, loops: list[str], mapping: dict[str, str],
                 rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="parallel", params={"loops": loops, "mapping": mapping},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def place(self, tensor: str, memory: str,
              rationale: str | None = None) -> Decision:
        return self.add_decision(Decision(
            kind="place", params={"tensor": tensor, "memory": memory},
            rationale=Rationale(text=rationale) if rationale else None,
        ))

    def compute_resource(self, warps: int | None = None,
                         stages: int | None = None,
                         rationale: str | None = None) -> Decision:
        """L2 decision: backend resource config (replaces v0.2.0 launch_config)."""
        params: dict = {}
        if warps is not None:
            params["warps"] = warps
        if stages is not None:
            params["stages"] = stages
        return self.add_decision(Decision(
            kind="compute_resource", params=params,
            rationale=Rationale(text=rationale) if rationale else None,
            level=2,
        ))

    def when(self, predicate: str, true_decisions: list[Decision],
             false_decisions: list[Decision] | None = None,
             rationale: str | None = None) -> ConditionalDecision:
        cd = ConditionalDecision(
            predicate=predicate,
            true_decisions=true_decisions,
            false_decisions=false_decisions or [],
            rationale=Rationale(text=rationale) if rationale else None,
        )
        return self.add_decision(cd)

    def summary(self) -> str:
        lines = [f"Strategy for {self.kernel_id} on {self.target_hw}:"]
        for d in self.decisions:
            if isinstance(d, ConditionalDecision):
                r = f" — {d.rationale.text}" if d.rationale else ""
                lines.append(f"  #{d.step} when({d.predicate}):{r}")
            else:
                r = f" — {d.rationale.text}" if d.rationale else ""
                lines.append(f"  #{d.step} {d.kind}({d.params}){r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {
            "version": self.version,
            "kernel_id": self.kernel_id,
            "target_hw": self.target_hw,
            "decisions": [_decision_to_dict(dec) for dec in self.decisions],
        }
        if self.shape_regimes:
            d["shape_regimes"] = [
                {"name": r.name, "predicate": r.predicate,
                 "decisions": [_decision_to_dict(dec) for dec in r.decisions]}
                for r in self.shape_regimes
            ]
        from dataclasses import asdict as _asdict
        c = _asdict(self.constraints)
        if any(v not in (0, 32) for v in c.values()):
            d["constraints"] = c
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> StrategyIR:
        """Deserialize -- handles v0.2.0 and v1.0 JSON."""
        ir = cls(
            version=data.get("version", "0.2.0"),
            kernel_id=data.get("kernel_id", ""),
            target_hw=data.get("target_hw", ""),
        )
        for d in data.get("decisions", []):
            if d.get("kind") == "__conditional__":
                true_ds = [_parse_decision(td) for td in d.get("true_decisions", [])]
                false_ds = [_parse_decision(fd) for fd in d.get("false_decisions", [])]
                rat = Rationale(**d["rationale"]) if d.get("rationale") else None
                ir.decisions.append(ConditionalDecision(
                    predicate=d["predicate"],
                    true_decisions=true_ds,
                    false_decisions=false_ds,
                    rationale=rat,
                    step=d.get("step", 0),
                ))
            else:
                ir.decisions.append(_parse_decision(d))
        return ir

    @classmethod
    def from_json(cls, json_str: str) -> StrategyIR:
        return cls.from_dict(json.loads(json_str))
```

### 7.4 Conditional Decision JSON Example

```json
{
  "version": "1.0.0",
  "kernel_id": "softmax_dynamic",
  "target_hw": "nvidia_ampere",
  "decisions": [
    {
      "kind": "__conditional__",
      "predicate": "dim(\"S\") <= 512",
      "true_decisions": [
        {"kind": "tile", "params": {"loop": "S", "factors": [256]},
         "rationale": {"text": "small S: single-pass fits in shared"},
         "step": 1, "level": 1}
      ],
      "false_decisions": [
        {"kind": "tile", "params": {"loop": "S", "factors": [512]},
         "rationale": {"text": "large S: multi-pass tiled reduce"},
         "step": 1, "level": 1}
      ],
      "step": 1
    },
    {
      "kind": "compute_resource",
      "params": {"warps": 4, "stages": 2},
      "rationale": {"text": "4 warps = 128 threads, 2-stage pipeline"},
      "step": 2, "level": 2
    }
  ]
}
```

### 7.5 MLIR Lowering by Strategy Level

Each StrategyIR level maps to specific MLIR dialect families:

| Level | StrategyIR Decisions | MLIR Standard Dialects | Lowering Example |
|-------|---------------------|----------------------|------------------|
| **L1** | `tile`, `fuse`, `vectorize`, `place` | `transform` dialect | `tile(M, [64])` → `transform.structured.tile_using_for %op tile_sizes [64]` |
| **L1** | `reorder` | `transform` dialect | `reorder([M,N,K])` → `transform.structured.interchange %op [0,1,2]` |
| **L2** | `ForNode`, `LoadTile`, `MAC` | `scf` / `affine` / `memref` | `ForNode("k", 0, K, 32)` → `scf.for %k = 0 to %K step 32` |
| **L2** | `MemoryBuffer(shared)` | `memref` + address spaces | `place(A, shared)` → `memref.alloc<workgroup>` |
| **L3** | `parallel`, `launch_config` | `gpu` dialect | `parallel(M, blockIdx.x)` → `gpu.launch blocks(%M)` |
| **L3** | `compute_resource` | `gpu` dialect | `num_warps=4` → `gpu.launch_func` params |

The `@rationale` annotation is preserved at all levels:
```mlir
transform.annotate %tiled "arke.rationale" = "M-tile=64: aligned to L2 cache lines"
```

---

## 8. Pass Infrastructure Design

### 8.1 Pass Protocol

```python
# arke/ir/passes/base.py
from __future__ import annotations
from typing import Protocol, Any
from dataclasses import dataclass, field


class Pass(Protocol):
    """Protocol for all Arke IR passes.

    A Pass transforms one IR object into another. Passes do NOT mutate input.
    """
    @property
    def name(self) -> str: ...

    def run(self, ir: Any, *args: Any) -> Any: ...

    def verify_pre(self, ir: Any) -> list[str]:
        return []

    def verify_post(self, ir: Any) -> list[str]:
        return []


@dataclass
class PassResult:
    ir: Any
    pass_name: str
    pre_violations: list[str] = field(default_factory=list)
    post_violations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.pre_violations and not self.post_violations


class IRInvariantError(Exception):
    """Raised when an IR invariant check fails in the pass pipeline."""
    pass
```

### 8.2 Pipeline Skeleton

```python
# arke/ir/passes/pipeline.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStage:
    name: str
    pass_obj: Any
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False


class Pipeline:
    """Dependency-aware, topologically-ordered pass pipeline."""

    def __init__(self, name: str):
        self.name = name
        self._stages: dict[str, PipelineStage] = {}
        self._order: list[str] = []

    def register(self, name: str, pass_obj: Any,
                 depends_on: list[str] | None = None,
                 optional: bool = False) -> None:
        self._stages[name] = PipelineStage(
            name=name, pass_obj=pass_obj,
            depends_on=depends_on or [], optional=optional,
        )
        self._order = self._topological_sort()

    def run(self, ir: Any, *extra_args: Any) -> Any:
        current = ir
        for stage_name in self._order:
            p = self._stages[stage_name].pass_obj
            pre_viol = p.verify_pre(current)
            if pre_viol:
                raise IRInvariantError(
                    f"Pass '{stage_name}' pre-conditions: {pre_viol}")
            current = p.run(current, *extra_args) if extra_args else p.run(current)
            post_viol = p.verify_post(current)
            if post_viol:
                raise IRInvariantError(
                    f"Pass '{stage_name}' post-conditions: {post_viol}")
        return current

    def _topological_sort(self) -> list[str]:
        in_deg = {n: 0 for n in self._stages}
        adj: dict[str, list[str]] = {n: [] for n in self._stages}
        for name, stage in self._stages.items():
            for dep in stage.depends_on:
                adj[dep].append(name)
                in_deg[name] += 1
        queue = [n for n, d in in_deg.items() if d == 0]
        result: list[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        if len(result) != len(self._stages):
            raise ValueError("Pass pipeline has a dependency cycle")
        return result
```

### 8.3 Lowering Pass Interfaces

```python
# arke/ir/passes/lowering.py  (Stage 1: interface definitions only)
from typing import Protocol


class SemanticToStrategyL2Pass(Protocol):
    """Lower SemanticIR + StrategyIR L1 -> StrategyIR L2.  (Stage 2)"""
    name = "semantic_to_strategy_l2"
    def run(self, semantic: "SemanticIR", strategy: "StrategyIR") -> "StrategyComputeIR": ...
    def verify_pre(self, ir: "SemanticIR") -> list[str]: ...
    def verify_post(self, ir: "StrategyComputeIR") -> list[str]: ...


class StrategyToHardwarePass(Protocol):
    """Lower StrategyIR L2/L3 -> HardwareIR.  (Stage 3)"""
    name = "strategy_to_hardware"
    def run(self, strategy: "StrategyComputeIR", target: str) -> "HardwareIR": ...
    def verify_pre(self, ir: "StrategyComputeIR") -> list[str]: ...
    def verify_post(self, ir: "HardwareIR") -> list[str]: ...


class HardwareToInstructionPass(Protocol):
    """Lower HardwareIR -> InstructionIR.  (Stage 4)"""
    name = "hardware_to_instruction"
    def run(self, hw: "HardwareIR") -> "InstructionIR": ...
    def verify_pre(self, ir: "HardwareIR") -> list[str]: ...
    def verify_post(self, ir: "InstructionIR") -> list[str]: ...
```

### 8.4 IR Invariant Checkers

```python
# arke/ir/passes/invariants.py

class SemanticIRInvariantChecker:
    """Post-pass invariant checker for SemanticIR."""

    def check(self, ir: "SemanticIR") -> list[str]:
        errors: list[str] = []
        seen_ids: set[str] = set()
        for n in ir.nodes:
            if n.id in seen_ids:
                errors.append(f"Duplicate node id: '{n.id}'")
            seen_ids.add(n.id)
        if ir.return_node and not ir.get_node(ir.return_node):
            errors.append(f"return_node '{ir.return_node}' not found")
        param_names = {p.name for p in ir.params}
        for n in ir.nodes:
            if not hasattr(n, "inputs"):
                continue
            for port, ref in n.inputs.items():
                if hasattr(ref, "name") and ref.name not in param_names:
                    errors.append(
                        f"Node '{n.id}' port '{port}': unknown param '{ref.name}'")
                if hasattr(ref, "id") and not ir.get_node(ref.id):
                    errors.append(
                        f"Node '{n.id}' port '{port}': unknown node '{ref.id}'")
        return errors


class StrategyIRInvariantChecker:
    """Post-pass invariant checker for StrategyIR."""

    def check(self, strategy: "StrategyIR",
              semantic: "SemanticIR | None" = None) -> list[str]:
        errors: list[str] = []
        steps = [d.step for d in strategy.decisions]
        if len(steps) != len(set(steps)):
            errors.append("Duplicate step numbers in StrategyIR decisions")
        if semantic:
            node_ids = {n.id for n in semantic.nodes}
            for d in strategy.decisions:
                if hasattr(d, "kind") and d.kind == "fuse":
                    for op in d.params.get("ops", []):
                        if op not in node_ids:
                            errors.append(
                                f"Strategy fuse references unknown node '{op}'")
        return errors
```

---

## 9. JSON Schema & Compact Format

### 9.1 Design Principles

1. **Omit defaults**: `layout=row_major`, `attrs={}`, `edges=[]`, `fusion_groups=[]` omitted when empty/default
2. **Shape shorthand**: `[1024, 512]` for concrete dims; `[8, {"sym": "S"}, 64]` for symbolic
3. **Compact InputRef**: `{"ref": "param", "name": "A"}` -- no extra derived fields
4. **Compact edges**: `{"from": "x", "to": "y", "tensor": "C"}` (short keys)
5. **Incremental patch**: LLM agents send `{"patch": [...]}` diffs instead of full IR

### 9.2 Compact vs Verbose Comparison

**Compact (default `to_dict()`):**
```json
{
  "version": "1.0.0",
  "kernel_id": "matmul",
  "params": [
    {"name": "A", "shape": [1024, 512], "dtype": "f16"},
    {"name": "B", "shape": [512, 2048], "dtype": "f16"}
  ],
  "nodes": [{
    "id": "matmul_0", "op": "matmul",
    "inputs": {"A": {"ref": "param", "name": "A"},
               "B": {"ref": "param", "name": "B"}},
    "output": {"shape": [1024, 2048], "dtype": "f16"},
    "semantics": {"computation": "C[i,j]=sum(A[i,k]*B[k,j],k)",
                  "reduction_axes": ["k"]}
  }],
  "return_node": "matmul_0"
}
```

**Verbose (all fields explicit):**
```json
{
  "version": "1.0.0",
  "kernel_id": "matmul",
  "symbolic_dims": [], "shape_constraints": [],
  "params": [{"name": "A", "shape": [1024, 512], "dtype": "f16", "layout": "row_major"}],
  "nodes": [{
    "id": "matmul_0", "op": "matmul",
    "inputs": {"A": {"ref": "param", "name": "A"}, "B": {"ref": "param", "name": "B"}},
    "output": {"shape": [1024, 2048], "dtype": "f16", "layout": "row_major"},
    "semantics": {"computation": "C[i,j]=sum(A[i,k]*B[k,j],k)",
                  "index_vars": ["i","j","k"], "reduction_axes": ["k"],
                  "properties": []},
    "attrs": {}
  }],
  "edges": [], "return_node": "matmul_0", "return_ports": [],
  "fusion_groups": [], "metadata": {}
}
```

### 9.3 Incremental Patch Format

```json
{
  "kernel_id": "matmul",
  "patch": [
    {"op": "add_node", "node": {
      "id": "relu_0", "op": "relu",
      "inputs": {"X": {"ref": "node", "id": "matmul_0"}},
      "output": {"shape": [1024, 2048], "dtype": "f16"},
      "semantics": {"computation": "Y[i,j]=max(X[i,j],0)"}
    }},
    {"op": "set_return", "node_id": "relu_0"},
    {"op": "add_fusion_group",
     "group": {"id": "fg_0", "nodes": ["matmul_0","relu_0"], "type": "epilogue"}}
  ]
}
```

---

## 10. MLIR Integration Design

### 10.1 Core Principle

Arke IR lowers to **MLIR standard dialects** (`linalg`, `transform`, `scf`, `affine`, `gpu`, `memref`, `llvm`). The MLIR ecosystem provides battle-tested optimization passes and hardware backends; Arke IR provides the LLM-native interface layer above it.

### 10.2 Layer-to-Dialect Mapping

Full field-level mapping: `docs/spec/ir-mlir-mapping.md`. Summary:

| Arke IR Layer | MLIR Standard Dialects | Role |
|---------------|----------------------|------|
| SemanticIR (Layer 4) | `linalg` / `tensor` | Math semantics: matmul → `linalg.matmul`, relu → `linalg.generic` |
| StrategyIR L1 decisions | `transform` | Optimization: tile → `transform.structured.tile_using_for` |
| StrategyIR L2 structures | `scf` / `affine` / `memref` | Loop nests: ForNode → `scf.for`; memory: `memref` + address spaces |
| StrategyIR L3 decisions | `gpu` | Hardware: `gpu.launch`, `gpu.thread_id`, block/warp mapping |
| HardwareIR (Layer 2) | `gpu` / `nvvm` / `rocdl` | Target-specific: shared memory, barriers, MMA instructions |
| InstructionIR (Layer 1) | `llvm` | Direct LLVM IR emission |
| `@rationale` | `transform.annotate` | Preserved as `"arke.rationale"` attribute |
| `FusionGroup` | `transform.structured.fuse_into_containing_op` | Epilogue fusion |

### 10.3 Stage-by-Stage MLIR Integration

```
Stage 1 ── MLIR framework + BL1 basic pathway:
  Primary: SemanticIR + StrategyIR L1 → Jinja2 templates → Triton Python → GPU
  MLIR:    MLIREmitter skeleton; BL1 ops (13) emit linalg/transform MLIR
           for correctness cross-check (verify via mlir-opt)

Stage 2 ── Full MLIR integration:
  StrategyIR L2 → scf/affine/memref MLIR
  Both Triton and MLIR paths active; MLIR for validation + alternative codegen

Stage 3 ── Complete integration, deeper hardware control:
  StrategyIR L3 + HardwareIR → gpu/nvvm/rocdl MLIR
  MLIR becomes primary codegen path for multi-target (NVIDIA + Ascend)

Stage 4 ── Direct LLVM IR, MLIR as optional target:
  InstructionIR → LLVM IR directly
  MLIR path remains available for targets where it is optimal
```

### 10.4 Arke IR vs MLIR Comparison

| Dimension | MLIR | Arke IR |
|-----------|------|---------|
| LLM legibility | Poor (C++ text IR, verbose) | First-class (JSON/Python) |
| Agent action space | Unbounded (any dialect op) | Bounded (structured decisions) |
| New op support | New C++ dialect | Row in `ops/catalog.py` |
| `@rationale` capture | External tooling | First-class field |
| Serialization | Custom text format | Standard JSON |
| Learning signals | None | Trajectory JSONL with rationale |
| Debugging | llvm-opt toolchain | Python `to_json()` |
| Hardware backends | Extensive ecosystem | Leverages MLIR ecosystem via lowering |

Arke IR and MLIR are complementary: Arke IR is the LLM-facing layer, MLIR is the compiler-facing layer. If future MLIR developments provide LLM-friendly interfaces, Arke can directly reuse them.

### 10.5 Specialized Target Examples

While Arke IR targets MLIR standard dialects, concrete compilation targets may involve specialized MLIR-based IRs. This section illustrates how Arke IR maps to them.

#### Triton TTIR / TTGPUIR

Triton's internal compilation pipeline uses MLIR dialects:
- **TTIR** (`tt` dialect): tile-level operations (`tt.dot`, `tt.reduce`, `tt.load`/`tt.store`)
- **TTGPUIR** (`triton_gpu` dialect): GPU-specific scheduling (coalesce, pipeline, accelerate_matmul)
- **TTNVGPUIR** (`triton_nvidia_gpu` dialect): NVIDIA-specific (TMA, fence, CTA planning)

```
Arke Layer              MLIR Standard          Triton Specialized
──────────────────────────────────────────────────────────────────────
SemanticIR            linalg/tensor        →  tt.dot, tt.reduce
StrategyIR L1         transform            →  (via Triton frontend)
StrategyIR L2         scf/memref           →  TTGPUIR layout/pipeline
StrategyIR L3         gpu                  →  TTNVGPUIR (TMA, CTA)
```

#### NVIDIA CUDA Tile IR (CuTe / CUTLASS)

CUDA Tile IR (from CUTLASS/CuTe) describes tile-level data movement between memory hierarchies:
- **Tile layouts**: describing how data tiles map from global → shared → register memory
- **MMA descriptors**: mapping tile shapes to hardware MMA instructions (HMMA, IMMA)
- **Copy atoms**: minimal data movement primitives between memory tiers

```
Arke Layer              MLIR Standard          CUDA Tile IR Concepts
──────────────────────────────────────────────────────────────────────
StrategyIR L2         memref/affine         →  Tile layouts, copy atoms
  LoadTile            memref.copy           →  CuTe copy_async (G→S)
  MAC                 linalg.matmul         →  CuTe MMA descriptor
StrategyIR L3         gpu                   →  Warp-group scheduling
  place(shared)       memref+workgroup      →  SMEM tile allocation
HardwareIR            gpu/nvvm              →  HMMA/IMMA instruction selection
```

> These specialized mappings are **implementation details** of specific backends.
> Arke IR's architecture is defined in terms of MLIR standard dialects;
> backend implementations may leverage specialized dialects as optimization targets.

---

## 11. Backward Compatibility

### 11.1 SemanticIR v0.2.0 -> v1.0

| Change | Compatibility Handling |
|--------|------------------------|
| `version` bumped to `"1.0.0"` | `from_dict()` accepts any version string |
| `return_type` removed | Ignored silently if present in old JSON |
| `params.shape` supports `SymbolicDim` | Old JSON `[int, ...]` still valid |
| `Node.attrs` added | Defaults to `{}` when absent |
| Edge uses compact keys `from`/`to`/`tensor` | Both old and new key names accepted |
| New node types `MultiOutputNode`, `ConditionalNode` | Old JSON has only plain nodes -- parsed correctly |
| `SemanticGraph` alias | Preserved: `SemanticGraph = SemanticIR` |

**Contract:** All existing v0.2.0 JSON files load without error via `SemanticIR.from_dict()`.

### 11.2 StrategyIR v0.2.0 -> v1.0

| Change | Compatibility Handling |
|--------|------------------------|
| `version` bumped to `"1.0.0"` | `from_dict()` accepts any version string |
| `Decision.level` added | Defaults to `1` when absent |
| `launch_config` kind | Auto-mapped to `compute_resource` (L2) by `_parse_decision()` |
| `autotune` kind | Preserved as-is (still valid L2 decision) |
| `ConditionalDecision` added | Old JSON has no `__conditional__` entries -- not affected |
| `shape_regimes` added | Defaults to `[]` when absent |

**Contract:** All 422 existing tests continue to pass.

### 11.3 Migration Code Patterns

```python
# v0.2.0 code -- still works unchanged
ir = StrategyIR(kernel_id="matmul_k", target_hw="nvidia_ampere")
ir.tile("M", [64], "tensor-core aligned")
ir.add_decision(Decision(kind="launch_config", params={"num_warps": 4}))

# v1.0 preferred style
ir = StrategyIR(kernel_id="matmul_k", target_hw="nvidia_ampere")
ir.tile("M", [64], "tensor-core aligned")
ir.compute_resource(warps=4, rationale="128 threads, Ampere optimal")

# Round-trip: v0.2.0 JSON -> v1.0 object (auto-migration)
old_json = '{"version":"0.2.0","kernel_id":"k","target_hw":"nvidia_ampere",' \
           '"decisions":[{"kind":"launch_config","params":{"num_warps":4},"step":1}]}'
strategy = StrategyIR.from_json(old_json)
# strategy.decisions[0].kind == "compute_resource"  <- auto-migrated
# strategy.decisions[0].params == {"warps": 4}
# strategy.decisions[0].level == 2
```

---

## 12. Stage 1 Implementation Scope

### 12.1 What Gets Implemented in Stage 1

| Component | Status | Location |
|-----------|--------|----------|
| **SemanticIR v1.0 full schema** | IMPLEMENT | `arke/ir/semantic.py` |
| **StrategyIR v1.0 full schema** | IMPLEMENT | `arke/ir/strategy.py` |
| `SymbolicDim`, `ShapeConstraint` | IMPLEMENT | `arke/ir/semantic.py` |
| `ConditionalNode` | IMPLEMENT | `arke/ir/semantic.py` |
| `MultiOutputNode` | IMPLEMENT | `arke/ir/semantic.py` |
| Enhanced `InputRef` (dtype/shape fields) | IMPLEMENT | `arke/ir/semantic.py` |
| `Node.attrs` field | IMPLEMENT | `arke/ir/semantic.py` |
| `ConditionalDecision` | IMPLEMENT | `arke/ir/strategy.py` |
| `compute_resource` decision kind | IMPLEMENT | `arke/ir/strategy.py` |
| `ShapeRegime` | IMPLEMENT | `arke/ir/strategy.py` |
| `SSAVerifier` (basic) | IMPLEMENT | `arke/ir/passes/invariants.py` |
| `SemanticIRInvariantChecker` | IMPLEMENT | `arke/ir/passes/invariants.py` |
| `StrategyIRInvariantChecker` | IMPLEMENT | `arke/ir/passes/invariants.py` |
| `Pass` protocol | IMPLEMENT | `arke/ir/passes/base.py` |
| `Pipeline` skeleton | IMPLEMENT | `arke/ir/passes/pipeline.py` |
| Lowering pass **interfaces** (protocols only) | SPEC ONLY | `arke/ir/passes/lowering.py` |
| **StrategyIR L2 schema** | SPEC ONLY | `arke/ir/strategy_compute.py` (StrategyIR L2 structures) |
| **HardwareIR interface** | SPEC ONLY | `arke/ir/hardware.py` (file exists, stub) |
| **InstructionIR interface** | SPEC ONLY | `arke/ir/instruction.py` (file exists, stub) |
| **MLIR framework** | IMPLEMENT | `arke/backend/mlir_emitter.py` (skeleton) |
| **BL1 MLIR pathway** | IMPLEMENT | 13 BL1 ops → linalg/transform MLIR + mlir-opt verify |
| `SemanticToStrategyL2Pass` | SPEC ONLY | `arke/ir/passes/lowering.py` |
| `StrategyToHardwarePass` | SPEC ONLY | `arke/ir/passes/lowering.py` |

### 12.2 What Is Deferred to Later Stages

| Component | Stage | Rationale |
|-----------|-------|-----------|
| `SemanticToStrategyL2Pass` implementation | Stage 2 | Requires StrategyIR L2 schema stable |
| `StrategyToHardwarePass` implementation | Stage 3 | Requires HardwareIR design complete |
| `HardwareToInstructionPass` implementation | Stage 4 | Requires LLVM IR binding design |
| Full MLIR codegen (beyond BL1 verify) | Stage 2-3 | Full integration after L2/L3 implemented |
| InstructionIR full schema | Stage 4 | Depends on LLVM Python binding choice |
| PTX/cubin direct emission | Stage 4 | Follows InstructionIR |
| `torch.compile` Inductor backend | Stage 3+ | Follows MLIR backend |

### 12.3 Stage 1 Pass Pipeline (Triton Path)

In Stage 1, the lowering pipeline is:

```
SemanticIR (Layer 4)
    +
StrategyIR v1.0
    |
    | [SSAVerifyPass]            <- Stage 1: implemented
    | [StrategyValidatePass]     <- Stage 1: implemented
    |
    v
TritonCodegenPass               <- Stage 1: existing Jinja2 engine
    |
    v
Triton Python code
    |
    v
triton.compile() -> GPU binary
```

The `Pipeline` class registers these passes:

```python
# Stage 1 pipeline (arke/compiler/pipeline.py)
from arke.ir.passes.pipeline import Pipeline
from arke.ir.passes.invariants import SemanticIRInvariantChecker, StrategyIRInvariantChecker
from arke.backend.triton_backend import TritonCodegenPass

def build_stage1_pipeline() -> Pipeline:
    p = Pipeline("stage1_triton")
    p.register("ssa_verify", SemanticIRInvariantChecker())
    p.register("strategy_validate", StrategyIRInvariantChecker())
    p.register("triton_codegen", TritonCodegenPass(),
               depends_on=["ssa_verify", "strategy_validate"])
    return p
```

### 12.4 Test Compatibility Matrix

All 422 existing tests must pass after the v0.2.0 -> v1.0 upgrade:

| Test Category | Expected Impact | Mitigation |
|---------------|----------------|------------|
| SemanticIR `from_dict()` tests | Zero breakage | Additive fields with defaults |
| StrategyIR `from_dict()` tests | Zero breakage | `_parse_decision()` maps old kinds |
| `launch_config` decision tests | Auto-migrated | `kind` becomes `compute_resource` |
| Codegen / Triton output tests | Zero breakage | Codegen reads same fields |
| Agent decision-making tests | Zero breakage | New fields are optional |
| Gate benchmark tests | Zero breakage | No schema-breaking changes |

---

## Appendix: File Map

```
arke/ir/
├── __init__.py
├── semantic.py          <- SemanticIR v1.0 (Layer 4)
├── strategy.py          <- StrategyIR v1.0
├── strategy_compute.py  <- StrategyIR L2 structures (Layer 3) [Stage 1: stub]
├── hardware.py          <- HardwareIR v1.0 (Layer 2) [Stage 1: stub]
├── instruction.py       <- InstructionIR v1.0 (Layer 1) [Stage 1: interface]
├── builder.py           <- IR builder utilities
├── shape_inference.py   <- Shape propagation
├── ops/
│   ├── catalog.py       <- OpDef catalog (45 ops)
│   └── ...
├── schemas/             <- JSON schema files
│   ├── semantic_v1.json
│   └── strategy_v1.json
├── targets/             <- Hardware target profiles
└── passes/              <- Pass infrastructure [Stage 1: new]
    ├── __init__.py
    ├── base.py          <- Pass protocol, PassResult, IRInvariantError
    ├── pipeline.py      <- Pipeline class
    ├── invariants.py    <- SemanticIR/StrategyIR invariant checkers
    └── lowering.py      <- Lowering pass interfaces (Stage 2-4)
```

---

*Document version: 1.0 | Created: 2026-04-06 | Status: Design Spec*
*Applies to: Arke Stage 1 (foundation) + Stage 2-4 roadmap*
