# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — SemanticIR (Layer 4).

Operator-level DAG: "what to compute."
Primary LLM Agent interface. Serializable to JSON.

See docs/spec/arke-ir-spec-design.md §3.2 for the complete schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from arke.version import IR_SCHEMA_VERSION, resolve_ir_schema_version

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
    """A named symbolic dimension (runtime variable or static symbolic dim).

    Examples:
        SymbolicDim("B")         # batch size
        SymbolicDim("S")         # sequence length
        SymbolicDim("H", min=1, max=128)  # bounded head count
        SymbolicDim("K", is_static=True)  # compile-time constant symbolic dim
    """
    name: str
    min: int | None = None   # optional lower bound (for compiler hints)
    max: int | None = None   # optional upper bound (for compiler hints)
    is_static: bool = False
    multiple_of: int | None = None
    default: int | None = None

    def to_dict(self) -> dict:
        d: dict = {"sym": self.name}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.is_static:
            d["is_static"] = True
        if self.multiple_of is not None:
            d["multiple_of"] = self.multiple_of
        if self.default is not None:
            d["default"] = self.default
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SymbolicDim:
        return cls(
            name=d["sym"],
            min=d.get("min"),
            max=d.get("max"),
            is_static=d.get("is_static", False),
            multiple_of=d.get("multiple_of"),
            default=d.get("default"),
        )


# A dimension can be a concrete int or a symbolic variable
Dim = int | SymbolicDim


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
            "shape": [dim_to_json(s) for s in self.shape],
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
            "shape": [dim_to_json(s) for s in self.shape],
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


InputRef = ParamRef | NodeRef


def input_ref_from_dict(d: dict) -> InputRef:
    """Parse InputRef from current structured form."""
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

    - attrs: op-specific attributes (eps, axis, k, groups, etc.)
    - output is a TensorDesc with Dim (may be symbolic)
    - inputs use structured typed InputRef records
    """
    id: str
    op: str
    inputs: dict[str, InputRef]
    output: TensorDesc
    semantics: Semantics
    attrs: dict[str, Any] = field(default_factory=dict)

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
    Both branches must produce the same output type.
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
AnyNode = Node | MultiOutputNode | ConditionalNode


# ─── Edges & Fusion ────────────────────────────────────────────────────────

@dataclass
class Edge:
    """Data flow edge between nodes (or param -> node)."""
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


# ─── Top-Level SemanticIR ──────────────────────────────────────────────────

@dataclass
class SemanticIR:
    """Layer 4 of Arke IR — the Semantic IR.

    Describes computation at operator level. Immutable after construction.
    The LLM Agent reads this; StrategyIR is what the LLM writes.
    """

    version: str = IR_SCHEMA_VERSION
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
        """Deserialize SemanticIR from the current structured format."""
        ir = cls(
            version=resolve_ir_schema_version(
                data.get("version"),
                artifact="SemanticIR",
            ),
            kernel_id=data.get("kernel_id", ""),
            return_node=data.get("return_node", ""),
            return_ports=data.get("return_ports", []),
        )

        # Symbolic dims
        for sd in data.get("symbolic_dims", []):
            ir.add_symbolic_dim(SymbolicDim.from_dict(sd))

        # Shape constraints
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


