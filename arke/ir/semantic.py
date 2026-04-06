# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Semantic IR (Layer 1).

The Semantic IR describes WHAT to compute, not HOW to compute it.
Each node represents an operator with explicit semantics, algebraic properties,
and data flow information.

See docs/spec/arke-ir-spec-v1.md §2 for the full specification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ============================================================
# Data Types
# ============================================================

@dataclass
class TensorDesc:
    """Description of a tensor (shape, dtype, layout)."""
    shape: list[int]
    dtype: str
    layout: str = "row_major"


@dataclass
class Param:
    """A kernel input parameter.

    Each param defines a named tensor that nodes can reference.
    """
    name: str
    shape: list[int]
    dtype: str
    layout: str = "row_major"

    def to_tensor_desc(self) -> TensorDesc:
        """Convert this parameter to a TensorDesc."""
        return TensorDesc(shape=self.shape, dtype=self.dtype, layout=self.layout)


# ============================================================
# Input References (unambiguous, no magic strings)
# ============================================================

@dataclass
class ParamRef:
    """Reference to a kernel parameter."""
    name: str

    def to_dict(self) -> dict:
        """Serialize this parameter reference to a dict."""
        return {"ref": "param", "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> ParamRef:
        """Deserialize a ParamRef from a dict."""
        return cls(name=d["name"])


@dataclass
class NodeRef:
    """Reference to a previous node's output."""
    id: str

    def to_dict(self) -> dict:
        """Serialize this node reference to a dict."""
        return {"ref": "node", "id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> NodeRef:
        """Deserialize a NodeRef from a dict."""
        return cls(id=d["id"])


InputRef = ParamRef | NodeRef


def input_ref_from_dict(d: dict | str) -> InputRef:
    """Parse an input reference from dict or legacy string format."""
    if isinstance(d, str):
        # Legacy format: "@node_id" or "param_name"
        if d.startswith("@"):
            return NodeRef(id=d[1:])
        return ParamRef(name=d)
    if d.get("ref") == "param":
        return ParamRef.from_dict(d)
    if d.get("ref") == "node":
        return NodeRef.from_dict(d)
    raise ValueError(f"Invalid input reference: {d}")


# ============================================================
# Semantic Components
# ============================================================

@dataclass
class Semantics:
    """Explicit semantic description of an operator."""
    computation: str  # e.g., "C[i,j] = sum(A[i,k] * B[k,j], axis=k)"
    index_vars: list[str] = field(default_factory=list)
    reduction_axes: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)


@dataclass
class Node:
    """A single operator node in the Semantic IR."""
    id: str
    op: str
    inputs: dict[str, InputRef]
    output: TensorDesc
    semantics: Semantics


@dataclass
class Edge:
    """Data flow edge between nodes."""
    from_node: str
    to_node: str
    tensor_name: str
    lifetime: str = "local"  # local | persistent


@dataclass
class FusionGroup:
    """A group of nodes that can be fused."""
    id: str
    nodes: list[str]
    fusion_type: str  # epilogue | prologue | horizontal | vertical
    reason: str = ""


# ============================================================
# SemanticIR (top-level)
# ============================================================

@dataclass
class SemanticIR:
    """Layer 1 of Arke IR — the Semantic IR.

    Describes the computation at operator level with explicit semantics.
    Immutable after creation — optimization does not modify it.
    """

    version: str = "0.2.0"
    kernel_id: str = ""
    params: list[Param] = field(default_factory=list)
    return_type: TensorDesc | None = None
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    return_node: str = ""
    fusion_groups: list[FusionGroup] = field(default_factory=list)

    def add_param(self, param: Param) -> None:
        """Add an input parameter."""
        self.params.append(param)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def add_fusion_group(self, group: FusionGroup) -> None:
        """Add a fusion group."""
        self.fusion_groups.append(group)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_param(self, name: str) -> Param | None:
        """Get a param by name."""
        for p in self.params:
            if p.name == name:
                return p
        return None

    # ─── Serialization ───

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        d: dict[str, Any] = {
            "version": self.version,
            "kernel_id": self.kernel_id,
            "params": [asdict(p) for p in self.params],
            "return_type": asdict(self.return_type) if self.return_type else None,
            "nodes": [],
            "edges": [asdict(e) for e in self.edges],
            "return_node": self.return_node,
            "fusion_groups": [asdict(fg) for fg in self.fusion_groups],
        }
        for node in self.nodes:
            nd: dict[str, Any] = {
                "id": node.id,
                "op": node.op,
                "inputs": {k: v.to_dict() for k, v in node.inputs.items()},
                "output": asdict(node.output),
                "semantics": asdict(node.semantics),
            }
            d["nodes"].append(nd)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: str) -> None:
        """Save to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> SemanticIR:
        """Deserialize from a plain dict."""
        ir = cls(
            version=data.get("version", "0.2.0"),
            kernel_id=data.get("kernel_id", ""),
            return_node=data.get("return_node", ""),
        )

        # Params
        for p in data.get("params", []):
            ir.add_param(Param(
                name=p["name"],
                shape=p["shape"],
                dtype=p["dtype"],
                layout=p.get("layout", "row_major"),
            ))

        # Return type
        rt = data.get("return_type")
        if rt:
            ir.return_type = TensorDesc(
                shape=rt["shape"], dtype=rt["dtype"],
                layout=rt.get("layout", "row_major"),
            )

        # Nodes
        for nd in data.get("nodes", []):
            inputs: dict[str, InputRef] = {}
            for k, v in nd.get("inputs", {}).items():
                inputs[k] = input_ref_from_dict(v)

            sem_data = nd.get("semantics", {})
            ir.add_node(Node(
                id=nd["id"],
                op=nd["op"],
                inputs=inputs,
                output=TensorDesc(
                    shape=nd["output"]["shape"],
                    dtype=nd["output"]["dtype"],
                    layout=nd["output"].get("layout", "row_major"),
                ),
                semantics=Semantics(
                    computation=sem_data.get("computation", ""),
                    index_vars=sem_data.get("index_vars", []),
                    reduction_axes=sem_data.get("reduction_axes", []),
                    properties=sem_data.get("properties", []),
                ),
            ))

        # Edges
        for e in data.get("edges", []):
            ir.add_edge(Edge(
                from_node=e["from_node"],
                to_node=e["to_node"],
                tensor_name=e["tensor_name"],
                lifetime=e.get("lifetime", "local"),
            ))

        # Fusion groups
        for fg in data.get("fusion_groups", []):
            ir.add_fusion_group(FusionGroup(
                id=fg["id"],
                nodes=fg["nodes"],
                fusion_type=fg["fusion_type"],
                reason=fg.get("reason", ""),
            ))

        return ir

    @classmethod
    def from_json(cls, json_str: str) -> SemanticIR:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> SemanticIR:
        """Load from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())


# ============================================================
# Backward compatibility alias
# ============================================================

SemanticGraph = SemanticIR  # Deprecated — use SemanticIR
