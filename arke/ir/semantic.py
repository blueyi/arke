# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Semantic Graph (Layer 1).

The Semantic Graph describes WHAT to compute, not HOW to compute it.
Each node represents an operator with explicit semantics, algebraic properties,
and data flow information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TensorDesc:
    """Description of a tensor (shape, dtype, layout)."""
    shape: list[int]
    dtype: str
    layout: str = "row_major"


@dataclass
class Semantics:
    """Explicit semantic description of an operator."""
    computation: str  # e.g., "C[i,j] = sum(A[i,k] * B[k,j], axis=k)"
    index_vars: list[str] = field(default_factory=list)  # e.g., ["i", "j", "k"]
    reduction_axes: list[str] = field(default_factory=list)  # e.g., ["k"]
    properties: list[str] = field(default_factory=list)  # e.g., ["associative", "elementwise"]


@dataclass
class Node:
    """A single operator node in the Semantic Graph."""
    id: str
    op: str
    inputs: dict[str, TensorDesc | str]  # str for references like "@matmul_0.output"
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


@dataclass
class SemanticGraph:
    """Layer 1 of Arke IR — the Semantic Graph.

    Describes the computation at operator level with explicit semantics.
    """

    version: str = "0.1.0"
    graph_id: str = ""
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    fusion_groups: list[FusionGroup] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def add_fusion_group(self, group: FusionGroup) -> None:
        """Add a fusion group."""
        self.fusion_groups.append(group)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=indent)

    def to_file(self, path: str) -> None:
        """Save to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> SemanticGraph:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        # TODO: Full deserialization with nested dataclass reconstruction
        graph = cls(
            version=data.get("version", "0.1.0"),
            graph_id=data.get("graph_id", ""),
        )
        return graph

    @classmethod
    def from_file(cls, path: str) -> SemanticGraph:
        """Load from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())
