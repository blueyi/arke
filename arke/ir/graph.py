# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Minimal IR for S6 Pass Pipeline.

Lightweight IR representation that the pass system operates on.
This expands in S7 into the full multi-layer Lang & IR architecture.

For S6, IR is a graph of nodes where each node is an op invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IRValue:
    """A value (tensor) in the IR graph.

    Identified by a unique name (SSA form: each name defined exactly once).
    """
    name: str
    dtype: str = "float32"  # "float32", "float16", "bfloat16", "int32", "int64", "int8"
    shape: list[int] = field(default_factory=list)
    is_input: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IRValue):
            return self.name == other.name
        return NotImplemented


@dataclass
class IRNode:
    """A single operation in the IR graph.

    Represents one op invocation with named inputs and outputs.
    """
    id: str  # unique node identifier
    op: str  # operator name (must exist in OpRegistry)
    inputs: dict[str, str]  # op_input_name -> IRValue.name
    outputs: list[str]  # list of IRValue.name produced
    attrs: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None  # @rationale annotation


@dataclass
class IRGraph:
    """A computation graph in SSA form.

    The graph consists of:
    - values: all tensor values (inputs + intermediate + outputs)
    - nodes: ordered list of operations
    - graph_inputs: names of input values
    - graph_outputs: names of output values
    """
    name: str = "unnamed"
    values: dict[str, IRValue] = field(default_factory=dict)
    nodes: list[IRNode] = field(default_factory=list)
    graph_inputs: list[str] = field(default_factory=list)
    graph_outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_input(self, name: str, dtype: str = "float32", shape: list[int] | None = None) -> IRValue:
        """Add a graph input value."""
        v = IRValue(name=name, dtype=dtype, shape=shape or [], is_input=True)
        self.values[name] = v
        self.graph_inputs.append(name)
        return v

    def add_node(self, node: IRNode) -> IRNode:
        """Add an operation node and register its output values."""
        self.nodes.append(node)
        for out_name in node.outputs:
            if out_name not in self.values:
                self.values[out_name] = IRValue(name=out_name)
        return node

    def set_outputs(self, names: list[str]) -> None:
        """Set graph output values."""
        self.graph_outputs = names

    def get_value(self, name: str) -> IRValue:
        """Get a value by name."""
        if name not in self.values:
            raise KeyError(f"Value {name!r} not found in graph")
        return self.values[name]

    def defined_names(self) -> set[str]:
        """Get all defined value names (inputs + node outputs)."""
        names = set(self.graph_inputs)
        for node in self.nodes:
            names.update(node.outputs)
        return names

    def used_names(self) -> set[str]:
        """Get all used value names (node inputs + graph outputs)."""
        names = set()
        for node in self.nodes:
            names.update(node.inputs.values())
        names.update(self.graph_outputs)
        return names
