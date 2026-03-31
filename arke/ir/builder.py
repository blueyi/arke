# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Builder.

Fluent API for constructing SemanticGraph from Python, without writing raw JSON.
"""

from __future__ import annotations

from arke.ir.ops.catalog import get_op, is_fusable_epilogue
from arke.ir.semantic import Edge, FusionGroup, Node, SemanticGraph, Semantics, TensorDesc


class KernelBuilder:
    """Fluent builder for constructing a SemanticGraph.

    Example:
        b = KernelBuilder("fused_matmul_relu")
        b.param("A", [1024, 512], "f16")
        b.param("B", [512, 2048], "f16")
        m = b.op("matmul", A="A", B="B")
        r = b.op("relu", X=m)
        b.returns(r, [1024, 2048], "f16")
        ir = b.build()
    """

    def __init__(self, name: str):
        self.name = name
        self.params: list[dict] = []
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.return_node: str | None = None
        self.return_shape: list[int] = []
        self.return_dtype: str = "f16"
        self._node_count = 0

    def param(self, name: str, shape: list[int], dtype: str,
              layout: str = "row_major") -> str:
        """Add an input parameter. Returns the parameter name for reference."""
        self.params.append({
            "name": name, "shape": shape, "dtype": dtype, "layout": layout,
        })
        return name

    def op(self, op_name: str, **inputs: str) -> str:
        """Add an operator node. Returns the node ID for chaining.

        Inputs can be parameter names (str) or prior node IDs (str).
        """
        node_id = f"{op_name}_{self._node_count}"
        self._node_count += 1

        op_def = get_op(op_name)

        # Build input dict: resolve parameter refs vs node refs
        resolved_inputs: dict = {}
        for key, ref in inputs.items():
            if any(p["name"] == ref for p in self.params):
                resolved_inputs[key] = TensorDesc(
                    shape=next(p["shape"] for p in self.params if p["name"] == ref),
                    dtype=next(p["dtype"] for p in self.params if p["name"] == ref),
                )
            else:
                resolved_inputs[key] = f"@{ref}"
                # Add edge from referenced node
                self.edges.append(Edge(
                    from_node=ref, to_node=node_id, tensor_name=f"{ref}_out",
                ))

        # Infer output shape (simplified — real impl needs shape inference)
        # For now: copy from first input or use last param shape
        if self.params:
            out_shape = self.params[0]["shape"]
            out_dtype = self.params[0]["dtype"]
        else:
            out_shape = [1]
            out_dtype = "f32"

        node = Node(
            id=node_id,
            op=op_name,
            inputs=resolved_inputs,
            output=TensorDesc(shape=out_shape, dtype=out_dtype),
            semantics=Semantics(
                computation=op_def.computation,
                index_vars=list(op_def.index_vars),
                reduction_axes=list(op_def.reduction_axes),
                properties=list(op_def.properties),
            ),
        )
        self.nodes.append(node)
        return node_id

    def returns(self, node_id: str, shape: list[int], dtype: str) -> None:
        """Set the return node and output shape."""
        self.return_node = node_id
        self.return_shape = shape
        self.return_dtype = dtype
        # Update the output node's shape
        for n in self.nodes:
            if n.id == node_id:
                n.output = TensorDesc(shape=shape, dtype=dtype)

    def build(self) -> SemanticGraph:
        """Build and return the SemanticGraph."""
        graph = SemanticGraph(graph_id=self.name)

        for node in self.nodes:
            graph.add_node(node)
        for edge in self.edges:
            graph.add_edge(edge)

        # Auto-detect fusion groups
        self._detect_fusion_groups(graph)

        return graph

    def _detect_fusion_groups(self, graph: SemanticGraph) -> None:
        """Auto-detect epilogue fusion opportunities."""
        for edge in graph.edges:
            to_node = graph.get_node(edge.to_node)
            if to_node and is_fusable_epilogue(to_node.op):
                graph.add_fusion_group(FusionGroup(
                    id=f"fg_{edge.from_node}_{edge.to_node}",
                    nodes=[edge.from_node, edge.to_node],
                    fusion_type="epilogue",
                ))
