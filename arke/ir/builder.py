# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Builder.

Fluent API for constructing SemanticIR from Python, without writing raw JSON.
"""

from __future__ import annotations

from arke.ir.ops.catalog import get_op, is_fusable_epilogue
from arke.ir.semantic import (
    Edge,
    FusionGroup,
    InputRef,
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    TensorDesc,
)
from arke.ir.shape_inference import infer_output_dtype, infer_output_shape


class KernelBuilder:
    """Fluent builder for constructing a SemanticIR.

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
        """Initialize a kernel builder with the given kernel name."""
        self.name = name
        self._params: list[Param] = []
        self._nodes: list[Node] = []
        self._edges: list[Edge] = []
        self._return_node: str | None = None
        self._return_shape: list[int] = []
        self._return_dtype: str = "f16"
        self._node_count = 0

    def param(self, name: str, shape: list[int], dtype: str,
              layout: str = "row_major") -> str:
        """Add an input parameter. Returns the parameter name for reference."""
        self._params.append(Param(
            name=name, shape=shape, dtype=dtype, layout=layout,
        ))
        return name

    def op(self, op_name: str, **inputs: str) -> str:
        """Add an operator node. Returns the node ID for chaining.

        Inputs can be parameter names (str) or prior node IDs (str).
        """
        node_id = f"{op_name}_{self._node_count}"
        self._node_count += 1

        op_def = get_op(op_name)

        # Build input dict: resolve parameter refs vs node refs
        # Non-string values (int, float, list, bool) are op attributes, skip
        resolved_inputs: dict[str, InputRef] = {}
        param_names = {p.name for p in self._params}
        for key, ref in inputs.items():
            if not isinstance(ref, str):
                # Scalar/attribute parameter (e.g., axis=-1, dims=[2,3])
                # Store as-is; these are op config, not tensor references
                continue
            if ref.startswith('__const_'):
                # Constant reference from literal assignment
                continue
            if ref in param_names:
                resolved_inputs[key] = ParamRef(name=ref)
            else:
                resolved_inputs[key] = NodeRef(id=ref)
                # Add edge from referenced node
                self._edges.append(Edge(
                    from_node=ref, to_node=node_id, tensor_name=f"{ref}_out",
                ))

        # Infer output shape (simplified — real impl needs proper shape inference)
        out_shape, out_dtype = self._infer_output(op_name, resolved_inputs)

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
        self._nodes.append(node)
        return node_id

    def returns(self, node_id: str, shape: list[int], dtype: str) -> None:
        """Set the return node and output shape."""
        self._return_node = node_id
        self._return_shape = shape
        self._return_dtype = dtype
        # Update the output node's shape
        for n in self._nodes:
            if n.id == node_id:
                n.output = TensorDesc(shape=shape, dtype=dtype)

    def build(self) -> SemanticIR:
        """Build and return the SemanticIR."""
        ir = SemanticIR(
            kernel_id=self.name,
            params=list(self._params),
            return_type=TensorDesc(
                shape=self._return_shape,
                dtype=self._return_dtype,
            ) if self._return_shape else None,
            return_node=self._return_node or "",
        )

        for node in self._nodes:
            ir.add_node(node)
        for edge in self._edges:
            ir.add_edge(edge)

        # Auto-detect fusion groups
        self._detect_fusion_groups(ir)

        return ir

    def _infer_output(self, op_name: str,
                      inputs: dict[str, InputRef]) -> tuple[list[int], str]:
        """Infer output shape and dtype using the shape inference module."""
        # Resolve shapes and dtypes from inputs
        input_shapes: dict[str, list[int]] = {}
        input_dtypes: dict[str, str] = {}

        for key, ref in inputs.items():
            if isinstance(ref, ParamRef):
                p = next((p for p in self._params if p.name == ref.name), None)
                if p:
                    input_shapes[key] = list(p.shape)
                    input_dtypes[key] = p.dtype
            elif isinstance(ref, NodeRef):
                n = next((n for n in self._nodes if n.id == ref.id), None)
                if n:
                    input_shapes[key] = list(n.output.shape)
                    input_dtypes[key] = n.output.dtype

        if not input_shapes:
            return [1], "f32"

        shape = infer_output_shape(op_name, input_shapes)
        dtype = infer_output_dtype(op_name, input_dtypes)
        return shape, dtype

    def _detect_fusion_groups(self, ir: SemanticIR) -> None:
        """Auto-detect epilogue fusion opportunities."""
        for edge in ir.edges:
            to_node = ir.get_node(edge.to_node)
            if to_node and is_fusable_epilogue(to_node.op):
                ir.add_fusion_group(FusionGroup(
                    id=f"fg_{edge.from_node}_{edge.to_node}",
                    nodes=[edge.from_node, edge.to_node],
                    fusion_type="epilogue",
                    reason=f"{to_node.op} is elementwise; can fuse into epilogue",
                ))
