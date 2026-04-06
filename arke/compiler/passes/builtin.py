# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Built-in Passes (S6 Track 2, Tasks C2.2-C2.4).

C2.2: ShapeInferencePass — infers output shapes for all nodes
C2.3: SSAValidationPass — validates SSA form, type consistency, op existence
C2.4: RationalePreservationPass — preserves @rationale annotations
"""

from __future__ import annotations

from arke.compiler.passes.base import ArkePass, PassContext, PassResult, Severity
from arke.ir.ops.shape_engine import SHAPE_ENGINE


class ShapeInferencePass:
    """Infers output shapes for all nodes in the IR graph.

    For each node, looks up the op's shape_rule via OpRegistry
    and computes the output shape from input shapes.

    Stores results in ctx.artifacts["shape_map"]: {value_name: [dims]}
    """

    name = "ShapeInference"

    def run(self, ctx: PassContext) -> PassResult:
        shape_map: dict[str, list[int]] = {}

        # Initialize with graph input shapes
        for inp_name in ctx.graph.graph_inputs:
            v = ctx.graph.get_value(inp_name)
            if v.shape:
                shape_map[inp_name] = list(v.shape)

        # Infer shapes node by node (topological order = list order in SSA)
        for node in ctx.graph.nodes:
            # Build input shapes for this node
            input_shapes = {}
            for op_input_name, value_name in node.inputs.items():
                if value_name in shape_map:
                    input_shapes[op_input_name] = shape_map[value_name]
                else:
                    ctx.add_warning(
                        self.name,
                        f"Input {value_name!r} has no known shape",
                        node.id,
                    )

            # Skip if we don't have all input shapes
            if len(input_shapes) < len(node.inputs):
                for out in node.outputs:
                    shape_map[out] = []
                continue

            try:
                output_shape = SHAPE_ENGINE.infer(node.op, input_shapes, node.attrs)
                for out in node.outputs:
                    shape_map[out] = output_shape
                    # Update the value in the graph
                    if out in ctx.graph.values:
                        ctx.graph.values[out].shape = output_shape
            except Exception as e:
                ctx.add_error(self.name, f"Shape inference failed for op={node.op!r}: {e}", node.id)
                return PassResult.fail(str(e))

        ctx.artifacts["shape_map"] = shape_map
        return PassResult.ok(modified=True)


class SSAValidationPass:
    """Validates IR graph is well-formed SSA.

    Checks:
    1. All ops exist in OpRegistry
    2. No duplicate value definitions
    3. All used values are defined before use
    4. Graph outputs are defined
    5. No self-referential nodes (cycles at node level)
    """

    name = "SSAValidation"

    def run(self, ctx: PassContext) -> PassResult:
        errors = 0

        # Track defined values
        defined: set[str] = set(ctx.graph.graph_inputs)

        # Track all definitions for duplicate detection
        all_defs: dict[str, str] = {}  # value_name -> defining node_id or "input"
        for inp in ctx.graph.graph_inputs:
            all_defs[inp] = "input"

        for node in ctx.graph.nodes:
            # 1. Check op exists in registry
            if node.op not in ctx.registry:
                ctx.add_error(self.name, f"Unknown op: {node.op!r}", node.id)
                errors += 1

            # 2. Check all inputs are defined
            for inp_name, value_name in node.inputs.items():
                if value_name not in defined:
                    ctx.add_error(
                        self.name,
                        f"Undefined value: {value_name!r} (input {inp_name!r})",
                        node.id,
                    )
                    errors += 1

            # 3. Check no self-reference (output used as own input)
            output_set = set(node.outputs)
            input_values = set(node.inputs.values())
            self_refs = output_set & input_values
            if self_refs:
                ctx.add_error(
                    self.name,
                    f"Self-referential node: {self_refs}",
                    node.id,
                )
                errors += 1

            # 4. Check no duplicate definitions
            for out in node.outputs:
                if out in all_defs:
                    ctx.add_error(
                        self.name,
                        f"Duplicate definition: {out!r} (already defined by {all_defs[out]!r})",
                        node.id,
                    )
                    errors += 1
                all_defs[out] = node.id
                defined.add(out)

        # 5. Check graph outputs are defined
        for out_name in ctx.graph.graph_outputs:
            if out_name not in defined:
                ctx.add_error(self.name, f"Graph output {out_name!r} is not defined")
                errors += 1

        if errors > 0:
            return PassResult.fail(f"SSA validation found {errors} error(s)")

        return PassResult.ok()


class RationalePreservationPass:
    """Preserves @rationale annotations through the compilation pipeline.

    Collects all rationale strings from IR nodes and stores them
    in ctx.artifacts["rationale_map"]: {node_id: rationale_text}
    """

    name = "RationalePreservation"

    def run(self, ctx: PassContext) -> PassResult:
        rationale_map: dict[str, str] = {}

        for node in ctx.graph.nodes:
            if node.rationale:
                rationale_map[node.id] = node.rationale

        ctx.artifacts["rationale_map"] = rationale_map

        if rationale_map:
            ctx.add_info(
                self.name,
                f"Preserved {len(rationale_map)} @rationale annotation(s)",
            )

        return PassResult.ok()
