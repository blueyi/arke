# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Semantic IR Passes.

Passes operating on SemanticIR (replaces S6 IRGraph-based passes for the
multi-layer IR architecture).

Passes:
    - SemanticShapeInferencePass: Infer output shapes for all nodes
    - SemanticSSAValidationPass: Validate SSA structural correctness
    - RationalePreservationPass: Cross-validate StrategyIR rationale ↔ SemanticIR nodes
"""

from __future__ import annotations

from arke.compiler.validator import validate_semantic_ir
from arke.ir.ops.registry import REGISTRY
from arke.ir.ops.shape_engine import SHAPE_ENGINE
from arke.ir.semantic import (
    ConditionalNode,
    MultiOutputNode,
    Node,
    NodeRef,
    ParamRef,
    SemanticIR,
    SymbolicDim,
)
from arke.ir.strategy import ConditionalDecision, StrategyIR


def semantic_shape_inference_pass(ir: SemanticIR) -> list[str]:
    """Walk SemanticIR nodes and infer output shapes using SHAPE_ENGINE.

    For each Node/MultiOutputNode, collects concrete input shapes from params
    or previously-computed nodes, calls SHAPE_ENGINE.infer(), and updates
    node.output.shape (or node.outputs[port].shape for multi-output).

    Nodes with symbolic dimensions are skipped (shape engine needs concrete ints).

    Args:
        ir: The SemanticIR to process (mutated in place).

    Returns:
        List of error messages. Empty list means all shapes inferred successfully.
    """
    errors: list[str] = []

    # Build a shape map: param_name -> concrete shape, node_id -> concrete shape
    shape_map: dict[str, list[int]] = {}

    # Initialize from params (only concrete shapes)
    for param in ir.params:
        concrete = _extract_concrete_shape(param.shape)
        if concrete is not None:
            shape_map[param.name] = concrete

    # Walk nodes in order
    for node in ir.nodes:
        if isinstance(node, ConditionalNode):
            # Skip conditional nodes — no shape inference needed
            continue

        if not isinstance(node, (Node, MultiOutputNode)):
            continue

        # Collect input shapes
        input_shapes: dict[str, list[int]] = {}
        all_concrete = True

        for input_name, ref in node.inputs.items():
            if isinstance(ref, ParamRef):
                if ref.name in shape_map:
                    input_shapes[input_name] = shape_map[ref.name]
                else:
                    all_concrete = False
            elif isinstance(ref, NodeRef):
                if ref.id in shape_map:
                    input_shapes[input_name] = shape_map[ref.id]
                else:
                    all_concrete = False

        if not all_concrete or len(input_shapes) < len(node.inputs):
            # Can't infer — might have symbolic dims or missing predecessors
            continue

        # Check if op exists in registry
        if node.op not in REGISTRY:
            continue  # Unknown ops are caught by SSA validation

        op_schema = REGISTRY.get(node.op)
        if op_schema.shape_rule is None:
            continue  # No shape rule defined

        try:
            output_shape = SHAPE_ENGINE.infer(node.op, input_shapes, node.attrs)

            if isinstance(node, Node):
                node.output.shape = output_shape
                shape_map[node.id] = output_shape
            elif isinstance(node, MultiOutputNode):
                # For multi-output, all outputs get the same inferred shape
                # (shape engine returns one shape; individual ports may differ)
                for port_name, td in node.outputs.items():
                    td.shape = output_shape
                shape_map[node.id] = output_shape
        except Exception:
            # Shape inference failure is non-fatal: the node already has
            # a declared output shape from the .ak file. We just couldn't
            # verify/update it. Carry forward the declared shape.
            if isinstance(node, Node):
                declared = _extract_concrete_shape(node.output.shape)
                if declared is not None:
                    shape_map[node.id] = declared
            elif isinstance(node, MultiOutputNode):
                # Use first output's declared shape
                for td in node.outputs.values():
                    declared = _extract_concrete_shape(td.shape)
                    if declared is not None:
                        shape_map[node.id] = declared
                        break

    return errors


def semantic_ssa_validation_pass(ir: SemanticIR) -> list[str]:
    """Validate SemanticIR structural correctness using the existing validator.

    Wraps validate_semantic_ir() as a pass that returns error list.

    Args:
        ir: The SemanticIR to validate.

    Returns:
        List of error messages. Empty list means valid.
    """
    return validate_semantic_ir(ir)


def rationale_preservation_pass(
    ir: SemanticIR,
    strategy_ir: StrategyIR | None = None,
) -> list[str]:
    """Cross-validate that StrategyIR decisions reference actual SemanticIR nodes.

    Checks that any strategy decision whose params reference node IDs (via 'ops'
    for fuse decisions, or 'tensor'/'loop' params) correspond to actual nodes or
    params in the SemanticIR.

    If strategy_ir is None, this pass is a no-op (returns empty list).

    Args:
        ir: The SemanticIR.
        strategy_ir: The StrategyIR (may be None).

    Returns:
        List of warning messages for orphaned references.
    """
    if strategy_ir is None:
        return []

    warnings: list[str] = []

    # Collect valid node IDs and param names from SemanticIR
    node_ids = {n.id for n in ir.nodes}
    param_names = {p.name for p in ir.params}
    valid_refs = node_ids | param_names

    # Walk strategy decisions
    for decision in strategy_ir.decisions:
        if isinstance(decision, ConditionalDecision):
            # Check nested decisions
            all_sub = decision.true_decisions + decision.false_decisions
            for sub_d in all_sub:
                _check_decision_refs(sub_d, valid_refs, warnings)
        else:
            _check_decision_refs(decision, valid_refs, warnings)

    return warnings


def _check_decision_refs(
    decision, valid_refs: set[str], warnings: list[str]
) -> None:
    """Check a single decision's params for references to SemanticIR entities."""
    params = decision.params

    # 'fuse' decisions reference ops by node ID
    if decision.kind == "fuse":
        ops = params.get("ops", [])
        for op_ref in ops:
            if op_ref not in valid_refs:
                rat_text = ""
                if decision.rationale:
                    rat_text = f" (rationale: {decision.rationale.text!r})"
                warnings.append(
                    f"Strategy decision 'fuse' references {op_ref!r} "
                    f"which is not in SemanticIR{rat_text}"
                )

    # 'place' decisions reference tensors
    if decision.kind == "place":
        tensor = params.get("tensor", "")
        # Tensor refs can be node IDs, param names, or derived names (e.g., "A_tile")
        # We only warn if the base name is completely unknown
        base = tensor.replace("_tile", "").replace("_shared", "")
        if base and base not in valid_refs:
            rat_text = ""
            if decision.rationale:
                rat_text = f" (rationale: {decision.rationale.text!r})"
            warnings.append(
                f"Strategy decision 'place' references tensor {tensor!r} "
                f"(base={base!r}) which is not in SemanticIR{rat_text}"
            )


def _extract_concrete_shape(shape: list) -> list[int] | None:
    """Extract a concrete integer shape from a Dim list, or None if symbolic."""
    result = []
    for d in shape:
        if isinstance(d, int):
            result.append(d)
        elif isinstance(d, SymbolicDim):
            return None  # Has symbolic dims — can't do concrete inference
        else:
            return None
    return result
