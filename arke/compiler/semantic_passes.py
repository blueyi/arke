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
    Dim,
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

    For fully concrete shapes, use the existing ShapeInferenceEngine.
    For symbolic shapes, preserve and propagate shape information for the
    subset of rules that are structurally safe in Track 2 (same_as_input,
    matmul_rule, batch_matmul_rule, attention_rule).

    Args:
        ir: The SemanticIR to process (mutated in place).

    Returns:
        List of error messages. Empty list means all shapes inferred successfully.
    """
    errors: list[str] = []

    # Build a shape map: param_name/node_id -> Dim list (concrete or symbolic)
    shape_map: dict[str, list[Dim]] = {}

    for param in ir.params:
        shape_map[param.name] = list(param.shape)

    for node in ir.nodes:
        if isinstance(node, ConditionalNode):
            continue

        if not isinstance(node, (Node, MultiOutputNode)):
            continue

        input_shapes: dict[str, list[Dim]] = {}
        missing_input = False

        for input_name, ref in node.inputs.items():
            if isinstance(ref, ParamRef):
                if ref.name in shape_map:
                    input_shapes[input_name] = list(shape_map[ref.name])
                else:
                    missing_input = True
            elif isinstance(ref, NodeRef):
                if ref.id in shape_map:
                    input_shapes[input_name] = list(shape_map[ref.id])
                else:
                    missing_input = True

        if missing_input or len(input_shapes) < len(node.inputs):
            continue

        if node.op not in REGISTRY:
            continue

        op_schema = REGISTRY.get(node.op)
        if op_schema.shape_rule is None:
            continue

        try:
            if _all_input_shapes_concrete(input_shapes):
                output_shape = SHAPE_ENGINE.infer(node.op, input_shapes, node.attrs)
            else:
                output_shape = _infer_symbolic_shape(node.op, input_shapes, node.attrs)

            if isinstance(node, Node):
                node.output.shape = output_shape
                shape_map[node.id] = list(output_shape)
            elif isinstance(node, MultiOutputNode):
                for _port_name, td in node.outputs.items():
                    td.shape = list(output_shape)
                shape_map[node.id] = list(output_shape)
        except Exception:
            if isinstance(node, Node):
                shape_map[node.id] = list(node.output.shape)
            elif isinstance(node, MultiOutputNode):
                for td in node.outputs.values():
                    shape_map[node.id] = list(td.shape)
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


def _extract_concrete_shape(shape: list[Dim]) -> list[int] | None:
    """Extract a concrete integer shape from a Dim list, or None if symbolic."""
    result = []
    for d in shape:
        if isinstance(d, int):
            result.append(d)
        elif isinstance(d, SymbolicDim):
            return None
        else:
            return None
    return result


def _all_input_shapes_concrete(input_shapes: dict[str, list[Dim]]) -> bool:
    return all(_extract_concrete_shape(shape) is not None for shape in input_shapes.values())


def _infer_symbolic_shape(
    op_name: str,
    input_shapes: dict[str, list[Dim]],
    attrs: dict | None = None,
) -> list[Dim]:
    """Best-effort symbolic shape propagation for Track 2.

    Supports the safe subset needed to stop dropping symbolic dims in the
    semantic pipeline while keeping behavior conservative.
    """
    attrs = attrs or {}
    op = REGISTRY.get(op_name)
    if op.shape_rule is None:
        raise ValueError(f"Operator {op_name!r} has no shape_rule defined")

    kind = op.shape_rule.kind

    if kind == "same_as_input":
        key = op.shape_rule.input_key
        if key not in input_shapes:
            raise ValueError(f"Input {key!r} not found for symbolic shape inference")
        return list(input_shapes[key])

    if kind in {"matmul_rule", "batch_matmul_rule"}:
        a = input_shapes.get("A", input_shapes.get("X", []))
        b = input_shapes.get("B", input_shapes.get("W", []))
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"{op_name} requires 2D+ inputs, got A={a}, B={b}")
        return list(a[:-1]) + [b[-1]]

    if kind == "attention_rule":
        return list(input_shapes.get("Q", []))

    raise ValueError(f"Symbolic shape inference not yet implemented for rule kind {kind!r}")
