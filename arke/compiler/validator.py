# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — SemanticIR Validation Passes.

Validates SemanticIR structural integrity before execution.
Equivalent to S6's SSAValidationPass but for the multi-layer IR.
"""

from __future__ import annotations

from arke.ir.ops.registry import REGISTRY
from arke.ir.semantic import (
    ConditionalNode,
    MultiOutputNode,
    Node,
    NodeRef,
    ParamRef,
    SemanticIR,
)


def validate_semantic_ir(ir: SemanticIR) -> list[str]:
    """Validate a SemanticIR for structural correctness.

    Checks performed:
    1. No duplicate node IDs
    2. All node input refs point to valid params or previously-defined nodes
    3. All ops exist in OpRegistry
    4. return_node points to a valid node or param
    5. No cycles in data flow (ensured by forward-reference check)

    Args:
        ir: The SemanticIR to validate.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    # Collect param names
    param_names: set[str] = {p.name for p in ir.params}

    # 1. Check for duplicate node IDs
    seen_ids: set[str] = set()
    for node in ir.nodes:
        if node.id in seen_ids:
            errors.append(f"Duplicate node ID: {node.id!r}")
        seen_ids.add(node.id)

    # 2 & 3. Walk nodes in order — check refs and ops
    defined_nodes: set[str] = set()  # nodes defined so far (forward-only)

    for node in ir.nodes:
        # 3. Check op exists in registry (skip special ops)
        if isinstance(node, (Node, MultiOutputNode)):
            if node.op not in REGISTRY:
                errors.append(
                    f"Node {node.id!r}: unknown op {node.op!r} "
                    f"(not in OpRegistry)"
                )

        # 2. Check input refs
        if isinstance(node, (Node, MultiOutputNode)):
            for input_name, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    if ref.name not in param_names:
                        errors.append(
                            f"Node {node.id!r}, input {input_name!r}: "
                            f"ParamRef {ref.name!r} not found in params "
                            f"(available: {sorted(param_names)})"
                        )
                elif isinstance(ref, NodeRef):
                    if ref.id not in defined_nodes:
                        if ref.id in seen_ids:
                            errors.append(
                                f"Node {node.id!r}, input {input_name!r}: "
                                f"NodeRef {ref.id!r} references a node "
                                f"defined later (forward reference / cycle)"
                            )
                        else:
                            errors.append(
                                f"Node {node.id!r}, input {input_name!r}: "
                                f"NodeRef {ref.id!r} not found in any "
                                f"defined node"
                            )

        elif isinstance(node, ConditionalNode):
            # Check branch refs
            for branch_name, branch_ids in [
                ("true_branch", node.true_branch),
                ("false_branch", node.false_branch),
            ]:
                for ref_id in branch_ids:
                    if ref_id not in defined_nodes and ref_id not in seen_ids:
                        errors.append(
                            f"ConditionalNode {node.id!r}, {branch_name}: "
                            f"references unknown node {ref_id!r}"
                        )

        # Mark this node as defined
        defined_nodes.add(node.id)

    # 4. Check return_node
    if ir.return_node:
        if (
            ir.return_node not in defined_nodes
            and ir.return_node not in param_names
        ):
            errors.append(
                f"return_node {ir.return_node!r} not found in "
                f"defined nodes or params"
            )

    return errors
