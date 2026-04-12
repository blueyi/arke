# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — AST to IR Converters.

Converts parser AST (from arke/lang/grammar.py) to SemanticIR and StrategyIR.
"""

from __future__ import annotations

from typing import Any

from arke.lang.ast import (
    Annotation,
    CompareCondition,
    BoolCondition,
    DimDecl,
    KernelDef,
    LetStmt,
    OpCall,
    Parameter,
    ReturnStmt,
    StrategyDef,
    StrategyStmt,
    TensorType,
    TupleType,
    InferType,
    WhenBlock,
    WhereClause,
)
from arke.ir.semantic import (
    Dim,
    Edge,
    InputRef,
    Node,
    Param,
    ParamRef,
    NodeRef,
    Semantics,
    SemanticIR,
    ShapeConstraint,
    SymbolicDim,
    TensorDesc,
)
from arke.ir.strategy import (
    ConditionalDecision,
    Decision,
    Rationale,
    StrategyIR,
)


def _extract_rationale(annotations: list[Annotation]) -> Rationale | None:
    """Extract @rationale annotation from a list of annotations."""
    for ann in annotations:
        if ann.name == "rationale":
            if ann.args:
                text = ann.args[0]
                if isinstance(text, tuple):
                    # keyword arg: ("text", "...") 
                    text = text[1]
                return Rationale(text=str(text))
    return None


def _tensor_type_to_dims(tt: TensorType) -> list[Dim]:
    """Convert AST TensorType shape to IR Dim list."""
    dims: list[Dim] = []
    for s in tt.shape:
        if isinstance(s, int):
            dims.append(s)
        elif isinstance(s, str):
            # symbolic dim reference — use SymbolicDim
            dims.append(SymbolicDim(name=s))
        else:
            dims.append(int(s))
    return dims


def _tensor_type_to_desc(tt: TensorType) -> TensorDesc:
    """Convert AST TensorType to IR TensorDesc."""
    return TensorDesc(
        shape=_tensor_type_to_dims(tt),
        dtype=tt.dtype.name,
        layout=tt.layout,
    )


def _get_op_semantics(op_name: str) -> Semantics:
    """Look up operator semantics from OpRegistry. Falls back to default."""
    try:
        from arke.ir.ops.registry import REGISTRY
        op = REGISTRY.get(op_name)
        return Semantics(
            computation=op.computation,
            index_vars=list(op.index_vars),
            reduction_axes=list(op.reduction_axes),
            properties=list(op.properties),
        )
    except (KeyError, ImportError):
        return Semantics(computation=f"{op_name}(...)")


def _resolve_output_shape(
    op_name: str,
    input_refs: dict[str, InputRef],
    attrs: dict[str, Any],
    param_map: dict[str, Parameter],
    node_map: dict[str, TensorDesc],
) -> TensorDesc:
    """Resolve output shape for an op.

    Uses the first tensor input's shape as default output shape.
    For matmul/batch_matmul, applies the matmul shape rule.
    """
    # Try shape inference from OpRegistry
    try:
        from arke.ir.ops.registry import REGISTRY
        op = REGISTRY.get(op_name)
        if op.shape_rule is not None:
            return _apply_shape_rule(
                op.shape_rule, input_refs, attrs, param_map, node_map
            )
    except (KeyError, ImportError):
        pass

    # Fallback: use first tensor input's shape and dtype
    first_desc = _get_first_input_desc(input_refs, param_map, node_map)
    if first_desc is not None:
        return TensorDesc(
            shape=list(first_desc.shape),
            dtype=first_desc.dtype,
            layout=first_desc.layout,
        )

    # Last resort
    return TensorDesc(shape=[], dtype="f16")


def _apply_shape_rule(
    shape_rule: Any,
    input_refs: dict[str, InputRef],
    attrs: dict[str, Any],
    param_map: dict[str, Parameter],
    node_map: dict[str, TensorDesc],
) -> TensorDesc:
    """Apply a ShapeRule to compute output TensorDesc."""
    kind = shape_rule.kind
    input_key = shape_rule.input_key

    # Get the reference tensor descriptor
    ref_desc = _get_input_desc(input_key, input_refs, param_map, node_map)
    if ref_desc is None:
        # Try fallback to first input
        ref_desc = _get_first_input_desc(input_refs, param_map, node_map)
    if ref_desc is None:
        return TensorDesc(shape=[], dtype="f16")

    if kind == "same_as_input":
        return TensorDesc(
            shape=list(ref_desc.shape),
            dtype=ref_desc.dtype,
            layout=ref_desc.layout,
        )
    elif kind == "matmul_rule":
        # [M,K] x [K,N] -> [M,N]
        a_desc = _get_input_desc("A", input_refs, param_map, node_map)
        b_desc = _get_input_desc("B", input_refs, param_map, node_map)
        if a_desc and b_desc and len(a_desc.shape) >= 2 and len(b_desc.shape) >= 2:
            return TensorDesc(
                shape=[a_desc.shape[0], b_desc.shape[-1]],
                dtype=a_desc.dtype,
                layout=a_desc.layout,
            )
        return TensorDesc(shape=list(ref_desc.shape), dtype=ref_desc.dtype)
    elif kind == "batch_matmul_rule":
        # [B,M,K] x [B,K,N] -> [B,M,N]
        a_desc = _get_input_desc("A", input_refs, param_map, node_map)
        b_desc = _get_input_desc("B", input_refs, param_map, node_map)
        if a_desc and b_desc and len(a_desc.shape) >= 3 and len(b_desc.shape) >= 3:
            return TensorDesc(
                shape=[a_desc.shape[0], a_desc.shape[1], b_desc.shape[-1]],
                dtype=a_desc.dtype,
                layout=a_desc.layout,
            )
        return TensorDesc(shape=list(ref_desc.shape), dtype=ref_desc.dtype)
    elif kind == "reduce_rule":
        # Drop reduction axes
        shape = list(ref_desc.shape)
        # shape_rule.axes may contain string axis names or ints
        # For simplicity, if attrs has 'axis', use that
        axis = attrs.get("axis", -1)
        if isinstance(axis, int) and shape:
            if axis < 0:
                axis = len(shape) + axis
            if 0 <= axis < len(shape):
                shape = shape[:axis] + shape[axis + 1:]
        return TensorDesc(shape=shape, dtype=ref_desc.dtype, layout=ref_desc.layout)
    else:
        # Default: same as input
        return TensorDesc(
            shape=list(ref_desc.shape),
            dtype=ref_desc.dtype,
            layout=ref_desc.layout,
        )


def _get_input_desc(
    input_name: str,
    input_refs: dict[str, InputRef],
    param_map: dict[str, Parameter],
    node_map: dict[str, TensorDesc],
) -> TensorDesc | None:
    """Get the TensorDesc for a named input."""
    ref = input_refs.get(input_name)
    if ref is None:
        return None
    if isinstance(ref, ParamRef):
        p = param_map.get(ref.name)
        if p:
            return _tensor_type_to_desc(p.type)
    elif isinstance(ref, NodeRef):
        return node_map.get(ref.id)
    return None


def _get_first_input_desc(
    input_refs: dict[str, InputRef],
    param_map: dict[str, Parameter],
    node_map: dict[str, TensorDesc],
) -> TensorDesc | None:
    """Get the TensorDesc of the first input."""
    for ref in input_refs.values():
        if isinstance(ref, ParamRef):
            p = param_map.get(ref.name)
            if p:
                return _tensor_type_to_desc(p.type)
        elif isinstance(ref, NodeRef):
            desc = node_map.get(ref.id)
            if desc:
                return desc
    return None


# ============================================================
# AST -> SemanticIR
# ============================================================

def ast_to_semantic(kernel_def: KernelDef) -> SemanticIR:
    """Convert a kernel AST to SemanticIR.

    Args:
        kernel_def: KernelDef AST node from the parser.

    Returns:
        SemanticIR v1.0 representation.
    """
    ir = SemanticIR(kernel_id=kernel_def.name)

    # Build param map for lookup
    ast_param_map: dict[str, Parameter] = {}

    # Process where clause -> symbolic dims
    sym_dim_set: set[str] = set()
    if kernel_def.where_clause:
        for dim_decl in kernel_def.where_clause.dims:
            sd = SymbolicDim(
                name=dim_decl.name,
                min=dim_decl.opts.get("min"),
                max=dim_decl.opts.get("max"),
                is_static=(dim_decl.kind == "static"),
                multiple_of=dim_decl.opts.get("multiple_of"),
                default=dim_decl.opts.get("default"),
            )
            ir.add_symbolic_dim(sd)
            sym_dim_set.add(dim_decl.name)

            if dim_decl.kind == "static":
                ir.add_shape_constraint(
                    ShapeConstraint(
                        expr=f"{dim_decl.name} is static",
                        reason=f"where {dim_decl.name}: static",
                    )
                )
            if "min" in dim_decl.opts:
                ir.add_shape_constraint(
                    ShapeConstraint(
                        expr=f"{dim_decl.name} >= {dim_decl.opts['min']}",
                        reason=f"where {dim_decl.name} min bound",
                    )
                )
            if "max" in dim_decl.opts:
                ir.add_shape_constraint(
                    ShapeConstraint(
                        expr=f"{dim_decl.name} <= {dim_decl.opts['max']}",
                        reason=f"where {dim_decl.name} max bound",
                    )
                )
            if "multiple_of" in dim_decl.opts:
                ir.add_shape_constraint(
                    ShapeConstraint(
                        expr=f"{dim_decl.name} % {dim_decl.opts['multiple_of']} == 0",
                        reason=f"where {dim_decl.name} alignment",
                    )
                )
            if "default" in dim_decl.opts:
                ir.add_shape_constraint(
                    ShapeConstraint(
                        expr=f"default({dim_decl.name}) == {dim_decl.opts['default']}",
                        reason=f"where {dim_decl.name} default value",
                    )
                )

    # Process params
    for p in kernel_def.params:
        dims = _tensor_type_to_dims(p.type)
        ir_param = Param(
            name=p.name,
            shape=dims,
            dtype=p.type.dtype.name,
            layout=p.type.layout,
        )
        ir.add_param(ir_param)
        ast_param_map[p.name] = p

    # Also collect symbolic dims from param shapes that aren't in where clause
    for p in kernel_def.params:
        for s in p.type.shape:
            if isinstance(s, str) and s not in sym_dim_set:
                ir.add_symbolic_dim(SymbolicDim(name=s))
                sym_dim_set.add(s)

    # Collect name -> binding for resolving data flow
    # Maps variable name -> (source_type, source_id)
    #   source_type: "param" or "node"
    name_map: dict[str, tuple[str, str]] = {}
    for p in kernel_def.params:
        name_map[p.name] = ("param", p.name)

    # Node output map: node_id -> TensorDesc
    node_output_map: dict[str, TensorDesc] = {}
    node_counter = 0

    # Process body statements
    for stmt in kernel_def.body:
        if isinstance(stmt, ReturnStmt):
            # Set return node
            val = stmt.value
            if isinstance(val, list):
                # tuple return — use the last node
                if val:
                    last = val[-1]
                    src = name_map.get(str(last))
                    if src and src[0] == "node":
                        ir.return_node = src[1]
                    ir.return_ports = [str(v) for v in val]
            else:
                val_str = str(val) if not isinstance(val, str) else val
                src = name_map.get(val_str)
                if src:
                    if src[0] == "node":
                        ir.return_node = src[1]
                    elif src[0] == "param":
                        # Return directly from param — need an identity node
                        ir.return_node = val_str
            continue

        if not isinstance(stmt, LetStmt):
            continue
        lhs_name = stmt.lhs if isinstance(stmt.lhs, str) else "_".join(stmt.lhs)
        op_call = stmt.op_call

        # Build node
        node_id = f"{op_call.op}_{node_counter}"
        node_counter += 1

        # Build inputs dict from OpCall args
        inputs: dict[str, InputRef] = {}
        attrs: dict[str, Any] = {}

        for arg_name, arg_val in op_call.args:
            if isinstance(arg_val, str) and arg_val in name_map:
                # This is a tensor reference
                src_type, src_id = name_map[arg_val]
                if src_type == "param":
                    inputs[arg_name] = ParamRef(name=src_id)
                else:
                    inputs[arg_name] = NodeRef(id=src_id)
            else:
                # This is an attribute value (int, float, bool, string, list)
                attrs[arg_name] = arg_val

        for k, v in op_call.kwargs.items():
            if isinstance(v, str) and v in name_map:
                src_type, src_id = name_map[v]
                if src_type == "param":
                    inputs[k] = ParamRef(name=src_id)
                else:
                    inputs[k] = NodeRef(id=src_id)
            else:
                attrs[k] = v

        # Get semantics from registry
        semantics = _get_op_semantics(op_call.op)

        # Resolve output shape
        output_desc = _resolve_output_shape(
            op_call.op, inputs, attrs, ast_param_map, node_output_map
        )

        # Create node
        node = Node(
            id=node_id,
            op=op_call.op,
            inputs=inputs,
            output=output_desc,
            semantics=semantics,
            attrs=attrs,
        )
        ir.add_node(node)

        # Register node output
        node_output_map[node_id] = output_desc

        # Map lhs name to node
        if isinstance(stmt, LetStmt) and isinstance(stmt.lhs, list):
            # Tuple destructuring — all names map to the same node
            for n in stmt.lhs:
                name_map[n] = ("node", node_id)
        else:
            name_map[lhs_name] = ("node", node_id)

        # Build edges from inputs
        for inp_name, inp_ref in inputs.items():
            if isinstance(inp_ref, ParamRef):
                from_id = f"param:{inp_ref.name}"
                tensor = inp_ref.name
            else:
                from_id = inp_ref.id
                tensor = inp_ref.id
            ir.add_edge(Edge(
                from_node=from_id,
                to_node=node_id,
                tensor_name=tensor,
            ))

    return ir


# ============================================================
# AST -> StrategyIR
# ============================================================

def _condition_to_predicate(cond: Any) -> str:
    """Convert an AST Condition to a predicate string."""
    if isinstance(cond, CompareCondition):
        return f'{cond.ident} {cond.op} {cond.value}'
    elif isinstance(cond, BoolCondition):
        left = _condition_to_predicate(cond.left)
        right = _condition_to_predicate(cond.right)
        return f'({left}) {cond.op} ({right})'
    return str(cond)


def _strategy_stmt_to_decision(stmt: StrategyStmt) -> Decision:
    """Convert a StrategyStmt AST node to a Decision."""
    rationale = _extract_rationale(stmt.annotations)

    kind = stmt.directive
    params = dict(stmt.kwargs)
    if kind == "tile" and "loop" not in params and "dim" in params:
        params["loop"] = params.pop("dim")
    level = 1

    # In v2, `compute(...)` is the only canonical resource-bearing directive.
    if kind == "compute":
        level = 2

    return Decision(
        kind=kind,
        params=params,
        rationale=rationale,
        level=level,
    )


def _process_strategy_body(
    body: list, ir: StrategyIR
) -> None:
    """Process a list of strategy body items (StrategyStmt / WhenBlock)."""
    for item in body:
        if isinstance(item, StrategyStmt):
            decision = _strategy_stmt_to_decision(item)
            ir.add_decision(decision)
        elif isinstance(item, WhenBlock):
            # Convert WhenBlock arms to ConditionalDecisions
            for cond, arm_body in item.arms:
                predicate = _condition_to_predicate(cond)
                true_decisions = []
                for arm_stmt in arm_body:
                    if isinstance(arm_stmt, StrategyStmt):
                        true_decisions.append(_strategy_stmt_to_decision(arm_stmt))

                false_decisions = []
                if item.otherwise_body:
                    for ow_stmt in item.otherwise_body:
                        if isinstance(ow_stmt, StrategyStmt):
                            false_decisions.append(_strategy_stmt_to_decision(ow_stmt))

                cd = ConditionalDecision(
                    predicate=predicate,
                    true_decisions=true_decisions,
                    false_decisions=false_decisions,
                )
                ir.add_decision(cd)


def ast_to_strategy(strategy_def: StrategyDef) -> StrategyIR:
    """Convert a strategy AST to StrategyIR.

    Args:
        strategy_def: StrategyDef AST node from the parser.

    Returns:
        StrategyIR v1.0 representation.
    """
    kernel_id = strategy_def.name
    if kernel_id.endswith("_strategy"):
        kernel_id = kernel_id[: -len("_strategy")]

    ir = StrategyIR(
        kernel_id=kernel_id,
        target_hw=strategy_def.target,
    )

    _process_strategy_body(strategy_def.body, ir)

    return ir
