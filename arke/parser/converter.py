# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convert Arke AST to Semantic IR.

Usage:
    from arke.parser.converter import ast_to_ir
    program = parse_string(source)
    ir = ast_to_ir(program.kernels[0])
"""

from __future__ import annotations

from arke.ir.builder import KernelBuilder
from arke.ir.ops.catalog import OP_CATALOG
from arke.ir.semantic import SemanticIR
from arke.parser.ast_nodes import (
    Literal,
    KernelDef,
    LetStmt,
    ReturnStmt,
)


class ConversionError(Exception):
    """Error converting AST to IR."""


def ast_to_ir(kernel: KernelDef) -> SemanticIR:
    """Convert a KernelDef AST node to SemanticIR.

    The conversion uses KernelBuilder to construct the IR,
    ensuring identical output to programmatic construction.

    Args:
        kernel: Parsed kernel definition

    Returns:
        SemanticIR equivalent to the kernel definition

    Raises:
        ConversionError: If the AST is invalid or unsupported
    """
    builder = KernelBuilder(kernel.name)

    # 1. Register parameters
    for param in kernel.params:
        builder.param(
            param.name,
            param.type.shape,
            param.type.dtype,
        )

    # 2. Process body statements
    # Track variable names → node IDs for resolving references
    var_to_node: dict[str, str] = {}

    # Parameters are also valid variable references
    for param in kernel.params:
        var_to_node[param.name] = param.name

    return_var = None

    for stmt in kernel.body:
        if isinstance(stmt, LetStmt):
            # Handle literal assignments (let scale = 0.125)
            if isinstance(stmt.value, Literal):
                # Store literal as a constant reference
                var_to_node[stmt.name] = f"__const_{stmt.name}"
                continue
            node_id = _process_let(builder, stmt, var_to_node)
            var_to_node[stmt.name] = node_id

        elif isinstance(stmt, ReturnStmt):
            return_var = stmt.name

    # 3. Set return
    if return_var is None:
        raise ConversionError(
            f"Kernel '{kernel.name}' has no return statement"
        )

    if return_var not in var_to_node:
        raise ConversionError(
            f"Return variable '{return_var}' not defined "
            f"in kernel '{kernel.name}'"
        )

    ret_node = var_to_node[return_var]
    builder.returns(
        ret_node,
        kernel.return_type.shape,
        kernel.return_type.dtype,
    )

    return builder.build()


def _process_let(
    builder: KernelBuilder,
    stmt: LetStmt,
    var_to_node: dict[str, str],
) -> str:
    """Process a let statement, returning the node ID."""
    op_call = stmt.value

    # Resolve positional args using op catalog input names
    positional = []
    named = {}
    for key, var_name in op_call.args.items():
        if key.startswith("_pos_"):
            positional.append(var_name)
        else:
            named[key] = var_name

    # If there are positional args, map them to op input names
    if positional:
        op_def = OP_CATALOG.get(op_call.op)
        if op_def is None:
            raise ConversionError(
                f"Unknown operator '{op_call.op}'"
            )
        input_names = list(op_def.inputs.keys())
        if len(positional) > len(input_names):
            raise ConversionError(
                f"Too many positional args for '{op_call.op}': "
                f"got {len(positional)}, expected ≤{len(input_names)}"
            )
        for i, var_name in enumerate(positional):
            named[input_names[i]] = var_name

    # Resolve variable references to node IDs
    resolved = {}
    for key, var_name in named.items():
        # Non-string values (int, float, list, bool) are literal parameters, pass through
        if not isinstance(var_name, str):
            resolved[key] = var_name
        elif var_name == "true":
            resolved[key] = True
        elif var_name == "false":
            resolved[key] = False
        elif var_name in var_to_node:
            resolved[key] = var_to_node[var_name]
        elif var_name.startswith('__const_') or any(
            v == f'__const_{var_name}' for v in var_to_node.values()
        ):
            # Constant reference — look up the constant
            const_key = f'__const_{var_name}'
            if const_key in var_to_node.values():
                resolved[key] = var_name  # pass through as constant ref
            else:
                resolved[key] = var_name
        else:
            raise ConversionError(
                f"Undefined variable '{var_name}' in "
                f"op call '{op_call.op}'"
            )

    return builder.op(op_call.op, **resolved)
