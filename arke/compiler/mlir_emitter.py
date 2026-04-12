# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal MLIR skeleton emitter for Stage 7 Track 4.

This module bridges the active multi-layer Arke IR stack to a verifiable MLIR
text skeleton path. It does not attempt full semantic lowering; it only proves
that BL1-style examples can traverse:

    SemanticIR -> StrategyIR -> ScheduleIR -> InstructionIR -> MLIR skeleton

The output is intentionally skeletal and suitable for tests / architecture
validation rather than code generation.
"""

from __future__ import annotations

from arke.ir.instruction import InstructionIR
from arke.ir.semantic import SemanticIR


def emit_mlir_skeleton(
    semantic_ir: SemanticIR,
    instruction_ir: InstructionIR | None,
) -> str:
    """Emit a minimal MLIR module for the active Arke IR stack.

    The skeleton contains:
    - module + func.func wrapper
    - comments describing SemanticIR nodes and Layer-1 instructions
    - minimal BL1 op mapping for matmul / relu / softmax
    - generic placeholder emission for unsupported ops
    """
    kernel_name = semantic_ir.kernel_id or "anonymous_kernel"
    if instruction_ir is not None and not kernel_name.endswith("_kernel"):
        kernel_name = f"{kernel_name}_kernel"
    lines: list[str] = ["module {"]

    args = []
    for idx, param in enumerate(semantic_ir.params):
        args.append(f"%arg{idx}: {_tensor_type(param.shape, param.dtype)}")
    ret_type = _infer_return_type(semantic_ir)
    lines.append(f"  func.func @{kernel_name}({', '.join(args)}) -> {ret_type} {{")

    value_map: dict[str, str] = {}
    for idx, param in enumerate(semantic_ir.params):
        value_map[param.name] = f"%arg{idx}"

    temp_idx = 0
    for node in semantic_ir.nodes:
        result_name = f"%v{temp_idx}"
        temp_idx += 1
        op_line = _emit_node_mlir(node.op, result_name, value_map, node.inputs, node.output.dtype)
        lines.append(f"    {op_line}")
        lines.append(f"    // semantic node: {node.id} op={node.op}")
        value_map[node.id] = result_name

    if instruction_ir is not None:
        for block in instruction_ir.blocks:
            lines.append(f"    // instruction block: {block.name}")
            for inst in block.instructions:
                lines.append(
                    f"    //   {inst.opcode} operands={inst.operands} attrs={inst.attrs}"
                )

    ret_value = value_map.get(semantic_ir.return_node, "%arg0")
    lines.append(f"    return {ret_value} : {ret_type}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _emit_node_mlir(op: str, result_name: str, value_map: dict[str, str], inputs: dict, dtype: str) -> str:
    operands = []
    for ref in inputs.values():
        if hasattr(ref, "name"):
            operands.append(value_map.get(ref.name, f"%{ref.name}"))
        elif hasattr(ref, "id"):
            operands.append(value_map.get(ref.id, f"%{ref.id}"))
    if op == "matmul" and len(operands) >= 2:
        return (
            f"{result_name} = linalg.matmul ins({operands[0]}, {operands[1]}) "
            f": {_opaque_tensor(dtype)}"
        )
    if op == "relu" and operands:
        return f"{result_name} = arith.maximumf {operands[0]}, %cst_zero : {dtype}"
    if op == "softmax" and operands:
        return f"{result_name} = \"arke.softmax\"({operands[0]}) : ({_opaque_tensor(dtype)}) -> {_opaque_tensor(dtype)}"
    joined = ", ".join(operands)
    return f"{result_name} = \"arke.{op}\"({joined}) : ({_opaque_tensor(dtype)}) -> {_opaque_tensor(dtype)}"


def _tensor_type(shape: list, dtype: str) -> str:
    dims = "x".join("?" if not isinstance(d, int) else str(d) for d in shape)
    return f"tensor<{dims}x{dtype}>"


def _opaque_tensor(dtype: str) -> str:
    return f"tensor<*x{dtype}>"


def _infer_return_type(semantic_ir: SemanticIR) -> str:
    for node in semantic_ir.nodes:
        if node.id == semantic_ir.return_node:
            return _tensor_type(node.output.shape, node.output.dtype)
    if semantic_ir.params:
        param = semantic_ir.params[0]
        return _tensor_type(param.shape, param.dtype)
    return "tensor<*xf32>"
