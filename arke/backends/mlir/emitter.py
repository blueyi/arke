"""MLIR code emitter for the active Arke IR stack.

This backend is intentionally minimal in Stage 7: it emits a readable MLIR-like
skeleton from SemanticIR, enough to verify the architectural seam required by
G7 criterion [5].
"""

from __future__ import annotations

from arke.ir.semantic import SemanticIR


class MLIREmitter:
    """Emit a minimal MLIR skeleton from SemanticIR."""

    def __init__(self):
        self.ops_registry: dict[str, object] = {}

    def emit(self, semantic_ir: SemanticIR) -> str:
        lines: list[str] = ["module {"]
        args = []
        for idx, param in enumerate(semantic_ir.params):
            args.append(f"%arg{idx}: {self._tensor_type(param.shape, param.dtype)}")
        ret_type = self._infer_return_type(semantic_ir)
        lines.append(
            f"  func.func @{semantic_ir.kernel_id}({', '.join(args)}) -> {ret_type} {{"
        )

        value_map: dict[str, str] = {p.name: f"%arg{i}" for i, p in enumerate(semantic_ir.params)}
        temp_idx = 0
        for node in semantic_ir.nodes:
            result_name = f"%v{temp_idx}"
            temp_idx += 1
            lines.append(f"    {self._emit_node(node.op, result_name, value_map, node.inputs, node.output.dtype)}")
            value_map[node.id] = result_name

        ret_value = value_map.get(semantic_ir.return_node, "%arg0")
        lines.append(f"    return {ret_value} : {ret_type}")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    def register_op(self, op_name: str, emitter: object) -> None:
        self.ops_registry[op_name] = emitter

    def _emit_node(self, op: str, result_name: str, value_map: dict[str, str], inputs: dict, dtype: str) -> str:
        operands = []
        for ref in inputs.values():
            if hasattr(ref, "name"):
                operands.append(value_map.get(ref.name, f"%{ref.name}"))
            elif hasattr(ref, "id"):
                operands.append(value_map.get(ref.id, f"%{ref.id}"))

        if op == "matmul" and len(operands) >= 2:
            return (
                f"{result_name} = linalg.matmul ins({operands[0]}, {operands[1]}) "
                f": tensor<*x{dtype}>"
            )
        if op == "relu" and operands:
            return f"{result_name} = arith.maximumf {operands[0]}, %cst_zero : {dtype}"
        if op == "softmax" and operands:
            return f"{result_name} = \"arke.softmax\"({operands[0]}) : (tensor<*x{dtype}>) -> tensor<*x{dtype}>"
        joined = ", ".join(operands)
        return f"{result_name} = \"arke.{op}\"({joined}) : (tensor<*x{dtype}>) -> tensor<*x{dtype}>"

    def _tensor_type(self, shape: list, dtype: str) -> str:
        dims = "x".join("?" if not isinstance(d, int) else str(d) for d in shape)
        return f"tensor<{dims}x{dtype}>"

    def _infer_return_type(self, semantic_ir: SemanticIR) -> str:
        for node in semantic_ir.nodes:
            if node.id == semantic_ir.return_node:
                return self._tensor_type(node.output.shape, node.output.dtype)
        if semantic_ir.params:
            p = semantic_ir.params[0]
            return self._tensor_type(p.shape, p.dtype)
        return "tensor<*xf32>"
