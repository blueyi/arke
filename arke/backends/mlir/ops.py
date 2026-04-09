"""Minimal MLIR operation registry for Stage 7."""

from __future__ import annotations


class MLIROp:
    """Placeholder MLIR op emitter entry."""

    def __init__(self, name: str):
        self.name = name

    def emit(self, *_args, **_kwargs) -> str:
        return f"// registered op: {self.name}"


MLIR_OPS = {
    "matmul": MLIROp("matmul"),
    "batch_matmul": MLIROp("batch_matmul"),
    "relu": MLIROp("relu"),
    "softmax": MLIROp("softmax"),
}


def register_mlir_ops(emitter) -> None:
    """Register known MLIR op placeholders with emitter."""
    for op_name, op_impl in MLIR_OPS.items():
        emitter.register_op(op_name, op_impl)
