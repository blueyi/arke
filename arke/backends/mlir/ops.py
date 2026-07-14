"""Stage 7 architectural-seam MLIR op registry (G7[5] validation).

⚠️ HERITAGE CODE — Stage 7 (2026-04).

This file provides a minimal placeholder registry that was used to validate
the S7 architectural seam: SemanticIR → MLIREmitter (arke/backends/mlir/emitter.py)
can receive registered ops. The 4 tests in tests/test_mlir_backend.py depend on it.

The REAL Phase 3 MLIR emitter (2992 LOC, 46 ops, GPU codegen) lives at:
    arke/backend/mlir_emitter.py

Do NOT extend this file for new ops — use arke/backend/mlir_emitter.py instead.
"""

from __future__ import annotations


class MLIROp:
    """Placeholder MLIR op emitter entry (S7 heritage)."""

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
    """Register known MLIR op placeholders with emitter (S7 heritage)."""
    for op_name, op_impl in MLIR_OPS.items():
        emitter.register_op(op_name, op_impl)
