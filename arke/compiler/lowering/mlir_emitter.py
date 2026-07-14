# DEPRECATED — Stage 7 early skeleton. DO NOT USE.
#
# This file is a 16-line placeholder created during Stage 7 scaffolding
# (commit 20bb863, 2026-04). It is NOT the real MLIR emitter.
#
# The REAL MLIR emitter lives at:
#   arke/backend/mlir_emitter.py   — 2992 LOC, Phase 3, 46 ops
#
# The S7 architectural-seam emitter (used by test_mlir_backend.py) lives at:
#   arke/backends/mlir/emitter.py  — MLIREmitter class with real emit logic
#
# This file has ZERO imports in the codebase. It survives only because
# there is no __init__.py in arke/compiler/lowering/ exporting it.
# Kept for git-history traceability; safe to delete when convenient.


class MLIREmitter:
    """DEPRECATED stub. See arke/backend/mlir_emitter.py for the real one."""

    def __init__(self):
        self.mlir_code = ""

    def emit(self, ir):
        self.mlir_code += f"emit: {ir}\n"

    def get_mlir(self):
        return self.mlir_code

    def __repr__(self):
        return f"MLIREmitter(mlir_code={self.mlir_code})"
