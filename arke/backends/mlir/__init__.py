"""Legacy S7-era MLIR emitter (BL1 matmul PoC).

⚠️ NAMING NOTE (audit R5, 2026-07-29): this package `arke.backends.mlir`
(plural *backends*) is the ORIGINAL S7 architectural-seam proof-of-concept —
a 77-LOC MLIREmitter + a BL1 matmul demo. It is NOT the production MLIR path.

The production, 46-op MLIR-GPU backend lives in the singular
`arke.backend.mlir_emitter` / `arke.backend.mlir_gpu`. New code and docs must
target `arke.backend.*`. This package is retained only because
`tests/test_mlir_backend.py` pins the S7 contract; do not extend it. Track its
removal (fold the contract test into the production suite) as a follow-up so
the plural/singular ambiguity can finally be deleted.
"""

from .emitter import MLIREmitter
from .ops import register_mlir_ops

__all__ = ["MLIREmitter", "register_mlir_ops"]
