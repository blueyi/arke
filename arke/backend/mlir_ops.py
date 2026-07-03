# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke → MLIR per-op body emitters (Phase 3, P3-S2).

This module holds the *scalable* op catalog for the MLIR backend: elementwise,
reduction, and data-movement ops expressed via ``linalg.generic`` /
``linalg.*`` structured ops. The P3-S1 module (``mlir_emitter``) owns the
graph-walk + matmul; this module extends coverage toward the 35-op P3-S2 gate.

Design:
  * **Elementwise** ops (OT0) share one ``linalg.generic`` skeleton (identity
    indexing map, all-parallel iterators); each op supplies only the scalar
    body via ``ELEMENTWISE`` (a list of MLIR body lines that compute ``%res``
    from operand SSA names ``%a0``, ``%a1``, …). n-ary elementwise (add, mul)
    just take more inputs.
  * **Reduction** ops (OT1 subset) use ``linalg.generic`` with a reduction
    iterator, or a two-pass structure (softmax/normalizations) built op-wise.
  * **Movement/dense** ops (OT2) map to dedicated structured ops (transpose,
    batch_matmul) or generic copies (permute/copy).

Every op here lowers through the same CPU pipeline in ``mlir_backend`` (linalg
→ loops → LLVM, with ``-convert-math-to-llvm`` for exp/tanh/rsqrt). f32-only in
P3-S2's correctness pass (matches the printMemrefF32 JIT harness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ── op spec ────────────────────────────────────────────────────

@dataclass
class OpSpec:
    """Declarative spec for one MLIR-lowerable op.

    category:
        "elementwise" — n-ary, output shape == input[0] shape; body via
                         ``ew_body`` (lines producing ``%res`` from %a0..%aK).
        "structured"  — custom emitter that returns full body lines given
                         (out_buf, in_bufs, out_ty, in_tys, attrs, shapes).
    """
    name: str
    num_inputs: int
    category: str
    ew_body: list[str] = field(default_factory=list)
    structured: Callable | None = None
    # shape rule: given input shapes (+ attrs) → output shape
    shape_rule: Callable | None = None


# ── elementwise scalar bodies ──────────────────────────────────
# Each body is a list of MLIR lines. Inputs are %a0, %a1, ...; the block yields
# via `linalg.yield %res : f32`. Bodies must define %res.

_EW_UNARY = {
    "relu": [
        "      %zero = arith.constant 0.0 : f32",
        "      %res = arith.maximumf %a0, %zero : f32",
    ],
    "neg": [
        "      %res = arith.negf %a0 : f32",
    ],
    "exp": [
        "      %res = math.exp %a0 : f32",
    ],
    "tanh": [
        "      %res = math.tanh %a0 : f32",
    ],
    "sigmoid": [
        # 1 / (1 + exp(-x))
        "      %one = arith.constant 1.0 : f32",
        "      %nx = arith.negf %a0 : f32",
        "      %e = math.exp %nx : f32",
        "      %d = arith.addf %one, %e : f32",
        "      %res = arith.divf %one, %d : f32",
    ],
    "silu": [
        # x * sigmoid(x)
        "      %one = arith.constant 1.0 : f32",
        "      %nx = arith.negf %a0 : f32",
        "      %e = math.exp %nx : f32",
        "      %d = arith.addf %one, %e : f32",
        "      %s = arith.divf %one, %d : f32",
        "      %res = arith.mulf %a0, %s : f32",
    ],
    "gelu": [
        # tanh approximation: 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3)))
        "      %half = arith.constant 0.5 : f32",
        "      %one = arith.constant 1.0 : f32",
        "      %c0 = arith.constant 0.7978845608028654 : f32",
        "      %c1 = arith.constant 0.044715 : f32",
        "      %x2 = arith.mulf %a0, %a0 : f32",
        "      %x3 = arith.mulf %x2, %a0 : f32",
        "      %t0 = arith.mulf %c1, %x3 : f32",
        "      %t1 = arith.addf %a0, %t0 : f32",
        "      %t2 = arith.mulf %c0, %t1 : f32",
        "      %th = math.tanh %t2 : f32",
        "      %t3 = arith.addf %one, %th : f32",
        "      %t4 = arith.mulf %half, %a0 : f32",
        "      %res = arith.mulf %t4, %t3 : f32",
    ],
    "rsqrt": [
        "      %res = math.rsqrt %a0 : f32",
    ],
}

_EW_BINARY = {
    "add": [
        "      %res = arith.addf %a0, %a1 : f32",
    ],
    "mul": [
        "      %res = arith.mulf %a0, %a1 : f32",
    ],
}


def _elementwise_specs() -> dict[str, OpSpec]:
    specs: dict[str, OpSpec] = {}
    for name, body in _EW_UNARY.items():
        specs[name] = OpSpec(name=name, num_inputs=1, category="elementwise",
                             ew_body=body)
    for name, body in _EW_BINARY.items():
        specs[name] = OpSpec(name=name, num_inputs=2, category="elementwise",
                             ew_body=body)
    return specs


ELEMENTWISE_SPECS = _elementwise_specs()
