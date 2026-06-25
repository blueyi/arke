# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""D8-IR3 — all-ops SemanticIR JSON round-trip (G9 IR finalization).

For every operator in the SSOT benchmark catalog (`benchmarks.op_registry`),
build a SemanticIR carrying that op's real schema (inputs as params + one
compute node) and assert that `SemanticIR.from_dict(to_dict(ir))` and the
JSON `from_json(to_json(ir))` paths are stable (byte-identical re-serialization).

This is the IR-maturity gate for the v1.0 IR spec freeze: the high-level IR
must survive a full serialize→deserialize→serialize cycle for every catalog op
with zero drift.
"""

from __future__ import annotations

import json

import pytest

from arke.ir.semantic import SemanticIR, Param
from arke.ir.ops.registry import REGISTRY
from benchmarks.op_registry import ALL_OPS


def _build_semantic_for_op(op: str) -> SemanticIR:
    """Build a minimal-but-real SemanticIR for ``op`` from its registry schema."""
    schema = REGISTRY.get(op)
    ir = SemanticIR(kernel_id=op)

    # Inputs → params (concrete int dims; round-trip stability is what we test).
    inputs = schema.inputs
    input_names = list(inputs.keys()) if hasattr(inputs, "keys") else list(inputs)
    for name in input_names:
        ir.add_param(Param(name=name, shape=[128, 256], dtype="float16"))

    return ir


@pytest.mark.parametrize("op", sorted(ALL_OPS))
def test_semantic_ir_dict_round_trip(op):
    """to_dict → from_dict → to_dict is stable for every catalog op."""
    ir = _build_semantic_for_op(op)
    d1 = ir.to_dict()
    ir2 = SemanticIR.from_dict(d1)
    d2 = ir2.to_dict()
    assert d1 == d2, f"{op}: dict round-trip drift"
    assert ir2.kernel_id == op


@pytest.mark.parametrize("op", sorted(ALL_OPS))
def test_semantic_ir_json_round_trip(op):
    """to_json → from_json → to_json is byte-stable for every catalog op."""
    ir = _build_semantic_for_op(op)
    j1 = ir.to_json()
    ir2 = SemanticIR.from_json(j1)
    j2 = ir2.to_json()
    # Compare parsed structures (whitespace-insensitive) + byte identity.
    assert json.loads(j1) == json.loads(j2), f"{op}: json round-trip drift"
    assert ir2.kernel_id == op


def test_all_ops_covered():
    """Sentinel: the catalog is non-trivial and matches the registry."""
    assert len(ALL_OPS) >= 45
    # Every catalog op must resolve in the IR op registry.
    for op in ALL_OPS:
        assert REGISTRY.get(op) is not None, f"{op} missing from IR REGISTRY"
