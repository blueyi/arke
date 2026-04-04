# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for G6.5 — IR ↔ JSON/dict round-trip completeness.

Validates that SemanticIR and StrategyIR survive full serialization
round-trips (to_dict → JSON → from_dict → to_dict) without data loss
for every operator in the OP_CATALOG.
"""

from __future__ import annotations

import json

import pytest

from arke.ir.builder import KernelBuilder
from arke.ir.ops.catalog import OP_CATALOG
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import Decision, Rationale, StrategyIR

# ─── Fixtures / helpers ──────────────────────────────────────────────────────

# Override default [64, 64] for ops that require specific shapes
_OP_SHAPES: dict[str, dict[str, list[int]]] = {
    "batch_matmul": {"A": [4, 32, 64], "B": [4, 64, 32]},
    "layernorm":    {"X": [32, 64], "W": [64]},
    "rmsnorm":      {"X": [32, 64], "W": [64]},
    "transpose":    {"X": [32, 64]},
    "matmul":       {"A": [32, 64], "B": [64, 32]},
}
_DEFAULT_SHAPE = [64, 64]


def _build_semantic_ir(op_name: str) -> SemanticIR:
    """Build a minimal SemanticIR for ``op_name``."""
    b = KernelBuilder(f"test_{op_name}")
    op_def = OP_CATALOG[op_name]
    custom = _OP_SHAPES.get(op_name, {})
    kwargs: dict[str, str] = {}
    for inp in op_def.inputs:
        shape = custom.get(inp, _DEFAULT_SHAPE)
        b.param(inp, shape, "f16")
        kwargs[inp] = inp
    nid = b.op(op_name, **kwargs)
    out_shape = b._params[0].shape
    b.returns(nid, out_shape, "f16")
    return b.build()


# ─── SemanticIR round-trip ────────────────────────────────────────────────────

@pytest.mark.parametrize("op_name", sorted(OP_CATALOG.keys()))
def test_semantic_ir_json_roundtrip(op_name: str) -> None:
    """SemanticIR.to_dict() → JSON → SemanticIR.from_dict() must be lossless."""
    ir = _build_semantic_ir(op_name)

    d1 = ir.to_dict()
    restored = SemanticIR.from_dict(json.loads(json.dumps(d1)))
    d2 = restored.to_dict()

    assert restored.kernel_id == ir.kernel_id, "kernel_id mismatch"
    assert len(restored.nodes) == len(ir.nodes), "node count mismatch"
    assert len(restored.params) == len(ir.params), "param count mismatch"
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True), (
        f"dict mismatch for op={op_name}\n"
        f"  before: {json.dumps(d1, sort_keys=True)[:200]}\n"
        f"  after:  {json.dumps(d2, sort_keys=True)[:200]}"
    )


# ─── StrategyIR round-trip ────────────────────────────────────────────────────

def _build_strategy_ir() -> StrategyIR:
    """Build a representative StrategyIR with diverse decision types."""
    ir = StrategyIR(kernel_id="test_matmul", target_hw="nvidia_ampere")
    ir.tile("M", [64], "aligned to tensor core 16×8")
    ir.tile("N", [64], "aligned to tensor core 16×8")
    ir.tile("K", [16], "shared memory budget")
    ir.reorder(["M", "N", "K"], "outer MN for parallelism, K for reduction")
    ir.add_decision(Decision(
        kind="fuse",
        params={"ops": ["matmul_0", "gelu_1"], "type": "epilogue"},
        rationale=Rationale("saves global mem roundtrip"),
    ))
    ir.add_decision(Decision(
        kind="launch_config",
        params={"num_warps": 4, "num_stages": 3, "block_sizes": {"BLOCK_M": 64, "BLOCK_N": 64}},
    ))
    ir.add_decision(Decision(
        kind="parallel",
        params={"loops": ["M", "N"], "mapping": {"M": "blockIdx.x", "N": "blockIdx.y"}},
        rationale=Rationale("each block owns one output tile"),
    ))
    return ir


def test_strategy_ir_json_roundtrip() -> None:
    """StrategyIR.to_dict() → JSON → StrategyIR.from_dict() must be lossless."""
    ir = _build_strategy_ir()
    d1 = ir.to_dict()
    restored = StrategyIR.from_dict(json.loads(json.dumps(d1)))
    d2 = restored.to_dict()

    assert restored.kernel_id == ir.kernel_id
    assert restored.target_hw == ir.target_hw
    assert restored.decision_count == ir.decision_count

    for orig, rest in zip(ir.decisions, restored.decisions):
        assert orig.kind == rest.kind
        assert orig.params == rest.params
        r1 = orig.rationale.text if orig.rationale else None
        r2 = rest.rationale.text if rest.rationale else None
        assert r1 == r2, f"rationale mismatch: {r1!r} != {r2!r}"

    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True), (
        "StrategyIR dict mismatch after round-trip"
    )


def test_strategy_ir_empty_roundtrip() -> None:
    """Empty StrategyIR should survive round-trip without errors."""
    ir = StrategyIR.from_dict({})
    assert ir.decisions == []
    d = ir.to_dict()
    ir2 = StrategyIR.from_dict(d)
    assert ir2.decisions == []


def test_strategy_ir_rationale_preserved() -> None:
    """@rationale text must survive round-trip exactly."""
    ir = StrategyIR(kernel_id="k", target_hw="nvidia_ampere")
    text = "K-tile=16: A+B tiles = 4096B ≤ smem/2 (24576B)"
    ir.tile("K", [16], text)
    restored = StrategyIR.from_dict(ir.to_dict())
    assert restored.decisions[0].rationale is not None
    assert restored.decisions[0].rationale.text == text


def test_strategy_ir_nested_params_preserved() -> None:
    """Nested dict/list params (maps, arrays) must survive round-trip."""
    ir = StrategyIR(kernel_id="k", target_hw="nvidia_ampere")
    ir.add_decision(Decision(
        kind="parallel",
        params={
            "loops": ["M", "N"],
            "mapping": {"M": "blockIdx.x", "N": "blockIdx.y"},
        },
    ))
    ir.add_decision(Decision(
        kind="autotune",
        params={
            "configs": [
                {"num_warps": 2, "num_stages": 3},
                {"num_warps": 4, "num_stages": 3},
            ],
            "key": ["M", "N", "K"],
        },
    ))
    restored = StrategyIR.from_dict(ir.to_dict())
    assert restored.decisions[0].params["mapping"] == {"M": "blockIdx.x", "N": "blockIdx.y"}
    assert restored.decisions[1].params["configs"][1]["num_warps"] == 4
    assert restored.decisions[1].params["key"] == ["M", "N", "K"]
