# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arke IR JSON Schema validation.

Validates that Semantic IR and Strategy IR data structures conform to
their JSON Schema definitions (draft-2020-12).
"""

from __future__ import annotations

import json
import pathlib

import pytest
from jsonschema import ValidationError, validate

from arke.ir.builder import KernelBuilder
from arke.ir.semantic import (
    Edge,
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    TensorDesc,
)
from arke.ir.strategy import Decision, HardwareConstraints, Rationale, StrategyIR

# ============================================================
# Schema loading helpers
# ============================================================

SCHEMA_DIR = pathlib.Path(__file__).resolve().parent.parent / "arke" / "ir" / "schemas"


def _load_schema(name: str) -> dict:
    path = SCHEMA_DIR / name
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def semantic_schema() -> dict:
    return _load_schema("semantic.schema.json")


@pytest.fixture(scope="module")
def strategy_schema() -> dict:
    return _load_schema("strategy.schema.json")


# ============================================================
# Semantic IR — positive tests
# ============================================================


class TestSemanticSchemaValid:
    """Valid Semantic IR instances must pass schema validation."""

    def test_builder_output_validates(self, semantic_schema):
        """SemanticIR produced by KernelBuilder passes the schema."""
        b = KernelBuilder("fused_matmul_relu")
        b.param("A", [1024, 512], "f16")
        b.param("B", [512, 2048], "f16")
        m = b.op("matmul", A="A", B="B")
        r = b.op("relu", X=m)
        b.returns(r, [1024, 2048], "f16")
        ir = b.build()

        data = ir.to_dict()
        validate(instance=data, schema=semantic_schema)

    def test_minimal_semantic_ir(self, semantic_schema):
        """Minimal valid Semantic IR with one node, no optional fields."""
        ir = SemanticIR(kernel_id="minimal")
        ir.add_param(Param(name="X", shape=[4], dtype="f32"))
        ir.return_type = TensorDesc(shape=[4], dtype="f32")
        ir.add_node(Node(
            id="relu_0",
            op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.return_node = "relu_0"

        validate(instance=ir.to_dict(), schema=semantic_schema)

    def test_all_dtypes_accepted(self, semantic_schema):
        """Every dtype enum value is accepted in a Param."""
        dtypes = [
            "f16", "f32", "f64", "bf16",
            "i8", "i16", "i32", "i64",
            "u8", "u16", "u32", "u64",
            "bool", "index",
        ]
        for dt in dtypes:
            ir = SemanticIR(kernel_id=f"dtype_{dt}")
            ir.add_param(Param(name="X", shape=[1], dtype=dt))
            ir.return_type = TensorDesc(shape=[1], dtype=dt)
            ir.add_node(Node(
                id="n0", op="identity",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[1], dtype=dt),
                semantics=Semantics(computation="Y = X"),
            ))
            ir.return_node = "n0"
            validate(instance=ir.to_dict(), schema=semantic_schema)

    def test_col_major_layout(self, semantic_schema):
        """col_major layout is a valid layout value."""
        ir = SemanticIR(kernel_id="col_test")
        ir.add_param(Param(name="A", shape=[4, 4], dtype="f32", layout="col_major"))
        ir.return_type = TensorDesc(shape=[4, 4], dtype="f32", layout="col_major")
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="A")},
            output=TensorDesc(shape=[4, 4], dtype="f32", layout="col_major"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.return_node = "n0"
        validate(instance=ir.to_dict(), schema=semantic_schema)

    def test_with_edges_and_fusion(self, semantic_schema):
        """Semantic IR with edges and fusion groups validates."""
        b = KernelBuilder("with_fusion")
        b.param("X", [8, 8], "f32")
        r0 = b.op("relu", X="X")
        r1 = b.op("relu", X=r0)
        b.returns(r1, [8, 8], "f32")
        ir = b.build()

        data = ir.to_dict()
        assert len(data["edges"]) >= 1
        assert len(data["fusion_groups"]) >= 1
        validate(instance=data, schema=semantic_schema)

    def test_edge_lifetime_persistent(self, semantic_schema):
        """Edge with lifetime='persistent' validates."""
        ir = SemanticIR(kernel_id="persist_test")
        ir.add_param(Param(name="X", shape=[4], dtype="f32"))
        ir.return_type = TensorDesc(shape=[4], dtype="f32")
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.add_node(Node(
            id="n1", op="relu",
            inputs={"X": NodeRef(id="n0")},
            output=TensorDesc(shape=[4], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.add_edge(Edge(
            from_node="n0", to_node="n1",
            tensor_name="intermediate", lifetime="persistent",
        ))
        ir.return_node = "n1"
        validate(instance=ir.to_dict(), schema=semantic_schema)


# ============================================================
# Semantic IR — InputRef tests
# ============================================================


class TestInputRefSchema:
    """InputRef (ParamRef / NodeRef) schema validation."""

    def test_param_ref_format(self, semantic_schema):
        """ParamRef with {ref: 'param', name: str} validates."""
        ir = SemanticIR(kernel_id="paramref_test")
        ir.add_param(Param(name="A", shape=[2], dtype="f32"))
        ir.return_type = TensorDesc(shape=[2], dtype="f32")
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="A")},
            output=TensorDesc(shape=[2], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.return_node = "n0"
        data = ir.to_dict()

        # Verify the serialized format
        ref = data["nodes"][0]["inputs"]["X"]
        assert ref == {"ref": "param", "name": "A"}
        validate(instance=data, schema=semantic_schema)

    def test_node_ref_format(self, semantic_schema):
        """NodeRef with {ref: 'node', id: str} validates."""
        ir = SemanticIR(kernel_id="noderef_test")
        ir.add_param(Param(name="X", shape=[2], dtype="f32"))
        ir.return_type = TensorDesc(shape=[2], dtype="f32")
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[2], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.add_node(Node(
            id="n1", op="relu",
            inputs={"X": NodeRef(id="n0")},
            output=TensorDesc(shape=[2], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.return_node = "n1"
        data = ir.to_dict()

        ref = data["nodes"][1]["inputs"]["X"]
        assert ref == {"ref": "node", "id": "n0"}
        validate(instance=data, schema=semantic_schema)

    def test_mixed_refs_in_one_node(self, semantic_schema):
        """A node can have both ParamRef and NodeRef inputs."""
        ir = SemanticIR(kernel_id="mixed_ref")
        ir.add_param(Param(name="A", shape=[4, 4], dtype="f32"))
        ir.add_param(Param(name="B", shape=[4, 4], dtype="f32"))
        ir.return_type = TensorDesc(shape=[4, 4], dtype="f32")
        ir.add_node(Node(
            id="relu_0", op="relu",
            inputs={"X": ParamRef(name="A")},
            output=TensorDesc(shape=[4, 4], dtype="f32"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        ))
        ir.add_node(Node(
            id="add_0", op="add",
            inputs={
                "A": NodeRef(id="relu_0"),
                "B": ParamRef(name="B"),
            },
            output=TensorDesc(shape=[4, 4], dtype="f32"),
            semantics=Semantics(computation="Y = A + B"),
        ))
        ir.return_node = "add_0"
        validate(instance=ir.to_dict(), schema=semantic_schema)

    def test_invalid_ref_type_rejected(self, semantic_schema):
        """An InputRef with ref='unknown' is rejected."""
        data = {
            "version": "0.2.0",
            "kernel_id": "bad_ref",
            "params": [{"name": "X", "shape": [1], "dtype": "f32"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [{
                "id": "n0", "op": "relu",
                "inputs": {"X": {"ref": "unknown", "name": "X"}},
                "output": {"shape": [1], "dtype": "f32"},
                "semantics": {"computation": "Y = X"},
            }],
            "edges": [],
            "return_node": "n0",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)


# ============================================================
# Semantic IR — negative tests
# ============================================================


class TestSemanticSchemaInvalid:
    """Invalid Semantic IR instances must fail schema validation."""

    def test_missing_kernel_id(self, semantic_schema):
        data = {
            "version": "0.2.0",
            "params": [],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_missing_version(self, semantic_schema):
        data = {
            "kernel_id": "test",
            "params": [],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_missing_return_type(self, semantic_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [],
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_invalid_dtype(self, semantic_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [{"name": "X", "shape": [1], "dtype": "float32"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_invalid_layout(self, semantic_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [{"name": "X", "shape": [1], "dtype": "f32", "layout": "NCHW"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_node_missing_semantics(self, semantic_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [{"name": "X", "shape": [1], "dtype": "f32"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [{
                "id": "n0", "op": "relu",
                "inputs": {"X": {"ref": "param", "name": "X"}},
                "output": {"shape": [1], "dtype": "f32"},
                # missing 'semantics'
            }],
            "edges": [],
            "return_node": "n0",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_node_missing_output(self, semantic_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [{"name": "X", "shape": [1], "dtype": "f32"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [{
                "id": "n0", "op": "relu",
                "inputs": {"X": {"ref": "param", "name": "X"}},
                # missing 'output'
                "semantics": {"computation": "Y = X"},
            }],
            "edges": [],
            "return_node": "n0",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_extra_top_level_field_rejected(self, semantic_schema):
        """additionalProperties=false rejects unknown fields."""
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
            "unknown_field": True,
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_empty_shape_rejected(self, semantic_schema):
        """Shape must have at least one dimension."""
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [{"name": "X", "shape": [], "dtype": "f32"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_negative_shape_rejected(self, semantic_schema):
        """Shape dimensions must be positive."""
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [{"name": "X", "shape": [-1], "dtype": "f32"}],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)

    def test_fusion_group_single_node_rejected(self, semantic_schema):
        """FusionGroup requires at least 2 nodes."""
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "params": [],
            "return_type": {"shape": [1], "dtype": "f32"},
            "nodes": [],
            "edges": [],
            "return_node": "",
            "fusion_groups": [{
                "id": "fg0",
                "nodes": ["only_one"],
                "fusion_type": "epilogue",
            }],
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=semantic_schema)


# ============================================================
# Strategy IR — positive tests
# ============================================================


class TestStrategySchemaValid:
    """Valid Strategy IR instances must pass schema validation."""

    def test_strategy_with_decisions(self, strategy_schema):
        """StrategyIR with multiple decisions validates."""
        s = StrategyIR(kernel_id="fmr", target_hw="nvidia_ampere")
        s.tile("i", [64, 16], rationale="L2 cache")
        s.fuse(["matmul_0", "relu_0"], rationale="epilogue fusion")
        s.place("A_tile", "shared", rationale="broadcast reuse")
        s.parallel(
            ["i_outer", "j_outer"],
            {"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"},
        )

        validate(instance=s.to_dict(), schema=strategy_schema)

    def test_empty_decisions(self, strategy_schema):
        """Strategy with zero decisions is valid."""
        s = StrategyIR(kernel_id="empty", target_hw="nvidia_ampere")
        validate(instance=s.to_dict(), schema=strategy_schema)

    def test_decision_without_rationale(self, strategy_schema):
        """Decisions with rationale=null validate."""
        s = StrategyIR(kernel_id="no_rat", target_hw="nvidia_ampere")
        s.add_decision(Decision(kind="tile", params={"loop": "i", "factors": [32]}))
        data = s.to_dict()
        assert data["decisions"][0]["rationale"] is None
        validate(instance=data, schema=strategy_schema)

    def test_all_decision_kinds(self, strategy_schema):
        """Every decision kind enum value is accepted."""
        kinds = ["tile", "reorder", "fuse", "parallel", "place", "vectorize", "unroll", "algorithm"]
        s = StrategyIR(kernel_id="all_kinds", target_hw="nvidia_ampere")
        for i, k in enumerate(kinds):
            s.add_decision(Decision(kind=k, params={"test": True}))
        validate(instance=s.to_dict(), schema=strategy_schema)

    def test_strategy_with_constraints(self, strategy_schema):
        """Custom hardware constraints validate."""
        s = StrategyIR(
            kernel_id="constrained",
            target_hw="nvidia_ampere",
            constraints=HardwareConstraints(
                shared_memory_limit=49152,
                register_limit=255,
                max_threads_per_block=1024,
                warp_size=32,
            ),
        )
        validate(instance=s.to_dict(), schema=strategy_schema)

    def test_strategy_json_roundtrip(self, strategy_schema):
        """Strategy IR survives JSON round-trip and still validates."""
        s = StrategyIR(kernel_id="rt", target_hw="nvidia_ampere")
        s.tile("i", [64], rationale="test")
        json_str = s.to_json()

        restored_data = json.loads(json_str)
        validate(instance=restored_data, schema=strategy_schema)


# ============================================================
# Strategy IR — negative tests
# ============================================================


class TestStrategySchemaInvalid:
    """Invalid Strategy IR instances must fail schema validation."""

    def test_missing_kernel_id(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "target_hw": "nvidia_ampere",
            "decisions": [],
            "constraints": {},
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_missing_target_hw(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "decisions": [],
            "constraints": {},
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_missing_constraints(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "target_hw": "nvidia_ampere",
            "decisions": [],
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_invalid_decision_kind(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "target_hw": "nvidia_ampere",
            "decisions": [{
                "step": 1,
                "kind": "magic_optimize",
                "params": {},
            }],
            "constraints": {},
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_decision_missing_step(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "target_hw": "nvidia_ampere",
            "decisions": [{
                "kind": "tile",
                "params": {"loop": "i", "factors": [64]},
            }],
            "constraints": {},
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_decision_missing_params(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "target_hw": "nvidia_ampere",
            "decisions": [{
                "step": 1,
                "kind": "tile",
            }],
            "constraints": {},
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_step_zero_rejected(self, strategy_schema):
        """Step must be >= 1."""
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "target_hw": "nvidia_ampere",
            "decisions": [{
                "step": 0,
                "kind": "tile",
                "params": {},
            }],
            "constraints": {},
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)

    def test_extra_top_level_field_rejected(self, strategy_schema):
        data = {
            "version": "0.2.0",
            "kernel_id": "test",
            "target_hw": "nvidia_ampere",
            "decisions": [],
            "constraints": {},
            "extra": True,
        }
        with pytest.raises(ValidationError):
            validate(instance=data, schema=strategy_schema)
