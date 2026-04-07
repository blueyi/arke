# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SemanticIR v1.0 (arke/ir/semantic.py)."""

import json
import pytest

from arke.ir.semantic import (
    AnyNode,
    ConditionalNode,
    Dim,
    Edge,
    FusionGroup,
    InputRef,
    MultiOutputNode,
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    ShapeConstraint,
    SymbolicDim,
    TensorDesc,
    dim_from_json,
    dim_to_json,
    input_ref_from_dict,
)


# ============================================================
# SymbolicDim
# ============================================================

class TestSymbolicDim:
    def test_basic(self):
        sd = SymbolicDim("B")
        assert sd.name == "B"
        assert sd.min is None
        assert sd.max is None

    def test_bounded(self):
        sd = SymbolicDim("S", min=1, max=65536)
        assert sd.min == 1
        assert sd.max == 65536

    def test_to_dict(self):
        sd = SymbolicDim("H", min=1, max=128)
        d = sd.to_dict()
        assert d == {"sym": "H", "min": 1, "max": 128}

    def test_to_dict_no_bounds(self):
        sd = SymbolicDim("B")
        d = sd.to_dict()
        assert d == {"sym": "B"}

    def test_round_trip(self):
        sd = SymbolicDim("S", min=1, max=4096)
        d = sd.to_dict()
        sd2 = SymbolicDim.from_dict(d)
        assert sd2.name == sd.name
        assert sd2.min == sd.min
        assert sd2.max == sd.max


class TestDim:
    def test_int_dim(self):
        assert dim_to_json(128) == 128
        assert dim_from_json(128) == 128

    def test_symbolic_dim(self):
        sd = SymbolicDim("S", max=4096)
        j = dim_to_json(sd)
        assert j == {"sym": "S", "max": 4096}
        sd2 = dim_from_json(j)
        assert isinstance(sd2, SymbolicDim)
        assert sd2.name == "S"
        assert sd2.max == 4096


# ============================================================
# ShapeConstraint
# ============================================================

class TestShapeConstraint:
    def test_basic(self):
        sc = ShapeConstraint("S % 128 == 0", "tile alignment")
        assert sc.expr == "S % 128 == 0"
        assert sc.reason == "tile alignment"

    def test_round_trip(self):
        sc = ShapeConstraint("H * D == model_dim", "head consistency")
        d = sc.to_dict()
        sc2 = ShapeConstraint.from_dict(d)
        assert sc2.expr == sc.expr
        assert sc2.reason == sc.reason

    def test_no_reason(self):
        sc = ShapeConstraint("S > 0")
        d = sc.to_dict()
        assert "reason" not in d
        sc2 = ShapeConstraint.from_dict(d)
        assert sc2.reason == ""


# ============================================================
# TensorDesc
# ============================================================

class TestTensorDesc:
    def test_basic(self):
        td = TensorDesc(shape=[128, 768], dtype="f16")
        assert td.shape == [128, 768]
        assert td.dtype == "f16"
        assert td.layout == "row_major"
        assert not td.is_symbolic()

    def test_symbolic(self):
        td = TensorDesc(shape=[8, SymbolicDim("S"), 512], dtype="f16")
        assert td.is_symbolic()

    def test_round_trip(self):
        td = TensorDesc(shape=[8, SymbolicDim("S", max=65536), 512], dtype="f16")
        d = td.to_dict()
        td2 = TensorDesc.from_dict(d)
        assert td2.dtype == td.dtype
        assert len(td2.shape) == 3
        assert isinstance(td2.shape[1], SymbolicDim)
        assert td2.shape[1].name == "S"

    def test_layout_omitted_when_default(self):
        td = TensorDesc(shape=[128], dtype="f32")
        d = td.to_dict()
        assert "layout" not in d

    def test_layout_included_when_not_default(self):
        td = TensorDesc(shape=[128], dtype="f32", layout="col_major")
        d = td.to_dict()
        assert d["layout"] == "col_major"


# ============================================================
# Param
# ============================================================

class TestParam:
    def test_basic(self):
        p = Param(name="X", shape=[128, 768], dtype="f16")
        assert p.name == "X"
        td = p.to_tensor_desc()
        assert td.shape == [128, 768]

    def test_round_trip(self):
        p = Param(name="A", shape=[SymbolicDim("M"), 768], dtype="f16")
        d = p.to_dict()
        p2 = Param.from_dict(d)
        assert p2.name == "A"
        assert isinstance(p2.shape[0], SymbolicDim)


# ============================================================
# InputRef
# ============================================================

class TestInputRef:
    def test_param_ref(self):
        pr = ParamRef(name="X")
        d = pr.to_dict()
        assert d == {"ref": "param", "name": "X"}
        pr2 = ParamRef.from_dict(d)
        assert pr2.name == "X"

    def test_node_ref(self):
        nr = NodeRef(id="matmul_0")
        d = nr.to_dict()
        assert d == {"ref": "node", "id": "matmul_0"}
        nr2 = NodeRef.from_dict(d)
        assert nr2.id == "matmul_0"

    def test_from_dict_legacy_at_prefix(self):
        ref = input_ref_from_dict("@matmul_0")
        assert isinstance(ref, NodeRef)
        assert ref.id == "matmul_0"

    def test_from_dict_legacy_param(self):
        ref = input_ref_from_dict("X")
        assert isinstance(ref, ParamRef)
        assert ref.name == "X"

    def test_from_dict_structured(self):
        ref = input_ref_from_dict({"ref": "node", "id": "relu_0"})
        assert isinstance(ref, NodeRef)
        assert ref.id == "relu_0"


# ============================================================
# Node
# ============================================================

class TestNode:
    def test_basic(self):
        node = Node(
            id="matmul_0",
            op="matmul",
            inputs={
                "A": ParamRef(name="A"),
                "B": ParamRef(name="B"),
            },
            output=TensorDesc(shape=[128, 256], dtype="f16"),
            semantics=Semantics(
                computation="C[i,j] = sum(A[i,k]*B[k,j], k)",
                index_vars=["i", "j", "k"],
                reduction_axes=["k"],
            ),
        )
        assert node.id == "matmul_0"
        assert node.op == "matmul"
        assert len(node.inputs) == 2

    def test_to_dict(self):
        node = Node(
            id="relu_0",
            op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[128, 768], dtype="f16"),
            semantics=Semantics(computation="Y = max(X, 0)"),
        )
        d = node.to_dict()
        assert d["id"] == "relu_0"
        assert d["op"] == "relu"
        assert "attrs" not in d  # empty attrs omitted

    def test_attrs(self):
        node = Node(
            id="softmax_0",
            op="softmax",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[128, 768], dtype="f16"),
            semantics=Semantics(computation="softmax"),
            attrs={"axis": -1},
        )
        d = node.to_dict()
        assert d["attrs"] == {"axis": -1}


# ============================================================
# MultiOutputNode
# ============================================================

class TestMultiOutputNode:
    def test_basic(self):
        mon = MultiOutputNode(
            id="topk_0",
            op="topk",
            inputs={"X": ParamRef(name="X")},
            outputs={
                "values": TensorDesc(shape=[128, 10], dtype="f16"),
                "indices": TensorDesc(shape=[128, 10], dtype="i64"),
            },
            semantics=Semantics(computation="topk(X, k=10)"),
            attrs={"k": 10},
        )
        assert mon.id == "topk_0"
        assert len(mon.outputs) == 2
        assert "values" in mon.outputs
        assert "indices" in mon.outputs

    def test_to_dict(self):
        mon = MultiOutputNode(
            id="split_0",
            op="split",
            inputs={"X": ParamRef(name="X")},
            outputs={
                "chunk_0": TensorDesc(shape=[64, 768], dtype="f16"),
                "chunk_1": TensorDesc(shape=[64, 768], dtype="f16"),
            },
            semantics=Semantics(computation="split(X, 2)"),
        )
        d = mon.to_dict()
        assert d["multi_output"] is True
        assert "outputs" in d
        assert "chunk_0" in d["outputs"]


# ============================================================
# ConditionalNode
# ============================================================

class TestConditionalNode:
    def test_basic(self):
        cn = ConditionalNode(
            id="dispatch_0",
            predicate='dim("S") <= 512',
            true_branch=["attn_short"],
            false_branch=["attn_long"],
            output=TensorDesc(shape=[8, SymbolicDim("S"), 64], dtype="f16"),
        )
        assert cn.id == "dispatch_0"
        assert cn.predicate == 'dim("S") <= 512'
        assert cn.true_branch == ["attn_short"]
        assert cn.false_branch == ["attn_long"]

    def test_to_dict(self):
        cn = ConditionalNode(
            id="dispatch_0",
            predicate='dim("S") <= 512',
            true_branch=["attn_short"],
            false_branch=["attn_long"],
            output=TensorDesc(shape=[8, 64], dtype="f16"),
        )
        d = cn.to_dict()
        assert d["op"] == "__conditional__"
        assert d["predicate"] == 'dim("S") <= 512'


# ============================================================
# Edge
# ============================================================

class TestEdge:
    def test_basic(self):
        e = Edge(from_node="param:X", to_node="relu_0", tensor_name="X")
        assert e.from_port == "output"
        assert e.lifetime == "local"

    def test_to_dict_compact(self):
        e = Edge(from_node="param:X", to_node="relu_0", tensor_name="X")
        d = e.to_dict()
        assert d == {"from": "param:X", "to": "relu_0", "tensor": "X"}
        assert "from_port" not in d  # default omitted
        assert "lifetime" not in d

    def test_to_dict_with_port(self):
        e = Edge(
            from_node="split_0", to_node="add_0",
            tensor_name="chunk_0", from_port="chunk_0"
        )
        d = e.to_dict()
        assert d["from_port"] == "chunk_0"

    def test_round_trip(self):
        e = Edge(
            from_node="matmul_0", to_node="gelu_0",
            tensor_name="C", from_port="output", lifetime="persistent"
        )
        d = e.to_dict()
        e2 = Edge.from_dict(d)
        assert e2.from_node == e.from_node
        assert e2.to_node == e.to_node
        assert e2.tensor_name == e.tensor_name
        assert e2.lifetime == "persistent"


# ============================================================
# FusionGroup
# ============================================================

class TestFusionGroup:
    def test_basic(self):
        fg = FusionGroup(
            id="fuse_0",
            nodes=["matmul_0", "gelu_0"],
            fusion_type="epilogue",
            reason="eliminate global memory round-trip",
        )
        assert fg.id == "fuse_0"
        assert fg.fusion_type == "epilogue"

    def test_round_trip(self):
        fg = FusionGroup(id="fuse_0", nodes=["a", "b"], fusion_type="vertical")
        d = fg.to_dict()
        fg2 = FusionGroup.from_dict(d)
        assert fg2.id == "fuse_0"
        assert fg2.fusion_type == "vertical"

    def test_reason_omitted_when_empty(self):
        fg = FusionGroup(id="f", nodes=["a"], fusion_type="epilogue")
        d = fg.to_dict()
        assert "reason" not in d


# ============================================================
# SemanticIR
# ============================================================

class TestSemanticIR:
    def _make_simple_ir(self) -> SemanticIR:
        """Create a simple matmul SemanticIR for testing."""
        ir = SemanticIR(kernel_id="matmul_test")
        ir.add_param(Param(name="A", shape=[128, 768], dtype="f16"))
        ir.add_param(Param(name="B", shape=[768, 256], dtype="f16"))
        ir.add_node(Node(
            id="matmul_0",
            op="matmul",
            inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
            output=TensorDesc(shape=[128, 256], dtype="f16"),
            semantics=Semantics(
                computation="C[i,j] = sum(A[i,k]*B[k,j], k)",
                index_vars=["i", "j", "k"],
                reduction_axes=["k"],
            ),
        ))
        ir.add_edge(Edge(from_node="param:A", to_node="matmul_0", tensor_name="A"))
        ir.add_edge(Edge(from_node="param:B", to_node="matmul_0", tensor_name="B"))
        ir.return_node = "matmul_0"
        return ir

    def test_construction(self):
        ir = self._make_simple_ir()
        assert ir.kernel_id == "matmul_test"
        assert ir.version == "1.0.0"
        assert len(ir.params) == 2
        assert len(ir.nodes) == 1
        assert len(ir.edges) == 2
        assert ir.return_node == "matmul_0"

    def test_get_node(self):
        ir = self._make_simple_ir()
        n = ir.get_node("matmul_0")
        assert n is not None
        assert n.op == "matmul"
        assert ir.get_node("nonexistent") is None

    def test_get_param(self):
        ir = self._make_simple_ir()
        p = ir.get_param("A")
        assert p is not None
        assert p.dtype == "f16"
        assert ir.get_param("nonexistent") is None

    def test_is_symbolic(self):
        ir = self._make_simple_ir()
        assert not ir.is_symbolic()
        ir.add_symbolic_dim(SymbolicDim("B"))
        assert ir.is_symbolic()

    def test_json_round_trip(self):
        ir = self._make_simple_ir()
        j1 = ir.to_json()
        ir2 = SemanticIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2

    def test_json_round_trip_with_symbolic(self):
        ir = SemanticIR(kernel_id="dynamic_test")
        ir.add_symbolic_dim(SymbolicDim("S", min=1, max=65536))
        ir.add_shape_constraint(ShapeConstraint("S % 64 == 0", "tile alignment"))
        ir.add_param(Param(name="X", shape=[8, SymbolicDim("S"), 512], dtype="f16"))
        ir.add_node(Node(
            id="softmax_0",
            op="softmax",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[8, SymbolicDim("S"), 512], dtype="f16"),
            semantics=Semantics(computation="softmax(X, axis=-1)"),
            attrs={"axis": -1},
        ))
        ir.return_node = "softmax_0"

        j1 = ir.to_json()
        ir2 = SemanticIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2

    def test_json_round_trip_with_multi_output(self):
        ir = SemanticIR(kernel_id="topk_test")
        ir.add_param(Param(name="X", shape=[128, 768], dtype="f16"))
        ir.add_node(MultiOutputNode(
            id="topk_0",
            op="topk",
            inputs={"X": ParamRef(name="X")},
            outputs={
                "values": TensorDesc(shape=[128, 10], dtype="f16"),
                "indices": TensorDesc(shape=[128, 10], dtype="i64"),
            },
            semantics=Semantics(computation="topk(X, k=10)"),
            attrs={"k": 10},
        ))
        ir.return_node = "topk_0"

        j1 = ir.to_json()
        ir2 = SemanticIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2
        # Verify deserialized node is MultiOutputNode
        node = ir2.get_node("topk_0")
        assert isinstance(node, MultiOutputNode)
        assert "values" in node.outputs

    def test_json_round_trip_with_conditional(self):
        ir = SemanticIR(kernel_id="cond_test")
        ir.add_symbolic_dim(SymbolicDim("S"))
        ir.add_node(ConditionalNode(
            id="dispatch_0",
            predicate='dim("S") <= 512',
            true_branch=["short_path"],
            false_branch=["long_path"],
            output=TensorDesc(shape=[8, SymbolicDim("S"), 64], dtype="f16"),
        ))
        ir.return_node = "dispatch_0"

        j1 = ir.to_json()
        ir2 = SemanticIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2
        node = ir2.get_node("dispatch_0")
        assert isinstance(node, ConditionalNode)
        assert node.predicate == 'dim("S") <= 512'

    def test_json_round_trip_with_fusion_groups(self):
        ir = self._make_simple_ir()
        ir.add_node(Node(
            id="gelu_0",
            op="gelu",
            inputs={"X": NodeRef(id="matmul_0")},
            output=TensorDesc(shape=[128, 256], dtype="f16"),
            semantics=Semantics(computation="gelu(X)"),
        ))
        ir.add_fusion_group(FusionGroup(
            id="fuse_0",
            nodes=["matmul_0", "gelu_0"],
            fusion_type="epilogue",
            reason="eliminate global memory round-trip",
        ))
        ir.return_node = "gelu_0"

        j1 = ir.to_json()
        ir2 = SemanticIR.from_json(j1)
        j2 = ir2.to_json()
        assert j1 == j2
        assert len(ir2.fusion_groups) == 1
        assert ir2.fusion_groups[0].fusion_type == "epilogue"

    def test_backward_compat_v020_format(self):
        """Test loading a v0.2.0-style JSON format."""
        v020_json = {
            "version": "0.2.0",
            "kernel_id": "old_kernel",
            "params": [
                {"name": "X", "shape": [128, 768], "dtype": "f16"}
            ],
            "nodes": [
                {
                    "id": "relu_0",
                    "op": "relu",
                    "inputs": {"X": "X"},  # legacy string format
                    "output": {"shape": [128, 768], "dtype": "f16"},
                    "semantics": {"computation": "max(X, 0)"},
                }
            ],
            "edges": [
                {"from": "param:X", "to": "relu_0", "tensor": "X"}
            ],
            "return_node": "relu_0",
        }
        ir = SemanticIR.from_dict(v020_json)
        assert ir.version == "0.2.0"
        assert ir.kernel_id == "old_kernel"
        assert len(ir.params) == 1
        node = ir.get_node("relu_0")
        assert node is not None
        # Legacy string "X" should become ParamRef
        assert isinstance(node.inputs["X"], ParamRef)
        assert node.inputs["X"].name == "X"

    def test_backward_compat_v020_node_ref(self):
        """Test loading v0.2.0 node reference with @ prefix."""
        v020_json = {
            "version": "0.2.0",
            "kernel_id": "chain",
            "params": [
                {"name": "X", "shape": [128], "dtype": "f32"}
            ],
            "nodes": [
                {
                    "id": "relu_0",
                    "op": "relu",
                    "inputs": {"X": "X"},
                    "output": {"shape": [128], "dtype": "f32"},
                    "semantics": {"computation": "relu(X)"},
                },
                {
                    "id": "gelu_0",
                    "op": "gelu",
                    "inputs": {"X": "@relu_0"},
                    "output": {"shape": [128], "dtype": "f32"},
                    "semantics": {"computation": "gelu(X)"},
                },
            ],
            "edges": [],
            "return_node": "gelu_0",
        }
        ir = SemanticIR.from_dict(v020_json)
        gelu = ir.get_node("gelu_0")
        assert isinstance(gelu.inputs["X"], NodeRef)
        assert gelu.inputs["X"].id == "relu_0"

    def test_metadata(self):
        ir = self._make_simple_ir()
        ir.metadata = {"source": "test", "version": 1}
        j1 = ir.to_json()
        ir2 = SemanticIR.from_json(j1)
        assert ir2.metadata == {"source": "test", "version": 1}

    def test_backward_compat_alias(self):
        from arke.ir.semantic import SemanticGraph
        assert SemanticGraph is SemanticIR
