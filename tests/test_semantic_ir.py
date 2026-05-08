# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SemanticIR (arke/ir/semantic.py)."""

import json

from arke.ir.semantic import (
    ConditionalNode,
    Edge,
    FusionGroup,
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

    def test_static_and_alignment_metadata(self):
        sd = SymbolicDim("K", is_static=True, multiple_of=32, default=128)
        assert sd.is_static is True
        assert sd.multiple_of == 32
        assert sd.default == 128

    def test_round_trip(self):
        sd = SymbolicDim("K", min=32, max=8192, is_static=True, multiple_of=32, default=256)
        sd2 = SymbolicDim.from_dict(sd.to_dict())
        assert sd2 == sd


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


class TestShapeConstraint:
    def test_basic(self):
        sc = ShapeConstraint("S % 128 == 0", "tile alignment")
        assert sc.expr == "S % 128 == 0"
        assert sc.reason == "tile alignment"

    def test_round_trip(self):
        sc = ShapeConstraint("H * D == model_dim", "head consistency")
        assert ShapeConstraint.from_dict(sc.to_dict()) == sc


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
        td2 = TensorDesc.from_dict(td.to_dict())
        assert td2.dtype == td.dtype
        assert isinstance(td2.shape[1], SymbolicDim)
        assert td2.shape[1].name == "S"


class TestParam:
    def test_basic(self):
        p = Param(name="X", shape=[128, 768], dtype="f16")
        td = p.to_tensor_desc()
        assert td.shape == [128, 768]

    def test_round_trip(self):
        p = Param(name="A", shape=[SymbolicDim("M"), 768], dtype="f16")
        p2 = Param.from_dict(p.to_dict())
        assert p2.name == "A"
        assert isinstance(p2.shape[0], SymbolicDim)


class TestInputRef:
    def test_param_ref(self):
        pr = ParamRef(name="X")
        assert ParamRef.from_dict(pr.to_dict()).name == "X"

    def test_node_ref(self):
        nr = NodeRef(id="matmul_0")
        assert NodeRef.from_dict(nr.to_dict()).id == "matmul_0"

    def test_from_dict_structured_param(self):
        ref = input_ref_from_dict({"ref": "param", "name": "X"})
        assert isinstance(ref, ParamRef)
        assert ref.name == "X"

    def test_from_dict_structured_node(self):
        ref = input_ref_from_dict({"ref": "node", "id": "relu_0"})
        assert isinstance(ref, NodeRef)
        assert ref.id == "relu_0"


class TestNode:
    def test_basic(self):
        node = Node(
            id="matmul_0",
            op="matmul",
            inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
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

    def test_attrs(self):
        node = Node(
            id="softmax_0",
            op="softmax",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[128, 768], dtype="f16"),
            semantics=Semantics(computation="softmax"),
            attrs={"axis": -1},
        )
        assert node.to_dict()["attrs"] == {"axis": -1}


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
        assert "values" in mon.outputs
        assert "indices" in mon.outputs


class TestConditionalNode:
    def test_basic(self):
        cn = ConditionalNode(
            id="dispatch_0",
            predicate='dim("S") <= 512',
            true_branch=["attn_short"],
            false_branch=["attn_long"],
            output=TensorDesc(shape=[8, SymbolicDim("S"), 64], dtype="f16"),
        )
        assert cn.true_branch == ["attn_short"]
        assert cn.false_branch == ["attn_long"]


class TestEdge:
    def test_round_trip(self):
        e = Edge(from_node="matmul_0", to_node="gelu_0", tensor_name="C", lifetime="persistent")
        assert Edge.from_dict(e.to_dict()) == e


class TestFusionGroup:
    def test_round_trip(self):
        fg = FusionGroup(id="fuse_0", nodes=["a", "b"], fusion_type="vertical")
        assert FusionGroup.from_dict(fg.to_dict()) == fg


class TestSemanticIR:
    def _make_simple_ir(self) -> SemanticIR:
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
        assert ir.version == "0.1.0"
        assert len(ir.params) == 2
        assert len(ir.nodes) == 1
        assert len(ir.edges) == 2

    def test_json_round_trip(self):
        ir = self._make_simple_ir()
        assert SemanticIR.from_json(ir.to_json()).to_json() == ir.to_json()

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
        assert SemanticIR.from_json(ir.to_json()).to_json() == ir.to_json()

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
        ir2 = SemanticIR.from_json(ir.to_json())
        assert isinstance(ir2.get_node("topk_0"), MultiOutputNode)

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
        ir2 = SemanticIR.from_json(ir.to_json())
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
        ir2 = SemanticIR.from_json(ir.to_json())
        assert len(ir2.fusion_groups) == 1
        assert ir2.fusion_groups[0].fusion_type == "epilogue"

    def test_metadata(self):
        ir = self._make_simple_ir()
        ir.metadata = {"source": "test", "version": 1}
        assert SemanticIR.from_json(ir.to_json()).metadata == {"source": "test", "version": 1}

    def test_requires_structured_input_refs(self):
        payload = {
            "version": "1.0.0",
            "kernel_id": "invalid_unstructured_payload",
            "params": [{"name": "X", "shape": [128], "dtype": "f32"}],
            "nodes": [{
                "id": "relu_0",
                "op": "relu",
                "inputs": {"X": "X"},
                "output": {"shape": [128], "dtype": "f32"},
                "semantics": {"computation": "relu(X)"},
            }],
            "edges": [],
            "return_node": "relu_0",
        }
        try:
            SemanticIR.from_dict(payload)
            raised = False
        except Exception:
            raised = True
        assert raised, "unstructured InputRef payload should be rejected"

    def test_json_is_valid(self):
        ir = self._make_simple_ir()
        json.loads(ir.to_json())
