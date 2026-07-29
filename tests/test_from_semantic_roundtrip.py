# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""K-H1 — Official SemanticIR ↔ IRGraph bridge round-trip golden tests.

``IRGraph.from_semantic`` is the canonical SemanticIR → backend-IRGraph
construction path (replaces the ad-hoc single-node graphs each backend entry
used to hand-build). These tests are the golden contract for that bridge:

1. Structural fidelity of ``from_semantic`` (params → inputs, nodes → IRNodes,
   InputRef → SSA value name, dtype vocabulary translation, symbolic-dim
   resolution).
2. Round-trip equivalence: SemanticIR → IRGraph → SemanticIR recovers the
   operator DAG (params, node ids/ops, output shapes/dtypes, input keys,
   return node/ports) with zero drift.
3. Every op in the SSOT benchmark catalog survives the round-trip.
"""

from __future__ import annotations

import pytest

from arke.ir.converters import ast_to_semantic
from arke.ir.graph import (
    IRGraph,
    graph_dtype_to_semantic,
    semantic_dtype_to_graph,
)
from arke.ir.ops.registry import REGISTRY
from arke.ir.semantic import (
    MultiOutputNode,
    Node,
    NodeRef,
    Param,
    ParamRef,
    Semantics,
    SemanticIR,
    SymbolicDim,
    TensorDesc,
)
from arke.lang.grammar import parse_string
from benchmarks.op_registry import ALL_OPS


# ─── dtype vocabulary bridge ────────────────────────────────────────────────

class TestDtypeBridge:
    def test_forward_map(self):
        assert semantic_dtype_to_graph("f16") == "float16"
        assert semantic_dtype_to_graph("bf16") == "bfloat16"
        assert semantic_dtype_to_graph("f32") == "float32"
        assert semantic_dtype_to_graph("i64") == "int64"

    def test_inverse_map(self):
        assert graph_dtype_to_semantic("float16") == "f16"
        assert graph_dtype_to_semantic("float32") == "f32"
        assert graph_dtype_to_semantic("int64") == "i64"

    def test_bijective_over_vocab(self):
        for sem in ["f16", "bf16", "f32", "f64", "i8", "i32", "i64", "bool"]:
            assert graph_dtype_to_semantic(semantic_dtype_to_graph(sem)) == sem

    def test_unknown_passthrough(self):
        # Already-verbose or unknown names pass through unchanged so
        # hand-built graphs with torch-style dtype names keep working.
        assert semantic_dtype_to_graph("float16") == "float16"
        assert graph_dtype_to_semantic("f16") == "f16"


# ─── from_semantic structural fidelity ──────────────────────────────────────

def _two_node_chain() -> SemanticIR:
    """matmul(A,B) → relu → out."""
    sem = SemanticIR(kernel_id="mm_relu")
    sem.add_param(Param(name="A", shape=[128, 64], dtype="f16"))
    sem.add_param(Param(name="B", shape=[64, 256], dtype="f16"))
    sem.add_node(Node(
        id="matmul_0", op="matmul",
        inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
        output=TensorDesc(shape=[128, 256], dtype="f16"),
        semantics=Semantics(computation="C=A@B"),
    ))
    sem.add_node(Node(
        id="relu_1", op="relu",
        inputs={"X": NodeRef(id="matmul_0")},
        output=TensorDesc(shape=[128, 256], dtype="f16"),
        semantics=Semantics(computation="relu(X)"),
    ))
    sem.return_node = "relu_1"
    return sem


class TestFromSemanticStructure:
    def test_params_become_inputs_with_translated_dtype(self):
        g = IRGraph.from_semantic(_two_node_chain())
        assert g.graph_inputs == ["A", "B"]
        assert g.values["A"].dtype == "float16"
        assert g.values["A"].is_input is True
        assert g.values["A"].shape == [128, 64]

    def test_nodes_and_ssa_value_names(self):
        g = IRGraph.from_semantic(_two_node_chain())
        assert [n.id for n in g.nodes] == ["matmul_0", "relu_1"]
        # NodeRef input resolves to producing node's SSA value name.
        assert g.nodes[1].inputs == {"X": "matmul_0"}
        # Each node produces a value named after its id.
        assert g.values["matmul_0"].shape == [128, 256]
        assert g.values["matmul_0"].dtype == "float16"

    def test_graph_outputs_from_return_node(self):
        g = IRGraph.from_semantic(_two_node_chain())
        assert g.graph_outputs == ["relu_1"]

    def test_metadata_stamped(self):
        g = IRGraph.from_semantic(_two_node_chain())
        assert g.metadata["source"] == "from_semantic"
        assert g.metadata["kernel_id"] == "mm_relu"

    def test_symbolic_dims_resolved_via_bindings_and_defaults(self):
        sem = SemanticIR(kernel_id="dyn")
        sem.add_symbolic_dim(SymbolicDim(name="S", default=512))
        sem.add_symbolic_dim(SymbolicDim(name="D"))
        sem.add_param(Param(
            name="X",
            shape=[SymbolicDim(name="S"), SymbolicDim(name="D")],
            dtype="f32",
        ))
        sem.add_node(Node(
            id="softmax_0", op="softmax",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(
                shape=[SymbolicDim(name="S"), SymbolicDim(name="D")],
                dtype="f32",
            ),
            semantics=Semantics(computation="softmax(X)"),
        ))
        sem.return_node = "softmax_0"
        # S resolves from default (512), D from explicit binding (64).
        g = IRGraph.from_semantic(sem, dim_bindings={"D": 64})
        assert g.values["X"].shape == [512, 64]

    def test_unresolved_symbolic_dim_is_minus_one(self):
        sem = SemanticIR(kernel_id="dyn2")
        sem.add_symbolic_dim(SymbolicDim(name="N"))  # no default, no binding
        sem.add_param(Param(name="X", shape=[SymbolicDim(name="N")], dtype="f32"))
        sem.add_node(Node(
            id="relu_0", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[SymbolicDim(name="N")], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        sem.return_node = "relu_0"
        g = IRGraph.from_semantic(sem)
        assert g.values["X"].shape == [-1]

    def test_conditional_node_rejected(self):
        from arke.ir.semantic import ConditionalNode
        sem = SemanticIR(kernel_id="cond")
        sem.add_node(ConditionalNode(
            id="c0", predicate='dim("S") <= 512',
            true_branch=["a"], false_branch=["b"],
            output=TensorDesc(shape=[128], dtype="f32"),
        ))
        with pytest.raises(ValueError, match="ConditionalNode"):
            IRGraph.from_semantic(sem)


# ─── Round-trip equivalence ──────────────────────────────────────────────────

def _assert_semantic_equivalent(a: SemanticIR, b: SemanticIR) -> None:
    """Assert two SemanticIRs are equivalent at the operator-DAG level."""
    assert a.kernel_id == b.kernel_id
    assert [(p.name, p.shape, p.dtype) for p in a.params] == \
           [(p.name, p.shape, p.dtype) for p in b.params]
    assert a.return_node == b.return_node
    assert set(a.return_ports) == set(b.return_ports)
    assert len(a.nodes) == len(b.nodes)
    for na, nb in zip(a.nodes, b.nodes):
        assert na.id == nb.id
        assert na.op == nb.op
        assert set(na.inputs.keys()) == set(nb.inputs.keys())
        # InputRef kinds & targets preserved.
        for k in na.inputs:
            ra, rb = na.inputs[k], nb.inputs[k]
            assert type(ra) is type(rb), f"{na.id}.{k} ref kind drift"
            assert getattr(ra, "name", getattr(ra, "id", None)) == \
                   getattr(rb, "name", getattr(rb, "id", None))
        if isinstance(na, MultiOutputNode):
            assert isinstance(nb, MultiOutputNode)
            assert set(na.outputs.keys()) == set(nb.outputs.keys())
            for port in na.outputs:
                assert na.outputs[port].shape == nb.outputs[port].shape
                assert na.outputs[port].dtype == nb.outputs[port].dtype
        else:
            assert na.output.shape == nb.output.shape
            assert na.output.dtype == nb.output.dtype


class TestRoundTrip:
    def test_two_node_chain_roundtrip(self):
        sem = _two_node_chain()
        sem2 = IRGraph.from_semantic(sem).to_semantic()
        _assert_semantic_equivalent(sem, sem2)

    def test_multi_output_roundtrip(self):
        sem = SemanticIR(kernel_id="topk_k")
        sem.add_param(Param(name="X", shape=[32, 100], dtype="f32"))
        sem.add_node(MultiOutputNode(
            id="topk_0", op="topk",
            inputs={"X": ParamRef(name="X")},
            outputs={
                "values": TensorDesc(shape=[32, 5], dtype="f32"),
                "indices": TensorDesc(shape=[32, 5], dtype="i64"),
            },
            semantics=Semantics(computation="topk(X,5)"),
            attrs={"k": 5},
        ))
        sem.return_node = "topk_0"
        sem.return_ports = ["values", "indices"]
        sem2 = IRGraph.from_semantic(sem).to_semantic()
        _assert_semantic_equivalent(sem, sem2)
        # Multi-output dtype preserved through the bridge.
        mo = sem2.get_node("topk_0")
        assert isinstance(mo, MultiOutputNode)
        assert mo.outputs["indices"].dtype == "i64"

    def test_from_ak_source_roundtrip(self):
        """End-to-end: .ak source → ast_to_semantic → bridge round-trip."""
        prog = parse_string('''
kernel mm_gelu(
    A: Tensor<[128, 64], f16>,
    B: Tensor<[64, 256], f16>
) -> Tensor<[128, 256], f16> {
    let C = matmul(A=A, B=B);
    let Y = gelu(X=C);
    return Y;
}
        ''')
        sem = ast_to_semantic(prog.kernels[0])
        sem2 = IRGraph.from_semantic(sem).to_semantic()
        _assert_semantic_equivalent(sem, sem2)


# ─── SSOT catalog coverage ───────────────────────────────────────────────────

def _build_single_node_semantic(op: str) -> SemanticIR:
    """Build a minimal real SemanticIR for ``op`` from its registry schema."""
    schema = REGISTRY.get(op)
    inputs = schema.inputs
    input_names = list(inputs.keys()) if hasattr(inputs, "keys") else list(inputs)
    sem = SemanticIR(kernel_id=op)
    for name in input_names:
        sem.add_param(Param(name=name, shape=[128, 256], dtype="f16"))
    node_inputs = {name: ParamRef(name=name) for name in input_names}
    sem.add_node(Node(
        id=f"{op}_0", op=op,
        inputs=node_inputs,
        output=TensorDesc(shape=[128, 256], dtype="f16"),
        semantics=Semantics(computation=f"{op}(...)"),
    ))
    sem.return_node = f"{op}_0"
    return sem


@pytest.mark.parametrize("op", sorted(ALL_OPS))
def test_every_catalog_op_roundtrips(op):
    """Every SSOT catalog op survives SemanticIR→IRGraph→SemanticIR."""
    sem = _build_single_node_semantic(op)
    g = IRGraph.from_semantic(sem)
    # from_semantic must produce a graph with one node carrying the op.
    assert len(g.nodes) == 1
    assert g.nodes[0].op == op
    sem2 = g.to_semantic()
    _assert_semantic_equivalent(sem, sem2)
