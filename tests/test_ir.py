# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arke IR data structures."""

from arke.ir.builder import KernelBuilder
from arke.ir.ops.catalog import OP_CATALOG, get_op, is_fusable_epilogue, list_ops
from arke.ir.semantic import Edge, FusionGroup, Node, SemanticGraph, Semantics, TensorDesc
from arke.ir.strategy import StrategyIR

# ============================================================
# Semantic IR Tests
# ============================================================

def test_semantic_graph_creation():
    """Test creating a basic semantic graph."""
    graph = SemanticGraph(graph_id="test_matmul_relu")

    matmul_node = Node(
        id="matmul_0",
        op="matmul",
        inputs={
            "A": TensorDesc(shape=[1024, 512], dtype="f16"),
            "B": TensorDesc(shape=[512, 2048], dtype="f16", layout="col_major"),
        },
        output=TensorDesc(shape=[1024, 2048], dtype="f16"),
        semantics=Semantics(
            computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
            index_vars=["i", "j", "k"],
            reduction_axes=["k"],
            properties=["associative", "distributive"],
        ),
    )
    graph.add_node(matmul_node)

    relu_node = Node(
        id="relu_0",
        op="relu",
        inputs={"X": "@matmul_0.output"},
        output=TensorDesc(shape=[1024, 2048], dtype="f16"),
        semantics=Semantics(
            computation="Y[i,j] = max(X[i,j], 0)",
            properties=["elementwise", "monotonic"],
        ),
    )
    graph.add_node(relu_node)

    graph.add_edge(Edge(
        from_node="matmul_0",
        to_node="relu_0",
        tensor_name="intermediate",
        lifetime="local",
    ))

    graph.add_fusion_group(FusionGroup(
        id="fg_0",
        nodes=["matmul_0", "relu_0"],
        fusion_type="epilogue",
    ))

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert len(graph.fusion_groups) == 1
    assert graph.get_node("matmul_0") is not None
    assert graph.get_node("relu_0") is not None
    assert graph.get_node("nonexistent") is None


def test_semantic_graph_serialization():
    """Test JSON serialization."""
    graph = SemanticGraph(graph_id="test_serialize")
    graph.add_node(Node(
        id="relu_0",
        op="relu",
        inputs={"X": "@input"},
        output=TensorDesc(shape=[1024], dtype="f32"),
        semantics=Semantics(computation="Y = max(X, 0)", properties=["elementwise"]),
    ))

    json_str = graph.to_json()
    assert '"graph_id": "test_serialize"' in json_str
    assert '"relu_0"' in json_str


# ============================================================
# Strategy IR Tests (renamed from ScheduleTree)
# ============================================================

def test_strategy_ir_creation():
    """Test creating a strategy IR with decisions."""
    strategy = StrategyIR(kernel_id="test_matmul_relu", target_hw="nvidia_ampere")

    strategy.tile("i", [64, 16], rationale="L2 cache line = 64, warp size = 16")
    strategy.tile("j", [128, 8], rationale="maximize memory coalescing")
    strategy.reorder(
        ["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"],
        rationale="outer parallel, inner reuse",
    )
    strategy.fuse(["matmul", "relu"], "epilogue", rationale="avoid extra global write")
    strategy.parallel(
        ["i_outer", "j_outer"],
        {"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"},
    )
    strategy.place("A_tile", "shared", rationale="broadcast along j")

    assert strategy.decision_count == 6
    assert strategy.decisions[0].kind == "tile"
    assert strategy.decisions[0].step == 1
    assert strategy.decisions[0].rationale.text == "L2 cache line = 64, warp size = 16"
    assert strategy.decisions[3].kind == "fuse"


def test_strategy_ir_rollback():
    """Test rollback removes decisions from the end."""
    strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
    strategy.tile("i", [64])
    strategy.tile("j", [128])
    strategy.fuse(["a", "b"])

    assert strategy.decision_count == 3
    removed = strategy.pop_decisions(2)
    assert len(removed) == 2
    assert strategy.decision_count == 1
    assert strategy.decisions[0].kind == "tile"


def test_strategy_ir_serialization():
    """Test Strategy IR JSON round-trip."""
    strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
    strategy.tile("i", [64], rationale="test rationale")
    strategy.fuse(["a", "b"], rationale="fuse reason")

    json_str = strategy.to_json()
    assert '"kind": "tile"' in json_str
    assert '"test rationale"' in json_str

    # Round-trip
    restored = StrategyIR.from_json(json_str)
    assert restored.decision_count == 2
    assert restored.decisions[0].kind == "tile"
    assert restored.decisions[0].rationale.text == "test rationale"
    assert restored.kernel_id == "test"


def test_strategy_ir_summary():
    """Test human-readable summary."""
    strategy = StrategyIR(kernel_id="matmul", target_hw="ampere")
    strategy.tile("i", [64], rationale="cache")
    summary = strategy.summary()
    assert "matmul" in summary
    assert "tile" in summary
    assert "cache" in summary


# ============================================================
# Op Catalog Tests
# ============================================================

def test_op_catalog_completeness():
    """P0 catalog should have 10 operators."""
    assert len(OP_CATALOG) == 10


def test_op_catalog_lookup():
    """Test operator lookup."""
    matmul = get_op("matmul")
    assert matmul.category == "compute"
    assert "associative" in matmul.properties
    assert matmul.numpy_ref == "np.matmul(A, B)"


def test_op_catalog_filter():
    """Test filtering by category."""
    elementwise = list_ops("elementwise")
    assert len(elementwise) == 4  # relu, gelu, add, mul
    assert all(op.category == "elementwise" for op in elementwise)


def test_fusable_epilogue():
    """Test epilogue fusion detection."""
    assert is_fusable_epilogue("relu") is True
    assert is_fusable_epilogue("add") is True
    assert is_fusable_epilogue("softmax") is False
    assert is_fusable_epilogue("matmul") is False


# ============================================================
# Builder Tests
# ============================================================

def test_kernel_builder_basic():
    """Test building a simple matmul + relu kernel."""
    b = KernelBuilder("fused_matmul_relu")
    b.param("A", [1024, 512], "f16")
    b.param("B", [512, 2048], "f16")
    m = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=m)
    b.returns(r, [1024, 2048], "f16")
    ir = b.build()

    assert ir.graph_id == "fused_matmul_relu"
    assert len(ir.nodes) == 2
    assert ir.nodes[0].op == "matmul"
    assert ir.nodes[1].op == "relu"
    assert len(ir.edges) == 1  # matmul → relu
    assert len(ir.fusion_groups) == 1  # auto-detected epilogue


def test_kernel_builder_single_op():
    """Test building a single-op kernel."""
    b = KernelBuilder("simple_relu")
    b.param("X", [1024], "f32")
    r = b.op("relu", X="X")
    b.returns(r, [1024], "f32")
    ir = b.build()

    assert len(ir.nodes) == 1
    assert len(ir.edges) == 0
    assert len(ir.fusion_groups) == 0
