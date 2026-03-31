# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Basic tests for Arke IR data structures."""

from arke.ir.semantic import SemanticGraph, Node, TensorDesc, Semantics, Edge, FusionGroup
from arke.ir.schedule import ScheduleTree


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
    """Test JSON serialization round-trip."""
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


def test_schedule_tree_creation():
    """Test creating a schedule tree with decisions."""
    schedule = ScheduleTree(
        target_graph="test_matmul_relu",
        target_hw="nvidia_ampere",
    )

    schedule.tile("i", [64, 16], rationale="L2 cache line = 64, warp size = 16")
    schedule.tile("j", [128, 8], rationale="maximize memory coalescing")
    schedule.reorder(
        ["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"],
        rationale="outer parallel, inner reuse",
    )
    schedule.fuse(["matmul", "relu"], "epilogue", rationale="avoid extra global write")
    schedule.parallel(
        ["i_outer", "j_outer"],
        {"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"},
    )
    schedule.place("A_tile", "shared", rationale="broadcast along j")

    assert len(schedule.decisions) == 6
    assert schedule.decisions[0].kind == "tile"
    assert schedule.decisions[0].rationale.text == "L2 cache line = 64, warp size = 16"
    assert schedule.decisions[3].kind == "fuse"


def test_schedule_tree_serialization():
    """Test schedule tree JSON serialization."""
    schedule = ScheduleTree(target_graph="test", target_hw="nvidia_ampere")
    schedule.tile("i", [64], rationale="test rationale")

    json_str = schedule.to_json()
    assert '"kind": "tile"' in json_str
    assert '"test rationale"' in json_str
