# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Pass Infrastructure + SSA Validator (S6 Track 2).

C2.1: PassPipeline basic operation
C2.2: ShapeInferencePass on matmul (2D), batch_matmul (3D), flash_attention (4D)
C2.3: SSAValidationPass — valid graphs + 5 invalid IR cases
C2.4: RationalePreservationPass
C2.5: Full pipeline integration
"""

import pytest

from arke.ir.graph import IRGraph, IRNode, IRValue
from arke.compiler.passes import (
    PassPipeline, PassContext, PassResult,
    ShapeInferencePass, SSAValidationPass, RationalePreservationPass,
)


# ── Helpers ────────────────────────────────────────────────────

def make_matmul_graph() -> IRGraph:
    """matmul: A[1024,512] @ B[512,2048] -> C[1024,2048]"""
    g = IRGraph(name="matmul_test")
    g.add_input("A", dtype="float32", shape=[1024, 512])
    g.add_input("B", dtype="float32", shape=[512, 2048])
    g.add_node(IRNode(
        id="n0", op="matmul",
        inputs={"A": "A", "B": "B"},
        outputs=["C"],
    ))
    g.set_outputs(["C"])
    return g


def make_batch_matmul_graph() -> IRGraph:
    """batch_matmul: A[4,64,32] @ B[4,32,128] -> C[4,64,128]"""
    g = IRGraph(name="batch_matmul_test")
    g.add_input("A", dtype="float32", shape=[4, 64, 32])
    g.add_input("B", dtype="float32", shape=[4, 32, 128])
    g.add_node(IRNode(
        id="n0", op="batch_matmul",
        inputs={"A": "A", "B": "B"},
        outputs=["C"],
    ))
    g.set_outputs(["C"])
    return g


def make_attention_graph() -> IRGraph:
    """flash_attention: Q,K,V [2,8,128,64] -> O[2,8,128,64]"""
    g = IRGraph(name="attention_test")
    g.add_input("Q", dtype="float32", shape=[2, 8, 128, 64])
    g.add_input("K", dtype="float32", shape=[2, 8, 128, 64])
    g.add_input("V", dtype="float32", shape=[2, 8, 128, 64])
    g.add_node(IRNode(
        id="n0", op="flash_attention",
        inputs={"Q": "Q", "K": "K", "V": "V"},
        outputs=["O"],
    ))
    g.set_outputs(["O"])
    return g


def make_multi_node_graph() -> IRGraph:
    """relu(matmul(A, B)): A[4,8] @ B[8,16] -> C -> relu(C) -> D"""
    g = IRGraph(name="multi_node")
    g.add_input("A", dtype="float32", shape=[4, 8])
    g.add_input("B", dtype="float32", shape=[8, 16])
    g.add_node(IRNode(id="n0", op="matmul", inputs={"A": "A", "B": "B"}, outputs=["C"]))
    g.add_node(IRNode(id="n1", op="relu", inputs={"X": "C"}, outputs=["D"]))
    g.set_outputs(["D"])
    return g


def make_rationale_graph() -> IRGraph:
    """matmul with @rationale annotation."""
    g = IRGraph(name="rationale_test")
    g.add_input("A", dtype="float32", shape=[1024, 512])
    g.add_input("B", dtype="float32", shape=[512, 2048])
    g.add_node(IRNode(
        id="n0", op="matmul",
        inputs={"A": "A", "B": "B"}, outputs=["C"],
        rationale="Tile 128x128 optimal for 6GB VRAM on RTX 3060",
    ))
    g.set_outputs(["C"])
    return g


# ── C2.1: PassPipeline ────────────────────────────────────────

class TestPassPipeline:

    def test_empty_pipeline(self):
        g = make_matmul_graph()
        pipeline = PassPipeline("empty")
        result = pipeline.run(g)
        assert result.success
        assert result.passes_run == []

    def test_pipeline_runs_passes_in_order(self):
        g = make_matmul_graph()
        pipeline = PassPipeline("ordered")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert result.success
        assert result.passes_run == ["SSAValidation", "ShapeInference"]

    def test_pipeline_stops_on_failure(self):
        """Pipeline with unknown op should fail at SSA validation."""
        g = IRGraph(name="bad")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="NONEXISTENT_OP", inputs={"X": "X"}, outputs=["Y"]))
        g.set_outputs(["Y"])
        pipeline = PassPipeline("fail")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert not result.success
        assert "SSAValidation" in result.passes_run
        assert "ShapeInference" not in result.passes_run


# ── C2.2: ShapeInferencePass ──────────────────────────────────

class TestShapeInferencePass:

    def test_matmul_2d(self):
        g = make_matmul_graph()
        ctx = PassContext(graph=g)
        result = ShapeInferencePass().run(ctx)
        assert result.success
        assert ctx.artifacts["shape_map"]["C"] == [1024, 2048]

    def test_batch_matmul_3d(self):
        g = make_batch_matmul_graph()
        ctx = PassContext(graph=g)
        result = ShapeInferencePass().run(ctx)
        assert result.success
        assert ctx.artifacts["shape_map"]["C"] == [4, 64, 128]

    def test_flash_attention_4d(self):
        g = make_attention_graph()
        ctx = PassContext(graph=g)
        result = ShapeInferencePass().run(ctx)
        assert result.success
        assert ctx.artifacts["shape_map"]["O"] == [2, 8, 128, 64]

    def test_multi_node_chain(self):
        g = make_multi_node_graph()
        ctx = PassContext(graph=g)
        result = ShapeInferencePass().run(ctx)
        assert result.success
        assert ctx.artifacts["shape_map"]["C"] == [4, 16]
        assert ctx.artifacts["shape_map"]["D"] == [4, 16]

    def test_softmax_shape(self):
        g = IRGraph(name="softmax")
        g.add_input("X", shape=[32, 1024])
        g.add_node(IRNode(id="n0", op="softmax", inputs={"X": "X"}, outputs=["Y"]))
        g.set_outputs(["Y"])
        ctx = PassContext(graph=g)
        result = ShapeInferencePass().run(ctx)
        assert result.success
        assert ctx.artifacts["shape_map"]["Y"] == [32, 1024]

    def test_layernorm_shape(self):
        g = IRGraph(name="layernorm")
        g.add_input("X", shape=[4, 768])
        g.add_input("W", shape=[768])
        g.add_input("B", shape=[768])
        g.add_node(IRNode(id="n0", op="layernorm", inputs={"X": "X", "W": "W", "B": "B"}, outputs=["Y"]))
        g.set_outputs(["Y"])
        ctx = PassContext(graph=g)
        result = ShapeInferencePass().run(ctx)
        assert result.success
        assert ctx.artifacts["shape_map"]["Y"] == [4, 768]


# ── C2.3: SSAValidationPass ───────────────────────────────────

class TestSSAValidationPass:

    def test_valid_matmul(self):
        g = make_matmul_graph()
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert result.success

    def test_valid_multi_node(self):
        g = make_multi_node_graph()
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert result.success

    def test_all_45_ops_single_node(self):
        """Each of the 45 ops should pass SSA validation as a single-node graph."""
        from arke.ir.ops.registry import REGISTRY
        for op in REGISTRY:
            g = IRGraph(name=f"test_{op.name}")
            for i, inp_name in enumerate(op.inputs.keys()):
                g.add_input(inp_name, shape=[4, 8])
            g.add_node(IRNode(
                id="n0", op=op.name,
                inputs={k: k for k in op.inputs.keys()},
                outputs=["out"],
            ))
            g.set_outputs(["out"])
            ctx = PassContext(graph=g)
            result = SSAValidationPass().run(ctx)
            assert result.success, f"SSA validation failed for op={op.name}: {ctx.errors()}"

    # ── 5 Invalid IR Cases ──

    def test_invalid_unknown_op(self):
        """Reject unknown op."""
        g = IRGraph(name="bad_op")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="FAKE_OP", inputs={"X": "X"}, outputs=["Y"]))
        g.set_outputs(["Y"])
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert not result.success

    def test_invalid_duplicate_def(self):
        """Reject duplicate value definition."""
        g = IRGraph(name="dup_def")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="relu", inputs={"X": "X"}, outputs=["Y"]))
        g.add_node(IRNode(id="n1", op="relu", inputs={"X": "X"}, outputs=["Y"]))  # dup Y
        g.set_outputs(["Y"])
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert not result.success
        assert any("Duplicate" in str(d) for d in ctx.errors())

    def test_invalid_undefined_use(self):
        """Reject use of undefined value."""
        g = IRGraph(name="undef_use")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="relu", inputs={"X": "NONEXISTENT"}, outputs=["Y"]))
        g.set_outputs(["Y"])
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert not result.success
        assert any("Undefined" in str(d) for d in ctx.errors())

    def test_invalid_self_reference(self):
        """Reject self-referential node (output = input)."""
        g = IRGraph(name="self_ref")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="relu", inputs={"X": "Y"}, outputs=["Y"]))  # Y uses Y
        g.set_outputs(["Y"])
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert not result.success

    def test_invalid_graph_output_undefined(self):
        """Reject graph output that is not defined."""
        g = IRGraph(name="bad_output")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="relu", inputs={"X": "X"}, outputs=["Y"]))
        g.set_outputs(["GHOST"])  # not defined
        ctx = PassContext(graph=g)
        result = SSAValidationPass().run(ctx)
        assert not result.success


# ── C2.4: RationalePreservationPass ───────────────────────────

class TestRationalePreservationPass:

    def test_preserves_rationale(self):
        g = make_rationale_graph()
        ctx = PassContext(graph=g)
        result = RationalePreservationPass().run(ctx)
        assert result.success
        assert "rationale_map" in ctx.artifacts
        assert ctx.artifacts["rationale_map"]["n0"] == "Tile 128x128 optimal for 6GB VRAM on RTX 3060"

    def test_no_rationale(self):
        g = make_matmul_graph()
        ctx = PassContext(graph=g)
        result = RationalePreservationPass().run(ctx)
        assert result.success
        assert ctx.artifacts["rationale_map"] == {}


# ── C2.5: Full Pipeline Integration ──────────────────────────

class TestFullPipeline:

    def test_standard_pipeline_matmul(self):
        """Full pipeline: SSA → Shape → Rationale on matmul."""
        g = make_matmul_graph()
        pipeline = PassPipeline("standard")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        pipeline.add_pass(RationalePreservationPass())
        result = pipeline.run(g)
        assert result.success
        assert result.passes_run == ["SSAValidation", "ShapeInference", "RationalePreservation"]
        assert result.artifacts["shape_map"]["C"] == [1024, 2048]

    def test_standard_pipeline_multi_node(self):
        """relu(matmul(A,B)) through full pipeline."""
        g = make_multi_node_graph()
        pipeline = PassPipeline("standard")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert result.success
        assert result.artifacts["shape_map"]["D"] == [4, 16]

    def test_standard_pipeline_with_rationale(self):
        g = make_rationale_graph()
        pipeline = PassPipeline("standard")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        pipeline.add_pass(RationalePreservationPass())
        result = pipeline.run(g)
        assert result.success
        assert result.artifacts["rationale_map"]["n0"].startswith("Tile")

    def test_pipeline_rejects_invalid_ir(self):
        """Pipeline should fail on invalid IR."""
        g = IRGraph(name="invalid")
        g.add_input("X", shape=[4, 8])
        g.add_node(IRNode(id="n0", op="FAKE", inputs={"X": "X"}, outputs=["Y"]))
        g.set_outputs(["Y"])
        pipeline = PassPipeline("standard")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(g)
        assert not result.success
