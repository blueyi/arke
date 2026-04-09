# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for arke.compiler.semantic_passes and arke.compiler.semantic_pipeline.

Tests:
    - Shape inference pass: verify inferred shapes match expected
    - SSA validation pass: error cases
    - Rationale preservation pass: cross-IR validation
    - Full pipeline with all passes
    - All 46 .ak files pass through semantic pipeline
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arke.compiler.semantic_passes import (
    rationale_preservation_pass,
    semantic_shape_inference_pass,
    semantic_ssa_validation_pass,
)
from arke.compiler.semantic_pipeline import SemanticPassPipeline, SemanticPassResult
from arke.compiler.pipeline import ArkePipeline
from arke.compiler.validator import validate_semantic_ir
from arke.ir.semantic import (
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    ShapeConstraint,
    SymbolicDim,
    TensorDesc,
)
from arke.ir.strategy import Decision, Rationale, StrategyIR

# ---- Fixtures ----

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"
ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))


@pytest.fixture
def pipeline():
    return ArkePipeline()


# ---- Helpers ----

def _make_relu_ir():
    """Create a minimal relu SemanticIR."""
    ir = SemanticIR(kernel_id="test_relu")
    ir.add_param(Param(name="X", shape=[128, 3072], dtype="f32"))
    ir.add_node(Node(
        id="n0",
        op="relu",
        inputs={"X": ParamRef(name="X")},
        output=TensorDesc(shape=[128, 3072], dtype="f32"),
        semantics=Semantics(computation="relu(X)"),
    ))
    ir.return_node = "n0"
    return ir


def _make_matmul_ir():
    """Create a matmul SemanticIR."""
    ir = SemanticIR(kernel_id="test_matmul")
    ir.add_param(Param(name="A", shape=[1024, 512], dtype="f32"))
    ir.add_param(Param(name="B", shape=[512, 2048], dtype="f32"))
    ir.add_node(Node(
        id="n0",
        op="matmul",
        inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
        output=TensorDesc(shape=[1024, 2048], dtype="f32"),
        semantics=Semantics(
            computation="C[i,j] = sum(A[i,k]*B[k,j], k)",
            index_vars=["i", "j", "k"],
            reduction_axes=["k"],
        ),
    ))
    ir.return_node = "n0"
    return ir


def _make_chain_ir():
    """Create a two-node chain: matmul -> relu."""
    ir = SemanticIR(kernel_id="test_chain")
    ir.add_param(Param(name="A", shape=[64, 128], dtype="f32"))
    ir.add_param(Param(name="B", shape=[128, 256], dtype="f32"))
    ir.add_node(Node(
        id="matmul_0",
        op="matmul",
        inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
        output=TensorDesc(shape=[64, 256], dtype="f32"),
        semantics=Semantics(computation="C = A @ B"),
    ))
    ir.add_node(Node(
        id="relu_1",
        op="relu",
        inputs={"X": NodeRef(id="matmul_0")},
        output=TensorDesc(shape=[64, 256], dtype="f32"),
        semantics=Semantics(computation="relu(C)"),
    ))
    ir.return_node = "relu_1"
    return ir


# ============================================================
# Test Class: SemanticShapeInferencePass
# ============================================================


class TestShapeInferencePass:
    """Test semantic_shape_inference_pass."""

    def test_relu_shape_inferred(self):
        """Shape inference should update relu output shape."""
        ir = _make_relu_ir()
        errors = semantic_shape_inference_pass(ir)
        assert errors == []
        assert ir.nodes[0].output.shape == [128, 3072]

    def test_matmul_shape_inferred(self):
        """Shape inference should compute matmul output shape correctly."""
        ir = _make_matmul_ir()
        errors = semantic_shape_inference_pass(ir)
        assert errors == []
        assert ir.nodes[0].output.shape == [1024, 2048]

    def test_chain_shapes_inferred(self):
        """Shape inference should propagate through a node chain."""
        ir = _make_chain_ir()
        errors = semantic_shape_inference_pass(ir)
        assert errors == []
        # matmul: [64,128] x [128,256] -> [64,256]
        assert ir.nodes[0].output.shape == [64, 256]
        # relu: same_as_input -> [64,256]
        assert ir.nodes[1].output.shape == [64, 256]

    def test_symbolic_dims_propagated_for_same_as_input(self):
        """same_as_input rules should preserve symbolic dims through inference."""
        ir = SemanticIR(kernel_id="test_symbolic")
        ir.add_param(Param(
            name="X",
            shape=[SymbolicDim("B"), 128],
            dtype="f32",
        ))
        ir.add_symbolic_dim(SymbolicDim("B"))
        ir.add_node(Node(
            id="n0",
            op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[1, 1], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.return_node = "n0"

        errors = semantic_shape_inference_pass(ir)
        assert errors == []
        assert isinstance(ir.nodes[0].output.shape[0], SymbolicDim)
        assert ir.nodes[0].output.shape[0].name == "B"
        assert ir.nodes[0].output.shape[1] == 128

    def test_symbolic_matmul_shape_inferred(self):
        """matmul_rule should propagate symbolic batch/row dims."""
        ir = SemanticIR(kernel_id="test_symbolic_matmul")
        ir.add_param(Param(name="A", shape=[SymbolicDim("M"), 64], dtype="f32"))
        ir.add_param(Param(name="B", shape=[64, 256], dtype="f32"))
        ir.add_symbolic_dim(SymbolicDim("M", min=1, multiple_of=32, default=128))
        ir.add_node(Node(
            id="n0",
            op="matmul",
            inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
            output=TensorDesc(shape=[1, 1], dtype="f32"),
            semantics=Semantics(computation="A @ B"),
        ))
        ir.return_node = "n0"

        errors = semantic_shape_inference_pass(ir)
        assert errors == []
        assert isinstance(ir.nodes[0].output.shape[0], SymbolicDim)
        assert ir.nodes[0].output.shape[0].name == "M"
        assert ir.nodes[0].output.shape[1] == 256

    def test_empty_ir_no_errors(self):
        """Empty IR should have no shape inference errors."""
        ir = SemanticIR(kernel_id="empty")
        errors = semantic_shape_inference_pass(ir)
        assert errors == []

    def test_unknown_op_skipped(self):
        """Unknown ops should be silently skipped (caught by SSA pass)."""
        ir = SemanticIR(kernel_id="test_unknown")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0",
            op="totally_fake_op",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="fake(X)"),
        ))
        ir.return_node = "n0"

        errors = semantic_shape_inference_pass(ir)
        assert errors == []  # Not an error in shape pass

    def test_shape_inference_updates_in_place(self):
        """Shape inference should mutate node.output.shape in place."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="A", shape=[32, 64], dtype="f32"))
        ir.add_param(Param(name="B", shape=[64, 16], dtype="f32"))
        # Start with a wrong output shape
        ir.add_node(Node(
            id="n0",
            op="matmul",
            inputs={"A": ParamRef(name="A"), "B": ParamRef(name="B")},
            output=TensorDesc(shape=[1, 1], dtype="f32"),
            semantics=Semantics(computation="A @ B"),
        ))
        ir.return_node = "n0"

        errors = semantic_shape_inference_pass(ir)
        assert errors == []
        # Should be updated to correct shape
        assert ir.nodes[0].output.shape == [32, 16]


# ============================================================
# Test Class: SemanticSSAValidationPass
# ============================================================


class TestSSAValidationPass:
    """Test semantic_ssa_validation_pass."""

    def test_valid_ir_no_errors(self):
        """A well-formed IR should produce no errors."""
        ir = _make_relu_ir()
        errors = semantic_ssa_validation_pass(ir)
        assert errors == []

    def test_duplicate_node_id(self):
        """Duplicate node IDs should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.return_node = "n0"

        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("Duplicate node ID" in e for e in errors)

    def test_unknown_op_detected(self):
        """Unknown ops should be reported."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0", op="nonexistent_op",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="fake(X)"),
        ))
        ir.return_node = "n0"

        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("unknown op" in e for e in errors)

    def test_invalid_symbolic_dim_constraints_detected(self):
        """Structural SymbolicDim constraint errors should be reported."""
        ir = SemanticIR(kernel_id="test_invalid_symbolic")
        ir.add_symbolic_dim(SymbolicDim("N", min=128, max=64, multiple_of=32, default=96))
        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("min 128 > max 64" in e for e in errors)
        assert any("default 96 < min 128" in e for e in errors)

    def test_invalid_shape_constraint_reference_detected(self):
        """ShapeConstraint expressions should reference known symbolic dims only."""
        ir = SemanticIR(kernel_id="test_bad_constraint_ref")
        ir.add_symbolic_dim(SymbolicDim("M", min=1))
        ir.add_shape_constraint(ShapeConstraint("N >= 64", "unknown dim"))
        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("unknown symbolic dim 'N'" in e for e in errors)

    def test_invalid_param_ref(self):
        """ParamRef to non-existent param should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="MISSING")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.return_node = "n0"

        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("ParamRef" in e and "MISSING" in e for e in errors)

    def test_invalid_node_ref(self):
        """NodeRef to non-existent node should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": NodeRef(id="ghost")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.return_node = "n0"

        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("NodeRef" in e and "ghost" in e for e in errors)

    def test_forward_reference_detected(self):
        """Forward references should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": NodeRef(id="n1")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.add_node(Node(
            id="n1", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.return_node = "n1"

        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("forward reference" in e.lower() or "defined later" in e for e in errors)

    def test_invalid_return_node(self):
        """Invalid return_node should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(Node(
            id="n0", op="relu",
            inputs={"X": ParamRef(name="X")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="relu(X)"),
        ))
        ir.return_node = "nonexistent"

        errors = semantic_ssa_validation_pass(ir)
        assert len(errors) >= 1
        assert any("return_node" in e for e in errors)


# ============================================================
# Test Class: RationalePreservationPass
# ============================================================


class TestRationalePreservationPass:
    """Test rationale_preservation_pass."""

    def test_no_strategy_noop(self):
        """No strategy should return empty warnings."""
        ir = _make_relu_ir()
        warnings = rationale_preservation_pass(ir, None)
        assert warnings == []

    def test_valid_fuse_reference(self):
        """Fuse referencing valid node IDs should produce no warnings."""
        ir = _make_chain_ir()
        strategy = StrategyIR(kernel_id="test_chain", target_hw="nvidia_ampere")
        strategy.fuse(
            ops=["matmul_0", "relu_1"],
            fusion_type="epilogue",
            rationale="Fuse matmul+relu",
        )

        warnings = rationale_preservation_pass(ir, strategy)
        assert warnings == []

    def test_invalid_fuse_reference(self):
        """Fuse referencing non-existent node should produce warning."""
        ir = _make_relu_ir()
        strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
        strategy.fuse(
            ops=["n0", "ghost_node"],
            fusion_type="epilogue",
            rationale="Bad fuse",
        )

        warnings = rationale_preservation_pass(ir, strategy)
        assert len(warnings) >= 1
        assert any("ghost_node" in w for w in warnings)

    def test_valid_place_reference(self):
        """Place referencing valid param should produce no warnings."""
        ir = _make_relu_ir()
        strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
        strategy.place(tensor="X_tile", memory="shared", rationale="Place X in shared")

        warnings = rationale_preservation_pass(ir, strategy)
        assert warnings == []  # "X_tile" base is "X" which is a param

    def test_invalid_place_reference(self):
        """Place referencing unknown tensor should produce warning."""
        ir = _make_relu_ir()
        strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
        strategy.place(
            tensor="UNKNOWN_tensor",
            memory="shared",
            rationale="Bad placement",
        )

        warnings = rationale_preservation_pass(ir, strategy)
        assert len(warnings) >= 1
        assert any("UNKNOWN_tensor" in w for w in warnings)

    def test_tile_no_cross_ref(self):
        """Tile decisions don't reference nodes — should produce no warnings."""
        ir = _make_relu_ir()
        strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
        strategy.tile(loop="row", factors=[4], rationale="Simple tile")

        warnings = rationale_preservation_pass(ir, strategy)
        assert warnings == []

    def test_rationale_text_in_warning(self):
        """Warning should include the rationale text."""
        ir = _make_relu_ir()
        strategy = StrategyIR(kernel_id="test", target_hw="nvidia_ampere")
        strategy.fuse(
            ops=["nonexistent"],
            fusion_type="epilogue",
            rationale="This should appear in the warning",
        )

        warnings = rationale_preservation_pass(ir, strategy)
        assert len(warnings) >= 1
        assert any("This should appear in the warning" in w for w in warnings)


# ============================================================
# Test Class: SemanticPassPipeline
# ============================================================


class TestSemanticPassPipeline:
    """Test SemanticPassPipeline orchestration."""

    def test_empty_pipeline(self):
        """Empty pipeline should succeed immediately."""
        pipe = SemanticPassPipeline("test")
        ir = _make_relu_ir()
        result = pipe.run(ir)
        assert result.success
        assert result.errors == []
        assert result.passes_run == []
        assert result.duration_ms >= 0

    def test_single_pass_success(self):
        """Pipeline with one passing pass should succeed."""
        pipe = SemanticPassPipeline("test")
        pipe.add_pass(semantic_ssa_validation_pass)

        ir = _make_relu_ir()
        result = pipe.run(ir)
        assert result.success
        assert result.errors == []
        assert "semantic_ssa_validation_pass" in result.passes_run

    def test_chained_passes(self):
        """Pipeline with multiple passes should run them in order."""
        pipe = SemanticPassPipeline("test")
        pipe.add_pass(semantic_ssa_validation_pass)
        pipe.add_pass(semantic_shape_inference_pass)

        ir = _make_relu_ir()
        result = pipe.run(ir)
        assert result.success
        assert len(result.passes_run) == 2
        assert result.passes_run[0] == "semantic_ssa_validation_pass"
        assert result.passes_run[1] == "semantic_shape_inference_pass"

    def test_pipeline_stops_on_error(self):
        """Pipeline should stop at the first failing pass."""
        # Create an IR that will fail SSA validation
        ir = SemanticIR(kernel_id="bad")
        ir.add_node(Node(
            id="n0", op="fake_op",
            inputs={"X": ParamRef(name="MISSING")},
            output=TensorDesc(shape=[4, 8], dtype="f32"),
            semantics=Semantics(computation="fake(X)"),
        ))
        ir.return_node = "n0"

        pipe = SemanticPassPipeline("test")
        pipe.add_pass(semantic_ssa_validation_pass)
        pipe.add_pass(semantic_shape_inference_pass)

        result = pipe.run(ir)
        assert not result.success
        assert len(result.errors) > 0
        # Should have only run the first pass
        assert len(result.passes_run) == 1
        assert result.passes_run[0] == "semantic_ssa_validation_pass"

    def test_pipeline_exception_handling(self):
        """Pipeline should handle passes that raise exceptions."""
        def bad_pass(ir: SemanticIR) -> list[str]:
            raise RuntimeError("Pass exploded!")

        pipe = SemanticPassPipeline("test")
        pipe.add_pass(bad_pass, name="exploding_pass")

        ir = _make_relu_ir()
        result = pipe.run(ir)
        assert not result.success
        assert any("exploding_pass" in e and "exploded" in e for e in result.errors)

    def test_add_pass_chaining(self):
        """add_pass should return self for method chaining."""
        pipe = SemanticPassPipeline("test")
        ret = pipe.add_pass(semantic_ssa_validation_pass)
        assert ret is pipe

    def test_passes_property(self):
        """passes property should return pass names."""
        pipe = SemanticPassPipeline("test")
        pipe.add_pass(semantic_ssa_validation_pass)
        pipe.add_pass(semantic_shape_inference_pass, name="shape_infer")

        names = pipe.passes
        assert names == ["semantic_ssa_validation_pass", "shape_infer"]

    def test_custom_pass_name(self):
        """Custom pass name should be used in results."""
        pipe = SemanticPassPipeline("test")
        pipe.add_pass(semantic_ssa_validation_pass, name="my_validator")

        ir = _make_relu_ir()
        result = pipe.run(ir)
        assert "my_validator" in result.passes_run

    def test_duration_tracked(self):
        """Pipeline should track execution duration."""
        pipe = SemanticPassPipeline("test")
        pipe.add_pass(semantic_ssa_validation_pass)
        pipe.add_pass(semantic_shape_inference_pass)

        ir = _make_relu_ir()
        result = pipe.run(ir)
        assert result.duration_ms >= 0


# ============================================================
# Test Class: Pipeline Integration in ArkePipeline
# ============================================================


class TestPipelineIntegration:
    """Test that ArkePipeline runs semantic passes during compilation."""

    def test_compile_runs_ssa_validation(self, pipeline):
        """Compilation should run SSA validation via semantic pipeline."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success

    def test_compile_runs_shape_inference(self, pipeline):
        """Compilation should run shape inference and update node shapes."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
        assert result.success
        # matmul [1024,1024] x [1024,1024] => [1024,1024]
        assert result.semantic_ir.nodes[0].output.shape == [1024, 1024]

    def test_compile_chain_shapes_propagated(self, pipeline):
        """Multi-node compilation should propagate shapes through chain."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "05_matmul_gelu.ak"))
        assert result.success
        # matmul_0: [128,768] x [768,3072] => [128,3072]
        assert result.semantic_ir.nodes[0].output.shape == [128, 3072]
        # gelu_1: same_as_input => [128,3072]
        assert result.semantic_ir.nodes[1].output.shape == [128, 3072]


# ============================================================
# Test Class: All 46 .ak Files Through Semantic Pipeline
# ============================================================


class TestAllFilesSemantic:
    """All 46 .ak files should pass through the semantic pipeline."""

    @pytest.mark.parametrize(
        "ak_file",
        ALL_AK_FILES,
        ids=[f.stem for f in ALL_AK_FILES],
    )
    def test_semantic_pipeline_all_files(self, pipeline, ak_file):
        """Each .ak file should compile and pass all semantic passes."""
        result = pipeline.compile_file(str(ak_file))
        assert result.success, (
            f"Semantic pipeline failed for {ak_file.name}: {result.errors}"
        )

    @pytest.mark.parametrize(
        "ak_file",
        ALL_AK_FILES,
        ids=[f.stem for f in ALL_AK_FILES],
    )
    def test_standalone_semantic_pipeline(self, pipeline, ak_file):
        """Each .ak file's SemanticIR should pass a standalone semantic pipeline."""
        result = pipeline.compile_file(str(ak_file))
        assert result.success

        # Run standalone pipeline
        pipe = SemanticPassPipeline("standalone_test")
        pipe.add_pass(semantic_ssa_validation_pass)
        pipe.add_pass(semantic_shape_inference_pass)

        pass_result = pipe.run(result.semantic_ir)
        assert pass_result.success, (
            f"Standalone semantic pipeline failed for {ak_file.name}: "
            f"{pass_result.errors}"
        )
        assert len(pass_result.passes_run) == 2
