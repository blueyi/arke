# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for arke.compiler.pipeline — End-to-end SemanticIR pipeline tests.

Tests the full path: .ak -> parse -> SemanticIR -> validate -> execute.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from arke.compiler.pipeline import ArkePipeline, CompilationResult
from arke.compiler.validator import validate_semantic_ir
from arke.ir.semantic import (
    Node,
    NodeRef,
    Param,
    ParamRef,
    SemanticIR,
    Semantics,
    SymbolicDim,
    TensorDesc,
)

# ---- Fixtures ----

OPERATORS_DIR = Path(__file__).resolve().parent.parent / "examples" / "operators"

ALL_AK_FILES = sorted(OPERATORS_DIR.glob("*.ak"))


@pytest.fixture
def pipeline():
    """Create a fresh ArkePipeline instance."""
    return ArkePipeline()


# ---- Helpers ----

def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute max absolute difference between two tensors."""
    return (a.float() - b.float()).abs().max().item()


# ============================================================
# Test Class: Compilation (all 46 .ak files)
# ============================================================


class TestCompileAll:
    """Test that all 46 .ak operator files compile without errors."""

    def test_compile_symbolic_where_kernel(self, pipeline):
        source = '''
kernel symbolic_relu(
    X: Tensor<[B, 128], f32>
) -> Tensor<[B, 128], f32>
where B: dynamic(min=32, multiple_of=32, default=128)
{
    let Y = relu(X=X);
    return Y;
}
        '''
        result = pipeline.compile_string(source)
        assert result.success, result.errors
        assert result.semantic_ir is not None
        assert len(result.semantic_ir.symbolic_dims) == 1
        assert result.semantic_ir.symbolic_dims[0].name == "B"
        assert isinstance(result.semantic_ir.nodes[0].output.shape[0], SymbolicDim)
        assert result.semantic_ir.nodes[0].output.shape[0].name == "B"

    def test_compile_invalid_symbolic_where_kernel_fails(self, pipeline):
        source = '''
kernel bad_symbolic_relu(
    X: Tensor<[B, 128], f32>
) -> Tensor<[B, 128], f32>
where B: dynamic(min=128, max=64, multiple_of=32, default=96)
{
    let Y = relu(X=X);
    return Y;
}
        '''
        result = pipeline.compile_string(source)
        assert not result.success
        assert any("min 128 > max 64" in e for e in result.errors)


    @pytest.mark.parametrize(
        "ak_file",
        ALL_AK_FILES,
        ids=[f.stem for f in ALL_AK_FILES],
    )
    def test_compile_ak_file(self, pipeline, ak_file):
        """Each .ak file should compile to a valid SemanticIR."""
        result = pipeline.compile_file(str(ak_file))

        assert result.success, (
            f"Compilation failed for {ak_file.name}: {result.errors}"
        )
        assert result.semantic_ir is not None
        assert result.kernel_name != ""
        assert len(result.semantic_ir.nodes) > 0
        assert len(result.semantic_ir.params) > 0
        assert result.semantic_ir.return_node != ""


# ============================================================
# Test Class: E2E Execution for Simple Ops
# ============================================================


class TestE2ESimpleOps:
    """End-to-end execution tests for simple ops.

    Compiles .ak file, executes with inputs, verifies output matches
    independent PyTorch baseline (torch.nn.functional or torch ops).
    """

    def test_e2e_relu(self, pipeline):
        """relu: Y = max(X, 0)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success

        X = torch.randn(128, 3072, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.nn.functional.relu(X)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"relu mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_add(self, pipeline):
        """add: Y = A + B"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "09_add.ak"))
        assert result.success

        A = torch.randn(128, 768, dtype=torch.float32)
        B = torch.randn(128, 768, dtype=torch.float32)
        outputs = pipeline.execute(result, {"A": A, "B": B})

        expected = A + B
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"add mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_matmul(self, pipeline):
        """matmul: C = A @ B"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
        assert result.success

        A = torch.randn(1024, 1024, dtype=torch.float32)
        B = torch.randn(1024, 1024, dtype=torch.float32)
        outputs = pipeline.execute(result, {"A": A, "B": B})

        expected = A @ B
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"matmul mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_gelu(self, pipeline):
        """gelu: Y = gelu(X)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "03_gelu.ak"))
        assert result.success

        X = torch.randn(1024, 4096, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.nn.functional.gelu(X)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"gelu mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_softmax(self, pipeline):
        """softmax: Y = softmax(X, axis=-1)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "02_softmax.ak"))
        assert result.success

        X = torch.randn(2048, 1024, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.nn.functional.softmax(X, dim=-1)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"softmax mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_silu(self, pipeline):
        """silu: Y = X * sigmoid(X)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "07_silu.ak"))
        assert result.success

        X = torch.randn(128, 11008, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.nn.functional.silu(X)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"silu mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_sigmoid(self, pipeline):
        """sigmoid: Y = 1 / (1 + exp(-X))"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "22_sigmoid.ak"))
        assert result.success

        X = torch.randn(128, 3072, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.sigmoid(X)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"sigmoid mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_tanh(self, pipeline):
        """tanh: Y = tanh(X)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "21_tanh.ak"))
        assert result.success

        X = torch.randn(128, 3072, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.tanh(X)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"tanh mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_neg(self, pipeline):
        """neg: Y = -X"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "25_neg.ak"))
        assert result.success

        X = torch.randn(128, 3072, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = -X
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"neg mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_exp(self, pipeline):
        """exp: Y = exp(X)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "26_exp.ak"))
        assert result.success

        # Use small values to avoid overflow
        X = torch.randn(128, 3072, dtype=torch.float32) * 0.5
        outputs = pipeline.execute(result, {"X": X})

        expected = torch.exp(X)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"exp mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_mul(self, pipeline):
        """mul: Y = A * B"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "10_mul.ak"))
        assert result.success

        A = torch.randn(128, 768, dtype=torch.float32)
        B = torch.randn(128, 768, dtype=torch.float32)
        outputs = pipeline.execute(result, {"A": A, "B": B})

        expected = A * B
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"mul mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_matmul_gelu_fused(self, pipeline):
        """matmul_gelu: Y = gelu(X @ W) -- multi-node pipeline."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "05_matmul_gelu.ak"))
        assert result.success
        assert len(result.semantic_ir.nodes) == 2  # matmul + gelu

        X = torch.randn(128, 768, dtype=torch.float32)
        W = torch.randn(768, 3072, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X, "W": W})

        expected = torch.nn.functional.gelu(X @ W)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-3, atol=1e-4
        ), f"matmul_gelu mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_batch_matmul(self, pipeline):
        """batch_matmul: C = Q @ K_T"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "08_batch_matmul.ak"))
        assert result.success

        Q = torch.randn(8, 128, 64, dtype=torch.float32)
        K_T = torch.randn(8, 64, 128, dtype=torch.float32)
        outputs = pipeline.execute(result, {"Q": Q, "K_T": K_T})

        expected = torch.bmm(Q, K_T)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"batch_matmul mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_transpose(self, pipeline):
        """transpose: Y = X.T"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "13_transpose.ak"))
        assert result.success

        X = torch.randn(512, 128, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = X.T
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"transpose mismatch: max_diff={_diff(outputs['output'], expected):.2e}"

    def test_e2e_reduce_sum(self, pipeline):
        """reduce_sum: Y = X.sum(axis=-1)"""
        result = pipeline.compile_file(str(OPERATORS_DIR / "11_reduce_sum.ak"))
        assert result.success

        X = torch.randn(128, 768, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X})

        expected = X.sum(dim=-1)
        assert torch.allclose(
            outputs["output"], expected, rtol=1e-4, atol=1e-5
        ), f"reduce_sum mismatch: max_diff={_diff(outputs['output'], expected):.2e}"


# ============================================================
# Test Class: Validation Error Cases
# ============================================================


class TestValidation:
    """Test SemanticIR validation catches errors."""

    def test_validate_valid_ir(self):
        """A well-formed IR should have no errors."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.return_node = "n0"
        errors = validate_semantic_ir(ir)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_validate_duplicate_node_id(self):
        """Duplicate node IDs should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.return_node = "n0"
        errors = validate_semantic_ir(ir)
        assert len(errors) == 1
        assert "Duplicate node ID" in errors[0]

    def test_validate_unknown_op(self):
        """Unknown ops should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(
            Node(
                id="n0",
                op="totally_fake_op",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="fake(X)"),
            )
        )
        ir.return_node = "n0"
        errors = validate_semantic_ir(ir)
        assert len(errors) == 1
        assert "unknown op" in errors[0]
        assert "totally_fake_op" in errors[0]

    def test_validate_invalid_param_ref(self):
        """ParamRef to a non-existent param should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": ParamRef(name="MISSING")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.return_node = "n0"
        errors = validate_semantic_ir(ir)
        assert len(errors) == 1
        assert "ParamRef" in errors[0]
        assert "MISSING" in errors[0]

    def test_validate_invalid_node_ref(self):
        """NodeRef to a non-existent node should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": NodeRef(id="nonexistent")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.return_node = "n0"
        errors = validate_semantic_ir(ir)
        assert len(errors) == 1
        assert "NodeRef" in errors[0]
        assert "nonexistent" in errors[0]

    def test_validate_forward_reference(self):
        """Forward references (use-before-def) should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        # n0 references n1 which is defined after n0
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": NodeRef(id="n1")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.add_node(
            Node(
                id="n1",
                op="relu",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.return_node = "n1"
        errors = validate_semantic_ir(ir)
        assert len(errors) == 1
        assert "forward reference" in errors[0].lower() or "defined later" in errors[0]

    def test_validate_invalid_return_node(self):
        """return_node pointing to non-existent node should be caught."""
        ir = SemanticIR(kernel_id="test")
        ir.add_param(Param(name="X", shape=[4, 8], dtype="f32"))
        ir.add_node(
            Node(
                id="n0",
                op="relu",
                inputs={"X": ParamRef(name="X")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="relu(X)"),
            )
        )
        ir.return_node = "nonexistent"
        errors = validate_semantic_ir(ir)
        assert len(errors) == 1
        assert "return_node" in errors[0]
        assert "nonexistent" in errors[0]

    def test_validate_multiple_errors(self):
        """Multiple errors should all be reported."""
        ir = SemanticIR(kernel_id="test")
        # No params at all
        ir.add_node(
            Node(
                id="n0",
                op="fake_op_1",
                inputs={"X": ParamRef(name="MISSING")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="fake(X)"),
            )
        )
        ir.add_node(
            Node(
                id="n0",  # duplicate
                op="fake_op_2",
                inputs={"Y": NodeRef(id="ghost")},
                output=TensorDesc(shape=[4, 8], dtype="f32"),
                semantics=Semantics(computation="fake(Y)"),
            )
        )
        ir.return_node = "wrong"
        errors = validate_semantic_ir(ir)
        # Should have at least: duplicate ID, 2 unknown ops, invalid param ref,
        # invalid node ref, invalid return_node
        assert len(errors) >= 4


# ============================================================
# Test Class: Strategy Preservation
# ============================================================


class TestStrategyPreservation:
    """Test that strategy blocks are correctly parsed and preserved."""

    def test_strategy_present(self, pipeline):
        """Files with strategy blocks should have StrategyIR populated."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        assert result.strategy_ir is not None
        assert result.strategy_ir.kernel_id != ""
        assert result.strategy_ir.target_hw != ""
        assert len(result.strategy_ir.decisions) > 0

    def test_strategy_absent(self, pipeline):
        """Files without strategy blocks should have StrategyIR = None."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
        assert result.success
        assert result.strategy_ir is None

    def test_strategy_decisions_have_rationale(self, pipeline):
        """Strategy decisions with @rationale should have text."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success
        assert result.strategy_ir is not None

        # At least one decision should have a rationale
        has_rationale = any(
            d.rationale is not None and d.rationale.text
            for d in result.strategy_ir.decisions
            if hasattr(d, "rationale")
        )
        assert has_rationale, "Expected at least one decision with @rationale"

    def test_strategy_target_hw(self, pipeline):
        """Strategy should have the correct target hardware."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "02_softmax.ak"))
        assert result.success
        assert result.strategy_ir is not None
        assert result.strategy_ir.target_hw == "nvidia_ampere"


# ============================================================
# Test Class: JSON Round-Trip
# ============================================================


class TestJSONRoundTrip:
    """Test compile -> serialize -> deserialize -> execute -> same result."""

    def test_json_round_trip_relu(self, pipeline):
        """SemanticIR JSON round-trip preserves semantics for relu."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success

        # Serialize and deserialize
        json_str = result.semantic_ir.to_json()
        ir_roundtrip = SemanticIR.from_json(json_str)

        # Validate the round-tripped IR
        errors = validate_semantic_ir(ir_roundtrip)
        assert errors == [], f"Round-trip validation errors: {errors}"

        # Build a new CompilationResult from the round-tripped IR
        result_rt = CompilationResult(
            semantic_ir=ir_roundtrip,
            success=True,
            kernel_name=ir_roundtrip.kernel_id,
        )

        # Execute both and compare
        X = torch.randn(128, 3072, dtype=torch.float32)
        out_orig = pipeline.execute(result, {"X": X})
        out_rt = pipeline.execute(result_rt, {"X": X})

        assert torch.allclose(
            out_orig["output"], out_rt["output"]
        ), "JSON round-trip changed execution result"

    def test_json_round_trip_matmul(self, pipeline):
        """SemanticIR JSON round-trip preserves semantics for matmul."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "01_matmul.ak"))
        assert result.success

        json_str = result.semantic_ir.to_json()
        ir_roundtrip = SemanticIR.from_json(json_str)
        errors = validate_semantic_ir(ir_roundtrip)
        assert errors == []

        result_rt = CompilationResult(
            semantic_ir=ir_roundtrip,
            success=True,
            kernel_name=ir_roundtrip.kernel_id,
        )

        A = torch.randn(1024, 1024, dtype=torch.float32)
        B = torch.randn(1024, 1024, dtype=torch.float32)
        out_orig = pipeline.execute(result, {"A": A, "B": B})
        out_rt = pipeline.execute(result_rt, {"A": A, "B": B})

        assert torch.allclose(
            out_orig["output"], out_rt["output"]
        ), "JSON round-trip changed matmul result"

    def test_json_round_trip_multi_node(self, pipeline):
        """SemanticIR JSON round-trip for multi-node kernel (matmul_gelu)."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "05_matmul_gelu.ak"))
        assert result.success

        json_str = result.semantic_ir.to_json()
        ir_roundtrip = SemanticIR.from_json(json_str)
        errors = validate_semantic_ir(ir_roundtrip)
        assert errors == []

        result_rt = CompilationResult(
            semantic_ir=ir_roundtrip,
            success=True,
            kernel_name=ir_roundtrip.kernel_id,
        )

        X = torch.randn(128, 768, dtype=torch.float32)
        W = torch.randn(768, 3072, dtype=torch.float32)
        out_orig = pipeline.execute(result, {"X": X, "W": W})
        out_rt = pipeline.execute(result_rt, {"X": X, "W": W})

        assert torch.allclose(
            out_orig["output"], out_rt["output"], rtol=1e-4, atol=1e-5
        ), "JSON round-trip changed matmul_gelu result"

    def test_json_preserves_structure(self, pipeline):
        """Serialized JSON should preserve all structural fields."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success

        json_str = result.semantic_ir.to_json()
        ir_rt = SemanticIR.from_json(json_str)

        # Check structural equality
        assert ir_rt.kernel_id == result.semantic_ir.kernel_id
        assert ir_rt.return_node == result.semantic_ir.return_node
        assert len(ir_rt.params) == len(result.semantic_ir.params)
        assert len(ir_rt.nodes) == len(result.semantic_ir.nodes)
        assert len(ir_rt.edges) == len(result.semantic_ir.edges)

        for p_orig, p_rt in zip(result.semantic_ir.params, ir_rt.params):
            assert p_orig.name == p_rt.name
            assert p_orig.dtype == p_rt.dtype

        for n_orig, n_rt in zip(result.semantic_ir.nodes, ir_rt.nodes):
            assert n_orig.id == n_rt.id
            assert n_orig.op == n_rt.op


# ============================================================
# Test Class: Pipeline API
# ============================================================


class TestPipelineAPI:
    """Test ArkePipeline API: compile_string, error handling, etc."""

    def test_compile_string_simple(self, pipeline):
        """compile_string should work for inline Arke source."""
        source = """
kernel simple_relu(
    X: Tensor<[64, 128], f16>
) -> Tensor<[64, 128], f16> {
    let Y = relu(X=X);
    return Y;
}
"""
        result = pipeline.compile_string(source)
        assert result.success
        assert result.kernel_name == "simple_relu"
        assert result.semantic_ir is not None
        assert len(result.semantic_ir.nodes) == 1
        assert result.semantic_ir.nodes[0].op == "relu"

    def test_compile_string_with_strategy(self, pipeline):
        """compile_string with strategy block."""
        source = """
kernel add_test(
    A: Tensor<[32, 32], f16>,
    B: Tensor<[32, 32], f16>
) -> Tensor<[32, 32], f16> {
    let C = add(A=A, B=B);
    return C;
}

strategy add_test_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("simple tile");
}
"""
        result = pipeline.compile_string(source)
        assert result.success
        assert result.strategy_ir is not None
        assert result.strategy_ir.target_hw == "nvidia_ampere"

    def test_compile_string_execute(self, pipeline):
        """compile_string then execute."""
        source = """
kernel inline_add(
    X: Tensor<[16, 16], f32>,
    Y: Tensor<[16, 16], f32>
) -> Tensor<[16, 16], f32> {
    let Z = add(A=X, B=Y);
    return Z;
}
"""
        result = pipeline.compile_string(source)
        assert result.success

        X = torch.randn(16, 16, dtype=torch.float32)
        Y = torch.randn(16, 16, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X, "Y": Y})

        expected = X + Y
        assert torch.allclose(outputs["output"], expected, rtol=1e-4, atol=1e-5)

    def test_compile_string_parse_error(self, pipeline):
        """Invalid syntax should return errors."""
        source = "this is not valid arke code"
        result = pipeline.compile_string(source)
        assert not result.success
        assert len(result.errors) > 0
        assert "Parse error" in result.errors[0]

    def test_execute_failed_compilation(self, pipeline):
        """Executing a failed compilation should raise ValueError."""
        result = CompilationResult(success=False, errors=["test error"])
        with pytest.raises(ValueError, match="compilation failed"):
            pipeline.execute(result, {})

    def test_execute_missing_input(self, pipeline):
        """Missing required input should raise KeyError."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success

        with pytest.raises(KeyError, match="Missing input parameter"):
            pipeline.execute(result, {})  # No inputs provided

    def test_intermediate_results_available(self, pipeline):
        """Multi-node kernels should expose intermediate results."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "05_matmul_gelu.ak"))
        assert result.success

        X = torch.randn(128, 768, dtype=torch.float32)
        W = torch.randn(768, 3072, dtype=torch.float32)
        outputs = pipeline.execute(result, {"X": X, "W": W})

        # Should have intermediate node outputs + "output"
        assert "output" in outputs
        # matmul_0 and gelu_1 are the node IDs
        assert len(outputs) >= 3  # at least 2 nodes + "output"

    def test_compilation_result_fields(self, pipeline):
        """CompilationResult should have all expected fields."""
        result = pipeline.compile_file(str(OPERATORS_DIR / "00_relu.ak"))
        assert result.success is True
        assert result.errors == []
        assert result.kernel_name == "relu_kernel"
        assert result.semantic_ir is not None
        assert result.strategy_ir is not None

    def test_compile_file_not_found(self, pipeline):
        """compile_file with non-existent path should return error."""
        result = pipeline.compile_file("/nonexistent/path.ak")
        assert not result.success
        assert len(result.errors) > 0