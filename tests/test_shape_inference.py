# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for shape inference module."""

import pytest

from arke.ir.shape_inference import (
    infer_output_dtype,
    infer_output_shape,
    validate_shapes,
)

# ============================================================
# infer_output_shape — happy paths for all 10 operators
# ============================================================

class TestInferOutputShape:
    """Test output shape inference for every P0 operator."""

    def test_matmul(self):
        result = infer_output_shape("matmul", {"A": [1024, 512], "B": [512, 2048]})
        assert result == [1024, 2048]

    def test_matmul_square(self):
        result = infer_output_shape("matmul", {"A": [64, 64], "B": [64, 64]})
        assert result == [64, 64]

    def test_batch_matmul(self):
        result = infer_output_shape("batch_matmul", {"A": [8, 64, 128], "B": [8, 128, 256]})
        assert result == [8, 64, 256]

    def test_relu(self):
        result = infer_output_shape("relu", {"X": [1024, 2048]})
        assert result == [1024, 2048]

    def test_relu_1d(self):
        result = infer_output_shape("relu", {"X": [256]})
        assert result == [256]

    def test_relu_3d(self):
        result = infer_output_shape("relu", {"X": [8, 64, 128]})
        assert result == [8, 64, 128]

    def test_gelu(self):
        result = infer_output_shape("gelu", {"X": [512, 512]})
        assert result == [512, 512]

    def test_gelu_1d(self):
        result = infer_output_shape("gelu", {"X": [1024]})
        assert result == [1024]

    def test_add(self):
        result = infer_output_shape("add", {"A": [1024, 2048], "B": [1024, 2048]})
        assert result == [1024, 2048]

    def test_mul(self):
        result = infer_output_shape("mul", {"A": [64, 64], "B": [64, 64]})
        assert result == [64, 64]

    def test_softmax(self):
        result = infer_output_shape("softmax", {"X": [128, 64]})
        assert result == [128, 64]

    def test_reduce_sum(self):
        result = infer_output_shape("reduce_sum", {"X": [128, 64]})
        assert result == [128]

    def test_reduce_max(self):
        result = infer_output_shape("reduce_max", {"X": [256, 512]})
        assert result == [256]

    def test_reduce_sum_3d(self):
        """Reduce last dim of 3D tensor."""
        result = infer_output_shape("reduce_sum", {"X": [8, 64, 128]})
        assert result == [8, 64]

    def test_transpose(self):
        result = infer_output_shape("transpose", {"X": [128, 256]})
        assert result == [256, 128]

    def test_transpose_square(self):
        result = infer_output_shape("transpose", {"X": [64, 64]})
        assert result == [64, 64]


# ============================================================
# Shape mismatch detection
# ============================================================

class TestValidateShapes:
    """Test shape error detection for all operators."""

    def test_matmul_inner_dim_mismatch(self):
        errors = validate_shapes("matmul", {"A": [1024, 512], "B": [256, 2048]})
        assert len(errors) == 1
        assert "inner dimensions mismatch" in errors[0]

    def test_matmul_wrong_rank(self):
        errors = validate_shapes("matmul", {"A": [1024, 512, 3], "B": [512, 2048]})
        assert len(errors) >= 1
        assert "2D" in errors[0]

    def test_matmul_missing_input(self):
        errors = validate_shapes("matmul", {"A": [1024, 512]})
        assert len(errors) >= 1
        assert "requires" in errors[0]

    def test_matmul_valid(self):
        errors = validate_shapes("matmul", {"A": [1024, 512], "B": [512, 2048]})
        assert errors == []

    def test_batch_matmul_batch_mismatch(self):
        errors = validate_shapes("batch_matmul", {"A": [8, 64, 128], "B": [4, 128, 256]})
        assert len(errors) >= 1
        assert "batch" in errors[0]

    def test_batch_matmul_inner_mismatch(self):
        errors = validate_shapes("batch_matmul", {"A": [8, 64, 128], "B": [8, 64, 256]})
        assert len(errors) >= 1
        assert "inner" in errors[0]

    def test_batch_matmul_wrong_rank(self):
        errors = validate_shapes("batch_matmul", {"A": [64, 128], "B": [8, 128, 256]})
        assert len(errors) >= 1
        assert "3D" in errors[0]

    def test_add_shape_mismatch(self):
        errors = validate_shapes("add", {"A": [1024, 512], "B": [1024, 256]})
        assert len(errors) == 1
        assert "must match" in errors[0]

    def test_mul_shape_mismatch(self):
        errors = validate_shapes("mul", {"A": [64], "B": [128]})
        assert len(errors) == 1
        assert "must match" in errors[0]

    def test_transpose_wrong_rank(self):
        errors = validate_shapes("transpose", {"X": [8, 64, 128]})
        assert len(errors) == 1
        assert "2D" in errors[0]

    def test_softmax_too_few_dims(self):
        errors = validate_shapes("softmax", {"X": [128]})
        assert len(errors) >= 1
        assert "2D" in errors[0]

    def test_unknown_op(self):
        errors = validate_shapes("nonexistent_op", {"X": [128]})
        assert len(errors) == 1
        assert "Unknown" in errors[0]

    def test_add_valid(self):
        errors = validate_shapes("add", {"A": [64, 64], "B": [64, 64]})
        assert errors == []

    def test_elementwise_unary_valid(self):
        errors = validate_shapes("relu", {"X": [1024]})
        assert errors == []

    def test_reduce_valid(self):
        errors = validate_shapes("reduce_sum", {"X": [128, 64]})
        assert errors == []

    # Test that infer_output_shape raises on mismatch
    def test_infer_raises_on_mismatch(self):
        with pytest.raises(ValueError, match="inner dimensions"):
            infer_output_shape("matmul", {"A": [1024, 512], "B": [256, 2048]})

    def test_infer_raises_on_unknown_op(self):
        with pytest.raises(ValueError, match="Unknown"):
            infer_output_shape("fake_op", {"X": [128]})


# ============================================================
# Dtype inference
# ============================================================

class TestInferOutputDtype:
    """Test dtype inference."""

    def test_single_input_f16(self):
        result = infer_output_dtype("relu", {"X": "f16"})
        assert result == "f16"

    def test_single_input_f32(self):
        result = infer_output_dtype("relu", {"X": "f32"})
        assert result == "f32"

    def test_two_inputs_first_wins(self):
        result = infer_output_dtype("matmul", {"A": "f16", "B": "f32"})
        assert result == "f16"  # first input dtype

    def test_f64(self):
        result = infer_output_dtype("transpose", {"X": "f64"})
        assert result == "f64"

    def test_integer_dtype(self):
        result = infer_output_dtype("relu", {"X": "i32"})
        assert result == "i32"

    def test_no_inputs_raises(self):
        with pytest.raises(ValueError, match="No input"):
            infer_output_dtype("relu", {})
