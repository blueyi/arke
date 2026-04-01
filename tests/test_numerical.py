# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for V1 numerical validation."""

import numpy as np
import pytest

from arke.engine.numerical_check import (
    NumericalValidator,
    _to_numpy_dtype,
)
from arke.ir.builder import KernelBuilder
from arke.ir.semantic import (
    SemanticIR,
)

# ============================================================
# Helpers — build common IR structures
# ============================================================

def _build_matmul_ir(m=64, k=32, n=128, dtype="f32") -> SemanticIR:
    """Build a simple matmul IR."""
    b = KernelBuilder("test_matmul")
    b.param("A", [m, k], dtype)
    b.param("B", [k, n], dtype)
    mm = b.op("matmul", A="A", B="B")
    b.returns(mm, [m, n], dtype)
    return b.build()


def _build_relu_ir(shape=None, dtype="f32") -> SemanticIR:
    """Build a simple relu IR."""
    if shape is None:
        shape = [64, 128]
    b = KernelBuilder("test_relu")
    b.param("X", shape, dtype)
    r = b.op("relu", X="X")
    b.returns(r, shape, dtype)
    return b.build()


def _build_softmax_ir(m=32, n=64, dtype="f32") -> SemanticIR:
    """Build a simple softmax IR."""
    b = KernelBuilder("test_softmax")
    b.param("X", [m, n], dtype)
    s = b.op("softmax", X="X")
    b.returns(s, [m, n], dtype)
    return b.build()


def _build_fused_matmul_relu_ir(m=64, k=32, n=128, dtype="f32") -> SemanticIR:
    """Build a matmul+relu fused kernel IR."""
    b = KernelBuilder("test_fused_matmul_relu")
    b.param("A", [m, k], dtype)
    b.param("B", [k, n], dtype)
    mm = b.op("matmul", A="A", B="B")
    r = b.op("relu", X=mm)
    b.returns(r, [m, n], dtype)
    return b.build()


# ============================================================
# NumPy reference generation
# ============================================================

class TestGenerateReference:
    """Test that NumPy reference generation produces correct results."""

    def test_matmul_reference(self):
        ir = _build_matmul_ir(4, 3, 5)
        validator = NumericalValidator()

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        B = np.ones((3, 5), dtype=np.float32)
        inputs = {"A": A, "B": B}

        result = validator.generate_reference(ir, inputs)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(result, expected)

    def test_relu_reference(self):
        ir = _build_relu_ir([4, 4])
        validator = NumericalValidator()

        X = np.array([[-1, 2, -3, 4], [5, -6, 7, -8],
                       [-9, 10, -11, 12], [13, -14, 15, -16]], dtype=np.float32)
        inputs = {"X": X}

        result = validator.generate_reference(ir, inputs)
        expected = np.maximum(X, 0)
        np.testing.assert_allclose(result, expected)

    def test_softmax_reference(self):
        ir = _build_softmax_ir(2, 4)
        validator = NumericalValidator()

        X = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        inputs = {"X": X}

        result = validator.generate_reference(ir, inputs)

        # Manual softmax computation
        x_max = np.max(X, axis=-1, keepdims=True)
        e_x = np.exp(X - x_max)
        expected = e_x / np.sum(e_x, axis=-1, keepdims=True)

        np.testing.assert_allclose(result, expected, atol=1e-6)

        # Softmax rows should sum to 1
        row_sums = np.sum(result, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_fused_matmul_relu_reference(self):
        ir = _build_fused_matmul_relu_ir(4, 3, 5)
        validator = NumericalValidator()

        A = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12]], dtype=np.float32)
        B = np.ones((3, 5), dtype=np.float32)
        inputs = {"A": A, "B": B}

        result = validator.generate_reference(ir, inputs)
        expected = np.maximum(np.matmul(A, B), 0)
        np.testing.assert_allclose(result, expected)

    def test_add_reference(self):
        b = KernelBuilder("test_add")
        b.param("A", [4, 4], "f32")
        b.param("B", [4, 4], "f32")
        a = b.op("add", A="A", B="B")
        b.returns(a, [4, 4], "f32")
        ir = b.build()

        validator = NumericalValidator()
        A = np.ones((4, 4), dtype=np.float32)
        B = np.full((4, 4), 2.0, dtype=np.float32)
        result = validator.generate_reference(ir, {"A": A, "B": B})
        np.testing.assert_allclose(result, A + B)

    def test_transpose_reference(self):
        b = KernelBuilder("test_transpose")
        b.param("X", [3, 5], "f32")
        t = b.op("transpose", X="X")
        b.returns(t, [5, 3], "f32")
        ir = b.build()

        validator = NumericalValidator()
        X = np.arange(15, dtype=np.float32).reshape(3, 5)
        result = validator.generate_reference(ir, {"X": X})
        np.testing.assert_allclose(result, X.T)

    def test_reduce_sum_reference(self):
        b = KernelBuilder("test_reduce_sum")
        b.param("X", [4, 8], "f32")
        r = b.op("reduce_sum", X="X")
        b.returns(r, [4], "f32")
        ir = b.build()

        validator = NumericalValidator()
        X = np.ones((4, 8), dtype=np.float32)
        result = validator.generate_reference(ir, {"X": X})
        np.testing.assert_allclose(result, np.sum(X, axis=-1))

    def test_reduce_max_reference(self):
        b = KernelBuilder("test_reduce_max")
        b.param("X", [4, 8], "f32")
        r = b.op("reduce_max", X="X")
        b.returns(r, [4], "f32")
        ir = b.build()

        validator = NumericalValidator()
        X = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = validator.generate_reference(ir, {"X": X})
        np.testing.assert_allclose(result, np.max(X, axis=-1))


# ============================================================
# Validation (full pipeline)
# ============================================================

class TestValidation:
    """Test the full validation pipeline."""

    def test_matmul_passes(self):
        ir = _build_matmul_ir()
        validator = NumericalValidator()
        result = validator.validate(ir, trials=3)
        assert result.passed
        assert result.trials == 3
        assert result.max_absolute_error == 0.0
        assert len(result.errors) == 0

    def test_relu_passes(self):
        ir = _build_relu_ir()
        validator = NumericalValidator()
        result = validator.validate(ir, trials=3)
        assert result.passed
        assert result.trials == 3

    def test_softmax_passes(self):
        ir = _build_softmax_ir()
        validator = NumericalValidator()
        result = validator.validate(ir, trials=3)
        assert result.passed

    def test_fused_matmul_relu_passes(self):
        ir = _build_fused_matmul_relu_ir()
        validator = NumericalValidator()
        result = validator.validate(ir, trials=5)
        assert result.passed
        assert result.trials == 5

    def test_validation_result_has_tolerance(self):
        ir = _build_matmul_ir(dtype="f32")
        validator = NumericalValidator()
        result = validator.validate(ir)
        assert "atol" in result.tolerance
        assert "rtol" in result.tolerance
        assert result.tolerance["atol"] == 1e-5
        assert result.tolerance["rtol"] == 1e-5

    def test_validation_f16_tolerance(self):
        ir = _build_matmul_ir(dtype="f16")
        validator = NumericalValidator()
        result = validator.validate(ir)
        assert result.tolerance["atol"] == 1e-2

    def test_single_trial(self):
        ir = _build_relu_ir()
        validator = NumericalValidator()
        result = validator.validate(ir, trials=1)
        assert result.passed
        assert result.trials == 1


# ============================================================
# Random input generation
# ============================================================

class TestRandomInputGeneration:
    """Test random input generation respects shapes and dtypes."""

    def test_shapes_match(self):
        ir = _build_matmul_ir(m=32, k=16, n=64)
        validator = NumericalValidator()
        inputs = validator.generate_random_inputs(ir, seed=42)

        assert "A" in inputs
        assert "B" in inputs
        assert inputs["A"].shape == (32, 16)
        assert inputs["B"].shape == (16, 64)

    def test_dtypes_match_f32(self):
        ir = _build_matmul_ir(dtype="f32")
        validator = NumericalValidator()
        inputs = validator.generate_random_inputs(ir)

        assert inputs["A"].dtype == np.float32
        assert inputs["B"].dtype == np.float32

    def test_dtypes_match_f16(self):
        ir = _build_matmul_ir(dtype="f16")
        validator = NumericalValidator()
        inputs = validator.generate_random_inputs(ir)

        assert inputs["A"].dtype == np.float16
        assert inputs["B"].dtype == np.float16

    def test_dtypes_match_f64(self):
        ir = _build_matmul_ir(dtype="f64")
        validator = NumericalValidator()
        inputs = validator.generate_random_inputs(ir)

        assert inputs["A"].dtype == np.float64

    def test_reproducible_with_seed(self):
        ir = _build_relu_ir()
        validator = NumericalValidator()
        inputs1 = validator.generate_random_inputs(ir, seed=123)
        inputs2 = validator.generate_random_inputs(ir, seed=123)
        np.testing.assert_array_equal(inputs1["X"], inputs2["X"])

    def test_different_seeds_different_data(self):
        ir = _build_relu_ir()
        validator = NumericalValidator()
        inputs1 = validator.generate_random_inputs(ir, seed=1)
        inputs2 = validator.generate_random_inputs(ir, seed=2)
        assert not np.array_equal(inputs1["X"], inputs2["X"])

    def test_integer_dtypes(self):
        b = KernelBuilder("test_int")
        b.param("X", [64, 64], "i32")
        r = b.op("relu", X="X")
        b.returns(r, [64, 64], "i32")
        ir = b.build()

        validator = NumericalValidator()
        inputs = validator.generate_random_inputs(ir)
        assert inputs["X"].dtype == np.int32

    def test_bf16_upcast(self):
        """bf16 should be upcast to float32 in NumPy."""
        b = KernelBuilder("test_bf16")
        b.param("X", [32, 32], "bf16")
        r = b.op("relu", X="X")
        b.returns(r, [32, 32], "bf16")
        ir = b.build()

        validator = NumericalValidator()
        inputs = validator.generate_random_inputs(ir)
        # bf16 maps to float32 in NumPy (no native bf16)
        assert inputs["X"].dtype == np.float32


# ============================================================
# Dtype mapping
# ============================================================

class TestDtypeMapping:
    """Test Arke dtype to NumPy dtype conversion."""

    def test_f16(self):
        assert _to_numpy_dtype("f16") == np.float16

    def test_f32(self):
        assert _to_numpy_dtype("f32") == np.float32

    def test_f64(self):
        assert _to_numpy_dtype("f64") == np.float64

    def test_bf16_upcast(self):
        assert _to_numpy_dtype("bf16") == np.float32

    def test_i8(self):
        assert _to_numpy_dtype("i8") == np.int8

    def test_i32(self):
        assert _to_numpy_dtype("i32") == np.int32

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _to_numpy_dtype("float128")
