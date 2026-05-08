# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpRegistry (S6 Track 1, Task C1.1).

Validates:
- OpRegistry basic access (get, contains, iter)
- Category queries
- Coverage validation
- Smoke test with relu (OT0) and matmul (OT2)
"""

import pytest

from arke.ir.ops.registry import REGISTRY


def test_registry_basic_access():
    """Test basic registry access methods."""
    # Get by name
    relu = REGISTRY.get("relu")
    assert relu.name == "relu"
    assert relu.category == "elementwise"

    matmul = REGISTRY.get("matmul")
    assert matmul.name == "matmul"
    assert matmul.category == "compute"

    # Unknown op raises KeyError
    with pytest.raises(KeyError, match="Unknown operator"):
        REGISTRY.get("nonexistent_op")


def test_registry_membership():
    """Test membership testing."""
    assert "relu" in REGISTRY
    assert "matmul" in REGISTRY
    assert "gelu" in REGISTRY
    assert "nonexistent_op" not in REGISTRY


def test_registry_iteration():
    """Test iteration over all ops."""
    ops = list(REGISTRY)
    assert len(ops) == 45  # All 45 ops from catalog.py

    names = [op.name for op in ops]
    assert "relu" in names
    assert "matmul" in names
    assert "flash_attention" in names


def test_registry_names():
    """Test names() returns sorted list."""
    names = REGISTRY.names()
    assert len(names) == 45
    assert names == sorted(names)  # Must be sorted
    assert "relu" in names
    assert "matmul" in names


def test_registry_len():
    """Test __len__ returns correct count."""
    assert len(REGISTRY) == 45


def test_registry_categories():
    """Test category queries."""
    categories = REGISTRY.categories()
    assert "compute" in categories
    assert "elementwise" in categories
    assert "reduce" in categories
    assert "attention" in categories

    # Query by category
    compute_ops = REGISTRY.ops_by_category("compute")
    compute_names = [op.name for op in compute_ops]
    assert "matmul" in compute_names
    assert "batch_matmul" in compute_names
    assert "grouped_matmul" in compute_names

    elementwise_ops = REGISTRY.ops_by_category("elementwise")
    elementwise_names = [op.name for op in elementwise_ops]
    assert "relu" in elementwise_names
    assert "gelu" in elementwise_names
    assert "silu" in elementwise_names


def test_registry_validate_coverage():
    """Test coverage validation — all 45 ops fully annotated after C1.2."""
    missing = REGISTRY.validate_coverage()

    # All fields should be fully covered after C1.2
    assert len(missing["shape_rule"]) == 0, f"Missing shape_rule: {missing['shape_rule']}"
    assert len(missing["template_hint"]) == 0, f"Missing template_hint: {missing['template_hint']}"
    assert len(missing["reference_impl"]) == 0, f"Missing reference_impl: {missing['reference_impl']}"
    assert len(missing["input_gen"]) == 0, f"Missing input_gen: {missing['input_gen']}"


def test_registry_stats():
    """Test stats() returns correct counts."""
    stats = REGISTRY.stats()

    assert stats["total"] == 45
    assert stats["with_template"] == 45   # All covered after C1.2
    assert stats["with_reference"] == 45  # All covered after C1.2
    assert stats["with_shape_rule"] == 45  # All covered after C1.2

    # Category counts
    assert stats["category_compute"] >= 3
    assert stats["category_elementwise"] >= 10
    assert stats["category_reduce"] >= 8


def test_registry_ops_with_filters():
    """Test filtered queries — all 45 ops covered after C1.2."""
    assert len(REGISTRY.ops_with_template()) == 45
    assert len(REGISTRY.ops_with_reference()) == 45
    assert len(REGISTRY.ops_with_shape_rule()) == 45


def test_relu_op_definition():
    """Smoke test: relu (OT0) operator definition."""
    relu = REGISTRY.get("relu")

    # Original fields
    assert relu.name == "relu"
    assert relu.category == "elementwise"
    assert "X" in relu.inputs
    assert relu.output == "Tensor[...]"
    assert relu.computation == "Y = max(X, 0)"
    assert "elementwise" in relu.properties
    assert relu.can_fuse_as == "epilogue"
    assert relu.numpy_ref == "np.maximum(X, 0)"

    # Extended fields — all populated after C1.2
    assert relu.shape_rule is not None
    assert relu.shape_rule.kind == "same_as_input"
    assert relu.template_hint is not None
    assert relu.template_hint.template_name == "elementwise"
    assert relu.reference_impl is not None
    assert relu.input_gen is not None


def test_matmul_op_definition():
    """Smoke test: matmul (OT2) operator definition."""
    matmul = REGISTRY.get("matmul")

    # Original fields
    assert matmul.name == "matmul"
    assert matmul.category == "compute"
    assert "A" in matmul.inputs
    assert "B" in matmul.inputs
    assert matmul.output == "Tensor[M,N]"
    assert "i" in matmul.index_vars
    assert "j" in matmul.index_vars
    assert "k" in matmul.index_vars
    assert "k" in matmul.reduction_axes
    assert "associative" in matmul.properties
    assert matmul.can_fuse_as == "prologue"
    assert matmul.numpy_ref == "np.matmul(A, B)"

    # Extended fields — all populated after C1.2
    assert matmul.shape_rule is not None
    assert matmul.shape_rule.kind == "matmul_rule"
    assert matmul.template_hint is not None
    assert matmul.template_hint.template_name == "matmul"
    assert matmul.reference_impl is not None
    assert matmul.input_gen is not None
