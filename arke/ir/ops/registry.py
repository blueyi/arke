# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Operator Registry (S6 Compiler Infrastructure).

OpRegistry provides typed access to the operator catalog with derived views
and validation utilities. It serves as the Single Source of Truth for all
operator metadata, replacing scattered if/elif dispatch tables.

Usage:
    from arke.ir.ops.registry import REGISTRY

    # Get operator definition
    op = REGISTRY.get("matmul")
    print(op.shape_rule.kind)  # "matmul_rule"

    # Query by category
    compute_ops = REGISTRY.ops_by_category("compute")

    # Validate coverage
    missing = REGISTRY.validate_coverage()
    if missing["shape_rule"]:
        print(f"Ops missing shape_rule: {missing['shape_rule']}")
"""

from __future__ import annotations

from typing import Iterator

from arke.ir.ops.catalog import OP_CATALOG
from arke.ir.ops.schema import OpSchema


class OpRegistry:
    """Typed registry wrapper around OP_CATALOG with derived views.

    Provides:
    - Typed access to operator definitions
    - Category-based queries
    - Coverage validation for S6 extended fields
    - Iteration and membership testing
    """

    def __init__(self, catalog: dict[str, OpSchema] | None = None) -> None:
        """Initialize registry from catalog.

        Args:
            catalog: Operator catalog dict. Defaults to OP_CATALOG.
        """
        self._ops: dict[str, OpSchema] = catalog if catalog is not None else OP_CATALOG

    def get(self, name: str) -> OpSchema:
        """Get operator definition by name.

        Args:
            name: Operator name (e.g., "matmul", "relu")

        Returns:
            OpSchema for the operator

        Raises:
            KeyError: If operator not found
        """
        try:
            return self._ops[name]
        except KeyError:
            available = ", ".join(sorted(self._ops.keys()))
            raise KeyError(
                f"Unknown operator: {name!r}. "
                f"Available operators: {available}"
            ) from None

    def __contains__(self, name: str) -> bool:
        """Check if operator is registered.

        Args:
            name: Operator name

        Returns:
            True if operator exists in registry
        """
        return name in self._ops

    def __iter__(self) -> Iterator[OpDefinition]:
        """Iterate over all operator definitions.

        Yields:
            OpDefinition instances
        """
        return iter(self._ops.values())

    def __len__(self) -> int:
        """Return number of registered operators.

        Returns:
            Count of operators in registry
        """
        return len(self._ops)

    def names(self) -> list[str]:
        """Get sorted list of all operator names.

        Returns:
            Sorted list of operator names
        """
        return sorted(self._ops.keys())

    def ops_by_category(self, category: str) -> list[OpDefinition]:
        """Get all operators in a category.

        Args:
            category: Category name (e.g., "compute", "elementwise", "reduce")

        Returns:
            List of OpDefinition instances in the category
        """
        return [op for op in self._ops.values() if op.category == category]

    def ops_with_template(self) -> list[OpDefinition]:
        """Get operators with template_hint defined.

        Returns:
            List of OpDefinition instances with template_hint
        """
        return [
            op for op in self._ops.values()
            if hasattr(op, "template_hint") and op.template_hint is not None
        ]

    def ops_with_reference(self) -> list[OpDefinition]:
        """Get operators with reference_impl defined.

        Returns:
            List of OpDefinition instances with reference_impl
        """
        return [
            op for op in self._ops.values()
            if hasattr(op, "reference_impl") and op.reference_impl is not None
        ]

    def ops_with_shape_rule(self) -> list[OpDefinition]:
        """Get operators with shape_rule defined.

        Returns:
            List of OpDefinition instances with shape_rule
        """
        return [
            op for op in self._ops.values()
            if hasattr(op, "shape_rule") and op.shape_rule is not None
        ]

    def validate_coverage(self) -> dict[str, list[str]]:
        """Validate S6 extended field coverage across all operators.

        Returns:
            Dict mapping field name to list of operator names missing that field.
            Fields checked: shape_rule, template_hint, reference_impl, input_gen

        Example:
            >>> missing = REGISTRY.validate_coverage()
            >>> if missing["shape_rule"]:
            ...     print(f"Missing shape_rule: {missing['shape_rule']}")
        """
        missing: dict[str, list[str]] = {
            "shape_rule": [],
            "template_hint": [],
            "reference_impl": [],
            "input_gen": [],
        }

        for op in self._ops.values():
            for field in missing:
                if not hasattr(op, field) or getattr(op, field) is None:
                    missing[field].append(op.name)

        return missing

    def categories(self) -> list[str]:
        """Get sorted list of all categories.

        Returns:
            Sorted list of unique category names
        """
        return sorted({op.category for op in self._ops.values()})

    def stats(self) -> dict[str, int]:
        """Get registry statistics.

        Returns:
            Dict with counts: total, by_category, with_template, with_reference, etc.
        """
        stats = {
            "total": len(self._ops),
            "with_template": len(self.ops_with_template()),
            "with_reference": len(self.ops_with_reference()),
            "with_shape_rule": len(self.ops_with_shape_rule()),
        }

        # Per-category counts
        for cat in self.categories():
            stats[f"category_{cat}"] = len(self.ops_by_category(cat))

        return stats


# Module-level singleton
REGISTRY = OpRegistry()
