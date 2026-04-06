# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Shape Inference Engine (S6 Track 1, Task C1.3).

Declarative shape inference driven by OpSchema.shape_rule.
Replaces the old 401-line if/elif chain in shape_inference.py.

Usage:
    from arke.ir.ops.shape_engine import ShapeInferenceEngine
    engine = ShapeInferenceEngine()
    output_shape = engine.infer("matmul", {"A": [1024, 512], "B": [512, 2048]})
    # => [1024, 2048]
"""

from __future__ import annotations

from arke.ir.ops.registry import REGISTRY


class ShapeInferenceEngine:
    """Declarative shape inference driven by OpSchema.shape_rule.

    Each shape rule kind maps to a handler method.
    Adding a new kind requires only one new method + updating ShapeRule docs.
    """

    def infer(
        self,
        op_name: str,
        input_shapes: dict[str, list[int]],
        attrs: dict | None = None,
    ) -> list[int]:
        """Infer output shape for an operator.

        Args:
            op_name: Operator name (e.g., "matmul", "relu")
            input_shapes: Mapping from input name to shape list
            attrs: Optional operator attributes

        Returns:
            Output shape as list of ints

        Raises:
            KeyError: Unknown operator
            ValueError: Missing or invalid shape rule
        """
        attrs = attrs or {}
        op = REGISTRY.get(op_name)

        if op.shape_rule is None:
            raise ValueError(f"Operator {op_name!r} has no shape_rule defined")

        kind = op.shape_rule.kind
        handler = getattr(self, f"_infer_{kind}", None)
        if handler is None:
            raise ValueError(f"Unknown shape rule kind: {kind!r} for op {op_name!r}")

        return handler(op.shape_rule, input_shapes, attrs)

    # ── Rule Handlers ─────────────────────────────────────────

    def _infer_same_as_input(self, rule, shapes, attrs) -> list[int]:
        key = rule.input_key
        if key not in shapes:
            raise ValueError(f"Input {key!r} not found in shapes: {list(shapes.keys())}")
        return list(shapes[key])

    def _infer_matmul_rule(self, rule, shapes, attrs) -> list[int]:
        a = shapes.get("A", shapes.get("X", []))
        b = shapes.get("B", shapes.get("W", []))
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"matmul requires 2D+ inputs, got A={a}, B={b}")
        return a[:-1] + [b[-1]]

    def _infer_batch_matmul_rule(self, rule, shapes, attrs) -> list[int]:
        a = shapes.get("A", shapes.get("X", []))
        b = shapes.get("B", shapes.get("W", []))
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"batch_matmul requires 2D+ inputs, got A={a}, B={b}")
        return a[:-1] + [b[-1]]

    def _infer_reduce_rule(self, rule, shapes, attrs) -> list[int]:
        key = rule.input_key
        shape = list(shapes.get(key, shapes.get("X", [])))
        axes = rule.axes or [-1]
        # Normalize negative axes
        ndim = len(shape)
        resolved = sorted({a % ndim for a in axes}, reverse=True)
        for ax in resolved:
            shape.pop(ax)
        return shape

    def _infer_topk_rule(self, rule, shapes, attrs) -> list[int]:
        shape = list(shapes.get("X", []))
        k = attrs.get(rule.k_attr, attrs.get("k", 1))
        shape[-1] = k
        return shape

    def _infer_concat_rule(self, rule, shapes, attrs) -> list[int]:
        a = shapes.get("A", [])
        b = shapes.get("B", [])
        axis = attrs.get(rule.axis_attr, -1)
        if axis < 0:
            axis = len(a) + axis
        result = list(a)
        result[axis] = a[axis] + b[axis]
        return result

    def _infer_split_rule(self, rule, shapes, attrs) -> list[int]:
        shape = list(shapes.get("X", []))
        axis = attrs.get(rule.axis_attr, -1)
        if axis < 0:
            axis = len(shape) + axis
        shape[axis] = shape[axis] // 2
        return shape

    def _infer_gather_rule(self, rule, shapes, attrs) -> list[int]:
        key = rule.input_key
        return list(shapes.get(key, shapes.get("idx", [])))

    def _infer_embedding_rule(self, rule, shapes, attrs) -> list[int]:
        indices = shapes.get("indices", [])
        weight = shapes.get("weight", [])
        if len(weight) >= 2:
            return list(indices) + [weight[-1]]
        return list(indices)

    def _infer_permute_rule(self, rule, shapes, attrs) -> list[int]:
        x = shapes.get("X", [])
        dims = attrs.get(rule.dims_attr, attrs.get("dims", list(range(len(x)))))
        return [x[d] for d in dims]

    def _infer_gated_halve_rule(self, rule, shapes, attrs) -> list[int]:
        key = rule.input_key
        shape = list(shapes.get(key, shapes.get("X", [])))
        shape[-1] = shape[-1] // 2
        return shape

    def _infer_attention_rule(self, rule, shapes, attrs) -> list[int]:
        return list(shapes.get("Q", []))

    def _infer_custom(self, rule, shapes, attrs) -> list[int]:
        if rule.fn is None:
            raise ValueError("Custom shape rule requires fn")
        return rule.fn(shapes, attrs)


# Module-level singleton
SHAPE_ENGINE = ShapeInferenceEngine()
