# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Semantic Interpreter (S6 Track 1, Task C1.4).

Executes operator semantics via PyTorch eager, driven by OpSchema.reference_impl.
Replaces the old 667-line numerical_check.py with 45 hand-written NumPy functions.

Usage:
    from arke.ir.ops.interpreter import SemanticInterpreter
    interp = SemanticInterpreter()
    result = interp.execute("relu", {"X": torch.randn(4, 8)})
    # => torch.Tensor of shape [4, 8]
"""

from __future__ import annotations

from typing import Any

import torch

from arke.ir.ops.registry import REGISTRY
from arke.ir.ops.shape_engine import SHAPE_ENGINE


class SemanticInterpreter:
    """PyTorch eager executor for all 45 operators.

    Driven entirely by OpSchema.reference_impl — no per-op dispatch.
    """

    def execute(
        self,
        op_name: str,
        inputs: dict[str, torch.Tensor],
        attrs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Execute an operator's reference implementation.

        Args:
            op_name: Operator name (e.g., "matmul", "relu")
            inputs: Mapping from input name to torch.Tensor
            attrs: Optional operator attributes

        Returns:
            Output tensor from PyTorch reference execution

        Raises:
            KeyError: Unknown operator
            ValueError: Missing reference implementation
            RuntimeError: Execution error
        """
        op = REGISTRY.get(op_name)
        attrs = attrs or {}

        # Merge default attrs from schema
        merged_attrs = {**op.attrs, **attrs}

        if op.reference_impl is None:
            raise ValueError(
                f"Operator {op_name!r} has no reference_impl. "
                f"Cannot execute via SemanticInterpreter."
            )

        ref = op.reference_impl

        # Apply dtype promotion if specified
        promoted_inputs = self._promote_dtypes(inputs, ref.dtype_map)

        try:
            result = ref.fn(promoted_inputs, merged_attrs)
        except Exception as e:
            raise RuntimeError(
                f"SemanticInterpreter failed on {op_name!r}: {e}"
            ) from e

        return result

    def validate(
        self,
        op_name: str,
        inputs: dict[str, torch.Tensor],
        expected: torch.Tensor,
        attrs: dict[str, Any] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> dict[str, Any]:
        """Execute and compare against expected output.

        Args:
            op_name: Operator name
            inputs: Input tensors
            expected: Expected output tensor
            attrs: Optional attributes
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Dict with keys: correct (bool), max_diff (float), mean_diff (float)
        """
        result = self.execute(op_name, inputs, attrs)

        if result.dtype != expected.dtype:
            result = result.to(expected.dtype)

        if result.shape != expected.shape:
            return {
                "correct": False,
                "error": f"Shape mismatch: got {list(result.shape)}, expected {list(expected.shape)}",
                "max_diff": float("inf"),
                "mean_diff": float("inf"),
            }

        # Integer tensors: exact match
        if not expected.is_floating_point():
            correct = torch.equal(result, expected)
            diff = (result.float() - expected.float()).abs()
            return {
                "correct": correct,
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
            }

        # Float tensors: allclose
        correct = torch.allclose(result, expected, rtol=rtol, atol=atol)
        diff = (result - expected).abs()
        return {
            "correct": correct,
            "max_diff": diff.max().item(),
            "mean_diff": diff.mean().item(),
        }

    def infer_shape(
        self,
        op_name: str,
        input_shapes: dict[str, list[int]],
        attrs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Infer output shape (delegates to ShapeInferenceEngine).

        Args:
            op_name: Operator name
            input_shapes: Mapping from input name to shape
            attrs: Optional attributes

        Returns:
            Output shape as list of ints
        """
        return SHAPE_ENGINE.infer(op_name, input_shapes, attrs)

    @staticmethod
    def _promote_dtypes(
        inputs: dict[str, torch.Tensor],
        dtype_map: dict[str, str],
    ) -> dict[str, torch.Tensor]:
        """Promote input dtypes for numerical stability.

        Args:
            inputs: Original input tensors
            dtype_map: Mapping from source dtype to target dtype
                       e.g., {"bf16": "f32", "f16": "f32"}

        Returns:
            New dict with promoted tensors (originals unchanged)
        """
        if not dtype_map:
            return inputs

        _DTYPE_MAP = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "f16": torch.float16, "float16": torch.float16,
            "f32": torch.float32, "float32": torch.float32,
            "f64": torch.float64, "float64": torch.float64,
        }

        promoted = {}
        for name, tensor in inputs.items():
            promoted_tensor = tensor
            for src_str, tgt_str in dtype_map.items():
                src_dtype = _DTYPE_MAP.get(src_str)
                tgt_dtype = _DTYPE_MAP.get(tgt_str)
                if src_dtype and tgt_dtype and tensor.dtype == src_dtype:
                    promoted_tensor = tensor.to(tgt_dtype)
                    break
            promoted[name] = promoted_tensor
        return promoted


# Module-level singleton
INTERPRETER = SemanticInterpreter()
