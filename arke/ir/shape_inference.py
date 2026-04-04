# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Shape Inference (v0.2.0).

Infers output shapes and dtypes for all P0 operators.
No broadcasting in v0.2.0 — elementwise ops require exact shape match.
"""

from __future__ import annotations

# ============================================================
# Operator categories for dispatch
# ============================================================

_ELEMENTWISE_UNARY = {"relu", "gelu", "silu"}
_ELEMENTWISE_BINARY = {"add", "mul"}
_SAME_SHAPE_OPS = {"softmax"}
_NORM_OPS = {"layernorm", "rmsnorm"}
_REDUCE_OPS = {"reduce_sum", "reduce_max"}


# ============================================================
# Public API
# ============================================================


def infer_output_shape(op_name: str, input_shapes: dict[str, list[int]]) -> list[int]:
    """Infer output shape from op name and input shapes.

    Args:
        op_name: Operator name (must be in the P0 catalog).
        input_shapes: Mapping of input-name → shape list.
            Accepts both canonical names (W, B) and semantic aliases
            (weight, bias, gamma, beta, scale).

    Returns:
        The inferred output shape.

    Raises:
        ValueError: On shape mismatch or unknown operator.
    """
    # Normalize aliases first so all downstream code uses canonical names
    input_shapes = _normalize_inputs(input_shapes)
    errors = validate_shapes(op_name, input_shapes)  # validate also normalizes (idempotent)
    if errors:
        raise ValueError("; ".join(errors))

    if op_name == "matmul":
        a = input_shapes["A"]
        b = input_shapes["B"]
        return [a[0], b[1]]

    if op_name == "batch_matmul":
        a = input_shapes["A"]
        b = input_shapes["B"]
        return [a[0], a[1], b[2]]

    if op_name in _ELEMENTWISE_UNARY:
        # Single input, passthrough shape
        x = _first_value(input_shapes)
        return list(x)

    if op_name in _ELEMENTWISE_BINARY:
        a = input_shapes["A"]
        return list(a)

    if op_name in _SAME_SHAPE_OPS:
        x = input_shapes["X"]
        return list(x)

    if op_name in _NORM_OPS:
        x = input_shapes["X"]
        return list(x)

    if op_name in _REDUCE_OPS:
        x = input_shapes["X"]
        # Remove last dimension
        return list(x[:-1])

    if op_name == "transpose":
        x = input_shapes["X"]
        return [x[1], x[0]]

    raise ValueError(f"Unknown operator: {op_name}")


def infer_output_dtype(op_name: str, input_dtypes: dict[str, str]) -> str:
    """Infer output dtype — v0.2.0 simplification: same as first input dtype.

    Args:
        op_name: Operator name.
        input_dtypes: Mapping of input-name → dtype string.

    Returns:
        The output dtype string.

    Raises:
        ValueError: If no inputs provided.
    """
    if not input_dtypes:
        raise ValueError("No input dtypes provided")
    return next(iter(input_dtypes.values()))


# Canonical input-name aliases: long/semantic names → internal short names
_INPUT_ALIASES: dict[str, str] = {
    "weight": "W",
    "bias": "B",
    "scale": "W",    # some ops use 'scale' instead of 'weight'
    "gamma": "W",
    "beta": "B",
}


def _normalize_inputs(
    input_shapes: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Normalize semantic input names to internal canonical names."""
    result: dict[str, list[int]] = {}
    for k, v in input_shapes.items():
        result[_INPUT_ALIASES.get(k, k)] = v
    return result


def validate_shapes(op_name: str, input_shapes: dict[str, list[int]]) -> list[str]:
    """Return list of shape errors (empty list = valid).

    Args:
        op_name: Operator name.
        input_shapes: Mapping of input-name → shape list.
            Accepts both canonical names (W, B) and semantic aliases
            (weight, bias, gamma, beta, scale).

    Returns:
        List of human-readable error strings. Empty if valid.
    """
    # Normalize aliases before validation
    input_shapes = _normalize_inputs(input_shapes)
    errors: list[str] = []

    if op_name == "matmul":
        if "A" not in input_shapes or "B" not in input_shapes:
            errors.append("matmul requires inputs 'A' and 'B'")
            return errors
        a, b = input_shapes["A"], input_shapes["B"]
        if len(a) != 2:
            errors.append(f"matmul: A must be 2D, got {len(a)}D shape {a}")
        if len(b) != 2:
            errors.append(f"matmul: B must be 2D, got {len(b)}D shape {b}")
        if len(a) == 2 and len(b) == 2 and a[1] != b[0]:
            errors.append(
                f"matmul: inner dimensions mismatch: A[{a[0]},{a[1]}] @ B[{b[0]},{b[1]}] "
                f"— A's K={a[1]} != B's K={b[0]}"
            )

    elif op_name == "batch_matmul":
        if "A" not in input_shapes or "B" not in input_shapes:
            errors.append("batch_matmul requires inputs 'A' and 'B'")
            return errors
        a, b = input_shapes["A"], input_shapes["B"]
        if len(a) != 3:
            errors.append(f"batch_matmul: A must be 3D, got {len(a)}D shape {a}")
        if len(b) != 3:
            errors.append(f"batch_matmul: B must be 3D, got {len(b)}D shape {b}")
        if len(a) == 3 and len(b) == 3:
            if a[0] != b[0]:
                errors.append(
                    f"batch_matmul: batch dimensions mismatch: A batch={a[0]} != B batch={b[0]}"
                )
            if a[2] != b[1]:
                errors.append(
                    f"batch_matmul: inner dimensions mismatch: A's K={a[2]} != B's K={b[1]}"
                )

    elif op_name in _ELEMENTWISE_UNARY:
        if not input_shapes:
            errors.append(f"{op_name}: requires at least one input")

    elif op_name in _ELEMENTWISE_BINARY:
        if "A" not in input_shapes or "B" not in input_shapes:
            errors.append(f"{op_name} requires inputs 'A' and 'B'")
            return errors
        a, b = input_shapes["A"], input_shapes["B"]
        if a != b:
            errors.append(
                f"{op_name}: shapes must match (no broadcasting in v0.2.0): "
                f"A={a} != B={b}"
            )

    elif op_name in _SAME_SHAPE_OPS:
        if "X" not in input_shapes:
            errors.append(f"{op_name} requires input 'X'")
            return errors
        x = input_shapes["X"]
        if len(x) < 2:
            errors.append(f"{op_name}: input must be at least 2D, got {len(x)}D")

    elif op_name in _NORM_OPS:
        if "X" not in input_shapes:
            errors.append(f"{op_name} requires input 'X'")
            return errors
        if "W" not in input_shapes:
            errors.append(f"{op_name} requires input 'W'")
            return errors
        x = input_shapes["X"]
        w = input_shapes["W"]
        if len(x) < 2:
            errors.append(f"{op_name}: X must be at least 2D, got {len(x)}D")
        if len(w) != 1:
            errors.append(f"{op_name}: W must be 1D, got {len(w)}D")
        if len(x) >= 2 and len(w) == 1 and x[-1] != w[0]:
            errors.append(
                f"{op_name}: W dim ({w[0]}) must match last dim of X ({x[-1]})"
            )

    elif op_name in _REDUCE_OPS:
        if "X" not in input_shapes:
            errors.append(f"{op_name} requires input 'X'")
            return errors
        x = input_shapes["X"]
        if len(x) < 1:
            errors.append(f"{op_name}: input must be at least 1D")

    elif op_name == "transpose":
        if "X" not in input_shapes:
            errors.append("transpose requires input 'X'")
            return errors
        x = input_shapes["X"]
        if len(x) != 2:
            errors.append(f"transpose: input must be 2D, got {len(x)}D shape {x}")

    else:
        errors.append(f"Unknown operator: {op_name}")

    return errors


# ============================================================
# Helpers
# ============================================================


def _first_value(d: dict[str, list[int]]) -> list[int]:
    """Get the first value from a dict."""
    return next(iter(d.values()))
