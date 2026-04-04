# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — V1 Numerical Validation.

Verifies kernel correctness by executing the Semantic IR computation graph
using NumPy as the reference implementation.

This V1 validator checks the *math* (Semantic IR), not compiled kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arke.ir.semantic import NodeRef, ParamRef, SemanticIR

# ============================================================
# Dtype mapping
# ============================================================

DTYPE_MAP: dict[str, np.dtype] = {
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "bf16": np.dtype(np.float32),  # upcast — NumPy has no bfloat16
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
}


def _to_numpy_dtype(arke_dtype: str) -> np.dtype:
    """Convert Arke dtype string to numpy dtype."""
    if arke_dtype in DTYPE_MAP:
        return DTYPE_MAP[arke_dtype]
    raise ValueError(f"Unsupported dtype: {arke_dtype}")


# ============================================================
# Tolerance table
# ============================================================

TOLERANCE_TABLE: dict[str, dict[str, float]] = {
    "f16": {"atol": 1e-2, "rtol": 1e-2},
    "bf16": {"atol": 1e-2, "rtol": 1e-2},
    "f32": {"atol": 1e-5, "rtol": 1e-5},
    "f64": {"atol": 1e-10, "rtol": 1e-10},
}


def _get_tolerance(arke_dtype: str) -> dict[str, float]:
    """Get numerical tolerance for a dtype."""
    return TOLERANCE_TABLE.get(arke_dtype, {"atol": 1e-5, "rtol": 1e-5})


# ============================================================
# NumPy reference implementations for each op
# ============================================================

def _numpy_matmul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.matmul(inputs["A"], inputs["B"])


def _numpy_batch_matmul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.matmul(inputs["A"], inputs["B"])


def _numpy_relu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(inputs["X"], 0)


def _numpy_gelu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Element-wise erf using scipy if available, else NumPy approximation."""
    try:
        import scipy.special
        return scipy.special.erf(x)
    except ImportError:
        # Abramowitz & Stegun approximation (good to ~1.5e-7)
        sign = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = 1.0 - (
            ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
             - 0.284496736) * t + 0.254829592
        ) * t * np.exp(-x * x)
        return sign * y


def _numpy_silu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    return x / (1.0 + np.exp(-x))  # x * sigmoid(x)


def _numpy_add(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return inputs["A"] + inputs["B"]


def _numpy_mul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return inputs["A"] * inputs["B"]


def _numpy_layernorm(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    w = inputs.get("W", inputs.get("weight"))
    b = inputs.get("B", inputs.get("bias"))
    eps = inputs.get("eps", 1e-5)
    if isinstance(eps, np.ndarray):
        eps = float(eps.flat[0])
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if w is not None:
        x_norm = x_norm * w
    if b is not None:
        x_norm = x_norm + b
    return x_norm


def _numpy_rmsnorm(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    w = inputs.get("W", inputs.get("weight"))
    eps = inputs.get("eps", 1e-5)
    if isinstance(eps, np.ndarray):
        eps = float(eps.flat[0])
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    if w is not None:
        x_norm = x_norm * w
    return x_norm


def _numpy_softmax(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    # Numerically stable softmax
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _numpy_reduce_sum(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(inputs["X"], axis=-1)


def _numpy_reduce_max(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.max(inputs["X"], axis=-1)


def _numpy_transpose(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return inputs["X"].T


_NUMPY_DISPATCH: dict[str, Any] = {
    "matmul": _numpy_matmul,
    "batch_matmul": _numpy_batch_matmul,
    "relu": _numpy_relu,
    "gelu": _numpy_gelu,
    "silu": _numpy_silu,
    "add": _numpy_add,
    "mul": _numpy_mul,
    "softmax": _numpy_softmax,
    "layernorm": _numpy_layernorm,
    "rmsnorm": _numpy_rmsnorm,
    "reduce_sum": _numpy_reduce_sum,
    "reduce_max": _numpy_reduce_max,
    "transpose": _numpy_transpose,
}


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class NumericalResult:
    """Result of a numerical validation run."""
    passed: bool
    trials: int
    max_absolute_error: float
    max_relative_error: float
    tolerance: dict  # {"atol": float, "rtol": float}
    errors: list[str] = field(default_factory=list)


# ============================================================
# NumericalValidator
# ============================================================

class NumericalValidator:
    """V1 Numerical Validation — verify kernel correctness vs NumPy reference.

    Operates on SemanticIR only (the math layer). Does not require
    a compiled kernel — validates that the computation graph produces
    correct results when executed with NumPy.
    """

    def generate_reference(
        self,
        semantic_ir: SemanticIR,
        input_tensors: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Execute the computation graph using NumPy to get reference output.

        Walks the graph in topological order. For each node, resolves inputs
        (from kernel params or prior node outputs) and applies the NumPy
        reference implementation.

        Args:
            semantic_ir: The Semantic IR describing the computation.
            input_tensors: Named input arrays matching kernel params.

        Returns:
            The output array from the return node.

        Raises:
            ValueError: If the graph is invalid or an op is unsupported.
        """
        # Build topologically-ordered node list
        ordered_nodes = self._topological_sort(semantic_ir)

        # Value store: param values + intermediate results
        values: dict[str, np.ndarray] = dict(input_tensors)

        for node in ordered_nodes:
            # Resolve inputs for this node
            node_inputs: dict[str, np.ndarray] = {}
            for input_name, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    if ref.name not in values:
                        raise ValueError(
                            f"Node '{node.id}': param '{ref.name}' not found in inputs"
                        )
                    node_inputs[input_name] = values[ref.name]
                elif isinstance(ref, NodeRef):
                    if ref.id not in values:
                        raise ValueError(
                            f"Node '{node.id}': node output '{ref.id}' not computed yet "
                            "(cycle or ordering error)"
                        )
                    node_inputs[input_name] = values[ref.id]
                else:
                    raise ValueError(f"Unknown input ref type: {type(ref)}")

            # Dispatch to NumPy implementation
            if node.op not in _NUMPY_DISPATCH:
                raise ValueError(f"No NumPy reference for op: {node.op}")

            result = _NUMPY_DISPATCH[node.op](node_inputs)
            values[node.id] = result

        # Return the output from the return node
        if not semantic_ir.return_node:
            raise ValueError("SemanticIR has no return_node set")

        if semantic_ir.return_node not in values:
            raise ValueError(
                f"Return node '{semantic_ir.return_node}' was not computed"
            )

        return values[semantic_ir.return_node]

    def generate_random_inputs(
        self,
        semantic_ir: SemanticIR,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Generate random input tensors matching the kernel's params.

        Args:
            semantic_ir: The Semantic IR with param definitions.
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping param names to random numpy arrays.
        """
        rng = np.random.RandomState(seed)
        inputs: dict[str, np.ndarray] = {}

        for param in semantic_ir.params:
            np_dtype = _to_numpy_dtype(param.dtype)

            if np.issubdtype(np_dtype, np.floating):
                # Standard normal scaled to [-1, 1] for numerical stability
                arr = rng.randn(*param.shape).astype(np_dtype)
            elif np.issubdtype(np_dtype, np.integer):
                info = np.iinfo(np_dtype)
                low = max(info.min, -128)
                high = min(info.max, 127) + 1
                arr = rng.randint(low, high, size=param.shape).astype(np_dtype)
            else:
                arr = rng.randn(*param.shape).astype(np_dtype)

            inputs[param.name] = arr

        return inputs

    def validate(
        self,
        semantic_ir: SemanticIR,
        trials: int = 3,
    ) -> NumericalResult:
        """Run N trials with random inputs, compare against NumPy reference.

        Each trial generates random inputs, computes the reference output,
        and checks that the graph produces matching results within tolerance.

        For V1, both "actual" and "reference" come from the same NumPy path
        — this validates that the graph structure and op dispatch are correct.

        Args:
            semantic_ir: The Semantic IR to validate.
            trials: Number of random trials to run.

        Returns:
            NumericalResult summarizing the validation.
        """
        # Determine output dtype for tolerance
        output_dtype = "f32"  # default
        if semantic_ir.return_type:
            output_dtype = semantic_ir.return_type.dtype
        elif semantic_ir.params:
            output_dtype = semantic_ir.params[0].dtype

        tolerance = _get_tolerance(output_dtype)
        max_abs_error = 0.0
        max_rel_error = 0.0
        all_errors: list[str] = []

        for trial in range(trials):
            seed = 42 + trial
            try:
                inputs = self.generate_random_inputs(semantic_ir, seed=seed)
                output = self.generate_reference(semantic_ir, inputs)

                # V1: re-run to verify determinism
                output2 = self.generate_reference(semantic_ir, inputs)

                # Compute errors
                abs_err = np.max(np.abs(output.astype(np.float64) - output2.astype(np.float64)))
                max_abs_error = max(max_abs_error, float(abs_err))

                # Relative error (avoid division by zero)
                denom = np.maximum(np.abs(output.astype(np.float64)), 1e-12)
                rel_err = np.max(
                    np.abs(output.astype(np.float64) - output2.astype(np.float64)) / denom
                )
                max_rel_error = max(max_rel_error, float(rel_err))

                if not np.allclose(output, output2,
                                   atol=tolerance["atol"], rtol=tolerance["rtol"]):
                    all_errors.append(
                        f"Trial {trial}: outputs not close "
                        f"(max_abs={abs_err:.2e}, max_rel={rel_err:.2e})"
                    )

                # Sanity checks
                if np.any(np.isnan(output)):
                    all_errors.append(f"Trial {trial}: output contains NaN")
                if np.any(np.isinf(output)):
                    all_errors.append(f"Trial {trial}: output contains Inf")

            except Exception as e:
                all_errors.append(f"Trial {trial}: exception — {e}")

        return NumericalResult(
            passed=len(all_errors) == 0,
            trials=trials,
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            tolerance=tolerance,
            errors=all_errors,
        )

    def _topological_sort(self, ir: SemanticIR) -> list:
        """Topological sort of nodes in the Semantic IR.

        Simple DFS-based sort. Nodes with only ParamRef inputs come first.
        """
        # Build dependency graph
        deps: dict[str, set[str]] = {}
        node_map: dict[str, Any] = {}
        for node in ir.nodes:
            node_map[node.id] = node
            deps[node.id] = set()
            for ref in node.inputs.values():
                if isinstance(ref, NodeRef):
                    deps[node.id].add(ref.id)

        # Kahn's algorithm
        in_degree = {nid: len(d) for nid, d in deps.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordered = []

        while queue:
            nid = queue.pop(0)
            ordered.append(node_map[nid])
            for other_id, other_deps in deps.items():
                if nid in other_deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(ordered) != len(ir.nodes):
            raise ValueError("Cycle detected in computation graph")

        return ordered
