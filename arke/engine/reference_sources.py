# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — Reference Sources for accuracy comparison.

Pluggable reference implementations:
- NumPyCPUSource: NumPy on CPU (default, highest precision)
- TorchGPUSource: PyTorch on GPU (for GPU-vs-GPU comparison)
- CustomSource: User-provided reference tensors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from arke.ir.semantic import SemanticIR


class ReferenceSource(ABC):
    """Abstract base for reference data sources."""

    name: str = "unknown"

    @abstractmethod
    def generate_reference(
        self,
        semantic_ir: SemanticIR,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute reference output for given inputs."""
        ...

    @abstractmethod
    def generate_inputs(
        self,
        semantic_ir: SemanticIR,
        seed: int = 42,
        input_type: str = "normal",
    ) -> dict[str, np.ndarray]:
        """Generate input tensors."""
        ...


class NumPyCPUSource(ReferenceSource):
    """NumPy CPU reference — highest precision, always available."""

    name = "numpy_cpu"

    def __init__(self, compute_dtype: str = "f64"):
        """Args:
            compute_dtype: Precision for reference computation.
                "f64" (default) for maximum precision, "f32" for faster.
        """
        self.compute_dtype = compute_dtype
        self._dtype_map = {"f64": np.float64, "f32": np.float32}

    def generate_reference(
        self,
        semantic_ir: SemanticIR,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        from arke.engine.numerical_check import NumericalValidator
        validator = NumericalValidator()

        # Upcast inputs to compute_dtype for higher precision reference
        np_dtype = self._dtype_map.get(self.compute_dtype, np.float64)
        upcast_inputs = {k: v.astype(np_dtype) for k, v in inputs.items()}

        return validator.generate_reference(semantic_ir, upcast_inputs)

    def generate_inputs(
        self,
        semantic_ir: SemanticIR,
        seed: int = 42,
        input_type: str = "normal",
    ) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        inputs: dict[str, np.ndarray] = {}

        for param in semantic_ir.params:
            shape = param.shape
            if input_type == "normal":
                arr = rng.randn(*shape).astype(np.float32)
            elif input_type == "uniform":
                arr = rng.uniform(-1, 1, size=shape).astype(np.float32)
            elif input_type == "edge":
                # Mix of normal values + edge cases
                arr = rng.randn(*shape).astype(np.float32)
                flat = arr.ravel()
                n = len(flat)
                # Inject edge cases at random positions
                edge_count = max(1, n // 100)
                edge_vals = [0.0, -0.0, 1e-7, -1e-7, 1e4, -1e4, 65504.0, -65504.0]
                for i in range(edge_count):
                    idx = rng.randint(0, n)
                    flat[idx] = edge_vals[i % len(edge_vals)]
                arr = flat.reshape(shape)
            else:
                arr = rng.randn(*shape).astype(np.float32)

            inputs[param.name] = arr

        return inputs


class TorchGPUSource(ReferenceSource):
    """PyTorch GPU reference — for GPU-vs-GPU comparison."""

    name = "torch_gpu"

    def __init__(self, device: str = "cuda"):
        self.device = device

    def generate_reference(
        self,
        semantic_ir: SemanticIR,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        import torch

        # Convert to GPU tensors
        gpu_inputs = {}
        for name, arr in inputs.items():
            gpu_inputs[name] = torch.from_numpy(arr.astype(np.float32)).to(
                device=self.device, dtype=torch.float32
            )

        # Execute using PyTorch ops
        output = self._execute_graph(semantic_ir, gpu_inputs)
        return output.cpu().numpy()

    def generate_inputs(
        self,
        semantic_ir: SemanticIR,
        seed: int = 42,
        input_type: str = "normal",
    ) -> dict[str, np.ndarray]:
        # Delegate to NumPy source for input generation
        cpu_source = NumPyCPUSource()
        return cpu_source.generate_inputs(semantic_ir, seed, input_type)

    def _execute_graph(self, ir: SemanticIR, inputs: dict) -> Any:
        """Execute SemanticIR using PyTorch ops on GPU."""
        import torch
        from arke.ir.semantic import NodeRef, ParamRef

        values = dict(inputs)
        for node in ir.nodes:
            node_inputs = {}
            for input_name, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    node_inputs[input_name] = values[ref.name]
                elif isinstance(ref, NodeRef):
                    node_inputs[input_name] = values[ref.id]

            result = self._dispatch_op(node.op, node_inputs)
            values[node.id] = result

        return values[ir.return_node]

    @staticmethod
    def _dispatch_op(op: str, inputs: dict) -> Any:
        import torch
        import torch.nn.functional as F

        dispatch = {
            "matmul": lambda i: torch.matmul(i["A"], i["B"]),
            "batch_matmul": lambda i: torch.matmul(i["A"], i["B"]),
            "relu": lambda i: F.relu(i["X"]),
            "gelu": lambda i: F.gelu(i["X"]),
            "add": lambda i: i["A"] + i["B"],
            "mul": lambda i: i["A"] * i["B"],
            "softmax": lambda i: F.softmax(i["X"], dim=-1),
            "reduce_sum": lambda i: torch.sum(i["X"], dim=-1),
            "reduce_max": lambda i: torch.max(i["X"], dim=-1).values,
            "transpose": lambda i: i["X"].T,
        }
        if op not in dispatch:
            raise ValueError(f"Unsupported op for TorchGPU: {op}")
        return dispatch[op](inputs)


class CustomSource(ReferenceSource):
    """User-provided reference data."""

    name = "custom"

    def __init__(self, ref_data: dict[str, np.ndarray] | None = None):
        self._ref_data = ref_data or {}

    def set_reference(self, inputs: dict[str, np.ndarray], output: np.ndarray) -> None:
        self._ref_data["output"] = output
        self._ref_data.update(inputs)

    def generate_reference(
        self,
        semantic_ir: SemanticIR,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        if "output" not in self._ref_data:
            raise ValueError("No reference output set. Call set_reference() first.")
        return self._ref_data["output"]

    def generate_inputs(
        self,
        semantic_ir: SemanticIR,
        seed: int = 42,
        input_type: str = "normal",
    ) -> dict[str, np.ndarray]:
        # Return stored inputs, or fall back to random
        stored = {k: v for k, v in self._ref_data.items() if k != "output"}
        if stored:
            return stored
        return NumPyCPUSource().generate_inputs(semantic_ir, seed, input_type)


# Registry
REFERENCE_SOURCES: dict[str, type[ReferenceSource]] = {
    "numpy_cpu": NumPyCPUSource,
    "torch_gpu": TorchGPUSource,
    "custom": CustomSource,
}


def get_reference_source(name: str = "numpy_cpu", **kwargs: Any) -> ReferenceSource:
    """Get a reference source by name."""
    cls = REFERENCE_SOURCES.get(name)
    if cls is None:
        raise ValueError(f"Unknown reference source: {name}. Available: {list(REFERENCE_SOURCES.keys())}")
    return cls(**kwargs)
