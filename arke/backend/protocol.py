# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — Protocol + Artifact types (S6 Track 3, Task C3.1).

Defines the ArkeBackend protocol that all backends must implement.
Phase 1: TritonBackend (NVIDIA GPU via Triton)
Phase 2: TritonBackend (Ascend) + MLIR skeleton
Phase 3: MLIRBackend (full compiler control)
Phase 4: LLVMBackend (direct LLVM IR)

Design ref: docs/architecture/arke-compiler-infrastructure.md §7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from arke.ir.graph import IRGraph


@dataclass
class BackendArtifact:
    """Intermediate artifact from the lower() step.

    Contains generated source code (e.g., Triton Python, MLIR)
    and metadata about the generation.
    """
    source_code: str
    backend_name: str
    op_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledKernel:
    """Result of compile() — a ready-to-launch kernel.

    For Triton: the compiled_fn is a callable Triton kernel.
    For MLIR/LLVM: binary_path points to compiled shared object.
    """
    success: bool
    compiled_fn: Any = None  # callable kernel (Triton JIT)
    binary_path: str | None = None
    error: str | None = None
    backend_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, fn: Any = None, backend_name: str = "", **meta: Any) -> CompiledKernel:
        return cls(success=True, compiled_fn=fn, backend_name=backend_name, metadata=meta)

    @classmethod
    def fail(cls, error: str) -> CompiledKernel:
        return cls(success=False, error=error)


@runtime_checkable
class ArkeBackend(Protocol):
    """Protocol for Arke compilation backends.

    Backends implement a three-phase pipeline:
    1. lower(graph) → BackendArtifact (source code generation)
    2. compile(artifact) → CompiledKernel (JIT / AOT compilation)
    3. run(kernel, inputs) → outputs (execution)
    """
    name: str

    def lower(self, graph: IRGraph) -> BackendArtifact:
        """Generate target-specific source code from IR graph."""
        ...

    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """Compile source code to executable kernel."""
        ...

    def run(self, kernel: CompiledKernel, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute compiled kernel with given inputs."""
        ...

    def supports_op(self, op_name: str) -> bool:
        """Check if this backend can handle a given operator."""
        ...


# ── Backend Registry ──────────────────────────────────────────

class BackendRegistry:
    """Route target hardware strings to ArkeBackend instances.

    Usage:
        registry = BackendRegistry()
        registry.register(triton_backend, ["nvidia_sm86", "nvidia_generic"])
        backend = registry.get("nvidia_sm86")
    """

    def __init__(self) -> None:
        self._backends: dict[str, ArkeBackend] = {}
        self._target_map: dict[str, str] = {}

    def register(self, backend: ArkeBackend, targets: list[str]) -> None:
        """Register a backend for one or more target hardware strings."""
        self._backends[backend.name] = backend
        for t in targets:
            self._target_map[t] = backend.name

    def get(self, target: str) -> ArkeBackend:
        """Get backend for a target hardware string."""
        if target in self._target_map:
            name = self._target_map[target]
            return self._backends[name]
        if target in self._backends:
            return self._backends[target]
        raise KeyError(
            f"No backend for target {target!r}. "
            f"Available: {list(self._target_map.keys())}"
        )

    def list_backends(self) -> list[str]:
        return list(self._backends.keys())

    def list_targets(self) -> list[str]:
        return list(self._target_map.keys())

    def __contains__(self, target: str) -> bool:
        return target in self._target_map or target in self._backends
