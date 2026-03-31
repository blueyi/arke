# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — Abstract base class for code generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


@dataclass
class CompileResult:
    """Result of code generation + compilation."""
    success: bool
    code: str = ""              # Generated source code
    error: str | None = None
    binary_path: str | None = None


@dataclass
class ProfileResult:
    """Result of GPU profiling."""
    latency_us: float = 0.0
    tflops: float = 0.0
    roofline_efficiency: float = 0.0
    vs_baseline: dict[str, Any] | None = None


class ArkeBackend(ABC):
    """Abstract backend for code generation and execution.

    Each backend implements:
    1. translate: Strategy IR → target source code
    2. compile: source code → executable binary
    3. run: binary + inputs → outputs
    4. profile: binary → performance metrics
    """

    name: str

    @abstractmethod
    def translate(self, semantic: SemanticIR, strategy: StrategyIR) -> str:
        """Generate target source code from IR."""
        ...

    @abstractmethod
    def compile(self, source_code: str) -> CompileResult:
        """Compile source code to executable."""
        ...

    @abstractmethod
    def run(self, compiled: CompileResult, inputs: dict) -> dict:
        """Execute compiled kernel with given inputs."""
        ...

    @abstractmethod
    def profile(self, compiled: CompileResult, inputs: dict,
                warmup: int = 5, runs: int = 20) -> ProfileResult:
        """Profile kernel performance."""
        ...


# ============================================================
# Backend Registry
# ============================================================

_BACKENDS: dict[str, type[ArkeBackend]] = {}


def register_backend(name: str, backend_cls: type[ArkeBackend]) -> None:
    """Register a code generation backend."""
    _BACKENDS[name] = backend_cls


def get_backend(name: str) -> type[ArkeBackend]:
    """Get a registered backend by name."""
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Backend '{name}' not found. Available: {available}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    """List all registered backend names."""
    return list(_BACKENDS.keys())
