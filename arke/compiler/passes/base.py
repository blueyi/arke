# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Pass Infrastructure (S6 Track 2, Task C2.1).

Defines the Pass protocol, PassContext, PassPipeline, and CompilationResult.
Design ref: docs/architecture/arke-compiler-infrastructure.md §4

Usage:
    pipeline = PassPipeline("default")
    pipeline.add_pass(ShapeInferencePass())
    pipeline.add_pass(SSAValidationPass())
    result = pipeline.run(ir_graph)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from arke.ir.graph import IRGraph
from arke.ir.ops.registry import REGISTRY, OpRegistry

logger = logging.getLogger(__name__)


# ── Diagnostics ───────────────────────────────────────────────

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Diagnostic:
    """A diagnostic message from a pass."""
    severity: Severity
    pass_name: str
    message: str
    node_id: str | None = None

    def __str__(self) -> str:
        loc = f" [node={self.node_id}]" if self.node_id else ""
        return f"[{self.severity.value.upper()}] {self.pass_name}{loc}: {self.message}"


# ── Hardware Profile ──────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Target hardware description."""
    name: str = "nvidia_ampere"
    compute_capability: tuple[int, int] = (8, 6)  # RTX 3060
    shared_memory_bytes: int = 49152
    max_threads_per_block: int = 1024
    warp_size: int = 32
    num_sms: int = 30
    peak_tflops_f16: float = 12.74


# ── Pass Context ──────────────────────────────────────────────

@dataclass
class PassContext:
    """Shared mutable context threading through all passes.

    The IRGraph may be replaced by transform passes.
    Diagnostics and artifacts accumulate across passes.
    """
    graph: IRGraph
    registry: OpRegistry = field(default_factory=lambda: REGISTRY)
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def add_error(self, pass_name: str, msg: str, node_id: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.ERROR, pass_name, msg, node_id))

    def add_warning(self, pass_name: str, msg: str, node_id: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.WARNING, pass_name, msg, node_id))

    def add_info(self, pass_name: str, msg: str, node_id: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.INFO, pass_name, msg, node_id))

    def has_errors(self) -> bool:
        return any(d.severity == Severity.ERROR for d in self.diagnostics)

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]


# ── Pass Result ───────────────────────────────────────────────

@dataclass
class PassResult:
    """Result from a single pass execution."""
    success: bool
    modified: bool = False
    error: str | None = None

    @classmethod
    def ok(cls, modified: bool = False) -> PassResult:
        return cls(success=True, modified=modified)

    @classmethod
    def fail(cls, error: str) -> PassResult:
        return cls(success=False, error=error)


# ── Pass Protocol ─────────────────────────────────────────────

@runtime_checkable
class ArkePass(Protocol):
    """A single compilation pass — stateless; all mutable state in PassContext."""
    name: str

    def run(self, ctx: PassContext) -> PassResult:
        """Execute this pass on the context."""
        ...


# ── Compilation Result ────────────────────────────────────────

@dataclass
class CompilationResult:
    """Result of running a full pipeline."""
    success: bool
    graph: IRGraph | None = None
    source_code: str = ""
    error: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    passes_run: list[str] = field(default_factory=list)


# ── Pass Pipeline ─────────────────────────────────────────────

class PassPipeline:
    """Ordered sequence of compilation passes.

    Usage:
        pipeline = PassPipeline("default")
        pipeline.add_pass(ShapeInferencePass())
        pipeline.add_pass(SSAValidationPass())
        result = pipeline.run(ir_graph)
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._passes: list[ArkePass] = []
        self._registry: OpRegistry = REGISTRY
        self._hardware: HardwareProfile = HardwareProfile()

    def add_pass(self, p: ArkePass) -> PassPipeline:
        """Add a pass to the pipeline."""
        self._passes.append(p)
        return self

    def with_registry(self, r: OpRegistry) -> PassPipeline:
        self._registry = r
        return self

    def with_hardware(self, hw: HardwareProfile) -> PassPipeline:
        self._hardware = hw
        return self

    @property
    def passes(self) -> list[ArkePass]:
        return list(self._passes)

    def run(self, graph: IRGraph) -> CompilationResult:
        """Execute all passes in order.

        Stops on first pass failure or if context accumulates errors.

        Args:
            graph: Input IR graph

        Returns:
            CompilationResult with final state
        """
        t0 = time.monotonic()
        ctx = PassContext(
            graph=graph,
            registry=self._registry,
            hardware=self._hardware,
        )

        passes_run = []

        for p in self._passes:
            logger.debug(f"Running pass: {p.name}")

            try:
                result = p.run(ctx)
            except Exception as e:
                return CompilationResult(
                    success=False,
                    graph=ctx.graph,
                    error=f"Pass {p.name!r} raised: {e}",
                    diagnostics=ctx.diagnostics,
                    artifacts=ctx.artifacts,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    passes_run=passes_run,
                )

            passes_run.append(p.name)

            if not result.success:
                return CompilationResult(
                    success=False,
                    graph=ctx.graph,
                    error=result.error or f"Pass {p.name!r} failed",
                    diagnostics=ctx.diagnostics,
                    artifacts=ctx.artifacts,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    passes_run=passes_run,
                )

            if ctx.has_errors():
                return CompilationResult(
                    success=False,
                    graph=ctx.graph,
                    error=f"Errors after pass {p.name!r}: {[str(e) for e in ctx.errors()]}",
                    diagnostics=ctx.diagnostics,
                    artifacts=ctx.artifacts,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    passes_run=passes_run,
                )

        return CompilationResult(
            success=True,
            graph=ctx.graph,
            diagnostics=ctx.diagnostics,
            artifacts=ctx.artifacts,
            duration_ms=(time.monotonic() - t0) * 1000,
            passes_run=passes_run,
        )
