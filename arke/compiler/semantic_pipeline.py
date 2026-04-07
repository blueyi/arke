# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Semantic Pass Pipeline.

Pass pipeline operating on SemanticIR, replacing the S6 PassPipeline for
the multi-layer IR architecture.

Usage:
    from arke.compiler.semantic_pipeline import SemanticPassPipeline

    pipeline = SemanticPassPipeline()
    pipeline.add_pass(semantic_ssa_validation_pass)
    pipeline.add_pass(semantic_shape_inference_pass)
    result = pipeline.run(ir)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from arke.ir.semantic import SemanticIR


@dataclass
class SemanticPassResult:
    """Result from running the semantic pass pipeline.

    Attributes:
        success: True if all passes completed without errors.
        errors: Combined error messages from all passes.
        passes_run: Names of passes that were executed.
        duration_ms: Total pipeline execution time in milliseconds.
    """
    success: bool = True
    errors: list[str] = field(default_factory=list)
    passes_run: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class SemanticPassPipeline:
    """Pass pipeline operating on SemanticIR.

    Replaces S6 PassPipeline for the multi-layer IR architecture.
    Each pass is a callable: (SemanticIR) -> list[str] (error messages).

    Usage:
        pipeline = SemanticPassPipeline()
        pipeline.add_pass(semantic_ssa_validation_pass)
        pipeline.add_pass(semantic_shape_inference_pass)
        result = pipeline.run(ir)
    """

    def __init__(self, name: str = "semantic_default") -> None:
        self.name = name
        self._passes: list[tuple[str, Callable[[SemanticIR], list[str]]]] = []

    def add_pass(
        self, pass_fn: Callable[[SemanticIR], list[str]], name: str | None = None,
    ) -> "SemanticPassPipeline":
        """Add a pass to the pipeline.

        Args:
            pass_fn: A callable that takes SemanticIR and returns a list of
                error/warning messages.
            name: Optional name for the pass. Defaults to the function name.

        Returns:
            self for chaining.
        """
        pass_name = name or getattr(pass_fn, "__name__", str(pass_fn))
        self._passes.append((pass_name, pass_fn))
        return self

    @property
    def passes(self) -> list[str]:
        """Return list of pass names."""
        return [name for name, _ in self._passes]

    def run(self, ir: SemanticIR) -> SemanticPassResult:
        """Execute all passes in order on the SemanticIR.

        Stops on first pass that produces errors.

        Args:
            ir: The SemanticIR to process (may be mutated by passes).

        Returns:
            SemanticPassResult with combined results.
        """
        t0 = time.monotonic()
        result = SemanticPassResult()

        for pass_name, pass_fn in self._passes:
            try:
                errors = pass_fn(ir)
            except Exception as e:
                result.success = False
                result.errors.append(f"Pass {pass_name!r} raised: {e}")
                result.passes_run.append(pass_name)
                break

            result.passes_run.append(pass_name)

            if errors:
                result.success = False
                result.errors.extend(errors)
                break  # Stop on first failure

        result.duration_ms = (time.monotonic() - t0) * 1000
        return result
