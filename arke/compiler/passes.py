# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Pass Infrastructure.

Unified pass interface and pipeline for SemanticIR transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


@dataclass
class PassContext:
    """Context passed through the pass pipeline."""
    semantic_ir: SemanticIR
    strategy_ir: StrategyIR | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ArkePass(ABC):
    """Base class for all Arke compiler passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Pass name (e.g., 'ShapeInference')."""
        pass

    @abstractmethod
    def run(self, ctx: PassContext) -> list[str]:
        """Execute the pass.
        
        Args:
            ctx: Pass context with IR and metadata
        
        Returns:
            List of error messages. Empty list means success.
        """
        pass


class ShapeInferencePass(ArkePass):
    """Infer output shapes for all SemanticIR nodes."""

    @property
    def name(self) -> str:
        return "ShapeInference"

    def run(self, ctx: PassContext) -> list[str]:
        """Run shape inference on SemanticIR."""
        try:
            from arke.compiler.semantic_passes import semantic_shape_inference_pass
            return semantic_shape_inference_pass(ctx.semantic_ir)
        except Exception as e:
            return [f"ShapeInferencePass error: {e}"]


class SSAValidationPass(ArkePass):
    """Validate SSA structural correctness."""

    @property
    def name(self) -> str:
        return "SSAValidation"

    def run(self, ctx: PassContext) -> list[str]:
        """Run SSA validation on SemanticIR."""
        try:
            from arke.compiler.semantic_passes import semantic_ssa_validation_pass
            return semantic_ssa_validation_pass(ctx.semantic_ir)
        except Exception as e:
            return [f"SSAValidationPass error: {e}"]


class RationalePreservationPass(ArkePass):
    """Preserve @rationale annotations through pipeline."""

    @property
    def name(self) -> str:
        return "RationalePreservation"

    def run(self, ctx: PassContext) -> list[str]:
        """Validate rationale preservation."""
        if ctx.strategy_ir is None:
            return ["StrategyIR not available for rationale validation"]
        try:
            from arke.compiler.semantic_passes import rationale_preservation_pass
            return rationale_preservation_pass(ctx.semantic_ir, ctx.strategy_ir)
        except Exception as e:
            return [f"RationalePreservationPass error: {e}"]


class PassPipeline:
    """Ordered execution of compiler passes."""

    def __init__(self):
        """Initialize with default passes."""
        self.passes = [
            ShapeInferencePass(),
            SSAValidationPass(),
            RationalePreservationPass(),
        ]

    def add_pass(self, pass_: ArkePass) -> None:
        """Add a pass to the pipeline."""
        self.passes.append(pass_)

    def run(self, ctx: PassContext) -> dict[str, list[str]]:
        """Execute all passes in order.
        
        Args:
            ctx: Pass context
        
        Returns:
            Dict mapping pass name to error list
        """
        results: dict[str, list[str]] = {}
        
        for pass_ in self.passes:
            errors = pass_.run(ctx)
            results[pass_.name] = errors
        
        return results

    def __len__(self) -> int:
        """Number of passes in pipeline."""
        return len(self.passes)
