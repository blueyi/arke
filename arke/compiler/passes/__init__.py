# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — Passes Package."""

from arke.compiler.passes.base import (
    ArkePass,
    CompilationResult,
    Diagnostic,
    HardwareProfile,
    PassContext,
    PassPipeline,
    PassResult,
    Severity,
)
from arke.compiler.passes.builtin import (
    RationalePreservationPass,
    SSAValidationPass,
    ShapeInferencePass,
)

__all__ = [
    "ArkePass", "CompilationResult", "Diagnostic", "HardwareProfile",
    "PassContext", "PassPipeline", "PassResult", "Severity",
    "ShapeInferencePass", "SSAValidationPass", "RationalePreservationPass",
]
