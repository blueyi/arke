# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — Triton backend implementation.

Implements ArkeBackend for Triton code generation, compilation, and execution.
"""

from __future__ import annotations

from arke.backend.base import (
    ArkeBackend,
    CompileResult,
    ProfileResult,
    register_backend,
)
from arke.backend.compiler import TritonCompiler
from arke.backend.triton_template_engine import TritonTemplateEngine
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


class TritonBackend(ArkeBackend):
    """Triton code generation backend for NVIDIA GPUs."""

    name = "triton"

    def __init__(self) -> None:
        self._engine = TritonTemplateEngine()
        self._compiler = TritonCompiler()

    def translate(self, semantic: SemanticIR, strategy: StrategyIR) -> str:
        """Generate Triton Python code from IR + strategy."""
        return self._engine.translate(semantic, strategy)

    def compile(self, source_code: str) -> CompileResult:
        """Compile Triton source code."""
        return self._compiler.compile(source_code)

    def run(self, compiled: CompileResult, inputs: dict) -> dict:
        """Execute compiled kernel with given inputs.

        Args:
            compiled: Result from compile().
            inputs: Dict mapping tensor names to torch.Tensor values.

        Returns:
            Dict with 'output' key containing the result tensor.
        """
        output = self._compiler.run(compiled, inputs)
        return {"output": output}

    def profile(
        self,
        compiled: CompileResult,
        inputs: dict,
        warmup: int = 5,
        runs: int = 20,
    ) -> ProfileResult:
        """Profile kernel performance."""
        assert compiled.code, "CompileResult must contain source code"
        return self._compiler.profile(
            compiled.code, inputs, warmup=warmup, runs=runs
        )


# Auto-register on import
register_backend("triton", TritonBackend)
