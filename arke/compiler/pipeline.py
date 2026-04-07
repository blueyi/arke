# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Compiler — New Compilation Pipeline for SemanticIR.

Full pipeline: .ak → parse → SemanticIR + StrategyIR → validate → execute.

This is the new pipeline replacing the S6 IRGraph-based PassPipeline for
the multi-layer IR architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from arke.compiler.validator import validate_semantic_ir
from arke.ir.akir import akir_from_dict, akir_to_dict
from arke.ir.akir import load_akir as _load_akir
from arke.ir.akir import save_akir
from arke.ir.converters import ast_to_semantic, ast_to_strategy
from arke.ir.ops.interpreter import INTERPRETER
from arke.ir.semantic import (
    MultiOutputNode,
    Node,
    NodeRef,
    ParamRef,
    SemanticIR,
    SymbolicDim,
)
from arke.ir.strategy import StrategyIR
from arke.lang.grammar import parse_file, parse_string


# ─── Dtype Mapping ──────────────────────────────────────────────────────────

_DTYPE_MAP: dict[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "bool": torch.bool,
}


def _arke_dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Convert Arke dtype string to PyTorch dtype."""
    return _DTYPE_MAP.get(dtype_str, torch.float32)


# ─── Compilation Result ────────────────────────────────────────────────────

@dataclass
class CompilationResult:
    """Result of compiling an Arke source.

    Attributes:
        semantic_ir: The SemanticIR representation.
        strategy_ir: The StrategyIR representation (None if no strategy block).
        success: Whether compilation succeeded without errors.
        errors: List of error messages from validation.
        kernel_name: Name of the compiled kernel.
    """
    semantic_ir: SemanticIR | None = None
    strategy_ir: StrategyIR | None = None
    success: bool = False
    errors: list[str] = field(default_factory=list)
    kernel_name: str = ""

    def save_akir(self, path: str, indent: int = 2) -> None:
        """Save the compiled IR to a .akir JSON file.

        Args:
            path: Output file path.
            indent: JSON indentation (default 2).

        Raises:
            ValueError: If compilation was not successful or semantic_ir is None.
        """
        if self.semantic_ir is None:
            raise ValueError("Cannot save .akir: semantic_ir is None")
        save_akir(self.semantic_ir, self.strategy_ir, path, indent=indent)


# ─── Pipeline ──────────────────────────────────────────────────────────────

class ArkePipeline:
    """Full Arke compilation pipeline: .ak → SemanticIR → StrategyIR → execution.

    This is the new pipeline replacing the S6 IRGraph-based PassPipeline for
    the multi-layer IR architecture.

    Usage:
        pipeline = ArkePipeline()
        result = pipeline.compile_file("examples/operators/00_relu.ak")
        if result.success:
            outputs = pipeline.execute(result, {"X": torch.randn(128, 3072)})
    """

    def compile_file(self, path: str) -> CompilationResult:
        """Parse .ak file → SemanticIR + StrategyIR → validate → ready to execute.

        Args:
            path: Path to the .ak file.

        Returns:
            CompilationResult with semantic_ir, strategy_ir, and validation status.
        """
        result = CompilationResult()

        # Parse
        try:
            program = parse_file(path)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        return self._compile_program(program, result)

    def compile_string(self, source: str) -> CompilationResult:
        """Parse .ak source → SemanticIR + StrategyIR → validate → ready to execute.

        Args:
            source: Arke language source code string.

        Returns:
            CompilationResult with semantic_ir, strategy_ir, and validation status.
        """
        result = CompilationResult()

        # Parse
        try:
            program = parse_string(source)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        return self._compile_program(program, result)

    def _compile_program(self, program: Any, result: CompilationResult) -> CompilationResult:
        """Compile a parsed Program AST into SemanticIR + StrategyIR.

        Args:
            program: Program AST from the parser.
            result: CompilationResult to populate.

        Returns:
            Populated CompilationResult.
        """
        if not program.kernels:
            result.errors.append("No kernel definition found in source")
            return result

        kernel_def = program.kernels[0]
        result.kernel_name = kernel_def.name

        # Convert AST → SemanticIR
        try:
            result.semantic_ir = ast_to_semantic(kernel_def)
        except Exception as e:
            result.errors.append(f"SemanticIR conversion error: {e}")
            return result

        # Convert AST → StrategyIR (optional)
        if program.strategies:
            try:
                result.strategy_ir = ast_to_strategy(program.strategies[0])
            except Exception as e:
                result.errors.append(f"StrategyIR conversion error: {e}")
                # Non-fatal: strategy is optional

        # Run semantic pass pipeline (SSA validation + shape inference)
        from arke.compiler.semantic_passes import (
            semantic_shape_inference_pass,
            semantic_ssa_validation_pass,
        )
        from arke.compiler.semantic_pipeline import SemanticPassPipeline

        sem_pipeline = SemanticPassPipeline("compile")
        sem_pipeline.add_pass(semantic_ssa_validation_pass)
        sem_pipeline.add_pass(semantic_shape_inference_pass)
        pass_result = sem_pipeline.run(result.semantic_ir)

        result.errors.extend(pass_result.errors)
        result.success = len(result.errors) == 0
        return result

    def execute(
        self,
        result: CompilationResult,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Execute a compiled kernel with the given inputs.

        Walks semantic_ir.nodes in order, resolving inputs from params or
        previous node outputs, executing via INTERPRETER.execute().

        Args:
            result: A successful CompilationResult from compile_file/compile_string.
            inputs: Mapping from parameter name to input tensor.

        Returns:
            Mapping from output name to output tensor. The return node's
            output is keyed as "output". For multi-node kernels, all
            intermediate results are also available keyed by node ID.

        Raises:
            ValueError: If compilation failed or SemanticIR is None.
            KeyError: If a required input parameter is missing.
            RuntimeError: If execution fails.
        """
        if not result.success or result.semantic_ir is None:
            raise ValueError(
                f"Cannot execute: compilation failed with errors: "
                f"{result.errors}"
            )

        sir = result.semantic_ir

        # Validate all required params are provided
        for param in sir.params:
            if param.name not in inputs:
                raise KeyError(
                    f"Missing input parameter {param.name!r}. "
                    f"Required params: {[p.name for p in sir.params]}"
                )

        # Execute nodes in order
        node_outputs: dict[str, torch.Tensor] = {}

        for node in sir.nodes:
            if isinstance(node, (Node, MultiOutputNode)):
                # Resolve inputs
                resolved_inputs: dict[str, torch.Tensor] = {}
                for input_name, ref in node.inputs.items():
                    if isinstance(ref, ParamRef):
                        if ref.name not in inputs:
                            raise KeyError(
                                f"Node {node.id!r}: ParamRef {ref.name!r} "
                                f"not found in provided inputs"
                            )
                        resolved_inputs[input_name] = inputs[ref.name]
                    elif isinstance(ref, NodeRef):
                        if ref.id not in node_outputs:
                            raise RuntimeError(
                                f"Node {node.id!r}: NodeRef {ref.id!r} "
                                f"not found in executed node outputs"
                            )
                        resolved_inputs[input_name] = node_outputs[ref.id]

                # Execute via interpreter
                output = INTERPRETER.execute(
                    node.op, resolved_inputs, node.attrs
                )
                node_outputs[node.id] = output

        # Build result dict
        outputs: dict[str, torch.Tensor] = dict(node_outputs)

        # Add "output" key for the return node
        if sir.return_node and sir.return_node in node_outputs:
            outputs["output"] = node_outputs[sir.return_node]
        elif sir.return_node and sir.return_node in inputs:
            # Direct param return (identity)
            outputs["output"] = inputs[sir.return_node]

        return outputs

    @staticmethod
    def load_akir(path: str) -> CompilationResult:
        """Load a .akir file and create a CompilationResult.

        Args:
            path: Path to the .akir file.

        Returns:
            CompilationResult with the loaded SemanticIR and StrategyIR.
        """
        result = CompilationResult()
        try:
            semantic_ir, strategy_ir = _load_akir(path)
            result.semantic_ir = semantic_ir
            result.strategy_ir = strategy_ir
            result.kernel_name = semantic_ir.kernel_id

            # Validate the loaded IR
            validation_errors = validate_semantic_ir(semantic_ir)
            result.errors.extend(validation_errors)
            result.success = len(result.errors) == 0
        except Exception as e:
            result.errors.append(f"Failed to load .akir: {e}")
        return result
