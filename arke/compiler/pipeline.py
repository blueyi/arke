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
from arke.compiler.lowering import lower_full_stack
from arke.compiler.mlir_emitter import emit_mlir_skeleton
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
from arke.ir.strategy import ConditionalDecision, Decision, Rationale, StrategyIR
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
    schedule_ir: Any | None = None
    instruction_ir: Any | None = None
    mlir_module: str | None = None
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
        save_akir(
            self.semantic_ir,
            self.strategy_ir,
            path,
            indent=indent,
            schedule_ir=self.schedule_ir,
            instruction_ir=self.instruction_ir,
        )


# ─── Pipeline ──────────────────────────────────────────────────────────────

def _synthesize_strategy_from_compile_advice(
    semantic_ir: SemanticIR,
    strategy_ir: StrategyIR,
) -> None:
    advice = strategy_ir.metadata.get("compile_advice")
    if not advice or advice.get("allow_compile", True):
        return
    if strategy_ir.decisions:
        return

    node_ops = {getattr(node, "op", "") for node in semantic_ir.nodes}
    dim_names = {dim.name for dim in semantic_ir.symbolic_dims}

    if "paged_attention" in node_ops and {"NB", "MB"}.issubset(dim_names):
        strategy_ir.when(
            "NB <= 1024 and MB <= 128",
            [
                Decision(kind="tile", params={"loop": "NB", "factors": [128]}, rationale=Rationale(text="synthesized from compile advice: smaller page table working set")),
                Decision(kind="tile", params={"loop": "MB", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: smaller block-table slice")),
                Decision(kind="compute", params={"warps": 4, "num_stages": 2, "shared_memory": 32768}, rationale=Rationale(text="synthesized from compile advice: paged attention conservative resources"), level=2),
            ],
            [
                Decision(kind="tile", params={"loop": "NB", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: larger page regime guard")),
                Decision(kind="tile", params={"loop": "MB", "factors": [32]}, rationale=Rationale(text="synthesized from compile advice: narrower block-table guard")),
                Decision(kind="compute", params={"warps": 2, "num_stages": 2, "shared_memory": 16384}, rationale=Rationale(text="synthesized from compile advice: paged attention low-memory guard"), level=2),
            ],
            rationale="auto-synthesized paged-attention strategy from compile advice",
        )
        return

    if "multi_latent_attention" in node_ops and "D_c" in dim_names:
        strategy_ir.when(
            "D_c <= 64",
            [
                Decision(kind="tile", params={"loop": "D_c", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: compact KV branch")),
                Decision(kind="compute", params={"warps": 4, "num_stages": 2, "shared_memory": 32768}, rationale=Rationale(text="synthesized from compile advice: compact-KV resources"), level=2),
            ],
            [
                Decision(kind="tile", params={"loop": "D_c", "factors": [32]}, rationale=Rationale(text="synthesized from compile advice: larger compressed-KV guard")),
                Decision(kind="compute", params={"warps": 2, "num_stages": 2, "shared_memory": 16384}, rationale=Rationale(text="synthesized from compile advice: compressed-KV low-memory guard"), level=2),
            ],
            rationale="auto-synthesized MLA strategy from compile advice",
        )
        return

    if "cross_attention" in node_ops and {"S_q", "S_kv"}.issubset(dim_names):
        strategy_ir.when(
            "S_kv <= 4096",
            [
                Decision(kind="tile", params={"loop": "S_q", "factors": [128]}, rationale=Rationale(text="synthesized from compile advice: shorter KV branch")),
                Decision(kind="tile", params={"loop": "S_kv", "factors": [128]}, rationale=Rationale(text="synthesized from compile advice: symmetric query/kv window")),
                Decision(kind="compute", params={"warps": 4, "num_stages": 2, "shared_memory": 32768}, rationale=Rationale(text="synthesized from compile advice: cross-attention balanced resources"), level=2),
            ],
            [
                Decision(kind="tile", params={"loop": "S_q", "factors": [128]}, rationale=Rationale(text="synthesized from compile advice: keep query tile stable")),
                Decision(kind="tile", params={"loop": "S_kv", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: reduce KV tile under long KV context")),
                Decision(kind="compute", params={"warps": 2, "num_stages": 2, "shared_memory": 16384}, rationale=Rationale(text="synthesized from compile advice: cross-attention KV-heavy guard"), level=2),
            ],
            rationale="auto-synthesized cross-attention strategy from compile advice",
        )
        return

    if "rope" in node_ops and "D" in dim_names:
        strategy_ir.when(
            "D <= 128",
            [
                Decision(kind="tile", params={"loop": "D", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: vector-friendly rope branch")),
                Decision(kind="compute", params={"warps": 4, "num_stages": 2, "shared_memory": 16384}, rationale=Rationale(text="synthesized from compile advice: rope compact-dim resources"), level=2),
            ],
            [
                Decision(kind="tile", params={"loop": "D", "factors": [32]}, rationale=Rationale(text="synthesized from compile advice: narrower rope vector width guard")),
                Decision(kind="compute", params={"warps": 2, "num_stages": 2, "shared_memory": 8192}, rationale=Rationale(text="synthesized from compile advice: rope high-dim low-memory guard"), level=2),
            ],
            rationale="auto-synthesized rope strategy from compile advice",
        )
        return

    is_attention = bool(node_ops & {"flash_attention", "grouped_query_attention", "multi_latent_attention", "cross_attention", "paged_attention", "rope"})
    if not (is_attention and "S" in dim_names):
        return

    strategy_ir.when(
        "S <= 4096",
        [
            Decision(kind="tile", params={"loop": "Br", "factors": [128]}, rationale=Rationale(text="synthesized from compile advice: short-context branch")),
            Decision(kind="tile", params={"loop": "Bc", "factors": [128]}, rationale=Rationale(text="synthesized from compile advice: short-context branch")),
            Decision(kind="compute", params={"warps": 8, "num_stages": 2, "shared_memory": 65536}, rationale=Rationale(text="synthesized from compile advice: short-context resources"), level=2),
        ],
        [
            Decision(kind="tile", params={"loop": "Br", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: long-context guard")),
            Decision(kind="tile", params={"loop": "Bc", "factors": [64]}, rationale=Rationale(text="synthesized from compile advice: long-context guard")),
            Decision(kind="compute", params={"warps": 4, "num_stages": 2, "shared_memory": 32768}, rationale=Rationale(text="synthesized from compile advice: long-context resources"), level=2),
        ],
        rationale="auto-synthesized conditional strategy from compile advice",
    )


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

        if result.success and result.semantic_ir is not None:
            if result.strategy_ir is not None:
                _synthesize_strategy_from_compile_advice(result.semantic_ir, result.strategy_ir)
                try:
                    result.schedule_ir, result.instruction_ir = lower_full_stack(
                        result.semantic_ir,
                        result.strategy_ir,
                    )
                except Exception as e:
                    result.errors.append(f"Lowering error: {e}")
                    result.success = False
            if result.success:
                result.mlir_module = emit_mlir_skeleton(
                    result.semantic_ir,
                    result.instruction_ir,
                )

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
            semantic_ir, strategy_ir, schedule_ir, instruction_ir = _load_akir(path)
            result.semantic_ir = semantic_ir
            result.strategy_ir = strategy_ir
            result.schedule_ir = schedule_ir
            result.instruction_ir = instruction_ir
            result.kernel_name = semantic_ir.kernel_id
            result.mlir_module = emit_mlir_skeleton(semantic_ir, instruction_ir)

            # Validate the loaded IR
            validation_errors = validate_semantic_ir(semantic_ir)
            result.errors.extend(validation_errors)
            result.success = len(result.errors) == 0
        except Exception as e:
            result.errors.append(f"Failed to load .akir: {e}")
        return result
