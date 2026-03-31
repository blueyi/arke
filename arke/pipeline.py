# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Pipeline — End-to-end orchestration.

Connects all components:
  SemanticIR → ArkeEnv → Strategy decisions → Codegen → Compile → Verify → Profile

This is the top-level entry point for both CLI and programmatic use.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arke.engine.env import ArkeEnv
from arke.engine.numerical_check import NumericalValidator
from arke.ir.builder import KernelBuilder
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


@dataclass
class PipelineResult:
    """Result of an end-to-end pipeline run."""
    kernel_id: str
    target_hw: str
    decisions: int
    semantic_ir: dict[str, Any]
    strategy_ir: dict[str, Any]
    numerical_validation: dict[str, Any] | None = None
    codegen_source: str | None = None
    gpu_result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ArkePipeline:
    """End-to-end pipeline for kernel optimization.

    Usage:
        pipeline = ArkePipeline()
        ir = pipeline.build_matmul_relu(1024, 512, 2048, "f16")
        result = pipeline.run(ir, "nvidia_ampere", decisions=[
            ("tile", {"loop": "i", "factors": [64, 16]}, "cache alignment"),
            ("tile", {"loop": "j", "factors": [128, 16]}, "coalescing"),
            ("fuse", {"ops": ["matmul_0", "relu_1"], "type": "epilogue"}, "eliminate write"),
        ])
    """

    def __init__(self) -> None:
        self.numerical_validator = NumericalValidator()

    # ─── Convenience builders ───

    @staticmethod
    def build_matmul(M: int, K: int, N: int, dtype: str = "f16") -> SemanticIR:
        """Build a simple matmul kernel."""
        b = KernelBuilder("matmul")
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        m = b.op("matmul", A="A", B="B")
        b.returns(m, [M, N], dtype)
        return b.build()

    @staticmethod
    def build_matmul_relu(M: int, K: int, N: int, dtype: str = "f16") -> SemanticIR:
        """Build a fused matmul+relu kernel."""
        b = KernelBuilder("fused_matmul_relu")
        b.param("A", [M, K], dtype)
        b.param("B", [K, N], dtype)
        m = b.op("matmul", A="A", B="B")
        r = b.op("relu", X=m)
        b.returns(r, [M, N], dtype)
        return b.build()

    @staticmethod
    def build_softmax(M: int, N: int, dtype: str = "f16") -> SemanticIR:
        """Build a softmax kernel."""
        b = KernelBuilder("softmax")
        b.param("X", [M, N], dtype)
        s = b.op("softmax", X="X")
        b.returns(s, [M, N], dtype)
        return b.build()

    # ─── Pipeline execution ───

    def run(
        self,
        semantic_ir: SemanticIR,
        target_hw: str,
        decisions: list[tuple[str, dict, str]] | None = None,
        validate_numerical: bool = True,
        codegen: bool = False,
        profile: bool = False,
    ) -> PipelineResult:
        """Run the end-to-end pipeline.

        Args:
            semantic_ir: The kernel to optimize
            target_hw: Hardware target
            decisions: List of (kind, params, rationale) tuples
            validate_numerical: Run V1 numerical check on Semantic IR
            codegen: Generate Triton code (requires backend)
            profile: Run GPU profiling (requires GPU)
        """
        start = time.time()
        result = PipelineResult(
            kernel_id=semantic_ir.kernel_id,
            target_hw=target_hw,
            decisions=0,
            semantic_ir=semantic_ir.to_dict(),
            strategy_ir={},
        )

        # 1. Create ArkeEnv
        env = ArkeEnv(semantic_ir, target_hw)

        # 2. Apply decisions
        if decisions:
            for kind, params, rationale in decisions:
                apply_result = env.apply_decision(kind, params, rationale)
                if not apply_result.get("success", False):
                    result.errors.append(
                        f"Decision '{kind}' failed: {apply_result.get('validation', {}).get('violations', [])}"
                    )
                    break
                result.decisions += 1

        result.strategy_ir = env.strategy.to_dict()

        # 3. V1 Numerical validation (on Semantic IR — checks the math)
        if validate_numerical:
            try:
                num_result = self.numerical_validator.validate(semantic_ir)
                result.numerical_validation = {
                    "passed": num_result.passed,
                    "trials": num_result.trials,
                    "max_absolute_error": num_result.max_absolute_error,
                    "max_relative_error": num_result.max_relative_error,
                    "tolerance": num_result.tolerance,
                    "errors": num_result.errors,
                }
            except Exception as e:
                result.numerical_validation = {
                    "passed": False,
                    "error": str(e),
                }

        # 4. Codegen (if requested and backend available)
        if codegen:
            try:
                from arke.backend.triton_backend import TritonBackend
                backend = TritonBackend()
                source = backend.translate(semantic_ir, env.strategy)
                result.codegen_source = source
            except ImportError:
                result.errors.append("Triton backend not available")
            except Exception as e:
                result.errors.append(f"Codegen failed: {e}")

        # 5. GPU profiling (if requested)
        if profile and result.codegen_source:
            try:
                from arke.backend.triton_backend import TritonBackend
                backend = TritonBackend()
                compiled = backend.compile(result.codegen_source)
                if compiled.success:
                    # Generate test inputs
                    import torch
                    inputs = {}
                    for p in semantic_ir.params:
                        dtype_map = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16}
                        t_dtype = dtype_map.get(p.dtype, torch.float16)
                        inputs[p.name] = torch.randn(p.shape, device="cuda", dtype=t_dtype)
                    prof = backend.profile(compiled, inputs)
                    result.gpu_result = {
                        "latency_us": prof.latency_us,
                        "tflops": prof.tflops,
                        "roofline_efficiency": prof.roofline_efficiency,
                    }
            except Exception as e:
                result.errors.append(f"Profiling failed: {e}")

        result.duration_seconds = round(time.time() - start, 3)
        return result

    # ─── Serialization ───

    @staticmethod
    def save_result(result: PipelineResult, path: str) -> None:
        """Save pipeline result to JSON."""
        data = {
            "kernel_id": result.kernel_id,
            "target_hw": result.target_hw,
            "decisions": result.decisions,
            "semantic_ir": result.semantic_ir,
            "strategy_ir": result.strategy_ir,
            "numerical_validation": result.numerical_validation,
            "codegen_source": result.codegen_source,
            "gpu_result": result.gpu_result,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        }
        Path(path).write_text(json.dumps(data, indent=2))
