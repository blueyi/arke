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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arke.compiler.default_strategy import DefaultStrategyGenerator
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

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def correct(self) -> bool:
        """True if numerical validation passed (or was not run)."""
        if self.numerical_validation is None:
            return True
        return bool(self.numerical_validation.get("passed", True))

    @property
    def latency_us(self) -> float | None:
        """Kernel latency in microseconds, or None if not profiled."""
        if self.gpu_result is None:
            return None
        return self.gpu_result.get("latency_us")

    @property
    def strategy_source(self) -> str:
        """Source of the strategy (user-provided or auto-generated)."""
        return self.strategy_ir.get("_source", "unknown")


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
        """Initialize the pipeline with a numerical validator."""
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
                        f"Decision '{kind}' failed: "
                        f"{apply_result.get('validation', {}).get('violations', [])}"
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
                        dtype_map = {
                            "f16": torch.float16,
                            "f32": torch.float32,
                            "bf16": torch.bfloat16,
                        }
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

    @classmethod
    def from_ak_file(
        cls,
        ak_path: str,
        target_hw: str = "nvidia_ampere",
        **run_kwargs,
    ) -> PipelineResult:
        """Compile and run a .ak file end-to-end.

        If the .ak file has no ``strategy`` block, the DefaultStrategyGenerator
        automatically produces a hardware-aware baseline Strategy IR.

        Args:
            ak_path:    Path to the .ak source file.
            target_hw:  Target hardware name (default: ``nvidia_ampere``).
            **run_kwargs: Forwarded to :meth:`run` (codegen, profile, etc.).

        Example::

            result = ArkePipeline.from_ak_file("mykernel.ak", target_hw="nvidia_ampere")
            print(result.strategy_ir)   # auto-generated if no strategy block
        """
        from arke.parser.converter import ast_to_ir
        from arke.parser.parser import parse_file
        from arke.engine.env import ArkeEnv
        from pathlib import Path
        import json

        prog = parse_file(ak_path)
        if not prog.kernels:
            raise ValueError(f"No kernel found in {ak_path}")

        kernel = prog.kernels[0]
        sem_ir = ast_to_ir(kernel)

        # ── Strategy resolution ──────────────────────────────────────────
        if prog.strategies:
            # User provided a strategy block — use ast_to_strategy converter
            from arke.compiler.ast_to_strategy import program_to_strategy
            strategy = program_to_strategy(prog, kernel.name, kernel_id=sem_ir.kernel_id)
            if strategy is None:
                # Fallback: use first strategy definition
                from arke.compiler.ast_to_strategy import ast_to_strategy
                strategy = ast_to_strategy(prog.strategies[0], kernel_id=sem_ir.kernel_id)
            source_note = "user-provided strategy block"
        else:
            # No strategy block — auto-generate from hardware profile
            hw_path = (
                Path(__file__).parent / "ir" / "targets" / f"{target_hw}.json"
            )
            hw_profile = json.loads(hw_path.read_text()) if hw_path.exists() else {"name": target_hw}
            gen = DefaultStrategyGenerator(hw_profile)
            strategy = gen.generate(sem_ir)
            source_note = f"auto-generated (no strategy block in {Path(ak_path).name})"

        # Inject strategy into env directly and run
        env = ArkeEnv(sem_ir, target_hw)
        env.strategy = strategy

        pipeline = cls()
        decisions_list = [
            (d.kind, d.params, d.rationale.text if d.rationale else "")
            for d in strategy.decisions
        ]
        result = pipeline.run(
            sem_ir,
            target_hw,
            decisions=None,   # already applied via env.strategy
            **run_kwargs,
        )
        result.strategy_ir = strategy.to_dict()
        result.strategy_ir["_source"] = source_note
        return result

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
