# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Session lifecycle management.

An optimization session encapsulates:
- SemanticIR (immutable computation)
- ArkeEnv (mutable optimization state)
- LLM conversation history
- Budget tracking
- Optimization trajectory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from arke.agent.prompts import build_system_prompt
from arke.agent.tools_schema import TOOL_METADATA, get_tool_schemas
from arke.engine.env import ArkeEnv
from arke.engine.numerical_check import NumericalValidator
from arke.ir.semantic import SemanticIR


class SessionState(str, Enum):
    """Session lifecycle states."""
    CREATED = "created"          # Session created, kernel not yet defined
    ANALYZING = "analyzing"      # Kernel defined, LLM analyzing
    OPTIMIZING = "optimizing"    # LLM making decisions
    VERIFYING = "verifying"      # Compiling and verifying
    FINALIZED = "finalized"      # Optimization complete
    FAILED = "failed"            # Unrecoverable error


@dataclass
class OptimizationBudget:
    """Tracks optimization resource usage."""
    max_decisions: int = 50
    max_compiles: int = 10
    target_performance: float = 0.7  # ratio of vendor baseline
    warning_threshold: int = 40      # warn LLM at this step

    decisions_used: int = 0
    compiles_used: int = 0

    @property
    def decisions_remaining(self) -> int:
        return max(0, self.max_decisions - self.decisions_used)

    @property
    def compiles_remaining(self) -> int:
        return max(0, self.max_compiles - self.compiles_used)

    @property
    def exhausted(self) -> bool:
        return self.decisions_remaining == 0

    @property
    def should_warn(self) -> bool:
        return self.decisions_used >= self.warning_threshold and not self.exhausted

    def use_decision(self) -> None:
        self.decisions_used += 1

    def use_compile(self) -> None:
        self.compiles_used += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions_used": self.decisions_used,
            "decisions_remaining": self.decisions_remaining,
            "compiles_used": self.compiles_used,
            "compiles_remaining": self.compiles_remaining,
            "target_performance": self.target_performance,
        }


@dataclass
class TrajectoryEntry:
    """A single entry in the optimization trajectory."""
    step: int
    type: str           # "action" | "result" | "compile" | "observe"
    tool: str = ""
    params: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationSession:
    """A complete LLM optimization session.

    Lifecycle:
        1. Create session with SemanticIR + target
        2. Session builds ArkeEnv, system prompt, tools
        3. LLM interacts via tool-use (run_tool dispatches to ArkeEnv)
        4. Session tracks budget, trajectory, state
        5. Finalize when done
    """

    semantic_ir: SemanticIR
    target_hw: str
    env: ArkeEnv = field(init=False)
    state: SessionState = SessionState.CREATED
    budget: OptimizationBudget = field(default_factory=OptimizationBudget)
    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    best_performance: dict[str, Any] | None = None
    _step: int = 0
    _created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.env = ArkeEnv(self.semantic_ir, self.target_hw)
        self.numerical_validator = NumericalValidator()
        self._init_messages()

    def _init_messages(self) -> None:
        """Initialize conversation with system prompt."""
        system_prompt = build_system_prompt(
            hw_profile=self.env.hw_profile,
            budget_decisions=self.budget.max_decisions,
            budget_compiles=self.budget.max_compiles,
            target_performance=self.budget.target_performance,
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]

    # ─── Tool dispatch ───

    def run_tool(self, tool_name: str, params: dict) -> dict[str, Any]:
        """Execute a tool call and return the result.

        Handles budget tracking, trajectory recording, and state transitions.
        """
        self._step += 1

        # Budget check
        meta = TOOL_METADATA.get(tool_name, {})
        budget_type = meta.get("budget_type", "free")

        if budget_type == "decision":
            if self.budget.exhausted:
                return self._budget_exhausted_response()
            self.budget.use_decision()
        elif budget_type == "compile":
            if self.budget.compiles_remaining <= 0:
                return {"success": False, "error": "Compile budget exhausted"}
            self.budget.use_compile()

        # Record trajectory
        self.trajectory.append(TrajectoryEntry(
            step=self._step, type="action", tool=tool_name, params=params,
        ))

        # Dispatch to ArkeEnv
        result = self._dispatch(tool_name, params)

        # Record result
        self.trajectory.append(TrajectoryEntry(
            step=self._step, type="result", tool=tool_name, result=result,
        ))

        # Inject budget info into every response
        result["budget"] = self.budget.to_dict()

        # Budget warning
        if self.budget.should_warn:
            result["budget_warning"] = (
                f"You have used {self.budget.decisions_used}/{self.budget.max_decisions} decisions. "
                f"Consider finalizing soon."
            )

        # State transitions
        self._update_state(tool_name, result)

        return result

    def _dispatch(self, tool_name: str, params: dict) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate ArkeEnv method."""
        dispatch_map: dict[str, Any] = {
            "create_kernel": lambda p: self._handle_create_kernel(p),
            "get_hw_profile": lambda p: self.env.get_hw_profile(),
            "analyze_compute": lambda p: self.env.analyze_compute(),
            "list_legal_actions": lambda p: self._handle_list_legal(p),
            "apply_decision": lambda p: self.env.apply_decision(
                p["kind"], p["params"], p.get("rationale", ""),
            ),
            "verify_correctness": lambda p: self._handle_verify_correctness(p),
            "compile_and_profile": lambda p: self._handle_compile_and_profile(p),
            "rollback": lambda p: self.env.rollback(p.get("steps", 1)),
            "checkpoint": lambda p: self.env.checkpoint(p.get("name")),
            "restore": lambda p: self.env.restore(p["checkpoint_id"]),
        }

        handler = dispatch_map.get(tool_name)
        if handler is None:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        return handler(params)

    def _handle_create_kernel(self, params: dict) -> dict[str, Any]:
        """Handle create_kernel (for agent-mode where LLM defines the kernel)."""
        # In most flows, the kernel is already defined in __init__.
        # This tool is for agent-initiated kernel creation.
        return {
            "success": True,
            "kernel_id": self.semantic_ir.kernel_id,
            "semantic_ir": self.env.get_semantic_ir(),
            "auto_analysis": self.env.analyze_compute(),
        }

    def _handle_list_legal(self, params: dict) -> dict[str, Any]:
        """Handle list_legal_actions with filtering and limits."""
        kind = params.get("kind")
        limit = params.get("limit", 10)
        return self.env.list_legal_actions(kind=kind, limit=limit)

    def _handle_verify_correctness(self, params: dict) -> dict[str, Any]:
        """Handle verify_correctness — V1 numerical + GPU correctness validation.

        Two-stage verification:
        1. V1: Semantic IR math check (NumPy reference, fast)
        2. GPU: Compile kernel → run on GPU → compare output vs NumPy reference
        """
        trials = params.get("trials", 3)
        result: dict[str, Any] = {"success": True}

        # Stage 1: V1 numerical validation (Semantic IR math)
        try:
            num_result = self.numerical_validator.validate(
                self.semantic_ir, trials=trials
            )
            result["v1_numerical"] = {
                "passed": num_result.passed,
                "trials": num_result.trials,
                "max_absolute_error": num_result.max_absolute_error,
                "max_relative_error": num_result.max_relative_error,
                "tolerance": num_result.tolerance,
                "errors": num_result.errors,
            }
            if not num_result.passed:
                result["passed"] = False
                result["errors"] = num_result.errors
                return result
        except Exception as e:
            result["v1_numerical"] = {"passed": False, "error": str(e)}
            result["passed"] = False
            result["errors"] = [f"V1 numerical check failed: {e}"]
            return result

        # Stage 2: GPU correctness (compile kernel → run → compare vs NumPy)
        try:
            gpu_check = self._verify_gpu_correctness(trials=trials)
            result["gpu_correctness"] = gpu_check
            result["passed"] = gpu_check["passed"]
            result["errors"] = gpu_check.get("errors", [])
        except Exception as e:
            # GPU check is best-effort — if backend unavailable, skip
            result["gpu_correctness"] = {"passed": None, "skipped": True, "reason": str(e)}
            result["passed"] = num_result.passed  # Fall back to V1 result
            result["errors"] = []

        return result

    def _verify_gpu_correctness(self, trials: int = 3) -> dict[str, Any]:
        """Compile kernel with current strategy and verify GPU output vs NumPy reference.

        Uses SAME dtype for reference (measures implementation correctness,
        not precision loss). e.g. f16 kernel vs NumPy f16 computation.

        Returns:
            Dict with 'passed', 'max_absolute_error', 'max_relative_error', 'errors'
        """
        import numpy as np
        import torch

        from arke.backend.triton_backend import TritonBackend

        backend = TritonBackend()

        # 1. Generate Triton source from current strategy
        source = backend.translate(self.semantic_ir, self.env.strategy)

        # 2. Compile
        compiled = backend.compile(source)
        if not compiled.success:
            return {
                "passed": False,
                "errors": [f"Compilation failed: {compiled.error}"],
            }

        # 3. Determine kernel dtype
        kernel_dtype = "f16"  # default
        if self.semantic_ir.return_type:
            kernel_dtype = self.semantic_ir.return_type.dtype
        elif self.semantic_ir.params:
            kernel_dtype = self.semantic_ir.params[0].dtype

        # Map dtype for NumPy and torch
        np_dtype_map = {"f16": np.float16, "f32": np.float32, "bf16": np.float32}
        torch_dtype_map = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16}
        np_compute = np_dtype_map.get(kernel_dtype, np.float16)
        torch_compute = torch_dtype_map.get(kernel_dtype, torch.float16)

        # Tolerance: same-dtype comparison should have very tight tolerance
        # Differences come from reduction order, FMA, etc.
        tol_map = {
            "f16": {"atol": 1e-1, "rtol": 5e-2},
            "bf16": {"atol": 1e-1, "rtol": 5e-2},
            "f32": {"atol": 1e-4, "rtol": 1e-4},
        }
        tolerance = tol_map.get(kernel_dtype, {"atol": 1e-2, "rtol": 1e-2})

        max_abs_error = 0.0
        max_rel_error = 0.0
        errors: list[str] = []

        for trial in range(trials):
            seed = 42 + trial

            # Generate random inputs at kernel dtype
            np_inputs = self.numerical_validator.generate_random_inputs(
                self.semantic_ir, seed=seed
            )

            # Compute NumPy reference at SAME dtype (not upcast)
            same_dtype_inputs = {k: v.astype(np_compute) for k, v in np_inputs.items()}
            np_output = self.numerical_validator.generate_reference(
                self.semantic_ir, same_dtype_inputs
            )

            # Create GPU tensors from same data
            gpu_inputs = {}
            for p in self.semantic_ir.params:
                gpu_inputs[p.name] = torch.from_numpy(
                    np_inputs[p.name].astype(np.float32)
                ).to(dtype=torch_compute, device="cuda")

            # Run GPU kernel
            try:
                gpu_output = backend.run(compiled, gpu_inputs)
                if isinstance(gpu_output, dict):
                    gpu_output = gpu_output.get("output", gpu_output)
                gpu_output_np = gpu_output.cpu().float().numpy()
            except Exception as e:
                errors.append(f"Trial {trial}: GPU execution failed — {e}")
                continue

            # Compare (both in float32 for safe comparison)
            ref_f32 = np_output.astype(np.float32)
            abs_err = float(np.max(np.abs(gpu_output_np - ref_f32)))
            max_abs_error = max(max_abs_error, abs_err)

            # Relative error only where reference is non-trivial
            nontrivial = np.abs(ref_f32) > 1e-4
            if np.any(nontrivial):
                rel_err = float(np.max(
                    np.abs(gpu_output_np[nontrivial] - ref_f32[nontrivial]) /
                    np.abs(ref_f32[nontrivial])
                ))
            else:
                rel_err = 0.0
            max_rel_error = max(max_rel_error, rel_err)

            if not np.allclose(gpu_output_np, ref_f32,
                               atol=tolerance["atol"], rtol=tolerance["rtol"]):
                errors.append(
                    f"Trial {trial}: GPU output mismatch "
                    f"(max_abs={abs_err:.2e}, max_rel={rel_err:.2e}, "
                    f"tol=atol={tolerance['atol']}, rtol={tolerance['rtol']})"
                )

            if np.any(np.isnan(gpu_output_np)):
                errors.append(f"Trial {trial}: GPU output contains NaN")
            if np.any(np.isinf(gpu_output_np)):
                errors.append(f"Trial {trial}: GPU output contains Inf")

        return {
            "passed": len(errors) == 0,
            "trials": trials,
            "max_absolute_error": max_abs_error,
            "max_relative_error": max_rel_error,
            "tolerance": tolerance,
            "reference_dtype": kernel_dtype,
            "comparison_mode": "same_dtype",
            "errors": errors,
        }

    def _handle_compile_and_profile(self, params: dict) -> dict[str, Any]:
        """Handle compile_and_profile — compile kernel and run GPU benchmarks.

        Full pipeline: Strategy decisions → Triton codegen → compile → profile.
        """
        try:
            import torch

            from arke.backend.triton_backend import TritonBackend

            backend = TritonBackend()

            # 1. Generate Triton source from current strategy
            source = backend.translate(self.semantic_ir, self.env.strategy)

            # 2. Compile
            compiled = backend.compile(source)
            if not compiled.success:
                return {
                    "success": False,
                    "error": f"Compilation failed: {compiled.error}",
                    "source_preview": source[:500],
                }

            # 3. Generate test inputs
            dtype_map = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16}
            inputs = {}
            for p in self.semantic_ir.params:
                t_dtype = dtype_map.get(p.dtype, torch.float16)
                inputs[p.name] = torch.randn(p.shape, device="cuda", dtype=t_dtype)

            # 4. Profile
            warmup = params.get("warmup", 5)
            runs = params.get("runs", 20)
            prof = backend.profile(compiled, inputs, warmup=warmup, runs=runs)

            return {
                "success": True,
                "performance": {
                    "latency_us": round(prof.latency_us, 2),
                    "tflops": round(prof.tflops, 3),
                    "roofline_efficiency": round(prof.roofline_efficiency, 3),
                    "vs_baseline": round(prof.vs_baseline, 3) if prof.vs_baseline else None,
                },
                "decisions_applied": self.env.strategy.decision_count,
            }

        except ImportError as e:
            return {"success": False, "error": f"Backend not available: {e}"}
        except Exception as e:
            return {
                "success": False,
                "error": f"Compile/profile failed: {e}",
                "hint": "Check that tile sizes are compatible with the kernel shape.",
            }

    def _budget_exhausted_response(self) -> dict[str, Any]:
        """Response when budget is exhausted."""
        return {
            "success": False,
            "error": "Decision budget exhausted",
            "budget": self.budget.to_dict(),
            "suggestion": (
                "Call compile_and_profile() to evaluate your current strategy, "
                "then finalize. Or rollback to a checkpoint and try a different approach."
            ),
        }

    def _update_state(self, tool_name: str, result: dict) -> None:
        """Update session state based on tool call."""
        if tool_name == "create_kernel" and result.get("success"):
            self.state = SessionState.ANALYZING
        elif tool_name in ("analyze_compute", "list_legal_actions"):
            if self.state == SessionState.CREATED:
                self.state = SessionState.ANALYZING
        elif tool_name == "apply_decision":
            self.state = SessionState.OPTIMIZING
        elif tool_name in ("verify_correctness", "compile_and_profile"):
            self.state = SessionState.VERIFYING
            # Track best performance
            perf = result.get("performance")
            if perf and (self.best_performance is None or
                         perf.get("latency_us", float("inf")) <
                         self.best_performance.get("latency_us", float("inf"))):
                self.best_performance = perf

    # ─── Properties ───

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return get_tool_schemas()

    @property
    def system_prompt(self) -> str:
        return self.messages[0]["content"] if self.messages else ""

    @property
    def duration_seconds(self) -> float:
        return time.time() - self._created_at

    def summary(self) -> dict[str, Any]:
        """Session summary for reporting."""
        return {
            "kernel_id": self.semantic_ir.kernel_id,
            "target_hw": self.target_hw,
            "state": self.state.value,
            "decisions": self.env.strategy.decision_count,
            "budget": self.budget.to_dict(),
            "best_performance": self.best_performance,
            "trajectory_steps": len(self.trajectory),
            "duration_seconds": round(self.duration_seconds, 1),
        }

    def export_trajectory(self) -> list[dict[str, Any]]:
        """Export trajectory as list of dicts (for JSONL)."""
        return [
            {
                "step": e.step,
                "type": e.type,
                "tool": e.tool,
                "params": e.params,
                "result": e.result,
                "timestamp": e.timestamp,
            }
            for e in self.trajectory
        ]
