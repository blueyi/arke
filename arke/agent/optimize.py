# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 8 autonomous optimization MVP.

This module intentionally starts with a deterministic strategy generator rather
than a live LLM call.  The public contract mirrors the G8 agent path:

    kernel-only .ak -> generated StrategyIR -> compile/lower validation
    -> compile/profile/adjust trajectory JSONL

The heuristic keeps the StrategyIR bounded and backend-agnostic at its core so a
future LLM runner can replace only the proposal step without changing the CLI or
trajectory artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arke.compiler.lowering import lower_full_stack
from arke.compiler.mlir_emitter import emit_mlir_skeleton
from arke.compiler.pipeline import ArkePipeline, CompilationResult
from arke.ir.akir import akir_to_dict
from arke.ir.semantic import SemanticIR, SymbolicDim
from arke.ir.strategy import StrategyIR
from arke.learn.trajectory import TrajectoryWriter


@dataclass(frozen=True)
class OptimizeResult:
    """Machine-readable result for ``arke optimize``."""

    success: bool
    kernel_id: str
    input_path: str
    output_dir: str
    strategy_path: str
    akir_path: str
    trajectory_path: str
    summary_path: str
    cycles_completed: int
    decision_count: int
    errors: list[str]
    warnings: list[str]
    best_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "kernel_id": self.kernel_id,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "strategy_path": self.strategy_path,
            "akir_path": self.akir_path,
            "trajectory_path": self.trajectory_path,
            "summary_path": self.summary_path,
            "cycles_completed": self.cycles_completed,
            "decision_count": self.decision_count,
            "best_score": self.best_score,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class HeuristicStrategyGenerator:
    """Small deterministic bounded-strategy generator for S8 bootstrapping."""

    target_hw = "nvidia_ampere"

    def generate(self, semantic_ir: SemanticIR) -> StrategyIR:
        """Generate a conservative StrategyIR from semantic metadata."""
        strategy = StrategyIR(
            kernel_id=semantic_ir.kernel_id,
            target_hw=self.target_hw,
            metadata={
                "generated_by": "heuristic-s8-mvp",
                "bounded_action_space": [
                    "tile",
                    "reorder",
                    "parallel",
                    "vectorize",
                    "place",
                    "compute",
                ],
            },
        )

        loops = self._infer_loops(semantic_ir)
        ops = {getattr(node, "op", "") for node in semantic_ir.nodes}

        if self._is_attention(ops):
            self._add_attention_strategy(strategy, semantic_ir)
        elif self._is_matmul_like(ops):
            self._add_matmul_strategy(strategy, loops)
        elif self._is_reduction_like(ops):
            self._add_reduction_strategy(strategy, loops)
        else:
            self._add_elementwise_strategy(strategy, loops)

        return strategy

    def refine(
        self,
        strategy: StrategyIR,
        *,
        cycle: int,
        profile: dict[str, Any],
    ) -> None:
        """Apply a deterministic mock adjustment for trajectory validation."""
        bottleneck = profile.get("bottleneck", "unknown")
        if cycle == 1 and bottleneck == "memory_bandwidth":
            if not _has_decision(strategy, "vectorize"):
                strategy.add_decision(_decision(
                    "vectorize",
                    {"loop": _first_loop(strategy) or "elem", "width": 4},
                    "cycle 1 adjustment: vectorize memory-bound traversal",
                ))
        elif cycle == 2 and bottleneck == "shared_memory_pressure":
            strategy.compute(
                warps=4,
                num_stages=2,
                shared_memory=32768,
                rationale="cycle 2 adjustment: cap shared-memory use for occupancy",
            )
        elif cycle >= 3 and not _has_decision(strategy, "unroll"):
            strategy.add_decision(_decision(
                "unroll",
                {"loop": _first_loop(strategy) or "elem", "factor": 2},
                "cycle 3 adjustment: expose instruction-level parallelism",
            ))

    def _infer_loops(self, semantic_ir: SemanticIR) -> list[str]:
        names = [dim.name for dim in semantic_ir.symbolic_dims]
        if names:
            return names[:3]

        if semantic_ir.params:
            shape = semantic_ir.params[0].shape
        elif semantic_ir.nodes:
            shape = semantic_ir.nodes[0].output.shape
        else:
            shape = []

        canonical = ["M", "N", "K", "D"]
        loops: list[str] = []
        for idx, dim in enumerate(shape[:3]):
            if isinstance(dim, SymbolicDim):
                loops.append(dim.name)
            elif idx < len(canonical):
                loops.append(canonical[idx])
            else:
                loops.append(f"dim{idx}")
        return loops or ["elem"]

    def _add_matmul_strategy(self, strategy: StrategyIR, loops: list[str]) -> None:
        ordered = _pad_loops(loops, ["M", "N", "K"])
        for loop, factor in zip(ordered[:3], [128, 128, 32]):
            strategy.tile(loop, [factor], f"heuristic matmul tile for {loop}")
        strategy.reorder(ordered[:3], "keep output-tile loops outermost")
        strategy.parallel(
            ordered[:2],
            {ordered[0]: "blockIdx.x", ordered[1]: "blockIdx.y"},
            "map output tiles to CTA grid",
        )
        strategy.place("A_tile", "shared", "stage lhs tile in shared memory")
        strategy.place("B_tile", "shared", "stage rhs tile in shared memory")
        strategy.compute(
            warps=4,
            num_stages=3,
            shared_memory=49152,
            rationale="conservative Ampere tensor-core resource seed",
        )

    def _add_reduction_strategy(self, strategy: StrategyIR, loops: list[str]) -> None:
        outer = loops[0]
        reduce_loop = loops[-1]
        strategy.tile(outer, [128], f"batch rows for reduction loop {outer}")
        strategy.tile(reduce_loop, [64], f"bounded reduction tile for {reduce_loop}")
        strategy.parallel([outer], {outer: "blockIdx.x"}, "parallelize independent reductions")
        strategy.compute(warps=4, num_stages=2, rationale="reduction-friendly warp budget")

    def _add_elementwise_strategy(self, strategy: StrategyIR, loops: list[str]) -> None:
        loop = loops[-1]
        strategy.tile(loop, [256], f"coalesced elementwise tile for {loop}")
        strategy.vectorize(loop, 4, "vectorize contiguous elementwise lanes")
        strategy.parallel([loop], {loop: "blockIdx.x"}, "map flat tiles across CTAs")
        strategy.compute(warps=4, num_stages=1, rationale="single-stage memory-bound kernel")

    def _add_attention_strategy(self, strategy: StrategyIR, semantic_ir: SemanticIR) -> None:
        dims = {dim.name for dim in semantic_ir.symbolic_dims}
        seq = "S" if "S" in dims else "S_q" if "S_q" in dims else "seq"
        strategy.when(
            f"{seq} <= 2048",
            [
                _decision("tile", {"loop": "Br", "factors": [128]}, "short-context query block"),
                _decision("tile", {"loop": "Bc", "factors": [128]}, "short-context key block"),
                _decision(
                    "compute",
                    {"warps": 4, "num_stages": 2, "shared_memory": 49152},
                    "short-context attention resource seed",
                    level=2,
                ),
            ],
            [
                _decision("tile", {"loop": "Br", "factors": [64]}, "long-context query block"),
                _decision("tile", {"loop": "Bc", "factors": [64]}, "long-context key block"),
                _decision(
                    "compute",
                    {"warps": 2, "num_stages": 2, "shared_memory": 32768},
                    "long-context low-memory resource guard",
                    level=2,
                ),
            ],
            rationale="shape-aware attention strategy seed",
        )

    @staticmethod
    def _is_matmul_like(ops: set[str]) -> bool:
        return bool(ops & {"matmul", "batch_matmul", "grouped_matmul"})

    @staticmethod
    def _is_reduction_like(ops: set[str]) -> bool:
        reduction_ops = {
            "softmax",
            "reduce_sum",
            "reduce_max",
            "reduce_mean",
            "layernorm",
            "rmsnorm",
        }
        return bool(ops & reduction_ops)

    @staticmethod
    def _is_attention(ops: set[str]) -> bool:
        return bool(ops & {
            "flash_attention",
            "grouped_query_attention",
            "multi_latent_attention",
            "cross_attention",
            "paged_attention",
        })


def optimize_file(
    input_path: str | Path,
    *,
    output_dir: str | Path = "benchmarks/results/phase1/stage8/track1/optimize",
    cycles: int = 3,
    dry_run: bool = True,
    target_hw: str = "nvidia_ampere",
) -> OptimizeResult:
    """Run the S8 MVP optimization flow for a kernel-only ``.ak`` file."""
    input_path = Path(input_path)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ArkePipeline()
    compile_result = pipeline.compile_file(str(input_path))
    errors = list(compile_result.errors)
    warnings: list[str] = []

    kernel_id = compile_result.kernel_name or input_path.stem
    trajectory_path = run_dir / "trajectory.jsonl"
    strategy_path = run_dir / "strategy.json"
    akir_path = run_dir / "result.akir"
    summary_path = run_dir / "summary.json"

    if not compile_result.success or compile_result.semantic_ir is None:
        summary = _summary_dict(
            success=False,
            kernel_id=kernel_id,
            input_path=input_path,
            run_dir=run_dir,
            cycles_completed=0,
            decision_count=0,
            best_score=None,
            errors=errors,
            warnings=warnings,
        )
        _write_json(summary_path, summary)
        return OptimizeResult(**summary)

    generator = HeuristicStrategyGenerator()
    strategy = generator.generate(compile_result.semantic_ir)
    strategy.target_hw = target_hw

    best_score: float | None = None
    cycles_completed = 0

    with TrajectoryWriter(trajectory_path) as writer:
        writer.write_header({
            "kernel_id": kernel_id,
            "input_path": str(input_path),
            "mode": "dry-run" if dry_run else "compile",
            "target_hw": target_hw,
            "schema": "s8-compile-profile-adjust-v1",
            "required_cycle_order": ["compile", "profile", "adjust"],
        })
        writer.write_observation(0, {
            "semantic_ir": {
                "kernel_id": compile_result.semantic_ir.kernel_id,
                "node_count": len(compile_result.semantic_ir.nodes),
                "param_count": len(compile_result.semantic_ir.params),
                "symbolic_dims": [d.name for d in compile_result.semantic_ir.symbolic_dims],
            },
        })

        for cycle in range(1, cycles + 1):
            step = cycle
            compile_event = _compile_cycle(
                compile_result,
                strategy,
                dry_run=dry_run,
            )
            writer.write_action(step, "compile", {
                "cycle": cycle,
                "decision_count": len(strategy.decisions),
                "dry_run": dry_run,
            })
            writer.write_result(step, compile_event["success"], compile_event)
            if not compile_event["success"]:
                errors.extend(compile_event.get("errors", []))
                break

            profile = _mock_profile(strategy, cycle=cycle)
            writer.write_action(step, "profile", {"cycle": cycle, "source": "mock"})
            writer.write_result(step, True, profile)
            best_score = max(best_score or 0.0, profile["score"])

            writer.write_action(step, "adjust", {
                "cycle": cycle,
                "bottleneck": profile["bottleneck"],
            })
            before = len(strategy.decisions)
            generator.refine(strategy, cycle=cycle, profile=profile)
            adjustment = {
                "cycle": cycle,
                "decisions_before": before,
                "decisions_after": len(strategy.decisions),
                "changed": len(strategy.decisions) != before,
            }
            writer.write_result(step, True, adjustment)
            cycles_completed = cycle

    strategy.to_file(str(strategy_path))
    _save_optimized_akir(compile_result, strategy, akir_path)

    summary = _summary_dict(
        success=cycles_completed == cycles and not errors,
        kernel_id=kernel_id,
        input_path=input_path,
        run_dir=run_dir,
        cycles_completed=cycles_completed,
        decision_count=len(strategy.decisions),
        best_score=best_score,
        errors=errors,
        warnings=warnings,
    )
    _write_json(summary_path, summary)
    return OptimizeResult(**summary)


def _compile_cycle(
    result: CompilationResult,
    strategy: StrategyIR,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if result.semantic_ir is None:
        return {"success": False, "errors": ["missing SemanticIR"]}
    try:
        schedule_ir, instruction_ir = lower_full_stack(result.semantic_ir, strategy)
        mlir_module = emit_mlir_skeleton(result.semantic_ir, instruction_ir)
        return {
            "success": True,
            "dry_run": dry_run,
            "schedule_loop_count": len(schedule_ir.loop_nests) if schedule_ir else 0,
            "instruction_block_count": len(instruction_ir.blocks) if instruction_ir else 0,
            "mlir_lines": len(mlir_module.splitlines()) if mlir_module else 0,
            "errors": [],
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {"success": False, "dry_run": dry_run, "errors": [str(exc)]}


def _mock_profile(strategy: StrategyIR, *, cycle: int) -> dict[str, Any]:
    bottlenecks = ["memory_bandwidth", "shared_memory_pressure", "instruction_latency"]
    score = round(0.70 + min(cycle, 3) * 0.08 + min(len(strategy.decisions), 12) * 0.01, 4)
    return {
        "cycle": cycle,
        "source": "mock_profile",
        "latency_ms": round(1.0 / score, 6),
        "score": score,
        "bottleneck": bottlenecks[(cycle - 1) % len(bottlenecks)],
        "target_score": 0.95,
    }


def _save_optimized_akir(
    result: CompilationResult,
    strategy: StrategyIR,
    path: Path,
) -> None:
    if result.semantic_ir is None:
        raise ValueError("Cannot save optimized .akir without SemanticIR")
    schedule_ir, instruction_ir = lower_full_stack(result.semantic_ir, strategy)
    combined = akir_to_dict(
        result.semantic_ir,
        strategy,
        schedule_ir=schedule_ir,
        instruction_ir=instruction_ir,
    )
    _write_json(path, combined)


def _summary_dict(
    *,
    success: bool,
    kernel_id: str,
    input_path: Path,
    run_dir: Path,
    cycles_completed: int,
    decision_count: int,
    best_score: float | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "success": success,
        "kernel_id": kernel_id,
        "input_path": str(input_path),
        "output_dir": str(run_dir),
        "strategy_path": str(run_dir / "strategy.json"),
        "akir_path": str(run_dir / "result.akir"),
        "trajectory_path": str(run_dir / "trajectory.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
        "cycles_completed": cycles_completed,
        "decision_count": decision_count,
        "best_score": best_score,
        "errors": list(errors),
        "warnings": list(warnings),
    }


def _decision(kind: str, params: dict[str, Any], rationale: str, *, level: int = 1):
    from arke.ir.strategy import Decision, Rationale

    return Decision(
        kind=kind,
        params=params,
        rationale=Rationale(text=rationale),
        level=level,
    )


def _has_decision(strategy: StrategyIR, kind: str) -> bool:
    return any(getattr(decision, "kind", None) == kind for decision in strategy.decisions)


def _first_loop(strategy: StrategyIR) -> str | None:
    for decision in strategy.decisions:
        params = getattr(decision, "params", {})
        if "loop" in params:
            return str(params["loop"])
    return None


def _pad_loops(loops: list[str], fallback: list[str]) -> list[str]:
    result = list(loops)
    for loop in fallback:
        if len(result) >= len(fallback):
            break
        if loop not in result:
            result.append(loop)
    return result


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


__all__ = [
    "HeuristicStrategyGenerator",
    "OptimizeResult",
    "optimize_file",
]
