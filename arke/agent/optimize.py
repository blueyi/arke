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
import re
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
    input_kind: str = "ak_file"
    normalized_source_path: str | None = None
    source_text_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "kernel_id": self.kernel_id,
            "input_path": self.input_path,
            "input_kind": self.input_kind,
            "normalized_source_path": self.normalized_source_path,
            "source_text_path": self.source_text_path,
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


@dataclass(frozen=True)
class OptimizeInput:
    """Normalized input payload for the Stage 8 optimization flow."""

    kind: str
    display_path: str
    source: str
    kernel_id_hint: str | None = None
    source_text: str | None = None


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
        strategy.add_decision(_decision(
            "vectorize",
            {"loop": loop, "width": 4},
            "vectorize contiguous elementwise lanes",
        ))
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


_OP_ALIASES = {
    "relu": "relu",
    "gelu": "gelu",
    "softmax": "softmax",
    "matmul": "matmul",
    "matrix_multiply": "matmul",
    "matrix_multiplication": "matmul",
    "linear": "matmul",
    "matmul_gelu": "matmul_gelu",
    "linear_gelu": "matmul_gelu",
    "gemm_gelu": "matmul_gelu",
    "reduce_sum": "reduce_sum",
    "sum": "reduce_sum",
    "reduce_max": "reduce_max",
    "max": "reduce_max",
    "reduce_mean": "reduce_mean",
    "mean": "reduce_mean",
}


@dataclass(frozen=True)
class _StructuredOptimizeSpec:
    op: str
    shape: tuple[int, ...]
    dtype: str = "f16"
    kernel: str | None = None


class OptimizeInputRouter:
    """Route ``arke optimize`` inputs into a compile-ready ``.ak`` source."""

    def route(
        self,
        value: str | Path | None,
        *,
        kernel: str | None = None,
        shape: str | list[int] | tuple[int, ...] | None = None,
        dtype: str = "f16",
    ) -> OptimizeInput:
        if kernel is not None:
            return self._from_structured(kernel=kernel, shape=shape, dtype=dtype)

        if value is None:
            raise ValueError("optimize input is required unless --kernel is provided")

        raw = str(value)
        path = Path(raw)
        if path.exists() and path.is_file():
            source = path.read_text(encoding="utf-8")
            if path.suffix == ".ak" or _looks_like_ak_source(source):
                return OptimizeInput(kind="ak_file", display_path=str(path), source=source)
            return self._from_code(source, display_path=str(path), kind="code_file", dtype=dtype)

        if _looks_like_ak_source(raw):
            return OptimizeInput(
                kind="ak_source",
                display_path="<inline-ak>",
                source=raw,
                source_text=raw,
            )

        if _looks_like_code_source(raw):
            return self._from_code(
                raw,
                display_path="<code-snippet>",
                kind="code_snippet",
                dtype=dtype,
            )

        return self._from_natural_language(raw, dtype=dtype)

    def _from_code(
        self,
        code: str,
        *,
        display_path: str,
        kind: str,
        dtype: str,
    ) -> OptimizeInput:
        spec = _parse_code_spec(code, dtype=dtype)
        source = _ak_source_from_spec(spec)
        return OptimizeInput(
            kind=kind,
            display_path=display_path,
            source=source,
            kernel_id_hint=spec.kernel or f"{spec.op}_kernel",
            source_text=code,
        )

    def _from_structured(
        self,
        *,
        kernel: str,
        shape: str | list[int] | tuple[int, ...] | None,
        dtype: str,
    ) -> OptimizeInput:
        op = _normalize_op(kernel)
        if shape is None:
            raise ValueError("--shape is required when --kernel is provided")
        spec = _StructuredOptimizeSpec(op=op, shape=_parse_shape(shape), dtype=dtype)
        source = _ak_source_from_spec(spec)
        return OptimizeInput(
            kind="structured_args",
            display_path=f"<structured:{op}>",
            source=source,
            kernel_id_hint=spec.kernel or f"{op}_kernel",
        )

    def _from_natural_language(self, text: str, *, dtype: str) -> OptimizeInput:
        spec = _parse_natural_language_spec(text, dtype=dtype)
        source = _ak_source_from_spec(spec)
        return OptimizeInput(
            kind="natural_language",
            display_path="<natural-language>",
            source=source,
            kernel_id_hint=spec.kernel or f"{spec.op}_kernel",
            source_text=text,
        )


def optimize(
    input_value: str | Path | None = None,
    *,
    kernel: str | None = None,
    shape: str | list[int] | tuple[int, ...] | None = None,
    dtype: str = "f16",
    output_dir: str | Path = "benchmarks/results/phase1/stage8/track1/optimize",
    cycles: int = 3,
    dry_run: bool = True,
    target_hw: str = "nvidia_ampere",
) -> OptimizeResult:
    """Run Stage 8 optimization for .ak, inline source, NL, or structured input."""
    routed = OptimizeInputRouter().route(
        input_value,
        kernel=kernel,
        shape=shape,
        dtype=dtype,
    )
    return _optimize_routed(
        routed,
        output_dir=output_dir,
        cycles=cycles,
        dry_run=dry_run,
        target_hw=target_hw,
    )


def optimize_file(
    input_path: str | Path,
    *,
    output_dir: str | Path = "benchmarks/results/phase1/stage8/track1/optimize",
    cycles: int = 3,
    dry_run: bool = True,
    target_hw: str = "nvidia_ampere",
) -> OptimizeResult:
    """Run the S8 MVP optimization flow for a kernel-only ``.ak`` file."""
    return optimize(
        input_path,
        output_dir=output_dir,
        cycles=cycles,
        dry_run=dry_run,
        target_hw=target_hw,
    )


def _optimize_routed(
    routed: OptimizeInput,
    *,
    output_dir: str | Path = "benchmarks/results/phase1/stage8/track1/optimize",
    cycles: int = 3,
    dry_run: bool = True,
    target_hw: str = "nvidia_ampere",
) -> OptimizeResult:
    """Run the S8 MVP optimization loop for a normalized input payload."""
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    normalized_source_path = run_dir / "input.ak"
    normalized_source_path.write_text(routed.source, encoding="utf-8")
    source_text_path: Path | None = None
    if routed.source_text is not None:
        source_text_path = run_dir / "input.txt"
        source_text_path.write_text(routed.source_text, encoding="utf-8")

    pipeline = ArkePipeline()
    compile_result = pipeline.compile_string(routed.source)
    errors = list(compile_result.errors)
    warnings: list[str] = []

    kernel_id = compile_result.kernel_name or routed.kernel_id_hint or Path(routed.display_path).stem
    trajectory_path = run_dir / "trajectory.jsonl"
    strategy_path = run_dir / "strategy.json"
    akir_path = run_dir / "result.akir"
    summary_path = run_dir / "summary.json"

    if not compile_result.success or compile_result.semantic_ir is None:
        summary = _summary_dict(
            success=False,
            kernel_id=kernel_id,
            input_path=routed.display_path,
            input_kind=routed.kind,
            normalized_source_path=normalized_source_path,
            source_text_path=source_text_path,
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
        # Trajectory v1.0 (D8-F3): header → stream events → adjust marker.
        # The header carries the legacy `schema` pin plus the locked
        # `trajectory_version` / `contract_id` via build_header_data().
        writer.write_header({
            "kernel_id": kernel_id,
            "input_path": routed.display_path,
            "input_kind": routed.kind,
            "normalized_source_path": str(normalized_source_path),
            "source_text_path": str(source_text_path) if source_text_path is not None else None,
            "mode": "dry-run" if dry_run else "compile",
            "target_hw": target_hw,
            "required_cycle_order": ["compile", "profile", "adjust"],
            "semantic_ir": {
                "kernel_id": compile_result.semantic_ir.kernel_id,
                "node_count": len(compile_result.semantic_ir.nodes),
                "param_count": len(compile_result.semantic_ir.params),
                "symbolic_dims": [d.name for d in compile_result.semantic_ir.symbolic_dims],
            },
        })

        for cycle in range(1, cycles + 1):
            compile_event = _compile_cycle(
                compile_result,
                strategy,
                dry_run=dry_run,
            )
            # Emit a single `compile` event per cycle carrying the
            # producer-side build outcome (D8-F2 stream kind).
            writer.write_compile({
                "backend": "mock" if dry_run else "triton",
                "success": bool(compile_event["success"]),
                "cycle": cycle,
                "decision_count": len(strategy.decisions),
                "dry_run": dry_run,
                **{k: v for k, v in compile_event.items() if k != "success"},
            })
            if not compile_event["success"]:
                errors.extend(compile_event.get("errors", []))
                break

            profile = _mock_profile(strategy, cycle=cycle)
            # Map mock profile fields onto the D8-F2 `profile` kind:
            # latency_ms + vs_baseline are the required pair, score is
            # carried as vs_baseline so downstream best-score tracking
            # remains a single-field read.
            writer.write_profile({
                "latency_ms": float(profile.get("latency_ms", 0.0)),
                "vs_baseline": float(profile.get("score", 0.0)),
                "baseline_name": "mock",
                "bottleneck": profile.get("bottleneck", ""),
                "cycle": cycle,
                "source": "mock",
            })
            best_score = max(best_score or 0.0, profile["score"])

            before = len(strategy.decisions)
            generator.refine(strategy, cycle=cycle, profile=profile)
            # Record-only `adjust` marks the cycle boundary.
            writer.write_adjust({
                "cycle": cycle,
                "decisions_before": before,
                "decisions_after": len(strategy.decisions),
                "changed": len(strategy.decisions) != before,
                "bottleneck": profile.get("bottleneck", ""),
            })
            cycles_completed = cycle

        # Terminal `done` event closes the trajectory deterministically.
        writer.write_done({
            "final_score": float(best_score or 0.0),
            "decisions": len(strategy.decisions),
            "compiles": cycles_completed,
            "termination": (
                "hard_error" if errors else "llm_no_more_tool_use"
            ),
            "chosen": "heuristic_floor",
        })

    strategy.to_file(str(strategy_path))
    _save_optimized_akir(compile_result, strategy, akir_path)

    summary = _summary_dict(
        success=cycles_completed == cycles and not errors,
        kernel_id=kernel_id,
        input_path=routed.display_path,
        input_kind=routed.kind,
        normalized_source_path=normalized_source_path,
        source_text_path=source_text_path,
        run_dir=run_dir,
        cycles_completed=cycles_completed,
        decision_count=len(strategy.decisions),
        best_score=best_score,
        errors=errors,
        warnings=warnings,
    )
    _write_json(summary_path, summary)
    return OptimizeResult(**summary)


def _looks_like_ak_source(text: str) -> bool:
    return "kernel" in text and "Tensor" in text and "{" in text and "}" in text


def _looks_like_code_source(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["def ", "torch.", "triton", "tl.", "@triton"])


def _normalize_op(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _OP_ALIASES:
        return _OP_ALIASES[key]
    if key in set(_OP_ALIASES.values()):
        return key
    raise ValueError(f"Unsupported optimize kernel/op: {name!r}")


def _parse_shape(shape: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape, str):
        values = [int(part) for part in re.findall(r"\d+", shape)]
    else:
        values = [int(part) for part in shape]
    if not values:
        raise ValueError(f"Could not parse shape from {shape!r}")
    return tuple(values)


def _parse_natural_language_spec(text: str, *, dtype: str) -> _StructuredOptimizeSpec:
    lowered = text.lower()
    op = _detect_op(lowered)
    detected_dtype = _detect_dtype(lowered) or dtype
    shape = _parse_shape_from_text(lowered, op=op)
    return _StructuredOptimizeSpec(op=op, shape=shape, dtype=detected_dtype)


def _parse_code_spec(code: str, *, dtype: str) -> _StructuredOptimizeSpec:
    lowered = code.lower()
    op = _detect_op(lowered)
    detected_dtype = _detect_dtype(lowered) or dtype
    shape = _parse_shape_from_text(lowered, op=op)
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    kernel = match.group(1) if match else None
    return _StructuredOptimizeSpec(op=op, shape=shape, dtype=detected_dtype, kernel=kernel)


def _detect_op(text: str) -> str:
    if re.search(r"matmul\s*\([^)]*\)\s*(?:\)|\s*)\+?\s*gelu|gelu\s*\([^)]*matmul", text):
        return "matmul_gelu"
    ordered = [
        ("matmul_gelu", ["matmul_gelu", "linear_gelu", "gemm_gelu"]),
        ("softmax", ["softmax"]),
        ("reduce_sum", ["reduce_sum", "sum(", " sum "]),
        ("reduce_max", ["reduce_max", "max(", " max "]),
        ("reduce_mean", ["reduce_mean", "mean(", " mean "]),
        ("matmul", ["matmul", "matrix multiply", "matrix multiplication", "torch.mm", "@"]),
        ("gelu", ["gelu"]),
        ("relu", ["relu"]),
    ]
    for op, needles in ordered:
        if any(needle in text for needle in needles):
            return op
    raise ValueError("Could not infer optimize input op; provide --kernel and --shape")


def _detect_dtype(text: str) -> str | None:
    for dtype in ["bf16", "f16", "f32", "f64", "i8", "i32", "i64"]:
        if dtype in text:
            return dtype
    aliases = {
        "float16": "f16",
        "fp16": "f16",
        "half": "f16",
        "float32": "f32",
        "fp32": "f32",
        "bfloat16": "bf16",
    }
    for alias, dtype in aliases.items():
        if alias in text:
            return dtype
    return None


def _parse_shape_from_text(text: str, *, op: str) -> tuple[int, ...]:
    labeled = _parse_labeled_dims(text)
    if op in {"matmul", "matmul_gelu"}:
        if all(key in labeled for key in ["m", "n", "k"]):
            return (labeled["m"], labeled["n"], labeled["k"])
        values = _all_ints(text)
        if len(values) >= 3:
            return tuple(values[:3])
        if len(values) == 2:
            return (values[0], values[1], values[1])
        raise ValueError("Matmul-like optimize input requires M,N,K or three dimensions")

    if op in {"softmax", "reduce_sum", "reduce_max", "reduce_mean"}:
        values = _all_ints(text)
        if len(values) >= 2:
            return tuple(values[:2])
        if len(values) == 1:
            return (values[0], values[0])
        raise ValueError(f"{op} optimize input requires at least one dimension")

    values = _all_ints(text)
    if len(values) >= 2:
        return tuple(values[:2])
    if len(values) == 1:
        return (values[0],)
    raise ValueError(f"{op} optimize input requires a tensor shape")


def _parse_labeled_dims(text: str) -> dict[str, int]:
    dims: dict[str, int] = {}
    for key, value in re.findall(r"\b([mnkbdsh])\s*=\s*(\d+)", text):
        dims[key] = int(value)
    return dims


def _all_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", text)]


def _ak_source_from_spec(spec: _StructuredOptimizeSpec) -> str:
    dtype = spec.dtype
    if spec.op == "matmul":
        m, n, k = _shape3(spec.shape, default_k_from_n=True)
        return _render_kernel(
            name=spec.kernel or "matmul_kernel",
            params=[("A", [m, k], dtype), ("B", [k, n], dtype)],
            output_shape=[m, n],
            body=["let C = matmul(A=A, B=B);", "return C;"],
        )
    if spec.op == "matmul_gelu":
        m, n, k = _shape3(spec.shape, default_k_from_n=True)
        return _render_kernel(
            name=spec.kernel or "matmul_gelu_kernel",
            params=[("A", [m, k], dtype), ("B", [k, n], dtype)],
            output_shape=[m, n],
            body=["let Z = matmul(A=A, B=B);", "let Y = gelu(X=Z);", "return Y;"],
        )
    if spec.op == "softmax":
        shape = _shape_at_least_2(spec.shape)
        return _render_kernel(
            name=spec.kernel or "softmax_kernel",
            params=[("X", list(shape), dtype)],
            output_shape=list(shape),
            body=["let Y = softmax(X=X, axis=-1);", "return Y;"],
        )
    if spec.op in {"reduce_sum", "reduce_max", "reduce_mean"}:
        shape = _shape_at_least_2(spec.shape)
        return _render_kernel(
            name=spec.kernel or f"{spec.op}_kernel",
            params=[("X", list(shape), dtype)],
            output_shape=list(shape[:-1]),
            body=[f"let Y = {spec.op}(X=X, axis=-1);", "return Y;"],
        )
    if spec.op in {"relu", "gelu"}:
        shape = spec.shape
        return _render_kernel(
            name=spec.kernel or f"{spec.op}_kernel",
            params=[("X", list(shape), dtype)],
            output_shape=list(shape),
            body=[f"let Y = {spec.op}(X=X);", "return Y;"],
        )
    raise ValueError(f"Unsupported optimize op: {spec.op!r}")


def _shape3(shape: tuple[int, ...], *, default_k_from_n: bool) -> tuple[int, int, int]:
    if len(shape) >= 3:
        return shape[0], shape[1], shape[2]
    if len(shape) == 2 and default_k_from_n:
        return shape[0], shape[1], shape[1]
    raise ValueError("Expected at least two dimensions")


def _shape_at_least_2(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) >= 2:
        return shape
    if len(shape) == 1:
        return (shape[0], shape[0])
    raise ValueError("Expected at least one dimension")


def _render_kernel(
    *,
    name: str,
    params: list[tuple[str, list[int], str]],
    output_shape: list[int],
    body: list[str],
) -> str:
    param_lines = []
    for param_name, shape, dtype in params:
        shape_text = ", ".join(str(dim) for dim in shape)
        param_lines.append(f"    {param_name}: Tensor<[{shape_text}], {dtype}>")
    joined_params = ",\n".join(param_lines)
    output_text = ", ".join(str(dim) for dim in output_shape)
    body_text = "\n".join(f"    {line}" for line in body)
    return (
        f"kernel {name}(\n"
        f"{joined_params}\n"
        f") -> Tensor<[{output_text}], {params[0][2]}>\n"
        "{\n"
        f"{body_text}\n"
        "}\n"
    )


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
    input_path: str,
    input_kind: str,
    normalized_source_path: Path | None,
    source_text_path: Path | None,
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
        "input_path": input_path,
        "input_kind": input_kind,
        "normalized_source_path": str(normalized_source_path) if normalized_source_path is not None else None,
        "source_text_path": str(source_text_path) if source_text_path is not None else None,
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
