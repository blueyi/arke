# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Agent — Tool Declarative Interface (S6 Track 4, Agent-G6-M2).

Defines ToolMeta + ArkeTool ABC for self-declaring tool capabilities.
The orchestrator uses these declarations for concurrent batching,
budget tracking, and serial/parallel execution decisions.

Design ref: docs/architecture/arke-harness.md §6 (Tools — declarative ToolMeta)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BudgetType(Enum):
    """Tool budget classification for orchestrator cost tracking."""
    FREE = "free"          # No resource cost
    DECISION = "decision"  # Strategy mutation cost
    COMPILE = "compile"    # GPU compilation + execution cost


class CostLevel(Enum):
    """Approximate execution cost."""
    CHEAP = "cheap"        # <10ms, read-only
    MEDIUM = "medium"      # ~100ms-1s
    EXPENSIVE = "expensive"  # 1-5s+ (GPU compile/profile)


@dataclass(frozen=True)
class ToolMeta:
    """Declarative metadata for an Arke agent tool.

    The orchestrator reads these fields to decide:
    - concurrent_safe tools can batch with asyncio.gather
    - mutates_strategy tools break batches and execute serially
    - budget_type tracks resource usage per session
    - idempotent tools can be retried safely on failure
    """
    concurrent_safe: bool = True
    idempotent: bool = True
    requires_compile: bool = False
    mutates_strategy: bool = False
    budget_type: BudgetType = BudgetType.FREE
    cost: CostLevel = CostLevel.CHEAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrent_safe": self.concurrent_safe,
            "idempotent": self.idempotent,
            "requires_compile": self.requires_compile,
            "mutates_strategy": self.mutates_strategy,
            "budget_type": self.budget_type.value,
            "cost": self.cost.value,
        }


@dataclass
class ToolResult:
    """Structured result from tool execution.

    All tools return ToolResult for consistent JSON serialization.
    """
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self, indent: int | None = None) -> str:
        d = {"success": self.success}
        if self.data:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.warnings:
            d["warnings"] = self.warnings
        return json.dumps(d, indent=indent, default=str)


class ArkeTool(ABC):
    """Abstract base class for Arke agent tools.

    Subclasses must define:
    - name: tool identifier (matches function-calling schema)
    - description: human-readable description
    - meta: ToolMeta with capability declarations
    - execute(): actual tool logic
    - parameters_schema(): JSON Schema for parameters
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g., 'compile_and_profile')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM function-calling."""
        ...

    @property
    @abstractmethod
    def meta(self) -> ToolMeta:
        """Declarative metadata for orchestrator."""
        ...

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            params: Validated parameters matching parameters_schema()

        Returns:
            ToolResult with structured output
        """
        ...

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters.

        Used for OpenAI/Anthropic function-calling schema generation.
        """
        ...

    def to_function_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


# ── Built-in Tools ────────────────────────────────────────────

class GetHWProfileTool(ArkeTool):
    """Read hardware profile — cheap, concurrent-safe, idempotent."""

    @property
    def name(self) -> str:
        return "get_hw_profile"

    @property
    def description(self) -> str:
        return "Get the target hardware profile (compute capability, VRAM, SM count, etc.)"

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=True, idempotent=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        from arke.compiler.passes.base import HardwareProfile
        hw = HardwareProfile()
        return ToolResult(success=True, data={
            "name": hw.name,
            "compute_capability": list(hw.compute_capability),
            "shared_memory_bytes": hw.shared_memory_bytes,
            "max_threads_per_block": hw.max_threads_per_block,
            "warp_size": hw.warp_size,
            "num_sms": hw.num_sms,
            "peak_tflops_f16": hw.peak_tflops_f16,
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


class AnalyzeComputeTool(ArkeTool):
    """Analyze operator compute characteristics."""

    @property
    def name(self) -> str:
        return "analyze_compute"

    @property
    def description(self) -> str:
        return "Analyze an operator's compute pattern, shape rule, and template hints"

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=True, idempotent=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        from arke.ir.ops.registry import REGISTRY

        op_name = params.get("op_name", "")
        if op_name not in REGISTRY:
            return ToolResult(success=False, error=f"Unknown op: {op_name!r}")

        op = REGISTRY.get(op_name)
        data = {
            "name": op.name,
            "category": op.category,
            "inputs": op.inputs,
            "output": op.output,
            "computation": op.computation,
            "properties": op.properties,
            "index_vars": op.index_vars,
            "reduction_axes": op.reduction_axes,
        }

        if op.shape_rule:
            data["shape_rule"] = {"kind": op.shape_rule.kind}
        if op.template_hint:
            data["template_hint"] = {"template_name": op.template_hint.template_name}

        return ToolResult(success=True, data=data)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op_name": {"type": "string", "description": "Operator name"},
            },
            "required": ["op_name"],
        }


class BenchmarkAdviceSummaryTool(ArkeTool):
    """Summarize benchmark artifact rows into agent-consumable guidance."""

    @property
    def name(self) -> str:
        return "benchmark_advice_summary"

    @property
    def description(self) -> str:
        return "Summarize benchmark CSV rows into structured advice for agent planning and Stage 7 triage"

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=True, idempotent=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        import csv
        from benchmarks.advice import build_agent_advice_summary

        csv_path = Path(params.get("csv_path", ""))
        gpu_memory_mb = int(params.get("gpu_memory_mb", 0))
        if not csv_path.exists():
            return ToolResult(success=False, error=f"CSV not found: {csv_path}")
        rows = list(csv.DictReader(csv_path.open()))
        summary = build_agent_advice_summary(rows, gpu_memory_mb=gpu_memory_mb)
        return ToolResult(success=True, data=summary)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "csv_path": {"type": "string", "description": "Path to PERF_ALL.csv or perf_*.csv"},
                "gpu_memory_mb": {"type": "integer", "description": "GPU memory in MB for context"},
            },
            "required": ["csv_path", "gpu_memory_mb"],
        }


class CompileAndProfileTool(ArkeTool):
    """Compile and profile a kernel — expensive, serial.

    Optionally env-aware (P5-S5 Step 5a): when constructed with an ArkeEnv
    (as done by ``ToolRegistry.with_env``), the accumulated decision_log is
    passed as the strategy to strategy-aware backends (cuda_c / llvm) so
    Agent decisions — including L3 instruction-level kinds like wmma_tile /
    block_threads / pipeline_stages — actually configure the generated
    kernel. Constructed without an env (``ToolRegistry.default()``),
    behavior is identical to before: lower with no strategy. The Facade
    v1.0 schema is unchanged (env binding is a construction detail, not a
    parameter).
    """

    def __init__(self, env: Any = None) -> None:
        self._env = env
        # Default-baseline latency cache: (op_name, shapes_key, backend_label)
        # -> median default (strategy=None) latency in ms, measured with the
        # same strict discipline as the agent kernel (P5-S5-T fidelity fix).
        self._default_cache: dict[tuple, float] = {}

    @property
    def name(self) -> str:
        return "compile_and_profile"

    @property
    def description(self) -> str:
        return (
            "Compile an operator from IR graph through the pass pipeline "
            "and MockBackend, then validate correctness against reference implementation"
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=False, idempotent=True,
            requires_compile=True, mutates_strategy=False,
            budget_type=BudgetType.COMPILE, cost=CostLevel.EXPENSIVE,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        import torch
        from arke.compiler.passes import (
            PassPipeline, SSAValidationPass, ShapeInferencePass,
        )
        from arke.ir.graph import IRGraph
        from arke.ir.ops.interpreter import INTERPRETER
        from arke.ir.ops.registry import REGISTRY

        op_name = params.get("op_name", "")
        if op_name not in REGISTRY:
            return ToolResult(success=False, error=f"Unknown op: {op_name!r}")

        op = REGISTRY.get(op_name)

        # Backend selection (D8-F1.3 / P0-B): use the real TritonBackend on
        # CUDA so the agent measures real kernels; fall back to MockBackend
        # only when CUDA is unavailable (CPU CI). The chosen backend is
        # reported in the result so the trajectory records measurement
        # provenance honestly.
        # Optional backend override (non-breaking additive param): the caller
        # may request a specific backend via params["backend"] ∈ {"triton",
        # "cuda_c", "llvm"}. Defaults to "triton" on CUDA (unchanged behavior),
        # MockBackend on CPU. This lets the Agent drive the Phase-4 CUDA-C and
        # Phase-5 LLVM backends through the same Façade tool (StrategyIR →
        # backend consumption path, including L3 instruction-level decisions).
        use_cuda = torch.cuda.is_available() and params.get("force_mock") is not True
        requested_backend = params.get("backend", "triton")
        # P5-S5 auto-routing guardrail: L3 instruction-level decisions are
        # consumed only by the llvm backend. If the bound strategy carries
        # L3 decisions and the caller did NOT explicitly choose a backend,
        # route to llvm automatically — the tool owns the decision→backend
        # activation binding, rather than relying on the agent to remember
        # it (live runs showed prompt-only binding is unreliable). An
        # explicit `backend` param always wins.
        auto_routed = False
        if "backend" not in params and use_cuda and self._env is not None:
            try:
                has_l3 = any(
                    getattr(d, "level", 1) == 3
                    for d in self._env.state.decision_log
                )
            except Exception:
                has_l3 = False
            if has_l3:
                requested_backend = "llvm"
                auto_routed = True
        if use_cuda and requested_backend == "cuda_c":
            from arke.backend.cuda_c_backend import CudaCBackend
            backend = CudaCBackend(chip="sm_86")
            backend_label = "cuda_c"
        elif use_cuda and requested_backend == "llvm":
            from arke.backend.llvm_backend import LLVMBackend
            backend = LLVMBackend(chip="sm_86")
            backend_label = "llvm"
        elif use_cuda:
            from arke.backend.triton_backend import TritonBackend
            backend = TritonBackend(device="cuda")
            backend_label = "triton"
        else:
            from arke.backend.mock_backend import MockBackend
            backend = MockBackend()
            backend_label = "mock"

        # Build a single-node graph (constructed below via
        # IRGraph.single_node once shapes are resolved).
        shapes = params.get("shapes", {})

        # Default shapes
        default_shapes = {k: [4, 8] for k in op.inputs}
        if op_name == "matmul":
            default_shapes = {"A": [64, 32], "B": [32, 128]}
        elif op_name == "batch_matmul":
            default_shapes = {"A": [4, 16, 32], "B": [4, 32, 64]}
        elif op_name in ("flash_attention", "grouped_query_attention", "cross_attention", "multi_latent_attention"):
            default_shapes = {"Q": [1, 2, 16, 32], "K": [1, 2, 16, 32], "V": [1, 2, 16, 32]}
        elif op_name == "layernorm":
            default_shapes = {"X": [4, 768], "W": [768], "B": [768]}
        elif op_name == "rmsnorm":
            default_shapes = {"X": [4, 128], "W": [128]}
        elif op_name == "rmsnorm_residual":
            default_shapes = {"X": [4, 64], "residual": [4, 64], "W": [64]}
        elif op_name == "embedding":
            default_shapes = {"indices": [2, 8], "weight": [100, 32]}
        elif op_name == "cross_entropy":
            default_shapes = {"logits": [8, 100], "labels": [8]}
        elif op_name == "fused_linear_cross_entropy":
            default_shapes = {"X": [8, 64], "W": [100, 64], "labels": [8]}
        elif op_name == "grouped_matmul":
            default_shapes = {"X": [4, 16, 32], "W": [8, 32, 64]}

        merged_shapes = {**default_shapes, **shapes}

        # Official single-node construction (K-H1): route through
        # IRGraph.single_node → SemanticIR → from_semantic. Keyed by the op's
        # schema input names so from_semantic maps them by identity.
        node_shapes = {
            inp_name: merged_shapes.get(inp_name, [4, 8])
            for inp_name in op.inputs
        }
        graph = IRGraph.single_node(op_name, node_shapes, output_name="output",
                                    name=f"profile_{op_name}")

        # Run pipeline
        pipeline = PassPipeline("compile_and_profile")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        result = pipeline.run(graph)

        if not result.success:
            return ToolResult(success=False, error=f"Pipeline failed: {result.error}")

        # Generate inputs for execution. On CUDA we place tensors on the
        # device with the op's natural dtype (f16 for matmul/attention) so
        # the real Triton kernel runs; on CPU/mock we keep f32 on host.
        torch.manual_seed(42)
        dev = "cuda" if use_cuda else "cpu"
        # f16 is the Phase-1 perf dtype for dense/attention ops; norms etc.
        # tolerate f16 too. Reductions over indices stay integer.
        compute_dtype = torch.float16 if use_cuda else torch.float32
        inputs = {}
        for inp_name in op.inputs:
            shape = merged_shapes.get(inp_name, [4, 8])
            if op.input_gen and inp_name in op.input_gen.distributions:
                dist = op.input_gen.distributions[inp_name]
                if dist == "randint":
                    rng = op.input_gen.ranges.get(inp_name, (0, 10))
                    inputs[inp_name] = torch.randint(int(rng[0]), int(rng[1]) + 1, shape, device=dev)
                elif dist == "uniform":
                    rng = op.input_gen.ranges.get(inp_name, (0, 1))
                    inputs[inp_name] = torch.empty(shape, device=dev, dtype=compute_dtype).uniform_(rng[0], rng[1])
                elif dist == "ones":
                    inputs[inp_name] = torch.ones(shape, device=dev, dtype=compute_dtype)
                elif dist == "bool_mask":
                    inputs[inp_name] = torch.randint(0, 2, shape, dtype=torch.bool, device=dev)
                else:
                    inputs[inp_name] = torch.randn(shape, device=dev, dtype=compute_dtype)
            else:
                inputs[inp_name] = torch.randn(shape, device=dev, dtype=compute_dtype)

        # Lower → compile → run through the selected backend.
        # Strategy injection (P5-S5 Step 5a): strategy-aware backends
        # (cuda_c / llvm — both accept lower(graph, strategy=...)) receive
        # the env's accumulated decision_log so applied decisions (L1 tile
        # … L3 wmma_tile/block_threads/pipeline_stages) configure codegen.
        # The decision_log is passed (not state.strategy) because ScheduleIR
        # ignores unknown kinds while extract_l3_params / MatmulConfig read
        # bare decision lists directly. Triton path unchanged (its lower()
        # does not take a strategy).
        strategy_decisions = None
        if self._env is not None:
            try:
                decision_log = self._env.state.decision_log
                if decision_log:
                    strategy_decisions = list(decision_log)
            except Exception:
                strategy_decisions = None
        try:
            if backend_label in ("cuda_c", "llvm") and strategy_decisions:
                artifact = backend.lower(result.graph, strategy=strategy_decisions)
            else:
                artifact = backend.lower(result.graph)
            compiled = backend.compile(artifact)
            if not getattr(compiled, "success", True):
                return ToolResult(success=False, error=f"Compile failed: {getattr(compiled, 'error', '?')}")
            outputs = backend.run(compiled, inputs)
        except Exception as e:
            return ToolResult(success=False, error=f"Backend {backend_label} execution failed: {e}")

        # Normalize output handle (graph output name may vary).
        output_tensor = None
        if isinstance(outputs, dict):
            output_tensor = outputs.get("output")
            if output_tensor is None and outputs:
                output_tensor = next(iter(outputs.values()))

        # CUDA-C backend returns numpy arrays; normalize to torch tensors so the
        # downstream V1/V2 code (which uses torch methods) works uniformly.
        import numpy as _np
        if isinstance(output_tensor, _np.ndarray):
            output_tensor = torch.from_numpy(output_tensor).to(dev)

        # V1: validate against the reference interpreter (fp64 CPU escape to
        # avoid the FlagGems aten::mm hijack when comparing on GPU).
        try:
            ref_result = INTERPRETER.execute(op_name, {k: v.float().cpu() if v.is_floating_point() else v.cpu()
                                                       for k, v in inputs.items()})
            if output_tensor is not None and ref_result is not None:
                cand = output_tensor.float().cpu()
                ref = ref_result.float().cpu() if ref_result.is_floating_point() else ref_result.cpu()
                if ref.is_floating_point():
                    # f16 compute → looser tolerance
                    rtol, atol = (1e-2, 1e-2) if use_cuda else (1e-3, 1e-5)
                    correct = bool(torch.allclose(cand, ref, rtol=rtol, atol=atol))
                    max_diff = (cand - ref).abs().max().item()
                else:
                    correct = bool(torch.equal(cand.long(), ref.long()))
                    max_diff = 0.0
            else:
                correct = True
                max_diff = 0.0
        except Exception as e:
            correct = None
            max_diff = None
            validation_note = f"reference validation skipped: {e}"
        else:
            validation_note = None

        # V2: real GPU profiling (only meaningful on CUDA with a real kernel).
        #
        # P5-S5-T measurement-fidelity fix: on WSL the GPU downclocks after a
        # few seconds idle (LLM thinking pauses), and a single benchmark()
        # call with warmup=25 finishes before the clocks ramp back up —
        # measured 165-195us for a 9us rmsnorm kernel after 10-30s idle.
        # Discipline (mirrors benchmarks/l3_sweep.py): (1) ~150ms busy-loop
        # clock ramp, (2) median-of-3 interleaved passes, (3) kernel-only
        # CUDA events via prepare/run_fast/benchmark_cached (llvm/cuda_c),
        # (4) spread recorded so the agent can judge measurement quality.
        # Also measures the strategy=None DEFAULT kernel interleaved in the
        # same passes and reports vs_default — the gate's actual criterion.
        latency_ms = None
        baseline_ratio = None
        default_latency_ms = None
        vs_default = None
        meas_spread = None
        strategy_noop = None
        if use_cuda and output_tensor is not None:
            try:
                from benchmarks.measure import bench_fn
                fine_grained = (
                    backend_label in ("cuda_c", "llvm")
                    and callable(getattr(backend, "prepare", None))
                    and callable(getattr(backend, "benchmark_cached", None))
                )
                if fine_grained:
                    import statistics as _stats
                    import time as _time
                    agent_cached = backend.prepare(compiled)
                    default_cached = None
                    try:
                        backend.run_fast(agent_cached, inputs)
                        # Default (strategy=None) kernel for the SAME graph,
                        # interleaved in the same thermal window. Only compile
                        # it when the agent kernel differs from default.
                        if strategy_decisions:
                            try:
                                default_artifact = backend.lower(result.graph)
                                default_compiled = backend.compile(default_artifact)
                                if getattr(default_compiled, "success", True):
                                    # No-op detection: if the agent's decisions
                                    # produced a byte-identical cubin (e.g.
                                    # block_threads(512) when 512 IS the
                                    # default, or L1 kinds the rowwise emitter
                                    # doesn't consume), the strategy changed
                                    # nothing — tell the agent instead of
                                    # letting it chase noise between two
                                    # measurements of the same kernel.
                                    try:
                                        agent_cubin = compiled.metadata.get("cubin")
                                        default_cubin = default_compiled.metadata.get("cubin")
                                        if agent_cubin and default_cubin:
                                            strategy_noop = bool(agent_cubin == default_cubin)
                                    except Exception:
                                        strategy_noop = None
                                    if not strategy_noop:
                                        default_cached = backend.prepare(default_compiled)
                                        backend.run_fast(default_cached, inputs)
                            except Exception:
                                default_cached = None
                        # Clock ramp: ~150ms of back-to-back launches so the
                        # idle-downclocked GPU returns to a high-clock state
                        # before any recorded measurement.
                        t0 = _time.perf_counter()
                        while (_time.perf_counter() - t0) < 0.15:
                            backend.benchmark_cached(agent_cached, iters=200, warmup=0)
                        # Resolution probe: small kernels (<50us) are dominated
                        # by launch jitter — use more iters and passes so a
                        # 10-15% config difference is resolvable.
                        probe = backend.benchmark_cached(agent_cached, iters=50, warmup=10)
                        small = probe < 0.05  # ms
                        n_iters = 300 if small else 100
                        n_passes = 5 if small else 3
                        agent_passes, default_passes = [], []
                        for p in range(n_passes):
                            # Alternate measurement order (D,A / A,D / ...) so
                            # monotone clock ramp-up within the window doesn't
                            # systematically favor whichever runs second.
                            if default_cached is None:
                                agent_passes.append(
                                    backend.benchmark_cached(agent_cached, iters=n_iters, warmup=10))
                            elif p % 2 == 0:
                                default_passes.append(
                                    backend.benchmark_cached(default_cached, iters=n_iters, warmup=10))
                                agent_passes.append(
                                    backend.benchmark_cached(agent_cached, iters=n_iters, warmup=10))
                            else:
                                agent_passes.append(
                                    backend.benchmark_cached(agent_cached, iters=n_iters, warmup=10))
                                default_passes.append(
                                    backend.benchmark_cached(default_cached, iters=n_iters, warmup=10))
                        latency_ms = round(float(_stats.median(agent_passes)), 6)
                        if min(agent_passes) > 0:
                            meas_spread = round(max(agent_passes) / min(agent_passes) - 1.0, 4)
                        if default_passes:
                            default_latency_ms = round(float(_stats.median(default_passes)), 6)
                            # Pairwise per-pass ratio median: each pass measures
                            # default and agent back-to-back, so their ratio
                            # cancels slow clock/thermal drift much better than
                            # the ratio of two medians taken across the window.
                            pair_ratios = [a / d for a, d in zip(agent_passes, default_passes) if d > 0]
                            if pair_ratios:
                                vs_default = round(float(_stats.median(pair_ratios)), 4)
                        else:
                            # Agent kernel IS the default (no decisions applied).
                            default_latency_ms = latency_ms
                            vs_default = 1.0
                    finally:
                        backend.release(agent_cached)
                        if default_cached is not None:
                            backend.release(default_cached)
                    cache_key = (op_name, json.dumps(merged_shapes, sort_keys=True), backend_label)
                    if default_latency_ms is not None:
                        self._default_cache[cache_key] = default_latency_ms
                    if vs_default is None and default_latency_ms and latency_ms:
                        vs_default = round(latency_ms / default_latency_ms, 4)
                else:
                    bench_method = getattr(backend, "benchmark", None)
                    if backend_label == "cuda_c" and callable(bench_method):
                        # CudaCBackend has no fine-grained prepare/benchmark_cached
                        # API; its benchmark() is still kernel-only CUDA events.
                        # Apply the same discipline coarsely: one throwaway ramp
                        # call, then median-of-3, default interleaved.
                        import statistics as _stats
                        bench_method(compiled, inputs, iters=100, warmup=30)  # ramp
                        default_compiled = None
                        if strategy_decisions:
                            try:
                                _dc = backend.compile(backend.lower(result.graph))
                                if getattr(_dc, "success", True):
                                    default_compiled = _dc
                            except Exception:
                                default_compiled = None
                        agent_passes, default_passes = [], []
                        for _ in range(3):
                            if default_compiled is not None:
                                default_passes.append(float(bench_method(
                                    default_compiled, inputs, iters=100, warmup=10)))
                            agent_passes.append(float(bench_method(
                                compiled, inputs, iters=100, warmup=10)))
                        latency_ms = round(_stats.median(agent_passes), 6)
                        if min(agent_passes) > 0:
                            meas_spread = round(max(agent_passes) / min(agent_passes) - 1.0, 4)
                        default_latency_ms = (round(_stats.median(default_passes), 6)
                                              if default_passes else latency_ms)
                        if default_latency_ms and latency_ms:
                            vs_default = round(latency_ms / default_latency_ms, 4)
                    else:
                        arke_fn = lambda: backend.run(compiled, inputs)  # noqa: E731
                        arke_bench = bench_fn(arke_fn, warmup=25, reps=100, trials=3)
                        latency_ms = round(arke_bench.latency_us / 1000.0, 6)
                # PyTorch-eager baseline via the interpreter on-device.
                try:
                    base_fn = lambda: INTERPRETER.execute(op_name, inputs)  # noqa: E731
                    base_bench = bench_fn(base_fn, warmup=25, reps=100, trials=3)
                    base_ms = base_bench.latency_us / 1000.0
                    if latency_ms and latency_ms > 0:
                        baseline_ratio = round(base_ms / latency_ms, 4)
                except Exception:
                    baseline_ratio = None
            except Exception as e:
                validation_note = (validation_note or "") + f" | profiling skipped: {e}"

        # D2: discrete robust reward (anti-reward-hacking, CUDA Agent schedule).
        # baseline_ratio here is the eager ratio; strong_ratio is not separately
        # measured in this tool yet (Same-Backend-Fairness denominator lives in
        # the bench harness), so we pass it as the strong ratio too when > 1 to
        # avoid over-rewarding. This is recorded for trajectory/RL consumption.
        from arke.agent.verification import robust_reward as _robust_reward
        reward_tier = int(_robust_reward(
            correct=correct,
            eager_ratio=baseline_ratio,
            strong_ratio=baseline_ratio,
        ))

        data = {
            "op_name": op_name,
            "pipeline_passes": result.passes_run,
            "output_shape": list(output_tensor.shape) if output_tensor is not None else [],
            "correct": correct,
            "max_diff": max_diff,
            "latency_ms": latency_ms,
            "baseline_ratio": baseline_ratio,
            # P5-S5-T fidelity fields: vs_default is the gate criterion
            # (<1.0 = beats the strategy=None default kernel); meas_spread
            # is measurement quality (max/min - 1 over the 3 passes).
            "default_latency_ms": default_latency_ms,
            "vs_default": vs_default,
            "meas_spread": meas_spread,
            # True when the applied decisions produced a byte-identical cubin
            # to the default — the strategy is a NO-OP for this op/backend
            # (any vs_default delta you think you saw was pure noise).
            "strategy_noop": strategy_noop,
            "robust_reward": reward_tier,
            "backend": backend_label,
            "strategy_decisions": len(strategy_decisions) if strategy_decisions else 0,
            "num_real_kernels": artifact.metadata.get("num_real_kernels"),
            "num_fallback": artifact.metadata.get("num_fallback"),
        }
        # Env-bound accounting (P5-S5 Step 5b): record the profile into
        # OptimizationState so it (a) spends compile budget honestly,
        # (b) lands in state.json / trajectory as an auditable V2 record,
        # (c) can promote best_result with a REAL latency. Without this,
        # profiles were invisible to the state (compiles_used stayed 0 and
        # best_result never carried latency — the run2 observability bug).
        if self._env is not None:
            from arke.agent.state import CompileResult as _CR
            profile_record = _CR(
                success=True, backend=backend_label,
                correct=correct, max_diff=max_diff,
                latency_ms=latency_ms, baseline_ratio=baseline_ratio,
                metadata={
                    "validation_tier": "V2_profile",
                    "requested_backend": requested_backend,
                    "strategy_decisions": data["strategy_decisions"],
                    "default_latency_ms": default_latency_ms,
                    "vs_default": vs_default,
                    "meas_spread": meas_spread,
                },
            )
            try:
                self._env.state.record_compile(profile_record)
            except Exception as e:  # budget exhausted → honest error
                return ToolResult(
                    success=False,
                    error=f"compile budget exhausted: {e}",
                )
            budget = self._env.state.budget
            data["compiles_used"] = budget.compiles_used
            data["compiles_remaining"] = budget.compiles_remaining
        if auto_routed:
            data["backend_auto_routed"] = (
                "strategy contains L3 decisions -> auto-routed to llvm "
                "(pass backend explicitly to override)"
            )
        warnings = [validation_note] if validation_note else []
        return ToolResult(success=True, data=data, warnings=warnings)

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op_name": {"type": "string", "description": "Operator name"},
                "shapes": {
                    "type": "object",
                    "description": "Optional shape overrides per input",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
            },
            "required": ["op_name"],
        }


# ── Stateful Façade tools (D8-F1.2) ───────────────────────────
#
# Tools 3/4/5/7/8 from arke-harness.md §6. Unlike the four S6 tools
# above, these read/write OptimizationState — they require a bound
# ArkeEnv at construction. ToolRegistry.with_env(env) wires them up.
#
# Design ref: docs/architecture/arke-harness.md §6
# Stage tracker: docs/phase1/stage8-plan.md D8-F1.2


class _EnvBoundTool(ArkeTool):
    """Base helper for tools that read/write a bound ArkeEnv."""

    def __init__(self, env: Any) -> None:
        # `env` typed loosely to avoid circular import (env.py imports tools? no — but keep loose)
        from arke.agent.env import ArkeEnv  # local import
        if not isinstance(env, ArkeEnv):
            raise TypeError(f"_EnvBoundTool requires ArkeEnv, got {type(env).__name__}")
        self._env = env

    @property
    def env(self) -> Any:
        return self._env


class ListLegalActionsTool(_EnvBoundTool):
    """Enumerate top-N legal next-decisions for the current state."""

    @property
    def name(self) -> str:
        return "list_legal_actions"

    @property
    def description(self) -> str:
        return (
            "List legal next-decisions for the current optimization state. "
            "Returns candidates of kind tile/unroll/vectorize/parallel/place "
            "(filterable via filter_kind). Redundant candidates already in "
            "decision_log are filtered out."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=True, idempotent=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        top_n = int(params.get("top_n", 10))
        filter_kind = params.get("filter_kind")
        try:
            candidates = self._env.list_legal_actions(top_n=top_n, filter_kind=filter_kind)
        except Exception as e:
            return ToolResult(success=False, error=f"list_legal_actions failed: {e}")
        data: dict[str, Any] = {
            "count": len(candidates),
            "candidates": [
                {"kind": d.kind, "params": d.params, "level": d.level}
                for d in candidates
            ],
        }
        # Read side of the @rationale loop (LT-7): surface prior decisions +
        # their MEASURED outcomes for this op so the Agent ranks candidates
        # with accumulated experience, not from scratch. This closes the
        # feedback loop the 390-entry KB was previously write-only for. The
        # candidate generator (legality surface) is untouched; priors are an
        # additive, advisory field on the returned data (NOT the frozen Façade
        # schema/description/meta — contract v1 is preserved). Best-effort:
        # any KB failure leaves list_legal_actions fully functional.
        priors = self._recall_priors(filter_kind=filter_kind)
        if priors:
            data["rationale_priors"] = priors
        return ToolResult(success=True, data=data)

    def _recall_priors(
        self, *, filter_kind: str | None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Best-effort @rationale KB recall for the current op (never raises)."""
        try:
            from arke.learn.rationale_kb import RationaleKB
            op = getattr(self._env, "op_name", None)
            if not op:
                return []
            kb = RationaleKB()
            if kb.count() == 0:
                return []
            recalled = kb.recall(op, decision_kind=filter_kind, top_k=top_k)
            return [
                {
                    "decision_kind": p.decision_kind,
                    "params": p.params,
                    "rationale": p.rationale,
                    "baseline_ratio": p.baseline_ratio,
                    "correct": p.correct,
                }
                for p in recalled
            ]
        except Exception:
            return []

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Max candidates to return", "default": 10},
                "filter_kind": {
                    "type": "string",
                    "description": "Restrict to one decision kind (tile/unroll/vectorize/parallel/place)",
                },
            },
            "required": [],
        }


class ApplyDecisionTool(_EnvBoundTool):
    """Apply a decision: mutates strategy, advances decision budget."""

    @property
    def name(self) -> str:
        return "apply_decision"

    @property
    def description(self) -> str:
        return (
            "Apply an optimization decision to the current strategy. Mutates "
            "OptimizationState (strategy + decision_log) and consumes 1 decision budget unit."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=False, idempotent=False,
            requires_compile=False, mutates_strategy=True,
            budget_type=BudgetType.DECISION, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        from arke.ir.strategy import Decision, Rationale
        kind = params.get("kind")
        d_params = params.get("params", {})
        if not isinstance(kind, str) or not kind:
            return ToolResult(success=False, error="missing required field: kind")
        if not isinstance(d_params, dict):
            return ToolResult(success=False, error="params must be an object")

        rationale = None
        rat_input = params.get("rationale")
        if isinstance(rat_input, str) and rat_input:
            rationale = Rationale(text=rat_input)
        elif isinstance(rat_input, dict) and rat_input.get("text"):
            rationale = Rationale(text=rat_input["text"], lang=rat_input.get("lang", "en"))

        # S4 (2026-06-26): @rationale is an execution-enforced contract.
        # A non-trivial decision (level >= 1 — tile/unroll/vectorize/parallel/
        # place/fuse) MUST carry a non-empty rationale. This closes the soft-
        # contract gap: the schema keeps `rationale` optional (Façade v1.0 is
        # frozen — we do NOT change the required-set), but the tool BEHAVIOR
        # now rejects a non-trivial decision with no WHY. Trivial level-0
        # decisions (if any) are exempt. The locked thesis pillar "@rationale
        # is a contract" is now enforced at the boundary, not just documented.
        #
        # P5-S5: the decision LEVEL is derived from the kind, not trusted
        # from the caller. Agents never pass `level`; defaulting to 1 caused
        # L3 kinds (wmma_tile/...) to be stored as level=1, which made
        # extract_l3_params() silently ignore them — the kernel never saw
        # the agent's decision. Kind is the semantic identity; level is
        # bookkeeping the tool owns. An explicit `level` param still wins.
        from arke.ir.strategy import L3_KINDS
        default_level = 3 if kind in L3_KINDS else 1
        level = int(params.get("level", default_level))
        if level >= 1 and (rationale is None or not rationale.text.strip()):
            return ToolResult(
                success=False,
                error=(
                    "missing required @rationale: a non-trivial decision "
                    f"(kind={kind!r}, level={level}) must include a non-empty "
                    "`rationale` explaining WHY. This is a hard Arke contract — "
                    "re-issue apply_decision with a rationale string."
                ),
            )

        decision = Decision(kind=kind, params=dict(d_params), rationale=rationale, level=level)
        try:
            self._env.state.apply_decision(decision)
        except Exception as e:
            return ToolResult(success=False, error=f"apply_decision failed: {e}")

        budget = self._env.state.budget
        return ToolResult(success=True, data={
            "applied": {"kind": decision.kind, "params": decision.params, "step": decision.step, "level": decision.level},
            "decisions_used": budget.decisions_used,
            "decisions_remaining": budget.decisions_remaining,
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Decision kind (tile/unroll/vectorize/parallel/place/fuse/reorder/compute/algorithm)"},
                "params": {"type": "object", "description": "Kind-specific parameters (e.g. {loop, factors} for tile)"},
                "rationale": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "object", "properties": {"text": {"type": "string"}, "lang": {"type": "string"}}},
                    ],
                    "description": "Optional natural-language justification (required by @rationale convention for non-trivial decisions)",
                },
                "level": {"type": "integer", "description": "1 = L1 backend-agnostic, 2 = L2 backend-specific", "default": 1},
            },
            "required": ["kind", "params"],
        }


class VerifyCorrectnessTool(_EnvBoundTool):
    """V0/V1 numeric correctness check via SemanticInterpreter reference.

    Trial-balloon semantics: if `decision` is provided, applies it on a
    temporary checkpoint, validates, then rolls back. The probe does NOT
    consume decision budget but consumes 1 compile budget unit.

    With no `decision`, validates the *current* strategy state.

    D8-F1.2 phase: candidate backend is not yet wired through this tool
    (that arrives in D8-F1.3 with real Triton compile). For now the
    candidate equals the reference (V0_mock tier), which always returns
    correct=True with max_diff=0. This keeps the pipeline plumbed end to
    end and lets agent code rely on the tool contract today.
    """

    _VERIFY_CHECKPOINT_LABEL = "__verify_tmp__"

    @property
    def name(self) -> str:
        return "verify_correctness"

    @property
    def description(self) -> str:
        return (
            "Numerically verify that the current strategy (optionally after applying "
            "a trial decision) produces outputs matching the operator's reference "
            "implementation. Trial decisions are rolled back automatically."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=False, idempotent=True,
            requires_compile=True, mutates_strategy=False,
            budget_type=BudgetType.COMPILE, cost=CostLevel.MEDIUM,
        )

    def _default_tolerance(self, dtype: Any) -> tuple[float, float]:
        import torch
        if dtype in (torch.float16, torch.bfloat16):
            return 1e-2, 1e-3
        return 1e-3, 1e-5

    def _validate_real(self, seed: int, rtol: float, atol: float) -> tuple[bool, float]:
        """Compile + run the current op on the real TritonBackend (CUDA) and
        compare against the reference interpreter. Returns (correct, max_diff).

        Builds a single-node IRGraph for the env's op, lowers through
        TritonBackend, executes on f16 CUDA tensors, and compares against the
        interpreter run on an fp64 CPU copy of the same inputs (the fp64 CPU
        escape sidesteps the global FlagGems aten::mm hijack). Raises on any
        backend/codegen failure so the caller can record an honest failure.
        """
        import torch
        from arke.backend.triton_backend import TritonBackend
        from arke.compiler.passes import (
            PassPipeline, SSAValidationPass, ShapeInferencePass,
        )
        from arke.ir.graph import IRGraph
        from arke.ir.ops.interpreter import INTERPRETER
        from arke.ir.ops.registry import REGISTRY

        op_name = self._env.op_name
        op = REGISTRY.get(op_name)
        shapes = self._env.op_inputs

        # Official single-node construction (K-H1): IRGraph.single_node routes
        # through SemanticIR → from_semantic, so the input-mapping/dtype logic
        # is not re-derived here. Fill any missing input shapes with the env
        # default, keyed by the op's schema input names.
        merged = {name: shapes.get(name, [4, 8]) for name in op.inputs}
        graph = IRGraph.single_node(op_name, merged, output_name="output",
                                    name=f"verify_{op_name}")

        pipeline = PassPipeline("verify_real")
        pipeline.add_pass(SSAValidationPass())
        pipeline.add_pass(ShapeInferencePass())
        pres = pipeline.run(graph)
        if not pres.success:
            raise RuntimeError(f"pipeline failed: {pres.error}")

        torch.manual_seed(seed)
        inputs = {}
        for inp_name in op.inputs:
            shape = shapes.get(inp_name, [4, 8])
            if op.input_gen and inp_name in op.input_gen.distributions:
                dist = op.input_gen.distributions[inp_name]
                if dist == "randint":
                    rng = op.input_gen.ranges.get(inp_name, (0, 10))
                    inputs[inp_name] = torch.randint(int(rng[0]), int(rng[1]) + 1, shape, device="cuda")
                elif dist == "bool_mask":
                    inputs[inp_name] = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
                else:
                    inputs[inp_name] = torch.randn(shape, device="cuda", dtype=torch.float16)
            else:
                inputs[inp_name] = torch.randn(shape, device="cuda", dtype=torch.float16)

        backend = TritonBackend(device="cuda")
        artifact = backend.lower(pres.graph)
        compiled = backend.compile(artifact)
        if not getattr(compiled, "success", True):
            raise RuntimeError(f"compile failed: {getattr(compiled, 'error', '?')}")
        outputs = backend.run(compiled, inputs)
        out = outputs.get("output") if isinstance(outputs, dict) else None
        if out is None and isinstance(outputs, dict) and outputs:
            out = next(iter(outputs.values()))
        if out is None:
            raise RuntimeError("backend produced no output")

        ref_inputs = {k: (v.float().cpu() if v.is_floating_point() else v.cpu())
                      for k, v in inputs.items()}
        ref = INTERPRETER.execute(op_name, ref_inputs)
        cand = out.float().cpu()
        ref_c = ref.float().cpu() if ref.is_floating_point() else ref.cpu()
        if ref_c.is_floating_point():
            correct = bool(torch.allclose(cand, ref_c, rtol=max(rtol, 1e-2), atol=max(atol, 1e-2)))
            max_diff = float((cand - ref_c).abs().max().item())
        else:
            correct = bool(torch.equal(cand.long(), ref_c.long()))
            max_diff = 0.0
        return correct, max_diff

    def execute(self, params: dict[str, Any]) -> ToolResult:
        from arke.agent.inputs import generate_inputs
        from arke.agent.state import CompileResult
        from arke.ir.ops.interpreter import INTERPRETER
        from arke.ir.strategy import Decision, Rationale

        rtol_override = params.get("rtol")
        atol_override = params.get("atol")
        seed = int(params.get("seed", self._env.seed))

        # Validate trial payload up front (before any side-effect)
        trial = params.get("decision")
        if trial is not None and (not isinstance(trial, dict) or "kind" not in trial):
            return ToolResult(success=False, error="decision must be an object with at least 'kind'")

        # Generate reproducible inputs FIRST (cheapest validation)
        try:
            inputs = generate_inputs(self._env.op_name, self._env.op_inputs, seed=seed)
        except Exception as e:
            return ToolResult(success=False, error=f"input generation failed: {e}")

        # Reference execution
        try:
            ref_output = INTERPRETER.execute(self._env.op_name, inputs)
        except Exception as e:
            return ToolResult(success=False, error=f"reference execution failed: {e}")

        # Tolerance defaults per dtype
        rtol_def, atol_def = self._default_tolerance(ref_output.dtype)
        rtol = float(rtol_override) if rtol_override is not None else rtol_def
        atol = float(atol_override) if atol_override is not None else atol_def

        # ── Ordering rationale (D8-F1.2) ─────────────────────────────────
        # We must run record_compile BEFORE the trial-balloon checkpoint,
        # so that subsequent rollback() restores us back to the
        # post-compile state (compiles_used preserved, decisions reverted).
        # Reversing this order would erase the compile we just recorded.
        # ────────────────────────────────────────────────────────────────

        # ── V1 numeric validation (P0-B / D8-F1.3) ───────────────────────
        # On CUDA: compile + run the op through the real TritonBackend and
        # compare against the reference interpreter (fp64 CPU escape avoids
        # the FlagGems aten::mm hijack). On CPU/no-GPU: fall back to the
        # V0_mock tier (candidate == reference) so CI stays green.
        import torch as _torch
        use_cuda = _torch.cuda.is_available() and params.get("force_mock") is not True

        if use_cuda:
            tier = "V1_triton"
            try:
                correct, max_diff = self._validate_real(seed, rtol, atol)
                backend_label = "triton"
            except Exception as e:
                # Real path failed → record an honest failed compile, not a
                # silent success. The agent sees correct=False and can adapt.
                result = CompileResult(
                    success=False, backend="triton", correct=None,
                    error=f"real-verify failed: {e}",
                    metadata={"validation_tier": tier, "rtol": rtol, "atol": atol},
                )
                try:
                    self._env.state.record_compile(result)
                except Exception:
                    pass
                return ToolResult(success=False, error=f"verify_correctness (real) failed: {e}")
        else:
            tier = "V0_mock"
            backend_label = "mock"
            correct, max_diff = True, 0.0

        result = CompileResult(
            success=True, backend=backend_label,
            correct=correct, max_diff=max_diff,
            metadata={"validation_tier": tier, "rtol": rtol, "atol": atol},
        )
        try:
            self._env.state.record_compile(result)
        except Exception as e:
            return ToolResult(success=False, error=f"verify_correctness failed: {e}")

        # Optional trial-balloon: snapshot post-compile state, apply, then rollback.
        if trial is not None:
            try:
                self._env.state.checkpoint(self._VERIFY_CHECKPOINT_LABEL)
                rat = trial.get("rationale")
                rationale = None
                if isinstance(rat, str) and rat:
                    rationale = Rationale(text=rat)
                elif isinstance(rat, dict) and rat.get("text"):
                    rationale = Rationale(text=rat["text"], lang=rat.get("lang", "en"))
                d = Decision(
                    kind=trial["kind"],
                    params=dict(trial.get("params", {})),
                    rationale=rationale,
                    level=int(trial.get("level", 1)),
                )
                self._env.state.apply_decision(d)
            except Exception as e:
                # Clean up temp checkpoint if it exists
                self._env.state.rollback(self._VERIFY_CHECKPOINT_LABEL)
                self._env.state.checkpoints.pop(self._VERIFY_CHECKPOINT_LABEL, None)
                return ToolResult(success=False, error=f"trial apply failed: {e}")
            # Roll back: restores decisions + compiles_used (which were already
            # bumped before the checkpoint, so they survive).
            self._env.state.rollback(self._VERIFY_CHECKPOINT_LABEL)
            self._env.state.checkpoints.pop(self._VERIFY_CHECKPOINT_LABEL, None)

        budget = self._env.state.budget
        return ToolResult(success=True, data={
            "correct": correct,
            "max_diff": max_diff,
            "validation_tier": tier,
            "backend": backend_label,
            "rtol": rtol, "atol": atol,
            "compiles_used": budget.compiles_used,
            "compiles_remaining": budget.compiles_remaining,
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "description": "Optional trial decision (applied + rolled back automatically). Same shape as apply_decision params.",
                    "properties": {
                        "kind": {"type": "string"},
                        "params": {"type": "object"},
                        "rationale": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                        "level": {"type": "integer"},
                    },
                    "required": ["kind"],
                },
                "rtol": {"type": "number", "description": "Relative tolerance (default per-dtype)"},
                "atol": {"type": "number", "description": "Absolute tolerance (default per-dtype)"},
                "seed": {"type": "integer", "description": "Input generation seed (default: env.seed)"},
            },
            "required": [],
        }


class CheckpointTool(_EnvBoundTool):
    """Snapshot current state under a label."""

    @property
    def name(self) -> str:
        return "checkpoint"

    @property
    def description(self) -> str:
        return "Snapshot the current optimization state under a label. Free operation."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=False, idempotent=False,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        label = params.get("label")
        if not isinstance(label, str) or not label:
            return ToolResult(success=False, error="label must be a non-empty string")
        try:
            cp = self._env.state.checkpoint(label)
        except Exception as e:
            return ToolResult(success=False, error=f"checkpoint failed: {e}")
        return ToolResult(success=True, data={
            "label": cp.label,
            "decision_count_at": cp.decision_count_at,
            "compile_count_at": cp.compile_count_at,
            "total_checkpoints": len(self._env.state.checkpoints),
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "Checkpoint label (overwrites if exists)"}},
            "required": ["label"],
        }


class RollbackTool(_EnvBoundTool):
    """Restore state from a previous checkpoint."""

    @property
    def name(self) -> str:
        return "rollback"

    @property
    def description(self) -> str:
        return (
            "Restore optimization state from a labelled checkpoint. Mutates "
            "strategy, decision_log, best_result, and budget counters back to "
            "the snapshot. Compile history is preserved (audit trail)."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            concurrent_safe=False, idempotent=True,
            requires_compile=False, mutates_strategy=True,
            budget_type=BudgetType.FREE, cost=CostLevel.CHEAP,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        label = params.get("label")
        if not isinstance(label, str) or not label:
            return ToolResult(success=False, error="label must be a non-empty string")
        try:
            self._env.state.rollback(label)
        except Exception as e:
            return ToolResult(success=False, error=f"rollback failed: {e}")
        budget = self._env.state.budget
        return ToolResult(success=True, data={
            "restored_to": label,
            "decisions_used": budget.decisions_used,
            "compiles_used": budget.compiles_used,
        })

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "Checkpoint label to restore"}},
            "required": ["label"],
        }


# ── Tool Registry ─────────────────────────────────────────────

class ToolRegistry:
    """Registry of all available agent tools.

    Provides lookup, schema generation, and concurrent partitioning.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ArkeTool] = {}

    def register(self, tool: ArkeTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ArkeTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all_schemas(self) -> list[dict[str, Any]]:
        """Generate function-calling schemas for all tools."""
        return [t.to_function_schema() for t in self._tools.values()]

    def partition_for_execution(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> list[list[tuple[str, dict[str, Any], bool]]]:
        """Partition tool calls into concurrent/serial batches.

        Args:
            tool_calls: List of (tool_name, params) tuples

        Returns:
            List of batches. Each batch is a list of (name, params, concurrent).
            A batch with concurrent=True can be executed with asyncio.gather.
        """
        if not tool_calls:
            return []

        batches: list[list[tuple[str, dict[str, Any], bool]]] = []
        current_batch: list[tuple[str, dict[str, Any], bool]] = []
        current_concurrent = None

        for name, params in tool_calls:
            tool = self._tools.get(name)
            is_concurrent = tool.meta.concurrent_safe if tool else False

            if current_concurrent is None:
                current_concurrent = is_concurrent
            elif is_concurrent != current_concurrent:
                batches.append(current_batch)
                current_batch = []
                current_concurrent = is_concurrent

            current_batch.append((name, params, is_concurrent))

        if current_batch:
            batches.append(current_batch)

        return batches

    @classmethod
    def default(cls) -> ToolRegistry:
        """Create registry with the 3 stateless Façade v1.0 tools.

        For the full Façade v1.0 contract (stateless + env-bound 8 tools),
        use `ToolRegistry.with_env(env)`.

        Note: `BenchmarkAdviceSummaryTool` is intentionally NOT registered here.
        It is a Phase-1 internal helper used by `benchmarks/` CLI flows, not part
        of the locked Façade v1.0 contract (arke-harness.md §6.1). The class
        remains importable for those internal call sites.
        """
        reg = cls()
        reg.register(GetHWProfileTool())
        reg.register(AnalyzeComputeTool())
        reg.register(CompileAndProfileTool())
        return reg

    @classmethod
    def with_env(cls, env: Any) -> ToolRegistry:
        """Create the full Façade v1.0 registry bound to an ArkeEnv.

        Wires up the 8 locked tools from arke-harness.md §6.1
        (Façade contract version: `arke-harness-facade-v1.0.0`):

          1. get_hw_profile           (stateless)
          2. analyze_compute          (stateless)
          3. list_legal_actions       (env-bound)
          4. apply_decision           (env-bound, mutates)
          5. verify_correctness       (env-bound)
          6. compile_and_profile      (stateless; D8-F1.3 will upgrade backend)
          7. checkpoint               (env-bound)
          8. rollback                 (env-bound, mutates)

        No additional tools are registered — the Façade is exactly 8.
        """
        reg = cls.default()
        # P5-S5 Step 5b fix: the profile tool must be the env-bound instance,
        # otherwise the decision_log is never injected as strategy (Step 5a
        # added the env parameter but with_env kept the stateless instance —
        # live run3 showed strategy_decisions=0 on every profile).
        reg.register(CompileAndProfileTool(env))
        reg.register(ListLegalActionsTool(env))
        reg.register(ApplyDecisionTool(env))
        reg.register(VerifyCorrectnessTool(env))
        reg.register(CheckpointTool(env))
        reg.register(RollbackTool(env))
        return reg


# Module-level default registry (stateless tools only)
TOOL_REGISTRY = ToolRegistry.default()
