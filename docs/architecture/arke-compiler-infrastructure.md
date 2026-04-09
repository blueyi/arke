# Arke Compiler Infrastructure Design

**Status:** Draft  
**Author:** Kitty (Arke Lead Engineer)  
**Date:** 2026-04-06  
**Version:** 0.1.0  

---

## 1. Executive Summary

This document specifies the compiler infrastructure that serves Arke IR's **LLM-Native multi-layer architecture** (see `arke-ir-spec-design.md` (in `docs/spec/`)). The compiler pipeline transforms `.ak` source through SemanticIR and StrategyIR, lowering to multiple backends (Triton, MLIR, LLVM IR) via a composable pass system.

Arke's compiler has a structural problem: the knowledge about each of its 45 ops is scattered across **6 separate files** (~3000 lines of redundant code total). Adding one new op requires touching 6 files and writing ~100 lines. The fix is a single architectural change — a unified **OpRegistry as the Single Source of Truth** — combined with a **Pass Infrastructure** that organizes compilation into composable, testable stages.

This document specifies:

1. **OpRegistry** — Extended `OpDef` with `shape_rule`, `template_hint`, `reference_impl`, `input_gen`, and `attrs`. All per-op dispatch tables become derived from this single registry.
2. **Pass Infrastructure** — `Pass` protocol + `Pipeline` + `PassContext`, replacing the current single-step lowering with a composable pipeline.
3. **SemanticInterpreter** — PyTorch-eager IR graph executor replacing 45 hand-written NumPy functions.
4. **ShapeInferenceEngine** — Declarative rule-based shape inference replacing the 401-line if/elif chain.
5. **Backend Abstraction** — `ArkeBackend` protocol formalizing the existing ABC, plus `BackendRegistry` for target routing.
6. **SSA Validator** — IR integrity checker for reference validity, type consistency, DAG structure, and symbolic dim consistency.
7. **Implementation Rollout Notes** — active compiler infrastructure evolution notes for the current mainline.

**Target architecture:** Adding a new op should require editing the canonical operator registry entry plus only the backend-specific implementation pieces that truly differ.

---

## 2. Architecture Overview

### 2.1 Current Architecture (Problem State)

```
.ak file
  │
  ▼
parse_file() ──→ Program (AST)
                     │
            ast_to_ir()
                     │
                     ▼
                SemanticIR ──→ shape_inference.py  (if/elif × 45 ops)
                     │
            strategy (default / LLM)
                     │
                     ▼
                StrategyIR
                     │
            TritonBackend.translate()
                     │
            template_engine.py  (if/elif × 45 ops)
                     │
                     ▼
                Triton source → TritonCompiler.compile() → GPU result

Parallel paths (all with per-op if/elif):
  numerical_check.py  — 45 NumPy functions
  kernel_cache.py     — _build_ir() if/elif
  arke_runner.py      — benchmark if/elif
```

**Per-file op knowledge:**

| File | Lines | Role | Problem |
|------|-------|------|---------|
| `arke/ir/ops/catalog.py` | 650 | OpDef definitions (45 ops) | Single source ✓ |
| `arke/ir/shape_inference.py` | 401 | Shape inference | if/elif × every op |
| `arke/integration/kernel_cache.py` | 553 | IR build + GPU exec | `_build_ir()` if/elif |
| `arke/engine/numerical_check.py` | 667 | NumPy reference | 45 hand-written fns |
| `benchmarks/baselines/arke_runner.py` | 323 | Benchmark runner | if/elif dispatch |
| `arke/backend/triton_template_engine.py` | 375 | Template routing | if/elif selection |

### 2.2 Target Architecture

```
.ak file
  │
  ▼
parse_file() ──→ Program (AST) ──→ ast_to_ir()
                                         │
                                   SemanticIR
                                         │
                         ┌───────────────┴──────────────────┐
                         │          Pass Pipeline           │
                         │  Analysis:  SSAValidationPass    │
                         │             ShapeInferencePass ◄─┼── OpRegistry
                         │             TypeCheckPass        │   (single
                         │  Transform: FusionPass           │    source of
                         │             TilingPass           │    truth)
                         │  Lowering:                       │
                         │    Phase 1: TritonCodegenPass ───┼── BackendRegistry
                         │    Phase 1: MLIRCodegenPass  ────┼── (BL1 verify)
                         │    Phase 2+: MLIRCodegenPass ────┼── (full codegen)
                         │    Phase 4: LLVMCodegenPass  ────┼── (direct LLVM)
                         │  Verify:    PostLowerCheckPass   │
                         └───────────────┬──────────────────┘
                                         │
                                 CompilationResult
                                  (source_code, artifacts)
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                   Triton compile  MLIR verify   LLVM emit
                          │              │              │
                       GPU result   correctness   GPU result
                       (Phase 1)    cross-check   (Phase 4)

Validation path (replaces numerical_check.py):
  SemanticInterpreter ── executes IR graph via OpDef.reference_impl (PyTorch eager)
```

**Impact under the target architecture:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Files to edit for new op | 6 | 1–2 | −67% |
| Lines for new op | ~100 | ~10 | −90% |
| if/elif dispatch tables | 5 | 0 | −100% |
| Hand-written NumPy fns | 45 | 0 | −100% |
| Test regression risk | High | Low (incremental) | ↓ |

---

## 3. OpRegistry Design

### 3.1 Extended OpDef Schema

Extended fields are optional so operators can declare only the metadata they need.

```python
# arke/ir/ops/catalog.py  (new additions)

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShapeRule:
    """Declarative shape inference rule.

    kind options:
      "same_as_input"     output = shape of input_key tensor
      "matmul_rule"       [M,K] x [K,N] -> [M,N]
      "batch_matmul_rule" [B,M,K] x [B,K,N] -> [B,M,N]
      "reduce_rule"       drop axes in self.axes from input shape
      "topk_rule"         replace last dim with k (from k_attr)
      "concat_rule"       join along axis_attr
      "split_rule"        split into n parts along axis_attr
      "gather_rule"       shape from index tensor
      "embedding_rule"    [vocab,dim] indexed by [seq] -> [seq,dim]
      "permute_rule"      reorder dims per dims_attr
      "gated_halve_rule"  halve last dim (swiglu/geglu)
      "attention_rule"    [B,H,S,D] from Q shape
      "custom"            delegate to fn(input_shapes, attrs) -> list[int]
    """
    kind: str
    input_key: str = "X"
    axes: list[str | int] = field(default_factory=list)
    k_attr: str = "k"
    axis_attr: str = "axis"
    dims_attr: str = "dims"
    fn: Callable[[dict[str, list[int]], dict], list[int]] | None = None


@dataclass(frozen=True)
class TemplateHint:
    """Routing hint for the Triton template engine.

    template_name: Jinja2 template filename without .j2 extension
    primary_op:    anchor op for fused kernels; defaults to op.name
    extra_ctx:     static key=value pairs injected into Jinja2 context
    """
    template_name: str
    primary_op: str = ""
    extra_ctx: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceImpl:
    """PyTorch eager reference for numerical validation.

    fn signature: (inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor
    dtype_map:    promote dtypes before running, e.g. {"bf16": "f32"}
    """
    fn: Callable
    dtype_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InputGen:
    """Rules for generating test inputs.

    distributions: per-input "uniform"|"normal"|"ones"|"eye"|"randint"|"bool_mask"
    ranges:        per-input (low, high) for uniform/randint
    dtype_override: force dtype for all inputs in tests
    constraints:   informational strings for the test harness
    """
    distributions: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    dtype_override: str | None = None
    constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpDefinition:
    """Complete operator definition — Single Source of Truth.
    """
    # existing fields (unchanged)
    name: str
    category: str
    inputs: dict[str, str]
    output: str
    computation: str
    index_vars: list[str] = field(default_factory=list)
    reduction_axes: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    can_fuse_as: str | None = None
    numpy_ref: str = ""
    # new fields
    shape_rule: ShapeRule | None = None
    template_hint: TemplateHint | None = None
    reference_impl: ReferenceImpl | None = None
    input_gen: InputGen | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
```

### 3.2 Example Annotated OpDef Entries

```python
# Reference implementations (in catalog.py or a companion module)

import torch
import torch.nn.functional as F


def _ref_matmul(inputs: dict, attrs: dict) -> torch.Tensor:
    return torch.matmul(inputs["A"], inputs["B"])

def _ref_relu(inputs: dict, attrs: dict) -> torch.Tensor:
    return F.relu(inputs["X"])

def _ref_layernorm(inputs: dict, attrs: dict) -> torch.Tensor:
    x = inputs["X"].float()
    w = inputs.get("W")
    b = inputs.get("B")
    return F.layer_norm(x, [x.shape[-1]], w, b, attrs.get("eps", 1e-5))

def _ref_softmax(inputs: dict, attrs: dict) -> torch.Tensor:
    return F.softmax(inputs["X"], dim=-1)

def _ref_reduce_sum(inputs: dict, attrs: dict) -> torch.Tensor:
    return inputs["X"].sum(dim=attrs.get("axis", -1))

def _ref_topk(inputs: dict, attrs: dict) -> torch.Tensor:
    return torch.topk(inputs["X"], attrs.get("k", 1), dim=-1).values

def _ref_swiglu(inputs: dict, attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    return F.silu(gate) * up


# Annotated entries (representative sample)

MATMUL = _register(OpDefinition(
    name="matmul",
    category="compute",
    inputs={"A": "Tensor[M,K]", "B": "Tensor[K,N]"},
    output="Tensor[M,N]",
    computation="C[i,j] = sum(A[i,k]*B[k,j], axis=k)",
    index_vars=["i", "j", "k"], reduction_axes=["k"],
    properties=["associative", "distributive"],
    can_fuse_as="prologue", numpy_ref="np.matmul(A, B)",
    shape_rule=ShapeRule(kind="matmul_rule"),
    template_hint=TemplateHint(template_name="matmul"),
    reference_impl=ReferenceImpl(fn=_ref_matmul, dtype_map={"bf16": "f32"}),
    input_gen=InputGen(
        distributions={"A": "normal", "B": "normal"},
        constraints=["A.shape[1] == B.shape[0]"],
    ),
))

RELU = _register(OpDefinition(
    name="relu", category="elementwise",
    inputs={"X": "Tensor[...]"}, output="Tensor[...]",
    computation="Y = max(X, 0)",
    properties=["elementwise", "monotonic"],
    can_fuse_as="epilogue", numpy_ref="np.maximum(X, 0)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise"),
    reference_impl=ReferenceImpl(fn=_ref_relu),
    input_gen=InputGen(distributions={"X": "normal"}),
))

LAYERNORM = _register(OpDefinition(
    name="layernorm", category="norm",
    inputs={"X": "Tensor[B,S,H]", "W": "Tensor[H]", "B": "Tensor[H]"},
    output="Tensor[B,S,H]",
    computation="Y = (X-mean)/sqrt(var+eps)*W+B",
    properties=["normalization"], numpy_ref="",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="layernorm"),
    reference_impl=ReferenceImpl(fn=_ref_layernorm, dtype_map={"bf16": "f32", "f16": "f32"}),
    input_gen=InputGen(distributions={"X": "normal", "W": "ones", "B": "ones"}),
    attrs={"eps": 1e-5},
))

SWIGLU = _register(OpDefinition(
    name="swiglu", category="elementwise",
    inputs={"X": "Tensor[...,2H]"}, output="Tensor[...,H]",
    computation="Y = silu(X[:H]) * X[H:]",
    properties=["gated"], can_fuse_as="epilogue", numpy_ref="",
    shape_rule=ShapeRule(kind="gated_halve_rule", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise",
                                extra_ctx={"op_variant": "swiglu"}),
    reference_impl=ReferenceImpl(fn=_ref_swiglu),
    input_gen=InputGen(distributions={"X": "normal"}),
))
```

### 3.3 OpRegistry Class

```python
# arke/ir/ops/registry.py  (new file)

from __future__ import annotations
from typing import Iterator
from arke.ir.ops.catalog import OP_CATALOG, OpDefinition


class OpRegistry:
    """Thin wrapper around OP_CATALOG with typed access and derived views."""

    def __init__(self, catalog: dict[str, OpDefinition] | None = None) -> None:
        self._ops: dict[str, OpDefinition] = catalog if catalog is not None else OP_CATALOG

    def get(self, name: str) -> OpDefinition:
        try:
            return self._ops[name]
        except KeyError:
            raise KeyError(
                f"Unknown op: {name!r}. Registered: {sorted(self._ops)}"
            )

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __iter__(self) -> Iterator[OpDefinition]:
        return iter(self._ops.values())

    def names(self) -> list[str]:
        return sorted(self._ops)

    def ops_by_category(self, cat: str) -> list[OpDefinition]:
        return [op for op in self._ops.values() if op.category == cat]

    def ops_with_template(self) -> list[OpDefinition]:
        return [op for op in self._ops.values() if op.template_hint is not None]

    def ops_with_reference(self) -> list[OpDefinition]:
        return [op for op in self._ops.values() if op.reference_impl is not None]

    def validate_coverage(self) -> dict[str, list[str]]:
        """Return {field: [op_names_missing_it]} for the four new fields."""
        missing: dict[str, list[str]] = {
            "shape_rule": [], "template_hint": [],
            "reference_impl": [], "input_gen": [],
        }
        for op in self._ops.values():
            for f in missing:
                if getattr(op, f) is None:
                    missing[f].append(op.name)
        return missing


REGISTRY = OpRegistry()  # module-level singleton
```

### 3.4 Derived System Summary

| Current file | Current mechanism | Derived replacement | Driven by |
|---|---|---|---|
| `shape_inference.py` | if/elif chain | `ShapeInferenceEngine` | `op_def.shape_rule` |
| `triton_template_engine.py` | if/elif + template map | `TemplateRouter` | `op_def.template_hint` |
| `numerical_check.py` | 45 NumPy functions | `SemanticInterpreter` | `op_def.reference_impl.fn` |
| `kernel_cache._build_ir()` | manual IR assembly | Removed; use parser pipeline | — |
| `arke_runner.py` | if/elif dispatch | `KernelCache.run_op()` | generic |

---

## 4. Pass Infrastructure

### 4.1 Pass Protocol and Context

```python
# arke/compiler/passes/base.py  (new file)

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from arke.ir.ops.registry import OpRegistry
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Diagnostic:
    severity: Severity
    pass_name: str
    message: str
    node_id: str | None = None

    def __str__(self) -> str:
        loc = f" [node={self.node_id}]" if self.node_id else ""
        return f"[{self.severity.value.upper()}] {self.pass_name}{loc}: {self.message}"


@dataclass
class HardwareProfile:
    name: str = "nvidia_generic"
    compute_capability: tuple[int, int] = (8, 0)
    shared_memory_bytes: int = 49152
    max_threads_per_block: int = 1024
    warp_size: int = 32
    num_sms: int = 1
    peak_tflops_f16: float = 0.0


@dataclass
class PassContext:
    """Shared mutable context threading through all passes.

    - SemanticIR / StrategyIR are replaced (not mutated) by transform passes.
    - artifacts accumulate across passes (e.g. shape_map, triton_source).
    """
    semantic: SemanticIR
    strategy: StrategyIR
    registry: OpRegistry
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    modified: bool = False

    def add_error(self, p: str, msg: str, node_id: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.ERROR, p, msg, node_id))

    def add_warning(self, p: str, msg: str, node_id: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(Severity.WARNING, p, msg, node_id))

    def has_errors(self) -> bool:
        return any(d.severity == Severity.ERROR for d in self.diagnostics)

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]


@dataclass
class PassResult:
    success: bool
    modified: bool = False
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, modified: bool = False, **kw: Any) -> "PassResult":
        return cls(success=True, modified=modified, artifacts=kw)

    @classmethod
    def fail(cls, error: str) -> "PassResult":
        return cls(success=False, error=error)


@runtime_checkable
class Pass(Protocol):
    """A single compilation pass — stateless; all mutable state in PassContext."""
    name: str

    def run(self, ctx: PassContext) -> PassResult: ...
```

### 4.2 Pipeline

```python
# arke/compiler/pipeline.py  (new file)

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any

from arke.compiler.passes.base import HardwareProfile, Pass, PassContext
from arke.ir.ops.registry import OpRegistry, REGISTRY
from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    success: bool
    source_code: str = ""
    error: str | None = None
    diagnostics: list = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Ordered sequence of compilation passes.

    Usage:
        result = Pipeline.default().run(semantic_ir, strategy_ir)
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._passes: list[Pass] = []
        self._registry = REGISTRY
        self._hardware = HardwareProfile()

    def add_pass(self, p: Pass) -> "Pipeline":
        self._passes.append(p)
        return self

    def with_registry(self, r: OpRegistry) -> "Pipeline":
        self._registry = r
        return self

    def with_hardware(self, hw: HardwareProfile) -> "Pipeline":
        self._hardware = hw
        return self

    def run(self, ir: SemanticIR, strategy: StrategyIR) -> CompilationResult:
        ctx = PassContext(
            semantic=ir, strategy=strategy,
            registry=self._registry, hardware=self._hardware,
        )
        for p in self._passes:
            try:
                result = p.run(ctx)
            except Exception as exc:
                logger.exception(f"Pass {p.name!r} raised")
                return CompilationResult(
                    success=False,
                    error=f"Pass {p.name} raised: {exc}",
                    diagnostics=ctx.diagnostics,
                    artifacts=ctx.artifacts,
                )
            ctx.artifacts.update(result.artifacts)
            if result.modified:
                ctx.modified = True
            if not result.success:
                ctx.add_error(p.name, result.error or "pass failed")
                return CompilationResult(
                    success=False,
                    error=f"Pass {p.name}: {result.error}",
                    diagnostics=ctx.diagnostics,
                    artifacts=ctx.artifacts,
                )

        source = ctx.artifacts.get("triton_source", "")
        return CompilationResult(
            success=True, source_code=source,
            diagnostics=ctx.diagnostics, artifacts=ctx.artifacts,
        )

    @classmethod
    def default(cls, hardware: HardwareProfile | None = None) -> "Pipeline":
        """Standard Phase 1 pipeline."""
        from arke.compiler.passes.analysis import (
            SSAValidationPass, ShapeInferencePass, TypeCheckPass,
        )
        from arke.compiler.passes.transform import FusionPass
        from arke.compiler.passes.lowering import TritonCodegenPass

        p = cls("phase1_default")
        if hardware:
            p.with_hardware(hardware)
        return (
            p.add_pass(SSAValidationPass())
             .add_pass(ShapeInferencePass())
             .add_pass(TypeCheckPass())
             .add_pass(FusionPass())
             .add_pass(TritonCodegenPass())
        )
```

### 4.3 Analysis, Transform, and Lowering Passes

```python
# arke/compiler/passes/analysis.py  (new file)

from __future__ import annotations
from arke.compiler.passes.base import PassContext, PassResult
from arke.ir.semantic import NodeRef, ParamRef


class SSAValidationPass:
    name = "ssa_validation"

    def run(self, ctx: PassContext) -> PassResult:
        from arke.compiler.ssa_validator import SSAValidator
        errors = SSAValidator(ctx.registry).validate(ctx.semantic)
        for e in errors:
            ctx.add_error(self.name, e)
        return PassResult.ok() if not errors else PassResult.fail(errors[0])


class ShapeInferencePass:
    name = "shape_inference"

    def run(self, ctx: PassContext) -> PassResult:
        from arke.compiler.shape_engine import ShapeInferenceEngine
        try:
            shape_map = ShapeInferenceEngine(ctx.registry).infer_all(ctx.semantic)
        except ValueError as exc:
            ctx.add_error(self.name, str(exc))
            return PassResult.fail(str(exc))
        ctx.artifacts["shape_map"] = shape_map
        return PassResult.ok(shape_map=shape_map)


class TypeCheckPass:
    name = "type_check"

    def run(self, ctx: PassContext) -> PassResult:
        errors: list[str] = []
        param_dtypes = {p.name: p.dtype for p in ctx.semantic.params}
        node_dtypes: dict[str, str] = {}

        for node in ctx.semantic.nodes:
            input_dtypes = []
            for ref in node.inputs.values():
                if isinstance(ref, ParamRef):
                    input_dtypes.append(param_dtypes.get(ref.name, "unknown"))
                elif isinstance(ref, NodeRef):
                    input_dtypes.append(node_dtypes.get(ref.id, "unknown"))

            out_dtype = (
                node.output.dtype if node.op == "cast"
                else (input_dtypes[0] if input_dtypes else node.output.dtype)
            )
            node_dtypes[node.id] = out_dtype

            if "unknown" in input_dtypes:
                errors.append(f"Node {node.id}: unresolved input dtype")

        for e in errors:
            ctx.add_error(self.name, e)
        ctx.artifacts["node_dtypes"] = node_dtypes
        return PassResult.ok() if not errors else PassResult.fail(errors[0])


# arke/compiler/passes/transform.py  (new file)

from arke.compiler.passes.base import PassContext, PassResult


class FusionPass:
    """Identify epilogue/prologue fusion opportunities."""
    name = "fusion"

    def run(self, ctx: PassContext) -> PassResult:
        fusion_plan: list[dict] = []
        consumer_count: dict[str, int] = {n.id: 0 for n in ctx.semantic.nodes}
        for edge in ctx.semantic.edges:
            if edge.from_node in consumer_count:
                consumer_count[edge.from_node] += 1

        for node in ctx.semantic.nodes:
            if consumer_count.get(node.id, 0) == 1:
                try:
                    op_def = ctx.registry.get(node.op)
                    if op_def.can_fuse_as == "epilogue":
                        fusion_plan.append({"node": node.id, "kind": "epilogue"})
                except KeyError:
                    pass
        ctx.artifacts["fusion_plan"] = fusion_plan
        return PassResult.ok(fusion_plan=fusion_plan)


# arke/compiler/passes/lowering.py  (new file)

from arke.compiler.passes.base import PassContext, PassResult


class TritonCodegenPass:
    """Lower SemanticIR + StrategyIR to Triton source via TemplateRouter."""
    name = "triton_codegen"

    def run(self, ctx: PassContext) -> PassResult:
        from arke.compiler.template_router import TemplateRouter
        try:
            source = TemplateRouter(ctx.registry).translate(
                ctx.semantic, ctx.strategy
            )
        except Exception as exc:
            ctx.add_error(self.name, str(exc))
            return PassResult.fail(str(exc))
        ctx.artifacts["triton_source"] = source
        return PassResult.ok(triton_source=source)


class MLIRCodegenPass:
    """Lower SemanticIR + StrategyIR to MLIR standard dialects.

    Phase 1: BL1 basic pathway — emit linalg/transform MLIR for 13 ops,
             verify via mlir-opt (correctness cross-check, not primary codegen).
    Phase 2+: Full codegen — emit complete MLIR for all ops, alternative
              compilation path alongside Triton.
    """
    name = "mlir_codegen"

    def run(self, ctx: PassContext) -> PassResult:
        from arke.backend.mlir_emitter import MLIREmitter
        try:
            mlir_source = MLIREmitter(ctx.registry).emit(
                ctx.semantic, ctx.strategy
            )
        except Exception as exc:
            ctx.add_error(self.name, str(exc))
            return PassResult.fail(str(exc))
        ctx.artifacts["mlir_source"] = mlir_source
        return PassResult.ok(mlir_source=mlir_source)


class LLVMCodegenPass:
    """Lower Arke IR directly to LLVM IR (Phase 4).

    Bypasses both Triton and MLIR. Requires ScheduleIR + InstructionIR
    to be fully implemented.
    """
    name = "llvm_codegen"

    def run(self, ctx: PassContext) -> PassResult:
        # Phase 4 stub — not implemented in Phase 1
        raise NotImplementedError("LLVMCodegenPass requires Phase 4 InstructionIR")
```

### 4.4 Pass Categories Reference

| Category | Modifies IR? | Adds artifacts? | Examples |
|---|---|---|---|
| Analysis | No | Yes | SSAValidationPass, ShapeInferencePass, TypeCheckPass |
| Transform | Yes (replaces) | Yes | FusionPass, TilingPass |
| Lowering | No | Yes (source) | TritonCodegenPass, MLIRCodegenPass, LLVMCodegenPass |
| Verification | No | Maybe | PostLowerCheckPass |

---

## 5. SemanticInterpreter Design

### 5.1 Purpose

The `SemanticInterpreter` is a PyTorch-eager executor for `SemanticIR` graphs. It replaces the 667-line `numerical_check.py` (45 hand-written NumPy functions) with a single generic dispatcher. Because it calls `OpDef.reference_impl.fn`, it **automatically supports new ops** without any code changes beyond the OpDef registration.

### 5.2 Implementation

```python
# arke/engine/semantic_interpreter.py  (new file)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch

from arke.ir.ops.registry import OpRegistry, REGISTRY
from arke.ir.semantic import NodeRef, ParamRef, SemanticIR


DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "f16":  torch.float16,
    "f32":  torch.float32,
    "f64":  torch.float64,
    "bf16": torch.bfloat16,
    "i8":   torch.int8,
    "i16":  torch.int16,
    "i32":  torch.int32,
    "i64":  torch.int64,
    "u8":   torch.uint8,
    "bool": torch.bool,
}


@dataclass
class InterpreterResult:
    success: bool
    outputs: dict[str, torch.Tensor]
    error: str | None = None


class SemanticInterpreter:
    """Execute a SemanticIR graph using PyTorch eager mode.

    Replaces arke/engine/numerical_check.py entirely.
    Each node dispatches to its OpDef.reference_impl.fn.

    Usage:
        interp = SemanticInterpreter()
        result = interp.run(ir, {"A": tensor_a, "B": tensor_b})
        output = result.outputs[ir.return_node]
    """

    def __init__(self, registry: OpRegistry | None = None, device: str = "cpu") -> None:
        self._registry = registry or REGISTRY
        self._device = device

    def run(
        self,
        ir: SemanticIR,
        inputs: dict[str, torch.Tensor],
        attrs: dict[str, dict[str, Any]] | None = None,
    ) -> InterpreterResult:
        """Execute the IR graph. Returns InterpreterResult with per-node outputs.

        Args:
            ir:     SemanticIR to execute.
            inputs: {param_name: tensor} — must match ir.params.
            attrs:  {node_id: {attr_name: value}} — per-node attributes.
        """
        attrs = attrs or {}
        env: dict[str, torch.Tensor] = {}

        # seed environment with param tensors
        for param in ir.params:
            if param.name not in inputs:
                return InterpreterResult(
                    success=False, outputs={},
                    error=f"Missing input for param {param.name!r}",
                )
            t = inputs[param.name].to(self._device)
            env[param.name] = t

        # execute nodes in topological order (IR nodes are already ordered)
        for node in ir.nodes:
            try:
                op_def = self._registry.get(node.op)
            except KeyError as exc:
                return InterpreterResult(success=False, outputs={}, error=str(exc))

            if op_def.reference_impl is None:
                return InterpreterResult(
                    success=False, outputs={},
                    error=f"Op {node.op!r} has no reference_impl registered",
                )

            # Resolve inputs
            node_inputs: dict[str, torch.Tensor] = {}
            for key, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    tensor = env.get(ref.name)
                    if tensor is None:
                        return InterpreterResult(
                            success=False, outputs={},
                            error=f"Node {node.id}: param {ref.name!r} not in env",
                        )
                    node_inputs[key] = tensor
                elif isinstance(ref, NodeRef):
                    tensor = env.get(ref.id)
                    if tensor is None:
                        return InterpreterResult(
                            success=False, outputs={},
                            error=f"Node {node.id}: node {ref.id!r} not in env",
                        )
                    node_inputs[key] = tensor

            # Apply dtype promotion for reference run
            ref_impl = op_def.reference_impl
            if ref_impl.dtype_map:
                promoted: dict[str, torch.Tensor] = {}
                for k, t in node_inputs.items():
                    arke_dtype = node.output.dtype  # use output dtype as heuristic
                    target_arke = ref_impl.dtype_map.get(arke_dtype)
                    if target_arke and target_arke in DTYPE_TO_TORCH:
                        promoted[k] = t.to(DTYPE_TO_TORCH[target_arke])
                    else:
                        promoted[k] = t
                node_inputs = promoted

            node_attrs = {**op_def.attrs, **attrs.get(node.id, {})}

            try:
                output = ref_impl.fn(node_inputs, node_attrs)
            except Exception as exc:
                return InterpreterResult(
                    success=False, outputs={},
                    error=f"Node {node.id} ({node.op}): {exc}",
                )

            env[node.id] = output

        return InterpreterResult(success=True, outputs=env)


# ---------------------------------------------------------------------------
# Backward-compat shim: expose the same interface as numerical_check.py
# ---------------------------------------------------------------------------

def check_numerical(
    ir: SemanticIR,
    inputs: dict[str, torch.Tensor],
    kernel_output: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple[bool, str]:
    """Drop-in replacement for the old numerical_check API.

    Returns (passed, message). Uses SemanticInterpreter for reference.
    """
    interp = SemanticInterpreter()
    ref = interp.run(ir, inputs)
    if not ref.success:
        return False, f"Reference execution failed: {ref.error}"

    expected = ref.outputs.get(ir.return_node)
    if expected is None:
        return False, f"Return node {ir.return_node!r} not in outputs"

    # Promote for comparison
    if kernel_output.dtype != expected.dtype:
        kernel_output = kernel_output.to(expected.dtype)

    passed = torch.allclose(kernel_output, expected, atol=atol, rtol=rtol)
    if not passed:
        diff = (kernel_output - expected).abs()
        msg = (
            f"Numerical check failed: max_diff={diff.max().item():.6f}, "
            f"mean_diff={diff.mean().item():.6f}"
        )
        return False, msg
    return True, "OK"
```

### 5.3 Migration from numerical_check.py

`numerical_check.py` currently exports:
- `NumericalChecker.check(ir, inputs, output)` → `bool`
- `execute_reference(op_name, inputs)` → `np.ndarray`

Migration steps:
1. Add `reference_impl` to all 45 `OpDef` entries in `catalog.py`.
2. Replace `NumericalChecker.check` calls with `check_numerical` from `semantic_interpreter.py`.
3. Keep `numerical_check.py` as a thin re-export shim until all call sites are migrated.
4. Delete shim after all tests pass.

---

## 6. ShapeInferenceEngine Design

### 6.1 Purpose

Replaces the 401-line `shape_inference.py` if/elif chain with a declarative rule engine driven by `OpDef.shape_rule`.

### 6.2 Built-in Rules

| Rule kind | Logic | Example ops |
|---|---|---|
| `same_as_input` | `output = input_shapes[input_key]` | relu, gelu, silu, layernorm, softmax |
| `matmul_rule` | `[A[0], B[1]]` from `[M,K]×[K,N]` | matmul |
| `batch_matmul_rule` | `[B, A[1], C[2]]` from `[B,M,K]×[B,K,N]` | batch_matmul |
| `reduce_rule` | remove `axes` from input shape | reduce_sum, reduce_max, reduce_mean |
| `topk_rule` | replace last dim with `attrs[k_attr]` | topk, argmax |
| `concat_rule` | sum sizes along `attrs[axis_attr]` | concat |
| `split_rule` | divide size along `attrs[axis_attr]` by n | split |
| `gather_rule` | shape from index tensor | gather, scatter |
| `embedding_rule` | `[*index_shape, embed_dim]` | embedding |
| `permute_rule` | reorder dims per `attrs[dims_attr]` | permute, transpose |
| `gated_halve_rule` | halve last dim | swiglu, geglu |
| `attention_rule` | `[B, H, S, S]` from Q shape | flash_attention, etc. |
| `custom` | delegate to `shape_rule.fn` | special cases |

### 6.3 Implementation

```python
# arke/compiler/shape_engine.py  (new file)

from __future__ import annotations
from typing import Any

from arke.ir.ops.catalog import ShapeRule
from arke.ir.ops.registry import OpRegistry, REGISTRY
from arke.ir.semantic import NodeRef, ParamRef, SemanticIR


class ShapeInferenceEngine:
    """Declarative shape inference driven by OpDef.shape_rule.

    Replaces arke/ir/shape_inference.py entirely.
    """

    def __init__(self, registry: OpRegistry | None = None) -> None:
        self._registry = registry or REGISTRY

    def infer_all(self, ir: SemanticIR) -> dict[str, list[int]]:
        """Infer shapes for all nodes. Returns {node_id: shape}."""
        shape_env: dict[str, list[int]] = {
            p.name: p.shape for p in ir.params
        }

        for node in ir.nodes:
            op_def = self._registry.get(node.op)
            if op_def.shape_rule is None:
                # Fallback: use existing shape_inference.infer_output_shape
                from arke.ir.shape_inference import infer_output_shape
                input_shapes = self._resolve_input_shapes(node, shape_env)
                shape = infer_output_shape(node.op, input_shapes)
            else:
                input_shapes = self._resolve_input_shapes(node, shape_env)
                node_attrs = dict(op_def.attrs)
                # Pull concrete attr values from node.inputs (non-tensor args)
                shape = self._apply_rule(op_def.shape_rule, input_shapes, node_attrs)
            shape_env[node.id] = shape

        return shape_env

    def infer_node(
        self,
        op_name: str,
        input_shapes: dict[str, list[int]],
        attrs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Infer shape for a single op by name (public utility API)."""
        op_def = self._registry.get(op_name)
        if op_def.shape_rule is None:
            from arke.ir.shape_inference import infer_output_shape
            return infer_output_shape(op_name, input_shapes)
        return self._apply_rule(op_def.shape_rule, input_shapes, attrs or {})

    # ── private helpers ──────────────────────────────────────────────────────

    def _resolve_input_shapes(
        self,
        node: Any,
        shape_env: dict[str, list[int]],
    ) -> dict[str, list[int]]:
        shapes: dict[str, list[int]] = {}
        for key, ref in node.inputs.items():
            if isinstance(ref, ParamRef) and ref.name in shape_env:
                shapes[key] = shape_env[ref.name]
            elif isinstance(ref, NodeRef) and ref.id in shape_env:
                shapes[key] = shape_env[ref.id]
        return shapes

    def _apply_rule(
        self,
        rule: ShapeRule,
        input_shapes: dict[str, list[int]],
        attrs: dict[str, Any],
    ) -> list[int]:
        k = rule.kind

        if k == "same_as_input":
            return list(input_shapes[rule.input_key])

        elif k == "matmul_rule":
            a, b = input_shapes["A"], input_shapes["B"]
            return [a[0], b[1]]

        elif k == "batch_matmul_rule":
            a, b = input_shapes["A"], input_shapes["B"]
            return [a[0], a[1], b[2]]

        elif k == "reduce_rule":
            shape = list(input_shapes[rule.input_key])
            axes = rule.axes or [attrs.get(rule.axis_attr, -1)]
            ndim = len(shape)
            normalized = [ax % ndim for ax in axes]
            return [s for i, s in enumerate(shape) if i not in normalized]

        elif k == "topk_rule":
            shape = list(input_shapes[rule.input_key])
            k_val = int(attrs.get(rule.k_attr, 1))
            return shape[:-1] + [k_val]

        elif k == "concat_rule":
            axis = int(attrs.get(rule.axis_attr, 0))
            shapes = list(input_shapes.values())
            result = list(shapes[0])
            result[axis] = sum(s[axis] for s in shapes)
            return result

        elif k == "split_rule":
            shape = list(input_shapes[rule.input_key])
            axis = int(attrs.get(rule.axis_attr, 0))
            n = attrs.get("n", 2)
            shape[axis] = shape[axis] // n
            return shape

        elif k == "gather_rule":
            idx_shape = input_shapes.get("idx", input_shapes.get("index", []))
            src_shape = input_shapes[rule.input_key]
            return list(idx_shape) + src_shape[1:]

        elif k == "embedding_rule":
            idx_shape = input_shapes.get("idx", input_shapes.get("index", []))
            embed_shape = input_shapes.get("W", input_shapes.get("weight", [0, 1]))
            return list(idx_shape) + [embed_shape[-1]]

        elif k == "permute_rule":
            shape = list(input_shapes[rule.input_key])
            dims = attrs.get(rule.dims_attr, list(range(len(shape))))
            return [shape[d] for d in dims]

        elif k == "gated_halve_rule":
            shape = list(input_shapes[rule.input_key])
            return shape[:-1] + [shape[-1] // 2]

        elif k == "attention_rule":
            q_shape = input_shapes.get("Q", input_shapes.get("query", []))
            # [B, H, S, D] -> output [B, H, S, S] (attention scores)
            # Actual flash_attention returns [B, H, S, D] (same as Q)
            return list(q_shape)

        elif k == "custom":
            if rule.fn is None:
                raise ValueError("ShapeRule kind='custom' requires fn")
            return rule.fn(input_shapes, attrs)

        else:
            raise ValueError(f"Unknown ShapeRule kind: {k!r}")
```

### 6.4 Incremental rollout behavior

During implementation rollout, the engine may fall back to the existing `infer_output_shape()` when `shape_rule` is `None`. This is an implementation bridge, not part of the active user-facing architecture contract.

---

## 7. Backend Abstraction Design

### 7.1 Formalized ArkeBackend Protocol

The existing `ArkeBackend` ABC in `backend/base.py` already provides the right interface. The changes here are:
1. Convert from `ABC` to a `Protocol` so TritonBackend doesn't need to inherit.
2. Add `BackendRegistry` for target_hw → backend routing.
3. Define `BackendArtifact` and `CompiledKernel` as typed dataclasses.
4. Declare `MLIRBackend` and `LLVMBackend` protocol stubs for future stages.

```python
# arke/backend/protocol.py  (new file — replaces ABC in base.py)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from arke.ir.semantic import SemanticIR
from arke.ir.strategy import StrategyIR


@dataclass
class BackendArtifact:
    """Intermediate artifact from the lower() step."""
    source_code: str
    backend_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledKernel:
    """Result of compile() — a ready-to-run kernel."""
    success: bool
    binary_path: str | None = None
    error: str | None = None
    backend_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ArkeBackend(Protocol):
    """Protocol for Arke compilation backends.

    Phase 1:   TritonBackend  (GPU via Triton)
    Phase 1:   MLIRBackend   (framework + BL1 verify)
    Phase 2:   TritonBackend  (Ascend via Triton) + MLIRBackend (full integration)
    Phase 3:   MLIRBackend    (primary codegen, deeper hardware control)
    Phase 4:   LLVMBackend    (direct LLVM IR emission)
    """
    name: str

    def lower(
        self, semantic: SemanticIR, strategy: StrategyIR
    ) -> BackendArtifact: ...

    def compile(self, artifact: BackendArtifact) -> CompiledKernel: ...

    def run(
        self, kernel: CompiledKernel, inputs: dict[str, Any]
    ) -> dict[str, Any]: ...

    def profile(
        self,
        kernel: CompiledKernel,
        inputs: dict[str, Any],
        warmup: int = 5,
        runs: int = 20,
    ) -> dict[str, float]: ...


# Phase 2-4 stubs (protocol only, no implementation required for Phase 1)

@runtime_checkable
class MLIRBackend(Protocol):
    """MLIR backend — Phase 1: framework + BL1 verify; Phase 2-3: full integration."""
    name: str
    def lower(self, semantic: SemanticIR, strategy: StrategyIR) -> BackendArtifact: ...
    def compile(self, artifact: BackendArtifact) -> CompiledKernel: ...
    def run(self, kernel: CompiledKernel, inputs: dict) -> dict: ...
    def profile(self, kernel: CompiledKernel, inputs: dict, **kwargs) -> dict: ...


@runtime_checkable
class LLVMBackend(Protocol):
    """Phase 4 backend via LLVM IR."""
    name: str
    def lower(self, semantic: SemanticIR, strategy: StrategyIR) -> BackendArtifact: ...
    def compile(self, artifact: BackendArtifact) -> CompiledKernel: ...
    def run(self, kernel: CompiledKernel, inputs: dict) -> dict: ...
    def profile(self, kernel: CompiledKernel, inputs: dict, **kwargs) -> dict: ...
```

### 7.2 BackendRegistry

```python
# arke/backend/registry.py  (new file)

from __future__ import annotations
from typing import Any
from arke.backend.protocol import ArkeBackend


class BackendRegistry:
    """Route target_hw strings to ArkeBackend instances.

    Usage:
        registry = BackendRegistry.default()
        backend = registry.get("nvidia_sm86")
    """

    def __init__(self) -> None:
        self._backends: dict[str, ArkeBackend] = {}
        self._target_map: dict[str, str] = {}  # target_hw -> backend name

    def register(self, backend: ArkeBackend, targets: list[str]) -> None:
        """Register a backend and map it to one or more target strings."""
        self._backends[backend.name] = backend
        for t in targets:
            self._target_map[t] = backend.name

    def get(self, target_hw: str) -> ArkeBackend:
        """Return the appropriate backend for a hardware target."""
        backend_name = self._target_map.get(target_hw)
        if backend_name is None:
            # Default: fall back to triton for unknown NVIDIA targets
            if "nvidia" in target_hw.lower() or "cuda" in target_hw.lower():
                backend_name = "triton"
            else:
                raise KeyError(
                    f"No backend registered for target {target_hw!r}. "
                    f"Registered targets: {sorted(self._target_map)}"
                )
        return self._backends[backend_name]

    @classmethod
    def default(cls) -> "BackendRegistry":
        """Build the default registry with TritonBackend for NVIDIA."""
        from arke.backend.triton_backend import TritonBackend
        reg = cls()
        tb = TritonBackend()
        reg.register(tb, [
            "triton", "nvidia", "cuda",
            "nvidia_sm80", "nvidia_sm86", "nvidia_sm89", "nvidia_sm90",
        ])
        return reg


REGISTRY = BackendRegistry.default()
```

### 7.3 TritonBackend Adapter

The existing `TritonBackend` in `backend/triton_backend.py` uses the old `translate` / `compile` / `run` interface. A thin adapter makes it conform to the new protocol without breaking existing tests:

```python
# Add to arke/backend/triton_backend.py (new methods, no changes to existing)

from arke.backend.protocol import ArkeBackend, BackendArtifact, CompiledKernel

class TritonBackend(ArkeBackend):  # keep existing ABC inheritance too
    ...

    def lower(self, semantic: SemanticIR, strategy: StrategyIR) -> BackendArtifact:
        """New protocol interface: lower to Triton source."""
        source = self.translate(semantic, strategy)  # existing method
        return BackendArtifact(source_code=source, backend_name=self.name)

    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """New protocol interface: compile Triton source."""
        result = self._compiler.compile(artifact.source_code)  # existing
        return CompiledKernel(
            success=result.success,
            binary_path=result.binary_path,
            error=result.error,
            backend_name=self.name,
        )
```

---

## 8. SSA Validator Design

### 8.1 Validation Rules

The SSA Validator runs as the **first pass** in every pipeline. It checks four classes of invariants:

| Class | Check | Error example |
|---|---|---|
| Reference validity | All `InputRef` point to defined params/nodes | `"Node relu_0 references undefined param 'X'"` |
| Type consistency | Referenced tensor dtype/shape matches declared output | `"Edge matmul_0→relu_0: dtype mismatch f32 vs f16"` |
| DAG structure | Node graph has no cycles | `"Cycle detected: relu_0 → add_0 → relu_0"` |
| Symbolic dim consistency | Dims declared in `index_vars` are consistently used | `"Dim 'k' used but not declared in index_vars"` |

### 8.2 Implementation

```python
# arke/compiler/ssa_validator.py  (new file)

from __future__ import annotations
from collections import defaultdict

from arke.ir.ops.registry import OpRegistry, REGISTRY
from arke.ir.semantic import NodeRef, ParamRef, SemanticIR


class SSAValidator:
    """Validate SemanticIR structural integrity.

    Returns a list of error strings. Empty list = valid.
    """

    def __init__(self, registry: OpRegistry | None = None) -> None:
        self._registry = registry or REGISTRY

    def validate(self, ir: SemanticIR) -> list[str]:
        errors: list[str] = []
        errors.extend(self._check_references(ir))
        errors.extend(self._check_dag(ir))
        errors.extend(self._check_return_node(ir))
        errors.extend(self._check_op_names(ir))
        return errors

    # ── Reference validity ────────────────────────────────────────────────────

    def _check_references(self, ir: SemanticIR) -> list[str]:
        errors: list[str] = []
        defined_params = {p.name for p in ir.params}
        defined_nodes: set[str] = set()

        for node in ir.nodes:
            for key, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    if ref.name not in defined_params:
                        errors.append(
                            f"Node {node.id!r} input {key!r}: "
                            f"param {ref.name!r} not defined"
                        )
                elif isinstance(ref, NodeRef):
                    if ref.id not in defined_nodes:
                        errors.append(
                            f"Node {node.id!r} input {key!r}: "
                            f"node {ref.id!r} not yet defined (forward reference)"
                        )
            defined_nodes.add(node.id)

        return errors

    # ── DAG structure (cycle detection) ──────────────────────────────────────

    def _check_dag(self, ir: SemanticIR) -> list[str]:
        """Topological sort; any back-edge is a cycle."""
        adj: dict[str, list[str]] = defaultdict(list)
        for node in ir.nodes:
            for ref in node.inputs.values():
                if isinstance(ref, NodeRef):
                    adj[ref.id].append(node.id)

        visited: set[str] = set()
        in_stack: set[str] = set()
        cycles: list[str] = []

        def dfs(n: str) -> None:
            visited.add(n)
            in_stack.add(n)
            for child in adj.get(n, []):
                if child not in visited:
                    dfs(child)
                elif child in in_stack:
                    cycles.append(f"Cycle detected involving nodes {n!r} → {child!r}")
            in_stack.discard(n)

        for node in ir.nodes:
            if node.id not in visited:
                dfs(node.id)

        return cycles

    # ── Return node ──────────────────────────────────────────────────────────

    def _check_return_node(self, ir: SemanticIR) -> list[str]:
        node_ids = {n.id for n in ir.nodes}
        if ir.return_node and ir.return_node not in node_ids:
            return [f"return_node {ir.return_node!r} not found in nodes"]
        return []

    # ── Op names ─────────────────────────────────────────────────────────────

    def _check_op_names(self, ir: SemanticIR) -> list[str]:
        errors: list[str] = []
        for node in ir.nodes:
            if node.op not in self._registry:
                errors.append(
                    f"Node {node.id!r}: unknown op {node.op!r}. "
                    f"Register it in catalog.py first."
                )
        return errors
```

---

## 9. Migration Plan

Implementation should remain incremental, but active docs should describe the target architecture directly rather than preserving compatibility framing.

### Phase 0 — Foundation (no behavior change)

| Step | Action | Files | Tests |
|------|--------|-------|-------|
| 0.1 | Add new dataclasses (`ShapeRule`, `TemplateHint`, `ReferenceImpl`, `InputGen`) to `catalog.py` — all `None` defaults | `catalog.py` | No change |
| 0.2 | Create `arke/ir/ops/registry.py` with `OpRegistry` + `REGISTRY` singleton | new file | No change |
| 0.3 | Create `arke/compiler/passes/base.py` (`Pass`, `PassContext`, `PassResult`) | new file | No change |
| 0.4 | Create `arke/compiler/pipeline.py` (`Pipeline`, `CompilationResult`) | new file | No change |
| 0.5 | Create `arke/compiler/ssa_validator.py` | new file | No change |

### Phase 1 — Annotate OpDefs (data only)

| Step | Action | Files | Tests |
|------|--------|-------|-------|
| 1.1 | Add `shape_rule` to all 45 OpDefs | `catalog.py` | No change |
| 1.2 | Add `template_hint` to all 45 OpDefs | `catalog.py` | No change |
| 1.3 | Add `reference_impl` functions (PyTorch) to all 45 OpDefs | `catalog.py` | No change |
| 1.4 | Add `input_gen` to all 45 OpDefs | `catalog.py` | No change |

### Phase 2 — New Engines (dual-path, old path still active)

| Step | Action | Files | Tests |
|------|--------|-------|-------|
| 2.1 | Implement `ShapeInferenceEngine` with fallback to old `infer_output_shape` | `compiler/shape_engine.py` | Add new tests |
| 2.2 | Implement `SemanticInterpreter` with `check_numerical` shim | `engine/semantic_interpreter.py` | Verify matches old output |
| 2.3 | Implement `SSAValidationPass`, `ShapeInferencePass`, `TypeCheckPass` | `compiler/passes/analysis.py` | Add pass unit tests |
| 2.4 | Implement `FusionPass`, `TritonCodegenPass` | `compiler/passes/transform.py`, `lowering.py` | Add pass unit tests |
| 2.5 | Implement `TemplateRouter` (uses `OpDef.template_hint`) alongside old engine | `compiler/template_router.py` | Verify same output |

### Phase 3 — Cutover (swap old paths for new, keep shims)

| Step | Action | Files | Tests |
|------|--------|-------|-------|
| 3.1 | Replace `shape_inference.py` calls with `ShapeInferenceEngine` in `builder.py` | `ir/builder.py` | All 422 must pass |
| 3.2 | Replace `TritonTemplateEngine` with `TemplateRouter` in `TritonBackend` | `backend/triton_backend.py` | All 422 must pass |
| 3.3 | Replace `NumericalChecker` in `accuracy.py` with `check_numerical` shim | `engine/accuracy.py` | All 422 must pass |
| 3.4 | Rework `KernelCache._build_ir()` to use parser pipeline; remove if/elif | `integration/kernel_cache.py` | All 422 must pass |
| 3.5 | Simplify `arke_runner.py` to use `KernelCache.run_op()` generically | `benchmarks/baselines/arke_runner.py` | All bench tests pass |
| 3.6 | Wire `Pipeline.default()` into `TritonBackend.translate()` entry point | `backend/triton_backend.py`, `compiler/pipeline.py` | All 422 must pass |

### Phase 4 — Cleanup (delete dead code)

| Step | Action | Files |
|------|--------|-------|
| 4.1 | Delete if/elif body from `shape_inference.py`; keep as thin wrapper for 1 release | `ir/shape_inference.py` |
| 4.2 | Delete if/elif body from `triton_template_engine.py` | `backend/triton_template_engine.py` |
| 4.3 | Delete `numerical_check.py` NumPy functions; keep `check_numerical` shim | `engine/numerical_check.py` |
| 4.4 | Delete `kernel_cache._build_ir()` and its if/elif chains | `integration/kernel_cache.py` |
| 4.5 | Run full test suite; confirm 422 pass, 6 skip | All |

---

## 10. Task Breakdown

### 10.1 Tasks with Estimates

Work unit: one capable LLM Agent (opus-level) with full codebase context.  
Time unit: hours of agent wall-clock execution.

| Task | Description | Files | Est. | Deps |
|------|-------------|-------|------|------|
| T01 | Add new dataclasses to `catalog.py`; extend `OpDefinition` | `catalog.py` | 1h | — |
| T02 | Create `OpRegistry` class + `REGISTRY` singleton | `registry.py` | 0.5h | T01 |
| T03 | Write `Pass`, `PassContext`, `PassResult` protocol + dataclasses | `passes/base.py` | 1h | — |
| T04 | Write `Pipeline` + `CompilationResult` | `pipeline.py` | 1h | T03 |
| T05 | Annotate `shape_rule` for all 45 ops | `catalog.py` | 2h | T01 |
| T06 | Annotate `template_hint` for all 45 ops | `catalog.py` | 1h | T01 |
| T07 | Write 45 PyTorch `reference_impl` functions; annotate OpDefs | `catalog.py` | 3h | T01 |
| T08 | Annotate `input_gen` for all 45 ops | `catalog.py` | 1.5h | T01 |
| T09 | Implement `ShapeInferenceEngine` with fallback | `shape_engine.py` | 2h | T02, T05 |
| T10 | Implement `SemanticInterpreter` + `check_numerical` shim | `semantic_interpreter.py` | 2h | T02, T07 |
| T11 | Implement `SSAValidator` | `ssa_validator.py` | 1.5h | T02 |
| T12 | Implement `SSAValidationPass`, `ShapeInferencePass`, `TypeCheckPass` | `passes/analysis.py` | 1h | T09, T11 |
| T13 | Implement `FusionPass`, `TritonCodegenPass` | `passes/transform.py`, `lowering.py` | 1.5h | T03 |
| T14 | Implement `TemplateRouter` using `OpDef.template_hint` | `template_router.py` | 2h | T02, T06 |
| T15 | Add `BackendRegistry` + typed artifacts | `backend/registry.py`, `protocol.py` | 1h | — |
| T16 | Add `lower()`/`compile()` adapters to `TritonBackend` | `backend/triton_backend.py` | 0.5h | T15 |
| T16b | `MLIREmitter` skeleton + BL1 pathway (13 ops → linalg/transform MLIR, verify via mlir-opt) | `backend/mlir_emitter.py`, `tests/test_mlir_emitter.py` | 2h | T02, T15 |
| T17 | Integrate `ShapeInferenceEngine` into `builder.py` | `ir/builder.py` | 1h | T09 |
| T18 | Swap `TemplateEngine` → `TemplateRouter` in `TritonBackend` | `backend/triton_backend.py` | 1h | T14 |
| T19 | Swap `NumericalChecker` → `SemanticInterpreter` in `accuracy.py` | `engine/accuracy.py` | 1h | T10 |
| T20 | Refactor `KernelCache._build_ir()` to use parser pipeline | `integration/kernel_cache.py` | 2h | T09 |
| T21 | Simplify `arke_runner.py` to use `KernelCache.run_op()` | `benchmarks/baselines/arke_runner.py` | 1h | T20 |
| T22 | Wire `Pipeline.default()` into main compilation path | `backend/triton_backend.py` | 1h | T04, T12, T13 |
| T23 | Write unit tests for all new passes and engines | `tests/` | 3h | T09–T14 |
| T24 | Full regression: run all 422 tests; fix failures | all | 2h | all |
| T25 | Delete dead code (phase 4 cleanup) | multiple | 1h | T24 |

**Total estimate: ~33 hours** (can be compressed with parallel execution)

### 10.2 Phase Grouping

```
Phase A (Foundation)   — T01, T02, T03, T04, T15        [~4h, parallel]
Phase B (Annotation)   — T05, T06, T07, T08              [~7.5h, sequential in catalog.py]
Phase C (Engines)      — T09, T10, T11, T14, T16, T16b   [~10h, parallel]
Phase D (Passes)       — T12, T13                        [~2.5h, after T09/T11]
Phase E (Cutover)      — T17, T18, T19, T20, T21, T22   [~7h, sequential per subsystem]
Phase F (Verify+Clean) — T23, T24, T25                  [~6h]
```

### 10.3 Critical Path

```
T01 ──→ T05 ──→ T09 ──→ T17 (builder)
                  └──→ T12 (passes) ──→ T22 (pipeline wire)
     └→ T07 ──→ T10 ──→ T19 (accuracy)
T03 ──→ T04 ────────────────────────────────────────────┘
T01 ──→ T06 ──→ T14 ──→ T18 (backend swap)
```

Critical path: **T01 → T05 → T09 → T17 → T22 → T24** ≈ 11h sequential.

---

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PyTorch reference fns diverge from NumPy for edge cases (bf16, reductions) | Medium | High | Cross-check both on 5 shapes per op during T07; keep NumPy shim in parallel until T24 confirms |
| `shape_rule` for complex ops (attention, gather, embedding) is wrong | Medium | Medium | Add per-op shape unit tests in Phase C before Phase E cutover |
| `TemplateRouter` misroutes fused ops (e.g. matmul+relu) | Low | High | Port existing template selection tests verbatim; require identical Triton source output |
| `KernelCache` refactor breaks fast-path dispatch latency | Medium | High | Keep `_generic_cache` dict; benchmark before/after T20 |
| `SSAValidator` too strict — rejects valid IRs | Low | Medium | Start with `_check_op_names` only; add checks incrementally |
| Circular import between `catalog.py` and `registry.py` | Low | Low | `registry.py` imports `catalog.py` only; never the reverse |
| Phase B annotation (45 ops × 4 fields) introduces regressions | Low | Medium | All new fields are additive (`None` default); no existing field changes |

---

## 12. Validation Discipline

### 12.1 Contract

> **The active architecture rewrite is only acceptable if the validation slices stay green.**  
> Active docs should describe the target architecture directly; implementation-bridge notes should be minimized and isolated.

### 12.2 Engineering discipline

- keep public APIs stable only where the active mainline still uses them
- remove dead compatibility shims once replacement paths are proven
- prefer rewriting tests to canonical current behavior over preserving migration-only assertions
- record validation checkpoints in Stage 7 planning docs when architecture-facing changes land

### 12.3 Validation checkpoints

```bash
cd /home/blueyi/workspace/repos/arke
source ~/.venvs/arke/bin/activate
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

---

## Appendix A: File Map

### New Files

```
arke/
├── ir/ops/
│   └── registry.py               # OpRegistry + REGISTRY singleton
├── compiler/
│   ├── __init__.py
│   ├── pipeline.py               # Pipeline + CompilationResult
│   ├── shape_engine.py           # ShapeInferenceEngine
│   ├── template_router.py        # TemplateRouter
│   ├── ssa_validator.py          # SSAValidator
│   └── passes/
│       ├── __init__.py
│       ├── base.py               # Pass, PassContext, PassResult, HardwareProfile
│       ├── analysis.py           # SSAValidationPass, ShapeInferencePass, TypeCheckPass
│       ├── transform.py          # FusionPass, TilingPass
│       └── lowering.py           # TritonCodegenPass
├── backend/
│   ├── protocol.py               # ArkeBackend Protocol + BackendArtifact + CompiledKernel
│   └── registry.py               # BackendRegistry
└── engine/
    └── semantic_interpreter.py   # SemanticInterpreter + check_numerical shim
```

### Modified Files

```
arke/ir/ops/catalog.py              — extend OpDefinition; add 4 new classes; annotate 45 ops
arke/ir/builder.py                  — use ShapeInferenceEngine (Phase E)
arke/backend/triton_backend.py      — add lower()/compile() adapters; swap template engine
arke/engine/accuracy.py             — use SemanticInterpreter shim
arke/integration/kernel_cache.py    — refactor _build_ir() to use parser pipeline
benchmarks/baselines/arke_runner.py — simplify to KernelCache.run_op()
```

### Deprecated (thin wrappers, kept for one release)

```
arke/ir/shape_inference.py           — delegates to ShapeInferenceEngine
arke/engine/numerical_check.py       — delegates to check_numerical
arke/backend/triton_template_engine.py — delegates to TemplateRouter
```

---

## Appendix B: Adding a New Op After Migration

Under the target architecture, adding `gelu_approx`:

**Step 1: Add to `catalog.py` (~10 lines)**

```python
def _ref_gelu_approx(inputs: dict, attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))

GELU_APPROX = _register(OpDefinition(
    name="gelu_approx",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = 0.5*X*(1+tanh(sqrt(2/pi)*(X+0.044715*X^3)))",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise",
                                extra_ctx={"op_variant": "gelu_approx"}),
    reference_impl=ReferenceImpl(fn=_ref_gelu_approx),
    input_gen=InputGen(distributions={"X": "normal"}),
))
```

**Step 2:** If `elementwise.j2` already handles `op_variant`, no template change needed.

**Result:** Shape inference, numerical validation, template routing, benchmark runner, and kernel cache all work automatically. **0 other files modified.**

---

*End of document — Arke Compiler Infrastructure Design v0.1.0*
