# G6 Exit Criteria v2 — Architecture Completeness Edition

> **Document type:** Gate Exit Criteria (Updated)
> **Version:** 2.0
> **Gate:** G6 — Arke Lang & IR Completeness
> **Author:** Arke Architecture Team
> **Created:** 2026-04-06
> **Status:** Active — supersedes G6 section in `stage1-gate-design.md`
>
> **Key change:** G6 now validates not just "can it run" but "is the foundation solid".
> Performance/correctness bars are unchanged; architecture completeness is added as a new mandatory gate.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Updated Exit Criteria — G6 PASS Combined](#2-updated-exit-criteria--g6-pass-combined)
3. [G6-LI Updated — Lang & IR Completeness](#3-g6-li-updated--lang--ir-completeness)
4. [G6-ARCH New — Architecture Completeness](#4-g6-arch-new--architecture-completeness)
5. [Verification Plan](#5-verification-plan)
6. [G6 → G7/G8 Dependency Update](#6-g6--g7g8-dependency-update)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Risk Assessment](#8-risk-assessment)
9. [Appendix: Comparison with Original G6](#9-appendix-comparison-with-original-g6)

---

## 1. Executive Summary

### From "Can Run" to "Solid Foundation"

G0–G5 proved that Arke works: the pipeline runs, correctness is high, LLM agents can generate kernels. G6 already PASSED under the original criteria (commit `fd2cbe0`, 9/9, 46/46 E2E correct, 422 tests passed). So why update the criteria?

**Because "passing" and "production-ready" are not the same thing.**

The original G6 verified that Arke can handle all 45 ops across all shapes. What it did NOT verify is that Arke's internal architecture is clean enough to support the G7/G8 roadmap — autonomous engineering, multi-model E2E, and eventually multi-hardware (Stage 2). Specifically:

| Problem | Impact |
|:--------|:-------|
| Op knowledge split across 6 files | Adding one op requires ~100-line changes in 6 files |
| NumPy reference functions with inconsistent signatures | Fused graph validation is impossible without a unified interpreter |
| Triton-specific concepts leaking into StrategyIR | `num_warps`/`num_stages` in IR breaks Stage 2 (Ascend) |
| No SSA guarantees in IR | LLM-generated IR can silently produce incorrect dataflow |
| No Backend Protocol | Switching from Triton to MLIR requires rearchitecting the pipeline |
| Static shapes only in `.ak` | G7 requires dynamic shape for real LLM inference (S=1 decode vs S=2048 prefill) |

**G6 v2 adds a new mandatory gate — G6-ARCH — that addresses these structural risks before they become G7/G8 blockers.**

### What G6 v2 Is NOT

- G6-ARCH does **not** require full MLIR integration (that's Stage 2-3)
- G6-ARCH does **not** require dynamic shape JIT compilation (that's G7-G8)
- G6-ARCH does **not** raise performance/correctness bars (those are inherited unchanged)
- G6-ARCH is **not** a full rewrite — it's surgical refactoring with full test coverage as the safety net

### G6 v2 Positioning in the Stage 1 Roadmap

```
G5 ✅  → Arke runs for all basic ops (OT0-2), GPT-2 E2E correct
           │
G6 v2 ⬜  → Arke runs for ALL 45 ops + architecture is stable and extensible
           │   ↳ Performance/correctness: unchanged bars
           │   ↳ Architecture: OpRegistry, Pass Infra, SemanticInterpreter,
           │     SSA Validator, Backend Abstraction, Lang v2.0 where clause
           │
G7 ⬜     → LLM Agent autonomously generates strategies, LLaMA-2 + DS-V2 E2E
           │   ↳ Depends on: dynamic shape (G6-ARCH.8 foundation),
           │     Pass Infra (G6-ARCH.4), Backend Abstraction (G6-ARCH.7)
           │
G8 ⬜     → 4-model E2E, BL5 regression, Stage 1 Final
```

---

## 2. Updated Exit Criteria — G6 PASS Combined

### G6 PASS Combined Criteria v2

```
AND ALL:

  ── Performance & Correctness (unchanged from v1) ──────────────────────────
  [1] L1 BL5 correctness: 100%(ST1-3) + ≥95%(ST4, excl. OOM) for all OT0-OT4
  [2] L1 BL5 performance weighted_score ≥ 0.83
        weighted_score = 0.25×score(OT0-1) + 0.30×score(OT2)
                       + 0.20×score(OT3) + 0.25×score(OT4)
        where score(OTn) = geomean pass rate for that OT group (0.0~1.0)
  [3] L2 BL5: ≥3/4 fusion combinations pass
        (matmul+relu/gelu, swiglu/geglu, linear+cross_entropy, QKV+flash_attn)

  ── Lang & IR Completeness (updated from v1) ────────────────────────────────
  [4] Lang&IR: G6-LI.1~LI.8 all pass
        (G6-LI.1~LI.6 retained; G6-LI.7 and G6-LI.8 are new additions)

  ── Architecture Completeness (new in v2) ────────────────────────────────────
  [5] Architecture: G6-ARCH.1~ARCH.7 and ARCH.10 pass (ARCH.8, ARCH.9 are MVP-scoped)
        See §4 for full definitions and MVP scope rationale
```

### Quick Reference: All G6 Gate Conditions

| ID | Category | Condition | Threshold | Status |
|:---|:---------|:----------|:----------|:------:|
| [1] | Perf/Correctness | L1 BL5 correctness | 100%(ST1-3) + ≥95%(ST4) | Inherited ✅* |
| [2] | Perf/Correctness | L1 BL5 performance | weighted_score ≥ 0.83 | Inherited ✅* |
| [3] | Perf/Correctness | L2 BL5 fusion | ≥3/4 combinations | Inherited ✅* |
| [4] | Lang & IR | G6-LI.1~LI.8 | All pass | Partially ✅, new items ⬜ |
| [5] | Architecture | G6-ARCH.1~ARCH.7, ARCH.10 | All pass | ⬜ New |

> *✅ Inherited: passed under original G6 commit `fd2cbe0`. Must not regress under v2 implementation.

---

## 3. G6-LI Updated — Lang & IR Completeness

### Retained Criteria (G6-LI.1~LI.6)

These criteria are unchanged in definition. Their status must be re-verified after architecture changes.

| ID | Criterion | Verification Command | Pass Condition |
|:---|:----------|:--------------------|:--------------|
| **G6-LI.1** | All 45 ops expressible and parseable in `.ak` | `for f in examples/*.ak; do arke parse $f; done` | All exit code 0 |
| **G6-LI.2** | `.ak → SemanticIR → StrategyIR → Pass pipeline` full round-trip | `python -m arke.compiler.pipeline --ak examples/<op>.ak --dry-run` | Pass for all 45 ops |
| **G6-LI.3** | `@rationale` annotations preserved through full pipeline | `python scripts/check_rationale_chain.py examples/` | ≥3 verified examples |
| **G6-LI.4** | Token efficiency: `.ak` ≤ Triton line count | `python scripts/token_compare.py --ak examples/ --triton benchmarks/triton/` | OT0-OT4: `.ak` lines ≤ Triton lines |
| **G6-LI.5** | Python interop IR round-trip | `python -m pytest tests/test_ir_roundtrip.py -v` | All 45 ops pass |
| **G6-LI.6** | Grammar completeness: 0 parse failures across all `.ak` files | `arke parse examples/ --strict` | 0 failures |

> **G6-LI.2 update note:** The pipeline now explicitly includes Pass Infrastructure. The pass pipeline (even with no-op passes) must be traversable for all 45 ops, validating the Pass Protocol integration.

### New Criteria (G6-LI.7~LI.8)

These are new additions based on architectural decisions made after the original G6 definition.

| ID | Criterion | Verification Command | Pass Condition |
|:---|:----------|:--------------------|:--------------|
| **G6-LI.7** | Symbolic shape `.ak` → SemanticIR round-trip with `where` clause | `python -m pytest tests/test_symbolic_shape.py -v` | ≥5 ops with `where` clause parse+IR-represent correctly |
| **G6-LI.8** | Backend-agnostic strategy `.ak` round-trip (no Triton-specific fields in StrategyIR core) | `python scripts/check_backend_agnostic.py examples/` | 0 Triton-specific fields in `StrategyIR.decisions[]`; `launch_config` is backend extension only |

### G6-LI Rationale for New Items

**G6-LI.7 (symbolic shape):** G7 requires dynamic shape for real LLM inference. Without `where` clause support in G6, G7 would need to implement both the language feature and the optimization logic simultaneously — a high-risk parallel. G6 establishes the parse + IR representation layer; G7 adds the codegen + runtime layer.

**G6-LI.8 (backend-agnostic strategy):** The StrategyIR currently contains `num_warps`, `num_stages`, and autotune configs that are Triton/NVIDIA-specific. Before G7 adds more strategy decisions, the core StrategyIR schema must be cleaned up so that Stage 2 (Ascend backend) can reuse the same IR without Triton pollution.

---

## 4. G6-ARCH New — Architecture Completeness

### G6-ARCH Priority Classification

Not all architecture items have equal urgency. They are classified by when they become blockers:

| Priority | Meaning | G6 Required? |
|:---------|:--------|:------------|
| **P0-G6** | Blocks G6 perf/correctness goals or creates G7 hard blockers | ✅ Must complete |
| **P1-G6** | Needed for G6 architecture health; G7 depends on it | ✅ Must complete |
| **P2-G7** | Important but G7 can start without it; recommended in G6 | ⚠️ MVP scope |
| **P3-G8** | Required for G8; G7 doesn't strictly need it | ❌ Defer to G7/G8 |

### G6-ARCH Criteria Table

| ID | Criterion | Verification | Priority | Gate |
|:---|:----------|:------------|:---------|:-----|
| **G6-ARCH.1** | Arke Lang Spec v2.0 document finalized | `docs/spec/arke-lang-spec-v2.md` exists + covers where clause, multi-return, type inference, backend-agnostic strategy, annotation extension | P1-G6 | ✅ Required |
| **G6-ARCH.2** | Arke IR Multi-Layer Architecture spec finalized | `docs/spec/arke-ir-spec-v2.md` exists + defines Layer 4/3/2/1 with interface contracts | P1-G6 | ✅ Required |
| **G6-ARCH.3** | OpRegistry implemented, replaces 6 separate op lists | `python scripts/verify_op_registry.py` — adding a new op requires changes to ≤2 files | P0-G6 | ✅ Required |
| **G6-ARCH.4** | Pass Infrastructure skeleton implemented | `python -m pytest tests/test_pass_infra.py -v` — Pass protocol + Pipeline runnable with ≥2 example passes | P1-G6 | ✅ Required |
| **G6-ARCH.5** | SemanticInterpreter implemented | `python -m pytest tests/test_semantic_interpreter.py -v` — replaces `numerical_check.py`, all 45 ops correct | P0-G6 | ✅ Required |
| **G6-ARCH.6** | SSA Validator implemented | `python -m pytest tests/test_ssa_validator.py -v` — all 45 ops' IR round-trips pass SSA validation | P1-G6 | ✅ Required |
| **G6-ARCH.7** | Backend Abstraction interface defined | `python -m pytest tests/test_backend_protocol.py -v` — `ArkeBackend` protocol exists + `TritonBackend` implements it | P1-G6 | ✅ Required |
| **G6-ARCH.8** | Arke Lang v2.0 key features implemented | `arke parse examples/matmul_symbolic.ak` — `where` clause parses + SemanticIR `symbolic_dims` field populated | P2-G7 | ⚠️ MVP scope |
| **G6-ARCH.9** | Layer 3/2/1 spec documents complete | `docs/spec/arke-ir-layer3-spec.md`, `layer2-spec.md`, `layer1-spec.md` + MLIR mapping updated | P2-G7 | ⚠️ MVP scope |
| **G6-ARCH.10** | All existing tests do not regress | `python -m pytest tests/ -q` — ≥422 passed, ≤6 skipped, 0 new failures | P0-G6 | ✅ Required |

### G6-ARCH.8 and ARCH.9 MVP Scope Rationale

**G6-ARCH.8 (Lang v2.0 implementation):** Full Arke Lang v2.0 implementation (multi-return, type inference, etc.) is substantial work. For G6, the MVP is: `where` clause syntax is parseable and produces `symbolic_dims` in SemanticIR. The remaining v2.0 features (multi-return, type inference, full backend-agnostic strategy) are design-complete in ARCH.1 spec, but implementation can extend into G7. Gate condition: ARCH.8 passes if `where` clause basic implementation works.

**G6-ARCH.9 (Layer 3/2/1 spec):** Layer 3 (Compute IR), Layer 2 (Hardware IR), and Layer 1 (Instruction IR) are defined as spec-only in Stage 1. They need interface contracts and MLIR mapping documented for Stage 2 planning, but they do NOT need implementation in G6. Gate condition: ARCH.9 passes if the three spec documents exist with interface definitions.

### G6-ARCH Detail Definitions

#### G6-ARCH.3 — OpRegistry

**Problem it solves:** Currently, adding one op requires modifying `catalog.py`, `shape_inference.py`, `numerical_check.py`, `triton_template_engine.py`, `kernel_cache.py`, and potentially `semantic.py` — 6 files, ~100 lines.

**Required outcome:**
```python
# Single registration point:
@op_registry.register("cross_attention")
class CrossAttentionOp:
    schema = OpSchema(inputs=[...], outputs=[...])
    shape_fn = lambda inputs: ...
    reference_fn = lambda inputs: torch.nn.functional.scaled_dot_product_attention(...)
    template = "cross_attention.py.j2"
```

**Verification:**
```bash
python scripts/verify_op_registry.py
# Expected output:
# ✅ All 45 ops registered in OpRegistry
# ✅ No op-specific logic found in shape_inference.py (if/elif removed)
# ✅ No op-specific logic found in numerical_check.py (if/elif removed)
# ✅ Adding new op: file change count = 1 (ops/new_op.py only)
```

#### G6-ARCH.4 — Pass Infrastructure

**Problem it solves:** Currently, there's no formal mechanism to run transformation passes over the IR. G7's autonomous optimization loop requires a structured pass pipeline to run, inspect, and compose transformations.

**Required outcome:**
```python
# Pass protocol
class ArkePass(Protocol):
    name: str
    def run(self, ir: SemanticIR, ctx: PassContext) -> SemanticIR: ...
    def verify(self, ir: SemanticIR) -> list[str]: ...  # returns error list

# Pipeline
pipeline = PassPipeline([
    ShapeInferencePass(),
    SSAValidationPass(),
    RationalePreservationPass(),
])
result = pipeline.run(semantic_ir)
```

**Verification:**
```bash
python -m pytest tests/test_pass_infra.py -v
# Must pass: PassProtocol interface test, PassPipeline compose test,
# ShapeInferencePass runs on all 45 ops, SSAValidationPass runs on all 45 ops
```

#### G6-ARCH.5 — SemanticInterpreter

**Problem it solves:** `numerical_check.py` has per-op hand-written NumPy functions with inconsistent signatures (some use `inputs["X"]`, others use `inputs.get("X", fallback)` which can hide IR construction bugs). Cannot validate fused graphs. SemanticInterpreter executes IR graph nodes using PyTorch eager, which automatically handles fused graphs.

**Required outcome:**
```python
# Execute any SemanticIR graph using PyTorch eager
interpreter = SemanticInterpreter()
outputs = interpreter.run(semantic_ir, inputs={"X": torch.randn(128, 256)})
# Works for single ops AND fused graphs
```

**Verification:**
```bash
python -m pytest tests/test_semantic_interpreter.py -v
# Must pass: all 45 single ops, ≥4 fused graphs (L2 BL5 combinations),
# Numerical tolerance: fp16 rtol=1e-2 atol=1e-3 (matches current numerical_check.py standard)
```

#### G6-ARCH.6 — SSA Validator

**Problem it solves:** Without SSA guarantees, LLM-generated IR can have use-before-define errors, duplicate variable names, or dead computations that look valid but produce wrong results.

**Required outcome:**
```python
validator = SSAValidator()
errors = validator.validate(semantic_ir)
# Returns: [] on valid IR, or list of SSAError with location + message
# Error types: use_before_define, duplicate_def, dead_code_with_side_effects, shape_mismatch
```

**Verification:**
```bash
python -m pytest tests/test_ssa_validator.py -v
# Must pass: all 45 ops valid IR passes validation (0 errors),
# ≥5 intentionally invalid IR examples correctly rejected with meaningful error messages
```

#### G6-ARCH.7 — Backend Abstraction

**Problem it solves:** `ArkePipeline.run()` hardcodes `from arke.backend.triton_backend import TritonBackend`. When Stage 2 adds an Ascend backend, this requires rearchitecting the pipeline instead of simply adding a new backend.

**Required outcome:**
```python
# Protocol (backend-agnostic interface)
class ArkeBackend(Protocol):
    backend_id: str
    supported_targets: list[str]
    def translate(self, semantic_ir: SemanticIR, strategy_ir: StrategyIR) -> BackendArtifact: ...
    def compile(self, artifact: BackendArtifact) -> CompiledKernel: ...
    def profile(self, kernel: CompiledKernel, inputs: dict) -> ProfilingResult: ...

# Existing TritonBackend adapted to implement ArkeBackend
class TritonBackend:
    backend_id = "triton"
    supported_targets = ["nvidia_ampere", "nvidia_hopper"]
    def translate(self, ...): ...  # wraps existing TritonTemplateEngine
    def compile(self, ...): ...
    def profile(self, ...): ...

# Pipeline uses backend via protocol, not concrete class
pipeline = ArkePipeline(backend=TritonBackend())
```

**Verification:**
```bash
python -m pytest tests/test_backend_protocol.py -v
# Must pass: ArkeBackend protocol defined, TritonBackend implements protocol,
# MockBackend (test stub) implements protocol and runs through pipeline,
# Pipeline works with both TritonBackend and MockBackend
```

---

## 5. Verification Plan

### 5.1 Automated Verification Script

```bash
#!/usr/bin/env bash
# G6 v2 full gate check
# Usage: ./scripts/verify_g6_v2.sh [--skip-bench]

set -e
source ~/.venvs/arke/bin/activate
cd /home/blueyi/workspace/repos/arke

echo "=== G6 v2 Gate Verification ==="

# ── [1] L1 BL5 Correctness ──────────────────────────────────────────────────
echo "[1] L1 BL5 Correctness..."
python -m benchmarks.gate G6 --tier 2 --check correctness
# Pass: 100% ST1-3, ≥95% ST4 for all OT0-OT4

# ── [2] L1 BL5 Performance ──────────────────────────────────────────────────
echo "[2] L1 BL5 Performance..."
python -m benchmarks.gate G6 --tier 2 --check performance
# Pass: weighted_score ≥ 0.83

# ── [3] L2 BL5 Fusion ───────────────────────────────────────────────────────
echo "[3] L2 BL5 Fusion..."
python -m benchmarks.gate G6 --tier 2 --check fusion
# Pass: ≥3/4 fusion combinations

# ── [4] G6-LI Lang & IR Completeness ────────────────────────────────────────
echo "[4] G6-LI Completeness..."
# LI.1: Parse all .ak files
for f in examples/*.ak; do arke parse "$f" || { echo "FAIL: $f"; exit 1; }; done
# LI.2: Full pipeline round-trip
python -m pytest tests/test_pipeline_roundtrip.py -v -x
# LI.3: @rationale preservation
python scripts/check_rationale_chain.py examples/ --min-examples 3
# LI.4: Token efficiency
python scripts/token_compare.py --ak examples/ --triton benchmarks/triton/ --check
# LI.5: IR round-trip
python -m pytest tests/test_ir_roundtrip.py -v -x
# LI.6: Grammar completeness
arke parse examples/ --strict
# LI.7: Symbolic shape round-trip
python -m pytest tests/test_symbolic_shape.py -v -x
# LI.8: Backend-agnostic strategy
python scripts/check_backend_agnostic.py examples/

# ── [5] G6-ARCH Architecture Completeness ───────────────────────────────────
echo "[5] G6-ARCH Architecture..."
# ARCH.1: Lang Spec v2.0
test -f docs/spec/arke-lang-spec-v2.md || { echo "FAIL: arke-lang-spec-v2.md missing"; exit 1; }
# ARCH.2: IR Multi-Layer Spec
test -f docs/spec/arke-ir-spec-v2.md || { echo "FAIL: arke-ir-spec-v2.md missing"; exit 1; }
# ARCH.3: OpRegistry
python scripts/verify_op_registry.py
# ARCH.4: Pass Infrastructure
python -m pytest tests/test_pass_infra.py -v -x
# ARCH.5: SemanticInterpreter
python -m pytest tests/test_semantic_interpreter.py -v -x
# ARCH.6: SSA Validator
python -m pytest tests/test_ssa_validator.py -v -x
# ARCH.7: Backend Abstraction
python -m pytest tests/test_backend_protocol.py -v -x
# ARCH.10: No regression
python -m pytest tests/ -q --tb=short
# Pass: ≥422 passed, ≤6 skipped, 0 new failures

echo "=== G6 v2 PASS ==="
```

### 5.2 Per-Criterion Verification Details

#### [1] L1 BL5 Correctness Verification

```bash
# Run BL5 L1 benchmark, check correctness column
python -m benchmarks.gate G6 --tier 2 --layer l1 --report correctness

# Manual check per OT group:
python -m benchmarks.bench_l1 --ot 0 --shapes st1,st2,st3,st4 --check-correctness
python -m benchmarks.bench_l1 --ot 1 --shapes st1,st2,st3,st4 --check-correctness
python -m benchmarks.bench_l1 --ot 2 --shapes st1,st2,st3,st4 --check-correctness
python -m benchmarks.bench_l1 --ot 3 --shapes st1,st2,st3,st4 --check-correctness
python -m benchmarks.bench_l1 --ot 4 --shapes st1,st2,st3,st4 --check-correctness --oom-ok
```

**Pass condition:**
- OT0-OT3 × ST1-ST3: 100% (zero failures)
- OT4 × ST1-ST4: 100% excluding OOM-marked shapes
- OT0-OT4 × ST4: ≥95% (OOM excluded from denominator)

#### [2] L1 BL5 Performance Verification

```bash
python -m benchmarks.gate G6 --tier 2 --layer l1 --report performance
# Output: weighted_score = X.XX (must be ≥ 0.83)

# Component breakdown:
# score(OT0-1) = geomean(relu/gelu/silu.../softmax/layernorm/... vs P1 FlagGems)
# score(OT2)   = geomean(matmul vs P0 cuBLAS; others vs P3 torch.eager)
# score(OT3)   = geomean(swiglu/rope vs P1 Liger/FlagGems)
# score(OT4)   = geomean(flash_attn/GQA vs P1 FlashAttn-2)
```

#### [3] L2 BL5 Fusion Verification

```bash
python -m benchmarks.bench_l2 --combinations all --check-fusion-benefit
# Each combination: Arke fused ≥ threshold vs unfused/P1 baseline
# matmul+relu/gelu:     ≥ 1.05× unfused (P3)
# swiglu/geglu:         ≥ 0.90× Liger (P1)
# linear+cross_entropy: ≥ 1.05× unfused (P3)
# QKV+flash_attention:  ≥ 0.80× FlashAttn-2 (P1)
# Pass: ≥3/4 combinations meet threshold
```

#### G6-ARCH.3 OpRegistry Verification

```bash
python scripts/verify_op_registry.py
# Checks:
# 1. All 45 op names present in OpRegistry
# 2. Each op has: schema, shape_fn, reference_fn, template
# 3. shape_inference.py has no op-specific if/elif chains
# 4. numerical_check.py has no _numpy_* functions
# 5. kernel_cache.py _build_ir() uses parser, not manual IR construction
# 6. triton_template_engine.py _select_template() uses registry lookup, not if/elif
```

#### G6-ARCH.5 SemanticInterpreter Verification

```bash
python -m pytest tests/test_semantic_interpreter.py -v
# Required test cases:
# - test_single_op_all_45: each op runs through interpreter, matches PyTorch reference
# - test_fused_matmul_relu: fused graph interpreter output matches unfused result
# - test_fused_swiglu: SwiGLU fused output matches reference
# - test_fused_linear_cross_entropy: fused loss matches torch.nn.CrossEntropyLoss
# - test_attention_variants: FA/GQA/MLA interpreter output matches reference
# Tolerance: fp16 rtol=1e-2, atol=1e-3
```

#### G6-ARCH.6 SSA Validator Verification

```bash
python -m pytest tests/test_ssa_validator.py -v
# Valid IR tests (must pass with 0 errors):
# - test_valid_all_45_ops: generate IR for each op, validate
# - test_valid_fused_graphs: validate 4 L2 BL5 fusion combinations
# Invalid IR tests (must be rejected with correct error type):
# - test_reject_use_before_define: ref to undefined var → SSAError.use_before_define
# - test_reject_duplicate_def: same var defined twice → SSAError.duplicate_def
# - test_reject_shape_mismatch: matmul([128,256], [512,128]) → SSAError.shape_mismatch
# - test_reject_missing_output: return refers to non-existent node → SSAError
# - test_reject_cycle: circular dependency in graph → SSAError.cycle_detected
```

---

## 6. G6 → G7/G8 Dependency Update

### 6.1 What G7 Depends on from G6 v2

G7's core goal is **Autonomous Engineering**: LLM Agent generates strategies without human strategy blocks. The following G6 v2 outputs are direct prerequisites:

| G6 v2 Output | G7 Dependency | Why Critical |
|:-------------|:-------------|:------------|
| **G6-ARCH.3 OpRegistry** | G7-AE.1 (auto strategy generation) | Agent needs a single authoritative source to query op semantics when generating strategy |
| **G6-ARCH.4 Pass Infrastructure** | G7-AE.2 (iterative optimization ≥3 rounds) | Each optimization round is a pass; the pipeline must be composable |
| **G6-ARCH.5 SemanticInterpreter** | G7-AE.2 + G7 BL5 correctness | Correctness check in the agent loop uses interpreter, not hand-coded NumPy |
| **G6-ARCH.6 SSA Validator** | G7-AE.1 (auto IR generation) | LLM-generated IR must be validated before execution; without SSA guarantees, agent loop silently produces wrong outputs |
| **G6-ARCH.7 Backend Abstraction** | G7 BL5 inherit (no regression) | Pipeline must not regress when TritonBackend is refactored behind the protocol |
| **G6-LI.7 Symbolic shape** | G7-AE.1 + G7 LLaMA-2/DS-V2 E2E | Real LLM inference requires dynamic seq_len; G6 must establish the IR representation layer |
| **G6-LI.8 Backend-agnostic strategy** | G7 DS-V2 (MoE) + G8 Ascend planning | StrategyIR must be clean before G7 adds MoE-specific strategy decisions |

### 6.2 What G8 Verifies from G6 Infrastructure

| G6 v2 Infrastructure | G8 Verification | How G8 Uses It |
|:--------------------|:----------------|:--------------|
| **OpRegistry** | BL5 full regression suite | G8 regression CI calls OpRegistry to enumerate all ops for batch testing |
| **SemanticInterpreter** | 4-model E2E correctness check | G8 uses interpreter to validate correctness before running GPU benchmark |
| **Pass Infrastructure** | Arke vs LLM-direct comparison | G8 comparison pipeline uses pass pipeline to normalize both Arke and LLM-direct outputs |
| **SSA Validator** | IR Spec v1.0 freeze (D8-IR1) | G8 freezes IR spec only after SSA validator confirms all 45 ops produce valid IR |
| **Backend Abstraction** | Stage 2 preparation | G8's language-decision.md assessment requires backend protocol to measure dispatch overhead |

### 6.3 Dynamic Shape: What G6 Must Establish for G7

Dynamic shape is a **G7 prerequisite** but G6 does the foundation work:

```
G6 (foundation layer):             G7 (execution layer):
─────────────────────              ─────────────────────
.ak where clause parse        →    LLM generates where clause for real ops
SemanticIR symbolic_dims      →    Shape analysis pass runs on LLM-generated IR
Shape constraint representation→    Constraint-driven strategy selection

G6 does NOT need:                  G7 adds:
─────────────────────              ─────────────────────
JIT compilation                →    Runtime bucket selection
Shape-parametric codegen       →    Triton @triton.jit + tl.constexpr
Runtime dispatch               →    KernelCache by shape bucket
```

**G6 minimum for dynamic shape (G6-LI.7 + G6-ARCH.8 MVP):**
1. `where` clause parses without error
2. SemanticIR node has `symbolic_dims: {"M": {"dynamic": true, "range": [1, 4096]}}` or equivalent
3. Shape inference pass propagates symbolic dims through matmul/attention compute graphs
4. Round-trip: `.ak` → SemanticIR → JSON → SemanticIR preserves symbolic dims

**What G6 does NOT need for dynamic shape:**
- Triton codegen for symbolic shapes (that's G7)
- Runtime JIT compilation (that's G7)
- LLM agent integration (that's G7)

---

## 7. Implementation Roadmap

### 7.1 Phase Structure

G6 v2 work is organized into 4 phases. Phases A and B (from original G6) are already complete. Phases C and D are the new architecture work.

```
Phase A ✅  .ak files for all 45 ops + grammar fixes           (DONE: commit fd2cbe0)
Phase B ✅  OT3/OT4 Triton templates + E2E correctness         (DONE: 46/46 E2E correct)
Phase C ⬜  Architecture refactoring                            (NEW in v2)
Phase D ⬜  Spec documents + validation + non-regression        (NEW in v2)
```

### 7.2 Phase C — Architecture Refactoring

**Goal:** Implement the infrastructure that makes G6-ARCH.3~ARCH.7 pass.

**Can run in parallel:**
- Track C1: OpRegistry + SemanticInterpreter (replaces 6-file pattern)
- Track C2: Pass Infrastructure + SSA Validator (new pipeline layer)
- Track C3: Backend Abstraction (wraps TritonBackend behind protocol)

**Dependency:** C1 must complete before C2 can integrate the interpreter into passes. C3 is independent.

| Task | Track | Description | Estimate (LLM Agent) | Priority |
|:-----|:------|:------------|:---------------------|:--------|
| C1.1 | C1 | Design OpSchema dataclass + OpRegistry class | 0.5d | P0 |
| C1.2 | C1 | Migrate all 45 ops from catalog.py to OpRegistry | 1d | P0 |
| C1.3 | C1 | Remove op-specific if/elif from shape_inference.py | 0.5d | P0 |
| C1.4 | C1 | Implement SemanticInterpreter (PyTorch eager executor) | 1d | P0 |
| C1.5 | C1 | Migrate numerical_check.py to use SemanticInterpreter | 0.5d | P0 |
| C1.6 | C1 | Update kernel_cache.py to use parser instead of _build_ir() | 0.5d | P1 |
| C1.7 | C1 | Update triton_template_engine.py to use registry lookup | 0.5d | P0 |
| C2.1 | C2 | Define ArkePass protocol + PassContext + PassPipeline | 0.5d | P1 |
| C2.2 | C2 | Implement ShapeInferencePass (wraps shape_inference.py) | 0.5d | P1 |
| C2.3 | C2 | Implement SSAValidator + SSAValidationPass | 1d | P1 |
| C2.4 | C2 | Implement RationalePreservationPass | 0.5d | P1 |
| C2.5 | C2 | Integrate PassPipeline into ArkePipeline.run() | 0.5d | P1 |
| C3.1 | C3 | Define ArkeBackend protocol + BackendArtifact hierarchy | 0.5d | P1 |
| C3.2 | C3 | Wrap TritonBackend to implement ArkeBackend | 0.5d | P1 |
| C3.3 | C3 | Update ArkePipeline to use backend via protocol | 0.5d | P1 |
| C3.4 | C3 | Implement MockBackend for testing | 0.5d | P1 |

**Phase C total estimate:** ~9-10 days (LLM Agent, parallelized to ~5-6 calendar days)

### 7.3 Phase D — Spec Documents + where Clause + Validation

**Goal:** Complete G6-ARCH.1, ARCH.2, ARCH.8 (MVP), ARCH.9 (MVP), G6-LI.7, G6-LI.8, and full non-regression.

| Task | Description | Estimate (LLM Agent) | Priority |
|:-----|:------------|:---------------------|:--------|
| D1 | Write `docs/spec/arke-lang-spec-v2.md` | 1d | P1 |
| D2 | Write `docs/spec/arke-ir-spec-v2.md` (Layer 4 upgraded, Layer 3/2/1 interfaces) | 1.5d | P1 |
| D3 | Implement `where` clause in Lark grammar | 0.5d | P2 |
| D4 | Add `symbolic_dims` field to SemanticIR + converter | 0.5d | P2 |
| D5 | Add shape propagation for symbolic dims in ShapeInferencePass | 1d | P2 |
| D6 | Write `tests/test_symbolic_shape.py` (G6-LI.7 verification) | 0.5d | P2 |
| D7 | Write `scripts/check_backend_agnostic.py` (G6-LI.8 verification) | 0.5d | P1 |
| D8 | Full non-regression run, fix any regressions | 1d | P0 |
| D9 | Update Layer 3/2/1 spec stubs (ARCH.9 MVP) | 1d | P2 |

**Phase D total estimate:** ~7-8 days (LLM Agent, some parallelism possible)

### 7.4 Key Milestones

| Milestone | Completion Condition | Target (LLM Agent days) |
|:----------|:--------------------|:-----------------------|
| **M1: OpRegistry live** | C1.1-C1.5 done; `verify_op_registry.py` passes | Day 3 |
| **M2: Pass Pipeline live** | C2.1-C2.5 done; `test_pass_infra.py` passes | Day 5 |
| **M3: Backend Protocol live** | C3.1-C3.4 done; `test_backend_protocol.py` passes | Day 5 |
| **M4: Non-regression confirmed** | 422+ tests pass after Phase C | Day 7 |
| **M5: Spec documents done** | D1+D2 complete; peer review done | Day 9 |
| **M6: where clause MVP** | D3-D6 done; `test_symbolic_shape.py` passes | Day 11 |
| **M7: G6 v2 PASS** | All criteria verified; gate check passes | Day 13-15 |

### 7.5 Parallelism Strategy

```
Day 1-3:   [C1: OpRegistry + SemanticInterpreter] ║ [C3: Backend Abstraction]
Day 3-5:   [C2: Pass Infra + SSA Validator]       ║ [D1: Lang Spec v2.0]
Day 5-7:   [C2.5: Integrate passes into pipeline] ║ [D2: IR Spec v2.0]
Day 7-9:   [D8: Non-regression + fixes]           ║ [D9: Layer spec stubs]
Day 9-11:  [D3-D6: where clause MVP]              ║ [D7: backend-agnostic check]
Day 11-15: [Final verification + gate check]
```

---

## 8. Risk Assessment

### 8.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| **OpRegistry migration breaks existing tests** | Medium | High | Migrate incrementally; keep old catalog.py as alias during migration; run test suite after each op batch |
| **SemanticInterpreter has different numerical behavior than numpy** | Medium | High | Start with loose tolerance (fp16 rtol=1e-2); document known differences; use PyTorch eager as ground truth |
| **Pass Infrastructure integration causes pipeline regression** | Low | High | New passes start as no-ops; add to pipeline only after unit tests pass; feature flag to bypass pass pipeline |
| **SSA Validator too strict (false positives on valid IR)** | Medium | Medium | Start with `strict=False` mode for warnings; promote to errors only after all 45 ops confirmed valid |
| **where clause parsing conflicts with existing grammar** | Low | Medium | Use Lark's `?` optional rule; `where` clause is opt-in; old `.ak` files unchanged |
| **Backend Protocol refactor changes TritonBackend behavior** | Low | High | TritonBackend wraps existing code without logic changes; profile before/after to confirm identical output |
| **Phase C takes longer than estimated, delays G7** | Medium | Medium | G6-ARCH items are P0/P1; can ship G6 without ARCH.8/ARCH.9 (they're P2-G7 scope); G7 can start with G6-ARCH.1~7 |
| **422 test count assumption wrong (tests added after fd2cbe0)** | Low | Low | Use `pytest --co -q | wc -l` to get actual count before finalizing threshold |

### 8.2 Rollback Plan

If Phase C causes unexpected regressions:

1. **OpRegistry rollback:** Keep `catalog.py` as-is; OpRegistry becomes a read-only view over catalog.py (adapter pattern, no migration needed)
2. **SemanticInterpreter rollback:** Keep `numerical_check.py` as primary; SemanticInterpreter becomes secondary validation (not gate-blocking)
3. **Pass Pipeline rollback:** Remove pass pipeline integration from `ArkePipeline.run()`; passes become standalone utilities (not gating)
4. **Backend Protocol rollback:** Revert `ArkePipeline` to concrete `TritonBackend` import; protocol exists as future interface without enforcement

**Rollback decision trigger:** If >10 tests regress after Phase C and root cause is not immediately clear → rollback to adapter pattern and proceed with Phase D (spec + where clause) which has no regression risk.

### 8.3 G6 v2 vs Original G6 Risk

The original G6 already PASSED under commit `fd2cbe0`. The risk introduced by v2 is:
- **Low risk:** Performance/correctness bars unchanged; no risk of lowering the bar
- **Medium risk:** Architecture refactoring could introduce regressions in the 422-test suite
- **Low risk:** New spec documents cannot cause regressions
- **Low risk:** where clause is purely additive (opt-in syntax)

**Net assessment:** G6 v2 adds meaningful value (architecture stability for G7/G8) at medium implementation risk, with clear rollback paths. The architecture work is mandatory before G7 because fixing a broken architecture at G7 time (when autonomous agent loops are being built on top) would be much higher cost.

---

## 9. Appendix: Comparison with Original G6

### 9.1 New vs Old PASS Conditions

| # | Original G6 | G6 v2 | Change |
|:--|:-----------|:------|:------|
| [1] | L1 BL5 correctness: 100%(ST1-3) + ≥95%(ST4) | **Same** | Unchanged |
| [2] | L1 BL5 performance weighted_score ≥ 0.83 | **Same** | Unchanged |
| [3] | L2 BL5: ≥3/4 fusion combinations | **Same** | Unchanged |
| [4] | G6-LI.1~LI.6 all pass | **G6-LI.1~LI.8 all pass** | Added LI.7 (symbolic shape), LI.8 (backend-agnostic) |
| [5] | _(not present)_ | **G6-ARCH.1~ARCH.7, ARCH.10 pass** | Entirely new |

### 9.2 G6-LI Comparison

| ID | Original | v2 | Change |
|:---|:---------|:---|:------|
| G6-LI.1 | 45 ops parseable in .ak | Same | Unchanged |
| G6-LI.2 | .ak → SemanticIR → StrategyIR pipeline | .ak → SemanticIR → StrategyIR → **Pass pipeline** | Updated: Pass pipeline now part of definition |
| G6-LI.3 | @rationale preserved through pipeline | Same | Unchanged |
| G6-LI.4 | Token efficiency .ak ≤ Triton | Same | Unchanged |
| G6-LI.5 | Python interop IR round-trip | Same | Unchanged |
| G6-LI.6 | Grammar completeness | Same | Unchanged |
| G6-LI.7 | _(not present)_ | Symbolic shape where clause round-trip | **New** |
| G6-LI.8 | _(not present)_ | Backend-agnostic strategy verification | **New** |

### 9.3 G6-ARCH New Items — Summary Rationale

| ID | Why Added in G6 v2 | Why NOT deferred to G7/G8 |
|:---|:-------------------|:--------------------------|
| ARCH.1 Lang Spec v2.0 | LLM Agent needs canonical spec to generate correct .ak files | Without spec, G7 agent generates inconsistent syntax |
| ARCH.2 IR Multi-Layer Spec | Defines Layer 4/3/2/1 contracts needed for Stage 2 planning | G8 requires Stage 2 preparation (language-decision.md) |
| ARCH.3 OpRegistry | Adding G7/G8 ops (MoE, paged_attn) would require 6-file changes | Each G7 new op multiplies tech debt |
| ARCH.4 Pass Infra | G7 iterative loop IS a pass pipeline | Cannot build G7 optimization loop without this foundation |
| ARCH.5 SemanticInterpreter | G7 agent correctness checks need unified interpreter | LLM-generated fused graphs cannot be validated with per-op NumPy |
| ARCH.6 SSA Validator | LLM will generate invalid IR; must catch before GPU execution | Silent IR bugs in G7 are extremely hard to debug |
| ARCH.7 Backend Abstraction | G8 requires Stage 2 backend planning; TritonBackend must be replaceable | Protocol retrofit at G8 time requires full pipeline retest |
| ARCH.8 (MVP) where clause | G7 needs dynamic shape; G6 must have IR layer ready | LLM cannot write dynamic-shape .ak if language doesn't support it |
| ARCH.9 (MVP) Layer specs | Stage 2 planning (MLIR, LLVM) requires interface contracts | Without layer specs, Stage 2 is blank-page architecture |
| ARCH.10 Non-regression | Refactoring without regression safety net is unacceptable | Obvious requirement; explicitly stated for gate verification |

### 9.4 G6 Status Before vs After v2

| Dimension | Before v2 (fd2cbe0) | After v2 (target) |
|:----------|:-------------------|:------------------|
| Performance/Correctness | ✅ PASS 9/9 | ✅ Must maintain |
| .ak files | ✅ 46/46 parseable | ✅ Must maintain |
| GPU verified ops | ✅ 45/45 | ✅ Must maintain |
| Test suite | ✅ 422 passed, 6 skipped | ✅ ≥422 passed, ≤6 skipped |
| OpRegistry | ❌ 6-file op knowledge | ⬜ Single source of truth |
| SemanticInterpreter | ❌ NumPy per-op (inconsistent) | ⬜ PyTorch eager graph executor |
| Pass Infrastructure | ❌ Not present | ⬜ Pass protocol + pipeline |
| SSA Validator | ❌ Not present | ⬜ Validates all IR round-trips |
| Backend Abstraction | ❌ Hardcoded TritonBackend | ⬜ ArkeBackend protocol |
| Dynamic Shape (.ak) | ❌ Static only | ⬜ where clause MVP |
| Lang Spec v2.0 | ❌ v1.0 (static shapes only) | ⬜ v2.0 (symbolic dims, backend-agnostic) |
| IR Spec v2.0 | ❌ v1.0 (single-layer SemanticIR+StrategyIR) | ⬜ v2.0 (Layer 4/3/2/1 defined) |

---

*Document end. For implementation task board, see `docs/design/stage1/g6-task-board.md` (to be created).*
*For original G6 section, see `docs/design/stage1/stage1-gate-design.md` §5.*
*For architecture discussion reference, see `docs/design/stage1/g6-redesign-reference/`.*