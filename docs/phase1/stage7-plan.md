# Phase 1 — Stage 7: Lang & IR v2

> Gate G7 exit criteria → [plan.md](../roadmap/plan.md#stage-7-g7-lang--ir-v2-)

**Objective:** Land the finalized Arke Lang v2.0 and Arke IR v2.0 architecture in code: `where` clause, symbolic dimensions, conditional/backend-agnostic StrategyIR, compiler round-trip for all 45 ops, and MLIR skeleton for future lowering. **Stage 7 is v2-only: do not preserve v1/v0.2 compatibility in code, tests, or examples. Remove legacy surface/tests instead of carrying migration shims. Stage 7 must end with Lang + IR + compiler support sufficient for BL5 full operator coverage and full shape coverage, and that support must be strong enough to pass the required L2 fused benchmarks rather than only L1 single-op checks.**

**Depends on:** S6 (Pass pipeline, OpRegistry, Backend abstraction)
**Blocks:** S8 (Agent Autonomy needs v2 Lang/IR semantics, symbolic shapes, MLIR skeleton, and full op coverage)

---

## Stage Intent

S7 is the bridge from the S6 compiler-infrastructure baseline to the fully LLM-native architecture required by S8/S9.

This stage is no longer about drafting the design. The Arke Lang Spec v2.0 and Arke IR Spec v2.0 are already finalized. The work now is to **make the implementation and verification stack match the spec**.

Concretely, S7 focuses on six outcomes:

1. **Spec-to-code alignment** — parser, SemanticIR, StrategyIR, examples, and tests match the finalized v2.0 definitions.
2. **Symbolic shape system, not just MVP syntax** — `where` clause and `symbolic_dims` work end-to-end in a way that can express and carry BL5 shape coverage requirements.
3. **Backend-agnostic optimization semantics** — StrategyIR remains target-neutral in its core representation, with target-specific lowering pushed below Layer 3.
4. **Layered compiler path** — explicit bridge from Layer 4/3 into MLIR-oriented Layer 2/1 skeleton.
5. **BL5 full-coverage enablement** — Lang + IR must be expressive enough for **all BL5 operators and all BL5 shapes**, not only representative subsets.
6. **L2-capable implementation target** — the final Stage 7 implementation must support the fused/operator-composition requirements needed by BL5 L2, especially the benchmark-design.md fusion set.

---

## Spec Alignment Summary

### Arke Lang v2.0 implications for S7

From `docs/spec/arke-lang-spec-v2.md`, S7 implementation must support:

- `kernel` + `strategy` as the canonical `.ak` structure
- `where` clause for symbolic dimensions and shape constraints
- tuple returns / destructuring for multi-output ops
- `_` return type inference where legal
- backend-agnostic directives in strategy blocks
- conditional strategies via `when` / `otherwise`
- `@rationale` preserved as first-class optimization metadata
- operator definitions remaining algorithm-agnostic and Op Registry driven

### Arke IR v2.0 implications for S7

From `docs/spec/arke-ir-spec-v2.md`, S7 implementation must establish:

- **Layer 4 SemanticIR** as immutable operator semantics
- **Layer 3 StrategyIR** as bounded, rationale-carrying optimization decisions
- **Layer 2 ScheduleIR** as compiler-generated schedule mapping layer
- **Layer 1 InstructionIR** as fully automated low-level representation
- first-class `SymbolicDim` / symbolic shape propagation across layers
- target-neutral StrategyIR core, with target-specific details deferred to lowering
- deterministic, verifiable lowering boundaries between layers

### Immediate planning consequence

Track planning should treat the spec docs as **done artifacts** and move effort to parser/IR/compiler/test/benchmark convergence.

More importantly, S7 planning must be evaluated against the benchmark system's real requirement surface:

- **BL5 = OT0–OT4 × ST1–ST4 full coverage**
- **ST4 applies to OT2–OT4 and carries the true production-shape pressure**
- **L2 at BL5 is not optional decoration**; Stage 7 Lang/IR choices must support the fused benchmark set in `benchmark-design.md`
- therefore, S7 cannot stop at “feature exists for sample ops” — it must establish a representation and lowering path that scales to the full BL5 matrix

---

## Gate Criteria Breakdown

**BL Exit:** BL5×L1+L2 — This is the first Gate requiring full benchmark coverage across all 45 ops and all shape tiers.

For S7 planning purposes, this means the final Lang/IR design must support:
- OT0–OT4 full operator coverage
- ST1–ST4 full shape coverage where defined by the benchmark system
- OT2–OT4 production-shape support strong enough to reach BL5, not merely BL4-style standard-shape support
- the L2 fusion set required by BL5 (`matmul+relu`, `matmul+gelu`, `swiglu`, `geglu`, `linear+cross_entropy`, `QKV+flash_attention`)

> Reference: `docs/benchmark/benchmark-design.md` for BL/OT/ST/L definitions; `docs/deprecated/phase1-gate-design.md` §5 for original G6 BL5 derivation.

### Benchmark Requirements (from Gate-Purpose Mapping)

#### L1 @ BL5 (OT0-4, ST1-4) — Single Operator Performance

| Op Group | Correctness Requirement | Performance Requirement | Baseline | Measurement |
|:---------|:------------------------|:------------------------|:---------|:------------|
| **OT0** Elementwise (12 ops) | 100%(ST1-4, excl. OOM) | geomean ≥ 1.05× P1 (FlagGems elem) | P1 | `arke bench --bl 5 --ot 0 --layer l1` |
| **OT1** Reduction (10 ops) | 100%(ST1-4, excl. OOM) | geomean ≥ 0.95× P1 (FlagGems norm/softmax) | P1 | `arke bench --bl 5 --ot 1 --layer l1` |
| **OT2** Compute-Dense (11 ops) | 100%(ST1-4, excl. OOM) | matmul geomean ≥ 1.00× P0 (cuBLAS); others ≥ 0.95× P1 | P0, P1 | `arke bench --bl 5 --ot 2 --layer l1` |
| **OT3** Gated Activation (7 ops) | 100%(ST1-4, excl. OOM) | swiglu/rope geomean ≥ 0.95× P1 (Liger/FlagGems) | P1 | `arke bench --bl 5 --ot 3 --layer l1` |
| **OT4** Attention (5 ops) | 100%(ST1-4, excl. OOM) | FA geomean ≥ 0.90× P1 (FlashAttn-2); GQA ≥ 0.90 | P1 | `arke bench --bl 5 --ot 4 --layer l1` |

> **OOM note:** BL5 cannot rely on “accept incomplete coverage” logic. If a shape is inherently impossible on 6GB VRAM, the benchmark harness must record it consistently and the stage must provide a shape/memory strategy compatible with the Gate definition.

#### L2 @ BL5 — Fused Operator Performance

| Fusion Combination | Requirement | Baseline | Measurement |
|:-------------------|:------------|:---------|:------------|
| matmul+relu, matmul+gelu | ≥ 1.10× unfused (fusion benefit verifiable) | P3 unfused | `arke bench --bl 5 --layer l2 --fusion matmul_relu,matmul_gelu` |
| swiglu, geglu | ≥ 0.95× Liger | P1 | `arke bench --bl 5 --layer l2 --fusion swiglu,geglu` |
| linear+cross_entropy | ≥ 1.10× unfused | P3 | `arke bench --bl 5 --layer l2 --fusion linear_ce` |
| QKV+flash_attention | ≥ 0.85× FlashAttn-2 | P1 | `arke bench --bl 5 --layer l2 --fusion qkv_fa` |

#### G7 Combined PASS Formula

```text
G7 PASS = AND ALL:
  [BL5-L1] L1 BL5 correctness: 100%(ST1-4, excl. OOM) for all OT0-OT4
  [BL5-L1] L1 BL5 performance weighted_score ≥ 0.95
           weighted_score = 0.25×score(OT0-1) + 0.30×score(OT2) + 0.20×score(OT3) + 0.25×score(OT4)
           where score(OTn) = geomean pass rate for that OT group (0.0~1.0)
  [BL5-L2] L2 BL5: 4/4 fusion combinations pass
  [Spec]   Criteria [1]-[5] below
  [Impl]   Criteria [6]-[9] below
```

### Gate Criteria Detail

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | Arke Lang Spec v2.0 finalized and used as implementation contract | `docs/spec/arke-lang-spec-v2.md` exists and current implementation matches required surface syntax |
| 2 | Arke IR Spec v2.0 finalized and used as implementation contract | `docs/spec/arke-ir-spec-v2.md` exists and Layer 4/3/2/1 terminology maps to code |
| 3 | `where` clause + symbolic shape system supports BL5-relevant shape expression and propagation | `pytest tests/test_symbolic_shape.py` + BL5-oriented shape cases for OT2–OT4 |
| 4 | Dynamic shape feasibility assessment complete | `docs/phase1/dynamic-shape-feasibility.md` exists |
| 5 | MLIR framework skeleton exists with BL1 matmul path verified | MLIREmitter / lowering skeleton exists; BL1 matmul verified through skeleton path |
| 6 | All 45 BL5 ops: `.ak → SemanticIR → StrategyIR` full round-trip passes | `python -m arke.compiler.pipeline --ak examples/<op>.ak --dry-run` passes all 45 ops |
| 7 | Lang expressiveness covers the full BL5 operator/shape surface, not just demos | parse/round-trip tests cover `where`, tuple returns, `_`, conditional strategy, multi-output, attention-family, quantization-family, and BL5 production-shape examples |
| 8 | StrategyIR / lowering surface can represent the BL5 L2 fusion set | dry-run / lowering tests cover `matmul+relu`, `matmul+gelu`, `swiglu`, `geglu`, `linear+cross_entropy`, `QKV+flash_attention` |
| 9 | Backend-agnostic StrategyIR core contains 0 Triton-specific fields | `scripts/check_backend_agnostic.py` passes against StrategyIR core |
| 10 | Non-regression suite remains green | `pytest tests/ -q` — no new failures |

---

## Pre-Refactor Reference (from G6 v1)

> ⚠️ All items below were completed under the old architecture. After the Lang/IR/Compiler redesign, they need re-implementation and re-validation. Tasks that overlap with S7 scope are marked ⬜ Reset.

| ID | Description | Status |
|:---|:------------|:------:|
| D6-L1 | `.ak` 4D tensor syntax extension | ✅ Done |
| D6-L2 | gather/scatter semantic nodes | ✅ Done |
| D6-L3 | quantize primitive syntax | ✅ Done |
| D6-L4 | paged memory semantic annotation (stub) | ✅ Done |
| D6-L5 | grammar fix (array literal, float constant) | ✅ Done |
| D6-L6 | `.ak` example files for all 46 ops | ✅ Done |
| D6-IR1 | SemanticIR op catalog → 46 ops | ✅ Done |
| D6-IR2 | AttentionSemanticIR (mask_type, num_kv_heads, head_dim) | ⬜ Reset |
| D6-IR3 | RopeSemanticIR (theta, base, rotary_dim) | ⬜ Reset |
| D6-IR4 | QuantizeSemanticIR (scale_dtype, group_size, zero_point) | ⬜ Reset |
| D6-IR5 | `ast_to_strategy()` converter | ✅ Done |
| D6-IR6 | StrategyIR JSON round-trip (all 46 ops) | ✅ Done |
| D6-IR7 | MLA-specific fields (latent_dim, kv_lora_rank) | ⬜ Reset |
| D6-E1 | 10 Triton template classes (OT3/OT4 full) | ⬜ Reset |
| D6-E6 | V1 validator extension (attention + quantization tolerance) | ⬜ Reset |

---

## Tracks

### Track 1: Spec-to-Code Alignment (P0)

This track converts the finalized v2.0 specs into executable implementation contracts.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T1.1 | Audit parser / AST / examples against Lang v2.0 (`where`, tuple returns, `_`, conditional strategy, imports) | P0 | 1d | ✅ |
| T1.2 | Audit SemanticIR / StrategyIR data model against IR v2.0 layer vocabulary and required fields | P0 | 1d | ✅ |
| T1.3 | Replace outdated v1/v1.5 terminology in code/docs/tests with Layer 4/3/2/1 naming where applicable | P1 | 0.5d | ✅ |
| T1.4 | Build a spec conformance checklist covering language features and IR structures used by all examples | P1 | 0.5d | 🟨 — Stage 7 tests now cover canonical v2 parser/IR/example behavior, but a single explicit checklist artifact is still not split out as its own document |

### Track 2: Symbolic Shape System for BL5 (`where` + `symbolic_dims`) (P0)

This is the core functional delta introduced by Lang/IR v2.0 and the primary unblocker for full BL5 operator/shape coverage.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T2.1 | Implement `where` clause in grammar / parser / AST | P0 | 0.5d | ✅ |
| T2.2 | Lower `where` declarations into SemanticIR `symbolic_dims` and constraints | P0 | 0.5d | ✅ |
| T2.3 | Extend shape inference / validation to preserve symbolic constraints end-to-end | P0 | 1d | ✅ |
| T2.4 | Add `.ak` examples covering BL5 production-shape OT2/OT4 operators, not only representative toy cases | P0 | 0.5d | ✅ |
| T2.5 | Add `tests/test_symbolic_shape.py` with BL5-oriented shape cases across OT2–OT4 | P0 | 0.5d | ✅ |
| T2.6 | Verify symbolic constraints are sufficient to encode the ST4 production-shape families used in BL5 | P0 | 0.5d | 🟨 — representable in current parser/SemanticIR, but benchmark execution readiness still blocked by OT4 memory constraints on 6GB VRAM |

### Track 3: StrategyIR v2 and Backend-Agnostic Decisions (P0)

Layer 3 must remain LLM-facing and target-neutral. Any Triton-specific configuration belongs below StrategyIR core.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T3.1 | Normalize StrategyIR decision types to v2.0 core directives / annotations | P0 | 0.5d | ✅ |
| T3.2 | Implement conditional strategy representation for `when` / `otherwise` | P0 | 0.5d | ✅ |
| T3.3 | Preserve `@rationale` on all StrategyIR decisions through parse → IR → serialization | P0 | 0.5d | ✅ |
| T3.4 | Eliminate Triton-specific fields from StrategyIR core and move them to lowering adapters | P0 | 0.5d | 🟨 — active Lang/IR/tests now use canonical `compute(...)`, but lowering/backends still carry Triton-internal resource fields by design |
| T3.5 | Strengthen `scripts/check_backend_agnostic.py` to enforce the v2.0 boundary | P1 | 0.5d | ✅ |

### Track 4: Layered Lowering + MLIR Skeleton (P1)

The goal is not full MLIR functionality yet; it is to establish the architectural seam from Layer 3 downwards.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T4.1 | Define minimal in-code ScheduleIR / InstructionIR skeletons aligned with IR spec terminology | P1 | 1d | ⬜ |
| T4.2 | Implement StrategyIR → ScheduleIR lowering skeleton for BL1 matmul path | P1 | 1d | ⬜ |
| T4.3 | Implement MLIREmitter / stub bridge from lower layers to MLIR-oriented output | P1 | 1d | ⬜ |
| T4.4 | Add verification that BL1 matmul can traverse the new skeleton path without regressions | P1 | 0.5d | ⬜ |

### Track 5: Full BL5 Surface Coverage (P0)

S7 must convert the spec-aligned representation into full BL5 coverage, not just feature demos or operator-only coverage.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T5.1 | Refresh all 45 BL5 operator examples to valid v2.0 `.ak` surface syntax | P0 | 1d | ✅ |
| T5.2 | Verify all 45 BL5 ops pass `.ak → SemanticIR → StrategyIR` dry-run pipeline | P0 | 0.5d | ✅ |
| T5.3 | Revalidate multi-output, attention, rope, quantize, and MLA-specific examples under v2.0 IR | P0 | 0.5d | ✅ |
| T5.4 | Add coverage audit ensuring each BL5 shape family is representable in Lang + IR for the relevant ops | P0 | 0.5d | 🟨 — language/IR representation covered, but full benchmark-level executability still blocked for OT4 memory-heavy cases |
| T5.5 | Keep token-efficiency checks meaningful under new syntax features and larger production-shape annotations | P1 | 0.5d | ⬜ |

### Track 6: BL5 Benchmark, L2 Fusion, and Memory-Readiness (P0)

This is the practical closure track for S7. The architectural work is only complete if it enables full BL5 execution, including the required L2 fused/operator-composition paths.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T6.1 | Extend benchmark routing to all 45 ops and the full BL5 shape registry | P0 | 1d | 🟨 |
| T6.2 | Implement / adapt L2 fused benchmark runners for the full required fusion set from `benchmark-design.md` | P0 | 0.5d | 🟨 |
| T6.3 | Rewrite active architecture design docs to canonical v2-only references (remove migration/backward-compat narrative from current docs) | P1 | 0.5d | ✅ |
| T6.3 | Ensure Lang + IR + lowering can express the six BL5 L2 fusion cases end-to-end | P0 | 0.5d | ⬜ |
| T6.4 | Align baselines: cuBLAS / FlashAttn-2 / Liger / FlagGems / eager fallback where needed | P0 | 0.5d | 🟨 |
| T6.5 | Define memory-aware execution strategy for OT4 / large OT2 shapes on 6GB VRAM without reducing BL5 scope | P0 | 1d | ⬜ |
| T6.6 | Produce stable perf artifacts (`perf_{op}.csv`, summaries, standard result dirs) for gate verification | P1 | 0.5d | 🟨 |

### Track 7: Non-Regression and Gate Closure (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T7.1 | Keep parser / IR / compiler / benchmark tests green during migration | P0 | continuous | 🟨 — current Stage 7 parser/IR/roundtrip/backend-agnostic slices pass; benchmark/gate suite not yet all-green |
| T7.2 | Add regression coverage for symbolic dims, conditional strategy, and rationale persistence | P0 | 0.5d | ✅ |
| T7.3 | Run full stage verification checklist and record evidence in standard result locations | P0 | 0.5d | ⬜ |

---

## Exit-Oriented Milestones

| Milestone | Scope | Exit Signal |
|:----------|:------|:------------|
| M1 | Track 1 complete | Spec terminology and required v2.0 features mapped to code | ✅ |
| M2 | Tracks 2+3 complete | `where` / `symbolic_dims` / conditional backend-agnostic StrategyIR working end-to-end | ✅ |
| M3 | Track 4 complete | BL1 matmul traverses Layer 4 → 3 → 2/1 skeleton → MLIR bridge | ⬜ |
| M4 | Track 5 complete | All 45 BL5 ops pass v2.0 dry-run round-trip and BL5 shape families are representable | 🟨 — dry-run coverage is green; full benchmark executability remains partially blocked by memory-heavy OT4 cases |
| M5 | Track 6 complete | BL5 L1/L2 benchmark harness ready with full fusion coverage and 6GB-aware execution strategy | 🟨 |
| M6 | Track 7 complete | Gate evidence assembled; no regressions; ready to run G7 verification | ⬜ |

**Critical path:** Track 1 → Tracks 2/3 → Track 4 → Track 5 → Track 6 → Track 7

---

## Priority Order for Execution

1. **Track 1: Spec-to-code alignment**
2. **Track 2: Symbolic shape MVP**
3. **Track 3: StrategyIR v2 cleanup**
4. **Track 5: Full-op example + round-trip coverage**
5. **Track 4: Layered lowering / MLIR skeleton**
6. **Track 6: BL5 benchmark + memory readiness**
7. **Track 7: Non-regression + gate closure**

Rationale for this order:
- Lang/IR surface alignment is the foundation for all later work.
- Symbolic shapes and backend-agnostic StrategyIR are the biggest true deltas from S6.
- Full operator round-trip must stabilize before benchmark closure work is trustworthy.
- MLIR skeleton should be architecturally correct, but not block earlier parser/IR convergence.
- BL5 closure is the final proof that the S7 redesign solves the practical coverage problem left by S6.
- “Support BL5” here means full operator coverage, full relevant shape coverage, and L2 fusion-capable Lang/IR/lowering — not just passing a subset of single-op demos.

---

## Current Verification Notes

- **Stage 7 parser/IR validation currently green:** `tests/test_parser.py`, `tests/test_strategy_ir.py`, `tests/test_strategy_converter.py`, `tests/test_converters.py`, `tests/test_semantic_ir.py`, `tests/test_stage7_roundtrip.py`, `tests/test_symbolic_shape.py`, and `tests/test_backend_agnostic.py` → `412 passed, 6 skipped`.
- **Current skip reasons are explicit, not ignored:**
  - `tests/test_backend_agnostic.py` has 6 skips, all for `01_matmul.ak`, because that example intentionally omits an explicit strategy block and therefore has no authored `StrategyIR` to validate.
  - `tests/test_rationale_e2e.py` has 1 skip for `01_matmul.ak`, because that file intentionally has no `@rationale` annotation.
- **Benchmark/test harness status for the current active slices is green:** `tests/test_backend_agnostic_script.py`, `tests/test_rationale_e2e.py`, `tests/test_bench*.py`, `tests/test_benchmark*.py`, and `benchmarks/` slice pass, with the single documented skip above.
- **Remaining non-green area is benchmark/gate readiness, not parser/IR correctness:** OT4 / large attention-family benchmark execution still hits 6GB VRAM limits in full gate runs, so any FAIL/SKIP there must be recorded in plan/status with the concrete operator and memory reason instead of being hidden as partial completion.
- **Active spec/benchmark-interface cleanup was revalidated after removing compat wording/aliases:** `tests/test_benchmark_cli.py`, `tests/test_op_registry.py`, `tests/test_converters.py`, `tests/test_semantic_ir.py`, `tests/test_symbolic_shape.py`, and `tests/test_stage7_roundtrip.py` passed together (`109 passed`).
- **Active architecture docs were also rewritten as v2-only references:** `docs/architecture/arke-lang-spec-design.md`, `docs/architecture/arke-ir-spec-design.md`, `docs/architecture/arke-compiler-infrastructure.md`, and `docs/architecture/naming-system.md` no longer act as migration-preservation docs for the current mainline; related Stage 7 slice revalidated with `344 passed, 6 skipped`.

## Dependencies

- **Depends on:** S6 (Pass pipeline, OpRegistry, Backend abstraction)
- **Blocks:** S8 (Agent Autonomy needs v2 Lang/IR semantics, symbolic shapes, full op coverage, and MLIR skeleton)
