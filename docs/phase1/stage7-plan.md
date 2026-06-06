# Phase 1 — Stage 7: Lang & IR v0.1.0

> Gate G7 exit criteria → [plan.md](../roadmap/plan.md#stage-7-g7-lang--ir-v010-)

**Objective:** Land the finalized Arke Lang v0.1.0 and Arke IR v0.1.0 architecture in code: `where` clause, symbolic dimensions, conditional/backend-agnostic StrategyIR, compiler round-trip for all 45 ops, and MLIR skeleton for future lowering. **Stage 7 must keep code, tests, and examples on the canonical current surface rather than preserving non-canonical aliases or translation shims. Stage 7 must end with Lang + IR + compiler support sufficient for BL5 full operator coverage and full shape coverage, and that support must be strong enough to pass the required L2 fused benchmarks rather than only L1 single-op checks.**

**Depends on:** S6 (Pass pipeline, OpRegistry, Backend abstraction)
**Blocks:** S8 (Agent Autonomy needs v0.1.0 Lang/IR semantics, symbolic shapes, MLIR skeleton, and full op coverage)

---

## Stage Intent

S7 is the bridge from the S6 compiler-infrastructure baseline to the fully LLM-native architecture required by S8/S9.

This stage is no longer about drafting the design. The Arke Lang Spec v0.1.0 and Arke IR Spec v0.1.0 are already finalized. The work now is to **make the implementation and verification stack match the spec**.

Concretely, S7 focuses on six outcomes:

1. **Spec-to-code alignment** — parser, SemanticIR, StrategyIR, examples, and tests match the finalized v0.1.0 definitions.
2. **Symbolic shape system, not just MVP syntax** — `where` clause and `symbolic_dims` work end-to-end in a way that can express and carry BL5 shape coverage requirements.
3. **Backend-agnostic optimization semantics** — StrategyIR remains target-neutral in its core representation, with target-specific lowering pushed below Layer 3.
4. **Layered compiler path** — explicit bridge from Layer 4/3 into MLIR-oriented Layer 2/1 skeleton.
5. **BL5 full-coverage enablement** — Lang + IR must be expressive enough for **all BL5 operators and all BL5 shapes**, not only representative subsets.
6. **L2-capable implementation target** — the final Stage 7 implementation must support the fused/operator-composition requirements needed by BL5 L2, especially the benchmark-design.md fusion set.

---

## Spec Alignment Summary

### Arke Lang v0.1.0 implications for S7

From `docs/spec/arke-lang-spec.md`, S7 implementation must support:

- `kernel` + `strategy` as the canonical `.ak` structure
- `where` clause for symbolic dimensions and shape constraints
- tuple returns / destructuring for multi-output ops
- `_` return type inference where legal
- backend-agnostic directives in strategy blocks
- conditional strategies via `when` / `otherwise`
- `@rationale` preserved as first-class optimization metadata
- operator definitions remaining algorithm-agnostic and Op Registry driven

### Arke IR v0.1.0 implications for S7

From `docs/spec/arke-ir-spec.md`, S7 implementation must establish:

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
- the L2 fusion set required by BL5 (`matmul+relu`, `matmul+gelu`, `silu_and_mul`, `gelu_and_mul`, `linear+cross_entropy`, `QKV+flash_attention`)

> Reference: `docs/benchmark/benchmark-design.md` for BL/OT/ST/L definitions. The active Gate contract lives in `docs/roadmap/plan.md`; non-normative derivation notes are outside the active Stage 7 contract.

### Benchmark Requirements (from Gate-Purpose Mapping)

> **Locked 2026-05-16 — Same-Backend Triton Fairness:** the L1/L2 BL5 performance denominator is the **best Triton-only implementation per op** in the ladder (FlagGems / Liger / Unsloth / vLLM-Triton / flash-attn / etc.), not a cross-backend reference. Pass criterion: `arke_latency ≤ triton_ref_latency × (1 + ε)` with `ε = 0.03`. Rows with no Triton-only reference are **audit-only** (excluded from Gate scoring). See `docs/roadmap/plan.md` § Same-Backend Fairness and `docs/benchmark/benchmark-protocol.md` for the full protocol.

#### L1 @ BL5 (OT0-4, ST1-4) — Single Operator Performance (Same-Backend Triton)

| Op Group | Correctness Requirement | Performance Requirement (Same-Backend Triton) | Triton Reference | Measurement |
|:---------|:------------------------|:----------------------------------------------|:-----------------|:------------|
| **OT0** Elementwise (12 ops) | 100%(ST1-4, excl. OOM) | per-row `arke ≤ triton_ref × 1.03`; ≥97% pass rate | FlagGems Triton elem | `arke bench --bl 5 --ot 0 --layer l1` |
| **OT1** Reduction (10 ops) | 100%(ST1-4, excl. OOM) | per-row `arke ≤ triton_ref × 1.03`; ≥97% pass rate | FlagGems Triton norm/softmax | `arke bench --bl 5 --ot 1 --layer l1` |
| **OT2** Compute-Dense (11 ops) | 100%(ST1-4, excl. OOM) | per-row `arke ≤ triton_ref × 1.03`; ≥97% pass rate | best Triton matmul/conv in ladder (FlagGems / vLLM-Triton matmul) | `arke bench --bl 5 --ot 2 --layer l1` |
| **OT3** Gated Activation (7 ops) | 100%(ST1-4, excl. OOM) | per-row `arke ≤ triton_ref × 1.03`; ≥97% pass rate | Liger / FlagGems Triton (silu_and_mul/gelu_and_mul/rope) | `arke bench --bl 5 --ot 3 --layer l1` |
| **OT4** Attention (5 ops) | 100%(ST1-4, excl. OOM) | per-row `arke ≤ triton_ref × 1.03`; ≥97% pass rate | flash-attn Triton / FlashInfer Triton / vLLM-Triton attention | `arke bench --bl 5 --ot 4 --layer l1` |

> **OOM note:** BL5 cannot rely on “accept incomplete coverage” logic. If a shape is inherently impossible on 6GB VRAM, the benchmark harness must record it consistently and the stage must provide a shape/memory strategy compatible with the Gate definition.

> **Audit-only note:** an op-shape with no Triton-only reference in the ladder is recorded with `perf_oracle_unavailable_triton=true` and excluded from Gate scoring. These rows must still appear in PERF_ALL (Benchmark design is frozen — only the Gate scoring layer changes).

#### L2 @ BL5 — Fused Operator Performance (Same-Backend Triton)

| Fusion Combination | Requirement (Same-Backend Triton) | Triton Reference | Measurement |
|:-------------------|:----------------------------------|:-----------------|:------------|
| matmul+relu, matmul+gelu | per-row `arke ≤ triton_ref × 1.03` (fusion must beat unfused Triton) | best Triton fused matmul+act in ladder; if absent → unfused Triton sequential | `arke bench --bl 5 --layer l2 --fusion matmul_relu,matmul_gelu` |
| silu_and_mul, gelu_and_mul | per-row `arke ≤ triton_ref × 1.03` | Liger Triton silu_and_mul/gelu_and_mul | `arke bench --bl 5 --layer l2 --fusion silu_and_mul,gelu_and_mul` |
| linear+cross_entropy | per-row `arke ≤ triton_ref × 1.03` | Liger Triton fused_linear_cross_entropy (canonical L2 slot name: `linear_ce`) | `arke bench --bl 5 --layer l2 --fusion linear_ce` |
| QKV+flash_attention | per-row `arke ≤ triton_ref × 1.03` | flash-attn Triton (FA-2 Triton kernel) | `arke bench --bl 5 --layer l2 --fusion qkv_fa` |

#### G7 Combined PASS Formula

```text
G7 PASS = AND ALL:
  [BL5-L1] L1 BL5 correctness: 100%(ST1-4, excl. OOM) for all OT0-OT4
  [BL5-L1] L1 BL5 performance under Same-Backend Triton Fairness:
           per-row: arke_latency ≤ triton_ref_latency × 1.03 (rows without Triton-ref are audit-only)
           per-group pass rate: ≥ 0.97 for each of OT0_1 / OT2 / OT3 / OT4
           weighted_score = 0.25×rate(OT0_1) + 0.30×rate(OT2) + 0.20×rate(OT3) + 0.25×rate(OT4) ≥ 0.95
  [BL5-L2] L2 BL5: 4/4 fusion combinations pass under Same-Backend Triton Fairness
  [Spec]   Criteria [1]-[5] below
  [Impl]   Criteria [6]-[9] below
```

### Gate Criteria Detail

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | Arke Lang Spec v0.1.0 finalized and used as implementation contract | `docs/spec/arke-lang-spec.md` exists and current implementation matches required surface syntax |
| 2 | Arke IR Spec v0.1.0 finalized and used as implementation contract | `docs/spec/arke-ir-spec.md` exists and Layer 4/3/2/1 terminology maps to code |
| 3 | `where` clause + symbolic shape system supports BL5-relevant shape expression and propagation | `pytest tests/test_symbolic_shape.py` + BL5-oriented shape cases for OT2–OT4 |
| 4 | Dynamic shape feasibility assessment complete | `docs/phase1/dynamic-shape-feasibility.md` exists |
| 5 | MLIR framework skeleton exists with BL1 matmul path verified | MLIREmitter / lowering skeleton exists; BL1 matmul verified through skeleton path |
| 6 | All 45 BL5 ops: `.ak → SemanticIR → StrategyIR` full round-trip passes | `python -m arke.compiler.pipeline --ak examples/<op>.ak --dry-run` passes all 45 ops |
| 7 | Lang expressiveness covers the full BL5 operator/shape surface, not just demos | parse/round-trip tests cover `where`, tuple returns, `_`, conditional strategy, multi-output, attention-family, quantization-family, and BL5 production-shape examples |
| 8 | StrategyIR / lowering surface can represent the BL5 L2 fusion set | dry-run / lowering tests cover `matmul+relu`, `matmul+gelu`, `silu_and_mul`, `gelu_and_mul`, `linear+cross_entropy`, `QKV+flash_attention` |
| 9 | Backend-agnostic StrategyIR core contains 0 Triton-specific fields | `scripts/check_backend_agnostic.py` passes against StrategyIR core |
| 10 | Non-regression suite remains green | `pytest tests/ -q` — no new failures |

---

## Pre-Refactor Reference (from G6 baseline)

> ⚠️ All items below were completed before the current Lang/IR/Compiler architecture. After the redesign, they need re-implementation and re-validation. Tasks that overlap with S7 scope are marked ⬜ Reset.

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
| D6-E6 | Validator extension (attention + quantization tolerance) | ⬜ Reset |

---

## Tracks

### Track 1: Spec-to-Code Alignment (P0)

This track converts the finalized v0.1.0 specs into executable implementation contracts.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T1.1 | Audit parser / AST / examples against Lang v0.1.0 (`where`, tuple returns, `_`, conditional strategy, imports) | P0 | 1d | ✅ |
| T1.2 | Audit SemanticIR / StrategyIR data model against IR v0.1.0 layer vocabulary and required fields | P0 | 1d | ✅ |
| T1.3 | Align outdated layer terminology in code/docs/tests with Layer 4/3/2/1 naming where applicable | P1 | 0.5d | ✅ |
| T1.4 | Build a spec conformance checklist covering language features and IR structures used by all examples | P1 | 0.5d | ✅ — explicit checklist artifact added at `docs/phase1/stage7-conformance-checklist.md` |

### Track 2: Symbolic Shape System for BL5 (`where` + `symbolic_dims`) (P0)

This is the core functional delta captured by Lang/IR v0.1.0 and the primary unblocker for full BL5 operator/shape coverage.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T2.1 | Implement `where` clause in grammar / parser / AST | P0 | 0.5d | ✅ |
| T2.2 | Lower `where` declarations into SemanticIR `symbolic_dims` and constraints | P0 | 0.5d | ✅ |
| T2.3 | Extend shape inference / validation to preserve symbolic constraints end-to-end | P0 | 1d | ✅ |
| T2.4 | Add `.ak` examples covering BL5 production-shape OT2/OT4 operators, not only representative toy cases | P0 | 0.5d | ✅ |
| T2.5 | Add `tests/test_symbolic_shape.py` with BL5-oriented shape cases across OT2–OT4 | P0 | 0.5d | ✅ |
| T2.6 | Verify symbolic constraints are sufficient to encode the ST4 production-shape families used in BL5 | P0 | 0.5d | ✅ — ST4 production-shape families are representable in current parser/SemanticIR; execution-pressure issues are tracked separately under T6.5 |

### Track 3: StrategyIR v0.1.0 and Backend-Agnostic Decisions (P0)

Layer 3 must remain LLM-facing and target-neutral. Any Triton-specific configuration belongs below StrategyIR core.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T3.1 | Normalize StrategyIR decision types to v0.1.0 core directives / annotations | P0 | 0.5d | ✅ |
| T3.2 | Implement conditional strategy representation for `when` / `otherwise` | P0 | 0.5d | ✅ |
| T3.3 | Preserve `@rationale` on all StrategyIR decisions through parse → IR → serialization | P0 | 0.5d | ✅ |
| T3.4 | Eliminate Triton-specific fields from StrategyIR core and move them to lowering adapters | P0 | 0.5d | ✅ — backend-agnostic StrategyIR core is in place; backend-specific resource binding now lives below Layer 3 |
| T3.5 | Strengthen `scripts/check_backend_agnostic.py` to enforce the v0.1.0 boundary | P1 | 0.5d | ✅ |

### Track 4: Layered Lowering + MLIR Skeleton (P1)

The goal is not full MLIR functionality yet; it is to establish the architectural seam from Layer 3 downwards.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T4.1 | Define minimal in-code ScheduleIR / InstructionIR skeletons aligned with IR spec terminology | P1 | 1d | ✅ — `arke/ir/schedule.py` and `arke/ir/instruction.py` define the Layer 2/1 skeletons with loop nests, fusion groups, resources, provenance, and instruction blocks. |
| T4.2 | Implement StrategyIR → ScheduleIR lowering skeleton for BL1 matmul path | P1 | 1d | ✅ — `arke/compiler/lowering.py` lowers StrategyIR decisions (including conditional branches) into ScheduleIR and then InstructionIR, exercised by `tests/test_track4_lowering.py` and `tests/test_stage7_lowering.py`. |
| T4.3 | Implement MLIREmitter / stub bridge from lower layers to MLIR-oriented output | P1 | 1d | ✅ — `arke/compiler/mlir_emitter.py` emits `module { func.func ... }` skeletons with `linalg.matmul` / op attrs plus schedule + instruction-block provenance comments. |
| T4.4 | Add verification that BL1 matmul can traverse the new skeleton path without regressions | P1 | 0.5d | ✅ — `tests/test_stage7_lowering.py::TestStage7Lowering::test_bl1_matmul_traverses_full_stage7_skeleton` exercises SemanticIR → StrategyIR → ScheduleIR → InstructionIR → MLIR on `examples/operators/01_matmul.ak`. |

### Track 5: Full BL5 Surface Coverage (P0)

S7 must convert the spec-aligned representation into full BL5 coverage, not just feature demos or operator-only coverage.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T5.1 | Refresh all 45 BL5 operator examples to valid v0.1.0 `.ak` surface syntax | P0 | 1d | ✅ |
| T5.2 | Verify all 45 BL5 ops pass `.ak → SemanticIR → StrategyIR` dry-run pipeline | P0 | 0.5d | ✅ |
| T5.3 | Revalidate multi-output, attention, rope, quantize, and MLA-specific examples under v0.1.0 IR | P0 | 0.5d | ✅ |
| T5.4 | Add coverage audit ensuring each BL5 shape family is representable in Lang + IR for the relevant ops | P0 | 0.5d | ✅ — coverage audit and shape-family evidence are recorded in `coverage_ledger.json` / `audit_report.json` |
| T5.5 | Keep token-efficiency checks meaningful under new syntax features and larger production-shape annotations | P1 | 0.5d | ✅ — `scripts/check_token_efficiency.py` and `tests/test_token_efficiency.py` now include BL5 L2 fused examples (`examples/operators/l2/*.ak`) with dedicated `l2_fused` category and Triton reference lines; full suite passes with aggregate ratio ≈ 0.26 under v0.1.0 syntax (`where`, conditional strategy, production-shape annotations). |
| T5.6 | Audit every BL5 operator / shape family against Stage 7 Lang surface (`where`, conditional strategy) and log unsupported cases | P0 | 0.5d | ✅ — `python -m benchmarks.stage7_audit_report` now consumes `coverage_ledger.json` and emits `audit_report.json` with missing-example / missing-strategy / missing-shape evidence plus unsupported-case queues for T5.7/T6 follow-up. |
| T5.7 | Add or update `.ak` examples for any patterns uncovered by the audit | P0 | 0.5d | ✅ — added explicit Stage 7 L2 surface examples for `matmul_relu`, `linear_ce`, and `qkv_fa` under `examples/operators/l2/`, and restored `examples/operators/01_matmul.ak` to an explicit strategy-backed L1 surface |
| T5.8 | Maintain machine-readable coverage ledger linking BL5 targets to `.ak` artefacts (feeds Track 6 automation) | P0 | 0.5d | ✅ — `python -m benchmarks.stage7_coverage_ledger` emits `coverage_ledger.json` linking target-matrix entries to `.ak` examples, dry-run pipeline evidence, and Track 6 PERF_ALL coverage |

### Track 6: BL5 Benchmark, L2 Fusion, and Memory-Readiness (P0)

This is the practical closure track for S7. The architectural work is only complete if it enables full BL5 execution, including the required L2 fused/operator-composition paths.

**Reverse-decomposition rule for Track 6:** treat `BL5 × (L1 + L2)` as the real contract, then derive the remaining Lang / IR / compiler / benchmark tasks backward from that target instead of forward from local implementation convenience.

That means every unfinished task in S7 should be justified by one of these benchmark-facing needs:
- **Coverage need:** every BL5 operator and every required shape family must be representable in Lang + IR and runnable by the benchmark harness.
- **Fusion need:** the full BL5 L2 fusion set must exist as first-class compiler/lowering/benchmark paths, not just ad hoc eager tests.
- **Artifact need:** every benchmark run needed by G7 must emit stable gate-readable artifacts (`perf_*.csv`, `PERF_ALL.csv`, `summary.json`, hardware/config/source manifests).
- **Memory need:** 6GB-VRAM-blocked OT4 / large OT2 cases must be handled by memory-aware strategy selection, shape dispatch, or consistently recorded OOM policy compatible with the gate contract.
- **Design need:** if a benchmark requirement cannot be expressed cleanly, the Lang / IR design must be strengthened rather than the benchmark target weakened.

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T6.1 | Extend benchmark routing to all 45 ops and the full BL5 shape registry | P0 | 1d | ✅ — benchmark routing is wired to the full BL5 op registry and shape matrix |
| T6.2 | Implement / adapt L2 fused benchmark runners for the full required fusion set from `benchmark-design.md` | P0 | 0.5d | ✅ — L2 fusion runners exist for `matmul_relu`, `matmul_gelu`, `silu_and_mul`, `gelu_and_mul`, `linear_ce`, and `qkv_fa` |
| T6.3 | Ensure Lang + IR + lowering can express the six BL5 L2 fusion cases end-to-end | P0 | 0.5d | ✅ — explicit `.ak` surface coverage and lowering evidence now exist for the six BL5 fusion cases |
| T6.4 | Align baselines: cuBLAS / FlashAttn-2 / Liger / FlagGems / eager fallback where needed | P0 | 0.5d | ✅ — baselines are aligned to cuBLAS / FlashAttn-2 / Liger / FlagGems / eager fallback where needed |
| T6.5 | Define memory-aware execution strategy for OT4 / large OT2 shapes on 6GB VRAM without reducing BL5 scope | P0 | 1d | ✅ — memory-aware preflight and artifact fields are in place for OT4 and dense/logit-pressure rows |
| T6.6 | Produce stable perf artifacts (`perf_{op}.csv`, `PERF_ALL.csv`, `summary.json`, manifests) for gate verification | P1 | 0.5d | ✅ — stable `perf_{op}.csv`, `PERF_ALL.csv`, `summary.json`, and manifest artifacts are produced under Track 6 |
| T6.7 | Drive remaining Lang / IR / compiler refinements from BL5 benchmark gaps instead of local code convenience | P0 | continuous | ✅ — benchmark-gap-driven refinement loop is now embodied by the coverage ledger, gap computation, dashboard, and gate evidence |
| T6.8 | Persist correctness metrics & tolerances across benchmark artifacts (`PERF_ALL.csv`, summaries, per-op files) | P0 | 0.5d | ✅ |
| T6.9 | Persist performance target evaluation fields (`perf_target`, `perf_actual`, `perf_pass`, `perf_gap`) across artifacts | P0 | 0.5d | ✅ — `benchmarks.artifacts` now writes + aggregates these fields into `PERF_ALL.csv` / `summary.json` |
| T6.10 | Build automation script to compute L1/L2 coverage gaps from `stage7_bl5_target_matrix.json` | P0 | 0.5d | ✅ — `python -m benchmarks.stage7_coverage_gap` |
| T6.11 | Generate machine-readable coverage dashboards / reports from automation outputs | P1 | 0.5d | ✅ — `python -m benchmarks.stage7_dashboard` now consolidates `coverage_gap.json`, `audit_report.json`, and `stage7_operator_shape_stats.json` into `dashboard.json` with focus slices for evidence gaps, unsupported surface cases, perf-field gaps, memory-pressure ops, and priority actions. |
| T6.12 | Integrate artifact field checks into Gate verification (scripts / CI) | P1 | 0.5d | ✅ — `benchmarks.gate_g7.check_stage7_track6_artifacts()` enforces the Track 6 result-tree contract for `l1/`, `l2/`, and root dashboard artifacts, and `G7.8b`/`G7.8c`/`G7.8d` now fail on incomplete BL5 coverage/correctness/performance evidence instead of treating artifact presence as closure. |

### Track 7: Non-Regression and Gate Closure (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| T7.1 | Keep parser / IR / compiler / benchmark tests green while aligning implementation with the active spec | P0 | continuous | 🚧 — parser/IR/roundtrip/backend-agnostic/non-regression slices are green; G7 currently 12/14 — `G7.8c` ✅ PASS after Q5a (rope fp32) + Q6a (gated odd-N typed-unsupported); `G7.8b` (coverage) and `G7.8d` (perf) remain open. |
| T7.2 | Add regression coverage for symbolic dims, conditional strategy, and rationale persistence | P0 | 0.5d | ✅ |
| T7.3 | Run full stage verification checklist and record evidence in standard result locations | P0 | 0.5d | 🚧 — checklist/report artifacts exist; current `python -m benchmarks.gate G7 --tier 2` is `12/14` (G7.8b coverage + G7.8d perf still open). Final G7 closure pending P1 + P3 in `docs/plans/2026-05-14-g78bd-coverage-and-perf.md`. |

---

## Exit-Oriented Milestones

| Milestone | Scope | Exit Signal |
|:----------|:------|:------------|
| M1 | Track 1 complete | Spec terminology and required v0.1.0 features mapped to code | ✅ |
| M2 | Tracks 2+3 complete | `where` / `symbolic_dims` / conditional backend-agnostic StrategyIR working end-to-end | ✅ |
| M3 | Track 4 complete | BL1 matmul traverses Layer 4 → 3 → 2/1 skeleton → MLIR bridge | ✅ |
| M4 | Track 5 complete | All 45 BL5 ops pass v0.1.0 dry-run round-trip and BL5 shape families are representable | ✅ — dry-run coverage is green; benchmark executability is tracked separately under Track 6 |
| M5 | Track 6 complete | BL5 L1/L2 benchmark harness ready with full fusion coverage and 6GB-aware execution strategy | 🚧 — harness/artifact plumbing is ready and `G7.8c` ✅ PASS, but `G7.8b` (coverage) + `G7.8d` (perf) still open |
| M6 | Track 7 complete | Gate evidence assembled; no regressions; ready to run G7 verification | 🚧 — non-regression slices green; G7 at 12/14, blocked by coverage (P1) + perf (P3) |

**Critical path:** Track 1 → Tracks 2/3 → Track 4 → Track 5 → Track 6 → Track 7

---

## Priority Order for Execution

1. **Track 1: Spec-to-code alignment**
2. **Track 2: Symbolic shape MVP**
3. **Track 3: StrategyIR v0.1.0 cleanup**
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
- Once Track 6 starts, **benchmark gaps become the task generator**: missing BL5 evidence should directly create Lang / IR / lowering / runtime work, rather than being treated as a separate downstream validation phase.

---

## Current Verification Notes

- **Stage 7 parser/IR validation currently green, but G7 is not closed:** focused parser/IR/roundtrip/backend-agnostic slices remain green, and `pytest -q tests/test_gate_g7.py` passes (`8 passed`). The stricter `python -m benchmarks.gate G7 --tier 2` now reports `11/14` criteria passed and fails `G7.8b`/`G7.8c`/`G7.8d` against the canonical Track 6 evidence; this is expected until BL5 coverage, correctness, and performance are complete.
- **Track 6 memory-policy slice currently green:** `tests/test_memory_policy.py`, `tests/test_benchmark_artifacts.py`, `tests/test_benchmark_cli.py`, `tests/test_benchmark_l1_attention_preflight.py`, `tests/test_benchmark_l2.py`, `tests/test_benchmark_l2_fused_ce.py`, `tests/test_benchmark_l2_qkv_fa.py`, `tests/test_benchmark_advice.py`, `tests/test_compiler_advice.py`, `tests/test_arke_runner_advice.py`, `tests/test_stage7_report.py`, `tests/test_stage7_coverage_gap.py`, `tests/phase1/stage7/test_stage7_dashboard.py`, `tests/phase1/stage7/test_track6_contract.py`, and Stage 7 strategy/advice slices → `52 passed`.
- **Current skip reasons are explicit, not ignored:**
  - `tests/test_backend_agnostic.py` has 6 skips, all for `01_matmul.ak`, because that example intentionally omits an explicit strategy block and therefore has no authored `StrategyIR` to validate.
  - `tests/test_rationale_e2e.py` has 1 skip for `01_matmul.ak`, because that file intentionally has no `@rationale` annotation.
- **Track 6 memory-readiness is now broader than OT4:** the benchmark policy layer now emits explicit preflight evidence for OT4 attention, OT2 dense matmul/grouped matmul, and OT3 linear+cross_entropy-style logits pressure. L1/L2 artifacts persist `memory_bytes_required`, `memory_bytes_budget`, `memory_ratio`, and `memory_policy`, and summary/dashboard checks surface `memory_pressure_rows` / `memory_policy_counts` instead of hiding those shapes as silent omissions.
- **Benchmark/test harness status for the current active slices is green:** `tests/test_backend_agnostic_script.py`, `tests/test_rationale_e2e.py`, `tests/test_bench*.py`, `tests/test_benchmark*.py`, and `benchmarks/` slice pass, with the single documented skip above.
- **Remaining non-green area is benchmark/gate readiness, not parser/IR correctness:** OT4 / large attention-family benchmark execution still hits 6GB VRAM limits in full gate runs, so any FAIL/SKIP there must be recorded in plan/status with the concrete operator and memory reason instead of being hidden as partial completion.
- **Active spec/benchmark-interface cleanup was revalidated after removing non-canonical wording/aliases:** `tests/test_benchmark_cli.py`, `tests/test_op_registry.py`, `tests/test_converters.py`, `tests/test_semantic_ir.py`, `tests/test_symbolic_shape.py`, and `tests/test_stage7_roundtrip.py` passed together (`109 passed`).
- **Active architecture docs were also rewritten as current-surface references:** `docs/architecture/arke-lang-spec-design.md`, `docs/architecture/arke-ir-spec-design.md`, `docs/architecture/arke-compiler-infrastructure.md`, and `docs/architecture/naming-system.md` document the current mainline directly; related Stage 7 slice revalidated with `344 passed, 6 skipped`.
- **Residual wording cleanup completed after the architecture rewrite:** `docs/spec/arke-lang-vs-python-triton.md` and remaining non-canonical examples/phrasing inside architecture docs were aligned to canonical `compute(...)` / current-surface wording, with the same Stage 7 validation slice staying green (`344 passed, 6 skipped`).
- **Stage 7 closure artifacts are present but not final closure evidence:** `docs/phase1/stage7-conformance-checklist.md` records the explicit v0.1.0 conformance checklist, and `benchmarks/results/phase1/stage7/track6/report.md` captures Track 6 summaries. The current gate runner now treats these as inputs to evidence checks rather than proof of completion; final Stage 7 closure requires `G7.8b`/`G7.8c`/`G7.8d` to pass on canonical artifacts.

## BL5 Gap Snapshot & Phased Closure

> Absorbed from the former `stage7-bl5-gap-report.md` (now retired). The machine-readable source of truth remains `benchmarks/stage7_bl5_target_matrix.json`; the numbers below are the last-recorded snapshot and are expected to move as Track 5/6 progresses.

### Coverage snapshot (reference)

| Dimension | Required | Observed | Ratio |
|:----------|---------:|---------:|------:|
| L1 ops | 45 | 45 | 1.0000 |
| L1 required shapes | 685 | 124 | 0.1810 |
| L2 fusions | 6 | 6 | 1.0000 |
| L2 required shapes | 120 | 6 | 0.0500 |

- **Performance artifacts present:** yes
- **Correctness / accuracy artifacts present:** yes (partial; live for L2 `matmul_relu` and a growing L1 subset across dense linear algebra, elementwise, reduction, normalization, activations, gated fused activations, loss ops, batched/grouped GEMM, positional encoding, data-movement/indexing, quantization, and attention — including `matmul`, `grouped_matmul`, `gelu`, `silu`, `silu_and_mul`, `gelu_and_mul`, `softmax`, `layernorm`, `cross_entropy`, `fused_linear_cross_entropy`, `rope`, `cross_attention`, `flash_attention`, `grouped_query_attention`; `Liger-Kernel` `rope` remains correctness-unsupported because it has no `run_with_inputs(...)` hook).

### Reverse decomposition into the Arke 4-piece suite

Remaining BL5 work is tracked under four component lenses, each feeding existing tracks:

- **A. Lang / Parser** → feeds Track 2 / Track 5. Audit every BL5 op-family against current surface syntax, add `.ak` examples for any unsupported benchmark-driven patterns, and keep a coverage audit linking BL5 targets to parseable examples (T5.6–T5.8).
- **B. SemanticIR / StrategyIR** → feeds Track 3 / Track 5. Validate every target-matrix op against SemanticIR / StrategyIR generation, add metadata hooks needed for benchmark evidence/diagnosability, and ensure fusion cases are represented canonically (T5.3, T6.3).
- **C. Lowering / Compiler pipeline** → feeds Track 4 / Track 6. Build a per-op / per-fusion lowering coverage audit from the target matrix, classify failures by stage (parse / semantic / strategy / lowering / emitter / runtime), and drive remaining lowering fixes directly from benchmark gaps (T6.1–T6.3, T6.7).
- **D. Benchmark / Gate / Evidence** → feeds Track 6 / Track 7. Persist correctness and performance fields into all artifacts, generate coverage dashboards from the target matrix, and promote these checks into formal gate criteria (T6.8–T6.12, T7.3).

### Phased closure roadmap

| Phase | Deliverables | Success condition |
|:------|:-------------|:------------------|
| **S7-A** Target matrix & gap accounting | `benchmarks/stage7_bl5_target_matrix.json`, coverage snapshot above | Every required L1 op and L2 fusion has a machine-readable required-shape inventory; observed coverage is measurable, not anecdotal |
| **S7-B** Correctness-first artifact schema | Benchmark runners write correctness metrics/tolerances; `PERF_ALL.csv` and per-op files include correctness fields; summaries aggregate correctness pass/fail | Every benchmark point has machine-readable correctness evidence (T6.8 ✅) |
| **S7-C** Coverage closure | Stage 7 benchmark routing covers all BL5 L1 ops and all Stage 7 L2 fusions; missing points surfaced automatically | 🚧 Required operator coverage is present, but required shape coverage is not yet 100%; current canonical evidence is L1 `478/685` shapes (`0.6978`) and L2 `120/120` shapes (`1.0000`) |
| **S7-D** Performance contract enforcement | Point-wise performance targets persisted in artifacts (T6.9); group/fusion pass logic implemented (T6.10–T6.12); gate-readable summaries expose coverage + correctness + performance status | 🚧 Machine checks are implemented and now fail honestly: current L1 weighted score is `0.6203 < 0.9500`, with incomplete `matmul_relu` / `matmul_gelu` L2 fusion performance |
| **S7-E** Final BL5 exit | — | ⬜ Full required coverage, correctness, and performance are not yet achieved; Stage 7 remains open until `G7.8b`/`G7.8c`/`G7.8d` pass |

### Immediate next actions (carried over from gap report)

1. Continue extending `run_with_inputs(...)` / reference coverage from the current verified L1 subset into the remaining unsupported BL5 operators (drives T5.6–T5.8, T6.1).
2. Audit write/index ops for probe-semantics pitfalls (e.g. repeated-index nondeterminism) before counting mismatches as implementation bugs (drives T6.7).
3. Extend artifact writing so performance pass/fail is preserved in `PERF_ALL.csv` and summaries alongside the new correctness fields (T6.9). ✅ Completed in `benchmarks.artifacts`; summaries now aggregate `perf_target` / `perf_actual` / `perf_pass` / `perf_gap`.
4. Wire the coverage gap automation (`python -m benchmarks.stage7_coverage_gap`, written by T6.10) into Stage 7 dashboards and gate verification (drives T6.11 → T6.12). ✅ Gate integration is now substantive: `G7.8b` checks coverage/audit completeness, `G7.8c` checks correctness rows with only explicit memory-policy exclusions, and `G7.8d` checks L1 weighted performance plus L2 fusion performance. Current canonical evidence fails those checks, so this item is implementation-complete but closure-blocking evidence remains.
5. Run targeted missing-shape benchmarks into temporary directories, then merge only new/updated evidence into canonical Track 6 artifacts with `python -m benchmarks.artifacts merge-evidence --source <tmp>/phase1/stage7/track*/l1 --target benchmarks/results/phase1/stage7/track6/l1`. This avoids partial per-op reruns shrinking `PERF_ALL.csv` / `summary.json` by overwriting canonical evidence with an incomplete local run.

## Dependencies

- **Depends on:** S6 (Pass pipeline, OpRegistry, Backend abstraction)
- **Blocks:** S8 (Agent Autonomy needs v0.1.0 Lang/IR semantics, symbolic shapes, full op coverage, and MLIR skeleton)
