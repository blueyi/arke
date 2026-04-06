# Arke — Execution Plan

> Authoritative Gate exit criteria are defined in `stage1-gate-design.md` (and future stage-N series).
> This document records execution history, Phase summaries, and long-term Stage 2-4 roadmap.

## Design Principles

1. **AI-First** — LLM agents make optimization decisions; compilers verify
2. **Minimal-Token** — Minimize total token consumption across the full pipeline: kernel definition, strategy search, compilation, and iterative optimization. Concise IR, structured decisions, and compiler verification eliminate the verbose trial-and-error of direct code generation
3. **Semantic/Strategy Separation** — *What to compute* and *how to optimize* are independent
4. **Compiler-Verified** — Every decision validated by deterministic checks

---

## Current Snapshot (2026-04-05)

### Completed
- ✅ GPU environment (PyTorch 2.6.0+cu124, Triton 3.2.0, RTX 3060)
- ✅ IR system (Semantic IR + Strategy IR, JSON Schema, 10 ops)
- ✅ Builder + Shape Inference (all 10 ops)
- ✅ Validation system (V0 static + V1 numerical + resource estimation)
- ✅ Legal action enumeration engine
- ✅ ArkeEnv full implementation
- ✅ Triton codegen (matmul + softmax, template engine)
- ✅ E2E pipeline (IR → strategy → codegen → GPU)
- ✅ LLM Runner (Anthropic + OpenAI API, fallback, retry)
- ✅ LLM closed-loop optimization (Sonnet 4.6, 23 tool calls, 106% cuBLAS)
- ✅ GPU correctness verification (same-dtype reference)
- ✅ Accuracy benchmark framework (10 metrics, 3-tier verdict)
- ✅ Trajectory JSONL export
- ✅ 237 tests passing

### Gate Status
- G0 ✅ — Triton matmul runs on RTX 3060
- G1 ✅ (⚠️ G1.4 needs 100% revalidation) — IR + Validation
- G2 ✅ — Manual strategy → codegen → 105-160% cuBLAS
- G3 ✅ — LLM tool-use → 106% cuBLAS + softmax correct
- G4 ✅ — Arke vs LLM-direct comparison
- G5 ✅ (3 known-fail perf) — E2E GPT-2 integration
- G6 ⬜ — BL5×L1+L2 — Lang & IR Completeness
- G7 ⬜ — Arke Autonomous Engineering
- G8 ⬜ — BL6×L3 (4 models) + Stage 1 Final Acceptance

→ Details: [stage1-gate-design.md](stage1-gate-design.md)

---

## Phase Overview

```
Phase 1.0 ✅  Environment setup
Phase 1.1 ✅  IR + Validation foundation (⚠️ G1.4 needs 100% fix)
Phase 1.2 ✅  Codegen + E2E pipeline
Phase 1.3 ✅  LLM agent integration
Phase 1.4 ✅  LLM closed-loop optimization
Phase 1.5 ✅  Evaluation framework + comparison
Phase 1.6 ✅  .ak Parser + CLI
Phase 1.7 ✅  Whole-model E2E (G5: 3 known-fail perf criteria)
Phase 1.8 ✅  MVP release (v0.1.0)
Phase 1.9 ⬜  Arke Lang & IR Completeness (Gate G6) ← CURRENT
Phase 1.10 ⬜ Arke Autonomous Engineering (Gate G7)
Phase 1.11 ⬜ Stage 1 Final Validation (Gate G8)
```

---

## Phase 1.0: Environment Setup ✅

**Gate G0** | One-click reproducible development environment with GPU verification.
`make setup` → venv + deps + CUDA smoke test | 237 tests passing

---

## Phase 1.1: IR + Validation Foundation ✅

**Gate G1** | Semantic IR + Strategy IR covering ≥10 operators, static + numerical validation.
10 ops, 6 strategy kinds, JSON Schema round-trip, V0 <1ms, V1 numerical | ✅ (⚠️ G1.4 threshold upgrade pending)

---

## Phase 1.2: Codegen + E2E Pipeline ✅

**Gate G2** | Triton code generation from Strategy IR; GPU perf ≥70% cuBLAS.
matmul + softmax + fused codegen correct | 105-160% cuBLAS | E2E pipeline connected

---

## Phase 1.3: LLM Agent Integration ✅

LLM autonomously completes optimization loop via tool-use, zero human intervention.
10 tools used, 13 decisions, fallback + multi-provider functional

---

## Phase 1.4: LLM Closed-Loop Optimization ✅

**Gate G3** | LLM-optimized kernels ≥50% cuBLAS, correctness verified.
matmul 106.1% cuBLAS | softmax + fused_matmul_relu correct | rollback on failure | 237 tests

---

## Phase 1.5: Evaluation Framework + Comparison ✅

**Gate G4** | Arke vs LLM-direct Triton across ≥5 benchmark tasks.
Arke correctness 100% vs direct 83% | Arke 115.7% vs direct 118.3% cuBLAS (marginal) | variance: Direct fails vary
**Decision:** Proceed — reliability win (100% vs 83%); performance within noise.

---

## Phase 1.6: .ak Parser + CLI ✅

Human-readable `.ak` syntax → Semantic IR; CLI parse/optimize/inspect.
3 .ak examples E2E: matmul, softmax, fused_matmul_relu | `arke parse/optimize/inspect` working

---

## Phase 1.7: Whole-Model End-to-End ✅

**Gate G5** | Replace kernels in real model; inference correctness verified.
GPT-2 Small correctness ✅ | memory ≤6GB ✅ | latency ⚠️ known-fail (1.7-2.3× eager; root cause: Triton dispatch overhead, no graph fusion)
Full analysis: `benchmarks/results/stage1/gates/G5/REPORT.md`

---

## Phase 1.8: MVP Release ✅

v0.1.0 — one-click setup, CI green (3 Python versions), complete docs, reproducible evaluation.
`make setup` verified | CI ×3 Python | API docs 99% | evaluation report published | v0.1.0 tagged

---

## Phase 1.9: Arke Lang & IR Completeness (Gate G6) ⬜

**Gate G6 — BL5×L1+L2** | Arke Lang & IR Completeness Validation

→ Exit criteria, capability backtrack, and dev items: [stage1-gate-design.md §5](stage1-gate-design.md#5-g6--bl5l1l2-lang--ir-completeness-current-target)

---

## Phase 1.10: Arke Autonomous Engineering (Gate G7) ⬜

**Gate G7** | Autonomous kernel generation + model E2E validation

→ Details: [stage1-gate-design.md §6](stage1-gate-design.md#6-g7--arke-autonomous-engineering)

---

## Phase 1.11: Stage 1 Final Validation (Gate G8) ⬜

**Gate G8 — BL6×L3 (4 models)** | Stage 1 Final Acceptance

→ Details: [stage1-gate-design.md §7](stage1-gate-design.md#7-g8--stage-1-final-acceptance)

---

## Stage 2: Ascend Backend — SIMD Architecture Validation

> **Goal:** Verify Arke Lang/IR works on SIMD architecture (Ascend NPU) via Ascend Triton backend.
> Arke-generated Ascend Triton kernels must outperform FlagGems on Ascend.
> Simultaneously complete Arke Lang/IR to cover Category B-E operators.
>
> **Hardware target:** Huawei Ascend 910B (SIMD, CANN)
> **Backend:** triton-ascend (Ascend Triton)
> **Benchmark baseline:** FlagGems/Ascend (P1), CANN library (P0)

### Phase 2.1: Ascend Environment + Arke Lang/IR Completeness

**Objective:** Set up Ascend Triton environment; extend Arke Lang/IR to express Category B-E operators
needed for Stage 2 Gate coverage.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.1.1 | Ascend Triton environment ready | `import torch_npu; triton-ascend` smoke test passes | ⬜ |
| 2.1.2 | Arke Semantic IR: Category B ops | `flash_attention`, `grouped_query_attention`, `multi_latent_attention` in OP_CATALOG | ⬜ |
| 2.1.3 | Arke Semantic IR: Category D ops | `swiglu`, `geglu` in OP_CATALOG | ⬜ |
| 2.1.4 | Arke Semantic IR: Category E ops | `rope`, `yarn_rope` in OP_CATALOG | ⬜ |
| 2.1.5 | Strategy IR: SIMD decision kinds | ≥3 SIMD-specific decision kinds (vector_tile, lane_mapping, simd_width) defined | ⬜ |
| 2.1.6 | Ascend hardware profile | `targets/ascend_910b.json` with SIMD width, memory hierarchy, CANN constraints | ⬜ |

### Phase 2.2: Ascend Triton Codegen

**Objective:** Arke IR → Ascend Triton → NPU execution, correctness verified on Category A+C ops.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.2.1 | matmul correctness on Ascend | Tier 2 (15 shapes): 100% pass V1 numerical | ⬜ |
| 2.2.2 | rmsnorm correctness on Ascend | Tier 2 (7 shapes): 100% pass | ⬜ |
| 2.2.3 | silu/swiglu correctness on Ascend | Tier 2 (5 shapes): 100% pass | ⬜ |
| 2.2.4 | LLM decides SIMD-aware strategy | LLM chooses different tile/mapping for Ascend vs NVIDIA on same kernel | ⬜ |

**Gate S2-G1:** matmul + rmsnorm + swiglu on Ascend — correctness 100% (Tier 2) + LLM SIMD decision verified

### Phase 2.3: Ascend Performance vs FlagGems

**Objective:** Arke/Ascend performance exceeds FlagGems/Ascend on Category A+C+D operators.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.3.1 | matmul Arke/FlagGems ≥ 1.1× (Ascend Tier 2) | `bench_l1 --op matmul --tier 2 --backend ascend` | ⬜ |
| 2.3.2 | rmsnorm Arke/FlagGems ≥ 1.0× (Ascend Tier 2) | `bench_l1 --op rmsnorm --tier 2 --backend ascend` | ⬜ |
| 2.3.3 | swiglu Arke/FlagGems ≥ 1.0× (Ascend Tier 4) | Tier 4 SwiGLU shapes (8 shapes) | ⬜ |

**Gate S2-G2:** Cat A+C+D on Ascend — Arke geomean ≥ FlagGems (Tier 2)

### Phase 2.4: Category B — FlashAttention + GQA

**Objective:** Arke expresses and generates FlashAttention/GQA on both NVIDIA and Ascend.
This is the first complex fused operator gate.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.4.1 | FlashAttention Arke Lang expression | `.ak` kernel for flash_attention compiles and runs correctly | ⬜ |
| 2.4.2 | FlashAttention NVIDIA correctness | Tier 4 FA shapes ≥12/22 (excl. OOM): 100% pass V1 | ⬜ |
| 2.4.3 | FlashAttention NVIDIA performance | ≥0.7× flash-attn-2 baseline (Tier 4 geomean) | ⬜ |
| 2.4.4 | FlashAttention Ascend correctness | Tier 4 FA shapes ≥8/22 (excl. OOM): 100% pass | ⬜ |
| 2.4.5 | GQA correctness (NVIDIA) | Tier 4 GQA 6 shapes: 100% pass | ⬜ |
| 2.4.6 | DeepSeek shapes included | ds-v2/v3 seq=512~16384 shapes pass | ⬜ |

**Gate S2-G3:** FlashAttention (Cat B) — NVIDIA ≥12 shapes correct + ≥0.7× FA-2; Ascend ≥8 shapes correct

### Phase 2.5: Category E — RoPE/YaRN + @rationale cross-arch

**Objective:** RoPE/YaRN codegen for NVIDIA + Ascend; validate @rationale improves cross-arch optimization.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 2.5.1 | RoPE correctness (NVIDIA) | Tier 4 RoPE 9 shapes: 100% pass | ⬜ |
| 2.5.2 | RoPE correctness (Ascend) | Tier 4 RoPE ≥5 shapes: 100% pass | ⬜ |
| 2.5.3 | YaRN RoPE long-context | DeepSeek YaRN ds-v2-8k ~ ds-v2-32k correct | ⬜ |
| 2.5.4 | @rationale cross-arch lift | With SIMD rationale > without: ≥10% perf lift on Ascend (Tier 2 matmul) | ⬜ |

**Gate S2-G4:** RoPE/YaRN (Cat E) correctness + @rationale cross-arch effect verified

### Stage 2 Summary Gate: S2-G_FINAL

| Criterion | Requirement |
|-----------|------------|
| Category coverage | A + B + C + D + E all passing |
| Shape coverage | Tier 2 (15) + Tier 4 sampled (≥20 shapes across all cats) |
| Ascend performance | matmul+rmsnorm+swiglu geomean ≥ FlagGems/Ascend |
| FlashAttention | ≥12 Tier 4 shapes correct on NVIDIA, ≥0.7× FA-2 |
| @rationale | Cross-arch improvement ≥10% verified |
| H4 (cross-hardware) | Same .ak → NVIDIA + Ascend both correct |

---

## Stage 3: MLIR Backend — Complete Compiler Control

> **Goal:** Replace Triton backend with Arke MLIR Dialect for both NVIDIA and Ascend.
> MLIR removes Triton's abstraction ceiling, enabling finer-grained optimization and
> more complete operator support. Performance must match or exceed Stage 2 Triton path.
>
> **Why MLIR after Triton:** Triton's Python abstraction is not AI-First — it constrains
> LLM decisions to Triton's fixed optimization model. MLIR gives Arke full control over
> the lowering pipeline, letting LLM decisions penetrate to loop nest and memory level.
>
> **Hardware target:** NVIDIA + Ascend (via NVVM dialect + AscendNPU IR)
> **Benchmark baseline:** Stage 2 Triton path results

### Phase 3.1: Arke MLIR Dialect Design

**Objective:** Define `arke.kernel` + `arke.strategy` MLIR dialect ops that can represent
all Stage 1+2 optimizations.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.1.1 | `arke.kernel` dialect defined | All Stage 1 ops expressible in arke.kernel | ⬜ |
| 3.1.2 | `arke.strategy` dialect defined | All Stage 1+2 decision kinds representable | ⬜ |
| 3.1.3 | Strategy IR → MLIR transform dialect | Decision kinds map to MLIR transform ops | ⬜ |
| 3.1.4 | Strategy IR Level 2 fields added | loop_nest, memory_access_pattern, pipeline_stage fields in Strategy IR | ⬜ |
| 3.1.5 | LLM can produce Level 2 decisions | LLM session generates ≥1 Level 2 decision in closed-loop | ⬜ |

### Phase 3.2: MLIR Path Correctness (NVIDIA)

**Objective:** All Stage 1+2 operators pass correctness via MLIR path on NVIDIA.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.2.1 | Cat A (matmul) via MLIR | Tier 2+4 (27 shapes): 100% correct | ⬜ |
| 3.2.2 | Cat C (rmsnorm, layernorm) via MLIR | Tier 2 (7 shapes each): 100% correct | ⬜ |
| 3.2.3 | Cat D (swiglu, gelu) via MLIR | Tier 4 SwiGLU (8 shapes): 100% correct | ⬜ |
| 3.2.4 | Cat B (FlashAttention) via MLIR | Tier 4 FA ≥12 shapes: 100% correct | ⬜ |
| 3.2.5 | Cat E (RoPE) via MLIR | Tier 4 RoPE 9 shapes: 100% correct | ⬜ |

**Gate S3-G1:** Cat A+C+D via MLIR — Tier 2+4 correctness 100%

### Phase 3.3: MLIR Path Performance ≥ Stage 2 Triton

**Objective:** MLIR path matches or exceeds Stage 2 Triton path on all categories.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.3.1 | matmul MLIR ≥ Triton path | Tier 2+4 geomean ≥ Stage 2 result | ⬜ |
| 3.3.2 | FlashAttention MLIR ≥ Triton path | Tier 4 FA geomean ≥ Stage 2 result | ⬜ |
| 3.3.3 | rmsnorm/swiglu MLIR ≥ Triton | Tier 2+4 geomean ≥ Stage 2 result | ⬜ |

**Gate S3-G2:** All Cat A+B+C+D MLIR geomean ≥ Stage 2 Triton path

### Phase 3.4: LLM Level-2 Decisions via MLIR

**Objective:** LLM makes loop-nest and memory-access decisions (Level 2) via Strategy IR,
and MLIR path reflects these decisions with measurable performance benefit.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.4.1 | LLM produces loop_nest decisions | LLM session uses Level 2 loop_nest field in ≥3 shapes | ⬜ |
| 3.4.2 | LLM Level 1+2 > Level 1 + default | Tier 2 matmul: LLM L1+2 ≥ LLM L1+default L2 by ≥15% | ⬜ |
| 3.4.3 | FlashAttention Level 2 optimization | FA Tier 4: LLM Level 2 decisions improve vs default by ≥20% | ⬜ |

**Gate S3-G3:** LLM Level-2 decision value verified (matmul ≥15%, FA ≥20%)

### Phase 3.5: Ascend via MLIR

**Objective:** MLIR path generates correct Ascend code via AscendNPU IR dialect.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 3.5.1 | MLIR → AscendNPU IR | matmul + rmsnorm correct on Ascend via MLIR path | ⬜ |
| 3.5.2 | Ascend MLIR ≥ Ascend Triton path | Tier 2 geomean: MLIR path ≥ Stage 2 Triton result | ⬜ |

**Gate S3-G4:** Ascend via MLIR correct + performance ≥ Stage 2

### Stage 3 Summary Gate: S3-G_FINAL

| Criterion | Requirement |
|-----------|------------|
| Category coverage | A + B + C + D + E all via MLIR path |
| Shape coverage | Tier 2 + Tier 4 across all categories |
| Performance | MLIR path geomean ≥ Stage 2 Triton (NVIDIA + Ascend) |
| LLM Level-2 | Verified benefit ≥15% on matmul, ≥20% on FlashAttention |
| Multi-hardware | NVIDIA + Ascend both via MLIR |

---

## Stage 4: LLVM IR Backend — 100% Hardware Completeness

> **Goal:** Arke IR → LLVM IR direct emission, achieving 100% hardware expression completeness
> and maximum performance headroom beyond what MLIR lowering allows.
>
> **Why LLVM IR after MLIR:** MLIR still imposes abstraction constraints in its lowering pipeline.
> Direct LLVM IR emission gives Arke full control: register allocation hints, barrier placement,
> instruction scheduling. This is where LLM Level-3 decisions become possible.
>
> **Hardware target:** NVIDIA + Ascend + AMD (≥3 backends via LLVM)
> **Benchmark baseline:** Stage 3 MLIR path results

### Phase 4.1: LLVM IR Emission Foundation

**Objective:** Arke IR → Loop Nest IR → LLVM IR, correctness verified on Cat A.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.1.1 | Loop Nest IR design | Intermediate representation between Strategy IR and LLVM IR defined | ⬜ |
| 4.1.2 | matmul → LLVM IR correct | Tier 2 matmul (15 shapes): 100% correct via LLVM path | ⬜ |
| 4.1.3 | Strategy IR Level 3 fields | register_hints, barrier_placement, instruction_scheduling fields defined | ⬜ |
| 4.1.4 | LLVM path on ≥2 backends | matmul correct on NVIDIA (NVPTX) + AMD (AMDGCN) | ⬜ |

**Gate S4-G1:** matmul via LLVM correct on ≥2 backends (Tier 2)

### Phase 4.2: LLVM Path Performance ≥ Stage 3 MLIR

**Objective:** LLVM path achieves better performance than MLIR path on Cat A+C+D.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.2.1 | matmul LLVM ≥ MLIR + 5% | Tier 2+4 geomean: LLVM ≥ Stage 3 MLIR + 5% | ⬜ |
| 4.2.2 | rmsnorm/swiglu LLVM ≥ MLIR | Tier 2+4 geomean: LLVM ≥ Stage 3 MLIR | ⬜ |

**Gate S4-G2:** Cat A+C+D LLVM geomean ≥ MLIR + 5% (NVIDIA)

### Phase 4.3: LLM Level-3 Decisions

**Objective:** LLM makes register/barrier/scheduling decisions (Level 3) via Strategy IR Level 3 fields.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.3.1 | LLM produces Level 3 decisions | LLM session uses register_hints / barrier_placement in ≥3 shapes | ⬜ |
| 4.3.2 | LLM L1+2+3 > L1+2+default L3 | Tier 2 matmul: LLM full stack ≥ L1+2+default by ≥5% | ⬜ |

**Gate S4-G3:** LLM Level-3 decision value verified (≥5% over L1+2+default)

### Phase 4.4: FlashAttention via LLVM + Multi-Hardware Parity

**Objective:** FlashAttention via LLVM path on ≥3 backends; performance within 90% of vendor libraries.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.4.1 | FlashAttention LLVM correct | Tier 4 FA ≥12 shapes on NVIDIA via LLVM: 100% | ⬜ |
| 4.4.2 | FlashAttention LLVM performance | ≥0.85× flash-attn-2 baseline (better than Stage 2's 0.7×) | ⬜ |
| 4.4.3 | MLA (DeepSeek) LLVM correct | Tier 4 MLA 8 shapes: 100% correct | ⬜ |
| 4.4.4 | ≥3 hardware backends | NVIDIA + Ascend + AMD all pass Cat A correctness | ⬜ |
| 4.4.5 | Multi-hardware perf within 90% vendor | Each backend: matmul geomean ≥90% of respective vendor lib | ⬜ |

**Gate S4-G4:** FlashAttention + MLA via LLVM correct; multi-hardware ≥90% vendor parity

### Phase 4.5: Production Release v1.0.0

**Objective:** Stable public API, comprehensive benchmark suite across ≥3 hardware platforms.

#### Completion Criteria
| # | Criterion | Verification | Status |
|---|-----------|-------------|:------:|
| 4.5.1 | pip package + stable API | `pip install arke` + API docs complete | ⬜ |
| 4.5.2 | Benchmark suite ≥3 platforms | NVIDIA + Ascend + AMD results published | ⬜ |
| 4.5.3 | @rationale knowledge base ≥200 entries | Covers Cat A-E across NVIDIA + Ascend + AMD | ⬜ |
| 4.5.4 | v1.0.0 tag | `git tag v1.0.0` | ⬜ |

**Gate S4-G_FINAL:** v1.0.0 — multi-hardware parity + @rationale KB + LLM Level 1-3 full stack

### Stage 4 Summary Gate: S4-G_FINAL

| Criterion | Requirement |
|-----------|------------|
| Category coverage | A + B + C + D + E + F (lm_head/vocab) |
| Shape coverage | Tier 2 + Tier 4 all categories incl. MLA/GQA/YaRN |
| Performance vs Stage 3 | LLVM geomean ≥ MLIR + 5% (Cat A+C+D) |
| FlashAttention | ≥0.85× FA-2 baseline via LLVM |
| Multi-hardware | ≥3 backends ≥90% respective vendor libs |
| LLM Level-3 | Verified benefit ≥5% over L1+2+default |
| @rationale KB | ≥200 entries, cross-hardware coverage |

---

## Stage Dependency Chain

```
Stage 1 ✅  SIMT feasibility (NVIDIA Triton)      → proves H1/H2/H3
    ↓ G5 PASS → Gate details: stage1-gate-design.md
Stage 2     SIMD feasibility (Ascend Triton)      → proves H4 (cross-arch)
    ↓ S2-G_FINAL PASS
Stage 3     MLIR backend (full compiler control)  → deeper LLM decisions (Level 2)
    ↓ S3-G_FINAL PASS
Stage 4     LLVM IR backend (100% HW completeness)→ LLM Level 1-3 full stack
    ↓ S4-G_FINAL PASS
  v1.0.0
```

**Gate benchmark coverage requirement (Stage 2+):**
- Every Gate must cover ≥3 Operator Categories (A-G, see benchmark-design.md §2)
- Shape coverage must include Tier 4 shapes from `docs/design/benchmark-design.md`
- DeepSeek shapes (seq=512~163840) must be included in FlashAttention + MLA Gates

---

## Risk Matrix

| Risk | Affects | Mitigation |
|------|:-------:|-----------|
| LLM decisions don't map to Triton templates | Phase 1.4 | Expand template coverage + parameter adaptation layer |
| compile_and_profile errors in LLM session | Phase 1.4 | Better error messages + graceful degradation |
| 6GB VRAM insufficient for large shapes | Phase 1.7 | Limit shapes to ≤2048 |
| Arke doesn't outperform direct Triton | Phase 1.5 | Gate G4 decision matrix |
| API timeout / rate limit | Phase 1.4-1.5 | Retry + fallback + prefer Sonnet over Opus |

---

*Last updated: 2026-04-05*
