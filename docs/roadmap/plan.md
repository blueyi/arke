# Arke Development Plan

> Single source of truth for development roadmap.
> Detailed stage plans → docs/phase1/stage{N}-plan.md
> Architecture → docs/architecture/; Specs → docs/spec/

## Hierarchy

Roadmap > Phase > Stage > Feature > Task

## Core Thesis — Three-Level Decomposition

> **Locked principle (2026-05-17, Leon-approved):** the Arke project's claim — *"LLM-as-decision-maker + structured IR + compiler verification produces correct, high-performance AI operators across architectures and abstraction layers"* — is split into three falsifiable sub-theses. Each level has an independent **kill criterion**. This prevents the global thesis from being indefinitely deferred to "the next Phase".

### Thesis L1 — Single-Architecture, Single-Backend (Phase 1)

**Claim:** The paradigm produces correct + high-performance AI operators on **SIMT** architecture via the **Triton** backend (one abstraction layer).

- **Validation window:** Phase 1 (S0–S9)
- **Pass evidence:** G8 + G9 — GPT-2 / LLaMA-2 / DS-V2 all meet correctness=100% + perf thresholds; BL5 inheritance holds; autonomous engineering (G7-AE) reproducible end-to-end.
- **Kill criterion:** If G8[4]/[5]/[6] cannot pass *with reasonable engineering* (≤ 2 Stage 8 extensions), or LLM-best ≤ heuristic-floor in ≥ 50% of trajectories, the LLM-decision pillar is falsified for SIMT/Triton.
- **Current status (2026-06-26):** L1 **validated** on dev-HW — Phase 1 CLOSED (2026-06-25). G7 ✅, G8 ✅ (gate 6/6 PASS), G9 ✅ CLOSED. GPT-2 1.0296× / LLaMA-family 1.239× / Qwen2.5-family 1.2796× eager, correctness 100%; Arke vs LLM-direct 1.263× @ 0 tokens; @rationale KB 292 entries. Only deferred: full 7-8B parameter scale (6GB VRAM limit, family-substitute口径 accepted by project lead); release tags G9[4]/[6] waived.

### Thesis L2 — Cross-Architecture, Same Abstraction Layer (Phase 2)

**Claim:** The same `SemanticIR + StrategyIR` schema, the same harness, and the same trajectory corpus can be **re-used** to drive a **SIMD/heterogeneous-pipeline** backend (Ascend 910B Cube/Vector via Triton-Ascend) without rewriting the IR.

- **Validation window:** Phase 2 (Stage 10–11+, TBD)
- **Pass evidence:** ≥80% of Phase 1 ops port to Ascend with **no new `Decision` kinds**, and ≥1 LLM-decided strategy on Ascend outperforms the Ascend heuristic floor.
- **Kill criterion:** If StrategyIR must add ≥3 architecture-specific decision kinds (e.g. `cube_pipeline_stage`, `vector_buffer_double_buffer`) just to express Ascend strategy, or if SIMT trajectories provide **zero** transfer learning to SIMD decisions, L2 is falsified — IR is not architecture-agnostic, must redesign.
- **Status:** ⬜ unstarted.

### Thesis L3 — Cross-Abstraction-Layer (Phase 3-5)

**Claim:** As Arke lowers further down the stack (MLIR → vendor-DSL → LLVM-IR), **performance improves monotonically** while correctness and LLM-decision quality hold.

- **Validation window:** Phase 3 / Phase 4 / Phase 5
- **Pass evidence:** Per Phase, geomean perf on the BL6 model set strictly improves vs the previous Phase's same-backend baseline (e.g. Phase 3 MLIR-direct geomean ≥ 1.05× Phase 1 Triton geomean on identical hardware).
- **Kill criterion:** If **lowering loss** (information lost across each compiler stage) exceeds **LLM-decision gain** at any Phase — i.e. deeper stack yields worse performance — the "cross-abstraction-layer" claim collapses; the project halts or backtracks to the last winning Phase.
- **Status:** ⬜ unstarted.

### Why three levels matter

- **L1 is the floor.** If L1 fails, the entire thesis fails — no need to start Phase 2.
- **L2 tests IR generality.** This is where the bet "structured IR can abstract over architectures" gets tested with hard data, not aspirational wording.
- **L3 tests compiler-stack value.** Without L3, Arke could collapse into "yet another Triton frontend"; L3 forces the value-add of going deeper to be measured.

Each Stage Gate cites which Thesis level it advances. See per-Stage docs for the explicit mapping.

---

## Gate Governance

> **Gates are the contract between design and development.**
>
> Gate exit criteria define verifiable acceptance standards for each Stage. Once a Phase/Stage's Gate criteria are finalized, they are **locked** — any adjustment requires explicit approval from the project lead.
>
> Development should be **Gate-driven**: work backward from Gate exit criteria to determine what Arke Lang, IR, Compiler, and Agent need to deliver. Gates drive design and implementation, not the other way around.
>
> All Gate criteria that involve operator-level performance or correctness **must** align to the BL/OT/ST/L benchmark system defined in `docs/benchmark/benchmark-design.md`. The Gate-Purpose Mapping below specifies which benchmark levels each Stage's Gate must satisfy.

### Benchmark vs Gate Thresholds — Separation of Concerns

> **Locked principle (2026-05-16, Leon-approved):** the Benchmark **measurement** layer and the Gate **acceptance** layer are decoupled.
>
> - **Benchmark design is frozen** — BL/OT/ST/L framework, measurement protocol, shape sets, baseline-ladder collection, and PERF_ALL schema do **not** change to accommodate Gate pass/fail outcomes.
> - **Gate thresholds may be re-calibrated against theoretical performance** — when a Phase/Stage/Track's compiler-backend ceiling is bounded by the chosen backend's physical capability (e.g. Triton kernel-launch overhead vs PyTorch fused CUDA dispatch on tiny elementwise), the **Gate exit criteria** may be adjusted to reflect what is *physically achievable on that backend*, without weakening the benchmark itself.
> - Every Gate threshold adjustment must (a) cite the physical/theoretical reason, (b) preserve the benchmark measurement faithfully (no shape removal, no op exclusion from PERF_ALL), and (c) carry the project lead's explicit approval before merging.

### Same-Backend Fairness — Triton Path

> **Locked principle (2026-05-16, Leon-approved):** when the compiler backend under test is **Triton**, the Gate performance comparison **denominator is the corresponding operator's Triton implementation**, not a cross-backend reference.
>
> - **Numerator:** Arke-generated Triton kernel latency
> - **Denominator:** the **fastest Triton-only implementation** of the same operator available in the ladder for this op (e.g. FlagGems Triton kernel, Liger Triton kernel, Unsloth Triton kernel, vLLM Triton kernel, flash-attn Triton kernel — picked per-op by the ladder's PRIMARY+FALLBACK ordering in `docs/benchmark/golden-kernel-ladder.md`)
> - **Pass criterion:** `arke_latency ≤ triton_reference_latency × (1 + ε)` with `ε = 0.03` (3% measurement-noise tolerance)
> - **Audit-only:** when no Triton-only implementation exists for the op-shape, the row is **audit-only** (excluded from Gate scoring) and recorded with `perf_oracle_unavailable_triton=true` for future review.
> - **Why same-backend:** cross-backend comparisons (Arke-Triton vs cuBLAS / FlagGems-CUDA / torch.compile-Inductor / PyTorch-eager-fused-dispatch) conflate compiler-quality with backend-architectural advantages. Same-backend fairness isolates Arke's compilation quality.
> - **When the backend changes:** Phase 2 (Triton-Ascend) compares against Ascend-Triton baselines; Phase 3 (MLIR) compares against MLIR-native baselines; Phase 4 (C-like kernel DSL) compares against hand-tuned CUDA-C / CCE-C / Bang-C references; Phase 5 (LLVM IR) compares against LLVM-IR-direct references. The same-backend principle applies uniformly.

---

## Phase 1: Arke → Triton → NVIDIA GPU (SIMT Validation)

**Goal:** Prove LLM + structured IR + compiler verification produces correct, fast kernels on SIMT architecture.
**Hardware:** NVIDIA Ampere (RTX 3060 Laptop, 6GB VRAM) · CUDA 12.4 · PyTorch 2.6.0+cu124 · Triton 3.2.0

### Design Principles

1. **AI-First** — LLM agents make optimization decisions; compilers verify
2. **Peak Performance** — Pursue maximum hardware utilization; every abstraction layer must justify its overhead
3. **Minimal-Token** — Minimize total token consumption across the full pipeline
4. **Semantic/Strategy Separation** — *What to compute* and *how to optimize* are independent
5. **Compiler-Verified** — Every decision validated by deterministic checks

---

### Stage Summary

| Stage | Gate | BL Exit | L Layer | Objective | Status |
|:-----:|:----:|:--------|:-------:|:----------|:------:|
| S0 | G0 | — | — | GPU Environment | ✅ |
| S1 | G1 | — | — | IR + Validation | ✅ |
| S2 | G2 | BL1×L1 (matmul) | L1 | Codegen + E2E Pipeline | ✅ |
| S3 | G3 | BL1×L1 (matmul+softmax) | L1 | LLM Agent Closed Loop | ✅ |
| S4 | G4 | BL2×L1 (6 tasks) | L1 | Arke vs LLM-direct | ✅ |
| S5 | G5 | BL3×L1 + BL6/GPT-2×L3 | L1+L3 | Whole-Model E2E | ✅ |
| S6 | G6 | BL4×L1 (46 ops correctness + ≥1.00× P3) | L1 | Compiler Infrastructure | ✅ 7/7 |
| S7 | G7 | BL5×L1+L2 | L1+L2 | Lang & IR v0.1.0 (high-level IR ready for Phase-3 MLIR consumption) | ✅ 13/14 closed (G7.8d honest-gap accepted per Gate Governance v2; S7.followup.1–3 open) |
| S8 | G8 | **Harness system (Tier 1)** + BL5 inherit + BL6×L3 endpoint validation (Tier 2) | L1+L2+L3 | **Build extensible Arke Harness** for LLM-driven op auto-gen/tune (Stage 8 primary deliverable); BL6×L3 = Thesis L1 endpoint validation | ✅ **gate 6/6 PASS (2026-06-25):** Tier-1 all green (Façade v1.0 frozen, live-LLM loop, Demo A+B); Tier-2 [1][2][3]✅ / [4a] GPT-2 geomean 0.9517 (D3) / [4b] bridge 48×/forward / [5] LLaMA-family 1.239× / [6] DS-V2 audit-only / [7] BL5 no-regression. |
| S9 | G9 | BL6×L3 (4 models) + BL5 regression | L1+L2+L3 | Phase 1 Final | ✅ CLOSED 2026-06-25 (dev-HW): G9[1] 4-model family-substitute accepted, G9[2] Arke 1.263×>LLM-direct +0 tokens, G9[3] KB 292; G9[4]/[6] release tags waived by project lead. Thesis L1 validated. |

### Gate-Purpose Mapping

> Aligns each Gate to the BL/OT/ST/L benchmark system. See `docs/benchmark/benchmark-design.md` for definitions.

| Gate | BL Exit | L Layer | Key Benchmark Requirement | Baselines |
|:----:|:--------|:-------:|:--------------------------|:----------|
| G0 | — | — | Environment prerequisite | — |
| G1 | — | — | IR infrastructure prerequisite | — |
| G2 | BL1×L1 | L1 | matmul ≥70% P0 (cuBLAS) | P0 |
| G3 | BL1×L1 | L1 | LLM-driven matmul ≥100% P0 | P0 |
| G4 | BL2×L1 | L1 | 6 tasks: Arke correctness ≥ LLM-direct, geomean ~P1 | P0, P1, P5 |
| G5 | BL3×L1 + BL6/GPT-2×L3 | L1+L3 | OT0-2×ST1-3 correctness 100%; GPT-2 top-1 correct | P0, P3 |
| **G6** | **BL4×L1** | **L1** | 46 ops correctness 100% via SemanticInterpreter; perf ≥1.00× P3 | P3 |
| **G7** | **BL5×L1+L2** | **L1+L2** | OT0-4×ST1-4 correctness 100%; **same-backend Triton fairness**: Arke-Triton ≥ (1−ε)·Triton-ref with ε=0.03; 4/4 fusions; per-op denominator = best Triton-only implementation in ladder | Triton-only ladder (FlagGems / Liger / Unsloth / vLLM-Triton / flash-attn) |
| **G8** | **Harness system (Tier 1) + BL5 inherit + BL6×L3 endpoint (Tier 2)** | **L1+L2+L3** | **Tier 1**: Façade v1.0 + LLM autonomy + extensibility (1 new op ≤300 LOC + 1 new baseline ≤200 LOC). **Tier 2**: GPT-2 ≥0.95× eager; LLaMA-2 ≥0.90×; DS-V2 ≥0.85×; auto-strategy ≥0.95× P0 | P0, P1, P3 |
| **G9** | **BL6×L3 (4 models) + BL5 regression** | **L1+L2+L3** | GPT-2 ≥1.00× eager; LLaMA-2/3 ≥0.95×; Qwen2.5 ≥0.90×; Arke ≥1.05× P5 | P0, P1, P5 |


---

### Completed Stages (S0-S5)

#### Stage 0 (G0): GPU Environment ✅

Established a reproducible one-click GPU development environment. `make setup` produces a working venv with PyTorch + Triton + CUDA verification on a fresh clone. All 237 tests pass with no import errors. **Key result:** RTX 3060 Laptop 6GB (Ampere SM 8.6) confirmed functional as the Phase 1 development target.

#### Stage 1 (G1): IR + Validation ✅

Built Semantic IR and Strategy IR covering 10 operators with 6 decision kinds, JSON Schema round-trip, V0 static validation (<1ms), and V1 numerical validation against NumPy reference. Shape inference works for all 10 ops. **Key result:** 237 tests passing; IR infrastructure proven capable of expressing and validating operator optimization decisions. (⚠️ G1.4 threshold upgrade pending full revalidation.)

#### Stage 2 (G2): Codegen + E2E Pipeline ✅

Hand-written strategy blocks translate through Triton codegen to GPU execution. matmul, softmax, and fused matmul+relu all pass V1 numerical validation; GPU performance reaches 105–160% cuBLAS. The full pipeline (IR → strategy → codegen → compile → profile) runs end-to-end in a single call. **Key result:** BL1×L1 gate passed; H1 (structured protocol improves correctness) validated.

#### Stage 3 (G3): LLM Agent Closed Loop ✅

LLM autonomously completes the optimization loop using all 10 tools, 23 tool calls, 13 strategy decisions, zero human intervention. Multi-provider support (Anthropic + OpenAI-compatible) and fallback/retry functional. **Key result:** matmul 2048² → 151.4% cuBLAS; lm-head (50257) → 116.5% cuBLAS. H2 (structured search superior to manual) validated.

#### Stage 4 (G4): Arke vs LLM-direct ✅

Quantitative comparison across 6 benchmark tasks. Arke correctness 100% vs LLM-direct 83%; Arke token consumption ≤60% of LLM-direct; Arke/FlagGems geomean = 0.991. Performance slightly lower (115.7% vs 118.3% cuBLAS) but variance significantly smaller. **Key result:** Gate decision: proceed — Arke wins on reliability and token efficiency; performance gap is within noise.

#### Stage 5 (G5): Whole-Model E2E ✅

GPT-2 Small E2E inference: top-1 token correctness 100%, 49/48 Conv1D replacements, memory ≤1100MB/6144MB. **Known-fail (recorded, non-blocking):** E2E latency 1.71–2.20× eager due to monkey-patch dispatch ~60µs/call × 49 calls (root cause: Python dispatch overhead, not kernel quality). Single matmul: Arke 76µs vs cuBLAS 44µs. Resolution: S8 torch.compile Inductor backend. Full analysis: `benchmarks/results/phase1/gates/G5/REPORT.md`.

---

### Stage 6 (G6): Compiler Infrastructure ✅

**Objective:** Refactor the compiler toolchain into a clean, extensible architecture. OpRegistry as single source of truth, Pass pipeline for composable transformations, Backend abstraction for multi-target support.

**Why this comes first:** All subsequent stages (IR/Lang, Agent autonomy, multi-model E2E) depend on a solid compiler foundation. Without OpRegistry, adding ops requires touching 6 files. Without Pass infrastructure, IR transformations are ad-hoc. Without Backend abstraction, Triton is hardcoded everywhere.

**BL Exit:** BL4×L1 — Full 46 ops correctness 100% via SemanticInterpreter + performance ≥1.00× P3 (eager).

**Gate G6 PASS Criteria:**

```
AND ALL:
  [1] OpRegistry: single source of truth for all 46 ops (adding op ≤ 2 files)
  [2] SemanticInterpreter: PyTorch eager executor, 46 ops correctness 100%
  [3] Pass Infrastructure: ArkePass protocol + PassPipeline with ≥2 passes
  [4] SSA Validator: validates all 46 ops; rejects ≥5 invalid IR examples
  [5] Backend Abstraction: ArkeBackend protocol + TritonBackend implements it
  [6] Codegen + GPU execution: 46 ops via TritonBackend, correctness 100%, perf ≥1.00× P3
  [7] Non-regression: ≥422 tests passed, ≤6 skipped, 0 new failures
```

**Closed:** 2026-06-06 on `feat/g6-closure` — 7/7 PASS at 46/46 ops after D8-X1 catalog growth (silu_and_mul / gelu_and_mul / swiglu_packed) + OT4 attention tier-3 re-run + qkv_fa-shape3 probe fix (CPU fp64 escape from FlagGems aten::mm hijack).

→ Detailed plan: [docs/phase1/stage6-plan.md](../phase1/stage6-plan.md)

---

### Stage 7 (G7): Lang & IR v0.1.0 ✅

**Objective:** Land the finalized Arke Lang v0.1.0 and the **high-level Arke IR v0.1.0** (Layer 4 SemanticIR + Layer 3 StrategyIR) as the implementation contract for Phase 1's Triton path **and** as the IR surface that Phase 3 will lower through MLIR. Implement the `where` clause + symbolic shape system end-to-end. Upgrade StrategyIR to be backend-agnostic in its core. Complete spec documents. Assess dynamic shape feasibility. Establish the MLIR framework skeleton as a forward-compatibility checkpoint (concrete MLIR lowering work belongs to Phase 3).

**Why this follows S6:** Pass pipeline (from S6) is needed for IR layer transformations. Backend abstraction (from S6) is needed for backend-agnostic strategy validation. OpRegistry (from S6) is needed for spec completeness verification.

**Scope clarification (locked 2026-05-16):** Stage 7 owns the **high-level IR spec + its in-tree implementation**. Stage 7 does **not** own concrete MLIR lowering — that is Phase 3's contract. G7.5 (MLIR framework skeleton) remains in scope as a forward-compatibility artifact: it proves the high-level IR can be lowered to MLIR-shaped surface, but full MLIR-dialect engineering is deferred.

**BL Exit:** BL5×L1+L2 — All 45 ops (OT0-4) × all shapes (ST1-4) correctness + performance at L1 single-op and L2 fused-op levels, evaluated under the **Same-Backend Fairness (Triton)** rule (see Gate Governance).

**Gate G7 PASS Criteria:**

```
AND ALL:
  [1] Arke Lang Spec v0.1.0 document finalized
  [2] Arke IR Spec v0.1.0 document finalized (Layer 4/3/2/1 defined; Layer 4+3 implemented)
  [3] where clause MVP: parses + SemanticIR symbolic_dims populated
  [4] Dynamic Shape feasibility assessment document complete
  [5] MLIR framework skeleton: MLIREmitter exists, BL1 matmul verified (forward-compatibility checkpoint; full MLIR lowering deferred to Phase 3)
  [6] All 45 ops: .ak → SemanticIR → StrategyIR full round-trip
  [7] Token efficiency: .ak lines < Triton lines for all OT0-OT4
  [8] Backend-agnostic strategy: 0 Triton-specific fields in StrategyIR core
  [9] L1 BL5 correctness: 100%(ST1-4, excl. OOM) for all OT0-OT4
  [10] L1 BL5 performance (Same-Backend Triton Fairness): for each OT group, ≥97% of evaluable rows pass `arke_latency ≤ triton_ref_latency × 1.03`; weighted_score ≥ 0.95 with the same OT0_1/OT2/OT3/OT4 = 0.25/0.30/0.20/0.25 weights. Rows with no Triton-only reference in the ladder are audit-only.
  [11] L2 BL5: 4/4 fusion combinations pass under Same-Backend Triton Fairness
  [12] Non-regression: ≥422 tests, 0 new failures
```

→ Detailed plan: [docs/phase1/stage7-plan.md](../phase1/stage7-plan.md)

**Track 6 artifact status (current):** Stage 7 benchmark automation emits a consolidated root-level dashboard artifact set under `benchmarks/results/phase1/stage7/track6/` — `coverage_gap.json`, `audit_report.json`, `stage7_operator_shape_stats.json`, and `dashboard.json` — plus per-layer `l1/` and `l2/` benchmark manifests. `benchmarks.gate_g7` now validates both the result-tree contract and the substantive BL5 evidence contract: coverage completeness, correctness rows, memory-policy exclusions, L1 weighted performance, and L2 fusion performance.

**Current G7 evidence status (2026-05-16, final):** Implementation/test slices for Lang, IR, MLIR skeleton, examples, backend-agnostic StrategyIR, and non-regression are green (13/14 sub-gates PASS). After the 2026-05-14 rope-fp32 (commit `82b635b`) + gated odd-N typed-unsupported (commit `160ebf4`) + 2026-05-15 G7.8b coverage closure (commit `13d42e6`) fixes, canonical Track 6 evidence passes G7.8c (correctness 863/863 ok across all baselines) and G7.8b (BL5 L1 coverage 45/45 ops, 685/685 shapes). G7.8d has been re-calibrated under the **Same-Backend Triton Fairness** rule locked on 2026-05-16 (commits `c366116` docs + `3c89a23` scoring): the perf denominator is now `min(latency_us)` over Triton-only baselines per `(layer, op, shape_tag)`, with ε=0.03 tolerance and audit-only handling when no Triton reference is available. The 2026-05-16 evening honest-gap closure round (commits `a5431c5` rmsnorm dedicated template + `25b279c` layernorm refresh + `ee48c35` rope/batch_matmul/matmul refresh + `a2c659b` summary docs) refreshed five ops' PERF_ALL rows against the live Triton ladder. Under this honest ruler with fresh data, current Track 6 PERF_ALL (2502 rows) yields **L1 weighted_score=0.3006** (per-OT pass rates: ot0_1=62.7% [220/351], ot2=26.8% [19/71], ot3=31.8% [7/22], ot4=0/0 due to missing Triton attention ref data) and L2 fusions=0 evaluable. Audit-only rows: `perf_oracle_unavailable_triton=438`, `non_arke_baseline_skipped=1826`.

The G7.8d weighted-score threshold (≥0.95) is **mathematically unreachable** under the current weights (`{ot0_1: 0.25, ot2: 0.30, ot3: 0.20, ot4: 0.25}` in `benchmarks/gate_g7.py:673`): with ot4=0 by construction (no Triton attention baseline exists in our ladder), the maximum achievable weighted score is 0.75 < 0.95. **Project lead decision (2026-05-16):** Per Gate Governance v2, accept G7=13/14 closed as honest Stage 7 completion; G7.8d gap is real distance vs ladder-fastest Triton kernels and is tracked as Stage 7 follow-up rather than blocking. Three follow-up streams remain open (will land in S8 or as S7.followup):

- **S7.followup.1** — L1 Triton-template perf ladder upgrade + dead-kernel revival + harness measurement-fidelity fixes. **✅ SUBSTANTIALLY LANDED (2026-06-28).** The original "rope 35.7% / batch_matmul 0% / matmul 35.3%" pass rates were largely **measurement artifacts**, not codegen gaps — root-caused and fixed:
  - **Harness perf-path shape-encoding bug** (`2baa0bb`, `9ef96a9`): every baseline runner's `get_fn` perf path built inputs with a squashed `(M,N,K)` convention that mismatched shape-encoded ops (batch_matmul B/M/K/N, grouped_matmul B/E/M/K/N, gated seq/ffn) — measuring the *wrong workload* (e.g. batch_matmul reported 146998µs vs real ~2400µs). Extracted a shared `_shared_inputs.py` builder consulting `get_current_shape()`; all runners harmonized → Arke-vs-baseline ratios now apples-to-apples. Full diagnosis: `docs/benchmark/harness-perf-shape-encoding-bug.md`.
  - **transpose audit-only + materialized eager baseline** (`c29e5d8`): bogus 0.011× was a lazy-view eager denominator; baselines now materialize (`.contiguous()`), transpose marked audit-only (no Triton-only golden).
  - **8 DEAD kernels revived** (`020823a` quantize_per_token + dequantize_per_channel; `fcd1160` cast/split/topk/copy_/embedding/permute): all were silently falling back to the PyTorch interpreter (scalar-arg launchers the tensor-only backend can't dispatch, `tl.math.nearbyint` absent in Triton 3.2, missing wrapper defs, unhashable extra_ctx in kernel cache). Dead-kernel count **8→0**; every op now runs a real Triton kernel. Detection: `_try_triton_runner(...) is None` ⇒ silent fallback.
  - **Codegen perf-gap closures** (real, post-harmonization): rope (`08e095f`) dead→**1.6–2.2× faster than Liger Triton**; softmax (`3dd38d4`) 0.75→~0.9 (worst wide-vocab-llama3 0.384→0.88) via single-pass threshold 4096→16384; embedding (`8cb5d66`) 0.28→0.78 via one-program-per-row + drop index cast.
  - **gather + scatter correctness bugs fixed** (`8dff103`, `7df586f`): both implemented a *row* gather/scatter but the schema is `torch.gather/scatter_(-1, idx)` *column* ops; gather additionally caused a CUDA illegal-memory-access crash at large shapes. Both rewritten to correct column semantics with OOB-clamp, bit-exact vs torch.
  - **Residual gaps are architectural, not codegen:** matmul (0.91× vs cuBLAS), batch_matmul (0.85×), reduce_sum/mean (~0.92×) sit near the cuBLAS / launch-overhead floor on small-M shapes — documented, low-ROI. Result: 22+ ops pass (Arke ≥ Triton golden), 0 correctness regressions, full suite 2016 tests pass.
- **S7.followup.2** — L2 Triton fusion baseline collection (currently 0 evaluable).
- **S7.followup.3** — OT4 attention Triton baseline integration (hard prerequisite to unlock the BL5-inherit gate at G8; the 0/0 ot4 group is the single largest contributor to the math-impossible 0.75 ceiling). **✅ LANDED (2026-06-07):** branch `feat/s7-followups` commits `3016ffa` (C1: FlagGems P1 golden, cuBLAS attention de-listed) + C2 (bench rerun + PERF_ALL refresh) close OT4. FlagGems was lifted into the OT4 golden slot for `flash_attention` / `grouped_query_attention` / `cross_attention` under the Same-Backend Triton Fairness rule — FlagGems is the only audited library shipping a production-quality Triton SDPA on sm 8.6 (RTX 3060). A same-backend-fairness lie in the prior ladder was also cleaned up: cuBLAS/cuDNN P0 used to declare `supports('flash_attention')` but its `get_fn` called `F.scaled_dot_product_attention`, which dispatches via ATen and gets globally hijacked into FlagGems Triton the moment `flag_gems.enable()` runs — name said "cuBLAS/cuDNN", backend was actually Triton. `multi_latent_attention` + `paged_attention` remain `audit-degraded` (no production Triton kernel exists for them in the 9 audited community libraries). **G7.8d outcome:** `ot4=26/26 (1.000)`, `weighted_score=0.5506` (up from frozen 0.3006, +83%); OT4 group fully evaluable. ot0_1/ot2/ot3 gaps remain — handled by S7.followup.1 / .2.

See [docs/phase1/stage7-completion-summary.md](../phase1/stage7-completion-summary.md) §9 for the evening honest-gap closure trajectory (weighted_score 0.3000 → 0.3214 mid-pass → 0.3006 final after fresh data exposed previously-stale failures).

**Memory evidence note (current):** skipped benchmark rows now carry memory preflight metadata in artifact CSVs, including `memory_bytes_required`, `memory_bytes_budget`, `memory_ratio`, and `memory_policy`. The evidence path is no longer attention-only; OT2 / OT3 pressure is represented the same way as OT4 attention pressure, which keeps BL5 coverage accounting honest under 6GB VRAM constraints.

**Golden Kernel protocol (2026-05-11):** L1 correctness/perf now follows
the locked Golden Kernel ladder: per op, one designated production kernel
(picked by priority + supports() at runtime) acts simultaneously as the
correctness oracle and the perf denominator. New PERF_ALL columns
`golden_runner` / `golden_priority` make the choice visible per row;
audit statuses (`golden_unavailable_pending_baseline`,
`mla_golden_degraded=true`) keep gaps observable without relaxing G7's
thresholds. Specification: [`docs/benchmark/benchmark-protocol.md`](../benchmark/benchmark-protocol.md) and [`docs/benchmark/golden-kernel-ladder.md`](../benchmark/golden-kernel-ladder.md).

---

### Stage 8 (G8): Agent Autonomy ✅

**Objective:** Build a **highly extensible Arke Harness system** for LLM-driven auto-generation and auto-tuning of AI operators. The Harness (Façade + Substrate) is the **primary Stage 8 deliverable**. Multi-model BL6 end-to-end results (GPT-2 / LLaMA-2 / DS-V2) serve as **Thesis L1 endpoint validation**, demonstrating that the Harness produces real wins on real LLMs.

**Why this follows S7:** Agent needs the v0.1.0 IR/Lang (from S7) to generate backend-agnostic strategies. torch.compile integration needs Backend abstraction (from S6) and Pass pipeline (from S6). Multi-model E2E needs full operator coverage and MLIR skeleton (from S7).

**BL Exit:** BL5 inherited (no regression) + BL6×L3 (GPT-2 + LLaMA-2 + DeepSeek-V2).

**Stage 8 Core Mission (locked 2026-05-17, Leon-approved):**
Stage 8 builds a **highly extensible Arke Harness system** capable of LLM-driven auto-generation and auto-tuning of AI operators. The Harness itself (Façade + Substrate) is the **primary deliverable**; multi-model BL6 end-to-end results are **Thesis L1 endpoint validation**, not the Stage 8 product. G8 PASS therefore splits into two tiers:

**Gate G8 PASS Criteria (Tier 1 + Tier 2, both must hold):**

```
Tier 1 — Harness system (Stage 8 primary deliverable):
  [HARNESS-1] Façade interfaces locked: 8 tools schema + OptimizationEvent stream
              + trajectory schema v1.0 frozen (≤1 breaking change budget for the tier).
  [HARNESS-2] LLM autonomy:  G7-AE.1~AE.5 reproducibly pass — Agent independently
              completes op-generation + autotune trajectory end-to-end.
  [HARNESS-3] Extensibility (mid-tier, Leon-approved 2026-05-17, Q3=b;
              LOC cap raised 300→400 on 2026-05-22 to absorb D8-X1 rename
              refactor + true-fused swiglu_packed in a single demo — see
              D8-X1 entry below):
              - Onboard 1 *new* operator end-to-end: ≤400 LOC (incl. tests)
                + 1 SKILL.md + 1 audit entry + registered in op_registry.py,
                runs through BL1.
              - Onboard 1 *new* baseline runner: ≤200 LOC + documented
                BaselineRunner subclass protocol + plugged into benchmarks/baselines/.
              - Both demos shipped under benchmarks/results/phase1/stage8/extensibility/.

              D8-X1 (Demo A, 2026-05-22 Leon-approved Aa1/Bb1/Cc2;
                     ✅ completed 2026-06-06):
                Discovered legacy "swiglu" / "geglu" benchmark impls were
                misnomers — they actually compute silu(x)*y and gelu(x)*y
                (no input split, no down_proj). Demo A path:
                  1. Hard-rename legacy `swiglu` benchmark op → `silu_and_mul`,
                     and legacy `geglu` benchmark op → `gelu_and_mul` across
                     op_registry + downstream (no aliases kept).
                  2. Register new OT3 op `swiglu_packed`: true fused
                     split → silu*mul → matmul(down_proj).
                  3. Single-commit-chain demo; total LOC budget 400 (above
                     original 300 — justified by rename-refactor surface).

                Outcome:
                  - Commit chain on `feat/op-count-ssot`:
                      3917493 C1 rename swiglu → silu_and_mul
                      911bd9d C2 rename geglu  → gelu_and_mul
                      4c4b118 C3 onboard swiglu_packed (46th catalog op)
                      8cea142 C4 SKILL.md (skills/swiglu-packed-fusion/)
                      b2e8bd1 C5 BL1 evidence (correctness + perf rows)
                  - Catalog: 45 → 46 ops, OT3: 7 → 8 ops, audit-degraded
                    rows: 1 → 2 (joined dequantize_per_channel).
                  - 5/5 acceptance items pass (stage8-plan.md §"Tier 1
                    Extensibility Acceptance" Demo A).

Tier 2 — Thesis L1 endpoint validation (Harness produces real wins):
  [1] Auto strategy: kernel-only .ak → LLM generates strategy → codegen → ≥0.95× P0 (cuBLAS)
  [2] Iterative optimization: ≥3 compile→profile→adjust cycles in trajectory
  [3] Multi-input: .ak file + natural language + code snippet → all work E2E
  [4a] Vanilla torch.compile baseline: GPT-2 correctness 100% + perf ≥0.95× eager
       (perf evaluated as **geomean-over-seq** of compile/eager ratios across the
        measured seq-len set — Leon-approved口径 2026-06-25, D3=geomean. Rationale:
        per-seq min over-penalizes a single hardware-bound shape; on a 6GB SM8.6
        laptop GPT-2 seq=256 is launch/guard-overhead-bound where vanilla
        torch.compile genuinely loses to eager, while 128≈break-even and 512 wins.
        Geomean reflects whole-model throughput honestly without relaxing the
        0.95× bar or excluding any seq. The seq=256 row remains a recorded
        known-fail, not deleted. Benchmark measurement layer unchanged.)
  [4b] Arke→torch.compile bridge active: ≥1 Arke kernel on GPT-2's critical path,
        correctness 100% + perf ≥0.95× eager + bridge-invocation-evidence.
        ⚠ Bridge is a *transient Substrate artifact* scoped to ≤3 ops, inference-only,
        single-file under arke/integration/torch_bridge.py. It is NOT part of the
        Façade and is not a permanent product capability — see stage8-plan.md
        "D7-E1.4 scope guardrails".
  [5] LLaMA-2 7B: correctness 100% + perf ≥0.90× eager
  [6] DeepSeek-V2 16B: correctness 100% + perf ≥0.85× eager (seq≤512, quantized;
       gated on D7-E3.0 reachability probe)
  [7] BL5 no regression: L1+L2 correctness and performance ≥ G7 results
```

> **Tiering rationale:** Tier 1 measures whether Stage 8 *built the right thing*
> (a usable, extensible LLM-driven optimization Harness). Tier 2 measures whether
> what was built *actually delivers* on the L1 thesis (end-to-end perf wins on
> real LLMs). Tier 1 cannot be degraded; Tier 2 model selection (e.g. DS-V2)
> may be relaxed per D7-E3.0 outcome with Leon approval.

→ Detailed plan: [docs/phase1/stage8-plan.md](../phase1/stage8-plan.md)

**Stage 8 MVP status (current):** the initial G8 bootstrap path is implemented and validated by `python -m benchmarks.gate G8`. It is intentionally an MVP subset of the locked full G8 criteria: `arke optimize <file.ak>` now generates bounded backend-agnostic StrategyIR with a deterministic heuristic, records three compile→profile→adjust cycles in `trajectory.jsonl`, writes `strategy.json` / `result.akir` / `summary.json`, and `benchmarks.bench_l3` now emits the GPT-2 eager vs `torch.compile` artifact contract with a CPU-safe `--mock` path. Full G8 remains open until the live LLM strategy path, multi-input routing, BL5 performance inheritance, LLaMA-2, and DeepSeek-V2 criteria above are satisfied.

**Stage 8 entry-scout findings (2026-05-17):** Gate G8 contract validation tier 1/2 → 4/4 PASS (MVP.1 trajectory contract, MVP.2 multi-input cases, MVP.3 L3 artifact contract, MVP.4 pytest gate). First real GPT-2 measurement on the laptop RTX 3060 (seq=128, runs=5, warmup=3) showed **eager 7.26 ms / torch.compile 8.95 ms = ratio 0.811× ❌** against the locked G8[4] target ≥0.95×. **D7-E1.1 diagnostic (2026-05-17, runs=20, warmup=10) overturned this reading**: seq=128 ratio 1.006 ✅, seq=512 ratio 1.024 ✅, seq=256 ratio 0.865 ❌. The original 0.811× was a measurement artifact (warmup=3 below compile-overhead noise floor), not a real regression. The single remaining gap (seq=256) is **dynamic-shape recompile thrash** hitting `torch._dynamo.config.cache_size_limit=8` when multiple seq_lens are exercised in the same process, evicting GPT-2 layer-0..11 cache entries. The 12 reported `torch._dynamo.explain()` graph breaks are all intra-layer `_collections_abc.Mapping.__contains__` skipfile breaks and are NOT the dominant cost. Stage-8 M1 critical path re-routed to **D7-E1.6** (bump `cache_size_limit`, compile with `dynamic=True`) instead of the originally planned graph-break elim + CUDA Graph chain. Two cheap-fix bugs also closed in the same commit pack: (a) `bench_l1 --no-resume` was silently appending to per-op CSVs instead of truncating (commit `7b74abc`); (b) `bench_l3 build_summary` filtered on `r.mode == "torch.compile"` while CLI `--modes eager,torch_compile` wrote rows with `mode="torch_compile"`, producing `compile_rows=0` in summary.json despite a successful compile run (commit `b33d9ce`). Full diagnosis artifacts under `benchmarks/results/phase1/stage8/track4/diagnose_2026-05-16/`. **D7-E1.6 closure (same day):** patched `bench_l3.py` to set `torch._dynamo.config.cache_size_limit=64` at module load + bump `DEFAULT_WARMUP_RUNS=10` + compile with `dynamic=True`, then reran `python -m benchmarks.bench_l3 --seq-len 128,256,512 --model gpt2 --modes eager,torch.compile --warmup 10 --runs 20`. All three seq_lens passed: **seq=128 → 1.024, seq=256 → 1.070** (from 0.865, **+23.7%**), **seq=512 → 1.072**, `summary.json: g8_gpt2_pass=true, min_compile_ratio=1.024, geomean=1.055`. Artifact at `benchmarks/results/phase1/stage8/track4/l3/2026-05-17_112120/`. G8[4] gate criterion **met**; M1/M4 milestones closed.

---

### Stage 9 (G9): Phase 1 Final ✅

**Objective:** Final acceptance across 4 models. Automated Arke-vs-LLM-direct comparison. Spec freeze. Evaluation report. v1.0 release tag.

**Why this is separate from S8:** S8 validates capability (can it work?). S9 validates maturity (is it reliable, competitive, documented?). Spec freeze requires all features to be stable. 4-model validation requires LLaMA-3 and Qwen2.5 in addition to S8's models.

**BL Exit:** BL6×L3 (4 models) + BL5 regression (no regression).

**Gate G9 PASS Criteria:**

```
AND ALL:
  [1] 4 models E2E correctness 100%: GPT-2 ≥1.00× eager, LLaMA-2 ≥0.95×, LLaMA-3 ≥0.95×, Qwen2.5 ≥0.90×
  [2] Arke vs LLM-direct: correctness 100%, tokens ≤ 0.70×, perf ≥ 1.05× P5
  [3] @rationale KB: ≥50 Phase 1 entries
  [4] Spec freeze: Lang v1.0 + IR v1.0 tagged
  [5] Phase 1 evaluation report published
  [6] v1.0.0 tag
```

→ Detailed plan: [docs/phase1/stage9-plan.md](../phase1/stage9-plan.md)

---

### Phase Dependency Chain

```
S0-S5 ✅ → S6 (Compiler Infra) → S7 (Lang & IR v0.1.0) → S8 (Agent Autonomy) → S9 (Final)
```

---

## Phase 2: Arke → Triton-Ascend/MLIR → Ascend NPU (SIMD Validation)

> ## ⏭️ SKIPPED (2026-07-02, Leon-approved)
>
> **Phase 2 (Ascend / SIMD validation) is SKIPPED in the active roadmap.**
> Decision (Leon, 2026-07-02): jump directly from Phase 1 (CLOSED) into
> **Phase 3 (Arke → MLIR Dialect)** and drive full development + test
> validation there. The entire Phase-2 section below is kept **dormant, not
> deleted** — it remains the design-of-record for if/when cross-architecture
> (Thesis L2) validation on Ascend is resumed. Gate numbering is **unchanged**
> (no renumber) per Leon decision 1a.
>
> **Consequence for Thesis L2:** cross-architecture IR-generality is now tested
> *inside Phase 3* via MLIR multi-target lowering (NVIDIA PTX first), not via
> the Ascend-Triton path. The Ascend-specific stage **P3-S4 (Ascend via MLIR)
> is likewise SKIPPED** — see the ⏭️ marker on P3-S4 in the Phase 3 table below.
>
> **Backend extensibility is still preserved** — `arke/backend/protocol.py`
> (`ArkeBackend` Protocol + `BackendRegistry`) remains the clean extension seam;
> a future Ascend backend plugs in without a core refactor.
>
> ---
>
> _Prior status (superseded):_ ⏸️ PAUSED (2026-06-24, Leon-approved)
>
> **Ascend / Phase-2 development is paused.** All current effort goes to the
> NVIDIA/Triton path (Phase 1) to validate the LLM-Native thesis at L1 (SIMT).
> This entire Phase-2 section is kept **dormant, not deleted** — it remains the
> design-of-record for when cross-architecture validation (Thesis L2) resumes.
>
> **Backend extensibility is preserved**, not dropped: the extension seam lives
> in `arke/backend/protocol.py` (`ArkeBackend` Protocol + `BackendRegistry`).
> A future Ascend backend plugs in by implementing the 4-method Protocol and
> registering target strings — no core refactor required. See that module's
> docstring and `docs/architecture/arke-compiler-infrastructure.md` §7.
>
> **Do not invest implementation effort here until Leon un-pauses Phase 2.**

**Goal:** Verify Arke Lang/IR works on SIMD architecture (Ascend NPU) via Ascend Triton backend. Arke-generated Ascend Triton kernels must outperform FlagGems on Ascend. Simultaneously complete Arke Lang/IR to cover Category B-E operators.

**Hardware target:** Huawei Ascend 910B (SIMD, CANN)
**Backend:** triton-ascend (Ascend Triton)
**Benchmark baseline:** FlagGems (Ascend port)

### Stage Structure


| Stage          | Milestone                 | Exit Criteria                                |
| -------------- | ------------------------- | -------------------------------------------- |
| **P2-S1**      | Ascend Triton environment | matmul runs on 910B, ≥70% CANN cuBLAS        |
| **P2-S2**      | Cat A+B ops on Ascend     | 20 ops correct + ≥0.85× FlagGems             |
| **P2-S3**      | Cat C+D ops on Ascend     | 15 ops correct + ≥0.80× FlagGems             |
| **P2-S4**      | LLaMA-2 on Ascend         | E2E correct + ≤1.40× eager                   |
| **P2-S_FINAL** | Phase 2 acceptance        | H4 validated: same Arke IR → NVIDIA + Ascend |


### Key Insights from phase2-3-review.md

**What Phase 1 taught us:**

- Triton dispatch overhead (~60µs) is real but not Arke's fault
- Small-M performance gap is architectural, not IR design issue
- E2E latency requires torch.compile backend (graph fusion + zero-overhead dispatch)

**Phase 2 adjusted goals:**

- Primary: Validate H4 (cross-architecture generalization) — same SemanticIR → NVIDIA Triton + Ascend Triton
- Secondary: Expand operator coverage (Cat B-E)
- Defer: Full MLIR integration to Phase 3 (Ascend Triton is sufficient for H4 validation)

**Phase 2 does NOT need:**

- Custom MLIR dialect (Ascend Triton backend is enough)
- Performance parity with Phase 1 NVIDIA (SIMD vs SIMT architectural difference expected)
- Full E2E latency optimization (torch.compile backend deferred to Phase 3)

---

## Phase 3: Arke → MLIR Dialect (Full Compiler Control)

> ## 🚧 ACTIVE (2026-07-02, Leon-approved) — Phase 1 → Phase 3 direct jump
>
> Phase 2 (Ascend) is SKIPPED (see marker above). Phase 3 is now the **active
> development phase**. Cross-architecture IR-generality (Thesis L2) is validated
> here through MLIR multi-target lowering on the NVIDIA path first.
>
> **Toolchain (locked 2026-07-02, Leon decision 2+3):** MLIR 18.1.3 installed
> user-local (no-root, `dpkg-deb -x` of Ubuntu `mlir-18-tools` + `libmlir-18`
> into `~/opt/mlir18`; source `~/opt/mlir18/env.sh`). Path C (CLI:
> `mlir-opt` / `mlir-translate` / `mlir-cpu-runner`) is the primary lowering +
> JIT-execution driver; a future in-process bindings path (A) requires a
> from-source LLVM build and is deferred. Correctness is validated **CPU first**
> (mlir-cpu-runner JIT, bit-correct vs torch) **then GPU** (`gpu` dialect → PTX).
> Verified 2026-07-02: `linalg.matmul` lowers end-to-end and JIT-executes
> correct numerics on CPU (2×3 @ 3×2 = [[6,6],[6,6]]).
>
> **P3-S1 (MLIR lowering framework) is ✅ COMPLETE (2026-07-03):**
> - CPU: `SemanticIR → linalg → mlir-cpu-runner JIT`, matmul bit-correct vs numpy (7 shapes).
> - **transform dialect:** `transform.structured.tile_using_for` tiling (full/partial/non-dividing tiles), bit-correct; the P3-S5 StrategyIR-L2 seam.
> - **GPU:** `SemanticIR → gpu dialect → NVVM → PTX → CUDA driver launch` (cuda-python), matmul bit-correct on RTX 3060 (sm_86) vs numpy + torch.cuda (6 shapes). Ubuntu mlir-18-tools ships NVPTX; PTX launched via driver API (no libmlir_cuda_runtime needed).
> - **GPU elementwise + transcendental (2026-07-04):** GPU path extended from matmul-only to 10 elementwise ops. Pure-arith (relu/neg/add/mul) lower directly; **transcendentals (exp/tanh/sigmoid/silu/gelu/rsqrt) lower via libdevice** — `gpu-module-to-binary` is passed `l=<libdevice.10.bc>`, which inlines `math.*`/`__nv_*` calls to native PTX (exp → `ex2.approx`), so the driver-only load succeeds (no `CUDA_ERROR_INVALID_PTX`). All bit-correct vs torch on RTX 3060 (max_err ≤1e-6). This is the correct libdevice-linking path, chosen over restricting the GPU set to a pure-arith subset. 33 new tests: `tests/backend/test_mlir_gpu_elementwise_p3s2.py`.
> - 25 P3-S1 tests (14 CPU + tiling wired into 14, 11 GPU) + 33 GPU elementwise. Commits 56d1f84, 313da7d, f1be537, + libdevice commit.
>
> **P3-S4 (Ascend via MLIR) is ⏭️ SKIPPED** — consistent with the Phase-2 skip.
>
> **P3-S2 correctness half ✅ (2026-07-03):** 35 ops correct via MLIR
> (OT0 12/12, OT1 9/10, OT2 11/11 — the "Cat A+B+C" scope) on the CPU
> mlir-runner JIT path, all bit-correct vs torch (max_err ≤5e-6). Built on a
> declarative op catalog (`arke/backend/mlir_ops.py`: elementwise via
> `linalg.generic`, reductions/norms/movement/gated/index ops via
> multi-linalg + scf composites). Remaining P3-S2 work = perf half
> (geomean ≥ Phase 2 Triton). Toolchain now **MLIR 20.1.2** (aligns Triton 3.2
> / PyTorch 2.6). Tests: `tests/backend/test_mlir_ops_p3s2.py`.

**Goal:** Remove Triton's abstraction ceiling. Arke IR lowers to standard MLIR dialects (linalg, transform, scf, gpu), enabling deeper hardware control and more complete operator support. Performance must match or exceed Phase 2 Triton path.

**Backend:** MLIR standard dialects → LLVM IR → PTX/AMDGPU/CANN
**Benchmark baseline:** Phase 2 Triton performance

### Stage Structure


| Stage          | Milestone                 | Exit Criteria                                                           |
| -------------- | ------------------------- | ----------------------------------------------------------------------- |
| **P3-S1**      | MLIR lowering framework   | SemanticIR → linalg + transform dialect, matmul correct                 |
| **P3-S2**      | Cat A+B+C via MLIR        | 35 ops correct + geomean ≥ Phase 2 Triton                               |
| **P3-S3**      | MLIR performance ≥ Triton | All Cat A+B+C+D MLIR geomean ≥ Phase 2 Triton                           |
| **P3-S4**      | ~~Ascend via MLIR~~ ⏭️ SKIPPED | ~~matmul+rmsnorm correct on Ascend via MLIR~~ — skipped (Phase-2 Ascend skip, 2026-07-02) |
| **P3-S5**      | LLM Level 2 decisions     | StrategyIR L2 (loop nests) → MLIR transform dialect, verified on ≥3 ops |
| **P3-S_FINAL** | Phase 3 acceptance        | MLIR path performance ≥ Triton + multi-hardware via MLIR                |


### Key Design Points

- **No custom Arke MLIR dialect**: Use standard linalg/transform/scf/gpu dialects
- **StrategyIR L2 → transform dialect**: Loop nest decisions map to MLIR transform ops
- **Multi-hardware via MLIR**: NVIDIA (PTX), AMD (AMDGPU), Ascend (CANN) all via MLIR lowering
- **torch.compile backend**: Integrate Arke as Inductor backend, eliminating Python dispatch overhead

---

## Phase 4: Arke → C-like Kernel Language (CUDA C / CCE-C / Bang-C / etc.)

**Goal:** Generate vendor-supplied C-like kernel languages directly from Arke IR. This phase fills the gap between the MLIR abstraction layer (Phase 3) and the bare LLVM IR layer (Phase 5): vendor C-like DSLs (CUDA C for NVIDIA, CCE-C for Ascend, Bang-C for Cambricon, etc.) are the productionized, vendor-stable kernel surface, with the largest body of hand-tuned reference kernels and the most direct access to vendor toolchains, intrinsics, and tuning practices.

**Backend:** Arke IR → C-like kernel source (vendor DSL) → vendor compiler (nvcc / ccec / cncc) → executable
**Benchmark baseline:** vendor hand-tuned C-like kernels for each operator (e.g. CUTLASS CUDA C kernels, vendor sample CCE-C / Bang-C kernels) — denominator under the Same-Backend Fairness rule

### Why a separate Phase between MLIR and LLVM IR

- **MLIR (Phase 3)** gives compiler-level control but the dialect ecosystem and lowering paths are still maturing; vendor kernel ecosystems live primarily in C-like DSLs, not in MLIR yet
- **LLVM IR (Phase 5)** is the lowest level, but writing LLVM IR directly bypasses vendor-optimization expertise encoded in their C-like SDKs
- **C-like vendor DSLs** are where the vast majority of production kernel engineering happens (CUTLASS, Cutlass3, vendor samples, FlagAttention, FlashInfer source), and where Arke can leverage the most existing tuned reference material

### Stage Structure

| Stage | Milestone | Exit Criteria |
| --- | --- | --- |
| **P4-S1** | CUDA-C lowering framework | SemanticIR + StrategyIR → CUDA C source for matmul; correctness verified |
| **P4-S2** | Cat A+B+C via CUDA-C | 30 ops correct + geomean ≥ Phase 3 MLIR (Same-Backend CUDA-C fairness) |
| **P4-S3** | CCE-C / Bang-C cross-vendor | matmul + rmsnorm + flash_attention correct on ≥1 non-NVIDIA vendor C-like DSL |
| **P4-S4** | Performance ≥ MLIR | C-like geomean ≥ MLIR geomean across Cat A+B+C (per-vendor) |
| **P4-S_FINAL** | Phase 4 acceptance | Multi-vendor C-like DSL coverage validated; H5 (vendor-DSL portability via Arke IR) demonstrated |

### Key Design Points

- **Backend abstraction extension:** add `CLikeBackend` siblings to `TritonBackend` / `MLIRBackend`; pluggable per-vendor codegen
- **Same-Backend Fairness:** Arke-CUDA-C vs hand-tuned CUDA-C reference; Arke-CCE-C vs CCE-C reference; etc. — cross-vendor comparisons audit-only
- **@rationale grounding:** vendor-DSL optimization patterns (CUTLASS-style tiling, vendor-specific memory hints) feed back into @rationale KB
- **Phase ordering rationale:** the high-level Arke IR (Layer 4 + Layer 3) from Phase 1 → Phase 2 stays unchanged; Phase 3 (MLIR) consumes it; Phase 4 (C-like) consumes the same IR through a parallel codegen path; Phase 5 (LLVM IR) provides the lowest-level escape hatch

---

## Phase 5: Arke → LLVM IR (100% Hardware Completeness)

**Goal:** Achieve maximum hardware expression completeness and performance headroom. Arke IR lowers directly to LLVM IR, bypassing all high-level abstractions. Support 100% of hardware ISA features. This is the final phase of the multi-backend roadmap.

**Backend:** LLVM IR → PTX/AMDGPU/CANN/ROCm
**Benchmark baseline:** Phase 4 C-like kernel performance (per-vendor) under Same-Backend Fairness for the LLVM-IR target.

### Stage Structure

| Stage | Milestone | Exit Criteria |
| --- | --- | --- |
| **P5-S1** | LLVM lowering framework | SemanticIR → LLVM IR, matmul correct |
| **P5-S2** | Cat A-F via LLVM | All 60+ ops correct + geomean ≥ Phase 4 C-like (per-vendor) |
| **P5-S3** | LLVM performance ≥ C-like | LLVM geomean ≥ C-like + 5% (Cat A+C+D) |
| **P5-S4** | Multi-hardware LLVM | ≥3 backends ≥90% respective vendor libs |
| **P5-S5** | LLM Level 3 decisions | StrategyIR L3 (instruction-level) → LLVM IR, verified benefit ≥5% |
| **P5-S_FINAL** | v1.0.0 release | @rationale KB ≥200 entries, cross-hardware coverage |

### Key Design Points

- **StrategyIR L3 → LLVM IR:** Instruction-level decisions (e.g., warp shuffle, tensor core intrinsics) map directly to LLVM intrinsics
- **100% ISA coverage:** No abstraction ceiling — full access to PTX/AMDGPU/CANN instruction sets
- **LLM Level 1-3 full stack:** LLM makes decisions at all three StrategyIR layers
- **@rationale knowledge base:** ≥200 cross-hardware optimization patterns

---

## Long-term TODOs

> Items tracked for future optimization beyond Phase 1 Gate requirements.

| ID | Category | Description | Target Phase |
|:---|:---------|:-----------|:-------------|
| LT-1 | Performance | Model integration overhead optimization — reduce torch.compile dispatch overhead toward zero | P2+ |
| LT-2 | Performance | Graph-level fusion — cross-op fusion beyond L2 pairwise (e.g., full attention block fusion) | P2+ |
| LT-3 | Performance | Memory optimization — activation checkpointing, KV cache compression for long-context | P2+ |
| LT-4 | Toolchain | Profiling integration — automated bottleneck identification with roofline analysis | P2 |
| LT-5 | Toolchain | CI/CD performance regression — automated nightly BL5 benchmark with alerting | P1-S9 |
| LT-6 | Architecture | Multi-GPU support — tensor/pipeline parallelism for models > 6GB VRAM | P3+ |
| LT-7 | Agent | @rationale knowledge base distillation — cross-hardware pattern extraction | P2+ |

---

## Risk Matrix


| Risk                                        | Affects       | Mitigation                                                        |
| ------------------------------------------- | ------------- | ----------------------------------------------------------------- |
| LLM decisions don't map to Triton templates | Phase 1 S7-S8 | Expand template coverage + parameter adaptation layer             |
| compile_and_profile errors in LLM session   | Phase 1 S8    | Better error messages + graceful degradation                      |
| 6GB VRAM insufficient for large shapes      | Phase 1 S5-S9 | Limit shapes to ≤2048; mark OOM as non-blocking                   |
| Arke doesn't outperform direct Triton       | Phase 1 S9    | Gate decision matrix: reliability + token efficiency win          |
| API timeout / rate limit                    | Phase 1 S7-S9 | Retry + fallback + prefer Sonnet over Opus                        |
| Ascend Triton backend unavailable           | Phase 2       | Fallback: validate H4 on AMD via ROCm Triton                      |
| MLIR learning curve too steep               | Phase 3       | Hire MLIR expert consultant; allocate 2× time buffer              |
| C-like vendor SDK lock-in / availability    | Phase 4       | Start with CUDA C (most stable SDK); add CCE-C / Bang-C incrementally |
| LLVM IR complexity explosion                | Phase 5       | Incremental: start with matmul only, expand gradually             |
| torch.compile integration breaks            | Phase 1 S8    | Maintain standalone CLI as fallback; Inductor backend is optional |


---

*Last updated: 2026-06-26 (Phase 1 CLOSED 2026-06-25 on dev-HW: S0–S6 ✅, S7 ✅ 13/14 closed + followups landed, S8 ✅ gate 6/6 PASS, S9 ✅ CLOSED — Thesis L1 SIMT validated. G9[4]/[6] release tags waived by project lead. Phase 2/Ascend remains PAUSED — backend extensibility preserved via `arke/backend/protocol.py`. Footer status synced to plan status table L100–110.)*