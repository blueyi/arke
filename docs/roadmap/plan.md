# Arke Development Plan

> Single source of truth for development roadmap.
> Detailed stage plans → docs/phase1/stage{N}-plan.md
> Architecture → docs/architecture/; Specs → docs/spec/

## Hierarchy

Roadmap > Phase > Stage > Feature > Task

## Gate Governance

> **Gates are the contract between design and development.**
>
> Gate exit criteria define verifiable acceptance standards for each Stage. Once a Phase/Stage's Gate criteria are finalized, they are **locked** — any adjustment requires explicit approval from the project lead.
>
> Development should be **Gate-driven**: work backward from Gate exit criteria to determine what Arke Lang, IR, Compiler, and Agent need to deliver. Gates drive design and implementation, not the other way around.
>
> All Gate criteria that involve operator-level performance or correctness **must** align to the BL/OT/ST/L benchmark system defined in `docs/benchmark/benchmark-design.md`. The Gate-Purpose Mapping below specifies which benchmark levels each Stage's Gate must satisfy.

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
| S6 | G6 | BL4×L1 (45 ops correctness + ≥1.00× P3) | L1 | Compiler Infrastructure | ✅ 7/7 |
| S7 | G7 | BL5×L1+L2 | L1+L2 | Lang & IR v2 | ⬜ |
| S8 | G8 | BL5 inherit + BL6×L3 (GPT-2+LLaMA-2+DS-V2) | L1+L2+L3 | Agent Autonomy | ⬜ |
| S9 | G9 | BL6×L3 (4 models) + BL5 regression | L1+L2+L3 | Phase 1 Final | ⬜ |

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
| **G6** | **BL4×L1** | **L1** | 45 ops correctness 100% via SemanticInterpreter; perf ≥1.00× P3 | P3 |
| **G7** | **BL5×L1+L2** | **L1+L2** | OT0-4×ST1-4 correctness 100%; OT0 ≥1.05 P1, matmul ≥1.00 P0; 4/4 fusions | P0, P1 |
| **G8** | **BL5 inherit + BL6×L3** | **L1+L2+L3** | GPT-2 ≥0.95× eager; LLaMA-2 ≥0.90×; DS-V2 ≥0.85×; auto-strategy ≥0.95× P0 | P0, P1, P3 |
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

### Stage 6 (G6): Compiler Infrastructure ⬜ ← CURRENT

**Objective:** Refactor the compiler toolchain into a clean, extensible architecture. OpRegistry as single source of truth, Pass pipeline for composable transformations, Backend abstraction for multi-target support.

**Why this comes first:** All subsequent stages (IR v2, Agent autonomy, multi-model E2E) depend on a solid compiler foundation. Without OpRegistry, adding ops requires touching 6 files. Without Pass infrastructure, IR transformations are ad-hoc. Without Backend abstraction, Triton is hardcoded everywhere.

**BL Exit:** BL4×L1 — Full 45 ops correctness 100% via SemanticInterpreter + performance ≥1.00× P3 (eager).

**Gate G6 PASS Criteria:**

```
AND ALL:
  [1] OpRegistry: single source of truth for all 45 ops (adding op ≤ 2 files)
  [2] SemanticInterpreter: PyTorch eager executor, 45 ops correctness 100%
  [3] Pass Infrastructure: ArkePass protocol + PassPipeline with ≥2 passes
  [4] SSA Validator: validates all 45 ops; rejects ≥5 invalid IR examples
  [5] Backend Abstraction: ArkeBackend protocol + TritonBackend implements it
  [6] Codegen + GPU execution: 45 ops via TritonBackend, correctness 100%, perf ≥1.00× P3
  [7] Non-regression: ≥422 tests passed, ≤6 skipped, 0 new failures
```

→ Detailed plan: [docs/phase1/stage6-plan.md](../phase1/stage6-plan.md)

---

### Stage 7 (G7): Lang & IR v2 ⬜

**Objective:** Implement the multi-layer IR architecture (Layer 4/3/2/1), upgrade Arke Lang with where clause and backend-agnostic strategy, complete spec documents, assess dynamic shape feasibility, establish MLIR framework skeleton.

**Why this follows S6:** Pass pipeline (from S6) is needed for IR layer transformations. Backend abstraction (from S6) is needed for backend-agnostic strategy validation. OpRegistry (from S6) is needed for spec completeness verification.

**BL Exit:** BL5×L1+L2 — All 45 ops (OT0-4) × all shapes (ST1-4) correctness + performance at L1 single-op and L2 fused-op levels.

**Gate G7 PASS Criteria:**

```
AND ALL:
  [1] Arke Lang Spec v2.0 document finalized
  [2] Arke IR Spec v2.0 document finalized (Layer 4/3/2/1 defined)
  [3] where clause MVP: parses + SemanticIR symbolic_dims populated
  [4] Dynamic Shape feasibility assessment document complete
  [5] MLIR framework skeleton: MLIREmitter exists, BL1 matmul verified
  [6] All 45 ops: .ak → SemanticIR → StrategyIR full round-trip
  [7] Token efficiency: .ak lines < Triton lines for all OT0-OT4
  [8] Backend-agnostic strategy: 0 Triton-specific fields in StrategyIR core
  [9] L1 BL5 correctness: 100%(ST1-4, excl. OOM) for all OT0-OT4
  [10] L1 BL5 performance: OT0 ≥1.05 P1, OT1 ≥0.95 P1, OT2 matmul ≥1.00 P0, OT3 ≥0.95 P1, OT4 ≥0.90 P1
  [11] L2 BL5: 4/4 fusion combinations pass
  [12] Non-regression: ≥422 tests, 0 new failures
```

→ Detailed plan: [docs/phase1/stage7-plan.md](../phase1/stage7-plan.md)

---

### Stage 8 (G8): Agent Autonomy ⬜

**Objective:** Validate that the Arke Agent can autonomously generate strategies, iterate optimization, and produce correct kernels for real LLMs. Integrate torch.compile backend to eliminate dispatch overhead. Validate on LLaMA-2 7B and DeepSeek-V2 16B.

**Why this follows S7:** Agent needs the v2 IR/Lang (from S7) to generate backend-agnostic strategies. torch.compile integration needs Backend abstraction (from S6) and Pass pipeline (from S6). Multi-model E2E needs full operator coverage and MLIR skeleton (from S7).

**BL Exit:** BL5 inherited (no regression) + BL6×L3 (GPT-2 + LLaMA-2 + DeepSeek-V2).

**Gate G8 PASS Criteria:**

```
AND ALL:
  [1] Auto strategy: kernel-only .ak → LLM generates strategy → codegen → ≥0.95× P0 (cuBLAS)
  [2] Iterative optimization: ≥3 compile→profile→adjust cycles in trajectory
  [3] Multi-input: .ak file + natural language + code snippet → all work E2E
  [4] torch.compile backend: GPT-2 correctness 100% + perf ≥0.95× eager (fixes S5 known-fail)
  [5] LLaMA-2 7B: correctness 100% + perf ≥0.90× eager
  [6] DeepSeek-V2 16B: correctness 100% + perf ≥0.85× eager (seq≤512, quantized)
  [7] BL5 no regression: L1+L2 correctness and performance ≥ G7 results
```

→ Detailed plan: [docs/phase1/stage8-plan.md](../phase1/stage8-plan.md)

---

### Stage 9 (G9): Phase 1 Final ⬜

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
S0-S5 ✅ → S6 (Compiler Infra) → S7 (Lang & IR v2) → S8 (Agent Autonomy) → S9 (Final)
```

---

## Phase 2: Arke → Triton-Ascend/MLIR → Ascend NPU (SIMD Validation)

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

**Goal:** Remove Triton's abstraction ceiling. Arke IR lowers to standard MLIR dialects (linalg, transform, scf, gpu), enabling deeper hardware control and more complete operator support. Performance must match or exceed Phase 2 Triton path.

**Backend:** MLIR standard dialects → LLVM IR → PTX/AMDGPU/CANN
**Benchmark baseline:** Phase 2 Triton performance

### Stage Structure


| Stage          | Milestone                 | Exit Criteria                                                           |
| -------------- | ------------------------- | ----------------------------------------------------------------------- |
| **P3-S1**      | MLIR lowering framework   | SemanticIR → linalg + transform dialect, matmul correct                 |
| **P3-S2**      | Cat A+B+C via MLIR        | 35 ops correct + geomean ≥ Phase 2 Triton                               |
| **P3-S3**      | MLIR performance ≥ Triton | All Cat A+B+C+D MLIR geomean ≥ Phase 2 Triton                           |
| **P3-S4**      | Ascend via MLIR           | matmul+rmsnorm correct on Ascend via MLIR; perf ≥ Phase 2               |
| **P3-S5**      | LLM Level 2 decisions     | StrategyIR L2 (loop nests) → MLIR transform dialect, verified on ≥3 ops |
| **P3-S_FINAL** | Phase 3 acceptance        | MLIR path performance ≥ Triton + multi-hardware via MLIR                |


### Key Design Points

- **No custom Arke MLIR dialect**: Use standard linalg/transform/scf/gpu dialects
- **StrategyIR L2 → transform dialect**: Loop nest decisions map to MLIR transform ops
- **Multi-hardware via MLIR**: NVIDIA (PTX), AMD (AMDGPU), Ascend (CANN) all via MLIR lowering
- **torch.compile backend**: Integrate Arke as Inductor backend, eliminating Python dispatch overhead

---

## Phase 4: Arke → LLVM IR (100% Hardware Completeness)

**Goal:** Achieve maximum hardware expression completeness and performance headroom. Arke IR lowers directly to LLVM IR, bypassing all high-level abstractions. Support 100% of hardware ISA features.

**Backend:** LLVM IR → PTX/AMDGPU/CANN/ROCm
**Benchmark baseline:** Phase 3 MLIR performance

### Stage Structure


| Stage          | Milestone               | Exit Criteria                                                     |
| -------------- | ----------------------- | ----------------------------------------------------------------- |
| **P4-S1**      | LLVM lowering framework | SemanticIR → LLVM IR, matmul correct                              |
| **P4-S2**      | Cat A-F via LLVM        | All 60+ ops correct + geomean ≥ Phase 3 MLIR                      |
| **P4-S3**      | LLVM performance ≥ MLIR | LLVM geomean ≥ MLIR + 5% (Cat A+C+D)                              |
| **P4-S4**      | Multi-hardware LLVM     | ≥3 backends ≥90% respective vendor libs                           |
| **P4-S5**      | LLM Level 3 decisions   | StrategyIR L3 (instruction-level) → LLVM IR, verified benefit ≥5% |
| **P4-S_FINAL** | v1.0.0 release          | @rationale KB ≥200 entries, cross-hardware coverage               |


### Key Design Points

- **StrategyIR L3 → LLVM IR**: Instruction-level decisions (e.g., warp shuffle, tensor core intrinsics) map directly to LLVM intrinsics
- **100% ISA coverage**: No abstraction ceiling — full access to PTX/AMDGPU/CANN instruction sets
- **LLM Level 1-3 full stack**: LLM makes decisions at all three StrategyIR layers
- **@rationale knowledge base**: ≥200 cross-hardware optimization patterns

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
| LLVM IR complexity explosion                | Phase 4       | Incremental: start with matmul only, expand gradually             |
| torch.compile integration breaks            | Phase 1 S8    | Maintain standalone CLI as fallback; Inductor backend is optional |


---

*Last updated: 2026-04-06 (Gate criteria finalized with high performance targets; S6 Compiler Infrastructure in progress)*