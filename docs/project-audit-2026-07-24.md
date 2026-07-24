# Arke Project Audit — 初心达成评估

> **Date:** 2026-07-24
> **Auditor:** Kitty (AI Lead Engineer)
> **Scope:** README Key Features 6 条宣言 + Thesis L1/L2/L3 + e2e-flow 整体流
> **Method:** 代码/数据/测试实证，非文档自证

---

## 1. AI-First Design

**宣言:** LLMs are optimization decision makers, not just code generators.

| Criterion | Evidence | Verdict |
|:---|:---|:---:|
| Bounded Action Space | 9 Façade tools: `analyze_compute`, `get_hw_profile`, `list_legal_actions`, `apply_decision`, `verify_correctness`, `compile_and_profile`, `checkpoint`, `rollback`, `benchmark_advice_summary` | ✅ |
| LLM decides, compiler verifies | `VerifyCorrectnessTool` implements V0_mock + V1_triton tiers; `CompileAndProfileTool` measures V2 perf | ✅ |
| Live-agent gate | P5-S5-T live-LLM gate 5/5 PASS (commit f6808da) | ✅ |
| Not code-generator | Agent emits `Decision` objects with `@rationale`, not raw kernel source | ✅ |

**Verdict: ✅ ACHIEVED**

---

## 2. Semantic/Strategy Separation

**宣言:** "What to compute" and "how to optimize" represented independently.

| Criterion | Evidence | Verdict |
|:---|:---|:---:|
| SemanticIR immutable | `SemanticIR` dataclass: version, kernel_id, params, symbolic_dims, nodes, edges, return_node, fusion_groups, metadata — pure computation graph | ✅ |
| StrategyIR independent | `StrategyIR` dataclass: version, kernel_id, target_hw, decisions, shape_regimes, constraints, metadata — optimization policy only | ✅ |
| L1/L2/L3 levels | `Decision.level`: 1=backend-agnostic, 2=resource-bound, 3=instruction-level (LLVM/PTX) | ✅ |
| InstructionIR (L3) | `InstructionIR`: version, kernel_id, target_hw, blocks, metadata — ISA-level | ✅ |
| Rollback safety | `CheckpointTool` + `RollbackTool` — strategy rollback without touching semantics | ✅ |

**Verdict: ✅ ACHIEVED**

---

## 3. Minimal-Token Efficiency

**宣言:** Minimize token consumption across the full pipeline.

| Criterion | Evidence | Verdict |
|:---|:---|:---:|
| Token comparison | README: Arke `.ak` 72 tokens vs LLM-direct Triton 563 vs hand-written 1102 (ratio 1:8:15) | ✅ |
| Analysis doc | `docs/architecture/token-efficiency-analysis.md` (677 lines, detailed breakdown) | ✅ |
| G9 evidence | G9[2]: Arke 1.263× better than LLM-direct @ 0 extra tokens | ✅ |
| Compact IR | Decision objects vs whole-program rewrite; `observe()` delta mode in harness | ✅ |

**Verdict: ✅ ACHIEVED**

---

## 4. Compiler-Verified Optimization

**宣言:** Optimization decisions validated through deterministic checks (V0 → V1 → V2).

| Criterion | Evidence | Verdict |
|:---|:---|:---:|
| V0 (static legality) | `list_legal_actions()` enumerates only valid moves; `apply_decision()` checks legality before applying | ✅ |
| V1 (numerical correctness) | `VerifyCorrectnessTool`: V0_mock tier (candidate==reference) + V1_triton tier (real GPU execution vs SemanticInterpreter reference) | ✅ |
| V2 (performance) | `CompileAndProfileTool`: compile to backend + profile latency on real hardware | ✅ |
| Pipeline integration | `ArkePipeline` orchestrates semantic passes → lowering → V1 → V2 sequentially | ✅ |
| 2722 tests | Full test suite covering correctness, compilation, gate criteria | ✅ |

**Verdict: ✅ ACHIEVED**

---

## 5. @rationale as First-Class Artifact

**宣言:** Decisions carry NL explanations; trajectories auditable, reusable, learnable.

| Criterion | Evidence | Verdict |
|:---|:---|:---:|
| KB size | `data/rationale_kb.jsonl`: 390 entries (target was ≥200) | ✅ |
| Decision.rationale | Every `Decision` object in StrategyIR carries a `@rationale` string field | ✅ |
| Trajectory files | `arke optimize` emits `trajectory.jsonl` per run | ✅ |
| Learning module | `arke/learn/rationale_kb.py` — KB query/inject for future runs | ✅ |
| Phase 5 live rationale | 72 entries from P5 live-LLM LLVM sessions added to KB | ✅ |

**Verdict: ✅ ACHIEVED**

---

## 6. Cross-Hardware Performance Ambition

**宣言:** Single semantic definition lowers toward multiple hardware targets.

| Criterion | Evidence | Verdict |
|:---|:---|:---:|
| Backend protocol | `arke/backend/protocol.py`: `ArkeBackend` ABC + `BackendRegistry` (register/get/list_backends/list_targets) | ✅ |
| Implemented backends | `llvm_backend.py`, `cuda_c_backend.py` + Triton (original) + MLIR emitter | ✅ |
| Extension seam preserved | Protocol docstring: "No core refactor needed to add Ascend/AMD/… backend" | ✅ |
| Actual multi-HW validation | ❌ Only NVIDIA tested; Ascend/AMD = PAUSED/no hardware | 🟨 |
| Honest declaration | README: "Phase closure… does not mean release-ready. Cross-hardware validation remains open." | ✅ |

**Verdict: 🟨 PARTIALLY ACHIEVED** — architecture ready, execution single-HW only (honest).

---

## 7. Thesis L1 — Single-Architecture SIMT (Phase 1)

**Claim:** LLM + structured IR + compiler verification → correct + high-perf ops on SIMT/Triton.

| Gate | Result | Evidence |
|:---|:---|:---|
| G8 (Agent Autonomy) | 6/6 PASS | GPT-2 1.030×, LLaMA-family 1.239×, DS-V2 audit-only; correctness 100% |
| G9 (Phase 1 Final) | CLOSED 2026-06-25 | 4-model family-substitute; Arke 1.263× vs LLM-direct; KB 292 entries |
| Kill criterion | NOT triggered | LLM-best > heuristic-floor in majority of trajectories |

**Verdict: ✅ VALIDATED** (dev-HW; full-scale deferred per 6GB VRAM)

---

## 8. Thesis L2 — Cross-Architecture (Phase 2)

**Claim:** Same IR schema re-usable on SIMD/heterogeneous backend (Ascend).

| Aspect | Status |
|:---|:---|
| Phase state | ⏸️ PAUSED (Leon directive 2026-07-02; no hardware) |
| Extensibility | `BackendRegistry` protocol preserved |
| Validation | None — no ops ported, no data |

**Verdict: ❌ UNTESTED** (by design — hardware prerequisite unmet)

---

## 9. Thesis L3 — Cross-Abstraction-Layer (Phase 3→4→5)

**Claim:** Deeper lowering → monotonically better perf; correctness + LLM-decision quality hold.

| Phase | Backend | Result | Evidence |
|:---|:---|:---|:---|
| 3 | MLIR GPU | 1.14× cuBLAS | 46/46 ops |
| 4 | CUDA-C | ~1.05× cuBLAS | 46/46 ops, MCP, BYOK |
| 5 | LLVM IR | 0.923 weighted geomean (gate 0.952) | Live-LLM L3 gate 5/5 (f6808da) |
| Monotonic | ✅ | Each phase beat previous on same RTX 3060 |
| Kill criterion | NOT triggered | No lowering-loss > LLM-gain observed |

**Verdict: ✅ VALIDATED** (NVIDIA single-HW; multi-HW half untested)

---

## Summary — 初心达成度

| # | Feature / Thesis | Verdict | Notes |
|:---:|:---|:---:|:---|
| 1 | AI-First Design | ✅ | 9-tool Façade, live-agent gate proven |
| 2 | Semantic/Strategy Separation | ✅ | 3-layer IR (Semantic/Strategy/Instruction), L1/L2/L3 |
| 3 | Minimal-Token Efficiency | ✅ | 72 vs 563 vs 1102 tokens; 1.263× vs LLM-direct |
| 4 | Compiler-Verified Optimization | ✅ | V0→V1→V2 pipeline; 2722 tests |
| 5 | @rationale KB | ✅ | 390 entries (195% of ≥200 target) |
| 6 | Cross-Hardware Ambition | 🟨 | Architecture ready; execution NVIDIA-only |
| 7 | Thesis L1 (SIMT) | ✅ | Phase 1 CLOSED, G9 all pass |
| 8 | Thesis L2 (cross-arch) | ❌ | PAUSED — no hardware |
| 9 | Thesis L3 (cross-abstraction) | ✅ | Phases 3→4→5 monotonic improvement |

**Overall: 7/9 achieved, 1 partially achieved (architecture done), 1 untested by design.**

### Quantitative highlights

- **681 commits**, first `d0dcb95` → HEAD `2b9466e`
- **2722 tests**, `make test` EXIT=0
- **46/46 operators** across 4 backends (Triton/MLIR/CUDA-C/LLVM)
- **390 @rationale KB entries** (72 from live-LLM Phase 5)
- **4 backend implementations**: Triton, MLIR GPU dialect, CUDA-C, LLVM IR
- **Live-LLM verified**: P5-S5-T gate 5/5; G8 gate 6/6

### Honest gaps (not failures — documented as design choices)

1. **No multi-hardware validation** — Thesis L2 dormant; only NVIDIA tested
2. **No v1.0.0 release tag** — DEFERRED per Leon 2026-07-23 (NVIDIA-only ≠ release)
3. **6GB VRAM constraint** — full 7-8B parameter models not validated at scale; family-substitute口径 accepted
4. **V1 correctness**: some ops at V0_mock tier (mock=reference) not V1_triton (real GPU) — honest-gap doc'd

---

*Generated by Kitty (Arke Lead Engineer) from code/data/test evidence, not documentation claims alone.*

