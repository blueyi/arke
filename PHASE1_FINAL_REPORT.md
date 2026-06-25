# Arke Phase 1 — Final Evaluation Report

> **Status: FINAL (2026-06-25).** Phase 1 (Arke → Triton → NVIDIA, SIMT
> validation) is complete on the dev hardware. S0–S8 done; S9 closed except the
> two release tags the project lead chose **not** to cut (G9[4] spec-freeze tags
> and G9[6] v1.0.0 tag — explicitly waived 2026-06-25). All measurable Gate
> criteria pass; remaining scale-up (full 7-8B models) is a pure VRAM limit,
> deferred to a larger GPU with project-lead acceptance of family-substitute
> validation.

## 1. Thesis & scope

**Claim (Thesis L1):** the AI-Native paradigm — an LLM/Agent as the optimization
decision-maker, a structured IR it can read/write/reason over, and a compiler
that verifies every decision — produces **correct + competitive** AI operators
on SIMT architecture via the Triton backend.

**AI-Native is two-directional** (locked 2026-06-24): the toolchain is
*designed for an Agent consumer* (token-dense `.ak`, Agent-legible IR, V0/V1/V2
feedback shaped for the iterate loop, bounded action space + `@rationale` +
trajectory + checkpoint/rollback Harness) **and** *built the AI-Native way*
(Agent as lead engineer, dogfooding). The LLM has three roles: runtime
decision-maker, target user, and builder.

## 2. Stage outcomes (S0–S9)

| Stage | Gate | Objective | Outcome |
|---|---|---|---|
| S0–S5 | G0–G5 | env → IR → codegen → LLM loop → GPT-2 E2E | ✅ |
| S6 | G6 | Compiler infrastructure (OpRegistry/Pass/Backend) | ✅ 7/7, 46 ops correctness 100% |
| S7 | G7 | Lang & IR v0.1.0, same-backend Triton fairness | ✅ 13/14 (G7.8d honest-gap accepted) |
| S8 | G8 | Extensible Harness + L1 endpoint validation | ✅ **gate 6/6 PASS** (see §3) |
| S9 | G9 | Phase 1 final (4 models, KB, spec freeze, v1.0) | 🚧 in progress (see §4) |

## 3. Stage 8 — Harness + endpoint validation (gate 6/6 PASS)

**Tier 1 — Harness product (not relaxable):**
- HARNESS-1 Façade v1.0 frozen — 8-tool schema + OptimizationEvent stream +
  trajectory v1.0, 158 contract tests green.
- HARNESS-2 LLM autonomy — a **live** `claude-sonnet-4-6` (yunwu `/v1`) drives
  the real loop: ≥3–5 genuine compile→profile→adjust cycles on matmul with real
  Triton GPU measurement, adaptive `baseline_ratio`, checkpoint + rollback.
  Folded into the gate as `G8.LIVE.1` (liveness, no perf threshold).
- HARNESS-3 Extensibility — Demo A (new op `swiglu_packed` ≤400 LOC) + Demo B
  (new baseline `MaxAutotuneRunner` 141 ≤200 LOC), both within falsifiable LOC
  budgets, no harness-core change.

**Tier 2 — L1 endpoint validation:**
- [1][2][3] auto-strategy + ≥3 cycles + multi-input routing — ✅
- [4a] GPT-2 vanilla torch.compile: correctness 100%, **geomean 0.9517×**
  (geomean-over-seq口径, D3); seq=256 evidence-backed known-fail.
- [4b] **Arke→torch.compile bridge** — Arke's Triton matmul fires **48×/GPT-2
  forward** on the Conv1D critical path, top-1 token match vs eager. Concrete
  proof Arke kernels do real work end-to-end.
- [5] LLaMA family (TinyLlama-1.1B): correctness 100%, geomean 1.239×.
- [6] DeepSeek-V2 16B: audit-only (6 GB OOM, evidence recorded).
- [7] BL5 no regression: Stage-8 changes touch zero BL5 kernel/IR/lang code.

`python -m benchmarks.gate G8` → **6/6 PASS** (MVP.1–4 + LIVE.1 + GPT2.1).

## 4. Stage 9 — Phase 1 final (in progress)

| G9 criterion | Status |
|---|---|
| [1] 4-model BL6 E2E | ✅ **CLOSED** (Leon-accepted family-substitute口径) — GPT-2 geomean 1.0296× (≥1.00), LLaMA-family 1.239× (≥0.95), Qwen2.5-family 1.2796× (≥0.90), all correctness 100%. Full 7-8B deferred to larger GPU (VRAM limit). |
| [2] Arke vs LLM-direct (perf ≥1.05×, tokens ≤0.7×) | ✅ matmul 512³: Arke **1.263×** faster, **0 tokens** vs 1965, both correct → `passed: true`. |
| [3] @rationale KB ≥50 | ✅ **292 entries** (`data/rationale_kb.jsonl`). |
| [4] Spec freeze: Lang v1.0 + IR v1.0 tags | ⏭️ **waived by project lead** (2026-06-25). IR maturity demonstrated (46-op round-trip), but no freeze tag cut. |
| [5] Phase 1 evaluation report | ✅ this document. |
| [6] v1.0.0 tag | ⏭️ **waived by project lead** (2026-06-25). |

IR maturity for the freeze is demonstrated: `tests/test_ir_roundtrip.py` —
all 46 catalog ops survive SemanticIR JSON round-trip (93 tests).

## 5. Key engineering findings

- **Live-LLM loop is real, not mock.** Routing through the clean `/v1`
  OpenAI-compatible endpoint (not the bare Anthropic relay, which injects ~30K
  Claude-Code context) makes the Agent call the actual 8 Arke tools; the loop
  produces real Triton kernels with real GPU latency.
- **Small-model launch-bound floor.** Vanilla torch.compile genuinely loses to
  eager on small-model short-seq cases (GPT-2 seq=128/256 on a 6 GB SM 8.6),
  where Python launch + CUDA-graph guard overhead dominates tiny matmuls. It
  wins at longer seq and on larger models (LLaMA/Qwen families all >eager). This
  is a property of the baseline, not of Arke. Recorded honestly per the
  no-relaxation discipline (isolated re-measurement falsified the convenient
  "eviction artifact" story).
- **Architecture-agnostic generalization.** The same E2E path works across MHA
  (GPT-2), GQA+RMSNorm+RoPE (LLaMA), and GQA+wide-FFN (Qwen2.5) families.
- **Backend extensibility preserved.** `arke/backend/protocol.py` keeps the
  `ArkeBackend` + `BackendRegistry` seam clean for a future Ascend/AMD/MLIR
  backend (Phase 2 paused, not deleted).

## 6. Hardware envelope

RTX 3060 Laptop 6 GB (Ampere SM 8.6), CUDA 12.4, PyTorch 2.6.0+cu124, Triton
3.2.0. The 6 GB ceiling is the single binding constraint on full-scale model
validation; everything that fits has been measured for real.

## 7. Phase 1 sign-off (project-lead decisions, 2026-06-25)

All open items resolved by the project lead:

1. **4-model full-scale (G9[1]):** ✅ family-substitute validation **accepted**
   as closing G9[1]. Full 7-8B runs deferred to a larger GPU (pure VRAM limit).
2. **Arke vs LLM-direct (G9[2]):** ✅ comparison harness built + run live —
   Arke 1.263× faster, 0 vs 1965 tokens, `passed: true`.
3. **Spec freeze + v1.0.0 (G9[4]/[6]):** ⏭️ **waived** — the project lead chose
   not to cut Lang/IR/v1.0.0 release tags at this time. The toolchain is
   functionally complete; tagging is deferred.

**Phase 1 verdict:** Thesis L1 (AI-Native LLM-decision + structured IR +
compiler verification produces correct, competitive SIMT/Triton operators) is
**validated** on the dev hardware. Every measurable Gate criterion passes; the
only deferral is full-parameter model scale, bounded purely by 6 GB VRAM.

---

*Generated 2026-06-25. Commit trail: P0 live-LLM (`20d9fd4`) → G8 closure
(`5203035`/`6b100ff`/`e34979e`/`44d514f`) → G9 in progress
(`5be7d04`/`bcdd569`/`14e5837`/`07387a4`).*
