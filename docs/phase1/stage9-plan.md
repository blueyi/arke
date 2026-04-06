# Phase 1 — Stage 9: Phase 1 Final

> Gate G9 exit criteria → [plan.md](../roadmap/plan.md#stage-9-g9-phase-1-final-)

**Objective:** Final acceptance across 4 models. Automated Arke-vs-LLM-direct comparison. Spec freeze. Evaluation report. v1.0 release tag.

**Depends on:** S8 (Agent Autonomy — proven auto-strategy, GPT-2/LLaMA-2/DS-V2 baselines)
**Blocks:** Phase 2 (cross-architecture validation on Ascend NPU)

---

## Gate Criteria Breakdown

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | 4 models E2E: GPT-2 ≤1.15×, LLaMA-2 ≤1.30×, LLaMA-3 ≤1.30×, Qwen2.5 ≤1.30× (all top-1 correct) | `arke bench --bl 6 --model gpt2 llama2 llama3 qwen25` |
| 2 | Arke vs LLM-direct: correctness ≥ direct, tokens ≤ 0.70×, perf ≥ 0.90× | `benchmarks/compare_arke_vs_direct.py` |
| 3 | @rationale KB: ≥50 Phase 1 entries | `wc -l data/rationale_kb.jsonl` ≥ 50 |
| 4 | Spec freeze: Lang v1.0 + IR v1.0 tagged | `git tag arke-lang-v1.0` + `git tag arke-ir-v1.0` exist |
| 5 | Phase 1 evaluation report published | `PHASE1_FINAL_REPORT.md` exists and complete |
| 6 | v1.0.0 tag | `git tag v1.0.0` exists |

---

## Tasks

### Track 1: Additional Model Integration (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D8-E1 | LLaMA-3 8B integration + bench_l3 runner | P0 | L | ⬜ |
| D8-E2 | Qwen2.5 7B integration + bench_l3 runner | P0 | L | ⬜ |
| D8-E3 | GPT-2 torch.compile backend E2E (≤1.15× eager, depends on S8 D7-E1) | P0 | M | ⬜ |

### Track 2: Lang Examples + Spec Freeze (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D8-L1 | `qwen25_forward.ak` example (GQA+SwiGLU+RMSNorm) | P0 | M | ⬜ |
| D8-L2 | `llama3_forward.ak` example (GQA, rope, RMSNorm) | P0 | M | ⬜ |
| D8-L3 | `arke-io-spec.md` (I/O contract document) | P1 | M | ⬜ |
| D8-L4 | Language Spec v1.0 freeze (document + tag) | P0 | — | ✅ Done |

### Track 3: IR Finalization (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D8-IR1 | IR Spec v1.0 freeze (document + tag) | P0 | — | ✅ Done |
| D8-IR2 | `ir-mlir-mapping.md` (Phase 2 preparation) | P1 | — | ✅ Done |
| D8-IR3 | `test_ir_roundtrip.py` (all 45 ops × JSON round-trip) | P0 | M | ⬜ |

### Track 4: Agent Maturity + Comparison (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D8-A1 | `arke optimize` unified entry with full 3-input-type support | P0 | L | ⬜ |
| D8-A2 | LLM auto-strategy maturity validation (all 45 ops, no human strategy) | P0 | L | ⬜ |
| D8-A3 | Iterative loop stable operation across 4 models | P0 | M | ⬜ |
| D8-A4 | @rationale knowledge base (≥50 Phase 1 entries) | P1 | M | ⬜ |
| D8-A5 | Arke vs LLM-direct automated comparison (`benchmarks/compare_arke_vs_direct.py`) | P0 | L | ⬜ |

### Track 5: Evaluation + Release (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D8-E4 | BL5 regression suite (CI): `ci/regression_bl5.py` | P1 | M | ⬜ |
| D8-E5 | Language evaluation benchmark + `language-decision.md` | P1 | M | ⬜ |
| D8-E6 | Phase 1 final evaluation report `PHASE1_FINAL_REPORT.md` | P0 | L | ⬜ |

---

## L3 @ BL6 Model Targets (4 Models)

| Model | Correctness | Performance | Memory | seq Coverage |
|:------|:-----------|:------------|:-------|:------------|
| **GPT-2 Small** | top-1 token 100% | ≤ **1.15×** eager (torch.compile backend) | ≤ 6GB | 128/512/1024 |
| **LLaMA-2 7B** | top-1 token 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 |
| **LLaMA-3 8B** | top-1 token 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 |
| **Qwen2.5 7B** | top-1 token 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 |

## Arke vs LLM-direct Comparison Criteria

| Dimension | Requirement |
|:----------|:-----------|
| Correctness | Arke ≥ LLM-direct (100% vs ≤90%) |
| Token efficiency | Arke ≤ 0.70× LLM-direct tokens |
| Performance | Arke geomean ≥ 0.90× LLM-direct |

---

## Key Milestones

| Milestone | Tracks | Day Estimate | Gate Criteria |
|:----------|:------:|:------------:|:-------------|
| M1: LLaMA-3 + Qwen2.5 integration | Track 1 | Day 3 | G9[1] partial |
| M2: 4-model E2E pass | Track 1, 2 | Day 5 | G9[1] |
| M3: Arke vs LLM-direct automated | Track 4 (D8-A5) | Day 7 | G9[2] |
| M4: @rationale KB ≥50 | Track 4 (D8-A4) | Day 8 | G9[3] |
| M5: Spec freeze + tags | Track 2, 3 | Day 4 | G9[4] |
| M6: Evaluation report + v1.0.0 | Track 5 | Day 10 | G9[5], G9[6] |

**Critical path:** Model integration → 4-model E2E → Arke vs LLM-direct → Evaluation report → v1.0.0 tag

---

## Dependencies

- **Depends on:** S8 (Agent Autonomy — proven auto-strategy, torch.compile, GPT-2/LLaMA-2/DS-V2 baselines)
- **Blocks:** Phase 2 (cross-architecture validation on Ascend NPU)
