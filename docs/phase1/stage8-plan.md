# Phase 1 — Stage 8: Agent Autonomy

> Gate G8 exit criteria → [plan.md](../roadmap/plan.md#stage-8-g8-agent-autonomy-)

**Objective:** Validate that the Arke Agent can autonomously generate strategies, iterate optimization, and produce correct kernels for real LLMs. Integrate torch.compile backend to eliminate dispatch overhead. Validate on LLaMA-2 7B and DeepSeek-V2 16B.

**Depends on:** S7 (v2 IR/Lang, MLIR skeleton, full op coverage)
**Blocks:** S9 (Phase 1 Final needs proven agent autonomy + 2 model baselines)

---

## Gate Criteria Breakdown

**BL Exit:** BL5 inherited (no regression vs G7) + BL6×L3 (GPT-2, LLaMA-2, DeepSeek-V2).

> Reference: `docs/benchmark/benchmark-design.md` for BL/OT/ST/L definitions; `docs/deprecated/phase1-gate-design.md` §6 for original G7 derivation.

### Benchmark Requirements

#### BL5 Inheritance (No Regression vs S7/G7)

| Dimension | Requirement | Measurement |
|:----------|:-----------|:------------|
| L1 BL5 correctness | ≥ G7 result (no regression) | `arke bench --bl 5 --layer l1` |
| L1 BL5 performance geomean | ≥ G7 result (no regression) | `arke bench --bl 5 --layer l1` |
| L2 BL5 fusion coverage | ≥ G7 fusion combination count | `arke bench --bl 5 --layer l2` |

#### L3 @ BL6 (3 Models) — Autonomous Generation Validation

| Model | Correctness | Performance | Memory | seq Coverage | Measurement |
|:------|:-----------|:------------|:-------|:------------|:------------|
| **GPT-2 Small** | top-1 100% | ≤ **1.20×** eager | ≤ 6GB | 128/512/1024 | `arke bench --bl 6 --model gpt2` |
| **LLaMA-2 7B** | top-1 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 | `arke bench --bl 6 --model llama2` |
| **DeepSeek-V2 16B** | top-1 100% | ≤ **1.40×** eager (MoE overhead) | ≤ 6GB (seq≤512, quantized) | 512/2048 | `arke bench --bl 6 --model deepseek` |

#### G8 Combined PASS Formula

```
G8 PASS = AND ALL:
  [G7-AE] Autonomous Engineering: G7-AE.1~AE.5 all pass (see below)
  [BL5]   BL5 inheritance: L1+L2 correctness and performance both ≥ G7 results
  [BL6]   L3 BL6 GPT-2: correctness 100% + latency ≤1.20× eager
  [BL6]   L3 BL6 LLaMA-2: correctness 100% + latency ≤1.30× eager
  [BL6]   L3 BL6 DS-V2: correctness 100% + latency ≤1.40× eager
```

### Gate Criteria Detail

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | Auto strategy: kernel-only .ak → LLM generates strategy → codegen → ≥80% cuBLAS | `arke optimize examples/matmul.ak --no-strategy` — auto-gen strategy, perf ≥80% cuBLAS |
| 2 | Iterative optimization: ≥3 compile→profile→adjust cycles in trajectory | trajectory JSONL contains ≥3 complete `compile→profile→adjust` cycles |
| 3 | Multi-input: .ak file + natural language + code snippet → all work E2E | ≥2 ops per input type validated E2E |
| 4 | torch.compile backend: GPT-2 latency ≤1.20× eager (fixes S5 known-fail) | `arke bench --bl 6 --model gpt2` — latency ≤1.20× eager |
| 5 | LLaMA-2 7B: top-1 correct + latency ≤1.30× eager | `arke bench --bl 6 --model llama2` — correct + ≤1.30× eager |
| 6 | DeepSeek-V2 16B: top-1 correct + latency ≤1.40× eager (seq≤512, quantized) | `arke bench --bl 6 --model deepseek` — correct + ≤1.40× eager |
| 7 | BL5 no regression: L1+L2 correctness and performance ≥ G7 results | `arke bench --bl 5 --layer l1 l2` — no regression vs G7 |

---

## Tasks

### Track 1: Autonomous Strategy Generation (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D7-A1 | Auto strategy generation (kernel-only `.ak` → LLM full strategy pipeline) | P0 | XL | ⬜ |
| D7-A2 | Iterative optimization loop (auto-trigger ≥3 rounds compile→profile→adjust) | P0 | L | ⬜ |
| D7-A3 | Multi-input type routing (`.ak` / natural language / existing code → unified parse) | P0 | L | ⬜ |
| D7-A4 | E2E profile → kernel feedback loop (bottleneck op → re-optimize) | P1 | L | ⬜ |
| D7-A5 | Batch optimize pipeline (full model op set batch optimization) | P1 | M | ⬜ |

### Track 2: Agent Architecture Migrations (from agent-design.md §7)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| Agent-G7-M1 | AsyncGenerator optimization loop (Migration 1) | P0 | L | ⬜ |
| Agent-G7-M3 | Segmented prompt cache (Migration 3) | P1 | M | ⬜ |
| Agent-G7-M4 | Context compact (predictive + reactive) (Migration 4) | P1 | M | ⬜ |
| Agent-G7-M5 | Large result delta compression (Migration 5) | P2 | M | ⬜ |
| Agent-G7-M6 | Provider fallback + retry chain (Migration 6) | P1 | M | ⬜ |
| Agent-G7-M7 | Cross-compact ground truth state (Migration 7) | P2 | M | ⬜ |

### Track 3: Agent Prompts + Knowledge (P1)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D7-A6 | Long-context agent prompt (seq>4K branch strategy) | P1 | M | ⬜ |
| D7-A7 | MoE-aware optimization prompt (top-k sparsity, load balance) | P1 | M | ⬜ |
| D7-A8 | Quantized inference agent prompt (W4A8, W8A8 strategy) | P2 | M | ⬜ |
| D7-A9 | @rationale knowledge base accumulation (≥30 G8 entries) | P2 | M | ⬜ |

### Track 4: torch.compile + E2E Engineering (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D7-E1 | torch.compile Inductor backend | P0 | XL | ⬜ |
| D7-E2 | LLaMA-2 7B integration + bench_l3 runner | P0 | L | ⬜ |
| D7-E3 | DeepSeek-V2 integration (seq≤512, quantized weights) | P2 | L | ⬜ |
| D7-E4 | Triton MLA template (compressed KV, lora project) | P1 | L | ⬜ |
| D7-E5 | Triton paged_attention template (block table scatter read) | P1 | L | ⬜ |
| D7-E6 | bench runner OOM guard + CSV annotation | P2 | S | ⬜ |
| D7-E7 | `bench_l3.py` (model forward + top-1 comparison + latency stats) | P0 | M | ⬜ |

### Track 5: IR + Lang Extensions (P1)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D7-IR1 | PipelineStageStrategy (prefill/decode separation) | P1 | M | ⬜ |
| D7-IR2 | MultiLatentAttentionIR (kv_lora_rank, qk_rope_head_dim) | P1 | S | ⬜ |
| D7-IR3 | GroupedMatmulSemanticIR expert_indices field | P1 | S | ⬜ |
| D7-IR4 | PaddingStrategy refinement (inherits D6-IR8) | P2 | S | ⬜ |
| D7-L1 | `.ak` `@context_len` annotation primitive | P2 | S | ⬜ |
| D7-L2 | paged memory semantic node (block_table, page_size) | P1 | M | ⬜ |
| D7-L3 | moe_dispatch/combine high-level primitives | P2 | M | ⬜ |
| D7-L4 | MLA parameter semantic nodes | P2 | S | ⬜ |
| D7-L5 | @dtype int8/fp8 annotation extension | P2 | S | ⬜ |

---

## Autonomous Engineering Capability (G7-AE) Reference

These are the core autonomy criteria that S8 must satisfy:

| ID | Criterion | Verification |
|:---|:----------|:------------|
| G7-AE.1 | LLM auto-generates strategy (no human strategy block) | kernel-only `.ak` → LLM generates strategy → codegen → ≥80% cuBLAS |
| G7-AE.2 | Iterative optimization loop ≥3 rounds | trajectory JSONL contains ≥3 complete `compile→profile→adjust` cycles |
| G7-AE.3 | Multi-input type support | `.ak` file, natural language, existing code snippet → ≥2 ops per type validated E2E |
| G7-AE.4 | `arke optimize <input>` unified entry point | CLI single command: input → LLM optimize → Triton → GPU → benchmark report |
| G7-AE.5 | E2E profile → kernel feedback loop | bottleneck op identification → re-optimize → latency improvement verifiable |

## L3 @ BL6 Model Targets

| Model | Correctness | Performance | Memory | seq Coverage |
|:------|:-----------|:------------|:-------|:------------|
| **GPT-2 Small** | top-1 token 100% | ≤ **1.20×** eager (torch.compile backend) | ≤ 6GB | 128/512/1024 |
| **LLaMA-2 7B** | top-1 token 100% matches eager | ≤ **1.30×** eager (torch.compile backend) | ≤ 6GB | 512/2048/4096 |
| **DeepSeek-V2 16B** | top-1 token 100% matches eager (seq∈{512,2048}) | ≤ **1.40×** eager (MoE dispatch overhead) | ≤ 6GB (seq≤512, quantized) | 512/2048 |

---

## Key Milestones

| Milestone | Tracks | Day Estimate | Gate Criteria |
|:----------|:------:|:------------:|:-------------|
| M1: torch.compile backend MVP | Track 4 (D7-E1) | Day 5 | G8[4] partial |
| M2: Auto strategy generation | Track 1 (D7-A1, D7-A2) | Day 8 | G8[1], G8[2] |
| M3: Multi-input support | Track 1 (D7-A3) | Day 10 | G8[3] |
| M4: GPT-2 latency fixed | Track 4 (D7-E1) | Day 6 | G8[4] |
| M5: LLaMA-2 E2E | Track 4 (D7-E2, D7-E7) | Day 12 | G8[5] |
| M6: DeepSeek-V2 E2E | Track 4 (D7-E3) | Day 15 | G8[6] |
| M7: BL5 non-regression | — | Day 16 | G8[7] |

**Critical path:** torch.compile backend → GPT-2 fix → LLaMA-2 E2E → DeepSeek-V2 E2E

---

## Dependencies

- **Depends on:** S7 (v2 IR/Lang, MLIR skeleton, full op coverage)
- **Blocks:** S9 (Phase 1 Final needs proven agent autonomy, GPT-2/LLaMA-2/DS-V2 baselines)
