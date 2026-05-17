# Phase 1 — Stage 8: Agent Autonomy

> Gate G8 exit criteria → [plan.md](../roadmap/plan.md#stage-8-g8-agent-autonomy-)

**Objective:** Validate that the Arke Agent can autonomously generate strategies, iterate optimization, and produce correct kernels for real LLMs. Integrate torch.compile backend to eliminate dispatch overhead. Validate on LLaMA-2 7B and DeepSeek-V2 16B.

**Depends on:** S7 (v2 IR/Lang, MLIR skeleton, full op coverage)
**Blocks:** S9 (Phase 1 Final needs proven agent autonomy + 2 model baselines)

---

## Gate Criteria Breakdown

**BL Exit:** BL5 inherited (no regression vs G7) + BL6×L3 (GPT-2, LLaMA-2, DeepSeek-V2).

> Reference: `docs/benchmark/benchmark-design.md` for BL/OT/ST/L definitions. BL5 inheritance is governed by the active Stage 7 / Stage 8 contracts in `docs/roadmap/plan.md`; the old derivation note was retired during the spec cleanup.

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
| **GPT-2 Small** | 100% | ≥ **0.95×** eager | ≤ 6GB | 128/512/1024 | `arke bench --bl 6 --model gpt2` |
| **LLaMA-2 7B** | 100% | ≥ **0.90×** eager | ≤ 6GB | 512/2048/4096 | `arke bench --bl 6 --model llama2` |
| **DeepSeek-V2 16B** | 100% | ≥ **0.85×** eager (MoE overhead) | ≤ 6GB (seq≤512, quantized) | 512/2048 | `arke bench --bl 6 --model deepseek` |

#### G8 Combined PASS Formula

```
G8 PASS = AND ALL:
  [G7-AE] Autonomous Engineering: G7-AE.1~AE.5 all pass (see below)
  [BL5]   BL5 inheritance: L1+L2 correctness and performance both ≥ G7 results
  [BL6]   L3 BL6 GPT-2 [4a] vanilla torch.compile: correctness 100% + perf ≥0.95× eager
  [BL6]   L3 BL6 GPT-2 [4b] Arke→torch.compile bridge: correctness 100% + perf ≥0.95× eager + bridge-invocation-evidence
  [BL6]   L3 BL6 LLaMA-2: correctness 100% + perf ≥0.90× eager
  [BL6]   L3 BL6 DS-V2: correctness 100% + perf ≥0.85× eager
```

### Gate Criteria Detail

| # | Criterion | Verification |
|:-:|:----------|:-------------|
| 1 | Auto strategy: kernel-only .ak → LLM generates strategy → codegen → ≥0.95× P0 (cuBLAS) | `arke optimize examples/matmul.ak --no-strategy` — auto-gen strategy, perf ≥0.95× cuBLAS |
| 2 | Iterative optimization: ≥3 compile→profile→adjust cycles in trajectory | trajectory JSONL contains ≥3 complete `compile→profile→adjust` cycles |
| 3 | Multi-input: .ak file + natural language + code snippet → all work E2E | ≥2 ops per input type validated E2E |
| 4a | **Vanilla `torch.compile` backend baseline** (no Arke path): GPT-2 correctness 100% + perf ≥0.95× eager via stock `torch.compile(model, mode="reduce-overhead", dynamic=True)` with `cache_size_limit=64`. This proves PyTorch's own Inductor stack works under the Arke bench harness — necessary but **not sufficient** evidence for "Arke is doing work". | `arke bench --bl 6 --model gpt2` (mode=torch_compile vanilla) — correctness 100% + perf ≥0.95× eager |
| 4b | **Arke→`torch.compile` bridge active**: ≥1 Arke-generated Triton kernel (rmsnorm or matmul) registered as `torch.library` custom op, picked up by `torch.compile`'d GPT-2 forward, and the resulting GPT-2 perf is ≥0.95× eager **with the Arke kernel measurably on the critical path** (trace shows ≥1 invocation per forward, evidenced by torch profiler / dynamo log). This is what makes G8[4] a real *Arke* gate criterion rather than a torch.compile smoke test. | `arke bench --bl 6 --model gpt2 --bridge arke` — correctness 100% + ratio ≥0.95× + bridge-invocation-evidence in artifact |
| 5 | LLaMA-2 7B: correctness 100% + perf ≥0.90× eager | `arke bench --bl 6 --model llama2` — correct + ≥0.90× eager |
| 6 | DeepSeek-V2 16B: correctness 100% + perf ≥0.85× eager (seq≤512, quantized) | `arke bench --bl 6 --model deepseek` — correct + ≥0.85× eager |
| 7 | BL5 no regression: L1+L2 correctness and performance ≥ G7 results | `arke bench --bl 5 --layer l1 l2` — no regression vs G7 |

> **G8[4] split rationale (Leon-approved 2026-05-17):** the original G8[4] passed (2026-05-17, D7-E1.6) using vanilla `torch.compile` with no Arke code on the critical path. That's a legitimate but **value-thin** PASS — it shows the bench harness works and the regression was a measurement artifact, not that Arke optimized anything. Splitting into [4a] vanilla (already ✅) + [4b] Arke bridge (⬜) makes "Arke is doing work" an explicit, machine-checkable contract rather than implicit hand-waving. [4b] is the **real** G8[4] success criterion; [4a] is its prerequisite floor.

### Stage 8 MVP Bootstrap (implemented)

The first Stage 8 bootstrap slice is intentionally narrower than the locked full G8 exit criteria. It establishes stable, machine-checkable contracts for the two P0 tracks before wiring live LLM calls and full-model GPU validation:

- `arke optimize <input>`: deterministic heuristic auto-strategy generation for kernel-only `.ak` input, with unified routing for inline `.ak`, natural language, code snippets, and structured `--kernel/--shape` input.
- Artifacts: `strategy.json`, `result.akir`, `trajectory.jsonl`, and `summary.json`.
- Trajectory schema: `s8-compile-profile-adjust-v1` with at least three ordered `compile → profile → adjust` cycles.
- `benchmarks.bench_l3`: GPT-2 eager vs `torch.compile` CSV/JSON artifact contract, with `--mock` for CPU-safe CI and contract testing.
- Gate hook: `python -m benchmarks.gate G8` validates this MVP artifact contract and regression slice.

This does **not** relax full G8. The remaining Stage 8 work is still the live LLM strategy path, multi-input routing, BL5 no-regression performance inheritance, GPT-2 GPU target validation, LLaMA-2, and DeepSeek-V2.

---

## Two-Layer Architecture Mapping (Façade vs Substrate)

> **Locked principle (2026-05-17, Leon-approved):** Stage 8 work is partitioned into **Public Harness Façade** workstreams (stable vendor-agnostic contract) and **Arke Substrate** workstreams (Arke-internal IR/compiler/codegen). See `docs/architecture/arke-harness.md §3.0` for the architectural rationale. This partitioning is documentation-only — it does **not** change Gate G8 thresholds or PASS criteria, only how milestones are grouped.

| Layer | Tracks here | What it delivers in S8 |
|:------|:------------|:-----------------------|
| **Public Façade** | Track 2 (Harness Migrations M1-M7), Track 6 (Harness Extensions X1-X9) | AsyncGenerator loop, ToolMeta concurrent batching, segmented prompt cache, compact policy, provider fallback, skills runtime, hooks runtime, subagents, MCP server, `arke.config.yaml`. All stable across LLM provider / agent runtime. |
| **Arke Substrate** | Track 1 (Autonomous Strategy D7-A1..A5), Track 3 (Agent Prompts D7-A6..A9), Track 4 (torch.compile + E2E D7-E1..E7), Track 5 (IR + Lang D7-IR1..IR4, D7-L1..L5) | LLM strategy generator, op-registry coupling, V0/V1/V2 wiring, `torch.library` bridge, model E2E adapters, StrategyIR extensions (MLA / paged / MoE), `.ak` annotations. Free to evolve per Stage. |

Track 4 D7-E1.4 (Arke→torch.compile bridge) is a **Substrate** workstream that produces a **Façade**-facing artifact: the registered `torch.library` custom op becomes part of the Façade tool surface in Stage 9.

---

## Tasks

### Track 1: Autonomous Strategy Generation (P0)

| ID | Task | Priority | Estimate | Status |
|:---|:-----|:--------:|:--------:|:------:|
| D7-A1 | Auto strategy generation (kernel-only `.ak` → LLM full strategy pipeline; MVP heuristic path implemented) | P0 | XL | 🚧 |
| D7-A2 | Iterative optimization loop (auto-trigger ≥3 rounds compile→profile→adjust; MVP trajectory implemented) | P0 | L | 🚧 |
| D7-A3 | Multi-input type routing (`.ak` / natural language / existing code → unified parse; MVP router implemented, ≥2 ops/type evidence still open) | P0 | L | 🚧 |
| D7-A4 | E2E profile → kernel feedback loop (bottleneck op → re-optimize) | P1 | L | ⬜ |
| D7-A5 | Batch optimize pipeline (full model op set batch optimization) | P1 | M | ⬜ |

### Track 2: Harness Architecture Migrations (from arke-harness.md §18.2)

> ID prefix renamed from `Agent-G7-*` → `Harness-G8-*` together with the
> agent→harness rename. Migration numbering (M1–M7) is preserved so prior
> cross-refs still resolve. M2 (declarative `ToolMeta`) was completed in S6 and
> is listed for traceability.

| ID | Task | Ref | Priority | Estimate | Status |
|:---|:-----|:---:|:--------:|:--------:|:------:|
| Harness-G8-M1 | AsyncGenerator optimization loop | §4 | P0 | L | ⬜ |
| Harness-G8-M2 | Tool self-description + concurrent batching | §6 | P0 | M | ✅ (S6) |
| Harness-G8-M3 | Segmented prompt cache (4-segment cache_control) | §7 | P1 | M | ⬜ |
| Harness-G8-M4 | Context compact (predictive + reactive) | §10.2 | P1 | M | ⬜ |
| Harness-G8-M5 | Large-result delta compression (top-N + filter) | §6, §13 | P2 | M | ⬜ |
| Harness-G8-M6 | Provider fallback + retry chain | §16.2 | P1 | M | ⬜ |
| Harness-G8-M7 | Cross-compact ground-truth state | §8 | P2 | M | ⬜ |

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
| D7-E1 | torch.compile Inductor backend artifact contract (full GPU backend target still open) — umbrella; see D7-E1.1..E1.6 | P0 | XL | 🚧 |
| D7-E1.1 | **Diagnose 0.811× regression**: profile GPT-2 eager vs torch.compile (seq=128/256/512) with `torch._dynamo.config.verbose=True` + `TORCH_LOGS=output_code` + `nvprof`; categorize overhead sources; produce `diagnosis.md` with ranked root causes. **Outcome (2026-05-17, see `benchmarks/results/phase1/stage8/track4/diagnose_2026-05-16/diagnosis.md`):** original 0.811× was a measurement artifact from warmup=3; under warmup=10/runs=20 → seq=128 ratio 1.006 ✅, seq=512 ratio 1.024 ✅, seq=256 ratio 0.865 ❌. Real root cause = dynamic-shape recompile thrash hitting `cache_size_limit=8`. | P0 | S | ✅ |
| D7-E1.6 | **Dynamic-shape recompile control** (new — supersedes E1.2/E1.3 as critical path): bump `torch._dynamo.config.cache_size_limit` to ~64, compile once with `dynamic=True`, isolate per-seq warmups in `bench_l3`. Target: seq=256 ratio ≥0.95×. **Outcome (2026-05-17, see `benchmarks/results/phase1/stage8/track4/l3/2026-05-17_112120/summary.json`):** seq=128 → 1.024 ✅, seq=256 → **1.070 ✅** (from 0.865 baseline), seq=512 → 1.072 ✅. `summary.json: g8_gpt2_pass=true, min_compile_ratio=1.024, geomean=1.055`. G8[4] gate criterion met. | P0 | S | ✅ |
| D7-E1.5 | **Hit G8[4] target**: rerun `bench_l3 --model gpt2 --seq-len 128,256,512 --runs 20 --warmup 10`; require ratio_vs_eager ≥0.95 at all three seq lens; commit fresh CSV+summary.json under `benchmarks/results/phase1/stage8/track4/l3/` and update M1/M4 status. **Outcome (2026-05-17): satisfied by E1.6 same-day rerun → `summary.json: g8_gpt2_pass=true`, all three seq_lens ≥ 1.024×.** | P0 | S | ✅ |
| D7-E1.4 | **Arke→torch.compile bridge MVP** (now G8[4b] critical path): implement minimum `arke.backend.torch_compile` shim that registers Arke-generated Triton kernels as `torch.library` custom ops, routable from `torch.compile`'d models; ship 2 ops end-to-end (rmsnorm + matmul) with correctness tests **+ trace evidence that the Arke kernel is invoked on GPT-2's critical path**. After 2026-05-17 G8[4] split this is what makes G8[4] a real Arke criterion (not a torch.compile smoke test). | P0 | L | ⬜ |
| D7-E1.2 | **Eliminate graph breaks** (downgraded P1 after E1.1 finding): 12 dynamo breaks all come from `_collections_abc.Mapping.__contains__` skipfile in transformers GPT-2 forward; intra-layer, not the dominant cost. Either patch the forward to use `is not None` checks or whitelist `collections.abc` via `torch._dynamo.config.skipfiles_inline_module_allowlist`. Revisit only if E1.6 isn't sufficient at long seq. | P1 | M | ⬜ |
| D7-E1.3 | **CUDA Graph + reduce-overhead tuning** (downgraded P1 after E1.1 finding): 128/512 already ≥1.0× under reduce-overhead, so per-token dispatch tax is NOT the dominant overhead at GPT-2 scale. Hold for LLaMA-2 (D7-E2) where decode-step dispatch may bite. | P1 | M | ⬜ |
| D7-E2 | LLaMA-2 7B integration + bench_l3 runner | P0 | L | ⬜ |
| D7-E3.0 | **DS-V2 16B reachability probe** (S size, Leon-approved 2026-05-17, Q2=a): before committing to D7-E3 full integration, run a 1-day probe — load DS-V2 16B in 4bit quant on RTX 3060 6GB, forward at seq=512 batch=1, measure peak VRAM + latency floor. Two outcomes: (i) fits & forward completes → confirm D7-E3 as G8[6] critical path; (ii) OOMs even at smallest config → escalate to Leon for decision (audit-only G8[6] vs reschedule to bigger GPU). Deliverable: `benchmarks/results/phase1/stage8/track4/dsv2_probe_YYYY-MM-DD/probe.md` with VRAM trace, latency, decision recommendation. | P0 | S | ⬜ |
| D7-E3 | DeepSeek-V2 integration (seq≤512, quantized weights) — gated on D7-E3.0 outcome | P2→**P0 if E3.0 (i)** / **audit-only if E3.0 (ii)** | L | ⬜ |
| D7-E4 | Triton MLA template (compressed KV, lora project) | **P1 (held — Leon-approved 2026-05-17, Q3=b)**: G8[6] strategy = run DS-V2 via transformers eager + Arke optimization at op level; MLA template promoted to P0 only if E3.0 reveals MLA-on-critical-path AND eager perf misses ≥0.85× threshold. | L | ⬜ |
| D7-E5 | Triton paged_attention template (block table scatter read) | **P1 (held — same rationale as D7-E4)**: paged_attention is decode-phase optimization, not strictly required for seq≤512 prefill bench. Promote to P0 only if E3.0 evidence forces it. | L | ⬜ |
| D7-E6 | bench runner OOM guard + CSV annotation | P2 | S | ⬜ |
| D7-E7 | `bench_l3.py` (model forward + top-1 comparison + latency stats; GPT-2 eager/torch.compile MVP implemented) | P0 | M | 🚧 |

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

### Track 6: Harness Extensions (from arke-harness.md §11–§14)

> Net-new runtime work introduced by the v0.2 Arke Harness redesign. None of
> these are blockers for G8; they are P1/P2 follow-ups that bring Claude-Code
> primitives (skills, hooks, subagents, MCP) into Arke. Track them here so the
> design doesn't bit-rot once the migrations (Track 2) land.

| ID | Task | Ref | Priority | Estimate | Status |
|:---|:-----|:---:|:--------:|:--------:|:------:|
| Harness-G8-X1 | `SKILL.md` loader + skill registry; lift `skills/arke-test-coverage` to the new schema | §11 | P1 | M | ⬜ |
| Harness-G8-X2 | Built-in skills: `operator-coverage`, `bench-gate`, `tier-promotion`, `flash-attn` | §11.2 | P2 | L | ⬜ |
| Harness-G8-X3 | Hook runtime (8 lifecycle points) + `arke.config.yaml` hook registration | §12 | P1 | M | ⬜ |
| Harness-G8-X4 | Default trajectory writer rewritten as a hook bundle (replaces in-line writes) | §12.2, §15 | P1 | S | ⬜ |
| Harness-G8-X5 | Subagent `spawn_subagents()` API with forked `OptimizationState` + budget split | §13 | P2 | L | ⬜ |
| Harness-G8-X6 | Subagent: parallel tile-size sweep example + trajectory layout under `subagents/<name>/` | §13.1 | P2 | M | ⬜ |
| Harness-G8-X7 | `arke mcp serve` (stdio transport) — auto-derive tool schemas from `ToolMeta` | §14.1, §14.2 | P2 | L | ⬜ |
| Harness-G8-X8 | MCP resources: `arke://kernels/*`, `arke://hw/*`, `arke://trajectory/*` | §14.1 | P2 | M | ⬜ |
| Harness-G8-X9 | `arke.config.yaml` schema + layered loader (`~/.arke/` → repo → CLI → env) | §17 | P1 | M | ⬜ |

---

## Autonomous Engineering Capability (G7-AE) Reference

These are the core autonomy criteria that S8 must satisfy:

| ID | Criterion | Verification |
|:---|:----------|:------------|
| G7-AE.1 | LLM auto-generates strategy (no human strategy block) | kernel-only `.ak` → LLM generates strategy → codegen → ≥0.95× P0 |
| G7-AE.2 | Iterative optimization loop ≥3 rounds | trajectory JSONL contains ≥3 complete `compile→profile→adjust` cycles |
| G7-AE.3 | Multi-input type support | `.ak` file, natural language, existing code snippet → ≥2 ops per type validated E2E |
| G7-AE.4 | `arke optimize <input>` unified entry point | CLI single command: input → LLM optimize → Triton → GPU → benchmark report |
| G7-AE.5 | E2E profile → kernel feedback loop | bottleneck op identification → re-optimize → latency improvement verifiable |

---

## Key Milestones

| Milestone | Tracks | Layer | Day Estimate | Gate Criteria |
|:----------|:------:|:-----:|:------------:|:-------------|
| M1: torch.compile backend MVP | Track 4 (D7-E1.1 ✅, E1.6 ✅) | Substrate | Day 5 | G8[4a] ✅ |
| M2: Auto strategy generation | Track 1 (D7-A1, D7-A2) | Substrate | Day 8 | G8[1], G8[2] |
| M3: Multi-input support | Track 1 (D7-A3) | Substrate | Day 10 | G8[3] |
| M4a: GPT-2 perf ≥0.95× eager (vanilla path) | Track 4 (D7-E1.6 → E1.5) | Substrate | Day 6 | G8[4a] ✅ (2026-05-17) |
| **M4b: GPT-2 Arke bridge active on critical path** | Track 4 (D7-E1.4) | Substrate (Façade-facing artifact) | Day 9 | **G8[4b]** ⬜ |
| M5: LLaMA-2 E2E | Track 4 (D7-E2, D7-E7) | Substrate | Day 12 | G8[5] |
| **M6a: DS-V2 reachability probe** | Track 4 (D7-E3.0) | Substrate | Day 13 | gates M6b |
| M6b: DeepSeek-V2 E2E (or audit-only escalation) | Track 4 (D7-E3) | Substrate | Day 15 | G8[6] |
| M7: BL5 non-regression | — | Substrate | Day 16 | G8[7] |

**Critical path:** torch.compile backend → GPT-2 vanilla [4a] ✅ → **GPT-2 Arke bridge [4b]** → LLaMA-2 E2E → DS-V2 probe → DS-V2 E2E

---

## Dependencies

- **Depends on:** S7 (v2 IR/Lang, MLIR skeleton, full op coverage)
- **Blocks:** S9 (Phase 1 Final needs proven agent autonomy, GPT-2/LLaMA-2/DS-V2 baselines)
