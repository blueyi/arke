# Arke Development Plan

> Single source of truth for all development planning.
> Architecture decisions → docs/architecture/; Specs → docs/spec/; This file → what to build and when.

## Hierarchy

Roadmap > Phase > Stage > Feature > Task

---

## Phase 1: Arke → Triton → NVIDIA GPU (SIMT Validation)

**Goal:** Prove LLM + structured IR + compiler verification produces correct, fast kernels on SIMT architecture.
**Hardware:** NVIDIA Ampere (RTX 3060 Laptop, 6GB VRAM) · CUDA 12.4 · PyTorch 2.6.0+cu124 · Triton 3.2.0

### Design Principles

1. **AI-First** — LLM agents make optimization decisions; compilers verify
2. **Minimal-Token** — Minimize total token consumption across the full pipeline
3. **Semantic/Strategy Separation** — *What to compute* and *how to optimize* are independent
4. **Compiler-Verified** — Every decision validated by deterministic checks

---

### Completed Stages

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

GPT-2 Small E2E inference: top-1 token correctness 100%, 49/48 Conv1D replacements, memory ≤1100MB/6144MB. **Known-fail (recorded, non-blocking):** E2E latency 1.71–2.20× eager due to monkey-patch dispatch ~60µs/call × 49 calls (root cause: Python dispatch overhead, not kernel quality). Single matmul: Arke 76µs vs cuBLAS 44µs. Resolution: G7 torch.compile Inductor backend. Full analysis: `benchmarks/results/phase1/gates/G5/REPORT.md`.

---

### Stage 6 (G6): Lang & IR Completeness ⬜ ← CURRENT

> **Core objective:** Verify that Arke Lang and Arke IR have complete expressibility, code generation capability, and performance competitiveness for all 45 operators across all shapes (ST1–ST4). Additionally (v2): verify the architecture is clean enough to support G7/G8. This is the watershed between Arke being "runnable" and "usable".

#### Exit Criteria

```bash
arke bench --bl 5 --layer l1    # OT0-4 × ST1-4, single op all shapes
arke bench --bl 5 --layer l2    # Fused op all shapes
```

##### L1 @ BL5 (OT0-4, ST1-4)

| Op Group | Correctness | Performance | Command |
|:---------|:------------|:------------|:--------|
| **OT0** Elementwise (12 ops) | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.90 P1 (FlagGems elem) | `bench_l1 --ot 0` |
| **OT1** Reduction (10 ops) | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.85 P1 (FlagGems norm/softmax) | `bench_l1 --ot 1` |
| **OT2** Compute-Dense (11 ops) | 100%(ST1-3) + ≥95%(ST4) | matmul geomean ≥ 0.90 P0; others ≥ P3 | `bench_l1 --ot 2` |
| **OT3** Gated Activation (7 ops) | 100%(ST1-3) + ≥95%(ST4) | swiglu/rope geomean ≥ 0.85 P1 (Liger/FlagGems) | `bench_l1 --ot 3` |
| **OT4** Attention (5 ops) | 100%(ST1-4, excl. OOM) | FA geomean ≥ 0.80 P1 (FlashAttn-2); GQA ≥ 0.80 | `bench_l1 --ot 4` |

> ST4 OOM note: OT4 may OOM on some large shapes with 6GB VRAM; mark `⚠️ OOM` and skip (not counted in pass-rate denominator).

##### L2 @ BL5 (Fused Operators)

| Fusion | Requirement | Baseline |
|:-------|:------------|:---------|
| matmul+relu, matmul+gelu | ≥ 1.05× unfused | P3 unfused |
| swiglu, geglu | ≥ 0.90× Liger | P1 |
| linear+cross_entropy | ≥ 1.05× unfused | P3 |
| QKV+flash_attention | ≥ 0.80× FlashAttn-2 | P1 |

##### Lang & IR Completeness (G6-LI)

| ID | Criterion | Verification |
|:---|:----------|:------------|
| G6-LI.1 | All 45 ops expressible and parseable in `.ak` | `arke parse examples/<op>.ak` all exit 0 |
| G6-LI.2 | `.ak → SemanticIR → StrategyIR → Pass pipeline` full round-trip | `python -m arke.compiler.pipeline --ak examples/<op>.ak --dry-run` passes all 45 ops |
| G6-LI.3 | `@rationale` annotations preserved through full pipeline | `scripts/check_rationale_chain.py` ≥3 examples verified |
| G6-LI.4 | Token efficiency: `.ak` ≤ Triton line count | OT0-OT4: `.ak` lines < Triton lines @ equivalent performance |
| G6-LI.5 | Python interop IR round-trip | `pytest tests/test_ir_roundtrip.py` — all 45 ops pass |
| G6-LI.6 | Grammar completeness: 0 parse failures | `arke parse examples/ --strict` |
| G6-LI.7 | Symbolic shape `.ak` → SemanticIR with `where` clause | `pytest tests/test_symbolic_shape.py` — ≥5 ops with `where` clause |
| G6-LI.8 | Backend-agnostic strategy (no Triton-specific fields in StrategyIR core) | `scripts/check_backend_agnostic.py` — 0 Triton-specific fields in `StrategyIR.decisions[]` |

##### Architecture Completeness (G6-ARCH) — New in v2

| ID | Criterion | Verification | Priority |
|:---|:----------|:------------|:--------|
| ARCH.1 | Arke Lang Spec v2.0 finalized | `docs/spec/arke-lang-spec-v2.md` exists | P1-G6 |
| ARCH.2 | Arke IR Multi-Layer Architecture spec finalized | `docs/spec/arke-ir-spec-v2.md` exists, defines Layer 4/3/2/1 | P1-G6 |
| ARCH.3 | OpRegistry implemented, replaces 6 separate op lists | `scripts/verify_op_registry.py` — adding op requires ≤2 file changes | P0-G6 |
| ARCH.4 | Pass Infrastructure skeleton | `pytest tests/test_pass_infra.py` — Pass protocol + Pipeline with ≥2 example passes | P1-G6 |
| ARCH.5 | SemanticInterpreter implemented | `pytest tests/test_semantic_interpreter.py` — all 45 ops correct | P0-G6 |
| ARCH.6 | SSA Validator implemented | `pytest tests/test_ssa_validator.py` — all 45 ops valid; ≥5 invalid examples rejected | P1-G6 |
| ARCH.7 | Backend Abstraction interface defined | `pytest tests/test_backend_protocol.py` — `ArkeBackend` protocol + `TritonBackend` implements it | P1-G6 |
| ARCH.8 | Arke Lang v2.0 `where` clause MVP | `arke parse examples/matmul_symbolic.ak` parses + SemanticIR `symbolic_dims` populated | P2-G7 |
| ARCH.9 | Layer 3/2/1 spec documents (stub) | `docs/spec/arke-ir-layer{3,2,1}-spec.md` + MLIR mapping updated | P2-G7 |
| ARCH.10 | No regression | `pytest tests/ -q` — ≥422 passed, ≤6 skipped, 0 new failures | P0-G6 |
| ARCH.11 | MLIR framework skeleton + BL1 pathway | MLIREmitter skeleton exists; BL1 matmul verified via MLIR skeleton | P1-G6 |
| ARCH.12 | Dynamic Shape feasibility assessment | `docs/phase1/dynamic-shape-feasibility.md` — covers where clause design, symbolic_dims IR, shape constraint propagation, Triton/MLIR integration points, risk assessment | P1-G6 |

> ARCH.8 and ARCH.9 are P2-G7 MVP scope: G6 can ship without them if ARCH.1–7, ARCH.10–12 pass.

##### G6 PASS Combined Criteria (v2)

```
AND ALL:
  [1] L1 BL5 correctness: 100%(ST1-3) + ≥95%(ST4, excl. OOM) for all OT0-OT4
  [2] L1 BL5 performance weighted_score ≥ 0.83
        weighted_score = 0.25×score(OT0-1) + 0.30×score(OT2)
                       + 0.20×score(OT3) + 0.25×score(OT4)
  [3] L2 BL5: ≥3/4 fusion combinations pass
  [4] G6-LI.1~LI.8 all pass
  [5] G6-ARCH.1~ARCH.7, ARCH.10, ARCH.11, ARCH.12 pass
      (ARCH.8, ARCH.9 are MVP-scoped — required if feasibility assessment positive)
```

---

#### G6 v1 Deliverables (DONE — commit fd2cbe0)

The original G6 criteria (performance/correctness/Lang&IR completeness) were fully satisfied:

- ✅ 46 `.ak` example files for all 45 ops (including `OT3_fused_linear_cross_entropy.ak` as bonus)
- ✅ Grammar fixes: array literals `[2,3]`, float constants `0.125`, 4D tensor support
- ✅ SemanticIR op catalog extended to 45 ops (OT3/OT4 all fields)
- ✅ AttentionSemanticIR fields: `mask_type`, `num_kv_heads`, `head_dim`
- ✅ RopeSemanticIR fields: `theta`, `base`, `rotary_dim`
- ✅ QuantizeSemanticIR fields: `scale_dtype`, `group_size`, `zero_point`
- ✅ MLA-specific fields: `latent_dim`, `kv_lora_rank`
- ✅ `ast_to_strategy()` converter: parser AST → StrategyIR
- ✅ StrategyIR JSON round-trip for all 45 ops
- ✅ 10 new Triton template classes: rope, flash_attention, GQA (OT3/OT4 full coverage)
- ✅ V1 validator extension: attention numerical tolerance, quantization precision standards
- ✅ 422 tests passing (up from 237), 6 skipped
- ✅ BL5×L1+L2 performance/correctness: 9/9 criteria pass, 46/46 E2E correct

---

#### G6 v2 Tasks

G6 v2 adds architecture completeness (ARCH.1–ARCH.12) on top of the already-passed v1 criteria. Work is organized into Phase C (architecture refactoring) and Phase D (spec documents + where clause + validation).

**Phase C — Architecture Refactoring**

*Track C1: OpRegistry + SemanticInterpreter*

| ID | Task | Priority | Estimate |
|:---|:-----|:--------:|:--------:|
| C1.1 | Design `OpSchema` dataclass + `OpRegistry` class | P0 | 0.5d |
| C1.2 | Migrate all 45 ops from `catalog.py` to `OpRegistry` | P0 | 1d |
| C1.3 | Remove op-specific if/elif from `shape_inference.py` | P0 | 0.5d |
| C1.4 | Implement `SemanticInterpreter` (PyTorch eager executor) | P0 | 1d |
| C1.5 | Migrate `numerical_check.py` to use `SemanticInterpreter` | P0 | 0.5d |
| C1.6 | Update `kernel_cache.py` to use parser instead of `_build_ir()` | P1 | 0.5d |
| C1.7 | Update `triton_template_engine.py` to use registry lookup | P0 | 0.5d |

*Track C2: Pass Infrastructure + SSA Validator (depends on C1)*

| ID | Task | Priority | Estimate |
|:---|:-----|:--------:|:--------:|
| C2.1 | Define `ArkePass` protocol + `PassContext` + `PassPipeline` | P1 | 0.5d |
| C2.2 | Implement `ShapeInferencePass` (wraps `shape_inference.py`) | P1 | 0.5d |
| C2.3 | Implement `SSAValidator` + `SSAValidationPass` | P1 | 1d |
| C2.4 | Implement `RationalePreservationPass` | P1 | 0.5d |
| C2.5 | Integrate `PassPipeline` into `ArkePipeline.run()` | P1 | 0.5d |

*Track C3: Backend Abstraction (independent)*

| ID | Task | Priority | Estimate |
|:---|:-----|:--------:|:--------:|
| C3.1 | Define `ArkeBackend` protocol + `BackendArtifact` hierarchy | P1 | 0.5d |
| C3.2 | Wrap `TritonBackend` to implement `ArkeBackend` | P1 | 0.5d |
| C3.3 | Update `ArkePipeline` to use backend via protocol | P1 | 0.5d |
| C3.4 | Implement `MockBackend` for testing | P1 | 0.5d |

**Phase D — Spec Documents + where Clause + Validation**

| ID | Task | Priority | Estimate |
|:---|:-----|:--------:|:--------:|
| D0 | Dynamic Shape feasibility assessment (`docs/phase1/dynamic-shape-feasibility.md`) — ARCH.12 | P1 | 1d |
| D1 | Write `docs/spec/arke-lang-spec-v2.md` — ARCH.1 | P1 | 1d |
| D2 | Write `docs/spec/arke-ir-spec-v2.md` (Layer 4 upgraded, Layer 3/2/1 interfaces) — ARCH.2 | P1 | 1.5d |
| D3 | Implement `where` clause in Lark grammar — ARCH.8 MVP | P2 | 0.5d |
| D4 | Add `symbolic_dims` field to SemanticIR + converter — ARCH.8 MVP | P2 | 0.5d |
| D5 | Shape propagation for symbolic dims in `ShapeInferencePass` — ARCH.8 MVP | P2 | 1d |
| D6 | Write `tests/test_symbolic_shape.py` — G6-LI.7 | P2 | 0.5d |
| D7 | Write `scripts/check_backend_agnostic.py` — G6-LI.8 | P1 | 0.5d |
| D8 | Full non-regression run + fix regressions — ARCH.10 | P0 | 1d |
| D9 | MLIR framework skeleton + Layer 3/2/1 spec stubs — ARCH.9, ARCH.11 | P2 | 1d |

**G6 v1 items — status refresh:**

| ID | Layer | Description | Status |
|:---|:------|:------------|:------:|
| D6-L1 | Lang | `.ak` 4D tensor syntax extension | ✅ Done (G6 v1) |
| D6-L2 | Lang | gather/scatter semantic nodes | ✅ Done (G6 v1) |
| D6-L3 | Lang | quantize primitive syntax | ✅ Done (G6 v1) |
| D6-L4 | Lang | paged memory semantic annotation (stub) | ✅ Done (G6 v1, stub) |
| D6-L5 | Lang | grammar fix (array literal, float constant) | ✅ Done (G6 v1) |
| D6-L6 | Lang | `.ak` example files for all 45 ops | ✅ Done (G6 v1, 46 files) |
| D6-IR1 | IR | SemanticIR op catalog → 45 ops | ✅ Done (G6 v1) |
| D6-IR2 | IR | AttentionSemanticIR (mask_type, num_kv_heads, head_dim) | ✅ Done (G6 v1) |
| D6-IR3 | IR | RopeSemanticIR (theta, base, rotary_dim) | ✅ Done (G6 v1) |
| D6-IR4 | IR | QuantizeSemanticIR (scale_dtype, group_size, zero_point) | ✅ Done (G6 v1) |
| D6-IR5 | IR | `ast_to_strategy()` converter | ✅ Done (G6 v1) |
| D6-IR6 | IR | StrategyIR JSON round-trip (all 45 ops) | ✅ Done (G6 v1) |
| D6-IR7 | IR | MLA-specific fields (latent_dim, kv_lora_rank) | ✅ Done (G6 v1) |
| D6-IR8 | IR | PaddingStrategy decision type | ⬜ Open |
| D6-A1 | Agent | attention prompt template | ⬜ Open |
| D6-A2 | Agent | rope prompt + rationale template | ⬜ Open |
| D6-A3 | Agent | fusion opportunity detection | ⬜ Open |
| D6-A4 | Agent | quantize/dequantize prompt template | ⬜ Open |
| D6-A5 | Agent | batch optimize pipeline (45 ops parallel sessions) | ⬜ Open |
| D6-A6 | Agent | non-aligned shape rationale template | ⬜ Open |
| D6-E1 | Eng | 10 Triton template classes | ✅ Done (G6 v1, OT3/OT4 full) |
| D6-E2 | Eng | bench_l1 routing extension (45 ops + shape_registry) | ⬜ Open |
| D6-E3 | Eng | bench_l2 OT3/OT4 fused benchmark runner | ⬜ Open |
| D6-E4 | Eng | Baseline adaptation (FlashAttn-2, Liger, FlagGems GQA) | ⬜ Open |
| D6-E5 | Eng | CSV output `L1/OT{n}/perf_{op}.csv` | ⬜ Open |
| D6-E6 | Eng | V1 validator extension (attention + quantization tolerance) | ✅ Done (G6 v1) |

**G6 v2 ARCH items — all new, all ⬜ Open:**

| ID | Description | Status |
|:---|:------------|:------:|
| ARCH.1 | Lang Spec v2.0 document | ⬜ Open |
| ARCH.2 | IR Multi-Layer Architecture spec | ⬜ Open |
| ARCH.3 | OpRegistry implementation | ⬜ Open |
| ARCH.4 | Pass Infrastructure skeleton | ⬜ Open |
| ARCH.5 | SemanticInterpreter | ⬜ Open |
| ARCH.6 | SSA Validator | ⬜ Open |
| ARCH.7 | Backend Abstraction protocol | ⬜ Open |
| ARCH.8 | `where` clause MVP (P2-G7, conditional on ARCH.12) | ⬜ Open |
| ARCH.9 | Layer 3/2/1 spec stubs (P2-G7) | ⬜ Open |
| ARCH.10 | Non-regression (≥422 tests) | ⬜ Open |
| ARCH.11 | MLIR framework skeleton | ⬜ Open |
| ARCH.12 | Dynamic Shape feasibility assessment | ⬜ Open |

**Agent items from agent-design.md §6 (G6-scoped):**

| ID | Description | Status |
|:---|:------------|:------:|
| Agent-G6-M2 | Declarative `ToolMeta` + `ArkeTool` ABC (Migration 2) | ⬜ Open |
| Agent-G6-CLI | Structured `--json-log` output + consistent exit codes | ⬜ Open |

---

### Stage 7 (G7): Autonomous Engineering ⬜

> **Core objective:** Validate Arke's Autonomous Engineering Capability. G7 verifies whether the Arke Agent can, **without human intervention**, automatically generate strategies, execute codegen, iterate optimization, and generate a complete kernel set for real LLMs using only kernel semantic descriptions. LLaMA-2 7B and DeepSeek-V2 16B are validation vehicles. Also resolves G5 known-fail (GPT-2 E2E latency ≤1.30×).

#### Exit Criteria

```bash
arke bench --bl 5 --layer l1 l2   # Inherit G6, BL5 all ops no regression
arke bench --bl 6 --model llama2  # LLaMA-2 7B E2E
arke bench --bl 6 --model deepseek # DeepSeek-V2 16B E2E
```

##### Autonomous Engineering Capability (G7-AE — Core)

| ID | Criterion | Verification |
|:---|:----------|:------------|
| G7-AE.1 | LLM auto-generates strategy (no human strategy block) | kernel-only `.ak` → LLM generates strategy → codegen → ≥80% cuBLAS |
| G7-AE.2 | Iterative optimization loop ≥3 rounds | trajectory JSONL contains ≥3 complete `compile→profile→adjust` cycles |
| G7-AE.3 | Multi-input type support | `.ak` file, natural language, existing code snippet → ≥2 ops per type validated E2E |
| G7-AE.4 | `arke optimize <input>` unified entry point | CLI single command: input → LLM optimize → Triton → GPU → benchmark report |
| G7-AE.5 | E2E profile → kernel feedback loop | bottleneck op identification → re-optimize → latency improvement verifiable |

##### BL5 Inheritance (No Regression)

| Dimension | Requirement |
|:----------|:-----------|
| L1 BL5 correctness | ≥ G6 result (no regression) |
| L1 BL5 performance geomean | ≥ G6 result (no regression) |
| L2 BL5 fusion coverage | ≥ G6 fusion combination count |

##### L3 @ BL6 (LLaMA-2 7B + DeepSeek-V2 16B)

| Model | Correctness | Performance | Memory | seq Coverage |
|:------|:-----------|:------------|:-------|:------------|
| **LLaMA-2 7B** | top-1 token 100% matches eager | Arke ≤ **1.30×** eager (torch.compile backend) | ≤ 6GB | 512/2048/4096 |
| **DeepSeek-V2 16B** | top-1 token 100% matches eager (seq∈{512,2048}) | Arke ≤ **1.40×** eager (MoE dispatch overhead) | ≤ 6GB (seq≤512, quantized) | 512/2048 |

> GPT-2 fix: once torch.compile backend is live, GPT-2 latency should simultaneously drop to ≤1.20×, fixing G5 known-fail.

##### G7 PASS Combined Criteria

```
AND ALL:
  [1] Autonomous engineering: G7-AE.1~AE.5 all pass
  [2] BL5 inheritance: L1+L2 correctness and performance ≥ G6 results
  [3] L3 BL6 LLaMA-2: correctness 100% + latency ≤1.30× eager
  [4] L3 BL6 DS-V2: correctness 100% + latency ≤1.40× eager
  [5] torch.compile Inductor backend: GPT-2 latency reduced to ≤1.20×
```

#### Tasks

**Arke LLM Agent (largest G7 group)**

| ID | Description | Priority | Estimate |
|:---|:------------|:--------:|:--------:|
| D7-A1 | Auto strategy generation (kernel-only `.ak` → LLM full strategy pipeline) | P0 | XL |
| D7-A2 | Iterative optimization loop (auto-trigger ≥3 rounds compile→profile→adjust) | P0 | L |
| D7-A3 | Multi-input type routing (`.ak` / natural language / existing code → unified parse) | P0 | L |
| D7-A4 | E2E profile → kernel feedback loop (bottleneck op → re-optimize) | P1 | L |
| D7-A5 | Batch optimize pipeline (full model op set batch optimization) | P1 | M |
| D7-A6 | Long-context agent prompt (seq>4K branch strategy) | P1 | M |
| D7-A7 | MoE-aware optimization prompt (top-k sparsity, load balance) | P1 | M |
| D7-A8 | Quantized inference agent prompt (W4A8, W8A8 strategy) | P2 | M |
| D7-A9 | @rationale knowledge base accumulation (≥30 G7 entries) | P2 | M |

*From agent-design.md §7 (G7 preparation):*

| ID | Description | Priority | Estimate |
|:---|:------------|:--------:|:--------:|
| Agent-G7-M1 | AsyncGenerator optimization loop (Migration 1) | P0 | L |
| Agent-G7-M3 | Segmented prompt cache (Migration 3) | P1 | M |
| Agent-G7-M4 | Context compact (predictive + reactive) (Migration 4) | P1 | M |
| Agent-G7-M5 | Large result delta compression (Migration 5) | P2 | M |
| Agent-G7-M6 | Provider fallback + retry chain (Migration 6) | P1 | M |
| Agent-G7-M7 | Cross-compact ground truth state (Migration 7) | P2 | M |

**Arke IR**

| ID | Description | Priority | Estimate |
|:---|:------------|:--------:|:--------:|
| D7-IR1 | PipelineStageStrategy (prefill/decode separation) | P1 | M |
| D7-IR2 | MultiLatentAttentionIR (kv_lora_rank, qk_rope_head_dim) | P1 | S |
| D7-IR3 | GroupedMatmulSemanticIR expert_indices field | P1 | S |
| D7-IR4 | PaddingStrategy refinement (inherits D6-IR8) | P2 | S |

**Arke Lang**

| ID | Description | Priority | Estimate |
|:---|:------------|:--------:|:--------:|
| D7-L1 | `.ak` `@context_len` annotation primitive | P2 | S |
| D7-L2 | paged memory semantic node (block_table, page_size) | P1 | M |
| D7-L3 | moe_dispatch/combine high-level primitives | P2 | M || D7-L4 | MLA parameter semantic nodes | P2 | S |
| D7-L5 | @dtype int8/fp8 annotation extension | P2 | S |

**Engineering**

| ID | Description | Priority | Estimate |
|:---|:------------|:--------:|:--------:|
| D7-E1 | torch.compile Inductor backend | P0 | XL |
| D7-E2 | LLaMA-2 7B integration + bench_l3 runner | P0 | L |
| D7-E3 | DeepSeek-V2 integration (seq≤512, quantized weights) | P2 | L |
| D7-E4 | Triton MLA template (compressed KV, lora project) | P1 | L |
| D7-E5 | Triton paged_attention template (block table scatter read) | P1 | L |
| D7-E6 | bench runner OOM guard + CSV annotation | P2 | S |
| D7-E7 | bench_l3.py (model forward + top-1 comparison + latency stats) | P0 | M |

---

### Stage 8 (G8): Phase 1 Final ⬜

> **Core objective:** Phase 1 Final Acceptance. Validate Arke's production readiness across 4 real-world LLMs (GPT-2, LLaMA-2, LLaMA-3, Qwen2.5). Verify that Arke Lang/IR/Agent/Compiler stack is stable, complete, and competitive. Freeze v1.0 specs. Produce final evaluation report.

#### Exit Criteria

```bash
arke bench --bl 6 --model gpt2 llama2 llama3 qwen25  # 4 models E2E
```

##### L3 @ BL6 (4 Models)

| Model | Correctness | Performance | Memory | seq Coverage |
|:------|:-----------|:------------|:-------|:------------|
| **GPT-2 Small** | top-1 token 100% | ≤ **1.15×** eager (torch.compile backend) | ≤ 6GB | 128/512/1024 |
| **LLaMA-2 7B** | top-1 token 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 |
| **LLaMA-3 8B** | top-1 token 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 |
| **Qwen2.5 7B** | top-1 token 100% | ≤ **1.30×** eager | ≤ 6GB | 512/2048/4096 |

##### Arke vs LLM-direct (Automated Comparison)

| Dimension | Requirement |
|:----------|:-----------|
| Correctness | Arke ≥ LLM-direct (100% vs ≤90%) |
| Token efficiency | Arke ≤ 0.70× LLM-direct tokens |
| Performance | Arke geomean ≥ 0.90× LLM-direct |

##### Spec Freeze

| Spec | Requirement |
|:-----|:-----------|
| Arke Lang Spec v1.0 | Frozen, tagged, published |
| Arke IR Spec v1.0 | Frozen, tagged, published |
| ir-mlir-mapping.md | Complete, Phase 2-ready |

##### G8 PASS Combined Criteria

```
AND ALL:
  [1] L3 BL6: 4 models all pass correctness + performance + memory
  [2] Arke vs LLM-direct: correctness ≥ direct, tokens ≤ 0.70×, perf ≥ 0.90×
  [3] Spec freeze: Lang v1.0 + IR v1.0 tagged
  [4] Phase 1 final evaluation report published
```

#### Tasks

**Arke Lang**

| ID | Description | Status |
|:---|:------------|:------:|
| D8-L1 | `qwen25_forward.ak` example (GQA+SwiGLU+RMSNorm) | ⬜ Open |
| D8-L2 | `llama3_forward.ak` example (GQA, rope, RMSNorm) | ⬜ Open |
| D8-L3 | `arke-io-spec.md` (I/O contract document) | ⬜ Open |
| D8-L4 | Language Spec v1.0 freeze (document + tag) | ✅ Done |

**Arke IR**

| ID | Description | Status |
|:---|:------------|:------:|
| D8-IR1 | IR Spec v1.0 freeze (document + tag) | ✅ Done |
| D8-IR2 | `ir-mlir-mapping.md` (Phase 2 preparation) | ✅ Done |
| D8-IR3 | `test_ir_roundtrip.py` (all 45 ops × JSON round-trip) | ⬜ Open |

**Arke Agent**

| ID | Description | Status |
|:---|:------------|:------:|
| D8-A1 | `arke optimize` unified entry with full 3-input-type support | ⬜ Open |
| D8-A2 | LLM auto-strategy maturity validation (all 45 ops, no human strategy) | ⬜ Open |
| D8-A3 | iterative loop stable operation across 4 models | ⬜ Open |
| D8-A4 | @rationale knowledge base (≥50 Phase 1 entries) | ⬜ Open |
| D8-A5 | Arke vs LLM-direct automated comparison (`benchmarks/compare_arke_vs_direct.py`) | ⬜ Open |

**Engineering**

| ID | Description | Status |
|:---|:------------|:------:|
| D8-E1 | LLaMA-3 8B integration + bench_l3 runner | ⬜ Open |
| D8-E2 | Qwen2.5 7B integration + bench_l3 runner | ⬜ Open |
| D8-E3 | GPT-2 torch.compile backend E2E (≤1.15× eager, depends on D7-E1) | ⬜ Open |
| D8-E4 | BL5 regression suite (CI): `ci/regression_bl5.py` | ⬜ Open |
| D8-E5 | Language evaluation benchmark + `language-decision.md` | ⬜ Open |
| D8-E6 | Phase 1 final evaluation report `PHASE1_FINAL_REPORT.md` | ⬜ Open |

---

## Phase 2: Arke → Triton/MLIR → Ascend NPU (SIMD Validation)

**Goal:** Verify Arke Lang/IR works on SIMD architecture (Ascend NPU) via Ascend Triton backend. Arke-generated Ascend Triton kernels must outperform FlagGems on Ascend. Simultaneously complete Arke Lang/IR to cover Category B-E operators.

**Hardware target:** Huawei Ascend 910B (SIMD, CANN)  
**Backend:** triton-ascend (Ascend Triton)  
**Benchmark baseline:** FlagGems (Ascend port)

### Stage Structure

| Stage | Milestone | Exit Criteria |
|:-----:|:----------|:--------------|
| **P2-S1** | Ascend Triton environment | matmul runs on 910B, ≥70% CANN cuBLAS |
| **P2-S2** | Cat A+B ops on Ascend | 20 ops correct + ≥0.85× FlagGems |
| **P2-S3** | Cat C+D ops on Ascend | 15 ops correct + ≥0.80× FlagGems |
| **P2-S4** | LLaMA-2 on Ascend | E2E correct + ≤1.40× eager |
| **P2-S_FINAL** | Phase 2 acceptance | H4 validated: same Arke IR → NVIDIA + Ascend |

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

| Stage | Milestone | Exit Criteria |
|:-----:|:----------|:--------------|
| **P3-S1** | MLIR lowering framework | SemanticIR → linalg + transform dialect, matmul correct |
| **P3-S2** | Cat A+B+C via MLIR | 35 ops correct + geomean ≥ Phase 2 Triton |
| **P3-S3** | MLIR performance ≥ Triton | All Cat A+B+C+D MLIR geomean ≥ Phase 2 Triton |
| **P3-S4** | Ascend via MLIR | matmul+rmsnorm correct on Ascend via MLIR; perf ≥ Phase 2 |
| **P3-S5** | LLM Level 2 decisions | StrategyIR L2 (loop nests) → MLIR transform dialect, verified on ≥3 ops |
| **P3-S_FINAL** | Phase 3 acceptance | MLIR path performance ≥ Triton + multi-hardware via MLIR |

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

| Stage | Milestone | Exit Criteria |
|:-----:|:----------|:--------------|
| **P4-S1** | LLVM lowering framework | SemanticIR → LLVM IR, matmul correct |
| **P4-S2** | Cat A-F via LLVM | All 60+ ops correct + geomean ≥ Phase 3 MLIR |
| **P4-S3** | LLVM performance ≥ MLIR | LLVM geomean ≥ MLIR + 5% (Cat A+C+D) |
| **P4-S4** | Multi-hardware LLVM | ≥3 backends ≥90% respective vendor libs |
| **P4-S5** | LLM Level 3 decisions | StrategyIR L3 (instruction-level) → LLVM IR, verified benefit ≥5% |
| **P4-S_FINAL** | v1.0.0 release | @rationale KB ≥200 entries, cross-hardware coverage |

### Key Design Points

- **StrategyIR L3 → LLVM IR**: Instruction-level decisions (e.g., warp shuffle, tensor core intrinsics) map directly to LLVM intrinsics
- **100% ISA coverage**: No abstraction ceiling — full access to PTX/AMDGPU/CANN instruction sets
- **LLM Level 1-3 full stack**: LLM makes decisions at all three StrategyIR layers
- **@rationale knowledge base**: ≥200 cross-hardware optimization patterns

---

## Risk Matrix

| Risk | Affects | Mitigation |
|------|:-------:|-----------|
| LLM decisions don't map to Triton templates | Phase 1 G6-G7 | Expand template coverage + parameter adaptation layer |
| compile_and_profile errors in LLM session | Phase 1 G7 | Better error messages + graceful degradation |
| 6GB VRAM insufficient for large shapes | Phase 1 G5-G8 | Limit shapes to ≤2048; mark OOM as non-blocking |
| Arke doesn't outperform direct Triton | Phase 1 G4 | Gate G4 decision matrix: reliability + token efficiency win |
| API timeout / rate limit | Phase 1 G6-G8 | Retry + fallback + prefer Sonnet over Opus |
| Ascend Triton backend unavailable | Phase 2 | Fallback: validate H4 on AMD via ROCm Triton |
| MLIR learning curve too steep | Phase 3 | Hire MLIR expert consultant; allocate 2× time buffer |
| LLVM IR complexity explosion | Phase 4 | Incremental: start with matmul only, expand gradually |
| torch.compile integration breaks | Phase 3 | Maintain standalone CLI as fallback; Inductor backend is optional |

---

*Last updated: 2026-04-06 (v0.1.1 tag, Phase 1 G0-G5 complete, G6 v2 in progress)*
