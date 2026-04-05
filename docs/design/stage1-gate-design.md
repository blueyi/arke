# Arke Stage 1 — Gate Design

> **Document purpose:** Using the BL/OT/ST/L layered system from `benchmark-design.md` as the sole
> measurement standard, this document defines the exit criteria for all nine Stage 1 Gates (G0-G8),
> and back-derives the required capabilities and development items for each layer from the Gate exit conditions.
>
> 中文版：[stage1-gate-design.zh.md](stage1-gate-design.zh.md)
>
> **Design principles:**
> - Gate exit = a specific BL × L combination, directly verifiable via the `arke bench` command
> - `benchmark-design.md` is the sole measurement Source of Truth; no custom metrics outside the BL system
> - Back-derive from Gate exit → layer capability requirements → concrete development items (ready for task board)
> - G6 = Lang/IR completeness (BL5×L1+L2); G7 = Arke Autonomous Engineering; G8 = Stage 1 final acceptance
>
> **Future reference:** Starting from Stage 2, create `stage2-gate-design.md` etc. in this format.
>
> *Created: 2026-04-05 | benchmark-design.md rev: 2026-04-05*

---

## Table of Contents

1. [BL System Review (Measurement Foundation)](#1-bl-system-review-measurement-foundation)
2. [Gate Summary Table](#2-gate-summary-table)
3. [Part I — G0-G4 (Passed Gates)](#3-part-i--g0-g4-passed-gates)
4. [Part II — G5 (Passed, Standard Retrospective Rewrite)](#4-part-ii--g5-passed-standard-retrospective-rewrite)
5. [G6 — BL5×L1+L2: Lang & IR Completeness (Current Target)](#5-g6--bl5l1l2-lang--ir-completeness-current-target)
6. [G7 — Arke Autonomous Engineering](#6-g7--arke-autonomous-engineering)
7. [G8 — Stage 1 Final Acceptance](#7-g8--stage-1-final-acceptance)
8. [Gate Dependency Chain](#8-gate-dependency-chain)
9. [Development Items Appendix](#9-development-items-appendix)
10. [Mapping to execution-plan.md](#10-mapping-to-execution-planmd)

---

## 1. BL System Review (Measurement Foundation)

### Benchmark Level (BL) Definitions

| Level | Operator Coverage | Shape Coverage | Description | Typical Use |
|:-----:|:-----------------|:--------------|:------------|:------------|
| **BL1** | OT0-OT2 | ST1 | Basic ops × micro shapes | Quick smoke test <30s |
| **BL2** | OT0-OT2 | ST1-ST2 | Basic ops × standard shapes | Daily CI ~5min |
| **BL3** | OT0-OT2 | ST1-ST3 | Basic ops × full shapes (incl. non-aligned) | Gate validation |
| **BL4** | OT0-OT4 | ST1-ST2 | **All ops** × standard shapes | Operator completeness |
| **BL5** | OT0-OT4 | ST1-ST4 | **All ops × all shapes** | Complete benchmark suite |
| **BL6** | Model-Complete | Model-Real | Real model graph: all ops + production shapes | E2E model validation |

**OT × ST Coverage Matrix**

```
              ST1(micro)  ST2(standard)  ST3(stress)  ST4(production)
OT0 (elem)      BL1          BL2            BL3            -
OT1 (reduce)    BL1          BL2            BL3            -
OT2 (dense)     BL1          BL2            BL3           BL5
OT3 (gated)     BL4          BL4            -             BL5
OT4 (attn)       -            -             -             BL5
Model-Complete  ──────────────── BL6 ────────────────────────
```

**Layer × BL Coverage Matrix**

```
         BL1  BL2  BL3  BL4  BL5  BL6
L1        ✓    ✓    ✓    ✓    ✓    -     Single operator perf
L2        -    -    -    ✓    ✓    -     Fused operator perf
L3 ≡ BL6  -    -    -    -    -    ✓     E2E model perf
```

> **L3 ≡ BL6**: L3 is the end-to-end forward inference execution over BL6's model-complete operator+shape set. BL6 defines the coverage scope; L3 defines the measurement method.

### Baseline Priority Tiers

| Tier | Name | Source |
|:----:|:-----|:-------|
| **P0** | Vendor-optimized | cuBLAS, cuDNN, CUTLASS |
| **P1** | Expert Triton | FlagGems, Liger-Kernel, FlashAttention-2 |
| **P2** | Reference Triton | Triton official tutorials |
| **P3** | PyTorch eager | `torch.nn.functional` |
| **P4** | Inductor-generated | `torch.compile` output |
| **P5** | LLM-direct | LLM writes Triton directly |

### Operator Coverage (45 ops, OT0-OT4)

| Tier | Count | Operators |
|:-----|:-----:|:----------|
| **OT0** Elementwise | 12 | `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `add`, `mul`, `where_`, `cast`, `neg`, `exp`, `rsqrt` |
| **OT1** Reduction | 10 | `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max`, `reduce_mean`, `argmax`, `topk`, `cumsum` |
| **OT2** Compute-Dense | 11 | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose`, `concat`, `split`, `gather`, `scatter`, `embedding`, `permute`, `copy_` |
| **OT3** Gated Activation | 7 | `swiglu`, `geglu`, `rope`, `fused_linear_cross_entropy`, `cross_entropy`, `quantize_per_token`, `dequantize_per_channel` |
| **OT4** Attention | 5 | `flash_attention`, `grouped_query_attention`, `multi_latent_attention`, `cross_attention`, `paged_attention` |
| **Total** | **45** | |

---

## 2. Gate Summary Table

| Gate | BL Exit | L Layer | Core Objective | Key Data | Status |
|:----:|:--------|:-------:|:--------------|:---------|:------:|
| **G0** | — | — | GPU environment validation | RTX 3060 6GB, CUDA 12.4, PyTorch 2.6.0, Triton 3.2.0 | ✅ |
| **G1** | — | — | IR + validation system | 10 ops, 6 decision types, 237 tests | ✅ |
| **G2** | BL1×L1 (matmul) | L1 | Manual strategy→codegen→GPU | 105-160% P0 | ✅ |
| **G3** | BL1×L1 (matmul+softmax) | L1 | LLM Agent closed loop | 106.1% P0, 23 tools, 13 decisions | ✅ |
| **G4** | BL2×L1 (6 tasks) | L1 | Arke vs LLM-direct | geomean=0.991, correctness 100% vs 83% | ✅ |
| **G5** | BL3×L1 + BL6/GPT-2×L3 | L1+L3 | All basic ops + E2E correctness | Latency known-fail 1.71-2.20× | ✅ |
| **G6** | BL5×L1+L2 | L1+L2 | **Lang & IR Completeness** | 45 ops all shapes | ⬜ |
| **G7** | BL5 inherited + BL6×L3(LLaMA-2+DS-V2) | L1+L2+L3 | **Arke Autonomous Engineering** | Auto-gen + ≥3 iter rounds + 2 model E2E | ⬜ |
| **G8** | BL6×L3(4 models) + BL5 regression | L1+L2+L3 | Stage 1 final acceptance | 4 models, Arke vs LLM-direct comparison | ⬜ |

---

## 3. Part I — G0-G4 (Passed Gates)

### G0 — GPU Environment Validation ✅

**Core objective:** Establish a reproducible GPU development environment.

#### Exit Criteria

| # | Criterion | Verification | Result |
|---|:----------|:------------|:------:|
| G0.1 | `make setup` completes without errors | Fresh clone → `make setup` | ✅ |
| G0.2 | PyTorch detects CUDA GPU | `torch.cuda.is_available() == True` | ✅ |
| G0.3 | Triton compiles and runs matmul | GPU smoke test exits with code 0 | ✅ |
| G0.4 | `pytest tests/ -q` no import errors | All tests collected successfully | ✅ |

**Hardware record:** RTX 3060 Laptop 6GB (Ampere, SM 8.6) · CUDA 12.4 · PyTorch 2.6.0+cu124 · Triton 3.2.0

**BL equivalent:** Prerequisite, not a BL tier. Hardware foundation required for H1/H2/H3/H4 validation.

---

### G1 — IR + Validation System ✅

**Core objective:** Semantic IR and Strategy IR covering ≥10 operators, with static and numerical validation.

#### Exit Criteria

| # | Criterion | Verification | Result |
|---|:----------|:------------|:------:|
| G1.1 | Semantic IR supports ≥10 operators | `len(OP_CATALOG) >= 10` | ✅ |
| G1.2 | Strategy IR supports ≥6 decision types | `{tile,fuse,place,parallel,reorder,algorithm}` fully covered | ✅ |
| G1.3 | JSON Schema IR round-trip | `jsonschema.validate(ir.to_json(), schema)` passes | ✅ |
| G1.4 | V0 static validation <1ms | Validator latency <1ms | ✅ |
| G1.5 | V1 numerical validation (NumPy reference) | matmul + softmax pass with 3 random seeds | ✅ |
| G1.6 | Shape inference for all 10 ops | `infer_shapes()` returns correct shapes | ✅ |
| G1.7 | ≥100 unit tests pass | pytest count ≥ 100 | ✅ (237) |

**BL equivalent:** IR infrastructure (prerequisite, not a BL tier). Required capability for BL1+.

**Core hypothesis validated:** H3 (explainability) — trajectory JSONL fully records all decisions and rationales.

---

### G2 — Manual Strategy → Codegen → GPU ✅

**Core objective:** Hand-written strategy block → Triton codegen → GPU execution ≥70% cuBLAS.

#### Exit Criteria

| # | Criterion | Verification | Result |
|---|:----------|:------------|:------:|
| G2.1 | matmul Triton codegen correct | Generated kernel passes V1 numerical validation | ✅ |
| G2.2 | softmax Triton codegen correct | Same as above | ✅ |
| G2.3 | fused matmul+relu codegen correct | Fused kernel passes numerical validation | ✅ |
| G2.4 | GPU execution **≥70% cuBLAS** | `compile_and_profile()` → `vs_baseline >= 0.7` | ✅ (105-160%) |
| G2.5 | Full pipeline end-to-end | IR → strategy → codegen → compile → profile in one call | ✅ |
| G2.6 | ≥9 GPU integration tests | GPU tests pass under `ARKE_GPU_TESTS=1` | ✅ |

**BL equivalent:** BL1×L1(matmul only). Validates H1 (structured protocol improves correctness).

---

### G3 — LLM Agent Closed Loop ✅

**Core objective:** LLM autonomously completes the optimization loop via tool calls, zero human intervention.

#### Exit Criteria

| # | Criterion | Result |
|---|:----------|:------:|
| G3.1 | LLM uses ≥8 distinct tools | ✅ (all 10) |
| G3.2 | LLM applies ≥4 strategy decisions | ✅ (13 times) |
| G3.3 | LLM calls verify_correctness | ✅ |
| G3.4 | LLM calls compile_and_profile | ✅ (5 times) |
| G3.5 | LLM uses checkpoint + rollback | ✅ |
| G3.6 | Fallback mechanism works | ✅ |
| G3.7 | Multi-provider support | ✅ (Anthropic + OpenAI-compatible) |
| G3.8 | Zero human intervention | ✅ |

**Key data:** matmul 2048² → 151.4% cuBLAS; lm-head (50257) → 116.5% cuBLAS; 23 tool calls, 13 strategy decisions.

**BL equivalent:** BL1×L1(LLM-driven). Validates H2 (structured search superior to manual).

---

### G4 — Arke vs LLM-direct ✅

**Core objective:** Quantitatively prove that Arke outperforms LLM-direct Triton in correctness, consistency, and performance.

#### Exit Criteria

| # | Criterion | Result |
|---|:----------|:------:|
| G4.1 | ≥5 benchmark tasks defined | ✅ (6 tasks) |
| G4.2 | Arke completes all tasks | ✅ 6/6 |
| G4.3 | LLM-direct completes all tasks | ✅ 6/6 |
| G4.4 | Arke correctness ≥ LLM-direct | ✅ 100% ≥ 83% |
| G4.5 | Arke mean performance ≥ LLM-direct | ⚠️ 115.7% < 118.3% (fusion task gap, variance significantly smaller) |
| G4.6 | Arke variance ≤ LLM-direct | ✅ (LLM-direct has high failure rate, large variance) |
| G4.7 | Evaluation report generated | ✅ `benchmarks/results/stage1/EVALUATION_REPORT.md` |
| G4.8 | Token efficiency comparison | ✅ Arke ≤60% LLM-direct token consumption |

**Key data:** Arke/FlagGems geomean=0.991; correctness 100% vs 83%; token efficiency ≤0.7×.

**BL equivalent:** BL2×L1(6 tasks). Validates H1+H2.

**Gate decision:** Proceed — Arke wins on reliability (100% vs 83%) and token efficiency; performance slightly lower but variance significantly smaller.

---

## 4. Part II — G5 (Passed, Standard Retrospective Rewrite)

### G5 — BL3×L1 + BL6/GPT-2×L3 ✅

**Core objective:** All basic operators (OT0-2) non-aligned shape correctness + GPT-2 Small E2E inference validation.

#### Exit Criteria (BL System Rewrite)

```bash
arke bench --bl 3 --ot 0-2          # OT0-2 × ST1-3, basic ops all shapes
arke bench --bl 6 --model gpt2      # GPT-2 Small E2E
```

**L1 @ BL3 (OT0-2, ST1-3)**

| Dimension | Requirement | Actual Result |
|:----------|:------------|:-------------|
| Correctness | OT0-2 × ST1-3 100% | ✅ All pass |
| Performance | geomean(OT0-2, ST1-3) ≥ P3 (torch eager) | ✅ Achieved |
| Note | P0/P1 performance targets deferred to G6 (limited by dispatch architecture) | — |

**L3 @ BL6 / GPT-2 Small**

| Dimension | Requirement | Actual Result |
|:----------|:------------|:-------------|
| Correctness | top-1 token matches eager for all seq_len | ✅ |
| Coverage | ≥48 Conv1D replacements | ✅ 49/48 |
| Memory | ≤ 6GB VRAM | ✅ 1100MB/6144MB |
| Latency | ≤ 1.15× eager | ⚠️ 1.71-2.20× **known-fail** |

#### G5 Known-Fail Analysis (Recorded, Non-blocking)

| Symptom | Root Cause | Resolution Timing |
|:--------|:-----------|:-----------------|
| E2E latency 1.7-2.3× eager | monkey-patch dispatch ~60µs/call × 49 calls accumulated | G7: torch.compile Inductor backend |
| Single matmul: Arke 76µs vs cuBLAS 44µs | L1 single op OK, Python dispatch overhead accumulates | G6: unified BL5 measurement for comparison |

> Detailed analysis report: `benchmarks/results/stage1/gates/G5/REPORT.md`

**BL equivalent:** BL3×L1 (OT0-2, 33 ops correctness) + BL6/GPT-2×L3 (E2E correctness).

**Core hypothesis validated:** H1 correctness validation (complete E2E pipeline correct).

---

## 5. G6 — BL5×L1+L2: Lang & IR Completeness (Current Target)

> **Core objective:** Verify that Arke Lang and Arke IR have complete expressibility, code generation capability,
> and performance competitiveness for all 45 operators across all shapes (including ST3 non-aligned + ST4 production scale).
> This is the watershed between Arke being "runnable" and "usable". **G6 does not include L3/BL6 (model E2E) — that is G7's responsibility.**

### Exit Criteria (BL5×L1+L2)

```bash
arke bench --bl 5 --layer l1    # OT0-4 × ST1-4, single op all shapes performance
arke bench --bl 5 --layer l2    # Fused op all shapes performance
```

#### L1 @ BL5 (OT0-4, ST1-4)

| Op Group | Correctness Requirement | Performance Requirement | Measurement Command |
|:---------|:------------------------|:------------------------|:--------------------|
| **OT0** Elementwise | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.90 P1 (FlagGems elem) | `bench_l1 --ot 0` |
| **OT1** Reduction | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.85 P1 (FlagGems norm/softmax) | `bench_l1 --ot 1` |
| **OT2** Compute-Dense | 100%(ST1-3) + ≥95%(ST4) | matmul geomean ≥ 0.90 P0; others ≥ P3 | `bench_l1 --ot 2` |
| **OT3** Gated Activation | 100%(ST1-3) + ≥95%(ST4) | swiglu/rope geomean ≥ 0.85 P1 (Liger/FlagGems) | `bench_l1 --ot 3` |
| **OT4** Attention | 100%(ST1-4, excl. OOM) | FA geomean ≥ 0.80 P1 (FlashAttn-2); GQA ≥ 0.80 | `bench_l1 --ot 4` |

> **ST4 OOM note:** OT4 may OOM on some large shapes with 6GB VRAM; mark `⚠️ OOM` and skip, not counted in correctness pass rate denominator.

#### L2 @ BL5 (Fused Operators)

| Fusion Combination | Requirement | Baseline |
|:-------------------|:------------|:---------|
| matmul+relu, matmul+gelu | ≥ 1.05× unfused (fusion benefit verifiable) | P3 unfused |
| swiglu, geglu | ≥ 0.90× Liger | P1 |
| linear+cross_entropy | ≥ 1.05× unfused | P3 |
| QKV+flash_attention | ≥ 0.80× FlashAttn-2 | P1 |

#### Lang & IR Completeness Additional Criteria (G6-LI)

| ID | Criterion | Verification |
|:---|:----------|:------------|
| **G6-LI.1** | All 45 ops expressible and parseable in `.ak` | `arke parse examples/<op>.ak` all exit 0 |
| **G6-LI.2** | `.ak → SemanticIR → StrategyIR` full pipeline | `ast_to_strategy()` passes round-trip for all example files |
| **G6-LI.3** | `@rationale` annotations preserved through full pipeline | `.ak @rationale → StrategyIR → codegen comments → trajectory/log` ≥3 examples verified |
| **G6-LI.4** | Token efficiency: `.ak` ≤ equivalent Triton line count | OT0-OT4 benchmark: `.ak` lines < Triton lines @ equivalent performance |
| **G6-LI.5** | Python interop IR round-trip | `from_json/to_json/from_dict/to_dict` passes for all 45 ops |
| **G6-LI.6** | Grammar completeness: all `.ak` files parse with 0 failures | Supports array literals, float constants, 4D tensor, all op param types |

#### G6 PASS Combined Criteria

```
AND ALL:
  [1] L1 BL5 correctness: 100%(ST1-3) + ≥95%(ST4, excl. OOM) for all OT0-OT4
  [2] L1 BL5 performance weighted_score ≥ 0.83
        weighted_score = 0.25×score(OT0-1) + 0.30×score(OT2) + 0.20×score(OT3) + 0.25×score(OT4)
        where score(OTn) = geomean pass rate for that OT group (0.0~1.0)
  [3] L2 BL5: ≥3/4 fusion combinations pass
  [4] Lang&IR: G6-LI.1~LI.6 all pass
```

### G6 Capability Back-Derivation

#### Arke Lang (.ak Language Layer)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D6-L1 | `.ak` 4D tensor op syntax extension | OT4 must be expressible | `.ak` syntax: 4D tensor, einsum annotation |
| D6-L2 | gather/scatter semantic nodes | OT2 new data-movement ops | gather/scatter semantic nodes |
| D6-L3 | quantize primitive syntax | OT3 quantization ops | `quantize`/`dequantize` syntax primitives |
| D6-L4 | paged KV / block_table parameter stub | OT4 paged_attention | paged memory semantic annotation (can defer to G7) |
| D6-L5 | grammar fix: array literal + float constant | G1.4 legacy syntax gap | Fix grammar, support `[2,3]`, `0.125`, etc. |
| D6-L6 | `.ak` example files for all 45 ops | G6-LI.1 completeness | `examples/<op>.ak` × 45 |

#### Arke IR (Semantic IR + Strategy IR)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D6-IR1 | SemanticIR op catalog extended to 45 ops | BL5 all-op correctness | OT3/OT4 all op node definitions |
| D6-IR2 | AttentionSemanticIR fields | GQA/MHA correctness | `mask_type`, `num_kv_heads`, `head_dim` |
| D6-IR3 | RopeSemanticIR fields | RoPE correctness | `theta`, `base`, `rotary_dim` |
| D6-IR4 | QuantizeSemanticIR fields | quantize correctness | `scale_dtype`, `group_size`, `zero_point` |
| D6-IR5 | `ast_to_strategy()` converter | G6-LI.2: full pipeline | parser AST → StrategyIR converter |
| D6-IR6 | StrategyIR JSON round-trip (all 45 ops) | G6-LI.5 Python interop | `from_json/to_json` full test coverage |
| D6-IR7 | MLA-specific fields | BL5 OT4 MLA correctness | `latent_dim`, `kv_lora_rank` fields |
| D6-IR8 | PaddingStrategy decision type | ST3 non-aligned performance | `pad_to_multiple`, `dynamic_padding` decisions |

#### Arke LLM Agent

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D6-A1 | attention prompt template | OT4 ≥0.80 P1 | causal mask, GQA group expansion strategy template |
| D6-A2 | rope prompt + rationale template | OT3 rope ≥ P3 | RoPE vectorized cos/sin, half-rotate strategy |
| D6-A3 | fusion opportunity detection | L2 fusion benefit | detect fuse opportunities in agent tools |
| D6-A4 | quantize/dequantize prompt template | OT3 quant ≥ P3 | per-token scale, vectorized quantization strategy |
| D6-A5 | batch optimize pipeline (45 ops parallel sessions) | BL5 cannot do manually one-by-one | op_list × shape_list parallel agent sessions |
| D6-A6 | non-aligned shape rationale template | ST3 performance target | padding vs masking trade-off rationale |

#### Arke Engineering (Infrastructure)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D6-E1 | 10 Triton template classes | BL5 correctness 100% | `rope`, `flash_attention`, `GQA`, `MLA`, `cross_attention`, `paged_attention`, `gather`, `scatter`, `embedding`, `quantize` |
| D6-E2 | `bench_l1` routing extension (45 ops + shape_registry) | BL5 executable | `bench_l1.py` routing extension + shape_registry integration |
| D6-E3 | `bench_l2` OT3/OT4 fused benchmark runner | L2 BL5 executable | bench_l2.py OT3/OT4 fused benchmark |
| D6-E4 | Baseline adaptation | OT4 P1 baselines | FlashAttn-2, Liger rope/quant, FlagGems GQA |
| D6-E5 | CSV output organized by BL/OT/L hierarchy | benchmark results traceable | `L1/OT{n}/perf_{op}.csv` output directory |
| D6-E6 | V1 validator extension (all 45 ops) | correctness 100% | attention numerical tolerance, quantization precision standards |

### G6 Critical Path

**Bottleneck:** D6-E1 (10 Triton template classes) is G6's largest workload; `flash_attention` and `MLA` templates are most complex.

```
D6-L5(grammar fix) + D6-IR1(catalog) → D6-IR5(ast_to_strategy) → G6-LI.1/2
D6-E1(10 Triton templates) + D6-E6(validator) → [1] correctness
D6-E2/E3/E4(bench runners) + D6-A1/A2/A5 → [2][3] performance
ALL → G6 PASS
```

---

## 6. G7 — Arke Autonomous Engineering

> **Core objective:** Validate Arke's **Autonomous Engineering Capability**.
> G7 is not merely "add two more model E2E tests" — it verifies whether the Arke Agent can,
> **without human intervention**, automatically generate strategies, execute codegen, iterate optimization,
> and ultimately generate a complete kernel set for real LLMs using only kernel semantic descriptions (`.ak` or natural language).
>
> **LLaMA-2 7B and DeepSeek-V2 16B are validation vehicles**, not core objectives in themselves.
> The core to validate is: whether the Arke Agent has reached production-grade autonomous engineering capability.

### Four Core Validation Dimensions of G7

1. **Autonomous kernel generation** — LLM Agent automatically generates strategies from kernel semantic descriptions alone, without human strategy blocks
2. **Iterative optimization loop** — LLM automatically executes ≥3 rounds of `compile → profile → adjust`
3. **Multi-input format support** — `.ak` files / natural language descriptions / existing code snippets → auto-routed to Arke pipeline
4. **E2E model validation** — LLaMA-2 7B + DS-V2 16B prove Arke can autonomously generate complete kernel sets for real models

Additionally, G7 resolves the G5 known-fail: reduces GPT-2 E2E latency to ≤1.30× via `torch.compile` Inductor backend.

### Exit Criteria

```bash
arke bench --bl 5 --layer l1 l2   # Inherit G6, BL5 all ops no regression
arke bench --bl 6 --model llama2  # LLaMA-2 7B E2E (validate autonomous generation)
arke bench --bl 6 --model deepseek # DeepSeek-V2 16B E2E (validate autonomous generation)
```

#### Autonomous Engineering Capability Validation Criteria (G7-AE, Core)

| ID | Criterion | Verification |
|:---|:----------|:------------|
| **G7-AE.1** | LLM auto-generates strategy (no human strategy block) | kernel-only `.ak` (no strategy block) → LLM generates strategy → codegen → ≥80% cuBLAS |
| **G7-AE.2** | Iterative optimization loop ≥3 rounds | trajectory JSONL contains ≥3 complete `compile→profile→adjust` cycles |
| **G7-AE.3** | Multi-input type support | (a) `.ak` file (b) natural language description (c) existing code snippet → ≥2 ops per type validated end-to-end |
| **G7-AE.4** | `arke optimize <input>` unified entry point | CLI single command completes input → LLM optimize → Triton → GPU → benchmark report |
| **G7-AE.5** | E2E profile → kernel feedback loop | bottleneck op identification → re-optimize → latency improvement verifiable (trajectory recorded) |

#### BL5 Inheritance Criteria (No Regression)

```bash
arke bench --bl 5 --layer l1    # ≥ G6 standard (correctness + performance no regression)
arke bench --bl 5 --layer l2    # ≥ G6 standard
```

| Dimension | Requirement |
|:----------|:-----------|
| L1 BL5 correctness | ≥ G6 result (no regression) |
| L1 BL5 performance geomean | ≥ G6 result (no regression) |
| L2 BL5 fusion coverage | ≥ G6 fusion combination count |

#### L3 @ BL6 (LLaMA-2 7B + DeepSeek-V2 16B) — Autonomous Generation Validation Vehicles

| Model | Correctness | Performance Threshold | Memory | seq Coverage |
|:------|:-----------|:---------------------|:-------|:------------|
| **LLaMA-2 7B** | top-1 token 100% matches eager (all seq_len) | Arke ≤ **1.30×** eager (torch.compile backend) | ≤ 6GB | 512/2048/4096 |
| **DeepSeek-V2 16B** | top-1 token 100% matches eager (seq∈{512,2048}) | Arke ≤ **1.40×** eager (MoE dispatch overhead) | ≤ 6GB (seq≤512, quantized) | 512/2048 |

> **GPT-2 E2E fix:** Once torch.compile backend is live, GPT-2 latency should simultaneously drop to ≤1.20×, fixing G5 known-fail.

#### G7 PASS Combined Criteria

```
AND ALL:
  [1] Autonomous engineering: G7-AE.1~AE.5 all pass
  [2] BL5 inheritance: L1+L2 BL5 correctness and performance both no lower than G6 results
  [3] L3 BL6 LLaMA-2: correctness 100% + latency ≤1.30× eager
  [4] L3 BL6 DS-V2: correctness 100% + latency ≤1.40× eager
  [5] torch.compile Inductor backend: GPT-2 latency reduced to ≤1.20× (G5 known-fail fix)
```

### G7 Capability Back-Derivation

#### ★ Arke LLM Agent (Largest G7 Development Item Group)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D7-A1 | Auto strategy generation (kernel-only .ak → LLM full strategy pipeline) | G7-AE.1 | kernel-only `.ak` input → LLM complete strategy generation pipeline |
| D7-A2 | Iterative optimization loop (auto-trigger ≥3 rounds compile→profile→adjust) | G7-AE.2 | Auto-trigger ≥3 iterations (based on performance delta threshold) |
| D7-A3 | Multi-input type routing (arke optimize unified entry) | G7-AE.3/4 | `.ak` / natural language / existing code → unified parse → Arke pipeline |
| D7-A4 | E2E profile → kernel feedback loop | G7-AE.5 | Bottleneck op identification (based on BL6 profile) → re-trigger optimization |
| D7-A5 | Batch optimize pipeline (full model op set batch optimization) | L3 BL6 autonomous generation | model forward graph → op list → batch agent sessions |
| D7-A6 | Long-context agent prompt (seq>4K branch strategy) | L3 BL6 LLaMA-2/DS-V2 latency thresholds | chunk prefill, KV cache split strategy templates |
| D7-A7 | MoE-aware optimization prompt | DS-V2 E2E (grouped_matmul+gather+scatter) | top-k sparsity, load balance strategy templates |
| D7-A8 | INT8/quantized inference agent prompt | L2 BL5 quant+matmul fusion | W4A8, W8A8 quantization path strategy templates |
| D7-A9 | @rationale knowledge base accumulation (≥30 G7 entries) | H3 explainability | trajectory → rationale_kb.jsonl distillation |

#### Arke IR

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D7-IR1 | PipelineStageStrategy decision type | OT4 ST4 performance (long context needs pipelining) | prefill/decode separation strategy |
| D7-IR2 | MultiLatentAttentionIR fields | BL5 MLA correctness + performance | `kv_lora_rank`, `qk_rope_head_dim` |
| D7-IR3 | GroupedMatmulSemanticIR expert_indices | DS-V2 MoE correctness | expert routing fields |
| D7-IR4 | PaddingStrategy (non-aligned shapes) | ST3 performance (inherits G6 D6-IR8) | `pad_to_multiple`, `dynamic_padding` |

#### Arke Lang (.ak Language Layer)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D7-L1 | `.ak` @context_len annotation primitive | ST4 OT4 + L3 BL6 | long-context semantic annotation |
| D7-L2 | paged memory semantic node (block_table, page_size) | BL5 OT4 paged_attention | paged KV cache complete semantics |
| D7-L3 | moe_dispatch / moe_combine high-level primitives | DS-V2 E2E | MoE dispatch/combine syntactic sugar |
| D7-L4 | MLA parameter semantic nodes | BL5 OT4 MLA | `latent_dim`, `kv_lora_rank` annotations |
| D7-L5 | @dtype int8/fp8 annotation extension | BL5 OT3 quant × ST4 | int8/fp8 dtype primitive extension |

#### Arke Engineering (Infrastructure)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D7-E1 | torch.compile Inductor backend | G7-AE latency thresholds + G5 known-fail fix | `arke/integration/torch_compile_backend.py` (Inductor custom op registration) |
| D7-E2 | LLaMA-2 7B integration + bench_l3 runner | L3 BL6 LLaMA-2 | `examples/llama2_arke.py` + L3 bench runner |
| D7-E3 | DeepSeek-V2 integration + bench_l3 runner | L3 BL6 DS-V2 | `examples/deepseek_v2_arke.py` (quantized weights, seq≤512) |
| D7-E4 | Triton MLA template (compressed KV, lora project) | BL5 OT4 MLA correctness | lora-style project + compressed KV |
| D7-E5 | Triton paged_attention template | BL5 OT4 paged_attention | block table scatter read |
| D7-E6 | bench runner OOM guard + CSV annotation | BL5 ST4 OOM handling | try/catch + `⚠️ OOM` written to CSV |
| D7-E7 | `bench_l3.py` — model forward pass multi-measurement | L3 automation | top-1 token comparison + latency statistics |

### G7 Critical Path

**Longest dependency chain:**
```
D7-A1(auto strategy gen) ────────────────────┐
D7-A2(iterative loop) ────────────────────────┤→ G7-AE.1~5
D7-A3(multi-input routing) + D7-E1(compile backend)┘
D7-E1(torch.compile backend) → D7-E2(LLaMA-2) → D7-E7(bench_l3) → [3][5]
D7-E3(DS-V2) → [4]
```

**Bottleneck:** D7-A1 (auto strategy generation) is G7's largest uncertainty; D7-E1 (torch.compile backend) is the architectural prerequisite for L3 performance.

---

## 7. G8 — Stage 1 Final Acceptance

> **Core objective:** Stage 1 final Gate. Arke can autonomously generate complete kernel sets for 4 real LLMs,
> all end-to-end performance meets production-ready thresholds, and quantitatively validates Arke vs LLM-direct advantages.
> Simultaneously completes language implementation assessment (Python vs hybrid approach) and lays the foundation for Stage 2.

### Exit Criteria

```bash
arke bench --bl 6 --model gpt2      # GPT-2 Small
arke bench --bl 6 --model llama2    # LLaMA-2 7B
arke bench --bl 6 --model llama3    # LLaMA-3 8B
arke bench --bl 6 --model qwen25    # Qwen2.5 7B
arke bench --bl 5 --layer l1 l2     # BL5 regression (no regression)
```

#### L3 @ BL6 (4 Models)

| Model | Correctness | Performance Threshold | Memory | seq Coverage |
|:------|:-----------|:---------------------|:-------|:------------|
| **GPT-2 Small** | top-1 100% | Arke ≤ **1.15×** eager (G5 known-fail fully fixed) | ≤ 4GB | 128/512/1024 |
| **LLaMA-2 7B** | top-1 100% | Arke ≤ **1.20×** eager | ≤ 6GB | 512/2048/4096 |
| **LLaMA-3 8B** | top-1 100% | Arke ≤ **1.20×** eager | ≤ 6GB | 512/2048/8192 |
| **Qwen2.5 7B** | top-1 100% | Arke ≤ **1.25×** eager (GQA 7:1 + very wide FFN) | ≤ 6GB | 512/2048 |

#### BL5 Full Regression (Inherits G7 Results, Must Not Regress)

| Dimension | Requirement |
|:----------|:-----------|
| L1 BL5 all 45 ops correctness | ≥ G7 standard, no regression |
| L1 BL5 OT0-4 performance geomean | ≥ G7 result (±1% noise allowed) |
| L2 BL5 fused op coverage | ≥ G7 coverage |

#### Arke vs LLM-direct Comparison (G8 New Addition)

| Metric | Arke Target | LLM-direct Baseline | Basis |
|:-------|:-----------|:--------------------|:------|
| Correctness | ≥ 98% (G8 all ops) | Historical ~83% | G4 data |
| Performance geomean (BL5 L1) | ≥ 1.05× LLM-direct | — | Arke structured search advantage |
| Performance variance (stddev) | ≤ 0.5× LLM-direct | — | Deterministic IR constraints reduce variance |
| Token consumption/kernel | ≤ 0.7× LLM-direct | — | IR constraints reduce exploration tokens |

#### Language Implementation Assessment (G8 Concurrent)

```
G8-Lang: Python vs hybrid approach data-driven evaluation
  Measurements: dispatch overhead (Python path vs Rust/C++ theoretical)
                parse latency (.ak → IR), memory footprint, LLM API integration cost
  Output: docs/design/language-decision.md (conclusion + data + Stage 2 migration strategy)
```

#### G8 PASS Combined Criteria

```
AND ALL:
  [1] 4 model L3 BL6 correctness 100%
  [2] 4 model E2E latency all ≤ threshold (GPT-2≤1.15×, LLaMA-2/3≤1.20×, Qwen2.5≤1.25×)
  [3] Arke vs LLM-direct: correctness ≥1.15×, performance geomean ≥1.05×, token ≤0.7×
  [4] BL5 L1 all 45 ops no performance regression
  [5] language-decision.md complete
```

### G8 Capability Back-Derivation

#### Arke Lang (.ak Language Layer)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D8-L1 | Qwen2.5 7B `.ak` example | BL6 Qwen2.5 L3 | `qwen25_forward.ak` (GQA+SwiGLU+RMSNorm complete description) |
| D8-L2 | LLaMA-3 8B `.ak` example | BL6 LLaMA-3 L3 | `llama3_forward.ak` (GQA, rope, RMSNorm) |
| D8-L3 | Arke I/O Spec document | G7-AE.3 multi-input types | `docs/spec/arke-io-spec.md` |
| D8-L4 | Language Spec v1.0 freeze | Stage 1 completion milestone | `arke-lang-spec.md` update + tag v1.0 |

#### Arke IR

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D8-IR1 | IR Spec v1.0 freeze | Stage 1 completion milestone | `arke-ir-spec.md` update + tag v1.0 |
| D8-IR2 | IR ↔ MLIR mapping document | Stage 2 preparation | `docs/spec/ir-mlir-mapping.md` |
| D8-IR3 | Complete round-trip validation (all 45 ops × JSON) | IR Spec v1.0 | `test_ir_roundtrip.py` |

#### Arke LLM Agent

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D8-A1 | Autonomous I/O contract (3 input types → Arke pipeline) | G7-AE.3/4 inherited | `arke optimize <input>` full support |
| D8-A2 | LLM auto-strategy maturity validation | G8 BL5 no regression | All 45 ops batch agent validation (no human strategy) |
| D8-A3 | ≥3 iterative optimization rounds (maturity) | G7-AE.2 inherited | iterative loop stable across 4 models |
| D8-A4 | @rationale knowledge base (≥50 Stage 1 entries) | H3 explainability | `trajectory → rationale_kb.jsonl` distillation |
| D8-A5 | Arke vs LLM-direct automated comparison | G8 comparison metrics | `benchmarks/compare_arke_vs_direct.py` |

#### Arke Engineering (Infrastructure)

| ID | Capability Requirement | Basis | Development Item |
|:---|:-----------------------|:------|:----------------|
| D8-E1 | LLaMA-3 8B integration + bench_l3 | BL6 LLaMA-3 L3 | `examples/llama3_arke.py` + bench_l3 runner |
| D8-E2 | Qwen2.5 7B integration + bench_l3 | BL6 Qwen2.5 L3 | `examples/qwen25_arke.py` + bench_l3 runner |
| D8-E3 | GPT-2 ≤1.15× eager (fully fixed) | G5 known-fail final resolution | torch.compile backend E2E integration (depends on D7-E1) |
| D8-E4 | BL5 full 45 ops regression suite (CI) | G8 BL5 no regression | `ci/regression_bl5.py` (runs BL5 L1 correctness each commit) |
| D8-E5 | Language evaluation benchmark | language-decision.md | dispatch overhead measurement script + memory profiler |
| D8-E6 | Stage 1 final evaluation report | Stage 1 completion milestone | `benchmarks/results/stage1/STAGE1_FINAL_REPORT.md` |

---

## 8. Gate Dependency Chain

```
G0(env) → G1(IR) → G2(Codegen) → G3(Agent) → G4(comparison) → G5(BL3+GPT-2)
                                                                       │
                                               G6(BL5 Lang/IR Completeness, current target)
                                                                       │
                                          G7(Arke Autonomous Engineering
                                               BL5 inherit + LLaMA-2/DS-V2 E2E)
                                                                       │
                                               G8(BL6×4 models, Stage 1 Final)
                                                                       │
                                                          Stage 2 (Ascend)
```

### Critical Blockers per Gate

| Gate | Critical Blockers (P0) | Estimate |
|:-----|:-----------------------|:---------|
| **G6** | D6-E1 (10 Triton templates); D6-IR1 (catalog 45 ops); D6-L5 (grammar fix) | XL+M+S |
| **G7** | D7-A1 (auto strategy generation); D7-E1 (torch.compile backend); D7-E2 (LLaMA-2 integration) | XL+XL+L |
| **G8** | D8-E1/E2 (LLaMA-3/Qwen2.5 integration); D8-E3 (GPT-2 fix depends on D7-E1) | L+L+M |

### Development Path Critical Chain

```
D6-IR1 → D6-E1(10 templates) → D6-E2(bench_l1) ───────────────── G6
                                       │
                       D7-A1(auto strategy) + D7-A2(iterative loop) ───────┐
                       D7-E1(torch.compile backend) ────────────────────────┤→ G7
                       D7-E2(LLaMA-2) + D7-E7(bench_l3) ───────────────────┘
                                       │
                       D8-E1(LLaMA-3) + D8-E2(Qwen2.5) ──────── G8
```

---

## 9. Development Items Appendix

> **Estimate legend:** S≤1d, M≤3d, L≤1w, XL>1w (single-person reference)

### G6 Development Items (26 items)

| ID | Layer | Description | Priority | Estimate |
|:---|:------|:------------|:--------:|:--------:|
| D6-L1 | Lang | `.ak` 4D tensor syntax extension (4D tensor, einsum annotation) | P2 | M |
| D6-L2 | Lang | gather/scatter semantic nodes | P2 | S |
| D6-L3 | Lang | quantize primitive syntax | P2 | S |
| D6-L4 | Lang | paged memory semantic annotation (stub, can defer to G7) | P2 | S |
| D6-L5 | Lang | grammar fix (array literal, float constant) | **P0** | S |
| D6-L6 | Lang | `.ak` example files for all 45 ops | P1 | L |
| D6-IR1 | IR | SemanticIR op catalog → 45 ops (OT3/OT4 all fields) | **P0** | M |
| D6-IR2 | IR | AttentionSemanticIR (mask_type, num_kv_heads, head_dim) | **P0** | S |
| D6-IR3 | IR | RopeSemanticIR (theta, base, rotary_dim) | **P0** | S |
| D6-IR4 | IR | QuantizeSemanticIR (scale_dtype, group_size, zero_point) | **P0** | S |
| D6-IR5 | IR | `ast_to_strategy()` converter | **P0** | M |
| D6-IR6 | IR | StrategyIR JSON round-trip (all 45 ops) | P1 | S |
| D6-IR7 | IR | MLA-specific fields (latent_dim, kv_lora_rank) | P1 | S |
| D6-IR8 | IR | PaddingStrategy decision type (pad_to_multiple, dynamic_padding) | P2 | S |
| D6-A1 | Agent | attention prompt template (causal mask, GQA group expansion) | P1 | M |
| D6-A2 | Agent | rope prompt + rationale template | P1 | S |
| D6-A3 | Agent | fusion opportunity detection | P1 | M |
| D6-A4 | Agent | quantize/dequantize prompt template | P2 | S |
| D6-A5 | Agent | batch optimize pipeline (45 ops parallel sessions) | P1 | L |
| D6-A6 | Agent | non-aligned shape rationale template | P2 | S |
| D6-E1 | Eng | 10 Triton template classes: rope/FA/GQA/MLA/cross_attn/paged_attn/gather/scatter/embedding/quantize | **P0** | XL |
| D6-E2 | Eng | bench_l1 routing extension (45 ops + shape_registry integration) | P1 | M |
| D6-E3 | Eng | bench_l2 OT3/OT4 fused benchmark runner | P1 | M |
| D6-E4 | Eng | baseline adaptation (FlashAttn-2, Liger rope/quant, FlagGems GQA) | P1 | M |
| D6-E5 | Eng | CSV output directory L1/OT{n}/perf_{op}.csv | P2 | S |
| D6-E6 | Eng | V1 validator extension (attention numerical tolerance, quantization precision standards) | **P0** | S |

### G7 Development Items (25 items)

| ID | Layer | Description | Priority | Estimate |
|:---|:------|:------------|:--------:|:--------:|
| D7-A1 | Agent | Auto strategy generation (kernel-only .ak → LLM full strategy pipeline) | **P0** | XL |
| D7-A2 | Agent | Iterative optimization loop (auto-trigger ≥3 rounds compile→profile→adjust) | **P0** | L |
| D7-A3 | Agent | Multi-input type routing (.ak / natural language / existing code → Arke pipeline) | **P0** | L |
| D7-A4 | Agent | E2E profile → kernel feedback loop (bottleneck op identification → re-optimize) | P1 | L |
| D7-A5 | Agent | Batch optimize pipeline (full model op set batch optimization) | P1 | M |
| D7-A6 | Agent | Long-context agent prompt (seq>4K branch strategy) | P1 | M |
| D7-A7 | Agent | MoE-aware optimization prompt (top-k sparsity, load balance) | P1 | M |
| D7-A8 | Agent | Quantized inference agent prompt (W4A8, W8A8 strategy) | P2 | M |
| D7-A9 | Agent | @rationale knowledge base accumulation (≥30 G7 entries) | P2 | M |
| D7-IR1 | IR | PipelineStageStrategy (prefill/decode separation) | P1 | M |
| D7-IR2 | IR | MultiLatentAttentionIR (kv_lora_rank, qk_rope_head_dim) | P1 | S |
| D7-IR3 | IR | GroupedMatmulSemanticIR expert_indices field | P1 | S |
| D7-IR4 | IR | PaddingStrategy refinement (inherits D6-IR8) | P2 | S |
| D7-L1 | Lang | `.ak` @context_len annotation primitive | P2 | S |
| D7-L2 | Lang | paged memory semantic node (block_table, page_size) | P1 | M |
| D7-L3 | Lang | moe_dispatch/combine high-level primitives | P2 | M |
| D7-L4 | Lang | MLA parameter semantic nodes | P2 | S |
| D7-L5 | Lang | @dtype int8/fp8 annotation extension | P2 | S |
| D7-E1 | Eng | torch.compile Inductor backend | **P0** | XL |
| D7-E2 | Eng | LLaMA-2 7B integration + bench_l3 runner | **P0** | L |
| D7-E3 | Eng | DeepSeek-V2 integration (seq≤512, quantized weights) | P2 | L |
| D7-E4 | Eng | Triton MLA template (compressed KV, lora project) | P1 | L |
| D7-E5 | Eng | Triton paged_attention template (block table scatter read) | P1 | L |
| D7-E6 | Eng | bench runner OOM guard + CSV annotation | P2 | S |
| D7-E7 | Eng | bench_l3.py (model forward + top-1 comparison + latency stats) | **P0** | M |

### G8 Development Items (18 items)

| ID | Layer | Description | Priority | Estimate |
|:---|:------|:------------|:--------:|:--------:|
| D8-L1 | Lang | `qwen25_forward.ak` example (GQA+SwiGLU+RMSNorm) | **P0** | S |
| D8-L2 | Lang | `llama3_forward.ak` example (GQA, rope, RMSNorm) | **P0** | S |
| D8-L3 | Lang | `arke-io-spec.md` (I/O contract document) | P1 | M |
| D8-L4 | Lang | Language Spec v1.0 freeze (document + tag) | P1 | M |
| D8-IR1 | IR | IR Spec v1.0 freeze (document + tag) | P1 | M |
| D8-IR2 | IR | `ir-mlir-mapping.md` (Stage 2 preparation) | P1 | M |
| D8-IR3 | IR | `test_ir_roundtrip.py` (all 45 ops × JSON round-trip) | P1 | S |
| D8-A1 | Agent | `arke optimize` unified entry with full 3-input-type support | P1 | L |
| D8-A2 | Agent | LLM auto-strategy maturity validation (all 45 ops, no human strategy) | P1 | M |
| D8-A3 | Agent | iterative loop stable operation across 4 models | P1 | M |
| D8-A4 | Agent | @rationale knowledge base (≥50 Stage 1 entries) | P2 | M |
| D8-A5 | Agent | Arke vs LLM-direct automated comparison (benchmarks/compare_arke_vs_direct.py) | P1 | M |
| D8-E1 | Eng | LLaMA-3 8B integration + bench_l3 runner | **P0** | L |
| D8-E2 | Eng | Qwen2.5 7B integration + bench_l3 runner | **P0** | L |
| D8-E3 | Eng | GPT-2 torch.compile backend E2E (≤1.15× eager, depends on D7-E1) | **P0** | M |
| D8-E4 | Eng | BL5 regression suite (CI): `ci/regression_bl5.py` | P1 | M |
| D8-E5 | Eng | Language evaluation benchmark + `language-decision.md` | P1 | M |
| D8-E6 | Eng | Stage 1 final evaluation report `STAGE1_FINAL_REPORT.md` | P1 | M |

---

## 10. Mapping to execution-plan.md

| execution-plan Phase | stage1-gate-design Gate | Key Differences |
|:---------------------|:------------------------|:----------------|
| Phase 1.0 (Environment) | G0 | No difference, direct correspondence |
| Phase 1.1 (IR+Validation) | G1 | No difference; G1.4 upgraded to full `.ak` file parsing (not ≥3/5) |
| Phase 1.2 (Codegen+E2E) | G2 | No difference; BL equivalent redefined as BL1×L1 |
| Phase 1.3 (LLM Agent) | G3 | No difference; BL equivalent BL1×L1(LLM-driven) |
| Phase 1.4 (Closed-loop) | G3/G4 | plan merges in Phase 1.4; gate splits into G3 (agent loop) + G4 (comparison) |
| Phase 1.5 (Eval Framework) | G4 | BL equivalent clarified as BL2×L1(6 tasks); geomean=0.991 recorded |
| Phase 1.6 (.ak Parser) | G6-LI.1/2/6 | plan as standalone Phase 1.6; merged into G6 Lang&IR completeness criteria |
| Phase 1.7 (BL3+GPT-2) | G5 | No difference; G5 retrospectively adopts BL3×L1+BL6/GPT-2×L3 format |
| Phase 2+ (G6-G8) | G6→G7→G8 | execution-plan did not break down G6-G8; this document provides full back-derivation |
