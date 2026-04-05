# Arke Benchmark System

Comprehensive GPU operator benchmark framework with multi-tier baselines,
automated provenance tracking, and a three-layer evaluation architecture.

---

## Table of Contents

1. [Overview & Benchmark Level System](#1-overview--benchmark-level-system)
2. [Operator Tier (OT)](#2-operator-tier-ot)
3. [Shape Tier (ST)](#3-shape-tier-st)
4. [Evaluation Layers (L1/L2/L3)](#4-evaluation-layers-l1l2l3)
5. [Baselines](#5-baselines)
6. [Measurement Protocol](./benchmark/benchmark-protocol.md#measurement-protocol)
7. [Scoring System](./benchmark/benchmark-protocol.md#scoring-system)
8. [Operator Source Registry](./benchmark/benchmark-sources.md)
9. [Output Structure & Provenance](./benchmark/benchmark-protocol.md#output-structure--provenance-tracking)
10. [CLI Interface](./benchmark/benchmark-protocol.md#cli-interface)
11. [Benchmark-Driven Development](./benchmark/benchmark-protocol.md#benchmark-driven-development)
12. [Implementation Status](./benchmark/benchmark-protocol.md#implementation-status)
13. [Dependencies](./benchmark/benchmark-protocol.md#dependencies)

**Sub-documents:**
- [`benchmark-ops.md`](./benchmark/benchmark-ops.md) — Op Tier definitions, operator catalog, per-op baseline selection
- [`benchmark-shapes.md`](./benchmark/benchmark-shapes.md) — Shape Tier definitions, full shape matrices per operator
- [`benchmark-protocol.md`](./benchmark/benchmark-protocol.md) — Measurement protocol, scoring, CLI, implementation status
- [`operator-source-registry.md`](./benchmark/operator-source-registry.md) — Complete baseline source catalog (14 sources: cuBLAS, FlagGems, Liger, FlashAttn, vLLM, ...)

---

## 1. Overview & Benchmark Level System

### Benchmark Level (BL)

A **Benchmark Level** is a specific combination of **Operator Tier** (OT) and **Shape Tier** (ST).
Higher levels cover more complex operators and/or larger/harder shapes.

```
Benchmark Level = Operator Tier (OT) × Shape Tier (ST)
```

| Level | Operator Coverage | Shape Coverage | Description | Typical Use |
|:-----:|:-----------------|:--------------|:------------|:------------|
| **BL1** | OT0–OT2 | ST1 | Basic ops × micro shapes | Quick smoke test, <30s |
| **BL2** | OT0–OT2 | ST1–ST2 | Basic ops × standard shapes | Daily CI, ~5 min |
| **BL3** | OT0–OT2 | ST1–ST3 | Basic ops × full shapes (incl. non-aligned) | Gate validation |
| **BL4** | OT0–OT4 | ST1–ST2 | **All ops** × standard shapes | Operator completeness |
| **BL5** | OT0–OT4 | ST1–ST4 | **All ops × all shapes** | Complete benchmark suite |
| **BL6** | Model-Complete | Model-Real | True model graph: all ops + exact production shapes | E2E model validation |

**BL6 (Model-Complete)** is special: instead of a Cartesian product of OT×ST, it captures the
exact operator+shape combinations that appear in a real model's forward pass:

| Model | Operators (from compute graph) | Shape Source |
|:------|:------------------------------|:-------------|
| GPT-2 Small | matmul, layernorm, gelu, softmax, add, transpose | Model params × seq ∈ {128, 512, 1024} |
| LLaMA-2 7B | matmul, rmsnorm, swiglu, flash_attention, rope, transpose | Model params × seq ∈ {512, 2048, 4096} |
| DeepSeek-V2 | matmul, grouped_matmul, rmsnorm, swiglu, multi_latent_attention, grouped_query_attention, rope | Model params × seq ∈ {512, 2048, 8192} |

> BL6 catches correctness and performance issues that BL5 misses because real models use
> specific dimension combinations (e.g. LLaMA's head_dim=128 with n_kv_heads=8) that may
> not be covered by the general shape matrix.

### OT × ST Coverage Matrix

```
              ST1(micro)  ST2(standard)  ST3(stress)  ST4(production)
OT0 (elem)      BL1          BL2            BL3            ─
OT1 (reduce)    BL1          BL2            BL3            ─
OT2 (dense)     BL1          BL2            BL3           BL5
OT3 (gated)     BL4          BL4            ─             BL5
OT4 (attn)       ─            ─             ─             BL5
Model-Complete  ──────────────── BL6 ────────────────────────
```

### Layer × BL Coverage Matrix

Benchmark Level (BL) defines **coverage scope**; Evaluation Layer (L) defines **what is measured**.
They are orthogonal — a benchmark run specifies both (e.g. "L1 at BL3").

```
         BL1  BL2  BL3  BL4  BL5  BL6
L1        ✓    ✓    ✓    ✓    ✓    ─     Single operator perf
L2        ─    ─    ─    ✓    ✓    ─     Fused operator perf
L3 ≡ BL6  ─    ─    ─    ─    ─    ✓     E2E model perf
```

> **L3 ≡ BL6**: L3 is the execution of BL6's model-complete operator+shape set as an
> end-to-end forward pass. They are the same concept viewed from two angles —
> BL6 defines the coverage, L3 defines the evaluation method.

---

## 2. Operator Tier (OT)

Operators are classified by **computational complexity and kernel design difficulty**.

> Full details: [`benchmark-ops.md`](./benchmark/benchmark-ops.md)

| Tier | Name | Operators | Core Characteristic |
|:----:|:-----|:----------|:--------------------|
| **OT0** | Elementwise | `relu`, `gelu`, `silu`, `add`, `mul` | No reduction, pure memory-bound |
| **OT1** | Reduction | `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max` | Row-wise reduction, warp-level cooperation |
| **OT2** | Compute-Dense | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose` | Matrix multiply, tensor core tiling, shared memory staging |
| **OT3** | Gated Activation | `swiglu`, `geglu` | Split + nonlinear + elementwise mul; output dim = input/2 |
| **OT4** | Attention | `flash_attention`, `grouped_query_attention`, `multi_latent_attention` | Multi-stage fused kernel, online softmax, causal mask, KV compression |

**Design rationale:**
- OT0→OT1: Adds reduction (shared memory, warp shuffle)
- OT1→OT2: Shifts from memory-bound to compute-bound (tensor cores, tiling strategy)
- OT2→OT3: Composite semantics — split + nonlinearity + multiply, output shape changes
- OT3→OT4: Multi-stage fusion (QKᵀ → online-softmax → AV), most complex tiling and memory management

---

## 3. Shape Tier (ST)

Shapes are classified by **scale, alignment, and production relevance**.

> Full shape matrices: [`benchmark-shapes.md`](./benchmark/benchmark-shapes.md)

| Tier | Name | Description | Count (approx) |
|:----:|:-----|:------------|:--------------:|
| **ST1** | Micro | Small, power-of-2 aligned shapes; launch-overhead dominated | ~15 shapes/op |
| **ST2** | Standard | Medium scale + typical LLM production shapes (GPT-2, LLaMA) | ~30 shapes/op |
| **ST3** | Stress | Non-power-of-2, off-by-one, extreme aspect ratios; full alignment coverage | ~50 shapes/op |
| **ST4** | Production | LLM production shapes at scale (DeepSeek-V2/V3, LLaMA-3, Qwen2.5); long-context variants | per-op, see shapes doc |

**ST4 applies only to OT2–OT4** (compute-dense and attention ops), since elementwise/reduction
operators at production scale are subsumed by ST3.

---

## 4. Evaluation Layers (L1/L2/L3)

Benchmark Level (BL) defines the **coverage scope** (how many ops × shapes).
Evaluation Layers define the **measurement dimension** (what question is answered).

| Layer | Question | What is Measured | BL Range | Metric |
|:-----:|:---------|:-----------------|:---------|:-------|
| **L1** | Can Arke match hand-tuned single-op kernels? | Individual operator perf | BL1–BL5 | μs latency, TFLOPS/GB·s, % of baseline |
| **L2** | Can Arke fuse operators as well as expert code? | Fused operator perf | BL4–BL5 | TFLOPS, vs best fused baseline |
| **L3** | Does it make a real model faster end-to-end? | Full model forward pass | **BL6** | Wall-clock latency, tok/s, peak memory |

> **L3 ≡ BL6**: L3 executes BL6's model-complete operator+shape set as an end-to-end
> forward pass. BL6 defines the coverage; L3 defines the evaluation method.

### L1: Single Operators

Benchmarks each of the 20 OP_CATALOG operators across the shape matrix.
Reports μs latency, TFLOPS (compute-bound ops), GB/s (memory-bound ops),
and ratio vs each available baseline tier.

### L2: Fused Operators

Compares three approaches:
1. **Separate ops** — sequential unfused kernels (baseline)
2. **torch.compile** — Inductor auto-fusion
3. **FlagGems / Liger** — expert Triton fusion

Currently supports: `matmul+relu`, `matmul+gelu`, `swiglu`, `geglu`

### L3: E2E Model (= BL6)

Full model forward pass benchmarks using BL6 model-complete shape sets:

| Mode | Description |
|:-----|:------------|
| **Eager** | PyTorch default (SDPA attention) |
| **torch.compile** | Inductor-optimized |
| **Arke** | Arke kernels via KernelCache patch |

Reports: latency (mean/min/max/median), correctness (logit diff, top-1 match), peak memory.

---

## 5. Baselines

### Baseline Priority Tiers

Each operator is benchmarked against multiple baseline tiers, ranked by expected performance:

| Tier | Name | Source | License |
|:----:|:-----|:-------|:--------|
| **P0** | Vendor-optimized | cuBLAS, cuDNN, CUTLASS | NVIDIA EULA / BSD-3 |
| **P1** | Expert Triton | FlagGems, Liger-Kernel, FlashAttention | Apache-2.0 / BSD-2 / BSD-3 |
| **P2** | Reference Triton | Triton official tutorials | MIT |
| **P3** | PyTorch eager | `torch.nn.functional` | BSD-3-Clause |
| **P4** | Inductor-generated | `torch.compile` output | BSD-3-Clause |
| **P5** | LLM-direct | LLM writes Triton directly | — |

### Where to find details

| What | Where |
|:-----|:------|
| **Per-operator baseline selection** (primary + expert baselines) | [`benchmark-ops.md`](./benchmark/benchmark-ops.md) — each op card lists its P0–P5 mapping |
| **Source installation, API, version, full op lists** | [`operator-source-registry.md`](./benchmark/operator-source-registry.md) — complete catalog of all 14 baseline sources |
| **Fused operator baselines** | [`benchmark-ops.md` §OT3/OT4](./benchmark/benchmark-ops.md) + table below |

### Fused Operator → Baseline Summary (L2)

| Fused Op | P0 Vendor | P1 Expert | P3 PyTorch | P5 LLM-direct |
|:---------|:----------|:----------|:-----------|:--------------|
| matmul + relu | — | FlagGems (ATen fusion) | `torch.compile` | ✓ |
| matmul + gelu | — | FlagGems | `torch.compile` | ✓ |
| linear + cross_entropy | — | Liger `fused_linear_ce` | separate ops | ✓ |
| QKV + attention | cuDNN SDPA | FlashAttention | `F.scaled_dot_product_attention` | ✓ |
| swiglu | — | Liger `swiglu` | manual impl | ✓ |
| geglu | — | Liger `geglu` | manual impl | ✓ |

---

*For full shape matrices, see [`benchmark-shapes.md`](./benchmark/benchmark-shapes.md).*
*For baseline source installation and API details, see [`operator-source-registry.md`](./benchmark/operator-source-registry.md).*
*For measurement protocol, scoring, and CLI, see [`benchmark-protocol.md`](./benchmark/benchmark-protocol.md).*

---

*Last updated: 2026-04-05*
