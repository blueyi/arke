# Arke Benchmark System

Comprehensive GPU operator benchmark framework with multi-tier baselines,
automated provenance tracking, and a three-layer evaluation architecture.

---

## Table of Contents

1. [Overview & Benchmark Level System](#1-overview--benchmark-level-system)
2. [Operator Tier (OT)](#2-operator-tier-ot)
3. [Shape Tier (ST)](#3-shape-tier-st)
4. [Three-Layer Architecture](#4-three-layer-architecture)
5. [Baseline Tiers & Operator→Baseline Matrix](#5-baseline-tiers--operatorbaseline-matrix)
6. [Measurement Protocol](./benchmark-protocol.md#measurement-protocol)
7. [Scoring System](./benchmark-protocol.md#scoring-system)
8. [Operator Source Registry](./benchmark-sources.md)
9. [Output Structure & Provenance](./benchmark-protocol.md#output-structure--provenance-tracking)
10. [CLI Interface](./benchmark-protocol.md#cli-interface)
11. [Benchmark-Driven Development](./benchmark-protocol.md#benchmark-driven-development)
12. [Implementation Status](./benchmark-protocol.md#implementation-status)
13. [Dependencies](./benchmark-protocol.md#dependencies)

**Sub-documents:**
- [`benchmark-ops.md`](./benchmark-ops.md) — Op Tier definitions, operator catalog, baseline matrix per op
- [`benchmark-shapes.md`](./benchmark-shapes.md) — Shape Tier definitions, full shape matrices per operator
- [`benchmark-protocol.md`](./benchmark-protocol.md) — Measurement protocol, scoring, CLI, implementation status
- [`benchmark-sources.md`](./benchmark-sources.md) — Operator source registry (cuBLAS, FlagGems, Liger, FlashAttn, ...)

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

### Coverage Matrix

```
              ST1(micro)  ST2(standard)  ST3(stress)  ST4(production)
OT0 (elem)      BL1          BL2            BL3            ─
OT1 (reduce)    BL1          BL2            BL3            ─
OT2 (dense)     BL1          BL2            BL3           BL5
OT3 (gated)     BL4          BL4            ─             BL5
OT4 (attn)       ─            ─             ─             BL5
Model-Complete  ──────────────── BL6 ────────────────────────
```

---

## 2. Operator Tier (OT)

Operators are classified by **computational complexity and kernel design difficulty**.

> Full details: [`benchmark-ops.md`](./benchmark-ops.md)

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

> Full shape matrices: [`benchmark-shapes.md`](./benchmark-shapes.md)

| Tier | Name | Description | Count (approx) |
|:----:|:-----|:------------|:--------------:|
| **ST1** | Micro | Small, power-of-2 aligned shapes; launch-overhead dominated | ~15 shapes/op |
| **ST2** | Standard | Medium scale + typical LLM production shapes (GPT-2, LLaMA) | ~30 shapes/op |
| **ST3** | Stress | Non-power-of-2, off-by-one, extreme aspect ratios; full alignment coverage | ~50 shapes/op |
| **ST4** | Production | LLM production shapes at scale (DeepSeek-V2/V3, LLaMA-3, Qwen2.5); long-context variants | per-op, see shapes doc |

**ST4 applies only to OT2–OT4** (compute-dense and attention ops), since elementwise/reduction
operators at production scale are subsumed by ST3.

---

## 4. Three-Layer Architecture

A unified benchmark measuring Arke-generated kernels across three evaluation layers:

```
┌──────────────────────────────────────────────────────┐
│  Layer 3 (L3): E2E Model Inference                   │
│  GPT-2 / LLaMA-2 7B / DeepSeek-V2 forward pass      │
│  Metric: wall-clock latency, throughput (tok/s)      │
│  Coverage: BL6 (Model-Complete)                      │
├──────────────────────────────────────────────────────┤
│  Layer 2 (L2): Fused Operators                       │
│  matmul+relu, matmul+gelu, swiglu, QKV+attn          │
│  Metric: TFLOPS, vs best fused baseline              │
│  Coverage: BL4–BL5                                   │
├──────────────────────────────────────────────────────┤
│  Layer 1 (L1): Single Operators                      │
│  All 20 OP_CATALOG operators                         │
│  Metric: μs latency, % of vendor-optimized baseline  │
│  Coverage: BL1–BL5                                   │
└──────────────────────────────────────────────────────┘
```

Each layer answers a different question:
- **L1:** Can Arke generate a kernel as fast as a hand-tuned one?
- **L2:** Can Arke fuse operators as well as expert-written fused kernels?
- **L3:** Does it actually make a real model faster end-to-end?

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

### L3: E2E Model

Full model forward pass benchmarks:

| Mode | Description |
|:-----|:------------|
| **Eager** | PyTorch default (SDPA attention) |
| **torch.compile** | Inductor-optimized |
| **Arke** | Arke kernels via KernelCache patch |

Reports: latency (mean/min/max/median), correctness (logit diff, top-1 match), peak memory.

---

## 5. Baseline Tiers & Operator→Baseline Matrix

### Baseline Priority Tiers

| Tier | Name | Source | License |
|:----:|:-----|:-------|:--------|
| **P0** | Vendor-optimized | cuBLAS, cuDNN, CUTLASS | NVIDIA EULA / BSD-3 |
| **P1** | Expert Triton | FlagGems, Liger-Kernel, FlashAttention | Apache-2.0 / BSD-2 / BSD-3 |
| **P2** | Reference Triton | Triton official tutorials | MIT |
| **P3** | PyTorch eager | `torch.nn.functional` | BSD-3-Clause |
| **P4** | Inductor-generated | `torch.compile` output | BSD-3-Clause |
| **P5** | LLM-direct | LLM writes Triton directly | — |

### Operator → Baseline Matrix (L1)

| Operator | OT | P0 Vendor | P1 Expert Triton | P2 Ref Triton | P3 PyTorch | P4 Inductor | P5 LLM-direct |
|:---------|:--:|:----------|:-----------------|:--------------|:-----------|:------------|:--------------|
| relu | 0 | — | FlagGems | — | `F.relu` | ✓ | ✓ |
| gelu | 0 | — | FlagGems | — | `F.gelu` | ✓ | ✓ |
| silu | 0 | — | FlagGems | — | `F.silu` | ✓ | ✓ |
| add | 0 | — | FlagGems | — | `torch.add` | ✓ | ✓ |
| mul | 0 | — | FlagGems | — | `torch.mul` | ✓ | ✓ |
| softmax | 1 | cuDNN (via PyTorch) | FlagGems | Tutorial 02 | `F.softmax` | ✓ | ✓ |
| layernorm | 1 | cuDNN (via PyTorch) | FlagGems | Tutorial 05 | `F.layer_norm` | ✓ | ✓ |
| rmsnorm | 1 | — | FlagGems, Liger | — | ✓ | ✓ | ✓ |
| rmsnorm_residual | 1 | — | Liger | — | manual | ✓ | ✓ |
| reduce_sum | 1 | — | FlagGems | — | `torch.sum` | ✓ | ✓ |
| reduce_max | 1 | — | FlagGems | — | `torch.max` | ✓ | ✓ |
| matmul | 2 | cuBLAS | FlagGems | Tutorial 03 | `torch.mm` | ✓ | ✓ |
| batch_matmul | 2 | cuBLAS | FlagGems | — | `torch.bmm` | ✓ | ✓ |
| grouped_matmul | 2 | CUTLASS | FlagGems (matmul_ogs) | — | — | ✓ | ✓ |
| transpose | 2 | — | FlagGems | — | `torch.transpose` | ✓ | ✓ |
| swiglu | 3 | — | Liger | — | manual | ✓ | ✓ |
| geglu | 3 | — | Liger | — | manual | ✓ | ✓ |
| flash_attention | 4 | cuDNN SDPA | FlashAttention | Tutorial 06 | SDPA | ✓ | ✓ |
| grouped_query_attention | 4 | cuDNN SDPA | FlashAttention | — | SDPA | ✓ | ✓ |
| multi_latent_attention | 4 | — | DeepSeek ref | — | — | — | ✓ |

### Fused Operator → Baseline Matrix (L2)

| Fused Op | P0 Vendor | P1 Expert | P3 PyTorch | P5 LLM-direct |
|:---------|:----------|:----------|:-----------|:--------------|
| matmul + relu | — | FlagGems (ATen fusion) | `torch.compile` | ✓ |
| matmul + gelu | — | FlagGems | `torch.compile` | ✓ |
| linear + cross_entropy | — | Liger `fused_linear_ce` | separate ops | ✓ |
| QKV + attention | cuDNN SDPA | FlashAttention | `F.scaled_dot_product_attention` | ✓ |
| swiglu | — | Liger `swiglu` | manual impl | ✓ |
| geglu | — | Liger `geglu` | manual impl | ✓ |

---

*For full shape matrices, see [`benchmark-shapes.md`](./benchmark-shapes.md).*
*For operator source details and installation, see [`benchmark-sources.md`](./benchmark-sources.md).*
*For measurement protocol, scoring, and CLI, see [`benchmark-protocol.md`](./benchmark-protocol.md).*

---

*Last updated: 2026-04-05*
