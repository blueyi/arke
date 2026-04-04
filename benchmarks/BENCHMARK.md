# Arke Benchmark System

Comprehensive GPU operator benchmark framework with multi-tier baselines,
automated provenance tracking, and three-layer evaluation architecture.

---

## Table of Contents

1. [Overview & Three-Layer Architecture](#1-overview--three-layer-architecture)
2. [Operator Categories](#2-operator-categories)
3. [Baseline Tiers & Operator → Baseline Matrix](#3-baseline-tiers--operator--baseline-matrix)
4. [Shape Matrix](#4-shape-matrix)
5. [Measurement Protocol](#5-measurement-protocol)
6. [Scoring System & Quality Gates](#6-scoring-system--quality-gates)
7. [Operator Source Registry](#7-operator-source-registry)
8. [Output Structure & Provenance Tracking](#8-output-structure--provenance-tracking)
9. [CLI Interface](#9-cli-interface)
10. [Benchmark-Driven Development](#10-benchmark-driven-development)
11. [Implementation Status](#11-implementation-status)
12. [Dependencies](#12-dependencies)

---

## 1. Overview & Three-Layer Architecture

A unified benchmark framework that measures Arke-generated kernels against
multiple baseline tiers, covering single-operator performance, fused-operator
quality, and whole-model end-to-end inference.

```
┌──────────────────────────────────────────────────────┐
│  Layer 3: E2E Model Inference                        │
│  GPT-2 / LLaMA-7B / Custom model forward pass       │
│  Metric: wall-clock latency, throughput (tok/s)      │
├──────────────────────────────────────────────────────┤
│  Layer 2: Fused Operators                            │
│  matmul+relu, matmul+gelu, linear+softmax, QKV+attn │
│  Metric: TFLOPS, vs best fused baseline              │
├──────────────────────────────────────────────────────┤
│  Layer 1: Single Operators                           │
│  matmul, softmax, layernorm, gelu, rmsnorm, rope ... │
│  Metric: μs latency, % of vendor-optimized baseline  │
└──────────────────────────────────────────────────────┘
```

Each layer answers a different question:
- **L1:** Can Arke generate a kernel as fast as a hand-tuned one?
- **L2:** Can Arke fuse operators as well as expert-written fused kernels?
- **L3:** Does it actually make a real model faster?

### L1: Single Operators

Benchmarks individual operators across a shape matrix covering tiny to LLaMA-scale.
Reports μs latency, TFLOPS (compute-bound), GB/s (memory-bound), and % of vendor baseline.

### L2: Fused Operators

Compares three approaches to operator fusion:
1. **Separate ops** — matmul then activation (baseline)
2. **torch.compile** — Inductor auto-fusion
3. **FlagGems** — ATen backend with Triton fusion

Currently supports: `matmul+relu`, `matmul+gelu`

### L3: E2E Model

GPT-2 Small end-to-end inference benchmark:

| Mode | Description |
|:-----|:------------|
| **Eager** | PyTorch default (SDPA attention) |
| **torch.compile** | Inductor-optimized |
| **Arke** | Conv1D layers patched with KernelCache |

Reports: latency (mean/min/max/median), correctness (logit diff, top-1 match), peak memory.

---

## 2. Operator Categories

All benchmark operators are organized by semantic role in modern LLM architectures.
Every Gate from **Stage 2 onward must cover at least 3 operator categories** across
multiple shape tiers — single-operator coverage is insufficient.

| Cat | Name | Operators | Architecture Relevance |
|:---:|:-----|:----------|:----------------------|
| **A** | Dense Linear | `matmul`, `batch_matmul`, `grouped_matmul` | All Transformer layers (QKV, FFN, lm_head, MoE expert) |
| **B** | Attention | `flash_attention`, `grouped_query_attention`, `multi_latent_attention` | All modern LLMs; DeepSeek MLA is unique |
| **C** | Normalization | `layernorm`, `rmsnorm`, `rmsnorm_residual` | Pre/post layer norm; LLaMA/DeepSeek use RMSNorm |
| **D** | Activation | `gelu`, `silu`, `swiglu`, `geglu`, `relu` | FFN non-linearity; SwiGLU dominant in LLaMA/DeepSeek |
| **E** | Positional Encoding | `rope`, `yarn_rope` | Rotary PE; YaRN for long-context (DeepSeek/Qwen) |
| **F** | Vocabulary | `embedding`, `cross_entropy` | Embedding lookup + language model head loss |
| **G** | MoE | `moe_gating`, `grouped_matmul` | DeepSeek-V2/V3 sparse MoE layers |

### Stage Coverage Requirements

| Stage | Minimum Category Coverage | Tier Requirement |
|:------|:--------------------------|:----------------|
| Stage 1 (G0–G5) | A + partial B/C/D | Tier 1–3 |
| Stage 2 | A + B + C + D + E (5 categories) | Tier 1–4 |
| Stage 3 | A–F (6 categories + G partial) | Tier 1–4 |
| Stage 4 | A–G (all 7 categories) | Tier 1–4 |

---

## 3. Baseline Tiers & Operator → Baseline Matrix

For each operator, baselines are ranked by **expected performance** (highest first).
The benchmark reports Arke's ratio against **every available baseline**.

### Priority Tiers

| Tier | Name | Source | License |
|:----:|:-----|:-------|:--------|
| **P0** | Vendor-optimized | cuBLAS, cuDNN, CUTLASS (gold standard) | NVIDIA EULA / BSD-3 |
| **P1** | Expert Triton | FlagGems, Liger-Kernel, FlashAttention (community best) | Apache-2.0 / BSD-2 / BSD-3 |
| **P2** | Reference Triton | Triton official tutorials (well-known, reproducible) | MIT |
| **P3** | PyTorch eager | `torch.nn.functional` / torch ops (user default) | BSD-3-Clause |
| **P4** | Inductor-generated | `torch.compile` output (auto-optimized) | BSD-3-Clause |
| **P5** | LLM-direct | LLM writes Triton directly (Arke's direct competitor) | — |

### Single Operator → Baseline Matrix (L1)

| Operator | P0 Vendor | P1 Expert Triton | P2 Ref Triton | P3 PyTorch | P4 Inductor | P5 LLM-direct |
|:---------|:----------|:-----------------|:--------------|:-----------|:------------|:---------------|
| matmul | cuBLAS (`torch.matmul`) | FlagGems `mm` | Tutorial 03 | `torch.mm` | `torch.compile` | ✓ |
| batch_matmul | cuBLAS (`torch.bmm`) | FlagGems `bmm` | — | `torch.bmm` | `torch.compile` | ✓ |
| softmax | cuDNN (via PyTorch) | FlagGems `softmax` | Tutorial 02 | `F.softmax` | `torch.compile` | ✓ |
| layernorm | cuDNN (via PyTorch) | FlagGems `layernorm` | Tutorial 05 | `F.layer_norm` | `torch.compile` | ✓ |
| rmsnorm | — | FlagGems `rmsnorm`, Liger | — | manual impl | `torch.compile` | ✓ |
| gelu | — | FlagGems `gelu` | — | `F.gelu` | `torch.compile` | ✓ |
| relu | — | FlagGems `relu` | — | `F.relu` | `torch.compile` | ✓ |
| silu | — | FlagGems `silu` | — | `F.silu` | `torch.compile` | ✓ |
| rope | — | Liger `rope` | — | manual impl | — | ✓ |
| cross_entropy | cuDNN (via PyTorch) | FlagGems, Liger (fused) | — | `F.cross_entropy` | `torch.compile` | ✓ |
| fused_attention | cuDNN SDPA | FlashAttention (Triton) | Tutorial 06 | SDPA | `torch.compile` | ✓ |
| dropout | — | FlagGems `dropout` | Tutorial 04 | `F.dropout` | `torch.compile` | ✓ |
| reduce_sum | — | FlagGems `sum` | — | `torch.sum` | `torch.compile` | ✓ |
| reduce_max | — | FlagGems `max` | — | `torch.max` | `torch.compile` | ✓ |
| embedding | — | FlagGems `embedding` | — | `nn.Embedding` | `torch.compile` | ✓ |

### Fused Operator → Baseline Matrix (L2)

| Fused Op | P0 Vendor | P1 Expert | P3 PyTorch | P5 LLM-direct |
|:---------|:----------|:----------|:-----------|:---------------|
| matmul + relu | — | FlagGems (via ATen fusion) | `torch.compile` | ✓ |
| matmul + gelu | — | FlagGems | `torch.compile` | ✓ |
| linear + cross_entropy | — | Liger `fused_linear_ce` | separate ops | ✓ |
| QKV + attention | cuDNN SDPA | FlashAttention | `F.scaled_dot_product_attention` | ✓ |
| swiglu | — | Liger `swiglu` | manual impl | ✓ |
| geglu | — | Liger `geglu` | manual impl | ✓ |

### Operator → Primary Source Mapping

Which source to use as primary baseline for each operator:

| Arke Op | Primary Baseline | Secondary Baselines | Notes |
|:--------|:-----------------|:--------------------|:------|
| **matmul** | cuBLAS (`torch.matmul`) | FlagGems, Triton tutorial | cuBLAS is the standard; FlagGems for Triton-vs-Triton |
| **batch_matmul** | cuBLAS (`torch.bmm`) | FlagGems | |
| **softmax** | PyTorch (`F.softmax`) | FlagGems, Triton tutorial | PyTorch uses cuDNN or custom CUDA |
| **layernorm** | PyTorch (`F.layer_norm`) | FlagGems, Triton tutorial | PyTorch calls cuDNN or custom |
| **rmsnorm** | FlagGems | Liger-Kernel | PyTorch has no built-in rmsnorm |
| **gelu** | PyTorch (`F.gelu`) | FlagGems | |
| **relu** | PyTorch (`F.relu`) | FlagGems | |
| **silu/swish** | PyTorch (`F.silu`) | FlagGems, Liger (swiglu) | |
| **cross_entropy** | PyTorch (`F.cross_entropy`) | FlagGems, Liger (fused) | Liger fuses linear+CE |
| **rope** | Liger-Kernel | FlagGems | No PyTorch built-in |
| **fused_attention** | PyTorch SDPA (cuDNN) | FlashAttention (CUDA), Triton tutorial | SDPA auto-selects best |
| **dropout** | PyTorch (`F.dropout`) | Triton tutorial | |
| **reduce_sum/max** | PyTorch (`torch.sum/max`) | FlagGems | |
| **embedding** | PyTorch (`nn.Embedding`) | FlagGems | |

---

## 4. Shape Matrix

Shapes are organized into **four tiers** for different use cases:

- **Tier 1** (15 shapes): Fast regression checks, daily development
- **Tier 2** (30 shapes): Standard CI, covers aligned + rectangular + LLM shapes
- **Tier 3** (50+ shapes per op): Full evaluation including **non-aligned shapes**, used for Gate G1–G5
- **Tier 4** (complex fused ops): FlashAttention, MLA, GQA, RoPE, SwiGLU with **LLM-production shapes** incl. DeepSeek; used for Stage 2+ Gates

### Matmul Shapes

| Tag | M | N | K | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `tiny` | 128 | 128 | 128 | 1 | micro | Launch overhead dominated |
| `small` | 128 | 768 | 768 | 1 | GPT-2 c_proj | Typical LLM hidden dim |
| `medium` | 128 | 2304 | 768 | 1 | GPT-2 c_attn | QKV projection |
| `square-1k` | 1024 | 1024 | 1024 | 1 | classic | Standard GEMM benchmark |
| `square-2k` | 2048 | 2048 | 2048 | 1 | classic | Compute-bound |
| `square-4k` | 4096 | 4096 | 4096 | 1 | classic | Large GEMM |
| `rect-wide` | 1024 | 4096 | 1024 | 2 | LLM FFN | Wide output |
| `rect-tall` | 4096 | 1024 | 1024 | 2 | LLM FFN | Tall input |
| `lm-head` | 128 | 50257 | 768 | 2 | GPT-2 lm_head | Vocabulary projection |
| `llama-q` | 4096 | 4096 | 4096 | 2 | LLaMA-7B | Attention Q/K/V |
| `llama-ffn` | 4096 | 11008 | 4096 | 2 | LLaMA-7B | FFN up-projection |
| `seq512` | 512 | 2304 | 768 | 2 | GPT-2 seq=512 | Triton sweet spot |
| `non-align-1` | 127 | 513 | 1000 | 3 | stress | Non-power-of-2 all dims |
| `non-align-2` | 333 | 777 | 555 | 3 | stress | Odd dimensions |
| `non-align-3` | 1023 | 1025 | 1024 | 3 | stress | Off-by-one M and N |
| `non-align-4` | 1000 | 1000 | 1000 | 3 | stress | Round non-power-of-2 |
| `non-align-5` | 384 | 640 | 1536 | 3 | stress | Non-power-of-2 real-world-ish |
| `non-align-6` | 2049 | 2047 | 2050 | 3 | stress | Off-by-one from 2048 |
| `non-align-7` | 513 | 2305 | 769 | 3 | stress | GPT-2 shapes +1 |
| `extreme-1row` | 1 | 1024 | 1024 | 3 | stress | Single-row matmul |
| `extreme-16` | 16 | 4096 | 4096 | 3 | stress | Very small M |
| `extreme-long` | 8192 | 64 | 4096 | 3 | stress | Extreme M/N ratio |

> Tier 3 includes 50 matmul shapes total (see `benchmarks/shapes.py` for the full list).

### Softmax Shapes

| Tag | M | N | Tier | Notes |
|:----|----:|----:|:----:|:------|
| `attn-small` | 12 | 128 | 1 | GPT-2 12-head, seq=128 |
| `attn-med` | 12 | 512 | 1 | GPT-2 12-head, seq=512 |
| `attn-256` | 12 | 256 | 1 | GPT-2 12-head, seq=256 |
| `attn-large` | 32 | 2048 | 2 | LLaMA 32-head, seq=2048 |
| `attn-64` | 12 | 64 | 2 | Short attention |
| `attn-4k` | 32 | 4096 | 2 | Long context |
| `attn-8k` | 32 | 8192 | 2 | Very long context |
| `square-1k` | 1024 | 1024 | 2 | Moderate stress test |
| `square-4k` | 4096 | 4096 | 2 | Large stress test |
| `wide-vocab` | 1 | 50257 | 2 | Vocabulary softmax |
| `wide-llama` | 1 | 128256 | 2 | LLaMA-3 vocabulary |
| `batch-large` | 128 | 4096 | 2 | Batch softmax |
| `batch-xlarge` | 1024 | 1024 | 2 | Large batch |
| `non-align-1` | 13 | 513 | 3 | Non-aligned head count + N |
| `non-align-2` | 7 | 511 | 3 | Prime heads, off-by-one N |
| `non-align-3` | 15 | 1023 | 3 | Non-power-of-2 dims |
| `non-align-4` | 32 | 2049 | 3 | Off-by-one from 2048 |
| `non-align-5` | 11 | 127 | 3 | Prime head, off-by-one N |
| `non-align-6` | 1 | 50261 | 3 | Non-aligned vocab (prime-ish) |
| `non-align-7` | 33 | 1000 | 3 | Non-power-of-2 both dims |
| `extreme-tiny` | 1 | 16 | 3 | Minimal softmax |
| `extreme-wide` | 1 | 1048576 | 3 | 1M-wide single row |
| `extreme-tall` | 65536 | 64 | 3 | Many short rows |
| `extreme-batch` | 4096 | 512 | 3 | Many medium rows |
| `mixed-1` | 100 | 3000 | 3 | Round non-power-of-2 |

> Tier 3 includes 25 softmax shapes total.

### LayerNorm / RMSNorm Shapes

| Tag | Batch | Hidden | Tier | Notes |
|:----|------:|-------:|:----:|:------|
| `gpt2` | 128 | 768 | 1 | GPT-2 |
| `gpt2-ffn` | 128 | 3072 | 1 | GPT-2 FFN intermediate |
| `llama` | 128 | 4096 | 2 | LLaMA-7B |
| `llama-13b` | 128 | 5120 | 2 | LLaMA-13B |
| `large` | 2048 | 4096 | 2 | Long sequence |
| `seq1k` | 1024 | 768 | 2 | GPT-2 long seq |
| `batch-large` | 4096 | 4096 | 2 | Stress test |
| `non-align-1` | 127 | 769 | 3 | Non-aligned both dims |
| `non-align-2` | 1000 | 3000 | 3 | Round non-power-of-2 |
| `non-align-3` | 333 | 4097 | 3 | Off-by-one hidden |
| `non-align-4` | 2049 | 4095 | 3 | Off-by-one both |
| `non-align-5` | 100 | 5121 | 3 | LLaMA-13B +1 |
| `extreme-small` | 1 | 768 | 3 | Single-sample norm |
| `extreme-large` | 8192 | 4096 | 3 | Very long sequence |
| `extreme-hidden` | 128 | 14336 | 3 | Mixtral FFN hidden |

> Tier 3 includes 15 layernorm/rmsnorm shapes total.

### Elementwise Shapes (relu, gelu, silu)

| Tag | M | N | Tier | Notes |
|:----|----:|----:|:----:|:------|
| `small` | 128 | 768 | 1 | GPT-2 hidden |
| `medium` | 128 | 3072 | 1 | GPT-2 FFN |
| `large` | 4096 | 4096 | 2 | Stress test |
| `llama-ffn` | 4096 | 11008 | 2 | LLaMA-7B FFN |
| `xlarge` | 8192 | 4096 | 2 | Very large |
| `seq1k` | 1024 | 768 | 2 | GPT-2 long seq |
| `non-align-1` | 127 | 769 | 3 | Off-by-one both dims |
| `non-align-2` | 1000 | 3000 | 3 | Round non-power-of-2 |
| `non-align-3` | 2049 | 4097 | 3 | Off-by-one from aligned |
| `non-align-4` | 333 | 11009 | 3 | LLaMA FFN +1, odd batch |
| `non-align-5` | 513 | 769 | 3 | Off-by-one, GPT-2 like |
| `extreme-flat` | 1 | 1048576 | 3 | Single-row 1M elements |
| `extreme-tall` | 65536 | 16 | 3 | Many very short rows |
| `extreme-wide` | 32768 | 128 | 3 | Many medium rows |
| `mixed-1` | 100 | 14336 | 3 | Mixtral FFN, round batch |

> Tier 3 includes 15 elementwise shapes total.

### Tier 4: Complex Fused Operators

Tier 4 covers operators with complex multi-stage semantics and hardware-specific
optimization requirements. Used for Stage 2+ Gate verification.

> Shape notation for attention: `(batch, seq_len, num_heads, head_dim)`.
> DeepSeek shapes include long-context variants (8K–163K tokens).

#### Reference LLM Architecture Parameters

| Model | Hidden | Q Heads | KV Heads | Head Dim | FFN | Layers | Vocab | Max Ctx |
|:------|-------:|--------:|---------:|---------:|----:|-------:|------:|--------:|
| GPT-2 Small | 768 | 12 | 12 | 64 | 3072 | 12 | 50257 | 1024 |
| GPT-2 Medium | 1024 | 16 | 16 | 64 | 4096 | 24 | 50257 | 1024 |
| LLaMA-2 7B | 4096 | 32 | 32 | 128 | 11008 | 32 | 32000 | 4096 |
| LLaMA-3 8B | 4096 | 32 | 8 | 128 | 14336 | 32 | 128256 | 8192 |
| DeepSeek-V2 16B | 5120 | 128 | 128 | 128 | 12288 | 60 | 102400 | 163840 |
| DeepSeek-V3 671B | 7168 | 128 | 128 | 128 | 18432 | 61 | 129280 | 163840 |
| Qwen2.5 7B | 3584 | 28 | 4 | 128 | 18944 | 28 | 151936 | 131072 |
| Mistral 7B | 4096 | 32 | 8 | 128 | 14336 | 32 | 32000 | 32768 |

#### FlashAttention Shapes

Baseline: `flash-attn` library (P1). dtype=f16/bf16.

| Tag | Batch | Seq | Heads | Head Dim | Tier | Source | Notes |
|:----|------:|----:|------:|---------:|:----:|:-------|:------|
| `gpt2-sm-128` | 1 | 128 | 12 | 64 | 4 | GPT-2 Small | Short context |
| `gpt2-sm-512` | 1 | 512 | 12 | 64 | 4 | GPT-2 Small | Medium context |
| `gpt2-sm-1k` | 1 | 1024 | 12 | 64 | 4 | GPT-2 Small | Max context |
| `gpt2-md-2k` | 1 | 2048 | 16 | 64 | 4 | GPT-2 Medium | Max context |
| `llama2-7b-512` | 1 | 512 | 32 | 128 | 4 | LLaMA-2 7B | Typical inference |
| `llama2-7b-2k` | 1 | 2048 | 32 | 128 | 4 | LLaMA-2 7B | Max context |
| `llama2-7b-4k` | 1 | 4096 | 32 | 128 | 4 | LLaMA-2 7B | Extended |
| `llama2-7b-batch` | 4 | 512 | 32 | 128 | 4 | LLaMA-2 7B | Batched inference |
| `llama3-8b-512` | 1 | 512 | 32 | 128 | 4 | LLaMA-3 8B | GQA (kv_heads=8) |
| `llama3-8b-8k` | 1 | 8192 | 32 | 128 | 4 | LLaMA-3 8B | Max context |
| `qwen25-7b-512` | 1 | 512 | 28 | 128 | 4 | Qwen2.5 7B | Standard |
| `qwen25-7b-2k` | 1 | 2048 | 28 | 128 | 4 | Qwen2.5 7B | — |
| `qwen25-7b-8k` | 1 | 8192 | 28 | 128 | 4 | Qwen2.5 7B | Long context |
| `ds-v2-512` | 1 | 512 | 128 | 128 | 4 | DeepSeek-V2 16B | Standard |
| `ds-v2-1k` | 1 | 1024 | 128 | 128 | 4 | DeepSeek-V2 16B | — |
| `ds-v2-2k` | 1 | 2048 | 128 | 128 | 4 | DeepSeek-V2 16B | — |
| `ds-v2-4k` | 1 | 4096 | 128 | 128 | 4 | DeepSeek-V2 16B | — |
| `ds-v2-8k` | 1 | 8192 | 128 | 128 | 4 | DeepSeek-V2 16B | Long context |
| `ds-v2-16k` | 1 | 16384 | 128 | 128 | 4 | DeepSeek-V2 16B | Long context |
| `ds-v3-32k` | 1 | 32768 | 128 | 128 | 4 | DeepSeek-V3 671B | Ultra-long ctx |
| `ds-v3-163k` | 1 | 163840 | 128 | 128 | 4 | DeepSeek-V3 671B | Max context limit |
| `ds-batch-8` | 8 | 512 | 128 | 128 | 4 | DeepSeek-V3 671B | Batched |

> Long-context shapes (≥8K) may OOM on 6GB VRAM; recorded as OOM, not fail.
> Stage 2 Gate requires passing ≥12 of 22 shapes (excluding OOM).

#### Multi-Latent Attention (MLA, DeepSeek-V2/V3)

DeepSeek-specific: compressed KV latent representation eliminates KV cache growth.
Shape: `(batch, seq, n_heads, head_dim, kv_lora_rank)`. Baseline: custom DeepSeek reference.

| Tag | Batch | Seq | Heads | Head Dim | KV LoRA Rank | Tier | Source |
|:----|------:|----:|------:|---------:|-------------:|:----:|:-------|
| `ds-v2-mla-512` | 1 | 512 | 128 | 128 | 512 | 4 | DeepSeek-V2 16B |
| `ds-v2-mla-2k` | 1 | 2048 | 128 | 128 | 512 | 4 | DeepSeek-V2 16B |
| `ds-v2-mla-4k` | 1 | 4096 | 128 | 128 | 512 | 4 | DeepSeek-V2 16B |
| `ds-v2-mla-8k` | 1 | 8192 | 128 | 128 | 512 | 4 | DeepSeek-V2 16B |
| `ds-v3-mla-512` | 1 | 512 | 128 | 128 | 1024 | 4 | DeepSeek-V3 671B |
| `ds-v3-mla-2k` | 1 | 2048 | 128 | 128 | 1024 | 4 | DeepSeek-V3 671B |
| `ds-v3-mla-4k` | 1 | 4096 | 128 | 128 | 1024 | 4 | DeepSeek-V3 671B |
| `ds-v3-mla-8k` | 1 | 8192 | 128 | 128 | 1024 | 4 | DeepSeek-V3 671B |

#### Grouped Query Attention (GQA)

Shape: `(batch, seq, n_q_heads, head_dim)` with `n_kv_heads < n_q_heads`.

| Tag | Batch | Seq | Q Heads | Head Dim | KV Heads | Tier | Source |
|:----|------:|----:|--------:|---------:|---------:|:----:|:-------|
| `llama3-8b-gqa-512` | 1 | 512 | 32 | 128 | 8 | 4 | LLaMA-3 8B |
| `llama3-8b-gqa-2k` | 1 | 2048 | 32 | 128 | 8 | 4 | LLaMA-3 8B |
| `llama3-8b-gqa-8k` | 1 | 8192 | 32 | 128 | 8 | 4 | LLaMA-3 8B |
| `mistral-7b-512` | 1 | 512 | 32 | 128 | 8 | 4 | Mistral 7B |
| `qwen25-7b-512` | 1 | 512 | 28 | 128 | 4 | 4 | Qwen2.5 7B |
| `qwen25-7b-2k` | 1 | 2048 | 28 | 128 | 4 | 4 | Qwen2.5 7B |

#### RoPE / YaRN Shapes

Shape: `(seq_len, n_heads, head_dim)`, dtype=f16/bf16. YaRN = long-context extension.

| Tag | Seq | Heads | Head Dim | Tier | Source | Notes |
|:----|----:|------:|---------:|:----:|:-------|:------|
| `llama2-512` | 512 | 32 | 128 | 4 | LLaMA-2 7B | Standard RoPE |
| `llama2-2k` | 2048 | 32 | 128 | 4 | LLaMA-2 7B | Max context |
| `llama2-4k` | 4096 | 32 | 128 | 4 | LLaMA-2 7B | Extended |
| `ds-v2-8k` | 8192 | 128 | 128 | 4 | DeepSeek-V2 YaRN | Long context |
| `ds-v2-16k` | 16384 | 128 | 128 | 4 | DeepSeek-V2 YaRN | — |
| `ds-v2-32k` | 32768 | 128 | 128 | 4 | DeepSeek-V2 YaRN | — |
| `ds-v3-65k` | 65536 | 128 | 128 | 4 | DeepSeek-V3 YaRN | Ultra-long |
| `ds-v3-163k` | 163840 | 128 | 128 | 4 | DeepSeek-V3 YaRN | Max context |
| `qwen25-8k` | 8192 | 28 | 128 | 4 | Qwen2.5 7B YaRN | — |

#### SwiGLU / GeGLU Shapes

Fused gate + activation used in LLaMA, DeepSeek FFN. Shape: `(seq_len, ffn_dim)`, dtype=f16.

| Tag | Seq | FFN Dim | Tier | Source | Notes |
|:----|----:|--------:|:----:|:-------|:------|
| `gpt2-sm` | 128 | 3072 | 4 | GPT-2 Small | GELU (reference) |
| `llama2-7b` | 512 | 11008 | 4 | LLaMA-2 7B | SwiGLU |
| `llama2-7b-long` | 2048 | 11008 | 4 | LLaMA-2 7B | SwiGLU long seq |
| `ds-v2-512` | 512 | 12288 | 4 | DeepSeek-V2 | SwiGLU |
| `ds-v2-2k` | 2048 | 12288 | 4 | DeepSeek-V2 | SwiGLU |
| `ds-v3-512` | 512 | 18432 | 4 | DeepSeek-V3 | SwiGLU (large FFN) |
| `ds-v3-2k` | 2048 | 18432 | 4 | DeepSeek-V3 | SwiGLU |
| `qwen25-7b` | 512 | 18944 | 4 | Qwen2.5 7B | SwiGLU |

#### Matmul Tier 4 (LLM-Scale + Long Context)

Extends Tier 3 with DeepSeek-V2/V3 and Qwen2.5 production shapes including long-context variants.

| Tag | M | N | K | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `ds-v2-attn` | 512 | 5120 | 5120 | 4 | DeepSeek-V2 | Attention proj |
| `ds-v2-ffn-up` | 512 | 12288 | 5120 | 4 | DeepSeek-V2 | FFN up |
| `ds-v2-ffn-down` | 512 | 5120 | 12288 | 4 | DeepSeek-V2 | FFN down |
| `ds-v2-lmhead` | 512 | 102400 | 5120 | 4 | DeepSeek-V2 | lm_head (huge vocab) |
| `ds-v3-attn` | 1024 | 7168 | 7168 | 4 | DeepSeek-V3 | Attention proj |
| `ds-v3-ffn-up` | 1024 | 18432 | 7168 | 4 | DeepSeek-V3 | FFN up |
| `ds-v3-lmhead` | 512 | 129280 | 7168 | 4 | DeepSeek-V3 | lm_head (huge vocab) |
| `qwen25-attn` | 512 | 3584 | 3584 | 4 | Qwen2.5 7B | Attention |
| `qwen25-ffn-up` | 512 | 18944 | 3584 | 4 | Qwen2.5 7B | FFN up (wide) |
| `qwen25-lmhead` | 512 | 151936 | 3584 | 4 | Qwen2.5 7B | lm_head (huge vocab) |
| `ds-v2-long-8k` | 8192 | 5120 | 5120 | 4 | DeepSeek-V2 | Long ctx seq=8K |
| `ds-v3-long-32k` | 32768 | 7168 | 7168 | 4 | DeepSeek-V3 | Long ctx seq=32K |

---

## 5. Measurement Protocol

### Single Operator (L1)

```python
# 1. Warmup: 200 iterations (triggers autotune, JIT)
for _ in range(200):
    kernel(inputs)
torch.cuda.synchronize()

# 2. Measure: CUDA events, 500 iterations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(500):
    kernel(inputs)
end.record()
torch.cuda.synchronize()
latency_us = start.elapsed_time(end) / 500 * 1000

# 3. Alternative: triton.testing.do_bench (FlagGems standard)
from triton.testing import do_bench
latency_ms = do_bench(lambda: kernel(inputs), warmup=200, rep=500)
```

### Fused Operator (L2)

Same protocol as L1, but input is the unfused sequence of ops vs the fused kernel.

### E2E Model (L3)

```python
# 1. Load model, apply patches
# 2. Warmup: 50 forward passes
# 3. Measure: 200 forward passes with CUDA events
# 4. Report: mean latency (ms), throughput (samples/s or tok/s)
# 5. Correctness: top-1 logit match + max absolute diff
```

### Correctness Tolerances

| Dtype | atol | rtol | Method |
|:------|-----:|-----:|:-------|
| f16 | 0.1 | 0.05 | `np.allclose` + max/mean diff |
| f32 | 1e-5 | 1e-4 | `np.allclose` |
| bf16 | 0.2 | 0.1 | `np.allclose` |

### Metrics Collected Per Run

| Metric | Unit | Description |
|:-------|:-----|:------------|
| `latency_us` | μs | Mean kernel latency |
| `latency_min_us` | μs | Minimum kernel latency |
| `tflops` | TFLOPS | Achieved throughput (for compute-bound ops) |
| `gbps` | GB/s | Achieved bandwidth (for memory-bound ops) |
| `vs_p0` | ratio | Arke latency / P0 vendor baseline latency |
| `vs_p1` | ratio | Arke latency / P1 expert Triton latency |
| `vs_p3` | ratio | Arke latency / P3 PyTorch eager latency |
| `vs_p5` | ratio | Arke latency / P5 LLM-direct latency |
| `correct` | bool | Passes numerical tolerance |
| `max_diff` | float | Maximum absolute difference from reference |
| `tokens_in` | int | LLM input tokens consumed (Arke/LLM-direct) |
| `tokens_out` | int | LLM output tokens consumed |
| `compile_time_s` | s | Time to generate + compile kernel |
| `arke_turns` | int | Number of LLM agent turns (Arke only) |

---

## 6. Scoring System & Quality Gates

### Single Operator Score (per shape)

```
op_score = Σ(weight_i × ratio_to_baseline_i) / Σ(weight_i)

where:
  ratio_to_baseline = baseline_latency / arke_latency  (>1 = Arke faster)
  weight: P0=3, P1=2, P3=1
```

### Layer Score

```
layer_score = geomean(op_scores)   # geometric mean across all ops
```

### Overall Arke Score

```
arke_score = 0.3 × L1_score + 0.3 × L2_score + 0.4 × L3_score
```

L3 weighted highest because real-world impact matters most.

### Report Indicators

| Indicator | Meaning |
|:---------:|:--------|
| 🟢 | ≥ 90% of baseline |
| 🟡 | ≥ 80% of baseline |
| 🔴 | < 80% of baseline |

### Quality Gates (G0–G5)

Quality gates follow the principle **Function > Accuracy > Performance** and use progressive performance thresholds across development stages.
See [gate-redesign.md](../docs/design/gate-redesign.md) for full SMART criteria per gate.

| Gate | Type | Core Question | Accuracy | Performance |
|:----:|:----:|:-------------|:---------|:------------|
| **G0** | Function | Environment works? | — | — |
| **G1** | Func+Acc | IR expressible & validation correct? | Tier 3 full 100% (4 ops) | — |
| **G2** | Func+Acc+Perf | Codegen produces correct, usable kernels? | Tier 3 full 100% (4 ops) | matmul ≥50% shapes ≥50% cuBLAS, geomean ≥60%; softmax ≥40% ≥50% cuDNN; elementwise ≥50% ≥50% PyTorch; layernorm ≥40% ≥50% cuDNN |
| **G3** | Func+Acc | LLM agent closed-loop optimization? | Tier 3 sampled 10 shapes 100% | Observe only |
| **G4** | Acc+Perf | Arke beats LLM-direct? | correct rate ≥ LLM-direct | ≥90% LLM-direct; ≥70% FlagGems; tokens ≤60% |
| **G5** | Acc+Perf | Works in real models? | Multi-config 100% | ≤1.15× eager (seq=128); ≤1.20× (seq=512); mem ≤6GB |

#### Performance Progression

```
           G2 (Template)  G3 (Agent)    G4 (Compare)   G5 (E2E)
           ────────────   ──────────    ────────────   ────────
Function:  ✓ required     ✓ core        —              ✓ coverage
Accuracy:  100% (4 ops)   100%          ≥ LLM-direct   100% multi-config
Perf goal: ≥50% cuBLAS    observe only  ≥90% direct    ≤1.15× eager
           ≥40% cuDNN                   ≥70% FlagGems
           ≥50% PyTorch
Perf type: absolute floor  no gate      relative edge  E2E overhead
```

#### Exclusion Rules

| Scenario | Handling | Reason |
|:---------|:---------|:-------|
| M ≤ 32 (matmul) | Accuracy must pass; perf excluded from stats | Triton ~55μs launch floor |
| N ≤ 32 (softmax) | Accuracy must pass; perf excluded from stats | Same |
| M×N ≤ 1024 (elementwise) | Accuracy must pass; perf excluded from stats | Kernel-launch dominated |
| Batch ≤ 1 (layernorm) | Accuracy must pass; perf excluded from stats | Single-sample overhead |
| OOM shapes | Skip, record "OOM" | 6GB VRAM limit |
| Triton compile timeout (>60s) | Record "TIMEOUT", accuracy marked fail | Template may need fix |

---

### Quality Gates: Stage 2–4 (S2-G1 – S4-G*)

All Stage 2+ Gates must cover **multiple operator categories** (see §2) and **multiple shape tiers**.
Single-operator-only Gates are not acceptable from Stage 2 onward.

#### Stage 2: SIMD Architecture Validation (Ascend backend)

| Gate | Category Coverage | Shape Tier | Accuracy Criterion | Performance Criterion |
|:----:|:-----------------|:----------:|:-------------------|:---------------------|
| **S2-G1** | A (matmul, batch_matmul) + C (rmsnorm) + D (swiglu) | Tier 2 (15 shapes) | 100% on NVIDIA + Ascend | Arke/FlagGems ≥1.1× on Ascend (Tier 2 geomean) |
| **S2-G2** | B (flash_attention, GQA) | Tier 4: ≥12/22 FlashAttention shapes (excl. OOM) | 100% correct | ≥0.7× flash-attn baseline |
| **S2-G3** | E (rope, yarn_rope) | Tier 4 RoPE (9 shapes, incl. DeepSeek YaRN) | 100% correct | ≥0.8× Liger baseline |
| **S2-G4** | D (swiglu, geglu) | Tier 4 SwiGLU (8 shapes) | 100% correct | ≥0.9× Liger swiglu baseline |
| **S2-G5** | A (matmul) — LLM-scale shapes | Tier 4 Matmul (12 shapes, incl. DeepSeek/Qwen) | 100% correct | Arke/cuBLAS ≥0.85× geomean |
| **S2-G6** | A+B+C+D+E combined | Tier 2 + Tier 4 sampled | 100% (all categories) | LLM w/  @rationale > w/o, ≥10% improvement |

> S2-G2 baseline: `flash-attn` CUDA library (P1).
> S2-G6 verifies the @rationale cross-architecture feedback loop (H4 validation).

#### Stage 3: MLIR Backend + Level-2 LLM Decisions

| Gate | Category Coverage | Shape Tier | Accuracy Criterion | Performance Criterion |
|:----:|:-----------------|:----------:|:-------------------|:---------------------|
| **S3-G1** | A (matmul) | Tier 2 (15 shapes) | 100% via MLIR path | MLIR ≥ Stage 1 Triton path (geomean) |
| **S3-G2** | A+C+D | Tier 2 + Tier 4 sampled (20 shapes) | 100% via MLIR path | MLIR ≥ Stage 1 Triton path (all 3 cats) |
| **S3-G3** | B (flash_attention, MLA) | Tier 4 FlashAttention + MLA (14 shapes) | 100% via MLIR path | MLIR ≥ Triton path |
| **S3-G4** | A (matmul) | Tier 2 (15 shapes) | 100% | LLM Level-1+2 decisions > LLM Level-1 + default Level-2 by ≥15% |
| **S3-G5** | B (flash_attention) | Tier 4 FlashAttention (10 shapes) | 100% | LLM Level-2 FlashAttention > default by ≥20% |
| **S3-G6** | A+B+C+D+E+F | Tier 2 + Tier 4 (all cats) | 100% | @rationale knowledge base (≥50 entries) improves new-op optimization ≥10% |

> S3-G4/G5 validate the core LLM-Native claim: LLM Level-2 decisions add measurable value.
> S3-G6 validates the @rationale knowledge accumulation loop.

#### Stage 4: Full LLM-Native Generation + LLVM Backend

| Gate | Category Coverage | Shape Tier | Accuracy Criterion | Performance Criterion |
|:----:|:-----------------|:----------:|:-------------------|:---------------------|
| **S4-G1** | A+B+C+D (4 cats) | Tier 4 (all) | LLM-generated Semantic IR passes V0+V1 (100%) | — |
| **S4-G2** | A+B+C+D (4 cats) | Tier 4 (all) | 100% | LLM end-to-end generated kernels ≥80% of human-written baseline |
| **S4-G3** | B (flash_attention, MLA, GQA) | Tier 4 FlashAttention+MLA+GQA (all) | 100% | LLM-generated FlashAttention ≥70% of flash-attn-2 |
| **S4-G4** | A+B+C+D+E+F | Tier 2 + Tier 4 | 100% | @rationale knowledge base (≥200 entries) > no knowledge base by ≥20% |
| **S4-G5** | A+C (matmul, layernorm, rmsnorm) | Tier 2 (15 shapes) | 100% via LLVM path | LLVM path ≥ MLIR path + 5% |

> S4-G3 is the ultimate FlashAttention challenge: LLM generates a custom FlashAttention
> variant from scratch with @rationale guidance, reaching 70% of the gold standard.
> S4-G4 validates the long-term "human experience → @rationale → LLM improvement" loop.

---

## 7. Operator Source Registry

Centralized registry of GPU operators used as benchmark baselines.
Each entry specifies: what it does, where it comes from, how to invoke it, and license.

### 6.1 NVIDIA Vendor Libraries (CUDA) — P0

The gold standard — closed-source, hardware-specific, vendor-tuned.

#### cuBLAS (via PyTorch)

| Field | Value |
|:------|:------|
| **Operators** | matmul, batch_matmul, addmm, gemm |
| **Access** | `torch.matmul()`, `torch.mm()`, `torch.addmm()` |
| **Install** | Bundled with PyTorch CUDA build |
| **Version** | CUDA 13.1 (via PyTorch 2.6.0+cu124) |
| **License** | NVIDIA EULA (proprietary) |
| **Notes** | PyTorch automatically dispatches to cuBLAS for GEMM ops. Best performance for matmul on NVIDIA GPUs. |

#### cuDNN (via PyTorch SDPA)

| Field | Value |
|:------|:------|
| **Operators** | fused_attention (SDPA), conv2d, batchnorm |
| **Access** | `torch.nn.functional.scaled_dot_product_attention()` |
| **Install** | Bundled with PyTorch CUDA build |
| **Version** | cuDNN 9.x (via PyTorch 2.6.0) |
| **License** | NVIDIA EULA (proprietary) |
| **Notes** | SDPA picks the best backend (FlashAttention, cuDNN, math). |

#### CUTLASS

| Field | Value |
|:------|:------|
| **Operators** | gemm, grouped_gemm, conv2d, fused_gemm_epilogue |
| **Repo** | https://github.com/NVIDIA/cutlass |
| **Install** | `pip install nvidia-cutlass` or build from source |
| **License** | BSD-3-Clause |
| **Notes** | Template library for GEMM. torch.compile/inductor uses CUTLASS templates for some matmul configs. Useful as a C++ baseline alternative to cuBLAS. |

### 6.2 Triton Official (triton-lang) — P2

Reference implementations from the Triton project itself.

#### Triton Tutorials

| Field | Value |
|:------|:------|
| **Operators** | vector_add, softmax, matmul, dropout, layer_norm, fused_attention, group_gemm, persistent_matmul |
| **Repo** | https://github.com/triton-lang/triton (`python/tutorials/`) |
| **URL** | https://triton-lang.org/main/getting-started/tutorials/ |
| **Install** | Part of `triton` package source |
| **License** | MIT |
| **Notes** | Official tutorial kernels. Well-documented, pedagogical quality. The matmul tutorial includes autotuning. Fused attention is based on FlashAttention v2 algorithm. |

#### triton-lang/kernels

| Field | Value |
|:------|:------|
| **Operators** | (early stage — check repo for current state) |
| **Repo** | https://github.com/triton-lang/kernels |
| **License** | MIT |
| **Notes** | Official kernel library under development. |

### 6.3 FlagGems (BAAI / FlagOS) — P1

Production-grade Triton operator library, 200+ ops, PyTorch ATen backend registration.

| Field | Value |
|:------|:------|
| **Operators** | matmul, softmax, layernorm, rmsnorm, gelu, silu, relu, dropout, cross_entropy, embedding, add, mul, div, exp, log, pow, sin, cos, where, sum, mean, max, min, cumsum, topk, sort, unique, scatter, gather, index_select, bmm, addmm, outer, mv, ... (200+ total) |
| **Repo** | https://github.com/flagos-ai/FlagGems |
| **Install** | `pip install flag-gems` |
| **License** | Apache-2.0 |
| **Notes** | Registers as PyTorch ATen backend — drop-in replacement for eager ops. Provides its own benchmark framework. Performance generally matches or exceeds PyTorch ATen on NVIDIA GPUs. Multi-backend (NVIDIA + others). |

**Key operators for Arke benchmarks:**

| Op | FlagGems module | Shape notes |
|:---|:----------------|:------------|
| matmul | `flag_gems.ops.mm` | Standard GEMM, autotuned |
| softmax | `flag_gems.ops.softmax` | Row-wise, fused |
| layernorm | `flag_gems.ops.layernorm` | Fused mean+var |
| gelu | `flag_gems.ops.gelu` | Elementwise, fused |
| rmsnorm | `flag_gems.ops.rmsnorm` | LLM-specific |
| cross_entropy | `flag_gems.ops.cross_entropy` | Fused log_softmax+nll |

**Usage:**

```python
import flag_gems

# Register FlagGems as ATen backend
flag_gems.enable()

# Now torch.matmul() uses FlagGems Triton kernel instead of cuBLAS
y = torch.matmul(a, b)

# Or use directly
from flag_gems.ops import mm as flaggems_mm
y = flaggems_mm(a, b)
```

### 6.4 Liger-Kernel (LinkedIn) — P1

LLM training-focused Triton kernels with aggressive operator fusion.

| Field | Value |
|:------|:------|
| **Operators** | rmsnorm, rope, swiglu, cross_entropy, fused_linear_cross_entropy, geglu, layernorm, fused_linear_jsd, kto_loss, dpo_loss, orpo_loss, cpo_loss, simpo_loss |
| **Repo** | https://github.com/linkedin/Liger-Kernel |
| **Paper** | https://arxiv.org/abs/2410.10989 |
| **Install** | `pip install liger-kernel` |
| **License** | BSD-2-Clause |
| **Notes** | Focused on training efficiency (memory + throughput). Kernels fuse forward+backward. Key innovation: chunked cross-entropy and fused linear+loss. Less relevant for inference-only benchmarks but excellent for fused-op comparisons. |

**Key operators for Arke benchmarks:**

| Op | Liger module | Notes |
|:---|:-------------|:------|
| rmsnorm | `liger_kernel.ops.rms_norm` | Fused, in-place grad |
| rope | `liger_kernel.ops.rope` | Rotary position embedding |
| swiglu | `liger_kernel.ops.swiglu` | Fused SwiGLU activation |
| cross_entropy | `liger_kernel.ops.cross_entropy` | Memory-efficient chunked |
| geglu | `liger_kernel.ops.geglu` | Fused GeGLU activation |

**Usage:**

```python
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
y = LigerRMSNormFunction.apply(x, weight, eps)
```

### 6.5 FlashAttention (Dao-AILab) — P1

The standard for fused attention kernels.

| Field | Value |
|:------|:------|
| **Operators** | flash_attention_forward, flash_attention_backward |
| **Repo** | https://github.com/Dao-AILab/flash-attention |
| **Triton version** | `flash_attn/flash_attn_triton.py` |
| **CUDA version** | `csrc/flash_attn/` (C++/CUDA, faster) |
| **Install** | `pip install flash-attn` |
| **License** | BSD-3-Clause |
| **Notes** | CUDA version is production standard (used by PyTorch SDPA). Triton version is reference/portable. Useful for comparing Arke's fused_attention against both implementations. |

### 6.6 PyTorch Inductor (torch.compile) — P4

Auto-generated Triton kernels from `torch.compile`.

| Field | Value |
|:------|:------|
| **Operators** | All elementwise, reductions, some matmul (via CUTLASS/Triton templates) |
| **Access** | `torch.compile(model)` → inspect via `TORCH_COMPILE_DEBUG=1` |
| **Source** | `torch/_inductor/codegen/triton.py` |
| **License** | BSD-3-Clause (PyTorch) |
| **Notes** | Inductor generates Triton for pointwise/reduction ops, dispatches matmul to cuBLAS/CUTLASS. The generated kernels include fusion optimizations. Useful as "what the compiler generates" baseline. |

**Extracting inductor kernels:**

```bash
TORCH_COMPILE_DEBUG=1 python script.py
# Kernels saved to torch_compile_debug/run_*/output_code.py
```

### 6.7 HuggingFace Kernels Community

Community-contributed Triton kernels hosted on HuggingFace.

| Field | Value |
|:------|:------|
| **Operators** | matmul_ogs, various community kernels |
| **Repo** | https://github.com/huggingface/kernels-community |
| **HF Hub** | https://huggingface.co/kernels-community/triton_kernels |
| **Install** | `pip install triton-kernels` |
| **License** | Apache-2.0 |
| **Notes** | Growing collection. `matmul_ogs` is an optimized grouped-scatter matmul. Check for new additions periodically. |

### 6.8 Unsloth

Fast LLM fine-tuning with custom Triton kernels.

| Field | Value |
|:------|:------|
| **Operators** | cross_entropy, rope, rms_layernorm, swiglu (internal, tightly coupled) |
| **Repo** | https://github.com/unslothai/unsloth |
| **License** | Apache-2.0 |
| **Notes** | Kernels are embedded in model-specific code, not easily extracted as standalone. Useful for E2E training throughput comparison rather than op-level benchmarks. |

---

## 8. Output Structure & Provenance Tracking

### Directory Layout

```
benchmarks/results/{run_id}/
├── config.json              # Run configuration (shapes, baselines, HW info)
├── hardware.json            # GPU info, driver version, CUDA version
│
├── L1_single_ops/
│   ├── matmul/
│   │   ├── results.csv      # All shape × baseline × trial results
│   │   ├── arke/            # Arke-generated kernel code
│   │   │   ├── square-1k.py
│   │   │   └── ...
│   │   └── llm_direct/      # LLM-direct-written kernel code
│   │       ├── square-1k.py
│   │       └── ...
│   ├── softmax/
│   │   ├── results.csv
│   │   └── ...
│   └── .../
│
├── L2_fused_ops/
│   ├── matmul_relu/
│   │   ├── results.csv
│   │   └── ...
│   └── .../
│
├── L3_e2e/
│   ├── gpt2/
│   │   ├── results.csv
│   │   ├── config.json      # Model, seq_len, patches applied
│   │   └── ...
│   └── .../
│
├── summary.json             # Aggregated results across all layers
├── summary.csv              # Flat CSV for easy analysis
└── report.md                # Human-readable report
```

### Current Implementation ✅

Each run archives results with timestamps:

```
benchmarks/results/
├── L1/{timestamp}/
│   ├── config.json          # Run parameters
│   ├── hardware.json        # GPU/system info
│   ├── sources.json         # Baseline provenance manifest
│   ├── matmul_results.csv   # Per-op results
│   ├── softmax_results.csv
│   └── ...
├── L2/{timestamp}/
│   ├── matmul_relu_results.csv
│   └── matmul_gelu_results.csv
├── L3/{timestamp}/
│   └── gpt2_results.csv
└── report.md                # Auto-generated summary
```

### Provenance Tracking ✅

Every benchmark result carries full source attribution:

- **CSV `source` column** — each row records the baseline's package name, version, URL, and license
- **`sources.json`** — per-run manifest listing all baselines used
- **`hardware.json`** — GPU name, CUDA version, driver, PyTorch/Triton versions

Example CSV row:
```
matmul,square-1k,1024,1024,1024,FlagGems,1,"FlagGems 5.0.0 (BAAI/FlagOS) | https://github.com/flagos-ai/FlagGems | License: Apache-2.0",104.8,89.0,20.49
```

---

## 9. CLI Interface

### Current Implementation ✅

```bash
# Run everything (L1 + L2 + L3)
python -m benchmarks --all

# Run specific layer
python -m benchmarks --layer L1          # Single operator benchmarks
python -m benchmarks --layer L2          # Fused operator benchmarks
python -m benchmarks --layer L3          # E2E model benchmarks

# Run specific operators
python -m benchmarks --op matmul         # Matmul only
python -m benchmarks --op matmul,softmax # Multiple ops

# Generate report from existing results
python -m benchmarks --report

# Custom parameters
python -m benchmarks --layer L1 --warmup 200 --reps 500  # More iterations
python -m benchmarks --layer L3 --seq-len 128,256,512     # Specific seq lengths
```

### Planned CLI ⬜

```bash
# Run everything
arke bench --all

# Run specific layer
arke bench --layer L1
arke bench --layer L2
arke bench --layer L3

# Run specific operator with shapes
arke bench --op matmul --shapes square-1k,square-2k
arke bench --op softmax --shapes attn-small,attn-med

# Run specific baselines only
arke bench --op matmul --baselines cublas,flaggems,arke

# Run E2E
arke bench --model gpt2 --seq-len 128,256,512

# Compare two runs
arke bench diff {run_id_1} {run_id_2}

# Generate report
arke bench report {run_id}

# Gate verification
arke gate G0                      # Environment check
arke gate G1                      # IR + validation + Tier 3 numerical
arke gate G2                      # L1 Tier 3 full bench
arke gate G2 --tier 1             # 15 shapes quick regression
arke gate G3                      # Agent 10 shapes closed-loop
arke gate G4                      # Tier 3 comparison + token stats
arke gate G5                      # L3 multi-config E2E
arke gate --all                   # All gates (~30-60 min)

# Benchmark execution modes
arke bench --mode baselines --op matmul,softmax    # Baselines only (no LLM, ~5 min)
arke bench --mode arke --op matmul --shape square-1k  # Arke vs baselines (~10 min/op)
arke bench --mode regression                       # CI regression (cached, ~2 min)
arke bench --mode full --trials 3                  # Full suite
```

---

## 10. Benchmark-Driven Development

### Core Idea

The benchmark is not a post-hoc validation tool — it is the **target state definition for Arke development**.
Each benchmark target maps directly to an Arke capability requirement.

```
Benchmark Target (what)  →  Arke Capability (how)  →  Development Task (code)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L1 matmul ≥80% cuBLAS    →  IR+Strategy+Codegen     →  Template autotune
L1 softmax ≥80% PyTorch  →  Memory-bound template   →  softmax.py.j2
L1 layernorm ≥90% FG     →  New op in OP_CATALOG    →  layernorm.py.j2
L2 matmul+gelu fused     →  Fusion in codegen       →  Epilogue fusion
L3 GPT-2 ≤1.1× eager    →  E2E integration         →  KernelCache + patch
```

### Target-Driven Development Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. Define Target                                       │
│     "matmul [4096,4096,4096] ≥ 90% FlagGems"          │
│                                                         │
│  2. Run Benchmark (Red)                                 │
│     arke bench --op matmul --shape square-4k            │
│     → Result: 72% FlagGems  ← FAIL                     │
│                                                         │
│  3. Diagnose                                            │
│     Why? Arke tile config missing L2 swizzle for 4K     │
│                                                         │
│  4. Fix Arke                                            │
│     Add swizzle config to matmul strategy space         │
│                                                         │
│  5. Run Benchmark (Green)                               │
│     → Result: 94% FlagGems  ← PASS                     │
│                                                         │
│  6. Commit + advance to next target                     │
└─────────────────────────────────────────────────────────┘
```

### Capability Mapping

What each benchmark target demands from Arke:

| Benchmark Target | Arke IR | Template | Strategy | Integration |
|:-----------------|:--------|:---------|:---------|:------------|
| L1 matmul | `matmul` op ✅ | `matmul.py.j2` ✅ | tile, split-k, swizzle ✅ | — |
| L1 batch_matmul | `batch_matmul` op ✅ | `batch_matmul.py.j2` ⬜ | batch dim handling | — |
| L1 softmax | `softmax` op ✅ | `softmax.py.j2` ✅ | rows_per_prog ✅ | — |
| L1 layernorm | `layernorm` op ⬜ | `layernorm.py.j2` ⬜ | block_size | — |
| L1 rmsnorm | `rmsnorm` op ⬜ | `rmsnorm.py.j2` ⬜ | block_size | — |
| L1 gelu | `gelu` op ✅ | elementwise fuse ✅ | — | — |
| L1 relu | `relu` op ✅ | elementwise fuse ✅ | — | — |
| L1 rope | `rope` op ⬜ | `rope.py.j2` ⬜ | interleave/rotate | — |
| L1 cross_entropy | `cross_entropy` op ⬜ | `cross_entropy.py.j2` ⬜ | online softmax | — |
| L2 matmul+relu | fusion ✅ | epilogue in matmul ✅ | fusion decision ✅ | — |
| L2 matmul+gelu | fusion ✅ | epilogue in matmul ✅ | fusion decision ✅ | — |
| L2 linear+softmax | multi-op ⬜ | chained templates ⬜ | pipeline ⬜ | — |
| L3 GPT-2 | all above ✅ | all above ✅ | all above ✅ | KernelCache ✅ |
| L3 LLaMA | + rmsnorm, rope, swiglu ⬜ | + new templates ⬜ | all above | Model-specific patch ⬜ |

### Priority Order

```
Priority 1 (blocks L1 core)     → layernorm, gelu standalone
Priority 2 (blocks L1 extended) → rmsnorm, rope, cross_entropy
Priority 3 (blocks L2)          → epilogue fusion, batch_matmul
Priority 4 (blocks L3 LLaMA)    → swiglu, rmsnorm template
```

### Integration Points

#### 1. Arke OP_CATALOG ← Benchmark op list

Every op in the benchmark MUST exist in Arke's `OP_CATALOG`.
The benchmark runner validates this at startup:

```python
from arke.ir.ops import OP_CATALOG
for op in benchmark_ops:
    assert op in OP_CATALOG, f"Op '{op}' needed by benchmark but missing from Arke"
```

#### 2. Arke Templates ← Benchmark code requirement

Every L1 op needs a Triton template. The benchmark checks:

```python
from arke.backend.triton_template_engine import TritonTemplateEngine
engine = TritonTemplateEngine()
for op in benchmark_ops:
    assert engine.has_template(op), f"No template for '{op}'"
```

#### 3. Arke KernelCache ← L3 E2E requirement

L3 benchmarks use `KernelCache` for model patching.
New ops in L1/L2 must be added to KernelCache dispatch.

#### 4. CI Gate ← Benchmark regression mode

```yaml
# .github/workflows/bench.yml
- name: Benchmark regression
  run: arke bench --mode regression
  # Fails PR if any op regresses >5%
```

---

## 11. Implementation Status

### Baseline Runner Architecture ✅

```python
class BaselineRunner(ABC):
    """Base class for all baseline implementations."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def priority(self) -> int: ...  # P0=0, P1=1, ..., P5=5

    @abstractmethod
    def supports(self, op: str, shape: tuple) -> bool: ...

    @abstractmethod
    def run(self, op: str, inputs: dict[str, Tensor]) -> Tensor: ...

    @abstractmethod
    def bench(self, op: str, inputs: dict[str, Tensor],
              warmup: int, reps: int) -> float: ...  # returns μs
```

### Runner Implementations

| Runner | Tier | Status | Ops Supported |
|:-------|:----:|:------:|:------|
| `CuBLASRunner` | P0 | ✅ | matmul, softmax, layernorm, gelu, relu, silu, dropout |
| `FlagGemsRunner` | P1 | ✅ | matmul, softmax, layernorm, rmsnorm, gelu, relu, silu, dropout |
| `LigerRunner` | P1 | ✅ | rmsnorm, gelu, silu, rope |
| `FlashAttnRunner` | P1 | ⬜ | (planned: fused_attention) |
| `TritonTutorialRunner` | P2 | ✅ | matmul, softmax |
| `PyTorchEagerRunner` | P3 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `InductorRunner` | P4 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `LLMDirectRunner` | P5 | ⬜ | (planned: all ops via LLM codegen) |
| `ArkeRunner` | — | ✅ | matmul, softmax |

### Benchmark Components

| Component | Status | Description |
|:----------|:------:|:------------|
| `baselines/` directory | ✅ | BaselineRunner ABC + 8 runner classes |
| `shapes.py` | ✅ | 77 shapes with Tier 1/2/3 tagging and `get_shapes(op, tier=N)` |
| `measure.py` | ✅ | CUDA event timing, warmup/reps configurable |
| `bench_l1.py` | ✅ | L1 single operator benchmarks with `--tier` filtering |
| `bench_l2.py` | ✅ | L2 fused operator benchmarks |
| `bench_l3.py` | ✅ | L3 E2E model benchmarks (GPT-2) |
| `gate.py` | ✅ | Gate verification CLI (`python -m benchmarks gate G0`) |
| `cli.py` | ✅ | Unified CLI entry point (`python -m benchmarks`) |
| `report.py` | ✅ | Markdown report generator |
| Hardware info collection | ✅ | `hardware.json` per run |
| Provenance tracking | ✅ | CSV source column + `sources.json` |
| `arke bench` CLI | ⬜ | Planned unified CLI (currently `python -m benchmarks`) |
| `arke gate` CLI | ⬜ | Planned unified CLI (currently `python -m benchmarks gate`) |
| Cross-run comparison | ⬜ | `arke bench diff` |
| CI integration | ⬜ | GitHub Actions regression mode |

### Gate Implementation Status

| Gate | Status | Description |
|:-----|:------:|:------------|
| G0 | ✅ | Environment Feasibility (4 criteria, all PASS) |
| G1 | ⬜ | IR Expressiveness & Validation Correctness |
| G2 | ⬜ | Codegen Correctness & Baseline Performance |
| G3 | ⬜ | LLM Agent Autonomous Optimization |
| G4 | ⬜ | Comparative Advantage over Direct LLM |
| G5 | ⬜ | End-to-End Model Integration |

### Development Roadmap (Benchmark-Driven)

| Sprint | Focus | Status |
|:-------|:------|:------:|
| Sprint 1 | Infrastructure + L1 Core (shapes, runners, measurement) | ✅ |
| Sprint 2 | L1 Extended (Tier 2/3 shapes, TritonTutorial runner, gate CLI) | ✅ |
| Sprint 3 | L2 Fused + L3 E2E refinement | ✅ (partial — L2/L3 run but limited ops) |
| Sprint 4 | Gate G1-G5 implementation | ⬜ |
| Sprint 5 | CI + Advanced (regression, cross-run, FlashAttn, LLMDirect) | ⬜ |

---

## 12. Dependencies

### Required (installed with Arke)

```bash
pip install torch triton
```

### Benchmark Baselines

```bash
# FlagGems — 200+ Triton operators (P1)
pip install flag-gems

# Liger-Kernel — LLM training kernels (P1)
pip install liger-kernel
```

### Optional

```bash
# FlashAttention — fused attention (requires CUDA build)
pip install flash-attn --no-build-isolation

# HuggingFace community kernels
pip install triton-kernels

# CUTLASS (for C++ GEMM baselines)
pip install nvidia-cutlass
```

### Graceful Degradation

If a baseline package is not installed, the runner skips that baseline
and logs a warning. Results table shows `N/A` for missing baselines.

```python
try:
    import flag_gems
    FLAGGEMS_AVAILABLE = True
except ImportError:
    FLAGGEMS_AVAILABLE = False
    logger.warning("flag-gems not installed, P1 FlagGems baseline disabled")
```

---

*Last updated: 2026-04-04*