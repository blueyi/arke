# Arke Benchmark — Shape Tiers & Shape Matrices

Full shape definitions for all operator tiers, organized by Shape Tier (ST1–ST4).

→ Parent: [`BENCHMARK.md`](../BENCHMARK.md)

---

## Shape Tier (ST) Overview

| Tier | Name | Description | Alignment | Target Use |
|:----:|:-----|:------------|:----------|:-----------|
| **ST1** | Micro | Small, power-of-2 aligned; launch-overhead dominated | Power-of-2 | Smoke test, <30s |
| **ST2** | Standard | Medium scale + LLM production shapes (GPT-2, LLaMA-2 7B) | Mixed | Daily CI, ~5 min |
| **ST3** | Stress | Non-power-of-2, off-by-one, extreme aspect ratios | Non-aligned | Gate validation |
| **ST4** | Production | Full LLM production shapes (DeepSeek-V2/V3, LLaMA-3, Qwen2.5); long-context | Mixed | Stage 2+ Gates |

> **ST4 applies to OT2–OT4 only.** Elementwise and reduction ops at production scale
> are covered by ST3 (their shapes don't differ qualitatively from ST2 at large scale).

---

## Reference LLM Architecture Parameters

Used as ground truth for ST2/ST4 shape derivation.

| Model | Hidden | Q Heads | KV Heads | Head Dim | FFN | Vocab | Max Ctx |
|:------|-------:|--------:|---------:|---------:|----:|------:|--------:|
| GPT-2 Small | 768 | 12 | 12 | 64 | 3072 | 50257 | 1024 |
| GPT-2 Medium | 1024 | 16 | 16 | 64 | 4096 | 50257 | 1024 |
| LLaMA-2 7B | 4096 | 32 | 32 | 128 | 11008 | 32000 | 4096 |
| LLaMA-3 8B | 4096 | 32 | 8 | 128 | 14336 | 128256 | 8192 |
| DeepSeek-V2 16B | 5120 | 128 | 128 | 128 | 12288 | 102400 | 163840 |
| DeepSeek-V3 671B | 7168 | 128 | 128 | 128 | 18432 | 129280 | 163840 |
| Qwen2.5 7B | 3584 | 28 | 4 | 128 | 18944 | 151936 | 131072 |
| Mistral 7B | 4096 | 32 | 8 | 128 | 14336 | 32000 | 32768 |

---

## OT0 — Elementwise Shapes (relu, gelu, silu, add, mul)

Shape: `[M, N]` (M = batch×seq, N = feature dim)

| Tag | M | N | Tier | Source | Notes |
|:----|----:|----:|:----:|:-------|:------|
| `micro-tiny` | 32 | 64 | 1 | synthetic | Launch overhead dominated |
| `micro-small` | 128 | 256 | 1 | synthetic | Small |
| `gpt2-hidden` | 128 | 768 | 1 | GPT-2 Small | Standard hidden dim |
| `gpt2-ffn` | 128 | 3072 | 1 | GPT-2 Small | FFN intermediate |
| `square-1k` | 1024 | 1024 | 1 | classic | Balanced |
| `llama-ffn` | 512 | 11008 | 2 | LLaMA-2 7B | SiLU FFN |
| `llama-hidden` | 512 | 4096 | 2 | LLaMA-2 7B | Hidden dim |
| `llama-long` | 4096 | 4096 | 2 | LLaMA-2 7B | Long sequence |
| `ds-ffn` | 512 | 12288 | 2 | DeepSeek-V2 | FFN dim |
| `ds-ffn-large` | 512 | 18432 | 2 | DeepSeek-V3 | Large FFN |
| `qwen-ffn` | 512 | 18944 | 2 | Qwen2.5 7B | Wide FFN |
| `xlarge` | 8192 | 4096 | 2 | stress | Very large |
| `non-align-1` | 127 | 769 | 3 | stress | Off-by-one both dims |
| `non-align-2` | 1000 | 3000 | 3 | stress | Round non-power-of-2 |
| `non-align-3` | 2049 | 4097 | 3 | stress | Off-by-one from aligned |
| `non-align-4` | 333 | 11009 | 3 | stress | LLaMA FFN+1, odd batch |
| `non-align-5` | 513 | 769 | 3 | stress | GPT-2 like, off-by-one |
| `extreme-flat` | 1 | 1048576 | 3 | stress | Single-row 1M elements |
| `extreme-tall` | 65536 | 16 | 3 | stress | Many very short rows |
| `extreme-wide` | 32768 | 128 | 3 | stress | Many medium rows |

---

## OT1 — Reduction Shapes

### Softmax: `[M, N]`

| Tag | M | N | Tier | Source | Notes |
|:----|----:|----:|:----:|:-------|:------|
| `attn-gpt2-128` | 12 | 128 | 1 | GPT-2 12-head, seq=128 | Attention score |
| `attn-gpt2-256` | 12 | 256 | 1 | GPT-2 12-head, seq=256 | — |
| `attn-gpt2-512` | 12 | 512 | 1 | GPT-2 12-head, seq=512 | — |
| `attn-llama-512` | 32 | 512 | 2 | LLaMA-2 7B | — |
| `attn-llama-2k` | 32 | 2048 | 2 | LLaMA-2 7B | Max context |
| `attn-llama-4k` | 32 | 4096 | 2 | LLaMA-2 7B | Extended |
| `wide-vocab-gpt2` | 1 | 50257 | 2 | GPT-2 | Vocabulary softmax |
| `wide-vocab-llama3` | 1 | 128256 | 2 | LLaMA-3 | Large vocab |
| `batch-large` | 128 | 4096 | 2 | stress | Batch softmax |
| `square-4k` | 4096 | 4096 | 2 | stress | Large stress test |
| `non-align-1` | 13 | 513 | 3 | stress | Non-aligned head + N |
| `non-align-2` | 7 | 511 | 3 | stress | Prime heads, off-by-one N |
| `non-align-3` | 15 | 1023 | 3 | stress | Non-power-of-2 |
| `non-align-4` | 32 | 2049 | 3 | stress | Off-by-one from 2048 |
| `non-align-5` | 1 | 50261 | 3 | stress | Non-aligned vocab |
| `extreme-tiny` | 1 | 16 | 3 | stress | Minimal |
| `extreme-wide` | 1 | 1048576 | 3 | stress | 1M-wide single row |
| `extreme-tall` | 65536 | 64 | 3 | stress | Many short rows |

### LayerNorm / RMSNorm / RMSNorm-Residual: `[B, H]`

> RMSNorm-Residual adds a `residual:[B,H]` input; shape matrix is identical.

| Tag | B | H | Tier | Source | Notes |
|:----|------:|-------:|:----:|:-------|:------|
| `gpt2-small` | 128 | 768 | 1 | GPT-2 Small | Standard |
| `gpt2-ffn` | 128 | 3072 | 1 | GPT-2 Small | FFN intermediate |
| `gpt2-medium` | 128 | 1024 | 1 | GPT-2 Medium | — |
| `llama-7b` | 128 | 4096 | 2 | LLaMA-2 7B | — |
| `llama-13b` | 128 | 5120 | 2 | LLaMA-13B | — |
| `llama-long` | 2048 | 4096 | 2 | LLaMA-2 7B | Long sequence |
| `ds-v2` | 128 | 5120 | 2 | DeepSeek-V2 | — |
| `ds-v3` | 128 | 7168 | 2 | DeepSeek-V3 | Large hidden |
| `mixtral-ffn` | 128 | 14336 | 2 | Mixtral 8x7B | FFN hidden |
| `non-align-1` | 127 | 769 | 3 | stress | Non-aligned both |
| `non-align-2` | 1000 | 3000 | 3 | stress | Round non-power-of-2 |
| `non-align-3` | 333 | 4097 | 3 | stress | Off-by-one hidden |
| `non-align-4` | 2049 | 4095 | 3 | stress | Off-by-one both |
| `extreme-small` | 1 | 768 | 3 | stress | Single-sample |
| `extreme-large` | 8192 | 4096 | 3 | stress | Very long sequence |

### Reduce Sum / Max: `[M, N] → [M]`

| Tag | M | N | Tier | Notes |
|:----|----:|----:|:----:|:------|
| `small` | 128 | 768 | 1 | GPT-2 hidden |
| `medium` | 128 | 4096 | 1 | LLaMA hidden |
| `large` | 1024 | 4096 | 2 | Stress test |
| `wide` | 1 | 50257 | 2 | Vocabulary |
| `non-align-1` | 127 | 769 | 3 | Off-by-one |
| `non-align-2` | 333 | 4097 | 3 | Off-by-one hidden |
| `extreme-tall` | 65536 | 64 | 3 | Many short rows |

---

## OT2 — Compute-Dense Shapes

### Matmul: `[M, K] × [K, N] → [M, N]`

#### ST1–ST3

| Tag | M | N | K | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `tiny` | 128 | 128 | 128 | 1 | synthetic | Launch overhead |
| `gpt2-c_proj` | 128 | 768 | 768 | 1 | GPT-2 Small | c_proj |
| `gpt2-c_attn` | 128 | 2304 | 768 | 1 | GPT-2 Small | QKV projection |
| `square-1k` | 1024 | 1024 | 1024 | 1 | classic | Standard GEMM |
| `square-2k` | 2048 | 2048 | 2048 | 1 | classic | Compute-bound |
| `square-4k` | 4096 | 4096 | 4096 | 1 | classic | Large GEMM |
| `rect-wide` | 1024 | 4096 | 1024 | 2 | LLM FFN | Wide output |
| `rect-tall` | 4096 | 1024 | 1024 | 2 | LLM FFN | Tall input |
| `lm-head-gpt2` | 128 | 50257 | 768 | 2 | GPT-2 | Vocab projection |
| `llama-q` | 4096 | 4096 | 4096 | 2 | LLaMA-2 7B | Attention Q/K/V |
| `llama-ffn` | 4096 | 11008 | 4096 | 2 | LLaMA-2 7B | FFN up |
| `seq512` | 512 | 2304 | 768 | 2 | GPT-2 seq=512 | Triton sweet spot |
| `non-align-1` | 127 | 513 | 1000 | 3 | stress | Non-power-of-2 all |
| `non-align-2` | 333 | 777 | 555 | 3 | stress | Odd dimensions |
| `non-align-3` | 1023 | 1025 | 1024 | 3 | stress | Off-by-one M and N |
| `non-align-4` | 1000 | 1000 | 1000 | 3 | stress | Round non-power-of-2 |
| `non-align-5` | 384 | 640 | 1536 | 3 | stress | Non-power-of-2 real-world |
| `non-align-6` | 2049 | 2047 | 2050 | 3 | stress | Off-by-one from 2048 |
| `non-align-7` | 513 | 2305 | 769 | 3 | stress | GPT-2 shapes +1 |
| `extreme-1row` | 1 | 1024 | 1024 | 3 | stress | Single-row matmul |
| `extreme-16` | 16 | 4096 | 4096 | 3 | stress | Very small M |
| `extreme-long` | 8192 | 64 | 4096 | 3 | stress | Extreme M/N ratio |

#### ST4 — Production Scale

| Tag | M | N | K | Source | Notes |
|:----|----:|----:|----:|:-------|:------|
| `ds-v2-attn` | 512 | 5120 | 5120 | DeepSeek-V2 | Attention proj |
| `ds-v2-ffn-up` | 512 | 12288 | 5120 | DeepSeek-V2 | FFN up |
| `ds-v2-ffn-down` | 512 | 5120 | 12288 | DeepSeek-V2 | FFN down |
| `ds-v2-lmhead` | 512 | 102400 | 5120 | DeepSeek-V2 | lm_head |
| `ds-v3-attn` | 1024 | 7168 | 7168 | DeepSeek-V3 | Attention proj |
| `ds-v3-ffn-up` | 1024 | 18432 | 7168 | DeepSeek-V3 | FFN up |
| `ds-v3-lmhead` | 512 | 129280 | 7168 | DeepSeek-V3 | lm_head |
| `qwen25-attn` | 512 | 3584 | 3584 | Qwen2.5 7B | Attention |
| `qwen25-ffn-up` | 512 | 18944 | 3584 | Qwen2.5 7B | FFN up (wide) |
| `qwen25-lmhead` | 512 | 151936 | 3584 | Qwen2.5 7B | lm_head |
| `ds-v2-long-8k` | 8192 | 5120 | 5120 | DeepSeek-V2 | Long ctx seq=8K |
| `ds-v3-long-32k` | 32768 | 7168 | 7168 | DeepSeek-V3 | Long ctx seq=32K |

### Batch Matmul: `[B, M, K] × [B, K, N] → [B, M, N]`

| Tag | B | M | K | N | Tier | Source | Notes |
|:----|--:|--:|--:|--:|:----:|:-------|:------|
| `gpt2-attn-128` | 8 | 128 | 64 | 128 | 1 | GPT-2 H=8, seq=128 | Attention score |
| `gpt2-attn-512` | 12 | 512 | 64 | 512 | 1 | GPT-2 H=12, seq=512 | — |
| `llama-attn-512` | 32 | 512 | 128 | 512 | 2 | LLaMA-2 7B seq=512 | — |
| `llama-attn-2k` | 32 | 2048 | 128 | 2048 | 2 | LLaMA-2 7B seq=2k | — |
| `batched-8` | 8 | 64 | 64 | 64 | 2 | synthetic | Batched inference |
| `batched-32` | 32 | 128 | 128 | 128 | 2 | synthetic | — |
| `non-align-1` | 7 | 127 | 65 | 129 | 3 | stress | Non-aligned all |
| `non-align-2` | 15 | 513 | 129 | 513 | 3 | stress | Off-by-one |
| `extreme-bsz` | 64 | 32 | 64 | 32 | 3 | stress | Large batch, small seq |

### Grouped Matmul: `[B, M, K] × [E, K, N] × indices[B] → [B, M, N]`

| Tag | B | E | M | K | N | Tier | Source | Notes |
|:----|--:|--:|--:|--:|--:|:----:|:-------|:------|
| `moe-tiny` | 4 | 8 | 32 | 64 | 128 | 1 | synthetic | Minimal MoE |
| `moe-small` | 8 | 8 | 64 | 256 | 256 | 1 | synthetic | Small MoE |
| `moe-medium` | 16 | 8 | 128 | 768 | 3072 | 2 | GPT-2 scale MoE | — |
| `ds-moe-512` | 4 | 64 | 512 | 5120 | 5120 | 4 | DeepSeek-V2 | Expert count=64 |
| `ds-moe-2k` | 4 | 64 | 2048 | 5120 | 5120 | 4 | DeepSeek-V2 MoE | Long seq |
| `ds-v3-moe` | 4 | 256 | 512 | 7168 | 7168 | 4 | DeepSeek-V3 | Large expert count |
| `non-align-1` | 7 | 8 | 127 | 769 | 769 | 3 | stress | Non-aligned dims |

### Transpose: `[M, N] → [N, M]`

| Tag | M | N | Tier | Notes |
|:----|----:|----:|:----:|:------|
| `small` | 128 | 512 | 1 | Typical |
| `square-1k` | 1024 | 1024 | 1 | Square |
| `llama-kv` | 512 | 4096 | 2 | KV head reshape |
| `wide` | 64 | 8192 | 2 | Wide matrix |
| `non-align-1` | 127 | 513 | 3 | Off-by-one |
| `extreme-tall` | 65536 | 64 | 3 | Extreme aspect ratio |

---

## OT3 — Gated Activation Shapes (swiglu, geglu)

Input: `[seq, ffn*2]` → Output: `[seq, ffn]`

| Tag | Seq | FFN×2 | FFN | Tier | Source | Notes |
|:----|----:|------:|----:|:----:|:-------|:------|
| `gpt2-sm` | 128 | 6144 | 3072 | 1 | GPT-2 Small | GeGLU reference |
| `llama-7b-512` | 512 | 22016 | 11008 | 2 | LLaMA-2 7B | SwiGLU |
| `llama-7b-2k` | 2048 | 22016 | 11008 | 2 | LLaMA-2 7B | SwiGLU long seq |
| `llama3-8b` | 512 | 28672 | 14336 | 2 | LLaMA-3 8B | SwiGLU |
| `ds-v2-512` | 512 | 24576 | 12288 | 4 | DeepSeek-V2 | SwiGLU |
| `ds-v2-2k` | 2048 | 24576 | 12288 | 4 | DeepSeek-V2 | — |
| `ds-v3-512` | 512 | 36864 | 18432 | 4 | DeepSeek-V3 | Large FFN |
| `ds-v3-2k` | 2048 | 36864 | 18432 | 4 | DeepSeek-V3 | — |
| `qwen25-7b` | 512 | 37888 | 18944 | 4 | Qwen2.5 7B | Wide FFN |
| `non-align-1` | 127 | 6145 | 3073 | 3 | stress | Off-by-one all |
| `non-align-2` | 333 | 22017 | 11009 | 3 | stress | LLaMA FFN+1 |

---

## OT4 — Attention Shapes

dtype = f16/bf16. Long-context (≥8K) shapes may OOM on 6GB VRAM; recorded as OOM, not fail.

### FlashAttention: `[B, H, S, D]`

| Tag | B | H | S | D | Tier | Source | Notes |
|:----|--:|--:|--:|--:|:----:|:-------|:------|
| `gpt2-sm-128` | 1 | 12 | 128 | 64 | 4 | GPT-2 Small | Short context |
| `gpt2-sm-512` | 1 | 12 | 512 | 64 | 4 | GPT-2 Small | — |
| `gpt2-sm-1k` | 1 | 12 | 1024 | 64 | 4 | GPT-2 Small | Max context |
| `gpt2-md-2k` | 1 | 16 | 2048 | 64 | 4 | GPT-2 Medium | Max context |
| `llama2-7b-512` | 1 | 32 | 512 | 128 | 4 | LLaMA-2 7B | Typical |
| `llama2-7b-2k` | 1 | 32 | 2048 | 128 | 4 | LLaMA-2 7B | Max context |
| `llama2-7b-4k` | 1 | 32 | 4096 | 128 | 4 | LLaMA-2 7B | Extended |
| `llama2-7b-batch` | 4 | 32 | 512 | 128 | 4 | LLaMA-2 7B | Batched |
| `ds-v2-512` | 1 | 128 | 512 | 128 | 4 | DeepSeek-V2 | Many-head |
| `ds-v2-2k` | 1 | 128 | 2048 | 128 | 4 | DeepSeek-V2 | — |
| `ds-v2-4k` | 1 | 128 | 4096 | 128 | 4 | DeepSeek-V2 | — |
| `ds-v2-8k` | 1 | 128 | 8192 | 128 | 4 | DeepSeek-V2 | Long context |
| `ds-v2-16k` | 1 | 128 | 16384 | 128 | 4 | DeepSeek-V2 | Long context |
| `ds-v3-32k` | 1 | 128 | 32768 | 128 | 4 | DeepSeek-V3 | Ultra-long |
| `ds-v3-163k` | 1 | 128 | 163840 | 128 | 4 | DeepSeek-V3 | Max context |
| `ds-batch-8` | 8 | 128 | 512 | 128 | 4 | DeepSeek | Batched |

### Grouped Query Attention (GQA): `Q:[B,Hq,S,D], K/V:[B,Hkv,S,D]`

| Tag | B | Hq | Hkv | S | D | Tier | Source | Notes |
|:----|--:|---:|----:|--:|--:|:----:|:-------|:------|
| `llama3-8b-512` | 1 | 32 | 8 | 512 | 128 | 4 | LLaMA-3 8B | GQA 4:1 |
| `llama3-8b-2k` | 1 | 32 | 8 | 2048 | 128 | 4 | LLaMA-3 8B | — |
| `llama3-8b-8k` | 1 | 32 | 8 | 8192 | 128 | 4 | LLaMA-3 8B | Max context |
| `mistral-7b-512` | 1 | 32 | 8 | 512 | 128 | 4 | Mistral 7B | — |
| `qwen25-7b-512` | 1 | 28 | 4 | 512 | 128 | 4 | Qwen2.5 7B | 7:1 ratio |
| `qwen25-7b-2k` | 1 | 28 | 4 | 2048 | 128 | 4 | Qwen2.5 7B | — |
| `qwen25-7b-8k` | 1 | 28 | 4 | 8192 | 128 | 4 | Qwen2.5 7B | Long context |
| `ds-v2-mha-512` | 1 | 128 | 128 | 512 | 128 | 4 | DeepSeek-V2 | MHA (Hq=Hkv) |

### Multi-Latent Attention (MLA): DeepSeek-V2/V3

Shape: `Q:[B,H,S,D]`, `KV_c:[B,S,D_c]`, `W_uk/W_uv:[D_c,H,D]`

| Tag | B | H | S | D | D_c | Tier | Source | Notes |
|:----|--:|--:|--:|--:|----:|:----:|:-------|:------|
| `ds-v2-mla-512` | 1 | 128 | 512 | 128 | 512 | 4 | DeepSeek-V2 | Standard |
| `ds-v2-mla-2k` | 1 | 128 | 2048 | 128 | 512 | 4 | DeepSeek-V2 | — |
| `ds-v2-mla-4k` | 1 | 128 | 4096 | 128 | 512 | 4 | DeepSeek-V2 | — |
| `ds-v2-mla-8k` | 1 | 128 | 8192 | 128 | 512 | 4 | DeepSeek-V2 | Long context |
| `ds-v3-mla-512` | 1 | 128 | 512 | 128 | 1024 | 4 | DeepSeek-V3 | Larger latent |
| `ds-v3-mla-2k` | 1 | 128 | 2048 | 128 | 1024 | 4 | DeepSeek-V3 | — |
| `ds-v3-mla-4k` | 1 | 128 | 4096 | 128 | 1024 | 4 | DeepSeek-V3 | — |
| `ds-v3-mla-8k` | 1 | 128 | 8192 | 128 | 1024 | 4 | DeepSeek-V3 | Long context |

---

## BL6 — Model-Complete Shape Sets

BL6 captures the exact operator+shape pairs from real model forward passes,
not a Cartesian product of OT×ST.

### GPT-2 Small (BL6)

Sequence lengths: 128, 512, 1024

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| Token embedding | — | [512, 768] |
| QKV projection | matmul | [512, 768] × [768, 2304] |
| Attention score | batch_matmul | [12, 512, 64] × [12, 64, 512] |
| Attention softmax | softmax | [12, 512] |
| Attention output | batch_matmul | [12, 512, 512] × [12, 512, 64] |
| c_proj | matmul | [512, 768] × [768, 768] |
| Residual add | add | [512, 768] |
| LayerNorm | layernorm | [512, 768] |
| FFN fc1 | matmul | [512, 768] × [768, 3072] |
| FFN gelu | gelu | [512, 3072] |
| FFN fc2 | matmul | [512, 3072] × [3072, 768] |
| LM head | matmul | [512, 768] × [768, 50257] |

### LLaMA-2 7B (BL6)

Sequence lengths: 512, 2048, 4096

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| QKV projection | matmul | [512, 4096] × [4096, 4096] |
| Attention (MHA) | flash_attention | [1, 32, 512, 128] |
| Attention output | matmul | [512, 4096] × [4096, 4096] |
| Residual add | add | [512, 4096] |
| RMSNorm | rmsnorm | [512, 4096] |
| FFN gate/up proj | matmul | [512, 4096] × [4096, 11008] (×2) |
| FFN swiglu | swiglu | [512, 22016] |
| FFN down proj | matmul | [512, 11008] × [11008, 4096] |
| LM head | matmul | [512, 4096] × [4096, 32000] |

### DeepSeek-V2 16B (BL6)

Sequence lengths: 512, 2048, 8192

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| QKV projection | matmul | [512, 5120] × [5120, 5120] |
| MLA attention | multi_latent_attention | Q:[1,128,512,128], KV_c:[1,512,512], W_uk/W_uv:[512,128,128] |
| Attention output | matmul | [512, 5120] × [5120, 5120] |
| Residual add | add | [512, 5120] |
| RMSNorm (pre/post) | rmsnorm | [512, 5120] |
| MoE gate | matmul | [512, 5120] × [5120, 64] |
| Expert dispatch | grouped_matmul | [4, 512, 5120] × [64, 5120, 12288], idx:[4] |
| FFN swiglu | swiglu | [512, 24576] |
| Expert combine | grouped_matmul | [4, 512, 12288] × [64, 12288, 5120], idx:[4] |
| LM head | matmul | [512, 5120] × [5120, 102400] |

---

*Last updated: 2026-04-05*
