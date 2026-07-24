# Arke Benchmark — Shape Tiers & Shape Matrices

Full shape definitions for all operator tiers, organized by Shape Tier (ST1–ST4).

**Total: ~350 benchmark shapes across 46 operators** (OT0×12 + OT1×10 + OT2×11 + OT3×7 + OT4×5); BL6 covers 5 reference models

→ Parent: [`benchmark-design.md`](./benchmark-design.md)

---

## Shape Tier (ST) Overview

| Tier | Name | Description | Alignment | Target Use |
|:----:|:-----|:------------|:----------|:-----------|
| **ST1** | Micro | Small, power-of-2 aligned; launch-overhead dominated | Power-of-2 | Smoke test, <30s |
| **ST2** | Standard | Medium scale + LLM production shapes (GPT-2, LLaMA-2 7B, LLaMA-3 8B, Qwen2.5 7B) | Mixed | Daily CI, ~5 min |
| **ST3** | Stress | Non-power-of-2, off-by-one, extreme aspect ratios | Non-aligned | Gate validation |
| **ST4** | Production | Full LLM production shapes (DeepSeek-V2/V3, LLaMA-3, Qwen2.5); long-context | Mixed | Phase 2+ Gates |

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

## OT0 — Elementwise Shapes

**Operators covered (12):** `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `add`, `mul`, `where_`, `cast`, `neg`, `exp`, `rsqrt`

All 12 ops are shape-preserving `[M, N] → [M, N]` and share the same shape table.
- `tanh`, `sigmoid`, `neg`, `exp`, `rsqrt` — unary, same access pattern as `relu`/`gelu`/`silu`
- `add`, `mul` — binary elementwise; A/B share same shape
- `where_` — ternary: `cond:[M,N]`, `X:[M,N]`, `Y:[M,N]` → `Z:[M,N]`
- `cast` — dtype conversion; same M×N, bandwidth differs by dtype pair

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

> **`copy_` (OT2)** reuses this shape table — pure memcpy, same M×N footprint.

---

## OT1 — Reduction Shapes

**Operators covered (10):** `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max`, `reduce_mean`, `argmax`, `topk`, `cumsum`

### Softmax: `softmax(X:[M,N]) → Y:[M,N]`

> Also used by: `reduce_mean`, `argmax` (same `[M,N]→[M]` row-reduction pattern; see note below)

| Tag | M | N | Tier | Source | Notes |
|:----|----:|----:|:----:|:-------|:------|
| `attn-gpt2-128` | 12 | 128 | 1 | GPT-2 Small | Attention score (12-head, seq=128) |
| `attn-gpt2-256` | 12 | 256 | 1 | GPT-2 Small | 12-head, seq=256 |
| `attn-gpt2-512` | 12 | 512 | 1 | GPT-2 Small | 12-head, seq=512 |
| `attn-llama-512` | 32 | 512 | 2 | LLaMA-2 7B | — |
| `attn-llama-2k` | 32 | 2048 | 2 | LLaMA-2 7B | Max context |
| `attn-llama-4k` | 32 | 4096 | 2 | LLaMA-2 7B | Extended |
| `wide-vocab-gpt2` | 1 | 50257 | 2 | GPT-2 | Vocabulary softmax |
| `wide-vocab-llama3` | 1 | 128256 | 2 | LLaMA-3 8B | Large vocab |
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

#### ST4 — Production Scale (softmax / norm)

> These shapes arise in long-context inference and production-scale training batches.

| Tag | M | N | Source | Notes |
|:----|----:|----:|:-------|:------|
| `ds-v2-attn-8k` | 128 | 8192 | DeepSeek-V2 | 128 heads × seq=8K |
| `ds-v2-attn-16k` | 128 | 16384 | DeepSeek-V2 | Long context |
| `ds-v3-attn-32k` | 128 | 32768 | DeepSeek-V3 | Ultra-long context |
| `llama3-attn-8k` | 32 | 8192 | LLaMA-3 8B | Max context |
| `qwen25-attn-32k` | 28 | 32768 | Qwen2.5 7B | 7:1 GQA, long ctx |
| `wide-vocab-ds-v2` | 512 | 102400 | DeepSeek-V2 | Batched vocab softmax |
| `wide-vocab-qwen25` | 512 | 151936 | Qwen2.5 7B | Widest vocab |

### LayerNorm / RMSNorm / RMSNorm-Residual: `[B, H]`

> `rmsnorm_residual` adds a `residual:[B,H]` input; shape matrix is identical.

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

#### ST4 — Production Scale (layernorm / rmsnorm)

| Tag | B | H | Source | Notes |
|:----|------:|-------:|:-------|:------|
| `llama3-8b-norm` | 512 | 4096 | LLaMA-3 8B | Standard seq=512 |
| `llama3-8b-long` | 8192 | 4096 | LLaMA-3 8B | Max context seq=8K |
| `qwen25-7b-norm` | 512 | 3584 | Qwen2.5 7B | Standard |
| `qwen25-7b-long` | 8192 | 3584 | Qwen2.5 7B | Long context |
| `ds-v2-long` | 8192 | 5120 | DeepSeek-V2 | Long context |
| `ds-v3-long` | 4096 | 7168 | DeepSeek-V3 | Production seq |

### Reduce Sum / Max / Mean: `[M, N] → [M]`

> `reduce_sum`, `reduce_max`, `reduce_mean` all share this shape table.
> `argmax` also uses this table (same access pattern, int64 output).

| Tag | M | N | Tier | Source | Notes |
|:----|----:|----:|:----:|:-------|:------|
| `small` | 128 | 768 | 1 | GPT-2 Small | GPT-2 hidden |
| `medium` | 128 | 4096 | 1 | LLaMA-2 7B | LLaMA hidden |
| `large` | 1024 | 4096 | 2 | stress | Stress test |
| `wide` | 1 | 50257 | 2 | GPT-2 | Vocabulary |
| `llama3-wide` | 1 | 128256 | 2 | LLaMA-3 8B | Large vocab |
| `qwen25-wide` | 1 | 151936 | 2 | Qwen2.5 7B | Widest vocab |
| `non-align-1` | 127 | 769 | 3 | stress | Off-by-one |
| `non-align-2` | 333 | 4097 | 3 | stress | Off-by-one hidden |
| `non-align-3` | 2049 | 11009 | 3 | stress | LLaMA FFN+1, long seq |
| `extreme-tall` | 65536 | 64 | 3 | stress | Many short rows |
| `extreme-flat` | 1 | 1048576 | 3 | stress | 1M single-row reduction |

### TopK: `topk(X:[M,N], k) → (values:[M,k], indices:[M,k])`

> Key variable: `k`. Performance depends on both N (row width) and k (output size).

| Tag | M | N | k | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `moe-top2-small` | 128 | 64 | 2 | 1 | synthetic | MoE top-2 routing, small |
| `moe-top2-gpt2` | 512 | 768 | 2 | 1 | GPT-2 scale | MoE top-2 gate logits |
| `moe-top4-llama` | 512 | 64 | 4 | 2 | LLaMA-2 7B scale | MoE top-4, E=64 |
| `moe-top2-ds` | 512 | 256 | 2 | 2 | DeepSeek-V2 | Top-2, E=256 |
| `sampling-top50` | 1 | 50257 | 50 | 2 | GPT-2 | Top-50 sampling |
| `sampling-top50-llama3` | 1 | 128256 | 50 | 2 | LLaMA-3 8B | Large vocab top-50 |
| `sampling-top100-qwen` | 1 | 151936 | 100 | 2 | Qwen2.5 7B | Widest vocab top-100 |
| `non-align-1` | 127 | 769 | 3 | 3 | stress | Non-aligned, small k |
| `non-align-2` | 333 | 4097 | 7 | 3 | stress | Off-by-one, prime k |
| `non-align-3` | 513 | 50261 | 51 | 3 | stress | Non-aligned vocab |
| `extreme-k1` | 1024 | 4096 | 1 | 3 | stress | k=1 degenerates to argmax |
| `extreme-kmax` | 512 | 1024 | 512 | 3 | stress | k=N/2, large output |
| `extreme-wide` | 1 | 1048576 | 10 | 3 | stress | 1M-wide, small k |

### CumSum: `cumsum(X:[M,N], dim) → Y:[M,N]`

> Shape-preserving but `dim` parameter changes scan axis and parallelism strategy.
> `dim=1` (row-scan) is the typical LLM use case; `dim=0` (col-scan) tests a different reduction direction.

| Tag | M | N | dim | Tier | Source | Notes |
|:----|----:|----:|:---:|:----:|:-------|:------|
| `small-row` | 128 | 512 | 1 | 1 | synthetic | Row-scan, small |
| `gpt2-row` | 128 | 768 | 1 | 1 | GPT-2 Small | Hidden dim row-scan |
| `llama-row` | 512 | 4096 | 1 | 2 | LLaMA-2 7B | Hidden row-scan |
| `llama-col` | 512 | 4096 | 0 | 2 | LLaMA-2 7B | Column-scan |
| `sampling-probs` | 1 | 50257 | 1 | 2 | GPT-2 | Nucleus sampling cum-probs |
| `sampling-probs-llama3` | 1 | 128256 | 1 | 2 | LLaMA-3 8B | Large vocab |
| `non-align-1` | 127 | 769 | 1 | 3 | stress | Off-by-one row-scan |
| `non-align-2` | 333 | 4097 | 0 | 3 | stress | Off-by-one col-scan |
| `non-align-3` | 2049 | 50261 | 1 | 3 | stress | Non-aligned vocab-size |
| `extreme-long-row` | 1 | 1048576 | 1 | 3 | stress | 1M element prefix sum |
| `extreme-tall-col` | 65536 | 64 | 0 | 3 | stress | 64K-depth col-scan |

---

## OT2 — Data Movement & Dense Shapes

**Operators covered (11):** `matmul`, `batch_matmul`, `grouped_matmul`, `transpose`, `concat`, `split`, `gather`, `scatter`, `embedding`, `permute`, `copy_`

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

### Concat: `concat([X₁:[M,N₁], X₂:[M,N₂], ...], dim) → [M, N₁+N₂+...]`


> `dim=1` (concat along feature dim) is the dominant LLM pattern (QKV merge, KV cache append).
> For benchmarking, shape is represented as total output `[M, N_total]` with the split noted.

| Tag | M | N₁ | N₂ | N_total | Tier | Source | Notes |
|:----|----:|----:|----:|--------:|:----:|:-------|:------|
| `tiny-2` | 64 | 64 | 64 | 128 | 1 | synthetic | 2-tensor concat |
| `gpt2-qkv-merge` | 128 | 768 | 1536 | 2304 | 1 | GPT-2 Small | Q+KV merge |
| `llama-qkv-merge` | 512 | 4096 | 4096 | 8192 | 2 | LLaMA-2 7B | Q+KV merge |
| `llama3-qkv-gqa` | 512 | 4096 | 1024 | 5120 | 2 | LLaMA-3 8B | Q(32h)+KV(8h) merge |
| `kvcache-append-512` | 512 | 4096 | 512 | 4608 | 2 | LLaMA-2 7B | KV cache append |
| `kvcache-append-2k` | 2048 | 4096 | 2048 | 6144 | 2 | LLaMA-2 7B | KV cache grow |
| `batch-concat-4` | 512 | 1024 | 1024 | 4096 | 2 | synthetic | 4-tensor concat |
| `non-align-1` | 127 | 769 | 769 | 1538 | 3 | stress | Off-by-one dims |
| `non-align-2` | 333 | 4097 | 4097 | 8194 | 3 | stress | Non-power-of-2 |
| `extreme-many-small` | 1024 | 64 | 64 | 512 | 3 | stress | 8 tiny tensors concat |
| `extreme-unbalanced` | 512 | 32768 | 128 | 32896 | 3 | stress | Extreme size imbalance |


### Split: `split(X:[M,N], sizes, dim) → [Y₁:[M,N₁], Y₂:[M,N₂], ...]`

> Inverse of concat. Dominant use: QKV split after projection (N→Q,K,V chunks).

| Tag | M | N | N₁ | N₂ | Tier | Source | Notes |
|:----|----:|----:|----:|----:|:----:|:-------|:------|
| `tiny-2` | 64 | 128 | 64 | 64 | 1 | synthetic | Equal split |
| `gpt2-qkv-split` | 128 | 2304 | 768 | 1536 | 1 | GPT-2 Small | Q / KV split |
| `llama-qkv-split` | 512 | 8192 | 4096 | 4096 | 2 | LLaMA-2 7B | Q / KV equal |
| `llama3-qkv-gqa` | 512 | 5120 | 4096 | 1024 | 2 | LLaMA-3 8B | Q(32h) / KV(8h) |
| `qwen25-qkv-gqa` | 512 | 4608 | 3584 | 1024 | 2 | Qwen2.5 7B | Q(28h) / KV(4h) |
| `ffn-gate-up` | 512 | 22016 | 11008 | 11008 | 2 | LLaMA-2 7B | SwiGLU gate/up |
| `non-align-1` | 127 | 769 | 385 | 384 | 3 | stress | Unequal odd split |
| `non-align-2` | 333 | 4097 | 2049 | 2048 | 3 | stress | Off-by-one unequal |
| `extreme-many` | 512 | 4096 | 512 | 512 | 3 | stress | 8-way equal split |


### Gather: `gather(X:[M,N], indices:[M,K]) → Y:[M,K]`

> Key dimension: K (index width). Used for MoE token dispatch and KV cache read.
> M = tokens/batch, N = source dim, K = gather width (# of selected elements per row).

| Tag | M | N | K | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `tiny` | 64 | 256 | 32 | 1 | synthetic | Small gather |
| `moe-dispatch-top2` | 512 | 4096 | 2 | 1 | LLaMA-2 7B | MoE top-2 dispatch |
| `moe-dispatch-top4` | 512 | 4096 | 4 | 2 | LLaMA-2 7B | Top-4 dispatch |
| `kvcache-read-256` | 256 | 4096 | 64 | 2 | LLaMA-2 7B | KV cache read, seq=64 |
| `kvcache-read-2k` | 2048 | 4096 | 128 | 2 | LLaMA-2 7B | KV cache read |
| `ds-moe-top2` | 512 | 5120 | 2 | 2 | DeepSeek-V2 | MoE E=64 top-2 |
| `embed-lookup-512` | 512 | 128256 | 1 | 2 | LLaMA-3 8B | Token embedding lookup |
| `non-align-1` | 127 | 769 | 3 | 3 | stress | Non-aligned, small K |
| `non-align-2` | 333 | 4097 | 7 | 3 | stress | Off-by-one N, prime K |
| `extreme-k-large` | 512 | 4096 | 1024 | 3 | stress | Large gather width |
| `extreme-m-large` | 65536 | 64 | 4 | 3 | stress | Many tokens, small N |

### Scatter: `scatter(X:[M,N], indices:[M,K], src:[M,K]) → Y:[M,N]`

> Inverse of gather. Used for MoE expert output combine and KV cache write.

| Tag | M | N | K | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `tiny` | 64 | 256 | 32 | 1 | synthetic | Small scatter |
| `moe-combine-top2` | 512 | 4096 | 2 | 1 | LLaMA-2 7B | MoE top-2 combine |
| `moe-combine-top4` | 512 | 4096 | 4 | 2 | LLaMA-2 7B | Top-4 combine |
| `kvcache-write-256` | 256 | 4096 | 64 | 2 | LLaMA-2 7B | KV cache write |
| `kvcache-write-2k` | 2048 | 4096 | 128 | 2 | LLaMA-2 7B | — |
| `ds-moe-combine` | 512 | 5120 | 2 | 2 | DeepSeek-V2 | Expert combine |
| `non-align-1` | 127 | 769 | 3 | 3 | stress | Non-aligned |
| `non-align-2` | 333 | 4097 | 7 | 3 | stress | Off-by-one |
| `extreme-k-large` | 512 | 4096 | 1024 | 3 | stress | Dense scatter |
| `extreme-atomic` | 65536 | 64 | 4 | 3 | stress | High atomic contention |


### Embedding: `embedding(W:[V,D], idx:[B,S]) → Y:[B,S,D]`

> Table lookup — performance driven by V (vocab size, cache pressure), D (hidden dim, bandwidth), and S (seq len).
> Key distinction from matmul: access is index-driven (sparse rows), not tiled dense compute.

| Tag | V | D | B | S | Tier | Source | Notes |
|:----|----:|----:|--:|--:|:----:|:-------|:------|
| `tiny` | 1024 | 64 | 1 | 32 | 1 | synthetic | Small vocab, short seq |
| `gpt2-small` | 50257 | 768 | 1 | 128 | 1 | GPT-2 Small | Standard |
| `gpt2-medium` | 50257 | 1024 | 1 | 512 | 1 | GPT-2 Medium | — |
| `llama2-seq512` | 32000 | 4096 | 1 | 512 | 2 | LLaMA-2 7B | — |
| `llama2-seq2k` | 32000 | 4096 | 1 | 2048 | 2 | LLaMA-2 7B | — |
| `llama3-seq512` | 128256 | 4096 | 1 | 512 | 2 | LLaMA-3 8B | Large vocab |
| `llama3-seq2k` | 128256 | 4096 | 1 | 2048 | 2 | LLaMA-3 8B | — |
| `qwen25-seq512` | 151936 | 3584 | 1 | 512 | 2 | Qwen2.5 7B | Widest vocab |
| `batched-b4` | 50257 | 768 | 4 | 512 | 2 | GPT-2 | Batch inference |
| `non-align-1` | 50261 | 769 | 1 | 127 | 3 | stress | Non-aligned V and D |
| `non-align-2` | 32003 | 4097 | 1 | 333 | 3 | stress | Off-by-one all |
| `extreme-large-vocab` | 151936 | 3584 | 4 | 2048 | 3 | stress | Max vocab, batched |
| `extreme-short-seq` | 128256 | 4096 | 64 | 1 | 3 | stress | Decode: 1 token, large batch |

### Permute: `permute(X:[d₀,d₁,...], order) → Y:[reordered]`

> Dominant use: multi-head reshape in attention (`[B,S,H,D]↔[B,H,S,D]`) and
> KV cache layout transforms. Shape table captures the most frequent 3D/4D patterns.

| Tag | Shape (input) | Order | Output | Tier | Source | Notes |
|:----|:-------------|:------|:-------|:----:|:-------|:------|
| `small-3d` | [4, 64, 128] | 0,2,1 | [4,128,64] | 1 | synthetic | Swap last 2 dims |
| `gpt2-bhsd` | [1,128,12,64] | 0,2,1,3 | [1,12,128,64] | 1 | GPT-2 | [B,S,H,D]→[B,H,S,D] |
| `gpt2-bshd` | [1,12,128,64] | 0,2,1,3 | [1,128,12,64] | 1 | GPT-2 | [B,H,S,D]→[B,S,H,D] |
| `llama-bhsd-512` | [1,512,32,128] | 0,2,1,3 | [1,32,512,128] | 2 | LLaMA-2 7B | Standard permute |
| `llama-bhsd-2k` | [1,2048,32,128] | 0,2,1,3 | [1,32,2048,128] | 2 | LLaMA-2 7B | Longer seq |
| `llama3-gqa-512` | [1,512,32,128] | 0,2,1,3 | [1,32,512,128] | 2 | LLaMA-3 8B | Q-head permute |
| `llama3-kv-512` | [1,512,8,128] | 0,2,1,3 | [1,8,512,128] | 2 | LLaMA-3 8B | KV GQA permute |
| `qwen25-q` | [1,512,28,128] | 0,2,1,3 | [1,28,512,128] | 2 | Qwen2.5 7B | Q permute |
| `qwen25-kv` | [1,512,4,128] | 0,2,1,3 | [1,4,512,128] | 2 | Qwen2.5 7B | KV permute |
| `non-align-1` | [1,127,13,64] | 0,2,1,3 | [1,13,127,64] | 3 | stress | Non-aligned S and H |
| `non-align-2` | [2,333,15,65] | 0,2,1,3 | [2,15,333,65] | 3 | stress | Off-by-one D |
| `extreme-5d` | [2,32,16,8,64] | 0,1,3,2,4 | [2,32,8,16,64] | 3 | stress | 5D tensor swap |


---

## OT3 — Fused Compound Shapes

**Operators covered (7):** `silu_and_mul`, `gelu_and_mul`, `rope`, `fused_linear_cross_entropy`, `cross_entropy`, `quantize_per_token`, `dequantize_per_channel`

### SiLU-and-Mul / GELU-and-Mul: `[B, 2H] → [B, H]`

| Tag | Seq | FFN×2 | FFN | Tier | Source | Notes |
|:----|----:|------:|----:|:----:|:-------|:------|
| `gpt2-sm` | 128 | 6144 | 3072 | 1 | GPT-2 Small | GELU-and-Mul reference |
| `llama-7b-512` | 512 | 22016 | 11008 | 2 | LLaMA-2 7B | SwiGLU |
| `llama-7b-2k` | 2048 | 22016 | 11008 | 2 | LLaMA-2 7B | SwiGLU long seq |
| `llama3-8b` | 512 | 28672 | 14336 | 2 | LLaMA-3 8B | SwiGLU |
| `ds-v2-512` | 512 | 24576 | 12288 | 4 | DeepSeek-V2 | SwiGLU |
| `ds-v2-2k` | 2048 | 24576 | 12288 | 4 | DeepSeek-V2 | — |
| `ds-v3-512` | 512 | 36864 | 18432 | 4 | DeepSeek-V3 | Large FFN |
| `ds-v3-2k` | 2048 | 36864 | 18432 | 4 | DeepSeek-V3 | — |
| `qwen25-7b` | 512 | 37888 | 18944 | 4 | Qwen2.5 7B | Wide FFN |
| `qwen25-7b-2k` | 2048 | 37888 | 18944 | 4 | Qwen2.5 7B | Long seq |
| `non-align-1` | 127 | 6145 | 3073 | 3 | stress | Off-by-one all |
| `non-align-2` | 333 | 22017 | 11009 | 3 | stress | LLaMA FFN+1 |

### RoPE: `rope(Q:[B,H,S,D], K:[B,H,S,D], pos:[S]) → (Q':[B,H,S,D], K':[B,H,S,D])`

> Shape parameters: B=batch, H=heads, S=seq_len, D=head_dim.
> Performance scales with H×S×D (total elements rotated).
> GQA models apply RoPE to Q (Hq heads) and K (Hkv heads) separately.

| Tag | B | H | S | D | Tier | Source | Notes |
|:----|--:|--:|--:|--:|:----:|:-------|:------|
| `gpt2-sm-128` | 1 | 12 | 128 | 64 | 1 | GPT-2 Small | Short seq |
| `gpt2-sm-512` | 1 | 12 | 512 | 64 | 1 | GPT-2 Small | — |
| `llama2-512` | 1 | 32 | 512 | 128 | 2 | LLaMA-2 7B | MHA |
| `llama2-2k` | 1 | 32 | 2048 | 128 | 2 | LLaMA-2 7B | Max context |
| `llama3-q-512` | 1 | 32 | 512 | 128 | 2 | LLaMA-3 8B | Q-heads (GQA) |
| `llama3-kv-512` | 1 | 8 | 512 | 128 | 2 | LLaMA-3 8B | KV-heads (GQA) |
| `qwen25-q-512` | 1 | 28 | 512 | 128 | 2 | Qwen2.5 7B | Q 28-heads |
| `qwen25-kv-512` | 1 | 4 | 512 | 128 | 2 | Qwen2.5 7B | KV 4-heads |
| `ds-v2-512` | 1 | 128 | 512 | 128 | 4 | DeepSeek-V2 | Many-head MHA |
| `ds-v3-2k` | 1 | 128 | 2048 | 128 | 4 | DeepSeek-V3 | — |
| `llama3-8k` | 1 | 32 | 8192 | 128 | 4 | LLaMA-3 8B | Max context |
| `qwen25-32k` | 1 | 28 | 32768 | 128 | 4 | Qwen2.5 7B | Long context |
| `non-align-1` | 1 | 13 | 127 | 65 | 3 | stress | Non-aligned H, S, D |
| `non-align-2` | 2 | 15 | 333 | 127 | 3 | stress | Off-by-one D |
| `extreme-long` | 1 | 32 | 65536 | 128 | 3 | stress | Extreme long context |
| `extreme-batch` | 8 | 32 | 512 | 128 | 3 | stress | Batch RoPE |


### Cross Entropy: `cross_entropy(logits:[B*S,V], target:[B*S]) → loss:scalar`

> Key dimension: V (vocab size). Memory-bound at large V; compute-bound at large B*S.

| Tag | B*S | V | Tier | Source | Notes |
|:----|----:|----:|:----:|:-------|:------|
| `tiny` | 64 | 1024 | 1 | synthetic | Small vocab |
| `gpt2-seq128` | 128 | 50257 | 1 | GPT-2 Small | Standard |
| `gpt2-seq512` | 512 | 50257 | 1 | GPT-2 Small | — |
| `llama2-seq512` | 512 | 32000 | 2 | LLaMA-2 7B | — |
| `llama2-seq2k` | 2048 | 32000 | 2 | LLaMA-2 7B | Long seq |
| `llama3-seq512` | 512 | 128256 | 2 | LLaMA-3 8B | Large vocab |
| `qwen25-seq512` | 512 | 151936 | 2 | Qwen2.5 7B | Widest vocab |
| `ds-v2-seq512` | 512 | 102400 | 2 | DeepSeek-V2 | — |
| `batch4-seq512` | 2048 | 50257 | 2 | GPT-2 | B=4 training |
| `non-align-1` | 127 | 50261 | 3 | stress | Non-aligned |
| `non-align-2` | 333 | 128259 | 3 | stress | Off-by-one vocab |
| `extreme-large-vocab` | 512 | 151936 | 3 | stress | Widest vocab stress |
| `extreme-large-bs` | 8192 | 32000 | 3 | stress | Large batch×seq |

### Fused Linear Cross Entropy: `fused_linear_ce(X:[B*S,H], W:[V,H], target:[B*S]) → (loss, logits_grad)`

> Fused lm_head+CE (Liger Kernel style). Key dims: H (hidden) and V (vocab).
> Memory savings vs unfused: ~60% (no materialized logits tensor at V×B*S).
> ST4 needed because V=102K–152K shapes are the primary motivation for fusion.

| Tag | B*S | H | V | Tier | Source | Notes |
|:----|----:|----:|----:|:----:|:-------|:------|
| `gpt2-seq128` | 128 | 768 | 50257 | 1 | GPT-2 Small | Standard |
| `gpt2-seq512` | 512 | 768 | 50257 | 1 | GPT-2 Small | — |
| `llama2-seq512` | 512 | 4096 | 32000 | 2 | LLaMA-2 7B | — |
| `llama2-seq2k` | 2048 | 4096 | 32000 | 2 | LLaMA-2 7B | Long seq |
| `llama3-seq512` | 512 | 4096 | 128256 | 2 | LLaMA-3 8B | Large vocab |
| `llama3-seq2k` | 2048 | 4096 | 128256 | 4 | LLaMA-3 8B | Long seq, large vocab |
| `qwen25-seq512` | 512 | 3584 | 151936 | 4 | Qwen2.5 7B | Max vocab |
| `qwen25-seq2k` | 2048 | 3584 | 151936 | 4 | Qwen2.5 7B | ⚠️ may OOM 6GB |
| `ds-v2-seq512` | 512 | 5120 | 102400 | 4 | DeepSeek-V2 | Large vocab |
| `non-align-1` | 127 | 769 | 50261 | 3 | stress | Non-aligned |
| `non-align-2` | 333 | 4097 | 128259 | 3 | stress | Off-by-one vocab |
| `extreme-large-vocab` | 512 | 3584 | 151936 | 3 | stress | Widest vocab |


### Quantize per Token: `quantize_per_token(X:[M,N], dtype) → (X_q:[M,N], scale:[M])`

> One scale per row (token). Performance depends on N (channels per token) and dtype target.
> Typical targets: INT8 (scale=fp32), FP8 (scale=fp32), INT4 (with group_size).

| Tag | M | N | dtype | Tier | Source | Notes |
|:----|----:|----:|:------|:----:|:-------|:------|
| `tiny-int8` | 64 | 256 | int8 | 1 | synthetic | Small INT8 |
| `gpt2-int8` | 128 | 768 | int8 | 1 | GPT-2 Small | Standard INT8 |
| `llama2-int8-512` | 512 | 4096 | int8 | 2 | LLaMA-2 7B | Activation quant |
| `llama2-fp8-512` | 512 | 4096 | fp8 | 2 | LLaMA-2 7B | FP8 quant |
| `llama3-int8-512` | 512 | 4096 | int8 | 2 | LLaMA-3 8B | — |
| `ds-v2-int8` | 512 | 5120 | int8 | 2 | DeepSeek-V2 | — |
| `ds-v3-int8` | 512 | 7168 | int8 | 2 | DeepSeek-V3 | — |
| `qwen25-fp8` | 512 | 3584 | fp8 | 2 | Qwen2.5 7B | FP8 inference |
| `non-align-1` | 127 | 769 | int8 | 3 | stress | Non-aligned channels |
| `non-align-2` | 333 | 4097 | int8 | 3 | stress | Off-by-one |
| `extreme-wide` | 512 | 18944 | int8 | 3 | stress | Widest FFN |
| `extreme-long-batch` | 8192 | 4096 | int8 | 3 | stress | Long sequence quant |

### Dequantize per Channel: `dequantize_per_channel(X_q:[M,N], scale:[N], zero_point:[N]) → Y:[M,N]`

> One scale+zero per column (channel/output neuron). Typical for weight dequant (INT4/INT8 weights).
> N = out_features (weight columns), M = tokens.

| Tag | M | N | src_dtype | Tier | Source | Notes |
|:----|----:|----:|:----------|:----:|:-------|:------|
| `tiny-int8` | 64 | 256 | int8 | 1 | synthetic | Small dequant |
| `gpt2-int8` | 128 | 768 | int8 | 1 | GPT-2 Small | Weight dequant |
| `llama2-int4-ffn` | 512 | 11008 | int4 | 2 | LLaMA-2 7B | INT4 FFN weight |
| `llama2-int8-hidden` | 512 | 4096 | int8 | 2 | LLaMA-2 7B | — |
| `llama3-int4-ffn` | 512 | 14336 | int4 | 2 | LLaMA-3 8B | INT4 FFN |
| `ds-v2-int4` | 512 | 12288 | int4 | 2 | DeepSeek-V2 | INT4 expert weight |
| `ds-v3-int4` | 512 | 18432 | int4 | 2 | DeepSeek-V3 | — |
| `qwen25-int4-ffn` | 512 | 18944 | int4 | 2 | Qwen2.5 7B | Wide FFN |
| `non-align-1` | 127 | 769 | int8 | 3 | stress | Non-aligned |
| `non-align-2` | 333 | 4097 | int4 | 3 | stress | Off-by-one channels |
| `extreme-ffn-large` | 512 | 18944 | int4 | 3 | stress | Widest FFN dequant |
| `extreme-lmhead` | 512 | 151936 | int4 | 3 | stress | Vocab dequant |


---

## OT4 — Attention Shapes

**Operators covered (5):** `flash_attention`, `grouped_query_attention`, `multi_latent_attention`, `cross_attention`, `paged_attention`

dtype = f16/bf16. Long-context (≥8K) shapes may OOM on 6GB VRAM; recorded as OOM, not fail.

### Flash Attention: `flash_attention(Q:[B,H,S,D], K:[B,H,S,D], V:[B,H,S,D]) → Y:[B,H,S,D]`

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
| `ds-v3-163k` | 1 | 128 | 163840 | 128 | 4 | DeepSeek-V3 | ⚠️ OOM 6GB |
| `ds-batch-8` | 8 | 128 | 512 | 128 | 4 | DeepSeek | Batched |

### Grouped Query Attention (GQA): `gqa(Q:[B,Hq,S,D], K:[B,Hkv,S,D], V:[B,Hkv,S,D]) → Y:[B,Hq,S,D]`

| Tag | B | Hq | Hkv | S | D | Tier | Source | Notes |
|:----|--:|---:|----:|--:|--:|:----:|:-------|:------|
| `llama3-8b-512` | 1 | 32 | 8 | 512 | 128 | 4 | LLaMA-3 8B | GQA 4:1 |
| `llama3-8b-2k` | 1 | 32 | 8 | 2048 | 128 | 4 | LLaMA-3 8B | — |
| `llama3-8b-8k` | 1 | 32 | 8 | 8192 | 128 | 4 | LLaMA-3 8B | Max context |
| `mistral-7b-512` | 1 | 32 | 8 | 512 | 128 | 4 | Mistral 7B | — |
| `qwen25-7b-512` | 1 | 28 | 4 | 512 | 128 | 4 | Qwen2.5 7B | 7:1 ratio |
| `qwen25-7b-2k` | 1 | 28 | 4 | 2048 | 128 | 4 | Qwen2.5 7B | — |
| `qwen25-7b-8k` | 1 | 28 | 4 | 8192 | 128 | 4 | Qwen2.5 7B | Long context |
| `qwen25-7b-32k` | 1 | 28 | 4 | 32768 | 128 | 4 | Qwen2.5 7B | ⚠️ may OOM |
| `ds-v2-mha-512` | 1 | 128 | 128 | 512 | 128 | 4 | DeepSeek-V2 | MHA (Hq=Hkv) |
| `llama3-batch4` | 4 | 32 | 8 | 512 | 128 | 4 | LLaMA-3 8B | Batched GQA |

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


### Cross Attention: `cross_attention(Q:[B,Hq,Sq,D], K:[B,Hk,Skv,D], V:[B,Hk,Skv,D]) → Y:[B,Hq,Sq,D]`

> Key distinction: `Sq ≠ Skv` (query seq ≠ key/value seq).
> Used in encoder-decoder models (T5, Whisper, BART) and diffusion (UNet cross-attn).

| Tag | B | Hq | Sq | Skv | D | Tier | Source | Notes |
|:----|--:|---:|---:|----:|--:|:----:|:-------|:------|
| `tiny` | 1 | 4 | 32 | 64 | 64 | 4 | synthetic | Minimal |
| `whisper-enc-dec` | 1 | 8 | 1500 | 1500 | 64 | 4 | Whisper | Audio encoder→decoder |
| `whisper-decode-1` | 1 | 8 | 1 | 1500 | 64 | 4 | Whisper | Autoregressive decode |
| `t5-small-enc128` | 1 | 8 | 128 | 128 | 64 | 4 | T5-Small | Encoder context |
| `t5-small-dec128` | 1 | 8 | 128 | 512 | 64 | 4 | T5-Small | Decoder→encoder |
| `t5-base-dec512` | 1 | 12 | 512 | 512 | 64 | 4 | T5-Base | — |
| `sdxl-unet-64` | 2 | 8 | 4096 | 77 | 64 | 4 | SDXL UNet | Image→text cross-attn |
| `sdxl-unet-128` | 2 | 8 | 16384 | 77 | 64 | 4 | SDXL UNet | High-res level |
| `llava-vision` | 1 | 32 | 576 | 576 | 128 | 4 | LLaVA | Visual tokens |
| `batch4-cross` | 4 | 8 | 512 | 512 | 64 | 4 | synthetic | Batched |
| `non-align-1` | 1 | 7 | 127 | 513 | 65 | 4 | stress | Non-aligned Sq, Skv, D |
| `non-align-2` | 2 | 13 | 333 | 1025 | 63 | 4 | stress | Off-by-one all |

### Paged Attention: `paged_attention(Q:[B,H,1,D], block_tables:[B,max_blk], kv_cache:[num_blk,2,H,blk_sz,D]) → Y:[B,H,1,D]`

> Decode-phase attention (single query token per request) over paged KV cache (vLLM-style).
> Key variables: context length (num_blocks × block_size), H, D, block_size.
> Measured latency captures: block table lookup + scattered KV reads + attention.

| Tag | B | H | D | blk_sz | context_len | Tier | Source | Notes |
|:----|--:|--:|--:|-------:|------------:|:----:|:-------|:------|
| `tiny` | 1 | 4 | 64 | 16 | 256 | 4 | synthetic | Minimal paged |
| `llama2-ctx512` | 1 | 32 | 128 | 16 | 512 | 4 | LLaMA-2 7B | Short context |
| `llama2-ctx2k` | 1 | 32 | 128 | 16 | 2048 | 4 | LLaMA-2 7B | Standard |
| `llama2-ctx4k` | 1 | 32 | 128 | 16 | 4096 | 4 | LLaMA-2 7B | Full context |
| `llama3-ctx8k` | 1 | 32 | 128 | 16 | 8192 | 4 | LLaMA-3 8B | Max context |
| `llama3-ctx8k-gqa` | 1 | 8 | 128 | 16 | 8192 | 4 | LLaMA-3 8B | KV-heads (GQA) |
| `qwen25-ctx32k` | 1 | 4 | 128 | 16 | 32768 | 4 | Qwen2.5 7B | Long context KV |
| `vllm-batch16` | 16 | 32 | 128 | 16 | 2048 | 4 | vLLM | Multi-request batch |
| `vllm-batch32-ctx4k` | 32 | 32 | 128 | 16 | 4096 | 4 | vLLM | ⚠️ may OOM |
| `blksize32-ctx4k` | 1 | 32 | 128 | 32 | 4096 | 4 | vLLM | Larger block size |
| `non-align-1` | 1 | 13 | 65 | 16 | 512 | 4 | stress | Non-aligned H, D |
| `non-align-2` | 3 | 15 | 127 | 16 | 1024 | 4 | stress | Non-aligned B, H, D |


---

## BL6 — Model-Complete Shape Sets

BL6 captures the exact operator+shape pairs from real model forward passes,
not a Cartesian product of OT×ST.

### GPT-2 Small (BL6)

Sequence lengths: 128, 512, 1024

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| Token embedding | embedding | W:[50257,768], idx:[1,512] → [1,512,768] |
| QKV projection | matmul | [512,768] × [768,2304] |
| Attention score | batch_matmul | [12,512,64] × [12,64,512] |
| Attention softmax | softmax | [12,512] |
| Attention output | batch_matmul | [12,512,512] × [12,512,64] |
| c_proj | matmul | [512,768] × [768,768] |
| Residual add | add | [512,768] |
| LayerNorm | layernorm | [512,768] |
| FFN fc1 | matmul | [512,768] × [768,3072] |
| FFN gelu | gelu | [512,3072] |
| FFN fc2 | matmul | [512,3072] × [3072,768] |
| LM head | matmul | [512,768] × [768,50257] |
| CE loss | cross_entropy | [512,50257] |

### LLaMA-2 7B (BL6)

Sequence lengths: 512, 2048, 4096

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| Token embedding | embedding | W:[32000,4096], idx:[1,512] → [1,512,4096] |
| QKV projection | matmul | [512,4096] × [4096,4096] |
| RoPE | rope | Q/K:[1,32,512,128], pos:[512] |
| Attention (MHA) | flash_attention | [1,32,512,128] |
| Attention output | matmul | [512,4096] × [4096,4096] |
| Residual add | add | [512,4096] |
| RMSNorm | rmsnorm | [512,4096] |
| FFN gate/up proj | matmul | [512,4096] × [4096,11008] (×2) |
| FFN silu_and_mul | silu_and_mul | [512,22016] |
| FFN down proj | matmul | [512,11008] × [11008,4096] |
| LM head | matmul | [512,4096] × [4096,32000] |
| CE loss | cross_entropy | [512,32000] |

### LLaMA-3 8B (BL6)

Sequence lengths: 512, 2048, 8192. **GQA: Hq=32, Hkv=8.**

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| Token embedding | embedding | W:[128256,4096], idx:[1,512] → [1,512,4096] |
| Q projection | matmul | [512,4096] × [4096,4096] |
| KV projection | matmul | [512,4096] × [4096,1024] |
| QKV permute | permute | Q:[1,512,32,128]→[1,32,512,128], KV:[1,512,8,128]→[1,8,512,128] |
| RoPE | rope | Q:[1,32,512,128], K:[1,8,512,128], pos:[512] |
| Attention (GQA) | grouped_query_attention | Q:[1,32,512,128], K/V:[1,8,512,128] |
| Attention output | matmul | [512,4096] × [4096,4096] |
| Residual add | add | [512,4096] |
| RMSNorm | rmsnorm | [512,4096] |
| FFN gate/up proj | matmul | [512,4096] × [4096,14336] (×2) |
| FFN silu_and_mul | silu_and_mul | [512,28672] |
| FFN down proj | matmul | [512,14336] × [14336,4096] |
| LM head | matmul | [512,4096] × [4096,128256] |
| CE loss (fused) | fused_linear_cross_entropy | X:[512,4096], W:[128256,4096] |

### Qwen2.5 7B (BL6)

Sequence lengths: 512, 2048, 8192, 32768. **GQA: Hq=28, Hkv=4 (7:1 ratio). FFN=18944 (widest).**

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| Token embedding | embedding | W:[151936,3584], idx:[1,512] → [1,512,3584] |
| Q projection | matmul | [512,3584] × [3584,3584] |
| KV projection | matmul | [512,3584] × [3584,512] |
| QKV permute | permute | Q:[1,512,28,128]→[1,28,512,128], KV:[1,512,4,128]→[1,4,512,128] |
| RoPE | rope | Q:[1,28,512,128], K:[1,4,512,128], pos:[512] |
| Attention (GQA) | grouped_query_attention | Q:[1,28,512,128], K/V:[1,4,512,128] |
| Attention output | matmul | [512,3584] × [3584,3584] |
| Residual add | add | [512,3584] |
| RMSNorm | rmsnorm | [512,3584] |
| FFN gate/up proj | matmul | [512,3584] × [3584,18944] (×2) |
| FFN silu_and_mul | silu_and_mul | [512,37888] |
| FFN down proj | matmul | [512,18944] × [18944,3584] |
| LM head | matmul | [512,3584] × [3584,151936] |
| CE loss (fused) | fused_linear_cross_entropy | X:[512,3584], W:[151936,3584] |

### DeepSeek-V2 16B (BL6)

Sequence lengths: 512, 2048, 8192

| Layer | Operator | Shape (seq=512) |
|:------|:---------|:----------------|
| Token embedding | embedding | W:[102400,5120], idx:[1,512] → [1,512,5120] |
| QKV projection | matmul | [512,5120] × [5120,5120] |
| RoPE | rope | Q:[1,128,512,128], pos:[512] |
| MLA attention | multi_latent_attention | Q:[1,128,512,128], KV_c:[1,512,512], W_uk/W_uv:[512,128,128] |
| Attention output | matmul | [512,5120] × [5120,5120] |
| Residual add | add | [512,5120] |
| RMSNorm (pre/post) | rmsnorm | [512,5120] |
| MoE gate | matmul | [512,5120] × [5120,64] |
| Gate topk | topk | [512,64], k=2 |
| Expert dispatch | gather | X:[512,5120], idx:[512,2] |
| Expert compute | grouped_matmul | [4,512,5120] × [64,5120,12288], idx:[4] |
| FFN silu_and_mul | silu_and_mul | [512,24576] |
| Expert combine | scatter | X:[512,5120], idx:[512,2], src:[512,5120] |
| LM head | matmul | [512,5120] × [5120,102400] |
| CE loss (fused) | fused_linear_cross_entropy | X:[512,5120], W:[102400,5120] |

---

*Last updated: 2026-04-05*
