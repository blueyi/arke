# Arke Benchmark — Operator Tiers

Full operator catalog with tier classification, complexity rationale, and baseline mapping.

→ Parent: [`BENCHMARK.md`](./BENCHMARK.md)

---

## Operator Tier (OT) Summary

| Tier | Name | Operators | Complexity Driver |
|:----:|:-----|:----------|:-----------------|
| **OT0** | Elementwise | `relu`, `gelu`, `silu`, `add`, `mul` | Memory-bound; 1:1 element map |
| **OT1** | Reduction | `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max` | Warp-level reduction; shared memory |
| **OT2** | Compute-Dense | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose` | Tensor core tiling; shared memory staging |
| **OT3** | Gated Activation | `swiglu`, `geglu` | Split semantics; output shape ≠ input shape |
| **OT4** | Attention | `flash_attention`, `grouped_query_attention`, `multi_latent_attention` | Multi-stage fused kernel; online softmax |

---

## OT0 — Elementwise

No reduction, no data dependencies across elements. Pure memory-bound.
Kernel structure: 1D grid of blocks, each element processed independently.

### relu

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `relu(X) → Y`, shape-preserving |
| **Computation** | `Y[i] = max(0, X[i])` |
| **Memory pattern** | Read X, write Y (1:1) |
| **Primary baseline** | `F.relu` (P3) |
| **Expert baseline** | FlagGems `relu` (P1) |

### gelu

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `gelu(X) → Y`, shape-preserving |
| **Computation** | `Y[i] = X[i] * Φ(X[i])` (Gaussian CDF approx) |
| **Memory pattern** | Read X, write Y (1:1) |
| **Primary baseline** | `F.gelu` (P3) |
| **Expert baseline** | FlagGems `gelu` (P1) |

### silu

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `silu(X) → Y`, shape-preserving |
| **Computation** | `Y[i] = X[i] * sigmoid(X[i])` |
| **Memory pattern** | Read X, write Y (1:1) |
| **Primary baseline** | `F.silu` (P3) |
| **Expert baseline** | FlagGems `silu` (P1), Liger (P1) |
| **Notes** | Used in LLaMA FFN (SiLU component of SwiGLU) |

### add

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `add(A, B) → Y`, shape-preserving |
| **Computation** | `Y[i] = A[i] + B[i]` |
| **Primary baseline** | `torch.add` (P3) |
| **Expert baseline** | FlagGems `add` (P1) |
| **Notes** | Used for residual connections |

### mul

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `mul(A, B) → Y`, shape-preserving |
| **Computation** | `Y[i] = A[i] * B[i]` |
| **Primary baseline** | `torch.mul` (P3) |
| **Expert baseline** | FlagGems `mul` (P1) |

---

## OT1 — Reduction

Row-wise or column-wise reduction. Requires warp-level cooperation and shared memory.
Kernel structure: one program per row (or block of rows).

### softmax

| Field | Value |
|:------|:------|
| **Category** | reduce |
| **Signature** | `softmax(X: [M, N]) → Y: [M, N]` |
| **Computation** | `Y[i,:] = exp(X[i,:] - max(X[i,:])) / sum(exp(...))` |
| **Memory pattern** | Read row, compute reduction, write row |
| **Algorithm** | Online softmax (single-pass, numerically stable) |
| **Primary baseline** | `F.softmax` / cuDNN (P0/P3) |
| **Expert baseline** | FlagGems `softmax` (P1), Triton Tutorial 02 (P2) |
| **Notes** | Used in attention score normalization |

### layernorm

| Field | Value |
|:------|:------|
| **Category** | reduce |
| **Signature** | `layernorm(X: [B, H], W: [H], bias: [H]) → Y: [B, H]` |
| **Computation** | Normalize X row-wise: `(X - mean) / std * W + bias` |
| **Memory pattern** | Two-pass or online (mean + variance) |
| **Primary baseline** | `F.layer_norm` / cuDNN (P0/P3) |
| **Expert baseline** | FlagGems `layernorm` (P1), Triton Tutorial 05 (P2) |
| **Notes** | Used in GPT-2, BERT |

### rmsnorm

| Field | Value |
|:------|:------|
| **Category** | reduce |
| **Signature** | `rmsnorm(X: [B, H], W: [H]) → Y: [B, H]` |
| **Computation** | `Y = X / rms(X) * W`, where `rms(X) = sqrt(mean(X²) + ε)` |
| **Primary baseline** | FlagGems `rmsnorm` (P1) |
| **Expert baseline** | Liger `rms_norm` (P1) |
| **Notes** | Used in LLaMA, DeepSeek (no bias term) |

### rmsnorm_residual

| Field | Value |
|:------|:------|
| **Category** | reduce |
| **Signature** | `rmsnorm_residual(X: [B, H], residual: [B, H], W: [H]) → Y: [B, H]` |
| **Computation** | `Z = X + residual; Y = rmsnorm(Z, W)` (fused add + rms) |
| **Primary baseline** | Liger `rms_norm` (P1) |
| **Notes** | Fused variant used in DeepSeek; saves one memory round-trip |

### reduce_sum

| Field | Value |
|:------|:------|
| **Category** | reduce |
| **Signature** | `reduce_sum(X: [M, N]) → Y: [M]` |
| **Computation** | `Y[i] = sum(X[i, :])` |
| **Primary baseline** | `torch.sum` (P3) |
| **Expert baseline** | FlagGems `sum` (P1) |

### reduce_max

| Field | Value |
|:------|:------|
| **Category** | reduce |
| **Signature** | `reduce_max(X: [M, N]) → Y: [M]` |
| **Computation** | `Y[i] = max(X[i, :])` |
| **Primary baseline** | `torch.max` (P3) |
| **Expert baseline** | FlagGems `max` (P1) |

---

## OT2 — Compute-Dense

Matrix multiply and data movement. Compute-bound for large shapes.
Requires tensor core tiling (BLOCK_M × BLOCK_N × BLOCK_K), shared memory staging,
L2 cache swizzle for large matrices.

### matmul

| Field | Value |
|:------|:------|
| **Category** | compute |
| **Signature** | `matmul(A: [M, K], B: [K, N]) → Y: [M, N]` |
| **Computation** | Standard GEMM |
| **Strategy params** | BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, split_k |
| **Primary baseline** | cuBLAS via `torch.matmul` (P0) |
| **Expert baseline** | FlagGems `mm` (P1), Triton Tutorial 03 (P2) |
| **Notes** | Core of QKV/FFN/lm_head projections |

### batch_matmul

| Field | Value |
|:------|:------|
| **Category** | compute |
| **Signature** | `batch_matmul(A: [B, M, K], B: [B, K, N]) → Y: [B, M, N]` |
| **Computation** | Independent GEMM per batch element |
| **Primary baseline** | cuBLAS via `torch.bmm` (P0) |
| **Expert baseline** | FlagGems `bmm` (P1) |
| **Notes** | Used in attention score computation (Q @ Kᵀ) |

### grouped_matmul

| Field | Value |
|:------|:------|
| **Category** | compute |
| **Signature** | `grouped_matmul(X: [B, M, K], W: [E, K, N], indices: [B]) → Y: [B, M, N]` |
| **Computation** | For each batch i: `Y[i] = X[i] @ W[indices[i]]` |
| **Primary baseline** | CUTLASS `grouped_gemm` (P0) |
| **Expert baseline** | FlagGems `matmul_ogs` (P1) |
| **Notes** | MoE expert dispatch; `indices` must be in `[0, E)` |

### transpose

| Field | Value |
|:------|:------|
| **Category** | compute (move) |
| **Signature** | `transpose(X: [M, N]) → Y: [N, M]` |
| **Computation** | `Y[j, i] = X[i, j]` |
| **Memory pattern** | Non-coalesced read or write; needs shared memory tiling |
| **Primary baseline** | `torch.transpose` (P3) |
| **Expert baseline** | FlagGems (P1) |

---

## OT3 — Gated Activation

Input tensor is split in half along the last dimension; one half is the gate signal.
Output has half the last dimension of input: `X: [B, 2H] → Y: [B, H]`.

### swiglu

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `swiglu(X: [B, 2H]) → Y: [B, H]` |
| **Computation** | `gate, val = split(X, 2, dim=-1); Y = silu(gate) * val` |
| **Primary baseline** | Liger `swiglu` (P1) |
| **Notes** | Dominant FFN non-linearity in LLaMA/DeepSeek families |

### geglu

| Field | Value |
|:------|:------|
| **Category** | elementwise |
| **Signature** | `geglu(X: [B, 2H]) → Y: [B, H]` |
| **Computation** | `gate, val = split(X, 2, dim=-1); Y = gelu(gate) * val` |
| **Primary baseline** | Liger `geglu` (P1) |
| **Notes** | Used in PaLM, some BERT variants |

---

## OT4 — Attention

Multi-stage fused kernels. Each requires careful tiling of Q/K/V dimensions,
online softmax for numerical stability, and hardware-specific memory management.
Most complex operator tier.

### flash_attention

| Field | Value |
|:------|:------|
| **Category** | attention |
| **Signature** | `flash_attention(Q: [B, H, S, D], K: [B, H, S, D], V: [B, H, S, D]) → Y: [B, H, S, D]` |
| **Computation** | Fused scaled dot-product attention with online softmax |
| **Algorithm** | FlashAttention v2 tiling (tiled Q outer loop, KV inner loop) |
| **Primary baseline** | cuDNN SDPA (P0) |
| **Expert baseline** | FlashAttention CUDA (P1), Triton Tutorial 06 (P2) |
| **Notes** | Causal mask optional; O(S) memory vs O(S²) naive |

### grouped_query_attention (GQA)

| Field | Value |
|:------|:------|
| **Category** | attention |
| **Signature** | `gqa(Q: [B, Hq, S, D], K: [B, Hkv, S, D], V: [B, Hkv, S, D]) → Y: [B, Hq, S, D]` where `Hq > Hkv` |
| **Computation** | Each KV head is shared by `Hq/Hkv` query heads |
| **Primary baseline** | cuDNN SDPA (P0) |
| **Expert baseline** | FlashAttention (GQA mode, P1) |
| **Notes** | Used in LLaMA-3 (Hq=32, Hkv=8), Mistral, Qwen2.5 |

### multi_latent_attention (MLA)

| Field | Value |
|:------|:------|
| **Category** | attention |
| **Signature** | `mla(Q: [B, H, S, D], KV_c: [B, S, D_c], W_uk: [D_c, H, D], W_uv: [D_c, H, D]) → Y: [B, H, S, D]` |
| **Computation** | Decompress KV from latent: `K = KV_c @ W_uk; V = KV_c @ W_uv`, then standard attention |
| **Primary baseline** | DeepSeek reference implementation |
| **Notes** | DeepSeek-V2/V3 architecture; KV cache size ∝ D_c not H×D |

---

*Last updated: 2026-04-05*
