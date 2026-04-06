# Arke Benchmark — Operator Tiers

Full operator catalog with tier classification, complexity rationale, and baseline mapping.

→ Parent: `[benchmark-design.md](../benchmark-design.md)`

---

## Operator Tier (OT) Summary


| Tier    | Name                  | Count | Operators                                                                                                                        | Complexity Driver                                             |
| ------- | --------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **OT0** | Elementwise           | 12    | `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `add`, `mul`, `where_`, `cast`, `neg`, `exp`, `rsqrt`                                 | Memory-bound; 1:1 element map                                 |
| **OT1** | Reduction             | 10    | `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max`, `reduce_mean`, `argmax`, `topk`, `cumsum`     | Warp-level reduction; shared memory                           |
| **OT2** | Data Movement & Dense | 11    | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose`, `concat`, `split`, `gather`, `scatter`, `embedding`, `permute`, `copy_` | Tensor core tiling; memory layout transformation              |
| **OT3** | Fused Compound        | 7     | `swiglu`, `geglu`, `rope`, `fused_linear_cross_entropy`, `cross_entropy`, `quantize_per_token`, `dequantize_per_channel`         | Split semantics; multi-op fusion; output shape ≠ input shape  |
| **OT4** | Attention             | 5     | `flash_attention`, `grouped_query_attention`, `multi_latent_attention`, `cross_attention`, `paged_attention`                     | Multi-stage fused kernel; online softmax; KV cache management |


**Total: 45 operators** 

---

## Coverage Gap Analysis

Below summarizes what was missing from the original 20-op catalog and why each addition matters.

### What was missing


| Gap                         | Examples Added                                                          | Why It Matters                                                                                                                                                                        |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Activation completeness** | `tanh`, `sigmoid`, `exp`, `rsqrt`, `neg`                                | `tanh` used in GPT-2 GELU approx & Mamba; `sigmoid` is SiLU's component; `exp`/`rsqrt` decompose softmax/rmsnorm                                                                      |
| **Conditional / mask ops**  | `where_`, `cast`                                                        | `where_` implements causal mask & padding mask; `cast` (dtype conversion) critical for mixed-precision training/quant                                                                 |
| **Reduction variety**       | `reduce_mean`, `argmax`, `topk`, `cumsum`                               | `topk` = MoE routing & sampling; `argmax` = greedy decode; `cumsum` = prefix sums in RoPE/sorting                                                                                     |
| **Data movement**           | `concat`, `split`, `gather`, `scatter`, `embedding`, `permute`, `copy_` | These dominate non-compute time: `concat`/`split` for QKV merge/split, `gather`/`scatter` for MoE dispatch & KV cache, `embedding` for token→hidden, `permute` for multi-head reshape |
| **Position encoding**       | `rope`                                                                  | RoPE is the dominant position encoding in LLaMA/Qwen/Mistral; structurally a gated rotation                                                                                           |
| **Loss / cross-entropy**    | `fused_linear_cross_entropy`, `cross_entropy`                           | Fused variant (Liger Kernel) saves 60% memory; regular CE is universal training loss                                                                                                  |
| **Quantization**            | `quantize_per_token`, `dequantize_per_channel`                          | INT4/INT8/FP8 weight-only and activation quantization are standard in inference                                                                                                       |
| **Attention variants**      | `cross_attention`, `paged_attention`                                    | Cross-attention for encoder-decoder (T5, Whisper); paged attention for vLLM serving                                                                                                   |


### What was intentionally NOT added


| Category                                | Examples                                     | Rationale                                                                                                                                                                                                 |
| --------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sparse ops** (SpMM, sparse attention) | `sparse_matmul`, `block_sparse_attention`    | Sparse kernel patterns are architecturally distinct (index-driven memory access, load balancing); better as a separate OT5 in Stage 3 when sparse models (Mixtral top-k, Switch Transformer) are targeted |
| **Convolution**                         | `conv1d`, `conv2d`                           | Minimal in decoder-only LLMs; relevant for vision encoders (VLM Stage 3)                                                                                                                                  |
| **Communication**                       | `all_reduce`, `all_gather`, `reduce_scatter` | Distributed primitives, not kernel-level ops (NCCL/RCCL handles)                                                                                                                                          |
| **DMA / explicit memory**               | `async_copy`, `prefetch`                     | These are hardware-level scheduling, not user-facing operators                                                                                                                                            |


---

## OT0 — Elementwise

No reduction, no data dependencies across elements. Pure memory-bound.
Kernel structure: 1D grid of blocks, each element processed independently.

### relu


| Field                | Value                           |
| -------------------- | ------------------------------- |
| **Category**         | elementwise                     |
| **Signature**        | `relu(X) → Y`, shape-preserving |
| **Computation**      | `Y[i] = max(0, X[i])`           |
| **Memory pattern**   | Read X, write Y (1:1)           |
| **Primary baseline** | `F.relu` (P3)                   |
| **Expert baseline**  | FlagGems `relu` (P1)            |


### gelu


| Field                | Value                                         |
| -------------------- | --------------------------------------------- |
| **Category**         | elementwise                                   |
| **Signature**        | `gelu(X) → Y`, shape-preserving               |
| **Computation**      | `Y[i] = X[i] * Φ(X[i])` (Gaussian CDF approx) |
| **Memory pattern**   | Read X, write Y (1:1)                         |
| **Primary baseline** | `F.gelu` (P3)                                 |
| **Expert baseline**  | FlagGems `gelu` (P1)                          |


### silu


| Field                | Value                                        |
| -------------------- | -------------------------------------------- |
| **Category**         | elementwise                                  |
| **Signature**        | `silu(X) → Y`, shape-preserving              |
| **Computation**      | `Y[i] = X[i] * sigmoid(X[i])`                |
| **Memory pattern**   | Read X, write Y (1:1)                        |
| **Primary baseline** | `F.silu` (P3)                                |
| **Expert baseline**  | FlagGems `silu` (P1), Liger (P1)             |
| **Notes**            | Used in LLaMA FFN (SiLU component of SwiGLU) |


### tanh *(new)*


| Field                | Value                                                                         |
| -------------------- | ----------------------------------------------------------------------------- |
| **Category**         | elementwise                                                                   |
| **Signature**        | `tanh(X) → Y`, shape-preserving                                               |
| **Computation**      | `Y[i] = tanh(X[i])`                                                           |
| **Memory pattern**   | Read X, write Y (1:1)                                                         |
| **Primary baseline** | `torch.tanh` (P3)                                                             |
| **Expert baseline**  | FlagGems `tanh` (P1)                                                          |
| **Notes**            | Used in GPT-2 `gelu_new` approximation, Mamba gate activation, some RNN cells |


### sigmoid *(new)*


| Field                | Value                                                      |
| -------------------- | ---------------------------------------------------------- |
| **Category**         | elementwise                                                |
| **Signature**        | `sigmoid(X) → Y`, shape-preserving                         |
| **Computation**      | `Y[i] = 1 / (1 + exp(-X[i]))`                              |
| **Memory pattern**   | Read X, write Y (1:1)                                      |
| **Primary baseline** | `torch.sigmoid` (P3)                                       |
| **Expert baseline**  | FlagGems `sigmoid` (P1)                                    |
| **Notes**            | Component of SiLU; gate function in LSTM/Mamba/MoE routing |


### add


| Field                | Value                             |
| -------------------- | --------------------------------- |
| **Category**         | elementwise                       |
| **Signature**        | `add(A, B) → Y`, shape-preserving |
| **Computation**      | `Y[i] = A[i] + B[i]`              |
| **Primary baseline** | `torch.add` (P3)                  |
| **Expert baseline**  | FlagGems `add` (P1)               |
| **Notes**            | Used for residual connections     |


### mul


| Field                | Value                             |
| -------------------- | --------------------------------- |
| **Category**         | elementwise                       |
| **Signature**        | `mul(A, B) → Y`, shape-preserving |
| **Computation**      | `Y[i] = A[i] * B[i]`              |
| **Primary baseline** | `torch.mul` (P3)                  |
| **Expert baseline**  | FlagGems `mul` (P1)               |


### where_ *(new)*


| Field                | Value                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Category**         | elementwise (conditional)                                                                                                                  |
| **Signature**        | `where_(cond: [M, N], X: [M, N], Y: [M, N]) → Z: [M, N]`                                                                                   |
| **Computation**      | `Z[i] = X[i] if cond[i] else Y[i]`                                                                                                         |
| **Memory pattern**   | Read cond + X + Y, write Z (3:1 read ratio)                                                                                                |
| **Primary baseline** | `torch.where` (P3)                                                                                                                         |
| **Expert baseline**  | FlagGems `where` (P1)                                                                                                                      |
| **Notes**            | Implements causal mask (`where_(mask, scores, -inf)`), padding mask, conditional updates. High frequency in attention and loss computation |


### cast *(new)*


| Field                | Value                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Category**         | elementwise (type conversion)                                                                                     |
| **Signature**        | `cast(X: dtype_a) → Y: dtype_b`, shape-preserving                                                                 |
| **Computation**      | `Y[i] = dtype_b(X[i])`                                                                                            |
| **Memory pattern**   | Read X (sizeof_a), write Y (sizeof_b)                                                                             |
| **Primary baseline** | `tensor.to(dtype)` (P3)                                                                                           |
| **Expert baseline**  | FlagGems `to.dtype` (P1)                                                                                          |
| **Notes**            | fp32↔fp16↔bf16 in mixed-precision training (AMP); fp16→int8/int4 in quantization. Bandwidth differs by dtype pair |


### neg *(new)*


| Field                | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| **Category**         | elementwise                                             |
| **Signature**        | `neg(X) → Y`, shape-preserving                          |
| **Computation**      | `Y[i] = -X[i]`                                          |
| **Memory pattern**   | Read X, write Y (1:1)                                   |
| **Primary baseline** | `torch.neg` (P3)                                        |
| **Expert baseline**  | FlagGems `neg` (P1)                                     |
| **Notes**            | Used in attention score negation, gradient manipulation |


### exp *(new)*


| Field                | Value                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | elementwise                                                                                                                  |
| **Signature**        | `exp(X) → Y`, shape-preserving                                                                                               |
| **Computation**      | `Y[i] = exp(X[i])`                                                                                                           |
| **Memory pattern**   | Read X, write Y (1:1)                                                                                                        |
| **Primary baseline** | `torch.exp` (P3)                                                                                                             |
| **Expert baseline**  | FlagGems `exp` (P1)                                                                                                          |
| **Notes**            | Core building block of softmax and cross-entropy; frequently appears as a standalone op in decomposed graphs (torch.compile) |


### rsqrt *(new)*


| Field                | Value                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------- |
| **Category**         | elementwise                                                                           |
| **Signature**        | `rsqrt(X) → Y`, shape-preserving                                                      |
| **Computation**      | `Y[i] = 1 / sqrt(X[i])`                                                               |
| **Memory pattern**   | Read X, write Y (1:1)                                                                 |
| **Primary baseline** | `torch.rsqrt` (P3)                                                                    |
| **Expert baseline**  | FlagGems `rsqrt` (P1)                                                                 |
| **Notes**            | Core of RMSNorm/LayerNorm normalization step; appears standalone in decomposed graphs |


---

## OT1 — Reduction

Row-wise or column-wise reduction. Requires warp-level cooperation and shared memory.
Kernel structure: one program per row (or block of rows).

### softmax


| Field                | Value                                                |
| -------------------- | ---------------------------------------------------- |
| **Category**         | reduce                                               |
| **Signature**        | `softmax(X: [M, N]) → Y: [M, N]`                     |
| **Computation**      | `Y[i,:] = exp(X[i,:] - max(X[i,:])) / sum(exp(...))` |
| **Memory pattern**   | Read row, compute reduction, write row               |
| **Algorithm**        | Online softmax (single-pass, numerically stable)     |
| **Primary baseline** | `F.softmax` / cuDNN (P0/P3)                          |
| **Expert baseline**  | FlagGems `softmax` (P1), Triton Tutorial 02 (P2)     |
| **Notes**            | Used in attention score normalization                |


### layernorm


| Field                | Value                                                 |
| -------------------- | ----------------------------------------------------- |
| **Category**         | reduce                                                |
| **Signature**        | `layernorm(X: [B, H], W: [H], bias: [H]) → Y: [B, H]` |
| **Computation**      | Normalize X row-wise: `(X - mean) / std * W + bias`   |
| **Memory pattern**   | Two-pass or online (mean + variance)                  |
| **Primary baseline** | `F.layer_norm` / cuDNN (P0/P3)                        |
| **Expert baseline**  | FlagGems `layernorm` (P1), Triton Tutorial 05 (P2)    |
| **Notes**            | Used in GPT-2, BERT                                   |


### rmsnorm


| Field                | Value                                                     |
| -------------------- | --------------------------------------------------------- |
| **Category**         | reduce                                                    |
| **Signature**        | `rmsnorm(X: [B, H], W: [H]) → Y: [B, H]`                  |
| **Computation**      | `Y = X / rms(X) * W`, where `rms(X) = sqrt(mean(X²) + ε)` |
| **Primary baseline** | FlagGems `rmsnorm` (P1)                                   |
| **Expert baseline**  | Liger `rms_norm` (P1)                                     |
| **Notes**            | Used in LLaMA, DeepSeek (no bias term)                    |


### rmsnorm_residual


| Field                | Value                                                               |
| -------------------- | ------------------------------------------------------------------- |
| **Category**         | reduce                                                              |
| **Signature**        | `rmsnorm_residual(X: [B, H], residual: [B, H], W: [H]) → Y: [B, H]` |
| **Computation**      | `Z = X + residual; Y = rmsnorm(Z, W)` (fused add + rms)             |
| **Primary baseline** | Liger `rms_norm` (P1)                                               |
| **Notes**            | Fused variant used in DeepSeek; saves one memory round-trip         |


### reduce_sum


| Field                | Value                            |
| -------------------- | -------------------------------- |
| **Category**         | reduce                           |
| **Signature**        | `reduce_sum(X: [M, N]) → Y: [M]` |
| **Computation**      | `Y[i] = sum(X[i, :])`            |
| **Primary baseline** | `torch.sum` (P3)                 |
| **Expert baseline**  | FlagGems `sum` (P1)              |


### reduce_max


| Field                | Value                            |
| -------------------- | -------------------------------- |
| **Category**         | reduce                           |
| **Signature**        | `reduce_max(X: [M, N]) → Y: [M]` |
| **Computation**      | `Y[i] = max(X[i, :])`            |
| **Primary baseline** | `torch.max` (P3)                 |
| **Expert baseline**  | FlagGems `max` (P1)              |


### reduce_mean *(new)*


| Field                | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| **Category**         | reduce                                                             |
| **Signature**        | `reduce_mean(X: [M, N]) → Y: [M]`                                  |
| **Computation**      | `Y[i] = mean(X[i, :])`                                             |
| **Memory pattern**   | Same as reduce_sum + division                                      |
| **Primary baseline** | `torch.mean` (P3)                                                  |
| **Expert baseline**  | FlagGems `mean` (P1)                                               |
| **Notes**            | Used in normalization layers (decomposed), loss averaging, pooling |


### argmax *(new)*


| Field                | Value                                              |
| -------------------- | -------------------------------------------------- |
| **Category**         | reduce                                             |
| **Signature**        | `argmax(X: [M, N]) → Y: [M]` (dtype=int64)         |
| **Computation**      | `Y[i] = argmax_j(X[i, j])`                         |
| **Memory pattern**   | Read row, track max value + index                  |
| **Primary baseline** | `torch.argmax` (P3)                                |
| **Expert baseline**  | FlagGems `argmax` (P1)                             |
| **Notes**            | Greedy token decoding; MoE routing top-1 selection |


### topk *(new)*


| Field                | Value                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| **Category**         | reduce                                                                                                   |
| **Signature**        | `topk(X: [M, N], k) → (values: [M, k], indices: [M, k])`                                                 |
| **Computation**      | Returns k largest elements per row with indices                                                          |
| **Memory pattern**   | Read row, partial sort; write k values + k indices                                                       |
| **Primary baseline** | `torch.topk` (P3)                                                                                        |
| **Expert baseline**  | FlagGems `topk` (P1)                                                                                     |
| **Notes**            | MoE expert routing (top-2/top-4 gating); nucleus sampling (top-p/top-k); beam search candidate selection |


### cumsum *(new)*


| Field                | Value                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| **Category**         | reduce (prefix)                                                                                |
| **Signature**        | `cumsum(X: [M, N], dim) → Y: [M, N]`                                                           |
| **Computation**      | `Y[i, j] = sum(X[i, 0:j+1])` (along dim)                                                       |
| **Memory pattern**   | Sequential dependency along scan axis                                                          |
| **Algorithm**        | Blelloch parallel prefix sum (work-efficient)                                                  |
| **Primary baseline** | `torch.cumsum` (P3)                                                                            |
| **Expert baseline**  | FlagGems `cumsum` (P1)                                                                         |
| **Notes**            | Nucleus sampling (cumulative probability), position index generation, causal mask construction |


---

## OT2 — Data Movement & Dense Compute

Matrix multiply, data reorganization, and indexing. Compute-bound for large matmul;
memory-bound for movement ops. Requires tensor core tiling (BLOCK_M × BLOCK_N × BLOCK_K)
for matmul, coalesced access patterns for data movement.

### matmul


| Field                | Value                                                     |
| -------------------- | --------------------------------------------------------- |
| **Category**         | compute                                                   |
| **Signature**        | `matmul(A: [M, K], B: [K, N]) → Y: [M, N]`                |
| **Computation**      | Standard GEMM                                             |
| **Strategy params**  | BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, split_k |
| **Primary baseline** | cuBLAS via `torch.matmul` (P0)                            |
| **Expert baseline**  | FlagGems `mm` (P1), Triton Tutorial 03 (P2)               |
| **Notes**            | Core of QKV/FFN/lm_head projections                       |


### batch_matmul


| Field                | Value                                                     |
| -------------------- | --------------------------------------------------------- |
| **Category**         | compute                                                   |
| **Signature**        | `batch_matmul(A: [B, M, K], B: [B, K, N]) → Y: [B, M, N]` |
| **Computation**      | Independent GEMM per batch element                        |
| **Primary baseline** | cuBLAS via `torch.bmm` (P0)                               |
| **Expert baseline**  | FlagGems `bmm` (P1)                                       |
| **Notes**            | Used in attention score computation (Q @ Kᵀ)              |


### grouped_matmul


| Field                | Value                                                                     |
| -------------------- | ------------------------------------------------------------------------- |
| **Category**         | compute                                                                   |
| **Signature**        | `grouped_matmul(X: [B, M, K], W: [E, K, N], indices: [B]) → Y: [B, M, N]` |
| **Computation**      | For each batch i: `Y[i] = X[i] @ W[indices[i]]`                           |
| **Primary baseline** | CUTLASS `grouped_gemm` (P0)                                               |
| **Expert baseline**  | FlagGems `matmul_ogs` (P1)                                                |
| **Notes**            | MoE expert dispatch; `indices` must be in `[0, E)`                        |


### transpose


| Field                | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| **Category**         | data movement                                           |
| **Signature**        | `transpose(X: [M, N]) → Y: [N, M]`                      |
| **Computation**      | `Y[j, i] = X[i, j]`                                     |
| **Memory pattern**   | Non-coalesced read or write; needs shared memory tiling |
| **Primary baseline** | `torch.transpose` (P3)                                  |
| **Expert baseline**  | FlagGems (P1)                                           |


### concat *(new)*


| Field                | Value                                                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | data movement                                                                                                                    |
| **Signature**        | `concat([X₁: [M, N₁], X₂: [M, N₂], ...], dim) → Y: [M, N₁+N₂+...]`                                                               |
| **Computation**      | Copies input tensors contiguously along specified dimension                                                                      |
| **Memory pattern**   | Multi-source read, contiguous write; bandwidth-bound                                                                             |
| **Primary baseline** | `torch.cat` (P3)                                                                                                                 |
| **Expert baseline**  | FlagGems `cat` (P1)                                                                                                              |
| **Notes**            | Merging QKV projections, combining MoE expert outputs, KV cache append. One of the highest-frequency ops in LLM profiling traces |


### split *(new)*


| Field                | Value                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | data movement                                                                                                                             |
| **Signature**        | `split(X: [M, N], sizes, dim) → [Y₁: [M, N₁], Y₂: [M, N₂], ...]`                                                                          |
| **Computation**      | Inverse of concat; slices tensor into chunks                                                                                              |
| **Memory pattern**   | Contiguous read, scattered write (or zero-copy view)                                                                                      |
| **Primary baseline** | `torch.split` / `torch.chunk` (P3)                                                                                                        |
| **Expert baseline**  | FlagGems `split` (P1)                                                                                                                     |
| **Notes**            | Splitting fused QKV into Q/K/V; gated activation input decomposition; often zero-copy (view) in eager but materializes in compiled graphs |


### gather *(new)*


| Field                | Value                                                                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | data movement (indexed)                                                                                                                                         |
| **Signature**        | `gather(X: [M, N], indices: [M, K]) → Y: [M, K]`                                                                                                                |
| **Computation**      | `Y[i, j] = X[i, indices[i, j]]`                                                                                                                                 |
| **Memory pattern**   | Random read (scatter-gather); poor coalescing                                                                                                                   |
| **Primary baseline** | `torch.gather` / `torch.index_select` (P3)                                                                                                                      |
| **Expert baseline**  | FlagGems `gather` (P1)                                                                                                                                          |
| **Notes**            | KV cache page lookup (paged attention), MoE token-to-expert routing, embedding lookup (generalized). Memory access pattern is the kernel challenge, not compute |


### scatter *(new)*


| Field                | Value                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Category**         | data movement (indexed)                                                                                                              |
| **Signature**        | `scatter(X: [M, N], indices: [M, K], src: [M, K]) → Y: [M, N]`                                                                       |
| **Computation**      | `Y[i, indices[i, j]] = src[i, j]` (or += for scatter_add)                                                                            |
| **Memory pattern**   | Random write; potential write conflicts (atomics)                                                                                    |
| **Primary baseline** | `torch.scatter` / `torch.scatter_add` (P3)                                                                                           |
| **Expert baseline**  | FlagGems `scatter` (P1)                                                                                                              |
| **Notes**            | MoE expert output aggregation (scatter_add), sparse gradient accumulation, one-hot encoding. Atomic operations required for add mode |


### embedding *(new)*


| Field                | Value                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | data movement (lookup)                                                                                                      |
| **Signature**        | `embedding(W: [V, D], idx: [B, S]) → Y: [B, S, D]`                                                                          |
| **Computation**      | `Y[b, s, :] = W[idx[b, s], :]`                                                                                              |
| **Memory pattern**   | Irregular read from weight table; contiguous write                                                                          |
| **Primary baseline** | `F.embedding` (P3)                                                                                                          |
| **Expert baseline**  | FlagGems `embedding` (P1)                                                                                                   |
| **Notes**            | Token embedding (V=32k-128k), position embedding. Weight table can be huge (128k × 4096 × 2B = 1GB); cache locality matters |


### permute *(new)*


| Field                | Value                                                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | data movement (layout)                                                                                                                         |
| **Signature**        | `permute(X: [dims...], order) → Y: [reordered dims...]`                                                                                        |
| **Computation**      | Reorders tensor dimensions (generalized transpose)                                                                                             |
| **Memory pattern**   | Non-contiguous access; may need tiled copy for efficiency                                                                                      |
| **Primary baseline** | `tensor.permute` / `tensor.contiguous` (P3)                                                                                                    |
| **Expert baseline**  | FlagGems `permute` (P1)                                                                                                                        |
| **Notes**            | Multi-head reshape `[B, S, H, D] → [B, H, S, D]`; appears before/after every attention layer. Often fused with contiguous() in compiled graphs |


### copy_ *(new)*


| Field                | Value                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **Category**         | data movement (memory)                                                                              |
| **Signature**        | `copy_(dst, src) → dst` (in-place)                                                                  |
| **Computation**      | `dst[i] = src[i]` (with potential dtype conversion)                                                 |
| **Memory pattern**   | Pure bandwidth benchmark; read src + write dst                                                      |
| **Primary baseline** | `tensor.copy`_ (P3)                                                                                 |
| **Expert baseline**  | FlagGems `copy_` (P1)                                                                               |
| **Notes**            | KV cache write, gradient accumulation, checkpoint restore. Baseline for achievable memory bandwidth |


---

## OT3 — Fused Compound

Multi-operation fusion in a single kernel. Each involves non-trivial data flow
(split, rotation, or chained matmul+activation). Key to training/inference efficiency.

### swiglu


| Field                | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| **Category**         | gated activation                                        |
| **Signature**        | `swiglu(X: [B, 2H]) → Y: [B, H]`                        |
| **Computation**      | `gate, val = split(X, 2, dim=-1); Y = silu(gate) * val` |
| **Primary baseline** | Liger `swiglu` (P1)                                     |
| **Notes**            | Dominant FFN non-linearity in LLaMA/DeepSeek families   |


### geglu


| Field                | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| **Category**         | gated activation                                        |
| **Signature**        | `geglu(X: [B, 2H]) → Y: [B, H]`                         |
| **Computation**      | `gate, val = split(X, 2, dim=-1); Y = gelu(gate) * val` |
| **Primary baseline** | Liger `geglu` (P1)                                      |
| **Notes**            | Used in PaLM, some BERT variants                        |


### rope *(new)*


| Field                | Value                                                                                                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | position encoding (fused)                                                                                                                                                                     |
| **Signature**        | `rope(Q: [B, H, S, D], K: [B, H, S, D], pos: [S]) → (Q': [B, H, S, D], K': [B, H, S, D])`                                                                                                     |
| **Computation**      | Split each head into even/odd pairs, apply 2D rotation: `q_r = q_even * cos(θ) - q_odd * sin(θ)`, `q_i = q_even * sin(θ) + q_odd * cos(θ)`                                                    |
| **Memory pattern**   | Read Q + K + freq table; write rotated Q' + K'                                                                                                                                                |
| **Primary baseline** | HuggingFace `apply_rotary_pos_emb` (P3)                                                                                                                                                       |
| **Expert baseline**  | Liger `rope` (P1), FlagGems `apply_rotary_pos_emb` (P1)                                                                                                                                       |
| **Notes**            | Used in every LLaMA/Qwen/Mistral/DeepSeek layer; structurally a gated element-wise rotation. High frequency (called once per layer per Q and K). NTK-aware and dynamic scaling variants exist |


### fused_linear_cross_entropy *(new)*


| Field                | Value                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Category**         | fused loss                                                                                                               |
| **Signature**        | `fused_linear_cross_entropy(X: [B*S, H], W: [V, H], target: [B*S]) → (loss, logits_grad)`                                |
| **Computation**      | Fused: `logits = X @ W.T → softmax → -log(p[target]) → loss`. Chunked to avoid V×B*S memory                              |
| **Memory pattern**   | Chunked matmul + online cross-entropy; never materializes full logit tensor                                              |
| **Primary baseline** | `F.cross_entropy(X @ W.T, target)` (P3)                                                                                  |
| **Expert baseline**  | Liger `FusedLinearCrossEntropyLoss` (P1)                                                                                 |
| **Notes**            | Saves ~60% memory for large vocabularies (V=128k). Liger Kernel's flagship fused op. Essential for long-context training |


### cross_entropy *(new)*


| Field                | Value                                                               |
| -------------------- | ------------------------------------------------------------------- |
| **Category**         | loss                                                                |
| **Signature**        | `cross_entropy(logits: [B*S, V], target: [B*S]) → loss: scalar`     |
| **Computation**      | `loss = -log(softmax(logits)[target])` with label smoothing         |
| **Memory pattern**   | Read logits (large V dimension), reduction to scalar                |
| **Primary baseline** | `F.cross_entropy` (P3)                                              |
| **Expert baseline**  | FlagGems `cross_entropy_loss` (P1), Liger `CrossEntropyLoss` (P1)   |
| **Notes**            | Standard training loss; V can be 128k+ making this memory-intensive |


### quantize_per_token *(new)*


| Field                | Value                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Category**         | quantization                                                                                                                                                                               |
| **Signature**        | `quantize_per_token(X: [M, N], dtype) → (X_q: [M, N], scale: [M])`                                                                                                                         |
| **Computation**      | `scale[i] = max(abs(X[i,:])) / max_dtype; X_q[i,:] = round(X[i,:] / scale[i])`                                                                                                             |
| **Memory pattern**   | Read X, compute per-row absmax (reduction), write quantized output + scale                                                                                                                 |
| **Primary baseline** | `torch.quantize_per_tensor` (P3)                                                                                                                                                           |
| **Expert baseline**  | vLLM / TensorRT-LLM quantize kernels (P1)                                                                                                                                                  |
| **Notes**            | Activation quantization for W8A8 / W4A16 / FP8 inference. Per-token (dynamic) is more accurate than per-tensor (static). Combines a reduction (absmax) with an elementwise (scale + round) |


### dequantize_per_channel *(new)*


| Field                      | Value                                                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Category**               | quantization                                                                                                               |
| **Signature**              | `dequantize_per_channel(X_q: [M, N], scale: [N], zero_point: [N]) → Y: [M, N]`                                             |
| **Computation**            | `Y[i, j] = (X_q[i, j] - zero_point[j]) * scale[j]`                                                                         |
| **Memory pattern (quant)** | Read X_q + broadcast scale/zero_point, write Y                                                                             |
| **Primary baseline**       | `torch.dequantize` (P3)                                                                                                    |
| **Expert baseline**        | GPTQ Triton dequant kernel (P1), vLLM `cutlass_scaled_mm` (P1)                                                             |
| **Notes**                  | Weight dequantization for INT4/INT8 weight-only quantization (GPTQ, AWQ, AutoRound). Often fused with matmul as W4A16 GEMM |


---

## OT4 — Attention

Multi-stage fused kernels. Each requires careful tiling of Q/K/V dimensions,
online softmax for numerical stability, and hardware-specific memory management.
Most complex operator tier.

### flash_attention


| Field                | Value                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------- |
| **Category**         | attention                                                                              |
| **Signature**        | `flash_attention(Q: [B, H, S, D], K: [B, H, S, D], V: [B, H, S, D]) → Y: [B, H, S, D]` |
| **Computation**      | Fused scaled dot-product attention with online softmax                                 |
| **Algorithm**        | FlashAttention v2 tiling (tiled Q outer loop, KV inner loop)                           |
| **Primary baseline** | cuDNN SDPA (P0)                                                                        |
| **Expert baseline**  | FlashAttention CUDA (P1), Triton Tutorial 06 (P2)                                      |
| **Notes**            | Causal mask optional; O(S) memory vs O(S²) naive                                       |


### grouped_query_attention (GQA)


| Field                | Value                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| **Category**         | attention                                                                                         |
| **Signature**        | `gqa(Q: [B, Hq, S, D], K: [B, Hkv, S, D], V: [B, Hkv, S, D]) → Y: [B, Hq, S, D]` where `Hq > Hkv` |
| **Computation**      | Each KV head is shared by `Hq/Hkv` query heads                                                    |
| **Primary baseline** | cuDNN SDPA (P0)                                                                                   |
| **Expert baseline**  | FlashAttention (GQA mode, P1)                                                                     |
| **Notes**            | Used in LLaMA-3 (Hq=32, Hkv=8), Mistral, Qwen2.5                                                  |


### multi_latent_attention (MLA)


| Field                | Value                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| **Category**         | attention                                                                                         |
| **Signature**        | `mla(Q: [B, H, S, D], KV_c: [B, S, D_c], W_uk: [D_c, H, D], W_uv: [D_c, H, D]) → Y: [B, H, S, D]` |
| **Computation**      | Decompress KV from latent: `K = KV_c @ W_uk; V = KV_c @ W_uv`, then standard attention            |
| **Primary baseline** | DeepSeek reference implementation                                                                 |
| **Notes**            | DeepSeek-V2/V3 architecture; KV cache size ∝ D_c not H×D                                          |


### cross_attention *(new)*


| Field                | Value                                                                                                                                                                |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | attention                                                                                                                                                            |
| **Signature**        | `cross_attention(Q: [B, Hq, Sq, D], K: [B, Hk, Skv, D], V: [B, Hk, Skv, D]) → Y: [B, Hq, Sq, D]` where `Sq ≠ Skv`                                                    |
| **Computation**      | Q attends to K/V from a different sequence (no causal mask)                                                                                                          |
| **Algorithm**        | Same FlashAttention tiling, but no causal mask and Sq ≠ Skv                                                                                                          |
| **Primary baseline** | cuDNN SDPA (P0)                                                                                                                                                      |
| **Expert baseline**  | FlashAttention (cross-attention mode, P1)                                                                                                                            |
| **Notes**            | Encoder-decoder models: T5, BART, Whisper. Also used in VLMs (image tokens → text queries). Distinct from self-attention because Sq and Skv can differ significantly |


### paged_attention *(new)*


| Field                | Value                                                                                                                                                                                                                |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Category**         | attention (inference)                                                                                                                                                                                                |
| **Signature**        | `paged_attention(Q: [B, H, 1, D], block_tables: [B, max_blocks], kv_cache: [num_blocks, 2, H, block_size, D]) → Y: [B, H, 1, D]`                                                                                     |
| **Computation**      | Attention over paged (non-contiguous) KV cache blocks; supports variable-length sequences                                                                                                                            |
| **Algorithm**        | Block-level gather + flash attention kernel with page table indirection                                                                                                                                              |
| **Primary baseline** | vLLM paged attention CUDA kernel (P0)                                                                                                                                                                                |
| **Expert baseline**  | FlashInfer (P1)                                                                                                                                                                                                      |
| **Notes**            | The core inference optimization for serving LLMs: eliminates KV cache fragmentation, enables continuous batching. Combines `gather` (page lookup) with `flash_attention` (score computation). vLLM's defining kernel |


---

## Appendix: OT/Operator Cross-Reference

### Full Operator Index (45 ops)


| OT  | Operator                     | Category          | Status        |
| --- | ---------------------------- | ----------------- | ------------- |
| 0   | `relu`                       | elementwise       | ✅ template    |
| 0   | `gelu`                       | elementwise       | ✅ template    |
| 0   | `silu`                       | elementwise       | ✅ template    |
| 0   | `tanh`                       | elementwise       | 📋 new        |
| 0   | `sigmoid`                    | elementwise       | 📋 new        |
| 0   | `add`                        | elementwise       | ✅ template    |
| 0   | `mul`                        | elementwise       | ✅ template    |
| 0   | `where`_                     | conditional       | 📋 new        |
| 0   | `cast`                       | type conversion   | 📋 new        |
| 0   | `neg`                        | elementwise       | 📋 new        |
| 0   | `exp`                        | elementwise       | 📋 new        |
| 0   | `rsqrt`                      | elementwise       | 📋 new        |
| 1   | `softmax`                    | reduce            | ✅ template    |
| 1   | `layernorm`                  | reduce            | ✅ template    |
| 1   | `rmsnorm`                    | reduce            | ✅ template    |
| 1   | `rmsnorm_residual`           | reduce            | ⬜ no template |
| 1   | `reduce_sum`                 | reduce            | ⬜ no template |
| 1   | `reduce_max`                 | reduce            | ⬜ no template |
| 1   | `reduce_mean`                | reduce            | 📋 new        |
| 1   | `argmax`                     | reduce            | 📋 new        |
| 1   | `topk`                       | reduce            | 📋 new        |
| 1   | `cumsum`                     | prefix scan       | 📋 new        |
| 2   | `matmul`                     | compute           | ✅ template    |
| 2   | `batch_matmul`               | compute           | ✅ template    |
| 2   | `grouped_matmul`             | compute           | ⬜ no template |
| 2   | `transpose`                  | data movement     | ⬜ no template |
| 2   | `concat`                     | data movement     | 📋 new        |
| 2   | `split`                      | data movement     | 📋 new        |
| 2   | `gather`                     | indexed movement  | 📋 new        |
| 2   | `scatter`                    | indexed movement  | 📋 new        |
| 2   | `embedding`                  | lookup            | 📋 new        |
| 2   | `permute`                    | layout            | 📋 new        |
| 2   | `copy_`                      | memory            | 📋 new        |
| 3   | `swiglu`                     | gated activation  | ⬜ no template |
| 3   | `geglu`                      | gated activation  | ⬜ no template |
| 3   | `rope`                       | position encoding | 📋 new        |
| 3   | `fused_linear_cross_entropy` | fused loss        | 📋 new        |
| 3   | `cross_entropy`              | loss              | 📋 new        |
| 3   | `quantize_per_token`         | quantization      | 📋 new        |
| 3   | `dequantize_per_channel`     | quantization      | 📋 new        |
| 4   | `flash_attention`            | attention         | ⬜ no template |
| 4   | `grouped_query_attention`    | attention         | ⬜ no template |
| 4   | `multi_latent_attention`     | attention         | ⬜ no template |
| 4   | `cross_attention`            | attention         | 📋 new        |
| 4   | `paged_attention`            | attention         | 📋 new        |


**Legend:** ✅ = Arke Triton template exists | ⬜ = in catalog, no template | 📋 = newly added

### Deferred to Stage 3 (OT5 candidates)


| Category       | Operators                                                        | Rationale                                                           |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| Sparse         | `sparse_matmul`, `block_sparse_attention`, `sparse_moe_dispatch` | Index-driven memory, load balancing; needs separate kernel patterns |
| Convolution    | `conv1d`, `conv2d`, `depthwise_conv`                             | Vision/VLM workloads                                                |
| Communication  | `all_reduce`, `all_gather`, `reduce_scatter`                     | NCCL primitives, not kernel-level                                   |
| Sliding window | `sliding_window_attention`                                       | Variant of flash_attention with window mask; Mistral/Gemma2         |


---

*Last updated: 2026-04-05 (expanded 20→45 operators)*