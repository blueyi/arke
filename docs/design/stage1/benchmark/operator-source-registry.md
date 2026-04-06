# Arke Benchmark — Full Operator Source Registry

→ Parent: [`benchmark-design.md`](../benchmark-design.md) §5 Baselines

> **Purpose:** Comprehensive catalog of all obtainable GPU operator sources for
> Arke benchmark baselines. Covers vendor libraries, Triton operator libraries,
> inference/training frameworks, and community kernels.
>
> **Status:** Reference plan — not yet implemented. See [§ Integration Plan](#integration-plan)
> for the phased approach.

---

## Summary

| # | Source | License | Standalone Ops | Framework-bound Ops | Total Triton Files | Relevance |
|---|--------|---------|---------------:|--------------------:|-------------------:|:---------:|
| 1 | **cuBLAS / cuDNN** (via PyTorch) | NVIDIA EULA | ~15 | — | 0 (closed) | P0 |
| 2 | **CUTLASS** | BSD-3 | ~8 | — | 0 (C++/CUDA) | P0 |
| 3 | **FlagGems** | Apache-2.0 | **200+** | — | 200+ | P1 |
| 4 | **Liger-Kernel** | BSD-2 | **15** | — | 15 | P1 |
| 5 | **FlashAttention** | BSD-3 | 2 | — | 2 | P1 |
| 6 | **Triton Tutorials** | MIT | **10** | — | 10 | P2 |
| 7 | **Triton Kernels** (triton-lang/kernels) | MIT | **6** | — | 6 | P2 |
| 8 | **HuggingFace Kernels-Community** | Apache-2.0 | ~5 | — | ~10 | P2 |
| 9 | **vLLM** | Apache-2.0 | ~12 | ~73 | 85+ | P1★ |
| 10 | **SGLang** | Apache-2.0 | ~10 | ~70 | 80+ | P1★ |
| 11 | **verl** | Apache-2.0 | 2 | 1 | 3 | P1★ |
| 12 | **PyTorch Eager** | BSD-3 | all | — | 0 | P3 |
| 13 | **torch.compile (Inductor)** | BSD-3 | all | — | auto-gen | P4 |
| 14 | **Unsloth** | Apache-2.0 | ~4 | tightly coupled | ~6 | Ref |

> **P1★** = P1-tier quality but most kernels are framework-bound (require vLLM/SGLang data
> structures). Only standalone-extractable ops are benchmarkable.

**Grand total unique operator implementations: ~370+**
**Standalone-benchmarkable (no framework dependency): ~280+**

---

## 1. NVIDIA Vendor Libraries — P0

### cuBLAS / cuDNN (via PyTorch)

| Category | Operators | Count |
|:---------|:----------|------:|
| BLAS | matmul, mm, bmm, addmm, addmv, gemm | 6 |
| Normalization | layer_norm, batch_norm, group_norm, instance_norm | 4 |
| Activation | softmax, log_softmax | 2 |
| Attention | scaled_dot_product_attention (SDPA) | 1 |
| Pooling | avg_pool2d, max_pool2d | 2 |
| **Total** | | **15** |

- **Repo:** https://pytorch.org (bundled)
- **Access:** `torch.matmul()`, `F.softmax()`, `F.layer_norm()`, etc.
- **License:** NVIDIA EULA (proprietary, bundled with PyTorch CUDA)

### CUTLASS

| Category | Operators | Count |
|:---------|:----------|------:|
| GEMM | gemm, grouped_gemm, batched_gemm, gemm_with_epilogue | 4 |
| Convolution | conv2d, depthwise_conv2d | 2 |
| Fused | fused_gemm_epilogue (bias+act), splitk_gemm | 2 |
| **Total** | | **8** |

- **Repo:** https://github.com/NVIDIA/cutlass
- **License:** BSD-3-Clause
- **Note:** C++/CUDA templates. PyTorch Inductor uses CUTLASS for some matmul configs.

---

## 2. FlagGems — P1 (Primary Triton Baseline)

**200+ operators** registered as PyTorch ATen backend. Most comprehensive standalone Triton library.

| Category | Operators | Count |
|:---------|:----------|------:|
| **BLAS** | mm, bmm, addmm, addmm_out, bmm_out, dot, addr, addmv, addmv_out, mv | 10 |
| **Math (Unary)** | abs, ceil, cos, exp, exp2, log, sin, erf, floor, neg, reciprocal, rsqrt, sqrt, sigmoid, tanh, acos, asin, atan, cosh, sinh, sign, round, trunc, digamma, i0 | 25 |
| **Math (Binary)** | add, sub, mul, div, pow, fmod, remainder, maximum, minimum, fmin, hypot, logaddexp, atan2, bitwise_and/or/not/xor, logical_and/or/not/xor | 22 |
| **Comparison** | eq, ne, gt, ge, lt, le, isnan, isinf, isfinite, isclose, allclose, equal | 12 |
| **Reduction** | sum, mean, max, min, amax, argmax, argmin, all, any, count_nonzero, cumsum, cummax, cummin, prod | 14 |
| **Activation** | gelu, relu, silu, elu, celu, glu, leaky_relu, hardsigmoid, hardswish, mish, softplus, softsign, prelu, threshold | 14 |
| **Fused Activation** | gelu_and_mul, silu_and_mul, geglu, swiglu (+ backward variants) | 8 |
| **Normalization** | layer_norm, group_norm, batch_norm, fused_add_rms_norm, rms_norm (+ backward) | 10 |
| **Attention** | flash_attention_forward, flash_attn_varlen_func, flash_mla, concat_and_cache_mla | 4 |
| **Embedding** | embedding, embedding_backward, embedding_dense_backward | 3 |
| **Loss** | cross_entropy_loss, mse_loss, nll_loss | 3 |
| **RoPE** | apply_rotary_pos_emb | 1 |
| **Tensor** | cat, clone, flip, gather, index_select, masked_fill, scatter, stack, transpose, permute, reshape, where, triu, diag, eye, full, zeros, ones, arange, linspace | 20+ |
| **Convolution** | conv1d, conv2d, conv3d, _conv_depthwise2d | 4 |
| **Dropout** | dropout, dropout_backward | 2 |
| **MoE** | dispatch_fused_moe_kernel, moe_align_block_size, grouped_topk, inplace_fused_experts | 4 |
| **Sampling** | apply_repetition_penalties, topk, multinomial | 3 |
| **Quantization** | cutlass_scaled_mm | 1 |
| **Other** | clamp, lerp, sort, unique, bincount, upsample, interpolate, etc. | 40+ |
| **Total** | | **200+** |

- **Repo:** https://github.com/flagos-ai/FlagGems
- **Install:** `pip install flag-gems`
- **License:** Apache-2.0
- **Docs:** https://flagos-ai.github.io/FlagGems/references/operators/

---

## 3. Liger-Kernel — P1 (LLM Training Specialist)

**15 fused operators** focused on training memory efficiency. Forward + backward.

| Category | Operators | Count |
|:---------|:----------|------:|
| **Normalization** | rms_norm, layer_norm, group_norm | 3 |
| **Positional** | rope | 1 |
| **Activation** | swiglu, geglu | 2 |
| **Loss** | cross_entropy, fused_linear_cross_entropy | 2 |
| **Post-Training Loss** | dpo_loss, cpo_loss, orpo_loss, simpo_loss, kto_loss, jsd_loss, fused_linear_jsd | 7 |
| **Total** | | **15** |

- **Repo:** https://github.com/linkedin/Liger-Kernel
- **Install:** `pip install liger-kernel`
- **License:** BSD-2-Clause
- **Paper:** https://arxiv.org/abs/2410.10989

---

## 4. FlashAttention — P1 (Attention Specialist)

| Category | Operators | Count |
|:---------|:----------|------:|
| **Attention** | flash_attention_forward (Triton), flash_attention_backward (Triton) | 2 |
| **Attention (CUDA)** | flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache | 3 |
| **Total Triton** | | **2** |

- **Repo:** https://github.com/Dao-AILab/flash-attention
- **Install:** `pip install flash-attn`
- **License:** BSD-3-Clause

---

## 5. Triton Official — P2 (Reference Implementations)

### 5.1 Triton Tutorials

| # | Operator | Notes |
|---|----------|-------|
| 01 | vector_add | Minimal example |
| 02 | softmax (fused) | Row-wise, online |
| 03 | matmul | With autotuning |
| 04 | dropout (low-memory) | Seed-based |
| 05 | layer_norm | Fused mean+var |
| 06 | fused_attention | FlashAttention v2 algorithm |
| 07 | libdevice ops | Special functions (sin, cos, etc.) |
| 08 | grouped_gemm | Batched, variable size |
| 09 | persistent_matmul | TMA, warp specialization |
| 10 | block_scaled_mm | MX format |
| **Total** | **10** | |

- **Repo:** https://github.com/triton-lang/triton (`python/tutorials/`)
- **License:** MIT

### 5.2 Triton Kernels Library (triton_kernels)

| Operator | Notes |
|:---------|:------|
| matmul (persistent + standard) | Production-quality |
| swiglu | Fused activation |
| reduce (generic) | Configurable reduction |
| topk | Selection |
| compaction | Tensor compaction |
| tensor ops | Utilities |
| **Total: 6** | |

- **Repo:** https://github.com/triton-lang/triton (`python/triton_kernels/`)
- **License:** MIT

---

## 6. HuggingFace Kernels-Community — P2

| Operator | Notes |
|:---------|:------|
| matmul_ogs | Optimized grouped-scatter matmul |
| gpt-oss MoE kernels | MoE routing + compute |
| Community contributions | Growing |
| **Total: ~5** | |

- **Repo:** https://github.com/huggingface/kernels-community
- **Hub:** https://huggingface.co/kernels-community
- **License:** Apache-2.0

---

## 7. vLLM — P1★ (Inference Engine)

**85+ Triton files.** Deeply optimized for inference.

### Standalone-Extractable Operators (12)

| Category | Operator | Path |
|:---------|:---------|:-----|
| **Activation** | silu_and_mul (SwiGLU) | `layers/activation.py` |
| **RoPE** | rotary_pos_emb | `v1/worker/gpu/mm/rope.py` |
| **RoPE** | mrope (multi-resolution) | `layers/rotary_embedding/mrope.py` |
| **Norm** | layernorm_gated | `layers/mamba/ops/layernorm_gated.py` |
| **Sampling** | topk_topp | `v1/sample/ops/topk_topp_triton.py` |
| **Sampling** | min_p | `v1/worker/gpu/sample/min_p.py` |
| **Sampling** | penalties (rep/freq/presence) | `v1/worker/gpu/sample/penalties.py` |
| **Logprob** | logprob | `v1/worker/gpu/sample/logprob.py` |
| **Quantization** | fp8_scaled_mm | `quantization/compressed_tensors/triton_scaled_mm.py` |
| **Quantization** | awq_triton | `quantization/awq_triton.py` |
| **Quantization** | int8_kernel | `quantization/utils/int8_utils.py` |
| **Quantization** | fp8_kernel | `quantization/utils/fp8_utils.py` |

### Framework-Bound Operators (~73)

| Category | Count | Examples |
|:---------|------:|:--------|
| Attention | 6 | triton_decode/prefill/unified_attention, merge_attn_states |
| MoE | 3 | fused_moe, fused_batched_moe |
| LoRA | 8 | lora_shrink/expand (fp8 + normal) |
| Mamba/SSM | 7 | causal_conv1d, mamba_ssm, ssd ops |
| FLA | 10 | chunk_delta_h, fused_recurrent, kda |
| Speculative | 3 | eagle speculator, rejection_sampler |
| Buffer/Memory | 4 | block_table, buffer_utils |
| Other | 5+ | batch_invariant, structured_outputs |

- **Repo:** https://github.com/vllm-project/vllm
- **License:** Apache-2.0

---

## 8. SGLang — P1★ (Inference Engine)

**80+ Triton files.** Significant overlap with vLLM, plus unique diffusion kernels.

### Standalone-Extractable Operators (10)

| Category | Operator | Path |
|:---------|:---------|:-----|
| **RoPE** | rotary_pos_emb | `layers/rotary_embedding/triton_kernels.py` |
| **Norm** | rmsnorm_onepass | `jit_kernel/diffusion/triton/rmsnorm_onepass.py` |
| **Norm** | layernorm_gated | `layers/attention/fla/layernorm_gated.py` |
| **Norm** | norm (diffusion) | `jit_kernel/diffusion/triton/norm.py` |
| **Elementwise** | fused elementwise | `layers/elementwise.py` |
| **Sampling** | fused_sampling | `layers/fused_sampling.py` |
| **Quantization** | fp8_kernel | `layers/quantization/fp8_kernel.py` |
| **Quantization** | int8_kernel | `layers/quantization/int8_kernel.py` |
| **Activation** | scale_shift | `jit_kernel/diffusion/triton/scale_shift.py` |
| **Activation** | gdn_fused_proj | `jit_kernel/triton/gdn_fused_proj.py` |

### Framework-Bound Operators (~70)

| Category | Count | Examples |
|:---------|------:|:--------|
| Attention | 5+ | decode/extend/prefill_attention |
| NSA | 4 | Native Sparse Attention kernels |
| MoE | 3 | fused_moe_triton_kernels |
| FLA | 12 | (shared with vLLM) |
| Mamba | 8 | causal_conv1d_triton, ssd ops |
| LoRA | 8 | chunked_sgmv, sgemm_lora |
| Other | 5+ | bitmask_ops, hash |

- **Repo:** https://github.com/sgl-project/sglang
- **License:** Apache-2.0

---

## 9. verl — P1★ (RL Training Framework)

**3 Triton files.** High-quality but limited.

| Category | Operator | Standalone? |
|:---------|:---------|:-----------:|
| **Loss** | fused_linear_cross_entropy (w/ token entropy) | ✅ |
| **Quantization** | fp8 kernel | ✅ |
| **Quantization** | QAT linear | ⚠️ coupled |

- **Repo:** https://github.com/verl-project/verl
- **License:** Apache-2.0

---

## 10. PyTorch Eager — P3

All standard PyTorch ops. No Triton — dispatches to cuBLAS/cuDNN/CUDA internally.

- **Access:** `torch.*`, `torch.nn.functional.*`
- **License:** BSD-3-Clause

---

## 11. torch.compile (Inductor) — P4

Auto-generated Triton from `torch.compile`. "What the compiler gives for free."

- Generates Triton for pointwise + reduction
- Dispatches matmul to cuBLAS/CUTLASS
- Fuses elementwise chains automatically
- **Extract:** `TORCH_COMPILE_DEBUG=1 python script.py`
- **License:** BSD-3-Clause

---

## 12. Unsloth — Reference Only

| Operator | Notes |
|:---------|:------|
| cross_entropy, rope, rms_layernorm, swiglu | Tightly coupled to model code |

- **Repo:** https://github.com/unslothai/unsloth
- **License:** Apache-2.0

---

## Cross-Source Coverage Matrix

Standalone-benchmarkable operators by source:

| Operator | cuBLAS | FlagGems | Liger | Triton | vLLM | SGLang | verl |
|:---------|:------:|:--------:|:-----:|:------:|:----:|:------:|:----:|
| matmul | ✅ | ✅ | — | ✅ | — | — | — |
| bmm | ✅ | ✅ | — | ✅ | — | — | — |
| softmax | ✅ | ✅ | — | ✅ | — | — | — |
| layer_norm | ✅ | ✅ | ✅ | ✅ | ✅† | ✅† | — |
| rms_norm | — | ✅ | ✅ | — | — | ✅ | — |
| gelu | — | ✅ | — | — | — | — | — |
| relu | — | ✅ | — | — | — | — | — |
| silu | — | ✅ | — | — | — | — | — |
| silu_and_mul | — | ✅ | ✅‡ | — | ✅ | — | — |
| swiglu | — | ✅ | ✅ | ✅ | — | — | — |
| geglu | — | ✅ | ✅ | — | — | — | — |
| rope | — | ✅ | ✅ | — | ✅ | ✅ | — |
| cross_entropy | ✅ | ✅ | ✅ | — | — | — | ✅ |
| fused_linear_ce | — | — | ✅ | — | — | — | ✅ |
| dropout | — | ✅ | — | ✅ | — | — | — |
| embedding | — | ✅ | — | — | — | — | — |
| flash_attention | ✅ | ✅ | — | ✅ | — | — | — |
| topk | — | ✅ | — | ✅ | ✅ | ✅ | — |
| fp8_scaled_mm | — | ✅ | — | — | ✅ | ✅ | ✅ |
| group_norm | ✅ | ✅ | ✅ | — | — | — | — |

> ✅ = standalone callable  ✅† = gated layernorm variant  ✅‡ = as part of swiglu

---

## Integration Plan

### Phase 1: Consolidate Current Sources (Stage 1 immediate)

Formalize runners for the 4 sources already in the benchmark:

| Task | Runner | Ops Covered |
|:-----|:-------|:------------|
| 1.1 | `CuBLASRunner` | matmul, softmax (via PyTorch) |
| 1.2 | `FlagGemsRunner` | matmul, softmax, gelu, relu, silu, layernorm, rmsnorm |
| 1.3 | `PyTorchEagerRunner` | all (reference) |
| 1.4 | `InductorRunner` | all (auto-fusion reference) |
| 1.5 | `ArkeRunner` | matmul, softmax (current templates) |

Run full Tier 3 for all 4 op categories.

### Phase 2: Add Liger + Triton Baselines

| Task | Runner | Ops Covered |
|:-----|:-------|:------------|
| 2.1 | `LigerRunner` | rms_norm, rope, swiglu, geglu, cross_entropy, fused_linear_ce |
| 2.2 | `TritonTutorialRunner` | matmul, softmax, layer_norm, fused_attention, dropout |
| 2.3 | `TritonKernelsRunner` | matmul (persistent), swiglu, topk |

### Phase 3: Extract vLLM / SGLang Standalone Ops

**Extraction approach:**
```
1. Pin a specific git commit (reproducibility)
2. Copy the @triton.jit kernel + wrapper into benchmarks/baselines/extracted/
3. Minimal modifications: remove framework imports, replace with direct torch ops
4. Document: source commit, file path, modifications, license
5. Verify numerical equivalence with PyTorch reference
```

| Task | Source | Ops to Extract |
|:-----|:-------|:---------------|
| 3.1 | vLLM | silu_and_mul, rope, layernorm_gated |
| 3.2 | vLLM | topk_topp, min_p, penalties |
| 3.3 | vLLM | fp8_scaled_mm, awq_triton |
| 3.4 | SGLang | rmsnorm_onepass, rope |
| 3.5 | SGLang | fused_sampling, elementwise |
| 3.6 | SGLang | fp8_kernel, int8_kernel |
| 3.7 | verl | fused_linear_cross_entropy |

Creates `VLLMRunner` and `SGLangRunner` classes.

### Phase 4: LLM-Direct Baseline (P5)

| Task | Description |
|:-----|:------------|
| 4.1 | Define prompt template for each op + shape |
| 4.2 | LLM generates Triton code → compile → verify → benchmark |
| 4.3 | Track token count per op (for G4 token efficiency gate) |
| 4.4 | Support multiple LLM providers (Anthropic, OpenAI, DeepSeek) |

### Phase 5: Full Coverage Matrix

| Task | Description |
|:-----|:------------|
| 5.1 | Auto-discover available runners at startup |
| 5.2 | For each (op, shape), run all applicable runners |
| 5.3 | Generate cross-source comparison table |
| 5.4 | Identify "best known" performance per (op, shape) pair |
| 5.5 | CI regression: alert if any runner regresses > 5% |

### Architecture

```
benchmarks/
├── baselines/
│   ├── base.py              # BaselineRunner ABC
│   ├── cublas_runner.py      # P0: cuBLAS via PyTorch
│   ├── flaggems_runner.py    # P1: FlagGems
│   ├── liger_runner.py       # P1: Liger-Kernel
│   ├── triton_tutorial_runner.py  # P2: Official tutorials
│   ├── triton_kernels_runner.py   # P2: triton_kernels library
│   ├── vllm_runner.py        # P1★: vLLM extracted ops
│   ├── sglang_runner.py      # P1★: SGLang extracted ops
│   ├── verl_runner.py        # P1★: verl extracted ops
│   ├── pytorch_eager_runner.py  # P3: PyTorch eager
│   ├── inductor_runner.py    # P4: torch.compile
│   ├── llm_direct_runner.py  # P5: LLM-direct Triton
│   ├── arke_runner.py        # P5: Arke pipeline
│   └── extracted/            # Extracted standalone kernels
│       ├── vllm/
│       │   ├── silu_and_mul.py
│       │   ├── rope.py
│       │   └── PROVENANCE.md   # commit hash, modifications
│       └── sglang/
│           ├── rmsnorm_onepass.py
│           └── PROVENANCE.md
├── benchmark-design.md              # Design document (existing)
├── shapes.py                 # Shape matrix (existing)
├── tasks.py                  # Task definitions (existing)
└── results/                  # Benchmark outputs
```

---

*Last updated: 2026-04-02*
