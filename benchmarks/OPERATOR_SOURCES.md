# Arke Benchmark Operator Sources

Centralized registry of GPU operators used as benchmark baselines.  
Each entry specifies: what it does, where it comes from, how to invoke it, and license.

---

## 1. NVIDIA Vendor Libraries (CUDA)

The gold standard — closed-source, hardware-specific, vendor-tuned.

### 1.1 cuBLAS (via PyTorch)

| Field | Value |
|:------|:------|
| **Operators** | matmul, batch_matmul, addmm, gemm |
| **Access** | `torch.matmul()`, `torch.mm()`, `torch.addmm()` |
| **Install** | Bundled with PyTorch CUDA build |
| **Version** | CUDA 13.1 (via PyTorch 2.6.0+cu124) |
| **License** | NVIDIA EULA (proprietary) |
| **Notes** | PyTorch automatically dispatches to cuBLAS for GEMM ops. Best performance for matmul on NVIDIA GPUs. |

### 1.2 cuDNN (via PyTorch SDPA)

| Field | Value |
|:------|:------|
| **Operators** | fused_attention (SDPA), conv2d, batchnorm |
| **Access** | `torch.nn.functional.scaled_dot_product_attention()` |
| **Install** | Bundled with PyTorch CUDA build |
| **Version** | cuDNN 9.x (via PyTorch 2.6.0) |
| **License** | NVIDIA EULA (proprietary) |
| **Notes** | SDPA picks the best backend (FlashAttention, cuDNN, math). |

### 1.3 CUTLASS

| Field | Value |
|:------|:------|
| **Operators** | gemm, grouped_gemm, conv2d, fused_gemm_epilogue |
| **Repo** | https://github.com/NVIDIA/cutlass |
| **Install** | `pip install nvidia-cutlass` or build from source |
| **License** | BSD-3-Clause |
| **Notes** | Template library for GEMM. torch.compile/inductor uses CUTLASS templates for some matmul configs. Useful as a C++ baseline alternative to cuBLAS. |

---

## 2. Triton Official (triton-lang)

Reference implementations from the Triton project itself.

### 2.1 Triton Tutorials

| Field | Value |
|:------|:------|
| **Operators** | vector_add, softmax, matmul, dropout, layer_norm, fused_attention, group_gemm, persistent_matmul |
| **Repo** | https://github.com/triton-lang/triton (`python/tutorials/`) |
| **URL** | https://triton-lang.org/main/getting-started/tutorials/ |
| **Install** | Part of `triton` package source |
| **License** | MIT |
| **Notes** | Official tutorial kernels. Well-documented, pedagogical quality. The matmul tutorial includes autotuning. Fused attention is based on FlashAttention v2 algorithm. |

### 2.2 triton-lang/kernels

| Field | Value |
|:------|:------|
| **Operators** | (early stage — check repo for current state) |
| **Repo** | https://github.com/triton-lang/kernels |
| **License** | MIT |
| **Notes** | Official kernel library under development. |

---

## 3. FlagGems (BAAI / FlagOS)

Production-grade Triton operator library, 200+ ops, PyTorch ATen backend registration.

| Field | Value |
|:------|:------|
| **Operators** | matmul, softmax, layernorm, rmsnorm, gelu, silu, relu, dropout, cross_entropy, embedding, add, mul, div, exp, log, pow, sin, cos, where, sum, mean, max, min, cumsum, topk, sort, unique, scatter, gather, index_select, bmm, addmm, outer, mv, ... (200+ total) |
| **Repo** | https://github.com/flagos-ai/FlagGems |
| **Install** | `pip install flag-gems` |
| **License** | Apache-2.0 |
| **Notes** | Registers as PyTorch ATen backend — drop-in replacement for eager ops. Provides its own benchmark framework. Performance generally matches or exceeds PyTorch ATen on NVIDIA GPUs. Multi-backend (NVIDIA + others). |

### Key operators for Arke benchmarks:

| Op | FlagGems module | Shape notes |
|:---|:----------------|:------------|
| matmul | `flag_gems.ops.mm` | Standard GEMM, autotuned |
| softmax | `flag_gems.ops.softmax` | Row-wise, fused |
| layernorm | `flag_gems.ops.layernorm` | Fused mean+var |
| gelu | `flag_gems.ops.gelu` | Elementwise, fused |
| rmsnorm | `flag_gems.ops.rmsnorm` | LLM-specific |
| cross_entropy | `flag_gems.ops.cross_entropy` | Fused log_softmax+nll |

---

## 4. Liger-Kernel (LinkedIn)

LLM training-focused Triton kernels with aggressive operator fusion.

| Field | Value |
|:------|:------|
| **Operators** | rmsnorm, rope, swiglu, cross_entropy, fused_linear_cross_entropy, geglu, layernorm, fused_linear_jsd, kto_loss, dpo_loss, orpo_loss, cpo_loss, simpo_loss |
| **Repo** | https://github.com/linkedin/Liger-Kernel |
| **Paper** | https://arxiv.org/abs/2410.10989 |
| **Install** | `pip install liger-kernel` |
| **License** | BSD-2-Clause |
| **Notes** | Focused on training efficiency (memory + throughput). Kernels fuse forward+backward. Key innovation: chunked cross-entropy and fused linear+loss. Less relevant for inference-only benchmarks but excellent for fused-op comparisons. |

### Key operators for Arke benchmarks:

| Op | Liger module | Notes |
|:---|:-------------|:------|
| rmsnorm | `liger_kernel.ops.rms_norm` | Fused, in-place grad |
| rope | `liger_kernel.ops.rope` | Rotary position embedding |
| swiglu | `liger_kernel.ops.swiglu` | Fused SwiGLU activation |
| cross_entropy | `liger_kernel.ops.cross_entropy` | Memory-efficient chunked |
| geglu | `liger_kernel.ops.geglu` | Fused GeGLU activation |

---

## 5. FlashAttention (Dao-AILab)

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

---

## 6. PyTorch Inductor (torch.compile)

Auto-generated Triton kernels from `torch.compile`.

| Field | Value |
|:------|:------|
| **Operators** | All elementwise, reductions, some matmul (via CUTLASS/Triton templates) |
| **Access** | `torch.compile(model)` → inspect via `TORCH_COMPILE_DEBUG=1` |
| **Source** | `torch/_inductor/codegen/triton.py` |
| **License** | BSD-3-Clause (PyTorch) |
| **Notes** | Inductor generates Triton for pointwise/reduction ops, dispatches matmul to cuBLAS/CUTLASS. The generated kernels include fusion optimizations. Useful as "what the compiler generates" baseline. |

### Extracting inductor kernels:

```bash
TORCH_COMPILE_DEBUG=1 python script.py
# Kernels saved to torch_compile_debug/run_*/output_code.py
```

---

## 7. HuggingFace Kernels Community

Community-contributed Triton kernels hosted on HuggingFace.

| Field | Value |
|:------|:------|
| **Operators** | matmul_ogs, various community kernels |
| **Repo** | https://github.com/huggingface/kernels-community |
| **HF Hub** | https://huggingface.co/kernels-community/triton_kernels |
| **Install** | `pip install triton-kernels` |
| **License** | Apache-2.0 |
| **Notes** | Growing collection. `matmul_ogs` is an optimized grouped-scatter matmul. Check for new additions periodically. |

---

## 8. Unsloth

Fast LLM fine-tuning with custom Triton kernels.

| Field | Value |
|:------|:------|
| **Operators** | cross_entropy, rope, rms_layernorm, swiglu (internal, tightly coupled) |
| **Repo** | https://github.com/unslothai/unsloth |
| **License** | Apache-2.0 |
| **Notes** | Kernels are embedded in model-specific code, not easily extracted as standalone. Useful for E2E training throughput comparison rather than op-level benchmarks. |

---

## Operator → Source Mapping (for Arke benchmarks)

Which source to use as baseline for each operator type:

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
| **elementwise (add/mul)** | PyTorch native | FlagGems | Trivially memory-bound |
| **embedding** | PyTorch (`nn.Embedding`) | FlagGems | |
| **conv1d/conv2d** | cuDNN (via PyTorch) | FlagGems | |

---

## Installation Guide

```bash
# Core (already installed)
pip install torch triton

# FlagGems — 200+ Triton operators
pip install flag-gems

# Liger-Kernel — LLM training kernels
pip install liger-kernel

# FlashAttention — fused attention (requires CUDA build)
pip install flash-attn --no-build-isolation

# HuggingFace community kernels
pip install triton-kernels

# CUTLASS (optional, for C++ GEMM baselines)
pip install nvidia-cutlass
```

---

## Usage in Arke Benchmarks

```python
import torch
import flag_gems

# Register FlagGems as ATen backend
flag_gems.enable()

# Now torch.matmul() uses FlagGems Triton kernel instead of cuBLAS
y = torch.matmul(a, b)

# Or use directly
from flag_gems.ops import mm as flaggems_mm
y = flaggems_mm(a, b)

# Liger-Kernel (standalone ops)
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
y = LigerRMSNormFunction.apply(x, weight, eps)

# Triton tutorial matmul
# Copy from triton/python/tutorials/03-matrix-multiplication.py
```

---

*Last updated: 2026-04-02*
*Maintain this file when adding new benchmark operators or discovering new sources.*
