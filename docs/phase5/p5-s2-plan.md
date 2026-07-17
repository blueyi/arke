# Phase 5 — P5-S2 Plan: Full Op Coverage via LLVM IR

> **Goal:** All 46 ops correct via LLVM IR backend + geomean ≥ Phase 4 C-like
> **Exit criteria:** `arke run --backend llvm --kernel <any_op>` → success for all 46 ops, correctness verified vs torch reference.

## Architecture Extension

```
P5-S1 (matmul only):
  llvm_emitter.py → emit_llvm_ir_matmul()

P5-S2 (all 46 ops):
  llvm_elementwise.py  → 12 OT0 ops (relu, gelu, silu, etc.)
  llvm_rowwise.py      → 10 OT1 ops (softmax, layernorm, reduce_*, etc.)
  llvm_dense.py        → 10 OT2 ops (batch_matmul, transpose, etc.) [matmul in llvm_emitter.py]
  llvm_fused.py        → 8 OT3 ops (silu_and_mul, rope, cross_entropy, etc.)
  llvm_attention.py    → 5 OT4 ops (flash_attention, gqa, paged, etc.)
  llvm_backend.py      → dispatch table (like CudaCBackend._EMITTERS)
```

## Op Coverage by Category

### OT0 — Elementwise (12 ops)
| Op | Pattern | Complexity |
|---|---|---|
| relu, gelu, silu, tanh, sigmoid, exp, neg, rsqrt | Unary point-wise | Low |
| add, mul | Binary point-wise | Low |
| cast | Type conversion | Low |
| where_ | Ternary select | Low |

### OT1 — Reduction (10 ops)
| Op | Pattern | Complexity |
|---|---|---|
| softmax | Row reduce (max + exp + sum) | Medium |
| layernorm, rmsnorm | Row normalize | Medium |
| reduce_sum, reduce_max, reduce_mean | Row reduce | Low-Medium |
| argmax | Row argmax | Medium |
| cumsum | Row prefix sum | Medium |
| topk | Row top-k selection | High |
| rmsnorm_residual | RMSNorm + residual add | Medium |

### OT2 — Data Movement & Dense (11 ops)
| Op | Pattern | Complexity |
|---|---|---|
| matmul | Already done (P5-S1) | — |
| batch_matmul | 3-D loop matmul | Medium |
| grouped_matmul | Variable-size batched | High |
| transpose, permute | Index remapping | Low |
| copy_, embedding, gather, scatter | Memory movement | Medium |
| concat, split | Tensor join/split | Medium |

### OT3 — Fused Compound (8 ops)
| Op | Pattern | Complexity |
|---|---|---|
| silu_and_mul, gelu_and_mul, swiglu_packed | Gated activation | Low-Medium |
| cross_entropy | Softmax + log + nll | Medium |
| fused_linear_cross_entropy | Matmul + CE | High |
| rope | Rotary position encoding | Medium |
| quantize_per_token, dequantize_per_channel | Quant/dequant | Medium |

### OT4 — Attention (5 ops)
| Op | Pattern | Complexity |
|---|---|---|
| flash_attention | QKV attention | High |
| grouped_query_attention | GQA variant | High |
| cross_attention | Cross-sequence | High |
| paged_attention | KV-cache paged | High |
| multi_latent_attention | MLA (DeepSeek) | High |

## Implementation Strategy

1. **Correctness first, performance later** (P5-S3 is for perf tuning)
2. **Elementwise: one thread per element** — simplest possible mapping
3. **Reductions: one block per row** — each block reduces N elements
4. **Attention: naive nested loop** — correct but slow; optimization in S3
5. **Reuse CudaCKernel dataclass** — same execution infrastructure as P5-S1

## Performance Target (P5-S2)

- **Correctness:** 46/46 ops pass vs torch reference
- **Perf floor:** geomean ≥ Phase 4 CUDA-C (measurement in bench pass at end)
- Note: many ops will naturally match or exceed C-like because llc optimizes well

## Verification

```bash
# Quick check: all ops supported
python -c "from arke.backend.llvm_backend import LLVMBackend; b = LLVMBackend(); ..."

# Full correctness suite
python -m pytest tests/backend/test_llvm_backend.py -v

# Benchmark comparison (after correctness passes)
python -m benchmarks.bench_l1 --backend llvm --tier 1
```

---
*Created: 2026-07-17*
