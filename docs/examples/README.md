# Arke Examples

Organized examples demonstrating Arke's capabilities from language syntax to IR to end-to-end optimization.

## Directory Structure

```
examples/
├── operators/     .ak operator definitions (46 files, OT0-OT4)
├── ir/            Arke IR examples (SemanticIR + StrategyIR JSON)
├── pipelines/     End-to-end walkthrough docs (NL → .ak → IR → GPU)
└── README.md      This file
```

## operators/

Arke Language (`.ak`) source files for all 46 operators across 5 categories:

| Category | Ops | Examples |
|:---------|:----|:--------|
| **OT0** Elementwise | relu, gelu, silu, tanh, sigmoid, add, mul, neg, exp, rsqrt, cast, where | `00_relu.ak`, `03_gelu.ak`, ... |
| **OT1** Reduction | softmax, layernorm, rmsnorm, reduce_sum/max/mean, argmax, topk, cumsum | `02_softmax.ak`, `04_layernorm.ak`, ... |
| **OT2** Compute-Dense | matmul, batch_matmul, grouped_matmul, transpose, concat, split, gather, scatter, embedding, permute, copy | `01_matmul.ak`, `08_batch_matmul.ak`, ... |
| **OT3** Gated/Fused | matmul_gelu, rmsnorm_residual, swiglu, geglu, rope, cross_entropy, fused_linear_cross_entropy, quantize/dequantize | `05_matmul_gelu.ak`, `19_swiglu.ak`, ... |
| **OT4** Attention | flash_attention, grouped_query_attention, multi_latent_attention, cross_attention, paged_attention | `15_flash_attention.ak`, `16_grouped_query_attention.ak`, ... |

## ir/

Arke IR examples in `.arke-ir.json` format. Each file contains:
- `semantic_ir` — What to compute (pure math, no optimization)
- `strategy_ir` — How to optimize (tile, fuse, place decisions) — when a strategy block is defined

Generated from `.ak` sources via `arke.parser` + `arke.compiler.ast_to_strategy`.

| File | Category | Description |
|:-----|:---------|:------------|
| `matmul.arke-ir.json` | Cat A | Matrix multiplication — SemanticIR only (no strategy block in source) |
| `softmax.arke-ir.json` | Cat C | Softmax normalization — SemanticIR + StrategyIR |
| `matmul_gelu.arke-ir.json` | Fused | matmul + GELU — SemanticIR + StrategyIR (2-node fusion graph) |
| `flash_attention.arke-ir.json` | Cat B | Flash Attention — SemanticIR + StrategyIR |
| `swiglu.arke-ir.json` | Cat D | SwiGLU gated activation — SemanticIR + StrategyIR |
| `rmsnorm.arke-ir.json` | Cat C | RMSNorm — SemanticIR + StrategyIR |
| `rope.arke-ir.json` | Cat E | Rotary Position Embedding — SemanticIR + StrategyIR |

## pipelines/

End-to-end walkthrough documents showing the complete Arke pipeline:

| File | Description |
|:-----|:------------|
| `01_matmul.md` | Natural language → `.ak` → SemanticIR → StrategyIR → Triton kernel → GPU execution |

---

*To regenerate IR examples: `python scripts/generate_ir_examples.py` (TODO)*
