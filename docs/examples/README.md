# Arke Examples

Organized examples demonstrating Arke's capabilities from language syntax to IR to end-to-end optimization.

## Directory Structure

```
examples/
├── operators/     .ak operator definitions (46 files, OT0-OT4)
├── ir/            Arke IR examples (.akir) — TODO: generate after G6 v2
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

> **TODO:** Arke IR examples (`.akir` format) will be generated after G6 v2 implementation completes the multi-layer IR architecture. See `docs/spec/arke-ir-spec-design.md` for the IR design.

## pipelines/

End-to-end walkthrough documents showing the complete Arke pipeline:

| File | Description |
|:-----|:------------|
| `01_matmul.md` | Natural language → `.ak` → SemanticIR → StrategyIR → Triton kernel → GPU execution |
