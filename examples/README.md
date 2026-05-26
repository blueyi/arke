# Arke Examples

Organized examples demonstrating Arke's capabilities from language syntax to IR to end-to-end optimization.

## Directory Structure

```
examples/
├── operators/     .ak operator definitions (46+ files, OT0-OT4 + Stage 7 L2 surfaces)
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
| **OT3** Gated/Fused | matmul_gelu, rmsnorm_residual, silu_and_mul, gelu_and_mul, rope, cross_entropy, fused_linear_cross_entropy, quantize/dequantize | `05_matmul_gelu.ak`, `19_silu_and_mul.ak`, ... |
| **OT4** Attention | flash_attention, grouped_query_attention, multi_latent_attention, cross_attention, paged_attention | `15_flash_attention.ak`, `16_grouped_query_attention.ak`, ... |

### Stage 7 L2 surface examples

`examples/operators/l2/` contains file-backed fusion-surface kernels used by the Stage 7 coverage ledger and audit flow:

- `matmul_relu.ak` — explicit matmul + relu epilogue fusion surface
- `linear_ce.ak` — streamed linear + cross entropy fusion surface
- `qkv_fa.ak` — QKV projection + flash attention producer/consumer fusion surface

The gated activation examples are also Stage 7 L2 fusion surfaces while staying in the canonical OT3 example set:

- `19_silu_and_mul.ak` — compact SwiGLU op with explicit `fuse(ops=["silu", "mul"], fusion_type="epilogue")`
- `20_gelu_and_mul.ak` — compact GeGLU op with explicit `fuse(ops=["gelu", "mul"], fusion_type="epilogue")`

## ir/

> **TODO:** Arke IR examples (`.akir` format) will be generated after G6 v2 implementation completes the multi-layer IR architecture. See `../docs/spec/arke-ir-spec-design.md` for the IR design.

## pipelines/

End-to-end walkthrough documents showing the complete Arke pipeline:

| File | Description |
|:-----|:------------|
| `01_matmul.md` | Natural language → `.ak` → SemanticIR → StrategyIR → Triton kernel → GPU execution |
