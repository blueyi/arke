# Arke Benchmark System

GPU operator benchmark framework with multi-tier baselines, automated provenance tracking,
and a classification system based on Operator Tier (OT), Shape Tier (ST), and Benchmark Level (BL).

> Design document: [`docs/design/benchmark-design.md`](../docs/design/benchmark-design.md)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-benchmark.txt

# Run default benchmark (BL2: basic ops × standard shapes)
python -m benchmarks

# Quick smoke test
python -m benchmarks --bl 1

# Full suite (all ops × all shapes)
python -m benchmarks --bl 5

# E2E model validation
python -m benchmarks --bl 6
```

---

## CLI Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--bl N` | **Benchmark Level** (1–6). Primary control. | `2` |
| `--ot N,N` | Operator Tier filter (0–4). Overrides BL. | from BL |
| `--st N,N` | Shape Tier filter (1–4). Overrides BL. | from BL |
| `--layer L` | Evaluation Layer (L1/L2/L3). Overrides BL. | from BL |
| `--op name` | Specific operator(s), comma-separated. | from OT |
| `--warmup N` | Warmup iterations. | `200` |
| `--reps N` | Measurement repetitions. | `500` |
| `--report` | Generate report from existing results. | — |
| `-v` | Verbose output. | — |

### Benchmark Level Expansion

| BL | Operator Tiers | Shape Tiers | Layers | Time |
|:--:|:--------------|:-----------|:-------|:-----|
| 1 | OT0–OT2 | ST1 | L1 | <30s |
| 2 | OT0–OT2 | ST1–ST2 | L1 | ~5 min |
| 3 | OT0–OT2 | ST1–ST3 | L1 | ~15 min |
| 4 | OT0–OT4 | ST1–ST2 | L1, L2 | ~30 min |
| 5 | OT0–OT4 | ST1–ST4 | L1, L2 | ~60 min |
| 6 | Model-Complete | Model-Real | L1, L2, L3 | ~90 min |

### Operator Tiers

| OT | Name | Operators |
|:--:|:-----|:----------|
| 0 | Elementwise | relu, gelu, silu, add, mul |
| 1 | Reduction | softmax, layernorm, rmsnorm, rmsnorm_residual, reduce_sum, reduce_max |
| 2 | Compute-Dense | matmul, batch_matmul, grouped_matmul, transpose |
| 3 | Gated Activation | swiglu, geglu |
| 4 | Attention | flash_attention, grouped_query_attention, multi_latent_attention |

---

## Examples

```bash
# Only elementwise operators
python -m benchmarks --ot 0

# Only attention operators with production shapes
python -m benchmarks --ot 4 --st 4

# Matmul with stress shapes
python -m benchmarks --op matmul --st 3

# L2 fused operators
python -m benchmarks --layer L2

# L3 E2E model (GPT-2)
python -m benchmarks --layer L3

# Generate report from existing results
python -m benchmarks --report
```

---

## Output Structure

```
benchmarks/results/{run_id}/
├── config.json          # Run configuration (bl, ot, st, layer)
├── hardware.json        # GPU, CUDA, PyTorch/Triton versions
├── L1/
│   └── {op}_results.csv
├── L2/
│   └── {fused_op}_results.csv
├── L3/
│   └── {model}/
│       └── results.csv
├── PERF_ALL.csv         # Unified CSV (41-column schema)
├── summary.json         # Aggregated scores
└── report.md            # Human-readable report
```

CSV schema: [`docs/design/benchmark/benchmark-csv-spec.md`](../docs/design/benchmark/benchmark-csv-spec.md)

---

## Gate System

```bash
# Run specific gate
python -m benchmarks.gate G0 --tier 2

# Run G6 (language completeness)
python -m benchmarks.gate G6 --tier 2
```

---

## Dependencies

**Core** (in `requirements.txt`):
- numpy, jinja2, httpx, lark, click, rich

**Benchmark** (in `requirements-benchmark.txt`):
- torch, triton (GPU runtime)
- flag-gems (P1 Expert Triton baseline)
- liger-kernel (P1 LLM training kernels)
- flash-attn (P1 FlashAttention, optional)

Missing baselines are gracefully skipped with a warning.
