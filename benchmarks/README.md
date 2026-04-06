# Arke Benchmark System

GPU operator benchmark framework with multi-tier baselines, automated provenance tracking,
and a classification system based on Operator Tier (OT), Shape Tier (ST), and Benchmark Level (BL).

> Design document: [`docs/design/benchmark-design.md`](../docs/design/benchmark-design.md)
> Protocol: [`docs/design/benchmark/benchmark-protocol.md`](../../docs/design/benchmark/benchmark-protocol.md)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-benchmark.txt

# Run default benchmark (BL2: OT0-2 × ST1-2, L1)
arke bench

# Quick smoke test (<30s)
arke bench --bl 1

# Complete suite (all ops × all shapes, L1+L2)
arke bench --bl 5

# E2E model validation (L1+L2+L3)
arke bench --bl 6
arke bench --bl 6 --model gpt2
```

> **Alternative entry point:** `python -m benchmarks` is equivalent to `arke bench`.

---

## CLI Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--bl N` | **Benchmark Level** (1–6). Primary control. | `2` |
| `--ot N,N` | Operator Tier filter (0–4, comma-separated). | from BL |
| `--st N,N` | Shape Tier filter (1–4, comma-separated). | from BL |
| `--layer L` | Evaluation Layer (L1/L2/L3). | from BL |
| `--op name` | Specific operator(s), comma-separated. | from OT |
| `--shapes tag,tag` | Specific shape tags, comma-separated. | all |
| `--baselines name,name` | Baseline methods (comma-separated, or `all`). | all available |
| `--model name` | Model for L3/BL6 (e.g. `gpt2`). | — |
| `--warmup N` | Warmup iterations. | `200` |
| `--reps N` | Measurement repetitions. | `500` |
| `--seq-len N,N` | Sequence lengths for L3 (comma-separated). | `128,256,512` |
| `-v` | Verbose output. | — |

### Benchmark Level Expansion

| BL | Operator Tiers | Shape Tiers | Layers | Time |
|:--:|:---------------|:------------|:-------|:-----|
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

### Validation Rules

- `--layer L3` automatically implies BL6
- `--layer L2` requires OT3+ (auto-expanded)
- `--ot 4` implies ST4 shapes (attention ops)

---

## Examples

```bash
# Filter by Operator Tier
arke bench --ot 0                           # Elementwise only
arke bench --ot 2,4                         # Dense + Attention only
arke bench --bl 5 --ot 4                    # All shapes, attention only

# Filter by Shape Tier
arke bench --st 4                           # Production shapes only
arke bench --bl 3 --st 3                    # Stress shapes only

# Filter by specific operator
arke bench --op matmul                      # All shapes for matmul
arke bench --op matmul --st 4              # Matmul production shapes
arke bench --op matmul,softmax --bl 3      # Matmul+softmax stress shapes

# Specific shapes
arke bench --op matmul --shapes square-1k,square-4k

# Baseline control
arke bench --baselines cublas,flaggems,arke
arke bench --baselines all

# Report & comparison
arke bench report                           # Latest results
arke bench report {run_id}                  # Specific run
arke bench diff {run_id_1} {run_id_2}       # Compare runs (planned)
arke bench history --op matmul              # Performance trend (planned)
```

---

## Output Structure

Each run produces a timestamped directory:

```
benchmarks/results/{run_id}/
├── config.json          # Run parameters (bl, ot, st, layer, baselines)
├── hardware.json        # GPU, CUDA, PyTorch/Triton versions
├── L1/
│   └── OT{n}/
│       └── perf_{op}.csv
├── L2/
│   └── perf_{fused_op}.csv
├── L3/
│   └── {model}/
│       └── perf_e2e.csv
├── summary.json         # Aggregated geomean scores
├── PERF_ALL.csv         # Unified CSV (41-column schema)
└── report.md            # Human-readable report
```

CSV schema: [`docs/design/benchmark/benchmark-csv-spec.md`](../../docs/design/benchmark/benchmark-csv-spec.md)

---

## Gate System

```bash
# Run specific gate
python -m benchmarks.gate G0 --tier 2
python -m benchmarks.gate G3 --tier 2 --live --archive
python -m benchmarks.gate G6 --tier 2
```

---

## Dependencies

**Core** (`requirements.txt`):
numpy, jinja2, httpx, lark, click, rich

**Benchmark** (`requirements-benchmark.txt`):
torch, triton, flag-gems, liger-kernel

**Optional:**
```bash
pip install flash-attn --no-build-isolation   # FlashAttention (P1)
pip install triton-kernels                     # HuggingFace community kernels
pip install nvidia-cutlass                     # CUTLASS C++ GEMM baselines
```

Missing baselines are gracefully skipped with a warning.
