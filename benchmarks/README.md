# Arke Benchmark System

Comprehensive GPU operator benchmark framework with multi-tier baselines,
automated provenance tracking, and three-layer evaluation architecture.

## Quick Start

```bash
# Run everything (L1 + L2 + L3)
python -m benchmarks --all

# Run specific layer
python -m benchmarks --layer L1          # Single operator benchmarks
python -m benchmarks --layer L2          # Fused operator benchmarks
python -m benchmarks --layer L3          # E2E model benchmarks

# Run specific operators
python -m benchmarks --op matmul         # Matmul only
python -m benchmarks --op matmul,softmax # Multiple ops

# Generate report from existing results
python -m benchmarks --report

# Custom parameters
python -m benchmarks --layer L1 --warmup 200 --reps 500  # More iterations
python -m benchmarks --layer L3 --seq-len 128,256,512     # Specific seq lengths
```

## Three-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│  L3: E2E Model Inference                             │
│  GPT-2 / LLaMA forward pass                         │
│  Metric: latency (ms), throughput, correctness       │
├──────────────────────────────────────────────────────┤
│  L2: Fused Operators                                 │
│  matmul+relu, matmul+gelu                            │
│  Metric: μs latency, vs torch.compile auto-fusion    │
├──────────────────────────────────────────────────────┤
│  L1: Single Operators                                │
│  matmul, softmax, layernorm, gelu, relu, silu        │
│  Metric: μs latency, TFLOPS, % of vendor baseline    │
└──────────────────────────────────────────────────────┘
```

**Each layer answers a different question:**
- **L1:** Can Arke generate a kernel as fast as a hand-tuned one?
- **L2:** Can Arke fuse operators as well as expert-written fused kernels?
- **L3:** Does it actually make a real model faster?

## Baseline Tiers

Every result is compared against multiple baselines, ranked by expected performance:

| Tier | Name | Source | License |
|:----:|:-----|:-------|:--------|
| **P0** | cuBLAS / cuDNN | NVIDIA vendor libraries via PyTorch | NVIDIA EULA |
| **P1** | FlagGems | 200+ Triton ops (BAAI/FlagOS) | Apache-2.0 |
| **P1** | Liger-Kernel | LLM training Triton ops (LinkedIn) | BSD-2-Clause |
| **P3** | PyTorch eager | `torch.nn.functional` default dispatch | BSD-3-Clause |
| **P4** | torch.compile | Inductor auto-generated kernels | BSD-3-Clause |
| **P5** | Arke | Arke KernelCache Triton codegen | Apache-2.0 |

## Provenance Tracking

Every benchmark result carries full source attribution:

- **CSV `source` column** — each row records the baseline's package name, version, URL, and license
- **`sources.json`** — per-run manifest listing all baselines used
- **`hardware.json`** — GPU name, CUDA version, driver, PyTorch/Triton versions

Example CSV row:
```
matmul,square-1k,1024,1024,1024,FlagGems,1,"FlagGems 5.0.0 (BAAI/FlagOS) | https://github.com/flagos-ai/FlagGems | License: Apache-2.0",104.8,89.0,20.49
```

## Output Structure

Each run archives results with timestamps:

```
benchmarks/results/
├── L1/{timestamp}/
│   ├── config.json          # Run parameters
│   ├── hardware.json        # GPU/system info
│   ├── sources.json         # Baseline provenance manifest
│   ├── matmul_results.csv   # Per-op results
│   ├── softmax_results.csv
│   └── ...
├── L2/{timestamp}/
│   ├── matmul_relu_results.csv
│   └── matmul_gelu_results.csv
├── L3/{timestamp}/
│   └── gpt2_results.csv
└── report.md                # Auto-generated summary
```

## L1: Single Operators

Benchmarks individual operators across a shape matrix covering tiny to LLaMA-scale:

**Matmul shapes** (12):
| Tag | M×N×K | Notes |
|:----|:------|:------|
| tiny | 128×128×128 | Launch overhead test |
| small | 128×768×768 | GPT-2 c_proj |
| square-1k | 1024³ | Classic GEMM |
| square-4k | 4096³ | Large GEMM |
| llama-ffn | 4096×11008×4096 | LLaMA-7B FFN |
| ... | ... | 12 shapes total |

**Other ops:** softmax (5 shapes), layernorm (3), gelu/relu/silu (3 each)

## L2: Fused Operators

Compares three approaches to operator fusion:

1. **Separate ops** — matmul then activation (baseline)
2. **torch.compile** — Inductor auto-fusion
3. **FlagGems** — ATen backend with Triton fusion

Currently supports: `matmul+relu`, `matmul+gelu`

## L3: E2E Model

GPT-2 Small end-to-end inference benchmark:

| Mode | Description |
|:-----|:------------|
| **Eager** | PyTorch default (SDPA attention) |
| **torch.compile** | Inductor-optimized |
| **Arke** | Conv1D layers patched with KernelCache |

Reports: latency (mean/min/max/median), correctness (logit diff, top-1 match), peak memory.

## Scoring System

The report generator computes geometric-mean ratios:

| Indicator | Meaning |
|:---------:|:--------|
| 🟢 | ≥ 90% of baseline |
| 🟡 | ≥ 80% of baseline |
| 🔴 | < 80% of baseline |

## Dependencies

```bash
# Required (installed with Arke)
pip install torch triton

# Benchmark baselines
pip install flag-gems       # FlagGems (P1)
pip install liger-kernel    # Liger-Kernel (P1)

# Optional
pip install triton-kernels  # HuggingFace community kernels
```

Missing dependencies are handled gracefully — unavailable baselines are skipped with a warning.

## Design Documents

| Document | Description |
|:---------|:------------|
| [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) | Full design — architecture, scoring, quality gates, implementation phases |
| [OPERATOR_SOURCES.md](OPERATOR_SOURCES.md) | Complete operator source registry — 8 categories, installation guide |
| [SYNERGY.md](SYNERGY.md) | Benchmark ↔ Arke co-development strategy — target-driven development loop |
