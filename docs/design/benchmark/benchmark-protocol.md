# Arke Benchmark — Protocol, Scoring & Implementation

Measurement protocol, scoring system, CLI interface, output structure, and implementation status.

→ Parent: [`benchmark-design.md`](../benchmark-design.md)

---

## Design Goal

**Default target: all operators correct on all shapes, performance ≥ P0 (vendor-optimized).**

```
Correctness: 100% pass rate across all OT × ST combinations
Performance: Arke latency ≤ P0 vendor baseline (ratio ≥ 1.0)
```

When P0 is unavailable for an operator (e.g. rmsnorm, swiglu), the primary
baseline falls back to P1 (expert Triton). See
[`benchmark-ops.md`](./benchmark-ops.md) for per-op primary baseline.

---

## Measurement Protocol

### L1: Single Operator

```python
# 1. Warmup: 200 iterations (triggers autotune, JIT)
for _ in range(200):
    kernel(inputs)
torch.cuda.synchronize()

# 2. Measure: CUDA events, 500 iterations
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(500):
    kernel(inputs)
end.record()
torch.cuda.synchronize()
latency_us = start.elapsed_time(end) / 500 * 1000

# Alternative: triton.testing.do_bench (FlagGems standard)
from triton.testing import do_bench
latency_ms = do_bench(lambda: kernel(inputs), warmup=200, rep=500)
```

### L2: Fused Operator

Same protocol as L1. Compares: (a) unfused sequential, (b) torch.compile fusion,
(c) expert fusion (FlagGems/Liger), (d) Arke fusion.

### L3: E2E Model (= BL6)

```python
# 1. Load model, apply Arke kernel patches (KernelCache)
# 2. Warmup: 50 forward passes
# 3. Measure: 200 forward passes with CUDA events
# 4. Report: mean latency (ms), throughput (tok/s)
# 5. Correctness: top-1 logit match + max absolute diff
```

### Correctness Tolerances

| Dtype | atol | rtol | Method |
|:------|-----:|-----:|:-------|
| f16 | 0.1 | 0.05 | `np.allclose` + max/mean diff |
| f32 | 1e-5 | 1e-4 | `np.allclose` |
| bf16 | 0.2 | 0.1 | `np.allclose` |

### Metrics Collected Per Run

See [`benchmark-csv-spec.md`](./benchmark-csv-spec.md) for the full 41-column CSV schema.
Key metrics:

| Metric | Unit | Description |
|:-------|:-----|:------------|
| `latency_us` | μs | Median kernel latency |
| `tflops` | TFLOPS | Achieved throughput (compute-bound ops) |
| `gbps` | GB/s | Achieved bandwidth (memory-bound ops) |
| `ratio_vs_baseline` | ratio | `baseline_latency / arke_latency` (>1 = Arke faster) |
| `correct` | bool | Passes numerical tolerance |
| `compile_time_s` | s | Time to generate + compile kernel |

---

## Scoring System

### Correctness Gate (binary)

Every (operator, shape, dtype) must pass correctness. **No exceptions.**
A single correctness failure blocks the entire benchmark level from passing.

### Performance Score (per shape)

```
ratio = P0_baseline_latency / arke_latency     (>1.0 = Arke faster than vendor)
```

When P0 is unavailable, use the primary baseline defined in benchmark-ops.md.

### Aggregation

```
op_score     = geomean(ratio across all shapes for one operator)
tier_score   = geomean(op_scores across all operators in one OT tier)
level_score  = geomean(tier_scores across all OT tiers in one BL level)
arke_score   = 0.3 × L1_level_score + 0.3 × L2_level_score + 0.4 × L3_level_score
```

L3 weighted highest because real-world E2E impact matters most.

### Report Indicators

| Indicator | Meaning |
|:---------:|:--------|
| 🟢 | ratio ≥ 1.0 (Arke ≥ vendor) |
| 🟡 | ratio ≥ 0.8 (within 20%) |
| 🔴 | ratio < 0.8 |

### Exclusion Rules

| Scenario | Handling | Reason |
|:---------|:---------|:-------|
| M ≤ 32 (matmul) | Correctness required; perf excluded from score | Triton ~55μs launch floor |
| N ≤ 32 (softmax) | Correctness required; perf excluded from score | Same |
| M×N ≤ 1024 (elementwise) | Correctness required; perf excluded from score | Kernel-launch dominated |
| OOM shapes | Skip, record "OOM" | Hardware VRAM limit |
| Triton compile timeout (>60s) | Record "TIMEOUT", correctness = fail | Template may need fix |

---

## CLI Interface

### Design Principle

CLI parameters directly map to the benchmark classification system:

| Parameter | Maps to | Values |
|:----------|:--------|:-------|
| `--bl` | Benchmark Level | `1`–`6` (default: `2`) |
| `--ot` | Operator Tier filter | `0`–`4`, comma-separated |
| `--st` | Shape Tier filter | `1`–`4`, comma-separated |
| `--layer` | Evaluation Layer | `L1`, `L2`, `L3` |
| `--op` | Specific operator(s) | operator name, comma-separated |

**`--bl` is the primary control.** It determines the default OT and ST ranges.
`--ot`, `--st`, `--layer`, `--op` are overrides for fine-grained control.

### Default Behavior

```
arke bench              → BL2 (OT0–OT2 × ST1–ST2, L1 only)
arke bench --bl 5       → BL5 (OT0–OT4 × ST1–ST4, L1+L2)
arke bench --bl 6       → BL6 (Model-Complete, L1+L2+L3)
```

### BL → Default Expansion

| `--bl` | Default OT | Default ST | Default Layer | Description |
|:------:|:-----------|:-----------|:--------------|:------------|
| `1` | OT0–OT2 | ST1 | L1 | Smoke test, <30s |
| `2` | OT0–OT2 | ST1–ST2 | L1 | Daily CI, ~5 min |
| `3` | OT0–OT2 | ST1–ST3 | L1 | Gate validation |
| `4` | OT0–OT4 | ST1–ST2 | L1, L2 | Operator completeness |
| `5` | OT0–OT4 | ST1–ST4 | L1, L2 | Complete suite |
| `6` | Model-Complete | Model-Real | L1, L2, L3 | E2E model validation |

### Examples

```bash
# Quick smoke test (BL1)
arke bench --bl 1

# Daily CI (BL2, default)
arke bench

# Gate validation with full stress shapes
arke bench --bl 3

# All operators, standard shapes
arke bench --bl 4

# Complete benchmark (all ops × all shapes)
arke bench --bl 5

# E2E model validation
arke bench --bl 6
arke bench --bl 6 --model gpt2                     # Specific model
arke bench --bl 6 --model llama2-7b --seq-len 512,2048

# Filter by Operator Tier
arke bench --ot 0                                   # Elementwise only
arke bench --ot 2,4                                 # Dense + Attention only
arke bench --bl 5 --ot 4                            # All shapes, attention only

# Filter by Shape Tier
arke bench --st 4                                   # Production shapes only
arke bench --bl 3 --st 3                            # Stress shapes only

# Filter by Evaluation Layer
arke bench --layer L1                               # Single ops only
arke bench --layer L2                               # Fused ops only
arke bench --layer L3                               # E2E only (implies BL6)

# Filter by specific operator
arke bench --op matmul                              # All shapes for matmul
arke bench --op matmul --st 4                       # matmul production shapes
arke bench --op matmul,softmax --bl 3               # matmul+softmax, stress shapes

# Specific shapes
arke bench --op matmul --shapes square-1k,square-4k

# Baseline control
arke bench --baselines cublas,flaggems,arke         # Only these baselines
arke bench --baselines all                          # All available baselines

# Report & comparison
arke bench report {run_id}                          # Generate report
arke bench diff {run_id_1} {run_id_2}               # Compare two runs
arke bench history --op matmul --shape square-4k    # Performance trend
```

### Validation Rules

- `--layer L3` automatically sets `--bl 6` (L3 ≡ BL6)
- `--layer L2` requires `--bl ≥ 4` (L2 needs OT3+)
- `--ot 4` requires `--st 4` (attention ops only have ST4 shapes)
- `--bl 6 --op matmul` is valid (runs only matmul shapes from the model graph)

### Current Implementation (python -m benchmarks)

Existing CLI is not yet aligned with BL/OT/ST. Migration path:

```bash
# Current (legacy)                        # New equivalent
python -m benchmarks --all                → arke bench --bl 6
python -m benchmarks --layer L1           → arke bench --layer L1 --bl 2
python -m benchmarks --op matmul          → arke bench --op matmul
python -m benchmarks --op matmul --tier 2 → arke bench --op matmul --st 2
python -m benchmarks --report             → arke bench report latest
```

---

## Output Structure & Provenance Tracking

### Directory Layout

```
benchmarks/results/{run_id}/
├── config.json              # Run configuration (bl, ot, st, layer)
├── hardware.json            # GPU, driver, CUDA, PyTorch/Triton versions
├── L1/
│   ├── OT0/                 # Elementwise results
│   │   ├── perf_relu.csv
│   │   ├── perf_gelu.csv
│   │   └── ...
│   ├── OT1/                 # Reduction results
│   ├── OT2/                 # Compute-dense results
│   ├── OT3/                 # Gated activation results
│   └── OT4/                 # Attention results
├── L2/
│   ├── perf_matmul_relu.csv
│   └── ...
├── L3/
│   └── {model}/
│       ├── perf_e2e.csv
│       └── config.json      # Model, seq_len, patches
├── summary.json             # Aggregated scores by BL/OT/ST
├── PERF_ALL.csv             # All rows in unified CSV v2.0 schema
└── report.md                # Human-readable report
```

### Provenance Tracking

Every result carries full source attribution:
- **CSV schema** — unified 41-column format ([`benchmark-csv-spec.md`](./benchmark-csv-spec.md))
- **`config.json`** — run parameters: bl, ot, st, layer, baselines
- **`hardware.json`** — GPU name, CUDA version, driver, framework versions

---

## Benchmark-Driven Development

The benchmark is the **target state definition** for Arke development.

### Capability Mapping

> **Legend:** ✅ = done, 🔶 = IR defined but no codegen template, ⬜ = not started

| Benchmark Target | Primary Baseline | IR | Template | Codegen | Strategy |
|:-----------------|:-----------------|:--:|:--------:|:-------:|:--------:|
| **OT0 Elementwise** | | | | | |
| L1 relu | PyTorch `F.relu` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 gelu | PyTorch `F.gelu` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 silu | PyTorch `F.silu` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 add | PyTorch `torch.add` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| L1 mul | PyTorch `torch.mul` (P3) | ✅ | ✅ `elementwise.py.j2` | ✅ | — |
| **OT1 Reduction** | | | | | |
| L1 softmax | cuDNN/PyTorch (P0/P3) | ✅ | ✅ `softmax.py.j2` | ✅ | rows_per_prog ✅ |
| L1 layernorm | cuDNN/PyTorch (P0/P3) | ✅ | ✅ `layernorm.py.j2` | ✅ | block_size ✅ |
| L1 rmsnorm | FlagGems (P1) | ✅ | ✅ `layernorm.py.j2` | ✅ | block_size ✅ |
| L1 rmsnorm_residual | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 reduce_sum | PyTorch (P3) | ✅ | 🔶 | ❌ | ⬜ |
| L1 reduce_max | PyTorch (P3) | ✅ | 🔶 | ❌ | ⬜ |
| **OT2 Compute-Dense** | | | | | |
| L1 matmul ≥ P0 | cuBLAS (P0) | ✅ | ✅ `matmul.py.j2` | ✅ | tile, split-k, swizzle ✅ |
| L1 batch_matmul ≥ P0 | cuBLAS (P0) | ✅ | ✅ `matmul.py.j2` | ✅ | batch dim ✅ |
| L1 grouped_matmul | CUTLASS (P0) | ✅ | 🔶 | ❌ | ⬜ |
| L1 transpose | PyTorch (P3) | ✅ | 🔶 | ❌ | ⬜ |
| **OT3 Gated Activation** | | | | | |
| L1 swiglu ≥ P1 | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 geglu ≥ P1 | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| **OT4 Attention** | | | | | |
| L1 flash_attention ≥ P1 | FlashAttention (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 grouped_query_attention | FlashAttention (P1) | ✅ | 🔶 | ❌ | ⬜ |
| L1 multi_latent_attention | DeepSeek ref | ✅ | 🔶 | ❌ | ⬜ |
| **L2 Fused** | | | | | |
| L2 matmul+gelu ≥ P1 | FlagGems fusion (P1) | ✅ | ✅ epilogue | ✅ | fusion decision ✅ |
| L2 matmul+relu ≥ P1 | FlagGems fusion (P1) | ✅ | ✅ epilogue | ✅ | fusion decision ✅ |
| L2 swiglu ≥ P1 | Liger (P1) | ✅ | 🔶 | ❌ | ⬜ |
| **L3/BL6 E2E** | | | | | |
| GPT-2 ≤ eager | E2E eager | ✅ | ✅ | ✅ | KernelCache ✅ |
| LLaMA-2 7B ≤ eager | E2E eager | partial | 🔶 | ❌ | ⬜ |
| DeepSeek-V2 ≤ eager | E2E eager | partial | 🔶 | ❌ | ⬜ |

**Summary:** 11/20 operators have working codegen (Triton template). 9 operators
(reduce_sum/max, transpose, rmsnorm_residual, grouped_matmul, swiglu, geglu,
flash_attention, GQA, MLA) have IR + numerical validation but no Triton
template yet — this is the primary Stage 2 codegen gap.

---

## Implementation Status

### Runner Implementations

| Runner | Tier | Status | Ops Supported |
|:-------|:----:|:------:|:------|
| `CuBLASRunner` | P0 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `FlagGemsRunner` | P1 | ✅ | matmul, softmax, layernorm, rmsnorm, gelu, relu, silu |
| `LigerRunner` | P1 | ✅ | rmsnorm, gelu, silu, rope |
| `FlashAttnRunner` | P1 | ⬜ | flash_attention (planned) |
| `TritonTutorialRunner` | P2 | ✅ | matmul, softmax |
| `PyTorchEagerRunner` | P3 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `InductorRunner` | P4 | ✅ | matmul, softmax, layernorm, gelu, relu, silu |
| `LLMDirectRunner` | P5 | ⬜ | (planned: all ops via LLM codegen) |
| `ArkeRunner` | — | ✅ | matmul, softmax |

### Benchmark Components

| Component | Status | Description |
|:----------|:------:|:------------|
| `baselines/` | ✅ | BaselineRunner ABC + 8 runner classes |
| `shapes.py` | ✅ | Shape registry with ST1–ST4 tagging |
| `perf_csv.py` | ✅ | PerfRow + PerfCSVWriter (CSV v2.0, 41 columns) |
| `measure.py` | ✅ | CUDA event timing |
| `bench_l1.py` | ✅ | L1 single operator benchmarks |
| `bench_l2.py` | ✅ | L2 fused operator benchmarks |
| `bench_l3.py` | ✅ | L3 E2E model benchmarks (GPT-2) |
| `gate.py` | ✅ | Gate verification CLI |
| `cli.py` | ✅ | Unified CLI entry point |
| `op_registry.py` | ✅ | Parses benchmark-ops.md → OT_OPS / OP_TIER / ALL_OPS (single source of truth) |
| `report.py` | ✅ | Markdown report generator |
| Hardware info | ✅ | `hardware.json` per run |
| Provenance | ✅ | CSV source column + per-run manifest |
| BL/OT/ST CLI | ✅ | `arke bench --bl/--ot/--st/--layer` |
| Op catalog auto-sync | ✅ | `tests/conftest.py` + `scripts/sync_ops.py` detect md changes on every pytest run |
| `shape_registry.py` | ✅ | Parses benchmark-shapes.md → SHAPE_TABLES / SHAPES_BY_TIER / ALL_SHAPE_TAGS (single source of truth) |
| Shape catalog auto-sync | ✅ | `tests/conftest.py` + `scripts/sync_shapes.py` detect shape changes via SHA-256 tags hash |
| Cross-run diff | ⬜ | `arke bench diff` |
| CI integration | ⬜ | GitHub Actions regression mode |

---

## Dependencies

### Required

```bash
pip install torch triton
```

### Benchmark Baselines

```bash
pip install flag-gems        # FlagGems — 200+ Triton ops (P1)
pip install liger-kernel     # Liger — LLM training kernels (P1)
```

### Optional

```bash
pip install flash-attn --no-build-isolation   # FlashAttention (P1)
pip install triton-kernels                     # HuggingFace community kernels
pip install nvidia-cutlass                     # CUTLASS C++ GEMM baselines
```

### Graceful Degradation

Missing baseline packages are skipped with a warning; results show `N/A`.

---

*Last updated: 2026-04-05*
