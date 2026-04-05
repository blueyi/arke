# Arke Benchmark — Protocol, Scoring & Implementation

Measurement protocol, scoring system, CLI interface, output structure, and implementation status.

→ Parent: [`BENCHMARK.md`](../BENCHMARK.md)

---

## Measurement Protocol

### Single Operator (L1)

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

### Fused Operator (L2)

Same protocol as L1, but input is the unfused sequence of ops vs the fused kernel.

### E2E Model (L3)

```python
# 1. Load model, apply patches (KernelCache)
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

| Metric | Unit | Description |
|:-------|:-----|:------------|
| `latency_us` | μs | Mean kernel latency |
| `latency_min_us` | μs | Minimum kernel latency |
| `tflops` | TFLOPS | Achieved throughput (compute-bound ops) |
| `gbps` | GB/s | Achieved bandwidth (memory-bound ops) |
| `vs_p0` | ratio | Arke / P0 vendor baseline latency |
| `vs_p1` | ratio | Arke / P1 expert Triton latency |
| `vs_p3` | ratio | Arke / P3 PyTorch eager latency |
| `vs_p5` | ratio | Arke / P5 LLM-direct latency |
| `correct` | bool | Passes numerical tolerance |
| `max_diff` | float | Maximum absolute difference |
| `tokens_in` | int | LLM input tokens consumed (Arke/LLM-direct) |
| `tokens_out` | int | LLM output tokens consumed |
| `compile_time_s` | s | Time to generate + compile kernel |
| `arke_turns` | int | Number of LLM agent turns (Arke only) |

---

## Scoring System

### Single Operator Score (per shape)

```
op_score = Σ(weight_i × ratio_to_baseline_i) / Σ(weight_i)

where:
  ratio_to_baseline = baseline_latency / arke_latency  (>1 = Arke faster)
  weights: P0=3, P1=2, P3=1
```

### Layer Score

```
layer_score = geomean(op_scores)   # geometric mean across all ops
```

### Overall Arke Score

```
arke_score = 0.3 × L1_score + 0.3 × L2_score + 0.4 × L3_score
```

L3 weighted highest because real-world impact matters most.

### Report Indicators

| Indicator | Meaning |
|:---------:|:--------|
| 🟢 | ≥ 90% of baseline |
| 🟡 | ≥ 80% of baseline |
| 🔴 | < 80% of baseline |

### Exclusion Rules

| Scenario | Handling | Reason |
|:---------|:---------|:-------|
| M ≤ 32 (matmul) | Accuracy must pass; perf excluded | Triton ~55μs launch floor |
| N ≤ 32 (softmax) | Accuracy must pass; perf excluded | Same |
| M×N ≤ 1024 (elementwise) | Accuracy must pass; perf excluded | Kernel-launch dominated |
| Batch ≤ 1 (layernorm) | Accuracy must pass; perf excluded | Single-sample overhead |
| OOM shapes | Skip, record "OOM" | 6GB VRAM limit |
| Triton compile timeout (>60s) | Record "TIMEOUT", accuracy fail | Template may need fix |

---

## Output Structure & Provenance Tracking

### Directory Layout

```
benchmarks/results/{run_id}/
├── config.json              # Run configuration
├── hardware.json            # GPU info, driver, CUDA, PyTorch/Triton versions
├── L1_single_ops/
│   ├── {op}/
│   │   ├── results.csv      # shape × baseline × trial results
│   │   ├── arke/            # Arke-generated kernel code per shape
│   │   └── llm_direct/      # LLM-direct kernel code per shape
├── L2_fused_ops/
│   └── {fused_op}/
│       └── results.csv
├── L3_e2e/
│   └── {model}/
│       ├── results.csv
│       └── config.json      # Model, seq_len, patches
├── summary.json             # Aggregated results
├── summary.csv              # Flat CSV
└── report.md                # Human-readable report
```

### Provenance Tracking

Every result carries full source attribution:
- **CSV `source` column** — package name, version, URL, license per row
- **`sources.json`** — per-run manifest of all baselines used
- **`hardware.json`** — GPU name, CUDA version, driver, framework versions

---

## CLI Interface

### Current (python -m benchmarks)

```bash
python -m benchmarks --all                          # L1 + L2 + L3
python -m benchmarks --layer L1                     # Single operator
python -m benchmarks --layer L2                     # Fused operator
python -m benchmarks --layer L3                     # E2E model
python -m benchmarks --op matmul                    # Specific op
python -m benchmarks --op matmul --tier 2           # With shape tier
python -m benchmarks --report                       # Generate report
```

### Planned (arke bench)

```bash
arke bench --all
arke bench --layer L1 --op matmul --shapes square-1k,square-2k
arke bench --layer L1 --op matmul --tier 3          # All ST3 shapes
arke bench --layer L3 --model gpt2 --seq-len 128,512
arke bench --baselines cublas,flaggems,arke
arke bench diff {run_id_1} {run_id_2}
arke bench report {run_id}
```

---

## Benchmark-Driven Development

The benchmark is not post-hoc validation — it is the **target state definition** for Arke development.

### Capability Mapping

| Benchmark Target | Arke IR | Template | Strategy |
|:-----------------|:--------|:---------|:---------|
| L1 matmul ≥80% cuBLAS | `matmul` op ✅ | `matmul.py.j2` ✅ | tile, split-k, swizzle ✅ |
| L1 softmax ≥80% PyTorch | `softmax` op ✅ | `softmax.py.j2` ✅ | rows_per_prog ✅ |
| L1 layernorm ≥90% FG | `layernorm` op ✅ | `layernorm.py.j2` ⬜ | block_size ⬜ |
| L1 rmsnorm ≥90% FG | `rmsnorm` op ✅ | `rmsnorm.py.j2` ⬜ | block_size ⬜ |
| L1 flash_attention | `flash_attention` op ✅ | `flash_attn.py.j2` ⬜ | tiling ⬜ |
| L1 swiglu | `swiglu` op ✅ | `swiglu.py.j2` ⬜ | — ⬜ |
| L2 matmul+gelu | fusion ✅ | epilogue ✅ | fusion decision ✅ |
| L3 GPT-2 ≤1.15× eager | all above ✅ | all above ✅ | KernelCache ✅ |
| L3 LLaMA ≤1.15× eager | + rmsnorm, swiglu, flash_attn | + new templates | Model-specific patch |

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
| `shapes.py` | ✅ | Shape registry with Tier tagging |
| `measure.py` | ✅ | CUDA event timing |
| `bench_l1.py` | ✅ | L1 single operator benchmarks |
| `bench_l2.py` | ✅ | L2 fused operator benchmarks |
| `bench_l3.py` | ✅ | L3 E2E model benchmarks (GPT-2) |
| `gate.py` | ✅ | Gate verification CLI |
| `cli.py` | ✅ | Unified CLI entry point |
| `report.py` | ✅ | Markdown report generator |
| Hardware info | ✅ | `hardware.json` per run |
| Provenance | ✅ | CSV source column + `sources.json` |
| `arke bench` CLI | ⬜ | Planned unified CLI |
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
