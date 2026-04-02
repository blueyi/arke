# Arke Benchmark System — End-to-End Design

## Overview

A unified benchmark framework that measures Arke-generated kernels against
multiple baseline tiers, covering single-operator performance, fused-operator
quality, and whole-model end-to-end inference.

---

## 1. Benchmark Layers

```
┌──────────────────────────────────────────────────────┐
│  Layer 3: E2E Model Inference                        │
│  GPT-2 / LLaMA-7B / Custom model forward pass       │
│  Metric: wall-clock latency, throughput (tok/s)      │
├──────────────────────────────────────────────────────┤
│  Layer 2: Fused Operators                            │
│  matmul+relu, matmul+gelu, linear+softmax, QKV+attn │
│  Metric: TFLOPS, vs best fused baseline              │
├──────────────────────────────────────────────────────┤
│  Layer 1: Single Operators                           │
│  matmul, softmax, layernorm, gelu, rmsnorm, rope ... │
│  Metric: μs latency, % of vendor-optimized baseline  │
└──────────────────────────────────────────────────────┘
```

Each layer answers a different question:
- **L1:** Can Arke generate a kernel as fast as a hand-tuned one?
- **L2:** Can Arke fuse operators as well as expert-written fused kernels?
- **L3:** Does it actually make a real model faster?

---

## 2. Baseline Priority (per operator)

For each operator, baselines are ranked by **expected performance** (highest first).
The benchmark reports Arke's ratio against **every available baseline**.

### Priority Tiers

```
P0  Vendor-optimized    — cuBLAS, cuDNN, CUTLASS (gold standard)
P1  Expert Triton       — FlagGems, Liger-Kernel, FlashAttention (community best)
P2  Reference Triton    — Triton official tutorials (well-known, reproducible)
P3  PyTorch eager       — torch.nn.functional / torch ops (user default)
P4  Inductor-generated  — torch.compile output (auto-optimized)
P5  LLM-direct          — LLM writes Triton directly (Arke's direct competitor)
```

### Operator → Baseline Matrix

| Operator | P0 Vendor | P1 Expert Triton | P2 Ref Triton | P3 PyTorch | P4 Inductor | P5 LLM-direct |
|:---------|:----------|:-----------------|:--------------|:-----------|:------------|:---------------|
| matmul | cuBLAS (`torch.matmul`) | FlagGems `mm` | Tutorial 03 | `torch.mm` | `torch.compile` | ✓ |
| batch_matmul | cuBLAS (`torch.bmm`) | FlagGems `bmm` | — | `torch.bmm` | `torch.compile` | ✓ |
| softmax | cuDNN (via PyTorch) | FlagGems `softmax` | Tutorial 02 | `F.softmax` | `torch.compile` | ✓ |
| layernorm | cuDNN (via PyTorch) | FlagGems `layernorm` | Tutorial 05 | `F.layer_norm` | `torch.compile` | ✓ |
| rmsnorm | — | FlagGems `rmsnorm`, Liger | — | manual impl | `torch.compile` | ✓ |
| gelu | — | FlagGems `gelu` | — | `F.gelu` | `torch.compile` | ✓ |
| relu | — | FlagGems `relu` | — | `F.relu` | `torch.compile` | ✓ |
| silu | — | FlagGems `silu` | — | `F.silu` | `torch.compile` | ✓ |
| rope | — | Liger `rope` | — | manual impl | — | ✓ |
| cross_entropy | cuDNN (via PyTorch) | FlagGems, Liger (fused) | — | `F.cross_entropy` | `torch.compile` | ✓ |
| fused_attention | cuDNN SDPA | FlashAttention (Triton) | Tutorial 06 | SDPA | `torch.compile` | ✓ |
| dropout | — | FlagGems `dropout` | Tutorial 04 | `F.dropout` | `torch.compile` | ✓ |
| reduce_sum | — | FlagGems `sum` | — | `torch.sum` | `torch.compile` | ✓ |
| reduce_max | — | FlagGems `max` | — | `torch.max` | `torch.compile` | ✓ |
| embedding | — | FlagGems `embedding` | — | `nn.Embedding` | `torch.compile` | ✓ |

### Fused Operators (L2)

| Fused Op | P0 Vendor | P1 Expert | P3 PyTorch | P5 LLM-direct |
|:---------|:----------|:----------|:-----------|:---------------|
| matmul + relu | — | FlagGems (via ATen fusion) | `torch.compile` | ✓ |
| matmul + gelu | — | FlagGems | `torch.compile` | ✓ |
| linear + cross_entropy | — | Liger `fused_linear_ce` | separate ops | ✓ |
| QKV + attention | cuDNN SDPA | FlashAttention | `F.scaled_dot_product_attention` | ✓ |
| swiglu | — | Liger `swiglu` | manual impl | ✓ |
| geglu | — | Liger `geglu` | manual impl | ✓ |

---

## 3. Shape Matrix

Standard shapes covering small/medium/large and square/rectangular.
Benchmark every operator at every applicable shape.

### Matmul Shapes

| Tag | M | N | K | Source | Notes |
|:----|----:|----:|----:|:-------|:------|
| `tiny` | 128 | 128 | 128 | micro | Launch overhead dominated |
| `small` | 128 | 768 | 768 | GPT-2 c_proj | Typical LLM hidden dim |
| `medium` | 128 | 2304 | 768 | GPT-2 c_attn | QKV projection |
| `square-1k` | 1024 | 1024 | 1024 | classic | Standard GEMM benchmark |
| `square-2k` | 2048 | 2048 | 2048 | classic | Compute-bound |
| `square-4k` | 4096 | 4096 | 4096 | classic | Large GEMM |
| `rect-wide` | 1024 | 4096 | 1024 | LLM FFN | Wide output |
| `rect-tall` | 4096 | 1024 | 1024 | LLM FFN | Tall input |
| `lm-head` | 128 | 50257 | 768 | GPT-2 lm_head | Vocabulary projection |
| `llama-q` | 4096 | 4096 | 4096 | LLaMA-7B | Attention Q/K/V |
| `llama-ffn` | 4096 | 11008 | 4096 | LLaMA-7B | FFN up-projection |
| `seq512` | 512 | 2304 | 768 | GPT-2 seq=512 | Triton sweet spot |

### Softmax Shapes

| Tag | M | N | Notes |
|:----|----:|----:|:------|
| `attn-small` | 12 | 128 | GPT-2 12-head, seq=128 |
| `attn-med` | 12 | 512 | GPT-2 12-head, seq=512 |
| `attn-large` | 32 | 2048 | LLaMA 32-head, seq=2048 |
| `square-4k` | 4096 | 4096 | Classic stress test |
| `wide` | 1 | 50257 | Vocabulary softmax |

### LayerNorm / RMSNorm Shapes

| Tag | Batch | Hidden | Notes |
|:----|------:|-------:|:------|
| `gpt2` | 128 | 768 | GPT-2 |
| `llama` | 128 | 4096 | LLaMA-7B |
| `large` | 2048 | 4096 | Long sequence |

### Elementwise Shapes (relu, gelu, silu)

| Tag | Size | Notes |
|:----|-----:|:------|
| `small` | 128 × 768 | GPT-2 hidden |
| `medium` | 128 × 3072 | GPT-2 FFN |
| `large` | 4096 × 4096 | Stress test |

---

## 4. Measurement Protocol

### Single Operator (L1)

```python
# 1. Warmup: 200 iterations (triggers autotune, JIT)
for _ in range(200):
    kernel(inputs)
torch.cuda.synchronize()

# 2. Measure: CUDA events, 500 iterations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(500):
    kernel(inputs)
end.record()
torch.cuda.synchronize()
latency_us = start.elapsed_time(end) / 500 * 1000

# 3. Alternative: triton.testing.do_bench (FlagGems standard)
from triton.testing import do_bench
latency_ms = do_bench(lambda: kernel(inputs), warmup=200, rep=500)
```

### Fused Operator (L2)

Same protocol as L1, but input is the unfused sequence of ops vs the fused kernel.

### E2E Model (L3)

```python
# 1. Load model, apply patches
# 2. Warmup: 50 forward passes
# 3. Measure: 200 forward passes with CUDA events
# 4. Report: mean latency (ms), throughput (samples/s or tok/s)
# 5. Correctness: top-1 logit match + max absolute diff
```

### Correctness

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
| `tflops` | TFLOPS | Achieved throughput (for compute-bound ops) |
| `gbps` | GB/s | Achieved bandwidth (for memory-bound ops) |
| `vs_p0` | ratio | Arke latency / P0 vendor baseline latency |
| `vs_p1` | ratio | Arke latency / P1 expert Triton latency |
| `vs_p3` | ratio | Arke latency / P3 PyTorch eager latency |
| `vs_p5` | ratio | Arke latency / P5 LLM-direct latency |
| `correct` | bool | Passes numerical tolerance |
| `max_diff` | float | Maximum absolute difference from reference |
| `tokens_in` | int | LLM input tokens consumed (Arke/LLM-direct) |
| `tokens_out` | int | LLM output tokens consumed |
| `compile_time_s` | s | Time to generate + compile kernel |
| `arke_turns` | int | Number of LLM agent turns (Arke only) |

---

## 5. Output Structure

```
benchmarks/results/{run_id}/
├── config.json              # Run configuration (shapes, baselines, HW info)
├── hardware.json            # GPU info, driver version, CUDA version
│
├── L1_single_ops/
│   ├── matmul/
│   │   ├── results.csv      # All shape × baseline × trial results
│   │   ├── arke/            # Arke-generated kernel code
│   │   │   ├── square-1k.py
│   │   │   └── ...
│   │   └── llm_direct/      # LLM-direct-written kernel code
│   │       ├── square-1k.py
│   │       └── ...
│   ├── softmax/
│   │   ├── results.csv
│   │   └── ...
│   └── .../
│
├── L2_fused_ops/
│   ├── matmul_relu/
│   │   ├── results.csv
│   │   └── ...
│   └── .../
│
├── L3_e2e/
│   ├── gpt2/
│   │   ├── results.csv
│   │   ├── config.json      # Model, seq_len, patches applied
│   │   └── ...
│   └── .../
│
├── summary.json             # Aggregated results across all layers
├── summary.csv              # Flat CSV for easy analysis
└── report.md                # Human-readable report
```

---

## 6. CLI Interface

```bash
# Run everything
arke bench --all

# Run specific layer
arke bench --layer L1
arke bench --layer L2
arke bench --layer L3

# Run specific operator
arke bench --op matmul --shapes square-1k,square-2k
arke bench --op softmax --shapes attn-small,attn-med

# Run specific baselines only
arke bench --op matmul --baselines cublas,flaggems,arke

# Run E2E
arke bench --model gpt2 --seq-len 128,256,512

# Compare two runs
arke bench diff {run_id_1} {run_id_2}

# Generate report
arke bench report {run_id}
```

---

## 7. Baseline Runner Architecture

```python
class BaselineRunner(ABC):
    """Base class for all baseline implementations."""
    
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def priority(self) -> int: ...  # P0=0, P1=1, ..., P5=5
    
    @abstractmethod
    def supports(self, op: str, shape: tuple) -> bool: ...
    
    @abstractmethod
    def run(self, op: str, inputs: dict[str, Tensor]) -> Tensor: ...
    
    @abstractmethod
    def bench(self, op: str, inputs: dict[str, Tensor], 
              warmup: int, reps: int) -> float: ...  # returns μs


class CuBLASRunner(BaselineRunner):
    """P0: cuBLAS via torch.matmul/torch.mm"""
    
class CuDNNRunner(BaselineRunner):
    """P0: cuDNN via PyTorch (softmax, layernorm, SDPA)"""

class FlagGemsRunner(BaselineRunner):
    """P1: FlagGems Triton operators"""
    
class LigerRunner(BaselineRunner):
    """P1: Liger-Kernel operators"""
    
class FlashAttnRunner(BaselineRunner):
    """P1: FlashAttention Triton"""

class TritonTutorialRunner(BaselineRunner):
    """P2: Triton official tutorial kernels"""

class PyTorchEagerRunner(BaselineRunner):
    """P3: PyTorch eager mode ops"""

class InductorRunner(BaselineRunner):
    """P4: torch.compile generated kernels"""

class LLMDirectRunner(BaselineRunner):
    """P5: LLM writes Triton directly"""

class ArkeRunner(BaselineRunner):
    """Arke pipeline: IR → strategy → codegen → verify"""
```

---

## 8. Scoring System

### Single Operator Score (per shape)

```
op_score = Σ(weight_i × ratio_to_baseline_i) / Σ(weight_i)

where:
  ratio_to_baseline = baseline_latency / arke_latency  (>1 = Arke faster)
  weight: P0=3, P1=2, P3=1
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

### Quality Gates

| Gate | Criterion | Threshold |
|:-----|:----------|:----------|
| G-correct | All ops pass numerical tolerance | 100% |
| G-perf-P0 | vs vendor-optimized (cuBLAS/cuDNN) | ≥ 80% |
| G-perf-P1 | vs expert Triton (FlagGems/Liger) | ≥ 90% |
| G-perf-P5 | vs LLM-direct Triton | ≥ 100% (must beat) |
| G-e2e | E2E model latency | ≤ 1.1× eager |
| G-token | Token efficiency vs LLM-direct | ≤ 50% tokens |
| G-variance | Consistency across trials | σ/μ ≤ 5% |

---

## 9. Dependency Installation

```bash
# Required
pip install flag-gems          # FlagGems (200+ Triton ops)
pip install liger-kernel       # Liger-Kernel (LLM training ops)

# Optional (for extended baselines)
pip install flash-attn --no-build-isolation  # FlashAttention
pip install triton-kernels     # HF community kernels

# Already available
# torch, triton (core deps)
```

### Graceful Degradation

If a baseline package is not installed, the runner skips that baseline
and logs a warning. Results table shows `N/A` for missing baselines.

```python
try:
    import flag_gems
    FLAGGEMS_AVAILABLE = True
except ImportError:
    FLAGGEMS_AVAILABLE = False
    logger.warning("flag-gems not installed, P1 FlagGems baseline disabled")
```

---

## 10. Implementation Phases

### Phase A: Baseline Infrastructure (current)
- [ ] Implement `BaselineRunner` ABC + all runner classes
- [ ] Shape matrix as structured config (`benchmarks/shapes.py`)
- [ ] Unified measurement function (`benchmarks/measure.py`)
- [ ] Hardware info collection (`benchmarks/hardware.py`)

### Phase B: L1 Single Operators
- [ ] matmul across full shape matrix × all baselines
- [ ] softmax across full shape matrix × all baselines
- [ ] layernorm, gelu, relu, silu (elementwise)
- [ ] Results CSV + per-operator summary

### Phase C: L2 Fused Operators
- [ ] matmul+relu, matmul+gelu fusions
- [ ] Comparison vs torch.compile auto-fusion
- [ ] Comparison vs FlagGems ATen-fused equivalents

### Phase D: L3 E2E Models
- [ ] GPT-2 Small (current, refine)
- [ ] GPT-2 Medium (scale test)
- [ ] (Future) LLaMA-7B (Stage 2 target)

### Phase E: Reporting & CI
- [ ] CLI integration (`arke bench`)
- [ ] Auto-generated report.md with charts
- [ ] Cross-run comparison (`arke bench diff`)
- [ ] CI integration (run on every PR)

---

*Last updated: 2026-04-02*
