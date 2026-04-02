# Benchmark ↔ Arke Synergy Design

## Core Idea: Benchmark as Development Driver

Benchmark 不是事后验证工具，而是 **Arke 开发的目标态定义**。
每个 benchmark target 直接映射到一个 Arke 能力要求。

```
Benchmark Target (what)  →  Arke Capability (how)  →  Development Task (code)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L1 matmul ≥80% cuBLAS    →  IR+Strategy+Codegen     →  Template autotune
L1 softmax ≥80% PyTorch  →  Memory-bound template   →  softmax.py.j2
L1 layernorm ≥90% FG     →  New op in OP_CATALOG    →  layernorm.py.j2
L2 matmul+gelu fused     →  Fusion in codegen       →  Epilogue fusion
L3 GPT-2 ≤1.1× eager    →  E2E integration         →  KernelCache + patch
```

## Target-Driven Development Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. Define Target                                       │
│     "matmul [4096,4096,4096] ≥ 90% FlagGems"          │
│                                                         │
│  2. Run Benchmark (Red)                                 │
│     arke bench --op matmul --shape square-4k            │
│     → Result: 72% FlagGems  ← FAIL                     │
│                                                         │
│  3. Diagnose                                            │
│     Why? Arke tile config missing L2 swizzle for 4K     │
│                                                         │
│  4. Fix Arke                                            │
│     Add swizzle config to matmul strategy space         │
│                                                         │
│  5. Run Benchmark (Green)                               │
│     → Result: 94% FlagGems  ← PASS                     │
│                                                         │
│  6. Commit + advance to next target                     │
└─────────────────────────────────────────────────────────┘
```

## Capability Mapping

### What each benchmark target demands from Arke:

| Benchmark Target | Arke IR | Template | Strategy | Integration |
|:-----------------|:--------|:---------|:---------|:------------|
| L1 matmul | `matmul` op | `matmul.py.j2` | tile, split-k, swizzle | — |
| L1 batch_matmul | `batch_matmul` op | `batch_matmul.py.j2` ⬜ | batch dim handling | — |
| L1 softmax | `softmax` op | `softmax.py.j2` | rows_per_prog | — |
| L1 layernorm | `layernorm` op ⬜ | `layernorm.py.j2` ⬜ | block_size | — |
| L1 rmsnorm | `rmsnorm` op ⬜ | `rmsnorm.py.j2` ⬜ | block_size | — |
| L1 gelu | `gelu` op | elementwise fuse | — | — |
| L1 relu | `relu` op | elementwise fuse | — | — |
| L1 rope | `rope` op ⬜ | `rope.py.j2` ⬜ | interleave/rotate | — |
| L1 cross_entropy | `cross_entropy` op ⬜ | `cross_entropy.py.j2` ⬜ | online softmax | — |
| L2 matmul+relu | fusion | epilogue in matmul | fusion decision | — |
| L2 matmul+gelu | fusion | epilogue in matmul | fusion decision | — |
| L2 linear+softmax | multi-op | chained templates | pipeline | — |
| L3 GPT-2 | all above | all above | all above | KernelCache, monkey-patch |
| L3 LLaMA | + rmsnorm, rope, swiglu | + new templates | all above | Model-specific patch |

**⬜ = not yet implemented in Arke**

### Priority Order (what to build next based on benchmark impact)

```
Priority 1 (blocks L1 core)     → layernorm, gelu standalone
Priority 2 (blocks L1 extended) → rmsnorm, rope, cross_entropy  
Priority 3 (blocks L2)          → epilogue fusion, batch_matmul
Priority 4 (blocks L3 LLaMA)    → swiglu, rmsnorm template
```

## Benchmark Execution Modes

### Mode 1: Baseline-Only (no Arke, fast)

Runs **only** the external baselines against each other.
Purpose: Establish ground truth performance for the hardware.

```bash
arke bench --mode baselines --op matmul,softmax,layernorm
```

Output: `baselines.csv` with cuBLAS/FlagGems/Liger/PyTorch numbers.
No LLM calls. Runs in ~5 minutes.

### Mode 2: Arke vs Baselines (standard)

Runs Arke pipeline (IR → LLM → codegen → verify) + all baselines.

```bash
arke bench --mode arke --op matmul --shape square-1k,square-2k
```

Output: `results.csv` with Arke vs every baseline.
Requires LLM API. Takes ~10 min per op × shape.

### Mode 3: Regression (CI)

Runs Arke on a fixed set of shapes with cached strategies (no LLM).
Detects performance regressions from code changes.

```bash
arke bench --mode regression
```

Output: pass/fail against stored baseline numbers.
No LLM calls. Runs in ~2 minutes.

### Mode 4: Full Suite

All ops × all shapes × all baselines × all modes.

```bash
arke bench --mode full --trials 3
```

## Integration Points

### 1. Arke OP_CATALOG ← Benchmark op list

Every op in the benchmark MUST exist in Arke's `OP_CATALOG`.
The benchmark runner validates this at startup:

```python
from arke.ir.ops import OP_CATALOG
for op in benchmark_ops:
    assert op in OP_CATALOG, f"Op '{op}' needed by benchmark but missing from Arke"
```

### 2. Arke Templates ← Benchmark code requirement

Every L1 op needs a Triton template. The benchmark checks:

```python
from arke.backend.triton_template_engine import TritonTemplateEngine
engine = TritonTemplateEngine()
for op in benchmark_ops:
    assert engine.has_template(op), f"No template for '{op}'"
```

### 3. Arke KernelCache ← L3 E2E requirement

L3 benchmarks use `KernelCache` for model patching.
New ops in L1/L2 must be added to KernelCache dispatch.

### 4. CI Gate ← Benchmark regression mode

```yaml
# .github/workflows/bench.yml
- name: Benchmark regression
  run: arke bench --mode regression
  # Fails PR if any op regresses >5%
```

## Development Roadmap (benchmark-driven)

### Sprint 1: Infrastructure + L1 Core (matmul, softmax)
- [x] OPERATOR_SOURCES.md
- [x] BENCHMARK_DESIGN.md  
- [x] This synergy doc
- [ ] `benchmarks/baselines/` — BaselineRunner framework
- [ ] `benchmarks/shapes.py` — Shape matrix config
- [ ] `benchmarks/measure.py` — Unified measurement
- [ ] L1 matmul × 12 shapes × P0/P1/P3 baselines
- [ ] L1 softmax × 5 shapes × P0/P1/P3 baselines

### Sprint 2: L1 Extended (layernorm, elementwise)
- [ ] Add `layernorm` to OP_CATALOG + template
- [ ] Add `rmsnorm` to OP_CATALOG + template
- [ ] L1 layernorm × 3 shapes × P0/P1/P3
- [ ] L1 gelu/relu/silu × 3 shapes × P3
- [ ] Standalone elementwise template

### Sprint 3: L2 Fused + L3 E2E
- [ ] L2 matmul+gelu fusion benchmark
- [ ] L2 vs torch.compile auto-fusion
- [ ] L3 GPT-2 refined (multi seq_len)
- [ ] Scoring system + report generation

### Sprint 4: CI + Advanced
- [ ] Regression mode for CI
- [ ] Cross-run comparison
- [ ] rope, cross_entropy ops
- [ ] LLaMA-7B E2E target
