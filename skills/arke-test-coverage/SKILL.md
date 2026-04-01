---
name: arke-test-coverage
description: Run Arke benchmark test coverage across CUBE (matmul/batch_matmul) and Vector (elementwise/reduce) operators. Use when asked to run benchmarks, test coverage, performance evaluation, accuracy verification, or compare Arke vs Direct Triton. Supports three tiers — Tier 1 (core, ~12 tasks, default), Tier 2 (extended, ~40 tasks, important features), Tier 3 (full, 126+ tasks, release validation). Triggers on "run benchmarks", "test coverage", "run tier 1/2/3", "evaluate performance", "accuracy test", "Arke vs Direct".
---

# Arke Test Coverage

Run structured benchmark suites for Arke kernel optimization evaluation.

## Quick Start

```bash
cd /home/blueyi/workspace/repos/arke
source ~/.venvs/arke/bin/activate

# Tier 1 (default, ~12 tasks, ~1 hour)
python -m benchmarks.run --phase <phase_name> --method both --trials 1

# Tier 2 (extended, ~40 tasks)
python -m benchmarks.run --phase <phase_name> --method both --trials 1 --tier 2

# Tier 3 (full, all tasks)
python -m benchmarks.run --phase <phase_name> --method both --trials 3 --tier 3
```

## Tier Selection

| Tier | Tasks | When | Time (1 trial) |
|:-----|------:|:-----|:----------------|
| 1 | ~12 | Every run, CI, quick validation | ~1h |
| 2 | ~40 | Weekly, after major changes | ~3h |
| 3 | 126+ | Pre-release, full validation | ~10h |

Default to **Tier 1** unless user specifies otherwise.

## Task Categories

### CUBE Class (Compute-bound)

Matmul-family operators. Performance baseline: cuBLAS.

Read `references/cube-tasks.md` for the full task list with shapes and rationale.

**Tier 1 CUBE tasks** (always included):
- `matmul_1024` — 1024×1024×1024, standard benchmark
- `matmul_2048` — 2048×2048×2048, high compute intensity
- `matmul_rect` — 1024×2048×512, non-square tiling
- `matmul_unaligned` — 997×1009×1013, boundary handling
- `matmul_tall` — 4096×256×1024, extreme M/N ratio

### Vector Class (Memory-bound)

Elementwise and reduce operators. Performance baseline: PyTorch eager.

Read `references/vector-tasks.md` for the full task list.

**Tier 1 Vector tasks** (always included):
- `softmax_4096` — 4096×4096, standard reduce
- `softmax_short` — 4096×64, short reduction dim
- `relu_medium` — 1024×1024, elementwise baseline
- `add_large` — 4096×4096, binary elementwise

### Fusion Combinations

Read `references/fusion-tasks.md` for fusion combinations.

**Tier 1 Fusion tasks** (always included):
- `fused_matmul_relu` — matmul + ReLU epilogue
- `fused_matmul_gelu` — matmul + GELU epilogue
- `fused_matmul_add` — matmul + residual connection

## Execution Workflow

1. Select tier based on context (default: Tier 1)
2. Set phase name (e.g., `phase1.5_baseline`, `phase2_improved`)
3. Run benchmark: `python -m benchmarks.run --phase <phase> --method both --trials <n>`
4. Results auto-archived to `benchmarks/results/<phase>/<timestamp>/`
5. Verify output files exist:
   - `benchmark_results.csv` — flat results table
   - `task_catalog.csv` — task definitions
   - `arke_ir/*.json` — Arke IR source files
   - `triton_kernels/arke/*.py` — Arke-compiled kernels
   - `triton_kernels/direct/*.py` — Direct-written kernels
6. Report Gate G4 pass/fail and per-task breakdown

## Accuracy Standards

| dtype | atol | rtol | Notes |
|:------|:-----|:-----|:------|
| f16 | 0.1 | 0.05 | Default dtype |
| f32 | 1e-5 | 1e-4 | High precision |
| bf16 | 0.2 | 0.1 | Wide mantissa tolerance |

Special: softmax row-sum ≈ 1.0 (atol=0.01), reduce_sum large-N allows atol=0.5 for f16.

## Performance Targets

| Operator | Baseline | Target |
|:---------|:---------|:-------|
| matmul | cuBLAS | ≥ 90% |
| elementwise | PyTorch | ≥ 100% |
| softmax | PyTorch F.softmax | ≥ 90% |
| reduce | PyTorch sum/max | ≥ 90% |

## Comparing Results Across Runs

To compare two archived runs:

```python
from benchmarks.export import export_csv
# Load both CSVs and compare vs_baseline, correct, duration_s columns
```

Key metrics to compare:
- **Correctness rate**: Arke should be ≥ Direct
- **vs_baseline mean**: Higher is better
- **duration_s**: Lower is better (Arke optimization speed)
- **total_tokens**: Lower is better (LLM cost efficiency)

## Adding New Tasks

Edit `benchmarks/tasks.py`. Follow the pattern:

```python
BenchmarkTask(
    name="descriptive_name",
    description="What this tests",
    semantic_ir=_build_matmul("name", M, N, K),  # or _build_softmax, etc.
    tags=["category", "subcategory"],
)
```

After adding tasks, update tier assignments in `references/tier-assignments.md`.
