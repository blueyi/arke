# Stage 7 Track 6 Benchmark Report

Generated: 2026-05-09 HKT

## Verification Summary

- Focused G7 gate tests: `pytest -q tests/test_gate_g7.py` → `8 passed in 0.06s`
- Focused L2 fusion/audit tests: `pytest -q tests/test_stage7_l2_fusion_surface.py tests/phase1/stage7/test_coverage_ledger.py tests/phase1/stage7/test_audit_report.py tests/test_gate_g7.py` → `17 passed in 2.03s`
- Full G7 verification: `python -m benchmarks.gate G7 --tier 2` → ❌ `11/14` criteria passed
- Non-regression slices inside G7 remain green: `556 passed, 30 warnings in 9.43s`
- Result-tree contract: ✅ root dashboard artifacts and per-layer manifests are present
- BL5 evidence contract: ❌ `G7.8b`, `G7.8c`, and `G7.8d` fail against canonical Track 6 artifacts
- L2 audit unsupported-surface cases: ✅ cleared after explicit gated activation fusion strategy evidence for `swiglu` and `geglu`

## Failed G7 Evidence Criteria

### G7.8b — Coverage evidence

Current failure detail:

```text
l1: full-shape coverage 22/45; partial=softmax, layernorm, rmsnorm, rmsnorm_residual, topk, cumsum, matmul, concat, ... +15 more; l1: shape coverage 478/685 (0.6978); l1: missing_full_shape_evidence=23 (softmax, layernorm, rmsnorm, rmsnorm_residual, topk, cumsum, matmul, concat, ... +15 more)
```

### G7.8c — Correctness evidence

Current failure detail:

```text
correctness failures=481 checked=1945 memory_excluded=15; first=l1:add:micro-tiny correctness=unsupported, l1:add:micro-small correctness=unsupported, l1:add:gpt2-hidden correctness=unsupported, l1:add:gpt2-ffn correctness=unsupported, l1:add:square-1k correctness=unsupported, l1:add:llama-ffn correctness=unsupported, l1:add:llama-hidden correctness=unsupported, l1:add:llama-long correctness=unsupported, ... +473 more
```

### G7.8d — Performance evidence

Current failure detail:

```text
L1 weighted_score=0.6203; ot0_1=778/1349 (0.577); ot2=125/232 (0.539); ot3=60/100 (0.600); ot4=7/9 (0.778); L2 fusions=6; memory_excluded=15; malformed/non-ok perf rows=8; first=l1:gelu:extreme-flat status=error, l1:rope:gpt2-sm-128 status=error, l1:rope:gpt2-sm-512 status=error, l1:softmax:wide-vocab-llama3 status=error, l2:geglu:non-align-1 status=error, l2:geglu:non-align-2 status=error, l2:swiglu:non-align-1 status=error, l2:swiglu:non-align-2 status=error; L1 weighted performance score 0.6203 < 0.9500; ot0_1=778/1349 (0.577); ot2=125/232 (0.539); ot3=60/100 (0.600); ot4=7/9 (0.778); L2 fusion performance incomplete: matmul_gelu=39/102, matmul_relu=48/102
```

## L1 — Single Operator / Shape Evidence

- Operator coverage: `45/45` (`100.0%`)
- Shape coverage: `478/685` (`69.78%`)
- Fully covered operators: `22/45`
- Partial/missing full-shape operators: `23`
- Status counts: `error=4, ok=1690, skipped=10`
- Correctness counts: `error=31, mismatch=10, ok=1237, skipped=10, unknown=2, unsupported=414`
- Performance pass counts: `false=734, true=970`
- Memory-policy rows: `dense_matmul=122`, `linear_ce=12`, `attention=8`

## L2 — Fused Operator Evidence

- Fusion coverage: `6/6` (`100.0%`)
- Shape coverage: `120/120` (`100.00%`)
- Fully covered fusions: `6/6`
- Unsupported surface cases in audit: none
- Explicit gated activation fusion evidence: `swiglu` → `fuse(ops=["silu", "mul"], fusion_type="epilogue")`; `geglu` → `fuse(ops=["gelu", "mul"], fusion_type="epilogue")`
- Status counts: `error=4, ok=247, skipped=5`
- Correctness counts: `error=4, ok=227, skipped=5, unsupported=20`
- Performance pass counts: `false=126, true=130`
- Memory-policy rows: `dense_matmul=147`, `linear_ce=12`, `attention=16`

## Interpretation

- Stage 7 should not be marked complete yet. The stricter G7 runner now correctly distinguishes green implementation/test slices from incomplete BL5 benchmark evidence.
- The L2 strategy-surface audit gap for `swiglu`/`geglu` is closed: both compact gated activation examples now carry explicit fusion decisions and the audit report has no unsupported L2 surface cases.
- The remaining Stage 7 blocker is substantive BL5 closure: L1 full-shape coverage, correctness support for all non-memory-excluded rows, L1 weighted performance ≥ `0.9500`, and complete L2 fusion performance.
- L2 shape coverage is complete, but L2 correctness/performance evidence still has explicit non-align `swiglu` / `geglu` errors and incomplete `matmul_relu` / `matmul_gelu` performance.
- Memory-policy exclusions are counted only when rows carry explicit memory preflight evidence.

## Artifact references

- `benchmarks/results/phase1/stage7/track6/coverage_gap.json`
- `benchmarks/results/phase1/stage7/track6/audit_report.json`
- `benchmarks/results/phase1/stage7/track6/dashboard.json`
- `benchmarks/results/phase1/stage7/track6/stage7_operator_shape_stats.json`
- `benchmarks/results/phase1/stage7/track6/l1/PERF_ALL.csv`
- `benchmarks/results/phase1/stage7/track6/l2/PERF_ALL.csv`
- `benchmarks/results/phase1/stage7/track6/l1/summary.json`
- `benchmarks/results/phase1/stage7/track6/l2/summary.json`
