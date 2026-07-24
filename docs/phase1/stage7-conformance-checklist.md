# Stage 7 Conformance Checklist

> **Purpose:** Explicit closure artifact for Stage 7 Track 1.4 / Track 7.3.
> **Date:** 2026-04-26
> **Scope:** Lang v0.1.0, IR v0.1.0, BL5 coverage, and Gate G7 evidence.

## Summary

- ✅ Canonical `.ak` surface aligned to Lang v0.1.0
- ✅ SemanticIR / StrategyIR aligned to IR v0.1.0 layering
- ✅ `where` + symbolic dimensions implemented and tested
- ✅ Conditional strategy and backend-agnostic StrategyIR implemented
- ✅ MLIR skeleton path present for the Stage 7 bridge
- ✅ BL5 examples and coverage ledger are machine-readable
- ✅ Track 6 artifacts are present under `benchmarks/results/phase1/stage7/track6/`
- ✅ Latest Gate G7 refresh passed: `555 passed, 30 warnings, 0 failed`

## Checklist

| Area | Status | Evidence |
|:--|:--:|:--|
| Lang v0.1.0 canonical kernel/strategy surface | ✅ | `docs/spec/arke-lang-spec.md`, `tests/test_stage7_roundtrip.py` |
| `where` clause + symbolic dimensions | ✅ | `docs/spec/symbolic-dimension-spec.md`, `tests/test_symbolic_shape.py` |
| Conditional / shape-aware StrategyIR | ✅ | `arke/ir/strategy.py`, `tests/test_stage7_memory_aware_strategy.py` |
| Backend-agnostic StrategyIR core | ✅ | `tests/test_backend_agnostic.py`, `tests/test_backend_agnostic_script.py` |
| ScheduleIR / InstructionIR bridge | ✅ | `arke/ir/schedule.py`, `arke/ir/instruction.py`, `tests/test_stage7_lowering.py` |
| MLIR skeleton emission | ✅ | `arke/backends/mlir/emitter.py`, `tests/test_stage7_lowering.py` |
| BL5 examples for all 46 L1 ops | ✅ | `examples/operators/`, `benchmarks/stage7_coverage_ledger.py` |
| BL5 L2 fusion surface examples | ✅ | `examples/operators/l2/`, `tests/test_stage7_l2_fusion_surface.py` |
| Coverage ledger / audit / dashboard artifacts | ✅ | `benchmarks/results/phase1/stage7/track6/{coverage_ledger.json,audit_report.json,coverage_gap.json,dashboard.json,stage7_operator_shape_stats.json}` |
| Stable perf artifacts with correctness + perf fields | ✅ | `benchmarks/results/phase1/stage7/track6/{l1,l2}/PERF_ALL.csv`, `summary.json` |
| Gate verification recorded in standard location | ✅ | `benchmarks/results/phase1/stage7/track6/report.md` |

## Verification commands

```bash
source ~/.venvs/arke/bin/activate
pytest -q tests/test_stage7_roundtrip.py tests/test_symbolic_shape.py tests/test_backend_agnostic.py tests/test_stage7_lowering.py tests/test_stage7_report.py tests/test_stage7_coverage_gap.py tests/phase1/stage7/test_stage7_dashboard.py tests/phase1/stage7/test_track6_contract.py
python -m benchmarks gate G7 --tier 2
```

## Remaining benchmark gap

- **[blocked]** Full BL5 shape coverage is still memory-limited on the 6GB RTX 3060 for the most expensive ST4 / OT4 rows.
- The gap is not hidden: it is surfaced in `coverage_gap.json`, `audit_report.json`, `dashboard.json`, and the per-layer `PERF_ALL.csv` manifests.
- Memory policy metadata is preserved in benchmark artifacts (`memory_bytes_required`, `memory_bytes_budget`, `memory_ratio`, `memory_policy`) so the remaining work is measurable and retryable.

## Artifact index

- `docs/phase1/stage7-plan.md`
- `benchmarks/results/phase1/stage7/track6/report.md`
- `benchmarks/results/phase1/stage7/track6/dashboard.json`
- `benchmarks/results/phase1/stage7/track6/coverage_gap.json`
- `benchmarks/results/phase1/stage7/track6/audit_report.json`
- `benchmarks/results/phase1/stage7/track6/coverage_ledger.json`

