# Stage 7 Track 6 Benchmark Report

Generated: 2026-04-26 12:35:18 CST

## Verification Summary

- Focused regression test slice: `345 passed in 3.13s`
- Gate G7 verification: `555 passed, 30 warnings, 0 failed`
- Result-tree contract: ✅ root dashboard artifacts present

## L1 — Single Operator / Shape Evidence

- Operator coverage: `45/45` (`100%`)
- Shape coverage: `124/685` (`18.1%`)
- Overall geomean: `0.8124`
- Status counts: `ok=780`, `skipped=10`, `error=3`
- Correctness counts: `ok=566`, `unsupported=204`, `error=8`, `mismatch=3`, `skipped=10`, `unknown=2`
- Memory pressure rows: `10`
- Performance pass counts: `true=413`, `false=380`

### L1 op scores

| Operator | Score |
|:--|--:|
| flash_attention | 3.3941 |
| cast | 3.2112 |
| grouped_query_attention | 2.5051 |
| rope | 1.5418 |
| rmsnorm | 1.4342 |
| swiglu | 1.1783 |
| relu | 1.1047 |
| softmax | 0.7688 |
| matmul | 0.8208 |
| batch_matmul | 0.5038 |
| grouped_matmul | 0.1524 |
| transpose | 0.2522 |

## L2 — Fused Operator Evidence

- Fusion coverage: `6/6` (`100%`)
- Shape coverage: `6/120` (`5.0%`)
- Overall geomean: `0.8238`
- Status counts: `ok=155`, `error=4`
- Correctness counts: `ok=132`, `unsupported=23`, `error=4`
- Performance pass counts: `true=82`, `false=77`

### L2 op scores

| Fusion | Score |
|:--|--:|
| geglu | 1.0000 |
| linear_ce | 1.0000 |
| matmul_gelu | 0.9500 |
| matmul_relu | 0.6677 |
| qkv_fa | 1.0000 |
| swiglu | 1.0000 |

## Interpretation

- The Stage 7 Lang/IR stack is now verified end-to-end enough for gate closure evidence, with the current contract and docs recorded in standard locations.
- The remaining benchmark gap is **coverage**, not artifact presence: ST4-heavy BL5 rows are still memory-limited on the 6GB RTX 3060 and remain explicitly surfaced in `coverage_gap.json`, `audit_report.json`, and `dashboard.json`.
- The current artifacts preserve the required correctness/performance metadata (`perf_target`, `perf_actual`, `perf_pass`, `perf_gap`, plus memory-policy fields) so future reruns can compare directly without reformatting.

## Artifact references

- `benchmarks/results/phase1/stage7/track6/coverage_gap.json`
- `benchmarks/results/phase1/stage7/track6/audit_report.json`
- `benchmarks/results/phase1/stage7/track6/dashboard.json`
- `benchmarks/results/phase1/stage7/track6/stage7_operator_shape_stats.json`
- `benchmarks/results/phase1/stage7/track6/l1/PERF_ALL.csv`
- `benchmarks/results/phase1/stage7/track6/l2/PERF_ALL.csv`
- `benchmarks/results/phase1/stage7/track6/l1/summary.json`
- `benchmarks/results/phase1/stage7/track6/l2/summary.json`
