# Stage 7 Track 6 Benchmark Report

Generated: 2026-04-28 01:02:57 HKT

## Verification Summary

- Focused benchmark/advice/dashboard slice: `11 passed in 1.76s`
- Gate/dashboard/audit/L2 probe slice: `12 passed in 1.76s`
- Combined focused verification slice: `19 passed in 1.78s`
- Result-tree contract: ✅ root dashboard artifacts refreshed

## L1 — Single Operator / Shape Evidence

- Operator coverage: `45/45` (`100.0%`)
- Shape coverage: `215/685` (`31.39%`)
- Fully covered operators: `3/45`
- Overall geomean: `0.8124`
- Status counts: `error=3, ok=780, skipped=10`
- Correctness counts: `error=8, mismatch=3, ok=566, skipped=10, unknown=2, unsupported=204`
- Memory pressure rows: `10`
- Performance pass counts: `false=380, true=413`

### L1 op scores

| Operator | Score |
|:--|--:|
| add | 1.1047 |
| argmax | 0.8853 |
| batch_matmul | 0.5038 |
| cast | 3.2112 |
| concat | 0.9792 |
| copy_ | 1.1213 |
| cross_attention | 0.9448 |
| cross_entropy | 0.8146 |
| cumsum | 1.3554 |
| dequantize_per_channel | 1.0830 |
| embedding | 1.2642 |
| exp | 0.9877 |
| flash_attention | 3.3941 |
| fused_linear_cross_entropy | 0.9109 |
| gather | 1.1532 |
| geglu | 0.7534 |
| gelu | 0.4978 |
| grouped_matmul | 0.1524 |
| grouped_query_attention | 2.5051 |
| layernorm | 0.4999 |
| matmul | 0.8208 |
| mul | 0.9920 |
| neg | 0.9804 |
| permute | 1.0000 |
| quantize_per_token | 0.9935 |
| reduce_max | 1.0463 |
| reduce_mean | 0.8942 |
| reduce_sum | 0.7110 |
| relu | 0.6557 |
| rmsnorm | 1.4342 |
| rmsnorm_residual | 0.8811 |
| rope | 1.5418 |
| rsqrt | 0.9244 |
| scatter | 1.0000 |
| sigmoid | 0.9682 |
| silu | 0.7894 |
| softmax | 0.7688 |
| split | 1.0236 |
| swiglu | 1.1783 |
| tanh | 0.9912 |
| topk | 1.3734 |
| transpose | 0.2522 |
| where_ | 0.5339 |

## L2 — Fused Operator Evidence

- Fusion coverage: `6/6` (`100.0%`)
- Shape coverage: `120/120` (`100.00%`)
- Fully covered fusions: `6/6`
- Overall geomean: `0.6821`
- Status counts: `error=4, ok=247, skipped=5`
- Correctness counts: `error=4, ok=227, skipped=5, unsupported=20`
- Memory pressure rows: `5`
- Performance pass counts: `false=126, true=130`

### L2 op scores

| Operator | Score |
|:--|--:|
| geglu | 1.0000 |
| linear_ce | 1.0000 |
| matmul_gelu | 0.5993 |
| matmul_relu | 0.6606 |
| qkv_fa | 1.0000 |
| swiglu | 1.0000 |

## Interpretation

- L2 BL5 shape coverage is now complete: all `120/120` required fused-op shape tags have evidence rows in `l2/PERF_ALL.csv`.
- Several L2 rows remain explicit memory-policy skips/errors on the 6GB RTX 3060; those are preserved as benchmark evidence and surfaced in `summary.json`, `stage7_operator_shape_stats.json`, and `dashboard.json`.
- The remaining Stage 7 coverage gap is L1 shape breadth (`215/685`); L2 is no longer the blocker for shape coverage accounting.
- The artifacts preserve the required correctness/performance metadata (`perf_target`, `perf_actual`, `perf_pass`, `perf_gap`, plus memory-policy fields) for future reruns and gate checks.

## Artifact references

- `benchmarks/results/phase1/stage7/track6/coverage_gap.json`
- `benchmarks/results/phase1/stage7/track6/audit_report.json`
- `benchmarks/results/phase1/stage7/track6/dashboard.json`
- `benchmarks/results/phase1/stage7/track6/stage7_operator_shape_stats.json`
- `benchmarks/results/phase1/stage7/track6/l1/PERF_ALL.csv`
- `benchmarks/results/phase1/stage7/track6/l2/PERF_ALL.csv`
- `benchmarks/results/phase1/stage7/track6/l1/summary.json`
- `benchmarks/results/phase1/stage7/track6/l2/summary.json`
