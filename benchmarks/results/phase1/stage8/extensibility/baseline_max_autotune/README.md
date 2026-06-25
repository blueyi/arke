# Demo B (D8-X2) — New Baseline Runner Onboarding Evidence

**Tier 1 [HARNESS-3] Extensibility Demo B** — proves a new `BaselineRunner`
subclass onboards into the Arke benchmark ladder within the ≤200 LOC budget
**without touching the harness core**.

## What was onboarded

`MaxAutotuneRunner` (`benchmarks/baselines/max_autotune.py`, **141 LOC**) wraps
`torch.compile(mode="max-autotune")` — a baseline source **distinct** from the
existing `InductorRunner` (which uses `mode="reduce-overhead"`). The
max-autotune mode enables Inductor's autotuning Triton template search +
CUDA-graph capture, producing different kernels. It is therefore a legitimate
*alternate Triton-implementation source* under the Same-Backend Triton Fairness
rule.

## Acceptance checklist

| Item | Hard limit | Status |
|---|---|---|
| Total new LOC | ≤ 200 | ✅ 141 LOC (runner) |
| `BaselineRunner` subclass protocol documented | new § in `arke-harness.md` | ✅ §9.1 "Onboarding a new BaselineRunner" |
| Plugged into | `benchmarks/baselines/<runner>.py` | ✅ `max_autotune.py` + registered in `bench_l1.py` import block |
| BL1 cross-coverage | new runner in ≥1 BL1 result row | ✅ `bl1_matmul.csv` (6 rows) + `bl1_relu.csv` (5 rows) |
| Unit tests | — | ✅ `tests/test_max_autotune_runner.py` (4 pass) |

## Files in this directory

- `bl1_matmul.csv` — BL1 tier-1 matmul perf rows including the new runner
- `bl1_relu.csv` — BL1 tier-1 relu perf rows including the new runner

## Regenerate

```bash
cd ~/workspace/repos/arke
source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l1 --op matmul,relu --tier 1 \
    --phase 1 --stage 8 --track demoB_extensibility
cp benchmarks/results/phase1/stage8/trackdemoB_extensibility/l1/perf_matmul.csv \
   benchmarks/results/phase1/stage8/extensibility/baseline_max_autotune/bl1_matmul.csv
cp benchmarks/results/phase1/stage8/trackdemoB_extensibility/l1/perf_relu.csv \
   benchmarks/results/phase1/stage8/extensibility/baseline_max_autotune/bl1_relu.csv
```

## Note on hardware

On the RTX 3060 Laptop (6 GB, SM 8.6), Inductor logs
`Not enough SMs to use max_autotune_gemm mode` and falls back to its default
gemm template; for tiny elementwise shapes the CUDA-graph + dispatch overhead
dominates, so this runner is slow there (~24–30% of cuBLAS). That is expected
and irrelevant to the demo: Demo B measures **extensibility** (a new runner
onboards and participates in the ladder at the claimed LOC cost), not that this
particular runner wins any shape. On larger GPUs max-autotune autotunes real
GEMM templates and is competitive.
