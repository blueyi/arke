# D8-X1 BL1 Evidence — swiglu_packed Onboarding

This directory contains the frozen BL1 acceptance evidence for the
Stage 8 Track 0 D8-X1 extensibility demo (per
`docs/phase1/stage8-plan.md` Tier 1 Extensibility Acceptance row
"BL1 measurement").

## Files

| File | Purpose |
|:-----|:--------|
| `bl1_new_op.csv` | Frozen snapshot of the swiglu_packed BL1 result table (12 rows = 6 shapes × 2 baselines: PyTorch-eager + Arke) |

## How it was generated

```bash
cd /home/blueyi/workspace/repos/arke
source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l1 \
  --op swiglu_packed \
  --tier 1 \
  --phase 1 --stage 8 --track extensibility \
  --warmup 5 --reps 30 --no-resume

# Native bench_l1 output:
#   benchmarks/results/phase1/stage8/trackextensibility/l1/swiglu_packed_results.csv
# Frozen here as bl1_new_op.csv (verbatim copy) for D8-X1 acceptance.
```

Hardware: RTX 3060 Laptop 6GB (Ampere, SM 8.6), CUDA 12.4, PyTorch
2.6.0+cu124, fp16.

## Result summary (6 shapes, Tier 1)

| Shape         | M    | N    | K    | PyTorch-eager (μs) | Arke (μs)      | Notes                                |
|:--------------|-----:|-----:|-----:|-------------------:|:---------------|:-------------------------------------|
| tiny          |  128 |  128 |  128 |              135.9 | unsupported    | Launch-overhead-dominated            |
| gpt2-c_proj   |  128 |  768 |  768 |               48.5 | unsupported    | GPT-2 c_proj projection              |
| gpt2-c_attn   |  128 | 2304 |  768 |               57.7 | unsupported    | GPT-2 QKV (3H output)                |
| square-1k     | 1024 | 1024 | 1024 |              245.8 | unsupported    | Standard square GEMM                 |
| square-2k     | 2048 | 2048 | 2048 |             1002.7 | unsupported    | Compute-bound regime                 |
| square-4k     | 4096 | 4096 | 4096 |             6614.7 | unsupported    | Large GEMM (memory + compute heavy)  |

## Interpretation

- **Correctness column (`allclose`)** for all 6 PyTorch-eager rows = `true`.
  `max_abs_diff = mean_abs_diff = 0` — exact match against the golden
  oracle (which is itself PyTorch-eager, per
  `golden-kernel-ladder.md` OT3 row 8, audit-degraded).
- **`status = ok`** on all 6 PyTorch-eager rows. Each row carries a
  measured `latency_us` + `latency_min_us` over 30 reps with 5 warmups.
- **`status = unsupported`** on the 6 Arke rows. Reason:
  `"Arke.get_fn declined swiglu_packed@<shape>"`. This is **expected** —
  Arke's Triton codegen template for `swiglu_packed` is **not** in
  D8-X1 scope. The Arke kernel will be authored in a later stage
  (S7+) when the agent invokes the
  `skills/swiglu-packed-fusion/SKILL.md` recipe with a non-trivial
  budget. D8-X1 acceptance asks only for **catalog onboarding +
  baseline correctness + perf rows**, all of which are present.

## Acceptance

This artifact satisfies the BL1 row in the Stage 8 Tier 1 Extensibility
Acceptance table (`docs/phase1/stage8-plan.md` Demo A):

> *"new op appears in `benchmarks/results/phase1/stage8/extensibility/bl1_new_op.csv` with correctness + perf rows"*

✅ Op appears (12 rows for `op=swiglu_packed`).
✅ Correctness rows present (`allclose=true`, `correctness_status=ok`).
✅ Perf rows present (`latency_us`, `latency_min_us`).

Audit context: `docs/benchmark/audit/swiglu_packed_baseline_audit_2026-05-30.md`
(audit-degraded — no community single-kernel baseline; PyTorch-eager
P3 is the only available reference, recorded per Route-b precedent
shared with `dequantize_per_channel`).
