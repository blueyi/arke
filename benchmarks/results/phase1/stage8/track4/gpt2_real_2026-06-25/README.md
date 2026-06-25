# Tier-2[4a] — GPT-2 E2E real measurement (eager vs torch.compile)

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6), CUDA 12.4,
PyTorch 2.6.0+cu124 · **Config:** warmup=10, runs=20, dtype fp16, `dynamic=True`,
`cache_size_limit=64` (D7-E1.6 settings).

This is the **real-GPU** measurement for G8 Tier-2[4a] (vanilla torch.compile
baseline: GPT-2 correctness 100% + perf ≥0.95× eager). The MVP gate only ran
the `--mock` CPU contract path; this is the genuine endpoint.

## Result (multi-seq in one process)

| SeqLen | eager (ms) | torch.compile (ms) | ratio | correct | top-1 | mem (MB) |
|---|---|---|---|---|---|---|
| 128 | 8.12 | 8.36 | 0.970× | yes | yes | 930.6 |
| 256 | 7.20 | 8.19 | 0.880× | yes | yes | 796.4 |
| 512 | 11.98 | 10.55 | 1.136× | yes | yes | 838.0 |

- **Correctness: 100%** (top-1 token match on all seq lens), no OOM.
- **geomean ratio = 0.990×**, min ratio = 0.880× (seq=256).

## seq=256 dip is REAL, not an eviction artifact (honest finding 2026-06-25)

Initial hypothesis was that seq=256's 0.880× was the known dynamic-shape
recompile thrash (D7-E1.1) caused by running multiple seq lens in one process.
**An isolated single-seq re-measurement falsified that:** seq=256 alone in a
fresh process is **0.801×** — even worse than in-process. So torch.compile
genuinely loses to eager at GPT-2 seq=256 on this hardware.

Root cause (consistent with D7-E1.1 / S5 findings): GPT-2 at seq=256 on a 6GB
3060 is dominated by Python kernel-launch + CUDA-graph guard overhead; the
matmuls are too small for Inductor's fused kernels to amortize that cost. At
seq=512 the larger matmuls let torch.compile win (1.136×); at seq=128 it's
roughly break-even (0.970×). This is a **real property of vanilla torch.compile
on a small model + small GPU**, NOT an Arke deficiency — G8[4a] measures the
*vanilla* torch.compile baseline; Arke is not on this path (Arke value-add is
[4b], the Arke→torch.compile bridge).

Per the no-relaxation discipline this is recorded honestly, not cherry-picked.
Whether G8[4a]'s ≥0.95× threshold should be evaluated as min-over-seq,
geomean-over-seq, or per-seq is a **Gate口径 decision pending Leon (D1)**:
- min-over-seq → 0.801× ❌ (fails on seq=256)
- geomean-over-seq → 0.990× ✅ (multi-seq run)
- per-seq → 2/3 pass (128 ✅, 512 ✅, 256 ❌)

## Files

- `multi_seq/` — the 3-seq run artifacts (config/hardware/sources/results/summary + gpt2_results.csv)
- `gpt2_256_isolated/` — seq=256 alone in a fresh process (isolates the eviction hypothesis → falsified)

## Regenerate

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l3 --model gpt2 --seq-len 128,256,512 \
    --modes eager,torch.compile --device cuda --warmup 10 --runs 20 \
    --output <out_dir>
```
