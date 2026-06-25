# G9[1] — GPT-2 BL6 E2E real measurement (eager vs torch.compile)

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6), fp16,
warmup=10, runs=20, `dynamic=True`.

G9[1] requires GPT-2 ≥1.00× eager at the BL6 seq coverage (128/512/1024),
correctness 100%, ≤4GB. Evaluated under the **geomean-over-seq口径**
(Leon-approved 2026-06-25, D3) consistent with G8[4a].

## Result

| SeqLen | eager (ms) | torch.compile (ms) | ratio | correct | top-1 | mem (MB) |
|---|---|---|---|---|---|---|
| 128 | 7.25 | 8.73 | 0.831× | yes | yes | 539 |
| 512 | 12.93 | 10.76 | 1.201× | yes | yes | 665 |
| 1024 | 23.04 | 21.09 | 1.093× | yes | yes | 832 |

- **Correctness 100%** (top-1 token match all seq lens), peak mem 832 MB ≤ 4 GB.
- **geomean = 1.0296× ≥ 1.00×** → G9[1] GPT-2 bar met under geomean口径.
- min = 0.831× (seq=128) — the small-model launch-bound known-fail pattern
  (same root cause as the G8 seq=256 dip; documented in
  `../gpt2_real_2026-06-25/README.md`). torch.compile wins on the larger
  seq=512 (1.201×) and seq=1024 (1.093×) where matmuls amortize launch cost.

## Regenerate

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l3 --model gpt2 --seq-len 128,512,1024 \
    --modes eager,torch.compile --device cuda --warmup 10 --runs 20 \
    --output <out_dir>
```
