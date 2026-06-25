# Tier-2[5] — LLaMA-family E2E real measurement (D4=L2: TinyLlama-1.1B)

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6), CUDA 12.4,
PyTorch 2.6.0+cu124 · **Config:** warmup=10, runs=20, dtype fp16.

D4=L2 (Leon-approved 2026-06-25): substitute an **ungated LLaMA-architecture**
model that fits 6GB to validate the LLaMA-family E2E path on the dev box; full
gated LLaMA-2 7B is deferred to a larger GPU (blocked here by gated download +
missing `bitsandbytes`/`accelerate` for quant — see
`../multimodel_probe_2026-06-25/probe.md`).

Model: **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (LLaMA architecture, ungated,
~2.1 GB fp16). Run via the new generic `_load_causal_lm` loader
(`bench_l3 --arch llama`).

## Result

| SeqLen | eager (ms) | torch.compile (ms) | ratio | correct | top-1 | mem (MB) |
|---|---|---|---|---|---|---|
| 128 | 27.27 | 19.00 | **1.435×** | yes | yes | 4478 |
| 256 | 37.38 | 31.69 | **1.179×** | yes | yes | 4493 |
| 512 | 75.12 | 66.86 | **1.124×** | yes | yes | 4521 |

- **Correctness 100%** (top-1 token match on all seq lens), no OOM (~4.5 GB peak).
- **geomean = 1.239×**, min = 1.124× — **every seq len beats eager**, well above
  the G8 Tier-2[5] LLaMA bar (≥0.90× eager).

## Why this is cleaner than GPT-2

Unlike GPT-2 (where seq=256 is a known-fail because the small matmuls are
launch-overhead-bound), TinyLlama's larger 1.1B matmuls give torch.compile room
to win at **every** seq len. This confirms the GPT-2 seq=256 dip is a
small-model artifact, not a pipeline/measurement problem — the LLaMA-family E2E
path is sound.

## Status vs Gate

- G8 Tier-2[5] target = LLaMA-2 7B ≥0.90× eager. This is validated on the
  **LLaMA architecture** via TinyLlama (geomean 1.239×, all ≥1.12×). Full
  LLaMA-2 7B remains deferred to a larger GPU per D4=L2.

## Regenerate

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l3 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --arch llama --seq-len 128,256,512 --modes eager,torch.compile \
    --device cuda --warmup 10 --runs 20 --output <out_dir>
```
