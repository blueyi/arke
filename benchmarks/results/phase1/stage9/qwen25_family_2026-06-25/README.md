# G9[1] — Qwen2.5-family E2E (GQA) via Qwen2.5-0.5B-Instruct

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6), fp16,
warmup=10, runs=20.

Qwen2.5-family endpoint on the 6GB dev box (same substitute logic as D4=L2 for
LLaMA): the ungated **Qwen2.5-0.5B-Instruct** (Qwen2 GQA architecture, ~1 GB
fp16) validates the GQA + wide-FFN path that G9[1]'s "Qwen2.5 7B" row targets.
Full Qwen2.5 7B (~6 GB fp16 weights) is deferred to a larger GPU.

## Result

| SeqLen | eager (ms) | torch.compile (ms) | ratio | correct | top-1 |
|---|---|---|---|---|---|
| 128 | 27.09 | 19.07 | 1.420× | yes | yes |
| 512 | 38.68 | 33.55 | 1.153× | yes | yes |

- **Correctness 100%** (top-1 match), no OOM (~2.4 GB peak).
- **geomean = 1.2796×**, min = 1.153× — both seq lens beat eager, well above
  the G9[1] Qwen2.5 bar (≥0.90× eager).

The GQA (grouped-query attention) + wide-FFN compute pattern Qwen represents
compiles cleanly and runs faster than eager — the Arke/torch.compile E2E path
generalizes across the GPT-2 (MHA), LLaMA (GQA+RMSNorm+RoPE), and Qwen2.5
(GQA+wide-FFN) architecture families on this hardware.

## Regenerate

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l3 --model Qwen/Qwen2.5-0.5B-Instruct --arch qwen2 \
    --seq-len 128,512 --modes eager,torch.compile --device cuda \
    --warmup 10 --runs 20 --output <out_dir>
```
