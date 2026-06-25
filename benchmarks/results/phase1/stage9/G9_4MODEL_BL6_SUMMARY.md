# G9[1] — 4-Model BL6 E2E Summary (Phase 1 Final)

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6), CUDA 12.4,
PyTorch 2.6.0+cu124, fp16, warmup=10 runs=20. Perf口径: **geomean-over-seq**
(Leon-approved D3, consistent with G8[4a]).

## G9[1] requirement

> 4 models E2E correctness 100%: GPT-2 ≥1.00×, LLaMA-2 ≥0.95×, LLaMA-3 ≥0.95×,
> Qwen2.5 ≥0.90× eager.

## Hardware reality (6 GB VRAM)

The full target models (LLaMA-2/3 7-8B, Qwen2.5 7B) do **not** fit a 6 GB
laptop GPU even before activations (7B fp16 ≈ 14 GB; 4-bit needs
`bitsandbytes`+`accelerate`, absent here — see
`../../stage8/track4/multimodel_probe_2026-06-25/probe.md`). Per the
Leon-approved substitute pattern (D4=L2, applied uniformly): validate each
**architecture family** with the largest ungated model that fits, and mark the
full-size target audit-only/deferred. No standard is relaxed; nothing is
silently excluded.

## Results — per architecture family

| G9[1] target | Family validated by | seq | correctness | geomean ratio | bar | status |
|---|---|---|---|---|---|---|
| GPT-2 (MHA) | **GPT-2** (actual model) | 128/512/1024 | 100% | **1.0296×** | ≥1.00× | ✅ |
| LLaMA-2 7B (GQA) | TinyLlama-1.1B (LLaMA arch) | 128/256/512 | 100% | **1.239×** | ≥0.95× | ✅ family |
| LLaMA-3 8B (GQA) | TinyLlama-1.1B (LLaMA arch) | 128/256/512 | 100% | **1.239×** | ≥0.95× | ✅ family / 8B deferred |
| Qwen2.5 7B (GQA+wide-FFN) | Qwen2.5-0.5B-Instruct (Qwen2 arch) | 128/512 | 100% | **1.2796×** | ≥0.90× | ✅ family / 7B deferred |

Evidence dirs:
- `gpt2_bl6_2026-06-25/` — GPT-2 (real target model)
- `tinyllama_2026-06-25/` (stage8/track4) — LLaMA family
- `qwen25_family_2026-06-25/` — Qwen2.5 family

## Reading

- **Correctness: 100% across every family**, no OOM.
- **Every family's geomean beats its eager bar**: GPT-2 1.03×, LLaMA 1.24×,
  Qwen 1.28×. The Arke/torch.compile E2E path generalizes across MHA, GQA, and
  GQA+wide-FFN architectures on the dev hardware.
- **Full-size 7-8B targets deferred to a larger GPU** (audit, hardware-bounded,
  not a quality failure). The architecture-family validation is the substance
  G9[1] tests — that the E2E path works across the model families Phase 1 cares
  about — which is demonstrated.

## Open口径 for project lead

Whether family-substitute validation **fully closes** G9[1] or whether the
full-size 7-8B models must run on a larger GPU before Phase 1 sign-off is a
Gate-acceptance call for Leon. The dev-box evidence is complete; the only gap
is full-parameter scale, which is purely a VRAM limit.
