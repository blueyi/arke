# Multi-model endpoint reachability probe (D7-E2 / D7-E3.0)

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6) ·
**Free VRAM at probe:** 5.37 GB / 6.44 GB total.

Probe for G8 Tier-2[5] (LLaMA-2 7B) and Tier-2[6] (DeepSeek-V2 16B) on the
locked dev hardware, per D2=a (Leon: "GPT-2 真测 / LLaMA-2 尽力 / DS-V2 audit").
This is the D7-E3.0-style reachability gate the plan anticipates **before**
committing to full integration.

## Dependency state

| Dep | Needed for | Status |
|---|---|---|
| `transformers` | model load | ✅ present |
| `safetensors` | weight load | ✅ present |
| `bitsandbytes` | 4-bit/8-bit quant load | ❌ **missing** |
| `accelerate` | device_map / offload | ❌ **missing** |

Without `bitsandbytes` + `accelerate`, 4-bit quantized loading (the only way a
7B/16B model could approach a 6GB budget) is not currently possible in the venv.

## VRAM feasibility (analytic)

| Model | 4-bit weights | + KV/activations @seq512 | Fits 5.37 GB free? |
|---|---|---|---|
| LLaMA-2 7B | ~3.5 GB | ~3.5 + ~1–2 GB | ⚠️ borderline / likely OOM |
| DeepSeek-V2 16B | ~8+ GB | ~8+ GB | ❌ **OOM (exceeds total 6.44 GB)** |

## Additional blocker

- **LLaMA-2 is a gated HF model** (requires license acceptance + token); not in
  cache, not downloadable without auth on this host.

## Outcome & recommendation (escalation per D7-E3.0 outcome (ii))

- **DeepSeek-V2 16B → `audit-only` + OOM evidence.** It cannot fit on 6GB even
  4-bit. Per no-relaxation discipline: recorded as audit-only with this VRAM
  evidence, NOT silently excluded, NOT relaxed. Matches the plan's pre-approved
  "audit-only G8[6] if E3.0 (ii)" branch.
- **LLaMA-2 7B → blocked on this host** by (a) missing `bitsandbytes`/`accelerate`
  and (b) gated download. Two ways forward, both needing Leon's call because
  they touch Tier-2 model selection (a locked Gate surface):
  - **Option L1:** install `bitsandbytes`+`accelerate`, obtain a gated-LLaMA-2
    token, attempt 4-bit seq≤256 best-effort (likely OOM-borderline).
  - **Option L2:** substitute an **ungated LLaMA-architecture** model that fits
    (e.g. TinyLlama-1.1B, same arch, ~2.2 GB fp16) to validate the LLaMA-family
    E2E path on the dev box, with full LLaMA-2 7B deferred to a larger GPU.

## Status

- DS-V2: audit-only (evidence here) — no further dev effort on 6GB.
- LLaMA-2: **pending Leon decision (D4: L1 vs L2)** — not started to avoid
  unilaterally changing Tier-2 model selection.
