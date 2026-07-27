# Golden Kernel Ladder — Per-Op Quick Reference

For each Arke catalog op, the **Golden Kernel** is the designated
production kernel that simultaneously serves as:

1. **Correctness oracle** — its output is the expected value for
   `_compare_l1_outputs` comparisons.
2. **Perf denominator** — its latency anchors `ratio_vs_baseline`.

Selection rule (encoded in
[`benchmarks/golden_ladder.py`](../../benchmarks/golden_ladder.py)):
iterate registered runners by priority ascending; return the first one
where `runner.supports(op) and runner.available`.

## Priority bands

| P  | Runner             | Source                                           |
|---:|:-------------------|:-------------------------------------------------|
| P0 | `cuBLAS/cuDNN`     | PyTorch vendor backends (`torch.matmul`, `F.scaled_dot_product_attention`, `F.layer_norm`, ...) |
| P1 | `FlagGems`         | https://github.com/flagos-ai/FlagGems (set `GEMS_VENDOR=nvidia`) |
| P1 | `Liger-Kernel`     | https://github.com/linkedin/Liger-Kernel (preferred for OT3 fused) |
| P2 | `flash-attn`       | https://github.com/Dao-AILab/flash-attention   |
| P2 | `FlashMLA`         | https://github.com/deepseek-ai/FlashMLA (Hopper sm_90+ only) |
| P2 | `vLLM`             | https://github.com/vllm-project/vllm (paged attention) |
| P3 | `PyTorch-eager`    | Plain `torch.*` — fallback for ops without P0/P1/P2 |
| P3 | `Triton-Tutorial`  | Hand-written Triton kernels (`benchmarks/baselines/triton_tutorial.py`) |
| P4 | `torch.compile`    | Inductor — separate runner, **never** the golden |
| P5 | `Arke`             | Our own — **never** the golden |
| P5 | `LLM-direct`       | LLM-generated raw code — **never** the golden |

## Per-op assignments

### OT0 — Elementwise (12)

| Op       | Golden                | Fallback           | Notes |
|:---------|:----------------------|:-------------------|:------|
| relu     | cuBLAS                | FlagGems           | |
| gelu     | cuBLAS                | FlagGems / Liger   | |
| silu     | cuBLAS                | FlagGems / Liger   | |
| tanh     | cuBLAS                | FlagGems           | |
| sigmoid  | cuBLAS                | FlagGems           | |
| add      | cuBLAS                | FlagGems           | |
| mul      | cuBLAS                | FlagGems           | |
| where_   | FlagGems (P1)         | PyTorch-eager (P3) | cuBLAS has no kernel |
| cast     | PyTorch-eager (P3)    | FlagGems `_to_copy` | ATen path; audit-degraded |
| neg      | cuBLAS                | FlagGems           | |
| exp      | cuBLAS                | FlagGems           | |
| rsqrt    | cuBLAS                | FlagGems           | |

### OT1 — Reduction (10)

| Op                | Golden                  | Fallback           |
|:------------------|:------------------------|:-------------------|
| softmax           | cuBLAS                  | FlagGems / Triton-Tutorial |
| layernorm         | cuBLAS                  | FlagGems / Liger   |
| rmsnorm           | Liger (P1)              | FlagGems           |
| rmsnorm_residual  | Liger (P1)              | PyTorch-eager      |
| reduce_sum        | cuBLAS                  | FlagGems           |
| reduce_max        | cuBLAS                  | FlagGems           |
| reduce_mean       | cuBLAS                  | FlagGems           |
| argmax            | FlagGems (P1)           | PyTorch-eager      |
| topk              | FlagGems (P1)           | PyTorch-eager      |
| cumsum            | FlagGems (P1)           | PyTorch-eager      |

### OT2 — Data Movement & Dense (11)

| Op             | Golden                  | Fallback           |
|:---------------|:------------------------|:-------------------|
| matmul         | cuBLAS                  | FlagGems / Triton-Tutorial |
| batch_matmul   | cuBLAS                  | FlagGems           |
| grouped_matmul | FlagGems (P1)           | PyTorch-eager      |
| transpose      | cuBLAS                  | FlagGems           | audit-only for Gate (no Triton-only golden; cuBLAS≠Triton). Baselines must materialize (.contiguous()) — a lazy .T view is O(1) and yields a bogus ~0.01× ratio vs Arke's materializing kernel. Fair micro-bench: Arke 3.09× vs eager-contiguous at 4096². See harness-perf-shape-encoding-bug.md |
| concat         | FlagGems (P1)           | PyTorch-eager      |
| split          | PyTorch-eager (P3)      | —                  | view op; bandwidth-only |
| gather         | FlagGems (P1)           | PyTorch-eager      |
| scatter        | FlagGems (P1)           | PyTorch-eager      |
| embedding      | FlagGems (P1)           | PyTorch-eager      |
| permute        | PyTorch-eager (P3)      | —                  | view + contiguous |
| copy_          | FlagGems (P1)           | PyTorch-eager      |

### OT3 — Fused Compound (7)

| Op                          | Golden                    | Fallback           |
|:----------------------------|:--------------------------|:-------------------|
| silu_and_mul                      | Liger (P1)                | PyTorch-eager      |
| gelu_and_mul                       | Liger (P1)                | PyTorch-eager      |
| rope                        | **PyTorch-eager (P3)** †  | Liger (P1, candidate) |
| fused_linear_cross_entropy  | Liger (P1)                | PyTorch-eager      |
| cross_entropy               | Liger (P1)                | FlagGems / PyTorch |
| quantize_per_token          | FlagGems (P1)             | PyTorch-eager      |
| dequantize_per_channel      | PyTorch-eager (P3)        | —                  | no production kernel; audit-degraded |
| swiglu_packed               | PyTorch-eager (P3)        | —                  | D8-X1 demo op; no production single-kernel baseline in 9 audited community libraries (FlagGems · Liger · vLLM · flash-attn · FlashMLA · xformers · TransformerEngine · Megatron-LM · FlashInfer · NVIDIA Apex · DeepSpeed); audit-degraded |

> † **G7.8c locked preference (2026-05-12).** rope is the one OT3 op whose
> Golden is *not* the strict P-winner. Liger-Kernel rope is the fastest
> production kernel (and remains a benchmark candidate), but it raises on
> odd-D head dimensions and a handful of non-aligned shape combinations
> (see `tests/test_benchmark_correctness_probe_linea12.py::test_rope_odd_head_dim*`
> and commits `ad28665`, `c80d182`). A Golden Kernel must be defined for
> every shape we measure; PyTorch-eager rope satisfies that across the
> entire OT3 shape grid and matches the analytical reference. The pin is
> encoded in `benchmarks/golden_ladder.LADDER_PREFERENCES` and exempt
> from the per-op P-order rule. To ad-hoc swap back to Liger for an
> experiment, use `--golden rope=Liger-Kernel`.

### OT4 — Attention (5)

| Op                       | Golden                    | Fallback                | NOTE |
|:-------------------------|:--------------------------|:------------------------|:-----|
| flash_attention          | **flash-attn (P2)** §     | FlagGems (bmm-decomposed)| fused FlashAttention2 (OT4 re-review 2026-07-27) |
| grouped_query_attention  | **flash-attn (P2)** §     | FlagGems (bmm-decomposed)| native GQA, no K/V expansion (OT4 re-review 2026-07-27) |
| cross_attention          | **FlagGems (P1)** ‡       | PyTorch-eager (hijacked)| flash_attn_func needs equal Q/KV seq; varlen API eval = follow-up |
| multi_latent_attention   | PyTorch-eager (P3)        | —                       | no production kernel; audit-degraded (FlashMLA Hopper-only, 9 audited libs lack sm 8.6 Triton MLA) |
| paged_attention          | PyTorch-eager (P3)        | —                       | no production kernel; audit-degraded (vLLM triton_attn is prefill-only; KV-cache layout mismatch) |

> ‡ **S7.followup.3 locked preference (2026-06-06) — superseded for
> FA/GQA on 2026-07-27, still governs cross_attention.** FlagGems was
> promoted as the OT4 golden on the premise it was the only audited
> library shipping a Triton-only fused SDPA on sm 8.6 (RTX 3060), and
> because the prior cuDNN P0 entry was a same-backend-fairness lie
> (its `get_fn` called `F.scaled_dot_product_attention`, which gets
> hijacked into FlagGems the moment `flag_gems.enable()` runs). MLA +
> paged_attention stay `audit-degraded` (no production kernel in the 9
> audited libraries; PyTorch-eager P3 catches them, same precedent as
> `swiglu_packed`).
>
> § **OT4 re-review locked preference (2026-07-27, Leon sign-off).**
> The S7.followup.3 premise no longer holds: flag_gems **5.0.0**'s SDPA
> is **bmm-decomposed**, not fused — profiler shows `bmm=True,
> flash-kernel=False, softmax=True`, and `flag_gems/ops/bmm.py`
> materializes the full `[B*H, S, S]` score buffer (observed 32 GiB /
> 112 GiB allocation attempts → OOM on tier-2 shapes, 6 GiB card).
> flash-attn **2.7.4.post1** (prebuilt wheel
> `cu12torch2.6cxx11abiFALSE`, no source build needed) is the only
> genuinely fused attention on this hardware: single
> `_flash_attn_forward` kernel, O(S) memory (16k-seq peak 0.16 GB),
> native GQA (`Hkv != Hq` without K/V `repeat_interleave`), fp16
> max_abs_diff 9.7e-4 vs CPU-fp64 reference, and immune to the
> flag_gems ATen hijack (direct extension call, no ATen dispatch).
> Pinned via `LADDER_PREFERENCES` (same mechanism as rope). Full
> evidence: `docs/benchmark/ot4-golden-review-rfc.md`. cross_attention
> keeps FlagGems because `flash_attn_func` requires equal Q/KV seq
> lens — evaluating the varlen API for it is a follow-up.

## Locking & change control

This table is part of the design contract. Modifying any per-op assignment
requires Leon's explicit sign-off (cf. `~/workspace/AGENTS.md` "Must
confirm with Leon"). Code-side, the assignment is *implicit* — it follows
from each runner's `supports()` set + `priority`. To shift a golden,
change the runner's `supports()` declaration, not this table.

## Locked ladder preferences (`LADDER_PREFERENCES`)

A small protocol-level dict in
[`benchmarks/golden_ladder.py`](../../benchmarks/golden_ladder.py) pins a
Golden for ops where the strict P0-first rule chooses a runner that is
*fast but not a stable oracle*. These pins are part of this design
contract — adding/removing entries requires Leon's sign-off, just like
table edits.

Current entries:

| Op   | Pinned Golden     | Why                                                |
|:-----|:------------------|:---------------------------------------------------|
| rope | PyTorch-eager (P3) | Liger rope odd-D & non-aligned shapes raise; eager covers full grid (G7.8c, 2026-05-12) |
| flash_attention | flash-attn (P2) | FlagGems 5.0.0 SDPA is bmm-decomposed (non-fused, OOMs on tier-2); flash-attn is the only true fused kernel on sm 8.6 (OT4 re-review, 2026-07-27) |
| grouped_query_attention | flash-attn (P2) | same as flash_attention + native GQA without K/V expansion (OT4 re-review, 2026-07-27) |

Caller-supplied `--golden` / `--golden-file` overrides take precedence
over `LADDER_PREFERENCES` so ad-hoc experiments aren't blocked.

## Overrides

Two CLI mechanisms support ad-hoc experimentation without changing
runner declarations:

```bash
python -m benchmarks.bench_l1 --op softmax --golden softmax=FlagGems
python -m benchmarks.bench_l1 --all --golden-file ./overrides.yaml
```

The override-pinned runner must declare `supports(op)` and be `available`;
otherwise `GoldenUnavailable` fires and the row is marked
`golden_unavailable_pending_baseline`.
