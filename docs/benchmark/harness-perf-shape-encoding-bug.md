# Harness perf-path shape-encoding bug — diagnostic report

**Date:** 2026-06-27
**Author:** Kitty (lead engineer)
**Status:** ArkeRunner fixed (`2baa0bb`); baseline runners NOT fixed — awaiting project-lead decision (touches cross-baseline fairness).
**Class:** s7f1-shape-encoding

## Summary

The benchmark **perf-measurement path** (`Runner.get_fn(op, M, N, K, dtype)`,
which builds its own input tensors for timing) uses the legacy squashed
`(M, N, K)` shape convention. For ops whose benchmark shape dataclass is NOT a
flat `(M, N, K)` — batch_matmul, grouped_matmul, silu_and_mul, gelu_and_mul,
dequantize_per_channel — this builds the **wrong-shaped workload**, so the
PERF_ALL latency for those ops is measuring something unrelated to the shape
tag it's filed under.

The **correctness path** (`build_inputs` in bench_l1.py + `run_with_inputs`)
already consults `get_current_shape()` and builds the canonical shape. The two
paths diverged: correctness is right, perf is wrong.

## Concrete symptoms (track6 / followup1 PERF_ALL, before fix)

| op | shape tag | what perf path built | real workload | reported Arke | real Arke |
|---|---|---|---|---|---|
| batch_matmul | llama-attn-2k | A[max(K,4)=128, M=32, N=2048] (K as batch) | B32 M2048 K128 N2048 | 146998 us | ~2400 us |
| batch_matmul | gpt2-attn-512 | wrong | B12 M512 K64 N512 | 28% cuBLAS | 82% cuBLAS |
| silu_and_mul | llama-7b-2k | randn(M,2N) microscopic | seq2048 ffn22016 | ~29 us flat | ~431 us |
| grouped_matmul | moe-medium | A[M,N,K] N/K swap, E=4 | B16 E8 M128 K768 N3072 | 12583 us | ~1068 us |

## Fix map — which (runner, op) pairs still use the squashed (buggy) path

(after `2baa0bb` fixed ArkeRunner)

| runner | batch_matmul | grouped_matmul | silu_and_mul | gelu_and_mul | dequantize |
|---|---|---|---|---|---|
| arke_runner.py | ✅ FIXED (2baa0bb) | ✅ FIXED | ✅ FIXED | ✅ FIXED | n/a (squash OK, M/N flat) |
| pytorch_eager.py | ❌ SQUASH | ❌ SQUASH | ❌ SQUASH | ❌ SQUASH | ❌ SQUASH |
| cublas.py | ❌ SQUASH | — | — | — | — |
| flaggems.py | ❌ SQUASH | — | — | — | — |
| inductor.py | ❌ SQUASH | — | — | — | — |
| liger.py | — | — | ✅ CTX | ✅ CTX | — |

(`flaggems.py` / `pytorch_eager.py` DO import `get_current_shape` but only use
it for the OT4 attention ops fixed in S7.followup.3 — NOT for bmm/gated.)

Attention ops (flash/gqa/cross/mla/paged) already consult get_current_shape in
all runners that support them (fixed during S7.followup.3) — not in scope here.

## Why this matters / doesn't

- **Cross-baseline ratios for these 5 ops are unreliable** until all runners
  building those ops are harmonized: Arke (fixed) now measures the real shape
  while eager/cuBLAS/FlagGems (unfixed) still measure the squashed shape →
  apples-to-oranges. E.g. silu now shows Arke 605us (real) vs eager 44us
  (microscopic) — a meaningless ratio.
- **Correctness is unaffected** (always used the right path).
- **The benchmark shape DEFINITIONS are correct** (shapes.py is fine). This is
  purely an input-builder bug in the timing harness, not a measurement-protocol
  or shape-set change. Fixing it makes perf measure what the shape tag says —
  it does not relax or alter any Gate threshold.

## Recommended fix (for project-lead decision)

**Preferred: extract a single shared `build_perf_inputs(op, shape, dtype)`** in
`benchmarks/baselines/_shared_inputs.py` that consults `get_current_shape()`
and returns canonical tensors, and have every runner's `get_fn` delegate to it.
This removes the per-runner duplication that let the bug diverge in the first
place (5+ runners each re-implement input building). Legacy squash stays as the
no-context fallback for unit tests.

Scope: ~5 runner files, ~5 ops. Mechanical once the shared builder exists.
Risk: low (changes only timing-input construction; correctness path untouched;
fallback preserves unit-test behavior).

## Decision needed

1. Harmonize all runners via shared builder (preferred, ~5 files).
2. Per-op patch in each runner (smaller diffs, keeps duplication).
3. Defer; use standalone micro-benchmarks as perf evidence for codegen work
   until a dedicated harness-harmonization pass.
