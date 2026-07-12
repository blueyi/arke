# Phase 4 — Known Issues & Follow-ups

**As of:** 2026-07-12 (post session 7)

This doc records non-blocking known issues and scoped follow-ups so future
sessions don't re-investigate settled findings.

---

## 1. Full-suite test pollution (PRE-EXISTING, not a Phase-4 regression)

**Symptom:** Running the *entire* `tests/` suite in one process yields ~62
failures in the external-baseline correctness probes
(`test_benchmark_correctness_probe_linea{6,7,8,9}.py`,
`test_triton_codegen_smoke.py`, `test_triton_backend_dispatch.py`) — typically
`Float did not match Half` or dtype/allclose mismatches on
Triton/Inductor/cuBLAS/Liger runners.

**Root cause:** global-state pollution across test modules — the FlagGems
`aten::mm` global hijack + shared CUDA context/dtype state leaks between
modules depending on import/run order. This is a long-standing test-infra
characteristic, **not** a codegen or backend correctness bug.

**Proof it's not a regression:**
- Every failing file **passes in isolation** (e.g. `pytest
  test_benchmark_correctness_probe_linea7.py` → 8/8 pass).
- Today's Phase-4 changes touched only `arke/backend/cuda_c_*`,
  `arke/agent/tools.py` (additive `backend=` param), and
  `arke/agent/verification.py` (new module) — none are imported by the
  polluted probe modules.
- All Phase-4 + Harness work areas pass **182/182** together.

**Fix (future, low priority):** add a FlagGems-hijack teardown fixture or run
the external-baseline probes in a separate pytest process (`-p forked` or CI
sharding). Tracked; not blocking Phase 4.

---

## 2. CUDA-C op coverage: 31/46 (P4-S2 gate = 30, MET)

**Covered (31, all 5 tiers):** see `stage-progress.md`.

**Not yet in CUDA-C (15):** argmax, topk, cumsum, grouped_matmul, gather,
scatter, swiglu_packed, rope, fused_linear_cross_entropy, quantize_per_token,
dequantize_per_channel, grouped_query_attention, multi_latent_attention,
cross_attention, paged_attention.

These are the exotic/variant ops (other attention flavors, quantization,
scatter/gather, rope). Natural follow-ups; P4-S2's 30-op gate is already met.

---

## 3. Performance follow-ups (week-level, scoped)

| Item | Current | Ceiling cause | Fix |
|---|---|---|---|
| matmul small shape (512) | 0.38× | wave-quantization (64 blocks / 30 SM) | different tile config for wave-fill (e.g. 32×32 or 128×32) |
| flash_attention large-seq | 0.18× | O(S²) per-warp, no cross-block K parallelism | FlashAttention-2 cross-block K-tile reduction + TC |
| matmul large shape | 0.96× (2048) | near parity already | diminishing returns |

Double-buffering (commit d138a45) already landed for the K-loop latency-hiding.

---

## 4. Harness D2 verification layer: 3/4 mechanisms landed

- ✅ robust_reward (CUDA Agent discrete schedule)
- ✅ staged_correctness_gate (AutoKernel 5-stage firewall)
- ✅ reflexion_feedback (GEAK error-trace self-correction)
- ⬜ Sakana LLM soft-verifier prefilter — needs live-LLM token budget (D3-adjacent)

---

*Last updated: 2026-07-12*
