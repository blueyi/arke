# K-ATT Plan — Triton flash-style attention to ≥0.35 (stage) / ≥0.5 (final)

**Approved decision (Leon 2026-07-29): `A1 B1 C2`**
**Status:** 📋 PLAN LOCKED — kernel iteration runs in a dedicated session (this doc is the contract)
**Owner:** Kitty (full implementation authority; only Gate thresholds are frozen — locked below)
**Backlog:** `docs/audit/kestrel-backlog.md` K-ATT (P1, 性能主线, 2-4w)
**Decision material:** `docs/kestrel/k-att-decision-material.md`

---

## 0. Locked frozen-layer parameters (Leon-approved, do NOT change without Leon)

| Param | Value | Source |
|:--|:--|:--|
| **A1 — Stage Gate** | Triton FA vs flash-attn golden **geomean ≥ 0.35** | = same-hardware CUDA-C v8 lower bound (0.35–0.42×); "Triton is no longer the short pole" |
| **A1 — Final Gate** | Triton FA vs flash-attn golden **geomean ≥ 0.50** | 1.4× of CUDA-C v8; Triton becomes an independently strong backend |
| **B1 — Path** | **Pure Triton flash-style** (online softmax + K/V double-buffer + `tl.dot` TC) | No CUDA-C bridge; Triton backend must stand alone for the AI-Native thesis |
| **C2 — Order** | **FA first, then GQA**; thresholds set **separately** per op | FA online-softmax is GQA's prerequisite skill |

**Correctness gate (implementation-layer, my call):** `max_abs_diff ≤ 5e-3` vs torch SDPA fp16 (same threshold used across the OT4 golden family). A kernel that fails correctness scores 0 regardless of latency — LIVE ≠ correct.

**Measurement protocol (locked to match the golden family):**
- Golden = flash-attn 2.7.4.post1 (`flash_attn_func`, causal=True for FA/GQA), via `golden_ladder`.
- Metric = geomean of (Arke latency / flash-attn latency) across the op's tier-1 shape set, kernel-only CUDA-events median (`benchmarks/measure.bench_fn`).
- RTX 3060 Laptop sm_86, fp16. Same-day A/B only (laptop eager baseline drifts 2–4× cross-day — never compare to historical PERF_ALL).

---

## 1. Baseline (honest starting point, 2026-07-27 clean rerun)

| op | Arke Triton vs flash-attn golden | gap |
|:--|--:|:--|
| flash_attention | **0.301×** | 3.3× slower |
| grouped_query_attention | **0.172×** | 5.8× slower |
| cross_attention (post K-XATT X1) | ~0.5–0.6× (non-causal) | 1.7–2× slower |

Current template `arke/backend/triton_templates/flash_attention.py.j2` (161 LOC) already has:
online softmax (`m_i`/`l_i`/`o_acc` running update), `tl.dot(q, kᵀ)` + `tl.dot(p, v)`,
causal mask, `BLOCK_N=64` fixed, single-buffered K/V loop, no `num_stages` pipelining.

**So K-ATT is an OPTIMIZATION of an existing correct kernel, not a greenfield build.**
The 0.301 baseline is correct but naive: no software pipeline, fixed tiles, no TC dtype control.

CUDA-C v8 (already production, `arke/backend/cuda_c_attention.py`) reached 0.35–0.42× on the
same hardware. Its winning lever — documented in `docs/phase5/c2-tensorcore-attention-2026-07-15.md §9/§11` —
was **deepening latency hiding (3-stage cp.async pipeline) over raw occupancy**. We port that
insight into Triton (`num_stages` + K/V double-buffer), not the CUDA code.

---

## 2. Task decomposition (each vN = one commit + push + geomean report)

### Phase FA (drive flash_attention 0.301 → ≥0.35 stage → ≥0.50 final)

- **FA-v1 — pipeline + tile sweep (highest lever).**
  Add `num_stages` (2/3/4) + `num_warps` (4/8) to the launch; sweep `BLOCK_N`∈{32,64,128}
  and `BLOCK_S`(K/V tile)∈{32,64}. Triton's `num_stages` gives us the cp.async software
  pipeline for free (the CUDA-C v8 lever). Expect the biggest single jump here.
  *Acceptance:* geomean improves AND correctness ≤5e-3; pick the Pareto-best config.

- **FA-v2 — TC dtype discipline.**
  Ensure `tl.dot` runs on Tensor Cores in fp16→fp32: `tl.dot(q, tl.trans(k), out_dtype=tl.float32)`,
  keep `p` in fp16 for the second dot (`tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)`).
  Verify via PTX/SASS dump that `mma.sync`/`wgmma` (or `hmma` on sm_86) is emitted, not fma.
  *Pitfall (from K-H3.1 lore):* fp32 sweep numbers mislead — always bench at real fp16 dtype.

- **FA-v3 — bucketed launch-config memo (K-H3.1 pattern).**
  Reuse the `_TILE_CFG_CACHE` + `next_pow2` bucket pattern from `matmul.py.j2` so dynamic
  S doesn't re-select config every call (ties into K-DYN cliff data: attention is where
  dynamic shapes hurt most). Config keyed on `(next_pow2(S), D, causal)`.

- **FA-v4 — mask/softmax micro-opts (if still short of 0.50).**
  Skip fully-masked K/V tiles under causal (early-exit `kv_len` bound is already there —
  verify it prunes); fuse the `1/l_i` normalize into the store; minimize `p` shared-memory
  round-trips. Only pursue if v1–v3 land between 0.35 and 0.50.

**FA exit:** stage ≥0.35 unlocks GQA start; final ≥0.50 closes FA.

### Phase GQA (after FA ≥0.35; drive grouped_query_attention 0.172 → its own thresholds)

- **GQA-v1 — native GQA (no K/V expansion).**
  Current impl expands K/V from Hkv→Hq (8× memory waste on llama-70b Hq=64/Hkv=8).
  Restructure the grid so each Hkv group's K/V tile is loaded once and reused across the
  Hq/Hkv query heads that map to it (Q-outer, K/V-inner-per-group). This is the root-cause
  fix, not a tweak.
  *Acceptance:* memory traffic drops ~Hq/Hkv×; geomean jumps off the 0.172 floor.

- **GQA-v2 — inherit FA's pipeline + TC + bucket config.**
  Fold FA-v1..v3 wins (num_stages, TC dtype, bucketed config) into the GQA kernel.

- **GQA thresholds (set separately per C2, proposed — needs Leon confirm at GQA start):**
  stage ≥0.30 / final ≥0.45. GQA's memory-reuse structure has a different ceiling than FA;
  I'll bring same-day baseline data to Leon before locking GQA numbers (frozen layer).

---

## 3. First session kickoff steps (when K-ATT session opens)

1. Same-day baseline: `run_op("flash_attention", tier=1)` + `run_op("grouped_query_attention", tier=1)`
   to re-anchor 0.301/0.172 on today's driver/clock (laptop drift discipline).
2. FA-v1: add num_stages/num_warps/tile sweep to `flash_attention.py.j2`; bench each config
   same-day A/B; commit the Pareto-best.
3. PTX/SASS dump on the winning config to confirm TC `mma` emission (feeds FA-v2).
4. Each vN: `commit + push + report geomean + commit id` (Leon authorization principle).
   Correctness ≤5e-3 checked every vN — a fast-but-wrong kernel is a fail, reported honestly.

## 4. Session isolation (AGENTS.md「一个 session 一件大事」)

K-ATT produces heavy output (Triton IR/PTX dumps, ncu/profiler, 3-way bench tables, multi-round
iteration). It gets its **own session**. This plan doc is the durable contract; the K-ATT session
starts by reading this file, not by re-deriving the decision. Write state back to INBOX.md +
daily memory when the window fills or a phase closes.

## 5. AI-Native tie-in (why pure Triton, not CUDA-C bridge)

The thesis is: **LLM decides via StrategyIR → Arke compiles to multiple backends**. If the Triton
backend can only reach attention parity by bridging to hand-written CUDA-C, the Triton path isn't
independently viable and the multi-backend AI-Native claim weakens. K-ATT proves the Agent-facing
compiler stack can drive a Triton flash kernel to competitive performance — the `num_stages` /
tile / TC-dtype choices are exactly the kind of bounded StrategyIR actions an Agent should own.

---

*Locked 2026-07-29 on Leon's `A1 B1 C2`. Kernel iteration → dedicated session.*
