# K-ATT Plan — Triton flash-style attention to ≥0.35 (stage) / ≥0.5 (final)

**Approved decision (Leon 2026-07-29): `A1 B1 C2`**
**Status:** 🎯 **BOTH GATES PASSED (2026-07-29, same day)** — FA geomean **0.846** (stage 0.35 ✅ / final 0.50 ✅), GQA geomean **0.802**. See §6 execution log.
**Owner:** Kitty (full implementation authority; only Gate thresholds are frozen — locked below)
**Backlog:** `docs/audit/kestrel-backlog.md` K-ATT (P1, 性能主线, 2-4w)
**Decision material:** `docs/kestrel/k-att-decision-material.md`

---

## 0. Locked frozen-layer parameters (Leon-approved, do NOT change without Leon)

| Param | Value | Source |
|:--|:--|:--|
| **A1 — Stage Gate** | Triton FA vs flash-attn golden **geomean ≥ 0.35** | = same-hardware CUDA-C v8 lower bound (0.35–0.42×); "Triton is no longer the short pole" |
| **A1 — Final Gate** | Triton FA vs flash-attn golden **geomean ≥ 0.50** | 1.4× of CUDA-C v8; Triton becomes an independently strong backend |
| **GQA — Stage Gate** | Triton GQA vs flash-attn golden **geomean ≥ 0.30** | **LOCKED (Leon 2026-07-30, "E OK")**; measured 0.863 at lock time (attention_refresh_2026-07-30) |
| **GQA — Final Gate** | Triton GQA vs flash-attn golden **geomean ≥ 0.45** | **LOCKED (Leon 2026-07-30)**; conservative sustainable bar — GQA's K/V-reuse structure has a different ceiling than FA; large cross-run drift margin over 0.863 actual |
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

- **GQA thresholds — LOCKED (Leon 2026-07-30 "E OK"):**
  stage ≥0.30 / final ≥0.45 (see §0 locked table). Measured 0.863 at lock time —
  both gates PASS. GQA's memory-reuse structure has a different ceiling than FA.

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

## 6. Execution log (2026-07-29 — plan executed same day, both gates passed)

| rev | commit | change | FA geomean | notes |
|:--|:--|:--|--:|:--|
| baseline | `1732532` | golden swap anchor | 0.301 | FlagGems→flash-attn denominator |
| FA-v1 | `3a20916` | bucketed tile heuristic from 8×42 fp16 sweep (`_cfg_override` seam + `_fa_cfg`/`_FA_CFG_CACHE`) | 0.496 | D=128: BLOCK_S=32 decisive (2.0-2.7×); num_stages>2 never helped on sm_86; stage gate 0.35 ✅ |
| FA-v2 | `647996f` | **TC dtype discipline** — drop `.to(tl.float32)` loads, `tl.dot(fp16,fp16,out_dtype=fp32)` | **0.846** | THE lever: fp32 loads had forced FFMA (TC idle). err improved 2e-3→4.9e-4. Final gate 0.50 ✅ |
| GQA-v1 | `336d29b` | runtime `gqa_groups = H//Hkv` (launcher bug: render-time constexpr 4 was wrong for qwen 28/4, ds-mha 128/128 → garbage err≈5.0) | GQA **0.802** | kernel was already group-native; correctness 8/8 ≤9.8e-4 incl. previously-broken shapes |

**Cross-attention side-effect (no extra work):** FA-v2 TC discipline lifted
cross_attention to geomean **1.081** vs flash-attn (llava-vision 1.25,
batch4 1.08, sdxl 1.03, t5 0.98) — Arke Triton now *beats* fused flash-attn
on non-causal Sq≠Skv shapes.

**Honest caveats:**
1. GQA's 0.172 baseline partly measured *incorrect* kernels (the gqa_groups=4
   bug) — the honest like-for-like is llama3-class (correct before and after),
   still a real ~4× perf win from FA-v1+v2 inheritance.
2. All numbers are same-day A/B on RTX 3060 Laptop (laptop clock drift
   discipline); ds-v3-163k class shapes not measured (6GB VRAM).
3. Remaining FA gap concentrates in D=64 short-S (gpt2-sm-512 0.66) and
   llama2-7b-2k (0.70) — candidate FA-v4 micro-opts if a later phase needs
   them; NOT pursued now (gate passed, diminishing returns).

**Measurement-honesty incident (recorded per Leon's principle):** the first
quick sweep fabricated a 6.4× "speedup" on gpt2-sm-1k because the default
config was timed first and absorbed GPU clock spin-up. Fixed with per-shape
warmup before any timed config; all distilled configs come from the fixed
harness. Phantom numbers never entered the heuristic or any commit.

*Locked 2026-07-29 on Leon's `A1 B1 C2`. Executed and gates passed same day.*
