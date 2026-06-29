# RFC — L2 Fusion Measurement Protocol (S7.followup.2)

> **Status:** 📋 PROPOSAL — awaiting project-lead review before any Gate-scoring change.
> **Author:** Kitty (lead engineer) · 2026-06-29
> **Scope:** Close S7.followup.2 — "L2 Triton fusion baseline collection (currently 0 evaluable)".
> **Governance:** This RFC proposes (a) **reversible implementation** work I can land now
> (wiring a Triton-only fused denominator) and (b) **one frozen-layer decision** that needs
> Leon's explicit approval before merge (how an L2 row becomes *evaluable* vs *audit-only*).
> The two are separated below so the reversible part can proceed independently.

---

## 1. Problem statement

`docs/roadmap/plan.md` S7.followup.2 is open: **L2 fusion performance has 0 evaluable rows.**
G7 sub-gate **[11]** requires *"4/4 fusion combinations pass under Same-Backend Triton Fairness"*,
and `benchmarks/gate_g7.py:701-709` enforces, per fused op:

```python
# each fused op must have total > 0 AND passed == total
if counts["total"] == 0 or counts["passed"] != counts["total"]: FAIL
if not l2_counts: FAIL  # "L2: no evaluable fusion performance rows"
```

So today L2 is structurally un-passable: there are **no evaluable rows at all**.

### Root cause (diagnosed 2026-06-29)

The L2 harness is *built* — `benchmarks/bench_l2.py` has fused ops, shapes, correctness
checks, and baselines. The blocker is the **denominator** under the locked
**Same-Backend Triton Fairness** rule (`plan.md` §"Same-Backend Fairness — Triton Path"):

> The Gate performance comparison **denominator is the corresponding operator's Triton
> implementation**, not a cross-backend reference. Audit-only when no Triton-only impl exists.

The current L2 baselines do **not** satisfy "Triton-only":

| L2 baseline today | What it actually dispatches | Triton-only? |
|:--|:--|:--:|
| FlagGems `act_fn(torch.matmul(A,B))` (`bench_l2.py:249`) | `torch.matmul` → **cuBLAS** (ATen), act → Triton | ❌ mixed |
| `torch.compile` auto-fusion | **Inductor**-generated, not a hand-Triton golden | ❌ |
| manual separate ops | eager | ❌ |

Because no baseline is a clean Triton-only **fused** kernel, every L2 row is correctly
classified **audit-only** → 0 evaluable → gate cannot score. This is the harness being
*honest*, not broken: it refuses to score Arke-Triton against a non-Triton denominator.

---

## 2. Key insight — the golden ladder already names the fix

`docs/benchmark/golden-kernel-ladder.md` already designates **Liger-Kernel (P1)** as the
golden **Triton** fused kernel for the OT3 fused family:

| Fused op | Golden (Triton) | Fallback |
|:--|:--|:--|
| `silu_and_mul` | **Liger (P1)** | PyTorch-eager |
| `gelu_and_mul` | **Liger (P1)** | PyTorch-eager |
| `fused_linear_cross_entropy` | **Liger (P1)** | PyTorch-eager |
| `cross_entropy` | **Liger (P1)** | FlagGems / PyTorch |

Liger ships *genuine Triton fused kernels* for these (`LigerSiLUMulFunction`,
`LigerGELUMulFunction`, `LigerFusedLinearCrossEntropyFunction`, `LigerCrossEntropyFunction`).
**The ladder protocol is already correct; the L2 runner just never wired Liger's Triton
fused kernels as the L2 denominator.** That is the implementation gap.

For the two pure-GEMM-epilogue fusions (`matmul_relu`, `matmul_gelu`) there is **no
Triton-only fused golden** in any of the 9 audited libraries (FlagGems/Liger/vLLM/… expose
GEMM via cuBLAS-dispatch, not a hand-Triton fused-epilogue kernel). Under the locked rule
these are legitimately **audit-only** — same as `transpose` / `multi_latent_attention`.

---

## 3. Proposed L2 fusion set (maps to G7[11] "4/4")

G7[11] says "4/4 fusion combinations". The L2 op list (`bench_l2.py:38-43`) is
`matmul_relu, matmul_gelu, silu_and_mul, gelu_and_mul, linear_ce, qkv_fa`. Proposal: the
**4 scored** fusions are the ones with a real Triton-only golden (Liger), the rest audit-only:

| # | Fusion | Triton golden (denominator) | Gate role |
|:--:|:--|:--|:--|
| 1 | `silu_and_mul` | Liger `LigerSiLUMulFunction` | **scored** |
| 2 | `gelu_and_mul` | Liger `LigerGELUMulFunction` | **scored** |
| 3 | `linear_ce` (`fused_linear_cross_entropy`) | Liger `LigerFusedLinearCrossEntropyFunction` | **scored** |
| 4 | `cross_entropy` | Liger `LigerCrossEntropyFunction` | **scored** |
| — | `matmul_relu`, `matmul_gelu` | none (no Triton fused-GEMM golden) | audit-only |
| — | `qkv_fa` | OT4 attention — covered by S7.followup.3 (FlagGems P1) | (already landed) |

→ "4/4" = the four Liger-backed fused ops. This is the **decision that needs Leon's
approval** (§5), because it fixes which fusions count toward the Gate.

---

## 4. Reversible implementation (I can land now, no Gate-scoring change)

These steps add a **Triton-only fused denominator** to the L2 harness without altering any
Gate pass/fail definition. They only make rows *measurable*; whether a measured row is
*scored* stays governed by the existing (frozen) audit-only logic until §5 is approved.

1. **Add `benchmarks/baselines/liger_fused.py`** — a thin runner exposing Liger's Triton
   fused kernels for `silu_and_mul`, `gelu_and_mul`, `fused_linear_cross_entropy`,
   `cross_entropy`, tagged `source="Liger <ver> Triton fused | <url> | Apache-2.0"`,
   `backend="triton"`. Guards: import-guard (skip if Liger absent), `GEMS_VENDOR` untouched.
2. **Wire it into `bench_l2.py`** as the per-fusion Triton golden, replacing the
   ATen-dispatch `act_fn(torch.matmul(...))` denominator for the four scored fusions.
3. **Correctness first** — every Liger fused output is checked against the SemanticInterpreter
   reference (the same V1 oracle), bit-for-bit within (rtol,atol), before any perf number is
   trusted. (Lesson: LIVE≠correct.)
4. **Emit PERF_ALL L2 rows** with `golden_runner="liger"`, `golden_priority=1`,
   `backend="triton"` so the existing scoring path can see a Triton-only denominator.
5. **`matmul_relu`/`matmul_gelu`** explicitly recorded `perf_oracle_unavailable_triton=true`
   (audit-only), mirroring the locked `transpose` precedent — **no op removed, no shape
   removed** (Benchmark-design-frozen invariant preserved).

All of the above is reversible engineering: it adds a baseline runner and populates rows.
It does **not** touch `gate_g7.py` scoring thresholds or the audit-only classifier.

---

## 5. Frozen-layer decision — needs Leon's explicit approval

The single thing I will **not** do without your sign-off (it changes Gate semantics):

> **Decision D-L2:** Accept Liger's Triton fused kernels as the **scored** Same-Backend
> denominator for the 4 OT3 fusions, making those L2 rows *evaluable* (counted toward
> G7[11] pass/fail), with `matmul_relu`/`matmul_gelu` remaining audit-only.

Why this is a Gate-semantics change (hence frozen-layer): it determines which fused ops
**count** toward "4/4 fusion combinations pass", i.e. it sets the L2 evaluable set. Per Gate
Governance, "every Gate threshold adjustment must carry the project lead's explicit approval
before merging." This is the adjustment.

**Sub-options for D-L2 (you pick):**

- **D-L2-a (recommended):** 4 scored = the 4 Liger fusions; matmul_relu/gelu audit-only.
  Cleanest, fully honest, uses the already-locked golden ladder.
- **D-L2-b:** also require a Triton fused-GEMM golden for matmul_relu/gelu before passing L2
  (would need me to *write* an Arke-independent reference Triton fused-GEMM — large, and
  arguably circular since Arke would be compared to a kernel I'd author). Not recommended.
- **D-L2-c:** keep L2 audit-only entirely (don't score fusion perf at G7) — but G7[11]
  explicitly requires 4/4, so this effectively requires re-wording G7[11]. Needs your call.

I recommend **D-L2-a**. It honors the frozen benchmark (no shape/op removed), uses the
existing golden ladder verbatim, and makes G7[11] honestly achievable on real Triton-vs-Triton
data instead of being structurally un-passable.

---

## 6. Acceptance (when this RFC is "done")

- `liger_fused.py` runner lands; L2 PERF_ALL has Triton-only denominator rows for the 4 fusions.
- Each fused op: correctness vs SemanticInterpreter = 100% before perf is reported.
- `matmul_relu`/`matmul_gelu` recorded audit-only (no removal).
- **After D-L2 approval only:** `gate_g7.py` L2 path scores the 4 fusions; re-run G7 and
  report the real `passed/total` per fusion + weighted impact, with commit id.
- Daily memory + `plan.md` S7.followup.2 status updated to reflect landed state.

---

## 7. What I will NOT do

- Not remove or alter any benchmark shape, op, or the PERF_ALL schema (benchmark-design frozen).
- Not change `gate_g7.py` scoring thresholds or the audit-only classifier until D-L2 is approved.
- Not author a "reference" Triton kernel that Arke is then compared against (circularity).
- Not touch Phase 2 (Ascend) — paused/skipped per your 2026-06-29 instruction.

---

*RFC v1 — 2026-06-29. The Triton-only-denominator wiring (§4) is reversible and I will land it
now; the evaluable-set decision (§5, D-L2) waits for your pick before any Gate-scoring merge.*
