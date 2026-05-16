# Stage 7 Completion Summary — Gate Governance Upgrade + Honest Distance Disclosure

**Date:** 2026-05-16
**Branch:** `feat/s7-gate-revision-2026-05-16`
**Commits:** `c366116` (docs) → `3c89a23` (scoring) → this summary
**Gate result:** **G7 = 13/14 PASS**, G7.8d single honest fail
**Status:** Stage 7 closed in **functional-spec-grade** at 13/14; perf gap to ladder-fastest Triton kernels disclosed as Phase-1-internal follow-up work (does not block Stage 8 entry per Gate Governance v2).

---

## 1. What Stage 7 Was Supposed To Deliver

Per `docs/roadmap/plan.md` Stage 7, the locked exit criteria for G7 are:

```
AND ALL:
  [G7.1]  Arke Lang Spec v0.1.0 finalized
  [G7.2]  Arke IR Spec v0.1.0 finalized (semantic + strategy + instruction layer)
  [G7.3]  Lang→SemanticIR parser + .ak round-trip
  [G7.4]  SemanticIR → StrategyIR full pipeline (≥45 ops, all OT tiers)
  [G7.5]  MLIR skeleton checkpoint (Phase-3 forward compatibility)
  [G7.6]  All 45 BL5 ops + 6 L2 fusions compile end-to-end
  [G7.7]  Examples library complete (.ak coverage 100%)
  [G7.8 / G7.8a]  Benchmark artifacts contract + dashboard
  [G7.8b]  BL5 L1/L2 coverage evidence complete
  [G7.8c]  BL5 correctness rows pass (modulo memory-policy exclusions)
  [G7.8d]  BL5 L1 weighted performance ≥ 0.95 + L2 fusion performance
  [G7.9]  Backend-agnostic StrategyIR core (zero Triton-specific fields)
  [G7.10] Non-regression suite stays green for the active S7 slice
```

This summary records that 13 of 14 criteria PASS; only G7.8d (performance) holds at honest distance under the newly-locked **Same-Backend Triton Fairness** ruler.

---

## 2. Gate Governance v2 — What Changed and Why

Leon's 2026-05-16 governance directive reset two foundational rules:

### Rule A — Same-Backend Fairness (locked)

> Arke→Triton kernels can only be benchmarked against **other Triton kernels** (`min(latency_us)` over Triton-only baselines per `(layer, op, shape_tag)`). Cross-backend comparisons against PyTorch-eager / cuBLAS / cuDNN / torch.compile are **audit-only** — they remain in the artifact but do not enter Gate scoring.

**Rationale:** Phase 1's backend is Triton. The fair scientific comparator is the best Triton implementation we can find in the community ladder (FlagGems, Liger-Kernel, Triton-Tutorial, Unsloth, flash-attn-triton, vllm-triton, FlashInfer-Triton). Beating PyTorch-eager is necessary but trivial; matching the best Triton is the real signal.

### Rule B — Benchmark Frozen, Gate Adjustable

> Benchmark measurement layer (`benchmarks/artifacts.py`) is **frozen** — `perf_pass` still writes against PyTorch-eager for backward-compat. Gate scoring layer (`benchmarks/gate_g7.py`) **re-scores** at gate-time per the locked ruler. Threshold tunes (epsilon, per-group floor, weights) live in the gate, not the artifact writer.

This decoupling means benchmark history stays comparable across gate revisions, and gate ruler changes don't invalidate past PERF_ALL.csv files.

### Rule C — Phase Restructure

Roadmap Phase 4 (was: LLVM IR) is renamed to **Phase 4: C-like Kernel Language** (CUDA C / CCE-C / Bang-C — direct hardware-vendor toolchain integration). Phase 5 = LLVM IR (true full-stack control). This makes the multi-hardware story credible: MLIR (Phase 3) gives compiler IR fluency; C-like kernels (Phase 4) give us a hatch into vendor toolchains where MLIR coverage is incomplete; LLVM IR (Phase 5) is the long-tail completeness target.

---

## 3. Scoring Implementation (commit `3c89a23`)

### Locked constants

```python
_TRITON_ONLY_BASELINES = frozenset({
    "flaggems", "liger-kernel", "triton-tutorial", "unsloth",
    "flash-attn-triton", "vllm-triton", "flashinfer-triton",
})
_PERF_EPSILON = 0.03           # tolerance band: arke_lat ≤ triton_ref_lat * 1.03
_PER_GROUP_FLOOR = 0.97        # any single OT group below this → fail
_WEIGHTED_THRESHOLD = 0.95     # OT0_1=0.25, OT2=0.30, OT3=0.20, OT4=0.25
```

### Scoring rule (per Arke row)

```
ref_lat = min(latency_us  for r in same-(layer,op,shape_tag) where r.baseline ∈ _TRITON_ONLY_BASELINES)
if ref_lat is None:
    → perf_oracle_unavailable_triton++ (audit-only, not scored)
else:
    passed = arke_lat ≤ ref_lat * (1 + 0.03)
```

### Audit-only categories (none affect Gate score)

- `non_arke_baseline_skipped` — reference rows themselves
- `memory_excluded` — OOM at 6 GB VRAM
- `perf_oracle_unavailable` — priority-1 baseline crashed, correctness verified via fallback
- `perf_oracle_unavailable_triton` — no Triton-only baseline at this (op, shape)
- legacy rows with empty `latency_us` — counted into `perf_oracle_unavailable`

---

## 4. Real-Data G7 --tier 2 Result

Run on Track 6 canonical PERF_ALL.csv (RTX 3060 Laptop 6GB · CUDA 12.4 · PyTorch 2.6.0 · Triton 3.2.0):

| Criterion | Result | Detail |
|---|---|---|
| G7.1 Lang Spec | ✅ PASS | v0.1.0 finalized |
| G7.2 IR Spec | ✅ PASS | semantic + strategy + instruction layer |
| G7.3 Lang→Semantic | ✅ PASS | .ak round-trip green |
| G7.4 Semantic→Strategy | ✅ PASS | 45 ops compile |
| G7.5 MLIR skeleton | ✅ PASS | Phase-3 forward compat |
| G7.6 Correctness pipeline | ✅ PASS | 46 .ak dry-run pass |
| G7.7 Examples | ✅ PASS | full coverage |
| G7.8 Artifacts | ✅ PASS | dashboard + per-layer manifests |
| G7.8a Artifact contract | ✅ PASS | result-tree shape valid |
| G7.8b Coverage | ✅ PASS | 45/45 ops, 685/685 shapes |
| G7.8c Correctness | ✅ PASS | 863 ok / 23 memory_excluded / 3 golden_exempted / 18 typed_unsupported / 1737 non_arke_baseline_skipped |
| **G7.8d Performance** | **❌ FAIL** | **see below** |
| G7.9 Backend-agnostic | ✅ PASS | 0 Triton-specific fields in StrategyIR core |
| G7.10 Non-regression | ✅ PASS | 563 tests green |

### G7.8d failure breakdown

```
L1 weighted_score = 0.3000  (threshold 0.95)
  · OT0_1  elementwise+reduction      : 190/351 = 54.1%   (floor 0.97 violated)
  · OT2    compute-dense              :  19/62  = 30.6%   (floor 0.97 violated)
  · OT3    gated activation / rope    :   4/11  = 36.4%   (floor 0.97 violated)
  · OT4    attention                  :   0/0            (no Triton ref data)
L2 fusions evaluable = 0              (no L2 Triton baseline collected yet)
perf_oracle_unavailable_triton = 438  (audit-only)
non_arke_baseline_skipped      = 1737 (audit-only)
memory_excluded                = 44   (audit-only)
```

The 24 L1 ops that **do** have Triton-only baseline coverage today:
`add, batch_matmul, cast, cumsum, embedding, exp, gelu, layernorm, matmul, mul, neg, reduce_max, reduce_mean, reduce_sum, relu, rmsnorm, rope, rsqrt, sigmoid, silu, softmax, tanh, transpose, where_`.

---

## 5. Honest Gap Analysis — Why Path 1 Was The Right Call

We considered three paths on 2026-05-16:

- **Path 1 (chosen):** Commit the honest ruler, accept 13/14, open follow-up subtasks for the perf gap.
- Path 2: Block Stage 7 close on shipping (a) Triton-template autotune across remaining ~20 templates, (b) L2 Triton fusion baseline collection. ETA 1–2 days.
- Path 3: Loosen ε / threshold until current data passes. **Rejected** — violates honest-measurement principle.

Path 1 is correct because:

1. **Stage 7 is functionally a Lang/IR/spec stage, not a perf stage.** Per Gate Governance v2, "Stage 7 core positioning is lang spec + high-order IR spec basic implementation (functional-orientation, perf strong-target not required)". The locked G7.8d perf threshold remains a useful North Star but does not retroactively rewrite Stage 7's design intent.
2. **The 30%/0.97-floor gap is the real engineering signal.** It tells us exactly where Arke's generated Triton code is slower than ladder-best Triton: dense compute (OT2) and gated activations (OT3) need Triton-template autotune; attention (OT4) needs Triton baseline collection before we can even score. Subtasks (a) autotune rollout, (b) L2 Triton baseline collection are well-defined and time-bounded. They become the first work items of the Stage 7→Stage 8 transition slice.
3. **The benchmark measurement layer is unchanged.** All historical PERF_ALL.csv files remain re-scorable under either old or new ruler. We didn't burn data.
4. **Gate Governance v2 explicitly allows "perf strong-target not required" for Stage 7.** Future Stages re-inherit BL5 perf via the Stage-N-inherits-BL5 chain (S8: "BL5 inherited (no regression)"). The honest 13/14 closure protects that inheritance — Stage 8's "no regression" check now starts from a real number, not a fabricated pass.

---

## 6. Stage 7 → Stage 8 Transition: Open Follow-Up Subtasks

These are **not blocking** Stage 7 close, but **must** be completed before Stage 8's BL5-inherit gate becomes meaningful:

### Subtask S7.followup.1 — Triton-template autotune rollout
- **Current state:** Only `matmul.j2` and `batch_matmul.j2` carry `@triton.autotune`. 20 other Jinja2 templates have hard-coded `BLOCK_SIZE` / `num_warps` / `num_stages`.
- **Target:** Add autotune configs to all OT2/OT3 templates (priority order: ones currently failing G7.8d).
- **Acceptance:** OT2 and OT3 pass rates climb from 30.6%/36.4% toward the 0.97 floor.

### Subtask S7.followup.2 — L2 Triton fusion baseline collection
- **Current state:** Zero Triton-side fusion baselines exist for L2 ops (`matmul_relu`, `matmul_gelu`, `swiglu`, `geglu`, `linear_ce`, `qkv_fa`). All L2 Arke rows are audit-only under the new ruler.
- **Target:** Author L2 Triton fusion baseline kernels (Liger / FlagGems / hand-written) for the 6 L2 ops.
- **Acceptance:** L2 fusions evaluable ≥ 6, all passing under ε=0.03.

### Subtask S7.followup.3 — OT4 attention Triton ref
- **Current state:** OT4 has 0/0 — no Triton attention baseline data in PERF_ALL.
- **Target:** Add flash-attn-triton / FlashInfer-Triton runs for the 5 OT4 ops at attention-relevant shapes.
- **Acceptance:** OT4 group becomes evaluable; rate ≥ 0.97 (or honest distance disclosed).

---

## 7. Artifacts Touched

| File | Change | Commit |
|---|---|---|
| `docs/roadmap/plan.md` | Gate Governance section + Stage 7 PASS criteria rewrite + Phase 4 C-like / Phase 5 LLVM IR | `c366116` |
| `docs/phase1/stage7-plan.md` | Benchmark Requirements L1/L2 + G7 weighted_score formula (same-backend Triton) | `c366116` |
| `docs/benchmark/benchmark-protocol.md` | Design Goal Same-Backend Fairness clause + ε=0.03 | `c366116` |
| `benchmarks/gate_g7.py` | `_TRITON_ONLY_BASELINES` / `_PERF_EPSILON` / `_build_triton_ref_index` / `_check_bl5_performance_evidence` rewrite + legacy-row oracle-gap handling | `3c89a23` |
| `tests/test_gate_g7.py` | Updated 2 existing tests to new fixture schema + 2 new tests (audit_only_when_no_triton_ref, epsilon_boundary) | `3c89a23` |
| `docs/roadmap/plan.md` | Stage Summary G7 row + evidence-status paragraph updated to honest-gap status | this summary |
| `docs/phase1/stage7-completion-summary.md` | This file | this summary |

Tests: `tests/test_gate_g7.py` 24/24 PASS. Full repo suite 1553 passed / 1 pre-existing GPU-contamination fail (unchanged by this work).

---

## 8. Verification Command

```bash
cd /home/blueyi/workspace/repos/arke
source ~/.venvs/arke/bin/activate
python -m benchmarks.gate G7 --tier 2
```

Expected: `❌ Gate G7 FAILED (1 criteria)` with G7.8d showing `L1 weighted_score=0.3000 (Same-Backend Triton, eps=0.03)` and the per-OT breakdown above.

---

*Closing note: Stage 7 ships its **spec deliverables** (Lang v0.1.0, IR v0.1.0, MLIR skeleton, 45-op pipeline) and its **honest perf ruler**. The remaining work is **engineering, not design** — autotune configs and Triton baseline collection, both well-scoped and gated.*
