> ⚠️ **HISTORICAL** — this plan is a point-in-time working document; status and numbers may be superseded. See docs/roadmap/plan.md for current SSOT.

# G7.8b + G7.8d Closure Plan

> **For Hermes / Kitty:** Use `subagent-driven-development` only for the
> mechanical sub-tasks below. The perf-investigation tasks (P1, P3 root-cause)
> require interactive reasoning and must stay in the main session.

**Goal:** Close the remaining three G7 evidence criteria — `G7.8b` (full-shape
coverage), `G7.8c` real failures (10 remaining post-1a), `G7.8d` (L1 weighted
perf ≥ 0.95, L2 fusions 102/102 each) — so Gate G7 reaches 14/14 with
canonical Track 6 evidence.

**Architecture:** three independent phases (P1 coverage, P2 correctness
fixups, P3 perf root-cause + ladder upgrades) that can be done in parallel.
Closure is gated by data, not by code-shipped; every phase ends with a fresh
`python -m benchmarks.gate G7 --tier 2` and a quantitative delta in the daily
memory log.

**Tech stack:** Arke 0.1.0 · Triton 3.x · PyTorch 2.6 · FlagGems · Liger ·
Stage 7 Track 6 benchmark harness.

---

## Reconnaissance summary (2026-05-14)

Initial reconnaissance after commit `d74beea` (1a typed-unsupported fix):

| Criterion | Status   | Quantitative gap                                         |
|-----------|----------|----------------------------------------------------------|
| G7.8b     | ❌ FAIL  | 10 ops × 60 missing shape rows                           |
| G7.8c     | ❌ FAIL  | 10 real failures (post-1a; was 46)                       |
| G7.8d     | ❌ FAIL  | L1 weighted=0.6921 (req ≥0.95); L2 matmul_gelu=39/102, matmul_relu=48/102 |

**Update 2026-05-14 14:30** — after Q5a (rope fp32) + Q6a (gated odd-N typed-unsupported):

| Criterion | Status   | Quantitative gap                                         |
|-----------|----------|----------------------------------------------------------|
| G7.8b     | ❌ FAIL  | 10 ops × 60 missing shape rows (unchanged)               |
| G7.8c     | ✅ PASS  | correctness=827/827 ok; memory_excluded=22, golden_exempted=3, typed_unsupported=18 |
| G7.8d     | ❌ FAIL  | L1 weighted=0.6866 (req ≥0.95); L2 matmul_gelu=39/102, matmul_relu=48/102; **3 malformed/non-ok perf rows** (gelu:extreme-flat status=error, topk:extreme-wide status=error, topk:extreme-wide perf_pass=<empty>) |

Gate G7 overall: **11/14 → 12/14** (78.6% → 85.7%). Remaining: G7.8b (P1) + G7.8d (P3).
Q5a commit `82b635b`, Q6a commit `160ebf4` (both pushed).

### G7.8b — 10 ops × 60 missing L1 shape rows

All missing shapes are large-model / long-context tags (DeepSeek-V2/V3,
Qwen2.5-7B, Llama3-8B, ds-v2-long, wide-vocab, …):

| op                          | OT | missing | required | missing tags (truncated)                                                                                                |
|-----------------------------|----|---------|----------|-------------------------------------------------------------------------------------------------------------------------|
| `matmul`                    | 2  | 12 / 34 | ds-v2-attn, ds-v2-ffn-down/up, ds-v2-lmhead, ds-v2-long-8k, ds-v3-attn/ffn-up/lmhead/long-32k, qwen25-attn/ffn-up/lmhead |
| `softmax`                   | 1  | 7 / 25  | ds-v2-attn-{8k,16k}, ds-v3-attn-32k, llama3-attn-8k, qwen25-attn-32k, wide-vocab-{ds-v2,qwen25}                          |
| `geglu`                     | 3  | 6 / 12  | ds-v2-{512,2k}, ds-v3-{512,2k}, qwen25-7b, qwen25-7b-2k                                                                  |
| `layernorm`                 | 1  | 6 / 21  | ds-v2-long, ds-v3-long, llama3-8b-{long,norm}, qwen25-7b-{long,norm}                                                     |
| `rmsnorm`                   | 1  | 6 / 21  | (same as layernorm)                                                                                                     |
| `rmsnorm_residual`          | 1  | 6 / 21  | (same)                                                                                                                  |
| `swiglu`                    | 3  | 6 / 12  | (same as geglu)                                                                                                         |
| `fused_linear_cross_entropy`| 3  | 4 / 12  | ds-v2-seq512, llama3-seq2k, qwen25-seq{512,2k}                                                                           |
| `rope`                      | 3  | 4 / 16  | ds-v2-512, ds-v3-2k, llama3-8k, qwen25-32k                                                                               |
| `grouped_matmul`            | 2  | 3 / 7   | ds-moe-{512,2k}, ds-v3-moe                                                                                               |

**Hypothesis:** most/all of these will OOM on the 6 GB GPU. After the a9
memory_policy upgrade (commit `77be321`), the harness should classify them
as `status=oom, memory_policy=<reason>` and the Gate would then count them
as `memory_excluded` rather than `missing`. They are MISSING from PERF_ALL
today because the harness never even attempted them.

→ **P1 is therefore primarily about running the missing shapes through the
new memory_policy preflight and letting them be recorded as either
`memory_excluded` (legit) or real perf rows (if they fit) — NOT about
making them all pass.**

### G7.8c — 10 real failures after 1a

| # | shape                                | failure mode                  | resolution                                    |
|---|--------------------------------------|-------------------------------|-----------------------------------------------|
| 1-3 | `flash_attention @ ds-v2-2k` (×3)  | `correctness=error`           | ✅ stale handle assert cleared (commit `31e7275`) |
| 4 | `gelu @ extreme-flat`                | `status=error`                | ⏭️ remains in G7.8d (perf row malformed) — Q6b/Q7 |
| 5 | `rope @ extreme-long`                | `correctness=mismatch`        | ✅ Q5a fp16→fp32 sin/cos (commit `82b635b`, 5 sites) |
| 6 | `topk @ extreme-wide`                | `status=error`                | ⏭️ remains in G7.8d (perf row malformed) — Q6b/Q7 |
| 7-8 | `geglu @ non-align-{1,2}` (L2)     | `status=error`                | ✅ Q6a typed-unsupported emit (commit `160ebf4`)   |
| 9-10 | `swiglu @ non-align-{1,2}` (L2)   | `status=error`                | ✅ Q6a typed-unsupported emit (commit `160ebf4`)   |

**Net G7.8c result:** correctness failures 10 → **0**, gate criterion now ✅ PASS. The
two `status=error` rows from gelu:extreme-flat / topk:extreme-wide migrate into G7.8d
as malformed perf rows; tracked under Q6b (triage) / Q7 (extreme-shape preflight scope).

### G7.8d — perf root cause

Arke vs PyTorch-eager `ratio_vs_baseline` p50 distribution per op (Arke
rows only, status=ok, L1):

| OT | op              | n  | p50    | p10    | min   | ≥1.0 count |
|----|-----------------|----|--------|--------|-------|------------|
| 0  | most elementwise| 20 | 0.97-1.00 | 0.85-0.95 | 0.55-0.92 | 7-13/20 |
| 1  | `layernorm`     | 15 | **0.18** | 0.15 | 0.13 | **0/15** |
| 1  | `rmsnorm`       | 15 | 0.70   | 0.61   | 0.61  | 0/15      |
| 1  | `rmsnorm_residual` | 15 | 0.80 | 0.65   | 0.57  | 0/15      |
| 1  | `softmax`       | 18 | 0.97   | 0.84   | 0.79  | 5/18      |
| 2  | `transpose`     | 6  | **0.01** | 0.00 | 0.003 | **0/6**  |
| 2  | `grouped_matmul`| 4  | **0.08** | 0.03 | 0.03  | **0/4**  |
| 2  | `batch_matmul`  | 9  | 0.73   | 0.02   | 0.02  | 0/9       |
| 2  | `matmul`        | 22 | 0.95   | 0.85   | 0.71  | 1/22      |
| 3  | `dequantize_per_channel` | 12 | 0.51 | 0.34 | 0.30 | 0/12  |
| 3  | `cross_entropy` | 13 | 0.68   | 0.67   | 0.56  | 0/13      |
| 3  | `geglu/swiglu`  | 6  | 0.85   | 0.81   | 0.81  | 0/6       |
| 4  | flash_attn etc. | —  | 5.9-11.4x | — | 1.4+  | all pass  |

**Two distinct perf problems:**

1. **Catastrophic ratio (≤0.20) → wrong kernel choice / launch overhead:**
   - `layernorm` 0.18 (5× slow) — Arke kernel doesn't use Welford / vectorized loads
   - `transpose` 0.01 (100× slow) — likely autotune compile per shape + tiny payload
   - `grouped_matmul` 0.08 — almost certainly falling back to a sequential loop
   - `batch_matmul` 0.02 worst — same family of issue

2. **Sub-target ratio (0.5-0.95) → kernel ladder primary is wrong:**
   - elementwise OT0 p50 ~0.98, but minimum 0.55-0.92 means some shapes
     hit a slow path
   - `dequantize_per_channel` 0.51, `cross_entropy` 0.68, `geglu/swiglu` 0.85,
     `rmsnorm` 0.70 — these are operators where FlagGems / Liger / custom
     Triton has known-faster implementations the ladder isn't using

`Arke` cannot reach OT0/1=0.95, OT2=0.95, OT3=0.95 weighted score
(currently 0.42, 0.36, 0.27 on Arke-only rows) without fixing both
problem classes.

---

## Phase P1 — G7.8b coverage closure via memory_policy

**Goal:** every (op, required shape) pair in the target matrix produces a row
in PERF_ALL with one of: `ok`, `oom` (memory_excluded), `unsupported` (typed).
No missing rows.

**Pre-condition:** commit `77be321` (a9 memory_policy) is on `main` ✅.

### P1.T1 — Audit which missing shapes the new memory_policy CAN exclude

**Objective:** dry-run the 60 missing (op, shape) pairs through
`benchmarks/preflight.py` (a9) to see how many would be classified
`oom / memory_excluded` vs `would-fit-but-not-yet-run`.

**Files (read-only):**
- `benchmarks/preflight.py` (memory_policy entry point — verify the actual
  function name; written by 77be321)
- `benchmarks/op_registry.py` — shape definitions
- `benchmarks/results/phase1/stage7/track6/l1/*_results.csv` — current state

**Step 1:** Read the preflight module to confirm the API:

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
grep -n "^def \|^class " benchmarks/preflight.py
```

**Step 2:** Write a one-shot audit script `benchmarks/scripts/audit_missing_shapes.py`:

```python
"""Audit P1: classify each missing (op, shape) under memory_policy."""
import csv, json
from pathlib import Path
from collections import defaultdict
from benchmarks.preflight import estimate_memory_required, MEMORY_BUDGET_BYTES  # adjust if names differ

m = json.loads(Path("benchmarks/stage7_bl5_target_matrix.json").read_text())
have = defaultdict(set)
with open("benchmarks/results/phase1/stage7/track6/l1/PERF_ALL.csv") as f:
    for row in csv.DictReader(f):
        have[row["operator"]].add(row["shape_tag"])

verdict = {"would_oom": [], "would_fit": [], "no_estimator": []}
for entry in m["l1"]:
    op = entry["op"]
    for tag in (entry.get("shape_tags_required") or []):
        if tag in have[op]: continue
        try:
            bytes_req = estimate_memory_required(op, tag)
        except Exception as e:
            verdict["no_estimator"].append((op, tag, str(e)[:60]))
            continue
        if bytes_req is None:
            verdict["no_estimator"].append((op, tag, "None"))
        elif bytes_req >= MEMORY_BUDGET_BYTES:
            verdict["would_oom"].append((op, tag, bytes_req))
        else:
            verdict["would_fit"].append((op, tag, bytes_req))

for k, v in verdict.items():
    print(f"\n=== {k} ({len(v)}) ===")
    for item in v[:20]:
        print(f"  {item}")
```

**Step 3:** Run + persist output:

```bash
python benchmarks/scripts/audit_missing_shapes.py | tee /tmp/p1_audit.txt
```

**Expected outcome:** breakdown like
`would_oom=45, would_fit=10, no_estimator=5` (numbers indicative). This
determines the work for P1.T2/T3.

**Verification:** counts sum to 60.

**Commit:** `chore(p1): add missing-shape audit script` if Leon wants
the script kept; otherwise discard.

### P1.T2 — Run the harness for `would_fit` shapes only

**Objective:** generate fresh PERF_ALL rows for shapes the preflight says
should fit in 6 GB. Use the per-op CLI so we don't re-run the entire L1.

**Files:**
- `benchmarks/cli.py` — `bench_l1` entry point (verify flag names)

**Step 1:** Inspect the CLI options:

```bash
python -m benchmarks.cli bench_l1 --help | head -40
```

**Step 2:** For each `(op, shape)` in `would_fit`, invoke:

```bash
python -m benchmarks.cli bench_l1 --op <op> --shape <tag> --append
```

(Driven by a small bash loop fed by `/tmp/p1_audit.txt`.)

**Step 3:** Verify rows landed:

```bash
python3 -c "
import csv
hits = 0
with open('benchmarks/results/phase1/stage7/track6/l1/PERF_ALL.csv') as f:
    for r in csv.DictReader(f):
        if (r['operator'], r['shape_tag']) in WOULD_FIT_SET: hits += 1
print('hits:', hits)
"
```

### P1.T3 — Mark `would_oom` shapes via memory_policy synthetic rows

**Objective:** the harness must EMIT a row for every required (op, shape)
even if preflight rejected it. Currently the missing shapes have no row at
all — the gate treats them as `missing_full_shape_evidence`. After 77be321
the harness should be writing `status=skipped, memory_policy=..., reason=oom`
rows for them; if it isn't, fix the harness to do so.

**Files:**
- `benchmarks/runners/<runner>.py` or `benchmarks/cli.py` (whichever holds
  the per-shape iteration loop — search for `MEMORY_BUDGET` or `preflight`)

**Step 1:** Grep for the integration point:

```bash
grep -rn "preflight\|memory_policy\|MEMORY_BUDGET" benchmarks/ --include='*.py'
```

**Step 2:** TDD — write a failing test under `tests/test_preflight_emits_skip_row.py`
that runs a tiny harness invocation against a shape too big to fit and asserts
`status=skipped, memory_policy != ""` lands in PERF_ALL.

**Step 3:** Fix the loop so that when preflight rejects, a synthetic
PERF_ALL row is emitted with:

- `status = "skipped"`
- `correctness_status = "skipped"`
- `memory_policy = <policy_name>`
- `memory_bytes_required`, `memory_bytes_budget`, `memory_ratio` populated
- `reason = "memory_preflight: requires N bytes > budget M bytes"`

**Step 4:** Run + verify gate gives a `memory_excluded` delta:

```bash
python -m benchmarks.cli bench_l1 --op layernorm --shape llama3-8b-long
python -m benchmarks.gate G7 --tier 2 2>&1 | grep -E "G7\.8b|memory_excluded"
```

**Expected:** G7.8b `missing_full_shape_evidence` drops by N (where N
= number of would_oom shapes for the op just run).

**Commit:** `fix(bench_l1): emit synthetic memory-skip rows for preflight rejections`

### P1.T4 — Repeat T2/T3 for all 60 shapes; re-check G7.8b

```bash
# After all per-op runs:
python -m benchmarks.gate G7 --tier 2 2>&1 | tail -30
```

**Exit criteria for P1:**
- G7.8b: `missing_full_shape_evidence = 0`
- Either `full_shape_coverage = 45/45` OR every gap is explicitly
  classified `memory_excluded` and surfaces in the audit report

**Commit:** `data(g7/l1): full-shape coverage via memory_policy preflight (commit X)`

---

## Phase P2 — G7.8c real-failure fixups ✅ CLOSED (2026-05-14)

**Goal:** Drive `correctness failures` from 10 → 0 by either (a) fixing the
underlying bug, or (b) classifying the failure as a typed/memory exclusion if
appropriate.

**Outcome:** ✅ G7.8c now PASSES. Correctness failures 10 → 0 across two commits
plus three prior commits (`a5508b2`, `fa14fd0`, `31e7275`):

| Commit    | Scope                                              | Class | Verified |
|-----------|----------------------------------------------------|:-----:|:--------:|
| `a5508b2` | Pin rope Golden to PyTorch-eager via LADDER_PREFERENCES | C | ✅ |
| `fa14fd0` | Regenerate rope L1 data (post-Golden pin)          | C     | ✅ |
| `31e7275` | flash_attention @ ds-v2-2k stale-handle assert cleared | B | ✅ |
| `82b635b` (Q5a) | rope fp16→fp32 sin/cos across 5 sites (extreme-long) | C | ✅ |
| `160ebf4` (Q6a) | bench_l2 emits typed-unsupported for gated odd-N (geglu/swiglu non-align-{1,2}) | D | ✅ |

The two remaining `status=error` rows (`gelu @ extreme-flat`, `topk @ extreme-wide`)
graduated from G7.8c (correctness count) into G7.8d (malformed perf rows). They are
tracked separately under Q6b (root-cause triage) and Q7 (extreme-shape preflight
scope narrowing — Leon flagged Q1c's "no oracle → audit" line as too aggressive for
point-wise ops like rope; rerun for `gelu:extreme-flat` and `topk:extreme-wide`
needs to repeat that distinction).

### P2.T1 — Triage each of the 10 failures (historical)

**Objective:** get a one-line root-cause for each. Run each failing op-shape
individually with full logging:

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.cli bench_l1 --op flash_attention --shape ds-v2-2k \
    --verbose 2>&1 | tee /tmp/p2_fa_ds-v2-2k.log
```

Repeat for: `gelu/extreme-flat`, `rope/extreme-long`, `topk/extreme-wide`,
`geglu/non-align-{1,2}` (L2), `swiglu/non-align-{1,2}` (L2).

**Step 2:** For each, classify into one of:

| Class | Action                                              |
|-------|-----------------------------------------------------|
| A) OOM / memory-policy issue          | Add memory_policy reason, will excl. |
| B) Runner traceback on edge shape     | Fix runner / add typed guard         |
| C) Numerical drift (rope mismatch)    | Investigate tolerance / impl bug     |
| D) Out-of-spec shape (non-align L2)   | Either fix Arke kernel or add guard  |

**Output:** triage table appended to this plan file.

### P2.T2 — Fix or guard each, one at a time

Each is a self-contained bite-sized TDD task. Format per item:

1. Write a failing test reproducing the bug (or asserting the typed guard).
2. Run to confirm RED.
3. Apply minimal fix.
4. Run to confirm GREEN.
5. Re-run the specific bench shape to confirm the failure clears.
6. Commit with `fix(p2/<op>): …` referencing the failure.

**Exit criteria for P2:**
- `python -m benchmarks.gate G7 --tier 2` shows G7.8c `correctness failures=0`.

---

## Phase P3 — G7.8d perf root-cause and ladder upgrades

**Goal:** L1 weighted ≥ 0.95 and L2 matmul_gelu=102/102, matmul_relu=102/102.

### P3.T1 — Sanity-check the Gate perf scoring semantics

**Objective:** confirm or refute the reconnaissance hypothesis that
`_check_bl5_performance_evidence` counts non-Arke baseline rows as
denominators. This affects what "ratio_vs_baseline" actually means
post-1a.

**Files:**
- `benchmarks/gate_g7.py:369-455` (perf checker)
- `benchmarks/runners/...` (where `perf_pass` is computed per row)

**Step 1:** Read both code paths; write a 10-line summary in
`/tmp/p3_perf_semantics.md` answering:

  - What `(op, shape, baseline)` rows go into `group_counts`?
  - Is `perf_pass` computed per-baseline or per-Arke-comparison?
  - What is `perf_target` set to?

**Step 2:** If the answer is "all baseline rows count", that is a
**semantic bug**: only `baseline=Arke` rows should contribute to perf
scoring. Discuss with Leon before changing — this is potentially a
Gate-standard change that needs his sign-off (per AGENTS.md
"Benchmark/Stage/Gate target/threshold changes need Leon").

**Expected outcome:** either confirm the scoring is correct as-is, or
escalate with a written proposal for Leon.

**This is decision-only, no code change without sign-off.**

### P3.T2 — Catastrophic-ratio kernel fixes (ratio ≤ 0.20)

**Targets** (each is its own ladder upgrade work item):

1. `layernorm` — Arke kernel p50=0.18. Action: switch primary to FlagGems
   layernorm or Liger-Kernel `LigerLayerNormFunction`. Verify ladder
   via `benchmarks/baselines/golden_ladder.py`.
2. `transpose` — Arke p50=0.01. Action: investigate per-shape autotune
   cache (likely root cause), then either use `torch.permute` baseline
   for small shapes or fix Arke's launch overhead.
3. `grouped_matmul` — Arke p50=0.08. Action: replace sequential loop with
   FlagGems / FlashInfer grouped matmul primary.
4. `batch_matmul` — Arke worst=0.02. Action: investigate which shapes
   trigger the slow path; likely K-dim or non-aligned batch.

For each, follow the SKILL convention from MEMORY:
> 每个 op 必查 GitHub 社区仓 (FlagGems/Liger/vLLM/FlashMLA/FlashInfer/flash-attn)
> 按性能 ladder 选 PRIMARY+FALLBACK，记 repo+commit+日期入 audit。

**Per-op micro-plan:**

  1. Read current ladder entry: `grep -A20 "<op>" benchmarks/baselines/golden_ladder.py`
  2. Search 6 community repos for `<op>` implementation; pick top-2 by claimed perf
  3. Add new runner under `benchmarks/runners/<repo>_<op>.py`
  4. Register in `golden_ladder.py` with `LADDER_PREFERENCES[<op>] = ("<new>", ..., "PyTorch-eager")`
  5. Re-bench: `python -m benchmarks.cli bench_l1 --op <op>` (writes per-op CSV)
  6. Verify p50 ratio: dump `ratio_vs_baseline` for new Arke rows
  7. Commit: `perf(p3/<op>): upgrade primary to <runner> — p50 ratio X → Y`

### P3.T3 — Sub-target ratio fixes (ratio 0.5-0.95)

Lower priority; only attack after T2 lands and re-baseline shows the gap
to 0.95 is still present:

- `dequantize_per_channel` (0.51)
- `cross_entropy` (0.68) — likely Liger CE
- `rmsnorm` (0.70) — likely FlagGems / Liger RMSNorm
- `rmsnorm_residual` (0.80)
- `geglu/swiglu` (0.85)
- elementwise OT0 long tail (silu/relu/gelu/tanh outliers below 0.90)

Same per-op micro-plan as T2.

### P3.T4 — L2 fusion completeness

`matmul_gelu = 39/102` and `matmul_relu = 48/102` are L2 fusion shape
coverage gaps, NOT perf gaps. After P1's memory_policy classification
runs L2 as well, recheck — most of the missing shapes likely OOM.

### P3.T5 — Final gate run

```bash
python -m benchmarks.gate G7 --tier 2 --live --archive
```

**Exit criteria for P3:**
- L1 weighted_score ≥ 0.95
- matmul_gelu = matmul_relu = 102/102 (or fully memory_excluded)
- Gate G7 returns 14/14 PASS

---

## Out-of-scope (explicitly deferred)

- `arke_score` aggregation polish (covered in Stage 9 final score work)
- Ascend / Triton-on-NPU prep (Phase 2)
- MLIR Dialect dialect-level optimizations (Phase 3)

## Decision points to escalate to Leon

| # | Decision                                                           | Phase | When        |
|---|---------------------------------------------------------------------|-------|-------------|
| D1 | If P3.T1 confirms non-Arke baseline rows are wrongly counted in scoring, change scoring? | P3 | After T1   |
| D2 | If P1.T1 shows would-fit > 0 but those shapes still OOM during run, raise memory budget? | P1 | After T1   |
| D3 | If P3.T2 ladder upgrades expose API breakage in existing tests, widen the runner ABI? | P3 | As-needed   |

## Per-phase exit criteria & expected commits

| Phase | Exit               | Expected commits |
|-------|--------------------|------------------|
| P1    | G7.8b PASS (or all gaps memory_excluded) | 2-4 commits: harness fix + per-op data |
| P2    | G7.8c correctness_failures=0 | 1 commit/failure, ~10 commits |
| P3    | G7.8d L1≥0.95, L2 102/102 | 1-2 per ladder upgrade, ~6-12 commits |

Every commit reports quantitative deltas in the commit message AND in
`memory/2026-MM-DD.md`.
