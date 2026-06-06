# D7-E1.1 — GPT-2 torch.compile Regression Diagnosis

**Captured:** 2026-05-17T03:01Z (UTC); local **2026-05-17 03:01** (run dir was named with the UTC-evening date and is kept as the canonical artifact path)
**Device:** NVIDIA GeForce RTX 3060 Laptop GPU (Ampere, SM 8.6, 6 GB)
**Torch:** 2.6.0+cu124
**Model:** `gpt2` (HF, fp16)
**Mode:** `torch.compile(mode="reduce-overhead")` vs eager
**Sampling:** warmup=10, runs=20 (vs earlier first-pass run with warmup=3, runs=5)

---

## Headline finding — original 0.811× was a measurement artifact, NOT a real regression

The Stage-8 entry-scout reading of **0.811× at seq=128 (warmup=3, runs=5)** does
not reproduce under proper warmup. With warmup=10/runs=20:

| seq_len | eager mean (ms) | compile mean (ms) | ratio_vs_eager | G8[4] ≥0.95 |
|:-------:|:---------------:|:-----------------:|:--------------:|:-----------:|
| 128 | 8.12 | 8.07 | **1.006** | ✅ |
| 256 | 8.41 | 9.73 | **0.865** | ❌ |
| 512 | 11.69 | 11.42 | **1.024** | ✅ |

**2/3 seq_lens already pass G8[4] ≥0.95×.** The 0.811× the entry-scout reported
was first-call compile overhead bleeding into the 5-run mean. Lesson re-learned:
torch.compile benchmarks need warmup ≥10 and runs ≥20 to be honest. The
`bench_l3` defaults (warmup=5, runs=20) are borderline; the seq=128 first-pass
used `--warmup 3 --runs 5` which is well below the noise floor for compile.

## What's *actually* still broken — the seq=256 ratio drop

256 is the **only** failing measurement, and the root cause is visible in the
dynamo log captured during the run (saved to the background-task output):

```
W torch._dynamo hit config.cache_size_limit (8)
  function: 'forward' (transformers/models/gpt2/modeling_gpt2.py:144)
  last reason: 11/0: L['self'].layer_idx == 0
```

Translation: while measuring multiple seq_lens **in the same Python process**,
dynamo runs into 12+ GPT-2 layers × 3 seq sizes worth of recompiles, hits the
default `cache_size_limit=8`, and starts evicting + recompiling cache lines.
The 256 measurement happens to land in the middle of that thrash window
(128 was first → cache empty, 512 was last → cache stabilized by then).

The 12 reported graph breaks are *all* duplicates of:
```
'inline in skipfiles: Mapping.__contains__ | __contains__
 _collections_abc.py, skipped according trace_rules.lookup SKIP_DIRS'
```
i.e. dynamo's skipfile rule against `collections.abc.Mapping.__contains__`,
triggered by `if "cache_position" in kwargs:` / `if cache_kwargs is not None`
style dict checks inside the transformers GPT-2 forward. These breaks are
*intra-layer* (12 layers × 1 break per layer), they don't introduce per-step
eager fallback overhead in the steady state (the surrounding graph still
compiles), and they're not the dominant cost — the cache-size-limit thrash is.

## Ranked Root Causes (re-calibrated)

| Rank | Score | Cause | Evidence | Fix path |
|:----:|:-----:|:------|:---------|:---------|
| **#1** | 90 | **Dynamic-shape recompile thrash hitting `cache_size_limit=8`** | dynamo log "hit config.cache_size_limit (8)"; only seq=256 fails; 128 & 512 already ≥1.0× | Raise `torch._dynamo.config.cache_size_limit` to ~64, OR compile once with `dynamic=True`, OR per-seq isolated runs. P0 — folded into new **D7-E1.6**. |
| #2 | 40 | **12 graph breaks from `Mapping.__contains__` skipfile in transformers GPT-2** | dynamo.explain → 13 graphs / 12 breaks, all same reason | Either patch transformers GPT-2 forward to use `getattr` / explicit `is not None` checks, or whitelist `collections.abc` via `torch._dynamo.config.skipfiles_inline_module_allowlist`. P1 — not the dominant cost; defer until D7-E1.6 lands and we re-measure. |
| #3 | 10 | **First-call compile cost ~3.2s** when measured cold | compile_metrics.json `first_call_s=3.165` | Already mitigated by warmup ≥10; document the floor in `bench_l3` help text. P2. |

**Original D7-E1.2 (graph break elim) and D7-E1.3 (CUDA Graph) hypotheses do
not match the evidence and should be deprioritized.** The actual critical-path
fix is dynamic-shape cache management.

## Re-planned Stage 8 M1 critical path

Replace the planned `D7-E1.2 → E1.3 → E1.4 → E1.5` chain with:

1. **D7-E1.6 (new, P0, S)** — Dynamic-shape recompile control: bump
   `cache_size_limit`, switch to `dynamic=True` compile, isolate per-seq
   warmups in `bench_l3`. Target: seq=256 ratio ≥0.95×.
2. **D7-E1.5 (existing, P0, S)** — Hit G8[4] target: rerun
   `bench_l3 --model gpt2 --seq-len 128,256,512 --warmup 10 --runs 20`,
   require ratio ≥0.95 at all three.
3. **D7-E1.4 (existing, P0, L)** — Arke→torch.compile bridge MVP (rmsnorm +
   matmul as `torch.library` custom ops); now justified by Stage 8 design, not
   by the (non-)regression.
4. **D7-E1.2 / E1.3 (existing, downgrade to P1)** — Graph break elim + CUDA
   Graph: only revisit if E1.6 isn't sufficient at longer seq_lens (1024+).

## Reproduction

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.scripts.diagnose_gpt2_torch_compile \
  --seq-lens 128,256,512 --warmup 10 --runs 20
```

## Artifacts (this run dir)

- `dynamo_explain.txt` — full `torch._dynamo.explain()` output (7320 lines)
- `dynamo_explain_summary.json` — 13 graphs / 12 breaks / 9 distinct reasons (all the same skipfile)
- `compile_metrics.json` — first_call=3.165s / second_call=5.203s (cache thrash visible in second_call > first_call)
- `timings.json` — per-seq_len eager vs compile timings
- (this file) `diagnosis.md` — re-calibrated root cause ranking
