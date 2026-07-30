# Dynamic-Shape Benchmark Track — the "Performance Cliff", measured

**Status:** measurement infrastructure ✅ LANDED · gate mode: **D2 soft gate ACTIVE (Leon 2026-07-30)** — `same_spec_geomean ≤ 5×` + spec_key-prediction consistency via `benchmarks/gate_dynamic_shape.py`; track itself stays threshold-free; D3 revisit needs more cross-run data
**Tool:** `python -m benchmarks.dynamic_shape --all`
**Tests:** `tests/benchmark/test_dynamic_shape.py` (25)
**First dataset:** `benchmarks/results/dynamic_shape/2026-07-29_191225/` (RTX 3060 Laptop 6GB, sm_86, fp16)

---

## 1. Why this track exists

The static benchmark grid (`bench_l1`) measures *steady-state* latency on a fixed
set of shapes. Production LLM inference is not steady-state: variable-length
decoding sweeps sequence length token-by-token, so kernels constantly meet
shapes they have never compiled for. Every genuinely-new shape pays a
**first-call cost** the static grid never sees:

1. launcher-side config selection (tile heuristic / launch-config cache), and
2. Triton's per-shape `@triton.jit` specialization compile.

The KESTREL audit flagged this cliff as *speculative* — asserted, never
measured. This track turns it into a measured curve.

## 2. What is measured

For each op, a **single production wrapper** (`KERNEL_CACHE.get_or_build_by_op`,
the exact object `TritonBackend` serves) is driven through a sweep of shapes
that mimics a dynamic workload (non-pow2 sizes included). Per shape:

| column | meaning |
|:--|:--|
| `first_call_ms` | wall-clock of the very first call for that shape (CUDA-synced `perf_counter` — the cost is host-side, CUDA events would miss it) |
| `steady_ms` | median of 50 warm calls |
| `cliff_ratio` | `first_call / steady` — the cliff magnitude |
| `spec_key` | *predicted* kernel-specialization class (op-aware, see below) |
| `new_spec` | prediction: "this shape should trigger a compile" |

`spec_key` models what actually drives a recompile: the launcher-selected
constexpr config **plus** Triton's per-int-arg specialization classes
(`==1` / `%16==0` / other). It is op-aware because each template caches
differently — matmul buckets tile configs by `next_pow2` (K-H3.1), softmax
derives BLOCK from `next_pow2(N)`, rmsnorm keys on exact N. The measured
ratio is the ground truth the prediction is checked against.

**Deliberate design choice:** we do *not* render a fresh module per shape
(as `benchmarks/probes/autotune_first_call.py` does for its narrower
question). A fresh render defeats Triton's JIT cache and over-reports the
cliff; the production wrapper reproduces deployment reality.

## 3. First results (2026-07-29, RTX 3060 Laptop, fp16)

| op | shapes | cliff geomean | median | max | new-spec geomean | same-spec geomean |
|:--|--:|--:|--:|--:|--:|--:|
| matmul | 15 | **3.31×** | 2.33× | 27.9× | 3.66× (13) | 1.71× (2) |
| softmax | 12 | **40.99×** | 68.7× | 130.7× | 51.5× (11) | 3.34× (1) |
| rmsnorm | 11 | **7.22×** | 6.39× | 86.4× | 77.2× (2) | 4.27× (9) |

Sweeps: matmul M=1…512 @ N=K=4096 (LLaMA token-batch); softmax M=32 heads,
N=128…8192 (attention logits); rmsnorm N=4096 fixed, M=128…4096.

### Honest findings

1. **The cliff is real and op-dependent.** softmax pays ~3.5–6 ms compile for
   nearly *every* new sequence length (geomean 41×) because its BLOCK
   constexpr tracks `next_pow2(N)` — 8 distinct N-classes in one sweep = 8
   compiles. In a token-by-token decode loop this is the dominant dynamic-
   shape cost.
2. **K-H3.1 bucketing demonstrably helps matmul.** matmul's geomean is 3.31×
   vs softmax's 40.99×. The tile-config bucket cache plus the fact that the
   tile constexprs (not raw M) feed the JIT key means many new shapes reuse
   compiled kernels (e.g. m48→m64 shared, cliff 2.4×/2.6× ≈ launch noise).
3. **rmsnorm's cliff is confined to the first shapes** (86×/69× for the first
   M-div-class pair), then flat ~2–6× — its kernel does not specialize on M
   beyond divisibility, exactly as `spec_key` predicts.
4. **Prediction vs measurement mostly agree** — new-spec rows carry the big
   ratios, same-spec rows sit near launch noise. Two caveats worth keeping:
   (a) same-spec baseline ratios are 1.7–4.3×, not 1.0× — first-call-for-a-
   *tensor-size* also pays allocator/caching-allocator work; (b) occasional
   outliers (driver/clock jitter on a laptop GPU) inflate single cells; the
   geomean split is the robust signal.
5. **Residual measurement noise:** the very first sweep entry after process
   start can absorb residual one-time costs despite the out-of-sweep warmup
   (observed m1 24× vs 688× across runs). Per-cell numbers are indicative;
   distribution stats are the contract.

## 4. Interpretation for the AI-Native thesis

An Agent-facing toolchain must expose this cost model to its Agent consumer:
"new shape ⇒ possible multi-ms compile" is exactly the kind of legality/cost
information StrategyIR feedback should carry. The `spec_key` predictor is a
first concrete step: it is cheap to compute, op-aware, and empirically aligned
with the measured cliff — a candidate for surfacing in compiler V2 feedback.

## 5. Gate threshold — deliberately NOT set here

Pass/fail semantics on this track (e.g. "new-spec geomean ≤ X×" or
"steady-state within Y% of static-grid latency") are **frozen-layer** gate
decisions. The module hard-guards against baking one in
(`test_no_gate_threshold_in_module`).

**Decision history:**
- 2026-07-29: **D1 approved** — measure-only, no pass/fail gate.
- 2026-07-30: **D2 approved** (Leon: "D推进D2并完成依赖") — soft gate live,
  threshold `same_spec_geomean ≤ 5×` is now a Leon-approved frozen parameter.

- **D1 (measure-only):** the track itself stays threshold-free
  (`test_no_gate_threshold_in_module` still guards this).
- **D2 (soft gate, ACTIVE):** `same_spec_geomean ≤ 5×` AND per-op `n_new_spec`
  must match the `spec_key` prediction (catches accidental despecialization).
  Implemented as a **consumer** of the track: `benchmarks/gate_dynamic_shape.py`
  (`python -m benchmarks.gate_dynamic_shape <run_dir>...`), threshold lives in
  the gate module, never in the track. Tests:
  `tests/benchmark/test_gate_dynamic_shape.py` (10 tests incl. frozen-param
  guard + separation guard).
- **D3 (hard gate, future):** additionally cap `new_spec_geomean` per op class —
  needs more cross-run data; `new_spec_geomean` CV measured at 53-59% (see
  variance table below), far too noisy to lock a per-class cap today.

### D2 dependency: cross-run variance data (2026-07-30, RTX 3060 fp16, 3 runs)

`benchmarks/results/dynamic_shape/variance_run{1,2,3}/` — same sweep, fresh
process each run, ~10s apart:

| op | metric | run1 | run2 | run3 | CV |
|:--|:--|--:|--:|--:|--:|
| matmul | same_spec_geomean | 1.66 | 1.36 | 1.65 | 11.1% |
| rmsnorm | same_spec_geomean | 4.34 | 4.28 | 4.35 | 0.9% |
| softmax | same_spec_geomean | 4.28 | 4.79 | 4.15 | 7.8% |
| matmul | new_spec_geomean | 3.29 | 2.61 | 2.49 | 15.5% |
| rmsnorm | new_spec_geomean | 62.6 | 33.4 | 22.2 | 53.0% |
| softmax | new_spec_geomean | 54.4 | 27.7 | 16.8 | 58.7% |

`n_new_spec`/`n_same_spec` identical across all runs (matmul 13/2, rmsnorm 2/9,
softmax 11/1) — the `spec_key` prediction is deterministic, which is what the
D2 consistency check leans on.

**D2 verdict at activation: PASS on all 3 runs × 3 ops** (worst
same_spec_geomean 4.79, limit 5.0).

**Honest margins:** (1) softmax has only n_same=1 in the sweep, so its
same-spec "geomean" is a single shape whose 4-5× is host-launch/clock jitter
on a ~0.1ms kernel — near the limit but stable (CV 7.8%); a richer sweep would
dilute this. (2) rmsnorm's 4.3× same-spec is the documented M-arg residual
(~1.4ms warm-N recompile on first novel M). Both are real, reproducible
behavior — the 5× limit accommodates them without masking a true
despecialization (which lands at compile scale, 20-100×+).

## 6. Mitigation — bucket-aware warmup (R3, audit 2026-07-29)

The audit confirmed the cliff mechanism empirically: it is **not** per-exact-N
recompilation. Triton specializes on `(BLOCK_N constexpr, N % 16)`, so two
different shapes in the same `(next_pow2(N), %16)` bucket **share** the
compiled kernel. Measured on RTX 3060: softmax N=512 first-call 253 ms, but
N=480 / N=496 (same bucket) first-call **0.16 ms** — full reuse. The cliff is
purely the **first-touch compile of each bucket**.

Fix: row-scan templates now expose `<kernel>_warmup_buckets(...)`, which
pre-compiles every distinct bucket a workload will hit, covering **both**
divisibility classes per pow2 bucket (aligned + ragged, so unaligned decode
lengths are also warm). Call once at serving startup with the model's
sequence-length / hidden-dim range; the variable-length decode path then never
hits a compile wall.

Measured cliff reduction (RTX 3060, fp16, fresh exact-N inside warmed buckets):

| op | cold cliff geomean | post-warmup geomean | how warmed |
|:--|--:|--:|:--|
| softmax | 40.99× | **1.33×** | `arke_softmax_warmup_buckets([128,256,512,1024,2048,4096,8192])` |
| rmsnorm | 7.22× | **2.61×** (residual ↓) | `arke_rmsnorm_warmup_buckets([hidden_dim])` — warms N (BLOCK_N) + M div reps; rmsnorm also specializes on the M row-count arg, so a *first novel M* still pays a small ~1.4ms warm-N recompile (vs 5ms cold), then reuses |

Regression test: `tests/backend/test_rowscan_warmup.py` (bucket-key semantics
CPU-side + GPU smoke asserting cliff collapse). matmul was already mitigated by
its `_TILE_CFG_CACHE` pow2 bucket (cliff 3.31×); flash_attention by
`_FA_CFG_CACHE` (K-ATT). Remaining honest residual: the *very first* real
workload after warmup can still show a single-shape spike (clock/launch
warmup, not compile) — geomeans above already include those.

Not yet warmed: layernorm and other row-scan variants (same pattern applies,
follow-up). No **runtime** "JIT-too-expensive → fall back to eager/interpreter
+ async compile" policy yet — that is the D2/D3 + serving-integration
follow-up, not this measure-only track.

## 7. Files

| path | role |
|:--|:--|
| `benchmarks/dynamic_shape.py` | track implementation (sweeps, spec_key, CSV) |
| `tests/benchmark/test_dynamic_shape.py` | 25 tests incl. GPU smoke + frozen-layer guard |
| `benchmarks/results/dynamic_shape/<ts>/` | per-op `*_cliff.csv` + `summary.json` |
| `benchmarks/probes/autotune_first_call.py` | narrower K-H3.1 probe (fresh-module render) — complementary, not superseded |
