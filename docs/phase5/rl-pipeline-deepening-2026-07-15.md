# RL/Learning Pipeline Deepening — Multi-Round Corpus Collection

**Status:** implemented (2026-07-15). **Phase:** 5 (M4/M5 groundwork).
**Scope:** deepen the Arke RL/learning pipeline from the minimal single-step
roundtrip closure to real multi-round tuning-corpus collection.

---

## 1. Current-state analysis (before this change)

### 1.1 The write → read → RL chain

```
                        WRITE                                READ / RL EXTRACT
 ┌───────────────────────────────────┐         ┌──────────────────────────────────────┐
 │ TrajectoryWriter                   │         │ extract_rl_samples(path)               │
 │  (arke/learn/trajectory.py)        │  JSONL  │  (arke/learn/rl_dataset.py)            │
 │  header → decision/compile/profile │ ──────▶ │  walk records:                         │
 │         /verify/adjust → done      │ traj-   │   decision → pair with NEXT profile    │
 │  v1.0 record contract              │ .jsonl  │   → RLStepSample(state, action, reward)│
 │  (trajectory_schema.py, 11 kinds)  │         │  final profile → RLTrajectorySample    │
 └───────────────────────────────────┘         └──────────────────────────────────────┘
        ▲            ▲                                          │
        │            │                                         ▼
   optimize.py   run_live_optimize.py                 build_rl_dataset → rl_corpus.jsonl
   (heuristic)   (live LLM, ad-hoc JSONL)             quality_gate (rl_quality.py)
```

**Producers.**
- `arke/agent/optimize.py` — heuristic path. Emits per-cycle
  `compile → profile → adjust`, but **never emits `decision` records**, so the
  miner extracts **zero step samples** from it.
- `benchmarks/live/run_live_optimize.py::_write_live_trajectory` — live-LLM
  path. Maps `apply_decision → decision`, `compile_and_profile → profile`. This
  is an **ad-hoc JSONL dump inside a benchmark script** — not reusable, not
  tested, and bypasses `TrajectoryWriter`/the v1.0 contract.

**Reader.** `extract_rl_samples` pairs each `decision` with the next
`profile`/`verify` and assigns the discrete `robust_reward` tier
(−1/1/2/3, anti-reward-hacking, from `arke/agent/verification.py`).

### 1.2 What was missing (single-step closure → multi-round corpus)

| Gap | Detail |
|---|---|
| **G1 — no credit assignment** | Step samples carried only an *instantaneous* reward. No `reward_delta`, no `return_to_go`, no discount. PPO/GRPO need per-step returns/advantage targets. |
| **G2 — thin trajectory sample** | `RLTrajectorySample` had only `final_reward`. No discounted episode return, no step-reward sequence, no step count. |
| **G3 — no reusable multi-round recorder** | The heuristic path emitted no `decision` records (0 step samples); the live path was an untested ad-hoc dump. There was no tested, contract-valid component to record a full multi-round session (multiple decisions + their intermediate profile outcomes). |
| **G4 — no e2e proof** | Nothing drove a real multi-round session end-to-end and landed a mineable corpus with returns computed + quality-gated. |

---

## 2. What was implemented

### 2.1 Step-wise reward + discounted return-to-go (`arke/learn/rl_dataset.py`)

`RLStepSample` gains four credit-assignment fields, and
`RLTrajectorySample` gains three aggregates:

- **`step_index`** — 0-based decision position in the episode.
- **`reward_delta`** — `reward − prior_step_reward` (the "did this move help?"
  signal).
- **`return_to_go`** — discounted Monte-Carlo return
  `G_t = Σ_{k≥t} γ^(k−t) · r_k`, computed by a reverse scan
  `G_t = r_t + γ·G_{t+1}`.
- **`discount`** (γ) + **`episode_len`** recorded for reproducibility.
- Trajectory: **`step_rewards`** (per-step reward sequence),
  **`discounted_return`** (`G_0`), **`num_steps`**.

`extract_rl_samples(path, *, discount=DEFAULT_DISCOUNT)` and
`build_rl_dataset(..., discount=...)` thread γ through (default `0.95`). The
header parse now also resolves `op`/`shape` from `semantic_ir` so recorder
headers mine correctly.

### 2.2 Multi-round session recorder (`arke/learn/session_recorder.py`, new)

A reusable, tested writer over `TrajectoryWriter`:

- **`RoundOutcome`** — the measured result of one round (correct, eager/strong
  ratios, latency, bottleneck); computes its own `robust_reward` tier.
- **`SessionRecorder`** — `start() → record_round(...)* → finish()`. Each
  `record_round` emits the full `decision → compile → profile → adjust` burst
  for that round, embedding the pre-computed `robust_reward` into the profile
  record so the miner reads it directly. Enforces the A5 `@rationale` contract
  (rejects empty rationale). Context-manager sugar included.

This closes **G3**: unlike the heuristic path it *does* emit `decision`
records, and unlike the live path it goes through the v1.0 contract and is
tested.

### 2.3 End-to-end runnable (`benchmarks/live/run_multiround_session.py`, new)

Drives multi-round sessions (deterministic improving mock policy — reproducible,
no LLM/GPU), records each via `SessionRecorder`, mines the corpus with returns,
prints per-step diagnostics, and runs the M3 quality gate. A live-LLM/GPU policy
plugs into the same recorder by swapping the scripted episode for a real one.

---

## 3. Evidence (runnable proof)

### 3.1 E2E script

```
$ python -m benchmarks.live.run_multiround_session --out /tmp/mr_session
=== recording 2 multi-round session(s) ===
  matmul: 3 rounds, rewards=[1, 2, 3], best=3
  softmax: 2 rounds, rewards=[1, 2], best=2
=== mined RL corpus (γ=0.95): {'steps': 5, 'trajectories': 2, 'files': 2} ===
  matmul: step_rewards=[1, 2, 3] discounted_return(G0)=5.6075 num_steps=3
      step[0] tile      reward=+1 delta=+1 return_to_go=5.6075
      step[1] tile      reward=+2 delta=+1 return_to_go=4.85
      step[2] compute   reward=+3 delta=+1 return_to_go=3.0
  softmax: step_rewards=[1, 2] discounted_return(G0)=2.9 num_steps=2
      step[0] tile      reward=+1 delta=+1 return_to_go=2.9
      step[1] vectorize reward=+2 delta=+1 return_to_go=2.0
  reward histogram (all steps): {1: 2, 2: 2, 3: 1}
=== corpus quality gate ===
RL Corpus Quality Gate: PASS
  samples: 7 (5 step + 2 traj)
  ✅ schema_sanity / deduplication / reward_distribution / tier_coverage
```

**Return math verified:** matmul `G_0 = 1 + 0.95·2 + 0.95²·3 = 5.6075` ✓.

### 3.2 Tests

```
$ python -m pytest tests/test_multiround_rl_pipeline.py tests/test_rl_dataset_d3.py -q
23 passed in 0.08s
```

`tests/test_multiround_rl_pipeline.py` (new, 17 tests) covers reward tiers,
v1.0-contract validity of the emitted trajectory, `@rationale` enforcement,
`reward_delta`/`return_to_go` math (incl. γ=0 myopic and γ=1 undiscounted-sum
edge cases), trajectory aggregates, and the full record→mine→quality-gate loop.

**No regressions:**
```
$ python -m pytest tests/ -q -k "rl or trajectory or rationale or quality"
194 passed, 2343 deselected in 3.76s
```

---

## 4. Files changed

| File | Change |
|---|---|
| `arke/learn/rl_dataset.py` | `DEFAULT_DISCOUNT`; step/traj return fields; γ-parameterized extraction; header `semantic_ir` fallback |
| `arke/learn/session_recorder.py` | **new** — `RoundOutcome`, `SessionRound`, `SessionRecorder` |
| `benchmarks/live/run_multiround_session.py` | **new** — e2e multi-round session → mined corpus |
| `tests/test_multiround_rl_pipeline.py` | **new** — 17 tests |
| `docs/phase5/rl-pipeline-deepening-2026-07-15.md` | **new** — this doc |

---

## 5. Follow-ups (not in this change)

- Wire `SessionRecorder` into the live-LLM harness to replace the ad-hoc
  `_write_live_trajectory` dump (unifies the live path onto the tested contract).
- Emit `decision` records from the heuristic `optimize.py` path so it also
  yields step samples.
- GAE (λ) advantage on top of `return_to_go` once a value baseline exists (M5).

*Reward schedule: `arke/agent/verification.py`. Corpus design:
`docs/architecture/arke-rl-corpus-design.md`.*
