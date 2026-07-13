# Arke RL Corpus — Design & Harness Integration

**Status:** design v1 (2026-07-13). **Owner:** Arke lead engineer.
**Leon-approved direction:** D3=yes (2026-07-12) — mine Arke optimization
trajectories into an agentic-RL training corpus, CUDA-Agent style.

---

## 1. Why Arke has a structural edge

Test-time-only kernel-gen systems (Astra, AutoKernel, GEAK) throw away every
run after producing a kernel. Arke's Harness emits a **frozen v1.0 trajectory**
for *every* optimization run — the (state → action → outcome) sequence is a
first-class, replayable artifact. That means an RL/SFT corpus is a **free
by-product** of normal operation, exactly the data shape CUDA Agent
(ByteDance/Tsinghua) trains its policy on.

The corpus captures the LLM's **role #1** (runtime optimization decision-maker)
so a future policy can be trained/distilled to make those decisions better —
and it does so with the **anti-reward-hacking discrete reward** (D2) that keeps
correctness dominant.

---

## 2. Data model

### 2.1 Source: trajectory v1.0 (already frozen)

Each `trajectory.jsonl` is a stream of `{"t", "kind", "data"}` records:
`header` → (`decision` | `verify` | `profile` | `checkpoint` | `rollback` |
`adjust` | `compact` | `fallback`)* → `done`. The record schema is locked in
`arke/learn/trajectory_schema.py`.

### 2.2 Two derived sample shapes (`arke/learn/rl_dataset.py`)

**Step sample** — one per decision, for step-level RL (PPO / GRPO advantage):
```json
{
  "type": "step",
  "op": "matmul",
  "shape": {"M": 512, "N": 512, "K": 512},
  "prior_decisions": [ {"kind","params","rationale"}, ... ],
  "action": {"kind": "tile", "params": {"loop":"i","factors":[128]}, "rationale": "..."},
  "reward": 3,
  "correct": true, "baseline_ratio": 1.49, "latency_ms": 0.066, "backend": "triton"
}
```
- **State** = op + shape + prior decision list (the StrategyIR context the LLM
  saw when it acted).
- **Action** = the decision it took (kind + params + @rationale).
- **Reward** = discrete `robust_reward` tier at the resulting kernel state.

**Trajectory sample** — full sequence + final reward, for trajectory-level
ranking / preference pairs (best-of-N, Sakana least-to-most):
```json
{
  "type": "trajectory",
  "op": "matmul", "shape": {...},
  "decisions": [ ... ],
  "final_reward": 3, "final_correct": true, "final_baseline_ratio": 1.49
}
```

### 2.3 Reward schedule (D2, anti-reward-hacking)

`robust_reward(correct, eager_ratio, strong_ratio)` →
| tier | value | condition |
|---|:---:|---|
| INCORRECT | −1 | not numerically correct (dominates — a fast wrong kernel is worthless) |
| CORRECT | 1 | correct, no speedup |
| BEATS_EAGER | 2 | correct + faster than torch eager |
| BEATS_BOTH | 3 | correct + beats a strong baseline (cuBLAS/cuDNN/FlagGems) |

Discrete tiers (not raw continuous speedup) prevent reward-hacking where a model
games a noisy latency delta. This mirrors CUDA Agent's schedule.

---

## 3. Corpus construction pipeline

```
 live/batch runs ─┐
                  ├─► *.trajectory.jsonl ─► build_rl_dataset() ─► rl_corpus.jsonl
 gate G8 runs ────┤                          (dedupe by traj id;   (step + trajectory
 heuristic runs ──┘                           reward from D2)        samples, JSONL)
                                                    │
                                                    ├─► reward_histogram()  (diagnostics)
                                                    └─► @rationale KB (parallel miner)
```

- **Idempotency:** `build_rl_dataset` rebuilds from scratch per training run
  (RL datasets are regenerated, not appended). The @rationale KB is the
  append-only, deduped long-term store.
- **Provenance:** every sample carries `source` = trajectory path.
- **Multi-source:** the same miner ingests live-LLM runs, G8 gate runs, and
  heuristic-baseline trajectories uniformly — heuristic runs give the policy a
  behavior-cloning floor; live-LLM runs give exploration diversity.

---

## 4. Harness integration

### 4.1 Emission (already wired)
- `LLMRunner.optimize()` writes `trajectory.jsonl` via `TrajectoryWriter`.
- `compile_and_profile` now emits the `robust_reward` field into the `profile`
  record, so the miner reads the reward directly (falls back to recomputing
  from `correct` + `baseline_ratio`).

### 4.2 Mining (this design)
- `arke/learn/rl_dataset.py` — `extract_rl_samples`, `build_rl_dataset`,
  `reward_histogram`. Pure Substrate reader; does **not** touch the frozen
  Façade or event schema.

### 4.3 Consumption (future, Phase 5+)
Two consumers, both outside the current repo scope:
1. **SFT / behavior cloning** — trajectory samples where `final_reward >= 2`
   become (prompt=state, completion=decision+rationale) supervised pairs.
2. **RL fine-tuning (PPO/GRPO)** — step samples become (state, action, reward)
   tuples; advantage estimated within a trajectory. The bounded action space
   makes the action distribution finite and maskable (only legal moves), which
   is exactly what a maskable-PPO head needs.

**Integration seam:** the corpus is a plain JSONL. A training harness (TRL,
verl, OpenRLHF) reads it directly; Arke does not depend on any training
framework. This keeps the D1=a decision intact — Arke owns the domain corpus,
external frameworks own the training loop.

---

## 5. Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M3 corpus schema + miner | rl_dataset.py, step+trajectory samples, D2 reward | ✅ (2026-07-13) |
| M2 corpus accumulation | batch live-LLM autotuning across ops/shapes → rl_corpus.jsonl | ✅ (2026-07-13) |
| **M3** | **语料质量门禁** | ✅ (2026-07-13) |
| M4 SFT export | filter final_reward>=2 → prompt/completion pairs | ⬜ Phase 5+ |
| M5 RL fine-tune harness | maskable-PPO on step samples via external framework | ⬜ Phase 5+ |

---

*Last updated: 2026-07-13. Implementation: `arke/learn/rl_dataset.py`
(+ tests `tests/test_rl_dataset_d3.py`). Reward: `arke/agent/verification.py`.*
