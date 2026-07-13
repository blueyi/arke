# Live-LLM Autotuning Batch — robustness + RL corpus (2026-07-13)

> **Trigger:** Leon task 2 — "结合跑更多 op 的调优验证并增强 Arke Harness 健壮性，
> 并积累 RL 语料".

Batch of 7 live-LLM autotuning runs (`benchmarks/live/batch_autotune.sh`) across
5 op types / 7 shapes, driven by `claude-sonnet-4-6` via the clean `/v1`
endpoint. Each run drove the 8 frozen Façade tools with real GPU profiling and
emitted a mineable trajectory.

## Robustness result — 7/7 clean

| Run | op | decisions | tool_calls | final_reward | best_ratio | errors | fallbacks |
|-----|----|:--:|:--:|:--:|:--:|:--:|:--:|
| matmul_512  | matmul    | 6 | 18 | 3 | 1.487× | 0 | 0 |
| matmul_1024 | matmul    | 5 | 16 | 3 | 1.368× | 0 | 0 |
| matmul_256  | matmul    | 5 | 14 | 1 | 1.012× | 0 | 0 |
| layernorm_2048 | layernorm | 4 | 14 | 3 | **3.672×** | 0 | 0 |
| softmax_4096 | softmax  | 3 | 13 | 1 | 0.316× | 0 | 0 |
| silu_1024   | silu      | 4 | 14 | 1 | 0.315× | 0 | 0 |
| add_512     | add       | 5 | 15 | 1 | 0.747× | 0 | 0 |

**Zero errors, zero provider fallbacks across all 7 runs.** Every run completed
cleanly at max_turns with a valid strategy and real GPU profile. This validates
Harness robustness across all 5 op tiers with a live frontier LLM.

## RL corpus accumulated

`benchmarks/results/phase4/live/rl_corpus.jsonl` — **12 samples** (5 step + 7
trajectory), op coverage `{add, layernorm, matmul, silu, softmax}`.

Reward distribution (D2 discrete robust_reward):
- **3/7 trajectories beat baseline** (reward 3): layernorm 3.67×, matmul_512
  1.49×, matmul_1024 1.37×.
- 4/7 correct-but-not-faster (reward 1) — still valuable negative/neutral
  signal for RL (the policy learns which strategies *don't* help on a shape).

## Observations / follow-ups

1. **All runs stop at max_turns, not self-convergence.** The LLM keeps exploring
   within budget rather than emitting an explicit "done" when satisfied. This is
   acceptable for a bounded-budget optimizer, but an optional plateau early-stop
   (via the existing `hooks` seam, not a frozen-runner change) would save tokens.
   Tracked as a future efficiency enhancement — not a robustness defect.
2. **Small-shape / memory-bound ops (add_512, silu_1024, softmax_4096) landed
   <1×** this batch — consistent with the known Triton-baseline ceiling on tiny
   shapes; the LLM correctly kept them correct (reward 1) without hacking speed.
3. Corpus is ready for M3 (quality gates) → M4 (SFT export) per
   `arke-rl-corpus-design.md`.

---

*Batch run 2026-07-13 via batch_autotune.sh. Corpus: rl_corpus.jsonl.
Per-run artifacts under benchmarks/results/phase4/live/<run>/.*
