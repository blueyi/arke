# Live-LLM Autotuning Demo — end-to-end closed loop (2026-07-13)

> **Trigger:** Leon — "启动一个新 session 进行端到端 live-llm autotuning demo，并以当前某个未达标的 kernel 优化为例"
>
> Demonstrates the full Arke thesis loop: a **live LLM agent** autonomously
> optimizes an **under-performing kernel** through the bounded-action Façade,
> producing a verified trajectory that feeds **both** the @rationale KB and the
> **D3 RL corpus**.

---

## Setup

- **Target kernel:** `matmul 512×512×512` — an under-target op (CUDA-C ~0.53×
  cuBLAS; the naive Triton baseline starts at 0.36×).
- **Live LLM:** `claude-sonnet-4-6` via the clean yunwu `/v1` OpenAI-compatible
  endpoint (ARKE_LLM_* overrides — avoids the bare-`yunwu.ai` Anthropic endpoint
  that injects ~30K tokens of Claude-Code context and hijacks tool-use).
- **Harness:** `benchmarks/live/run_live_optimize.py` → `LLMRunner.optimize()`
  driving the 8 frozen Façade tools, real Triton GPU profiling (V1/V2).

Command:
```bash
export ARKE_LLM_API_KEY="$OPENAI_API_KEY" \
       ARKE_LLM_BASE_URL="https://yunwu.ai/v1" \
       ARKE_LLM_PROTOCOL="openai" ARKE_LLM_MODEL="claude-sonnet-4-6"
python -m benchmarks.live.run_live_optimize \
  --op matmul --shape 512,512,512 --model yunwu/claude-sonnet-4-6 \
  --max-turns 15 --out benchmarks/results/phase4/live/matmul_512
```

---

## Result (real GPU, live LLM)

| Metric | Value |
|:--|:--|
| Decisions | 6 (all with @rationale — A5 audit clean) |
| Tool calls | 18 |
| compile_and_profile (correct) | 2/2 |
| Tokens (in/out) | 77,580 / 2,205 |
| Duration | 75.5 s |
| Best baseline_ratio | **1.487×** (beats torch eager) |
| Provider fallbacks | 0 |
| Errors | none |

The agent explored a real GEMM optimization strategy autonomously:
1. `tile(i=128)` — M-dim 128-row block tile
2. `tile(j=128)` — N-dim 128-col output tile (128×128/block)
3. `tile(k=32)` — K reduction tile for shared-mem pipelining
4. `parallel(i→threadblock.y)` — row-tile to grid.y
5. `parallel(j→threadblock.x)` — col-tile to grid.x
6. `vectorize(j, width=4)` — 128-bit vectorized loads

Each decision carried a substantive @rationale (the locked thesis pillar).

---

## Closed loop: trajectory → D2 reward → D3 RL corpus + @rationale KB

The emitted `trajectory.jsonl` was mined through both learning pipelines:

**D3 RL dataset** (`arke/learn/rl_dataset.py`):
- Trajectory sample: `op=matmul, final_reward=3, decisions=6`
- **final_reward=3 = BEATS_BOTH** — the discrete robust_reward (D2) correctly
  captured that the best config beat both eager and the baseline. A fast-but-
  wrong kernel would have scored −1; the anti-reward-hacking schedule held.
- Written to `rl_dataset.jsonl` (step + trajectory samples).

**@rationale KB** (`arke/learn/rationale_kb.py`):
- 6 entries mined (decision + rationale + outcome).

---

## What this validates

1. **Bounded-action-space autonomy** — the LLM only chose from compiler-legal
   moves and produced a monotonic optimization strategy.
2. **Staged verification (V1/V2)** — every profiled kernel was correctness-
   checked before its latency counted.
3. **Trajectory-as-learning-artifact** — one run fed both the @rationale KB
   (human-experience loop) and the D3 RL corpus (agentic-RL training data),
   for free. This is Arke's structural edge over test-time-only kernel-gen
   systems (CUDA Agent trains on exactly this kind of trajectory).
4. **Robust reward (D2)** — the discrete anti-reward-hacking schedule labeled
   the trajectory correctly (BEATS_BOTH).

The A (Harness integration) + D2 (verification layer) + D3 (RL pipeline) work
from this session is now demonstrated as a single coherent closed loop on a
real under-target kernel with a real frontier LLM.

---

## Artifacts

- `benchmarks/results/phase4/live/matmul_512/evidence.md` — run evidence card
- `.../trajectory.jsonl` — mineable v1.0 trajectory
- `.../rl_dataset.jsonl` — D3 RL corpus (step + trajectory samples)
- `.../result.json`, `.../state.json` — full result + resumable state

---

*Demo run 2026-07-13. Follow-up: drive the same loop through the CUDA-C backend
(`compile_and_profile(backend="cuda_c")`) to autotune the StrategyIR→CUDA-C
tile/TC selection with a live LLM.*
