# Arke Harness — Usage Guide

**Audience:** engineers and LLM agents driving Arke to auto-generate / auto-tune
GPU kernels. **Façade version:** v1.0.0 (locked 2026-05-18, contract
`arke-harness-facade-v1.0.0`).

The Harness is the **vendor-agnostic, agent-runtime-agnostic** surface of Arke.
It exposes exactly **8 tools** that any tool-use-capable LLM (Claude, GPT, or an
MCP client such as Claude Code / OpenClaw / Hermes / Cline) drives to optimize a
kernel through a compile → verify → profile → adjust loop.

---

## 1. Mental model

```
   ┌─────────────────────────── Façade v1.0 (8 frozen tools) ───────────────────────────┐
   │ get_hw_profile   analyze_compute   list_legal_actions   apply_decision              │
   │ verify_correctness   compile_and_profile   checkpoint   rollback                    │
   └───────────────────────────────────────────────────────────────────────────────────┘
              │ drives                                     ▲ observes
              ▼                                            │
   OptimizationState  ──(SemanticIR + StrategyIR)──►  Backend (triton | cuda_c | mlir_gpu)
              │                                            │
              └──────────► TrajectoryWriter (v1.0 JSONL) ◄─┘   → @rationale KB + D3 RL corpus
```

- **Bounded action space:** you may only apply moves returned by
  `list_legal_actions()`. The compiler computes what is legal given the current
  StrategyIR — you never invent a move.
- **@rationale required:** every `apply_decision` must carry a non-empty
  rationale (WHY, not just WHAT). Enforced by the A5 trajectory audit.
- **Staged verification:** V0 (syntax, in compile) → V1 (`verify_correctness`,
  numeric) → V2 (`compile_and_profile`, real GPU latency). Correctness gates
  performance — a fast-but-wrong kernel is rejected.
- **Budget:** decisions and compiles are metered. `get_hw_profile`,
  `analyze_compute`, `list_legal_actions`, `checkpoint`, `rollback` are free.

---

## 2. The 8 tools (frozen contract)

| Tool | Budget | Mutates strategy | Needs compile | Purpose |
|------|--------|:---:|:---:|---------|
| `get_hw_profile` | free | — | — | Target HW constraints (SM, smem, regs, warp size). |
| `analyze_compute` | free | — | — | Kernel compute characteristics (FLOPs, bytes, arithmetic intensity, reduction axes). |
| `list_legal_actions` | free | — | — | The compiler-legal move set for the current StrategyIR. |
| `apply_decision` | decision | ✅ | — | Apply one legal move + `@rationale`. Mutates StrategyIR. |
| `verify_correctness` | compile | — | ✅ | V1 numeric check vs reference. |
| `compile_and_profile` | compile | — | ✅ | V2 real-GPU latency + baseline_ratio + robust_reward. |
| `checkpoint` | free | — | — | Snapshot current strategy for safe exploration. |
| `rollback` | free | ✅ | — | Restore a prior checkpoint. |

**Concurrency:** the three `free`/idempotent read tools (`get_hw_profile`,
`analyze_compute`, `list_legal_actions`) are `concurrent_safe` and may be
batched in one turn. `apply_decision`/`checkpoint`/`rollback` mutate state and
must be serial.

---

## 3. Running an optimization loop

### 3a. Live-LLM driver (recommended entry point)

`benchmarks/live/run_live_optimize.py` wraps `LLMRunner.optimize()` and emits a
mineable trajectory + evidence card.

```bash
# CRITICAL: use the clean /v1 OpenAI-compatible endpoint. The bare Anthropic
# endpoint (https://yunwu.ai) injects ~30K tokens of Claude-Code context and
# hijacks custom tool-use. Always route local tool-use loops through /v1.
export ARKE_LLM_API_KEY="$OPENAI_API_KEY"
export ARKE_LLM_BASE_URL="https://yunwu.ai/v1"
export ARKE_LLM_PROTOCOL="openai"
export ARKE_LLM_MODEL="claude-sonnet-4-6"     # verified clean: claude-sonnet-4-6 / gpt-4o / deepseek-v3
export PATH=/usr/local/cuda/bin:$PATH          # nvcc for cuda_c backend

python -m benchmarks.live.run_live_optimize \
  --op matmul --shape 512,512,512 \
  --model yunwu/claude-sonnet-4-6 \
  --max-turns 15 --timeout 180 \
  --out benchmarks/results/phase4/live/matmul_512
```

Outputs in `--out`:
- `result.json` — full OptimizeResult (decisions, profiles, tokens, stop reason)
- `state.json` — resumable OptimizationState (S2 resume)
- `trajectory.jsonl` — v1.0 mineable trajectory (feeds KB + RL corpus)
- `evidence.md` — human-readable run card (A5 @rationale audit, GPU profiles)

**Resume a crashed run** (budget is not re-spent):
```bash
python -m benchmarks.live.run_live_optimize --op matmul --shape 512,512,512 \
  --resume-from benchmarks/results/phase4/live/matmul_512 --out <same-dir>
```

### 3b. Programmatic driver

```python
from arke.agent.llm_config import load_from_env
from arke.agent.runner import LLMRunner

config = load_from_env()          # reads ARKE_LLM_* / ANTHROPIC_* / OPENAI_*
with LLMRunner(config, timeout=180.0) as runner:
    result = runner.optimize(
        op_name="matmul",
        shapes={"A": [512, 512], "B": [512, 512]},
        target_hw="nvidia_ampere",
        max_turns=15,
        model_spec="yunwu/claude-sonnet-4-6",
        state_out="out/matmul_512",      # writes state.json for resume
    )
print(result.best_latency_ms, result.best_baseline_ratio)
```

`optimize()` accepts either `op_name` (+ optional `shapes`) or a `semantic_ir`
(IRGraph). Key kwargs: `max_turns`, `resume_from`, `state_out`, `on_event`
(streaming callback), `compact_after_chars` (context compaction), `hooks`.

---

## 4. Backend selection

`compile_and_profile` (and `verify_correctness`) default to the **Triton**
backend on CUDA. To drive the **CUDA-C** backend (Phase 4, 46/46 ops), pass a
`backend` param through the tool call:

```python
# inside the loop, the agent calls:
compile_and_profile(backend="cuda_c")   # or "triton" (default), "mlir_gpu"
```

The CUDA-C backend uses kernel-only CUDA-events timing via
`CudaCBackend.benchmark()` and honors StrategyIR decisions
(`MatmulConfig.from_strategy()` maps tile/unroll/algorithm=tensor_core into the
generated kernel). See `docs/phase4/stage-progress.md`.

Available backends: `triton` (Phase 1, validated), `mlir_gpu` (Phase 3, 1.03×
cuBLAS), `cuda_c` (Phase 4, 46/46 ops, 1.05× cuBLAS).

---

## 5. Verification tiers (D2 hardened)

`arke/agent/verification.py` provides borrowed SOTA mechanisms (Leon-approved
D2, 2026-07-12):

- **`robust_reward(correct, eager_ratio, strong_ratio)`** — discrete
  anti-reward-hacking reward tier:
  `-1` incorrect · `1` correct · `2` beats eager · `3` beats a strong baseline.
  A fast-but-wrong kernel always scores `-1`. Emitted in `compile_and_profile`
  output and recorded in the trajectory.
- **`staged_correctness_gate(...)`** — AutoKernel-style 5-stage firewall
  (smoke → shape-sweep → stability → determinism → edge).
- **`reflexion_feedback(stage, error_message, attempt, max_attempts)`** —
  GEAK-style categorized error feedback (compile / correctness / performance /
  timeout) for LLM self-correction on the next turn.

---

## 6. Learning artifacts

Every run emits a `trajectory.jsonl` (v1.0 record contract). Two miners consume it:

```python
# @rationale KB — decision → rationale → outcome
from arke.learn.rationale_kb import mine_trajectory, RationaleKB
entries = mine_trajectory("out/matmul_512/trajectory.jsonl")
RationaleKB().add_entries(entries)

# D3 RL corpus — (state, action, reward) samples with robust_reward labels
from arke.learn.rl_dataset import build_rl_dataset, extract_rl_samples
build_rl_dataset(["out/matmul_512/trajectory.jsonl"], "out/rl_corpus.jsonl")
```

See `docs/architecture/arke-rl-corpus-design.md` for the RL corpus schema and
training integration.

---

## 7. Quick reference — env & gotchas

| Item | Value |
|------|-------|
| venv | `source ~/.venvs/arke/bin/activate` |
| nvcc | `export PATH=/usr/local/cuda/bin:$PATH` (CUDA 13.2) |
| LLM endpoint | `https://yunwu.ai/v1` (OpenAI protocol) — **never** bare `https://yunwu.ai` for tool-use |
| clean models | `claude-sonnet-4-6`, `gpt-4o`, `deepseek-v3` |
| GPU | RTX 3060 Laptop, SM 8.6, 6 GB |
| contract test | `pytest tests/test_facade_contract_v1.py` (guards the frozen 8-tool surface) |

**Do not modify the Façade** (tool names/signatures/meta) — it is frozen at
v1.0. Additive changes only (new tools/event kinds) within MAJOR 1. Any contract
test failure signals an accidental Façade mutation.

---

*Last updated: 2026-07-13.*
