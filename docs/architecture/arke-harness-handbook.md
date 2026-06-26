# Arke Harness — 工程使用手册 (Engineering Handbook)

> The Arke Harness is the optimization layer that wraps an LLM (or a
> deterministic heuristic) around Arke's compiler so that **GPU/NPU kernel
> optimization becomes a bounded, auditable, resumable agentic loop**.
>
> This handbook documents the **shipped** Harness (Façade v1.0, `LLMRunner`,
> real Triton V1/V2 measurement). Proposed future capabilities live in
> `arke-harness-v2-rfc.md`. Style and section layout follow the Hermes Agent
> docs convention (Overview → Quick Start → Tools → Config → Cookbook →
> Troubleshooting → Reference).

**Version:** Façade `arke-harness-facade-v1.0.0` · Trajectory
`arke-trajectory-v1.0.0` · Events `arke-events-v1.0.0`
**Last updated:** 2026-06-26

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [The 8 Façade Tools](#3-the-8-façade-tools)
4. [Integration Modes (A / B / C)](#4-integration-modes)
5. [Configuration](#5-configuration)
6. [Live-LLM Autotuning (`LLMRunner`)](#6-live-llm-autotuning)
7. [Deterministic / Heuristic Path](#7-deterministic--heuristic-path)
8. [Trajectories & the @rationale KB](#8-trajectories--the-rationale-kb)
9. [Triton Kernel Generation & Tuning Cookbook](#9-triton-kernel-cookbook)
10. [Extending the Harness](#10-extending-the-harness)
11. [Troubleshooting](#11-troubleshooting)
12. [Reference](#12-reference)

---

## 1. Overview

### What the Harness is

Just as **Claude Code is not Claude** — it is the harness (tools, permissions,
memory, loop) that turns a chat model into an engineer — the **Arke Harness**
is everything that makes an LLM (Claude, GPT-4o, or a deterministic heuristic)
safe, productive, and reproducible at the task *"produce a high-performance,
correct GPU kernel"*.

The LLM is a **decision-maker, not a code generator**: it picks among legal
optimization moves the compiler surfaces; the compiler does the codegen and
the measurement.

### Two-layer architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1 — Public Façade  (vendor-agnostic, v1.0 FROZEN)      │
│   • 8 tools  • OptimizationEvent stream  • trajectory JSONL  │
│   Any MCP-compatible agent (Hermes, Claude Code, Cline…)     │
│   drives this surface. LLM provider is replaceable.          │
└───────────────────────────┬─────────────────────────────────┘
                            ↓ internal ABI (evolves per Phase)
┌─────────────────────────────────────────────────────────────┐
│ Layer 2 — Arke Substrate  (internal)                         │
│   • SemanticIR / StrategyIR  • V0/V1/V2 validators           │
│   • op registry + OT catalog  • HeuristicStrategyGenerator   │
│   • Triton backend codegen  • OptimizationState (ground truth)│
└─────────────────────────────────────────────────────────────┘
```

External agents talk **only** to the 8 Façade tools. They never import
`ArkeEnv`, `Decision`, or `StrategyIR` directly.

### Three roles of the LLM (do not collapse to #1)

1. **Runtime optimization decision-maker** — inside the Harness loop.
2. **Target user** of the Lang / IR / Compiler / Harness — the whole toolchain
   is designed for an Agent consumer.
3. **Builder** of the toolchain — the dev process is itself AI-Native.

---

## 2. Quick Start

### Prerequisites

- Linux / WSL2, Python 3.10+, NVIDIA GPU + CUDA (for V1/V2 on GPU).
- Dev reference box: RTX 3060 Laptop 6 GB (Ampere SM 8.6), CUDA 12.4.

### Install

```bash
git clone https://github.com/arke-lang/arke.git
cd arke
make setup            # or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

On the project dev box the venv is pre-built:

```bash
source ~/.venvs/arke/bin/activate     # MUST — pyenv 3.10.5; bare `python` may load wrong version
```

### First optimization run (deterministic, no LLM, no API cost)

```bash
arke optimize examples/operators/01_matmul.ak --cycles 3 --output /tmp/arke-matmul --json
```

Output (machine-readable):

```json
{
  "success": true,
  "kernel_id": "matmul",
  "cycles_completed": 3,
  "decision_count": 11,
  "best_score": 1.04,
  "strategy_path": "/tmp/arke-matmul/strategy.json",
  "akir_path": "/tmp/arke-matmul/result.akir",
  "trajectory_path": "/tmp/arke-matmul/trajectory.jsonl",
  "summary_path": "/tmp/arke-matmul/summary.json"
}
```

Four artifacts are written:

| File | What it is |
|:--|:--|
| `strategy.json` | The chosen StrategyIR (decisions + `@rationale`) |
| `result.akir` | The lowered `.akir` IR document |
| `trajectory.jsonl` | The replayable run record (`arke-trajectory-v1.0.0`) |
| `summary.json` | Budget, best result, decision log summary |

> **Note:** the deterministic path uses a *mock* profile formula (`best_score`
> is not real GPU latency). For real `baseline_ratio` use the live path (§6)
> or the benchmark harness (`benchmarks.bench_l1`).

---

## 3. The 8 Façade Tools

The Façade is **exactly 8 tools**, frozen at v1.0 (`arke/agent/facade.py`,
enforced by `tests/test_facade_contract_v1.py`). Ordered as the loop uses them:

| # | Tool | Purpose | Cost | Mutates strategy? |
|:-:|:--|:--|:--:|:--:|
| 1 | `get_hw_profile` | Hardware constraints (SM, shared mem, warps, dtypes) | ~0 | no |
| 2 | `analyze_compute` | Operator compute characteristics (FLOPs, mem-bound vs compute-bound, loops) | ~0 | no |
| 3 | `list_legal_actions` | The bounded set of legal optimization moves *right now* | ~0 | no |
| 4 | `apply_decision` | Apply one move **with mandatory `@rationale`** | low | **yes** |
| 5 | `verify_correctness` | V1 numeric check (Triton compile + ref compare on CUDA; `V0_mock` on CPU) | med | no |
| 6 | `compile_and_profile` | V2 real GPU latency + `baseline_ratio` | **high** | no |
| 7 | `checkpoint` | Snapshot state for safe exploration | ~0 | no |
| 8 | `rollback` | Restore a snapshot (undo a regression) | ~0 | yes (restores) |

### The `@rationale` contract

Every `apply_decision` should carry a human-readable `rationale` explaining
**why**, not just **what** — e.g. `"align tiles with the target's execution
structure to improve memory coalescing"`. The rationale is:

- Surfaced in `strategy.json` decisions and `trajectory.jsonl` `decision` records.
- Mined into the `@rationale` knowledge base for SFT/RL (§8).
- A locked thesis pillar (human-experience → LLM-optimization improvement loop).

### Budget awareness

Compile/profile (tools 5, 6) is expensive. The recommended pattern is **apply
1–3 related decisions, then immediately verify + profile** to close the loop on
a real number — do not stack many `apply_decision` calls before measuring. The
loop tracks `decision_max`, `compile_max`, and token budgets; when exhausted,
`stop_reason="budget_exhausted"`.

---

## 4. Integration Modes

All three modes consume the **same** 8-tool Façade.

| Mode | Who owns the loop | Entry point | Use case |
|:--|:--|:--|:--|
| **A. Built-in** | Arke | `arke optimize …` (CLI) / `arke.optimize(…)` (Python) | CI, batch, scheduled benchmark runs |
| **B. External agent** | The agent (Claude Code, Cursor, Hermes) | agent shells out to `arke <verb>`, reads JSON | AI assistants that own their own LLM access |
| **C. MCP server** | The MCP client (Cline, Continue, Hermes, Claude Desktop) | `arke mcp serve` | Any MCP client drives the 8 tools directly |

**Today:** Modes A and B ship. Mode C (MCP server) is proposed in the v2 RFC
(`arke-harness-v2-rfc.md` §3 N3).

### Mode A — Python API

```python
from arke.agent.runner import LLMRunner
from arke.agent.llm_config import load_from_env

config = load_from_env()           # resolves provider from env (§5)
with LLMRunner(config, timeout=300.0) as runner:
    result = runner.optimize(
        op_name="matmul",
        shapes={"A": [1024, 512], "B": [512, 2048]},
        target_hw="nvidia_ampere",
        max_turns=25,
        model_spec="yunwu/claude-sonnet-4-6",   # <provider-alias>/<model>
    )
print(result.to_dict())            # decisions, tool_calls, tokens, trajectory, …
```

### Mode B — shell out + read JSON

```bash
arke optimize examples/operators/06_rmsnorm.ak --cycles 3 --json > out.json
# parse out.json → strategy_path / trajectory_path
```

---

## 5. Configuration

### Environment-driven provider resolution

`load_from_env()` resolves the provider with this precedence (highest first):

1. Explicit kwargs to `load_from_env(api_key=…, base_url=…, model=…, protocol=…)`.
2. `ARKE_LLM_API_KEY` / `ARKE_LLM_BASE_URL` / `ARKE_LLM_MODEL` / `ARKE_LLM_PROTOCOL`.
3. `ANTHROPIC_API_KEY` (+ `ANTHROPIC_BASE_URL`) / `YUNWU_API_KEY`.
4. `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`).

`model_spec` forms:
- `"<alias>/<model>"` → that provider + that model (e.g. `yunwu/claude-sonnet-4-6`).
- `"<model>"` → primary provider + that model.
- `None` → primary provider + its default model.

### ⚠️ yunwu.ai endpoint rule (critical)

The **bare Anthropic endpoint** `https://yunwu.ai` injects a ~30K-token
Claude-Code system context into every request, which **hijacks the tool-use
loop** (the model calls injected tools instead of Arke's 8). The loader
auto-routes yunwu credentials through the **clean OpenAI-compatible `/v1`
surface** (`https://yunwu.ai/v1`, `protocol="openai"`), where `input_tokens`
drops from ~30,600 to ~500. **Always drive local LLM tool-use loops via `/v1`.**
Verified-clean models: `claude-sonnet-4-6`, `gpt-4o`, `deepseek-v3`.

### Model choice

Sonnet is the Phase-1 workhorse — fast enough for 25-turn tool-use loops.
**Opus is too slow** for long conversations. Default per protocol:
anthropic → `claude-sonnet-4-20250514`, openai → `gpt-4o`.

### Budget defaults

Set per-op in `ArkeEnv` (`decision_max`, `compile_max`). On a 6 GB box keep
shapes ≤ 2048 and treat OOM as a recorded non-blocking event.

---

## 6. Live-LLM Autotuning

`LLMRunner` (`arke/agent/runner.py`) is Arke's in-tree agent that puts a real
LLM in the driver's seat. Given a `SemanticIR` (or `op_name` + `shapes`) it:

1. Builds an `ArkeEnv` for the kernel.
2. Wires the locked 8-tool registry (`ToolRegistry.with_env(env)`).
3. Opens a tool-use conversation, exposing the 8 tools as function-calling
   schemas (anthropic *and* openai protocols supported).
4. Runs `propose → apply → verify → profile → adjust`, ≥3 profiled cycles,
   keeping the best correct strategy and rolling back regressions.
5. Records a **real trajectory** and returns an `OptimizeResult`.

```python
result = runner.optimize(op_name="rmsnorm", shapes={"X": [4096, 4096]},
                         max_turns=25, model_spec="yunwu/claude-sonnet-4-6")
# result.decisions, result.tool_calls, result.tokens_in/out,
# result.session_summary["best_performance"]["baseline_ratio"], result.stop_reason
```

**Real measurement (shipped):** `verify_correctness` performs a real Triton
compile + fp64-CPU-escape reference compare on CUDA (`V1_triton` tier);
`compile_and_profile` measures real Triton latency and computes
`baseline_ratio` via `benchmarks.measure.bench_fn`. On CPU these degrade to
`V0_mock`.

**Stop reasons:** `end_turn` (model done), `max_turns`, `budget_exhausted`,
`llm_error` (provider/network — all providers + retries exhausted).

**Provider resilience (S3, shipped 2026-06-26):** `load_from_env` auto-builds
a fallback chain from every resolved provider. On a *transient* error
(timeout / 429 / 5xx / connection), `LLMRunner` retries the same provider with
exponential backoff (1.5s, 3.0s), then fails over to the next same-protocol
provider, recording a `fallback{layer:"provider"}` entry in
`result.session_summary["fallback_events"]`. *Non-transient* errors (auth,
bad request, model-not-found) abort immediately — retrying won't help.

---

## 7. Deterministic / Heuristic Path

`HeuristicStrategyGenerator` (`arke/agent/optimize.py`) is the **always-on
floor**: a deterministic strategy generator that needs no LLM and no API cost.
Use it for:

- Reproducible CI / regression runs.
- The `@rationale` KB build (deterministic → 46 examples ≈ 292 entries).
- A guaranteed-correct kernel when the LLM path is unavailable.

```bash
arke optimize <op>.ak --cycles 3        # deterministic; mock profile
```

> The heuristic path's `@rationale` is written into `strategy.json` decisions
> (NOT into trajectory `decision` records — the trajectory only has
> `profile`/`adjust` for this path). Mine `strategy.json` for rationales.

---

## 8. Trajectories & the @rationale KB

### Trajectory schema (`arke-trajectory-v1.0.0`, frozen)

Every run yields a JSONL trajectory designed as a **first-class learning
artifact** for SFT / RL / `@rationale` accumulation. Same envelope
`{"t", "kind", "data"}` serves as both event stream and record log.

- First line: `header` (contract_id + SemanticIR snapshot).
- Body: strict `compile → profile → adjust` cycles.
- Last line: `done` (final_score, decisions, compiles, termination, chosen).

Record kinds = event kinds (9) ∪ `{header, adjust}`. Frozen by
`tests/test_facade_trajectory_contract_v1.py`.

### Building the @rationale KB

```bash
python -m benchmarks.build_rationale_kb     # runs arke optimize over all examples/operators/*.ak, mines rationales
```

Mines `strategy.json` decisions, pairs each with the run's best
`baseline_ratio` from the sibling `trajectory.jsonl`. Append-only;
dedup key = sha1(op, kind, params, rationale). Output:
`data/rationale_kb.jsonl`.

---

## 9. Triton Kernel Cookbook

### 9.1 The generation + tuning flow

```
.ak / op+shape → SemanticIR (WHAT)
   → [LLMRunner live | Heuristic floor]
   → 8 tools: hw_profile → analyze → list_legal_actions
        → apply_decision(@rationale) → verify (V1) → profile (V2)
        → checkpoint/rollback   (≥3 cycles)
   → best correct StrategyIR (HOW)
   → TritonBackend.lower → .triton kernel + launch config
   → trajectory.jsonl + strategy.json + result.akir
```

### 9.2 Worked example — matmul

```bash
arke optimize examples/operators/01_matmul.ak --cycles 3 --output /tmp/mm --json
read_strategy() { python -m json.tool /tmp/mm/strategy.json; }
```

Typical decisions: `tile(loop="i", factors=[64,16])`,
`tile(loop="j", factors=[128,8])`, `vectorize`, `place(shared)`.

### 9.3 Reading `baseline_ratio`

`baseline_ratio = arke_latency / reference_latency` is reported by V2. Per the
**Same-Backend Fairness** rule, the reference denominator is the **fastest
Triton-only** implementation of the op (FlagGems / Liger / Unsloth / vLLM /
flash-attn Triton variants), with `ε = 0.03`. A ratio ≤ 1.00 means Arke is at
or above parity; the Gate target for G6 is `≥1.00× P3`.

### 9.4 6 GB VRAM survival

- Keep shapes ≤ 2048 (batch=8 / seq=512 may OOM).
- Record OOM as a non-blocking event, do not silently drop the shape.
- `torch._dynamo.config.cache_size_limit` is bumped to 64 in `bench_l3` to
  avoid recompile thrash across seq-lens.

---

## 10. Extending the Harness

The HARNESS-3 extensibility contract (G8 Tier 1) proved two onboarding paths
with falsifiable LOC budgets:

### Onboard a new operator (≤400 LOC)

1. Edit the SSOT `docs/benchmark/benchmark-ops.md` (add the op + tier).
2. Register an `OpSchema` + a `ref_*` reference impl.
3. Register in `benchmarks/op_registry.py` (the SSOT parser).
4. Add a `SKILL.md` recipe + one audit entry; run through BL1.

> Demo A (`swiglu_packed`, commit chain on `feat/op-count-ssot`) onboarded the
> 46th catalog op end-to-end within budget.

### Onboard a new baseline runner (≤200 LOC)

Subclass the documented `BaselineRunner` protocol, plug into
`benchmarks/baselines/`. Demo B: `MaxAutotuneRunner` (141 LOC).

> **Sub-agent rule (Hermes file-tool discipline):** when delegating onboarding,
> pass file paths, error messages, and constraints explicitly — sub-agents have
> no conversation history. Use `write_file` (not heredoc), `patch` (not sed),
> `read_file` (not cat), `search_files` (not grep/find).

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|:--|:--|:--|
| LLM calls injected tools (`Skill`/`Glob`/`Bash`), `input_tokens` ~30K | Bare yunwu.ai Anthropic endpoint injects Claude-Code context | Route via clean `/v1` OpenAI surface (loader does this automatically for yunwu creds) |
| `verify_correctness` returns `correct=True, max_diff=0.0000` suspiciously | Stuck on `V0_mock` (CPU) or eager fallback (bridge) | Confirm CUDA available; check `tier` in result; for bridge confirm op input-name CASE (`X`/`W` not `x`/`weight`) |
| Reference latency looks like CPU fp64 | FlagGems `enable()` hijacks `aten::mm` globally | Use the CPU-fp64-escape reference path (built into V1) |
| `best_score` not real latency | Deterministic path uses `_mock_profile` | Use `LLMRunner` live path or `benchmarks.bench_l1` for real GPU numbers |
| Run aborts on one provider error | No fallback chain yet | (v2 S3 proposal) retry/backoff + heuristic floor degrade |
| OOM on large shape | 6 GB VRAM | Shapes ≤ 2048; record OOM non-blocking |
| `stop_reason="budget_exhausted"` | decision/compile/token budget hit | Raise budget in `ArkeEnv`, or accept best-so-far strategy |

---

## 12. Reference

### CLI verbs

```bash
arke compile <file.ak> [-o out.akir]     # compile .ak → Arke IR / .akir JSON
arke optimize <input> [--cycles N] [--kernel OP --shape S] [--dtype D]
                      [--target T] [--output DIR] [--json] [--dry-run]
```

`arke optimize` accepts: a `.ak` file path, inline `.ak` source, a
natural-language request, or a code snippet (multi-input routing).

### OptimizationEvent kinds (9, frozen `arke-events-v1.0.0`)

`header` · `decision` · `compile` · `profile` · `verify` · `checkpoint` ·
`rollback` · `fallback` · `done`
(authoritative list: `arke/agent/events.py`)

### Stop / termination reasons

`end_turn` · `max_turns` · `budget_exhausted` · `llm_error` ·
`llm_no_more_tool_use`

### Governance (frozen contracts — do not modify without project-lead ack)

- Façade v1.0 (8 tools) — `facade_v1_schema.json`, `test_facade_contract_v1.py`
- Event stream v1.0 — `test_facade_events_contract_v1.py`
- Trajectory v1.0 — `test_facade_trajectory_contract_v1.py`
- Same-Backend Fairness denominator (plan.md Locked Principle #2)
- Two-directional AI-Native thesis, three LLM roles (Locked Principle #6)

### Key source map

| Concern | File |
|:--|:--|
| Façade tool list + versioning | `arke/agent/facade.py` |
| Live LLM orchestrator | `arke/agent/runner.py` |
| Provider resolution | `arke/agent/llm_config.py` |
| 8 tools impl + ToolMeta | `arke/agent/tools.py` |
| Env / legal actions | `arke/agent/env.py` |
| OptimizationState / checkpoint | `arke/agent/state.py` |
| Events | `arke/agent/events.py` |
| Heuristic floor + CLI | `arke/agent/optimize.py` |
| Trajectory schema | `arke/learn/trajectory_schema.py` |
| @rationale KB | `arke/learn/rationale_kb.py` |
| Triton bridge (transient) | `arke/integration/torch_bridge.py` |

### Related docs

- `docs/architecture/arke-harness.md` — full architecture (v0.2 design)
- `docs/architecture/arke-harness-v2-rfc.md` — proposed v2 capability set
- `docs/architecture/e2e-flow.md` — end-to-end kernel generation flow
- `docs/roadmap/plan.md` — roadmap, Stages, Gates, locked principles
- `docs/benchmark/benchmark-design.md` — BL/OT/ST/L benchmark framework

---

*Arke Harness Handbook v1 — 2026-06-26. Documents Façade v1.0 as shipped.
Hermes-doc-style. For proposed future capabilities see arke-harness-v2-rfc.md.*
