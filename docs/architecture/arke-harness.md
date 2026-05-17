# Arke Harness

> The optimization harness that wraps an LLM (or a deterministic fallback) around
> Arke's compiler so that *kernel optimization* becomes a bounded, auditable,
> resumable agentic loop.
>
> Status: **v0.2 design** — supersedes the earlier `agent-design.md`. Implementation
> is partial; tracked under `docs/phase1/stage8-plan.md` Track 2 and the
> *Implementation status* matrix at the end of this document.

---

## 1. Why "Harness"?

Claude Code is not Claude. It is a **harness** that gives Claude a working set of
tools, a permission model, a project memory, lifecycle hooks, skills, and an
event-driven loop. Without the harness, the model is just a chat. With the harness,
it is an engineer.

Arke's optimization layer is structurally the same kind of thing. The model — Claude
Opus, GPT-4o, or a deterministic heuristic — is a decision-maker. **Arke Harness**
is everything that makes that decision-maker safe, productive, and reproducible
when the task is "produce a high-performance GPU kernel":

- A bounded action space sourced from the compiler (`list_legal_actions`).
- A three-tier validator that catches illegal moves *before* they touch state.
- Persistent ground-truth state (`StrategyIR`, decision log, best result) that
  survives compaction, provider fallback, and rollback.
- A trajectory recorder so every run is replayable and learnable.
- Hooks, skills, and subagents — borrowed wholesale from the Claude Code playbook.

We rename the abstraction from "Arke Agent" to **"Arke Harness"** to make this
relationship explicit and to free the word *agent* for the LLM-using-Arke role
documented in `AGENTS.md`.

### What we borrow from Claude Code

| Claude Code primitive | Why it fits Arke |
|---|---|
| Permission model (allowlist/denylist) | Arke already has the strongest version: legality is computed by the compiler |
| Hooks (`PreToolUse`, `PostToolUse`, `SessionEnd`) | Map naturally to `PreDecision`, `PostCompile`, `PostProfile`, `OnRollback` |
| Skills (`SKILL.md` + procedure) | Encodes recurring optimization recipes (e.g., "sweep one operator across all shape tiers") |
| Subagents (`Agent` tool) | Parallel exploration of design space (tile-size sweep, backend comparison) |
| MCP servers | Lets *any* MCP-compatible client drive Arke; symmetric to how Claude Code consumes MCP |
| `settings.json` | We use `arke.config.yaml` with the same layered-defaults idea |
| `TodoWrite` / Plan mode | Mirrored by `checkpoint` / `rollback` and the immutable decision log |

### What is different

| Aspect | Claude Code | Arke Harness |
|---|---|---|
| Permission source | User-configured allowlist | Compiler-computed legal actions (no user list to maintain) |
| Tool effects | Filesystem + shell — usually reversible | GPU compile + profile — *expensive*, must be budgeted |
| Verifier | Tests (run by user) | V0 static + V1 numerical + V2 performance (built into the loop) |
| "Truth" | The repository on disk | `OptimizationState` — kept outside the message log |
| Failure floor | None — model gives up | Compiler-generated heuristic strategy is always available |

---

## 2. Design Goals

1. **LLM as decision-maker, not code generator.** The LLM picks among legal moves
   surfaced by the compiler; the compiler does the codegen and the measurement.
2. **`@rationale` is a contract.** Every `Decision` carries a human-readable
   justification. Asserted by `tests/test_rationale_e2e.py`. Surfaced in trajectories.
3. **Bounded action space.** No free-form code generation. The set of legal moves
   for any given `(SemanticIR, StrategyIR, target)` is finite and enumerable.
4. **Hardware-aware, GPU-budget-aware.** Compile/profile is expensive — track and
   cap it. Decisions are cheap; profiles are not.
5. **Trajectory as a first-class output.** Every run yields a JSONL trajectory
   suitable for offline supervised fine-tuning, reinforcement learning, and
   `@rationale` knowledge-base accumulation.
6. **Graceful degradation.** Provider rate-limits, compaction, and LLM disagreement
   never produce *no* kernel — the heuristic floor always wins by default if the
   LLM does not improve on it.

---

## 3. Architecture Overview — Two-Layer Design

> **Locked principle (2026-05-17, Leon-approved):** Arke Harness is a **two-layer system**, not a monolith. The split between *Public Façade* (vendor-agnostic public contract) and *Arke Substrate* (Arke-internal compiler/IR coupling) is the **defining architectural commitment** that lets Arke simultaneously:
>
> 1. Plug into any MCP-compatible agent runtime (Claude Code, OpenClaw, Hermes, Cline, Continue, future Anthropic agents, custom in-house agents) — via the Public Façade.
> 2. Retain deep IR/Compiler/V0-V1-V2 coupling for decision quality and cross-architecture portability — via the Arke Substrate.
>
> Without this split, Arke either (a) becomes "yet another vendor-locked optimizer" with no ecosystem reach, or (b) gets dragged into the lowest common denominator of public agent API contracts and loses the IR-coupling advantage. The two-layer design preserves both.

### 3.0 The two layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1: Public Harness Façade                                      │
│  ─────────────────────────────                                       │
│  Vendor-agnostic, stable contract. Any MCP-compatible agent          │
│  (Claude Code, OpenClaw, Hermes, Cline, Continue, …) talks to this   │
│  surface. Version-locked. Backward-compatible across Arke releases   │
│  within the same Façade major version.                               │
│                                                                      │
│  • 8 tools (§6) — `ToolMeta`-described, JSON-schema discoverable     │
│  • OptimizationEvent stream (§4) — AsyncGenerator                    │
│  • Trajectory JSONL (§15) — schema `s8-compile-profile-adjust-v1`    │
│  • SKILL.md (§11) — Claude-Code-compatible recipe format             │
│  • Hook spec (§12) — 8 lifecycle points, MAY register externally     │
│  • MCP transport (§14) — stdio / sse, surfaces Tools/Resources/Prompts│
│  • CLI verbs (`arke optimize`, `arke bench`, `arke mcp serve`)        │
│  • Python API (`arke.optimize(...)`)                                  │
│                                                                      │
│  ↑ LLM provider is REPLACEABLE: Anthropic / OpenAI / OSS / in-house  │
│  ↑ Agent runtime is REPLACEABLE: any MCP-speaking client             │
└──────────────────────────────────────────────────────────────────────┘
                          ↓  (internal ABI — may evolve per Stage)
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2: Arke Substrate                                             │
│  ───────────────────────                                             │
│  Arke-internal. Not exposed as a stable public API. Free to evolve   │
│  across Phases (Triton → MLIR → vendor-DSL → LLVM-IR).                │
│                                                                      │
│  • SemanticIR / StrategyIR (docs/spec/arke-ir-spec-design.md)        │
│  • Op registry + 45-op OT catalog (benchmarks/op_registry.py)        │
│  • V0 (static) / V1 (numeric) / V2 (profile) validators              │
│  • HeuristicStrategyGenerator — the always-on floor                  │
│  • Backend codegen: Triton (Phase 1) → Ascend → MLIR → DSL → LLVM-IR │
│  • Per-target hardware envelopes + legality computation              │
│  • OptimizationState (ground truth outside the message log)          │
└──────────────────────────────────────────────────────────────────────┘
                          ↓
                  GPU / NPU / CPU hardware
```

### 3.0.1 What may cross the layer boundary

The Façade-to-Substrate boundary is enforced by **`arke/agent/tools.py` (ABC) + JSON-schema tool descriptors**. Crossing rules:

| Direction | Allowed | Forbidden |
|:---|:---|:---|
| Façade → Substrate | Tool calls (8 tools), config (§17), hooks | Direct `StrategyIR` mutation, direct op-registry inspection, raw codegen invocation |
| Substrate → Façade | Tool results (typed), `OptimizationEvent`s, structured errors with legal alternatives | Substrate-internal types (`Decision`, `StrategyIR`, op-registry handles) as opaque blobs in tool results |

**Rule of thumb:** if a third-party agent (Claude Code, OpenClaw) needs it, it crosses the boundary as a typed Façade artifact. If only Arke's own compiler/codegen consumes it, it stays in the Substrate.

### 3.0.2 Versioning policy

- **Façade contract** is versioned `arke-harness-facade-vX.Y.Z`. Within `X`, all changes are backward-compatible (add tools, add event kinds, add hook points — never remove or break-signature). Breaking changes bump `X`.
- **Substrate ABI** has no public contract. It evolves at Stage cadence. Each Phase may rewrite it entirely (e.g. Phase 3 introduces MLIR dialect; Substrate ABI re-shaped accordingly). The Façade absorbs the change.
- **Compatibility test suite** verifies that every Façade version supports the locked 8-tool semantics, event stream, and trajectory schema. Lives in `tests/test_facade_contract_v*.py` (added Stage 9).

### 3.1 The three integration modes (all on the same Façade)

| Mode | Owns the loop | Entry point | Use case |
|---|---|---|---|
| **A. Built-in** | Arke | `arke optimize ...` (CLI) / `arke.optimize(...)` (Python) | Automated CI, batch optimization, scheduled benchmark runs |
| **B. External agent** | The agent (Claude Code, Cursor, etc.) | Agent shells out to `arke <verb>` and reads JSON | AI dev assistants that already own LLM access and project context |
| **C. MCP server** | The MCP client (Claude Desktop, Cline, Continue, OpenClaw, Hermes, …) | `arke mcp serve --target ampere` | Any MCP-compatible client drives the 8-tool surface directly — no shell intermediation |

Modes A and B exist today (B at MVP fidelity via the heuristic generator). Mode C is a planned addition that closes the gap between Arke and the broader MCP-tooling ecosystem.

All three modes consume the **same Façade** — same 8 tools, same events, same trajectory schema. The Substrate sees no mode distinction.

### 3.2 Shared substrate

```
                    ┌─────────────────────────────────────┐
   Mode A  ───────► │              The Loop               │
   Mode B  ───────► │  (AsyncGenerator[OptimizationEvent])│
   Mode C  ───────► │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    Tools (8)    │  declarative ToolMeta
                              │  partitioned    │  (concurrent_safe,
                              │  by ToolMeta    │   mutates_strategy, …)
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │     ArkeEnv     │  current StrategyIR,
                              │  + HW profile   │  HW descriptor, budget
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
        ┌──────────┐            ┌──────────┐              ┌──────────┐
        │  V0      │            │  V1      │              │  V2      │
        │ static   │            │ numeric  │              │ profile  │
        │ ~0 ms    │            │ ~100ms-1s│              │ 1-5 s    │
        └──────────┘            └──────────┘              └──────────┘
                                       │
                                       ▼
                              ┌────────────────────┐
                              │ OptimizationState  │  ground truth,
                              │  strategy_ir       │  outside messages
                              │  decision_log      │
                              │  compile_results   │
                              │  best_result       │
                              └────────┬───────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Trajectory    │  JSONL,
                              │  (PostHook      │  s8-compile-profile-
                              │   writer)       │  adjust-v1
                              └─────────────────┘
```

Hooks tap every arrow. Skills inject prompts and shortcuts. Subagents fork the
substrate for parallel exploration with their own scoped state.

---

## 4. The Loop

The harness runs an **`AsyncGenerator[OptimizationEvent, None]`** — a single
primitive that the CLI, the Python API, the REST surface, the Jupyter renderer,
and the MCP server all consume.

```python
async for event in harness.run(env, llm, config):
    match event.kind:
        case "decision":  ...
        case "compile":   ...
        case "profile":   ...
        case "verify":    ...
        case "rollback":  ...
        case "compact":   ...
        case "fallback":  ...   # provider switched
        case "done":      ...
```

Termination conditions, in priority order:

1. LLM emits no further `tool_use` (signals completion) — finalize.
2. Decision budget or compile budget exhausted — inject "budget exhausted"
   message, request final summary, finalize.
3. The caller calls `agen.aclose()` — propagate cancellation, persist state,
   exit.
4. Hard error (provider unreachable after fallback chain exhausted, V2 baseline
   measurement fails) — finalize with the best result so far.

Today: a synchronous turn loop drives the heuristic path in
`arke/agent/optimize.py`. The AsyncGenerator form is the migration target — see
§18.

---

## 5. Bounded Action Space — Arke's Permission System

Claude Code's permission model relies on a user-curated allowlist plus runtime
prompts. Arke's permission model is **stronger and lower-overhead**: legality is
*computed* from the IR + hardware target, so the user maintains nothing.

### 5.1 The contract

`list_legal_actions(strategy_ir, hw_profile) → list[LegalAction]` is the **only**
source of truth. The LLM may not propose a decision outside this set; if it tries,
V0 rejects the call before any state mutates and returns a structured error with
the legal candidates inline.

### 5.2 What "legal" means

A move is legal iff all of:

- The decision kind is recognized by the op registry (`arke/ir/ops/registry.py`).
- The parameter values are inside the per-target hardware envelope (shared memory
  size, register pressure estimate, tensor-core shape multiples, warp size).
- The structural invariants of the current StrategyIR are preserved (no double
  fusion, no cyclic schedule, alignment math closes).

### 5.3 Comparison

| Aspect | Claude Code permission | Arke bounded action space |
|---|---|---|
| Source | User config (`settings.json` allowlist) | Compiler-computed at every step |
| Runtime check | Match against allowlist | V0 static gate (no runtime list to scan) |
| User maintenance | Manual list per project | Zero — the compiler knows |
| Failure mode | Prompt user, possibly approve | Auto-reject + return legal alternatives |

### 5.4 Why this matters

Because legality is computed, the harness can present the LLM with a *small*
top-N list rather than a flat allowlist. `list_legal_actions` returns
top-10-by-heuristic-score plus a `total_count` and a `kind=` filter for the
remainder. This compresses what would otherwise be a token-blowing combinatorial
enumeration.

---

## 6. Tools

Eight tools form the agent-facing surface. Their authoritative declaration lives
in `arke/agent/tools.py` (`ToolMeta` + `ArkeTool` ABC); this section is the
contract that file implements. *This section subsumes the former `agent-design.md
§5.1 Tool Declarative Interface`.*

### 6.1 The eight tools

| # | Tool | Purpose | Mutates state | Budget bucket |
|---|---|---|---|---|
| 1 | `get_hw_profile` | Returns target descriptor (SM count, shared mem, TC shape) | no | free |
| 2 | `analyze_compute` | Returns kernel characterization (compute/memory bound, intensity) | no | free |
| 3 | `list_legal_actions` | Returns legal next-decisions (top-N + filter) | no | free |
| 4 | `apply_decision` | Mutates StrategyIR with a single legal `Decision` | **yes** | decision |
| 5 | `verify_correctness` | V1 numeric check vs. NumPy reference | no | compile |
| 6 | `compile_and_profile` | V2 GPU compile + microbench vs. baseline | no | compile |
| 7 | `checkpoint` | Snapshot current StrategyIR + best metrics under a label | no | free |
| 8 | `rollback` | Restore a previous checkpoint | **yes** | free |

### 6.2 `ToolMeta` schema

Every tool self-declares (see `arke/agent/tools.py`):

```python
@dataclass(frozen=True)
class ToolMeta:
    concurrent_safe: bool      # may run in asyncio.gather batch with others
    idempotent: bool           # safe to retry on failure
    requires_compile: bool     # touches GPU
    mutates_strategy: bool     # changes StrategyIR — breaks concurrent batches
    budget_type: BudgetType    # FREE | DECISION | COMPILE
    cost: CostLevel            # CHEAP | MEDIUM | EXPENSIVE
```

### 6.3 Concurrent partitioning rule

Consecutive `concurrent_safe=True` tool calls batch through `asyncio.gather`. Any
`concurrent_safe=False` (state-mutating or GPU-using) call breaks the batch and
runs serially. Example partition:

```
[analyze_compute, get_hw_profile, apply_decision, list_legal_actions,
 compile_and_profile]

  → Batch([analyze_compute, get_hw_profile])    concurrent
  → Serial([apply_decision])                     mutates StrategyIR
  → Batch([list_legal_actions])                  concurrent (singleton)
  → Serial([compile_and_profile])                GPU
```

Adding a new tool is safe: declare its `ToolMeta`, the orchestrator does the rest.

---

## 7. System Prompt — Segmented Cache Topology

The system prompt is **four segments** with independent `cache_control`. The
segmentation matches change frequency, which maximizes Anthropic prompt-cache
hit rate across a multi-turn session.

| Seg | Content | Changes when | `cache_control` |
|---|---|---|---|
| 1 | Role + global kernel-optimization knowledge | Never (per Arke release) | global ephemeral |
| 2 | Hardware profile (SMs, shared mem, TC shape, peak TFLOPS) | Per `--target` | hardware-scoped |
| 3 | Semantic IR + auto-analysis of the kernel | Per kernel | kernel-scoped |
| 4 | Current StrategyIR + budget + decision log tail | Every turn | uncached |

Segments 1–3 hit cache on N-1 of N turns. For a 50-turn optimization, the savings
(per [token-efficiency-analysis.md](token-efficiency-analysis.md)) are on the
order of 175K input tokens vs. an unsegmented prompt.

**Persona injection.** Mode A also prepends `AGENTS.md`, `IDENTITY.md`, `SOUL.md`,
and `TOOLS.md` from the repo root into Segment 1 (or Segment 2 for the
hardware-flavored portions of `TOOLS.md`). Mode B agents read those files
themselves; the contract is the same in both directions.

---

## 8. State — Ground Truth Outside the Messages

`OptimizationState` is the **only** authoritative record of what has happened.
Messages are lossy — they get compacted, retried after provider fallback,
truncated by context limits. State is not.

```
OptimizationState
├── strategy_ir       : StrategyIR             # always complete, never summarized
├── decision_log      : list[Decision]         # all decisions with rationale
├── compile_results   : list[CompileResult]    # all V2 attempts
├── best_result       : CompileResult | None   # tracked separately for finalization
├── checkpoints       : dict[label, snapshot]
└── budget            : OptimizationBudget     # decisions_used, compiles_used
```

**Compact** rewrites the *messages*. It never touches state. After compact, the
authoritative state is re-injected into the new message context as a single
"current state" block, so the LLM resumes with full ground truth even if the
conversation was clipped.

This is what guarantees the final `compare(llm_best, fallback)` is always made on
complete data, regardless of how chaotic the conversation was.

---

## 9. Validation — V0 / V1 / V2

| Tier | When | Cost | What it checks | Failure handling |
|---|---|---|---|---|
| **V0** static | Every `apply_decision`, automatic | ~0 ms | Bounded-action membership, alignment, shared-mem bounds, structural invariants | Auto-reject with structured error + alternatives; rollback not needed (state never mutated) |
| **V1** numeric | On `verify_correctness` | ~100 ms – 1 s | NumPy reference comparison, max abs/rel error, NaN/Inf detection | Return error details; LLM decides whether to rollback |
| **V2** profile | On `compile_and_profile` | 1 – 5 s | Real GPU latency, TFLOPS, roofline efficiency, vs. vendor baseline (cuBLAS, cuDNN) | Return numbers; LLM decides next move |

`verify_correctness` is two-stage internally: NumPy fast-path first, then a GPU
compile-and-run when the strategy is potentially numerically distinct from the
reference (e.g., reduced-precision accumulators).

---

## 10. Budget & Compact

### 10.1 Budget

```
OptimizationBudget
├── decisions_used / max_decisions   default: 50    # apply_decision calls
└── compiles_used  / max_compiles    default: 10    # verify_correctness + compile_and_profile
```

Budget is injected into every tool result so the LLM always sees remaining
headroom. When either limit is reached, the harness injects a "budget exhausted —
finalize now" message and accepts only `checkpoint` / `rollback` thereafter.

### 10.2 Compact — two triggers, one function

Both call `compact_optimization_context(messages, state)`:

- **Predictive.** Token estimate before the next LLM call exceeds 80 % of the
  context window → compact proactively.
- **Reactive.** Provider returns `prompt_too_long` / `context_length_exceeded` →
  catch, compact, retry transparently.

Compact policy:

1. Summarize the middle of the message history (decisions, outcomes, errors).
2. Preserve the **last 5 messages** verbatim.
3. Re-inject the full ground-truth state from `OptimizationState`.

Tracked as a `compact` event so the caller (and the trajectory writer) can see
it happen.

---

## 11. Skills

A **skill** is a packaged optimization recipe. The harness loads one when the
kernel, the user request, or the current state matches its trigger. The format
mirrors Claude Code skills exactly: a directory under `skills/<name>/` with a
`SKILL.md` (frontmatter + procedure), optional `references/`, optional `scripts/`.

The existing `skills/arke-test-coverage/SKILL.md` is the template.

### 11.1 `SKILL.md` schema

```yaml
---
name: operator-coverage
when: |
  Kernel is a single op (OT0/OT1/OT2) and the user asks to "sweep all shapes" /
  "validate across tiers" / triggers `arke optimize ... --skill operator-coverage`.
inputs:
  - operator     # e.g., "matmul"
  - shape_tier   # ST1..ST4
budgets:
  decisions: 80
  compiles: 24
references:
  - benchmarks/bench_l1.py
  - docs/benchmark/benchmark-design.md
---

## Procedure

1. Resolve the shape set for the requested ST tier from `bench_l1.py`.
2. For each shape: invoke the inner harness with budget shares.
3. Aggregate per-shape `best_result` into a coverage report (CSV column set:
   `s8-coverage-v1`).
4. If geomean speedup < 0.95×, surface a bottleneck-shape recommendation in the
   trajectory.
```

### 11.2 Built-in catalog (planned)

| Skill | Trigger | What it does |
|---|---|---|
| `operator-coverage` | "sweep one op across all shapes" | Drive a per-shape inner harness session, aggregate |
| `bench-gate` | `--skill bench-gate G6` (or G7/G8) | Wrap `python -m benchmarks.gate G<n>` with structured artifacts |
| `tier-promotion` | Kernel passing at ST1, want ST2/3 | Re-run with promoted shape tier, compare regressions |
| `flash-attn` | OT4 op (`flash_attention`, `paged_attention`) | Inject FA-specific decision priors (block size, KV layout) |

### 11.3 Discovery

At session start, the harness scans `skills/*/SKILL.md`, parses frontmatter, and
keeps only a *one-line digest* (`name`, `when`) in Segment 1 of the system
prompt. The full body of a skill loads only when triggered, keeping prompt
overhead bounded regardless of how many user-authored skills coexist.

---

## 12. Hooks

Hooks are user-supplied callables invoked at fixed lifecycle points. Same model as
Claude Code hooks; the difference is *which* events Arke surfaces.

### 12.1 Lifecycle points

| Hook | Fires | Payload | May abort? |
|---|---|---|---|
| `PreDecision` | Before `apply_decision` runs V0 | `(decision, state)` | yes (raises → V0-style reject) |
| `PostDecision` | After `apply_decision` succeeds | `(decision, state, diff)` | no (annotation only) |
| `PreCompile` | Before `compile_and_profile` | `(strategy_ir, hw_profile)` | yes |
| `PostCompile` | After codegen, before profile run | `(kernel_blob, build_log)` | yes |
| `PostProfile` | After V2 measurement | `(profile_result, baseline)` | no |
| `OnRollback` | After `rollback` restores a checkpoint | `(label, restored_state)` | no |
| `OnCompact` | After messages compacted | `(removed_count, kept_count)` | no |
| `OnSessionEnd` | At loop termination (any cause) | `(final_state, exit_reason)` | no |

### 12.2 Use cases

- **Default trajectory writer** is itself a hook bundle — `PostDecision`,
  `PostCompile`, `PostProfile`, `OnRollback`, `OnCompact`, `OnSessionEnd` all
  emit JSONL records. This means *removing* trajectory recording is just
  unregistering the bundle; no special-case code path.
- **External monitoring.** A `PostProfile` hook ships `(latency_ms, gflops)` to
  Prometheus / OpenTelemetry.
- **Custom validation gate.** A `PreCompile` hook that rejects strategies whose
  estimated register pressure exceeds a project-specific limit.
- **Auto-checkpoint policy.** A `PostProfile` hook that calls `checkpoint(label)`
  whenever a new best result is reached.

### 12.3 Configuration

Hooks register via `arke.config.yaml` (see §17) or programmatically in Python:

```python
harness.hooks.register("PostProfile", my_prometheus_pusher)
harness.hooks.register("PreCompile", reject_high_register_pressure)
```

A failing hook *with `may_abort=True`* surfaces as a structured error to the LLM,
who decides the next move (typically retry with different parameters or rollback).

---

## 13. Subagents

Some optimization sub-problems are *embarrassingly parallel*: a tile-size sweep
across `{32, 64, 128, 256}`; a comparison of Triton vs. MLIR-backed lowering for
the same kernel; an A/B test of two fusion choices. The harness supports these
as **subagent sessions** — child harness instances that inherit a forked
`OptimizationState` and report results back.

### 13.1 Contract

```python
results = await harness.spawn_subagents(
    parent_state=state,
    variants=[
        {"name": "tile_M=32",  "decisions": [{"kind": "tile", "dim": "M", "factor": 32}]},
        {"name": "tile_M=64",  "decisions": [{"kind": "tile", "dim": "M", "factor": 64}]},
        {"name": "tile_M=128", "decisions": [{"kind": "tile", "dim": "M", "factor": 128}]},
        {"name": "tile_M=256", "decisions": [{"kind": "tile", "dim": "M", "factor": 256}]},
    ],
    per_child_budget=Budget(decisions=5, compiles=2),
    select="best_by_gflops",
)
```

Each child:

- Receives a *deep copy* of `OptimizationState` (no mutation leaks back).
- Has its own decision and compile budget, deducted up-front from the parent's
  pool.
- Runs the same loop, with the same hooks, but writes its trajectory under
  `<run>/subagents/<child_name>/trajectory.jsonl`.
- Returns a `SubagentResult` with its `best_result` and full final state.

The parent picks the winner per the `select=` strategy (`best_by_gflops`,
`best_by_latency`, or a custom callable) and merges the winner's
`OptimizationState` back into its own.

### 13.2 Why this beats sequential exploration

A sequential harness loop must *roll back* between variants, which costs
decisions and can't run compiles in parallel. Subagents let the LLM say "try all
four and keep the best" in one tool call, with `compile_and_profile` running
under `asyncio.gather` constrained by GPU concurrency settings.

### 13.3 Read-only mode

Subagents may be marked `read_only=True` to forbid `apply_decision`. This turns
them into *probes* — useful for "what would `analyze_compute` say if I imagined
a different shape?" without spending decision budget.

---

## 14. Mode C — MCP Server

`arke mcp serve` exposes the harness as a Model Context Protocol server. Any
MCP-compatible client (Claude Desktop, Cline, Continue, future Anthropic agents,
custom clients via the MCP SDK) can drive Arke without shell intermediation.

### 14.1 Surface

| MCP feature | Arke mapping |
|---|---|
| **Tools** | The 8 harness tools (§6). Schemas auto-derived from `ToolMeta` + the JSON-schema annotations on each `ArkeTool` subclass. |
| **Resources** | `arke://kernels/<id>` (45-op catalog), `arke://hw/<target>` (hardware profile), `arke://trajectory/<run>` (past run JSONL). |
| **Prompts** | Each built-in skill (§11) is exposed as an MCP prompt with the same `when`/`inputs` schema. |
| **Sampling** | Optional. When enabled, the harness can request the *client's* LLM for sub-decisions, eliminating the need for an Arke-side API key. |

### 14.2 Transport

| Transport | Use case |
|---|---|
| `stdio` | Local clients (Claude Desktop config, IDE plugins) |
| `sse` | Remote / multi-tenant; requires `--auth-token` |
| `http` | Future, when the MCP HTTP transport stabilizes |

### 14.3 Why a separate mode

Mode B already lets external agents shell out to `arke`. MCP is *better* for
agents that natively speak the protocol because:

- Tool schemas are typed and discoverable; no agent-side parsing of CLI output.
- Streaming events (the AsyncGenerator from §4) become MCP notifications — the
  client sees `decision`, `compile`, `profile` events in real time.
- Session state lives in the MCP server, so the client can disconnect and
  reconnect without losing the run.

Mode C is symmetric to how Claude Code consumes MCP servers from its side.

---

## 15. Trajectory & Learning

Every run produces `trajectory.jsonl` (schema `s8-compile-profile-adjust-v1`).
Records are emitted by the default hook bundle (§12) — never by special-case code.

```jsonl
{"t": 0.012, "kind": "decision",  "data": {"decision": {...}, "rationale": "..."}}
{"t": 0.014, "kind": "verify",    "data": {"v0": "pass"}}
{"t": 0.241, "kind": "compile",   "data": {"backend": "triton", "build_ms": 227}}
{"t": 1.840, "kind": "profile",   "data": {"latency_ms": 0.41, "vs_baseline": 1.18}}
{"t": 1.841, "kind": "checkpoint","data": {"label": "best", "score": 1.18}}
{"t": 2.001, "kind": "compact",   "data": {"removed": 12, "kept": 5}}
{"t": 2.512, "kind": "done",      "data": {"final_score": 1.18, "decisions": 17, "compiles": 4}}
```

Downstream consumers:

- **SFT corpus.** `(state_before → decision → outcome)` triples extracted from
  trajectories train a model to imitate successful sequences.
- **RL signal.** Final `vs_baseline` score is the reward; intermediate
  `vs_baseline` improvements are dense rewards.
- **`@rationale` knowledge base.** Decisions whose `vs_baseline` improved are
  archived under their `kind` (e.g., `tile`, `fuse`) so future runs can retrieve
  precedent rationales for similar contexts.

---

## 16. Fallback Chain

The harness has three layers of fallback. Each emits an event so the caller knows
which layer is in use.

### 16.1 Strategy fallback — the floor

`arke/agent/optimize.py::HeuristicStrategyGenerator` produces a *valid, if
unoptimized* `StrategyIR` from the SemanticIR with **no LLM in the loop**. This
is the floor — every run finishes with at least this strategy available.

At finalization:

```
if llm_best is None or llm_best.score <= heuristic.score:
    final = heuristic
    note  = "LLM did not improve over heuristic floor"
else:
    final = llm_best
```

### 16.2 Provider fallback — the chain

```yaml
llm:
  primary:   anthropic/claude-opus-4-7
  fallbacks:
    - anthropic/claude-sonnet-4-6
    - openai/gpt-4o
```

On `RateLimitError`, persistent timeout, or 5xx, the runner switches to the next
entry. Each switch emits a `fallback` event. Order is most-capable-first so the
optimizer degrades gracefully.

### 16.3 Tool-call fallback — the retry

Idempotent tools (`ToolMeta.idempotent=True`) auto-retry on transient failure
with exponential backoff. Mutating tools do not auto-retry — the LLM is asked to
re-decide.

---

## 17. Configuration — `arke.config.yaml`

One file, layered defaults: `~/.arke/config.yaml` → `./arke.config.yaml` → CLI
flags → env vars. Same precedence order as Claude Code's `settings.json`.

```yaml
llm:
  primary: anthropic/claude-opus-4-7
  fallbacks:
    - anthropic/claude-sonnet-4-6
    - openai/gpt-4o

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    timeout_seconds: 300
  openai:
    api_key: ${OPENAI_API_KEY}

harness:
  budget:
    decisions: 50
    compiles: 10

  hooks:
    PostProfile:
      - module: arke.hooks.prometheus
        config: { gateway: "http://pushgw.local:9091" }
    PreCompile:
      - module: my_project.hooks
        callable: reject_if_register_pressure_above
        config: { limit: 96 }

  skills:
    enabled:
      - operator-coverage
      - bench-gate
      - flash-attn
    user_skill_dir: ./skills

  mcp:
    serve:
      transport: stdio        # or "sse"
      target: ampere
      auth_token: ${ARKE_MCP_TOKEN}    # required for sse

  compact:
    predictive_threshold: 0.80
    keep_last_messages: 5
```

The loader is intentionally tiny: `pydantic` schema for validation, env-var
interpolation, layered merge, no plugins. Anything plugin-shaped lives in hooks
or skills.

---

## 18. Implementation Status & Roadmap

Honest assessment as of 2026-05-10. Contract is **frozen** for §1–§10, §15, §16;
§11–§14, §17 are **design** and tracked in `stage8-plan.md` Track 2.

### 18.1 Status matrix

| Area | State | Where it lives |
|---|---|---|
| Bounded action space | ✅ implemented | `arke/agent/tools.py`, `list_legal_actions` impls |
| ToolMeta declarative interface | ✅ implemented | `arke/agent/tools.py` |
| Heuristic strategy floor | ✅ implemented | `arke/agent/optimize.py::HeuristicStrategyGenerator` |
| Trajectory (`s8-compile-profile-adjust-v1`) | ✅ implemented | `arke/learn/trajectory.py` |
| `arke optimize` CLI MVP | ✅ implemented | `arke/agent/optimize.py` (deterministic path) |
| Sync turn loop with LLM | 🚧 partial | aspirational `LLMRunner` |
| AsyncGenerator loop | ⬜ planned | Migration M1 |
| Tool orchestrator (concurrent partition) | ⬜ planned | Migration M2 |
| Segmented prompt cache | ⬜ planned | Migration M3 |
| Compact (predictive + reactive) | ⬜ planned | Migration M4 |
| Large-result delta compression | ⬜ planned | Migration M5 |
| Provider fallback chain | ⬜ planned | Migration M6 |
| Cross-compact ground-truth state | ⬜ planned | Migration M7 |
| Skills runtime | ⬜ planned | Track "Harness extensions" |
| Hooks runtime | ⬜ planned | Track "Harness extensions" |
| Subagents | ⬜ planned | Track "Harness extensions" |
| MCP server (Mode C) | ⬜ planned | Track "Harness extensions" |

### 18.2 Migration map (anchors prior cross-refs)

The earlier `agent-design.md §7` named seven migrations. They survive under the
same numbering — `stage8-plan.md` Track 2 cross-refs are still valid.

| ID | Migration | Now described in |
|---|---|---|
| M1 | AsyncGenerator optimization loop | §4 |
| M2 | Tool self-description + concurrent batching | §6 |
| M3 | Segmented prompt cache | §7 |
| M4 | Predictive + reactive compact | §10.2 |
| M5 | Large-result delta compression | §6 (top-N + filter), §13 (subagent isolation) |
| M6 | Provider fallback chain | §16.2 |
| M7 | Cross-compact ground-truth state | §8 |

---

## 19. Glossary — Claude Code ↔ Arke Harness

| Claude Code | Arke Harness | Notes |
|---|---|---|
| Permission (allowlist/denylist) | Bounded action space (§5) | Compiler-computed, not user-curated |
| Skill (`SKILL.md`) | Harness skill (§11) | Same on-disk format |
| Hook (`PreToolUse`, `PostToolUse`, …) | Harness hook (§12) | Different event names, same model |
| Subagent (`Agent` tool) | Subagent session (§13) | Forked `OptimizationState`, isolated budget |
| MCP server | MCP server (§14) | Identical protocol; Arke as server |
| `settings.json` | `arke.config.yaml` (§17) | Same layered-defaults idea |
| `TodoWrite` / Plan mode | `checkpoint` / `rollback` + immutable `decision_log` (§8) | The decision log *is* the plan |
| Slash command (`/foo`) | `arke <verb>` CLI subcommand | `arke optimize`, `arke bench`, `arke mcp serve` |
| Cache control (`cache_control`) | Segmented prompt (§7) | We use the same primitive at finer granularity |
| File read/write | `apply_decision` / `rollback` | Both are reversible state mutators |

---

## 20. References

### Inbound (this document is referenced from)

- `README.md` — top-level architecture map
- `CLAUDE.md` — repo guidance for Claude Code
- `docs/phase1/stage6-plan.md` — Tool Declarative Interface design ref
- `docs/phase1/stage8-plan.md` — Track 2 (migrations) and Track for harness extensions
- `docs/phase1/spec-completeness-audit.md` — completeness checklist
- `docs/spec/arke-ir-spec.md` — IR spec back-reference
- `docs/spec/op-registry-interface.md` — op registry back-reference
- `docs/architecture/token-efficiency-analysis.md` — token math back-reference
- `arke/agent/tools.py` — `ToolMeta` source comment

### Outbound (this document references)

- `docs/architecture/e2e-flow.md` — end-to-end kernel generation flow
- `docs/architecture/token-efficiency-analysis.md` — segmented-cache token math
- `docs/architecture/naming-system.md` — terminology registry
- `docs/spec/arke-ir-spec.md` — SemanticIR / StrategyIR contract
- `docs/spec/op-registry-interface.md` — `list_legal_actions` source
- `docs/benchmark/benchmark-design.md` — BL/OT/ST/L taxonomy
- `arke/agent/tools.py` — `ToolMeta`, `BudgetType`, `CostLevel`
- `arke/agent/optimize.py` — `HeuristicStrategyGenerator` (the floor)
- `arke/learn/trajectory.py` — JSONL writer
- `skills/arke-test-coverage/SKILL.md` — skill format precedent
- `AGENTS.md` — external-agent contract

---

*v0.2 design — 2026-05-10. Renamed from `agent-design.md`. Implementation tracked
in `docs/phase1/stage8-plan.md` Track 2 and "Harness extensions".*
