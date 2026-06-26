# RFC: Arke Harness v2 — 能力补强与重构提案

> **Status:** 📋 PROPOSAL — awaiting project-lead (Leon) approval before any
> Façade-touching landing.
> **Author:** Kitty (lead engineer)
> **Date:** 2026-06-26
> **Scope governance:** Per `plan.md` Locked Principle #4 (Two-Layer Harness),
> any change to the **Public Façade** (8 tools / event stream / trajectory
> schema / SKILL.md / Hooks / MCP) is a locked architecture commitment and
> **requires Leon approval**. This RFC separates proposals into:
> **(N) non-breaking 1.x additions** (safe, no Gate impact),
> **(B) breaking changes** (bump Façade major, re-run/re-lock G8 — needs explicit
> Leon ack), and **(S) Substrate-only** (Kitty decides + executes directly).
>
> Companion deliverable: `docs/architecture/arke-harness-handbook.md` (usage
> manual, Hermes-doc-style).

---

## 0. TL;DR

Arke Harness's **contract scaffold is genuinely AI-Native and well-built**:
Façade v1.0 (8 tools, frozen + 158 contract tests), 9-kind OptimizationEvent
stream, frozen trajectory v1.0, real checkpoint/rollback, budget accounting,
and — since P0-A/P0-B (commit `20d9fd4`) — a live-LLM tool-use orchestrator
(`LLMRunner`) with real Triton V1/V2 measurement. G8 passed 6/6.

Benchmarked against the four most mature agent harnesses in the wild
(Hermes, OpenClaw, Codex CLI, Claude Code), Arke is **strong on the parts
unique to compiler-verified kernel optimization** (bounded action space from
the compiler, V0/V1/V2 staged verification, deterministic heuristic floor,
trajectory-as-learning-artifact) and **behind on the generic agent-harness
runtime ergonomics** they have converged on:

1. **Legal-action quality** — `list_legal_actions` is a static cartesian
   product of default factors, not a compiler/HW-computed legal set. (the #1
   AI-Native gap flagged in the 2026-06-24 review)
2. **Provider robustness** — no fallback chain, no retry/backoff, no
   rate-limit handling. One provider error aborts the run.
3. **Context/token discipline at scale** — no compaction, no delta/observe
   mode, no segmented prompt cache. Long loops will blow context.
4. **Extension runtimes** — Skills, Hooks, Subagents, MCP server are
   designed but not wired to a runtime.
5. **Resumability & durability** — a crashed/interrupted run is lost; no
   session persistence/resume (Codex & Claude Code both have this).
6. **Self-description for orchestration** — `ToolMeta` exists but isn't fully
   exploited for concurrent partitioning at runtime.

This RFC proposes a **non-breaking-first** roadmap (P0→P2) that closes these
gaps while keeping Façade v1.0 frozen, plus a concrete end-to-end **Triton
kernel generation & autotuning** path built on the upgraded Harness.

---

## 1. Current-state audit (what exists today)

| Subsystem | State | Evidence / location |
|:---|:---:|:---|
| Façade v1.0 — 8 tools, frozen | ✅ | `arke/agent/facade.py`, `facade_v1_schema.json`, 158 contract tests |
| OptimizationEvent stream (9 kinds) | ✅ | `arke/agent/events.py`, frozen schema |
| Trajectory v1.0 (`arke-trajectory-v1.0.0`) | ✅ | `arke/learn/trajectory.py` + `trajectory_schema.py` |
| Live LLM tool-use loop | ✅ | `arke/agent/runner.py::LLMRunner` (468 LOC) |
| Provider resolution (anthropic+openai, yunwu `/v1`) | ✅ | `arke/agent/llm_config.py` (228 LOC) |
| Real V1 numeric validation (Triton/CUDA) | ✅ | `verify_correctness` → TritonBackend compile+compare |
| Real V2 GPU profiling | ✅ | `compile_and_profile` → Triton latency + `baseline_ratio` |
| Bounded action space | ⚠️ static | `arke/agent/env.py` — default-factor cartesian product, **no HW/shape legality** |
| Heuristic strategy floor | ✅ | `arke/agent/optimize.py::HeuristicStrategyGenerator` |
| checkpoint / rollback | ✅ | `arke/agent/state.py` |
| `@rationale` contract | 🟡 soft | enforced in heuristic path; **optional in tool schema** (not execution-enforced) |
| Provider fallback / retry | ❌ | one error → `stop_reason="llm_error"`, run aborts |
| Compaction / delta / prompt cache | ❌ | full message log resent each turn |
| Skills runtime | ❌ planned | designed `arke-harness.md §11` |
| Hooks runtime | ❌ planned | designed §12 |
| Subagents | ❌ planned | designed §13 |
| MCP server (Mode C) | ❌ planned | designed §14 |
| Session persistence / resume | ❌ | trajectory is write-only; no resume |

**Live verification run (this RFC, 2026-06-26):**
`arke optimize examples/operators/01_matmul.ak --cycles 3` →
`success: true, cycles_completed: 3, decision_count: 11, best_score: 1.04`
(deterministic path; profile is `_mock_profile` formula — confirms loop +
trajectory contract, not real GPU latency. Real latency requires the live
or bench path.)

---

## 2. Cross-harness benchmark — what the four mature harnesses do

> Lens: which of their primitives map onto an Arke kernel-optimization
> harness, and what Arke is missing. (Hermes is the explicit reference for
> the usage manual.)

### 2.1 Capability matrix

| Capability | Hermes | OpenClaw | Codex CLI | Claude Code | **Arke today** | **Gap → proposal** |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Tool/function calling loop | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Declarative tool metadata | ✅ | ✅ | ✅ | ✅ | ✅ `ToolMeta` | exploit for concurrency (N2) |
| Permission / bounded actions | approvals | approvals | sandbox | allowlist | ✅ **compiler-computed** (strongest) | upgrade legality (S1) |
| Provider fallback chain | ✅ | ✅ | ✅ | ✅ (anthropic) | ❌ | **P0** S3 fallback |
| Retry / backoff on transient err | ✅ | ✅ | ✅ | ✅ | ❌ | **P0** S3 |
| Context compaction | ✅ predictive+reactive | ✅ | ✅ | ✅ | ❌ | P1 S5 |
| Delta / observe mode | ✅ | partial | — | — | ❌ | P1 S5 |
| Skills (SKILL.md procedural memory) | ✅ | ✅ | — | ✅ | ❌ runtime | P2 S6 |
| Hooks / lifecycle events | ✅ cron+webhook | ✅ | ✅ notify | ✅ Pre/Post | ❌ runtime | P2 S7 |
| Subagents / delegation | ✅ `delegate_task` | ✅ | — | ✅ Agent tool | ❌ | P2 S8 |
| MCP server/client | ✅ native | ✅ | ✅ | ✅ | ❌ Mode C | P1 S4 |
| Session persistence / resume | ✅ session DB | ✅ | ✅ rollout | ✅ | ❌ | **P0** S2 |
| Memory (cross-session facts) | ✅ | ✅ | — | ✅ CLAUDE.md | n/a (trajectory KB) | reuse `rationale_kb` |
| Streaming events to consumer | ✅ | ✅ | ✅ | ✅ | 🟡 sync only | P1 S5 (AsyncGen) |
| Cost / budget accounting | ✅ tokens | ✅ | ✅ | ✅ | ✅ decisions+compiles+tokens | — |
| Structured final report | ✅ | ✅ | ✅ | ✅ | ✅ `OptimizeResult` | — |

### 2.2 Distilled lessons (what to borrow)

- **Hermes — fallback chain + toolset gating + cron/delegation.** Hermes pins
  a provider chain (`fallback_providers`) and degrades gracefully; restricts
  toolsets per task to cut token overhead; uses `delegate_task` for parallel
  isolated subagents whose only the final summary re-enters context. → Arke
  should adopt **provider fallback** (P0) and **subagent design-space sweep**
  (P2). Toolset-gating maps onto Arke's already-bounded 8 tools (no-op win).
- **OpenClaw — gateway resilience + approvals.mode degradation.** Auxiliary-LLM
  failure degrades to a conservative-but-functional mode rather than hard
  stop. → Arke's **heuristic floor is the analog**, but it must actually be
  *triggered* on LLM failure (today it's the only path, never a fallback).
- **Codex CLI — rollout/resume + sandbox levels.** Every run is a resumable
  rollout; sandbox escalation is explicit. → Arke should make trajectory
  **resumable** (P0 S2): a crashed 8-turn run resumes from the last
  checkpoint instead of restarting (compile budget is expensive — this is
  high-value for a 6 GB dev box).
- **Claude Code — Pre/PostToolUse hooks + skills + plan mode.** Hooks gate
  tool calls (e.g. reject risky edits); skills encode recurring recipes;
  the immutable decision log *is* the plan. → Arke maps Pre/Post hooks onto
  **PreDecision / PostCompile / PostProfile / OnRollback** (already named in
  the design); skills onto **autotune recipes** (e.g. "sweep one op across
  all shape tiers"). The decision-log-as-plan is already true.

---

## 3. Proposed capability set — Arke Harness v2

Ordered by value-to-thesis and gated by governance class. **(N)** = non-breaking
1.x addition, **(S)** = Substrate-only (Kitty executes), **(B)** = breaking
(needs Leon ack + G8 re-lock).

### P0 — Robustness & honesty (highest value, mostly Substrate)

| ID | Proposal | Class | Why | Touches Façade? |
|:--|:--|:--:|:--|:--:|
| **S1** | **Compiler/HW-computed legal actions** — `list_legal_actions` filters by `HardwareProfile` + shape: tile must divide/≤ dim; `place(shared)` cumulative ≤ `hw.shared_memory_bytes`; warps ≤ HW cap. Add gate assertion: feed illegal factor, assert absent. **✅ LANDED 2026-06-26** — tile/unroll/vectorize were already shape/HW-aware (P1-b); this pass added `place(shared)` capacity legality (`_tensor_bytes` × `hw.shared_memory_bytes`, 48 KiB Ampere) + 3 assertion tests. | S | The #1 AI-Native differentiator ("bounded space *guaranteed legal by compiler*") is currently name-only. **Tool output content** changes, not signature → non-breaking. | content only |
| **S2** | **Resumable trajectory** — `LLMRunner.optimize(resume_from=<trajectory.jsonl>)` rehydrates `OptimizationState` from the last checkpoint record; skips already-spent compile budget. **⬜ NEXT** — deferred this pass: needs a new `OptimizationState.from_dict` public deserialization contract + `optimize()` signature extension (schema-first review warranted) and only the *live* path materially benefits (gated on D2). S1/S3/S4 landed first as zero-external-cost wins. | S | Compile/profile is the expensive resource on 6 GB; a crashed run today loses all GPU work. Codex-style rollout resume. | no |
| **S3** | **Provider fallback chain + retry/backoff** — `LLMConfig.fallback` list; on transient error (timeout/429/5xx) retry w/ exp backoff, then next provider; on terminal LLM failure emit real `fallback{layer:"provider"}` event and degrade to heuristic floor with `done.chosen="heuristic_floor"`. **✅ LANDED 2026-06-26** — `LLMConfig.fallback` + `provider_chain()`; `load_from_env` auto-populates fallback from sibling providers; `LLMRunner._call_llm_resilient` does same-provider exp-backoff retry (1.5/3.0s ×2) + same-protocol failover, records `fallback{layer:"provider"}` in `session_summary`; non-transient errors abort immediately. 16 tests. (Cross-protocol failover + heuristic-floor degrade-on-exhaustion deferred to a follow-up.) | S | Hermes/OpenClaw both have this; today one error aborts. Makes the heuristic floor an *actual* fallback, not the only path. | no |
| **S4** | **`@rationale` execution-enforced** — `apply_decision` rejects non-trivial (level≥1) decisions with empty rationale (`ToolResult(success=False)`). Gate assertion: every `decision` event has non-empty rationale. **✅ LANDED 2026-06-26** — `ApplyDecisionTool.execute` now rejects level≥1 decisions with missing/whitespace rationale; Façade v1.0 schema unchanged (rationale stays schema-optional); 2 assertion tests + 6 existing tests updated to pass rationale. | S | Closes the soft-contract gap; `@rationale` is a locked thesis pillar but currently bypassable. Tightening *behavior* of an existing tool → non-breaking (stricter validation only). | behavior |

### P1 — Scale & ecosystem reach

| ID | Proposal | Class | Why | Touches Façade? |
|:--|:--|:--:|:--|:--:|
| **N1** | **AsyncGenerator event loop** — stream `OptimizationEvent`s to the consumer as they happen (migration M1). Sync loop kept as a thin wrapper. | N | Enables live dashboards / MCP streaming; all 4 harnesses stream. New surface, old one intact. | +API |
| **N2** | **Concurrent tool partitioning** — use `ToolMeta.concurrent_safe` + `partition_for_execution` at runtime to batch independent read-only tools (e.g. `get_hw_profile` + `analyze_compute`) in one turn (migration M2). | N | Cuts turns/tokens; the metadata already exists, just unused at runtime. | no |
| **S5** | **Context compaction + delta observe** — predictive (token-threshold) + reactive compaction; an `observe(delta=true)` style result-shrinking for repeated state reads (migrations M3/M4/M5). | S | Long autotune loops (≥15 turns) will overflow context without this. | no |
| **N3** | **MCP server (Mode C)** — `arke mcp serve` exposes the 8 tools / trajectory resource / prompts over stdio+sse so any MCP client (Hermes, Cline, Claude Desktop) drives Arke directly. | N | Closes Arke ↔ MCP-ecosystem gap; the Façade was *designed* for this. New transport, same 8 tools. | +transport |

### P2 — Extension runtimes (design→runtime)

| ID | Proposal | Class | Why | Touches Façade? |
|:--|:--|:--:|:--|:--:|
| **N4** | **Skills runtime** — load `SKILL.md` autotune recipes (e.g. `sweep-op-all-tiers`, `flash-attn-blocking`) into the system prompt / as callable procedures. | N | Encodes recurring optimization expertise; Claude-Code-compatible format already specced. | +loader |
| **N5** | **Hooks runtime** — wire `PreDecision / PostCompile / PostProfile / OnRollback` to external Python callables (e.g. Prometheus push on PostProfile, reject-if-register-pressure on PreCompile). | N | Observability + policy injection; named in design, no runtime. | +registry |
| **N6** | **Subagent design-space sweep** — fork `OptimizationState` with isolated budget to explore tile-size/backend variants in parallel; only the winning strategy re-enters the parent (Hermes `delegate_task` analog). | N | Parallel exploration on the bounded space; isolated budgets prevent runaway. | +tool (additive) |

### Breaking-change candidates (HOLD — need Leon ack + G8 re-lock)

| ID | Proposal | Class | Note |
|:--|:--|:--:|:--|
| **B1** | Add a 9th Façade tool (e.g. `propose_strategy` for one-shot whole-strategy proposal) | B | Adding a tool is *technically* 1.x-compatible per versioning policy (§3.0.2 "add tools"), BUT it changes the locked "exactly 8 tools" assertion + frozen schema snapshot → I treat it as needing your explicit ack. |
| **B2** | Change `apply_decision` signature to make `rationale` schema-`required` | B | This is the "hard" version of S4. Breaks the frozen schema (required-set change) → Façade major bump. S4 (execution-enforced, schema unchanged) is the non-breaking alternative I recommend instead. |

**Recommendation:** land **P0 (S1–S4)** first — highest thesis value, all
Substrate/non-breaking, no Gate risk. Then P1. Hold all **(B)** items for an
explicit Façade v2.0 decision.

---

## 4. End-to-end Triton kernel generation & autotuning (built on v2)

The deliverable "基于 Arke Harness 实现 GPU Triton kernel 的生成与调优" maps to
this concrete flow on the upgraded Harness:

```
.ak source / op-name+shape
        │
        ▼  parse → SemanticIR (WHAT)
┌───────────────────────────────────────────────┐
│  LLMRunner.optimize(...)  [live]  OR            │
│  HeuristicStrategyGenerator [floor/deterministic]│
└───────────────┬───────────────────────────────┘
                │  drives the 8 Façade tools:
   get_hw_profile → analyze_compute → list_legal_actions(S1: HW-legal)
        → apply_decision(@rationale, S4-enforced)
        → verify_correctness (real V1: Triton compile + fp64-CPU ref compare)
        → compile_and_profile (real V2: Triton latency + baseline_ratio)
        → checkpoint / rollback   ← repeat ≥3 cycles
                │
                ▼  best correct StrategyIR (HOW)
        TritonBackend.lower → .triton kernel source + launch config
                │
                ▼  resumable trajectory.jsonl (S2) + strategy.json + result.akir
        @rationale → rationale_kb.jsonl  (learning artifact)
```

**Autotuning loop** = the compile→profile→adjust cycle driven by real
`baseline_ratio` from V2, with the heuristic floor (S3) guaranteeing a
correct kernel even if the LLM path errors. The **Same-Backend Fairness**
denominator (plan.md Locked Principle #2) is the fastest Triton-only
implementation of the op (FlagGems/Liger/…), `ε=0.03`.

**Phase-1 validation slice (this RFC ships the deterministic-path evidence;
live-path evidence is a follow-up that consumes yunwu `/v1` tokens):**
- Deterministic: `arke optimize <op>.ak --cycles 3` → strategy + trajectory
  (✅ run above: matmul 3 cycles / 11 decisions).
- Live (optional, token cost): `LLMRunner` over matmul/rmsnorm → real GPU
  `baseline_ratio` + `chosen="llm"` trajectory.

---

## 5. Usage manual outline (Hermes-doc-style)

`docs/architecture/arke-harness-handbook.md` — sections mirror Hermes docs:

1. **Overview** — what the Harness is, two-layer model, three roles of the LLM.
2. **Quick Start** — install, env, first `arke optimize` run, reading output.
3. **The 8 Façade Tools** — per-tool: purpose, params schema, result shape,
   cost tier, `@rationale` contract (Hermes "tools" reference style).
4. **Integration Modes** — A (built-in CLI/Python), B (external agent shells
   out), C (MCP server) — with copy-paste examples.
5. **Configuration** — `arke.config.yaml` (provider, budget, hooks, skills,
   mcp) + env precedence (`ARKE_LLM_*` / `ANTHROPIC_*` / yunwu `/v1`).
6. **Live LLM autotuning** — `LLMRunner` usage, model_spec, max_turns,
   budget, fallback chain (S3), resume (S2).
7. **Deterministic / heuristic path** — when to use, reproducibility.
8. **Trajectories & the @rationale KB** — schema, mining for SFT/RL.
9. **Extending the Harness** — onboard a new op, a new baseline runner,
   a skill, a hook (the HARNESS-3 extensibility demos).
10. **Triton kernel generation & tuning cookbook** — worked examples:
    matmul, rmsnorm, flash_attention; reading `baseline_ratio`; OOM on 6 GB.
11. **Troubleshooting** — yunwu endpoint injection, FlagGems aten::mm hijack,
    mock-vs-real profile, budget exhaustion, provider errors.
12. **Reference** — event kinds, trajectory record kinds, exit/stop reasons,
    CLI verbs, governance (frozen contracts).

---

## 6. Decision points for Leon

> Ack format e.g. `D1=a D2=yes D3=b`. Default if no reply in 10 min recorded
> in INBOX.

- **D1 — Landing scope.** (a) Land P0 (S1–S4, all Substrate/non-breaking) now;
  (b) only land the handbook + this RFC, hold all code; (c) land P0+P1.
  **My rec: (a)** — highest thesis value, zero Gate risk.
- **D2 — Live-path evidence.** Spend yunwu `/v1` tokens to produce a real
  live-LLM Triton autotuning trajectory (matmul + rmsnorm) as evidence?
  (yes/no). **My rec: yes, bounded to 2 ops** — proves the thesis pillar with
  real GPU `baseline_ratio`, small token cost.
- **D3 — Breaking items (B1/B2).** Authorize a future Façade v2.0 (9th tool /
  required-rationale schema), or keep Façade v1.0 frozen and use the
  non-breaking equivalents (S4)? **My rec: keep v1.0 frozen, use S4.**
- **D4 — MCP server priority (N3).** Build Mode C now (lets Hermes itself
  drive Arke) or defer to after P0? **My rec: defer to P1** — robustness
  before reach.

---

## 7. Non-negotiables this RFC respects

- Façade v1.0 stays frozen unless D3 explicitly authorizes a major bump.
- No Gate threshold / exit-criterion changes (Gate Governance).
- Same-Backend Fairness denominator unchanged (Locked Principle #2).
- Two-directional AI-Native thesis, three LLM roles (Locked Principle #6) —
  this RFC's center of gravity is "how AI-Native is the whole Harness for an
  Agent consumer", not just "does the decision loop run".
- Doc-sync: any landed code updates `arke-harness.md` + this RFC's status.

---

*RFC v1 — 2026-06-26. Companion: arke-harness-handbook.md. Supersedes nothing;
extends arke-harness.md §18 roadmap with a governance-classified, cross-harness-
benchmarked v2 plan.*
