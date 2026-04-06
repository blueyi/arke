# Arke Agent Design

> Architecture of the Arke Agent system, supporting dual-mode LLM integration.
> Date: 2026-04-05

---

## 1. Design Goals

### 1.1 Core Mission

The Arke Agent system exists to drive LLM-powered kernel optimization for AI accelerators (GPU, NPU, etc.) using Arke's
toolchain. Its core mission:

- **Generate high-performance, generalizable operators for diverse AI hardware** with minimal token cost
- **LLM as decision maker, not code generator** — the LLM explores optimization strategies
  through a Bounded Action Space; the compiler validates and executes
- **@rationale discipline** — every optimization decision carries a human-readable
  explanation, enabling learning, debugging, and knowledge transfer
- **Hardware-aware optimization** — decisions are grounded in real hardware constraints
  (shared memory limits, tensor core shapes, warp size)

The optimization loop is structurally identical to a general agentic loop:

```
while budget_remaining:
    response = llm.call(messages, tools)
    tool_results = execute_tools(response.tool_calls)
    messages.extend(tool_results)
    if llm_signals_done:
        break
compare(llm_best, fallback_strategy)
```

This isomorphism with other agentic systems (e.g., Claude Code) means well-validated
engineering patterns from those systems can be directly adapted.

### 1.2 Dual-Mode Integration

Two use cases demand different integration models:

| | Mode A: Built-in Agent | Mode B: External Agent |
|---|---|---|
| **LLM control** | Arke owns the call loop | External agent owns it |
| **Entry point** | `arke optimize ...` CLI / Python API | Agent calls `arke` as a tool |
| **Config needed** | API key + provider in `arke.config.yaml` | None (agent has its own LLM) |
| **Best for** | Automated CI, batch optimization, scripts | AI dev assistants, interactive workflows |
| **Examples** | Scheduled benchmark runs, GitHub Actions | OpenClaw, Claude Code, Cursor, Copilot |

**Why both?** Mode A delivers a standalone optimizer requiring no external AI framework.
Mode B lets existing AI coding assistants (which already have LLM access and project
context) use Arke as a specialized tool — the agent reads `AGENTS.md` to understand its
role, then drives Arke's optimization loop using CLI commands or the Python API.

---

## 2. Architecture Overview

### Mode A — API Key (Built-in Agent)

```
User (CLI / Python API)
        │
        ▼
  LLMRunner.optimize()
        │  build_system_prompt() + initial user message
        │
        ▼
  LLM Provider  ◄──────────────────────────────────────┐
  (Anthropic /                                          │
   OpenAI-compat)                                       │
        │                                               │
        │  tool_use response                            │
        ▼                                               │
  OptimizationSession.run_tool()                        │
        │  dispatch to ArkeEnv                          │
        ▼                                               │
  ArkeEnv ──────── V0 Static Validation                 │
        │           V1 Numerical Check                  │
        │           V2 HW Profiling (compile_and_profile)
        │                                               │
        │  tool_result JSON  ──────────────────────────►│
        │                         (appended to messages)
        ▼
  Budget check / nudge / compact
        │
        ▼
  RunResult (decisions, tokens, trajectory, generated_code)
```

### Mode B — External Agent (Agent-as-User)

```
External Agent (OpenClaw / Claude Code / Cursor / ...)
        │
        ├── reads AGENTS.md         → understands role + tool protocol
        ├── reads TOOLS.md          → environment-specific configuration
        ├── reads docs/architecture/ → architecture reference
        │
        ├── calls: arke optimize --kernel matmul --shape 1024,512,2048
        ├── calls: arke bench --bl 5 --layer l1
        └── calls: python -c "import arke; arke.optimize(...)"
                │
                ▼
          Arke CLI / Python API
                │
                ▼
          ArkeEnv + Codegen + Validation
                │
                ▼
          JSON output: strategy.json, report.json, trajectory.jsonl
```

### Shared Infrastructure

Both modes share the same underlying stack:

```
ArkeEnv  →  Tool Definitions  →  Semantic IR  →  Strategy IR
                                                      │
                                               Codegen (Triton)
                                                      │
                                               V0 / V1 / V2 Validation
                                                      │
                                               KernelCache / PyTorch integration
```

---

## 3. Mode A — API Key Integration (Built-in Agent)

### 3.1 LLM Provider Abstraction

Arke supports two provider APIs, unified under a common call interface in `LLMRunner`:

| Provider | API type | Config key |
|---|---|---|
| Anthropic Claude | `anthropic-messages` | `providers.anthropic` |
| OpenAI / compatible | `openai-completions` | `providers.openai` |

Configuration is read from `arke.config.yaml` or environment variables:

```yaml
# arke.config.yaml
llm:
  primary: "anthropic/claude-sonnet-4-6"
  fallbacks:
    - "openai/gpt-4o"
    - "openai/gpt-4o-mini"

providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com/v1"
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
```

The runner resolves the `primary` spec to `(ProviderConfig, ModelConfig)`, calls the
appropriate API, and normalizes the response into a common format before parsing.

**Provider fallback chain** *(Adapted from cc-inspired-update.md §6)*: on
`RateLimitError` or persistent timeout, the runner automatically retries with the next
entry in `fallbacks`. Each fallback attempt is recorded in `RunResult.errors` for
observability. The fallback list is intentionally ordered from most-capable to
least-capable, so the optimizer degrades gracefully rather than failing.

### 3.2 Optimization Loop

`LLMRunner.optimize()` drives a synchronous turn-based loop (current implementation):

```
for turn in range(max_turns):
    response = _call_llm(provider, model, messages)      # with retry on 429 / timeout
    assistant_msg, tool_uses = _parse_response(response)
    messages.append(assistant_msg)

    if not tool_uses:
        break   # LLM signals completion

    for tool_call in tool_uses:
        result = session.run_tool(tool_call.name, tool_call.input)
        # inject result into messages

    if session.budget.exhausted:
        # inject budget-exhausted message, force finalization
    if decisions >= 4 and not yet verified:
        # inject nudge: "call verify_correctness()"
```

**Planned refactor — AsyncGenerator loop** *(Adapted from cc-inspired-update.md §1)*:
The synchronous loop will be replaced with an `AsyncGenerator`-based model that yields
typed `OptimizationEvent` objects. This enables:

- Real-time streaming to CLI, REST API, and Jupyter without different code paths
- Clean cancellation via `.aclose()`
- Decoupled trajectory recording and event handling

```python
# Target interface (planned)
async for event in runner.optimization_stream(env, llm, config):
    match event.type:
        case "decision": print(f"  ▸ {event.data['kind']} — {event.data['rationale'][:60]}…")
        case "compile":  print(f"    HW: {event.data['vs_baseline']:.0%} baseline")
        case "done":     print(f"✅ Final: {event.data['performance']['vs_baseline']:.0%}")
```

**Budget management** is tracked in `OptimizationBudget`:
- `decisions_used` / `max_decisions` (default 50) — counts `apply_decision` calls
- `compiles_used` / `max_compiles` (default 10) — counts `verify_correctness` and
  `compile_and_profile` calls (hardware-expensive operations)
- Budget status is injected into every tool result so the LLM sees remaining headroom

### 3.3 System Prompt Design

The system prompt is built by `build_system_prompt()` in `arke/agent/prompts.py`.
It teaches the LLM its role, the workflow, decision priorities, and hardware constraints.

Current structure (single block):
1. Role definition — decision maker, not code generator
2. Hardware target — compute units, shared memory limit, warp size, peak TFLOPS, tensor core shapes
3. Workflow — 6-step sequence: analyze → plan → decide → verify → measure → iterate
4. Decision priority — compute-bound vs. memory-bound heuristics
5. Budget — decision count, compile count, target performance ratio
6. Key principles — tile alignment, epilogue fusion, shared memory bounds, rationale requirement
7. Error handling — what to do on validation failure, numerical failure, performance regression

**Planned: Segmented prompt cache** *(Adapted from cc-inspired-update.md §3)*:
Split into 4 segments with independent `cache_control` to reduce token cost across
repeated optimization sessions:

| Segment | Content | Cache behavior |
|---|---|---|
| 1 | Role + kernel optimization knowledge | Global cache (all sessions) |
| 2 | Hardware profile | Hardware-level cache (per target) |
| 3 | Semantic IR + auto-analysis | Kernel-level cache (per kernel) |
| 4 | Current Strategy IR state + budget | No cache (changes every step) |

Estimated savings for a 50-step session: ~176K tokens vs. no-cache baseline
(Segment 1–3 hit cache on 49 of 50 turns).

**Agent context injection**: `arke/agent/context.py` auto-loads `AGENTS.md`,
`IDENTITY.md`, `SOUL.md`, and `TOOLS.md` from the project root and prepends them to
the system prompt. This ensures the optimization agent always has its role definition
and project architecture in context, regardless of how the session is initiated.

### 3.4 Context Compact

For long optimization sessions (50+ turns), message history can exceed provider context
limits. Two compact strategies are planned *(Adapted from cc-inspired-update.md §4)*:

**Predictive compact**: estimate token count before each LLM call; if approaching 80%
of the context window, trigger compaction proactively.

**Reactive compact**: if the API returns a `prompt_too_long` / `context_length_exceeded`
error, catch it, compact, and retry transparently.

Compact logic:
1. Summarize the middle portion of message history (decisions, outcomes, errors)
2. Preserve the last N messages in full (recent context)
3. Always preserve the **ground truth state** from `OptimizationState` — the Strategy IR,
   compile results, and decision log are never lossy-compressed

**Ground truth state across compacts** *(Adapted from cc-inspired-update.md §7)*:
`OptimizationState` maintains authoritative records independently of the message list:
- `strategy_ir` — always complete, never summarized
- `compile_results` — all attempts with performance numbers
- `decision_log` — all decisions with rationale
- `best_result` — tracked separately for finalization

This ensures the final `compare(llm_best, fallback)` decision is always based on
complete data, even if the conversation was compacted multiple times.

---

## 4. Mode B — External Agent Integration

### 4.1 Agent Context Files

When an external AI agent (OpenClaw, Claude Code, Cursor, etc.) opens the Arke
repository, it discovers context files at the project root:

| File | Purpose | Audience |
|---|---|---|
| `AGENTS.md` | Role definition, tool protocol, operator coverage, key references | Optimization agents |
| `IDENTITY.md` | Agent identity and persona | Internal agent sessions |
| `SOUL.md` | Behavioral principles | Internal agent sessions |
| `TOOLS.md` | Environment-specific config (hardware, paths, aliases) | Both |

`AGENTS.md` is the primary entry point for external agents. It communicates:
- The agent's role (optimizer, not developer)
- The 8-tool protocol (`analyze_compute`, `list_legal_actions`, `apply_decision`, etc.)
- Optimization principles (Bounded Action Space, @rationale, budget awareness)
- Architecture map (which directories contain what)
- Operator coverage (45 ops, OT0–OT4)
- Key design documents and specs

**Auto-loading**: `arke/agent/context.py` also loads these files into the system prompt
for Mode A sessions, ensuring both modes operate with the same agent identity.

### 4.2 CLI / API as Tool Interface

External agents interact with Arke through shell commands or the Python API.
These are the primary "tools" available to external agents:

**CLI — Optimization**:
```bash
# Natural language entry (LLM-Native)
arke optimize "matmul kernel, 1024x512x2048, f16, Ampere"

# Structured entry
arke optimize --kernel matmul --shape 1024,512,2048 --dtype f16 --target ampere

# With explicit LLM provider
arke optimize --kernel matmul --shape 1024,512,2048 --llm anthropic
```

**CLI — Benchmark**:
```bash
arke bench --bl 5 --layer l1            # BL5 × L1: all 45 ops × all shapes
arke bench --bl 6 --model gpt2          # BL6: GPT-2 end-to-end
arke bench --bl 3 --layer l1 --archive  # BL3, save results
```

**CLI — Gate validation**:
```bash
python -m benchmarks.gate G6 --tier 2
python -m benchmarks.gate G7 --tier 2 --live --archive
```

**Python API** (programmatic integration):
```python
import arke

# Optimize a single kernel
result = arke.optimize(
    kernel="matmul",
    shape=[1024, 512, 2048],
    dtype="f16",
    target="ampere",
    llm="anthropic",
)
print(f"Performance: {result.vs_baseline:.0%} baseline")
print(f"Decisions: {result.decisions}, Tokens: {result.tokens_used}")

# Run benchmarks
bench_result = arke.bench(bl=5, layer="l1")
```

### 4.3 Agent-Readable Output

All Arke outputs are designed for machine consumption by external agents:

**Structured files** (written to `output/` by default):

| File | Format | Content |
|---|---|---|
| `strategy.json` | JSON | Full Strategy IR with all decisions and @rationale |
| `report.json` | JSON | Performance numbers, correctness, token cost, metadata |
| `trajectory.jsonl` | JSONL | Step-by-step (state, action, result) sequence |
| `kernel.py` | Python | Final optimized Triton kernel, ready to use |
| `metadata.json` | JSON | LLM provider, hardware, budget, timestamps |

**Exit codes**:
- `0` — success, performance target met
- `1` — success, but performance below target (fallback used or LLM did not improve)
- `2` — correctness failure
- `3` — configuration or environment error

**Structured error messages**: all errors include a `hint` field with actionable guidance,
so the external agent can diagnose and retry without human intervention.

**Progress feedback**: the CLI emits structured log lines parseable as JSONL when
`--json-log` is set, allowing the external agent to monitor progress in real time.

### 4.4 Integration with Agent Frameworks

**OpenClaw**: uses the `exec` tool to invoke `arke` CLI commands. The agent reads
`AGENTS.md` for role understanding, then drives optimization via shell commands.
Results are read back via the `read` tool on output JSON files.

**Claude Code / Cursor**: discovers context files by reading the project root, uses the
terminal to run `arke` commands, and parses structured output files.

**MCP (Model Context Protocol)**: a future integration path. Arke could expose its
10-tool interface as an MCP server, enabling any MCP-compatible agent to use Arke's
optimization tools directly without shell invocation. This would provide:
- Typed tool schemas discoverable by the agent
- Streaming tool results
- Session state management at the protocol level

---

## 5. Shared Infrastructure

### 5.1 Tool Declarative Interface

Tools are defined in `arke/agent/tools_schema.py` as OpenAI function-calling schemas,
compatible with Anthropic `tool_use` and other providers. Each tool schema includes
name, description, and a full JSON Schema for parameters.

`TOOL_METADATA` provides per-tool execution metadata used by the orchestrator:

```python
TOOL_METADATA = {
    "get_hw_profile":      {"concurrent_safe": True,  "budget_type": "free",     "cost": "cheap"},
    "analyze_compute":     {"concurrent_safe": True,  "budget_type": "free",     "cost": "cheap"},
    "list_legal_actions":  {"concurrent_safe": True,  "budget_type": "free",     "cost": "cheap"},
    "apply_decision":      {"concurrent_safe": False, "budget_type": "decision", "cost": "cheap"},
    "verify_correctness":  {"concurrent_safe": False, "budget_type": "compile",  "cost": "medium"},
    "compile_and_profile": {"concurrent_safe": False, "budget_type": "compile",  "cost": "expensive"},
    "rollback":            {"concurrent_safe": False, "budget_type": "free",     "cost": "cheap"},
    "checkpoint":          {"concurrent_safe": False, "budget_type": "free",     "cost": "cheap"},
}
```

**Planned: full `ToolMeta` + `ArkeTool` ABC** *(Adapted from cc-inspired-update.md §2)*:
Each tool will self-declare `concurrent_safe`, `idempotent`, `requires_compile`,
`mutates_strategy`, and `budget_type`. The orchestrator uses these declarations to
automatically partition tool calls into concurrent batches (read-only tools) vs.
serial execution (state-mutating tools).

**Concurrent partitioning algorithm**: consecutive `concurrent_safe=True` tools are
batched for `asyncio.gather`; any `concurrent_safe=False` tool breaks the batch and
executes serially. For example:
```
[analyze_compute, get_hw_profile, apply_decision, list_legal_actions]
  → Batch([analyze_compute, get_hw_profile], concurrent=True)
  → Batch([apply_decision], concurrent=False)
  → Batch([list_legal_actions], concurrent=True)
```

### 5.2 Large Result Management

Some tool results can be very large (full legal action lists, complete state dumps).
Large results are compressed before injecting into the message history to save tokens.

*(Adapted from cc-inspired-update.md §5)*

Key strategies:
- `list_legal_actions`: returns top 10 of N candidates + total count + hint.
  Full list available with `kind=` filter.
- `observe` / state dumps: return delta (changed fields) rather than full state.
  Full state on demand with `full=true`.
- `verify_correctness`: omits raw tensor data; returns pass/fail + max error + tolerance.

Threshold: results over ~3000 characters are automatically compressed.

### 5.3 Three-Level Validation

Every strategy change passes through a validation pipeline:

| Level | Name | When | Cost | What it checks |
|---|---|---|---|---|
| **V0** | Static validation | Every `apply_decision` call, automatic | ~0ms | Shared memory bounds, tile alignment, legal action constraints, structural validity |
| **V1** | Numerical validation | On `verify_correctness` call | ~100ms–1s | NumPy reference comparison, max absolute/relative error, NaN/Inf detection |
| **V2** | Performance validation | On `compile_and_profile` call | ~1–5s | Actual GPU latency, TFLOPS, roofline efficiency, vs. vendor baseline (cuBLAS) |

V0 failures trigger automatic rollback and return a structured error with guidance.
V1 and V2 failures return error details but leave rollback to the LLM's discretion.

The two-stage `verify_correctness` implementation:
1. V1 numerical check against NumPy reference (fast, no GPU compilation needed)
2. GPU correctness check: compile → run → compare output vs. NumPy reference at the
   same dtype (measures implementation correctness, not precision loss)

### 5.4 Fallback Strategy

Every optimization session has a fallback path:

1. **Compiler-generated baseline**: Arke's codegen produces a valid (if unoptimized)
   strategy from the Semantic IR without any LLM decisions. This is the floor.
2. **LLM vs. fallback comparison** at finalization: if `llm_best.performance >
   fallback.performance`, use LLM result; otherwise use fallback and note that
   "LLM did not improve over fallback strategy".
3. **Provider fallback chain**: primary LLM → fallback models, on rate limit or error.

This ensures Arke always produces a correct, runnable kernel even when the LLM
optimization fails or is unavailable.

---


> **Implementation roadmap** → [plan.md](../../roadmap/plan.md)

## Appendix A: Current `arke/agent/` Code Structure

```
arke/agent/
├── __init__.py           — Package exports
├── context.py            — Agent context file loader
│                           Auto-loads AGENTS.md, IDENTITY.md, SOUL.md, TOOLS.md
│                           from project root into system prompt
├── llm_config.py         — LLM configuration (providers, models, fallbacks)
│                           Reads from arke.config.yaml or environment variables
├── prompts.py            — System prompt builder + initial user message
│                           build_system_prompt(hw_profile, budget, target_perf)
├── runner.py             — LLM optimization loop driver
│                           LLMRunner.optimize() — turn-based tool-use loop
│                           Handles Anthropic + OpenAI APIs, retry, fallback
├── session.py            — Optimization session lifecycle
│                           OptimizationSession: ArkeEnv + budget + trajectory
│                           run_tool() dispatch, state machine, budget tracking
├── tools_schema.py       — Tool schemas (OpenAI function-calling format)
│                           10 tools: create_kernel → restore
│                           TOOL_METADATA: concurrent_safe, budget_type, cost
├── tools/
│   ├── __init__.py
│   ├── base.py           — (Planned) ArkeTool ABC + ToolMeta declarative interface
│   └── orchestrator.py   — (Planned) Tool partitioning + concurrent execution
└── providers/
    ├── __init__.py
    └── base.py           — LLMProvider ABC (planned expansion)
```

Key relationships:
- `LLMRunner` owns the HTTP calls; `OptimizationSession` owns the domain logic
- `context.py` is used by both Mode A (injected into system prompt) and
  Mode B (external agents read the same files directly from the repo)
- `tools_schema.py` is the single source of truth for the tool interface —
  both LLM providers and the session dispatch table reference it

---

## Appendix B: Key Design Decisions from cc-inspired-update.md

This appendix distills the core decisions from the deprecated `cc-inspired-update.md`
to avoid needing to re-read the full document.

**Decision 1 — AsyncGenerator as the loop primitive**
The optimization loop is an `AsyncGenerator[OptimizationEvent, None]` that yields
typed events. Consumers (CLI, API, Jupyter) process the same stream differently.
Cancellation is handled by generator cleanup (`aclose()`), not flags.

**Decision 2 — Tools are self-describing**
Each tool declares `concurrent_safe`, `idempotent`, `requires_compile`, `mutates_strategy`,
`budget_type`. The orchestrator makes scheduling decisions from these properties, not
from a hardcoded list. This makes adding new tools safe — the orchestrator automatically
does the right thing.

**Decision 3 — System prompt has cache topology**
The 4-segment structure maps to cache invalidation frequency: global knowledge
(never changes) → hardware profile (changes per target) → semantic IR (changes per
kernel) → dynamic state (changes every turn). Matching cache granularity to change
frequency maximizes token savings.

**Decision 4 — Two compact triggers, one compact function**
Predictive compact (token estimate > 80% of limit) and reactive compact (API error)
both call the same `compact_optimization_context()` function. The function always
preserves the last 5 messages in full and injects the ground truth state so the LLM
never loses track of where it is.

**Decision 5 — Large results use delta, not truncation**
Rather than cutting off large results mid-way, the result management layer returns
semantically complete but smaller representations: top-N lists with counts, state
deltas with full-state-on-demand, correctness results without raw tensors. The LLM
gets all the information it needs at minimal token cost.

**Decision 6 — Fallback chain is ordered by capability**
`primary → fallback[0] → fallback[1] → ...` is ordered from most capable to least.
This means the optimizer degrades gracefully on rate limits rather than failing.
Each fallback switch emits an observable event so the caller knows what model was
actually used.

**Decision 7 — Ground truth lives outside messages**
`OptimizationState` (Strategy IR, compile results, decision log) is maintained
independently of the conversation history. After compact, the authoritative state is
re-injected into the new context, so the LLM always sees a complete and accurate
picture of what has been decided.

---

*Document version: v1.0 | Created: 2026-04-05*
*References: e2e-flow.md, cc-inspired-update.md (deprecated), arke/agent/ source*
