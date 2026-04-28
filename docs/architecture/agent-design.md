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


> **Implementation roadmap** → [plan.md](../roadmap/plan.md)

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

---

## 7. Claude as Arke Agent LLM Backend

### 7.1 Overview

Claude (by Anthropic) is the recommended primary LLM backend for Arke Agent. This section details integration, configuration, and best practices for using Claude as the decision-making engine for kernel optimization.

### 7.2 Why Claude for Arke Agent

| Capability | Claude | Benefit for Arke |
|:---|:---|:---|
| **Structured reasoning** | Excellent at JSON/IR parsing and generation | Handles SemanticIR, StrategyIR, @rationale annotations natively |
| **Long context** | 200K tokens (Opus 4.6) | Supports full kernel context + optimization history |
| **Tool use** | Native tool_use format | Direct integration with ArkeEnv tool definitions |
| **Bounded reasoning** | Strong at constrained decision spaces | Excels at selecting from legal_actions enumerated by compiler |
| **Transparency** | Thinking mode + detailed reasoning | Enables auditability and learning from optimization decisions |
| **Multi-turn dialogue** | Excellent conversation management | Supports iterative refinement and rollback scenarios |
| **Cost efficiency** | Competitive pricing | Minimal token overhead for IR-based optimization |

### 7.3 Configuration

#### 7.3.1 API Key Setup

```bash
# Set Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Or add to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

#### 7.3.2 arke.config.yaml

```yaml
# Primary LLM configuration
llm:
  primary: "anthropic/claude-opus-4-6"  # Recommended: Opus 4.6 for best reasoning
  fallbacks:
    - "anthropic/claude-sonnet-4-6"     # Fallback: Sonnet 4.6 (faster, cheaper)
    - "openai/gpt-4o"                   # Last resort: GPT-4o

# Provider configuration
providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com/v1"
    timeout_seconds: 300
    max_retries: 3
    retry_backoff_factor: 2.0

# Agent-specific settings
agent:
  thinking_budget: 10000  # tokens for Claude thinking mode (Opus only)
  use_thinking: true      # enable extended thinking for complex decisions
  temperature: 0.3        # lower = more deterministic decisions
  max_tokens: 16000       # max output tokens per call
```

#### 7.3.3 Model Selection Guide

| Model | Use Case | Reasoning | Cost |
|:---|:---|:---|:---|
| **Claude Opus 4.6** | Complex kernels, novel optimizations | Best reasoning, extended thinking | Higher |
| **Claude Sonnet 4.6** | Standard kernels, batch optimization | Good balance of speed/quality | Medium |
| **Claude Haiku 3.5** | Simple ops, rapid iteration | Fast, minimal cost | Lower |

**Recommendation:** Start with Sonnet 4.6 for production; use Opus 4.6 for research/novel kernels.

### 7.4 Integration Flow

#### 7.4.1 Initialization

```python
from arke.agent import LLMRunner, OptimizationSession
from arke.env import ArkeEnv

# Initialize LLM runner with Claude backend
llm_runner = LLMRunner(
    provider="anthropic",
    model="claude-opus-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Create optimization session
session = OptimizationSession(
    kernel_id="matmul_relu",
    semantic_ir=semantic_ir,
    target_hw="nvidia_ampere",
    llm_runner=llm_runner
)
```

#### 7.4.2 Optimization Loop

```python
# Run optimization with Claude as decision-maker
result = session.optimize(
    budget_tokens=50000,
    max_iterations=20,
    strategy="bounded_actions"  # LLM selects from legal_actions
)

# Result contains:
# - decisions: list of optimization decisions with @rationale
# - trajectory: JSONL of all LLM interactions
# - best_strategy: highest-performing strategy found
# - metrics: throughput, memory, compilation time
```

#### 7.4.3 Tool Invocation

Claude calls Arke tools via native tool_use:

```json
{
  "type": "tool_use",
  "id": "toolu_01...",
  "name": "get_legal_actions",
  "input": {
    "kernel_id": "matmul_relu",
    "target_hw": "nvidia_ampere",
    "current_strategy_ir": {...}
  }
}
```

ArkeEnv responds with:

```json
{
  "legal_actions": [
    {"kind": "tile", "dim": "M", "legal_factors": [64, 128, 256]},
    {"kind": "fuse", "candidates": [["n1", "n2"]]},
    {"kind": "compute", "legal_threads": [128, 256, 512]}
  ],
  "constraints": {...},
  "hw_profile": {...}
}
```

Claude selects an action and provides @rationale:

```json
{
  "type": "tool_use",
  "name": "apply_decision",
  "input": {
    "decision": {
      "kind": "tile",
      "dim": "M",
      "factors": [128, 8],
      "rationale": "128 aligns with tensor core 16x8x16 shape, 8 for warp size 32"
    }
  }
}
```

### 7.5 Prompt Engineering for Claude

#### 7.5.1 System Prompt Structure

The system prompt is divided into 4 segments (as per §6.2):

```python
system_prompt = f"""
{GLOBAL_KNOWLEDGE}  # Arke IR spec, Op Registry, optimization principles

{HARDWARE_PROFILE}  # GPU specs, memory hierarchy, tensor core shapes

{SEMANTIC_IR}       # Current kernel semantics, input/output shapes

{OPTIMIZATION_STATE}  # Current strategy, decisions made, performance metrics
"""
```

#### 7.5.2 Key Prompt Principles

1. **Emphasize bounded action space**
   ```
   "You have access to legal_actions enumerated by the compiler.
    Select from these actions only. Do not propose arbitrary transformations."
   ```

2. **Require @rationale for every decision**
   ```
   "Every decision must include a rationale field explaining:
    - Why this decision improves performance
    - How it respects hardware constraints
    - What trade-offs it makes"
   ```

3. **Provide hardware context**
   ```
   "Target: NVIDIA Ampere (RTX 3060)
    - Shared memory: 96 KB per block
    - Max threads: 1024 per block
    - Tensor cores: 16x8x16 (FP32)
    - Warp size: 32"
   ```

4. **Show verification feedback**
   ```
   "Previous decision: tile(M, [256])
    V0 validation: PASS (constraints satisfied)
    V1 numerical: PASS (output matches reference)
    V2 performance: 120 GFLOPS (target: 150 GFLOPS)
    Suggestion: Increase parallelism or reduce memory pressure"
   ```

### 7.6 Extended Thinking (Opus Only)

For complex optimization problems, enable Claude's extended thinking:

```python
session = OptimizationSession(
    ...,
    use_thinking=True,
    thinking_budget=10000  # tokens for thinking
)
```

Claude will use thinking to:
- Analyze the optimization landscape
- Reason about trade-offs
- Plan multi-step strategies
- Verify constraint satisfaction

**Example thinking output:**
```
<thinking>
Let me analyze this matmul optimization:
1. Current bottleneck: memory bandwidth (40% utilization)
2. Legal actions: tile(M, [64,128,256]), tile(N, [64,128,256]), compute(threads=[128,256,512])
3. Strategy: Increase tile size to reduce memory pressure, then increase parallelism
4. Constraint check: 256x256 tile requires 256*256*4 bytes = 256KB shared memory (available: 96KB) → FAIL
5. Revised: 128x128 tile = 64KB (OK), then parallelize with 256 threads
</thinking>
```

### 7.7 Learning & Trajectory Logging

Every optimization run generates a trajectory JSONL for learning:

```jsonl
{"turn": 1, "llm_model": "claude-opus-4-6", "action": "get_legal_actions", "result": {...}}
{"turn": 2, "llm_model": "claude-opus-4-6", "decision": {"kind": "tile", "dim": "M", "factors": [128, 8], "rationale": "..."}}
{"turn": 3, "llm_model": "claude-opus-4-6", "verification": {"v0": "PASS", "v1": "PASS", "v2_gflops": 120}}
{"turn": 4, "llm_model": "claude-opus-4-6", "decision": {"kind": "fuse", "nodes": ["n1", "n2"], "rationale": "..."}}
```

These trajectories enable:
- **Supervised fine-tuning** — train Claude on successful optimization patterns
- **Reinforcement learning** — reward high-performance decisions
- **Knowledge transfer** — adapt strategies across kernels and hardware

### 7.8 Error Handling & Fallback

#### 7.8.1 Rate Limiting

```python
try:
    response = llm_runner.call(messages, tools)
except RateLimitError:
    logger.warning("Claude rate limited, switching to fallback")
    llm_runner.switch_to_fallback()  # → Sonnet 4.6
    response = llm_runner.call(messages, tools)
```

#### 7.8.2 Invalid Decisions

```python
if not verify_decision(decision):
    # V0 validation failed
    feedback = f"Decision invalid: {error_message}. Legal actions: {legal_actions}"
    messages.append({"role": "user", "content": feedback})
    # Claude re-reasons and proposes new decision
```

#### 7.8.3 Timeout Handling

```python
try:
    response = llm_runner.call(messages, tools, timeout=300)
except TimeoutError:
    logger.error("Claude call timed out")
    # Use best strategy found so far
    return result_with_best_strategy_so_far()
```

### 7.9 Cost Optimization

#### 7.9.1 Token Budgeting

```python
# Estimate tokens before calling Claude
estimated_tokens = estimate_tokens(
    system_prompt=system_prompt,
    messages=messages,
    tools=tools
)

if estimated_tokens > 150000:  # 75% of 200K limit
    compact_optimization_context()  # Reduce context size
```

#### 7.9.2 Prompt Caching

Use Anthropic's prompt caching for repeated kernels:

```python
# System prompt is cached (rarely changes)
system_prompt = {
    "type": "text",
    "text": GLOBAL_KNOWLEDGE + HARDWARE_PROFILE,
    "cache_control": {"type": "ephemeral"}
}

# Semantic IR is cached per kernel
semantic_ir_block = {
    "type": "text",
    "text": json.dumps(semantic_ir),
    "cache_control": {"type": "ephemeral"}
}
```

### 7.10 Example: End-to-End Optimization

```python
from arke.agent import LLMRunner, OptimizationSession
from arke.ir import SemanticIR
import json

# 1. Load kernel
with open("kernels/matmul_relu.json") as f:
    semantic_ir = SemanticIR.from_json(json.load(f))

# 2. Initialize Claude backend
llm_runner = LLMRunner(
    provider="anthropic",
    model="claude-opus-4-6",
    thinking_enabled=True
)

# 3. Create optimization session
session = OptimizationSession(
    kernel_id="matmul_relu",
    semantic_ir=semantic_ir,
    target_hw="nvidia_ampere",
    llm_runner=llm_runner
)

# 4. Run optimization
result = session.optimize(
    budget_tokens=50000,
    max_iterations=20
)

# 5. Inspect results
print(f"Best throughput: {result.best_strategy.metrics.throughput_gflops} GFLOPS")
print(f"Decisions made: {len(result.decisions)}")
for decision in result.decisions:
    print(f"  - {decision.kind}: {decision.rationale}")

# 6. Save trajectory for learning
with open("trajectory.jsonl", "w") as f:
    for event in result.trajectory:
        f.write(json.dumps(event) + "\n")
```

### 7.11 Best Practices

1. **Start with Sonnet 4.6** for most kernels; use Opus 4.6 for novel/complex optimizations
2. **Enable thinking mode** for kernels with >10 decision points
3. **Log all trajectories** for offline analysis and learning
4. **Use prompt caching** for repeated kernel families
5. **Monitor token usage** and compact context when approaching limits
6. **Provide hardware context** explicitly in system prompt
7. **Require @rationale** for every decision (enables auditability)
8. **Implement fallback chain** for production robustness
9. **Test on reference hardware** before deploying optimized kernels
10. **Collect feedback** from V2 performance profiling to improve future decisions

---

*Document version: v1.1 | Updated: 2026-04-09*
*References: e2e-flow.md, op-registry-interface.md, arke-ir-spec.md, arke/agent/ source*
