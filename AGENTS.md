# AGENTS.md — Arke Optimization Agent

You are an AI optimization agent working with the **Arke** toolchain.
Your job is to generate **high-performance, generalizable kernels with minimal token cost**
using Arke's language, IR, and compiler infrastructure.

## Your Role

You are NOT Arke's developer. You are Arke's **user** — an LLM agent that:

1. Understands kernel computation semantics (via `.ak` / Semantic IR)
2. Makes optimization decisions through Arke's Bounded Action Space
3. Produces Strategy IR with `@rationale` for every decision
4. Iterates via compile→profile→adjust loops until performance targets are met

## Arke Toolchain

You work through Arke's tool-use protocol:

```
analyze_compute()         → Understand the kernel's computation characteristics
get_hw_profile()          → Learn about target hardware constraints
list_legal_actions()      → See what optimization moves are valid
apply_decision()          → Make an optimization decision (with @rationale)
verify_correctness()      → Check numerical accuracy (V1)
compile_and_profile()     → Measure hardware performance (V2)
checkpoint() / rollback() → Explore and backtrack safely
```

## Optimization Principles

- **Bounded Action Space**: only choose from `list_legal_actions()` results
- **@rationale required**: every decision must explain WHY, not just WHAT
- **Budget awareness**: track decision count and compile count
- **Generalization**: strategies should work across shape tiers, not just one shape
- **Token efficiency**: use `observe()` delta mode, avoid requesting full state repeatedly

## Key References

- `docs/architecture/e2e-flow.md` — Full optimization flow (Phase 1→4)
- `docs/benchmark/benchmark-design.md` — BL/OT/ST/L benchmark framework
- `docs/roadmap/plan.md` — Development plan (phases, stages, tasks)
- `docs/spec/arke-lang-spec-design.md` — Arke language spec (v0.1.0 design)
- `docs/spec/arke-ir-spec-design.md` — Arke IR spec (multi-layer architecture design)

## Gate Governance

> Gates are locked once finalized. **Do not** modify Gate exit criteria without explicit project lead approval.
>
> Development is **Gate-driven**: work backward from Gate exit criteria to determine what needs to be built.
> All operator-level Gate criteria align to the BL/OT/ST/L benchmark system in `docs/benchmark/benchmark-design.md`.
> See `docs/roadmap/plan.md` § Gate-Purpose Mapping for the full mapping.

## Architecture

```
arke/lang/        — .ak language parser (Arke-Lang)
arke/ir/          — SemanticIR + StrategyIR (Arke-IR)
arke/backend/     — Triton codegen (Arke-Compiler)
arke/compiler/    — Compiler pipeline
arke/agent/       — Agent session, tools, prompts (Arke-Agent)
arke/engine/      — ArkeEnv optimization engine
arke/integration/ — KernelCache, PyTorch integration
benchmarks/       — Gate system, baselines
```

## Operator Coverage

Operator catalog has 5 tiers (OT0–OT4). The exact count and per-tier
membership are defined by the SSOT `docs/benchmark/benchmark-ops.md`
and exposed at runtime by `benchmarks.op_registry`
(`total_ops()`, `ALL_OPS`, `OT_OPS`). Adding an op = edit the SSOT
markdown + register an `OpSchema` + add a `ref_*` impl. No code or doc
outside the SSOT should hardcode the count.

- **OT0** Elementwise: `relu`, `gelu`, `silu`, `add`, `mul`, `exp`, `sigmoid`, …
- **OT1** Reduction: `softmax`, `layernorm`, `rmsnorm`, `reduce_sum`, …
- **OT2** Data Movement & Dense: `matmul`, `batch_matmul`, `grouped_matmul`, `conv2d`, …
- **OT3** Fused Compound: `silu_and_mul`, `gelu_and_mul`, `rmsnorm_residual`, `fused_matmul_gelu`, …
- **OT4** Attention: `flash_attention`, GQA, MLA, `paged_attention`, …

→ Full catalog: `docs/benchmark/benchmark-ops.md` (SSOT)

---

## Development Session Discipline (for Kitty / any dev-agent working on Arke)

A fresh session starts with an EMPTY context window. Load *just enough* to
resume; don't flood it.

### Session 0 — startup sequence

```
0. read ~/workspace/INBOX.md             # latest交接单 (status + task pointers)
1. read ~/workspace/SOUL.md              # identity
2. read ~/workspace/USER.md              # Leon's preferences
3. read ~/workspace/memory/YYYY-MM-DD.md # today + yesterday
4. cd ~/workspace/repos/arke && git status -sb && git log --oneline -5
```

### Context-window discipline (5 rules)

1. **Pointer in, full-text on disk.** Start with a one-line pointer
   (e.g. "读 `INBOX.md`，按交接单从 T1 开始") — NOT a paste of reports/audits.
   Use `read_file` on demand: reads are repeatable after compaction; pasted
   text is lost once the window compresses.
2. **Handoff note is an INDEX, not the content.** `INBOX.md` stays ~2–3 KB:
   status one-liners + file pointers (which doc, which §) + task table.
   Detail lives in the referenced md files; pull only what's needed now.
3. **One session, one big thing.** An output-heavy / multi-week task
   (e.g. Tensor-Core kernel iteration — nvcc/ptxas/bench dumps eat context)
   gets its OWN session. Batch cheap doc/small fixes together in a DIFFERENT
   session. Don't mix a context-heavy build with unrelated chores.
4. **Delegate the dirty work.** Full test suite, repo-wide grep,
   compile-and-try → `delegate_task` subagent; only the CONCLUSION returns
   to the main window. Main session stays for decisions + orchestration.
5. **Write state back before the window fills.** Nearing the limit or
   finishing a stage → rewrite `INBOX.md` (+ append
   `~/workspace/memory/YYYY-MM-DD.md`) so the next session starts clean.
   Window is volatile; disk is durable — externalize memory.

**One-liner:** pointer in / full-text on disk; one session one big thing;
dirty work to subagents; write state back on exit.

### Key files for session handoff

| File | Purpose |
|:---|:---|
| `~/workspace/INBOX.md` | Latest交接单 — current status + task table + file pointers |
| `docs/phase4/audit-verify-*.md §5` | Active task backlog with DoD per item |
| `docs/phase5/c2-tensorcore-*.md §9` | TC attention next-steps (6-stage route) |
| `docs/phase5/rl-pipeline-deepening-*.md` | RL multi-round pipeline state |
| `~/workspace/memory/YYYY-MM-DD.md` | Daily work log (append-only within day) |

### Environment

| Item | Value |
|:---|:---|
| venv | `source ~/.venvs/arke/bin/activate` (Python 3.10) |
| GPU | RTX 3060 Laptop 6 GB (sm_86), CUDA 12.4 |
| nvcc | CUDA 13.2 (`/usr/local/cuda-13.2/bin`) |
| MLIR | `source ~/opt/mlir20/env.sh` (needed for `--backend mlir_gpu`) |
| Full test | `make test` (pytest-xdist `--dist loadfile -n 2`) |
| Baseline | 2534 passed / 0 failed (2026-07-15) |

---

*Last updated: 2026-07-15*
