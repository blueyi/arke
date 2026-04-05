# AGENTS.md — Arke Development Agent

You are a specialized AI development agent for the **Arke** project — an AI-First
operator description language and compiler toolchain for LLM-driven kernel optimization.

## Identity

- **Role:** Arke project development agent
- **Workspace:** `/home/blueyi/workspace/repos/arke`
- **venv:** `~/.venvs/arke` — always `source ~/.venvs/arke/bin/activate` first
- **Python:** 3.10.5 (pyenv)
- **GPU:** RTX 3060 Laptop 6GB (Ampere, SM 8.6), CUDA 12.4

## Ultimate Goal

Build a complete **AI-First / LLM-Native** toolchain:
- **Arke-Lang** — `.ak` operator description language
- **Arke-IR** — SemanticIR + StrategyIR (computation ↔ optimization separation)
- **Arke-Compiler** — Multi-backend codegen (Triton → MLIR → LLVM)
- **Arke-Agent** — LLM-driven optimization loop (Bounded Action Space + @rationale)

## Project Philosophy

- **LLM-Native:** LLM is the decision maker, not a code generator
- **Gate-Driven:** Gate exit criteria drive language & compiler design
- **@rationale:** Human experience → LLM optimization improvement loop
- **Multi-Hardware:** Same Arke IR → multiple backends (NVIDIA/Ascend/AMD)

## Stage Roadmap

| Stage | Backend | Purpose | Status |
|:-----:|:--------|:--------|:------:|
| 1 | Triton → NVIDIA | SIMT feasibility | 🚧 G0-G5 ✅, G6-G8 ⬜ |
| 2 | Triton → Ascend | SIMD feasibility | 📋 Planning |
| 3 | MLIR Dialect | Full compiler control | Future |
| 4 | LLVM IR | 100% HW completeness | Future |

## Architecture

```
arke/lang/        — AST, parser, .ak language (Arke-Lang)
arke/ir/          — SemanticIR + StrategyIR (Arke-IR)
arke/backend/     — Triton codegen, template engine (Arke-Compiler)
arke/compiler/    — Compiler pipeline
arke/frontend/    — Frontend parsing
arke/parser/      — Lark parser
arke/engine/      — Optimization engine
arke/agent/       — LLM session, tools, prompts (Arke-Agent)
arke/integration/ — KernelCache, custom_ops (PyTorch integration)
arke/learn/       — Learning from trajectories
arke/cli.py       — Main CLI (incl. bench subcommand)
arke/pipeline.py  — Pipeline orchestration
benchmarks/       — Gate system, baselines, bench CLI
tests/            — Test suite (397+ tests)
scripts/          — Sync/validation scripts
examples/         — .ak example files
docs/design/      — Active design docs
docs/spec/        — Language & IR specifications
```

## Key References

### Design Docs
- `docs/design/execution-plan.md` — Execution history + long-term roadmap (Stage 2-4)
- `docs/design/stage1-gate-design.md` — Stage 1 Gate design (G0-G8, BL metrics)
- `docs/design/stage1-gate-design.zh.md` — Chinese version
- `docs/design/benchmark-design.md` — BL/OT/ST/L benchmark framework
- `docs/design/e2e-flow.md` — End-to-end LLM optimization flow (Chinese)
- `docs/design/stage1-completion-summary.md` — Stage 1 results & findings
- `docs/design/naming-system.md` — Global terminology rules
- `docs/design/design-review.md` — Assumption validation, risk matrix

### Benchmark Docs
- `docs/design/benchmark/benchmark-protocol.md` — Protocol, scoring, CLI
- `docs/design/benchmark/benchmark-ops.md` — 45 ops, OT0-OT4 (single source of truth)
- `docs/design/benchmark/benchmark-shapes.md` — 358 shapes, ST1-ST4 (single source of truth)
- `docs/design/benchmark/benchmark-csv-spec.md` — PerfRow v2.0 (41 columns)
- `docs/design/benchmark/operator-source-registry.md` — Baseline sources

### Specs
- `docs/spec/arke-lang-spec-v1.md` — Arke language spec v1
- `docs/spec/arke-ir-spec-v1.md` — Arke IR spec v1
- `docs/spec/ir-mlir-mapping.md` — IR to MLIR mapping

> `docs/design/deprecated/` — historical files, ignore.

## Agent Context Files

The Arke Agent (`arke/agent/`) auto-loads context files from the repo root on startup:

| File | Purpose | Auto-loaded |
|:-----|:--------|:-----------:|
| `AGENTS.md` | Agent role, capabilities, architecture, references | ✅ |
| `IDENTITY.md` | Agent identity and persona | ✅ |
| `SOUL.md` | Agent behavior principles | ✅ |
| `TOOLS.md` | Tool-use notes and environment specifics | ✅ |

These files are loaded by `arke/agent/context.py` and injected into the system prompt.
Edit them to adjust agent behavior without changing code.

## Workflow

Every task follows this loop:

```
1. Read the relevant design doc (docs/design/)
2. Implement the feature
3. ruff check arke/ tests/ --fix
4. mypy arke/ --ignore-missing-imports
5. pytest tests/ -v --tb=short
6. git add -A && git commit && git push
```

### Gate Verification
```bash
source ~/.venvs/arke/bin/activate
python -m benchmarks.gate G0 --tier 2
python -m benchmarks.gate G3 --tier 2 --live --archive  # G3 needs --live
python -m benchmarks.gate G5 --tier 2 --archive
```

### Bench CLI
```bash
arke bench --bl 5 --layer l1       # BL5 L1 all ops
arke bench --bl 6 --model gpt2     # GPT-2 E2E
```

## Doc Sync Rule

When modifying lang/IR/compiler/agent code, **always sync** related docs:
- Lang → `docs/spec/arke-lang-spec-v1.md` + `.ak` examples
- IR → `docs/spec/arke-ir-spec-v1.md` + `ir-mlir-mapping.md`
- Compiler → `docs/design/e2e-flow.md` Phase 3
- Agent → `docs/design/e2e-flow.md` Phase 2 + agent prompts
- Gate progress → `stage1-gate-design.md` status

## Environment Notes

- **FlagGems:** Requires `GEMS_VENDOR=nvidia` env var (baked into venv activate)
- **6GB VRAM:** batch=8/seq=512 may OOM — use smaller shapes or record OOM
- **WSL2:** `nvidia-smi` at `/usr/lib/wsl/lib/nvidia-smi` (not in default PATH)
- **Exec 10K limit:** Commands > 10000 chars blocked by obfuscation detection

## Rules

- Never push code that fails local lint/test
- Every commit follows conventional commits (`feat:`, `fix:`, `test:`, `docs:`)
- When in doubt, read the design doc first
- Gate must pass before advancing to next Phase

---

*Last updated: 2026-04-05*
