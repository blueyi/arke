# AGENTS.md — Arke Development Agent

You are a specialized AI development agent for the **Arke** project — an AI-First
operator description language and compiler toolchain for LLM-driven kernel optimization.

## Identity

- **Role:** Arke project development agent
- **Workspace:** `/home/blueyi/workspace/repos/arke`
- **venv:** `~/.venvs/arke` — always `source ~/.venvs/arke/bin/activate` first
- **Python:** 3.10.5 (pyenv)
- **GPU:** RTX 3060 Laptop 6GB (Ampere, SM 8.6), CUDA 12.4

## Project Philosophy

- **LLM-Native:** LLM is the decision maker, not a code generator
- **Gate-Driven:** Gate exit criteria drive language & compiler design
- **@rationale:** Human experience → LLM optimization improvement loop
- **Multi-Hardware:** Same Arke IR → multiple backends (NVIDIA/Ascend/AMD)

## Stage Roadmap

| Stage | Backend | Purpose | Status |
|:-----:|:--------|:--------|:------:|
| 1 | Triton → NVIDIA | SIMT feasibility | ✅ Complete |
| 2 | Triton → Ascend | SIMD feasibility | 📋 Planning |
| 3 | MLIR Dialect | Full compiler control | Future |
| 4 | LLVM IR | 100% HW completeness | Future |

## Key References

- **Execution plan:** `docs/design/plan-v3.0.md` — Phase definitions, SMART criteria, all Stages
- **Gate system:** `docs/design/gate-redesign.md` — Function > Accuracy > Performance, Tier verification
- **Benchmark:** `docs/design/BENCHMARK.md` — Operator categories, shapes, baselines, scoring, Stage 2-4 Gates
- **E2E flow:** `docs/design/e2e-flow.md` — User input to GPU execution walkthrough
- **Naming convention:** `docs/design/naming-system.md` — Global terminology rules
- **Design review:** `docs/design/design-review.md` — Assumption validation, risk matrix
- **Operator registry:** `docs/design/operator-source-registry.md` — Baseline sources
- **Language spec:** `docs/spec/arke-language-spec.md` — Syntax, type system, built-in ops
- **IR spec:** `docs/spec/arke-ir-spec.md` — Semantic IR / Strategy IR structure
- **Stage 1 summary:** `docs/design/stage1-completion-summary.md` — Stage 1 results & findings

### Deprecated (reference only)
- `docs/deprecated/plan-v2.1.md` — Superseded by plan-v3.0.md
- `docs/deprecated/detailed-design-v2.1.md` — Superseded by gate-redesign.md

## Workflow

Every task follows this loop:

```
1. Read the relevant design doc (docs/design/)
2. Implement the feature
3. ruff check arke/ arkec/ tests/ --fix
4. mypy arke/ arkec/ --ignore-missing-imports
5. pytest tests/ -v --tb=short
6. git add -A && git commit && git push
7. Check CI: gh run list --repo arke-lang/arke --limit 1
```

### Gate Verification
```bash
source ~/.venvs/arke/bin/activate
python -m benchmarks.gate G0 --tier 2
python -m benchmarks.gate G3 --tier 2 --live --archive  # G3 needs --live
python -m benchmarks.gate G5 --tier 2 --archive
```

## Architecture

```
arke/lang/        — AST, parser, .ak language
arke/ir/          — SemanticIR + StrategyIR
arke/backend/     — Triton codegen (template engine)
arke/agent/       — LLM session, tools, prompts
arke/integration/ — KernelCache, custom_ops (PyTorch integration)
benchmarks/       — gate system, baselines (cuBLAS/FlagGems/Inductor/LigerKernel/LLM-direct)
docs/design/      — active design docs, plans, BENCHMARK.md
docs/spec/        — language & IR specifications
docs/deprecated/  — superseded documents (reference only)
```

## Environment Notes

- **FlagGems:** Requires `GEMS_VENDOR=nvidia` env var (baked into venv activate)
- **6GB VRAM:** batch=8/seq=512 may OOM — use smaller shapes or record OOM
- **WSL2:** `nvidia-smi` at `/usr/lib/wsl/lib/nvidia-smi` (not in default PATH)

## Rules

- Never push code that fails local lint/test
- Every commit message follows conventional commits (`feat:`, `fix:`, `test:`, `docs:`)
- When in doubt, read the design doc first
- Gate must pass before advancing to next Phase
