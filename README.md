# Arke

> **Let LLMs write the kernels and choose the optimizations. Let compilers verify the result.**

---

## Overview

Arke is a native LLM programming language, IR, compiler toolchain, and agent engineering system for GPU/NPU kernels. It combines benchmark systems to drive LLM for extreme kernel of operator functionality and performance generalization.

## Status & Version Semantics

- **Python package / CLI release:** `0.2.0.dev0` — the current prerelease line exposed by `pyproject.toml` and `arke.__version__`.
- **Arke Language schema:** `2.0.0` — the canonical `.ak` surface documented in [docs/spec/arke-lang-spec.md](docs/spec/arke-lang-spec.md).
- **Arke IR / `.akir` schema:** `2.0.0` — the active multi-layer IR contract documented in [docs/spec/arke-ir-spec.md](docs/spec/arke-ir-spec.md).
- **Other version labels in the repo:** supporting specs may keep their own document revision numbers, and roadmap docs may mention project release tags such as `v1.0.0`; those are not the active `.ak` / `.akir` schema identifiers.
- **Repository policy:** the active tree is **v2-only** for language/IR semantics. Historical drafts and migration notes live in git history, not in the active docs/code path.

## Key Features

### Project-Level Features

- 🤖 **AI-First Design** — Arke treats LLMs as optimization decision makers, not just code generators.
- 🔗 **Semantic/Strategy Separation** — "What to compute" and "how to optimize" are represented independently, enabling safe and reversible strategy exploration.
- 🪙 **Minimal-Token Efficiency** — The path from kernel definition through optimization and verification is designed to minimize token consumption.
- 🛡️ **Compiler-Verified Optimization** — Optimization decisions are validated through deterministic checks, from static legality to numerical correctness and performance.
- 💬 **`@rationale` as a First-Class Artifact** — Decisions carry natural-language explanations that make optimization trajectories auditable, reusable, and learnable.
- ⚡ **Cross-Hardware Performance Ambition** — A single semantic definition can lower toward multiple hardware targets while preserving a consistent optimization model.

### The LLM-Native Stack

#### 📝 LLM-Native Language

Arke's `.ak` language is a compact operator description surface for both humans and LLMs. It separates `kernel` semantics from `strategy` decisions so the mathematical definition remains stable while optimization policy evolves independently.

#### 🧬 LLM-Native IR

Arke IR makes the split explicit: `Semantic IR` captures what to compute, while `Strategy IR` captures how to optimize. This separation is the foundation for bounded action spaces, staged verification, rollback, and multi-backend lowering.

#### 🧰 LLM-Native Compiler Toolchain

The compiler is more than code generation. It enumerates legal actions, checks IR validity, lowers to backend-specific representations, and measures correctness and performance under a structured verification flow.

#### 🤖 AI Agent System

Arke's agent layer drives the optimization loop itself: analyze the kernel, choose legal actions, apply decisions with `@rationale`, verify outcomes, rollback when necessary, and iterate under compiler-enforced constraints.

## Architecture At A Glance

At a high level: `Semantic IR` defines **what** to compute, `Strategy IR` defines **how** to optimize, the compiler validates and lowers, and the agent iterates within that structured space.

```text
  Natural language │ Python/Triton | CUDA/Ascend C │ .ak source │ Benchmarks │ ...
                             │
                             │ LLM translates
                             ▼
  ┌────────────────────────────────────────────────────────────┐
  │  .ak — Arke Language (AI-Native Operator Programming)      │
  │  kernel { semantics }    strategy { @rationale decisions } │
  └────────────────────────────┬───────────────────────────────┘
                               │ parse
                               ▼
  ┌────────────────────────────────────────────────────────────┐
  │            Semantic IR — WHAT to compute                   │
  │         (immutable math, graph structure, correctness)     │
  └────────────────────────────┬───────────────────────────────┘
                               │
  ┌────────────────────────────▼───────────────────────────────┐
  │   LLM(Agent loop) ◄══ Structured Protocol ══► Compiler     │
  │                                                            │
  │   analyze → choose → apply → verify → rollback → iterate   │
  │                                                            │
  │  LLM Agent (Decides)       ArkeEnv (Verifies)              │
  │  ┌──────────────────┐      ┌─────────────────────────────┐ │
  │  │ analyze kernel   │─────►│ enumerate legal_actions     │ │
  │  │ select action    │◄─────│ (bounded decision space)    │ │
  │  │ apply @rationale │─────►│ validate: V0(<1ms)→V1→V2    │ │
  │  │ iterate / stop   │◄─────│ checkpoint / rollback       │ │
  │  └──────────────────┘      └───────────────┬─────────────┘ │
  │                                            │               │
  │  ┌─────────────────────────────────────────▼─────────────┐ │
  │  │           Strategy IR — HOW to optimize               │ │
  │  |    explicit decisions, rationale, backend-aware flow  | │
  │  └───────────────────────────────────────────────────────┘ │
  └────────────────────────────┬───────────────────────────────┘
                               │
  ┌────────────────────────────▼───────────────────────────────┐
  │  Codegen Backends (progressive depth into hardware)        │
  │                                                            │
  │   Triton   │  MLIR Dialect  │   LLVM IR   │   HW ISA       │
  │  (Phase 1) │   (Phase 3)   │  (Phase 4)  │  (Future)       │
  │                                                            │
  │  ◄── deeper hardware control ── extreme performance ──►    │
  └────────────────────────────┬───────────────────────────────┘
                               │
  ┌────────────────────────────▼───────────────────────────────┐
  │      GPU / NPU Execution: NVIDIA │ Ascend │ AMD │ ...      │
  └────────────────────────────────────────────────────────────┘
```

- `Semantic IR` is the source of truth for correctness-oriented reasoning.
- `Strategy IR` keeps optimization decisions explicit instead of burying them in free-form backend code.
- The compiler owns legality, validation, lowering, and measurement.
- The agent operates inside a bounded, inspectable optimization loop instead of an open-ended code generation loop.

## Minimal Example

Arke separates pure computation from optimization policy. The `kernel` block says what to compute; the `strategy` block says how to optimize it for a target.

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A=A, B=B);
    let Y = relu(X=C);
    return Y;
}

strategy fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("align tiles with the target's execution structure");
    tile(loop="j", factors=[128, 8])
        @rationale("improve memory coalescing on the output path");
    fuse(ops=["matmul", "relu"], type=epilogue)
        @rationale("remove an intermediate write to global memory");
}
```

## Why This Design Works

- **Verifiable** — semantics stay separate from optimization, so correctness can be checked against a stable computation definition.
- **Searchable** — optimization is expressed as explicit decisions rather than hidden inside handwritten backend code.
- **LLM-friendly** — the language and IR reduce token-heavy boilerplate while preserving enough structure for planning and validation.
- **Portable** — semantics remain stable while lowering and strategy specialization adapt to hardware targets.

## Token Efficiency

Arke is designed to reduce token usage across the full optimization loop, not just the surface syntax of a kernel definition.

| Representation                      | Tokens | Ratio  |
| ----------------------------------- | ------ | ------ |
| **Arke `.ak`** (kernel only)        | 72     | **1x** |
| **Arke `.ak`** (kernel + strategy)  | 160    | 2x     |
| LLM direct-write Triton             | 563    | 8x     |
| Triton (autotuned, hand-written)    | 1,102  | 15x    |

- **Definition** — semantic intent is represented directly instead of backend boilerplate.
- **Search** — optimization steps become compact actions, not whole-program rewrites.
- **Verification** — deterministic checks replace long back-and-forth debugging loops.
- **Iteration** — invalid strategies rollback cleanly without regenerating everything.

For a deeper analysis, see [docs/architecture/token-efficiency-analysis.md](docs/architecture/token-efficiency-analysis.md).

## Quick Start

### Prerequisites

- Linux (tested on Ubuntu / WSL2)
- Python 3.10+
- NVIDIA GPU and CUDA for the GPU-oriented setup paths

### One-Click Setup

```bash
git clone https://github.com/arke-lang/arke.git
cd arke
make setup
```

Other setup profiles:

```bash
make setup-cpu
make setup-gpu
make setup-bench
```

You can also use the bootstrap script directly:

```bash
scripts/bootstrap_env.sh cpu-dev
scripts/bootstrap_env.sh gpu-dev
scripts/bootstrap_env.sh bench
```

### Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

### First Verified Commands

The current top-level CLI exposes the compiler-facing path and the Stage 8 MVP optimization path:

```bash
arke compile examples/operators/01_matmul.ak
arke optimize examples/operators/01_matmul.ak --output /tmp/arke-opt --cycles 3 --json
```

To write the resulting `.akir` JSON from the compiler path to a file:

```bash
arke compile examples/operators/01_matmul.ak -o /tmp/matmul.akir
```

For environment details and custom venv paths, see [docs/architecture/python-environment-setup.md](docs/architecture/python-environment-setup.md).

## Current CLI

Today, the documented package entry points in the current prerelease distribution are:

- `arke compile <file.ak>` — compile `.ak` source into Arke IR / `.akir` JSON
- `arke optimize <file.ak>` — Stage 8 MVP flow: generate bounded StrategyIR, validate/lower, and emit machine-readable optimization artifacts

`arke optimize` currently accepts `.ak` file input and uses a deterministic heuristic strategy generator by default. It emits `strategy.json`, `result.akir`, `trajectory.jsonl`, and `summary.json` so agent workflows can validate the compile→profile→adjust contract before the live LLM provider path is enabled.

Design documents describe richer optimization flows and agent-driven workflows; read those as architecture and roadmap material unless a specific interface is documented here and implemented in the package entry points.

If you are checking versions: the package stays on a `0.x` prerelease track while the active language and IR schemas are both `2.0.0`. See [docs/spec/arke-lang-spec.md#11-versioning](docs/spec/arke-lang-spec.md#11-versioning) and [docs/spec/arke-ir-spec.md#15-versioning](docs/spec/arke-ir-spec.md#15-versioning).

## Roadmap Snapshot

Arke is developed in four phases:

- **Phase 1** — Arke -> Triton -> NVIDIA GPU: validate the SIMT path, language/IR, compiler infrastructure, and benchmark system
- **Phase 2** — Arke -> Triton -> Ascend NPU: validate cross-architecture generalization on SIMD hardware
- **Phase 3** — Arke -> MLIR Dialect: gain deeper compiler control beyond Triton's abstraction boundary
- **Phase 4** — Arke -> LLVM IR: pursue lower-level backend completeness and broader hardware coverage

The active roadmap, Gate criteria, and stage details live in [docs/roadmap/plan.md](docs/roadmap/plan.md).

## Documentation Guide

### Start Here

The roadmap, Gate definitions, and benchmark terminology are maintained in the following docs:
- [docs/roadmap/plan.md](docs/roadmap/plan.md) — development roadmap, stages, Gates, and project-level status
- [docs/benchmark/benchmark-design.md](docs/benchmark/benchmark-design.md) — benchmark model and the `BL` / `OT` / `ST` / `L` terminology used throughout the project
- [docs/architecture/e2e-flow.md](docs/architecture/e2e-flow.md) — end-to-end system walkthrough

### Active Specs

- [docs/spec/arke-lang-spec.md](docs/spec/arke-lang-spec.md) — active Arke language specification
- [docs/spec/arke-ir-spec.md](docs/spec/arke-ir-spec.md) — active multi-layer IR specification
- [docs/spec/symbolic-dimension-spec.md](docs/spec/symbolic-dimension-spec.md) — symbolic dimension model
- [docs/spec/pass-infrastructure-spec.md](docs/spec/pass-infrastructure-spec.md) — pass system specification
- [docs/spec/op-registry-interface.md](docs/spec/op-registry-interface.md) — operator registry contract

### Architecture And Design

- [docs/architecture/arke-lang-spec-design.md](docs/architecture/arke-lang-spec-design.md) — language design rationale
- [docs/architecture/arke-ir-spec-design.md](docs/architecture/arke-ir-spec-design.md) — IR design rationale
- [docs/architecture/arke-compiler-infrastructure.md](docs/architecture/arke-compiler-infrastructure.md) — compiler infrastructure design
- [docs/architecture/agent-design.md](docs/architecture/agent-design.md) — agent architecture and integration modes
- [docs/architecture/naming-system.md](docs/architecture/naming-system.md) — canonical terminology and naming
- [docs/architecture/python-environment-setup.md](docs/architecture/python-environment-setup.md) — environment bootstrap details
- [docs/architecture/token-efficiency-analysis.md](docs/architecture/token-efficiency-analysis.md) — token-cost analysis

### Benchmark System

- [docs/benchmark/benchmark-design.md](docs/benchmark/benchmark-design.md) — benchmark overview
- [docs/benchmark/benchmark-ops.md](docs/benchmark/benchmark-ops.md) — operator tiers and catalog
- [docs/benchmark/benchmark-shapes.md](docs/benchmark/benchmark-shapes.md) — shape tiers and matrices
- [docs/benchmark/benchmark-protocol.md](docs/benchmark/benchmark-protocol.md) — measurement and scoring protocol
- [docs/benchmark/benchmark-csv-spec.md](docs/benchmark/benchmark-csv-spec.md) — benchmark result schema
- [docs/benchmark/operator-source-registry.md](docs/benchmark/operator-source-registry.md) — baseline source registry

### Stage Plans

- [docs/phase1/stage6-plan.md](docs/phase1/stage6-plan.md) — compiler infrastructure
- [docs/phase1/stage7-plan.md](docs/phase1/stage7-plan.md) — Lang and IR v2
- [docs/phase1/stage8-plan.md](docs/phase1/stage8-plan.md) — agent autonomy
- [docs/phase1/stage9-plan.md](docs/phase1/stage9-plan.md) — Phase 1 finalization
- [docs/phase1/design-review.md](docs/phase1/design-review.md) — design review and risk analysis
- [docs/phase1/completion-summary.md](docs/phase1/completion-summary.md) — completed Phase 1 summary for earlier stages

### Repository History Policy

The active tree documents the current v2 language and IR surfaces only. Historical migration notes and superseded design drafts were removed during the Stage 7 spec cleanup so they do not compete with active references. Use git history when older drafts must be inspected for archaeology or provenance.

## Project Structure

```text
arke/         core language, IR, compiler, backend, and agent packages
benchmarks/   benchmark runners, baselines, reports, and result artifacts
docs/         roadmap, specs, architecture notes, and stage plans
examples/     example `.ak` operators and walkthrough materials
tests/        automated coverage for language, compiler, benchmarks, and agent-adjacent flows
scripts/      bootstrap and project utility scripts
```

## About The Name

**Arke** (Ἄρκη) is named after the swift-footed messenger of Greek mythology. In the context of this project, the name reflects a bridge between semantic intent and hardware-specific execution strategy.

## License

[Apache License 2.0](LICENSE)
