# Arke

> **Let LLMs write the kernels. Let compilers check the math.**

---

**Arke** (*/ˈɑːrki/*) is an AI-native language and compiler toolchain for describing and optimizing GPU/NPU tensor operators — where LLM agents make optimization decisions and deterministic compilers verify every step.

## About the Name

**Arke** (Ἄρκη) — a swift-footed messenger goddess in Greek mythology. Zeus later gave her iridescent wings to Thetis as a wedding gift, symbolizing speed and brilliance.

In our context, Arke is the messenger between two worlds — translating **what to compute** (semantic intent) into **how to compute it** (hardware-specific strategy), through rapid, iterative AI-driven optimization cycles.

## Key Features

- 🤖 **AI-First Design** — LLM agents as optimization decision makers, not just code generators
- 🪙 **Minimal-Token Efficiency** — Arke minimizes end-to-end token consumption from kernel definition through optimization to peak performance
- 💬 **`@rationale` Annotations** — Every optimization decision carries a natural language explanation, making AI reasoning auditable
- ⚡ **Extreme Performance** — LLM-guided strategy search achieves vendor-library-level performance across hardware targets
- 🔗 **Semantic/Strategy Separation** — "What to compute" and "how to optimize" are independent, enabling safe exploration
- 🛡️ **Compiler-Verified** — Every LLM decision validated by deterministic checks (static → numerical → performance)
- 🎯 **Multi-Hardware** — Single kernel definition targets NVIDIA, Ascend, and beyond (Stage 2+)

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │              LLM Agent (Decision Maker)       │
                │   analyze → decide → verify → iterate         │
                └──────────┬──────────────┬────────────────────┘
                           │ tool-use     │ tool results
                ┌──────────▼──────────────▼────────────────────┐
                │              ArkeEnv (Compiler Env)            │
                │                                                │
                │  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │
                │  │Semantic  │→ │Strategy  │→ │  Codegen     │  │
                │  │IR        │  │IR        │  │  (Triton)    │  │
                │  └─────────┘  └──────────┘  └──────┬───────┘  │
                │                                     │          │
                │  ┌─────────────────────────────────▼────────┐ │
                │  │ Validation: V0 Static → V1 Numeric → V2  │ │
                │  │                                  Perf     │ │
                │  └───────────────────────────────────────────┘ │
                └────────────────────────────────────────────────┘
```

## Quick Example

Arke separates **what to compute** from **how to optimize** — the kernel author declares pure math, and the optimization strategy is a separate, machine-searchable artifact that an LLM agent can explore and refine.

```arke
// ─── Semantic Layer: WHAT to compute ───
// Pure math declaration. No tiling, no thread mapping, no hardware details.
// This is the single source of truth for correctness verification.
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);    // Matrix multiplication
    let Y = relu(C);          // Elementwise activation
    return Y;
}

// ─── Strategy Layer: HOW to optimize ───
// Separate from the kernel — can be searched, modified, or regenerated
// without changing the computation semantics.
// Each decision is a discrete, reversible action the LLM agent can explore.
strategy fused_matmul_relu for target("nvidia_ampere") {
    // Tile the i-loop into 64×16 blocks
    // → 64 maps to L2 cache lines, 16 maps to warp size
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");

    // Tile the j-loop for memory coalescing
    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing");

    // Fuse relu into matmul as an epilogue
    // → eliminates one global memory round-trip
    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

**Why this design:**
- **Correctness is verifiable** — the semantic layer is pure math, checkable against a NumPy reference
- **Strategy is searchable** — each decision (tile size, fusion, placement) is a discrete action an LLM can enumerate, apply, and rollback
- **`@rationale` is auditable** — every optimization carries a natural language explanation, making LLM reasoning transparent

### Token Efficiency

Arke minimizes token consumption across the full optimization pipeline — not just the kernel definition, but the entire path from specification to peak performance.

**Representation cost** (GPT-4 tokenizer, fused matmul+relu):

| Representation | Tokens | Ratio |
|:--------------|-------:|:-----:|
| **Arke `.ak`** (kernel only) | 72 | **1×** |
| **Arke `.ak`** (kernel + strategy) | 160 | 2× |
| LLM direct-write Triton | 563 | 8× |
| Triton (autotuned, hand-written) | 1,102 | 15× |

**End-to-end optimization cost** — beyond representation, Arke reduces tokens at every stage:

- **Kernel definition**: Semantic IR captures intent; no tiling/masking/memory boilerplate
- **Strategy search**: Each optimization decision is a structured action (~10 tokens), not a full code rewrite (~500 tokens)
- **Verification**: Compiler checks correctness deterministically — no multi-turn "fix this bug" debugging loops
- **Iteration**: Failed strategies roll back cleanly; the LLM tries the next option without regenerating the whole kernel

The result: LLMs reach peak-performance kernels spending a fraction of the tokens that direct code generation requires.

## Python API

```python
import arke

result = arke.optimize(
    kernel="matmul",
    shape=[1024, 512, 2048],
    target="nvidia_ampere",
    llm="anthropic"
)

print(result.performance)  # vs vendor library baseline
print(result.trajectory)   # Full optimization trace
```

---

## 🗺️ Roadmap

Three stages from hypothesis validation to full compiler stack:

```
Stage 1: Arke → Triton → GPU          Validate hypothesis    (current)
Stage 2: Arke → MLIR Dialect → Multi   Multi-hardware support
Stage 3: Arke → LLVM IR → All HW       Full compiler stack
```

---

### Stage 1: Arke → Triton → GPU — Validate Hypothesis

**Goal:** Prove that LLM + structured IR + compiler verification produces kernels that are both correct and **faster than LLM-written Triton**, using Triton as the codegen backend.

**Target hardware:** NVIDIA Ampere (RTX 3060)

| Phase | Objective | Gate | Exit Criteria | Status |
|:-----:|:----------|:----:|:-------------|:------:|
| **1.0** | Environment setup | G0 | `make setup` → venv + PyTorch + Triton + CUDA; GPU smoke test; ≥100 tests | ✅ |
| **1.1** | IR + Validation | G1 | ≥10 ops, ≥6 strategy types, IR round-trip 100%; **Tier 3 全量数值验证 100%** | ✅ |
| **1.2** | Codegen + Pipeline | G2 | IR → Triton → GPU; **Tier 3 精度 100%**; 性能 geomean ≥60% cuBLAS | ✅ |
| **1.3** | LLM agent integration | — | LLM uses ≥8 tools, applies ≥4 decisions, zero human intervention | ✅ |
| **1.4** | LLM closed-loop | G3 | Agent 闭环优化; **Tier 3 抽样 10 shapes 精度 100%**; 性能观测记录 | ✅ |
| **1.5** | Evaluation + comparison | G4 | Arke correct ≥ LLM-direct; perf ≥90% direct, ≥70% FlagGems; token ≤60% | ✅ |
| **1.6** | .ak parser + CLI | — | `.ak` → AST → IR for ≥3 kernels; `arke parse/optimize/inspect` CLI | ✅ |
| **1.7** | Whole-model E2E | G5 | **多配置精度 100%**; latency ≤1.15× eager; mem ≤6GB; ≥48 ops replaced | ✅ |
| **1.8** | MVP release | — | CI green ×3 Python; API docs 99%; evaluation report; v0.1.0 tag | ✅ |

> **Post Stage 1 TODO:** Evaluate implementation language for Stages 2–3. Consider compile-time performance, MLIR/LLVM C++ API integration ergonomics, deployment binary size, and whether a Rust/C++ rewrite of the compiler core (keeping Python for agent/LLM layer) is warranted.

---

### Stage 2: Arke → MLIR Dialect → Multi-Hardware

**Goal:** Replace Triton backend with a custom MLIR dialect. Generate hardware-specific code for multiple targets from a single Arke IR, **matching or exceeding Triton-backend performance**.

**Target hardware:** NVIDIA + Huawei Ascend

| Phase | Objective | Gate Criteria |
|:-----:|:----------|:-------------|
| **2.1** | MLIR dialect design | Define Arke MLIR dialect ops; lower elementwise Arke IR → MLIR → LLVM IR; generated code produces correct output |
| **2.2** | NVIDIA codegen via MLIR | matmul via MLIR path, correctness verified (same-dtype ref), **perf ≥ Stage 1 Triton-backend result on same hardware** |
| **2.3** | Ascend backend prototype | matmul on Ascend 910B, correctness verified against same-dtype NumPy reference, **perf ≥ 50% Ascend CANN library** |
| **2.4** | Cross-hardware evaluation | Same Arke kernel → NVIDIA + Ascend, both correct; **NVIDIA perf ≥ Stage 1 baseline; Ascend perf ≥ 50% CANN** |

---

### Stage 3: Arke → LLVM IR → All Hardware

**Goal:** Full compiler stack emitting LLVM IR directly. Maximum control over optimization passes and code generation, targeting all major accelerators with **performance within 90% of vendor libraries**.

**Target hardware:** NVIDIA + Ascend + AMD + Intel

| Phase | Objective | Gate Criteria |
|:-----:|:----------|:-------------|
| **3.1** | Direct LLVM IR emission | Emit PTX/AMDGCN/SPIR-V from Arke IR; matmul correct on ≥2 backends |
| **3.2** | Custom optimization passes | Arke-specific LLVM passes for tiling, fusion, memory placement; **perf ≥ Stage 2 MLIR-backend on same ops** |
| **3.3** | Multi-target parity | Same kernel → ≥3 backends, all correct, **perf within 90% of each platform's vendor library** |
| **3.4** | Production release | Stable public API; pip/cargo package; benchmark suite across ≥3 hardware platforms; v1.0.0 tag |

---

## 📋 Current Progress (Stage 1)

### Key Achievements

- **LLM closed-loop optimization** — Claude Sonnet 4.6 autonomously optimizes matmul+relu through 23 tool calls, zero errors
- **106% cuBLAS** — LLM-optimized kernel outperforms NVIDIA's hand-tuned library
- **164% cuBLAS at 1024³** — Arke autotuned matmul beats cuBLAS by 64% (L1 benchmark)
- **Gate G5 PASS** — GPT-2 E2E inference at 1.01× eager (seq=128)
- **Multi-tier benchmark system** — 6 baselines (cuBLAS → FlagGems → Arke), 3 layers (L1/L2/L3), full provenance tracking
- **GPU correctness verification** — Same-dtype comparison (Triton output vs NumPy reference at matching precision)
- **Accuracy benchmark framework** — 10 metrics, 3-tier verdict (accept/review/reject), per-dtype thresholds
- **280 tests passing** (including GPU correctness tests)

### Gate Status

> Gate design: **Function > Accuracy > Performance** — see [gate-redesign.md](docs/design/gate-redesign.md) for full SMART criteria with Tier 3 verification.

| Gate | Type | Validates | Key Criteria | Status |
|:----:|:----:|:----------|:-------------|:------:|
| G0 | 功能 | Environment | CUDA + Triton + GPU execution + ≥100 tests | ✅ |
| G1 | 功能+精度 | IR & Validation | ≥10 ops, ≥6 decision types, **Tier 3 全量数值验证 100%** | ✅ |
| G2 | 功能+精度+性能 | Codegen quality | **Tier 3 精度 100%**; 性能: ≥50% shapes ≥50% cuBLAS, geomean ≥60% | ✅ |
| G3 | 功能+精度 | LLM agent | ≥8 tools, ≥4 decisions, 闭环无人工; **Tier 3 抽样 10 shapes 精度 100%** | ✅ |
| G4 | 精度+性能 | Arke vs baselines | Arke correct ≥ LLM-direct; **perf ≥90% direct, ≥70% FlagGems**; token ≤60% | ✅ |
| G5 | 精度+性能 | E2E integration | **多配置精度 100%** (3 seq × 3 batch); latency ≤1.15× eager; mem ≤6GB | ✅ |

---

## Getting Started

### Prerequisites

- Linux (tested on Ubuntu 22.04 / WSL2)
- NVIDIA GPU with CUDA ≥ 12.1 (tested on RTX 3060)
- Python 3.10+

### One-Click Setup

```bash
git clone https://github.com/arke-lang/arke.git
cd arke
make setup    # Creates venv, installs PyTorch + Triton + deps, verifies GPU
```

### Manual Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
pytest tests/ -q  # Run tests
```

### Run LLM Optimization Demo

```bash
# Set LLM provider (Anthropic example)
export ANTHROPIC_API_KEY=your-key-here

# Run matmul optimization
python examples/agent_matmul.py
```

---

## 🧰 Project Structure

```
arke/
├── ir/                        # IR Layer
│   ├── semantic.py            # Semantic IR (what to compute)
│   ├── strategy.py            # Strategy IR (how to optimize)
│   ├── builder.py             # Python → IR builder
│   ├── ops/                   # Operator catalog (10 ops)
│   ├── schemas/               # JSON Schemas
│   └── targets/               # HW profiles (nvidia_ampere)
├── engine/                    # Core Engine
│   ├── env.py                 # ArkeEnv (LLM ↔ compiler interface)
│   ├── legal_actions.py       # Legal action enumeration
│   ├── validator.py           # V0 static validation
│   ├── numerical_check.py     # V1 numerical validation
│   ├── accuracy.py            # Accuracy benchmark framework
│   └── reference_sources.py   # Pluggable reference sources
├── agent/                     # LLM Agent Runtime
│   ├── runner.py              # LLM optimization loop
│   ├── session.py             # Session lifecycle + GPU verification
│   ├── prompts.py             # System prompt builder
│   ├── llm_config.py          # LLM provider config
│   └── tools_schema.py        # 10 tool definitions
├── backend/                   # Code Generation
│   ├── triton_backend.py      # Triton backend (translate + compile + run)
│   ├── triton_template_engine.py
│   └── base.py                # Backend ABC
├── learn/                     # Learning System
│   └── trajectory.py          # JSONL trajectory export
├── lang/                      # Language Frontend (Phase 1.6)
│   └── ast.py                 # AST node definitions
├── pipeline.py                # E2E pipeline
└── examples/
    └── agent_matmul.py        # LLM agent optimization demo

benchmarks/                    # Benchmark System
├── baselines/                 # Baseline runners (P0-P5)
│   ├── cublas.py              # P0: cuBLAS/cuDNN via PyTorch
│   ├── flaggems.py            # P1: FlagGems (200+ Triton ops)
│   ├── liger.py               # P1: Liger-Kernel
│   ├── pytorch_eager.py       # P3: PyTorch eager
│   ├── inductor.py            # P4: torch.compile
│   └── arke_runner.py         # P5: Arke KernelCache
├── bench_l1.py                # L1: Single operator benchmarks
├── bench_l2.py                # L2: Fused operator benchmarks
├── bench_l3.py                # L3: E2E model benchmarks
├── cli.py                     # Unified CLI entry point
├── report.py                  # Markdown report generator
├── shapes.py                  # Shape matrix definitions
├── measure.py                 # CUDA event measurement utils
└── results/                   # Archived benchmark results
```

## Documentation

| Document | Description |
|----------|-------------|
| [plan-v3.0.md](docs/design/plan-v3.0.md) | Execution plan — Phase definitions, SMART criteria, Gate milestones |
| [gate-redesign.md](docs/design/gate-redesign.md) | **Gate system v3** — Function > Accuracy > Performance, Tier 3 verification |
| [e2e-flow.md](docs/design/e2e-flow.md) | End-to-end flow — user input to GPU execution walkthrough |
| [design-review.md](docs/design/design-review.md) | Design review — assumption validation, risk matrix |
| [naming-system.md](docs/design/naming-system.md) | Naming conventions — global terminology rules |

| Spec | Description |
|------|-------------|
| [arke-language-spec.md](docs/spec/arke-language-spec.md) | Arke language spec — syntax, type system, built-in ops |
| [arke-ir-spec.md](docs/spec/arke-ir-spec.md) | Arke IR spec — Semantic IR / Strategy IR structure |

### Benchmark System

| Document | Description |
|----------|-------------|
| [benchmarks/README.md](benchmarks/README.md) | Benchmark usage guide — CLI, layers, baselines, output format |
| [benchmarks/BENCHMARK_DESIGN.md](benchmarks/BENCHMARK_DESIGN.md) | Design — three-layer architecture, scoring system, quality gates |
| [benchmarks/OPERATOR_SOURCES.md](benchmarks/OPERATOR_SOURCES.md) | Operator source registry — 8 categories of GPU kernels with provenance |
| [benchmarks/SYNERGY.md](benchmarks/SYNERGY.md) | Benchmark ↔ Arke co-development — target-driven development loop |
| [benchmarks/results/EVALUATION_REPORT.md](benchmarks/results/EVALUATION_REPORT.md) | **Stage 1 Evaluation Report** — all gates, L1/L2/L3 data, conclusions |

## License

[Apache License 2.0](LICENSE)
