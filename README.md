# Arke

> **Let LLMs write the kernels. Let compilers check the math.**

---

**Arke** (*/ˈɑːrki/*) is an AI-native language and compiler toolchain for describing and optimizing GPU/NPU tensor operators — where LLM agents make optimization decisions and deterministic compilers verify every step.

## About the Name

**Arke** (Ἄρκη) — a swift-footed messenger goddess in Greek mythology. Zeus later gave her iridescent wings to Thetis as a wedding gift, symbolizing speed and brilliance.

In our context, Arke is the messenger between two worlds — translating **what to compute** (semantic intent) into **how to compute it** (hardware-specific strategy), through rapid, iterative AI-driven optimization cycles.

## Key Features

- 🤖 **AI-First Design** — LLM agents as optimization decision makers, not just code generators
- 🪙 **Minimal-Token Efficiency** — `.ak` kernels express compute intent in an order of magnitude fewer tokens than hand-written low-level code. LLMs spend tokens on *what to compute*, not *how to implement it*, maximizing optimization throughput per token budget
- 🔗 **Semantic/Strategy Separation** — "What to compute" and "how to optimize" are independent, enabling safe exploration
- 🛡️ **Compiler-Verified** — Every LLM decision validated by deterministic checks (static → numerical → performance)
- ⚡ **Extreme Performance** — LLM-guided strategy search achieves vendor-library-level performance across hardware targets
- 🎯 **Multi-Hardware** — Single kernel definition targets NVIDIA, Ascend, and beyond (Stage 2+)
- 💬 **`@rationale` Annotations** — Every optimization decision carries a natural language explanation, making AI reasoning auditable

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

The same fused matmul+relu kernel in different representations (GPT-4 tokenizer):

| Representation | Tokens | Ratio | Result |
|:--------------|-------:|:-----:|:-------|
| **Arke `.ak`** (kernel only) | 72 | **1×** | ≥100% cuBLAS, compiler-verified |
| **Arke `.ak`** (kernel + strategy) | 160 | 2× | Full optimization specification |
| Triton (autotuned, hand-written) | 1,102 | 15× | Requires expert knowledge |
| LLM direct-write Triton | 563 | 8× | 83% correct, inconsistent perf |

Arke's semantic representation lets an LLM express **intent** in ~70 tokens,
then the compiler handles tiling, fusion, memory layout, and autotuning —
decisions that otherwise cost hundreds of tokens of fragile Triton code.

**Why this matters:**
- **Lower cost** — 8–15× fewer output tokens per kernel
- **Faster iteration** — More optimization attempts within the same context window
- **Higher reliability** — Compiler guarantees correctness; LLM only decides *what*, not *how*

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
Stage 2: Arke → MLIR Dialect → Multi   Break free from deps
Stage 3: Arke → LLVM IR → All HW       Full compiler stack
```

---

### Stage 1: Arke → Triton → GPU — Validate Hypothesis

**Goal:** Prove that LLM + structured IR + compiler verification produces kernels that are both correct and **faster than LLM-written Triton**, using Triton as the codegen backend.

**Target hardware:** NVIDIA Ampere (RTX 3060)

| Phase | Objective | Gate Criteria | Status |
|:-----:|:----------|:-------------|:------:|
| **1.0** | Environment setup | `make setup` bootstraps venv + PyTorch + Triton + CUDA on a fresh machine; GPU smoke test passes | ✅ |
| **1.1** | IR + Validation foundation | Semantic IR covers ≥10 ops; Strategy IR covers ≥6 decision types; V0 static + V1 numerical validators pass on all ops; ≥100 tests | ✅ |
| **1.2** | Codegen + E2E pipeline | Manual strategy → Triton → GPU execution, correctness verified (same-dtype NumPy ref), **perf ≥ 70% cuBLAS** | ✅ (105-160%) |
| **1.3** | LLM agent integration | LLM completes full tool-use optimization loop using ≥8 tools, applies ≥4 decisions, zero human intervention | ✅ |
| **1.4** | LLM closed-loop optimization | LLM-optimized matmul/softmax/fused kernels all pass GPU correctness, **perf ≥ 50% cuBLAS** | ✅ (106%) |
| **1.5** | Evaluation + comparison | ≥5 benchmark tasks; Arke correctness ≥ LLM-direct-Triton; **Arke mean perf ≥ LLM-direct-Triton mean perf**; variance ≤ direct | 🔨 |
| **1.6** | .ak parser + CLI | `.ak` → AST → Semantic IR for ≥3 kernels; `arke parse/optimize/inspect` CLI commands functional | ⬜ |
| **1.7** | Whole-model E2E | GPT-2 Small with ≥2 Arke-replaced ops, output matches PyTorch reference, **inference latency ≤ torch.compile** | ⬜ |
| **1.8** | MVP release | CI green on 3 Python versions; API docs complete; evaluation report with reproducible data; v0.1.0 tag | ⬜ |

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
- **GPU correctness verification** — Same-dtype comparison (Triton output vs NumPy reference at matching precision)
- **Accuracy benchmark framework** — 10 metrics, 3-tier verdict (accept/review/reject), per-dtype thresholds
- **237 tests passing** (including GPU correctness tests)

### Gate Status

| Gate | Validates | Criteria | Status |
|:----:|:----------|:---------|:------:|
| G0 | Environment feasibility | Triton matmul runs on RTX 3060 | ✅ |
| G1 | IR expressiveness | Known-good strategy representable in Arke IR | ✅ |
| G2 | E2E pipeline | Manual strategy → codegen → **perf ≥ 70% cuBLAS** | ✅ (105-160%) |
| G3 | LLM feasibility | LLM tool-use → **matmul perf ≥ 50% cuBLAS** + softmax correct | ✅ (106%) |
| G4 | Comparative advantage | **Arke perf ≥ LLM-direct-Triton perf** across ≥5 tasks | 🔨 |
| G5 | Whole-model benefit | GPT-2 Small w/ Arke kernels **latency ≤ torch.compile** | ⬜ |

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
```

## Documentation

| Document | Description |
|----------|-------------|
| [plan-v3.0.md](docs/design/plan-v3.0.md) | Execution plan — Phase definitions, SMART criteria, Gate milestones |
| [e2e-flow.md](docs/design/e2e-flow.md) | End-to-end flow — user input to GPU execution walkthrough |
| [design-review.md](docs/design/design-review.md) | Design review — assumption validation, risk matrix |
| [naming-system.md](docs/design/naming-system.md) | Naming conventions — global terminology rules |

| Spec | Description |
|------|-------------|
| [arke-language-spec.md](docs/spec/arke-language-spec.md) | Arke language spec — syntax, type system, built-in ops |
| [arke-ir-spec.md](docs/spec/arke-ir-spec.md) | Arke IR spec — Semantic IR / Strategy IR structure |

## License

[Apache License 2.0](LICENSE)
