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
- 💬 `**@rationale` Annotations** — Every optimization decision carries a natural language explanation, making AI reasoning auditable
- ⚡ **Extreme Performance** — LLM-guided strategy search achieves vendor-library-level performance across hardware targets
- 🔗 **Semantic/Strategy Separation** — "What to compute" and "how to optimize" are independent, enabling safe exploration
- 🛡️ **Compiler-Verified** — Every LLM decision validated by deterministic checks (static → numerical → performance)
- 🎯 **Multi-Hardware** — Single kernel definition targets NVIDIA, Ascend, and beyond 

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
- `**@rationale` is auditable** — every optimization carries a natural language explanation, making LLM reasoning transparent

### Token Efficiency

Arke minimizes token consumption across the full optimization pipeline — not just the kernel definition, but the entire path from specification to peak performance.

**Representation cost** (GPT-4 tokenizer, fused matmul+relu):


| Representation                     | Tokens | Ratio  |
| ---------------------------------- | ------ | ------ |
| **Arke `.ak`** (kernel only)       | 72     | **1×** |
| **Arke `.ak`** (kernel + strategy) | 160    | 2×     |
| LLM direct-write Triton            | 563    | 8×     |
| Triton (autotuned, hand-written)   | 1,102  | 15×    |


**End-to-end optimization cost** — beyond representation, Arke reduces tokens at every stage:

- **Kernel definition**: Semantic IR captures intent; no tiling/masking/memory boilerplate
- **Strategy search**: Each optimization decision is a structured action (~~10 tokens), not a full code rewrite (~~500 tokens)
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

Four stages from hypothesis validation to full compiler stack:

```
Stage 1: Arke → Triton → NVIDIA GPU     SIMT feasibility ✅  (complete)
Stage 2: Arke → Triton → Ascend NPU     SIMD feasibility     (next)
Stage 3: Arke → MLIR Dialect            Full compiler control
Stage 4: Arke → LLVM IR                 100% hardware completeness
```

**Core principle:** Each Stage validates Arke's capabilities on a new architectural dimension.
NVIDIA (Stage 1) proves SIMT feasibility. Ascend (Stage 2) proves cross-architecture
generalization. MLIR (Stage 3) removes Triton's abstraction ceiling. LLVM IR (Stage 4)
achieves maximum hardware expression completeness and performance headroom.

> Gate benchmark coverage: Stage 2+ Gates must cover ≥3 Operator Categories and
> include LLM-production shapes (LLaMA/DeepSeek/Qwen). See [BENCHMARK.md](docs/design/BENCHMARK.md).

---

## Documentation


| Document                                         | Description                                                                 |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| [plan-v3.0.md](docs/design/plan-v3.0.md)         | Execution plan — Phase definitions, SMART criteria, Gate milestones         |
| [gate-redesign.md](docs/design/gate-redesign.md) | **Gate system v3** — Function > Accuracy > Performance, Tier 3 verification |
| [e2e-flow.md](docs/design/e2e-flow.md)           | End-to-end flow — user input to GPU execution walkthrough                   |
| [design-review.md](docs/design/design-review.md) | Design review — assumption validation, risk matrix                          |
| [naming-system.md](docs/design/naming-system.md) | Naming conventions — global terminology rules                               |



| Spec                                                     | Description                                            |
| -------------------------------------------------------- | ------------------------------------------------------ |
| [arke-language-spec.md](docs/spec/arke-language-spec.md) | Arke language spec — syntax, type system, built-in ops |
| [arke-ir-spec.md](docs/spec/arke-ir-spec.md)             | Arke IR spec — Semantic IR / Strategy IR structure     |


### Benchmark System


| Document                                                                           | Description                                                                                                                             |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [docs/design/BENCHMARK.md](docs/design/BENCHMARK.md)                               | **Complete benchmark reference** — architecture, baselines, shapes, scoring, quality gates, operator sources, CLI, development workflow |
| [benchmarks/results/EVALUATION_REPORT.md](benchmarks/results/EVALUATION_REPORT.md) | **Stage 1 Evaluation Report** — all gates, L1/L2/L3 data, conclusions                                                                   |


### Stage 1: Arke → Triton → GPU — Validate Hypothesis

**Goal:** Prove that LLM + structured IR + compiler verification produces kernels that are both correct and **faster than LLM-written Triton**, using Triton as the codegen backend.

**Target hardware:** NVIDIA Ampere (RTX 3060)


| Phase   | Objective               | Gate | Exit Criteria                                                                             | Status |
| ------- | ----------------------- | ---- | ----------------------------------------------------------------------------------------- | ------ |
| **1.0** | Environment setup       | G0   | `make setup` → venv + PyTorch + Triton + CUDA; GPU smoke test; ≥100 tests                 | ✅      |
| **1.1** | IR + Validation         | G1   | ≥10 ops, ≥6 strategy types, IR round-trip 100%; **Tier 3 full numerical validation 100%** | ✅      |
| **1.2** | Codegen + Pipeline      | G2   | IR → Triton → GPU; **Tier 3 accuracy 100%**; perf geomean ≥60% cuBLAS                     | ✅      |
| **1.3** | LLM agent integration   | —    | LLM uses ≥8 tools, applies ≥4 decisions, zero human intervention                          | ✅      |
| **1.4** | LLM closed-loop         | G3   | Agent closed-loop optimization; **Tier 3 sampled 10 shapes accuracy 100%**; perf observed | ✅      |
| **1.5** | Evaluation + comparison | G4   | Arke correct ≥ LLM-direct; perf ≥90% direct, ≥70% FlagGems; token ≤60%                    | ✅      |
| **1.6** | .ak parser + CLI        | —    | `.ak` → AST → IR for ≥3 kernels; `arke parse/optimize/inspect` CLI                        | ✅      |
| **1.7** | Whole-model E2E         | G5   | **Multi-config accuracy 100%**; latency ≤1.15× eager; mem ≤6GB; ≥48 ops replaced          | ✅      |
| **1.8** | MVP release             | —    | CI green ×3 Python; API docs 99%; evaluation report; v0.1.0 tgitag                        | ✅      |


> **Post Stage 1 TODO:** Evaluate implementation language for Stages 2–3. Consider compile-time performance, MLIR/LLVM C++API integration ergonomics, deployment binary size, and whether a Rust/C++ rewrite of the compiler core (keeping Python for agent/LLM layer) is warranted.

---

### Stage 2: Arke → Ascend Triton — SIMD Architecture Validation

**Goal:** Verify Arke Lang/IR works on SIMD architecture (Ascend NPU) via Ascend Triton backend.
Arke-generated Ascend kernels must **outperform FlagGems on Ascend**.
Complete Arke Lang/IR to cover Operator Categories B–E (Attention, Norm, Activation, Positional Encoding).

**Hardware target:** Huawei Ascend 910B


| Phase   | Objective                      | Gate  | Key Criteria                                                                                               |
| ------- | ------------------------------ | ----- | ---------------------------------------------------------------------------------------------------------- |
| **2.1** | Ascend env + IR completeness   | —     | Ascend Triton smoke test; Cat B/D/E ops in OP_CATALOG; SIMD decision kinds in Strategy IR                  |
| **2.2** | Ascend Triton codegen          | S2-G1 | Cat A+C+D correctness 100% (Tier 2); LLM makes SIMD-aware decisions                                        |
| **2.3** | Ascend performance vs FlagGems | S2-G2 | matmul+rmsnorm+swiglu geomean ≥ FlagGems/Ascend                                                            |
| **2.4** | FlashAttention + GQA           | S2-G3 | FA Tier 4 ≥12 shapes correct on NVIDIA (≥0.7× FA-2); ≥8 shapes correct on Ascend; DeepSeek shapes included |
| **2.5** | RoPE/YaRN + @rationale         | S2-G4 | RoPE/YaRN correctness incl. DeepSeek 8K-32K; @rationale ≥10% cross-arch lift                               |


**Stage 2 Summary Gate (S2-G_FINAL):**
Cat A+B+C+D+E all passing • Tier 2 + Tier 4 shapes • Ascend perf ≥ FlagGems • H4 (cross-hardware) verified

---

### Stage 3: Arke → MLIR Dialect — Full Compiler Control

**Goal:** Replace Triton backend with Arke MLIR Dialect for NVIDIA + Ascend.
MLIR removes Triton's abstraction ceiling, enabling deeper LLM decisions (Level 2: loop nest, memory access)
and more complete operator support. Performance must **match or exceed Stage 2 Triton path**.

**Hardware target:** NVIDIA + Ascend (via NVVM dialect + AscendNPU IR)


| Phase   | Objective                 | Gate  | Key Criteria                                                                                           |
| ------- | ------------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| **3.1** | Arke MLIR Dialect design  | —     | `arke.kernel` + `arke.strategy` dialect; Strategy IR Level 2 fields (loop_nest, memory_access_pattern) |
| **3.2** | MLIR correctness (NVIDIA) | S3-G1 | Cat A+C+D via MLIR: Tier 2+4 100% correct                                                              |
| **3.3** | MLIR performance ≥ Triton | S3-G2 | All Cat A+B+C+D MLIR geomean ≥ Stage 2 Triton                                                          |
| **3.4** | LLM Level-2 decisions     | S3-G3 | LLM L1+2 ≥ L1+default L2 by ≥15% (matmul), ≥20% (FA)                                                   |
| **3.5** | Ascend via MLIR           | S3-G4 | matmul+rmsnorm correct on Ascend via MLIR; perf ≥ Stage 2                                              |


**Stage 3 Summary Gate (S3-G_FINAL):**
Cat A+B+C+D+E via MLIR • Tier 2+Tier 4 • MLIR ≥ Triton (NVIDIA+Ascend) • LLM Level-2 value verified

---

### Stage 4: Arke → LLVM IR — 100% Hardware Completeness

**Goal:** Full compiler stack emitting LLVM IR directly, achieving **100% hardware expression completeness**
and performance headroom beyond MLIR. Supports ≥3 hardware backends (NVIDIA + Ascend + AMD).
LLM Level-3 decisions (register, barrier, instruction scheduling) become possible.

**Hardware target:** NVIDIA + Ascend + AMD (≥3 backends)


| Phase   | Objective                     | Gate       | Key Criteria                                                  |
| ------- | ----------------------------- | ---------- | ------------------------------------------------------------- |
| **4.1** | LLVM IR emission              | S4-G1      | matmul via LLVM correct on ≥2 backends (Tier 2)               |
| **4.2** | LLVM performance ≥ MLIR       | S4-G2      | Cat A+C+D LLVM geomean ≥ MLIR + 5%                            |
| **4.3** | LLM Level-3 decisions         | S4-G3      | LLM L1+2+3 ≥ L1+2+default L3 by ≥5%                           |
| **4.4** | FlashAttention/MLA + multi-HW | S4-G4      | FA via LLVM ≥0.85× FA-2; MLA correct; ≥3 backends ≥90% vendor |
| **4.5** | Production release v1.0.0     | S4-G_FINAL | pip package; ≥3 platforms; @rationale KB ≥200 entries         |


**Stage 4 Summary Gate (S4-G_FINAL):**
Cat A+B+C+D+E+F via LLVM • ≥3 backends ≥90% vendor • LLM Level 1-3 full stack • v1.0.0

---

## 📋 Current Progress (Stage 1)

### Key Achievements

- **LLM closed-loop optimization** — Claude Sonnet 4.6 autonomously optimizes matmul+relu through 23 tool calls, zero errors
- **151% cuBLAS** — LLM Agent kernel reaches 151.4% cuBLAS at 2048²
- **Stage 1 Gates G0-G5 PASS** — All gates pass (G5: 3 known-fail perf criteria)
- **Multi-tier benchmark system** — 6 baselines, 3 layers (L1/L2/L3), 7 operator categories, 4 shape tiers
- **305 tests passing** (including GPU correctness tests)

### Current Focus: Gate G6 — Lang & IR Completeness

G6 validates the foundation for all subsequent development:
- Full `.ak → SemanticIR → StrategyIR → Triton → GPU` pipeline
- Cat A+B+C+D operator expression completeness at Tier 2 shapes
- Token efficiency:  lines < equivalent Triton kernel lines
- Python interop, MLIR structural mapping, Spec v1.0 freeze

### Gate Status

> Gate design: **Function > Accuracy > Performance** — see [gate-redesign.md](docs/design/gate-redesign.md) for full SMART criteria with Tier 3 verification.


| Gate | Type          | Validates           | Key Criteria                                                                               | Status |
| ---- | ------------- | ------------------- | ------------------------------------------------------------------------------------------ | ------ |
| G0   | Function      | Environment         | CUDA + Triton + GPU execution + ≥100 tests                                                 | ✅      |
| G1   | Func+Acc      | IR & Validation     | ≥10 ops, ≥6 decision types, **Tier 3 numerical validation 100%**                           | ✅ ⚠️   |
| G2   | Func+Acc+Perf | Codegen quality     | **Tier 3 accuracy 100%**; perf: ≥50% shapes ≥50% cuBLAS, geomean ≥60%                      | ✅      |
| G3   | Func+Acc      | LLM agent           | ≥8 tools, ≥4 decisions, closed-loop autonomous; **Tier 3 sampled accuracy 100%**           | ✅      |
| G4   | Acc+Perf      | Arke vs baselines   | Arke correct ≥ LLM-direct; **perf ≥90% direct, ≥70% FlagGems**; token ≤60%                 | ✅      |
| G5   | Acc+Perf      | E2E integration     | **Multi-config accuracy 100%**; latency ≤1.15× eager (⚠️ known-fail); mem ≤6GB             | ✅ ⚠️   |
| G6   | Func+Acc      | Lang & IR complete  | `.ak` full E2E pipeline; Cat A-D expression; @rationale; token efficiency; spec v1.0       | ⬜      |
| G7   | Func          | Autonomous pipeline | Multi-input (.ak/NL/code) → auto kernel gen; I/O spec defined                             | ⬜      |
| G8   | Analysis      | Language decision   | Python vs alternatives; critical path; hybrid assessment; decision doc                     | ⬜      |


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

## License

[Apache License 2.0](LICENSE)