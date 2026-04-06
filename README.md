# Arke

> **Let LLMs write the kernels. Let compilers check the math.**

---

**Arke** (*/ˈɑːrki/*) is an AI-native operator programming language and compiler toolchain for GPU/NPU tensor operators. The entire pipeline — from kernel definition through LLM-driven strategy search to peak performance — is designed for minimal token consumption, with optional `@rationale` annotations capturing expert knowledge to guide optimization. A single kernel definition targets NVIDIA, Ascend, and beyond, achieving vendor-library-level performance across hardware targets.

## About the Name

**Arke** (Ἄρκη) — a swift-footed messenger goddess in Greek mythology. Zeus later gave her iridescent wings to Thetis as a wedding gift, symbolizing speed and brilliance.

In our context, Arke is the messenger between two worlds — translating **what to compute** (semantic intent) into **how to compute it** (hardware-specific strategy), through rapid, iterative AI-driven optimization cycles.

## Key Features

### LLM-Native Language

- **Semantic/Strategy Separation** — "What to compute" (immutable math) and "how to optimize" (searchable decisions) are independent layers, enabling LLMs to explore strategies without risking correctness
- **Minimal-Token End-to-End** — The entire pipeline — definition, search, verification, iteration — consumes an order of magnitude fewer tokens than direct code generation
- **Bounded Action Space** — LLMs select from compiler-enumerated legal actions, not free-form code — turning optimization into navigating a decision tree
- **`@rationale` Annotations** — Every optimization decision carries a natural language explanation, preserved in IR as a first-class construct for auditability and cross-hardware knowledge transfer

### LLM-Native Compiler Toolchain

- **Compiler-as-Verifier** — The compiler does not optimize; it verifies every LLM decision through progressive checks: V0 Static (<1ms) → V1 Numerical → V2 Performance
- **Structured LLM-Compiler Protocol** — LLM and compiler interact through a closed-loop tool-use API (analyze → decide → verify → iterate), not free-text generation
- **Safe Exploration** — Checkpoint/rollback with correctness-first gating (Function > Accuracy > Performance); LLMs explore boldly because invalid decisions are caught at V0 in under 1ms
- **Multi-Hardware** — Single kernel definition targets NVIDIA, Ascend, and beyond; strategy adapts per hardware, semantics stay fixed

## Architecture

```
      Python │ Triton │ CUDA │ Natural Language │ ...
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
  │         (immutable computation graph, pure math)           │
  └────────────────────────────┬───────────────────────────────┘
                               │
  ┌────────────────────────────▼───────────────────────────────┐
  │         LLM ◄══ Structured Protocol ══► Compiler           │
  │                                                            │
  │  LLM Agent (Decides)       ArkeEnv (Verifies)              │
  │  ┌──────────────────┐      ┌─────────────────────────────┐ │
  │  │ analyze kernel   │─────►│ enumerate legal_actions     │ │
  │  │ select action    │◄─────│ (bounded decision space)    │ │
  │  │ apply @rationale │─────►│ validate: V0(<1ms)→V1→V2    │ │
  │  │ iterate / stop   │◄─────│ checkpoint / rollback       │ │
  │  └──────────────────┘      └──────────────┬──────────────┘ │
  │                                            │               │
  │  ┌─────────────────────────────────────────▼──────────────┐│
  │  │  Strategy IR — HOW to optimize (decision-by-decision)  ││
  │  └────────────────────────────────────────────────────────┘│
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

Four phases from hypothesis validation to full compiler stack:

```
Phase 1: Arke → Triton → NVIDIA GPU     SIMT MVP ✅; G6–G8 (lang/IR + autonomy) in progress
Phase 2: Arke → Triton → Ascend NPU     SIMD feasibility     (next)
Phase 3: Arke → MLIR Dialect            Full compiler control
Phase 4: Arke → LLVM IR                 100% hardware completeness
```

**Core principle:** Each Phase validates Arke's capabilities on a new architectural dimension.
NVIDIA (Phase 1) proves SIMT feasibility. Ascend (Phase 2) proves cross-architecture
generalization. MLIR (Phase 3) removes Triton's abstraction ceiling. LLVM IR (Phase 4)
achieves maximum hardware expression completeness and performance headroom.

---

## Documentation


| Document                                                                       | Description                                                                 |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| [plan.md](docs/roadmap/plan.md)                                                  | Development plan — single source of truth for all phases/stages/tasks                                       |
| [completion-summary.md](docs/phase1/completion-summary.md)                        | Phase 1 G0-G5 completion summary                                                          |
| [benchmark-design.md](docs/benchmark/benchmark-design.md)                         | Complete benchmark reference — baselines, shapes, scoring, operator sources |
| [e2e-flow.md](docs/architecture/e2e-flow.md)                                         | End-to-end flow — user input to GPU execution walkthrough                   |
| [design-review.md](docs/phase1/design-review.md)                               | Design review — assumption validation, risk matrix                          |
| [naming-system.md](docs/architecture/naming-system.md)                               | Naming conventions — global terminology rules                               |
| [benchmarks/results/phase1/EVALUATION_REPORT.md](benchmarks/results/phase1/EVALUATION_REPORT.md) | Phase 1 Evaluation Report — all gates, L1/L2/L3 data, conclusions |


| Spec                                                     | Description                                            |
| -------------------------------------------------------- | ------------------------------------------------------ |
| [arke-lang-spec-design.md](docs/spec/arke-lang-spec-design.md) | Arke language spec v2.0 design |
| [arke-ir-spec-design.md](docs/spec/arke-ir-spec-design.md)     | Arke IR spec — multi-layer architecture design         |


### Phase 1: Arke → Triton → GPU — Validate Hypothesis

**Goal:** Prove LLM + structured IR + compiler verification produces correct, fast kernels.
**Hardware:** NVIDIA Ampere (RTX 3060 Laptop, 6GB)

| Gate | Status | Milestone |
|:----:|:------:|:----------|
| G0-G4 | ✅ | Environment, IR, Codegen, Agent, Evaluation |
| G5 | ✅ | OT0-2 full shapes + GPT-2 E2E (correctness ✅, latency ⚠️ known-fail) |
| G6 | ⬜ | **BL5×L1+L2 — Lang & IR completeness** (45 ops × all shapes) |
| G7 | ⬜ | **Autonomous Engineering** — self-directed kernel gen + LLaMA-2/DS-V2 E2E |
| G8 | ⬜ | **Phase 1 Final** — 4 model E2E + Arke vs LLM-direct |

→ Development plan: [plan.md](docs/roadmap/plan.md)

---

### Phase 2: Arke → Ascend Triton/AscendNPU IR — SIMD Architecture Validation

**Goal:** Verify Arke Lang/IR works on SIMD architecture (Ascend NPU) via Ascend Triton backend.
Arke-generated Ascend kernels must **outperform FlagGems on Ascend**.
Complete Arke Lang/IR to cover Operator Categories B–E (Attention, Norm, Activation, Positional Encoding).

**Hardware target:** Huawei Ascend 910B


| Phase   | Objective                      | Gate  | Key Criteria                                                                                               |
| ------- | ------------------------------ | ----- | ---------------------------------------------------------------------------------------------------------- |
| **2.1** | Ascend env + IR completeness   | —     | Ascend Triton smoke test; Cat B/D/E ops in OP_CATALOG; SIMD decision kinds in Strategy IR                  |
| **2.2** | Ascend Triton codegen          | P2-S1 | Cat A+C+D correctness 100% (Tier 2); LLM makes SIMD-aware decisions                                        |
| **2.3** | Ascend performance vs FlagGems | P2-S2 | matmul+rmsnorm+swiglu geomean ≥ FlagGems/Ascend                                                            |
| **2.4** | FlashAttention + GQA           | P2-S3 | FA Tier 4 ≥12 shapes correct on NVIDIA (≥0.7× FA-2); ≥8 shapes correct on Ascend; DeepSeek shapes included |
| **2.5** | RoPE/YaRN + @rationale         | P2-S4 | RoPE/YaRN correctness incl. DeepSeek 8K-32K; @rationale ≥10% cross-arch lift                               |


**Phase 2 Summary Gate (P2-S_FINAL):**
Cat A+B+C+D+E all passing • Tier 2 + Tier 4 shapes • Ascend perf ≥ FlagGems • H4 (cross-hardware) verified

---

### Phase 3: Arke → MLIR Dialect — Full Compiler Control

**Goal:** Replace Triton backend with Arke MLIR Dialect for NVIDIA + Ascend.
MLIR removes Triton's abstraction ceiling, enabling deeper LLM decisions (Level 2: loop nest, memory access)
and more complete operator support. Performance must **match or exceed Phase 2 Triton path**.

**Hardware target:** NVIDIA + Ascend (via NVVM dialect + AscendNPU IR)


| Phase   | Objective                 | Gate  | Key Criteria                                                                                           |
| ------- | ------------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| **3.1** | Arke MLIR Dialect design  | —     | `arke.kernel` + `arke.strategy` dialect; Strategy IR Level 2 fields (loop_nest, memory_access_pattern) |
| **3.2** | MLIR correctness (NVIDIA) | P3-S1 | Cat A+C+D via MLIR: Tier 2+4 100% correct                                                              |
| **3.3** | MLIR performance ≥ Triton | P3-S2 | All Cat A+B+C+D MLIR geomean ≥ Phase 2 Triton                                                          |
| **3.4** | LLM Level-2 decisions     | P3-S3 | LLM L1+2 ≥ L1+default L2 by ≥15% (matmul), ≥20% (FA)                                                   |
| **3.5** | Ascend via MLIR           | P3-S4 | matmul+rmsnorm correct on Ascend via MLIR; perf ≥ Phase 2                                              |


**Phase 3 Summary Gate (P3-S_FINAL):**
Cat A+B+C+D+E via MLIR • Tier 2+Tier 4 • MLIR ≥ Triton (NVIDIA+Ascend) • LLM Level-2 value verified

---

### Phase 4: Arke → LLVM IR — 100% Hardware Completeness

**Goal:** Full compiler stack emitting LLVM IR directly, achieving **100% hardware expression completeness**
and performance headroom beyond MLIR. Supports ≥3 hardware backends (NVIDIA + Ascend + AMD).
LLM Level-3 decisions (register, barrier, instruction scheduling) become possible.

**Hardware target:** NVIDIA + Ascend + AMD (≥3 backends)


| Phase   | Objective                     | Gate       | Key Criteria                                                  |
| ------- | ----------------------------- | ---------- | ------------------------------------------------------------- |
| **4.1** | LLVM IR emission              | P4-S1      | matmul via LLVM correct on ≥2 backends (Tier 2)               |
| **4.2** | LLVM performance ≥ MLIR       | P4-S2      | Cat A+C+D LLVM geomean ≥ MLIR + 5%                            |
| **4.3** | LLM Level-3 decisions         | P4-S3      | LLM L1+2+3 ≥ L1+2+default L3 by ≥5%                           |
| **4.4** | FlashAttention/MLA + multi-HW | P4-S4      | FA via LLVM ≥0.85× FA-2; MLA correct; ≥3 backends ≥90% vendor |
| **4.5** | Production release v1.0.0     | P4-S_FINAL | pip package; ≥3 platforms; @rationale KB ≥200 entries         |


**Phase 4 Summary Gate (P4-S_FINAL):**
Cat A+B+C+D+E+F via LLVM • ≥3 backends ≥90% vendor • LLM Level 1-3 full stack • v1.0.0

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