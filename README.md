# Arke

> **Let LLMs write the kernels. Let compilers check the math.**
>
> **让大模型写算子，让编译器验算术。**

---

**Arke** (*/ˈɑːrki/*) is an AI-native language and compiler toolchain for describing and optimizing GPU/NPU tensor operators — where LLM agents make optimization decisions and deterministic compilers verify every step.

## About the Name

**Arke** (Ἄρκη) — the twin sister of Iris in Greek mythology. Both were messenger goddesses of the rainbow, but while Iris served the Olympians, Arke chose the Titans.

In our context:
- **Iris** represents the established path — traditional compilers with hand-written optimization rules
- **Arke** represents the new path — AI-driven optimization where LLMs bridge the gap between human intent and hardware reality

Just as the mythological Arke was swift-footed (Zeus later gave her iridescent wings to Thetis as a wedding gift), the system is designed for rapid, iterative optimization cycles. And like a messenger between two worlds, Arke translates between **what to compute** (semantic intent) and **how to compute it** (hardware-specific strategy).

## Key Features

- 🤖 **AI-First Design** — Explicit semantics, structured representation, enumerable search spaces
- 🔗 **Semantic IR → Strategy IR → Hardware Code** — Multi-level IR separating "what" from "how"
- 🛡️ **Compiler-Verified** — Every LLM decision validated by deterministic checks (static → numerical → performance)
- 💬 **`@rationale` Annotations** — Every optimization decision carries a natural language explanation
- ⚡ **Tool-Use Agent Runtime** — LLM autonomously explores, applies, verifies, and profiles optimizations
- 🎯 **106% cuBLAS** — LLM-optimized matmul+relu kernel outperforms NVIDIA's library on RTX 3060

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

```arke
// Declare computation (what to compute)
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

// Declare optimization strategy (how to optimize)
strategy fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");
    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing");
    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

## Python API (Target)

```python
import arke

result = arke.optimize(
    kernel="matmul",
    shape=[1024, 512, 2048],
    target="nvidia_ampere",
    llm="anthropic"
)

print(result.performance)  # "106% of cuBLAS"
print(result.trajectory)   # Full optimization trace
```

---

## 🗺️ Long-Term Roadmap

Three stages, from validation to production:

```
Stage 1: Arke → Triton → GPU          验证假设 (current)
Stage 2: Arke → MLIR Dialect → 多硬件  摆脱依赖
Stage 3: Arke → LLVM IR → 全硬件       重构编译栈
```

### Stage 1: Arke → Triton → GPU (验证假设)

Validate the core thesis: **LLM + structured IR + compiler verification > LLM direct codegen.**

Use Triton as the code generation target. Prove that LLM-guided optimization through tool-use outperforms LLM-written Triton code on correctness, consistency, and performance.

**Target hardware:** NVIDIA Ampere (RTX 3060)

### Stage 2: Arke → MLIR Dialect → 多硬件 (摆脱依赖)

Replace Triton backend with a custom MLIR dialect. Enable multi-hardware targeting without depending on external code generation frameworks.

**Target hardware:** NVIDIA + Huawei Ascend

### Stage 3: Arke → LLVM IR → 全硬件 (重构编译栈)

Full compiler stack from Arke IR directly to LLVM IR. Maximum control over code generation and optimization for all hardware targets.

**Target hardware:** NVIDIA + Ascend + AMD + Intel

---

## 📋 Stage 1 Development Progress

> Phase-based execution — each phase has SMART completion criteria and gate milestones.
> See [plan-v3.0.md](docs/design/plan-v3.0.md) for detailed criteria.

### Phase Overview

| Phase | Goal | Gate | Status |
|:-----:|:-----|:----:|:------:|
| **1** | IR + 验证基础 | G0: Triton matmul runs on GPU | ✅ |
| **2** | Codegen + E2E Pipeline | G2: Manual strategy → ≥70% cuBLAS | ✅ (105-160%) |
| **3** | LLM Runner 联调 | — | ✅ |
| **4** | LLM 闭环优化 | G3: LLM tool-use → matmul ≥50% cuBLAS | ✅ (106%) |
| **5** | 评估框架 + 对比实验 | G4: Arke ≥ direct Triton gen | 🔨 |
| **6** | .ak Parser + CLI | — | ⬜ |
| **7** | 整模型端到端 | G5: GPT-2 w/ Arke ≥ torch.compile | ⬜ |
| **8** | 多 LLM + 报告 | — | ⬜ |

### Key Achievements

- **LLM closed-loop optimization working** — Claude Sonnet 4.6 autonomously optimizes matmul+relu kernel through 23 tool calls with zero errors
- **106% cuBLAS** — LLM-optimized kernel outperforms NVIDIA's hand-tuned library
- **GPU correctness verification** — Same-dtype comparison (Triton kernel output vs NumPy CPU reference at matching precision)
- **Accuracy benchmark framework** — 10 metrics, 3-tier verdict (accept/review/reject), per-dtype thresholds
- **237 tests passing** (219 with GPU tests enabled)

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
│   ├── triton_template_engine.py  # Strategy → Triton param mapping
│   └── base.py                # Backend ABC
├── learn/                     # Learning System
│   └── trajectory.py          # JSONL trajectory export
├── lang/                      # Language Frontend (Phase 6)
│   ├── ast.py                 # AST node definitions
│   └── ...
├── pipeline.py                # E2E pipeline
└── examples/                  # Example scripts
    └── agent_matmul.py        # LLM agent matmul optimization demo
```

## Design Documents

| Document | Description |
|----------|-------------|
| [plan-v3.0.md](docs/design/plan-v3.0.md) | Phase-based execution plan — SMART criteria, Gate milestones |
| [e2e-flow.md](docs/design/e2e-flow.md) | End-to-end flow — from user input to GPU execution |
| [design-review.md](docs/design/design-review.md) | Design review — assumption validation, risk matrix |
| [naming-system.md](docs/design/naming-system.md) | Naming conventions — global terminology rules |

## Specifications

| Spec | Description |
|------|-------------|
| [arke-language-spec.md](docs/spec/arke-language-spec.md) | Arke language spec — syntax, type system, built-in ops |
| [arke-ir-spec.md](docs/spec/arke-ir-spec.md) | Arke IR spec — Semantic IR / Strategy IR structure |

## License

[Apache License 2.0](LICENSE)
