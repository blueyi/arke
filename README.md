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
- ⚡ **AsyncGenerator Runtime** — Streaming events for CLI / Python API / Jupyter (inspired by Claude Code)
- 🎯 **Multi-Target** — NVIDIA Ampere (Phase 1), Huawei Ascend A3 via triton-ascend (Phase 2)

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
schedule fused_matmul_relu for target("nvidia_ampere") {
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

print(result.performance)  # "82% of cuBLAS"
print(result.trajectory)   # Full optimization trace
```

---

## 🗺️ Development Roadmap

**MVP Target:** 8 weeks → v0.1.0
**Core Principle:** LLM protocol first, validation second, human syntax last

### Phase Overview

```
Week 1-2: Foundation (IR + Validation + Tool Schema)
Week 3-4: Engine (Codegen + Agent Runtime + LLM Integration)
Week 5-6: Integration (Parser + E2E Pipeline + Evaluation)
Week 7-8: Polish (Multi-LLM + Whole-model Eval + Report)
```

### Gate Milestones

| Gate | Week | What We're Validating | Pass Criteria | Fail = |
|:----:|:----:|:---------------------|:-------------|:-------|
| **G0** | W1 | Environment works | Triton matmul runs on RTX 3060 | Fix env |
| **G1** | W2 | IR is expressive enough | Known-good strategy expressible in Arke IR | IR redesign |
| **G2** | W3 | E2E path works | Manual strategy → codegen → ≥70% cuBLAS | **Foundation broken** |
| **G3** | W4 | LLM can optimize | LLM tool-use → matmul ≥50% cuBLAS + softmax correct | LLM can't reason about GPU |
| **G4** | W6 | Arke adds value | Arke correctness+perf ≥ direct Triton generation | **Kill or pivot** |
| **G5** | W8 | Whole-model benefit | GPT-2 Small w/ Arke kernels ≥ torch.compile | Analyze bottleneck |

---

## 📋 Development Progress

> Legend: ✅ Done | 🔨 In Progress | ⬚ Not Started | ❌ Blocked | 🔬 Experimental

### Week 1 — Foundation: IR + Schema + Environment

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W1-01 | Create venv + install PyTorch/Triton | Env | |
| ⬚ | W1-02 | GPU environment verification script | Env | Depends: W1-01 |
| ⬚ | W1-03 | Op catalog P0 (10 operators) | IR | matmul, relu, softmax, add, mul, etc. |
| ⬚ | W1-04 | Semantic IR JSON Schema + serialization | IR | |
| ⬚ | W1-05 | Strategy IR JSON Schema | IR | |
| ⬚ | W1-06a | Unified HW Profile Schema | IR | |
| ⬚ | W1-06b | HW Profile: nvidia_ampere_rtx3060.json | IR | Depends: W1-01, W1-06a |
| ⬚ | W1-06c | HW Profile: huawei_ascend_a3.json | IR | Depends: W1-06a |
| ⬚ | W1-06d | Backend abstract base + registry | IR | |
| ⬚ | W1-07 | Tool-use schema (all 10 tools) | Agent | Depends: W1-03 |
| ⬚ | W1-08 | Session lifecycle + system prompt template | Agent | Depends: W1-07 |
| ⬚ | W1-09 | IR Builder (Python → SemanticIR) | IR | Depends: W1-03, W1-04 |
| ⬚ | W1-10 | Integration: manual matmul IR → JSON roundtrip | ALL | **Gate G0 check** |
| ⬚ | W1-11 | Glossary + doc alignment | ALL | |
| ⬚ | W1-12 | Collect AscendC matmul samples | IR | Reference only |

### Week 2 — Validation + ArkeEnv Core

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W2-01 | V0 static validator (shape + constraints) | Validation | Depends: W1-04,05,06 |
| ⬚ | W2-02 | V1 numerical validator (NumPy ref + compare) | Validation | Depends: W1-03, W1-01 |
| ⬚ | W2-03 | ArkeEnv core framework | Agent | Depends: W1-07 |
| ⬚ | W2-04 | Legal actions enumeration engine | Agent | Depends: W1-03,04,06 |
| ⬚ | W2-05 | ArkeEnv observe/apply/rollback | Agent | Depends: W2-03,04,01 |
| ⬚ | W2-06 | Strategy IR rename (schedule→strategy) | IR | Depends: W1-05 |
| ⬚ | W2-07 | Unit tests: validator + legal_actions | Test | Depends: W2-01,04 |
| ⬚ | W2-08 | legal_actions codegen_support annotation | Agent | Depends: W2-04 |
| ⬚ | W2-09 | Declarative Tool interface (ToolMeta + base) | Agent | **CC-inspired** |
| ⬚ | W2-10 | Large result delta compression | Agent | **CC-inspired** |

### Week 3 — Codegen + Agent Runtime Infrastructure

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W3-01 | Triton matmul template (Jinja2) | Codegen | Depends: W1-01 |
| ⬚ | W3-02 | Triton matmul+relu fusion template | Codegen | Depends: W3-01 |
| ⬚ | W3-03 | Template engine (strategy→params mapping) | Codegen | Depends: W3-01, W1-05 |
| ⬚ | W3-04 | ArkeCompiler (compile + load + execute) | Codegen | Depends: W3-01 |
| ⬚ | W3-05 | E2E: manual IR → manual strategy → Triton → GPU | ALL | **Gate G2 check** |
| ⬚ | W3-06 | V2 performance profiler (vs cuBLAS) | Validation | Depends: W3-04, W1-01 |
| ⬚ | W3-07 | ArkeEnv ↔ codegen + verify + profile | ALL | Depends: W3-05,06, W2-05 |
| ⬚ | W3-08 | Triton softmax template | Codegen | |
| ⬚ | W3-09 | Tool concurrency partitioning (orchestrator.py) | Agent | **CC-inspired**. Depends: W2-09 |
| ⬚ | W3-10 | Segmented prompt cache (4-segment build) | Agent | **CC-inspired**. Depends: W1-08 |
| ⬚ | W3-11 | OptimizationState ground truth manager | Agent | **CC-inspired**. Depends: W2-05 |

### Week 4 — LLM Integration + Agent Runtime

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W4-01 | LLM Agent Runner (Claude API) | Agent | Depends: W2-05, W3-07 |
| ⬚ | W4-02 | matmul agent demo: full tool-use loop | Agent | **Gate G3 check** |
| ⬚ | W4-03 | Error recovery module | Agent | Depends: W4-01 |
| ⬚ | W4-03b | Fallback strategy mechanism | Agent | Depends: W4-02 |
| ⬚ | W4-04 | Benchmark task definitions T1-T5 | Eval | |
| ⬚ | W4-05 | cuBLAS/PyTorch baseline implementation | Eval | Depends: W1-01 |
| ⬚ | W4-06 | softmax agent demo | Agent | **Gate G3 check** |
| ⬚ | W4-07 | Multi-LLM backend (Qwen/GPT) | Agent | Depends: W4-01 |
| ⬚ | W4-08 | Context compact (predictive + reactive) | Agent | **CC-inspired**. Depends: W4-01, W3-11 |
| ⬚ | W4-09 | Three-tier fault tolerance (resilience.py) | Agent | **CC-inspired**. Depends: W4-03, W4-08 |
| ⬚ | W4-10 | AsyncGenerator optimization loop (runner.py) | Agent | **CC-inspired**. Depends: W3-09, W4-08 |

### Week 5 — LLM Codegen Experiment + Parser

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W5-01 | 🔬 Path B: LLM Triton codegen | Codegen | Experimental |
| ⬚ | W5-02 | 🔬 Path A vs B comparison | Eval | |
| ⬚ | W5-03 | .ak EBNF grammar (arke.lark) | Lang | |
| ⬚ | W5-04 | Lark parser implementation | Lang | Depends: W5-03 |
| ⬚ | W5-05 | AST → SemanticIR conversion | Lang | Depends: W5-04, W1-04 |
| ⬚ | W5-06 | Parse examples/*.ak → IR → codegen | Lang | |
| ⬚ | W5-07 | AscendC matmul template (skeleton) | Codegen | Phase 2 prep |
| ⬚ | W5-08 | AscendC backend skeleton (stub) | Codegen | Phase 2 prep |

### Week 6 — End-to-End Pipeline + Evaluation

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W6-01 | E2E pipeline assembly | ALL | |
| ⬚ | W6-02 | Baseline B: LLM direct Triton | Eval | **Gate G4 check** |
| ⬚ | W6-03 | Baseline C: LLM direct CUDA | Eval | |
| ⬚ | W6-04 | Run T1-T3 comparison experiments | Eval | **Gate G4 check** |
| ⬚ | W6-04b | Baseline D: Zero-LLM brute-force search | Eval | |
| ⬚ | W6-05 | fused_matmul_relu full E2E | Codegen | |
| ⬚ | W6-06 | Experiment analysis + preliminary conclusions | Eval | |

### Week 7 — Polish + Multi-LLM + Whole-Model Prep

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W7-01 | Trajectory recording system | Agent | |
| ⬚ | W7-02 | Run T4-T5 + multi-LLM comparison | Eval | |
| ⬚ | W7-03 | CLI polish (parse/inspect/optimize/codegen) | Lang | |
| ⬚ | W7-04 | IR visualization (inspect --visual) | Lang | |
| ⬚ | W7-05 | Integration testing + bug fixes | ALL | |
| ⬚ | W7-06 | GPT-2 Small baseline (eager + torch.compile) | Eval | |
| ⬚ | W7-07 | PyTorch custom op integration (torch.library) | Integration | |

### Week 8 — Whole-Model Validation + MVP Release

| Status | ID | Task | Stream | Notes |
|:------:|:---|:-----|:------:|:------|
| ⬚ | W8-01 | GPT-2 Small E2E (Arke kernel replacement) | Eval | **Gate G5 check** |
| ⬚ | W8-02 | Complete evaluation report | Eval | |
| ⬚ | W8-03 | Documentation (agent-protocol/, ir-spec/) | ALL | |
| ⬚ | W8-04 | README update + Quick Start guide | ALL | |
| ⬚ | W8-05 | Code cleanup + ruff/mypy pass | ALL | |
| ⬚ | W8-06 | Test coverage ≥ 60% | Test | |
| ⬚ | W8-07 | **MVP v0.1.0 Tag** 🎉 | ALL | |

---

## 📊 MVP v0.1.0 Success Criteria

| # | Criterion | Acceptance |
|---|-----------|------------|
| 1 | **LLM E2E works** | LLM tool-use optimizes matmul → GPU runs → correct result |
| 2 | **3-layer validation works** | Static + numerical + performance checks all functional |
| 3 | **Performance target** | LLM-optimized matmul ≥ 70% cuBLAS |
| 4 | **Comparative data** | Arke vs direct Triton vs brute-force quantitative comparison |
| 5 | **Multi-operator** | matmul + softmax + fused_matmul_relu all working |
| 6 | **Parser works** | .ak → IR → codegen → execute |
| 7 | **Trajectories exportable** | Full (state, action, reward) trace for each optimization |
| 8 | **Fallback works** | Auto-degrade when LLM search underperforms baseline |
| 9 | **Whole-model E2E** | GPT-2 Small inference with Arke kernels ≥ torch.compile |

---

## 🧰 Project Structure

```
arke/
├── ir/                        # IR Layer
│   ├── semantic.py            # Semantic IR (what to compute)
│   ├── strategy.py            # Strategy IR (how to optimize)
│   ├── builder.py             # Python → IR builder
│   ├── ops/                   # Operator catalog
│   ├── schemas/               # JSON Schemas
│   └── targets/               # HW profiles (nvidia_ampere, ascend_a3)
├── engine/                    # Core Engine
│   ├── env.py                 # ArkeEnv (LLM ↔ compiler interface)
│   ├── legal_actions.py       # Legal action enumeration
│   ├── state.py               # State management + checkpoints
│   ├── validator.py           # V0 static validation
│   ├── numerical_check.py     # V1 numerical validation
│   └── profiler.py            # V2 performance profiling
├── agent/                     # LLM Agent Runtime
│   ├── runner.py              # AsyncGenerator optimization loop
│   ├── tools/                 # Tool implementations + orchestrator
│   │   ├── base.py            # ArkeTool ABC + ToolMeta
│   │   ├── orchestrator.py    # Concurrency partitioning
│   │   └── result_management.py  # Delta compression
│   ├── prompts.py             # 4-segment prompt cache builder
│   ├── compact.py             # Context compact (predictive + reactive)
│   ├── resilience.py          # Three-tier fault tolerance
│   ├── state.py               # Ground truth state (survives compact)
│   ├── session.py             # Session lifecycle
│   ├── recovery.py            # Error recovery
│   └── providers/             # LLM provider adapters
├── backend/                   # Code Generation
│   ├── triton_templates/      # Jinja2 templates
│   ├── triton_template_engine.py
│   ├── triton_llm_gen.py      # 🔬 LLM codegen (experimental)
│   └── compiler.py            # Compile + load + execute
├── lang/                      # Language Frontend
│   ├── arke.lark              # EBNF grammar
│   ├── parser.py              # Lark parser
│   └── ast_to_ir.py           # AST → Semantic IR
├── learn/                     # Learning System
│   └── trajectory.py          # Optimization trace recording
├── integration/               # External Integration
│   └── torch_ops.py           # PyTorch custom op registration
├── pipeline.py                # E2E pipeline
└── frontend/                  # External imports (Phase 2+)
```

## Design Documents

| Document | Description |
|----------|-------------|
| [plan-v2.1.md](docs/design/plan-v2.1.md) | 执行计划 — 8 周 MVP 路线图，4 条并行 Stream，Gate 里程碑 |
| [detailed-design-v2.1.md](docs/design/detailed-design-v2.1.md) | 详细设计 — Tool Schema、IR 规范、验证系统、Codegen、Agent 运行时 |
| [e2e-flow.md](docs/design/e2e-flow.md) | 端到端流程 — 从用户输入到 GPU 执行的完整路径 walkthrough |
| [design-review.md](docs/design/design-review.md) | 设计审视 — 假设验证框架、风险矩阵、Gate 决策标准 |
| [naming-system.md](docs/design/naming-system.md) | 命名体系 — 全局术语规范、CLI/IR/Tool/目录命名规则 |

<details>
<summary>Archived documents (docs/design/deprecated/)</summary>

| Document | Status | Superseded by |
|----------|--------|---------------|
| cc-inspired-update.md | 已合并 | detailed-design-v2.1.md §9 (Agent 运行时架构) |
| multi-backend-design.md | 已合并 | plan-v2.1.md §12 + detailed-design-v2.1.md |
| patch-v2.1.2.md | 已合并 | plan-v2.1.md (LLM API) + detailed-design-v2.1.md |
| op-taxonomy.md | 参考存档 | 算子分类研究，供实现时参考 |
| license-comparison.md | 决策完成 | 已选定 Apache 2.0 |
| e2e-design-v1.md / v2.md | 早期版本 | e2e-flow.md |
| overview.md / plan.md | 早期版本 | plan-v2.1.md |

</details>

## License

[Apache License 2.0](LICENSE)
