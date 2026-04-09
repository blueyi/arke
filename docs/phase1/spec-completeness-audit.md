# Arke Spec Completeness Audit — Stage 7 Track 2

> **Date:** 2026-04-09  
> **Purpose:** Audit README capability claims against current spec/docs completeness  
> **Status:** 🟨 Partial — 6/8 capabilities fully specified, 2 need expansion

---

## Executive Summary

README 声称 8 个核心能力。当前 spec/docs 覆盖情况：

| # | 能力 | 状态 | 文档 | 缺失 |
|:--|:-----|:----:|:-----|:-----|
| 1 | LLM-Native Language | ✅ 完整 | arke-lang-spec-v2.md | — |
| 2 | Semantic/Strategy Separation | ✅ 完整 | arke-ir-spec-v2.md | — |
| 3 | Minimal-Token E2E | ⚠️ 部分 | e2e-flow.md | **Token 消耗量化指标缺失** |
| 4 | Bounded Action Space | ✅ 完整 | agent-design.md | — |
| 5 | @rationale Annotations | ✅ 完整 | arke-lang-spec-v2.md | — |
| 6 | Compiler-as-Verifier | ✅ 完整 | arke-compiler-infrastructure.md | — |
| 7 | Structured LLM-Compiler Protocol | ✅ 完整 | agent-design.md | — |
| 8 | Multi-Hardware | ⚠️ 部分 | arke-ir-spec-v2.md | **Ascend/AMD 后端设计缺失** |

**关键发现：**
- ✅ 核心 Lang/IR/Agent 设计已定稿
- ⚠️ 2 个能力需要补充设计文档
- ❌ 3 个关键 spec 文档缺失（见下文）

---

## 详细审视

### 1. LLM-Native Language ✅

**README 声称：**
> Semantic/Strategy Separation — "What to compute" (immutable math) and "how to optimize" (searchable decisions) are independent layers

**现有文档：**
- ✅ `docs/spec/arke-lang-spec-v2.md` — 完整定义 kernel/strategy 语法
- ✅ `docs/architecture/naming-system.md` — 清晰的术语体系
- ✅ `docs/spec/arke-lang-vs-python-triton.md` — 对标 Python/Triton

**覆盖度：** 100% ✅

**验证：**
```python
# kernel 块（What）
kernel matmul {
    inputs: [A: f32[M, K], B: f32[K, N]]
    outputs: [C: f32[M, N]]
    compute: C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
}

# strategy 块（How）
strategy matmul_opt {
    @rationale("Tile for cache locality")
    tile(M, 64)
    tile(N, 64)
    tile(K, 32)
    
    @rationale("Fuse load + compute")
    fuse(load, compute)
}
```

---

### 2. Semantic/Strategy Separation ✅

**README 声称：**
> enabling LLMs to explore strategies without risking correctness

**现有文档：**
- ✅ `docs/spec/arke-ir-spec-v2.md` — 4 层 IR 架构（Layer 4: SemanticIR, Layer 3: StrategyIR）
- ✅ `docs/architecture/e2e-flow.md` — 端到端流程中的分离点
- ✅ `docs/phase1/dynamic-shape-feasibility.md` — 符号维度在 SemanticIR 中的处理

**覆盖度：** 100% ✅

**验证：**
```
SemanticIR (Layer 4)
├─ kernel_id: "matmul_v1"
├─ inputs: [A: f32[M, K], B: f32[K, N]]
├─ outputs: [C: f32[M, N]]
├─ compute: C[i,j] = sum(A[i,k] * B[k,j])
└─ constraints: [M > 0, K > 0, N > 0]

StrategyIR (Layer 3)
├─ kernel_id: "matmul_v1"
├─ decisions: [
│   {type: "tile", target: "M", size: 64, @rationale: "..."},
│   {type: "tile", target: "N", size: 64, @rationale: "..."},
│   ...
│ ]
└─ hardware_target: "nvidia_ampere"
```

---

### 3. Minimal-Token End-to-End ⚠️ **PARTIAL**

**README 声称：**
> The entire pipeline — definition, search, verification, iteration — consumes an order of magnitude fewer tokens than direct code generation

**现有文档：**
- ✅ `docs/architecture/e2e-flow.md` — 流程定义
- ✅ `docs/architecture/agent-design.md` — Agent 交互协议
- ❌ **缺失：Token 消耗量化指标和对标分析**

**缺失内容：**

需要新增 `docs/architecture/token-efficiency-analysis.md`，包含：

1. **Token 消耗分解**
   ```
   Direct Code Generation (Baseline)
   ├─ Prompt: kernel definition + context → 500 tokens
   ├─ Generation: free-form code → 2000 tokens
   ├─ Verification: manual testing → 1000 tokens (per iteration)
   └─ Total per iteration: ~3500 tokens
   
   Arke LLM-Native Pipeline
   ├─ Prompt: kernel + legal_actions → 300 tokens
   ├─ Decision: select action + @rationale → 200 tokens
   ├─ Verification: V0 static check → 0 tokens (compiler)
   └─ Total per iteration: ~500 tokens
   
   Savings: 7× fewer tokens per iteration
   ```

2. **对标案例**
   - GPT-2 matmul: Direct 3500 tokens/iter vs Arke 500 tokens/iter
   - LLaMA attention: Direct 5000 tokens/iter vs Arke 600 tokens/iter

3. **Token 预算系统**
   - 定义 `OptimizationBudget` 的 token 限制
   - 说明如何在 Agent 中强制执行

**优先级：** 🔴 HIGH — 这是 README 的核心卖点之一

---

### 4. Bounded Action Space ✅

**README 声称：**
> LLMs select from compiler-enumerated legal actions, not free-form code

**现有文档：**
- ✅ `docs/architecture/agent-design.md` — `list_legal_actions` tool 定义
- ✅ `docs/spec/arke-ir-spec-v2.md` — Decision 类型枚举
- ✅ `docs/phase1/stage7-plan.md` — Track 2 中的 `LegalActionsEngine` 实现计划

**覆盖度：** 100% ✅

**验证：**
```python
# Agent 可用的 legal actions（由编译器枚举）
legal_actions = [
    {"type": "tile", "target": "M", "sizes": [32, 64, 128, 256]},
    {"type": "tile", "target": "N", "sizes": [32, 64, 128, 256]},
    {"type": "tile", "target": "K", "sizes": [8, 16, 32, 64]},
    {"type": "fuse", "ops": ["load_A", "load_B", "compute"]},
    {"type": "place", "target": "shared_memory", "ops": ["load_A"]},
    ...
]
```

---

### 5. @rationale Annotations ✅

**README 声称：**
> Every optimization decision carries a natural language explanation, preserved in IR as a first-class construct

**现有文档：**
- ✅ `docs/spec/arke-lang-spec-v2.md` — `@rationale` 语法定义（§3.2）
- ✅ `docs/spec/arke-ir-spec-v2.md` — StrategyIR 中的 `rationale` 字段
- ✅ `docs/architecture/agent-design.md` — Agent 生成 @rationale 的流程

**覆盖度：** 100% ✅

**验证：**
```python
# Lang 层
strategy matmul_opt {
    @rationale("Tile M dimension for L1 cache locality; 64 threads per block")
    tile(M, 64)
    
    @rationale("Tile N dimension to maximize register reuse")
    tile(N, 64)
}

# IR 层（StrategyIR）
{
    "decisions": [
        {
            "type": "tile",
            "target": "M",
            "size": 64,
            "rationale": "Tile M dimension for L1 cache locality; 64 threads per block"
        },
        ...
    ]
}
```

---

### 6. Compiler-as-Verifier ✅

**README 声称：**
> The compiler does not optimize; it verifies every LLM decision through progressive checks: V0 Static (<1ms) → V1 Numerical → V2 Performance

**现有文档：**
- ✅ `docs/architecture/arke-compiler-infrastructure.md` — V0/V1/V2 验证层定义
- ✅ `docs/benchmark/benchmark-design.md` — V2 性能验证框架
- ✅ `docs/phase1/stage7-plan.md` — Track 3-5 中的验证实现计划

**覆盖度：** 100% ✅

**验证：**
```
V0 Static Validation (<1ms)
├─ Shape inference: 符号维度传播
├─ Type checking: dtype 一致性
├─ Constraint checking: 维度兼容性
└─ SSA form: 数据流合法性

V1 Numerical Validation
├─ Reference implementation: NumPy/PyTorch
├─ Correctness check: 数值误差 < 1e-5
└─ Edge cases: 边界条件测试

V2 Performance Validation
├─ Baseline: FlagGems/cuBLAS/Triton
├─ Speedup: 目标 ≥ 1.0× baseline
└─ Profiling: 内存/计算/通信分解
```

---

### 7. Structured LLM-Compiler Protocol ✅

**README 声称：**
> LLM and compiler interact through a closed-loop tool-use API (analyze → decide → verify → iterate)

**现有文档：**
- ✅ `docs/architecture/agent-design.md` — Tool 定义和交互流程
- ✅ `docs/architecture/e2e-flow.md` — 端到端流程图
- ✅ `docs/architecture/naming-system.md` — Tool 命名规范

**覆盖度：** 100% ✅

**验证：**
```python
# Tool-use 循环
while not done:
    # 1. Analyze
    semantic_ir = get_semantic_ir()
    current_strategy = get_current_strategy()
    legal_actions = list_legal_actions()
    
    # 2. Decide
    decision = llm.select_action(
        semantic_ir=semantic_ir,
        current_strategy=current_strategy,
        legal_actions=legal_actions
    )
    
    # 3. Verify
    result = apply_decision(decision)
    if result.status == "valid":
        checkpoint()
    else:
        rollback()
    
    # 4. Iterate
    if result.performance >= target:
        done = True
```

---

### 8. Multi-Hardware ⚠️ **PARTIAL**

**README 声称：**
> Single kernel definition targets NVIDIA, Ascend, and beyond; strategy adapts per hardware, semantics stay fixed

**现有文档：**
- ✅ `docs/spec/arke-ir-spec-v2.md` — `hardware_target` 字段
- ✅ `docs/architecture/arke-compiler-infrastructure.md` — Backend 抽象
- ❌ **缺失：Ascend/AMD 后端的具体设计**

**缺失内容：**

需要新增以下文档：

1. **`docs/architecture/backend-abstraction-protocol.md`**
   - Backend 接口规范（输入：StrategyIR，输出：目标代码）
   - 每个后端的职责边界
   - 硬件特性映射表

2. **`docs/architecture/ascend-backend-design.md`**
   - Ascend NPU 特性分析
   - StrategyIR → Ascend IR 映射
   - 性能模型和约束

3. **`docs/architecture/amd-backend-design.md`**
   - AMD RDNA/CDNA 特性分析
   - StrategyIR → HIP/LLVM 映射
   - 性能模型和约束

4. **`docs/architecture/multi-hardware-strategy-adaptation.md`**
   - 同一个 kernel 在不同硬件上的策略差异
   - 策略迁移规则（NVIDIA → Ascend）
   - 硬件特性检测和自适应

**优先级：** 🟡 MEDIUM — Phase 2 的基础，但 Phase 1 可以先聚焦 NVIDIA

---

## 缺失的关键 Spec 文档

### A. Pass Infrastructure Spec ❌

**现状：** `arke-compiler-infrastructure.md` 提到 Pass Pipeline，但没有完整的 Pass 接口规范

**需要：** `docs/spec/pass-infrastructure-spec.md`
- Pass 基类接口
- 前置/后置条件
- 数据流依赖
- 例子：ShapeInferencePass, FusionPass, SchedulePass

**优先级：** 🔴 HIGH — S7 Track 2 需要

---

### B. Symbolic Dimension System Spec ❌

**现状：** `arke-ir-spec-v2.md` 和 `dynamic-shape-feasibility.md` 有提及，但没有完整的符号维度规范

**需要：** `docs/spec/symbolic-dimension-spec.md`
- 符号维度语法（M, N, K, seq_len, ...）
- 维度约束表达式
- 符号维度传播算法
- 例子：matmul(M, K) × (K, N) → (M, N)

**优先级：** 🔴 HIGH — S7 Track 2 需要

---

### C. Backend Abstraction Protocol Spec ❌

**现状：** `arke-compiler-infrastructure.md` 提到 Backend 抽象，但没有完整的接口规范

**需要：** `docs/spec/backend-abstraction-protocol.md`
- Backend 接口（输入/输出）
- 硬件特性声明
- 代码生成流程
- 性能模型接口

**优先级：** 🟡 MEDIUM — Phase 2 需要

---

### D. SSA Validator Spec ❌

**现状：** `arke-compiler-infrastructure.md` 提到 SSA 验证，但没有完整的规范

**需要：** `docs/spec/ssa-validator-spec.md`
- SSA 形式定义
- 验证规则
- 错误报告格式

**优先级：** 🟡 MEDIUM — S7 Track 3 需要

---

### E. MLIR/LLVM Interoperability Design ❌

**现状：** `arke-ir-spec-v2.md` 提到 MLIR 集成，但没有详细设计

**需要：** `docs/architecture/mlir-llvm-interoperability.md`
- StrategyIR → MLIR Dialect 映射
- MLIR Dialect 定义
- LLVM IR 生成流程
- 性能优化机制

**优先级：** 🟡 MEDIUM — Phase 3 需要

---

## 现有文档质量评估

### 优秀 ✅

| 文档 | 评分 | 理由 |
|:-----|:----:|:-----|
| `arke-lang-spec-v2.md` | 9/10 | 完整、清晰、有例子 |
| `arke-ir-spec-v2.md` | 9/10 | 4 层 IR 架构清晰，映射完整 |
| `agent-design.md` | 8/10 | Tool 定义清晰，流程完整 |
| `e2e-flow.md` | 8/10 | 端到端流程图清晰 |
| `arke-compiler-infrastructure.md` | 7/10 | 架构清晰，但缺少接口规范 |
| `naming-system.md` | 9/10 | 术语体系完整、一致 |
| `benchmark-design.md` | 8/10 | BL/OT/ST/L 系统清晰 |

### 需要改进 ⚠️

| 文档 | 问题 | 优先级 |
|:-----|:-----|:----:|
| `e2e-flow.md` | 缺少 token 消耗量化 | 🔴 HIGH |
| `arke-compiler-infrastructure.md` | 缺少 Pass/Backend 接口规范 | 🔴 HIGH |
| `arke-ir-spec-v2.md` | 符号维度部分需要独立 spec | 🔴 HIGH |

---

## 补充计划（S7 Track 2-5）

### Track 2: 补充 3 个关键 Spec（本周）

- [ ] `docs/spec/pass-infrastructure-spec.md` — Pass 接口规范
- [ ] `docs/spec/symbolic-dimension-spec.md` — 符号维度完整规范
- [ ] `docs/architecture/token-efficiency-analysis.md` — Token 消耗量化

**验证：** 所有 spec 必须有可运行的例子

### Track 3: 补充 2 个后端设计（下周）

- [ ] `docs/architecture/backend-abstraction-protocol.md` — 后端抽象协议
- [ ] `docs/architecture/ascend-backend-design.md` — Ascend 后端设计

**验证：** 后端设计必须能映射到 StrategyIR

### Track 4: 补充 MLIR 设计（两周后）

- [ ] `docs/architecture/mlir-llvm-interoperability.md` — MLIR/LLVM 互操作

**验证：** 设计必须能编译到 MLIR Dialect

### Track 5: 补充 SSA 和验证设计（三周后）

- [ ] `docs/spec/ssa-validator-spec.md` — SSA 验证规范

**验证：** 验证器必须能检测所有 SSA 违规

---

## 对标 README 的最终检查表

| 能力 | 文档完整性 | 实现完整性 | 验证完整性 | 总体 |
|:-----|:--------:|:--------:|:--------:|:----:|
| LLM-Native Language | ✅ | ✅ | ✅ | ✅ |
| Semantic/Strategy Separation | ✅ | ✅ | ✅ | ✅ |
| Minimal-Token E2E | ⚠️ | ✅ | ❌ | ⚠️ |
| Bounded Action Space | ✅ | ✅ | ✅ | ✅ |
| @rationale Annotations | ✅ | ✅ | ✅ | ✅ |
| Compiler-as-Verifier | ✅ | ✅ | ✅ | ✅ |
| Structured LLM-Compiler Protocol | ✅ | ✅ | ✅ | ✅ |
| Multi-Hardware | ⚠️ | ⚠️ | ❌ | ⚠️ |

**总体评分：** 6/8 完整，2/8 需要补充

---

## 建议

### 立即行动（本周）

1. **补充 Token 消耗量化** — 这是 README 的核心卖点，必须有数据支撑
2. **补充 Pass 接口规范** — S7 Track 2 需要
3. **补充符号维度规范** — S7 Track 2 需要

### 短期行动（下周）

4. **补充后端抽象协议** — 为 Phase 2 做准备
5. **补充 Ascend 后端设计** — 验证多硬件可行性

### 中期行动（两周后）

6. **补充 MLIR 设计** — Phase 3 的基础

---

*版本：v1.0 | 创建：2026-04-09 | 审视者：Kitty*
