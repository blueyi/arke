# Phase 2 & 3 目标合理性审视

> **基于 Phase 1 实测数据重新评估**
> 日期：2026-04-04

---

## 一、Phase 1 给我们的真实基线

在审视 Phase 2/3 之前，先明确 Phase 1 的真实状态：

### 什么被验证了

| 假设 | 验证结果 |
|------|---------|
| H1: 结构化协议让 LLM kernel 更正确 | ✅ 强验证 — 100% vs 83% 正确率 |
| H2: 结构化搜索让 LLM 优化更好 | ✅ 部分验证 — Agent 151% cuBLAS；Arke geomean 超 FlagGems 12% |
| H3: @rationale 让决策可追溯 | ✅ 验证 — JSONL trajectory 完整 |
| H4: 同一协议跨硬件 | ⏳ 仅在 RTX 3060 验证，无跨硬件数据 |

### 关键性能基线（matmul FP16）

- **M≥512 大 shape**：Arke 已超过 cuBLAS 和 FlagGems
- **M=128 小 shape**：Arke 比 cuBLAS 慢 1.5-3×（Triton 固有 dispatch overhead）
- **E2E GPT-2**：1.7-2.3× eager（monkey-patch 架构限制）
- **单算子 geomean**：Arke/cuBLAS=1.09×；Arke/FlagGems=0.88×（13 shapes）

### 架构层面的核心发现

Phase 1 证明了一个重要结论：**Triton 作为后端，在小 M 场景有 ~60µs 不可消除的 dispatch overhead**。这个 overhead 不是 Arke 的问题，而是 Triton Python 层的固有代价。

---

## 二、Phase 2 原始目标回顾

原始 plan-v2.1.md 中 Phase 2 (Phase 2) 的目标：

> **Arke IR → MLIR Dialect → 多硬件**
>
> - 定义 Arke MLIR Dialect (arke.kernel, arke.strategy, arke.rationale)
> - Strategy IR decisions → MLIR transform dialect ops
> - NVIDIA + AMD + Ascend 三硬件验证
> - Gate G6: MLIR 路径性能 ≥ Triton 路径
> - Gate G7: 多硬件可迁移

### 合理性评估

#### ✅ 合理的部分

1. **解决 E2E overhead 的方向正确**
   - Inductor 集成（custom_ops.py 已 prototype）是最近路径
   - Phase 1 实验证明：custom_ops + torch.compile 已从 1.75× 压到 1.49×
   - Inductor graph fusion 理论上可再降 30%，达到 ≤1.15× 目标

2. **MLIR 方向长期正确**
   - 摆脱 Triton 的 dispatch overhead 需要更底层的控制
   - MLIR 是工业级多硬件编译的事实标准
   - Triton 本身也在向 MLIR 迁移

#### ⚠️ 需要调整的部分

1. **Phase 2 的第一个目标不应该是 MLIR，而应该是 torch.compile 后端**

   原始 Phase 2 直接跳到 MLIR，但 Phase 1 发现了一个更紧迫的问题：**E2E 性能的 1.7× overhead 需要先解决**。而解决路径不需要 MLIR——直接用 `torch.library` + Inductor 后端就能做到，而且 `custom_ops.py` 已经做好了一半。

   建议拆分：
   - **Phase 2 Phase 1 (近期，2-4 周)**: torch.compile Inductor backend — 解决 E2E overhead
   - **Phase 2 Phase 2 (中期，6-12 周)**: MLIR Dialect — 获得完整编译控制 + 多硬件

2. **G6 的目标 "MLIR 路径性能 ≥ Triton 路径" 定义模糊**

   Phase 1 证明 Arke/Triton 已经超过 FlagGems（专家 Triton）。所以 G6 不应该是"MLIR ≥ Triton"，而应该是"MLIR 能达到或超过 Phase 1 Triton 路径的最佳水平"（即 cuBLAS 水平），同时在 E2E 场景消除 dispatch overhead。

3. **G7 多硬件目标需要降期望**

   Phase 1 未验证 H4。在没有第二块硬件的情况下，G7（同时支持 Ascend）的优先级应该低于先把 NVIDIA 路径做扎实。建议 G7 先只做 AMD（ROCm 路径，开源 MLIR 友好），Ascend 放到 Phase 3。

4. **Strategy IR Level 分层扩展需要提前**

   plan-v2.1.md 中提到 MLIR 路径需要 Level 2 策略（具体 loop nest 结构、memory access pattern）。Phase 2 Phase 2 开始前需要先扩展 Strategy IR，否则 MLIR lowering 无法表达足够的信息。

---

## 三、Phase 3 原始目标回顾

原始 plan-v2.1.md 中 Phase 3 (Phase 3) 的目标：

> **Arke IR → LLVM IR 直接路径**
>
> - 自研 Lowering Engine（Semantic IR → Loop Nest IR → 调度应用 → Hardware Mapping）
> - LLM-guided lowering（LLM 参与每一步 lowering 决策）
> - Gate G8: LLVM 直接路径性能 ≥ MLIR 路径
> - Gate G9: LLM Level 1-3 全层决策 > 传统 pass pipeline

### 合理性评估

#### ✅ 合理的部分

1. **"LLM 参与 lowering 决策" 是核心差异化**
   - 如果 LLM 能做 Level 1-2-3 全部决策，Arke 就成了真正的 AI-Native 编译器
   - Phase 1 已证明 LLM 在 Level 1 决策（tiling/fusion/算法选择）上有价值
   - Level 2-3 决策（循环顺序、内存访问模式、寄存器分配）的 LLM 效果尚未验证

2. **长期来看 LLVM 直接路径是正确方向**
   - 摆脱 MLIR 的限制，获得最大灵活性
   - 支持所有 LLVM 后端（CPU、GPU、NPU、Edge）

#### ⚠️ 需要调整的部分

1. **G9 "LLM Level 1-3 全层决策 > 传统 pass pipeline" 验证条件不清晰**

   什么是 "传统 pass pipeline"？LLVM 的 O3？Triton 的 autotune？需要定义明确的 baseline。建议：
   - G9 改为：LLM 指导的 lowering 在 ≥3 个 shape 上超过同一 Arke 算法的 "默认参数" 版本

2. **Phase 3 的时间线需要重新评估**

   自研 Lowering Engine（Semantic IR → LLVM IR）是编译器领域的重型工作，不亚于重写一个 Triton。需要：
   - Loop Nest IR 设计
   - Polyhedral analysis（或类似 affine scheduling）
   - Hardware Mapping（线程映射、内存层级）
   - Code emission（LLVM IR builder）

   这不是几个月的工作，更现实的时间线是 6-12 个月，且风险很高。

3. **Phase 3 有一个更务实的替代路径**

   不需要完全自研 lowering engine——可以：
   - 在 MLIR 路径上扩展 LLM 决策到 Level 2（循环策略）
   - 用 MLIR transform dialect 表达 LLM 的 lowering 决策
   - 这样 LLM 控制 Level 1-2，MLIR/LLVM 控制 Level 3

   这比 "完全自研 lowering" 风险低 10 倍，但能验证同样的核心假设。

---

## 四、综合建议：修订后的 Phase 2/3 目标

### Phase 2（修订版）

**总目标：** 在保持 Triton 路径的基础上，(1) 解决 E2E overhead，(2) 建立 MLIR 路径获得完整编译控制。

#### Phase 2a — torch.compile Inductor 后端（4-6 周）

| Gate | 目标 | 验证条件 |
|------|------|---------|
| P2-S1 | 注册 Arke 为 Inductor codegen backend | `torch.compile(model)` 能使用 Arke 算子（custom_ops.py 已 prototype） |
| P2-S2 | 启用 Inductor 跨 Arke ops fusion | GPT-2 forward 的 matmul+bias+activation 被融合为单 kernel |
| P2-S3 | E2E GPT-2 全编译性能 | seq≥256 时 ≤1.15× eager（Phase 1 known-fail 解决） |

**意义：** 直接解决 Phase 1 的 G5 known-fail，让 Arke 在 E2E 场景有实际价值。

#### Phase 2b — MLIR Dialect（8-12 周）

| Gate | 目标 | 验证条件 |
|------|------|---------|
| P2-S4 | 定义 Arke MLIR Dialect | arke.kernel + arke.strategy 能表达 Phase 1 的所有优化决策 |
| P2-S5 | MLIR 路径正确性 | 所有 G1/G2 的 Tier 3 shapes 通过 MLIR 路径验证 |
| P2-S6 | MLIR 路径性能 ≥ Triton 路径 | Geomean ≥ Phase 1 Triton 路径（即 1.09× cuBLAS 水平） |
| P2-S7 | AMD ROCm 路径可行 | 同一 .ak → MLIR → ROCm，matmul 正确且性能 ≥ 50% rocBLAS |

**意义：** 摆脱 Triton dispatch overhead 的根本解，同时迈出多硬件第一步。

**调整说明：**
- 原 G7 "Ascend 支持" → 降优先级，先做 AMD
- 原 G6 定义改为相对于 Phase 1 Triton 路径的水平
- Strategy IR 在 Phase 2b 开始前需要扩展 Level 2 字段

### Phase 3（修订版）

**总目标：** LLM 参与 lowering 决策（Level 2），验证 AI-Native 编译在中层决策的价值，并在 MLIR 基础上扩展多硬件覆盖。

#### 核心调整：Phase 3 不自研 Lowering Engine

| 原始目标 | 修订目标 | 原因 |
|---------|---------|------|
| 自研 Semantic IR → LLVM IR Lowering Engine | LLM 指导 MLIR transform dialect lowering | 风险降低 10×，验证同等假设 |
| LLM Level 1-2-3 全层决策 | LLM Level 1-2 决策，MLIR 做 Level 3 | Level 3（寄存器/barrier）LLM 价值存疑 |

| Gate | 目标 | 验证条件 |
|------|------|---------|
| P3-S1 | Strategy IR Level 2 决策 | LLM 能表达并优化 loop nest 结构、memory access pattern |
| P3-S2 | LLM Level 2 vs 默认参数 | LLM Level 2 决策在 ≥3 shapes 上超过默认参数 20%+ |
| P3-S3 | Ascend NPU 路径可行 | 同一 .ak → MLIR → AscendNPU，H4 假设验证 |
| P3-S4 | AI-Native lowering 优势 | LLM Level 1-2 全部决策 vs 固定 pass pipeline，geomean ≥ 10% 提升 |

**G9 重新定义：**
- 不是 "LLM 全层决策 > 传统 pass pipeline"（模糊）
- 而是 "LLM Level 1-2 联合优化 > LLM Level 1 + 默认 Level 2"（具体可测量）

---

## 五、总结对比表

| 维度 | 原始 Phase 2/3 | 修订后 Phase 2/3 |
|------|--------------|----------------|
| Phase 2 首要任务 | MLIR Dialect | torch.compile Inductor 后端（先解决 E2E） |
| G6 定义 | MLIR ≥ Triton（模糊） | MLIR ≥ Phase 1 Triton 最佳水平（具体） |
| G7 多硬件 | Ascend（难） | AMD ROCm（相对易，MLIR 友好） |
| Phase 3 lowering | 完全自研 | LLM 指导 MLIR transform（风险降低 10×） |
| G9 定义 | LLM Level 1-3 > 传统 pipeline（模糊） | LLM L1-2 > LLM L1 + 默认 L2（可测量） |
| Ascend 支持 | Phase 2 | Phase 3（延迟，风险可控） |

**核心逻辑：Phase 1 证明了 Arke 的 kernel 质量（H1/H2/H3），Phase 2 的第一优先级是把这个质量传递到 E2E 场景（解决 G5 known-fail），而不是立刻跳到 MLIR。**
