# Arke Project — Phase-Based Execution Plan v3.0

> 核心原则：每个 Phase 有 SMART 目标，必须达成后才能进入下一阶段
> 基于 v2.1.4 实际执行经验重构
> Date: 2026-04-01

---

## 设计理念

### 为什么从 Week 改为 Phase

Week 时间表把"时间"当驱动力，但实际执行中：
- 有些 Week 一天就完成了（如 Week 1 环境搭建）
- 有些 Week 花了 3 天也做不完（如 LLM 联调）
- 跨 Week 的依赖关系导致跳跃式推进

**Phase 以"目标达成"为驱动**：每个 Phase 定义了清晰的完成标准，达标才进入下一个。

### SMART 标准

每个 Phase 的目标必须：
- **S**pecific — 具体到可以写成 assert 语句
- **M**easurable — 有量化指标（测试数、性能数、正确率）
- **A**chievable — 基于当前代码和资源可实现
- **R**elevant — 直接服务于 MVP 验证假设
- **T**ime-bounded — 有预估时间（但不硬约束）

---

## 当前状态快照 (2026-04-01)

### 已完成
- ✅ GPU 环境 (PyTorch 2.6.0+cu124, Triton 3.2.0, RTX 3060)
- ✅ IR 体系 (Semantic IR + Strategy IR, JSON Schema, 10 ops)
- ✅ Builder + Shape Inference (全 10 op)
- ✅ 验证系统 (V0 静态 + V1 数值 + 资源估算)
- ✅ 合法动作枚举引擎
- ✅ ArkeEnv 完整实现
- ✅ Triton codegen (matmul + softmax, 模板引擎)
- ✅ E2E pipeline (IR → strategy → codegen → GPU)
- ✅ LLM Runner (Anthropic + OpenAI API, fallback, retry)
- ✅ LLM 首次联调成功 (Sonnet 4.6, 27 tool calls, 完整循环)
- ✅ 212 tests 通过

### 已验证的 Gate
- G0 ✅ — Triton matmul 在 RTX 3060 跑通
- G2 ✅ — 手动 strategy → codegen → 105-160% cuBLAS

### 关键文件
```
arke/ir/semantic.py          # Semantic IR
arke/ir/strategy.py          # Strategy IR  
arke/ir/builder.py           # IR Builder
arke/engine/env.py           # ArkeEnv
arke/engine/validator.py     # V0 Validator
arke/engine/numerical_check.py  # V1 Numerical
arke/engine/legal_actions.py # Legal Actions
arke/agent/tools_schema.py   # 10 Tools
arke/agent/session.py        # Session Manager
arke/agent/prompts.py        # System Prompt
arke/agent/runner.py         # LLM Runner
arke/agent/llm_config.py     # LLM Config
arke/backend/triton_backend.py    # Triton Backend
arke/backend/triton_template_engine.py
arke/pipeline.py             # E2E Pipeline
```

---

## Phase 定义

```
Phase 1 ✅  IR + 验证基础
Phase 2 ✅  Codegen + E2E Pipeline
Phase 3 ✅  LLM Runner 联调
Phase 4 ⬅  LLM 闭环优化 (当前)
Phase 5     评估框架 + 对比实验
Phase 6     .ak Parser + CLI
Phase 7     整模型端到端
Phase 8     MVP Release
```

---

## Phase 1: IR + 验证基础 ✅ 已完成

**目标**: 建立 Semantic IR / Strategy IR 体系，实现静态验证和数值验证

### 完成标准 (全部达成 ✅)
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.1 | Semantic IR 支持 ≥10 个算子 | `len(OP_CATALOG) >= 10` | ✅ |
| 1.2 | Strategy IR 支持 ≥6 种 decision | `kinds ⊇ {tile,fuse,place,parallel,reorder,algorithm}` | ✅ |
| 1.3 | JSON Schema 完整且可验证 | `jsonschema.validate(ir, schema)` 通过 | ✅ |
| 1.4 | V0 静态验证 <1ms | `validator.validate()` 耗时 <1ms | ✅ |
| 1.5 | V1 数值验证 (NumPy reference) | `numerical_check.validate()` 3 trials 通过 | ✅ |
| 1.6 | Shape inference 全 10 op | `infer_shapes()` 对所有 op 返回正确 shape | ✅ |
| 1.7 | ≥100 unit tests | `pytest` 通过数 ≥100 | ✅ (212) |

**Gate G1 ✅**: IR 能表达已知好的 strategy (matmul tiling + fusion)

---

## Phase 2: Codegen + E2E Pipeline ✅ 已完成

**目标**: Triton 代码生成 + 端到端 pipeline 从 IR 到 GPU 执行

### 完成标准 (全部达成 ✅)
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 2.1 | matmul Triton codegen 生成正确代码 | 生成的 kernel 通过 V1 数值验证 | ✅ |
| 2.2 | softmax Triton codegen 生成正确代码 | 同上 | ✅ |
| 2.3 | fused matmul+relu codegen | 融合 kernel 通过数值验证 | ✅ |
| 2.4 | GPU 端到端 ≥70% cuBLAS | `compile_and_profile()` 结果 ≥0.7 | ✅ (105-160%) |
| 2.5 | Pipeline 完整串联 | IR → strategy → codegen → compile → profile 一条龙 | ✅ |
| 2.6 | ≥9 GPU tests | `pytest -k gpu` 通过数 ≥9 | ✅ |

**Gate G2 ✅**: 手动 strategy → codegen → ≥70% cuBLAS

---

## Phase 3: LLM Runner 联调 ✅ 已完成

**目标**: LLM 通过 tool-use 自主完成优化循环

### 完成标准 (全部达成 ✅)
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 3.1 | LLM 能调用 ≥8 种 tools | 联调 trajectory 中出现 ≥8 个不同 tool | ✅ (全10种) |
| 3.2 | LLM 完成 ≥4 个 decisions | `result.decisions >= 4` | ✅ (13) |
| 3.3 | LLM 调用 verify_correctness | trajectory 中包含 verify_correctness | ✅ |
| 3.4 | LLM 调用 compile_and_profile | trajectory 中包含 compile_and_profile | ✅ (5次) |
| 3.5 | LLM 使用 checkpoint + rollback | trajectory 中包含 checkpoint 和 rollback | ✅ |
| 3.6 | Fallback 机制工作 | timeout/error 时自动尝试 fallback model | ✅ |
| 3.7 | 多 provider 支持 | Anthropic + OpenAI API 都可用 | ✅ |
| 3.8 | 零人工干预完成完整循环 | 从 start 到 finish 无需手动介入 | ✅ |

---

## Phase 4: LLM 闭环优化 ⬅ 当前阶段

**目标**: LLM 优化后的 kernel 在 GPU 上达到 ≥50% cuBLAS 性能，多算子验证

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 4.1 | LLM 优化 matmul → GPU 执行 → 正确 | `verify_correctness` 通过 + GPU 输出与 NumPy 一致 (atol=1e-2) | ⬜ |
| 4.2 | LLM 优化 matmul ≥50% cuBLAS | `compile_and_profile()` 返回 `vs_baseline >= 0.5` | ⬜ |
| 4.3 | LLM 优化 softmax → 正确 | 同 4.1 | ⬜ |
| 4.4 | LLM 优化 fused_matmul_relu → 正确 | 同 4.1 | ⬜ |
| 4.5 | compile_and_profile 返回真实 GPU 性能数据 | 返回 `latency_us`, `tflops`, `vs_baseline` 字段 | ⬜ |
| 4.6 | 错误恢复: LLM 遇到 validation failure 后自动调整 | trajectory 中有 failed decision → rollback → success | ✅ |
| 4.7 | 轨迹导出为 JSONL | `export_trajectory()` 输出包含 state/action/result | ⬜ |
| 4.8 | ≥220 tests 全部通过 | `pytest` 通过数 ≥220 | ⬜ |

### 关键差距分析

当前 Phase 3 联调中 `compile_and_profile()` 返回的是 fallback 错误信息，
不是真实 GPU 性能。需要修通 LLM decisions → Triton codegen → GPU execute 的完整链路。

核心问题: LLM 做的 strategy decisions 需要正确映射到 Triton 模板参数。

### 预估时间: 2-3 天

**Gate G3**: LLM tool-use 50 步 → matmul ≥50% cuBLAS + softmax 正确

---

## Phase 5: 评估框架 + 对比实验

**目标**: 定量证明 Arke (LLM + tool-use) 优于 LLM 直写 Triton

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 5.1 | 评估任务 ≥5 个定义完成 | `benchmarks/tasks.py` 中 ≥5 个 task | ⬜ |
| 5.2 | Baseline A: Arke 跑完全部 tasks | 每个 task 有 Arke 结果 | ⬜ |
| 5.3 | Baseline B: LLM 直写 Triton 跑完 | 每个 task 有直写结果 | ⬜ |
| 5.4 | 正确率: Arke ≥ 直写 Triton | `arke_correct_rate >= direct_correct_rate` | ⬜ |
| 5.5 | 性能: Arke 平均 ≥ 直写 Triton | `mean(arke_perf) >= mean(direct_perf)` | ⬜ |
| 5.6 | 一致性: Arke 方差 ≤ 直写 | `var(arke_results) <= var(direct_results)` | ⬜ |
| 5.7 | 评估报告生成 | `benchmarks/report.md` 包含完整数据和分析 | ⬜ |
| 5.8 | Token 效率对比 | Arke vs 直写的总 token 消耗对比 | ⬜ |

### 预估时间: 3-4 天

**Gate G4**: Arke 正确率和性能 ≥ 直写 Triton

### Gate 4 决策矩阵
| 结果 | 结论 | 下一步 |
|------|------|--------|
| Arke 正确率高 + 性能好 | ✅ 继续 | Phase 6-8 |
| Arke 正确率高 + 性能差 | ⚠️ Arke 是验证框架 | Pivot 定位 |
| Arke ≈ 直写 Triton | ⚠️ 无明显优势 | 审视增量价值 |
| 两者都差 | ❌ | Kill 或根本 pivot |

---

## Phase 6: .ak Parser + CLI

**目标**: 人类可以通过 .ak 语法和 CLI 使用 Arke

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 6.1 | .ak parser 解析 matmul kernel | `parser.parse("examples/01_matmul.ak")` 返回 AST | ⬜ |
| 6.2 | .ak parser 解析 fused kernel | `parser.parse("examples/02_matmul_relu_fused.ak")` 返回 AST | ⬜ |
| 6.3 | AST → Semantic IR 转换正确 | `ast_to_ir(ast) == builder.build()` | ⬜ |
| 6.4 | CLI `arke parse` 可用 | `arke parse kernel.ak -o kernel.json` 输出正确 JSON | ⬜ |
| 6.5 | CLI `arke optimize` 可用 | `arke optimize kernel.json --target ampere` 启动 LLM session | ⬜ |
| 6.6 | CLI `arke inspect` 可用 | `arke inspect kernel.json` 输出人类可读 IR | ⬜ |
| 6.7 | ≥3 个 .ak examples 可跑通 | matmul, softmax, fused_matmul_relu 全链路 | ⬜ |

### 依赖: Phase 4 达标
### 预估时间: 3-4 天

---

## Phase 7: 整模型端到端

**目标**: 在真实模型推理中替换 kernel，验证端到端收益

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 7.1 | GPT-2 Small 推理正确 | Arke kernel 替换后输出与 PyTorch 一致 | ⬜ |
| 7.2 | 推理性能 ≥ torch.compile | Arke 替换后延迟 ≤ torch.compile 延迟 | ⬜ |
| 7.3 | 至少替换 2 个算子 | matmul + softmax 或 matmul + layernorm | ⬜ |
| 7.4 | 显存使用不超 6GB | RTX 3060 Laptop 6GB 限制内 | ⬜ |

### 依赖: Phase 5 Gate G4 通过
### 预估时间: 3-5 天

**Gate G5**: GPT-2 Small 推理性能 Arke ≥ torch.compile

---

## Phase 8: MVP Release

**目标**: 发布 MVP v0.1.0，完整文档和可复现结果

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 8.1 | README 完整 (安装 + 快速开始 + 示例) | 新人按 README 能跑通 | ⬜ |
| 8.2 | API 文档完整 | 所有公开 class/function 有 docstring | ⬜ |
| 8.3 | 评估报告完整 | benchmarks/report.md 有图表和结论 | ⬜ |
| 8.4 | CI 通过 | GitHub Actions 全绿 | ⬜ |
| 8.5 | v0.1.0 tag | `git tag v0.1.0` | ⬜ |
| 8.6 | 轨迹数据公开 | 优化轨迹 JSONL 文件可下载 | ⬜ |

### 依赖: Phase 7 达标
### 预估时间: 2-3 天

---

## Phase 进入/退出检查清单

### 进入下一 Phase 前必须确认：

```
□ 当前 Phase 所有完成标准达成 (100%)
□ 对应 Gate 通过
□ 所有 tests 仍然通过 (无回归)
□ 代码已 commit + push
□ daily notes 更新
```

### 异常处理

- **某项标准无法达成**: 分析原因，与 Leon 讨论是否降低标准或跳过
- **发现新的必要工作**: 加入当前 Phase 的标准列表（不拖到下一 Phase）
- **Gate 失败**: 按决策矩阵处理，可能 pivot 或 kill

---

## 假设验证 Gate 总览

| Gate | Phase | 验证假设 | 通过标准 |
|:----:|:-----:|---------|---------|
| G0 ✅ | P1 | 环境可行 | Triton 在 RTX 3060 跑通 |
| G1 ✅ | P1 | IR 表达力 | 已知好的 strategy 可表达 |
| G2 ✅ | P2 | 端到端通路 | 手动 strategy → ≥70% cuBLAS |
| G3 | P4 | LLM 可行性 | LLM → matmul ≥50% cuBLAS + softmax 正确 |
| G4 | P5 | 对比优势 | Arke ≥ LLM 直写 Triton |
| G5 | P7 | 整模型收益 | GPT-2 Small ≥ torch.compile |

---

## 风险与应对

| 风险 | 影响 Phase | 应对 |
|------|:----------:|------|
| LLM decisions 无法映射到 Triton 模板 | P4 | 增加模板覆盖 + 参数适配层 |
| compile_and_profile 在 LLM session 中报错 | P4 | 更好的错误信息 + 降级策略 |
| 6GB 显存不够跑大 shape | P7 | 限制 shape ≤2048 |
| Arke 不比直写 Triton 好 | P5 | Gate G4 决策矩阵 |
| API timeout/rate limit | P4-P5 | 重试 + fallback + 用 Sonnet 代替 Opus |

---

*计划版本: v3.0 | 创建: 2026-04-01 | 基于 v2.1.4 实际执行重构*
*核心变化: Week → Phase, 目标 SMART 化, Gate 驱动的阶段推进*
