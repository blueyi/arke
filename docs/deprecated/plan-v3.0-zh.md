# Arke — 执行计划 v3.0 (中文版)

> 每个 Phase 有 SMART 完成标准，Gate 通过后才能推进到下一阶段。
> 日期: 2026-04-01

---

## 当前状态快照 (2026-04-01)

### 已完成
- ✅ GPU 环境 (PyTorch 2.6.0+cu124, Triton 3.2.0, RTX 3060)
- ✅ IR 体系 (Semantic IR + Strategy IR, JSON Schema, 10 个算子)
- ✅ Builder + Shape 推断 (全 10 个算子)
- ✅ 验证系统 (V0 静态 + V1 数值 + 资源估算)
- ✅ 合法动作枚举引擎
- ✅ ArkeEnv 完整实现
- ✅ Triton 代码生成 (matmul + softmax, 模板引擎)
- ✅ 端到端流水线 (IR → strategy → codegen → GPU)
- ✅ LLM Runner (Anthropic + OpenAI API, fallback, 重试)
- ✅ LLM 闭环优化 (Sonnet 4.6, 23 次 tool call, 106% cuBLAS)
- ✅ GPU 正确性验证 (同精度标杆)
- ✅ 精度对比框架 (10 项指标, 三级判定)
- ✅ 轨迹 JSONL 导出
- ✅ 237 个测试通过

### Gate 状态
- G0 ✅ — Triton matmul 在 RTX 3060 跑通
- G1 ✅ — 已知好的 strategy 可用 Arke IR 表达
- G2 ✅ — 手动 strategy → codegen → 105-160% cuBLAS
- G3 ✅ — LLM tool-use → 106% cuBLAS + softmax 正确

---

## Phase 总览

```
Phase 1.0 ✅  环境搭建
Phase 1.1 ✅  IR + 验证基础
Phase 1.2 ✅  代码生成 + 端到端流水线
Phase 1.3 ✅  LLM Agent 联调
Phase 1.4 ✅  LLM 闭环优化
Phase 1.5 ⬅  评估框架 + 对比实验（当前阶段）
Phase 1.6     .ak 语法解析 + CLI
Phase 1.7     整模型端到端
Phase 1.8     MVP 发布
```

---

## Phase 1.0: 环境搭建 ✅

**目标：** 一键可复现的开发环境，含 GPU 验证。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.0.1 | `make setup` 创建 venv + 安装全部依赖 | 全新 clone → `make setup` → 无报错 | ✅ |
| 1.0.2 | PyTorch 检测到 CUDA GPU | `torch.cuda.is_available() == True` | ✅ |
| 1.0.3 | Triton 编译并运行 matmul | GPU 冒烟测试脚本退出码 0 | ✅ |
| 1.0.4 | `pytest tests/ -q` 无导入错误 | 所有测试可收集（无 GPU 时跳过 GPU 门控测试） | ✅ |

---

## Phase 1.1: IR + 验证基础 ✅

**目标：** 建立 Semantic IR 和 Strategy IR 体系，静态和数值验证覆盖 ≥10 个算子。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.1.1 | Semantic IR 支持 ≥10 个算子 | `len(OP_CATALOG) >= 10` | ✅ |
| 1.1.2 | Strategy IR 支持 ≥6 种决策类型 | `kinds ⊇ {tile,fuse,place,parallel,reorder,algorithm}` | ✅ |
| 1.1.3 | JSON Schema 验证 IR 往返序列化 | `jsonschema.validate(ir.to_json(), schema)` 通过 | ✅ |
| 1.1.4 | V0 静态验证 < 1ms | 验证器耗时实测 < 1ms | ✅ |
| 1.1.5 | V1 数值验证 (NumPy 标杆) | 3 个随机种子 trial 对 matmul、softmax 通过 | ✅ |
| 1.1.6 | 全 10 个算子的 Shape 推断 | `infer_shapes()` 返回正确 shape | ✅ |
| 1.1.7 | ≥100 个单元测试通过 | `pytest` 通过数 ≥ 100 | ✅ (237) |

**Gate G1：** IR 可表达已知好的 strategy（matmul tiling + fusion）✅

### 任务
- [x] 算子目录 P0（10 个算子）
- [x] Semantic IR 数据类 + JSON 序列化
- [x] Strategy IR 数据类 + JSON 序列化
- [x] JSON Schema 定义
- [x] KernelBuilder（Python → IR）
- [x] Shape 推断引擎（全 10 个算子）
- [x] V0 静态验证器（shape + 约束检查）
- [x] V1 数值验证器（NumPy 标杆对比）
- [x] 资源估算（共享内存、寄存器用量）
- [x] 硬件描述文件：`nvidia_ampere_rtx3060.json`

---

## Phase 1.2: 代码生成 + 端到端流水线 ✅

**目标：** 从 Strategy IR 生成 Triton 代码，端到端流水线产出可在 GPU 执行的 kernel，**性能 ≥ 70% cuBLAS**。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.2.1 | matmul Triton 代码生成正确 | 生成的 kernel 通过 V1 数值验证 | ✅ |
| 1.2.2 | softmax Triton 代码生成正确 | 同上 | ✅ |
| 1.2.3 | 融合 matmul+relu 代码生成正确 | 融合 kernel 通过数值验证 | ✅ |
| 1.2.4 | GPU 执行**性能 ≥ 70% cuBLAS** | `compile_and_profile()` 返回 `vs_baseline >= 0.7` | ✅ (105-160%) |
| 1.2.5 | 流水线完整串联 | IR → strategy → codegen → compile → profile 一条龙调用 | ✅ |
| 1.2.6 | ≥9 个 GPU 集成测试 | `ARKE_GPU_TESTS=1` 下 GPU 测试通过 | ✅ |

**Gate G2：** 手动 strategy → codegen → ≥70% cuBLAS ✅

### 任务
- [x] Triton matmul 模板（Jinja2）
- [x] Triton softmax 模板
- [x] Triton matmul+relu 融合模板
- [x] 模板引擎（strategy 参数 → Triton 模板参数映射）
- [x] TritonBackend（translate + compile + run）
- [x] cuBLAS 基线性能分析器（vs_baseline 计算）
- [x] 端到端流水线组装（`pipeline.py`）
- [x] GPU 集成测试

---

## Phase 1.3: LLM Agent 联调 ✅

**目标：** LLM 通过 tool-use 自主完成优化循环，全程零人工干预。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.3.1 | LLM 使用 ≥8 种不同 tool | 轨迹日志中出现 ≥8 个不同 tool 名 | ✅（全部 10 种）|
| 1.3.2 | LLM 应用 ≥4 个 strategy 决策 | `result.decisions >= 4` | ✅ (13) |
| 1.3.3 | LLM 调用 verify_correctness | tool 出现在轨迹中 | ✅ |
| 1.3.4 | LLM 调用 compile_and_profile | tool 出现在轨迹中 | ✅（5 次）|
| 1.3.5 | LLM 使用 checkpoint + rollback | 两个 tool 均出现在轨迹中 | ✅ |
| 1.3.6 | Fallback 机制工作 | 超时/错误时自动切换 fallback 模型 | ✅ |
| 1.3.7 | 多 provider 支持 | Anthropic + OpenAI 兼容 API 均可用 | ✅ |
| 1.3.8 | 零人工干预 | 从启动到完成无需手动步骤 | ✅ |

### 任务
- [x] Tool schema 定义（10 个 tool）
- [x] Session 生命周期管理器
- [x] System prompt 构建器（硬件感知）
- [x] LLM Runner（异步、多 provider）
- [x] LLM 配置（模型选择、fallback 链）
- [x] 错误恢复 + 重试逻辑
- [x] Agent matmul 示例脚本

---

## Phase 1.4: LLM 闭环优化 ✅

**目标：** LLM 优化后的 kernel 通过 GPU 正确性验证，**性能 ≥ 50% cuBLAS**，覆盖多个算子。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.4.1 | LLM 优化的 matmul → GPU 正确 | `verify_correctness` 通过 + GPU 输出匹配同精度 NumPy 标杆 | ✅ |
| 1.4.2 | LLM 优化的 matmul **≥ 50% cuBLAS** | `compile_and_profile()` 返回 `vs_baseline >= 0.5` | ✅ (106.1%) |
| 1.4.3 | LLM 优化的 softmax → GPU 正确 | 同 1.4.1 | ✅ |
| 1.4.4 | LLM 优化的 fused_matmul_relu → GPU 正确 | 同 1.4.1 | ✅ |
| 1.4.5 | compile_and_profile 返回真实 GPU 数据 | 返回包含 `latency_us`、`tflops`、`vs_baseline` 字段 | ✅ |
| 1.4.6 | 错误恢复：LLM 失败后自动回滚 | 轨迹中有 失败决策 → rollback → 成功 | ✅ |
| 1.4.7 | 轨迹导出为 JSONL | `export_trajectory()` 输出 state/action/result 记录 | ✅ |
| 1.4.8 | ≥220 个测试通过 | `pytest` 通过数 ≥ 220 | ✅ (237) |

**Gate G3：** LLM tool-use → matmul ≥50% cuBLAS + softmax 正确 ✅

### 任务
- [x] Strategy 决策 → Triton 模板参数映射
- [x] GPU 正确性验证（同精度 NumPy 标杆）
- [x] compile_and_profile 中的 vs_baseline 字段
- [x] 精度对比框架（10 项指标、三级判定）
- [x] 标杆数据源（NumPyCPU、TorchGPU、Custom）
- [x] 轨迹 JSONL 写入器
- [x] GPU 正确性测试
- [x] 端到端 agent 示例（matmul、softmax、fused）

---

## Phase 1.5: 评估框架 + 对比实验 🔨

**目标：** 定量证明 Arke（LLM + tool-use）产出的 kernel 比 LLM 直写 Triton 更正确、更稳定、**更快**，覆盖 ≥5 个基准任务。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.5.1 | 定义 ≥5 个基准任务 | `benchmarks/tasks.py` 中包含 ≥5 个 task 定义 | ⬜ |
| 1.5.2 | Arke 完成全部任务 | 每个 task 有 Arke 结果（正确性 + 性能） | ⬜ |
| 1.5.3 | LLM 直写 Triton 完成全部任务 | 每个 task 有直写结果 | ⬜ |
| 1.5.4 | Arke 正确率 ≥ 直写 Triton | `arke_correct_rate >= direct_correct_rate` | ⬜ |
| 1.5.5 | **Arke 平均性能 ≥ 直写 Triton** | `mean(arke_vs_cublas) >= mean(direct_vs_cublas)` | ⬜ |
| 1.5.6 | Arke 方差 ≤ 直写 Triton | `var(arke_results) <= var(direct_results)` | ⬜ |
| 1.5.7 | 生成评估报告 | `benchmarks/report.md` 含数据表格和分析 | ⬜ |
| 1.5.8 | Token 效率对比 | 记录 Arke vs 直写的总 token 消耗 | ⬜ |

**Gate G4：** Arke 正确率和**性能** ≥ LLM 直写 Triton

### Gate G4 决策矩阵
| 结果 | 结论 | 下一步 |
|------|------|--------|
| Arke 正确 + 快 | ✅ 继续推进 | Phase 1.6–1.8 |
| Arke 正确 + 慢 | ⚠️ Arke 定位为验证框架 | 调整定位 |
| Arke ≈ 直写 Triton | ⚠️ 无明显优势 | 审视增量价值 |
| 两者都差 | ❌ | 终止或根本性转型 |

### 精度对比设计

**同精度标杆（默认）：** 测试实现正确性，而非精度损失。
- GPU f16 kernel → NumPy CPU f16 标杆
- 差异来源：规约顺序、FMA、非确定性

**可插拔标杆源：**
- `NumPyCPUSource` — NumPy CPU，同精度（默认，无 GPU 时 fallback）
- `TorchGPUSource` — PyTorch GPU（GPU 对 GPU 比较）
- `CustomSource` — 用户提供（如昇腾标杆数据）

**度量指标（10 项）：** abs/rel 误差（max/mean/P90/P99）、ULP 误差、余弦相似度、符号翻转率、NaN/Inf 计数、完全匹配率

**三级判定：** Accept / Review / Reject，各 dtype 独立阈值

### 任务
- [ ] 定义 ≥5 个基准任务（不同 matmul 尺寸、softmax、融合算子、规约）
- [ ] 实现 Arke 基准运行器
- [ ] 实现 LLM 直写 Triton 基线运行器
- [ ] 运行全部任务 × 两种方法 × 3 次 trial
- [ ] 统计分析（均值、方差、显著性）
- [ ] Token 计数集成
- [ ] 生成评估报告（`benchmarks/report.md`）
- [ ] Gate G4 决策

---

## Phase 1.6: .ak 语法解析 + CLI ⬜

**目标：** 人类可读的 `.ak` 语法解析为 Semantic IR，CLI 支持 parse/optimize/inspect 工作流。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.6.1 | 解析 matmul kernel | `parser.parse("examples/01_matmul.ak")` 返回 AST | ⬜ |
| 1.6.2 | 解析融合 kernel | `parser.parse("examples/02_matmul_relu_fused.ak")` 返回 AST | ⬜ |
| 1.6.3 | AST → Semantic IR 正确 | `ast_to_ir(ast)` 等于 `KernelBuilder.build()` 输出 | ⬜ |
| 1.6.4 | CLI `arke parse` | `arke parse kernel.ak -o kernel.json` 输出合法 JSON | ⬜ |
| 1.6.5 | CLI `arke optimize` | `arke optimize kernel.json --target ampere` 启动 LLM session | ⬜ |
| 1.6.6 | CLI `arke inspect` | `arke inspect kernel.json` 输出人类可读 IR | ⬜ |
| 1.6.7 | ≥3 个 .ak 示例端到端跑通 | matmul、softmax、fused_matmul_relu：parse → optimize → GPU | ⬜ |

**依赖：** Phase 1.4 完成

### 任务
- [ ] EBNF 文法定义（`arke.lark`）
- [ ] Lark 解析器实现
- [ ] AST 节点定义
- [ ] AST → Semantic IR 转换器
- [ ] CLI 入口点（`arkec` 或 `arke`）
- [ ] `parse` 子命令
- [ ] `optimize` 子命令
- [ ] `inspect` 子命令
- [ ] 示例 .ak 文件（≥3 个）

---

## Phase 1.7: 整模型端到端 ⬜

**目标：** 在真实模型中替换 kernel 为 Arke 优化版本，推理正确性验证通过，**延迟 ≤ torch.compile**。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.7.1 | GPT-2 Small 推理正确 | Arke kernel 输出匹配 PyTorch eager 输出（同精度标杆） | ⬜ |
| 1.7.2 | **推理延迟 ≤ torch.compile** | `arke_latency <= torch_compile_latency`（同硬件） | ⬜ |
| 1.7.3 | ≥2 个算子被替换 | matmul + softmax（或 matmul + layernorm） | ⬜ |
| 1.7.4 | 显存 ≤ 6GB | 适配 RTX 3060 Laptop 6GB 显存 | ⬜ |

**Gate G5：** GPT-2 Small + Arke kernel — 正确且**延迟 ≤ torch.compile**

**依赖：** Phase 1.5 Gate G4 通过

### 任务
- [ ] GPT-2 Small 基线分析（eager + torch.compile）
- [ ] PyTorch 自定义算子注册（`torch.library`）
- [ ] Arke kernel 集成到 GPT-2 前向传播
- [ ] 正确性验证（token 级输出对比）
- [ ] 延迟对比（Arke vs eager vs torch.compile）
- [ ] 显存分析

---

## Phase 1.8: MVP 发布 ⬜

**目标：** 发布 v0.1.0，一键搭建、CI 通过、文档完整、评估结果可复现。

### 完成标准
| # | 标准 | 验证方式 | 状态 |
|---|------|----------|:----:|
| 1.8.1 | `make setup` 在全新 clone 上可用 | 在干净 Ubuntu 22.04 上测试通过 | ⬜ |
| 1.8.2 | CI 全绿（3 个 Python 版本） | GitHub Actions 在 3.10、3.11、3.12 上通过 | ⬜ |
| 1.8.3 | README 完整（安装 + 快速开始 + 示例） | 新用户按 README 可跑通 demo | ⬜ |
| 1.8.4 | API 文档完整 | 所有公开 class/function 有 docstring | ⬜ |
| 1.8.5 | 评估报告发布 | `benchmarks/report.md` 含表格、图表、结论 | ⬜ |
| 1.8.6 | 轨迹数据可下载 | 评估运行的 JSONL 文件公开可获取 | ⬜ |
| 1.8.7 | v0.1.0 tag | `git tag v0.1.0` | ⬜ |

### 任务
- [ ] Makefile 包含 `setup` / `test` / `lint` / `bench` 目标
- [ ] CI 工作流修复（lint + 类型检查 + 测试）
- [ ] README 快速开始在全新机器上验证
- [ ] API 文档审查
- [ ] 评估报告定稿
- [ ] 轨迹数据打包
- [ ] 版本 tag + GitHub release

---

## Phase 进入/退出检查清单

推进到下一 Phase 前必须确认：

```
□ 全部完成标准达成（100%）
□ 对应 Gate 通过（如适用）
□ 全部已有测试仍然通过（无回归）
□ 代码已 commit + push
```

### 异常处理

- **某项标准无法达成：** 分析根因，与 Leon 讨论是否降低标准或跳过
- **发现新的必要工作：** 加入当前 Phase 标准列表（不拖延到下一 Phase）
- **Gate 失败：** 按决策矩阵处理，可能调整方向或终止

---

## 风险矩阵

| 风险 | 影响阶段 | 应对措施 |
|------|:--------:|----------|
| LLM 决策无法映射到 Triton 模板 | Phase 1.4 | 扩展模板覆盖 + 参数适配层 |
| compile_and_profile 在 LLM session 中报错 | Phase 1.4 | 更好的错误信息 + 优雅降级 |
| 6GB 显存不够跑大尺寸 | Phase 1.7 | 限制 shape ≤ 2048 |
| Arke 不比直写 Triton 好 | Phase 1.5 | Gate G4 决策矩阵 |
| API 超时/限流 | Phase 1.4-1.5 | 重试 + fallback + 优先用 Sonnet |

---

*计划版本: v3.0 | 创建: 2026-04-01 | 最后更新: 2026-04-01*
