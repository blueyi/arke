# Stage 1 Gate System v3 — Function-First + Accuracy-Always + Performance-Progressive

## Design Principles

**Gate priority: Function > Accuracy > Performance**

> **Shape definitions:** All Tier 1/2/3 shapes (including non-aligned cases for every op category)
> are defined in [benchmarks/BENCHMARK.md § Shape Matrix](../../benchmarks/BENCHMARK.md#3-shape-matrix).
> Gate criteria reference those shapes by tier and count.

每个 Gate 的本质目标不同：
- 功能 Gate：验证某项能力是否存在（能不能做）
- 精度 Gate：验证涉及算子的数值正确性（做得对不对）
- 性能 Gate：验证在对应阶段的性能水位（做得快不快）

性能标准随 Arke 开发阶段渐进提高，不在早期 Gate 设不切实际的性能目标。

---

## Gate ↔ 本质目标映射

| Gate | 本质类型 | 核心问题 | 精度要求 | 性能要求 |
|:-----|:---------|:---------|:---------|:---------|
| G0 | **功能** | 环境能跑吗？ | — | — |
| G1 | **功能+精度** | IR 能表达 + 验证对吗？ | Tier 3 全量 100% | — |
| G2 | **功能+精度+性能** | Codegen 能生成正确且可用的 kernel 吗？ | Tier 3 全量 100% | 初始性能基线 |
| G3 | **功能+精度** | LLM 能自主完成闭环优化吗？ | Tier 3 抽样 100% | 初始性能基线 |
| G4 | **精度+性能** | Arke 比 LLM-direct 好在哪？| Tier 3 全量 100% | 对比优势 |
| G5 | **精度+性能** | 真实模型能用吗？| 多配置 100% | E2E 可接受 |

---

## G0: Environment Feasibility
**类型：功能**
**核心问题：** CUDA + Triton + PyTorch 工具链在目标硬件上可用吗？

| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G0.1 | CUDA 检测 | 功能 | `torch.cuda.is_available()` | `True` |
| G0.2 | Triton 编译 | 功能 | Triton matmul kernel 编译 | exit 0, 无编译错误 |
| G0.3 | GPU 执行 | 功能 | Triton matmul [128,128,128] 执行 | 返回非零 tensor |
| G0.4 | 测试框架 | 功能 | `pytest tests/ -q` | ≥ 100 passed, 0 failed |

**出口命令：** `arke gate G0`
**出口产物：** CI log（make test 通过）

---

## G1: IR Expressiveness & Validation Correctness
**类型：功能 + 精度**
**核心问题：** IR 系统能完整表达计算意图和优化策略吗？验证器在所有 shape 上都正确吗？

### 功能 Criteria
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G1.1 | OP_CATALOG 覆盖 | 功能 | `len(OP_CATALOG)` | ≥ 10 ops |
| G1.2 | Strategy 决策类型 | 功能 | 枚举 decision kinds | ≥ 6 种 |
| G1.3 | IR 序列化完备 | 功能 | 所有 10 ops × `from_json(to_json(ir))` | 100% round-trip 一致 |
| G1.4 | .ak 解析 → IR | 功能 | ≥ 3 .ak 文件 parse → AST → IR | 与 KernelBuilder 输出一致 |
| G1.5 | V0 静态验证可用 | 功能 | V0 validator 对 10 ops 执行 | 100% 完成, 延迟 < 1ms |
| G1.6 | 单元测试覆盖 | 功能 | `pytest tests/ -q` | ≥ 200 passed, 0 failed |

### 精度 Criteria
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G1.7 | V1 数值验证-matmul | 精度 | **Tier 3 matmul 50 shapes**, 3 random seeds, f16 | **100% pass** (atol=0.1, rtol=0.05) |
| G1.8 | V1 数值验证-softmax | 精度 | **Tier 3 softmax 25 shapes**, 3 random seeds, f16 | **100% pass** |
| G1.9 | V1 数值验证-elementwise | 精度 | **Tier 3 elementwise 15 shapes**, 3 seeds | **100% pass** |

> **无性能要求。** G1 只验证 IR 表达能力和数值正确性，不关心 kernel 快不快。

**出口命令：** `arke gate G1`
**出口产物：**
- `gate_results/G1/validation_matrix.csv` — shape × op × seed → pass/fail
- `gate_results/G1/unit_tests.log`

---

## G2: Codegen Correctness & Baseline Performance
**类型：功能 + 精度 + 性能（初始基线）**
**核心问题：** 代码生成能产出正确的 GPU kernel 吗？性能处于什么水位？

### 功能 Criteria
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G2.1 | Pipeline 连通 | 功能 | IR → Strategy → Codegen → Compile → Run | 单个 kernel 端到端通过 |
| G2.2 | 多算子模板 | 功能 | matmul + softmax + fused_matmul_relu 各生成 Triton 代码 | 3 个模板均编译通过 |

### 精度 Criteria（Tier 3 全量, hard gate）
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G2.3 | matmul 正确性 | 精度 | L1 Tier 3 **50 shapes**, f16, vs NumPy | **100% allclose** (atol=0.1) |
| G2.4 | softmax 正确性 | 精度 | L1 Tier 3 **25 shapes**, f16, vs NumPy | **100% allclose** |
| G2.5 | elementwise 正确性 | 精度 | L1 Tier 3 **15 shapes**, relu/gelu/silu | **100% allclose** |

### 性能 Criteria（初始基线 — 阈值较宽松）
> 这是 Arke 第一次出性能数据，阈值对应"模板基本可用"的水平。

| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G2.6 | matmul 性能通过率 | 性能 | Tier 3 50 shapes vs cuBLAS (排除 M≤32) | **≥ 50%** shapes 达到 ≥ 50% cuBLAS |
| G2.7 | matmul 性能 geomean | 性能 | Tier 3 50 shapes geomean (排除 M≤32) | **geomean ≥ 60%** cuBLAS |
| G2.8 | softmax 性能通过率 | 性能 | Tier 3 25 shapes vs cuDNN (排除 N≤32) | **≥ 40%** shapes 达到 ≥ 50% cuDNN |

> **性能阈值较低。** 这是 Phase 1.2 的出口 — 手动策略的模板 codegen，还没有 LLM 优化和 autotune。目标是"能用"而非"最快"。

**出口命令：** `arke gate G2`
**出口产物：**
- `gate_results/G2/matmul_tier3.csv` — 50 shapes × baselines
- `gate_results/G2/softmax_tier3.csv` — 25 shapes × baselines
- `gate_results/G2/summary.json` — 通过率/geomean/worst_case

---

## G3: LLM Agent Autonomous Optimization
**类型：功能 + 精度（+ 性能观测）**
**核心问题：** LLM 能在无人工干预下完成优化闭环吗？在不同 shape 上都能做到吗？

### 功能 Criteria
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G3.1 | 工具使用广度 | 功能 | 单次 session trajectory | ≥ 8 distinct tools |
| G3.2 | 决策能力 | 功能 | 单次 session trajectory | ≥ 4 strategy decisions |
| G3.3 | 闭环完整性 | 功能 | Agent runner | start → finish, 0 human steps |
| G3.4 | 错误恢复 | 功能 | trajectory 分析 | ≥ 1 次 rollback → 成功恢复 |
| G3.5 | 多 provider 支持 | 功能 | Anthropic + OpenAI 各跑一次 | 两个 provider 均完成 |
| G3.6 | 轨迹记录 | 功能 | JSONL output | header + ≥ 6 step records, 格式合法 |

### 精度 Criteria（Agent 产出的 kernel 必须全部正确）
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G3.7 | Agent kernel 正确性-泛化 | 精度 | Agent 在 **≥ 10 diverse shapes** 上生成 kernel (从 Tier 3 按类别抽样: 方阵×3, 矩形×2, 非对齐×2, LLM×3) | **100% 正确** |

### 性能观测（记录但不卡 gate）
| # | Criterion | 类型 | 验证 | 说明 |
|---|-----------|:----:|------|------|
| G3.P1 | Agent kernel 性能 | 观测 | Agent kernels vs cuBLAS geomean | 记录到 CSV，不设阈值 |
| G3.P2 | Agent vs G2 模板 | 观测 | Agent kernel vs 模板 kernel 性能比 | 观测 LLM 优化是否提升了模板基线 |

> **G3 的性能是观测不是门槛。** LLM agent 刚开始跑时性能不稳定是正常的。关键是证明"能闭环"+"产出正确"。性能提升是 G4 的目标。

**出口命令：** `arke gate G3`
**出口产物：**
- `gate_results/G3/agent_trajectories/` — 每个 shape 一个 JSONL
- `gate_results/G3/agent_kernels/` — 每个 shape 的生成 kernel
- `gate_results/G3/correctness.csv` — shape → correct/incorrect
- `gate_results/G3/performance.csv` — shape → latency → vs_cublas（观测数据）

---

## G4: Comparative Advantage over Direct LLM
**类型：精度 + 性能（对比）**
**核心问题：** Arke 路线比 LLM 直接写 Triton 好在哪里？好多少？

### 精度 Criteria（Tier 3 全量对比）
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G4.1 | Arke 正确率 ≥ LLM-direct | 精度 | Tier 3 matmul 50 shapes × 3 trials | `arke_correct_rate ≥ direct_correct_rate` |
| G4.2 | Arke 一致性 ≥ LLM-direct | 精度 | Tier 3 × 3 trials 的方差 | `arke_stddev ≤ direct_stddev` |

### 性能 Criteria（基于开发进度的渐进目标）
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G4.3 | vs LLM-direct 性能 | 性能 | Tier 3 matmul geomean (排除 M≤32) | Arke geomean **≥ 90%** LLM-direct geomean |
| G4.4 | vs P1 Expert (FlagGems) | 性能 | Tier 3 matmul geomean (排除 M≤32) | Arke geomean **≥ 70%** FlagGems geomean |
| G4.5 | Token 效率 | 性能 | 端到端 token 消耗统计 | Arke total tokens **≤ 60%** LLM-direct tokens |

### L2 观测（记录但不卡 gate）
| # | Criterion | 类型 | 验证 | 说明 |
|---|-----------|:----:|------|------|
| G4.P1 | L2 融合算子 | 观测 | matmul+gelu Tier 3 shapes | 记录 Arke fused vs separate vs FlagGems |

> **G4 的性能阈值比 v2 降低了。** 因为 Stage 1 的 Arke 还没有 autotune 和 MLIR 后端，模板 codegen 在小 shape 天然弱。vs FlagGems 70% 是务实目标。

**出口命令：** `arke gate G4`
**出口产物：**
- `gate_results/G4/comparison_tier3.csv` — Tier 3 × method × trial
- `gate_results/G4/token_efficiency.json`
- `gate_results/G4/summary.json`

---

## G5: End-to-End Model Integration
**类型：精度 + 性能（E2E）**
**核心问题：** Arke kernel 放进真实模型后，能用吗？性能可接受吗？

### 精度 Criteria（多配置, hard gate）
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G5.1 | 推理正确性-多 seq | 精度 | GPT-2, seq=128/256/512 | **100%** top-1 match, max_logit_diff < 5.0 |
| G5.2 | 推理正确性-多 batch | 精度 | GPT-2, batch=1/4/8, seq=128 | **100%** top-1 match |

### 性能 Criteria
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G5.3 | 延迟-seq=128 | 性能 | L3 benchmark | Arke ≤ **1.15×** eager |
| G5.4 | 延迟-seq=512 | 性能 | L3 benchmark | Arke ≤ **1.20×** eager |
| G5.5 | 延迟泛化 | 性能 | L3, 3 seq_lens | ≥ **2/3** seq_lens: Arke ≤ 1.15× eager |
| G5.6 | 内存 | 性能 | L3, 所有配置 | **100%** peak_mem ≤ 6144 MB |

### 功能 Criteria
| # | Criterion | 类型 | 验证 | 通过条件 |
|---|-----------|:----:|------|---------|
| G5.7 | 替换覆盖率 | 功能 | patch 统计 | ≥ 48 Conv1D/Linear 替换 |

> **E2E 性能阈值放宽。** seq=128 从 1.1× 调到 1.15×，seq=512 设 1.20×。原因：monkey-patching overhead 在 Stage 1 无法消除（需要 Stage 2 的 torch.compile backend 集成）。精度不妥协。

**出口命令：** `arke gate G5`
**出口产物：**
- `gate_results/G5/e2e_results.csv` — seq × batch × mode → latency/mem/correct
- `gate_results/G5/summary.json`

---

## Gate 渐进性能路线

展示性能标准如何随开发阶段提升：

```
           G2 (模板)    G3 (Agent)    G4 (对比)     G5 (E2E)
           ─────────    ──────────    ─────────     ────────
功能        ✓ 必须      ✓ 核心        —             ✓ 替换覆盖
精度        100%        100%          ≥ LLM-direct  100% 多配置
性能目标    ≥50% cuBLAS  观测记录     ≥90% direct   ≤1.15× eager
                                     ≥70% FlagGems
性能类型    绝对水位     不设门槛      相对优势       E2E overhead
```

**为什么 G3 不卡性能？**
- Agent 初次运行时策略质量不稳定
- Agent 的价值在于"能自主闭环"，不在于"第一次就最快"
- 性能提升是迭代过程，在 G4 通过对比验证

**为什么 G4 卡相对优势而非绝对水位？**
- G4 的问题是"Arke 比 LLM-direct 好吗"，不是"Arke 有多快"
- 绝对性能已在 G2 设定基线

---

## 排除规则

| 场景 | 处理 | 原因 |
|:-----|:-----|:-----|
| M ≤ 32 (matmul) | 精度必须通过，性能不计入统计 | Triton ~55μs launch floor |
| N ≤ 32 (softmax) | 精度必须通过，性能不计入统计 | 同上 |
| OOM shapes | 跳过，记录 "OOM" | 6GB VRAM 限制 |
| Triton 编译超时 (>60s) | 记录 "TIMEOUT"，精度标 fail | 模板可能需要修复 |

---

## 与 Benchmark CLI 的集成

```bash
# Gate 验证（默认 Tier 3）
arke gate G0                      # 环境检查
arke gate G1                      # IR + 验证器 + Tier 3 数值
arke gate G2                      # L1 Tier 3 全量 bench
arke gate G3                      # Agent 10 shapes 闭环
arke gate G4                      # Tier 3 对比 + token 统计
arke gate G5                      # L3 多配置 E2E
arke gate --all                   # 全部（预计 30-60 分钟）

# 快速检查（Tier 1, 日常开发用）
arke gate G2 --tier 1             # 15 shapes 快速回归

# 输出格式
arke gate G2

  G2: Codegen Correctness & Baseline Performance
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  功能:
    G2.1 Pipeline 连通              ✅ PASS
    G2.2 多算子模板                  ✅ PASS  3/3
  精度:
    G2.3 matmul 正确性 (50 shapes)   ✅ PASS  50/50 (100%)
    G2.4 softmax 正确性 (25 shapes)  ✅ PASS  25/25 (100%)
    G2.5 elementwise 正确性 (15)     ✅ PASS  15/15 (100%)
  性能:
    G2.6 matmul ≥50% rate           ✅ PASS  36/46 (78% ≥ 50%)
    G2.7 matmul geomean             ✅ PASS  72% cuBLAS (≥ 60%)
    G2.8 softmax ≥50% rate          ✅ PASS  12/22 (55% ≥ 40%)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  G2: PASS (8/8)
  产物: gate_results/G2/
```

---

## 与 plan-v3.0 的对应关系

| Gate | Phase | 原目标 | 新目标 |
|:-----|:------|:-------|:-------|
| G0 | 1.0 | "Triton matmul runs" | 功能：4 项环境检查 |
| G1 | 1.1 | "Known-good strategy representable" | 功能 6 项 + 精度 Tier 3 全量 3 项 |
| G2 | 1.2 | "perf ≥ 70% cuBLAS" | 功能 2 项 + 精度 Tier 3 全量 3 项 + 性能 3 项(宽松) |
| G3 | 1.3-1.4 | "matmul perf ≥ 50% cuBLAS" | 功能 6 项 + 精度 Tier 3 抽样 1 项 + 性能观测 |
| G4 | 1.5 | "Arke ≥ LLM-direct across ≥5" | 精度对比 2 项 + 性能对比 3 项 |
| G5 | 1.7 | "latency ≤ torch.compile" | 精度多配置 2 项 + 性能 4 项(放宽) + 功能 1 项 |
