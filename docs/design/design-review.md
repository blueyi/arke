# Arke — 端到端设计审视与假设验证框架

> 从项目全生命周期视角审视：每个阶段验证什么假设？什么信号触发 pivot/kill？
> Date: 2026-03-31

---

## 一、Arke 的核心赌注

整个项目其实就是在回答一个问题：

> **结构化的 AI-kernel 交互协议（Arke），是否比"直接让 LLM 写代码"更好？**

"更好"需要量化拆解。Arke 的价值主张有四层，每层都是可独立验证（或证伪）的假设：

```
假设 H1（正确性）：通过 Arke 的结构化协议，LLM 生成的 kernel 正确率更高
假设 H2（性能）：  通过 Arke 的策略搜索，LLM 优化的 kernel 性能更好
假设 H3（可解释）：@rationale 让优化决策可追溯，人类能审查和信任
假设 H4（可迁移）：同一套 Arke 协议能跨硬件（NVIDIA → Ascend），LLM 无需重新学习
```

如果 **H1 和 H2 都不成立**——Arke 没有存在价值。
如果 **H1 成立但 H2 不成立**——Arke 作为"验证框架"有价值，但作为"优化工具"无价值。
如果 **H1+H2 成立但 H4 不成立**——Arke 是一个好用的单硬件工具，但不是平台。

**这是整个项目最重要的思考框架。每一周的工作都应该指向验证（或证伪）这些假设。**

---

## 二、当前设计的逻辑链条

```
Arke 的端到端逻辑链：

[输入] 算子规格 + 硬件目标
    ↓
[L1] 语义表达：Semantic IR 描述"算什么"
    ↓ LLM 通过 tool-use
[L2] 策略生成：Strategy IR 描述"怎么优化"
    ↓ 后端翻译
[L3] 代码生成：Triton/AscendC 代码
    ↓ 编译执行
[输出] GPU/NPU 上跑出正确且高性能的 kernel
    ↓ 验证
[反馈] 正确性 + 性能 → 回到 L2 调整
```

**链条中的每个环节都可能断裂。** 下面逐个审视。

---

## 三、每个环节的风险与验证

### 环节 1：Semantic IR（L1）

**问题**：Semantic IR 能否准确描述所有目标算子？

**风险**：
- 低风险。matmul、softmax、attention 的语义描述是确定性的，不涉及 AI 判断
- 算子目录是人工定义的，正确性可保证

**验证**：
- W1 结束时：matmul/softmax/relu 的 Semantic IR 能 JSON 往返，能自动推导 output shape
- 验证方式：单元测试

**不太可能失败。跳过。**

---

### 环节 2：LLM + Tool-use 策略搜索（L2）

**问题**：LLM 能否通过 tool-use 做出有效的优化决策？

**这是整个项目最大的未知。** 细分为三个子问题：

#### 2a. LLM 能否理解 tool 返回的信息？

```
LLM 调用 analyze_compute() 得到：
  arithmetic_intensity = 341.3, bottleneck = compute_bound

LLM 能否据此推理出"应该优先 tiling 以最大化数据复用"？
```

**风险**：中。当前顶级 LLM（Claude Opus、GPT-4o）对这类分析有不错的推理能力，但不确定在多步决策中是否稳定。

**验证**：
- W4 第一次 LLM 联调就能看到
- 指标：LLM 是否按照合理的优先级做决策（fusion → tiling → placement）
- 如果 LLM 随机乱做 → 需要更强的 prompt engineering 或 few-shot 示例

#### 2b. LLM 做出的决策组合是否能收敛到好的结果？

```
LLM 做了 20 步决策后的 strategy，编译出来的 kernel：
  - 正确吗？（H1）
  - 性能好吗？（H2）
```

**风险**：高。这是核心赌注。

可能的失败模式：
1. **LLM 能做出每一步都"合理"的决策，但组合起来不好** → 需要更强的全局规划能力或 beam search
2. **LLM 陷入局部最优** → 需要 rollback + 探索机制
3. **搜索预算（50步）不够** → 可能需要更高效的搜索策略
4. **LLM 的"直觉"在 GPU 优化领域不准** → in-context learning 或 fine-tuning

**验证**：
- W4 末尾的 matmul agent demo 会给出第一个数据点
- W6 的对比实验会给出决定性数据

#### 2c. tool-use 循环的 token 效率如何？

```
每次 observe() + list_legal_actions() + apply_decision() 消耗多少 token？
50 步优化总共消耗多少 token？成本可接受吗？
```

**风险**：中。IR 的 JSON 表示可能很长。

**验证**：
- W4 联调时记录 token 使用量
- 如果 token 太多 → 需要 observe_diff()（增量观测）或压缩 IR 表示

---

### 环节 3：Codegen（L2 → L3）

**问题**：Strategy IR 能否可靠地转换为正确的 GPU 代码？

**双路径分析**：

**路径 A（模板）**：
- 风险低。模板是人写的，覆盖的模式有限但正确
- 问题：模板组合爆炸。(tiling × fusion × placement) 的组合是否都能正确处理？
- 验证：每种组合的 V1 数值验证

**路径 B（LLM 生成）**：
- 风险高。LLM 生成的 Triton/AscendC 代码可能有 bug
- 但 V1 验证是安全网——不正确的代码会被拒绝
- 验证：W5 的 A vs B 对比实验

**关键洞察**：路径 A 是"Arke 的下限"（最差也能跑到这个水平），路径 B 是"Arke 的上限"（如果 LLM 能可靠生成代码，表达力不受模板限制）。

---

### 环节 4：验证系统

**问题**：验证系统是否足够可靠，能区分"正确"和"不正确"？

**风险**：低。NumPy 参考实现是数学上精确的，容差阈值是已知的。

**但有一个隐含风险**：f16 的精度问题。
- 某些计算在 f16 下天然有较大的数值误差
- 如果容差太紧 → 正确的 kernel 被误判
- 如果容差太松 → 错误的 kernel 漏过

**验证**：
- W2 实现 V1 时就需要用 PyTorch f16 matmul 的结果校准容差阈值
- 不是理论值，是实测值

---

### 环节 5：多硬件抽象

**问题**：Arke 的抽象层是否真的能屏蔽硬件差异？

**风险**：中高。NVIDIA 和 Ascend 的编程模型差异很大：
- NVIDIA：隐式数据搬运（tl.load）；Ascend：显式 DMA
- NVIDIA：Thread-level 编程；Ascend：Core-level + 流水线
- NVIDIA：Warp 同步；Ascend：Pipeline Queue 同步

**可能的失败模式**：
- Arke 的 `place(fast_memory)` 抽象掩盖了 Ascend DMA 编程的复杂性
- LLM 为 Ascend 做的决策看起来"合理"但生成的代码无法正确处理 DMA 流水线
- 某些 Ascend 特有的优化（如三级流水线 CopyIn→Compute→CopyOut）在 Arke 的 strategy vocabulary 中无法表达

**验证**：
- Phase 1 无法完全验证（没有 Ascend 硬件）
- 但可以验证的部分：LLM 为 Ascend target 做 tool-use 时，生成的 strategy 是否"看起来合理"（tile 更大、用了 double_buffer 等）
- Phase 2 上 Ascend 硬件后才能端到端验证

**这里有一个诚实的问题**：如果 Phase 1 只在 NVIDIA 上做，Phase 2 上 Ascend 时发现抽象层需要大改怎么办？

**应对**：Phase 1 的 Ascend 骨架任务不是"做完了不管"，而是要定期拿真实的 AscendC 代码样例来验证 Arke 的抽象是否合理。如果发现不对，在 Phase 1 内就改。

---

## 四、阶段性验证里程碑（重新定义）

当前 plan 的 7 个里程碑偏向"技术交付"（IR 可用、Codegen 可用...）。但更重要的是**假设验证里程碑**：

### Gate 0：环境可行性（Phase 1）

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| GPU 环境可用 | Triton matmul 在 RTX 3060 上跑通 | 切换到云 GPU |
| cuBLAS baseline 可测 | torch.mm 性能可稳定复现 | 调整 profiling 方法 |

**这是最简单的 gate，几乎不会失败。**

### Gate 1：IR 表达力验证（Phase 1）

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| Semantic IR 能描述 P0 算子 | matmul, softmax, fused_matmul_relu IR 构建正确 | 扩展 IR schema |
| Strategy IR 能表达已知好的优化 | 手动构造"FlashAttention 风格"的 strategy 能通过 V0 验证 | 扩展 strategy vocabulary |
| 合法动作枚举合理 | matmul 的 tiling candidates 包含已知好的配置 (64×128×32) | 修正枚举逻辑 |

**验证假设**：Arke 的表达力足够。如果已知好的优化策略无法在 Arke 中表达，项目有根本问题。

### Gate 2：端到端通路验证（Phase 2）🔴 关键

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| 手动 strategy → 正确 kernel | 手工构造的 matmul strategy，codegen 后 V1 数值验证通过 | 修 codegen 模板 |
| 手动 strategy → 合理性能 | 手工构造的"好" strategy 达到 ≥ 70% cuBLAS | 修模板或调 tile 参数 |
| **端到端不经过 LLM 就能跑通** | IR → strategy → codegen → compile → run → verify 全链路 | 必须在此解决 |

**这是第一个真正的 gate。** 如果手动构造的好 strategy 都无法通过 codegen 转化为好的 kernel，那问题在 codegen 而非 LLM。必须先证明 "如果策略正确，Arke 能生成好代码"，然后再问"LLM 能否找到好策略"。

**关键洞察**：Gate 2 验证的是 **Arke 的下限**。如果下限就不行，后面的 LLM 部分全是空中楼阁。

### Gate 3：LLM 可行性验证（Phase 4）🔴🔴 最关键

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| LLM tool-use 循环能跑完 | LLM 在 50 步内完成 matmul 优化，不 crash/死循环 | 改 prompt / 简化 tool |
| LLM 的决策"看起来合理" | 决策序列符合专家逻辑（先 fuse → 再 tile → 再 place） | 加 few-shot 示例 |
| **LLM 优化后 kernel 正确** | V1 数值验证通过 | 加错误恢复引导 |
| **LLM 优化后性能 ≥ 50% cuBLAS** | 至少比完全不优化好很多 | 根本问题 → 见下文 |

**如果 LLM 的 50 步 tool-use 连 50% cuBLAS 都达不到，有两种可能：**

1. **Prompt/tool 设计问题** → 可修复。加 few-shot 示例、改 tool 返回信息、添加 hint
2. **LLM 根本不具备 GPU 优化的推理能力** → 项目核心假设不成立

**区分方法**：看 LLM 的 rationale。如果 rationale 的推理是对的但决策参数不好，是调参问题；如果 rationale 本身就是错的（"shared memory 越大越好所以我把所有东西都放进 shared"），说明 LLM 缺乏领域知识。

**应对**：
- 问题 1：优化 prompt、加领域知识到 system prompt、提供 few-shot 示例
- 问题 2：考虑 fine-tuning，或者 pivot 到"LLM 从预定义模式库中选择"而非自由搜索

### Gate 4：对比优势验证（Phase 5）🔴🔴🔴 生死判断

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| **Arke (H1) 正确率 > 直写 Triton 正确率** | 具体数字看实验 | 如果不成立 → pivot |
| **Arke (H2) 性能 ≥ 直写 Triton 性能** | 或至少相当 | 如果差很多 → pivot |
| Arke token 效率合理 | 不比直写 Triton 多 5x 以上 | 压缩 IR 表示 |

**这是整个项目的 existential gate。** 必须诚实面对数据：

| 实验结果 | 结论 | 下一步 |
|----------|------|--------|
| Arke 正确率高 + 性能好 | ✅ 假设成立，继续 | Phase 2：多算子、多硬件 |
| Arke 正确率高 + 性能差 | ⚠️ Arke 是好的验证框架，但不是好的优化工具 | **Pivot**：Arke 定位为"LLM kernel 生成的验证平台"而非"优化平台" |
| Arke 正确率低 + 性能好 | ⚠️ 不太可能出现（性能好通常意味着代码正确） | 检查验证系统 |
| Arke 正确率低 + 性能差 | ❌ 核心假设不成立 | **Kill** 或根本性 pivot |
| Arke ≈ 直写 Triton | ⚠️ 没有优势 | 审视 Arke 的增量价值（可解释性？可迁移性？） |

### Gate 5：整模型端到端验证（Phase 7）🔴🔴 最终验证

> 单算子 benchmark 可能给出虚假信心。算子快了 20% 不代表模型快了 20%——
> 框架集成开销、内存 layout 转换、kernel launch latency 都会吃掉收益。
> **必须在一个最小化的真实模型上验证端到端收益。**

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| **选定验证模型** | 一个最小但有代表性的模型（见下文） | - |
| **Arke kernel 可替换原生算子** | PyTorch custom op 接口可用 | 修 integration layer |
| **整模型推理正确** | 模型输出与 baseline 数值一致（容差内） | 修 kernel / 修接口 |
| **整模型端到端有性能收益** | 吞吐量或延迟相比 baseline 有可测量提升 | 如果无收益 → 分析瓶颈 |
| **无回退** | 不出现 Arke kernel 导致模型变慢的情况 | 分析 kernel launch 和内存开销 |

#### 验证模型选择原则

```
选择标准：
1. 小到能在 RTX 3060 (6GB) 上跑推理
2. 包含 Arke 优化的目标算子（matmul, softmax, attention）
3. 足够有代表性——不能是 toy model
4. 有公认的 baseline 性能数据
```

**候选模型（按优先级排序）**：

| 模型 | 参数量 | 关键算子 | 为什么选 |
|------|--------|---------|---------|
| GPT-2 Small (124M) | 124M | matmul + softmax + layernorm + GELU | 最经典的 transformer 验证模型，小到 3060 能跑 |
| BERT-base (110M) | 110M | matmul + softmax + layernorm | encoder-only，推理更简单 |
| ViT-B/16 (86M) | 86M | matmul + softmax + patch embedding | CV 方向，验证通用性 |
| Llama-2 7B (量化) | 7B→~3.5GB | matmul + RoPE + RMSNorm + attention | 真实 LLM，但需要 int4 量化才能放进 6GB |

**推荐**：GPT-2 Small 作为主验证模型。理由：
1. 124M 参数，FP16 只需 ~250MB 显存，RTX 3060 轻松
2. 12 层 transformer，每层含 matmul + softmax + layernorm + GELU——覆盖 Arke Phase 1 的所有目标算子
3. HuggingFace 上有标准实现，baseline 性能容易获取
4. 社区关注度高，结果有说服力

#### 端到端验证流程

```
Step 1: Baseline 测量
  标准 PyTorch GPT-2 Small 推理性能（tokens/sec）
  - torch.compile 优化版
  - 原生 eager 版
  记录：throughput, latency_p50, latency_p99, memory_peak

Step 2: 识别热点算子
  Profile GPT-2 推理，找出耗时 Top-5 算子
  通常：linear (matmul) > attention (softmax+matmul) > layernorm > GELU
  确定 Arke 优化哪些算子

Step 3: Arke 算子替换
  用 PyTorch custom op (torch.library) 将 Arke 优化后的 kernel 注入模型
  逐个替换，每替换一个测一次：
    替换 matmul → 测性能
    替换 matmul + softmax → 测性能
    替换 matmul + softmax + fused_attention → 测性能

Step 4: 正确性验证
  - 逐算子：Arke kernel 输出 vs PyTorch 原生输出，allclose(atol, rtol)
  - 整模型：GPT-2 生成文本质量不降（perplexity 对比）

Step 5: 性能对比
  | 配置 | tokens/sec | vs baseline |
  |------|-----------|-------------|
  | PyTorch eager | xxx | 1.0x |
  | torch.compile | xxx | y.yx |
  | Arke (matmul only) | xxx | z.zx |
  | Arke (all ops) | xxx | w.wx |
```

#### 什么算"端到端有收益"？

```
最低标准（必须达到）：
  Arke 优化后的 GPT-2 推理不比 torch.compile 慢

有意义的收益：
  Arke 优化后 ≥ 5% 的端到端 throughput 提升

优秀：
  Arke 优化后 ≥ 15% 的端到端 throughput 提升
```

**关键：如果单算子快了 30% 但整模型只快了 2%，需要分析为什么——可能是 kernel launch overhead、内存拷贝、或者该算子本身在模型中占比不高。这种分析本身就是有价值的输出。**

#### 整模型验证失败的可能原因与应对

| 原因 | 症状 | 应对 |
|------|------|------|
| Kernel launch overhead | 单算子快但整模型慢 | kernel 合并、减少 launch 次数 |
| 内存 layout 转换 | 输入输出需要 contiguous/transpose | 优化 memory layout 策略 |
| 框架集成开销 | custom op 调用成本 > 算子加速 | 优化 Python↔CUDA 边界 |
| 算子在模型中占比低 | matmul 只占 40% 耗时，快 30% 只有 12% 整体提升 | 扩大优化算子覆盖 |
| 优化的 shape 不匹配 | benchmark 用 [1024,512]，实际模型用 [768,768] | 针对模型实际 shape 优化 |

### Gate 6：多硬件验证（Phase 2，有 Ascend 硬件后）

| 验证项 | 标准 | 失败应对 |
|--------|------|----------|
| 同一个 LLM session 能为两种硬件做优化 | LLM 无需特殊训练就能优化 Ascend | 验证 H4 |
| Ascend kernel 正确 + 性能合理 | V1 通过 + ≥ 50% CANN baseline | 修 Ascend 后端 |
| 抽象层不需要大改 | Phase 1 的 HAL 设计基本不变 | 如果大改 → 说明抽象设计有问题 |

---

## 五、当前设计中发现的问题

### 问题 1：缺少 "Zero-LLM Baseline"

当前评估只比较 "Arke+LLM" vs "直写 Triton"。但缺少一个关键基线：

> **"Arke 模板 + 暴力搜索"（不用 LLM，纯枚举）的性能是多少？**

为什么重要：
- 如果 Arke 模板 + 暴力搜索就能达到 70% cuBLAS，而 LLM 只到 75%，那 LLM 的增量价值只有 5% → 不需要 LLM
- 这个基线能帮助区分 "Arke 的框架价值" vs "LLM 的智能价值"

**建议新增**：Group D baseline——对 matmul 的所有 legal_actions 组合做暴力枚举（tiling 候选 × placement 候选），取最优。估计对于 matmul 这种搜索空间不大的算子，枚举是可行的。

### 问题 2：模板 codegen 是瓶颈

当前路径 A 的模板 codegen 意味着：**Arke 能生成的代码只能和模板一样灵活**。

如果 LLM 做出了一个创新的 strategy（比如"对 k 维度做 split-k 并行"），但模板里没有 split-k 的支持，codegen 就会失败。

这意味着 LLM 的搜索空间其实被模板限制了，而不是被 `list_legal_actions()` 限制。

**建议**：
- 在 `list_legal_actions()` 中明确标注每个 action 的 "codegen_support" 状态
- 有模板支持的 action 标记为 `"codegen": "template"`
- 无模板支持的 action 标记为 `"codegen": "llm_gen_only"` 或 `"codegen": "unsupported"`
- LLM 可以选择是否使用 "llm_gen_only" 的动作（实验路径 B）

### 问题 3：matmul 太简单，可能给出虚假信心

matmul 是 GPU 优化中最"已解决"的问题：
- 搜索空间相对小
- 最优策略已知（tile + tensor core + shared memory）
- cuBLAS 已经高度优化

**风险**：Arke 在 matmul 上表现好，但在更复杂的算子上失败。

**建议**：Gate 3（W4）必须同时测试 softmax。Softmax 涉及 online 算法选择、数值稳定性、行级归约——这些比 matmul 的纯 tiling 问题复杂得多。如果 LLM 能通过 Arke 正确实现 online softmax，才是真正有意义的信号。

### 问题 4：缺少"LLM 失败时的降级路径"

当 LLM 的 tool-use 循环在 50 步内没有收敛到好结果，怎么办？

当前设计只有 "budget_warning" 提示。但产品化场景需要：
- **Fallback strategy**：预定义的"已知好"的 strategy（如 matmul 的标准 tiling）
- **Human-in-the-loop**：LLM 做不到的，flag 给人类
- **Best-effort 输出**：在搜索预算内找到的最好结果，即使不满足 target_performance

**建议**：在 ArkeEnv 中增加 `fallback_strategy` 机制——如果 LLM 搜索后性能不如 fallback，输出 fallback 并标记"LLM 未能改进"。

### 问题 5：v2 设计文档之间有术语不一致

| 概念 | v2 overview.md | v2 e2e-design-v2.md | plan-v2.1.md | multi-backend.md |
|------|---------------|---------------------|-------------|-----------------|
| 优化层 | Schedule Tree | Strategy | Strategy IR | Strategy IR |
| 语义层 | Semantic Graph | Semantic | Semantic IR | Semantic IR |
| 优化决策 | ScheduleDirective | decision | decision | decision |

**建议**：统一用 v2.1 的术语——Semantic IR、Strategy IR、decision。在 README 或 glossary 中明确。

### 问题 6：AscendC 模板的可行性未验证

v2.1 的 AscendC 模板是基于公开文档写的骨架代码。但 AscendC 的实际编程有很多隐含约束（DMA 对齐要求、L0/L1 切换开销、Cube Unit 的 shape 限制），这些不一定能从公开文档中完全获取。

**建议**：
- Phase 1 中找到至少一个 AscendC 的开源 matmul 实现作为参考（华为 CANN samples 仓库）
- 用真实代码校准 Arke 的 AscendC 模板和 HW Profile
- 在 Phase 6 做 AscendC 骨架时，如果发现抽象层有问题，立即修正

---

## 六、修正后的验证框架总结

```
Phase 1-2: Gate 0 + Gate 1 (环境 + 表达力)
  "Arke 能描述问题吗？"

Phase 2: Gate 2 (端到端通路)                ← 第一个真正的检查点
  "如果策略是对的，Arke 能生成好代码吗？"

Phase 4: Gate 3 (LLM 可行性)                ← 最关键的检查点
  "LLM 能通过 Arke 做有效的优化吗？"
  同时测试 matmul + softmax

Week 5-6: Gate 4 (对比优势)                ← 生死判断
  "Arke 比直写 Triton 更好吗？"
  新增 Group D: Arke 暴力搜索 baseline

Week 7-8: Gate 5 (整模型端到端)            ← 最终验证
  "单算子优势能转化为整模型收益吗？"
  GPT-2 Small 端到端推理性能对比

Phase 2: Gate 6 (多硬件)
  "Arke 的抽象能跨硬件吗？"
```

---

## 七、需要在 Plan 中新增的任务

| 新增任务 | Week | 理由 |
|----------|:----:|------|
| Zero-LLM baseline（暴力搜索） | W6 | 区分框架价值 vs LLM 价值 |
| legal_actions 标注 codegen_support | W2 | LLM 知道哪些动作有模板支持 |
| softmax 同步测试 | W4 | matmul 太简单，需更复杂算子 |
| Fallback strategy 机制 | W4 | LLM 失败时的降级路径 |
| AscendC 开源样例收集 | W1 | 校准 Ascend 抽象层 |
| 术语统一 + glossary | W1 | 消除文档歧义 |

---

## 八、一句话总结

**设计整体合理，但缺少"证伪自己"的勇气。**

当前 plan 的隐含假设是"只要按计划做就能成功"。真正 LLM-Native 的项目需要接受的是：**核心假设可能不成立**。每个 Gate 不只是"功能交付检查点"，更是"假设验证检查点"。在 Gate 4 如果数据说 Arke 不比直写 Triton 好，要有勇气承认并 pivot。

在此基础上的六项补充：
1. 新增 Zero-LLM 暴力搜索基线
2. 模板 codegen 瓶颈的显式标注
3. softmax 提前到 Gate 3 同步验证
4. LLM 失败时的 fallback 降级
5. AscendC 实际代码校准
6. 文档术语统一

---

*审视版本：v1.0 | 创建日期：2026-03-31*