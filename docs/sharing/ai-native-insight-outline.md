# 洞察分享 PPT 大纲（分 3 个大部分：洞察背景 → 案例趋势与难题 → Arke 技术构建与 TODO）

> 说明：本文为 **PPT 大纲**（不是逐页内容），用于确认叙事结构与要点覆盖范围。

---

## Part A｜背景：为什么需要“这份洞察”（Why this insight now）

### 1) 封面与目标
- 本次分享要回答的 3 个问题：**为什么现在、业界怎么走、我们怎么构建**

### 2) 变化的三重驱动
- **硬件**：SIMT→SIMD→多类 NPU，代际与异构叠加  
- **工作负载**：Attention/MLA/GQA/MoE、融合爆炸、动态 shape 常态化  
- **工程现实**：kernel 数量爆炸、专家稀缺、验证成本与回归成本上升  

### 3) 传统路径的失效点（对照基线）
- **人工手写/调优**的不可扩展性  
- 仅靠 **autotune/heuristics** 的上限与泛化问题  
- “**LLM 直写 Triton/CUDA**”的正确性、token 成本、可维护性问题  

### 4) 洞察的输出形态
- 用“案例证据”抽象出：**趋势 → 进展 → 难题 → 构建方案（可落地）**

---

## Part B｜业界趋势：从案例出发看“LLM-driven kernel generation/optimization”走向何处

### 1) 案例导读：7 个代表方案覆盖了哪些路线
- KernelEvolve / KernelAgent / AutoKernel / K-Search / AVO / CuTeGen / KernelGen-LM（各自定位一句话）

### 2) 案例 1：KernelEvolve（Meta）
- 代表趋势：**生产级异构 + RAG 硬件知识库 + 搜索式优化**

### 3) 案例 2：KernelAgent / KernelFalcon（PyTorch/Meta）
- 代表趋势：**Deep Agents 分层编排 + 硬件信号（NCU/roofline）驱动 + 严格验证门禁**

### 4) 案例 3：AutoKernel（RightNow AI）
- 代表趋势：**autoresearch 循环 + Amdahl 优先级 + 双后端（Triton/CUDA）**

### 5) 案例 4：K-Search（UC Berkeley）
- 代表趋势：**World-Model 规划式搜索（策略/实现解耦）+ 抗非单调优化路径**

### 6) 案例 5：AVO（NVIDIA）
- 代表趋势：**Agent-as-Variation-Operator（把“变异算子”升级为自主 agent）+ 长周期演化**

### 7) 案例 6：CuTeGen（U. Toronto）
- 代表趋势：**选“稳定抽象层”作为目标语言（CuTe）+ 单 kernel 渐进精炼（低 token）**

### 8) 案例 7：KernelGen-LM / AscendKernelGen（PCL）
- 代表趋势：**领域数据/领域模型（SFT+RLEF）补齐专用 DSL 数据稀缺**

### 9) 趋势归纳：业界正在收敛的 5 个方向（从案例抽象）
- **搜索化（非 one-shot）**
- **工具化（tool-use + harness + 可复现评测）**
- **硬件信号化（profile/roofline 进入 loop）**
- **抽象层选择（Triton/CuTe/DSL，稳定可迭代）**
- **知识注入与沉淀（RAG/playbook/后训练/轨迹）**

### 10) 已有的“好进展”（What’s working）
- 正确率门禁、并行探索、硬件反馈、生产部署、覆盖复杂 kernel 等（按 5–7 条列）

### 11) 仍待解决的“关键技术难题”（What’s hard）
> 统一列出并编号（建议 8–12 条），例如：

- **H1**：如何把“策略”从自由代码中抽出来变成可迁移资产？  
- **H2**：如何把验证从事后对拍变成逐步可剪枝的多级门禁（V0/V1/V2）？  
- **H3**：动态 shape/符号维度下的泛化与性能不退化  
- **H4**：跨硬件迁移（NVIDIA→Ascend/AMD）的知识与策略复用  
- **H5**：从单 kernel 到模型级自治优化（bottleneck→再优化→回归）  
- **H6**：Token/预算治理与长会话稳定性  
- **H7**：后端天花板（Triton→MLIR/LLVM）与更深层决策（寄存器/屏障/调度）  

### 12) 结论页：为什么“需要一套更体系化的编译栈”，而不是再堆一个 agent
- 用一句话把 Part B 引到 Part C

---

## Part C｜Arke 技术构建：如何承接进展、正面解决难题（含 TODO）

### 1) Arke 定位与总体架构（1 张总图）
- 用户入口 → Language → 多层 IR → Agent/Compiler 协议 → Backend → HW

### 2) 四件套总览：Language / IR / Compiler Toolchain / Agent Engineering
- 每件套一句话职责 + 与 Part B 趋势的对应关系

### 3) 逐条对齐：把 Part B 的“好进展”映射到 Arke 的设计
- 例如：确定性编排、benchmark/gate、硬件 profile、回滚等（映射表）

### 4) 逐条对齐：把 Part B 的“关键难题 H1–H7”映射到 Arke 的解决方案
- **H1** → Semantic/Strategy 分离 + Strategy IR + `@rationale`（知识资产化）  
- **H2** → V0/V1/V2 三级验证 + checkpoint/rollback  
- **H3** → where + symbolic dimension system + conditional strategy  
- **H4** → target-aware strategy + rationale 迁移  
  - **TODO**：跨架构 lift 的量化验证与 KB 检索体系  
- **H5** → KernelCache + benchmark BL6  
  - **TODO**：模型级瓶颈定位 → 自动回灌优化的闭环产品化  
- **H6** → budget + segmented prompt cache + compact  
  - **TODO**：跨 session 的长期记忆/KB 治理  
- **H7** → Triton→MLIR→LLVM 路线  
  - **TODO**：L2/L3 决策空间的动作枚举完整性  

### 5) Arke 路线图（Phase/Stage/Gate）如何保证“趋势落地”
- 用 Gate/Benchmark 解释为什么能持续收敛（避免只做 demo）

### 6) TODO 清单（显式标记未覆盖点）
- 按优先级列出：缺什么、为什么重要、建议的验证方式/里程碑

### 7) 讨论题 / Q&A
- 3–5 个研讨引导问题（有界动作空间边界、知识沉淀介质、跨硬件迁移等）

