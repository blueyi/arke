# Arke 技术洞察 PPT 大纲（聚焦"技术发展趋势")

> **定位**：本 PPT 是一次**技术趋势洞察**，不是 Arke 项目汇报。叙事主轴是"业界正在发生什么 → 难题在哪 → Arke 如何承接"，Arke 作为最后一章的"答卷"出现。
>
> **结构**：三大部分
> 1. **背景**：为什么需要这次洞察（模型能力 + 业界现状 + 传统路径失效）
> 2. **趋势 + 案例**：用 7 个代表性 LLM-driven kernel 案例，归纳趋势、提炼"已有进展"与"待解难题"
> 3. **Arke 的技术构建**：把进展和难题逐条映射到 Arke 的方案；未覆盖部分**显式标记 TODO**
>
> **使用方式**：先确认本大纲，确认后再生成各章节正文与最终 PPT。后续大纲调整也通过本 md 文件承载。
>
> **上次修订**：2026-04-26（聚焦"技术趋势洞察"叙事，强化模型能力角度和 TODO 标记）

---

## Part A｜背景：为什么现在需要这份洞察（Why this insight, why now）

### A1. 封面与目标
- 主题一句话：**"AI-Native 时代，算子工程的范式正在被重写"**
- 本次分享要回答的 3 个问题：
  1. 模型变强了，到底**强到能做什么**？
  2. 业界相关方向**已经走到哪、卡在哪**？
  3. 我们**应当如何构建**才能搭上这一波趋势？

### A2. 模型能力侧：LLM 能力跃迁带来的"新可能"
> 这一节回答"为什么以前做不了，今天可以做"。

- **代码理解与生成**：从片段补全 → 跨文件、跨抽象层的程序合成
- **结构化推理**：可被显式约束的"决策"型推理（tool-use / function-calling / structured output）
- **长上下文 + RAG**：把硬件手册、ISA、profile 报告、历史轨迹纳入决策上下文
- **可验证性闭环**：模型不再是"一次性写出"，而是与验证器/编译器/profiler 形成多轮闭环
- **多模态与代理化**：从单次回答 → Agent 编排（规划/执行/反思/回滚）
- **后训练范式**：SFT / RLHF / RLEF（execution feedback）/ trajectory replay 让"经验"变成可累计资产

> 小结：**模型从"会写代码"升级为"会做带反馈的工程决策"**——这是后续所有趋势的能力底座。

### A3. 问题域侧：算子工程的业界现状（Arke 要解决的问题域）
> 这一节回答"为什么这个领域亟需被 LLM 重塑"。

- **硬件维度**：SIMT（NVIDIA）→ SIMD（Ascend）→ 多类 NPU/DSA，**代际叠加 + 异构叠加**
- **工作负载维度**：Attention/MLA/GQA/MoE、融合爆炸、动态 shape 常态化、长上下文导致的 KV/Paged 模式
- **工程现实维度**：
  - kernel SKU 数爆炸（算子 × 形状 × 精度 × 后端 × 配置）
  - 顶级 kernel 工程师**稀缺且不可扩展**
  - 验证成本与回归成本随版本上升
  - 厂商库（cuBLAS/cuDNN/CUTLASS/FlagGems/Liger）覆盖不全且演进不同步
- **量化的业界现状**：用 1 张图同时呈现"模型规模/算子复杂度/硬件代际"三条曲线的发散

### A4. 传统路径为什么不够用（Failure modes of the status quo）
- **手写 + autotune** —— 不可扩展、跨硬件不可迁移、知识沉淀依赖个人
- **图编译器（XLA / Inductor / TVM）** —— 高层有效，但深层决策（tile / pipeline / 寄存器分配）仍依赖启发式或专家
- **"LLM 直写 Triton/CUDA"** —— 正确性差、token 成本高、可维护性差、知识无法跨 kernel 复用
- **结论**：单点工具不再够用，**"能力底座 + 问题域复杂度"**共同迫使我们重新思考算子工程栈

### A5. 这份洞察的输出形态
- 用"案例 → 趋势 → 进展 → 难题 → 构建方案"五段式给出**可落地的范式判断**
- 不只是"看热闹"，每条趋势都要回答："这个工程上**该不该跟、怎么跟**？"

---

## Part B｜业界趋势：从 LLM-driven kernel generation/optimization 看技术走向

> **本部分以"趋势"为主线，案例只是证据**。每个案例只用 1 页，落点在"它代表了什么趋势"，避免变成案例巡礼。

### B1. 总图：LLM-driven kernel 工作链路全景
- 一张全景图：**Spec/IR ↔ LLM Agent ↔ Toolchain（compile/profile/verify）↔ Hardware**
- 标出 7 个案例分别落在链路上的哪个位置（决策层 / 搜索层 / 数据层 / 后端层）

### B2. 7 个代表案例（一页一个，落点是"代表趋势"）

| # | 案例 | 来源 | 一句话定位 | 代表趋势 |
|---|------|------|-----------|----------|
| 1 | **KernelEvolve** | Meta | 生产级异构 + RAG 硬件知识库 + 搜索式优化 | T1 / T3 / T4 |
| 2 | **KernelAgent / KernelFalcon** | PyTorch / Meta | Deep Agents 分层编排 + 硬件信号驱动 + 严格门禁 | T2 / T3 |
| 3 | **AutoKernel** | RightNow AI | autoresearch 循环 + Amdahl 优先级 + 双后端（Triton/CUDA） | T1 / T2 |
| 4 | **K-Search** | UC Berkeley | World-Model 规划式搜索（策略/实现解耦）+ 抗非单调路径 | T1 / T5 |
| 5 | **AVO** | NVIDIA | Agent-as-Variation-Operator，把"变异算子"升级为自主 agent | T1 / T2 |
| 6 | **CuTeGen** | U. Toronto | 选稳定抽象层（CuTe）+ 单 kernel 渐进精炼 + 低 token | T4 |
| 7 | **KernelGen-LM / AscendKernelGen** | PCL | 领域数据 + 领域模型（SFT+RLEF）补齐 DSL 数据稀缺 | T6 |

> 每页固定模板：**做了什么 / 关键技术点 / 它代表的趋势 / 它没解决什么**

### B3. 趋势归纳：业界正在收敛的 6 个方向（核心页）
> 这是 Part B 的"灵魂页"，承上启下。

- **T1 · 搜索化**：从 one-shot 生成 → 演化/规划/MCTS 等结构化搜索
- **T2 · 工具化**：tool-use + harness + 可复现评测 = 把 LLM 关进编译器/profiler 的"控制室"
- **T3 · 硬件信号化**：profile / roofline / NCU 反馈进入 LLM 决策循环
- **T4 · 抽象层选择**：放弃直写 PTX，押注**稳定且可迭代**的抽象层（Triton / CuTe / DSL / IR）
- **T5 · 知识资产化**：RAG / playbook / @rationale / 轨迹回放 → 经验从"个人手感"变"可检索资产"
- **T6 · 领域模型化**：SFT + RLEF + 后训练，让小/中模型在 kernel 域具备专家级判断

### B4. 已有的"好进展"（What's working）
> 客观陈述案例已经验证可行的部分，作为后续 Part C 对齐的输入。

- **G1**：正确性门禁（numerical equivalence）已成为业界标准做法
- **G2**：并行探索 + 早停 + 成本预算已被多个工作验证有效
- **G3**：硬件 profile 反馈进入决策循环，能稳定带来双位数性能提升
- **G4**：Triton/CuTe 等"稳定抽象层"上的 LLM 生成已能落地生产
- **G5**：RAG + 知识库 + 后训练让"专家直觉"开始可累计、可迁移
- **G6**：Agent 编排（分层、规划、回滚）让长会话稳定性显著改善

### B5. 仍待解决的"关键技术难题"（What's hard）
> 用统一编号，便于 Part C 一一对齐。

- **H1 · 策略可迁移性**：如何把"策略"从自由代码中抽出来变成可迁移、可检索的资产？
- **H2 · 多级验证**：如何把验证从"事后对拍"变成**逐步可剪枝的多级门禁**（V0 静态 / V1 数值 / V2 性能）？
- **H3 · 动态 shape 与符号维度**：动态/符号 shape 下如何同时保证泛化与不退化？
- **H4 · 跨硬件迁移**：知识与策略如何从 NVIDIA 迁移到 Ascend / AMD / 其他 NPU？
- **H5 · 模型级自治优化**：从单 kernel 到模型级（bottleneck → 再优化 → 回归）的端到端闭环
- **H6 · Token 与预算治理**：长会话中 token / 预算 / 上下文的稳定性与可治理性
- **H7 · 后端天花板**：Triton 抽象层封顶 → 何时下沉到 MLIR / LLVM 才能解锁更深决策（寄存器 / 屏障 / 调度）
- **H8 · 数据稀缺与领域模型**：高质量 kernel × HW × profile 三元组数据稀缺，专用模型如何冷启动
- **H9 · 评测与可复现**：缺乏统一的 benchmark / 形状层级 / 基线协议，跨工作不可比

### B6. 小结页：为什么"需要一套体系化的编译栈"，而不是再堆一个 Agent
- 一句话过渡到 Part C：**Agent 是手段，IR + 验证 + 编译栈才是底盘**

---

## Part C｜Arke 的技术构建：承接进展、正面解决难题（含 TODO）

> 本部分**严格回应 Part B 的 G1–G6 与 H1–H9**，每一条都要给出 Arke 的对应方案；没有覆盖的部分**显式 TODO**。

### C1. Arke 定位与总图
- 一张总图：**`.ak` 语言 → 多层 IR（Semantic/Strategy/Schedule/Instruction）→ Agent 协议 → Compiler/Verifier → Backend → HW**
- 一句话总纲：**"让 LLM 写 kernel，让编译器把关数学；用 IR 把策略变成可迁移资产"**

### C2. 四件套总览（Language / IR / Compiler Toolchain / Agent Engineering）
- 每件套一句话职责
- **核心理念表**（与 Part B 趋势一一对应）：
  - 语义/策略分离 ↔ T1/T5
  - Bounded Action Space ↔ T2
  - 多层 IR + 多级验证 ↔ T2/T4
  - @rationale 一等公民 ↔ T5
  - 多后端可插拔 ↔ T4
  - Agent 协议（tool-use 闭环）↔ T2/T3

### C3. 把"好进展（G1–G6）"映射到 Arke 的设计

| Part B 进展 | Arke 中的体现 |
|------|------|
| G1 正确性门禁 | V0 静态（<1ms）→ V1 数值 → V2 性能 三级流水 |
| G2 并行探索 + 预算 | checkpoint/rollback + budget governance + 段式 prompt cache |
| G3 硬件信号 | `compile_and_profile` + roofline-aware decisions |
| G4 稳定抽象层 | Phase 1 Triton → Phase 3 MLIR → Phase 4 LLVM 渐进路线 |
| G5 知识资产 | `@rationale` 一等公民 + Strategy IR + KernelCache + 轨迹导出 |
| G6 Agent 编排 | Mode A/B 双模 + 工具协议 + nudge/compact/stop 机制 |

### C4. 把"难题（H1–H9）"映射到 Arke 的方案 + 显式 TODO

> 这是 Part C 的核心页。每条难题给出"Arke 怎么做"+"还差什么（TODO）"。

- **H1 · 策略可迁移** → ✅ Semantic/Strategy 分离 + Strategy IR + `@rationale`
  - **TODO-1**：跨 kernel 的 *Rationale Knowledge Base* 检索/合成尚未产品化（计划在 P4-S_FINAL 累计 ≥200 条）
- **H2 · 多级验证** → ✅ V0/V1/V2 + checkpoint/rollback + Compiler-as-Verifier
  - **TODO-2**：V2 性能门禁的"可剪枝早停"策略尚未形式化（当前依赖 budget，缺少基于增益梯度的主动剪枝）
- **H3 · 动态 / 符号 shape** → ✅ `where` 子句 + Symbolic Dimension System + conditional strategy（v2.0 已落地设计）
  - **TODO-3**：动态 shape 在 L2 fused / L3 模型级的端到端泛化验证尚未完成（S7 feasibility 文档 → S8 落地）
- **H4 · 跨硬件迁移** → ✅ target-aware strategy + 后端可插拔（Phase 2 Ascend）
  - **TODO-4**：跨架构 *strategy lift*（NVIDIA→Ascend→AMD）的量化验证 + KB 检索体系尚未建立
  - **TODO-5**：`@rationale` 跨架构有效性（≥10% lift）目前是 Gate P2-S4 目标，**尚未达成**
- **H5 · 模型级自治** → ⚠️ 部分覆盖：KernelCache + benchmark BL6 + Inductor 集成（S8）
  - **TODO-6**：模型级"瓶颈定位 → 自动回灌优化 → 回归"的**闭环产品化**尚未完成
  - **TODO-7**：跨 kernel 的全局优化（共享内存/流/调度协同）尚未纳入决策空间
- **H6 · Token / 预算治理** → ✅ budget tracker + segmented prompt cache + compact + observe(delta)
  - **TODO-8**：跨 session 的**长期记忆 / 知识库治理**（去重 / 失效 / 版本）尚未设计
- **H7 · 后端天花板** → ✅ 路线已定：Triton (P1) → MLIR (P3) → LLVM (P4)
  - **TODO-9**：Strategy IR Level 2 (loop_nest / memory_access_pattern) 与 Level 3 (register / barrier / instr_sched) 的**完整动作枚举**尚未实现
  - **TODO-10**：MLIR Dialect (`arke.kernel` / `arke.strategy`) 仅有骨架，尚未端到端打通
- **H8 · 领域模型与数据** → ⚠️ 当前未覆盖
  - **TODO-11**：领域模型路线（SFT / RLEF / 轨迹回放）尚未纳入 Arke 路线图，需要决策"自研 or 复用 KernelGen-LM 类外部模型"
  - **TODO-12**：Arke 自身的轨迹/rationale 数据如何形成训练集，尚无 schema 与采集流水
- **H9 · 评测与可复现** → ✅ BL/OT/ST/L benchmark 框架 + Gate 治理 + 多 baseline (P0–P5)
  - **TODO-13**：与业界（KernelEvolve / KernelAgent 等）的**横向 benchmark 对齐**尚未做（缺统一 reporting schema）

### C5. Arke 路线图（Phase / Stage / Gate）如何保证趋势"持续落地"
- 用 Gate 体系解释：每个 Stage 的 Gate 是"业界趋势 → Arke 验证"的落点
- 关键里程碑映射：
  - **G6（当前）**：Compiler Infrastructure，对应 T4（稳定抽象层）+ T2（工具化）
  - **G7**：Lang & IR v2，对应 T5（知识资产化）+ H3（符号 shape）
  - **G8**：Agent Autonomy，对应 T1/T2/T3，集中回应 H5/H6
  - **G9**：Phase 1 Final，4 模型 E2E + Arke vs LLM-direct，对应 H9（评测）
  - **Phase 2**：Ascend 验证，集中回应 H4
  - **Phase 3/4**：MLIR / LLVM，集中回应 H7

### C6. TODO 总览（按优先级 / 责任 Stage）

| TODO | 关联难题 | 优先级 | 建议落地 Stage / Phase |
|------|---------|------|----------|
| TODO-1 Rationale KB 产品化 | H1 | P1 | P4-S_FINAL（≥200 条目标） |
| TODO-2 V2 早停剪枝 | H2 | P2 | S8（Agent Autonomy） |
| TODO-3 动态 shape 端到端 | H3 | P1 | S7→S8 |
| TODO-4 跨架构 strategy lift 量化 | H4 | P1 | Phase 2 |
| TODO-5 @rationale 跨架构 ≥10% lift | H4 | P1 | P2-S4 |
| TODO-6 模型级闭环产品化 | H5 | P2 | S8/S9 |
| TODO-7 跨 kernel 全局优化 | H5 | P3 | Phase 3+ |
| TODO-8 跨 session 长期记忆 | H6 | P2 | Phase 3+ |
| TODO-9 L2/L3 决策动作完整枚举 | H7 | P1 | Phase 3 |
| TODO-10 MLIR Dialect 端到端 | H7 | P1 | Phase 3 |
| TODO-11 领域模型路线决策 | H8 | P2 | Phase 2/3 |
| TODO-12 训练集 schema + 采集流水 | H8 | P2 | Phase 2 |
| TODO-13 与业界横向 benchmark 对齐 | H9 | P1 | S9 |

### C7. 讨论题 / Q&A
- 3–5 个研讨引导问题，例如：
  - 有界动作空间的"边界"应当由谁定义？编译器枚举 vs 模型外推
  - `@rationale` 是终极的"知识沉淀介质"吗？还是只是过渡形态？
  - 自研领域模型 vs 复用通用前沿模型，分界点在哪里？
  - 跨硬件策略迁移：是 IR 层的事，还是模型层的事？
  - 评测的"业界横向可比"由谁主导？是否应当推动一个开放协议？

---

## 附录（Appendix，PPT 中放在最后或备用页）

- 附录 A：术语表（SemanticIR / StrategyIR / `@rationale` / V0–V2 / BL/OT/ST/L 等）
- 附录 B：7 个案例的关键参考链接 / paper / repo
- 附录 C：Arke 当前 Stage 状态速查（S0–S5 ✅，S6 进行中，S7–S9 ⬜）
- 附录 D：与 README 中"Key Features"的逐项映射表

---

## 修订记录

| 日期 | 版本 | 主要变化 |
|------|------|---------|
| 2026-04-25 | v0.1 | 初版大纲，三段式结构 |
| 2026-04-26 | v0.2 | 聚焦"技术趋势洞察"叙事；A 部分新增"模型能力侧"专章；B 部分由案例巡礼改为"趋势 + 案例证据"；C 部分新增 H8/H9，TODO 列表化并按优先级排序 |
