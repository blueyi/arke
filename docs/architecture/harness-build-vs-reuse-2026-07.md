# Arke Harness: Build vs Reuse — 2026-07 深度调研决策文档

> **触发:** Leon (2026-07-12): "关于 Harness 工程的方案,需要结合最新的业界演进深度调研之后确定,是 Arke 项目自研,还是使用业界已经有的...如果可以考虑复用 hermes 或者 openclaw 或者其他 agent 工程系统,也请详细对比"
>
> **作者:** Kitty (lead engineer)
> **状态:** 📋 调研完成 + 推荐,待 Leon 决策
> **前置文档:** `arke-harness-v2-rfc.md` (2026-06-26 首轮对比) · 本文档结合 2026 最新业界演进重做战略层分析

---

## 0. TL;DR — 推荐

**保留 Arke 自研 Harness 的领域核心层,通过 MCP 暴露给外部通用框架,不做整体替换。**

调研结论有一个清晰的分水岭:业界 2025-2026 的 SOTA kernel-gen 系统里——

- **复用通用框架的** (CUDA Agent→OpenHands, Astra→OpenAI Agents SDK) 都是把 domain 逻辑塞进 general coding-agent 的 Bash/Read/Write/Edit 工具抽象,**放弃了 bounded-action-space 和 staged-verification 的一等公民地位**。
- **做深度优化、拿到最高性能的** (AutoKernel 9.2k LOC custom, Sakana custom evolutionary, GEAK custom Reflexion) **全部自研 harness**。

Arke 的三个硬需求(compiler-computed bounded action space / V0-V1-V2 staged verification / trajectory-as-RL-artifact)恰好是通用框架**都不原生提供**的部分——这三点正是 Arke 论文命题的核心差异化。**整体替换会丢掉命题;整体自研已经做完了(~5800 LOC, G8 6/6 PASS)。正确答案是分层。**

---

## 1. 业界最新演进 (2025-2026)

### 1.1 Kernel-gen agent 系统 (领域内)

| System | 核心循环 | Harness 抽象 | 框架:复用 vs 自研 | 验证信号 | 训练? |
|---|---|---|---|---|---|
| **CUDA Agent** (ByteDance/清华) | ReAct 多轮: profile→write→compile→refine | Skill-augmented env; `SKILL.md` + 固定 workspace + 6000 任务数据合成 | **复用** OpenHands; Claude-Code 式工具集 | 离散 robust reward {−1,1,2,3} vs eager & torch.compile | **是 — agentic RL (PPO+RFT)** |
| **Astra** (Stanford) | 4-agent 固定 R=5 轮: plan→code→test→profile | 4 个专职 agent; 从 SGLang 生产 kernel 出发 | **复用** OpenAI Agents SDK, o4-mini | test-suite (≤ε) + geomean speedup | 否 (推理时) |
| **AutoKernel** (RightNow AI) | 单 agent keep/revert; 改 1 文件→bench→git keep/reset | 3 阶段; Amdahl 排序瓶颈; 909 行 playbook; git+TSV | **自研** (~9.2k LOC); 明确拒绝 multi-agent | 5 阶段 correctness gate 再测 perf | 否 |
| **Sakana AI CUDA Engineer** | 2 阶段: translate → 进化式 meta-generation (G 代, N=8) | `robust-kbench` (fwd+bwd+fusion); LLM ensemble | **自研** 进化 pipeline | LLM soft-verifier (~80%) 预筛 → 硬件测试 | 否 (test-time 进化) |
| **GEAK** (AMD) | Reflexion: Generate→Evaluate→Reflect→Optimize | 模块化; 1-shot 检索 + 知识注入; seq+parallel scaling | **自研** 基于 Reflexion 范式 | 单测 pass/fail + speedup, error 喂 Reflector | 否 |

**跨系统共性(五个系统全部遵守):**
1. **验证防火墙普遍存在** — 每个系统都把可变的 code generator 与不可变的 evaluator 隔离,大多显式加固防 reward hacking。
2. **correctness-before-performance gating** — perf 只在 correctness 通过后才测(GEAK/Astra 单测,AutoKernel 5 阶段,Sakana soft-verify,CUDA Agent 离散 reward 层级)。
3. **只有 CUDA Agent 训练策略** — 其余四个都是围绕 frozen frontier LLM 的 test-time-compute,创新在 search strategy(进化/keep-revert/multi-agent/Reflexion)而非模型。
4. **single vs multi-agent 有争议** — Astra 主张 multi-agent 随复杂度提升有益(1.32× vs 1.08×);AutoKernel 主张相反,单 agent 紧循环避免协调开销。

**对 Arke 的直接启示:**
- Arke 的 **V0/V1/V2 staged verification + bounded action space** 与这些系统的"验证防火墙 + correctness-first gating"完全同构——**Arke 早在 Stage 7/8 就把它做成一等公民,而业界很多系统是事后打补丁。**
- **CUDA Agent 的 RL 路线**是 Arke trajectory-as-learning-artifact 的天然下游:Arke 已有 trajectory v1.0 schema + rationale_kb,正是 RL SFT 语料的来源。这是 Arke 相对 test-time-only 系统的结构性优势。
- **Astra 复用 OpenAI Agents SDK** 证明通用框架能跑起来,但它放弃了 bounded action space(直接让 LLM 写 CUDA 文件)——这正是 Arke 不该退让的地方。

### 1.2 通用 agent 框架 (可复用 runtime)

| 维度 | OpenAI Agents SDK | Claude Agent SDK | LangGraph | MS Agent Framework | Google ADK | **Hermes** | **OpenClaw** |
|---|---|---|---|---|---|---|---|
| 核心原语 | Runner tool-loop + handoffs | Claude Code 引擎作库 | State graph + checkpoints | Actor→GraphFlow | Workflow+LLM agents | tool-loop + cron/deleg | gateway + approvals |
| 状态/checkpoint | Sessions (基础) | 自动 compaction | **业界最强(每步 checkpoint)** | session+telemetry | session/state svc | session DB | session |
| 可恢复/time-travel | 有限 | session resume | **全回放+fork+time-travel** | durable | resumable | session DB resume | resume |
| bounded action space | Guardrails | Hooks+permissions | 条件边(结构化) | middleware | **workflow agents(原生)** | approvals | approvals.mode |
| trajectory-as-artifact | 弱 | 弱 | **原生一等公民** | 一般 | 有 persistence | trajectory 非核心 | — |
| 嵌入为库? | 是 | 是(包 binary) | 是 | 是 | 是 | Hermes 是产品 | 是产品 |
| License | MIT | SDK MIT/用受 Anthropic ToS | core MIT / server Elastic 2.0 | MIT | **Apache 2.0** | 专有(自用) | 专有 |

**三个硬需求的 reuse-fit:**
1. **bounded action space (compiler-computed)** — 无框架原生提供,本质是 domain 逻辑。最佳复用 = 验证 hook + MCP 暴露"legal moves"工具。LangGraph 条件边 / ADK workflow agents 结构最贴合。
2. **staged verification V0/V1/V2** — 确定性多阶段 gate,正是 workflow/graph runtime 的用武之地。LangGraph(条件边失败回退)/ ADK(SequentialAgent 包 LoopAgent)直接表达。**但 verifier 本身永远是 custom。**
3. **trajectory-as-learning-artifact** — **唯一区分点**。只有 LangGraph 把它做成原生 load-bearing 特性(checkpointer + time-travel replay/fork)。其余都要自建持久层——**而 Arke 已经自建好了(trajectory v1.0 + rationale_kb)。**

---

## 2. 为什么不整体替换

### 2.1 沉没成本 vs 迁移成本
- Arke 自研 Harness 已 ~5800 LOC(arke/agent/ + arke/learn/),Façade v1.0 **frozen + 158 契约测试**,G8 **6/6 PASS**。这不是原型,是通过 Gate 的生产件。
- 迁移到任何通用框架都要:重写 8 tools 为该框架的 tool 抽象、重接 V0/V1/V2、把 trajectory schema 映射到框架的 state——**而这三样正是通用框架不原生支持、必须 custom 的部分**。迁移换来的只是通用 runtime 层(loop/fallback/compaction),而这些 Arke 的 v2 RFC 已借鉴四大 harness 补齐了(S1-S5, N1-N6 大部分 LANDED)。

### 2.2 命题风险
Arke 的论文命题是"为 Agent 从第一性原理设计 AI-Native 的 Language/IR/**Compiler/Harness**"。Harness 是命题的一等组成部分,不是可外包的脚手架。整体复用 = 承认 Harness 不是贡献点,直接削弱命题。Astra 的路径(复用 SDK + 让 LLM 裸写 CUDA)对 Arke 是命题倒退。

### 2.3 hermes / openclaw 具体评估
- **Hermes**:成熟 tool-loop + cron/delegation + session DB + provider fallback,是极好的**通用** agent 产品,但它是**产品不是嵌入库**,且 trajectory 非其核心;Arke 已在 v2 RFC 借鉴其 fallback/subagent 设计。**结论:借鉴设计模式(已做),不整体依赖。**
- **OpenClaw**:gateway resilience + approvals 降级是亮点(Arke heuristic floor 的类比),同样是产品形态。**结论:借鉴 degrade 模式,不整体依赖。**
- 两者都不解决 Arke 的三个 domain 硬需求,复用它们等于用重型通用产品包一层,反而增加耦合。

---

## 3. 推荐架构 — 分层复用

```
┌─────────────────────────────────────────────────────────┐
│  外部通用 Agent 框架 (可选, 通过 MCP 接入)                  │
│  Hermes / Claude Code / LangGraph / 任意 MCP client       │
└───────────────────────────┬─────────────────────────────┘
                            │ MCP (Nov-2025 spec, 企业级 auth/transport)
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Arke Harness Façade v1.0 (frozen, 自研核心 — 不替换)      │
│  8 tools · 9-kind event stream · trajectory v1.0          │
│  ── 这三样是 domain 一等公民, 通用框架都不原生提供 ──        │
│  1. bounded action space (compiler/HW-computed legal)     │
│  2. V0/V1/V2 staged verification (syntax/numeric/perf)    │
│  3. trajectory-as-RL-artifact (SFT/RL 语料源)             │
└───────────────────────────┬─────────────────────────────┘
                            ▼
        Arke IR (SemanticIR/StrategyIR) + Compiler backends
        (MLIR GPU 1.03× / CUDA-C 1.05× cuBLAS)
```

**具体动作(A 线 = Harness 集成):**
1. **保留** Arke 自研 Façade 核心(bounded actions + V0/V1/V2 + trajectory)——已 frozen,不动。
2. **强化 MCP server (N3, 已 LANDED)** 作为对外主接口——任何通用框架(含 Hermes)都能通过 MCP 驱动 Arke,而不是 Arke 依赖它们。这实现了"可复用性"而不牺牲 domain 核心。
3. **借鉴而非依赖** 四大 harness + CUDA Agent 的具体机制:
   - CUDA Agent 的**离散 robust reward {−1,1,2,3}** → 加固 Arke 的 V2 perf 反馈防 reward-hacking(当前是连续 baseline_ratio)。
   - AutoKernel 的 **5 阶段 correctness gate + benchmark 防火墙** → 强化 V1(当前单次数值比对 → 多形状/稳定性/determinism/edge)。
   - Sakana 的 **LLM soft-verifier 预筛** → V1 前加一层廉价 LLM 筛,省 GPU 编译预算(6GB 机器关键)。
   - GEAK 的 **Reflexion error-trace 回喂** → Arke 已有 V0/V1/V2 错误可恢复,补 error→reflect 回路。
4. **接通 CUDA-C backend 到 Harness**(Session 3-4 已打通 StrategyIR→CUDA-C)——让 Agent 通过 Façade 的 `apply_decision(algorithm='tensor_core')` 驱动 CUDA-C tile/TC 选择,`compile_and_profile` 用 `CudaCBackend.benchmark()`。这是 A 线的核心交付。

---

## 4. 决策点 (待 Leon)

- **D1 — 战略方向:** (a) 保留自研核心 + MCP 对外 + 选择性借鉴(**我的推荐**); (b) 整体迁移到某通用框架(LangGraph 最贴合但要重写三大 domain 层 + 命题风险); (c) 维持纯自研不接 MCP 外部框架。
- **D2 — 借鉴优先级:** 是否按 CUDA Agent robust-reward → AutoKernel 5 阶段 gate → Sakana soft-verify → GEAK reflexion 的顺序增强验证层?(我的推荐: 是, robust-reward + 5 阶段 gate 优先, 直接提升 trajectory 质量)
- **D3 — RL 下游:** 是否把 Arke trajectory 语料对接 CUDA Agent 式 agentic RL 作为 Phase 5+ 方向?(这是 Arke 相对 test-time-only 系统的结构性优势, 但是大工程)

---

*调研基于两份并行子调研(5 个 kernel-gen 系统架构 + 6 个通用框架 reuse-fit),2026-07-12。*
