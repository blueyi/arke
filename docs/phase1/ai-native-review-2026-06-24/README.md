# Arke AI-Native 架构审视汇总 (2026-06-24)

> **委任**:Leon 2026-06-24 — 审视 Arke 规划与进展、重构不合理架构实现(不降标准凑 Gate)、暂停 Ascend、全力 NVIDIA 验证 LLM-Native 命题。
> **透镜**:双向 AI-Native — 整套 Language/IR/Compiler/Harness 对 **Agent 消费者**的友好度,而非只验证"决策者"命题。
> **方法**:Kitty 自查(git/grep/测试)+ 3 个并行 audit 子任务(逐层源码级审视,交叉印证致命点)。
> **明细**:`lang.md` / `ir-compiler.md` / `harness.md`。

---

## 一、最致命发现(决定一切优先级)

### 🔴 P0-A:live LLM 决策闭环根本不存在 —— 命题命门

- `arke/agent/` 下 **0 个** runner/llm 文件;`LLMRunner` / `llm_config` 全仓库**零命中**。
- `examples/agents/agent_matmul.py` import `arke.agent.runner.LLMRunner` + `arke.agent.llm_config` → **import 即崩**(期望 API 已写好,实现缺失)。
- `benchmarks/baselines/llm_direct.py` 的 `generate_kernel()` 直接 `raise NotImplementedError`。
- `arke/cli.py` 中 LLM/provider 集成 **0 处**。
- 当前 `arke optimize` 跑的全是 **deterministic heuristic**:`optimize.py:566` `chosen:"heuristic_floor"` 字面量,profile 是 `_mock_profile()`(`optimize.py:820`)用公式 `0.70+cycle*0.08+...` 造的**假分数**,`source:"mock"`。
- `benchmarks/gate_g8.py` `run_g8(tier)` **完全忽略 tier**,永远只跑 heuristic-only 检查,自标注 MVP。

**含义**:Arke 当前是"一个建得很专业的 AI-Native 脚手架,但坐进驾驶座的 LLM 还没来"。LLM-Native 命题(三角色之一的运行时决策者)**尚未被任何真实数据验证**。这正是"全力验证 AI-Native"要打的第一靶。

### 🔴 P0-B:compiler 反馈环是 mock 桩 —— Agent 无法迭代

- `verify_correctness` / `compile_and_profile` 仍是 **V0_mock 桩**:candidate ≡ reference,恒返回 `correct=True / max_diff=0`,无真实 V2 性能数。
- Agent 的 compile→profile→adjust 自纠环**拿不到真实信号** → 即便接上 LLM 也无法真正迭代优化。
- P0-A 与 P0-B 是**绑定**的:LLM runner + 真实测量必须一起上,否则各自都没意义。

---

## 二、分层不合理点(按层)

### Arke-Lang (.ak) — 详见 lang.md
1. **parser 报错纯传统 LALR 风格**(最不合理):用内部 token 名(`RPAR/SEMICOLON/F16`)、无源码 caret、无修正建议 → 掐断 Agent compile→fix 自纠环。**不动 Gate**。
2. **`where` 维度名与签名 shape 无绑定校验**:能 parse 过但语义错位的"沉默陷阱"。**不动 Gate**。
3. **token 冗余**:返回重复 shape + 单算子 `let/return` 样板 + 人类糖关键字(`for/when/otherwise`)。纯语法糖层。**不动 Gate**。

正面:LALR(1) 确定性 + 全命名参 + 封闭 dtype 枚举 + `directive(kwargs)@rationale` 统一模式 = 好的 AI-Native 基础。

### Arke-IR + Compiler — 详见 ir-compiler.md
1. **反馈环是桩**(= P0-B,关联 G8/Benchmark)。
2. **真实后端走老 `IRGraph`,Layer 2/1 是死支路**(只喂 MLIR skeleton 无人消费)。
3. **同名占位空壳类**:`compiler/lowering/{schedule_ir,instruction_ir}.py` 与正式类同名的死占位。
4. **两套 Pass 框架并存**,新栈丢弃结构化 `Diagnostic` 退回裸字符串(诊断模型降级 → 对 Agent 不友好)。
5. **自由文本谓词** `ShapeConstraint.expr/predicate` 靠 eval,validator 几乎不校验;`MultiOutputNode` 缺 `from_dict`、`Semantics` 无序列化方法(写侧不对称);`FusionGroup`/`Edge.lifetime` 优化概念泄漏进 SemanticIR。

正面:**G7[8] 实质达成** — StrategyIR core 零 Triton 字段(`warps/num_stages` 是 GPU 通用、正确归 L2);OpRegistry 纪律优秀;读侧 JSON 干净。

### Arke-Harness — 详见 harness.md
- 🟢 **成熟**:trajectory(独立 `arke/learn/`,v1.0 frozen + contract_id + 契约测试)、Façade(8 tools `assert len==8` + frozen schema + 51 测试)、checkpoint/rollback 机制真实可用。
- 🟡 **半落实**:`@rationale` 在 heuristic 路径每决策都带,但 `apply_decision` schema 标 Optional、无"每 decision 必带"执行强制 + 测试断言;graceful degradation 只有文档承诺 + 空 `fallback` 事件槽,无运行时触发(因为没有可失败的 LLM 路径)。
- 🔴 **缺失**:list_legal_actions 是硬编码 `_DEFAULT_TILE_FACTORS` 笛卡尔积,非编译器/HW/shape 合法性计算(自承 future work)。

---

## 三、重构优先级(执行顺序)

| 优先级 | 项 | 动 Gate? | 谁拍板 |
|:---:|:---|:---:|:---:|
| **P0** | 实装 LLM tool-use orchestrator(`runner.py`+`llm_config.py`,API 已被 agent_matmul.py 期望)| 否(实现 G8 既有 Tier2 criteria,非改目标)| Kitty 全权 |
| **P0** | compile_and_profile / verify 从 mock 升级真实 Triton+GPU 测量(注意 3060 6GB OOM)| 否 | Kitty 全权 |
| **P1** | parser → `ArkeSyntaxError`(line/col/caret/expected回译/suggestion)| 否 | Kitty 全权 |
| **P1** | list_legal_actions 接 HW/shape 合法性 | 否 | Kitty 全权 |
| **P1** | `@rationale` 执行期强制 + 测试断言 + 真实 fallback 分支 | 否 | Kitty 全权 |
| **P2** | 清死支路:同名占位空壳类、Layer 2/1 死支路、两套 Pass 框架收敛、`where` 语义校验 | 否 | Kitty 全权 |
| **P2** | gate_g8 补被忽略的 Tier2 live 分支 | ⚠️ 触及 Gate 验收逻辑 | **需 Leon 确认** |
| **P3** | token 冗余语法糖、谓词 eval 收紧、序列化补全、SemanticIR 分层泄漏清理 | 否(除非改 spec 定义)| Kitty 全权 |
| 卫生 | benchmark 结果产物 gitignore(~1.4MB 未跟踪)| 否 | Kitty 全权 |

**关键判断**:P0/P1/P2 大部分是**实现既有 Gate 锁定的目标(G8 Tier2 要求 live LLM 闭环)**,不是修改 Gate——属我全权范围。唯一需你确认的是 **gate_g8 验收逻辑是否补 Tier2 分支**(因为它会改变 Gate 通过/失败的判定面)。

---

## 四、已落地(本批次)

- 双向 AI-Native 命题校正 → AGENTS.md + MEMORY.md + 持久记忆。
- Phase 2/Ascend 标 PAUSED(保留全文)+ `backend/protocol.py` 扩展缝注释 + plan.md 矛盾 footer 修复 → **commit `4762f81`**。
- 三份 AI-Native 审视归档 → 本目录。
