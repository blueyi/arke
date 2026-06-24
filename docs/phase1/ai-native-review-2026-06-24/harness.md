# Arke-Harness / Arke-Agent —「AI-Native」架构审视 + 进展核实

**审视对象**：`arke/agent/`（facade/optimize/tools/env/events/state/inputs）+ `benchmarks/gate_g8.py` + 设计文档 `docs/architecture/arke-harness.md` / `docs/phase1/stage8-plan.md`
**方法**：源码精读 + 实跑 `python -m benchmarks.gate G8` 与 `arke optimize`（venv: arke）
**结论速览**：契约层（Façade/Events/Trajectory 三套 frozen schema）做得很干净、很「AI-Native」；但**核心命题「LLM 作为运行时优化决策者」目前是空的——live LLM 闭环完全没接上，运行时只有 deterministic heuristic 一条路径，且 LLM runner 模块根本不存在（demo import 的模块缺失）。**

---

## 总体判定矩阵

| 问题 | 现状 | 证据强度 |
|---|---|---|
| (1) live LLM 决策闭环 | ❌ **未跑通，且 runner 模块不存在** | 强（缺文件 + 跑通输出 `chosen:heuristic_floor`） |
| (2) list_legal_actions 由编译器计算 | ⚠️ **硬编码默认因子，非编译器/合法性计算** | 强（env.py:37-98） |
| (3) @rationale 契约落实 | 🟡 **部分**：heuristic 路径每决策都带；契约层有但**无「每 decision 必带」断言** | 中 |
| (4) trajectory 一等公民 + schema 冻结 | ✅ **是，且 v1.0 frozen + 契约测试** | 强 |
| (5) Façade(8 tools)/Substrate 两层 | ✅ **干净落地 + frozen schema + 51 测试** | 强 |
| (6) checkpoint/rollback + heuristic floor | 🟡 **state 层真实可用；但 floor 是「唯一路径」而非「降级兜底」** | 强 |

---

## (1) live LLM 决策闭环：未跑通，runner 模块缺失 ❌

**证据**
- `arke/agent/optimize.py:6-14`（模块 docstring）：
  > "This module intentionally **starts with a deterministic strategy generator rather than a live LLM call**... a future LLM runner can replace only the proposal step"
- 唯一的策略生成器是 `HeuristicStrategyGenerator`（optimize.py:87-267），`refine()`（:124-152）按 `cycle` + 固定 `bottleneck` 字符串做 if/elif 死规则调整。profile 来自 `_mock_profile()`（:820-830，`score = 0.70 + cycle*0.08 + ...` 纯公式）。
- **实跑 `arke optimize examples/operators/01_matmul.ak --cycles 3`** 的 trajectory 终止事件：
  ```json
  {"kind":"done","data":{"final_score":1.04,"decisions":11,"compiles":3,
   "termination":"llm_no_more_tool_use","chosen":"heuristic_floor"}}
  ```
  `termination` / `chosen` 是 optimize.py:563-566 **硬编码字符串字面量**，并非真有 LLM 在 loop 里"决定不再调用工具"。所有 profile 事件 `"source":"mock"`。
- **关键：LLM runner 根本不存在。** `examples/agents/agent_matmul.py:11-12` import：
  ```python
  from arke.agent.llm_config import load_from_openclaw
  from arke.agent.runner import LLMRunner
  ```
  但 `search_files arke/agent runner.py` → **0 命中**；`llm_config` → **0 命中**。整个 `arke/` 内对 `LLMProvider/Session/Orchestrator/completion/api_key/base_url` 的搜索只命中 `benchmarks/baselines/llm_direct.py`，而该文件的 `generate_kernel()` 直接 `raise NotImplementedError`（llm_direct.py:187-191），且它是「LLM 直写 Triton」基线，**不是** Harness 内的 tool-use 决策循环。
- `benchmarks/gate_g8.py` 是自标注的 **MVP** gate（:4-9 "not the full final G8 acceptance suite"）。`run_g8(tier=2)` **完全忽略 tier 参数**，永远只跑 4 个 heuristic-only MVP 检查（MVP.1~4），没有任何 live LLM 检查。`stage8-plan.md:105` 明说：
  > "The remaining Stage 8 work is still the **live LLM strategy path**, ... GPT-2 GPU target validation, LLaMA-2, DeepSeek-V2."

**诊断**：项目命题「LLM 三角色之一 = Harness 内运行时优化决策者」**当前 0% 落地**。存在的是：① 完整的 tool-use 契约外壳（8 tools/9 events/trajectory，可被任意 MCP agent 驱动）；② 一条 deterministic heuristic 参考实现。两者之间**缺失把 LLM 接到 8 tools 上、跑 propose→apply→verify→profile→adjust 多轮循环的 orchestrator（`LLMRunner`/session）**。G8 Tier2[1][2][3] 要求的 live LLM 策略路径在代码和 gate 里都没有对应实现。

**建议**：
1. 实现 `arke/agent/runner.py`（`LLMRunner`）+ `arke/agent/llm_config.py`，对接 yunwu.ai 中转（OpenAI 兼容 chat.completions + function-calling，用 `ToolRegistry.with_env(env).all_schemas()` 喂 tool schema）。`agent_matmul.py` 已写好期望 API（`runner.optimize(semantic_ir, max_turns, model_spec)` 返回 `decisions/tool_calls/tokens/trajectory`），照此实现即可使 demo 立即可跑。
2. 在 optimize.py 加 `--engine {heuristic,llm}` 开关，让 `done.chosen` 真实反映胜者，而非硬编码 `heuristic_floor`。
3. 给 gate_g8 增加 Tier2 live 分支（`--live`），断言 trajectory 出现 ≥1 个由 LLM 产生的 `decision` 事件且 `chosen=="llm"`。

---

## (2) list_legal_actions：硬编码默认因子，非编译器计算 ⚠️

**证据**：`arke/agent/env.py`
- 候选完全来自模块级常量：`_DEFAULT_TILE_FACTORS=((16,),(32,),(64,),(128,),(16,16),...)`（:37-45），`_DEFAULT_UNROLL_FACTORS=(2,4,8)` 等。
- `list_legal_actions()`（:164-209）：`loops = op.index_vars or ["i","j"]`，对每个 loop × 每个默认因子笛卡尔积生成 `Decision`。
- docstring 自承（:181-183）："This is the *generator-of-candidates* — not a ranker... Future work (D7-A1+) will add heuristic pre-filtering and **shape-aware legality checks**."
- 唯一的「计算」是 `_filter_redundant()`（:211-237）——只去掉与已应用决策重复的项，**不做硬件/shape 合法性校验**（不查 shared memory 容量、不查 tile 是否整除维度、不查 warp 预算）。`_enum_fuse_candidates` 直接返回 `[]`（:88-90）。

**诊断**：动作空间是「**有界**」的（来自固定枚举），但**不是编译器根据 SemanticIR/HardwareProfile 计算出的合法集**。当前它是 op.index_vars 驱动的静态笛卡尔积 + 去重，会产出对具体 shape/HW 非法的动作（如 tile 因子超过维度、shared 放置超容量）。这削弱了「Bounded Action Space 由编译器保证合法」这一 AI-Native 卖点。

**建议**：在 `list_legal_actions` 内接入 `HardwareProfile`（env 已持有 `self.hw_profile`）+ shape 信息做合法性过滤：tile 因子需整除/不超过对应维度、`place(shared)` 需累计 ≤ `hw.shared_memory_bytes`、`compute(warps)` 需 ≤ 上限。把这条做成 gate 断言（喂非法因子，断言其不出现在候选中）。

---

## (3) @rationale 契约：部分落实，缺「每 decision 必带」断言 🟡

**证据**
- 设计契约明确（arke-harness.md:62-64）："**`@rationale` is a contract.** Every `Decision` carries a human-readable [rationale]"。
- heuristic 路径**事实上每个决策都带** rationale（实跑 strategy.json 11 个 decision 全部有非空 text，如 `"heuristic matmul tile for M"`、`"cycle 1 adjustment: vectorize..."`）。
- **但工具层 rationale 是可选的**：`ApplyDecisionTool.parameters_schema`（tools.py:578-584）把 rationale 标为 *Optional*，schema `required:["kind","params"]` **不含 rationale**；`apply_decision` execute（:552-559）允许 `rationale=None` 通过。`Decision` 构造也接受 `rationale=None`。
- 测试覆盖（`tests/test_rationale_e2e.py`）：只验证 **.ak 源码里写了 @rationale 的算子**经 `parse → ast_to_strategy → .akir round-trip` 后 rationale 文本不丢失/不变（:64-164）。**没有任何测试断言「agent/optimize 产生的每个 Decision 都必须带非空 rationale」**，也没有断言 `apply_decision` 拒绝无 rationale 的非平凡决策。

**诊断**：契约在文档和 heuristic 实现里成立，但在**工具接口层是软约束**——一个真实 LLM 完全可以调 `apply_decision(kind,params)` 不带 rationale 而成功。契约缺少「执行时强制 + 测试断言」这两道闸。

**建议**：① 在 `ApplyDecisionTool` 对 level≥1 的非平凡决策强制 rationale 非空（缺失则 `ToolResult(success=False)`）；② 加测试断言 trajectory 中每个 `decision` 事件 payload 的 `rationale` 字段非空（events.py:165-174 已把 rationale 列为 `decision` kind 的 required field，可直接用 `validate_payload` 落地断言）。

---

## (4) trajectory：一等公民 + schema 冻结 ✅

**证据**
- 独立模块 `arke/learn/trajectory.py` + `trajectory_schema.py` + frozen `trajectory_v1_schema.json`。
- 版本锁定（trajectory_schema.py:44-55）：`TRAJECTORY_VERSION="1.0.0"`，`contract_id="arke-trajectory-v1.0.0"`，legacy `schema="s8-compile-profile-adjust-v1"`，"existing kind names + required payload fields MUST NOT change"，由 `tests/test_facade_trajectory_contract_v1.py` 强制。
- 设计为 stream(9 kind) 的**严格超集**：`RECORD_KINDS_V1 = ("header",) + EVENT_KINDS_V1 + ("adjust",)`（:59）。同一 envelope `{"t","kind","data"}` 既是 stream 又是 record，"SFT/RL extraction code has exactly one schema to parse"（:40-42）。
- 实跑产物：trajectory.jsonl 首行 `header`（含 `contract_id`、SemanticIR snapshot），随后严格 `compile→profile→adjust` ×3，末行 `done`。gate_g8.py:73-88 会断言 cycle 顺序与 header `contract_id`。

**诊断**：这是整个 Harness **最成熟、最 AI-Native** 的部分——把 trajectory 明确定位为「学习产物」(SFT/RL 语料源)、schema 冻结、stream/record 双层统一。一等公民地位坐实。

**建议**：唯一隐患是当前 trajectory 全部来自 mock/heuristic（`source:"mock"`），真正用于 SFT/RL 时需要 live LLM 轨迹 + 真实 GPU profile。schema 本身无需改动，等 (1) 落地即可填充真数据。

---

## (5) Façade(8 tools)/Substrate 两层设计：干净落地 ✅

**证据**
- `arke/agent/facade.py`：`FACADE_V1_TOOLS` 恰好 8 个（get_hw_profile, analyze_compute, list_legal_actions, apply_decision, verify_correctness, compile_and_profile, checkpoint, rollback），`assert len==8`（:58），`FACADE_VERSION="1.0.0"` + frozen `facade_v1_schema.json` + `tests/test_facade_contract_v1.py`（51 测试，doc:324）。
- 边界容器 `ArkeEnv`（env.py:103-160）："the only object passed across the Façade-Substrate boundary. External agents NEVER touch ArkeEnv directly"。`ToolRegistry.with_env(env)`（tools.py:921-945）精确注册这 8 个、注释"the Façade is exactly 8"。
- `ToolMeta`（tools.py:37-62）声明式元数据（concurrent_safe/mutates_strategy/budget_type/cost）+ `partition_for_execution`（:865-901）按 concurrent_safe 分批，体现「编排器读声明做并发/串行决策」的 AI-Native 设计。
- Substrate 类型（Decision/ScheduleIR）read-only 借用，Façade 持 orchestration shell（state.py:11-13）。

**诊断**：两层切分清晰、契约冻结、有强测试护栏。这是 vendor-agnostic / agent-runtime-agnostic 的扎实底座。**唯一注意**：tool #6 `compile_and_profile` 当前是 stateless + MockBackend（tools.py:302-424，注释 "D8-F1.3 will upgrade backend"），verify_correctness 是 `V0_mock` tier（tools.py:674-680，恒返回 correct=True/max_diff=0）——即「测量」这一环也还是 mock，未接真 Triton/GPU。

**建议**：把 #5/#6 升级到真实 Triton 编译 + GPU microbench（D8-F1.3），否则即便接上 LLM，LLM 也是在 mock 信号上做决策，无法验证「真实性能提升」。

---

## (6) checkpoint/rollback + graceful degradation：state 真实可用，但 floor 非「降级」而是「唯一路径」🟡

**证据**
- `OptimizationState.checkpoint()/rollback()`（state.py:187-241）实现真实：deep-copy strategy/decision_log/best_result + budget 快照，rollback 经 `from_dict` round-trip 恢复并回退 budget 计数。`verify_correctness` 的 trial-balloon（tools.py:686-711）真实使用了 checkpoint→apply→rollback 模式做"试探-回滚"，且注释解释了 record_compile 必须在 checkpoint 之前的微妙排序（:667-672）。✅ 机制本身可用。
- graceful degradation / heuristic floor：设计文档（arke-harness.md:73-75）："Provider rate-limits, compaction, and LLM disagreement never produce *no* kernel — **the heuristic floor always wins by default if the LLM does not improve on it**"；:734-736 伪码 `final = heuristic; note="LLM did not improve over heuristic floor"`。events.py 也定义了 `fallback` kind（:239-248，layer:strategy/provider/tier）。
- **但实现层**：因为 (1) 没有 LLM 路径，所谓「降级到 heuristic floor」**当前等于「只有 heuristic」**——不存在「LLM 失败 → fallback → heuristic」的真实分支。搜索 `arke/` 中 `fallback/graceful/degradation` 只命中 events.py 的 schema 定义和 trajectory schema，**没有任何运行时代码真正 emit `fallback` 事件或在 provider 报错时切换路径**。optimize.py 的 `done.chosen` 永远硬编码 `"heuristic_floor"`。

**诊断**：checkpoint/rollback = 真·可用（且被 verify 工具实际使用）。但「graceful degradation（heuristic floor 兜底）」**只是文档承诺 + 一个空的 `fallback` event 槽位**，没有被任何 live 路径触发——因为根本没有可失败的 LLM 路径。

**建议**：(1) 落地后，在 `LLMRunner` 里实装：provider 异常/预算耗尽/LLM 最终分数 < heuristic floor → emit `fallback{layer:"strategy",from:"llm",to:"heuristic"}` + `done.chosen="heuristic_floor"`；正常胜出则 `chosen="llm"`。加 gate 断言注入 provider 故障后仍产出非空 strategy 且出现 `fallback` 事件。

---

## 最大 Gap（按优先级）—— 验证「LLM-Native」命题需要补什么

> 命题 = 「LLM 作为 Harness 内的运行时优化决策者，在有界合法动作空间里多轮决策，跑赢/兜底于 heuristic floor」。当前**契约外壳齐备、内核空缺**。

1. **【P0 阻断】实装 LLM tool-use orchestrator（`arke/agent/runner.py` + `llm_config.py`）。**
   这是命题的全部要害。没有它，"LLM 是决策者" 一句都验证不了。`examples/agents/agent_matmul.py` 已经按期望 API 写好（import 即崩，因模块不存在），是最直接的落地靶子。接 yunwu.ai（OpenAI 兼容 function-calling），喂 `ToolRegistry.with_env(env)` 的 8 tool schema，跑 propose→apply→verify→profile→adjust 多轮，产出真实 LLM trajectory + `chosen="llm"`。

2. **【P0 阻断】把「测量」从 mock 升级为真实信号（compile_and_profile / verify_correctness 接真 Triton + GPU microbench，D8-F1.3）。**
   否则 LLM 在 `_mock_profile` 的公式分数上做决策，"跑赢 heuristic" 无法证伪。注意 RTX 3060 6GB，需小 shape 防 OOM。

3. **【P1】list_legal_actions 升级为编译器/HW 计算的合法集（接 HardwareProfile + shape 校验），而非静态默认因子笛卡尔积。**
   「有界动作空间由编译器保证合法」是 AI-Native 的核心差异化卖点，目前名不副实。

4. **【P1】把 @rationale 从软约束升为执行期强制 + trajectory 断言；落地真实 `fallback`/graceful-degradation 分支**（依赖 #1）。让 `done.chosen` 真实反映胜者而非硬编码。

5. **【P2】给 gate_g8 增加被忽略的 Tier2 live 分支**（当前 `run_g8(tier)` 完全无视 tier），断言 live LLM 策略路径 + BL5/BL6 endpoint，使 G8 真正覆盖锁定的 exit criteria 而非仅 MVP 子集。

**一句话总结**：Arke-Harness 把「让 LLM 安全、可复现地做 GPU 优化决策」的**脚手架（8-tool Façade / 9-event stream / frozen trajectory / checkpoint-rollback / budget）建得相当专业且 AI-Native**，但**真正坐到驾驶座上的 LLM 还没来**——运行时是一条 deterministic heuristic 顶替，连接 LLM 的 runner 模块尚不存在。命题成立与否，全押在 P0 的两项（LLM orchestrator + 真实测量）上。
