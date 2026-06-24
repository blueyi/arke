# Arke-IR + Arke-Compiler — AI-Native / LLM-友好度 架构审视

> 透镜：使用者是 **LLM/Agent（非人类）**。评估对象 = `arke/ir/`（SemanticIR + StrategyIR + Schedule/Instruction）与 `arke/compiler/` + `arke/backend/`。
> 方法：每问给「证据（文件:行号+片段）→ 诊断 → 重构建议（标注是否动 Gate/Benchmark）」。仅分析，未改任何文件。
> 仓库：`/home/blueyi/workspace/repos/arke`，commit 当前 HEAD。

---

## 总览结论（先看这个）

Arke 的 **SemanticIR / StrategyIR 这两层（Layer 4/3）对 Agent 是真正友好的**：JSON 干净、字段语义明确、`@rationale` 一等公民、默认值省略使 token 经济。G7[8]「StrategyIR core 零 Triton-specific 字段」**实质达成**。

但有三处系统性问题严重削弱 AI-Native 价值：

1. **双 IR 并存 + 真实后端走老路**：`TritonBackend.lower()` 消费的是 S6 老的 `IRGraph`（`arke/ir/graph.py`），而 Layer 2/1（ScheduleIR/InstructionIR）只产出一份**永不被后端读取**的 MLIR *skeleton*。Layer 2/1 是事实上的死支路。
2. **Agent 反馈环未接通**：`verify_correctness` / `compile_and_profile` 仍是 `V0_mock` 桩，candidate≡reference，**永远返回 `correct=True, max_diff=0`**。Agent 拿不到真实数值/性能信号，无法迭代自修正。
3. **空壳与重复定义**：`arke/compiler/lowering/{schedule_ir,instruction_ir}.py` 是与正式 `arke/ir/{schedule,instruction}.py` 同名却完全无关的占位类。

---

## (IR-1) IR schema 能否被 Agent 低错误率读写？

**证据（正面）**
- `semantic.py:257-275` `Node.to_dict()`：输出结构扁平清晰 `{id, op, inputs, output, semantics, attrs}`，且**默认空字段全部省略**（`if self.semantics.index_vars: ...`），token 经济。
- `semantic.py:188-190` 注释 `# dtype/shape are derived — not serialized (avoid redundancy)`：`ParamRef/NodeRef` 只序列化 `{ref, name/id}`，消除冗余 → Agent 写时不需重复填类型，降低出错面。
- `strategy.py:30-53` `Decision` 的 docstring **把所有合法 kind + 参数 schema 写进类型注释**（tile/reorder/fuse/parallel/place/vectorize/unroll/algorithm/compute…），Agent 可直接据此构造。
- `strategy.py:162-223`：`tile()/reorder()/fuse()/compute()/when()` 提供 builder API，比手搓 dict 更不易错。

**证据（风险）**
- `semantic.py:101` `ShapeConstraint.expr: str`、`semantic.py:317` `ConditionalNode.predicate: str`、`strategy.py:63` `ConditionalDecision.predicate: str`：**谓词是自由字符串**（`'dim("S") <= 512'`、`"NB <= 1024 and MB <= 128"`），靠下游 `eval`/正则解析。`validator.py:151-160` 只做最弱的「大写 token 是否声明过」检查，**语法/类型完全不校验**。Agent 极易写出解析期才报错、甚至静默错误的谓词。
- **MultiOutputNode 的 `from_dict` 缺失**：`semantic.py:278-306` 有 `MultiOutputNode.to_dict()`，但全类**没有 `from_dict`**；反序列化逻辑硬塞在 `SemanticIR.from_dict`（`semantic.py:525-541`），与 `Node` 路径不对称 → Agent/工具往返一致性脆弱。
- `Semantics` dataclass（`semantic.py:229-237`）**无 `to_dict/from_dict`**，序列化散落在 `Node.to_dict`；而 `MultiOutputNode.to_dict`（`:299-304`）**无条件输出全部 semantics 字段**，与 `Node` 的「省略默认」策略不一致 → 同一份语义两种 JSON 形态。

**诊断**：读侧优秀；写侧的自由文本谓词是最大低级错误来源，且序列化路径不对称（Node vs MultiOutputNode、有无 from_dict）会让 round-trip 工具行为不一致。

**重构建议**
- 为谓词引入**结构化 mini-AST 或受限 DSL**（`{op:"<=", lhs:{dim:"S"}, rhs:512}`），并在 `validator` 中做语法+符号校验，把错误从「运行期 eval 崩溃」前移到「apply 即报错」。
- 统一序列化：给 `Semantics`/`MultiOutputNode`/`ConditionalNode` 补 `from_dict`，让所有 Node 类型走同一「省略默认」策略。
- **不动 Gate**（纯 schema/校验加固，G7[2] IR spec 仍满足，反而更稳）；JSON schema 文件 `arke/ir/schemas/` 需同步更新。

---

## (IR-2) Semantic / Strategy 分离是否干净（"算什么" vs "怎么优化"）？

**证据（正面）**
- `semantic.py:398-415` `SemanticIR` 字段纯描述计算（params/nodes/edges/semantics/symbolic_dims），`strategy.py:130-145` `StrategyIR` 纯描述决策（decisions/shape_regimes/constraints）。两文件 import 方向单一：`strategy.py` **不 import** `semantic.py`，物理解耦。
- `semantic.py:402` docstring 明确：「The LLM Agent reads this; StrategyIR is what the LLM writes.」职责切分对 Agent 清楚。

**证据（污染）**
- `semantic.py:371-392` `FusionGroup`（带 `fusion_type: "epilogue"|"prologue"|"horizontal"|"vertical"` + `reason`）**位于 SemanticIR**。Fusion 是「怎么优化」，却出现在「算什么」层；`SemanticIR.fusion_groups`（`:414`）让优化提示泄漏进语义层。
- `semantic.py:339-358` `Edge.lifetime: "local"|"persistent"` —— 内存生命周期是调度/优化概念，混进语义 DAG 边。
- 对偶：StrategyIR 的 `fuse` decision（`strategy.py:175-180`）与 SemanticIR.FusionGroup **语义重叠**，到底以哪个为准未定义 → Agent 可能两处都写或写矛盾。

**诊断**：核心字段解耦干净，但 **fusion / lifetime 两个「优化语义」漏进了 SemanticIR**，造成 (a) 概念越界，(b) 与 StrategyIR.fuse 的真相源冲突。对 Agent 而言「fusion 该写在哪层」是模糊的。

**重构建议**
- 把 `FusionGroup` 从 SemanticIR **降级为可选 hint** 或彻底移到 StrategyIR；`Edge.lifetime` 移到 ScheduleIR（它本就是调度层）。
- 文档化「fusion 单一真相源 = StrategyIR.fuse」，SemanticIR 至多保留 `can_fuse_as` 能力声明（已在 `OpSchema.can_fuse_as`，`schema.py:115`）。
- **不动 Gate 目标**（G7[2] 关注 spec 完整性，分层归位反而更符合 spec §262「backend-specific lowering 属于 StrategyIR 之下」）；但需同步 spec 文档 §3/§4。

---

## (IR-3) StrategyIR 是否 backend-agnostic（G7[8] 要求 core 0 个 Triton-specific 字段）？

**证据**
- 全 IR 目录搜 `triton|BLOCK_|num_warps`：`arke/ir/` 命中 **0 处 Triton 字面量**。
- `strategy.py:196-212` `compute(warps, num_stages, shared_memory)` —— 这些是 **CUDA/GPU 通用资源概念**（warp、shared memory、pipeline stages），非 Triton 私有；且明确标 `level=2`（L2 resource/backend-bound），与 L1 backend-agnostic 决策（tile/reorder/fuse…）分级。`strategy.py:44-47` docstring 把 L2 单独列为「resource / backend-bound, set by specialization passes」。
- `docs/roadmap/plan.md:209` G7[8] 原文：「0 Triton-specific fields in StrategyIR core」；`plan.md:220` 标 13/14 PASS、backend-agnostic StrategyIR 为 green。

**唯一可争议点（但不在 core）**
- `schema.py:54-66` `TemplateHint`「Routing hint for the **Triton** template engine」+ `template_name`(Jinja2) —— 这是 **Triton-specific**，但它在 `OpSchema`（kernel schema 注册表），**不在 StrategyIR**。属于 SemanticIR 周边的算子元数据，技术上未违反 G7[8]「StrategyIR core」字面，但从「AI-Native 分层洁癖」看，它让 SemanticIR 算子定义里带上了后端泄漏。

**诊断**：**G7[8] 实质达成 ✅**——StrategyIR core 无 Triton 字段，`warps/num_stages` 是 GPU 通用且正确归在 L2。唯一瑕疵是 `TemplateHint` 把 Triton 绑进了 OpSchema（非 StrategyIR）。

**重构建议**
- 维持现状即满足 Gate；若追求更纯净：把 `TemplateHint` 泛化为 `backend_hints: dict[str, ...]`（按 backend 名分桶），避免 OpSchema 里出现 `triton` 字面绑定。
- **不动 Gate**（G7[8] 已 PASS，此改动只增强可扩展性，不影响判定）。

---

## (IR-4) IR 层数（Layer 4/3/2/1）是否过度设计 / 有空壳？

**证据（设计意图）**
- `docs/architecture/arke-ir-spec-design.md:51-54`：Layer 4 Semantic → 3 Strategy → 2 Schedule → 1 Instruction（near-LLVM）。`:99` 明确 **Phase 1 只交付 Layer 4+3**，Layer 2/1 标注「mostly/fully automated」。

**证据（空壳 / 死支路）**
- **重复定义**：`arke/compiler/lowering/schedule_ir.py:3-18` 与 `arke/compiler/lowering/instruction_ir.py:3-12` 是 **18 行 / 12 行的占位类**（`ops=[]; dependencies={}`），与正式实现 `arke/ir/schedule.py`(305行)、`arke/ir/instruction.py`(118行) **同名却毫无关系**。纯死代码，且制造导入歧义。
- **Layer 2/1 产出无人消费**：`lowering.py:113-122` `lower_full_stack` 产出 ScheduleIR+InstructionIR；`pipeline.py:316-327` 把它们塞进 `CompilationResult`，但**唯一下游是 `mlir_emitter.py`**，而 MLIR 仅是 `emit_mlir_skeleton`（`mlir_emitter.py:4-13` 自述「does not attempt full semantic lowering... suitable for tests/architecture validation rather than code generation」）。
- **真实后端绕过 Layer 2/1**：`triton_backend.py:88` `lower(self, graph: IRGraph)` 与 `protocol.py:70` `lower(graph: IRGraph)` —— 后端消费的是**老 IRGraph**，ScheduleIR/InstructionIR 与真实 codegen **完全无连接**。
- `schedule.py:243` `apply_decision` 对未知 kind 静默记 `ignored:{kind}`，`lowering.py:169-184` 条件分支只物化 true 分支 → Layer 2 的语义本身也偏「占位」。

**诊断**：Layer 4/3 必要且活跃；**Layer 2/1 当前是「为 MLIR 前向兼容预留的形状证明」**，spec 已声明 Phase 3 才落地——这本身合理（非过度设计）。**真正的问题是工程卫生**：(a) 同名占位类 vs 正式类的重复定义是明确技术债；(b) 真实编译路径走 IRGraph 老路，使 multi-layer 栈与可执行结果脱节——对 Agent 而言，它在 StrategyIR 写的决策**最终是否影响 kernel 无法通过 Layer 2/1 验证**。

**重构建议**
- **立即删除** `arke/compiler/lowering/schedule_ir.py` + `instruction_ir.py` 占位类（或合并到正式实现），消除导入歧义。
- 中期：让 `TritonBackend` 改吃 InstructionIR（或至少让 SemanticIR→IRGraph 的桥接显式化），使 Layer 2/1 不再是死支路。
- **不动 Gate**（G7.5 只要求 MLIR skeleton 作前向兼容，Phase 3 才做真 lowering；清理死代码不影响 Gate 判定）。

---

## (Compiler-5) V0/V1/V2 反馈信号是否为 Agent 迭代循环设计？

**证据（设计意图，很好）**
- `docs/architecture/naming-system.md:163-171`：V0 静态<1ms / V1 数值~100ms / V2 性能~1-5s，并明确「LLM 可被告知 V0 便宜随便用、V2 贵节约用」——成本分级是 AI-Native 的正确直觉。
- `agent/state.py:32-52` `CompileResult` 字段设计良好：`correct/max_diff/latency_ms/baseline_ratio/error`，且 `to_dict` 省略 None → 反馈紧凑。
- `tools.py:629-633` `_default_tolerance` 按 dtype 分级（fp16/bf16 用 1e-2，否则 1e-3），数值判定对 Agent 有意义。

**证据（致命：环未接通）**
- `tools.py:674-680`：`verify_correctness` 注释直白「**candidate backend not wired yet → V0_mock tier**」，硬编码 `CompileResult(correct=True, max_diff=0.0, validation_tier="V0_mock")`，**无论 Agent 提交什么决策都返回正确**。
- `tools.py:600-604`：「candidate equals the reference (V0_mock tier), which **always returns correct=True with max_diff=0**」。
- `tools.py:402` `compile_and_profile` 的「真实」校验跑的是 `MockBackend.run_graph` 对 `INTERPRETER`（参考实现）—— 二者本质同源，`correct` 几乎恒真；**无 `latency_ms`/`baseline_ratio`（V2 完全缺位）**。
- 错误可恢复性方面：`tools.py:411-415` 参考校验失败时返回 `success=True` + warning（吞错），Agent 收到的是「跳过」而非可操作的失败信号。

**诊断**：**反馈环的 schema/成本模型设计得对，但信号本身是假的**。Agent 当前处在「怎么改都说对、没有性能数」的环境里——它**无法学习、无法自修正、无法做性能驱动的迭代**。这是 AI-Native 命题的核心功能缺口（代码自注为 D8-F1.3 待办）。错误信息在「真失败」时质量尚可（如 `pipeline.py:368-371` 缺参数会给出 required 列表），但成功路径的信号空洞。

**重构建议**
- 接通 V1：让 candidate 走真实 codegen（Triton/编译产物）而非 `INTERPRETER`，使 `max_diff` 反映真实数值偏差；失败时返回**结构化、可操作**的 error（哪个 node、哪个 tile factor 越界、建议方向）。
- 接通 V2：填充 `latency_ms` + `baseline_ratio`（对照 golden ladder），这是 Agent 性能迭代的唯一信号源。
- 错误**分级可恢复性**：区分「永久错误（op 不存在）」vs「可调错误（shared_memory 超限→建议降 tile）」，后者附 `suggestion` 字段供 Agent 直接消费。
- **直接关联 Gate/Benchmark**：这正是 Stage 8 / G8（`plan.md:312` 「BL5 no regression」+ HARNESS）与 `compile→profile→adjust` 闭环的硬前提。**会动 Benchmark 验收**（需 GPU 真实测量），建议作为最高优先工程项。

---

## (Compiler-6) Pass pipeline / OpRegistry / Backend 抽象有无不合理耦合 / 过度工程？

**证据（OpRegistry —— 设计克制，正面）**
- `registry.py:12-46` 用大段注释精确划清「kernel schema view」vs「kernel SSOT (`benchmarks/op_registry.py`)」vs「未来 IR dialect 注册表」三者边界，并**显式禁止** 在此文件加 `total_ops()`/`__len__==N` 不变量（`:41-45`）。这是优秀的反过度耦合纪律。
- `registry.py:190-214` `validate_coverage()` 让缺字段算子可被程序化发现 —— 对 Agent/CI 友好。

**证据（重复 / 耦合）**
- **两套 Pass 框架并存**：
  - `compiler/passes/base.py:143` `PassPipeline`（吃 `IRGraph`，重型：`PassContext`+`Diagnostic`+`HardwareProfile`+severity）；
  - `compiler/semantic_pipeline.py:43` `SemanticPassPipeline`（吃 `SemanticIR`，轻型：pass = `Callable[[SemanticIR], list[str]]`）。
  - 两者职责重叠（都是「有序跑 pass、首错即停」），但**接口/诊断模型完全不同**。`tools.py:359` 用前者，`pipeline.py:304` 用后者 → 维护者与 Agent 都要理解两套心智模型。
- **Diagnostic 富模型被浪费**：`base.py:38-48` 的 `Diagnostic`（severity/pass_name/node_id）本是给 Agent 的好反馈载体，但 `SemanticPassPipeline` 弃用它、退回裸 `list[str]`（`semantic_pipeline.py:98`）→ Agent 拿到的是无结构错误串，丢失 node 定位。
- **Backend 抽象耦合到老 IR**：`protocol.py:20,70` `ArkeBackend.lower(graph: IRGraph)` 把整个后端协议**钉死在 S6 IRGraph 上**，与 multi-layer IR 脱节（呼应 IR-4）。

**诊断**：OpRegistry **不过度工程，纪律良好**。真正问题是**两套 Pass 框架 + 两套诊断模型的重复**（S6 遗留与 S7 新栈未收敛），以及 Backend 协议绑定老 IR。对 Agent 的副作用：错误反馈结构在新栈里被降级为纯字符串。

**重构建议**
- 收敛为单一 Pass 框架：让 `SemanticPassPipeline` 复用 `Diagnostic`（带 node_id/severity），淘汰或下沉 `IRGraph` 版 `PassPipeline`。
- `ArkeBackend.lower` 的入参从 `IRGraph` 迁到 InstructionIR/SemanticIR，统一 IR 真相源。
- **不动 Gate 目标**（属内部收敛重构，不改 G7/G8 验收口径）；但需回归测试覆盖 `tools.py` 现用的 IRGraph 路径，避免迁移破坏 Stage 8 MVP。

---

## Top-5 不合理点排序（按对 AI-Native 命题的危害）

| # | 不合理点 | 证据 | 危害 | 是否动 Gate |
|---|---------|------|------|------------|
| **1** | **Agent 反馈环是桩**：verify/profile 恒返回 correct=True、无 V2 性能数 | `tools.py:600-604,674-680,402` | **致命**：Agent 无法迭代/自修正/性能优化，直接架空 AI-Native 命题 | 关联 **G8/Benchmark**（需真实测量） |
| **2** | **真实后端走老 IRGraph，Layer 2/1 是死支路** | `triton_backend.py:88` `protocol.py:70` vs `lowering.py:113` | 高：Agent 在 StrategyIR 的决策无法经 multi-layer 栈验证落到 kernel；架构两套真相源 | 不动 Gate（Phase 3 才接 MLIR）；属工程债 |
| **3** | **同名占位类重复定义（空壳）** | `compiler/lowering/schedule_ir.py`+`instruction_ir.py` vs `ir/schedule.py`+`ir/instruction.py` | 中高：导入歧义、误导维护者、纯死代码 | 不动 Gate；应立即删 |
| **4** | **两套 Pass 框架 + 诊断模型降级**，新栈丢失结构化 Diagnostic | `passes/base.py:143` vs `semantic_pipeline.py:43,98` | 中：Agent 拿裸字符串错误、丢 node 定位；双心智模型 | 不动 Gate；需回归测试 |
| **5** | **自由文本谓词 + SemanticIR 分层泄漏（fusion/lifetime）+ 序列化路径不对称** | `semantic.py:101,317,371-392,339-358`；缺 `MultiOutputNode.from_dict` | 中：Agent 写谓词/fusion 易错且静默；round-trip 脆弱 | 不动 Gate（schema 加固，G7[2] 更稳） |

---

### 一句话总结
**Layer 4/3 的 schema 设计是 AI-Native 的合格样板（G7[8] 实质达成）；但「反馈环是桩 + multi-layer 栈与真实后端脱节」使整个 Agent 迭代闭环目前无法真正运转——这是从「能编译」到「Agent 能学会优化」之间最关键、且 Gate 已识别（D8-F1.3 / G8）的缺口。**
