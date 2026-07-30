# Arke 架构级质量审计报告（2026-07-29）

**审计范围**：全仓源码（HEAD `c5e02b1`）+ 78 篇设计文档 + benchmark 代码与真实运行数据
**方法**：3 路并行取证（benchmark 数据 / 架构源码 / 工程规范）+ 主审亲验关键断言；
所有数字来自真实文件、命令输出或当日实测，未测项如实标注。
**取证底稿**：`/tmp/audit_dim1_benchmark.md`、`/tmp/audit_dim2_arch.md`、`/tmp/audit_dim35_eng.md`

---

## 0. 目标达成诊断（结论先行）

**总体达成比例预估：~72%**（按锁定命题"AI-Native 工具链 + NVIDIA L1 验证"口径；
若按题面"异构硬件 + 自定义 MLIR Dialect + 任意 DSA 扩展"的完整口径则 ~55%）。

| 维度 | 达成 | 一句话判定 |
|:--|:--:|:--|
| 一、Benchmark 闭环 | **85%** | 46 op/SSOT/golden ladder/动态shape track 完备；分母陷阱与超长 seq 空白扣分 |
| 二、架构与扩展性 | **60%** | 扩展 seam 真实干净；但无自定义 Dialect、4 层 IR 只有 2 层承重 |
| 三、动态 Shape 泛化 | **70%** | cliff 已量化+bucket 缓解（matmul/FA）；row-scan 未覆盖、无运行时降级策略 |
| 四、异构适配深度 | **35%** | HardwareModel/protocol 抽象成型；但唯一实例是 sm_86，异构=纸面承诺 |
| 五、工程与开源就绪 | **75%** | 2862 测试全绿、后端解耦好、README/LICENSE 有；缺 CONTRIBUTING/对外指南 |

---

## 一、Benchmark 深度解析与性能闭环

### 1.1 覆盖度：完备且有 SSOT 纪律

- **46 算子**（实测 `op_registry.py`）：OT0=12 / OT1=10 / OT2=11 / OT3=8 / OT4=5。
  计算密集 8（matmul 族 3 + attention 5）、访存密集 30、融合 8。
- Shape 三档极端边界：tiny、超长（1M 宽、S=163k）、非对齐（`non-align-*`）。
- conv / 稀疏是**文档记录的有意排除**（`benchmark-ops.md`），非疏漏。
- SSOT 链有测试强制（`test_ssot_op_registry.py`），算子清单不会漂移。
- **空白**：4 个 no-data 算子（MLA / paged_attention / permute / swiglu_packed，与
  audit-degraded 名单吻合）；163k 超长 seq 在 6GB 卡上从未实测。

### 1.2 性能基线（真实数字，vs 业界标杆）

| 算子族 | vs 标杆 | 数字（当日 A/B，correctness 全过） |
|:--|:--|:--|
| matmul (Triton) | cuBLAS | **~0.81×**（MLIR 路径 1.05×，CUDA-C 1.05×） |
| flash_attention | flash-attn 2.7.4 | **0.846**（今日 FA-v2 后；晨间还是 0.301） |
| GQA | flash-attn | **0.802**（修正 correctness bug 后） |
| cross_attention | flash-attn | **1.081（反超）** |
| 小 elementwise/reduce | cuBLAS/eager | 0.5–0.76×（relu 0.55 / softmax 0.58） |

- **无 <0.3 真回退**。唯一 0.207（split）是 view-op 基准不公平，文档已识别。
- ⚠️ **分母陷阱（审计发现）**：全量 PERF_ALL 快照里 flash-attn golden 曾因
  `IndexError` 静默回退到 PyTorch-eager，导致该快照 attention 数字（3.06×/2.31×）
  是 vs eager 的虚高值。可信 attention 数据只在 `k-att-plan.md §6`。
  **教训：golden 失效必须 fail-loud，不能静默降级分母。**

### 1.3 调优收益量化

- matmul live-agent：2 迭代 **0.38→1.28（3.4×）** — 最亮点样本。
- 但 live 收敛轨迹整体**样本浅**（matmul 仅 2 iter；FA 当时 2/3 correctness 失败）。
  "搜索空间收敛效率"的强结论证据不足——诚实标注，不夸大。
- default heuristic 已强（gate 加权 geomean 0.947），live-agent 边际收益 -2~-3.4%；
  其真实价值在：把 default 输的 case 拉回 ≤1.05× + config 记忆 + rationale KB 390 条。
- **今日最大调优事实**（非 LLM 调优，是 dtype 纪律）：FA 0.301→0.496（tile sweep）
  →**0.846**（TC dtype 修复）。杠杆排序：TC 利用率 ≫ tile/pipeline 微调。

---

## 二、核心架构与扩展性（MLIR/LLVM 生态）

### 2.1 Dialect 与 Pass —— 必须澄清的两个事实

**事实 A：Arke 没有自定义 MLIR Dialect。** 全仓 `.td`/tablegen/C++ = **0**。
MLIR 路径是 Python 字符串生成 upstream 标准 dialect（linalg/memref/gpu/nvvm/
nvgpu/transform）文本，交外部 `mlir-opt`/`llc`/`ptxas` lowering。Arke 是
"MLIR 文本前端 + emitter"，不是 dialect 作者。任何"自定义 Dialect 分层"
的表述与代码不符。

**事实 B：`compiler/passes.py` 的 Pass 是校验/分析，非优化变换。**
实际存在：ShapeInference / SSAValidation / RationalePreservation。真正的性能
变换（tile 选择、TC dtype、bucket 缓存、fusion 组织）全部在 backend 的 Jinja
模板 + launcher 启发式里，**绕过 pass pipeline**。"多级 Pass pipeline 高效闭环"
目前不成立——闭环的是 Agent 的 compile→profile→adjust 外环，不是 IR 内的
pass 变换环。

### 2.2 IR 映射：语义无损，调度非无损

- 4 层 IR 中 **SemanticIR(L4)/StrategyIR(L3) 真实承重**；ScheduleIR(L2)/
  InstructionIR(L1) 是"已填充结构骨架，字段不驱动 codegen"（K-H5.1 已诚实
  降格入 spec §3.4，本审计核实属实）。
- SemanticIR→linalg、StrategyIR→transform dialect 映射干净，`@rationale` 经
  `transform.annotate` 保留——**语义层信息无损成立**。
- thread/warp/bank/register 级调度决策在各 backend 内部完成，绕过 L2/L1——
  **端到端调度无损不成立**。

### 2.3 Tensor Core 利用率（结合 benchmark 佐证）

- ✅ flash_attention 模板已修（FA-v2）：fp16 dot + fp32 accumulate → HMMA 上线，
  geomean 0.496→0.846，精度反升（err 2e-3→4.9e-4）。
- 🔴 **审计新发现：`mla.py.j2` 两个 kernel 仍有同款 `.to(tl.float32)` 反模式**
  （L54/61/93/104/116，已亲验）——TC 全程闲置，正是 FA-v2 修复前的病根。
  MLA 恰好也是 4 个 no-data 算子之一，坏味道叠加。
- 🟡 batch/grouped_matmul `allow_tf32=False` 在 f32 输入时禁 TF32-TC（fp16 主路
  无影响，f32 路径待确认）。
- MLIR emitter 有 `nvgpu.mma.sync` 真 TC 路径但仅 single-matmul/f32/tile-aligned。

### 2.4 架构泛化 / 异构（维度四合并于此）

- `protocol.py`（ArkeBackend 4 方法 Protocol + BackendRegistry）+ `hardware.py`
  （HardwareModel：内存层级树/同步域/计算单元/对齐约束）是**真结构化的扩展 seam**，
  新后端插拔无需动核心。
- 但**唯一 HardwareModel 实例 = `nvidia_sm86()`**；4 个后端全指同一块 RTX 3060。
  Ascend/AMD 零代码。"屏蔽 NVIDIA vs Ascend 差异"是**未经任何异构硬件检验的
  设计承诺**。SIMT vs SIMD 的执行模型差异、Ascend 的 UB/L1/L0 内存层级是否能被
  现有抽象表达，无实证。（Ascend PAUSED 是 Leon 的战略决定，此处只陈述事实。）
- backend/ 无 stub：Triton/MLIR-GPU/CUDA-C/LLVM/mock 全真实可跑。

---

## 三、动态 Shape 鲁棒性与退化控制

### 3.1 Performance Cliff（实测，`dynamic_shape/2026-07-29_191225`）

| op | cliff geomean | max | 缓解机制 |
|:--|--:|--:|:--|
| softmax | **40.99×** | 130.7× | ❌ 无（每新 seq-len 付 3.5-6ms 编译） |
| rmsnorm | 7.22× | 86.4× | ❌ 无（cliff 集中头两 shape） |
| matmul | **3.31×** | 27.9× | ✅ `_TILE_CFG_CACHE` pow2 bucket |

**对 Llama3 变长解码 = 最坏场景**：attention logits 的 softmax 在逐 token
decode 的新长度路径累计 ~30-50ms 纯编译墙；SD 分辨率扫掠同理。matmul(FFN)
因 bucket 基本免疫——**bucket 方案有效性已被对照数据证明**，只是没铺开。

### 3.2 退化控制现状

- ✅ 三级正确性 fallback（triton_backend：未知 op/无 hint/codegen 异常 →
  INTERPRETER.execute）——保证不 crash 不错算。
- ✅ bucket cache 模式已在 matmul + flash_attention（`_FA_CFG_CACHE`，今日落地）。
- ✅ PlateauEarlyStop(patience=3) 控优化循环编译预算（优化层，非运行时）。
- ❌ **缺口**：softmax/rmsnorm/row-scan 族未套 bucket；**没有运行时"JIT 过贵
  则降级"策略**（如首调用 interpreter/eager 顶住、后台异步编译、编译预算超限
  自动切 AOT 常用 bucket）；dynamic-shape track 仅 D1 measure-only 无门禁。

---

## 五、工程落地与开源就绪度

- **测试**：make test = **2862 passed / 0 failed**（本审计期间三次全绿）。
- **解耦**：后端 Protocol+Registry 插拔干净；backend/ 32 文件按
  `triton_*/cuda_c_*/llvm_*/mlir_*` 四组水平切分，非散乱；无 god object
  （最大 mlir_emitter 2992 LOC 可接受）。
- **气味（AST 实测）**：①唯一包级循环依赖 `arke.agent ↔ arke.learn`
  （agent/optimize.py↔learn/rl_dataset.py 等，已亲验）；②双目录
  `arke/backend/`(18.5K) vs `arke/backends/mlir/`(235 LOC 遗留)命名歧义；
  ③AGENTS.md 架构图列了不存在的 arke/engine、arke/parser（文档漂移）。
- **开源就绪**：README(309行)/LICENSE(Apache-2.0)/pyproject ✅；
  **CONTRIBUTING ❌、对外开发者指南 ❌、API 教程 ❌**；内部 agent 流程文档
  （AGENTS/SOUL/phase*/kestrel）与对外文档未剥离。判定：**"内部研发中"，
  非"社区可加入"**。接口规范本身（protocol/SSOT/spec 三件套）质量足以吸引
  MLIR/IREE 背景开发者，缺的是入口不是里子。

---

## 突出技术亮点（数据佐证）

1. **Benchmark 治理是同类项目少见的强项**：SSOT + golden ladder（拒绝弱分母）+
   频谱化 gate + 测量诚实文化（假 6.4× speedup 事故被记录在案而非掩盖）。
2. **TC dtype 纪律的发现与修复链**（0.301→0.846 一日完成）证明了
   compile→profile→adjust 闭环真的能驱动大幅优化——AI-Native 命题的正面实证。
3. **cross_attention 1.081 反超 fused flash-attn**：非 causal Sq≠Skv 区间是
   当前性能优势区。
4. **bucket cache 对照实验**（matmul 3.3× vs softmax 41×）把动态 shape 缓解
   方案的有效性变成了可复制的量化证据。
5. **fallback 三级链 + V1 correctness gate**：错误的快 kernel 进不了结果
   （GQA gqa_groups bug 即被 correctness 闸截获修复）。

## 架构级隐患（≥3，按严重度排序）

### 隐患 1：宣称的 IR/Pass 分层与实际承重结构不符（诚实性债务）🔴
4 层 IR 只有 2 层承重、pass pipeline 只做校验、无自定义 Dialect。当前 spec
已加降格标注（K-H5.1），但**对外叙事（README/架构文档）若继续用"多级 IR +
Pass pipeline + MLIR Dialect"语言，就是宣传超过实现**。对内的风险是：新贡献
者会按 spec 去 ScheduleIR 找调度逻辑，找不到。

### 隐患 2：调度知识散落在模板字符串里，不可组合、不可复用 🔴
TC dtype 纪律在 flash_attention.py.j2 修了，mla.py.j2 里同款病根还活着——
**因为"fp16 dot + fp32 acc"这条知识没有一个统一的中间层可以一次性生效**。
每个 Jinja 模板是一座孤岛：tile heuristic、bucket cache、dtype 纪律都要逐
模板手抄。这正是 ScheduleIR 骨架化的直接代价，也是未来任何新 op/新后端的
重复劳动来源。

### 隐患 3：动态 shape 无运行时退化策略，row-scan 族裸奔 🟠
cliff 已测量（softmax 41×）但只有 measure-only 门禁；bucket 只铺了 2 个 op；
没有"JIT 过贵→先 eager/interpreter 顶住+异步编译"的运行时策略。对 serving
场景（变长 decode）这是可预测的 P99 事故源。

### 隐患 4：异构抽象零实证 + 单实例 HardwareModel 🟠
HardwareModel 只有 sm_86 一个实例，抽象的"泛化力"从未被第二个真实硬件拉练。
经验规律：单客户抽象在第二个客户出现时必然重构。Ascend 恢复时 protocol/
HardwareModel 大概率要动刀（尤其 SIMD 执行模型与 UB 内存层级的表达）。

### 隐患 5（工程级）：agent↔learn 循环依赖 + 双 backend 目录 + no-data 算子 🟡
单独都小，叠加是入口处的第一印象问题。

## 建设性重构与调优方案（对应隐患，可操作）

### R1（对隐患1）：叙事对齐 + L2 重生的"最小真实化"路径
- 短期（0.5d）：README/架构文档统一措辞——"SemanticIR+StrategyIR 双层承重 IR，
  MLIR emitter 走 upstream dialect"；删除/降格所有"自定义 Dialect"暗示。
  AGENTS.md 架构图修正（删 engine/parser 幽灵目录）。
- 中期（1-2w，若 Leon 批）：**不要**自底向上补全 ScheduleIR——把今日证明有效
  的三条横切知识（TC dtype 纪律 / bucket cache / tile heuristic）提为
  `KernelPolicy` 数据对象，由 StrategyIR→模板渲染上下文统一注入。这是
  "L2 最小真实化"：先让一小块 ScheduleIR 字段真正驱动 codegen，再谈全量。

### R2（对隐患2）：立即修 mla.py.j2 + 模板 lint 闸
- 立即（0.5d）：mla.py.j2 删 5 处 `.to(tl.float32)`，改 fp16 dot +
  `out_dtype=fp32`，同日 A/B 验收（FA 经验预期显著提升）。
- 制度化（1d）：新增 `tests/backend/test_template_tc_discipline.py`——静态扫描
  所有 `*.py.j2`：凡 `tl.dot` 的操作数出现 `.to(tl.float32)` 即 fail。
  把这条知识从"人肉记忆"变成"回归闸"。同类闸可扩展：bucket cache 存在性、
  launch config 走 `_cfg` 约定。

### R3（对隐患3）：row-scan bucket + 运行时降级
- 短期（1-2d）：softmax/rmsnorm 套 `_LAUNCH_CFG` 的 pow2-N bucket（matmul 模式
  照抄），预期 cliff 41×→个位数；用 dynamic_shape track 直接验收（该 track
  就是为此建的）。
- 中期（3-5d）：KernelCache 加"预算感知"模式——首调若预估编译>阈值（op 类
  查表），先走 interpreter/eager 返回 + 后台线程编译，编译完热切换。
  serving 语义从"首 token 卡 6ms"变"首 token 稍慢但无墙"。
- 门禁：D1→D2 升级（same_spec_geomean ≤5× 软 gate）等跨 run 方差数据齐后报批。

### R4（对隐患4）：用"纸上第二后端"拉练抽象
在不投入 Ascend 实现的前提下（尊重 PAUSED 决策），做一次 **1-2d 的
HardwareModel 纸面实例化**：按 Ascend 910B 公开规格填一份
`ascend_910b()` HardwareModel（UB/L1/L0 层级、cube unit、对齐约束），
让 StrategyIR 合法动作生成器对它跑一遍 dry-run。跑不通的字段就是抽象的
真实缺口清单——把"第二客户重构"的成本提前从实现期挪到设计期。

### R5（对隐患5）：卫生三连（各 0.5d 内）
- 删 `arke/backends/mlir/`（235 LOC 遗留，先确认无引用）。
- 断 agent↔learn 环：`robust_reward`/`events` 下沉到 `arke/common/`（或
  learn 侧改依赖注入）。
- 补 CONTRIBUTING.md + docs/README 索引（内外文档分层），4 个 no-data 算子
  给出数据或在 SSOT 标注原因。

### 调优策略改进（Auto-tuning 方向）
- **把杠杆排序变成搜索先验**：今日证据链（dtype 纪律 ≫ tile ≫ pipeline）应
  编码进 agent 的 action space 排序/prompt——先查 TC 利用率再扫 tile，
  避免 agent 在低杠杆维度烧预算。
- **收敛曲线补样本**：live-agent 轨迹当前太浅（matmul 2 iter），建议每 op
  至少 5 组不同 shape × 3 seed 的 trajectory 入库，才够支撑"收敛效率"结论。
- rationale KB(390 条) 是差异化资产：下一步让 agent 检索 KB 做 warm-start，
  量化"有 KB vs 无 KB"的收敛迭代数差——这是 AI-Native 命题最可发表的实验。

---

## 附：达成比例的计算口径

"~72%"= 五维加权（Benchmark 25%×0.85 + 架构 25%×0.60 + 动态shape 15%×0.70 +
异构 15%×0.35 + 工程 20%×0.75）。压分项集中在"异构实证为零"与"IR 宣称与
承重不符"；两者分别对应 R4 与 R1，均有明确解法。

*审计人：Kitty（主审）+ 3 取证 subagent；全部结论可溯源至文件/命令/当日实测。*

---

## 附二：整改执行记录（R1-R5 全部落地，2026-07-29 同日）

Leon 批「按识别的方案执行」→ 五项建议当日全部实现 + 测试 + commit + push。

| 建议 | 内容 | commit | 量化结果 |
|:--|:--|:--|:--|
| **R2** | 修 mla.py.j2 的 5 处 TC 反模式 + 建模板 TC-纪律 lint 闸 | `d73bfb7` | mla 启用 TC 路径（A/B 确认 err 0.122→0.141 同量级，非回归）；lint 闸 11 tests 扫全模板，防回归制度化 |
| **R3** | softmax/rmsnorm bucket-aware warmup 消动态-shape cliff | `97ddbdb` | **softmax cliff 40.99×→1.33×，rmsnorm 7.22×→2.61×**；6 tests；查明 cliff=每 bucket 首编译（非每 N），同 bucket 复用实证 |
| **R5** | 断 agent↔learn 循环依赖 + MLIR 双目录消歧 + 修架构图幽灵目录 + CONTRIBUTING.md | `d6fefd1` | AST 确认包级环消除；补开源贡献入口 |
| **R1** | README 措辞对齐：删「MLIR Dialect」→「emits upstream linalg/memref/gpu/nvgpu，非自定义 dialect」 | `d6fefd1` | 叙事与代码一致（IR spec 早有 K-H5.1 降格标注） |
| **R4** | 纸面 `ascend_910b()` HardwareModel + dry-run 暴露 4 个 schema 缺口 | `676fe11` | 3 tests 把「抽象能否泛化」变成可追踪契约；缺口清单入 §7.7.1，Ascend 重构成本从实现期挪到设计期 |

**整改后测试基线**：make test（全量）— 见 daily memo 最终数字。TC 纪律闸 + row-scan warmup 闸 + Ascend dry-run 闸共新增 20 个回归测试，把本次审计的三条关键知识（TC dtype 纪律 / bucket 首编译机制 / 硬件抽象缺口）从「人肉记忆」固化为「回归闸」。

**未做（诚实边界）**：
- 隐患1 的「L2 最小真实化」（让 ScheduleIR 字段真正驱动 codegen）是中期重构，需 Leon 批方向，未动。
- R3 的运行时「JIT 过贵→eager 顶住+异步编译」降级策略属 serving 集成层，未做（当前只提供 warmup API）。
- layernorm 等其余 row-scan 未套 warmup（同模式可跟进）。
- dynamic-shape D1→D2 gate 升级是 frozen 层，需跨-run 方差数据齐后报批。

---

## 附三：后续决策与推进记录（2026-07-30，Kitty 自主）

Leon 令「按 Arke 能达到的最佳效果决策并推进」→ 就附二「未做」的 7 项开放问题
自主拍板（决策矩阵见 `docs/audit/2026-07-29-followup-decisions.md`），实现层直接落地，
frozen 层备料等 Leon。

| # | 决策 | commit | 结果 |
|:--|:--|:--|:--|
| **C** | layernorm row-scan warmup（补齐第三个 row-scan norm） | `46f08c7` | cold first-touch 169.8ms→warmed 0.084ms（**2011×**）；`test_rowscan_warmup.py` 7 passed（+2 layernorm） |
| **分母陷阱 fail-loud** | perf CSV 加 `ratio_denominator` 列，显式标注 ratio 分母是哪个 runner | (本次) | 消除「eager 分母比值被误读为 vs-golden」歧义（审计 §1.2 教训）；`test_benchmark_artifacts.py` 5 passed（+1） |
| **A（L2 最小真实化）** | 设计先行 + 一条窄真实链，不做全量重构 | 待续 | 见决策文档 §A 理由 |
| **B（运行时降级）** | 延后到 serving 集成 Phase，先落 API 契约 | 决策 | 见决策文档 §B |
| **D（dynamic-shape gate）** | **frozen 硬停点** — 推荐维持 D1，待方差数据后评估 D2；不擅改 | 备料 Leon | 见决策文档 §D |
| **E（GQA 定阈）** | **frozen 硬停点** — 实测 0.802，推荐锁 stage≥0.30/final≥0.45；不擅改 | 备料 Leon | 见决策文档 §E |
| **F（attention 快照刷新）** | harness 官方口径重跑 attention 族（旧快照 attention 数字过时/曾有 eager 分母陷阱） | (本次) | **flash_attention geomean 0.950**（11 shapes，harness bench_l1，`attention_refresh_2026-07-30/`）；**GQA 0.863**、**cross_attention 1.090**（同日 ad-hoc 对照口径）。全部 vs flash-attn 真分母，K-ATT gate（≥0.50）大幅通过；FA-v4 后 D=128 short-S 修复可见（llama2-7b-512 0.778） |
| **G（FA-v4 micro-opt）** | 尝试 D=64 short-S gap，helps 则留否则诚实放弃 | 待续 | Gate 已过（0.846），锦上添花 |
