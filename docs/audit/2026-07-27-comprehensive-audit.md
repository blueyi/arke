# Arke 项目综合审计报告 — 目标达成诊断与架构级质量提升建议

**日期**: 2026-07-27
**审计方法**: 3 路并行子审计（benchmark 数据闭环 / 核心架构与异构 / shape 泛化与工程落地）
+ 主审计综合裁决。全程只读，未干扰运行中的生产 bench。
**子报告**: 本目录 `2026-07-27-{benchmark,architecture,engineering}-audit.md`（含全部文件行号证据）
**一手活体证据**: 当天 OT4 golden 重审事件全程（FlagGems 5.0.0 SDPA 非 fused 证伪 →
flash-attn 上位 → Arke attention 5-6× 差距首次暴露），作为审计结论的实证样本。

---

## 一、目标达成诊断（总评）

### 目标达成比例预估（按项目自身宣称的目标口径）

| 维度 | 达成度 | 一句话裁决 |
|:---|:---:|:---|
| L1 单算子生成（NVIDIA/Triton） | **~90%** | 42/46 op 有真实现+数据；4 op（swiglu_packed/permute/MLA/paged）declined 却计 1.0 分 |
| L2 融合收益 | **~85%** | vs Liger 融合 golden 2.33×（45 shape）是全项目最硬数据；5 个融合 op 缺 Arke 行 |
| L3 Agent 自治闭环 | **~75%** | live-agent gate 5/5、never-worse 守卫、@rationale 全链路；但收敛效率无曲线数据 |
| 多后端（Triton/MLIR/CUDA-C/LLVM） | **~80%** | 4 后端真实现、TC 三路径真 emit；但双 IR 并存、Schedule/Instruction 层无消费者 |
| 异构 DSA 泛化（含 Ascend） | **~35%** | 扩展缝干净（语法层），但硬件模型未抽象；SIMT↔SIMD 是**未验证命题**，项目文档对此诚实 |
| 动态 shape / 生产鲁棒性 | **~40%** | SymbolicDim 停在 IR 声明层 backend 零命中；无 bucketing；autotune 首调悬崖未量化 |
| 开源就绪度 | **~55%** | LICENSE/CI/docs 齐；缺 CONTRIBUTING/RFC/治理多元化；v1.0.0 自评 DEFERRED 是清醒的 |

**加权总评：约 70%**——作为"NVIDIA-scoped 的 AI-First 算子生成与调优研究工程"已接近完成且质量高；
作为"面向任意异构硬件的通用工具链"仍处于架构预留阶段。项目自身的 gate 治理与
v1.0.0 DEFERRED 决定表明内部认知与本审计一致，这本身是成熟度的标志。

### 最突出的技术亮点（数据支撑）

1. **L2 融合是最硬的真实收益**：Arke 融合 kernel vs Liger 融合 golden geomean **2.33×**
   （45 shape），vs eager 分离 **3.24×**。这是"AI 生成算子超过人工产线库"的直接证据。
2. **同后端公平性治理独一无二**：Golden ladder（P0-P5 + LADDER_PREFERENCES pin 机制）、
   typed row（unsupported/skipped/oom/timeout 全部带证据落盘）、memory preflight。
   当天 OT4 事件即范例：发现 golden 名不副实（FlagGems bmm 分解）→ RFC → 换 golden →
   重测，全链路 4 小时闭环，且旧结论以脚注保留而非抹除。
3. **退化控制成熟度罕见**：codegen 异常→interpreter fallback（有测试）、GoldenUnavailable
   P5 守卫（永不拿被测系统当 oracle）、watchdog、PlateauEarlyStop、V0/V1/V2 验证闸。
4. **Tensor Core 非口号**：MLIR `nvgpu.mma.sync`、CUDA-C `wmma::fragment`、LLVM inline-PTX
   wmma+软流水，三条 Arke 亲手 emit 的路径（非调库），Phase3/4 各 46 op 全绿有 gate 数据。
5. **测试与 SSOT 工程纪律**：25k LOC tests（1408 个 test 函数 ≈2534 pass）、op 目录
   SSOT 单点治理 + 一致性测试强制、slim-launch 把 softmax/reduce/layernorm/bmm 从
   0.85-0.97 拉到 >1.0（vs FlagGems）的可复现记录。

### 性能优势区间与劣势区间（基于 PERF_ALL 1629 行）

- **优势**: OT1 reduce/norm 族（vs FlagGems 1.2-1.5×，rmsnorm 9×）、OT0 elementwise
  （1.3× 带）、OT3 融合（2.3×）、小-中 shape host dispatch（slim-launch 后 32-39µs
  反超 FlagGems 42-58µs）。
- **劣势**: **attention 全族 vs 真 fused 基线 0.17-0.24**（即慢 4-6×，flash-attn 上位后
  首次诚实暴露）；split 0.207（eager 零拷贝 view vs Arke 物化，测量语义问题大于性能问题）；
  matmul 个别小 shape 0.76-0.88 波动带。

---

## 二、架构级隐患诊断（5 项，按严重度排序）

### 🔴 H1 — 双 IR 并存：Agent 世界与后端世界靠手搓转换桥接
**证据**: Agent 读写 SemanticIR/StrategyIR；后端消费 IRGraph；两者间是单节点图手工转换，
多节点/fusion/dtype 传播有损（architecture-audit §2）。
**风险**: 这是"信息无损映射"命题的最大裂缝。L2 融合已经绕过该桥（专用路径），
未来任何多算子子图工作（L3 E2E、图级调优）都会撞上。
**重构路径**:
1. 短期：为 IRGraph 增加 `from_semantic(sem_ir, strategy)` 官方构造器，废除散落的
   ad-hoc 转换，把 dtype/stride/fusion 边信息补进转换契约并加 golden 测试
   （SemanticIR→IRGraph→SemanticIR 往返等价）。
2. 中期：二选一——(a) IRGraph 降格为 Triton 后端私有中间体，其余后端直接吃
   SemanticIR+StrategyIR；(b) IRGraph 升格为第三层正式 IR 并写进 arke-ir-spec。
   当前"半正式"状态是技术债的根源。

### 🔴 H2 — 后端协议只抽象了 SIMT，硬件模型是隐式的
**证据**: `ArkeBackend.lower()` 签名已漂移（三个后端各自加了私参）；内存层级/同步原语/
指令粒度全部内化在各后端 codegen 内部；Triton 后端对 strategy 零消费（architecture-audit §3）。
**风险**: Ascend（SIMD、显式 DMA、cube unit）恢复时，现有协议提供的只有"注册一个类"
的语法便利，实际每个后端都要从头理解硬件——"屏蔽异构差异"退化为口号。
**重构路径**:
1. 把 `lower(sem_ir, strategy, hw: HardwareModel)` 提为协议正式签名，消除漂移。
2. 引入显式 `HardwareModel` 抽象（内存层级树 + 同步域 + 计算单元描述符 + 对齐/容量
   约束），先用 NVIDIA 填充实例并让 **StrategyIR 合法动作生成器消费它**（决定 tile
   上限、stage 数、TC 可用性），这样 Ascend 只需提供新 HardwareModel 实例 + codegen。
3. 增加 `backend.capabilities()` 能力查询（支持哪些 decision kind），让 engine 在
   动作空间生成期就裁剪掉后端不支持的分支——同时解决"Triton 零消费 strategy"问题。

### 🔴 H3 — 动态 shape 是声明性支持，生产会撞 autotune 悬崖
**证据**: SymbolicDim/ShapeConstraint 在 backend 中零命中；matmul `@triton.autotune(key=["M","N","K"])`
每个新 (M,N,K) 触发全量扫描；JIT 开销无系统量化（engineering-audit §1）。
当天活体证据：reduce 族 N>65536 静默截断 bug 潜伏至 tier-3 sweep 才暴露——tier 网格
之外就是无人区。
**风险**: Llama3 变长推理 / SD 变分辨率场景下，每个新 shape 首调付出全量 autotune 代价，
且无 bucketing 时 cache 无界增长。
**重构路径**:
1. 为 matmul/bmm 把 autotune key 改为 bucketed key（`next_pow2(M), next_pow2(N), K`），
   一次性消除同 bucket 内的重复扫描；配 probe 量化首调开销（现有 probes 只测稳态）。
2. 让 SymbolicDim 真正 lower：模板层已有 in-kernel stride 推导（slim-launch 成果），
   顺势把"contiguous fast-path kernel 对 seq 维免特化"制度化——reduce 族 TILE 循环
   （c2df77b）就是现成模式，推广到 softmax/norm 即可承接大部分动态 seq 场景。
3. 新增 dynamic-shape bench track（同 op 连续变 shape 测首调+稳态曲线），把
   "Performance Cliff"从推测变成有数据的 gate。

### 🟡 H4 — 无数据默认满分：4 个 declined op 计 score=1.0
**证据**: swiglu_packed/permute/MLA/paged_attention 全部 `get_fn declined`（Arke 无实现），
summary.json 仍打 1.0（benchmark-audit §2）。
**风险**: 覆盖叙事被污染——"46/46 op"的 gate 口径里混着 4 个无数据满分，外部审查
（或未来的你）会把它当真实覆盖。这与项目一贯的 honest-finding 文化直接冲突。
**重构路径**: `benchmarks/artifacts.py::write_summary` 把 declined-only op 的 score 改为
`None` + 独立 `no_data_ops` 列表；gate 聚合层显式区分"measured 1.0"与"no evidence"。
一行逻辑的修复，叙事诚实度收益极大。**建议本周内做**。

### 🟡 H5 — 四层 IR 的下两层是骨架；收敛效率无数据
**证据**: Schedule/Instruction IR 无任何后端消费（搜索命中=0）；L3 只有 budget 上限与
终态 geomean，无 per-iteration 收敛曲线（architecture-audit §1, benchmark-audit §3）。
**风险**: spec 宣称四层，实际两层在跑——文档与实现的差距会误导贡献者；调优"收敛效率"
命题（本审计任务点 1.3）目前无法用数据回答。
**重构路径**: (a) 诚实降格：spec 标注 Schedule/Instruction 为 Phase-future，或
(b) 真接降级：让 LLVM 后端的软流水/寄存器分配决策显式经过 ScheduleIR。
收敛数据侧：ArkeEnv 已有 trajectory 记录，补一个 `--emit-convergence-csv`（iteration
vs best-so-far ratio），一次跑 3 op 即可产出首批收敛曲线，成本半天。

---

## 三、量化补充：当天 OT4 事件对审计命题的直接回答

任务书问"与业界标杆相比处于什么水平、是否存在异常性能回退"——当天数据给出了
最诚实的回答样本：

| 事件 | 审计含义 |
|:---|:---|
| FlagGems 5.0.0 SDPA 被 profiler 证伪为 bmm 分解（非 fused） | 基线库本身会退化；golden ladder 的可重审性（pin 机制+RFC）是本项目护城河 |
| flash-attn 上位后 Arke attention 差距从"看似 >1.0"变为 0.17-0.24 | 之前的 OT4 优势部分是**弱分母幻觉**；分母诚实后差距 4-6× —— 这不是回退，是测量修正 |
| GQA@8k 假 mismatch（坏参考 NaN） | correctness oracle 与 perf 分母耦合于同一 golden 的设计，在 golden 自身失效时会产生假信号；建议 correctness 参考与 perf 分母允许分离配置 |
| flash-attn wheel ABI 踩坑（2.8.3 双变体均不匹配 torch2.6+cu124） | 基线供应链脆弱性应文档化进 operator-source-registry |

**Arke attention 4-6× 差距的技术归因**（供下一轮优化排期）: Arke L1 attention 是
非 flash 的 tile 模板（物化 score 或分段 softmax），缺 online-softmax + K/V 流水;
在 sm_86 上补齐 flash-style 模板（online softmax + 双缓冲 K/V tile + TC dot）
是唯一能收敛该差距的路径——工作量大（估 2-4 周),但 LLVM 后端 Phase5 的软流水
经验可直接迁移。建议列为 L1 天花板迭代第二季主线。

## 四、结论

项目以"NVIDIA 单目标 + Agent 全流程开发"的口径衡量是一次高完成度的验证：
gate 治理、honest-finding 文化、L2 融合收益与四后端 TC 路径都是真材实料。
三大结构性欠账——双 IR、隐式硬件模型、声明性动态 shape——决定了它距离
"任意 DSA 无缝扩展"的宣称还有一个架构迭代周期。建议的优先序：
**H4（一行诚实修复，本周）→ H3.1+H5 收敛 CSV（低成本高信息）→ H1（下个大版本
的地基）→ H2（Ascend 恢复前必须完成）→ attention flash 模板（性能主线）**。
