# Arke 核心架构与扩展性审计（MLIR/LLVM 生态 + 异构适配 + AI-Native 维度）

> 只读审计 · 2026-07-27 · 范围：`arke/ir/`, `arke/compiler/`, `arke/backend/`, `arke/agent/`, `docs/spec/`, `docs/architecture/`
> 审计透镜：两向 AI-Native（为 Agent 设计 + 用 Agent 建造）+ MLIR/LLVM 生态惯例对标 + 异构泛化真实能力
> 结论口径：诚实区分「设计意图」与「已验证事实」。

---

## 1. Dialect / Pass 设计

### 1.1 IR 分层实际是四层，不是两层 —— 且下两层是「骨架」

`docs/architecture/arke-ir-spec-design.md:47-57` 声明四层 IR：
- **Layer 4 SemanticIR**（"what to compute"，Agent 主接口）— `arke/ir/semantic.py`
- **Layer 3 StrategyIR**（"how to optimize"，Agent 决策）— `arke/ir/strategy.py`
- **Layer 2 ScheduleIR**（"schedule mapping"，自动）— `arke/ir/schedule.py`
- **Layer 1 InstructionIR**（"near-LLVM"，自动）— `arke/ir/instruction.py`

**评估：Semantic→Strategy 两级划分本身合理**，是全项目最干净的架构承诺：
- `semantic.py:32-142` 的 `SymbolicDim`/`TensorDesc`/`ShapeConstraint` 纯数学描述，无优化信息；`strategy.py:30-71` 的 `Decision(kind, params, rationale, level)` 纯优化决策。两者是独立 dataclass，SemanticIR 构造后不可变，Agent 只探索 StrategyIR —— 与 MLIR「计算 dialect（linalg/tensor）vs 变换（transform dialect）」的分工严格对齐（`ir-mlir-mapping.md:13-18`）。
- `strategy.py:71` 的 `level` 字段（1=L1 backend-agnostic / 2=L2 resource / 3=L3 instruction）是渐进式 lowering 的正确编码。L3（`wmma_tile`/`block_threads`/`fma_contract`/`pipeline_stages`，`strategy.py:234-275`）显式标注「仅对 ISA 级后端有意义」，边界清楚。

**隐患：Layer 2/1 是「结构骨架」，真实 codegen 路径完全绕过它们。**
- `arke/compiler/lowering.py:26-56` 的 `strategy_to_schedule` 和 `:59-110` 的 `schedule_to_instruction` 存在，但 `schedule_to_instruction` 产出的 InstructionIR 只是把 loop/placement/resource 转成 `Instruction(opcode="loop.configure"...)` 的**声明性记录**（`lowering.py:76-108`），并且 `:104-108` 在无内容时直接发一条 `nop`。
- **关键证据**：`search_files pattern='schedule_to_instruction|lower_full_stack|InstructionIR' path=arke/backend` → **0 命中**。即四个后端（Triton/MLIR/CUDA-C/LLVM）**没有任何一个消费 ScheduleIR 或 InstructionIR**。它们直接吃 `IRGraph`（下节详述）+ 可选 `StrategyIR`。
- 结论：InstructionIR/ScheduleIR 属于 spec 里「设计意图」，非「已走通的 lowering 环节」。`arke-ir-spec-design.md:19` 自称 Layer 2/1 是 "Implementation Notes for Active Mainline"，实际是 dead-ish 分层。这与 AI-Native 审计透镜里「IR 层是否 real or over-designed（空壳）」正好命中一条。

### 1.2 存在两套并行 IR，是最大的架构耦合味

- `arke/ir/semantic.py` 的 `SemanticIR/Node/ParamRef/NodeRef`（Agent 面向、JSON 友好、四层体系顶层）。
- `arke/ir/graph.py` 的 `IRGraph/IRNode/IRValue`（自称 "Minimal IR for S6 Pass Pipeline"，`graph.py:4-9`）—— **所有后端的 `lower(graph: IRGraph)` 签名吃的是这一套**（`protocol.py:82`）。
- `arke/compiler/passes/builtin.py` 的三个 Pass（ShapeInference/SSAValidation/RationalePreservation）也全部操作 `IRGraph`（`pipeline.py:24`, `builtin.py:32-68`）。
- 而 `arke/compiler/validator.py` 的 `validate_semantic_ir` 操作的是 `SemanticIR`。

**即：校验/合法性分散在两套 IR 上** —— `PassPipeline`(IRGraph) 走 SSA/shape/rationale 校验，`validate_semantic_ir`(SemanticIR) 走另一套结构校验。这是典型的历史遗留双层命名冲突（skill 已���录过 `mlir_emitter.py` 双份文件的同类问题）。对 Agent 而言，「它读写的 SemanticIR」和「编译器实际 lower 的 IRGraph」不是同一个对象，转换缝在 `arke/agent/backends.py:227-254`（临时手搓单节点 IRGraph）。

### 1.3 Pass pipeline 是否闭环

- `arke/compiler/pipeline.py:176-243` 的 `PassPipeline.run` 是干净的：顺序执行、首错即停（`:214-223`）、`ctx.has_errors()` 短路（`:225-234`）、计时+`passes_run` 追踪。**这一段是标准编译器 pass manager 的合格实现**。
- Diagnostics 分级（`pipeline.py:32-48` INFO/WARNING/ERROR + node_id 定位）对 Agent 自纠错友好 —— 命中 AI-Native「V0 静态反馈可恢复」要求。
- **但闭环止于 IRGraph 层**：pass pipeline 做完 shape/SSA/rationale 后，产物 `CompilationResult`（`pipeline.py:128-138`）并不驱动一个 `LoweringPass`/`CodegenPass`。`arke-compiler-infrastructure.md:84-107` 的目标架构画了 `TilingPass/FusionPass/TritonCodegenPass/MLIRCodegenPass` 进 pipeline，**但代码里 pipeline 只有 3 个 analysis pass，没有 transform/lowering/codegen pass**（`builtin.py` 仅 3 类）。codegen 由后端 `.lower()` 在 pipeline 之外单独完成。所以「Pass pipeline 闭环」目前是**半闭环：分析闭环、变换与降级不在 pipeline 内**。

### 1.4 与标准 MLIR dialect 惯例的差距

| MLIR 惯例 | Arke 现状 | 差距 |
|:--|:--|:--|
| Progressive lowering（dialect→dialect 逐级合法化） | 有分层设计，但 L2/L1 无后端消费，实际是 Semantic/Strategy → 直接 backend codegen 两跳 | **中-大**：分层是文档，非运行时 |
| Op interface（`verify()`, `InferTypeOpInterface` 等挂在 op 上） | 校验集中在两个自由函数（`validator.py`, `builtin.py:SSAValidationPass`），op 定义在 `OpRegistry` 里带 `shape_rule`（声明式，`arke-compiler-infrastructure.md:140-151`） | **中**：无 per-op verifier 接口，但声明式 shape_rule 比 if/elif 好 |
| Verifier（结构强制） | `SSAValidationPass`（`builtin.py:72-145`）查 op 存在/重复定义/use-before-def/自环/输出已定义 —— **覆盖了 MLIR verifier 的核心项**；`validate_semantic_ir` 补 symbolic dim 约束（`validator.py:118-160`） | **小**：验证覆盖不错，只是分散 |
| transform dialect 复用 | `strategy_to_transform.py:57-101` 真把 tile/reorder/vectorize 降成 `transform.structured.tile_using_for` 等，直接复用 MLIR transform 基础设施 | **优点**：真正对接了 MLIR 生态 |

**净评**：Semantic/Strategy 分层是项目最强的架构资产，对标 MLIR 计算/变换分工到位；但四层 IR 的下两层徒有其名，且并行的 SemanticIR/IRGraph 双 IR 让「Agent 眼中的 IR」和「编译器降级的 IR」割裂 —— 这是 progressive lowering 惯例的主要缺口。

---

## 2. IR 映射机制（.ak → 各后端）

### 2.1 前端 .ak → 后端的信息无损性

- `.ak` → AST → SemanticIR + StrategyIR：`arke/ir/converters.py`（`ast_to_semantic`/`ast_to_strategy`），`@rationale` 在此被提取（`converters.py:53-61`）。SemanticIR 完全 JSON 可序列化（`semantic.py:49-72,124-139`），round-trip 有 `to_dict/from_dict`，**这一跳无损**。
- **有损点在 SemanticIR → IRGraph**：后端吃 `IRGraph`（`protocol.py:82`），而 Agent 路径经 `arke/agent/backends.py:227-254` 手搓一个**单节点** IRGraph（`# Build a minimal IRGraph for the op`），dtype 硬编码 `"float32"`（`backends.py:238`），输入名靠 `zip(schema_input_names, shapes_keys)` 位置配对（`:247-249`），配不上时退化成 identity 映射（`:251-252`）。**多节点图 / fusion group / symbolic dim / layout 在这条 Agent 实测路径上丢失**。这也是 skill 里记过的 `_shapes_for()` binary-op 覆盖 bug 的同源风险。
- `@rationale` 传递：`RationalePreservationPass`（`builtin.py:148-172`）把 rationale 收进 `ctx.artifacts["rationale_map"]`；Triton 后端把它作为��释写进生成源（`triton_backend.py:130-131` `# @rationale:`）。`ir-mlir-mapping.md:155-163` 声明 MLIR 后端用 `transform.annotate "arke.rationale"` 保留。**rationale 到 codegen 注释无损，但不进二进制/不参与优化决策**（本就是设计如此）。

### 2.2 Tensor Core 利用 —— 真用上了，三个后端各一条真实路径

`search_files pattern='mma|wmma|tensor.?core|tl.dot' path=arke/backend` → **139 命中**。逐后端核实：

- **MLIR-GPU 后端（Phase 3）**：`mlir_gpu.py:127,160,203-205` 走 `nvgpu.mma.sync` 两阶段 lowering，`:203` 甚至硬校验 `if "nvgpu.mma.sync" not in s1: raise`（确保 `vector.contract` 真被分发到 tensor core，不是静默退化）。`mlir_gpu.py:613-634` matmul 默认走 `emit_gpu_matmul_mma`（f16 TC + f32 累加），小 shape 才退标量。**真 TC。**
- **CUDA-C 后端（Phase 4）**：`cuda_c_matmul_templates.py:146-190` `_emit_matmul_tensor_core` 用 `#include <mma.h>` + `wmma::fragment<...>` + 双缓冲 shared memory（`:184-186`）。`cuda_c_attention.py` 有 18 处 mma 命中。**真 WMMA。**
- **LLVM-IR 后端（Phase 5）**：`llvm_wmma.py`（29 命中）用 inline PTX `wmma.load/wmma.mma`（`llvm_wmma.py:63-66`），2×4 warp grid、128×128 tile、NSTAGE ring-buffer 软流水（`:76-98`），并记录了「fragment-level 双缓冲比 staging 慢 31%」的实测教训（`:35-55`）。**真 TC，且是最深度的手写路径。**
- **Triton 后端（Phase 1）**：`tl.dot` 仅在 `matmul.py.j2`(2)、`flash_attention.py.j2`(2)、`batch_matmul.py.j2`(2)、`grouped_matmul.py.j2`(1)、`mla.py.j2`(3) 命中 —— **靠 Triton 自身把 `tl.dot` 编译到 TC**，Arke 不直接控制 fragment。合理（Triton 层不该手写 mma）。

**净评**：Tensor Core 不是纸面，四个后端里三个有 Arke 亲自 emit 的 mma/wmma，一个（Triton）借道 `tl.dot`。这是全项目「已验证事实」最扎实的一块。

### 2.3 Phase3 MLIR (~1.05×) vs Phase4 CUDA-C (1.05×) 的架构证据

- **同一 `ArkeBackend` 协议，不同降级深度**：
  - MLIR：`IRGraph → linalg/scf/gpu dialect → nvvm → PTX/cubin`（`mlir_gpu.py:11-17` 用 `mlir-opt -convert-gpu-to-nvvm -gpu-module-to-binary`，驱动 API 加载），走**标准 MLIR dialect 栈**。
  - CUDA-C：`IRGraph → CUDA C 源 → nvcc --cubin → cuModuleLoadData`（`cuda_c_backend.py:9,490-527`），走**vendor-native DSL**，`cuda_c_backend.py:11-14` 明说这是为了「拿到完整 CUDA 编程模型（cooperative groups, wmma/mma intrinsics）」。
- **数值口径分歧（文档 drift，需修）**：`ir-mlir-mapping.md:204` 写 MLIR「OVERALL geomean **1.14×** cuBLAS」，而 `plan.md:459,469` 和 skill SSOT 都写 **1.05×**（L1 component metric）。同一后端两个数字，属已知 staleness hotspot。任务给的「~1.1× MLIR / 1.05× CUDA-C」与 plan.md 的 1.05×/1.05× 也不完全一致 —— **建议以 plan.md 的 1.05× 为准并统一 mapping 文档**。
- 架构证据结论：两个后端性能接近（都贴着 cuBLAS）**恰恰验证了 backend 协议抽象是有效的** —— 同一 IR 经两条完全不同的降级路径（标准 MLIR vs vendor DSL）都能逼近 cuBLAS，说明性能瓶颈不在 IR 抽象而在 kernel 手艺，这对 Thesis L3（越降越快）是正向证据。

---

## 3. 架构泛化 / 异构适配

### 3.1 protocol.py 扩展缝质量

`arke/backend/protocol.py:71-96` 的 `ArkeBackend` 是 4 方法 `Protocol`（`lower/compile/run/supports_op`）+ `runtime_checkable`；`:101-139` 的 `BackendRegistry` 用 `target_map` 路由硬件字符串；`:146-191` 的 `get_default_registry()` 懒加载注册 Triton/MLIR/CUDA-C/LLVM，import 失败静默跳过（`:163-164` 等）。

**评估：这是全项目最干净的扩展缝，接口抽象本身合格。** 新后端只需实现 4 方法 + 注册 target 串，无需改核心（`protocol.py:16-22` 明确写「if a core refactor is needed to add a backend, that is an architecture smell to fix here」）。四个后端都真实实现了这个协议（`triton_backend.py:88`, `mlir_backend.py:112`, `cuda_c_backend.py:443`, `llvm_backend.py:217`）—— 缝是活的，不是摆设。

**隐患 A：协议只覆盖了 SIMT，未覆盖 SIMD 差异。** 4 方法签名 `lower(graph)/compile(artifact)/run(kernel, inputs)` 完全没有表达：内存层级（NVIDIA shared/register vs Ascend Cube/Vector buffer）、同步原语（`__syncthreads` vs Ascend pipe barrier）、指令粒度（SIMT warp vs SIMD 向量）。这些差异目前全被塞进各后端**内部**（如 `cuda_c_matmul_templates.py` 硬编码 `__shared__`、warp 概念）。协议层是「硬件无关的空壳」，真正的硬件模型没有被抽象出来 —— 所以协议对 Ascend 的「兼容」是**语法兼容（能实现 4 方法）而非语义覆盖（能表达 SIMD 模型）**。

**隐患 B：`lower()` 签名已经开始漂移。** 基协议 `protocol.py:82` 是 `lower(self, graph)`，但 `mlir_backend.py:112` 是 `lower(graph, tile_sizes=None)`，`cuda_c_backend.py:443` 和 `llvm_backend.py:217` 是 `lower(graph, strategy=None)`。三个后端各自加了不兼容的可选参数 —— Protocol 没约束住，`runtime_checkable` 也查不出（只查方法名存在）。StrategyIR 如何进 backend 目前是**每后端各写各的**（CUDA-C 只在 matmul 消费 strategy：`cuda_c_backend.py:462-480`；LLVM 只在 L3-aware ops 消费：`llvm_backend.py:232-236`；Triton **完全不消费 strategy**：`triton_backend.py` 里 strategy/Strategy 0 命中）。这是扩展缝「看起来干净、实则契约松」的隐患。

### 3.2 Phase 2 Ascend 遗留设计与当前 protocol 的兼容性

- **`docs/phase2/` 目录不存在**（`search_files` 报 Path not found，仅有 phase1/3/4/5）。Ascend 遗留设计只活在 `plan.md:375-449` 的 dormant 段落里，非独立设计文档。
- `plan.md:24-31` 的 Thesis L2 定义了可证伪的 kill criterion：**「若 StrategyIR 必须加 ≥3 个架构专属 decision kind（如 `cube_pipeline_stage`, `vector_buffer_double_buffer`）才能表达 Ascend 策略，则 L2 被证伪」**。当前 StrategyIR decision kinds（`strategy.py:34-65`）全是 SIMT/通用概念（tile/reorder/fuse/parallel/place/vectorize/unroll + L3 的 wmma_tile 等）。**没有任何 Ascend 专属 kind** —— 从「未引入污染」角度，protocol 与遗留设计**形式兼容**。
- `arke/` 源码里 Ascend 渗透 = **0 行实现**（`search_files pattern='ascend|910b|cann'` 在 `arke/` 只命中 `protocol.py:10,15,18` 三行文档字符串 + `lowering.py:23`/`env.py` 等的 "cannot" 假匹配）。符合 skill 记录的「Ascend pause 成本 ≈ 0」判断。

**但兼容性是「未被证伪」而非「已验证」**：L2 从未在真实 Ascend 硬件上跑过（`plan.md:30` "L2 remains an open, untested claim"）。protocol 能否真正承接 SIMD，取决于 3.1 隐患 A —— 一旦真上 Ascend，`lower/compile/run` 三方法很可能不足以表达 Cube/Vector 双引擎调度，届时要么撑爆 `run(kernel, inputs)` 的 `inputs` dict（塞硬件专属配置），要么就得改协议（触发它自己定义的 "architecture smell"）。**当前的干净是「没被真实 SIMD 需求压过」的干净。**

### 3.3 「屏蔽异构差异」：设计意图 vs 已验证事实

**诚实评估：目前是设计意图，不是已验证事实。**
- 已验证的是 **SIMT 内跨抽象层**：同一 SemanticIR/StrategyIR 经 Triton→MLIR→CUDA-C→LLVM 四条降级路径都能生成正确且贴近 cuBLAS 的 kernel（Thesis L3 partially validated，`plan.md:468-471`）。这证明了「同一 IR 跨编译栈层」，**不是**「同一 IR 跨硬件架构」。
- 未验证的是**跨硬件架构（SIMT↔SIMD）**：Thesis L2 因无 Ascend 硬件 SKIPPED（`plan.md:31`），`P2-S_FINAL` 的退出标准「same Arke IR → NVIDIA + Ascend」（`plan.md:428`）从未达成。
- `arke-ir-spec-design.md:75-91` 反复用「LLM-Native, not MLIR-Native」措辞���但从未声称已屏蔽异构 —— 项目文档本身是诚实的（用 "open, untested claim"）。真正的风险是**外部叙述**容易把「4 后端跑通」误读成「跨硬件跑通」，二者是不同的命题。

---

## 4. AI-Native 双向命题维度

**(a) 为 Agent 设计的工具链：**
- `list_legal_actions` 是**编译器计算的、shape+hw-aware 的**合法动作空间，非硬编码：`agent/env.py:33-45,95-110` 的 `_enum_tile_candidates` 从算子真实维度推 tile factor 并用 `HardwareProfile` 的 max_threads/warp 过滤（`env.py:100`）。命中 AI-Native「bounded action space 真由编译器算」要求。
- `@rationale` 是**执行强制契约**，非软约定：`agent/tools.py:901-926` 在 `apply_decision` 里对 level≥1 的决策强制非空 rationale，缺失直接返回硬错误（`:919-926`），并给 Agent 可恢复的提示文本。这是把 locked thesis pillar 落到执行层的正确做法。
- V0/V1/V2 反馈：pass diagnostics（`pipeline.py:32-48`，静态 V0）+ `verify_correctness`（V1）+ `compile_and_profile`（V2）为 Agent 迭代循环设计，diagnostics 带 node_id 定位便于自纠错。

**隐患（AI-Native 维度）：** Agent 读写的顶层是 SemanticIR（四层体系），但真正驱动 codegen 的是手搓单节点 IRGraph（`agent/backends.py:227-254`）。Agent 的「决策心智模型」（四层 IR + StrategyIR level）和「实际生效路径」（IRGraph + 每后端各自消费 strategy 的方式，Triton 甚至不消费）之间有落差 —— 一个 Agent 在 Triton 后端上做的 tile/strategy 决策**根本不生效**（`triton_backend.py` 0 处消费 strategy）。skill `references/live-llm-l3-closed-loop-p5s5.md` 已记过同类「decision 只在特定 tool param 下生效」的坑。这是 AI-Native「Agent 决策是否真影响产物」的关键裂缝。

---

## 亮点（Top 5）

1. **Semantic/Strategy 两级分层是教科书级的计算/变换解耦** —— 独立不可变 dataclass、`level` 字段编码渐进降级、与 MLIR linalg/transform 分工严格对齐（`semantic.py` + `strategy.py` + `ir-mlir-mapping.md:13-18`）。
2. **Tensor Core 是真的**：MLIR(`nvgpu.mma.sync`)、CUDA-C(`wmma::fragment`)、LLVM(inline-PTX wmma + NSTAGE 软流水) 三条 Arke 亲手 emit 的 TC 路径，且带实测教训（fragment 双缓冲慢 31%，`llvm_wmma.py:35-55`）。
3. **`ArkeBackend` + `BackendRegistry` 扩展缝语法层干净**：4 方法 Protocol、target 路由、import 失败静默降级，四后端全部真实实现、无核心 refactor（`protocol.py:71-191`）。
4. **`@rationale` 执行强制 + `list_legal_actions` 编译器计算** —— AI-Native「bounded action space + 决策问责」落到了执行层而非文档层（`tools.py:901-926`, `env.py:95-110`）。
5. **Pass manager 本体合格**：顺序执行、首错短路、diagnostics 三级分类带 node 定位（`pipeline.py:176-243`），对 Agent 自纠错友好。

## 架构级隐患（Top 5，含重构方向）

1. **四层 IR 名不副实：Layer 2/1(Schedule/Instruction) 无任何后端消费。**
   证据：`search_files 'InstructionIR' in arke/backend` = 0；`lowering.py:76-108` 的 InstructionIR 只是声明性记录，产不出可执行物。
   *重构方向*：要么把 Schedule/Instruction 降级真接到某一后端（让 progressive lowering 落地），要么在 spec 里诚实降格为「可选中间表示」，不要在架构图上画成主干四层。当前对外叙述「四层渐进降级」经不起代码核对。

2. **双 IR 并存（SemanticIR vs IRGraph）造成 Agent 心智与运行时割裂。**
   证据：后端吃 `IRGraph`(`protocol.py:82`)，Agent 读写 `SemanticIR`，转换靠 `agent/backends.py:227-254` 手搓单节点图（dtype 硬编码 float32、位置配对输入、多节点/fusion/symbolic dim 丢失）。
   *重构方向*：统一到单一 IR（让后端直接吃 SemanticIR，或让 IRGraph 成为 SemanticIR 的正式 lowering 产物而非平行体系）��消灭 `backends.py` 的手搓缝；校验逻辑也随之从两套（`validator.py` + `SSAValidationPass`）合一。

3. **`lower()` 签名漂移 + strategy 消费三后端三样、Triton 零消费 —— 扩展缝契约太松。**
   证据：`lower(graph)` vs `lower(graph, tile_sizes=)` vs `lower(graph, strategy=)`；`triton_backend.py` 中 strategy 0 命中。
   *重构方向*：把 `lower(graph, strategy: StrategyIR | None)` 提为协议正式签名，并定义「后端如何声明它消费哪些 decision kind」的能力查询接口（类似 `supports_op` 的 `supports_decision_kind`）。否则 Agent 在不消费 strategy 的后端上做的优化决策是静默无效的。

4. **协议只抽象了 SIMT，未抽象硬件模型（内存层级/同步/指令粒度）—— 「屏蔽异构」是意图非事实。**
   证据：4 方法签名无任何硬件模型字段；SIMD 差异全塞后端内部；Ascend 从未上过真机（`plan.md:31`）。
   *重构方向*：在 protocol 层引入一个显式的 `HardwareModel` 抽象（内存层级枚举、同步原语类别、执行模型 SIMT/SIMD），让 StrategyIR 的 `place`/`parallel`/`compute` 决策对接抽象内存层级而非隐含 CUDA 概念。这是让 Thesis L2 未来可证伪的前置条件 —— 否则真上 Ascend 时必然触发它自己定义的 "architecture smell"（改协议）。

5. **文档数值 drift 污染架构证据可信度。**
   证据：MLIR geomean `ir-mlir-mapping.md:204` 写 1.14×，`plan.md:459/469` 写 1.05×；`arke-compiler-infrastructure.md:3` 仍标 "Phase 4 in progress"、`protocol.py:12-13` 标 Phase4 🚧/Phase5 future，实际 P4/P5 均 COMPLETE。
   *重构方向*：以 `plan.md` + `benchmark-ops.md` 为 SSOT 做一次 mapping/infrastructure/protocol docstring 的一致性 sweep（skill 已列为已知 hotspot）。架构审计里「1.1× vs 1.05×」这种矛盾会直接削弱对性能结论的信任。

---

## 一句话总结

Arke 的 **Semantic/Strategy 两级分层 + Tensor Core 三后端实证 + `@rationale`/合法动作空间的执行强制**是真资产，AI-Native「为 Agent 设计」在这几处落到了执行层；但**四层 IR 的下两层是骨架、双 IR 并存、扩展缝契约松、硬件模型未抽象**四条决定了「跨硬件屏蔽异构」目前是**设计意图**，已验证的只是「SIMT 内跨编译栈层」（Thesis L3 partial），Thesis L2（SIMT↔SIMD）因无硬件仍是 open/untested claim —— 项目文档对此是诚实的，风险主要在外部叙述的过度解读。
