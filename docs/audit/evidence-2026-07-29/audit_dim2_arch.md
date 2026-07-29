# Arke 维度二+四审计：MLIR/LLVM 生态、Dialect/Pass、IR 映射、架构泛化、异构抽象

> 审计日期：2026-07-29 | 审计范围：维度二（MLIR/LLVM 生态下的 Dialect/Pass 设计、IR 映射）+ 维度四（架构泛化、异构硬件抽象）
> 方法：只报从 `/home/blueyi/workspace/repos/arke` 源码亲眼读到的事实。不确定处标注"需进一步确认"。

---

## 0. 核心结论（TL;DR）

1. **Arke 没有自定义 MLIR Dialect。** 全仓 `.td`/tablegen 文件数量 = **0**。Arke 是纯 Python 端**字符串拼接生成 MLIR 文本**，交给外部 `mlir-opt` / `llc` / `ptxas` lowering。这是"MLIR 文本前端"，不是"MLIR dialect 作者"。文档自己也如实说明（`mlir_emitter.py` docstring：*"Emits MLIR text ... consumed by the CPU lowering pipeline ... JIT-executed via mlir-cpu-runner"*）。
2. **4 层 IR 是"2 层真实 + 2 层骨架"。** SemanticIR(L4)/StrategyIR(L3) 是承重的生产层；ScheduleIR(L2)/InstructionIR(L1) 是"已填充结构骨架，但字段不驱动 codegen"。这一诚实降格已写进 `docs/spec/arke-ir-spec.md §3.4`（K-H5.1，2026-07-29）。核实属实。
3. **IR→MLIR 映射设计干净但非全量无损。** 映射表把 SemanticIR→`linalg`/`tensor`、StrategyIR→`transform`+`scf` 对齐得很到位；但 L2/L3 的 thread/warp 映射、bank_conflicts、register_allocation 等字段在当前不参与 codegen，属"声明未落地"，因此严格意义上**当前不是端到端信息无损**。
4. **Tensor Core 利用：多数已修，但 `mla.py.j2` 仍有 FA-v2 之前的 `.to(tl.float32)` 反模式**——会把 `tl.dot` 逼到 FFMA，TC 闲置。这是审计发现的**具体、可复现的性能缺陷**。
5. **HardwareModel 抽象是真结构化的**（内存层级/同步域/计算单元/对齐约束都是数据模型），protocol + BackendRegistry 是干净的扩展 seam。但**所有具体实例只有 NVIDIA sm_86**（`nvidia_sm86()`），Ascend/AMD 是设计承诺，无任何实例。"异构屏蔽"目前是**设计承诺，非实测**。

---

## 1. 维度二：Dialect / Pass 设计 + MLIR/LLVM 生态

### 1.1 是否有自定义 MLIR Dialect？——**否**

**证据（一手）：**
- `search_files(*.td)` 全仓命中 **0** 个 tablegen 文件。没有 dialect 定义、没有 op 定义、没有 dialect 注册（C++/tablegen）。
- `arke/backend/mlir_emitter.py`（2992 行）纯 Python 字符串拼接产出 MLIR 文本：
  - `memref_type()` / `tensor_type()` 拼 `memref<4x8xf32>` 字符串。
  - `_emit_elementwise()` 拼 `linalg.generic` 文本。
  - 用的都是**上游标准 dialect**：`linalg`、`tensor`、`memref`、`gpu`、`nvvm`、`nvgpu`、`scf`、`transform`。
- `arke/backend/mlir_gpu.py` docstring 明确：lowering 靠外部 `mlir-opt -convert-scf-to-cf -convert-gpu-to-nvvm -gpu-module-to-binary=format=isa`，从输出里正则提取 `assembly = "..."`（PTX 文本），再用 CUDA driver API (`cuModuleLoadData`) JIT 加载。
- `docs/nvgpu-dialect-research.md` 佐证：所有 pass 都是"our mlir-opt build 里已有的" upstream pass（`--nvgpu-optimize-shared-memory`、`-gpu-lower-to-nvvm-pipeline` 等），不是 Arke 自研。

**结论：** Arke 的 MLIR 后端是**"Python codegen → 标准 dialect 文本 → 外部 mlir-opt lowering"**。它复用 MLIR 生态（这是合理工程选择），但**没有**贡献/定义任何自定义 dialect 或 tablegen。任何说 "Arke 定义了自己的 MLIR dialect" 的表述都是**不成立**的。

**值得肯定的一点：** MLIR emitter 确实走到了真 TC 路径 —— 代码里有 `nvgpu.mma.sync` matmul emitter（`mlir_emitter.py` §"Tensor-core (nvgpu.mma.sync) matmul"，L754+），并用 `#nvvm.target<chip="sm_86">` 标注 gpu.module。所以 MLIR 路径不是玩具，是能生成 TC MMA 的。但仅限单 matmul 节点、f32 输入、tile 对齐（多处 `raise NotImplementedError("... single matmul node only / f32 only / tile-aligned only")`）——覆盖面窄。

### 1.2 Pass Pipeline

**证据：**
- `arke/compiler/passes.py`：`ArkePass(ABC)` + `PassContext(semantic_ir, strategy_ir, metadata)`，pass 作用于 **SemanticIR**（如 `ShapeInferencePass`），返回 error list。
- `arke/compiler/passes/base.py`：另一套 `PassPipeline` + `Diagnostic`/`Severity`，作用于旧的 `IRGraph`（S6 遗留）。
- `arke/compiler/pipeline.py`：新管线 `.ak → parse → SemanticIR + StrategyIR → validate → execute`，注释明说"replacing the S6 IRGraph-based PassPipeline"。
- `arke/backend/strategy_to_transform.py`：把 StrategyIR 的 `tile`/`reorder`/`vectorize` decision 转成 `transform.structured.tile_using_for` / `interchange` / `vectorize` 文本，由 `mlir-opt -transform-interpreter` 当预处理 pass 跑。

**判断：**
- **存在两套 pass 基础设施并存**（`passes.py` 作用于 SemanticIR，`passes/base.py` 作用于旧 IRGraph）——有历史包袱/重复，但不致命。
- StrategyIR→transform-dialect 的映射是**真的落地到 MLIR transform 文本**并被 mlir-opt 消费的（`emit_transform_schedule` 生成 `transform.named_sequence @__transform_main`）——这条链在 MLIR 后端是通的。这是 Arke 一个真实亮点：**同一份 StrategyIR 既能驱动 Triton codegen 又能驱动 MLIR transform**（P3-S5 目标，文档 `strategy_to_transform.py` docstring 明述）。
- 但 Pass pipeline 的"高效性"无法从静态源码断言，需 benchmark 数据支撑（本审计未跑）——**需进一步确认**。

### 1.3 4 层 IR 是否逻辑闭环？——**上 2 层闭环，下 2 层是骨架**

**证据（`docs/spec/arke-ir-spec.md §3.4`，K-H5.1 honest downgrade，2026-07-29 原文）：**
> Layers 4 和 3 fully realized... Layers 2 和 1 (ScheduleIR/InstructionIR) **exist as populated structural skeletons but are not yet the load-bearing scheduling substrate**... thread/warp mapping、bank_conflicts、register_allocation、latency-driven software-pipelining 都是 **declared, not decision-driving**. Today the Triton/CUDA-C/MLIR backends make those choices inside their own codegen (e.g. matmul tile heuristic in `matmul.py.j2`), not by consuming ScheduleIR fields.

**代码核实：**
- `arke/compiler/lowering.py`：`strategy_to_schedule()` 和 `schedule_to_instruction()` 确实存在且会 materialize L2/L3 结构（docstring 自称"intentionally minimal but structured"、"Initial Track 4 skeleton"）。
- `arke/ir/schedule.py` / `instruction.py`：`LoopNest`/`MemoryPlacement`/`ResourceBinding`/`Instruction`/`InstructionBlock` 数据类齐全，有完整 to_dict/from_dict（可序列化）。**结构在，但字段是死数据**。
- 反向印证：各 backend 模板（`matmul.py.j2` 的 tile 是 Triton `@triton.heuristics` + constexpr，`flash_attention.py.j2` 的 launch config 是自己的 `_FA_CFG_CACHE`）——调度决策**在 backend 内部**，不读 ScheduleIR。

**"诚实降格"的实际影响：**
- **正面**：spec 明确标注 `[realized]`/`[skeleton]`/`[Phase-future]`，agent 读 IR 不会被误导以为改 ScheduleIR 能改代码。这是负责任的。
- **负面**：宣传中的"4 层 IR"实际是 **"2 层功能层 + 2 层未通电的展示层"**。L2/L3 目前是**架构占位符**，不是工作中的编译中间层。若对外宣称"完整 4 层 IR 编译栈"会**夸大**。真实描述应是："2 层 agent-facing IR（Semantic/Strategy）驱动生产，另有 2 层已定义但未接入 codegen 的下层 IR 骨架（future work）"。

---

## 2. 维度二：IR 映射无损性 + 硬件并行暴露

### 2.1 前端 DSL → 硬件 IR 是否信息无损？——**语义层无损，调度层未落地**

**证据（`docs/spec/ir-mlir-mapping.md`）：**
- SemanticIR→MLIR 映射表完整：`kernel_id→func.func`、`params→func args (tensor<...>)`、`matmul→linalg.matmul`、dtype 一一对应（f16→f16→half 等）。**语义层映射是无损的、可逆的**（有 matmul 的正反向示例）。
- StrategyIR→`transform` dialect 映射表也在（§3.1，decision kind → transform op），`@rationale` 用 `transform.annotate` 保留。

**但：**
- ir-mlir-mapping.md 自标 "Version 1.0 / Phase 1 specification / enabling **future** Phase 3 MLIR backend" —— 是**规格文档**，不代表全部字段都已在 codegen 里落地。
- 结合 §3.4 的 K-H5.1 降格：L2/L3 承载的调度信息（thread/warp/bank/register）当前**不流入** codegen。所以从 `.ak` 到最终 SASS，**调度决策的信息通道是断的**（backend 各自重做决策）。
- **结论：** 语义/数学层信息无损；**优化/调度层不是端到端无损**（信息在 L3→backend 之间"绕过" L2/L1 骨架）。严格说当前 pipeline **不满足"全字段无损"**，只满足"语义无损"。**需进一步确认**：是否有测试断言 SemanticIR round-trip 无损（本审计未查测试）。

### 2.2 Tensor Core 暴露 —— 逐模板核查（关键发现）

结合 `docs/kestrel/k-att-plan.md §6` 的 FA-v2 教训（模板曾因 `.to(float32)` 把 `tl.dot` 逼到 FFMA 让 TC 全程闲置；改 fp16 dot + fp32 accumulate 后 FA geomean **0.496→0.846**），逐一检查带 `tl.dot` 的模板：

| 模板 | dot 写法 | TC 状态 | 判断 |
|:--|:--|:--|:--|
| `flash_attention.py.j2` | `q` 保持 fp16；`tl.dot(q, tl.trans(k), out_dtype=tl.float32)`；`tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)` | ✅ **已修（FA-v2）** | 注释明确"loading fp32 forced the dot onto FFMA (no TC)"，已走 HMMA |
| `matmul.py.j2` | `a`/`b` 保持原 dtype load，`acc=fp32`，`acc += tl.dot(a, b)` | ✅ OK | fp16 输入 + fp32 累加，走 TC |
| `batch_matmul.py.j2` | `a`/`b` 原 dtype，`acc += tl.dot(a, b, allow_tf32=False)` | ✅ OK（fp16）/ ⚠️ f32 时无 TF32-TC | fp16 输入走 TC；若 f32 输入则 `allow_tf32=False` 禁掉 TF32-TC → FFMA |
| `grouped_matmul.py.j2` | 同上 `allow_tf32=False` | ✅ OK（fp16）/ ⚠️ f32 时无 TF32-TC | 同上 |
| **`mla.py.j2`** (第一个 kernel) | `kvc = tl.load(...).to(tl.float32)`；`w = tl.load(...).to(tl.float32)`；`acc += tl.dot(kvc, w)` | ❌ **未修 —— FA-v2 之前的反模式** | 两个操作数都被 `.to(tl.float32)`，`tl.dot` 会被逼到 FFMA，**TC 全程闲置**。这正是 k-att-plan §6 描述的病根 |
| `mla.py.j2` (attn kernel) | `q`/`k`/`v` 均 `.to(tl.float32)` 后 `tl.dot(q, tl.trans(k))` / `tl.dot(p.to(v.dtype), v)` | ❌ **同类问题** | Q/K/V 提前转 f32，dot 走 FFMA，TC 闲置 |

**关键发现（本审计新增，非既有文档记录）：**
- **`mla.py.j2`（Multi-head Latent Attention，DeepSeek 类模型的核心算子）存在与 FA-v2 修复前完全相同的 TC 未利用缺陷**：`.to(tl.float32)` 把 dot 操作数逼到 f32，`tl.dot` 落到 FFMA，Tensor Core 全程闲置。
- 按 FA 的经验（0.496→0.846，≈1.7×），mla 修同样的病（保持 fp16 输入 + `out_dtype=tl.float32`）预期有可观提速。**建议立即修复**：把 MLA 两个 kernel 里的 `.to(tl.float32)` load 改为保持 fp16、`tl.dot(..., out_dtype=tl.float32)`、accumulator 保持 fp32。
- **次要**：`batch_matmul`/`grouped_matmul` 的 `allow_tf32=False` 对 fp16 输入无害（fp16 本来就走 HMMA），但若用 f32 输入会禁掉 Ampere 的 TF32 TC 路径 → 退到 FFMA。**需进一步确认**这两个算子在 benchmark 里是否用 f32 输入；若是，`allow_tf32=False` 是 TC 未利用点。

---

## 3. 维度四：架构泛化 + 异构硬件抽象

### 3.1 protocol.py —— 扩展 seam 是真的（设计层面）

**证据（`arke/backend/protocol.py`）：**
- `ArkeBackend` Protocol：4 个核心方法 `lower(graph, hw=None) → BackendArtifact`、`compile → CompiledKernel`、`run`、`supports_op` + `capabilities`。`@runtime_checkable`。
- `BackendRegistry`：`register(backend, targets)` / `get(target)` / `list_backends` / `list_targets`，target 字符串路由到 backend 名。
- `get_default_registry()` 注册 4 个 backend：Triton(`nvidia_*`)、MLIR-GPU(`mlir_gpu`,`mlir`)、CUDA-C(`cuda_c`)、LLVM(`llvm`)。import 失败静默跳过（可选依赖友好）。
- `BackendCapabilities`：`tensor_core`/`async_copy`/`max_pipeline_stages`/`supported_dtypes` —— 供 engine 剪枝 legal action space（K-H2），backend 不支持的优化动作不会被 agent 选中。
- docstring 明确 Phase-2 Ascend surface "PAUSED but must stay pluggable"，"Do NOT delete the paused Phase-2 surface — keep it dormant"。

**判断：** 扩展 seam **在设计上是干净、成立的** —— 加一个新 DSA 后端理论上只需 (1) 实现 4 方法 Protocol，(2) 注册 target 字符串，(3) 提供一个 HardwareModel 实例。**无需改核心**。这是良好的架构。

### 3.2 hardware.py —— HardwareModel 是真结构化抽象

**证据（`arke/backend/hardware.py`）：**
- `HardwareModel` 是 backend-agnostic **纯数据模型**（docstring："no codegen, no Triton/MLIR/CUDA specifics"），包含：
  - `memory_levels: tuple[MemoryLevel]`：register/shared/l2/global，带 scope(thread/block/device)、size、bandwidth、latency。
  - `sync_domains: tuple[SyncDomain]`：warp(width=32,barrier_free)、block、device。
  - `compute_units: tuple[ComputeUnit]`：`simt` 和 `tensor_core`（含 supported_dtypes、peak_tflops）。
  - `alignment: AlignmentConstraints`：warp_size、`mma_tile=(16,8,16)`、vector_width、shared_bank_bytes。
  - 便捷查询：`has_tensor_core()`、`shared_memory_bytes()`、`sync_domain()` 等。
- **这确实抽象了内存层级 / 同步域 / 计算单元 / 对齐约束的差异**（正是异构硬件的关键差异维度），设计上不是 NVIDIA 硬编码。StrategyIR legal-action 生成器消费它来 bound tile factor / pipeline stage / TC 可用性。

**但（关键）：**
- **唯一的具体实例是 `nvidia_sm86()`（RTX 3060, SM 8.6）**，`DEFAULT_HARDWARE` 就是它。全仓**没有** Ascend/AMD/任何非 NVIDIA 的 HardwareModel 实例。
- docstring 自认："TritonBackend / MLIRGPUBackend / CudaCBackend / LLVMBackend **all target the same RTX 3060 SM 8.6 model today**"。**4 个 backend 全指向同一块 NVIDIA 卡**。
- Ascend 是 docstring 里的"extensibility seam"承诺（"a future Ascend/AMD supplies its own HardwareModel instance ... with no change to the model schema"）——**无代码**。

**结论：** HardwareModel **schema 层面**确实抽象了内存/指令/执行差异，不是 NVIDIA-only 硬编码的**结构**；但**实例层面 100% 是 NVIDIA sm_86**。因此"异构屏蔽 NVIDIA vs Ascend" **是设计承诺，未经任何非 NVIDIA 硬件验证**。抽象的"可扩展性"是纸面成立、实测为零。能否"无缝扩展到任意 DSA"—— **架构上不阻塞，但从未被证伪/证实，需真实第二个后端落地才能确认**。

### 3.3 `arke/backend/` —— 真实现 vs stub 清单

| 文件 | 性质 | 证据 |
|:--|:--|:--|
| `protocol.py` | ✅ 真实现（抽象层） | Protocol + Registry + Capabilities，完整 |
| `hardware.py` | ✅ 真实现（仅 NVIDIA 实例） | HardwareModel + `nvidia_sm86()`；无其他芯片 |
| `triton_backend.py` | ✅ 真实现（Phase 1 CLOSED） | 生产路径，Triton JIT |
| `triton_codegen.py` + `triton_templates/*.j2` (25 模板) | ✅ 真实现 | Jinja2 模板真产 Triton kernel；含 FA-v2 TC 修复（mla 除外） |
| `mlir_emitter.py` (2992 行) | ✅ 真实现（窄覆盖） | 产 linalg/gpu/nvgpu/transform MLIR 文本；含 nvgpu.mma.sync TC 路径；但多处 `NotImplementedError`（single matmul / f32 only / tile-aligned only） |
| `mlir_gpu.py` (859 行) | ✅ 真实现 | mlir-opt→PTX→CUDA driver launch，RTX 3060 验证过 matmul bit-correct |
| `mlir_backend.py` | ✅ 真实现（CPU 路径） | mlir-cpu-runner JIT |
| `cuda_c_backend.py` + `cuda_c_*.py` (10+ 文件) | ✅ 真实现（Phase 4 COMPLETE） | nvcc 编译 CUDA C |
| `llvm_backend.py` (719 行) + `llvm_*.py` (9 文件) | ✅ 真实现（Phase 5 COMPLETE） | LLVM IR→llc→ptxas→driver |
| `mock_backend.py` | ✅ 真实现（测试用） | CPU SemanticInterpreter (PyTorch eager)，无 GPU |
| `strategy_to_transform.py` | ✅ 真实现 | StrategyIR→transform dialect 文本 |
| `gpu_tuning.py` / `kernel_cache.py` / `mlir_ops.py` | ✅ 真实现（辅助） | launch policy / 缓存 / op catalog |
| **Ascend backend** | ❌ **不存在**（PAUSED） | protocol docstring 承诺的 seam，无任何文件/类 |
| **AMD / 其他 DSA backend** | ❌ **不存在** | 仅 docstring 提及 |
| **`arke/backends/mlir/`（复数目录）** | ⚠️ 疑似并行/遗留 | 另有一套 `MLIREmitter`/`register_mlir_ops`/`bl1_matmul`，与 `arke/backend/mlir_*` 并存，**需进一步确认**是否重复/废弃 |

**要点：** `arke/backend/` 下**没有 stub 后端** —— 4 个 NVIDIA 路径后端（Triton/MLIR/CUDA-C/LLVM）全是真能跑的实现，mock 是测试后端。**唯一"缺失"是异构（非 NVIDIA）后端**：Ascend/AMD 只有 protocol 里的"预留 seam"注释，零代码。所有已实现后端**同指一块 NVIDIA sm_86**。

---

## 4. 综合评价

### 做得好的（真实事实）
- **诚实**：K-H5.1 对 L2/L3 骨架的降格、AGENTS.md 里"v1.0.0 NOT cut — NVIDIA-only coverage far from release level"，都是难得的自我诚实。审计核实这些自述**属实**。
- **扩展 seam 干净**：protocol + BackendRegistry + HardwareModel（结构化数据模型）是良好架构，加后端不需改核心。
- **StrategyIR 双路复用**：同一 StrategyIR 既驱动 Triton 又能出 MLIR transform 文本，是真链路。
- **TC 纪律（大部分）**：FA-v2 教训已固化到 flash_attention/matmul/batch_matmul/grouped_matmul。

### 需要修正的（含夸大风险）
1. **"自定义 MLIR Dialect" —— 不成立。** Arke 是 MLIR 文本前端，复用 upstream dialect，`.td` 文件为 0。对外表述须改为"生成标准 MLIR dialect 文本供 mlir-opt lowering"。
2. **"4 层 IR 编译栈" —— 半真。** 实为 2 层承重 + 2 层未通电骨架。真实描述见 §1.3。
3. **"异构硬件屏蔽 / 泛化到任意 DSA" —— 设计承诺，实测为零。** 唯一 HardwareModel 实例是 NVIDIA sm_86，4 个后端全指同一块卡，Ascend/AMD 零代码。可扩展性纸面成立、未验证。
4. **IR 映射"无损" —— 仅语义无损。** 调度层字段绕过 L2/L1 骨架，非端到端无损。

### 具体可操作缺陷（本审计新发现）
- 🔴 **`mla.py.j2` 两个 kernel 的 `.to(tl.float32)` 反模式** —— TC 全程闲置，正是 FA-v2 修复前的病根。建议按 FA-v2 方案立即修（预期显著提速，参考 FA 0.496→0.846）。
- 🟡 **`batch_matmul.py.j2` / `grouped_matmul.py.j2` 的 `allow_tf32=False`** —— 若这些算子用 f32 输入，会禁掉 Ampere TF32 TC。需确认 benchmark 输入 dtype。
- 🟡 **两套 pass 基础设施并存**（`passes.py` on SemanticIR vs `passes/base.py` on IRGraph）——历史包袱，建议收敛。
- 🟡 **`arke/backend/` vs `arke/backends/mlir/`（单复数两个目录）** —— 疑似重复 MLIR emitter，需确认是否遗留。

### 未能从静态源码断言（需进一步确认）
- Pass pipeline / lowering 的"高效性"（需 benchmark）。
- 是否有 SemanticIR round-trip 无损性测试。
- `arke/backends/mlir/` 是否废弃。
- `batch_matmul`/`grouped_matmul` benchmark 实际输入 dtype。
