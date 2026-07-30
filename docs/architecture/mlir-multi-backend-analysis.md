# MLIR 多硬件后端：技术挑战、解法、可复用业界工作

**状态：** 架构分析（Kitty 2026-07-30，Leon 问题 msg 1532321993112813599）
**范围：** Arke 在 MLIR 层扩展到多硬件后端的技术路线判断
**前置事实核对：** 基于 `arke/backend/mlir_gpu.py` / `mlir_backend.py` / `hardware.py` / `protocol.py` 实际代码，非记忆推断

---

## 0. 先厘清 Arke MLIR backend 的真实现状

「MLIR 层支持多硬件」这句话要先落到 Arke 的实际代码状态，否则会误诊。核查结论：

| 事实 | 证据 | 含义 |
|:--|:--|:--|
| MLIR backend 已 COMPLETE 但 **NVIDIA-only** | `mlir_gpu.py` pipeline 硬编码 `-convert-nvgpu-to-nvvm -convert-gpu-to-nvvm -gpu-module-to-binary` | 是"能 lower 到 NVVM 一条腿"，非多腿 |
| **无自定义 Dialect** | `find arke -name '*.td'` = 0 个 | Arke 是 **upstream MLIR 的消费者**，不是方言定义者（走标准 `scf`/`gpu`/`nvgpu`/`nvvm`） |
| HardwareModel 抽象已建但仅一个实例 | `hardware.py` 有 `SyncDomain`/`warp_size`，但只实例化 `nvidia_sm86`；注释标 `SCHEMA MISFIT #1: CUDA-ism, no Ascend analog` | 抽象骨架在，但 NVIDIA 假设已泄漏进 schema |
| CPU JIT correctness path 真实存在 | `mlir_backend.py` `mlir-cpu-runner -e main`，linalg → CPU JIT | 有现成的**硬件无关 golden 源** |
| 扩展缝干净 | `protocol.py` `ArkeBackend` 4-method Protocol + `BackendRegistry` | 设计上"加 backend 无需重构核心" |

**真问题定义**：从"同一 Arke-IR 只能 lower 到 NVVM"变成"同一 Arke-IR 能 lower 到
`{nvvm, rocdl, spirv, 华为 CCE}` 多个 target dialect"。以下按挑战拆解。

---

## 1. 挑战：后端 lowering 分叉 `gpu → {nvvm, rocdl, spirv}`

**难点**：`mlir_gpu.py` 硬编码了 `-convert-gpu-to-nvvm`。多硬件要求同一份 `gpu`
dialect IR 分派到不同 target dialect——AMD 走 `-convert-gpu-to-rocdl`，Intel/Vulkan
走 `-convert-gpu-to-spirv`，各自的 binary 生成（`gpu-module-to-binary` 的 target
attribute）也不同。

**解法**：把硬编码 pass list 改成 **HardwareModel 驱动的 pass 选择器**——backend
声明 `target_dialect`，pipeline 按它组装。这与本项目已验证的"L2 ScheduleIR narrow
driver"（`docs/audit/2026-07-30-A-l2-narrow-driver.md`）同一模式：结构化 target
描述**真实驱动** codegen，而非各 backend 硬编码。

**可复用业界工作**：
- **upstream MLIR `gpu` dialect + `GPUToROCDL` / `GPUToSPIRV` pass**——生产可用
  （IREE、Triton 都在用），Arke 无需自写 lowering，只需**正确编排** pass。
- **`gpu-module-to-binary` 统一 target-attr**（`#nvvm.target` / `#rocdl.target`）——
  MLIR 20 已统一，一个 API 覆盖多 target 的 PTX/HSACO 生成。

---

## 2. 挑战：HardwareModel schema misfit（抽象泄漏）

**难点**：代码已诚实标了缺口——`compute_capability` 是 CUDA-ism，无 Ascend 类比。
SIMT（NVIDIA warp / AMD wavefront）与 SIMD/达芬奇（华为 NPU cube+vector 单元）的
资源模型根本不同，硬塞进 flat dataclass 必然泄漏。

**解法**：HardwareModel 分层——
- **通用层**：memory hierarchy（字节数）、并行度上限、支持的 dtype。
- **arch-family 层**：SIMT 的 `warp_size`/`shared_mem` vs SIMD 的 `vector_width`/
  `cube_dim`，用 **capability-set / tagged-union** 而非 flat 字段。

StrategyIR 的 legal-action 生成读 capability-set，天然屏蔽当前 target 不支持的动作
（如非 TC 硬件不出 `mma` 动作）。Arke 现有的 `SyncDomain(warp/block/device)` 已是
正确方向的雏形，需扩成 family-aware。

**可复用业界工作**：
- **IREE `#iree.gpu.target` / TargetAttr**——业界最成熟的多硬件 target 抽象，
  结构化了 wgp（workgroup processor）、mma intrinsics、subgroup size，覆盖
  NVIDIA/AMD/Apple。**Arke HardwareModel 应照它重构，而非从零猜。**

---

## 3. 挑战：TensorCore 类硬件原语可移植性

**难点**：Arke 现用 `nvgpu.mma.sync` / `ldmatrix` / `cp.async`——全 NVIDIA 专有。
AMD 是 `amdgpu.mfma`（matrix core），华为是 cube 指令。同一 matmul IR 到不同硬件，
MMA 原语完全不同。

**解法**：Arke-IR 层保持"矩阵乘意图"抽象（见 `docs/spec/ir-mlir-mapping.md`），把
MMA 原语选择**下沉到 lowering 最后一跳**，由 HardwareModel 的 mma-intrinsic
capability 决定。**绝不在 IR 里写死 `mma.sync`。**

**可复用业界工作**：
- **MLIR `vector.contract` + vector dialect**——硬件无关的矩阵乘表达，upstream 有
  `vector → {nvgpu.mma, amdgpu.mfma}` lowering。这是 Arke 该锚的抽象层。
- **Triton `tl.dot` → 各后端 MMA** 已证明这条路可行；Arke Triton backend 已享此
  红利，MLIR backend 应复制同样的抽象位置。

---

## 4. 挑战：跨硬件正确性验证（无黄金参照）

**难点**：Arke V1 correctness 现全靠 CUDA 上 PyTorch/SDPA 做 golden。换到 AMD/华为，
本地未必有等价 golden，6GB 单卡也测不了大 shape。

**解法**：**golden 与 target 解耦**——在 CPU（MLIR CPU JIT path，已存在）跑参照，
target 硬件只验 target-vs-CPU-golden 的数值一致。加新硬件不需要新 golden。

**可复用业界工作**：**Arke 自己的 MLIR CPU JIT correctness path**
（`mlir_backend.py`，Phase 3 已建）就是现成的硬件无关 golden 源。

---

## 5. 战略判断（Kitty 建议，供 Leon 拍板 — 不改任何 frozen 目标）

### 5.1 最该先做的：把 NVIDIA-only 假设从 MLIR pipeline 挤出去（挑战 1+2 抽象重构）

**低风险高杠杆**：不需要真拿到 AMD/华为卡，就能把 `mlir_gpu.py` 硬编码 pass +
HardwareModel schema misfit 重构成 target-driven。修完，加真实后端 = "填 capability
+ 注册 target dialect"，不是重构——与 `protocol.py` 设计意图（"加 backend 无需重构
核心"）一致。

### 5.2 该不该现在真接第二硬件？倾向**先不接**

AI-Native 命题的 win condition 是"整套工具链对 Agent 友好"，**不是"跑在 N 个硬件上"**。
多硬件的价值在于*证明 IR/抽象的可移植性*，而这个证明用 **一个 MLIR CPU target + 一个
NVIDIA GPU target 的双 target** 就能做（同一 IR 两条 lowering 腿），不必等真硬件。
真 AMD/华为接入应是抽象层验证过之后的事。

> ⚠️ **这是方向建议，涉及 backend 架构走向，属需 Leon 拍板的范畴**（是否投入抽象解耦、
> 是否引入 CPU-target 作第二腿）。不属实现层 cheap fix，故先出文档对齐，不擅自开工重构。

### 5.3 可复用清单（直接拿来，别重造）

| 复用项 | 来源 | 替代 Arke 自造什么 |
|:--|:--|:--|
| TargetAttr 多硬件抽象 | IREE `#iree.gpu.target` | HardwareModel 重构参照 |
| `gpu`+`vector` dialect 多后端 lowering pass | upstream MLIR | 免写各硬件 codegen |
| `gpu-module-to-binary` 统一 target-attr | MLIR 20 | 免写各硬件 binary 生成 |
| `vector.contract` → {nvgpu.mma, amdgpu.mfma} | upstream MLIR | MMA 原语可移植抽象层 |
| CPU JIT golden | Arke 自有 `mlir_backend.py` | 跨硬件 correctness 参照 |

---

## 6. 增量路线（若 5.1 方向获批）

1. **P-M1**：`mlir_gpu.py` pass list → HardwareModel.target_dialect 驱动（NVVM 仍是
   唯一实例，纯重构，行为不变，回归测试守住）。
2. **P-M2**：HardwareModel 分层为 capability-set（通用层 + SIMT family 层），
   照 IREE TargetAttr；StrategyIR legal-action 读 capability。
3. **P-M3**：引入 **CPU target 作第二腿**（复用 CPU JIT），证明同一 Arke-IR 双 target
   lowering + CPU-golden 跨 target 一致 = 可移植性的最小可信证明。
4. **P-M4（未来，真硬件）**：AMD ROCDL 或华为 CCE 接入 = 填 capability + 注册
   target dialect，无核心重构。

每步独立 commit + 回归全绿，与项目一贯 Gate 驱动 / 增量落地纪律一致。

*Analysis by Kitty, 2026-07-30. 待 Leon 就 §5.1/5.2 方向拍板后再推进实现。*
