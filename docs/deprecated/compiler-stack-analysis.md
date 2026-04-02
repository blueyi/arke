# Kernel 编程方式、编译栈全景与 Arke 定位分析

> Date: 2026-04-01
> 目的：分析 Arke 在编译栈中的最佳切入位置

---

## 一、常见 Kernel 编程方式

### 1.1 按抽象层级分类

```
高抽象                                                               低抽象
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PyTorch    Triton      Halide/     TVM TIR/    CUDA C++    PTX/SASS
 eager      Python      TVM TE      MLIR SCF    + intrinsics 手写汇编
 torch.     @triton.jit DSL         手动调度     cudnn API
 compile                                        cublas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 "写什么"                                                     "写怎么做"
 框架自动优化                                                  手动控制一切
```

| 方式 | 代表 | 特点 | 谁用 |
|:-----|:-----|:-----|:-----|
| **框架级** | PyTorch eager / torch.compile, JAX | 最高抽象，零 kernel 知识 | 应用开发者 |
| **Python DSL** | Triton, Taichi | Python 写 kernel，编译器做调度 | 算法工程师 |
| **算法+调度分离** | Halide, TVM TE | 算法和调度分开写 | 编译器研究者 |
| **低级 IR 手调** | TVM TIR, MLIR linalg/SCF | 直接操作循环嵌套 | 编译器专家 |
| **原生 GPU 编程** | CUDA C++, HIP, AscendC | 手写 kernel，用 intrinsics | 性能工程师 |
| **汇编级** | PTX, SASS, inline asm | 手写 GPU 汇编 | 极少数极客 |

### 1.2 每种方式到硬件的完整流程

#### A. Triton（当前 Arke 主 codegen 目标）

```
Triton Python (@triton.jit)
    │ Python AST 解析
    ▼
Triton AST
    │ Frontend lowering
    ▼
TTIR (Triton IR)           ← MLIR Dialect（硬件无关）
    │ TritonToTritonGPU pass
    ▼
TTGIR (TritonGPU IR)       ← MLIR Dialect（GPU 概念：shared mem, 线程映射）
    │ TritonGPU optimizations (coalescing, pipeline, prefetch...)
    │ TritonGPUToLLVM pass
    ▼
LLVM Dialect / NVVM        ← MLIR → LLVM IR
    │ LLVM backend
    ▼
PTX                        ← NVIDIA 虚拟 ISA
    │ ptxas (NVIDIA assembler)
    ▼
CUBIN (SASS)               ← GPU 可执行二进制
    │ CUDA Driver API 加载
    ▼
GPU 执行
```

#### B. CUDA C++ （传统路径）

```
CUDA C++ (.cu)
    │ nvcc (NVIDIA compiler driver)
    │   ├─ cicc (CUDA→PTX compiler, 基于 LLVM)
    │   └─ 分离 host/device code
    ▼
PTX
    │ ptxas
    ▼
CUBIN (SASS) → fatbin → 链接 → 可执行
```

#### C. XLA / JAX

```
Python (JAX)
    │ jax.jit / tracing
    ▼
StableHLO                  ← MLIR Dialect（~100 个高级算子）
    │
    ▼
HLO                        ← XLA 内部 IR
    │ XLA 优化 passes（融合、layout 优化、内存规划）
    ▼
Thunks / Emitters          ← XLA GPU codegen
    │ 用 LLVM IR 发射 kernel
    ▼
LLVM IR → PTX → CUBIN
```

#### D. TVM

```
PyTorch / ONNX / TFLite
    │ import
    ▼
Relay / Relax              ← 计算图 IR（高级，算子粒度）
    │ 图优化（算子融合、常量折叠）
    ▼
TIR (Tensor IR)            ← 循环嵌套 IR（低级）
    │ Schedule（tile, vectorize, unroll...）
    │ AutoTVM / Meta-Schedule / Ansor 自动调优
    ▼
Target codegen
    ├─ CUDA: TIR → CUDA C++ → nvcc → PTX → CUBIN
    ├─ LLVM: TIR → LLVM IR → machine code
    └─ Vulkan: TIR → SPIR-V
```

#### E. Halide

```
Halide C++ DSL
    │ 算法定义 (pure function)
    │ + Schedule 定义 (tile/parallel/vectorize)
    ▼
Halide IR                  ← 自有 IR（循环嵌套 + 边界）
    │ lowering passes
    ▼
LLVM IR
    │
    ├─ CPU: LLVM backend → x86/ARM 二进制
    └─ GPU: LLVM NVPTX backend → PTX → CUBIN
```

#### F. Ascend NPU（华为）

```
AscendC / Triton-Ascend
    │
    ├─ 路径 1: Triton Python → TTIR → triton-ascend 后端
    │           → AscendNPU IR (MLIR) → NPU binary
    │
    ├─ 路径 2: AscendC (C++ DSL) → 毕昇编译器
    │           → AscendNPU IR → NPU binary
    │
    └─ 路径 3: StableHLO / ONNX → CANN 图编译器
                → 融合+优化 → AscendNPU IR → NPU binary
```

---

## 二、必备编译环节

不管用什么编程方式，kernel 到硬件都必须经过这些环节：

```
① 计算语义定义     "算什么"（算子、shape、dtype、数据流）
        ↓
② 优化决策         "怎么优化"（tiling、fusion、placement、并行映射）
        ↓
③ 循环嵌套生成     将 ①+② 变成具体的 loop nest + memory access pattern
        ↓
④ 硬件映射         将 loop 映射到硬件执行单元（thread/warp/block/SM）
        ↓
⑤ 低级 IR 生成     LLVM IR 或者硬件 MLIR dialect
        ↓
⑥ 机器码生成       PTX/SASS (NVIDIA), NPU binary (Ascend), SPIR-V (Vulkan)
        ↓
⑦ 二进制加载执行   Driver API 加载 + kernel launch
```

| 环节 | Triton 在哪做 | TVM 在哪做 | CUDA C++ 在哪做 |
|:-----|:-------------|:-----------|:---------------|
| ① 计算语义 | Python 函数 | Relay/Relax | .cu 函数 |
| ② 优化决策 | 编译器自动 | Schedule/AutoTVM | 手写 |
| ③ 循环嵌套 | TTIR→TTGIR | TIR | 手写 |
| ④ 硬件映射 | TTGIR passes | TIR schedule | 手写 (blockIdx, threadIdx) |
| ⑤ 低级 IR | LLVM Dialect | LLVM IR / CUDA codegen | cicc (LLVM-based) |
| ⑥ 机器码 | ptxas | nvcc/ptxas | ptxas |
| ⑦ 加载执行 | Triton runtime | TVM runtime | CUDA driver API |

---

## 三、现有编译基础设施的层次

```
          ┌─────────────────────────────────────────────────────┐
          │                  MLIR 生态                           │
          │                                                     │
  Level 5 │  StableHLO / TOSA / Linalg          算子级         │
          │  (高级算子：matmul, conv, softmax)                   │
          │                                                     │
  Level 4 │  linalg.generic / tensor / memref    张量级         │
          │  (数据+循环结构，但未调度)                            │
          │                                                     │
  Level 3 │  SCF (structured control flow)       循环级         │
          │  affine / vector                                    │
          │  (调度后的循环嵌套 + 向量化)                          │
          │                                                     │
  Level 2 │  GPU dialect                         硬件映射级     │
          │  (gpu.launch, gpu.alloc, 线程映射)                   │
          │  TTGIR (Triton GPU IR)                              │
          │                                                     │
  Level 1 │  LLVM dialect / NVVM / ROCDL /       底层 IR       │
          │  AscendNPU IR                                       │
          │  (寄存器级，内存级)                                  │
          │                                                     │
  Level 0 │  PTX / SASS / NPU binary             机器码        │
          └─────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────────────────────┐
          │                  LLVM IR                             │
          │  (SSA, 类型系统, 优化 passes)                        │
          │  → NVPTX backend → PTX                              │
          │  → AMDGPU backend → GCN ISA                         │
          │  → AArch64 backend → ARM                            │
          └─────────────────────────────────────────────────────┘
```

---

## 四、Arke 当前设计在栈中的位置

```
当前 Arke 的位置：

  Arke Lang (.ak)  ──→  Semantic IR  ──→  Strategy IR
       ↑                     ↑                ↑
    人类写              LLM tool-use       LLM 决策
                             ↓
                    Triton Python 代码   ← Arke 的 codegen 输出
                             ↓
                    TTIR → TTGIR → LLVM → PTX → CUBIN

问题：Arke 输出 Triton Python，然后依赖 Triton 的完整编译栈。
      Arke 对 Triton 之下的层完全没有控制力。
```

---

## 五、问题分析：当前设计的局限

### 5.1 "Arke → Triton → GPU" 是一层间接

Arke 生成 Triton Python 代码，Triton 再编译成 TTIR → TTGIR → PTX。
这意味着：

- **Arke 的优化决策必须表达为 Triton 的语言** — Arke 的 `tile(i=[64,16])` 必须翻译成 Triton 的 `BLOCK_M=64` 参数。如果 Triton 不支持某种优化（比如 split-k），Arke 也做不到
- **Triton 会再次做自己的优化** — Arke 做了 tiling 决策，Triton 还会做 coalescing、pipelining 等。两者可能冲突
- **调试困难** — 性能不好时不知道是 Arke 的决策问题还是 Triton 的编译问题

### 5.2 多硬件扩展受限

- NVIDIA: Triton 成熟 ✅
- AMD: Triton 有 ROCm 后端 ✅
- Ascend: triton-ascend 正在开发中 ⚠️
- 其他硬件（Intel GPU, 自定义 ASIC）: 每个都需要 Triton 后端

### 5.3 无法复用 MLIR 基础设施

当前 Arke 和 MLIR 完全没有交集。MLIR 有丰富的：
- **linalg dialect**: 已经定义了 matmul, conv 等算子语义
- **transform dialect**: 可以表达 tile, fuse, vectorize 等变换
- **GPU dialect**: 硬件映射
- **各种 lowering pass**: 经过大量验证

这些 Arke 都没有用上。

---

## 六、Arke 与芯片交互的最佳结合界面

### 6.1 三个候选位置

```
                    Arke 可以切入的位置
                    ↓           ↓           ↓
        ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
Level 5 │StHLO│  │     │  │     │  │     │  │     │
        │TOSA │  │     │  │     │  │     │  │     │
        │linalg│  │     │  │     │  │     │  │     │
        └──┬──┘  └─────┘  └─────┘  └─────┘  └─────┘
           │        ↑
Level 4    │   位置 A: Arke IR → MLIR linalg + transform
           │   (最高层，最大控制力，最大工作量)
        ┌──┴──┐
Level 3 │ SCF │
        │affine│
        │vector│
        └──┬──┘     ↑
           │   位置 B: Arke IR → MLIR SCF/GPU
Level 2    │   (中间层，平衡控制力和复用)
        ┌──┴──┐
        │ GPU │
        │TTGIR│
        └──┬──┘     ↑
           │   位置 C: Arke IR → Triton Python (当前方案)
Level 1    │   (最低层切入，最少工作量，最少控制力)
        ┌──┴──┐
        │LLVM │
        │NVVM │
        └──┬──┘
Level 0 ┌──┴──┐
        │PTX  │
        │SASS │
        └─────┘
```

### 6.2 分析

| | 位置 A: linalg | 位置 B: SCF/GPU | 位置 C: Triton (当前) |
|:--|:---|:---|:---|
| **切入点** | Arke IR → MLIR linalg + transform dialect | Arke IR → MLIR SCF + GPU dialect | Arke IR → Triton Python |
| **控制力** | ⭐⭐⭐⭐⭐ 完全控制优化 | ⭐⭐⭐⭐ 控制循环和硬件映射 | ⭐⭐ Triton 再做自己的优化 |
| **复用** | 复用 MLIR 全栈 lowering | 复用 MLIR 下半栈 | 复用 Triton 全栈 |
| **多硬件** | 天然支持所有 MLIR 后端 | 支持 MLIR 后端 | 依赖 Triton 后端 |
| **工作量** | 极大（需要完整 MLIR 集成） | 大（需要 MLIR codegen） | 小（模板/LLM 生成文本） |
| **MVP 可行** | ❌ 8 周不够 | ❌ 8 周不够 | ✅ 当前方案可行 |
| **未来上限** | 可以替代 Triton | 可以补充 Triton | 永远受限于 Triton |

### 6.3 推荐：分阶段演进

```
Phase 1 (当前, 8周 MVP):
  位置 C — Arke IR → Triton Python → GPU
  目的：验证核心假设（LLM 能否做有效优化）
  不需要 MLIR，快速出结果

Phase 2 (MVP 验证后, 3-6月):
  位置 B — Arke IR → MLIR SCF/GPU dialect → LLVM → PTX/NPU
  目的：摆脱 Triton 依赖，获得完整控制力
  关键：Arke Strategy IR 的 decisions 直接翻译为 MLIR transform ops
  复用：MLIR GPU dialect → NVVM/ROCm/AscendNPU IR → 机器码

Phase 3 (长期):
  位置 A — Arke 成为 linalg 层的替代/补充
  目的：Arke 定义自己的算子语义 + 变换规则，走 MLIR 全栈
  这时 Arke 就是一个完整的 AI-Native 编译器
```

---

## 七、对 Arke IR 和 Arke Lang 设计的影响

### 7.1 当前设计已经为 MLIR 集成做好了准备吗？

**Semantic IR — 基本准备好了 ✅**

Arke Semantic IR 和 MLIR linalg 的映射很自然：

| Arke Semantic IR | MLIR linalg |
|:-----------------|:------------|
| `node: {op: "matmul", inputs: ...}` | `linalg.matmul ins(%A, %B) outs(%C)` |
| `semantics.computation` | linalg 算子的语义定义 |
| `semantics.index_vars` | linalg 的 iterator_types |
| `semantics.reduction_axes` | linalg 的 reduction dims |
| `edges` | SSA value 数据流 |

Semantic IR 的 **不可变性** 和 **纯计算语义** 与 linalg 的设计哲学一致。

**Strategy IR — 需要调整 ⚠️**

| Arke Strategy IR | MLIR transform dialect |
|:-----------------|:-----------------------|
| `decision: tile(loop="i", factors=[64,16])` | `transform.structured.tile_using_for %matmul [64, 16]` |
| `decision: fuse(nodes=[...])` | `transform.structured.fuse_into_containing_op` |
| `decision: parallel(...)` | `transform.structured.map_to_gpu_blocks/threads` |
| `@rationale(...)` | ❌ MLIR transform 没有 rationale 概念 |

**问题**：当前 Strategy IR 的 decision 参数命名和语义与 MLIR transform dialect
不完全对齐。为了未来迁移更顺畅，建议：

1. **Decision 参数命名向 MLIR transform 靠拢**（但保持 JSON 可读性）
2. **新增 `codegen_hint` 字段**：当前 codegen 目标（triton/mlir/cuda）
3. **@rationale 作为 Arke 的差异化特性保留** — MLIR 没有，但 Arke 有

### 7.2 Arke Lang 的 strategy 块设计

当前设计：
```arke
strategy mm for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("...");
}
```

这个设计**已经很接近 MLIR transform dialect 的表达方式**：

```mlir
// MLIR transform dialect (for comparison)
transform.sequence failures(propagate) {
^bb0(%matmul: !transform.any_op):
  %tiled, %loops = transform.structured.tile_using_for %matmul [64, 16]
}
```

差异在于：
- Arke 用命名参数（`loop="i"`），MLIR 用位置参数
- Arke 有 @rationale，MLIR 没有
- Arke 的 target 是声明式的，MLIR 的 target 是通过 pass pipeline 选择

**建议保持当前 Arke Lang 设计不变** — 它在人类可读性和 MLIR 可映射性之间取得了好平衡。

### 7.3 为 Phase 2 预留的设计决策

现在不需要实现，但 Arke IR 设计需要**不阻碍**以下未来能力：

1. **Arke IR → MLIR 双向转换**
   - Semantic IR ↔ linalg ops (可行，语义对齐)
   - Strategy IR → MLIR transform sequence (可行，但 rationale 需要 MLIR attribute 扩展)
   - MLIR → Arke IR (从现有 MLIR pipeline 导入，用于分析)

2. **自定义 MLIR Dialect**
   - `arke.kernel` op (包装 Semantic IR 的语义)
   - `arke.strategy` op (包装 Strategy IR 的决策)
   - `arke.rationale` attribute (MLIR 扩展)
   - 这些 dialect 走标准 MLIR lowering 到 linalg/SCF/GPU

3. **多后端统一**
   ```
   Arke IR → Arke MLIR Dialect → MLIR lowering pipeline
     ├─ → GPU dialect → NVVM → PTX → CUBIN (NVIDIA)
     ├─ → GPU dialect → ROCm → GCN ISA (AMD)
     ├─ → AscendNPU IR → NPU binary (Ascend)
     └─ → LLVM → 其他后端
   ```

---

## 八、结论

### Arke 与芯片的最佳结合界面

**短期 (Phase 1)：Triton Python 层** — 验证假设，快速出 MVP

**中期 (Phase 2)：MLIR SCF/GPU dialect 层** — 这是"最佳甜蜜点"
- 在这一层，Arke 的 Strategy IR decisions 可以直接翻译为 MLIR transform ops
- MLIR 之下的所有 lowering (→ LLVM → PTX/NPU binary) 全部复用
- 多硬件支持几乎免费（换个 MLIR 后端就行）
- 仍然保留 Arke 的差异化（@rationale, LLM tool-use protocol, 验证系统）

**长期 (Phase 3)：Arke 自有 MLIR Dialect** — 成为编译基础设施的一部分
- Arke 定义自己的 dialect（semantic + strategy + rationale）
- 走 MLIR 标准 lowering pipeline
- 其他项目可以用 Arke dialect 做 LLM-Native 优化
- Arke 从"工具"变成"基础设施"

### 当前 IR 设计的合理性

| 设计决策 | 合理性 | 为什么 |
|:---------|:------:|:-------|
| Semantic/Strategy 双层分离 | ✅ | 直接映射 MLIR linalg + transform |
| JSON 序列化 | ✅ | LLM 友好，MLIR 可以通过 JSON → MLIR 转换 |
| 算子目录 | ✅ | 与 linalg named ops 设计一致 |
| Decision kinds | ✅ | 与 MLIR transform dialect ops 大部分对应 |
| @rationale | ✅ | Arke 独有差异化，MLIR 可通过 attribute 扩展 |
| .ak 语法 (A+D 混合) | ✅ | kernel 映射 linalg, strategy 映射 transform |
| 静态 shape | ✅ | 与 MLIR 静态 shape 一致，动态 shape 是未来扩展 |

**结论：当前设计不需要大改。主要预留项是在 Strategy IR 的 decision 参数命名上尽量与 MLIR transform dialect 对齐。**

---

*分析版本：v1.0 | 创建日期：2026-04-01*
