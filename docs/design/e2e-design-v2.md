# Arke — 端到端概要方案设计 v2.0

> Version: v2.0
> Date: 2026-03-31
> Author: Leon + AI Co-design

---

## 〇、定位纠正：Arke 不在 Triton 上面，Arke 替代 Triton

### 之前的错误定位

```
v1.0 的错误架构（已废弃）：

  PyTorch
    ↓
  Arke（"高级抽象"）    ←── 错！Arke 不是在 Triton 上面的一层
    ↓
  Triton / CUDA / AscendC  ←── 错！不是生成它们
    ↓
  硬件
```

### 正确的定位

```
v2.0 的正确架构：

  PyTorch / AI Framework
    ↓
  ┌────────┬──────────┬──────────┬──────────┐
  │ Triton │   CUDA   │ AscendC  │  Arke    │  ← 同一层级！
  │ (人写) │  (人写)  │  (人写)  │ (AI写)   │
  └────┬───┴────┬─────┴────┬─────┴────┬─────┘
       ↓        ↓          ↓          ↓
     PTX     nvcc/ptxas   CANN      ???       ← 向下结合点（待探索）
       ↓        ↓          ↓          ↓
     GPU      GPU        Ascend     硬件
```

**关键认知：**
- Arke 不是"生成 Triton 代码的工具"——那只是套壳
- Arke 是**一种全新的 kernel 编程范式**，和 Triton/CUDA 同级
- Triton 是"让人更容易写 GPU kernel"，Arke 是"让 AI 更容易写 kernel"
- 向下与芯片指令的结合点是需要探索的开放问题

### 四项核心能力

| # | 能力 | 定义 |
|---|------|------|
| 1 | **AI-First Kernel Language** | 一门 AI 能直接读写和推理的 kernel 描述语言 |
| 2 | **AI-First IR（如需要）** | 语言的内部表示，是否需要独立于语言存在待论证 |
| 3 | **LLM-Native 端到端编译工具链** | LLM 作为编译流程的一等参与者 |
| 4 | **基于 LLM 的自动生成与自主调优验证系统** | LLM 生成 → 验证 → 调优 → 闭环 |

---

## 一、能力 1：AI-First Kernel Language

### 1.1 设计问题

不是"LLM 能写的语言"（任何语言 LLM 都能写），而是"LLM 写了之后**最不容易出错、最容易推理、最容易自我验证**的语言"。

### 1.2 LLM 写代码的核心困难及 Arke 应对

| 困难 | CUDA/Triton 中的表现 | Arke 的设计应对 |
|------|---------------------|-----------------|
| **隐式语义** | 线程索引隐含并行语义 | 语义显式声明 |
| **全局推理** | 改一行影响整个 kernel | 局部决策独立性 |
| **约束不可见** | shared memory 大小要人记住 | 约束作为一等公民 |
| **无法自检** | 不知道自己写的对不对 | 内置 @assert/@invariant |
| **上下文窗口** | 长 kernel 超出上下文 | 模块化 + 分层表达 |
| **数值精度** | 精度由程序员保证 | 精度约束显式标注 |

### 1.3 语言设计原则

| 原则 | 说明 |
|------|------|
| **显式优于隐式** | 数据依赖、并行性、内存层级、精度要求全部显式 |
| **局部可理解** | 每段代码的含义不依赖远处上下文 |
| **结构化决策** | 优化不是"自由修改代码"，而是"在明确维度上做选择" |
| **可自检** | @assert、@invariant 让 LLM 验证自己的输出 |
| **可解释** | @rationale 是一等语法，不是注释 |

### 1.4 语言分层结构

```
Arke Kernel Language 三个层面：

┌──────────────────────────────────────────────────────┐
│  Level 1: Semantic（语义层）                          │
│  描述"算什么" —— 纯数学表达                            │
│  LLM 在这一层理解计算的本质                             │
├──────────────────────────────────────────────────────┤
│  Level 2: Strategy（策略层）                          │
│  描述"怎么优化" —— 结构化的优化决策                      │
│  LLM 在这一层做搜索和推理                               │
├──────────────────────────────────────────────────────┤
│  Level 3: Mapping（映射层）                           │
│  描述"怎么执行" —— 与芯片指令的结合                      │
│  设计是开放问题（见第五章）                               │
└──────────────────────────────────────────────────────┘
```

### 1.5 完整语法示例

```arke
// ============================================================
// Level 1: Semantic — 描述"算什么"
// ============================================================

type Mat = Tensor<[M, N], f16>;

kernel attention(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>,
    scale: f32 = 0.125
) -> Tensor<[B, H, S, D], f16>
{
    // 每一步都是数学声明，不是指令序列
    scores[b,h,i,j] = reduce_sum(Q[b,h,i,d] * K[b,h,j,d], axis=d);
    scaled[b,h,i,j] = scores[b,h,i,j] * scale;
    weights[b,h,i,j] = softmax(scaled[b,h,i,:], axis=j);
    output[b,h,i,d]  = reduce_sum(weights[b,h,i,j] * V[b,h,j,d], axis=j);

    return output;

    // 内置语义断言：LLM 可以自检
    @assert output.shape == Q.shape;
    @assert all(weights[b,h,i,:].sum() ≈ 1.0, atol=1e-3);
    @invariant weights >= 0;
}

// ============================================================
// Level 2: Strategy — 描述"怎么优化"
// ============================================================

strategy attention for hw("nvidia_ampere") {

    algorithm(softmax = "online")
        @rationale("online softmax avoids O(S^2) memory, uses O(S) instead");

    fuse(scores, scaled, weights, output -> single_pass)
        @rationale("FlashAttention: fuse into single tiled pass over K/V blocks");

    tile(i = [Br: 128], j = [Bc: 64], d = [Bd: 64])
        @rationale("Br*Bd*sizeof(f16) = 16KB for Q block, fits in shared memory");

    parallel(b -> grid.z, h -> grid.y, i_outer -> grid.x)
        @rationale("batch and head independent; i blocks across SMs");

    place(Q_block -> shared, async_copy = true)
        @rationale("Q stays in shared for entire K/V block iteration");
    place(K_block -> shared, double_buffer = true)
        @rationale("double buffer K to overlap compute and memory");
    place(V_block -> shared, double_buffer = true);
    place(acc -> register, dtype = f32)
        @rationale("accumulate f32 for numerical stability");

    compute(scores -> tensor_core.mma.m16n8k16.f16)
        @rationale("MMA instruction matches tile dimensions");

    pipeline(stages = [load_K, compute_QK, load_V, compute_OV], depth = 2)
        @rationale("2-stage pipeline hides memory latency");

    @constraint shared_memory_usage <= 48 * 1024;
    @constraint registers_per_thread <= 255;
}

strategy attention for hw("ascend_a3") {

    algorithm(softmax = "online")
        @rationale("same algorithmic choice, hardware-independent");

    fuse(scores, scaled, weights, output -> single_pass);

    tile(i = [Br: 256], j = [Bc: 128], d = [Bd: 64])
        @rationale("Ascend L1 = 1MB >> GPU shared 48KB; larger tiles");

    parallel(b -> core_group, h -> ai_core)
        @rationale("32 AI cores, distribute batch and head");

    place(Q_block -> l1_buffer);
    place(K_block -> l1_buffer, dma_prefetch = true);
    place(acc -> l0_buffer, dtype = f32);

    compute(scores -> cube_unit.matmul.f16)
        @rationale("Cube unit is Ascend's matrix multiply engine");

    @constraint l1_buffer_usage <= 1024 * 1024;
}
```

### 1.6 Arke vs Triton vs CUDA

| 维度 | CUDA | Triton | **Arke** |
|------|------|--------|----------|
| **设计用户** | 人类 | 人类（更友好） | **LLM** |
| **编程模型** | 线程级 | Block 级 | **语义+策略分离** |
| **优化方式** | 手写 | 部分自动 | **LLM 策略搜索** |
| **并行表达** | threadIdx | tl.program_id | **声明式 parallel()** |
| **内存管理** | 手动 | 部分自动 | **声明式 place()** |
| **正确性** | 程序员负责 | 程序员负责 | **@assert + @invariant** |
| **可解释性** | 注释（可选） | 注释（可选） | **@rationale（一等语法）** |
| **输出** | PTX | PTX (LLVM) | **开放：见第五章** |

---

## 二、能力 2：AI-First IR（是否需要？）

### 2.1 必要性论证

| 场景 | 需要 IR？ | 理由 |
|------|:---------:|------|
| LLM 生成 .ak → 编译执行 | ✅ | 语言需被解析为结构化表示 |
| LLM 优化过程中观测中间状态 | ✅ | 需要结构化状态做决策 |
| 从 PyTorch FX Graph 导入 | ✅ | 需统一内部表示 |
| LLM 做增量修改 | ✅ | JSON patch 比重写 .ak 更可靠 |

**结论：IR 需要，但角色不同于传统 IR。**

### 2.2 定位差异

```
传统 IR（LLVM IR, MLIR）：
  人写代码 → 前端 → IR → 优化 pass → codegen
  IR 是编译器内部语言，用户不需要看到。

Arke IR：
  LLM 生成 .ak → 解析 → IR → LLM 读取/推理/修改 → 编译
  IR 是 LLM 和编译器之间的"共享工作语言"。
  LLM 需要能读懂 IR（做推理），有时直接操作 IR（跳过语言层）。
```

### 2.3 IR 设计原则

| 原则 | 说明 |
|------|------|
| **JSON-Native** | 序列化格式为 JSON，LLM 对 JSON 理解最可靠 |
| **Language-IR 同构** | .ak 和 JSON IR 双射，.ak 是"语法糖视图"，IR 是"结构化视图" |
| **两层结构** | Semantic IR（计算） + Strategy IR（决策），不预设 Level 3 |
| **增量可表示** | 支持 diff/patch，减少 LLM token 消耗 |

### 2.4 IR 示例

```json
{
  "arke_ir_version": "2.0",
  "semantic": {
    "kernel_name": "attention",
    "params": [
      {"name": "Q", "type": {"kind": "tensor", "shape": ["B","H","S","D"], "dtype": "f16"}}
    ],
    "computations": [
      {
        "id": "scores",
        "expr": "reduce_sum(Q[b,h,i,d] * K[b,h,j,d], axis=d)",
        "index_vars": ["b","h","i","j","d"],
        "reduction_axes": ["d"],
        "properties": ["associative"]
      }
    ],
    "assertions": [
      {"kind": "shape_eq", "lhs": "output", "rhs": "Q"}
    ]
  },
  "strategy": {
    "target_hw": "nvidia_ampere",
    "decisions": [
      {"kind": "algorithm", "params": {"softmax": "online"}, "rationale": "..."},
      {"kind": "fuse", "params": {"ops": ["scores","weights","output"], "pattern": "single_pass"}},
      {"kind": "tile", "params": {"i": [128], "j": [64], "d": [64]}},
      {"kind": "parallel", "params": {"b": "grid.z", "h": "grid.y"}},
      {"kind": "place", "params": {"Q_block": {"memory": "shared", "async": true}}},
      {"kind": "compute", "params": {"scores": "tensor_core.mma.m16n8k16.f16"}}
    ]
  }
}
```

### 2.5 Language ↔ IR 双射

```
LLM 可以选择在任一视图上工作：
  - 生成完整 kernel → 用 .ak 视图（数学表达直观）
  - 做增量优化      → 用 JSON IR 视图（结构化操作可靠）
  - 推理语义        → 用 .ak 视图
  - 跨 kernel 比较  → 用 JSON IR 视图
```

---

## 三、能力 3：LLM-Native 端到端编译工具链

### 3.1 传统 vs LLM-Native

```
传统：
  Source → Lex → Parse → IR → [Pass1 → Pass2 → ... → PassN] → CodeGen → Binary
  每个 pass 是确定性程序，由编译器工程师编写。

LLM-Native：
  Source → Parse → IR → [LLM Decision Loop] → Lowering → Binary
  优化决策由 LLM 做出。LLM 不是"调用编译器的脚本"，是"编译流程的一等参与者"。
```

### 3.2 LLM 在编译流程中的角色

```
.ak Source
    │
    ▼
┌──────────┐
│  Parser  │  确定性（传统编译器技术）
└────┬─────┘
     ▼
 Semantic IR (L1)
     │
     ▼
┌──────────────────────────────────┐
│     LLM Strategy Engine          │  ← LLM 核心角色
│                                  │
│  输入：Semantic IR + HW Profile  │
│  输出：Strategy IR (L2)          │
│                                  │
│  LLM 调用 tools 做决策：         │
│    analyze → decide → verify     │
│    → estimate → adjust → repeat  │
└───────────┬──────────────────────┘
            ▼
 Strategy IR (L2)
     │
     ▼
┌──────────────────────────────────┐
│     Lowering（降级）              │
│                                  │
│  Semantic + Strategy → 芯片指令   │
│  向下结合点是开放问题（第五章）     │
└───────────┬──────────────────────┘
            ▼
 Binary / Loadable Kernel
```

### 3.3 LLM Strategy Engine — Tool-Use 协议

LLM 作为 agent，编译器组件作为 tools：

```yaml
tools:
  # 观测
  - get_semantic_ir       # 获取计算语义
  - get_current_strategy  # 获取当前决策
  - get_hw_profile        # 获取硬件参数
  - get_resource_usage    # 获取资源使用

  # 分析
  - analyze_compute       # 分析计算特征（算术强度、瓶颈类型）
  - estimate_performance  # 代价模型预估
  - list_legal_actions    # 列出合法决策

  # 决策
  - apply_decision        # 应用决策（含 rationale）
  - rollback              # 回滚
  - checkpoint / restore  # 检查点

  # 验证
  - verify_equivalence    # 语义等价检查
  - verify_constraints    # 硬件约束检查

  # 编译
  - compile_and_profile   # 实际编译 + profiling（慢操作）
```

### 3.4 LLM 编译 attention 的典型流程

```
LLM: get_semantic_ir()
  → 看到 attention 的数学定义

LLM: analyze_compute()
  → compute-bound, O(S^2) 中间矩阵, softmax 数值问题

LLM: get_hw_profile()
  → Ampere: 48KB shared, tensor core, 108 SMs

LLM（推理）: "O(S^2) 中间矩阵是瓶颈，应该用 online softmax 避免物化"
LLM: apply_decision({kind: "algorithm", params: {softmax: "online"},
       rationale: "avoid O(S^2) memory by online softmax"})
  → ✅ verified

LLM: apply_decision({kind: "fuse", ...})
  → ✅ verified

LLM: list_legal_actions(kind="tile")
  → [{i: [64,128,256], j: [32,64,128], d: [32,64]}]

LLM（推理）: "Br=128, Bd=64: Q_block = 128*64*2 = 16KB, fits shared"
LLM: apply_decision({kind: "tile", params: {i: [128], j: [64], d: [64]},
       rationale: "Q_block = 16KB fits shared memory"})
  → ✅ verified, shared_memory = 32KB

LLM: estimate_performance()
  → 85% roofline efficiency

LLM（判断）: "85% 已经很好了，尝试 compile"
LLM: compile_and_profile()
  → actual: 82% roofline, 145 TFLOPS
  → vs cuBLAS: 0.91x

LLM（判断）: "91% of cuBLAS, 尝试调整 tile 看看能不能更好"
LLM: checkpoint()
LLM: apply_decision({kind: "tile", params: {i: [64], j: [128], d: [64]}})
LLM: compile_and_profile()
  → actual: 78% roofline, 127 TFLOPS  ← 更差了
LLM: restore(checkpoint)  ← 回滚

LLM: "之前的方案更好，最终确定"
  → 输出完整 Strategy IR
```

**关键点：LLM 不是一次性生成代码，而是通过 tool-use 循环做有反馈的决策。**

### 3.5 编译工具链的确定性部分

LLM 负责 Strategy 决策，以下部分由传统确定性编译器处理：

```
确定性组件（不需要 LLM）：
├── Parser: .ak → Semantic IR（标准编译器前端）
├── Verifier: 检查决策合法性（规则引擎）
├── Constraint Checker: 硬件约束检查（数值计算）
├── Cost Estimator: Analytical cost model（公式计算）
└── Lowering: Semantic + Strategy → 低层表示（确定性映射规则）

LLM 负责的部分（需要"智能"）：
├── 算法选择（online softmax vs standard？）
├── 融合策略（哪些 op 融合？什么模式？）
├── Tiling 参数选择
├── 并行映射策略
├── 内存放置决策
├── 当 cost model 不准时的经验判断
└── 生成 rationale（可解释性）
```

---

## 四、能力 4：基于 LLM 的自动生成与自主调优验证系统

### 4.1 系统总览

```
完整的自主闭环：

  [输入] PyTorch 模型 / 算子规格
     │
     ▼
  ┌─────────────────────────┐
  │  LLM Kernel Generator   │  ← 能力 1（Language）
  │  生成 .ak kernel        │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  LLM Strategy Optimizer │  ← 能力 3（工具链 LLM 部分）
  │  通过 tool-use 做决策    │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  Compiler Pipeline      │  ← 能力 3（工具链确定性部分）
  │  Lowering + CodeGen     │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  Verification Engine    │  ← 验证系统
  │  三层验证               │
  │  V1: 静态等价性         │
  │  V2: 数值正确性         │
  │  V3: 性能达标           │
  └───────────┬─────────────┘
              │
         ┌────┴────┐
         │         │
     Pass ✅    Fail ❌
         │         │
         ▼         ▼
     Deploy    ┌──────────────┐
               │ LLM Debugger │  ← 自主调试
               │ 分析失败原因  │
               │ 调整策略重试  │
               └──────┬───────┘
                      │
                      └──→ 回到 Strategy Optimizer
```

### 4.2 自动生成

**问题：给定一个算子规格，LLM 如何从零生成 kernel？**

```
输入方式 A：自然语言
  "Write a fused matmul+relu kernel for 1024x512 @ 512x2048 in f16"
  → LLM 生成 .ak kernel（Level 1 Semantic）

输入方式 B：PyTorch 代码
  def forward(x, w): return F.relu(x @ w)
  → torch.compile(backend="arke") 自动提取
  → 生成 Semantic IR

输入方式 C：算子签名
  op: matmul_relu
  inputs: A[1024,512,f16], B[512,2048,f16]
  output: C[1024,2048,f16]
  → LLM 生成完整 .ak

LLM 生成时利用：
  - 内置算子库（matmul, softmax, conv2d 等标准算子的语义模板）
  - 历史经验（Pattern Library 中的优化模式）
  - @assert/@invariant 自检（生成后立即验证基本正确性）
```

### 4.3 自主调优

**问题：LLM 如何在没有人类干预的情况下持续优化性能？**

```
调优循环：

while budget > 0:
    strategy = llm.generate_strategy(semantic_ir, hw_profile, history)
    result = compiler.compile_and_profile(semantic_ir, strategy)

    if result.correctness_check == FAIL:
        llm.analyze_failure(result.error)
        llm.adjust_strategy()
        continue

    if result.performance > best_performance:
        best = (strategy, result)
        llm.record_success(strategy, rationale="...")

    else:
        llm.analyze_regression(result, best)
        llm.adjust_strategy()

    budget -= 1

return best
```

**调优策略的三个层次：**

```
层次 1：参数搜索（Parameter Tuning）
  固定算法和融合策略，搜索 tile 大小、unroll factor 等
  类似 auto-tuning，但 LLM 有启发式（不是随机搜索）

层次 2：策略搜索（Strategy Search）
  尝试不同的融合策略、内存布局、流水线深度
  LLM 根据 analyze_compute() 的结果做有方向的搜索

层次 3：算法搜索（Algorithm Search）
  尝试不同的算法变体（online vs standard softmax, 分块 vs 全量 etc.）
  最有创造性的一层，LLM 的领域知识在这里发挥作用
```

### 4.4 验证系统

```
三层验证（从快到慢，从粗到细）：

V1: 静态验证（每次 apply 自动执行, <1ms）
├── @assert 和 @invariant 检查
├── 变换前后的语义等价性
├── 硬件资源约束
├── 数据依赖完整性
└── 类型一致性
→ 不合法的决策直接拒绝，LLM 立即收到反馈

V2: 数值验证（编译后执行, ~100ms）
├── 与参考实现（NumPy）逐元素对比
├── f16 容差 vs f32 容差
├── 边界条件（shape=1, shape=max）
├── 随机输入多轮测试
└── 数值稳定性检测（NaN, Inf, 精度退化）
→ 数值错误触发 LLM Debugger

V3: 性能验证（profiling, ~1s）
├── 与基准对比（cuBLAS, vendor library）
├── Roofline 分析（距理论上限的差距）
├── 资源利用率（SM utilization, bandwidth, occupancy）
├── 性能回归检测
└── 多 shape 泛化测试（这个策略对不同输入尺寸都有效吗）
→ 性能不达标触发调优循环
```

### 4.5 LLM Debugger（自主调试）

```
当验证失败时，LLM 不只是"换个参数再试"，而是分析失败原因：

数值错误示例：
  LLM: compile_and_profile()
  → V2 FAIL: output[123,456] = NaN, reference = 0.0312

  LLM: analyze_failure()
  → 推理："NaN 通常由 exp() 溢出引起，检查 softmax 实现"
  → 发现 strategy 没有设 algorithm(softmax="online")
  → 修正：添加 online softmax
  → 重新验证 → PASS

性能回归示例：
  LLM: compile_and_profile()
  → V3: 45 TFLOPS (vs baseline 130 TFLOPS = 0.35x) ← 严重回归

  LLM: analyze_regression()
  → 推理："0.35x 说明有严重的性能 bug"
  → 调用 get_resource_usage() → shared_memory = 49000 bytes (接近上限)
  → 推理："shared memory 几乎满了，可能导致 occupancy 极低"
  → 调用 estimate_performance() → occupancy = 0.125 (只有 1 block/SM)
  → 修正：减小 tile size
  → 重新编译 → 120 TFLOPS (0.92x baseline) ← 恢复
```

---

## 五、开放问题：向下与芯片指令的结合点

### 5.1 问题陈述

Arke 的 Level 1 (Semantic) 和 Level 2 (Strategy) 设计已经清晰。
**Level 3 (Mapping) 是最大的开放问题：Arke 的策略决策如何最终变成芯片可执行的指令？**

### 5.2 候选方案

```
方案 A：复用现有后端（Triton/LLVM）作为过渡
  Arke L1 + L2 → 翻译为 Triton-Python → Triton 编译 → PTX
  优点：最快落地，复用成熟后端
  缺点：受 Triton 表达力限制；本质上还是"在 Triton 上面套壳"
  定位：Phase 1 过渡方案，不是最终形态

方案 B：直接生成 LLVM IR
  Arke L1 + L2 → 确定性 lowering → LLVM IR → LLVM 后端 → PTX
  优点：最大控制力，不依赖 Triton
  缺点：工程量巨大（重写 Triton 已做的事）；LLVM IR 对 GPU 优化不够好
  定位：长期方案之一，但投入产出比存疑

方案 C：生成 MLIR
  Arke L1 + L2 → MLIR（自定义方言） → GPU 方言 → LLVM → PTX
  优点：利用 MLIR 的多层 lowering 体系；学术社区支持好
  缺点：MLIR 学习曲线陡；方言设计本身就是大工程
  定位：如果 Arke 进入编译器基础设施领域，这是合理方向

方案 D：LLM 参与 Lowering
  Arke L1 + L2 → LLM 生成 low-level 代码（CUDA C++ 或 PTX 片段）
  → 验证系统检查正确性
  优点：最灵活，LLM 可以处理边角情况和创新优化
  缺点：LLM 生成低层代码的可靠性？验证成本高
  定位：最激进的方案，也可能是最有潜力的

方案 E：混合方案
  常规路径：确定性 lowering（覆盖 80% 的标准模式）
  特殊路径：LLM 辅助 lowering（处理 20% 的非标准情况）
  优点：兼顾可靠性和灵活性
  缺点：两套路径的维护成本
  定位：最务实的方案
```

### 5.3 探索路径建议

```
Phase 1（MVP）：方案 A（复用 Triton 后端）
  目标：验证 Language + IR + LLM Strategy Engine 的设计
  Lowering 不是重点，先跑通端到端

Phase 2（探索）：方案 A + D（Triton 为主 + LLM lowering 实验）
  开始探索 LLM 在 lowering 中的能力边界
  小范围实验：LLM 生成 Triton-Python 代码而非模板映射

Phase 3（定型）：根据 Phase 2 结果选择
  如果 LLM lowering 效果好 → 方案 D/E
  如果 LLM lowering 不可靠 → 方案 B/C（传统编译器路径）

对 Ascend A3 同理：
  Phase 1 可以复用 AscendC 编译器
  长期需要探索 Arke → CANN 的直接路径
```

### 5.4 这个问题为什么现在不需要解答

```
理由：
1. Arke 的核心价值在 Level 1 + Level 2（语义表达 + 策略搜索），不在 Level 3
2. Level 3 是可替换的——同一个 (L1, L2) 可以通过不同的 lowering 路径到不同硬件
3. Phase 1 MVP 用 Triton 后端完全够用，不影响核心设计验证
4. 过早锁定 Level 3 会限制未来的可能性（也许 LLM 2 年后能直接生成 PTX）

但需要保证：
1. L1 + L2 的设计不假设特定的 lowering 方案
2. Lowering 接口是可插拔的
3. 在设计中预留 Level 3 的扩展点
```

---

## 六、硬件抽象层（HAL）

### 6.1 设计原则

AI 不应该为每种硬件重新学习。

```
┌──────────────────────┐
│  hw("nvidia_ampere")  │  ← 具体硬件
│  hw("ascend_a3")      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Hardware Profile    │  ← JSON 描述文件
│  (compute_unit,      │     LLM 在 observe 时读取
│   memory_hierarchy,  │     legal_actions 自动过滤
│   constraints)       │
└──────────────────────┘
```

### 6.2 抽象概念映射

| 抽象概念 | NVIDIA Ampere | Ascend A3 |
|----------|---------------|-----------|
| compute_unit | SM | AI Core |
| matrix_unit | Tensor Core | Cube Unit |
| fast_memory | Shared Mem (48KB) | L1 Buffer (1MB) |
| local_memory | Register (64K) | L0 Buffer (64K) |
| parallel_outer | blockIdx | ai_core_id |
| parallel_inner | threadIdx | 隐式向量化 |

### 6.3 HW Profile 示例

```json
{
  "name": "nvidia_ampere",
  "compute_units": 108,
  "matrix_unit": {"name": "tensor_core", "shape": [16,8,16], "dtypes": ["f16","bf16","tf32"]},
  "memory_hierarchy": [
    {"name": "register", "size_per_cu": 65536, "latency_cycles": 1},
    {"name": "shared",   "size_per_cu": 49152, "bandwidth_gbps": 19000, "latency_cycles": 20},
    {"name": "l2_cache", "size_total": 41943040, "bandwidth_gbps": 6000},
    {"name": "global",   "bandwidth_gbps": 2039, "latency_cycles": 500}
  ],
  "constraints": {
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
    "max_registers_per_thread": 255,
    "warp_size": 32
  }
}
```

---

## 七、学习系统

### 7.1 三个学习来源

```
来源 A：自我探索（Self-Play）
  LLM 在 tool-use 循环中积累 (state, action, reward) 轨迹
  → 微调 / in-context learning 数据

来源 B：专家经验（Expert Imitation）
  人类用 Python DSL 写优化方案 → 自动提取为模式
  → Pattern Library

来源 C：跨 kernel 迁移
  matmul 上学到的策略迁移到 conv2d
  → 模式泛化 + 元学习
```

### 7.2 Pattern Library

```json
{
  "pattern_id": "epilogue_fusion_elementwise",
  "description": "Fuse elementwise ops after compute-heavy ops",
  "applicability": {
    "preceding_op": {"properties": ["compute_bound"]},
    "following_op": {"properties": ["elementwise"]}
  },
  "action": {"kind": "fuse", "params": {"type": "epilogue"}},
  "confidence": 0.97,
  "evidence_count": 1247
}
```

### 7.3 Cost Model 演进

```
Phase 1: Analytical Model（规则 + 公式）
  Roofline model + latency model
  误差 30-50%，但零训练成本

Phase 2: ML Model（从轨迹中学习）
  MLP/GBT，输入 = (IR features, strategy params)，输出 = performance
  误差 < 20%

Phase 3: GNN on IR（端到端预测）
  直接在 IR graph 上做性能预测
  误差 < 10%
```

---

## 八、人类参与接口

人类不操作 IR，人类审查和注入知识。

| 模式 | 说明 | 工具 |
|------|------|------|
| **审查** | 看 AI 决策 + rationale，批准/否决 | `arke inspect --visual` |
| **注入** | 用 Python DSL 写已知好的方案 | `@arke.expert_strategy` |
| **约束** | 限制搜索空间 | `.ak` 中 `@constraint` |
| **评估** | 对比多个方案的 profiling 结果 | `arke compare a.json b.json` |

---

## 九、成功标准

### Phase 1 MVP

| 指标 | 目标 |
|------|------|
| 端到端 | matmul: .ak → parse → IR → LLM strategy → Triton → GPU 执行 |
| LLM 可操作 | tool-use 协议可用，决策+验证循环正常 |
| 性能 | LLM 调优后 matmul ≥ 70% cuBLAS |
| 双硬件 | Ampere codegen 可用；Ascend IR 层可用 |
| 自检 | @assert/@invariant 验证正常 |

### Phase 2

| 指标 | 目标 |
|------|------|
| 多算子 | matmul, softmax, attention, conv2d, layernorm |
| 性能 | ≥ 85% cuBLAS/cuDNN |
| 自主调优 | 完整的生成→验证→调优闭环 |
| Pattern Library | ≥ 30 个经验证模式 |

### Phase 3

| 指标 | 目标 |
|------|------|
| torch.compile | 作为 backend 接入 |
| Ascend 落地 | 核心算子可跑 |
| Lowering 探索 | LLM lowering 实验数据 |
| 学术发表 | 论文级别的实验结果 |

---

## 十、与 v1.0 的关键差异

| 维度 | v1.0 | **v2.0** |
|------|------|----------|
| **定位** | Arke 在 Triton 上面 | **Arke 和 Triton 同级** |
| **codegen** | 生成 Triton/CUDA/AscendC 代码 | **Lowering 是开放问题，Triton 仅为过渡** |
| **四项能力** | 隐式 | **显式拆分为 Language + IR + 工具链 + 调优系统** |
| **LLM 角色** | RL agent 做搜索 | **编译器的一等参与者（tool-use 协议）** |
| **工具链** | 传统 pass-based | **LLM-Native：LLM 做决策，编译器做验证** |
| **调优** | 外部循环 | **内置自主调优+调试闭环** |
| **Level 3** | 预设 codegen 路径 | **声明为开放问题，预留探索空间** |

---

*本文档为 Arke v2.0 顶层设计，后续基于此做详细设计拆解与开发任务分解。*
