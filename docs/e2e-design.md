# Arke — 端到端概要方案设计（AI-First 视角）

> Version: v1.0 Draft
> Date: 2026-03-31
> Author: Leon + AI Co-design

---

## 〇、这份文档是什么

这不是"先想人怎么用，再适配 AI"的设计。
这是一份**以 AI Agent 为第一用户**，从端到端视角回答以下问题的概要方案：

1. AI Agent 看到什么？（输入表示）
2. AI Agent 能做什么？（动作空间）
3. AI Agent 怎么学？（反馈与学习）
4. AI Agent 的决策如何变成高效代码？（落地路径）
5. 人类在哪里参与？（监督与注入）

---

## 一、核心问题：为什么现有工具链对 AI 不友好

### 1.1 现状

| 工具链 | AI 能读懂吗 | AI 能操作吗 | AI 能学到东西吗 |
|--------|:---------:|:---------:|:------------:|
| MLIR | ⚠️ 需理解方言体系 | ❌ 变换是 C++ pass，无法枚举 | ❌ 无 rationale |
| Triton | ✅ Python 语法 | ⚠️ 优化空间隐式 | ❌ 无结构化反馈 |
| TVM TensorIR | ⚠️ Schedule 语法特殊 | ⚠️ search space 要手写 | ⚠️ AutoTVM 有但不透明 |
| CUDA | ✅ 语料多 | ❌ 太低层，变换非结构化 | ❌ 无法追溯决策 |

### 1.2 核心矛盾

**AI Agent 需要的是：**
- 结构化的状态观测（知道现在是什么情况）
- 有限的、可枚举的动作空间（知道能做什么）
- 即时的、多维度的反馈（知道做得好不好）
- 可追溯的决策轨迹（知道为什么这样做）

**现有工具链提供的是：**
- 非结构化的文本代码
- 开放的、无边界的修改自由
- 延迟的、单维度的反馈（只有"编译成功/失败"或"运行时间"）
- 不可追溯的优化历史

### 1.3 Arke 的核心主张

> **把算子优化问题，从"代码编写问题"重新定义为"结构化搜索问题"。**

AI Agent 不是在"写代码"，而是在一个定义明确的搜索空间中，做有约束的决策序列。
Arke 的全部设计都围绕这一核心主张展开。

---

## 二、端到端流程设计

### 2.1 全局流程图

```
                              端到端优化流程
═══════════════════════════════════════════════════════════════

[阶段 0: 输入]        [阶段 1: 理解]        [阶段 2: 搜索]

  PyTorch Model        Arke Semantic       AI Agent
  ─────────────→       Graph (L1)        ┌─────────────┐
       │                   │             │ observe()   │
  torch.compile            │             │ actions()   │
       │                   │             │ apply()     │←─ RL Loop
  FX Graph ───→ arke-parse ──→ L1 IR ──→│ evaluate()  │
                                         │ rollback()  │
  Python DSL ──→                         └──────┬──────┘
                                                │
  .ak File ────→                                │
                                                ▼

[阶段 3: 决策]        [阶段 4: 生成]        [阶段 5: 验证]

  Arke Schedule          Target Code         Correctness
  Tree (L2)                                  Performance
      │                      │                   │
  decisions[] ──→ arke-codegen ──→ CUDA/Triton ──→ arke-verify
  rationale[]        │                   │           │
      │              │                   │           │
      └──── L3 HW Mapping               ▼           ▼
                                      Compile     Pass/Fail
                                      & Run       + Metrics

═══════════════════════════════════════════════════════════════

[反馈回路]

  arke-learn ←── trajectory(states, actions, rewards, rationales)
      │
      ├── → Pattern Library（优化模式库）
      ├── → Cost Model Training Data（代价模型训练数据）
      └── → RL Training Data（强化学习训练数据）
```

### 2.2 阶段详细说明

#### 阶段 0：输入获取

**问题：** AI Agent 从哪里获得要优化的算子？

```
三条输入路径（按优先级排序）：

路径 A：torch.compile 集成（主路径）
  PyTorch 模型 → torch.compile(backend="arke") → FX Graph → arke-parse → L1 IR

路径 B：.ak 文件直接编写
  AI Agent 或人类直接编写 .ak 文件 → arke-parse → L1 IR

路径 C：Python DSL
  Python @arke.kernel 装饰器 → AST 提取 → L1 IR
```

**AI-First 考量：**
- 路径 A 最重要——AI Agent 通常优化的是 PyTorch 模型中的子图
- AI Agent 不需要关心输入格式，只需要关心输出的 L1 IR
- 所有路径输出统一的 L1 IR，AI Agent 的后续操作完全相同

#### 阶段 1：语义理解（L1 IR）

**问题：** AI Agent 看到的"当前状态"长什么样？

```json
{
  "graph_id": "fused_matmul_relu",
  "nodes": [
    {
      "id": "matmul_0",
      "op": "matmul",
      "inputs": {"A": {"shape": [1024, 512], "dtype": "f16"}, "B": {"shape": [512, 2048], "dtype": "f16"}},
      "output": {"shape": [1024, 2048], "dtype": "f16"},
      "semantics": {
        "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
        "index_vars": ["i", "j", "k"],
        "reduction_axes": ["k"],
        "properties": ["associative", "distributive"],
        "arithmetic_intensity": 512,
        "data_reuse": {"A": "broadcast_j", "B": "broadcast_i"}
      }
    },
    {
      "id": "relu_0",
      "op": "relu",
      "inputs": {"X": "@matmul_0.output"},
      "output": {"shape": [1024, 2048], "dtype": "f16"},
      "semantics": {
        "computation": "Y[i,j] = max(X[i,j], 0)",
        "properties": ["elementwise", "monotonic", "idempotent"],
        "arithmetic_intensity": 0.25
      }
    }
  ],
  "data_flow": [
    {"from": "matmul_0", "to": "relu_0", "size_bytes": 4194304, "lifetime": "local"}
  ],
  "analysis": {
    "total_flops": 2147483648,
    "total_memory_bytes": 6291456,
    "compute_bound": true,
    "fusion_opportunities": [
      {"nodes": ["matmul_0", "relu_0"], "type": "epilogue", "benefit": "eliminate 4MB intermediate write"}
    ]
  }
}
```

**AI-First 设计原则：**

| 原则 | 体现 |
|------|------|
| **语义显式化** | `computation` 用数学公式描述，不是 opcode |
| **属性显式化** | `properties` 列出代数性质，AI 可直接推理变换合法性 |
| **分析预计算** | `arithmetic_intensity`、`data_reuse`、`fusion_opportunities` 预先计算好 |
| **引用清晰** | `@matmul_0.output` 明确指向，不是隐式参数位置 |

#### 阶段 2：搜索优化（AI Agent Core Loop）

**问题：** AI Agent 怎么做优化决策？

设计为**有限状态的马尔可夫决策过程（MDP）**：

```
State:   当前 IR 状态（L1 Graph + 已有 L2 Decisions）
Action:  一个结构化的变换操作
Reward:  性能预估变化 + 合法性 + 资源利用率
Done:    达到搜索预算或无更多改进空间
```

**2.2.1 状态表示（State）**

```json
{
  "step": 3,
  "graph": "<L1 IR>",
  "decisions_so_far": [
    {"kind": "fuse", "params": {"ops": ["matmul_0", "relu_0"], "type": "epilogue"}, "step": 1},
    {"kind": "tile", "params": {"loop": "i", "factors": [64, 16]}, "step": 2},
    {"kind": "tile", "params": {"loop": "j", "factors": [128, 8]}, "step": 3}
  ],
  "resource_usage": {
    "shared_memory": 16384, "shared_memory_limit": 49152,
    "registers_per_thread": 32, "register_limit": 255,
    "threads_per_block": 128, "thread_limit": 1024
  },
  "estimated_performance": {
    "gflops": 45.2, "memory_bandwidth_utilization": 0.72,
    "compute_utilization": 0.35, "roofline_efficiency": 0.48
  }
}
```

**2.2.2 动作空间（Actions）**

```json
{
  "legal_actions": [
    {
      "id": "tile_k_32",
      "kind": "tile",
      "params": {"loop": "k", "factors": [32]},
      "preconditions_met": true,
      "estimated_impact": {
        "shared_memory_delta": "+8192 bytes",
        "data_reuse_improvement": "+2.1x on A",
        "estimated_gflops_delta": "+12.3"
      }
    },
    {
      "id": "reorder_ikj",
      "kind": "reorder",
      "params": {"order": ["i_outer", "k_outer", "j_outer", "i_inner", "j_inner", "k_inner"]},
      "preconditions_met": true,
      "estimated_impact": {"memory_coalescing": "improved", "estimated_gflops_delta": "+5.1"}
    },
    {
      "id": "place_A_shared",
      "kind": "place",
      "params": {"tensor": "A_tile", "memory": "shared"},
      "preconditions_met": true,
      "estimated_impact": {"shared_memory_delta": "+16384 bytes", "global_memory_reduction": "-50% on A reads"}
    }
  ],
  "illegal_actions_summary": {
    "tile_i_again": "loop i already tiled (max 2 levels)",
    "fuse_more": "no more fusion opportunities"
  }
}
```

**AI-First 设计原则：**

| 原则 | 体现 |
|------|------|
| **搜索空间可枚举** | `legal_actions()` 返回完整列表，不是无限生成空间 |
| **前置条件显式** | `preconditions_met` + `blocked_by` 告诉 AI 为什么不能选某个动作 |
| **影响预估** | `estimated_impact` 让 AI 无需实际编译就能预判效果 |
| **非法动作解释** | `illegal_actions_summary` 帮助 AI 理解约束边界 |

**2.2.3 奖励信号（Reward）**

```json
{
  "reward": {
    "performance_delta": 0.15,
    "legality": true,
    "resource_efficiency": {
      "shared_memory_utilization": 0.67,
      "register_utilization": 0.45,
      "occupancy": 0.75
    },
    "code_quality_signals": {
      "memory_coalescing_ratio": 0.92,
      "bank_conflict_count": 0,
      "warp_divergence": false
    }
  },
  "cost_model_confidence": 0.85,
  "suggestion": "consider tiling k next for shared memory reuse"
}
```

**2.2.4 Agent 交互协议**

```python
class ArkeEnv:
    """AI Agent 的核心交互接口——设计为 MDP 环境"""

    # === 观测 ===
    def observe(self) -> State:
        """返回当前完整状态（JSON 可序列化）"""
    def observe_diff(self) -> StateDiff:
        """增量观测（减少 token 消耗）"""

    # === 动作 ===
    def legal_actions(self) -> List[Action]:
        """返回所有合法动作及其预估影响"""
    def legal_actions_filtered(self, kind: str) -> List[Action]:
        """按类型过滤（tile/fuse/reorder/...）"""
    def apply(self, action: Action) -> Reward:
        """执行动作，返回多维奖励"""
    def apply_sequence(self, actions: List[Action]) -> List[Reward]:
        """批量执行（减少交互轮次）"""

    # === 回溯 ===
    def rollback(self, steps: int = 1) -> State:
        """回滚 N 步"""
    def checkpoint(self) -> str:
        """保存检查点"""
    def restore(self, checkpoint_id: str) -> State:
        """恢复到检查点"""

    # === 评估 ===
    def estimate(self) -> CostEstimate:
        """快速代价模型预估（< 1ms）"""
    def compile_and_profile(self) -> ProfilingResult:
        """实际编译 + 运行 profiling（慢但精确）"""

    # === 学习 ===
    def export_trajectory(self) -> Trajectory:
        """导出完整的 (state, action, reward) 轨迹"""
    def explain_best(self) -> Explanation:
        """用自然语言解释当前最佳方案的决策链"""
```

#### 阶段 3：决策记录（L2 Schedule Tree）

**问题：** AI 的决策如何被记录和传递？

每个决策携带结构化的 `rationale`：

```json
{
  "schedule_id": "fused_matmul_relu_ampere_v3",
  "target_graph": "fused_matmul_relu",
  "target_hw": "nvidia_ampere",
  "decisions": [
    {
      "step": 1,
      "kind": "fuse",
      "params": {"ops": ["matmul_0", "relu_0"], "type": "epilogue"},
      "rationale": {
        "text": "relu is elementwise and monotonic, fusing eliminates 4MB intermediate write to global memory",
        "evidence": {"memory_saved_bytes": 4194304, "compute_overhead": "negligible"},
        "confidence": 0.95
      }
    },
    {
      "step": 2,
      "kind": "tile",
      "params": {"loop": "i", "factors": [64, 16]},
      "rationale": {
        "text": "L2 cache line = 128B, 64 * f16 = 128B per row; warp = 32 threads, 16 = half-warp for better occupancy",
        "evidence": {"l2_hit_rate_delta": "+0.35", "occupancy_delta": "+0.12"},
        "confidence": 0.82
      }
    }
  ],
  "metadata": {
    "agent": "arke-agent-v1",
    "search_budget": 100,
    "steps_used": 47,
    "total_reward": 2.34,
    "optimization_time_sec": 12.5
  }
}
```

**rationale 的三层结构：**
- `text`: 自然语言解释（人可读 + AI 可学习）
- `evidence`: 量化证据（可被 verify 验证的数字）
- `confidence`: 置信度（AI 对这个决策的把握程度，低置信度标记给人类复审）

#### 阶段 4：代码生成

**问题：** AI 的优化决策如何变成可执行代码？

```
L1 (Semantic Graph) + L2 (Schedule Tree) + L3 (HW Mapping)
    │
    ├─→ [Primary] Triton IR → Triton Compiler → PTX → CUDA Binary
    │   适用：大部分算子。复用 Triton 成熟后端。
    │
    ├─→ [Secondary] CUDA C++ → nvcc → CUDA Binary
    │   适用：Triton 无法表达的特殊模式
    │
    └─→ [Ascend] AscendC → CANN Compiler → Ascend Binary
        适用：华为昇腾 A3 硬件
```

**AI-First 考量：**
- AI Agent **不关心 codegen 细节**——它只操作 L1 + L2
- Codegen 是确定性映射：给定 (L1, L2, target) → 输出唯一确定
- AI 通过 `compile_and_profile()` 获取实际性能，但不需要理解生成的代码

#### 阶段 5：验证

```
三层验证：

V1: 静态验证（每次 apply 时自动执行，<1ms）
├── 变换前后语义等价性检查
├── 硬件资源约束检查（shared memory, registers）
├── 数据依赖完整性检查
└── 类型一致性检查

V2: 数值验证（codegen 后执行，~100ms）
├── 与参考实现（Python/NumPy）逐元素对比
├── 多精度容差检查（f16 容差 vs f32 容差）
└── 边界条件测试

V3: 性能验证（profiling 后执行，~1s）
├── 与基准性能对比（cuBLAS, Triton auto-tuned）
├── Roofline 模型分析
├── 资源利用率分析
└── 回归检测（是否比优化前更差）
```

---

## 三、Arke Language 语法设计

### 3.1 设计原则

语法 = 方案 A+D（Python-like 计算 + 声明式 Schedule + 大括号）

1. `kernel` 块描述"算什么"——命令式，接近 Python
2. `schedule` 块描述"怎么优化"——声明式，接近配置
3. 大括号界定作用域——消除缩进歧义，AI 生成更可靠
4. `@rationale` 是一等语法——每个决策可附带解释
5. 类型标注强制——`Tensor<shape, dtype, layout>`，无隐式推断

### 3.2 完整语法示例

```arke
// 类型别名
type Mat1024x512 = Tensor<[1024, 512], f16, row_major>;
type Mat512x2048 = Tensor<[512, 2048], f16, col_major>;
type MatOut       = Tensor<[1024, 2048], f16>;

// Kernel：描述"算什么"
kernel fused_matmul_relu(A: Mat1024x512, B: Mat512x2048) -> MatOut {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

// Schedule for NVIDIA Ampere：描述"怎么优化"
schedule fused_matmul_relu for target("nvidia_ampere") {
    fuse(ops=["matmul", "relu"], type=epilogue)
        @rationale("relu is elementwise, fusing saves 4MB global write");

    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 128B = 64 * f16; half-warp = 16");

    tile(loop="j", factors=[128, 8])
        @rationale("maximize coalescing: 128 * f16 = 256B = 2 cache lines");

    tile(loop="k", factors=[32])
        @rationale("shared mem ≤ 48KB: 2 * 64 * 32 * 2B = 8KB, fits well");

    reorder(["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"])
        @rationale("outer loops → GPU blocks, inner loops → data reuse");

    parallel(loops=["i_outer", "j_outer"],
             mapping={"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"});

    place(tensor="A_tile", memory=shared, prefetch=true)
        @rationale("A broadcast along j, reuse 128x across j iterations");

    place(tensor="B_tile", memory=shared, prefetch=true)
        @rationale("B broadcast along i, reuse 64x across i iterations");

    use_tensor_core(op="matmul", instruction="mma.m16n8k16.f16")
        @rationale("Ampere tensor core matches tile sizes");

    vectorize(loop="j_inner", width=8)
        @rationale("8 * f16 = 128 bit = 1 vector load");
}

// Schedule for Ascend A3：同一个 kernel，不同硬件
schedule fused_matmul_relu for target("ascend_a3") {
    fuse(ops=["matmul", "relu"], type=epilogue)
        @rationale("same fusion, hardware-independent decision");

    tile(loop="i", factors=[256, 16])
        @rationale("Ascend cube unit optimal tile: 256x256");

    tile(loop="j", factors=[256, 16])
        @rationale("symmetric tiling for cube unit");

    tile(loop="k", factors=[64])
        @rationale("L1 buffer = 1MB, 256*64*2B * 2 = 64KB per operand");

    parallel(loops=["i_outer", "j_outer"],
             mapping={"i_outer": "ai_core_id", "j_outer": "block_id"});

    place(tensor="A_tile", memory=l1_buffer)
        @rationale("Ascend L1 buffer replaces shared memory");

    use_cube_unit(op="matmul")
        @rationale("native f16 matrix multiply instruction");
}
```

### 3.3 语法元素总览

| 类别 | 元素 |
|------|------|
| **顶层声明** | `type`, `kernel`, `schedule`, `import` |
| **标量类型** | f16, f32, f64, bf16, i8, i16, i32, i64, u8, u16, u32, u64, bool, index |
| **复合类型** | `Tensor<[dims], dtype>`, `Tensor<[dims], dtype, layout>` |
| **Layout** | row_major, col_major, tiled(r,c) |
| **Kernel 语句** | `let`, `return`, `if/else`, `for` |
| **Schedule 指令** | tile, reorder, fuse, parallel, place, vectorize, unroll, pipeline |
| **硬件特化指令** | use_tensor_core (NVIDIA), use_cube_unit (Ascend) |
| **内存层级** | global, shared, local, register (NVIDIA); global, l1_buffer, l0_buffer, ub (Ascend) |
| **注解** | `@rationale(...)`, `@constraint(...)`, `@priority(...)` |

---

## 四、硬件抽象层（HAL）

### 4.1 问题：如何让 AI Agent 的决策跨硬件？

AI Agent 不应该为每种硬件重新学习。Arke 的解法：

```
                    硬件无关的抽象概念
                    ═══════════════
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
  NVIDIA Ampere       Ascend A3           Future HW
    │                     │                     │
  具体硬件映射          具体硬件映射           具体硬件映射
```

### 4.2 抽象概念 → 硬件映射表

| 抽象概念 | NVIDIA Ampere | Ascend A3 |
|----------|---------------|-----------|
| **compute_unit** | SM (Streaming Multiprocessor) | AI Core |
| **vector_unit** | CUDA Core | Vector Unit |
| **matrix_unit** | Tensor Core | Cube Unit |
| **fast_memory** | Shared Memory (48KB/SM) | L1 Buffer (1MB/Core) |
| **local_memory** | Registers (64K/SM) | L0 Buffer |
| **global_memory** | HBM | HBM |
| **parallel_outer** | blockIdx.x/y/z | ai_core_id, block_id |
| **parallel_inner** | threadIdx.x/y/z | 隐式向量化 |
| **warp_equiv** | Warp (32 threads) | 无直接等价（向量宽度=256B） |

### 4.3 硬件 Profile 文件

每种硬件有一个 JSON profile，AI Agent 在 `observe()` 中可见：

```json
{
  "name": "nvidia_ampere",
  "compute_capability": "8.0",
  "compute_units": 108,
  "vector_width": 32,
  "matrix_unit": {
    "name": "tensor_core",
    "shape": [16, 8, 16],
    "supported_dtypes": ["f16", "bf16", "tf32", "i8"],
    "throughput_tops": 312
  },
  "memory_hierarchy": [
    {"name": "register", "size_per_cu": 65536, "bandwidth_gbps": "infinite", "latency_cycles": 1},
    {"name": "shared",   "size_per_cu": 49152, "bandwidth_gbps": 19000,     "latency_cycles": 20},
    {"name": "l2_cache", "size_total": 41943040, "bandwidth_gbps": 6000,    "latency_cycles": 200},
    {"name": "global",   "size_total": null,     "bandwidth_gbps": 2039,    "latency_cycles": 500}
  ],
  "constraints": {
    "max_threads_per_block": 1024,
    "max_blocks_per_sm": 32,
    "max_shared_memory_per_block": 49152,
    "max_registers_per_thread": 255,
    "warp_size": 32
  }
}
```

```json
{
  "name": "ascend_a3",
  "compute_units": 32,
  "vector_width": 128,
  "matrix_unit": {
    "name": "cube_unit",
    "shape": [16, 16, 16],
    "supported_dtypes": ["f16", "f32", "i8"],
    "throughput_tops": 640
  },
  "memory_hierarchy": [
    {"name": "l0_buffer",  "size_per_cu": 65536,   "bandwidth_gbps": "infinite", "latency_cycles": 1},
    {"name": "l1_buffer",  "size_per_cu": 1048576,  "bandwidth_gbps": 48000,     "latency_cycles": 10},
    {"name": "l2_cache",   "size_total": 67108864,  "bandwidth_gbps": 12000,     "latency_cycles": 100},
    {"name": "global",     "size_total": null,       "bandwidth_gbps": 3200,      "latency_cycles": 300}
  ],
  "constraints": {
    "max_blocks_per_core": 8,
    "ub_size": 262144,
    "vector_calc_width": 128
  },
  "programming_model": "single_core_sequential_with_dma"
}
```

**AI-First 考量：**
- AI Agent 在 `observe()` 中同时看到 IR 状态和硬件 profile
- `legal_actions()` 自动根据硬件 profile 过滤非法动作
  - 例如：在 Ascend 上不会出现 `use_tensor_core` 动作
  - 例如：shared memory 超限的 tile 方案不会出现在合法动作中
- AI Agent 学到的是"在给定硬件约束下的优化策略"，而非硬编码的硬件知识

---

## 五、学习系统（arke-learn）

### 5.1 问题：AI Agent 怎么变得越来越好？

```
三个学习来源：

来源 A：自我探索（Self-Play）
  AI Agent 在 ArkeEnv 中做搜索，积累 (state, action, reward) 轨迹
  → RL 训练数据

来源 B：专家经验（Expert Imitation）
  人类专家用 Python DSL 写出优化后的 kernel
  → arke-parse 解析 → arke-diff 对比 → 自动提取优化模式
  → 监督学习训练数据

来源 C：跨 kernel 迁移（Transfer Learning）
  在 matmul 上学到的 tiling 策略，迁移到 conv2d
  → 优化模式库 + 元学习
```

### 5.2 轨迹数据格式

```json
{
  "trajectory_id": "traj_20260331_001",
  "kernel": "fused_matmul_relu",
  "target_hw": "nvidia_ampere",
  "steps": [
    {
      "step": 0,
      "state_hash": "abc123",
      "action": null,
      "reward": null,
      "performance": {"gflops": 12.3, "roofline_eff": 0.15}
    },
    {
      "step": 1,
      "state_hash": "def456",
      "action": {"kind": "fuse", "params": {"ops": ["matmul_0", "relu_0"]}},
      "reward": {"performance_delta": 0.08, "legality": true},
      "rationale": "eliminate intermediate global memory write",
      "performance": {"gflops": 15.1, "roofline_eff": 0.19}
    },
    {
      "step": 2,
      "state_hash": "ghi789",
      "action": {"kind": "tile", "params": {"loop": "i", "factors": [64, 16]}},
      "reward": {"performance_delta": 0.22, "legality": true},
      "rationale": "match L2 cache line size",
      "performance": {"gflops": 35.7, "roofline_eff": 0.44}
    }
  ],
  "final_performance": {"gflops": 89.2, "roofline_eff": 0.88},
  "baseline_comparison": {"vs_cublas": 0.92, "vs_triton_autotune": 1.05}
}
```

### 5.3 优化模式库

从大量轨迹中自动提取的可复用模式：

```json
{
  "pattern_id": "epilogue_fusion_elementwise",
  "description": "Fuse elementwise ops after compute-heavy ops as epilogue",
  "applicability": {
    "preceding_op": {"properties": ["compute_bound"]},
    "following_op": {"properties": ["elementwise"]},
    "data_flow": "direct_dependency"
  },
  "action": {"kind": "fuse", "params": {"type": "epilogue"}},
  "expected_benefit": "eliminate intermediate global memory write",
  "confidence": 0.97,
  "evidence_count": 1247
}
```

### 5.4 Cost Model 训练

```
数据来源：trajectory 中的 (state, action) → actual_performance
训练目标：给定 (state, action) 预测 performance_delta
模型选择：
  - 初期：基于规则的 analytical cost model（Roofline + latency）
  - 中期：轻量级 MLP/GBT，从轨迹数据中学习
  - 后期：GNN on IR graph，端到端预测
```

---

## 六、人类参与接口

### 6.1 设计原则

> 人类不操作 IR，人类审查和注入知识。

```
人类参与的四种模式：

模式 A：审查（Review）
  AI 完成优化 → 人类审查 rationale → 批准 / 否决 / 修改
  工具：arke inspect --visual（可视化 Schedule 决策树）

模式 B：注入（Inject）
  人类用 Python DSL 写出已知好的优化方案
  → arke-learn 自动提取为模式 → AI 学习
  工具：Python DSL + @arke.expert_schedule 装饰器

模式 C：约束（Constrain）
  人类设置搜索空间约束："不要用 tensor core" / "shared memory 不超过 32KB"
  → ArkeEnv 自动在 legal_actions 中过滤
  工具：.ak 文件中的 @constraint 注解

模式 D：评估（Evaluate）
  人类对比多个 AI 方案的 profiling 结果
  → 选择最佳方案 → 反馈给学习系统
  工具：arke compare schedule_a.json schedule_b.json
```

### 6.2 Python DSL 交互

```python
import arke

# 方式 1：声明 kernel
@arke.kernel
def fused_matmul_relu(
    A: arke.Tensor[1024, 512, arke.f16],
    B: arke.Tensor[512, 2048, arke.f16]
) -> arke.Tensor[1024, 2048, arke.f16]:
    C = arke.matmul(A, B)
    return arke.relu(C)

# 方式 2：手动 schedule（专家模式）
@arke.expert_schedule(fused_matmul_relu, target="nvidia_ampere")
def my_schedule(s):
    s.fuse("matmul", "relu", type="epilogue")
    s.tile("i", [64, 16])
    s.tile("j", [128, 8])
    s.tile("k", [32])
    s.reorder(["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"])
    s.parallel(["i_outer", "j_outer"], grid_mapping=True)
    s.place("A_tile", "shared", prefetch=True)
    s.place("B_tile", "shared", prefetch=True)

# 方式 3：AI 自动优化
result = arke.auto_optimize(
    fused_matmul_relu,
    target="nvidia_ampere",
    budget=100,                    # 搜索预算
    constraints={"shared_memory_max": 32768},  # 可选约束
)
print(result.best_schedule)
print(result.performance)
print(result.rationale_chain)      # AI 的决策理由链

# 方式 4：torch.compile 集成
import torch
model = MyModel()
compiled = torch.compile(model, backend="arke", options={"target": "nvidia_ampere"})
```

---

## 七、关键技术风险与应对

| 风险 | 严重度 | 应对策略 |
|------|:------:|----------|
| **Cost Model 不准** | 🔴 高 | 初期用 analytical model + compile_and_profile 校准；中期用学到的 ML model |
| **搜索空间爆炸** | 🔴 高 | 分层搜索（先 fusion，再 tiling，再 mapping）；pattern library 剪枝 |
| **Triton codegen 表达力不足** | 🟡 中 | Triton 为主路径，CUDA C++ 为兜底路径 |
| **Ascend A3 工具链不成熟** | 🟡 中 | CANN 6.x SDK 已支持 AscendC；先在 NVIDIA 上验证设计，再移植 |
| **AI Agent 产出的 schedule 不如人工** | 🟡 中 | 初期不追求超越人工，目标是 90%+ 人工性能；专家 schedule 作为 baseline |
| **Language 设计迭代成本** | 🟡 中 | .ak 语法只是语法糖，底层是 JSON IR；语法可以随时改，IR 稳定即可 |
| **跨硬件抽象泄漏** | 🟡 中 | HAL 只做概念映射，不隐藏硬件特性；硬件特化指令仍然可用 |

---

## 八、成功标准

### Phase 1 MVP（10 周）

| 指标 | 目标 |
|------|------|
| **端到端可用** | matmul kernel: .ak → parse → IR → optimize → Triton → GPU 执行 |
| **AI 可操作** | ArkeEnv 可用，legal_actions 正确，apply/rollback 正常 |
| **性能基线** | AI 搜索 100 步后的 matmul 性能 ≥ 70% cuBLAS |
| **双硬件** | NVIDIA Ampere codegen 可用；Ascend A3 IR 层可用，codegen 骨架 |
| **测试** | IR 单测 + 语义等价验证 + 端到端集成测试 |

### Phase 2 AI Agent 集成（+8 周）

| 指标 | 目标 |
|------|------|
| **多算子** | matmul, softmax, attention, conv2d, layernorm 全部可优化 |
| **AI 性能** | 搜索 100 步后 ≥ 85% cuBLAS / cuDNN |
| **轨迹系统** | 完整的 trajectory 记录 + 导出 + 可视化 |
| **Cost Model** | analytical model 预估误差 < 30% |

### Phase 3 生态打通（+12 周）

| 指标 | 目标 |
|------|------|
| **torch.compile** | 作为 backend 接入，端到端跑通 |
| **Pattern Library** | ≥ 50 个经验证的优化模式 |
| **Ascend 落地** | Ascend A3 codegen 可用，核心算子可跑 |

---

## 九、与 v0.1 设计文档的差异

| 维度 | v0.1 (Arke-Design.md) | v1.0 (本文档) |
|------|----------------------|---------------|
| **视角** | 工具链怎么做 | AI Agent 怎么用 |
| **核心主张** | 未明确 | "代码编写问题 → 结构化搜索问题" |
| **AI 交互协议** | 简单的 5 方法接口 | 完整的 MDP 环境 + 多维反馈 |
| **硬件抽象** | 未设计 | HAL + 硬件 Profile 文件 |
| **学习系统** | 概念性描述 | 轨迹格式 + 模式库 + Cost Model 训练 |
| **人类接口** | 未设计 | 四种参与模式 + Python DSL |
| **验证** | 简单提及 | 三层验证体系 |
| **Ascend A3** | 未涉及 | 并行设计 + HW Profile |
| **风险分析** | 无 | 7 项风险 + 应对 |

---

*本文档将作为 Arke 项目的顶层设计参考。*
*下一步：基于本文档做详细设计拆解 + 开发任务分解。*
