# Arke — LLM 端到端算子生成与优化流程

> 完整描述 Arke 系统如何通过 LLM 生成并优化 GPU kernel 的全过程
> Date: 2026-03-31

---

## 一、全景视图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        用户入口（四种方式）                            │
│                                                                      │
│  方式 A: 自然语言           方式 B: CLI 参数                          │
│  arke optimize \           arke optimize \                           │
│    "写一个高效的矩阵乘法     --kernel matmul \                        │
│     kernel，支持 f16 和      --shape 1024,512,2048 \                  │
│     tensor core"              --target ampere                        │
│    → LLM 生成 .ak           → 系统从算子目录构建 IR                   │
│                                                                      │
│  方式 C: .ak 文件           方式 D: Python API                       │
│  kernel matmul(              arke.optimize(                          │
│    A: Tensor...                kernel="matmul",                      │
│  ) + strategy {                shape=[1024,512,2048],                │
│    @rationale...               target="ampere")                      │
│  }                                                                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Phase 1: 构建 Semantic IR                         │
│                                                                      │
│  输入 → Parser / Builder → Semantic IR (JSON)                        │
│  "算什么" — 纯计算语义，不含任何优化决策                                │
│  自动分析：FLOPS、memory、arithmetic intensity、fusion opportunities  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Phase 2: LLM 优化循环（核心）                      │
│                                                                      │
│  ┌─────────────────┐         tool-use         ┌──────────────────┐  │
│  │   LLM Agent     │ ◄────────────────────► │   ArkeEnv        │  │
│  │  (决策者)        │   Bounded Action Space  │  (验证器+执行器)  │  │
│  │                 │                         │                  │  │
│  │  每步决策附      │   get_legal_actions()   │  V0 静态验证     │  │
│  │  @rationale     │   apply_decision()      │  (每步自动)      │  │
│  └─────────────────┘   compile_and_profile()  └──────────────────┘  │
│                                                                      │
│  LLM 逐步构建 Strategy IR（附 @rationale）                           │
│  编译器负责验证每一步，LLM 负责做决策                                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Phase 3: Codegen + 编译执行                       │
│                                                                      │
│  Strategy IR → Codegen（模板/LLM生成）→ Triton 代码 → GPU Binary     │
│  验证：V0 静态 → V1 数值 → V2 性能                                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Phase 4: 输出与集成                                │
│                                                                      │
│  优化后的 Triton kernel + 性能报告 + 优化轨迹（含 @rationale）        │
│  → KernelCache → PyTorch custom op → torch.compile backend (未来)    │
│  → Benchmark 验证（BL×L 体系）→ Gate 出口判定                         │
└──────────────────────────────────────────────────────────────────────┘
```

**核心设计理念：**
- **Bounded Action Space**：编译器提供合法动作集，LLM 在约束空间内探索
- **@rationale**：每个优化决策附人类可读的推理过程，支持学习与调试
- **Semantic IR ↔ Strategy IR 分离**：语义与优化决策解耦，支持多硬件后端
- **三级验证（V0→V1→V2）**：静态 → 数值 → 性能，逐层保证正确性


---

## 二、Phase 1 — 构建 Semantic IR

### 2.1 四种输入路径

```
路径 A: 自然语言描述（LLM-Native 入口）
  用户以自然语言描述需求（中文/英文）
  → LLM 引擎理解意图 → 自动生成 .ak 文件（kernel + strategy）
  → Lark Parser → AST → Semantic IR
  示例：arke optimize "写一个高效的 matmul kernel，1024x512x2048，f16，Ampere"

路径 B: CLI 参数（结构化入口）
  用户指定算子名 + shape + dtype + target
  → 系统从算子目录查找定义
  → 自动构建 Semantic IR

路径 C: .ak 文件（精确控制）
  用户编写 .ak 语法（kernel + strategy 分离）
  → Lark Parser → AST → Semantic IR

**示例：.ak 语法**

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

strategy fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");
    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing");
    tile(loop="k", factors=[32])
        @rationale("balance register pressure");
    fuse(ops=["matmul", "relu"], type=epilogue)
        @rationale("eliminate intermediate write, ~15% speedup");
}
```

**关键特性：**
- `kernel` 块：纯语义描述（"算什么"），不含优化决策
- `strategy` 块：优化决策（"怎么优化"），每个决策附 `@rationale`
- 分离设计让 LLM 可以只生成 strategy，kernel 由人工或工具生成

路径 D: Python API（编程集成）
  arke.optimize(kernel="matmul", shape=[1024,512,2048], target="ampere")
  → 系统从算子目录查找定义 → 自动构建 Semantic IR

路径内部: LLM Agent 模式（路径 A/B/C/D 均可触发）
  LLM 调用 create_kernel() tool
  → 系统验证并构建 Semantic IR
```

> **路径 A 是 Arke 的 LLM-Native 特色**：用户无需了解算子细节或 .ak 语法，
> LLM 引擎自动将自然语言意图转化为结构化的 kernel 描述和优化策略。
> 这也是 G8（Agent Autonomy）的核心验证目标之一。

### 2.2 Semantic IR 示例（matmul + relu 融合）

```json
{
  "kernel_id": "fused_matmul_relu",
  "params": [
    {"name": "A", "shape": [1024, 512], "dtype": "f16", "layout": "row_major"},
    {"name": "B", "shape": [512, 2048], "dtype": "f16", "layout": "col_major"}
  ],
  "return_type": {"shape": [1024, 2048], "dtype": "f16"},
  "nodes": [
    {
      "id": "matmul_0", "op": "matmul",
      "inputs": {"A": "A", "B": "B"},
      "output_shape": [1024, 2048],
      "semantics": {
        "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
        "index_vars": ["i", "j", "k"],
        "reduction_axes": ["k"],
        "properties": ["associative"]
      }
    },
    {
      "id": "relu_0", "op": "relu",
      "inputs": {"X": "@matmul_0"},
      "output_shape": [1024, 2048],
      "semantics": {
        "computation": "Y[i,j] = max(X[i,j], 0)",
        "properties": ["elementwise", "monotonic"]
      }
    }
  ],
  "edges": [{"from": "matmul_0", "to": "relu_0"}],
  "return_node": "relu_0"
}
```

### 2.3 自动分析

Semantic IR 构建完成后，系统自动计算特征，作为 LLM 做决策的输入：

```json
{
  "auto_analysis": {
    "total_flops": 2147483648,
    "memory_bytes": {"read": 4194304, "write": 4194304},
    "arithmetic_intensity": 256.0,
    "bottleneck": "compute_bound",
    "fusion_opportunities": [
      {
        "nodes": ["matmul_0", "relu_0"],
        "type": "epilogue",
        "reason": "relu is elementwise, can fuse into matmul epilogue",
        "estimated_benefit": "eliminate 4MB intermediate write, ~15% speedup"
      }
    ]
  }
}
```

---

## 二点五、算子覆盖（Operator Tier）

当前 Arke 覆盖 **45 个算子**，按复杂度分为 5 层（OT0-OT4）：

| Tier | 名称 | 算子数 | 代表算子 |
|:----:|:-----|:------:|:---------|
| **OT0** | 元素级 | 12 | `relu`, `gelu`, `silu`, `add`, `mul`, `exp`, `sigmoid`, `tanh`... |
| **OT1** | 规约 | 10 | `softmax`, `layernorm`, `rmsnorm`, `reduce_sum`, `reduce_max`... |
| **OT2** | 数据移动与计算密集 | 11 | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose`, `conv2d`... |
| **OT3** | 融合复合 | 7 | `swiglu`, `geglu`, `rmsnorm_residual`, `fused_matmul_gelu`... |
| **OT4** | 注意力 | 5 | `flash_attention`, `grouped_query_attention`, `multi_latent_attention`, `paged_attention`... |

**Shape Tier (ST)**：每个算子在 4 个 shape 层级验证：
- ST1: Micro（小规模对齐）
- ST2: Standard（中规模 + LLM 典型）
- ST3: Stress（非对齐 + 极端）
- ST4: Production（真实 LLM 生产 shape）

→ 完整算子列表见 [benchmark-ops.md](../benchmark/benchmark-ops.md)


## 三、Phase 2 — LLM 优化循环（核心）

### 3.1 架构角色

```
┌─────────────────────────┐     tool-use     ┌──────────────────────────┐
│        LLM Agent        │ ◄──────────────► │        ArkeEnv           │
│                         │    (JSON API)    │                          │
│  角色：decision maker   │                  │  角色：validator +       │
│  能力：推理、规划、学习  │                  │        executor          │
│  限制：不写代码         │                  │  能力：验证、编译、       │
│                         │                  │        profiling          │
│  输入：                  │                  │                          │
│  - tool 返回的 JSON     │                  │  组件：                   │
│  - system prompt        │                  │  - StaticValidator (V0)  │
│  - few-shot 示例        │                  │  - NumericalValidator(V1)│
│                         │                  │  - PerformanceProfiler(V2)│
│  输出：                  │                  │  - LegalActionsEngine    │
│  - tool 调用 (决策)      │                  │  - Codegen (模板/LLM)    │
│  - rationale (推理过程)  │                  │  - ArkeCompiler          │
└─────────────────────────┘                  └──────────────────────────┘
```

### 3.2 完整交互时序

```
┌─────────┐                                    ┌───────────┐
│   LLM   │                                    │  ArkeEnv  │
└────┬────┘                                    └─────┬─────┘
     │                                               │
     │ ══════════ 阶段 A：理解问题 ══════════════     │
     │                                               │
     │  ① create_kernel(matmul+relu spec)            │
     │──────────────────────────────────────────────>│
     │                                               │ 构建 Semantic IR
     │                                               │ 自动分析
     │  ◄── semantic_ir + auto_analysis              │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ② get_hw_profile("nvidia_ampere")            │
     │──────────────────────────────────────────────>│
     │  ◄── HW Profile JSON                         │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ③ analyze_compute()                          │
     │──────────────────────────────────────────────>│
     │  ◄── per-op 分析 + 融合建议 + 优先级建议      │
     │<──────────────────────────────────────────────│
     │                                               │
     │  LLM 内部推理：                                │
     │  "compute_bound, AI=256, tensor core shape    │
     │   16×8×16, shared=48KB → 先 fuse relu 进      │
     │   epilogue, 再 tile 对齐 tensor core"          │
     │                                               │
     │ ══════════ 阶段 B：逐步优化 ══════════════     │
     │                                               │
     │  ④ list_legal_actions(kind="fuse")            │
     │──────────────────────────────────────────────>│
     │  ◄── 可选融合方案 + 影响估计                   │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ⑤ apply_decision({                           │
     │      kind: "fuse",                            │
     │      params: {nodes: ["matmul_0","relu_0"],   │
     │               type: "epilogue"},              │
     │      rationale: "relu is elementwise after    │
     │        matmul; fusing eliminates 4MB write"   │
     │    })                                         │
     │──────────────────────────────────────────────>│
     │                                               │ V0 静态验证 → ✅
     │                                               │ 更新 Strategy IR
     │  ◄── success + state_delta                    │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ⑥ list_legal_actions(kind="tile")            │
     │──────────────────────────────────────────────>│
     │  ◄── tile 候选: [64,128,32], [128,128,32]...  │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ⑦ apply_decision({                           │
     │      kind: "tile",                            │
     │      params: {tiles: {i:[64,16],j:[128,16],   │
     │                       k:[32]}},               │
     │      rationale: "BLOCK_M=64, BLOCK_N=128      │
     │        aligns with tensor core 16×8×16.       │
     │        BLOCK_K=32 keeps shared mem under      │
     │        48KB with double-buffer room"          │
     │    })                                         │
     │──────────────────────────────────────────────>│
     │                                               │ V0: shared mem ~24KB < 48KB ✅
     │  ◄── success + resource_usage                 │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ⑧ apply_decision({kind:"place", ...})        │
     │──────────────────────────────────────────────>│
     │  ◄── success                                  │
     │                                               │
     │  ⑨ apply_decision({kind:"parallel", ...})     │
     │──────────────────────────────────────────────>│
     │  ◄── success                                  │
     │                                               │
     │ ══════════ 阶段 C：验证 + 迭代 ═══════════     │
     │                                               │
     │  ⑩ checkpoint("v1")                           │
     │──────────────────────────────────────────────>│
     │                                               │
     │  ⑪ verify_correctness()                       │
     │──────────────────────────────────────────────>│
     │                                               │ Strategy IR → Codegen
     │                                               │ → Triton 代码 → 编译
     │                                               │ → GPU 执行 → vs NumPy
     │  ◄── pass=true, max_err=0.00195              │
     │<──────────────────────────────────────────────│
     │                                               │
     │  ⑫ compile_and_profile()                      │
     │──────────────────────────────────────────────>│
     │                                               │ GPU profiling
     │  ◄── latency=142μs, 70% cuBLAS               │
     │<──────────────────────────────────────────────│
     │                                               │
     │  LLM 分析："70% 不够好。occupancy=62.5%，      │
     │  可能 tile 影响了寄存器压力。试试调 tile。"     │
     │                                               │
     │  ⑬ rollback("v1")                             │
     │──────────────────────────────────────────────>│
     │                                               │ 恢复到 checkpoint
     │                                               │
     │  ⑭-⑯ 调整 tile + place + parallel            │
     │──────────────────────────────────────────────>│
     │                                               │
     │  ⑰ verify_correctness() → pass               │
     │  ⑱ compile_and_profile()                      │
     │──────────────────────────────────────────────>│
     │  ◄── latency=125μs, 82% cuBLAS ✅            │
     │<──────────────────────────────────────────────│
     │                                               │
     │  LLM: "82% cuBLAS, 接受。"                    │
     │  （优化循环结束）                              │
```

### 3.3 LLM 典型决策序列

一个 matmul+relu 优化的典型决策流：

```
Step 1: analyze_compute()
  → compute_bound, AI=256, tensor core available
  → 推理出优先级：fuse → tile → place → parallel

Step 2: fuse(matmul_0 + relu_0, epilogue)
  → 消除 4MB 中间写入

Step 3: tile(i=64, j=128, k=32)
  → 对齐 tensor core shape
  → shared mem 估算 24KB < 48KB

Step 4: place(A_tile → fast_memory, B_tile → fast_memory)
  → 数据复用：A 复用 16x (2048/128), B 复用 16x (1024/64)

Step 5: parallel(i_outer → block.x, j_outer → block.y)
  → 16×16 = 256 blocks

Step 6: verify_correctness()
  → 数值正确

Step 7: compile_and_profile()
  → 性能不满意 → rollback → 调整 tile

Step 8-10: 迭代调整 tile 参数

Step 11: compile_and_profile()
  → 性能满意 → finalize
```

### 3.4 Strategy IR 的逐步构建过程

每次 `apply_decision()` 都在 Strategy IR 上追加一条决策记录：

```json
{
  "kernel_id": "fused_matmul_relu",
  "target_hw": "nvidia_ampere",
  "decisions": [
    {
      "step": 1, "kind": "fuse",
      "params": {"nodes": ["matmul_0","relu_0"], "type": "epilogue"},
      "rationale": "relu is elementwise; fusing eliminates 4MB intermediate write",
      "validation": {"pass": true, "resource_delta": {"memory_write": "-4MB"}}
    },
    {
      "step": 2, "kind": "tile",
      "params": {"tiles": {"i": [64,16], "j": [128,16], "k": [32]}},
      "rationale": "BLOCK_M=64, BLOCK_N=128 aligns with tensor core 16×8×16",
      "validation": {"pass": true, "shared_memory_estimated": 24576}
    },
    {
      "step": 3, "kind": "place",
      "params": {"A_tile": "fast_memory", "B_tile": "fast_memory"},
      "rationale": "A reused 16x across j, B reused 16x across i",
      "validation": {"pass": true, "shared_memory_total": 24576}
    },
    {
      "step": 4, "kind": "parallel",
      "params": {"i_outer": "block.x", "j_outer": "block.y"},
      "rationale": "16×16=256 blocks, good GPU occupancy",
      "validation": {"pass": true, "grid": [16, 16]}
    }
  ],
  "compile_results": [
    {
      "attempt": 1, "decisions_snapshot": [1,2,3,4],
      "correctness": true,
      "performance": {"latency_us": 142, "vs_cublas": 0.70}
    },
    {
      "attempt": 2, "decisions_snapshot": [1,2b,3b,4],
      "correctness": true,
      "performance": {"latency_us": 125, "vs_cublas": 0.82}
    }
  ],
  "final_performance": {"latency_us": 125, "tflops": 17.2, "vs_cublas": 0.82}
}
```

### 3.5 错误处理与降级

```
┌────────────────────────────────────────────────┐
│           LLM 决策失败时的处理流程               │
├────────────────────────────────────────────────┤
│                                                │
│  apply_decision()                              │
│       │                                        │
│       ▼                                        │
│  V0 静态验证                                    │
│       │                                        │
│   ┌───┴───┐                                    │
│   │ 通过？ │                                    │
│   └───┬───┘                                    │
│   Yes │   No                                   │
│       │    │                                    │
│       ▼    ▼                                    │
│   更新IR  自动rollback + 返回错误信息给LLM       │
│       │   "shared memory 52KB exceeds 48KB     │
│       │    limit. Try smaller tile or remove    │
│       │    one tensor from fast_memory."        │
│       │                                        │
│       ▼                                        │
│  verify_correctness()                          │
│       │                                        │
│   ┌───┴───┐                                    │
│   │ 通过？ │                                    │
│   └───┬───┘                                    │
│   Yes │   No                                   │
│       │    │                                    │
│       ▼    ▼                                    │
│   继续   rollback + 返回错误信息给LLM           │
│          "max_abs_err=0.15 > atol=0.01.        │
│           Common causes: accumulation dtype,    │
│           boundary masking, reduction order."   │
│       │                                        │
│       ▼                                        │
│  compile_and_profile()                         │
│       │                                        │
│   ┌───┴───────────────┐                        │
│   │ 性能 ≥ fallback？  │                        │
│   └───┬───────────────┘                        │
│   Yes │   No                                   │
│       │    │                                    │
│       ▼    ▼                                    │
│   继续   提示LLM但不rollback                    │
│          "performance regressed from 82% to    │
│           65% cuBLAS. Consider rollback."      │
│                                                │
│  预算耗尽？                                     │
│   Yes → 比较 LLM 最优 vs fallback_strategy     │
│         如果 LLM 更好 → 输出 LLM 结果          │
│         如果 fallback 更好 → 输出 fallback      │
│         并标记 "LLM 未能改进"                   │
└────────────────────────────────────────────────┘
```

### 3.6 预算管理

```python
class OptimizationBudget:
    max_decisions: int = 50     # 最多 50 步决策
    max_compiles: int = 10      # 最多 10 次编译+profiling（昂贵操作）
    target_performance: float = 0.7   # 目标 ≥ 70% cuBLAS
    warning_threshold: int = 40       # 40 步时提醒 LLM

    # 自动注入 budget 状态到每次 tool 返回
    # "budget": {"decisions_used": 12, "decisions_remaining": 38,
    #            "compiles_used": 2, "compiles_remaining": 8}
```

---

## 四、Phase 3 — Codegen + 编译执行

### 4.1 双路径 Codegen

```
                    Strategy IR
                        │
                ┌───────┴───────┐
                ▼               ▼
         路径 A: 模板       路径 B: LLM 生成
         (Jinja2 模板)     (LLM 写 Triton 代码)
                │               │
                ▼               ▼
           Triton 代码     Triton 代码
                │               │
                └───────┬───────┘
                        ▼
                V0 静态验证（代码级）
                        │
                        ▼
                Triton 编译 → GPU Binary
                        │
                        ▼
                V1 数值验证（vs NumPy）
                        │
                        ▼
                V2 性能验证（GPU profiling）
```

#### 路径 A：模板翻译

```python
class TritonTemplateEngine:
    """从 Strategy IR 翻译为 Triton 代码"""

    def generate(self, semantic_ir, strategy_ir) -> str:
        # 1. 匹配算子模式
        pattern = self._match_pattern(semantic_ir)  # "matmul", "matmul_relu", "softmax"

        # 2. 加载对应模板
        template = self._load_template(f"{pattern}.py.j2")

        # 3. 从 strategy_ir 提取模板参数
        params = {
            "BLOCK_M": strategy_ir.get_tile("i", 0),   # e.g., 64
            "BLOCK_N": strategy_ir.get_tile("j", 0),   # e.g., 128
            "BLOCK_K": strategy_ir.get_tile("k", 0),   # e.g., 32
            "fused_epilogue": strategy_ir.get_fusion_epilogue(),  # "relu" or None
            "use_tensor_core": strategy_ir.has_compute("matrix_unit"),
        }

        # 4. 渲染模板
        return template.render(**params)
```

#### 路径 B：LLM 生成

```python
class TritonLLMGenerator:
    """让 LLM 根据 Strategy IR 生成 Triton 代码"""

    def generate(self, semantic_ir, strategy_ir, hw_profile) -> str:
        prompt = f"""Generate a Triton GPU kernel.

## Computation
{json.dumps(semantic_ir, indent=2)}

## Optimization Strategy (apply these decisions)
{json.dumps(strategy_ir, indent=2)}

## Hardware
{json.dumps(hw_profile, indent=2)}

## Requirements
- @triton.jit decorator
- Accumulation in float32, output in float16
- Proper boundary masking
- Apply ALL optimization decisions from the strategy

Output only the complete Python code."""

        code = self.llm_provider.chat([{"role": "user", "content": prompt}])
        return self._extract_python_code(code.content)
```

### 4.2 编译 + 执行 + Profiling

```python
class ArkeCompiler:
    def compile_and_run(self, triton_code, semantic_ir):
        """完整的编译→执行→验证流程"""

        # 1. 写入临时文件并动态 import
        module = self._dynamic_import(triton_code)
        kernel_fn = module.kernel
        launcher_fn = module.launch

        # 2. 准备输入
        inputs = self._prepare_inputs(semantic_ir)  # GPU tensors

        # 3. 执行
        output = launcher_fn(*inputs)

        # 4. V1 数值验证
        ref = self._numpy_reference(semantic_ir, inputs)
        correctness = self._check_numerical(output, ref)

        # 5. V2 性能 profiling
        performance = self._profile(launcher_fn, inputs)

        return CompileResult(
            code=triton_code,
            correctness=correctness,
            performance=performance,
        )
```

### 4.3 多硬件 Codegen（Phase 2）

```
同一份 Strategy IR → 同一份 Triton 代码
                          │
                  ┌───────┴───────┐
                  ▼               ▼
            NVIDIA GPU       Ascend NPU
            (triton)         (triton-ascend)
                  │               │
                  ▼               ▼
               PTX/SASS       AscendNPU IR
                  │               │
                  ▼               ▼
              GPU Binary      NPU Binary
```

triton-ascend 让 Triton 代码直接跑在 Ascend 上，Arke 的 Triton codegen 是双硬件通用的。差异仅在编译路径和 profiling 接口。

---

## 五、Phase 4 — 输出与集成

### 5.1 输出物

一次完整的 Arke 优化产出：

```
output/
├── kernel.py                  # 优化后的 Triton kernel 代码
├── strategy.json              # 完整的 Strategy IR（含所有决策 + rationale）
├── trajectory.jsonl           # 优化轨迹：(state, action, reward) 序列
├── report.json                # 性能报告
└── metadata.json              # 元信息（LLM, hardware, budget, timestamps）
```

#### kernel.py（最终输出的 Triton 代码）

```python
import triton
import triton.language as tl

@triton.jit
def fused_matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # ... (LLM 或模板生成的完整 kernel)

def launch(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    fused_matmul_relu_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
    )
    return C
```

#### report.json

```json
{
  "kernel": "fused_matmul_relu",
  "hardware": "nvidia_ampere (RTX 3060 Laptop)",
  "shape": {"M": 1024, "N": 2048, "K": 512},
  "dtype": "f16",
  "optimization": {
    "decisions": 8,
    "compiles": 3,
    "total_time_sec": 45,
    "tokens_used": 12500,
    "llm_provider": "anthropic/claude-sonnet-4-6"
  },
  "correctness": {
    "pass": true,
    "max_absolute_error": 0.00195,
    "tolerance": {"atol": 0.01, "rtol": 0.01}
  },
  "performance": {
    "latency_us": 125,
    "tflops": 17.2,
    "vs_cublas": 0.82,
    "vs_pytorch": 1.15,
    "roofline_efficiency": 0.79
  }
}
```

#### trajectory.jsonl（轨迹数据，用于学习）

```jsonl
{"step":0,"type":"observe","state":{"kernel":"fused_matmul_relu","hw":"ampere","decisions":[]}}
{"step":1,"type":"action","tool":"apply_decision","params":{"kind":"fuse","nodes":["matmul_0","relu_0"]},"rationale":"..."}
{"step":1,"type":"result","success":true,"resource_delta":{"memory_write":"-4MB"}}
{"step":2,"type":"action","tool":"apply_decision","params":{"kind":"tile","tiles":{"i":[64,16],"j":[128,16],"k":[32]}},"rationale":"..."}
{"step":2,"type":"result","success":true,"shared_memory":24576}
...
{"step":7,"type":"compile","attempt":1,"correctness":true,"performance":{"vs_cublas":0.70}}
{"step":11,"type":"compile","attempt":2,"correctness":true,"performance":{"vs_cublas":0.82}}
{"step":11,"type":"final","accepted":true}
```

### 5.2 PyTorch 集成（整模型端到端）

```python
# arke/integration/torch_ops.py

import torch
from torch.library import Library

arke_lib = Library("arke", "DEF")

# 注册 Arke 优化的 matmul
arke_lib.define("matmul(Tensor A, Tensor B) -> Tensor")

@torch.library.impl(arke_lib, "matmul", "CUDA")
def arke_matmul(A, B):
    """调用 Arke 优化后的 Triton kernel"""
    from arke.output.fused_matmul_relu import launch
    return launch(A, B)

# 在模型中替换
class ArkeLinear(torch.nn.Module):
    """替换 nn.Linear 的 Arke 优化版本"""
    def __init__(self, original_linear):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias

    def forward(self, x):
        out = torch.ops.arke.matmul(x, self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out

# 自动替换模型中的算子
def patch_model(model, optimized_ops):
    """将模型中的目标算子替换为 Arke 优化版本"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "linear" in optimized_ops:
            parent = _get_parent(model, name)
            setattr(parent, name.split(".")[-1], ArkeLinear(module))
    return model
```

#### KernelCache（实际集成方式）

Arke 通过 `KernelCache` 管理优化后的 kernel，提供基于算子名 + shape 的自动 dispatch：

```python
from arke.integration import KernelCache

cache = KernelCache()
cache.register("matmul", shape=[1024, 1024], kernel_fn=optimized_matmul)

# 模型前向时自动查找并使用缓存的 kernel
output = cache.dispatch("matmul", A, B)
```

**未来集成路径（G8/G9 目标）**：
- 通过 `torch.compile` Inductor backend 可消除 Python dispatch 开销
- 实现图级 fusion + 缓存 kernel 的零开销调用
- 支持 `torch.library.custom_op` 注册为原生算子

### 5.3 整模型端到端流程

```
Step 1: 用户指定模型 + 目标
  arke model-optimize --model gpt2 --target ampere --llm anthropic

Step 2: Arke 自动 profile 模型，识别热点算子
  → linear (matmul): 65% time
  → softmax: 12% time
  → layernorm: 8% time
  → 其他: 15% time

Step 3: 对每个热点算子运行 LLM 优化循环
  → matmul [1024,768]×[768,768] → 优化 → 输出 kernel
  → matmul [1024,768]×[768,3072] → 优化 → 输出 kernel
  → softmax [1024,12,64,64] → 优化 → 输出 kernel
  （按模型实际 shape 优化，不是 benchmark shape）

Step 4: 用 torch.library 注册优化后的 kernel

Step 5: 逐个替换 + 测性能
  baseline (eager): 100 tokens/sec
  + Arke matmul:    112 tokens/sec (+12%)
  + Arke softmax:   118 tokens/sec (+18%)
  + Arke layernorm: 120 tokens/sec (+20%)

Step 6: 输出完整报告
```

---

## 六、完整代码路径追踪

以 `arke optimize matmul.json --target ampere --llm anthropic` 为例：

```python
# arke/cli.py
@cli.command()
def optimize(input_file, target, llm):
    # 1. 加载配置
    config = load_config()                        # arke.config.yaml
    llm_provider = create_provider(llm, config)   # LLM Provider 抽象

    # 2. 构建 Semantic IR
    if input_file.endswith('.ak'):
        ast = parse_ak_file(input_file)           # Lark Parser
        semantic_ir = ast_to_ir(ast)              # AST → Semantic IR
    elif input_file.endswith('.json'):
        semantic_ir = load_json(input_file)       # 直接加载 JSON IR
    else:
        raise ValueError("Unsupported file format")

    # 3. 加载硬件 profile
    hw_profile = load_hw_profile(target)          # nvidia_ampere.json

    # 4. 创建 ArkeEnv
    env = ArkeEnv(
        semantic_ir=semantic_ir,
        hw_profile=hw_profile,
        config=OptimizationConfig(
            max_decisions=config.optimization.max_decisions,
            max_compiles=config.optimization.max_compiles,
            target_performance=config.optimization.target_performance_ratio,
        ),
    )

    # 5. 创建 Agent Runner
    runner = AgentRunner(
        llm_provider=llm_provider,
        env=env,
        system_prompt=build_system_prompt(hw_profile),
        tools=env.get_tool_schemas(),             # 10 个 tool 的 JSON Schema
    )

    # 6. 运行优化循环
    result = runner.run()
    # runner 内部循环：
    #   while not done:
    #     response = llm.chat(messages, tools)
    #     if response.tool_calls:
    #       for call in response.tool_calls:
    #         tool_result = env.execute_tool(call.name, call.params)
    #         messages.append(tool_result)
    #     else:
    #       done = True

    # 7. 输出
    result.save("output/")
    print(f"✅ Performance: {result.performance.vs_cublas:.0%} cuBLAS")
    print(f"   Decisions: {result.decisions_count}, Compiles: {result.compiles_count}")
    print(f"   Tokens: {result.tokens_used}, Time: {result.total_time_sec:.0f}s")
```

### AgentRunner 核心循环

```python
class AgentRunner:
    def run(self) -> OptimizationResult:
        """执行完整的 LLM 优化循环"""

        # 初始化对话
        messages = [{"role": "system", "content": self.system_prompt}]

        # 让 LLM 开始（通常它会先调 create_kernel 或 analyze_compute）
        messages.append({"role": "user", "content":
            f"Optimize this kernel for {self.env.hw_profile['name']}.\n"
            f"Semantic IR:\n{json.dumps(self.env.semantic_ir, indent=2)}"
        })

        while True:
            # 调用 LLM
            response = self.llm_provider.chat(messages, tools=self.tools)

            if not response.tool_calls:
                # LLM 没有调用 tool → 优化结束
                break

            for tool_call in response.tool_calls:
                # 执行 tool
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                try:
                    result = self.env.execute_tool(tool_name, tool_args)
                except BudgetExhaustedError:
                    # 预算耗尽 → 强制结束
                    result = self._handle_budget_exhausted()
                    return self._finalize(result)

                # 将 tool 结果加入对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })

            # 检查是否达到性能目标
            if self.env.best_performance and \
               self.env.best_performance.vs_cublas >= self.env.config.target_performance:
                # 可以继续优化，也可以接受
                pass

        return self._finalize()

    def _finalize(self) -> OptimizationResult:
        """收尾：对比 fallback，输出最优结果"""
        llm_best = self.env.best_compile_result
        fallback = self.env.fallback.evaluate()

        if llm_best and llm_best.performance > fallback.performance:
            return OptimizationResult(source="llm", **llm_best)
        else:
            return OptimizationResult(source="fallback", **fallback,
                                      note="LLM did not improve over fallback strategy")
```

---

## 七、Benchmark 分层体系与 Gate 系统

### Benchmark Level (BL)

Arke 使用三维 benchmark 体系衡量验证完整度：

**BL = Operator Tier (OT) × Shape Tier (ST)**

| Level | 覆盖范围 | 用途 |
|:-----:|:---------|:-----|
| BL1 | OT0-2 × ST1 | 快速回归（基础算子 + 小 shape） |
| BL2 | OT0-2 × ST1-2 | 日常 CI（基础算子 + 标准 shape） |
| BL3 | OT0-2 × ST1-3 | Gate 验证（基础算子 + 全量 shape） |
| BL4 | OT0-4 × ST1-2 | 算子完整性（全算子 + 标准 shape） |
| BL5 | OT0-4 × ST1-4 | 完整基准（全算子 × 全 shape） |
| BL6 | Model-Complete | 真实模型端到端（GPT-2, LLaMA-2/3, Qwen2.5, DeepSeek-V2） |

### 评估层次 (L)

- **L1**: 单算子 benchmark（`benchmarks/bench_l1.py`）
- **L2**: 融合算子 benchmark（`benchmarks/bench_l2.py`）
- **L3**: 模型端到端 benchmark（`benchmarks/bench_l3.py`）

### Gate 系统

Phase 1 共 **10 个 Gate（G0-G9）**，每个 Gate 的出口条件由 **BL×L 组合**定义：

| Gate | 出口 | 核心目标 | 状态 |
|:----:|:-----|:---------|:----:|
| G0 | — | GPU 环境验证 | ✅ |
| G1 | — | IR + 验证系统 | ✅ |
| G2 | BL1×L1 | 手动策略 → Codegen → GPU | ✅ |
| G3 | BL1×L1 | LLM Agent 闭环优化 | ✅ |
| G4 | BL2×L1 | Arke vs LLM-direct 对比 | ✅ |
| G5 | BL3×L1 + BL6/GPT-2×L3 | 全基础算子 + E2E 正确性 | ✅ |
| G6 | BL5×L1+L2 | **Compiler Infrastructure**（编译器基础设施） | ⬜ |
| G7 | BL5×L1+L2 | **Lang & IR v0.1.0**（当前语言与 IR） | ⬜ |
| G8 | BL5×L1+L2 + BL6×L3 | **Agent Autonomy**（自主工程能力） | ⬜ |
| G9 | BL6×L3 (4模型) | **Phase 1 最终验收** | ⬜ |

→ 详见 [benchmark-design.md](../benchmark/benchmark-design.md) | [plan.md](../roadmap/plan.md)


## 八、数据流总结

```
用户输入（自然语言 / CLI 参数 / .ak 文件 / Python API）
  │
  ├── 自然语言 → LLM 生成 .ak       ← LLM-Native 入口
  ├── CLI/API → 算子目录查找        ← 结构化入口
  └── .ak 文件 → Lark Parser         ← 精确控制入口
  │
  ▼
Semantic IR (JSON)                      ← "算什么"（纯语义）
  │
  ├── auto_analysis                     ← 自动特征分析（FLOPS、memory、AI、fusion）
  │
  ▼
ArkeEnv + LLM 循环（Bounded Action Space）
  │
  │  get_legal_actions()                ← 编译器提供合法动作集
  │  apply_decision() × N               ← LLM 逐步做决策（附 @rationale）
  │    └── V0 静态验证（每步自动）       ← 保证决策合法性
  │  compile_and_profile()              ← V1 数值 + V2 性能验证
  │  rollback / checkpoint              ← 探索与回退
  │
  ▼
Strategy IR (JSON)                      ← "怎么优化"（LLM 构建，含 @rationale）
  │
  ├── 路径 A: 模板 Codegen
  │     └── Jinja2 渲染 → Triton 代码
  │
  ├── 路径 B: LLM Codegen
  │     └── LLM 生成 → Triton 代码
  │
  ▼
Triton 代码
  │
  ├── NVIDIA: triton.compile() → GPU Binary
  └── Ascend:  triton-ascend → NPU Binary (Phase 2)
  │
  ▼
输出与集成
  ├── kernel.py                         ← 可直接使用的 Triton kernel
  ├── strategy.json                     ← 优化策略（可复现）
  ├── trajectory.jsonl                  ← 学习数据（含 @rationale）
  ├── report.json                       ← 性能报告
  │
  ├── KernelCache                       ← 缓存管理（自动 dispatch）
  ├── PyTorch custom op                 ← 整模型集成（当前）
  └── torch.compile backend             ← 零开销集成（G8/G9）
  │
  ▼
Benchmark 验证（BL×L 体系）
  │
  ├── L1: 单算子 benchmark              ← 45 ops × ST1-4
  ├── L2: 融合算子 benchmark            ← matmul+gelu, rmsnorm+residual...
  └── L3: 模型端到端 benchmark          ← GPT-2, LLaMA-2/3, Qwen2.5, DS-V2
  │
  ▼
Gate 出口判定（G0-G9）
  └── Phase 1 完成 → Phase 2（MLIR Dialect + Ascend 后端）
```

---

*文档版本：v0.1.0 | 创建日期：2026-03-31 | 更新：2026-04-05（对齐 45 ops + BL/OT/ST/L 体系 + Gate 系统）*
