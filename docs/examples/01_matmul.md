# Example 01: 从自然语言到 GPU Kernel — 矩阵乘法

> 本文档以 matmul 为例，演示 Arke 如何从一句自然语言描述出发，经过 `.ak` 源码 → Semantic IR → Strategy IR → Triton kernel 的完整链路。

---

## 1. 自然语言描述

> **"计算两个 1024×1024 的半精度矩阵乘法 C = A × B"**

这就是用户的全部输入。Arke 的 NL→IR 前端会从中提取：
- **算子**：matmul
- **形状**：A=[1024, 1024], B=[1024, 1024], C=[1024, 1024]
- **数据类型**：f16 (半精度)

---

## 2. `.ak` 源文件

```arke
// Example 01: Matrix Multiplication (matmul)
//
// Natural Language:
//   "计算两个 1024×1024 的半精度矩阵乘法 C = A × B"
//
// Semantics: C[i,j] = sum(A[i,k] * B[k,j], axis=k)
// Category: A (compute)

kernel matmul(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 1024], f16>
) -> Tensor<[1024, 1024], f16> {
    let C = matmul(A=A, B=B);
    return C;
}

// optional
strategy matmul for target("nvidia_ampere") {
    tile(loop="M", factors=[128])
        @rationale("128 rows per thread block for L1 reuse");
    tile(loop="N", factors=[128])
        @rationale("128 cols for balanced occupancy");
    tile(loop="K", factors=[32])
        @rationale("K=32 for shared memory tiling");
    reorder(order=["M", "N", "K"])
        @rationale("outer M/N for parallelism, inner K for reduction");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("each thread block computes one 128×128 output tile");
    place(tensor="A_tile", memory="shared")
        @rationale("tile A in shared memory for K-loop reuse");
    place(tensor="B_tile", memory="shared")
        @rationale("tile B in shared memory for K-loop reuse");
}
```

`.ak` 文件由两个块组成：
- **`kernel` 块** — 声明计算（WHAT）：输入张量、算子调用、返回值
- **`strategy` 块** — 声明优化策略（HOW）：分块、排序、并行映射、内存放置

---

## 3. Semantic IR（算什么）

`.ak` 的 `kernel` 块经 parser 转换为 Semantic IR（JSON）：

```json
{
  "version": "0.2.0",
  "kernel_id": "matmul_1024",
  "params": [
    { "name": "A", "shape": [1024, 1024], "dtype": "f16", "layout": "row_major" },
    { "name": "B", "shape": [1024, 1024], "dtype": "f16", "layout": "row_major" }
  ],
  "return_type": { "shape": [1024, 1024], "dtype": "f16", "layout": "row_major" },
  "nodes": [
    {
      "id": "matmul_0",
      "op": "matmul",
      "inputs": {
        "A": { "ref": "param", "name": "A" },
        "B": { "ref": "param", "name": "B" }
      },
      "output": { "shape": [1024, 1024], "dtype": "f16" },
      "semantics": {
        "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
        "index_vars": ["i", "j", "k"],
        "reduction_axes": ["k"],
        "properties": ["associative", "distributive"]
      }
    }
  ],
  "edges": [],
  "return_node": "matmul_0",
  "fusion_groups": []
}
```

### 解读

| 字段 | 含义 |
|------|------|
| `params` | 输入张量声明：名字、形状、类型、内存布局 |
| `nodes[0].op` | 算子名，来自 OP_CATALOG（13 个已注册算子） |
| `nodes[0].inputs` | 数据流引用——`A` 和 `B` 指向 `params` 中的输入 |
| `nodes[0].semantics.computation` | **数学定义**：`C[i,j] = Σ_k A[i,k] × B[k,j]` |
| `nodes[0].semantics.reduction_axes` | 约减维度 `k`——沿 K 维求和 |
| `nodes[0].semantics.properties` | 代数性质：结合律、分配律——告诉优化器可以安全地重排/分块 |

**关键特征：没有任何关于 tiling、内存、线程的信息。** Semantic IR 是纯数学——它定义正确性的锚点，LLM Agent 读它来理解"要算什么"，但不能修改它。

---

## 4. Strategy IR（怎么优化）

`.ak` 的 `strategy` 块（或由 LLM Agent 生成）转换为 Strategy IR：

```json
{
  "version": "0.2.0",
  "kernel_id": "matmul_1024",
  "target_hw": "nvidia_ampere",
  "decisions": [
    {
      "step": 1,
      "kind": "tile",
      "params": { "loop": "M", "factors": [128] },
      "rationale": { "text": "128 rows per thread block for L1 reuse" }
    },
    {
      "step": 2,
      "kind": "tile",
      "params": { "loop": "N", "factors": [128] },
      "rationale": { "text": "128 cols for balanced occupancy" }
    },
    {
      "step": 3,
      "kind": "tile",
      "params": { "loop": "K", "factors": [32] },
      "rationale": { "text": "K=32 for shared memory tiling" }
    },
    {
      "step": 4,
      "kind": "reorder",
      "params": { "order": ["M", "N", "K"] },
      "rationale": { "text": "outer M/N for parallelism, inner K for reduction" }
    },
    {
      "step": 5,
      "kind": "parallel",
      "params": {
        "loops": ["M", "N"],
        "mapping": { "M": "blockIdx.x", "N": "blockIdx.y" }
      },
      "rationale": { "text": "each thread block computes one 128×128 output tile" }
    },
    {
      "step": 6,
      "kind": "place",
      "params": { "tensor": "A_tile", "memory": "shared" },
      "rationale": { "text": "tile A in shared memory for K-loop reuse" }
    },
    {
      "step": 7,
      "kind": "place",
      "params": { "tensor": "B_tile", "memory": "shared" },
      "rationale": { "text": "tile B in shared memory for K-loop reuse" }
    }
  ],
  "constraints": { "shared_memory_limit": 0, "warp_size": 32 }
}
```

### 逐步解读

| Step | Decision | 做了什么 | 为什么（@rationale） |
|:----:|----------|---------|---------------------|
| 1 | `tile(M, [128])` | 把 M=1024 切成 8 块，每块 128 行 | 128 行适合 L1 cache 大小 |
| 2 | `tile(N, [128])` | 把 N=1024 切成 8 块，每块 128 列 | 平衡 GPU 占用率 |
| 3 | `tile(K, [32])` | 把 K=1024 切成 32 块，每块 32 | shared memory 容量限制 |
| 4 | `reorder([M,N,K])` | 循环顺序：外层 M/N，内层 K | M/N 并行化，K 做约减 |
| 5 | `parallel(M→blockIdx.x, N→blockIdx.y)` | M/N 映射到 GPU 线程块网格 | 8×8=64 个线程块并行 |
| 6 | `place(A_tile→shared)` | A 的分块放 shared memory | K 循环内重复访问，避免每次从 global memory 读 |
| 7 | `place(B_tile→shared)` | B 的分块放 shared memory | 同上 |

**关键特征：每个决策都带 `@rationale`。** 如果性能不好，可以追问 Agent "step 3 为什么选 K=32 而不是 64？"，Agent 可以回溯推理链并调整。

---

## 5. Strategy 块是可选的

`strategy` 块**不是必须的**。只写 `kernel` 块就是合法的 `.ak` 文件：

```arke
kernel matmul(
    A: Tensor<[1024, 1024], f16>,
    B: Tensor<[1024, 1024], f16>
) -> Tensor<[1024, 1024], f16> {
    let C = matmul(A=A, B=B);
    return C;
}
// ← 无 strategy 块，编译器自动生成
```

当缺少 strategy 块时，`DefaultStrategyGenerator` 根据硬件 profile 自动生成：

```
输入: kernel 块 + 目标硬件
         ↓
  DefaultStrategyGenerator
    │  读取 hw_profile (arke/ir/targets/nvidia_ampere.json)
    │  识别主导算子类型 (compute / reduce / elementwise / move)
    │  根据 shared_mem 大小、tensor core shape 、warp_size 计算 tile 大小
    ↓
  输出: Strategy IR（每个 Decision 都含 @rationale）
```

自动生成的 7 个决策（RTX 3060 Laptop / Ampere SM 8.6）：

| Step | Decision | 值 | 依据 |
|:----:|---------|---|------|
| 1 | `tile(M)` | 64 | tensor core 16×8×16，对齐倍数 |
| 2 | `tile(N)` | 64 | 同上 |
| 3 | `tile(K)` | 16 | A+B tile = 4096B ≤ smem/2 (24576B) |
| 4 | `reorder([M,N,K])` | 外层M/N, 内层K | 并行局部性 |
| 5 | `parallel(M→blockIdx.x, N→blockIdx.y)` | 2D 网格 | 独立输出 tile |
| 6 | `place(A_tile→shared)` | shared mem | K 循环重用 A |
| 7 | `place(B_tile→shared)` | shared mem | K 循环重用 B |

如果不满意默认策略，可以任意添加 `strategy` 块覆盖它。**显式 strategy 块总是优先于自动生成的结果。**

使用示例：

```python
from arke.pipeline import ArkePipeline

# 有 strategy 块：使用用户提供的
# 无 strategy 块：自动生成
result = ArkePipeline.from_ak_file(
    "docs/examples/01_matmul.ak",
    target_hw="nvidia_ampere",
)
print(result.strategy_ir["_source"])  # → "auto-generated (no strategy block in 01_matmul.ak)"
print(len(result.strategy_ir["decisions"]))  # → 7
```

---

## 6. 两层 IR 的关系

```
自然语言                .ak 源码                  Semantic IR          Strategy IR         Triton Kernel
"计算 1024×1024   ──→   kernel matmul(         ──→  {                ──→  {              ──→  @triton.jit
 半精度矩阵乘法"          A: Tensor<...>,            "op": "matmul",      "decisions": [      def matmul_kernel(
                          B: Tensor<...>             "semantics":          tile(M,128),         A, B, C, ...):
                        ) { ... }                     "C[i,j]=Σ..."        tile(N,128),        ...
                        strategy matmul {          }                       tile(K,32),
                          tile(...)                                         ...
                          parallel(...)                                   ]
                        }                                                }
```

| 层 | 职责 | 谁写 | 可变？ |
|----|------|------|--------|
| 自然语言 | 用户意图 | 用户 | — |
| `.ak` 源码 | 结构化的计算+策略声明 | NL 前端 or 用户 | 是 |
| Semantic IR | 纯数学定义（正确性锚点） | parser 自动生成 | **不可变** |
| Strategy IR | 优化决策序列 | LLM Agent 生成/迭代 | **可迭代** |
| Triton Kernel | GPU 可执行代码 | codegen 自动生成 | 不直接编辑 |

**核心设计理念：LLM Agent 只操作 Strategy IR，永远不碰 Semantic IR。**
- 数学正确性由 Semantic IR 保证（不可变）
- 性能优化由 Strategy IR 驱动（LLM 可自由调整、回滚、搜索）
- `@rationale` 让每个决策可解释、可追溯

---

## 7. 复现

```bash
cd /path/to/arke
source ~/.venvs/arke/bin/activate

# Parse .ak → AST
python -c "
from arke.parser.parser import parse_file
prog = parse_file('docs/examples/01_matmul.ak')
print(f'Kernels: {len(prog.kernels)}, Strategies: {len(prog.strategies)}')
"

# Build Semantic IR
python -c "
from arke.ir.builder import KernelBuilder
b = KernelBuilder('matmul_1024')
b.param('A', [1024, 1024], 'f16')
b.param('B', [1024, 1024], 'f16')
n = b.op('matmul', A='A', B='B')
b.returns(n, [1024, 1024], 'f16')
print(b.build().to_json())
"

# Build Strategy IR
python -c "
from arke.ir.strategy import StrategyIR
s = StrategyIR(kernel_id='matmul_1024', target_hw='nvidia_ampere')
s.tile('M', [128], rationale='128 rows per thread block for L1 reuse')
s.tile('N', [128], rationale='128 cols for balanced occupancy')
s.tile('K', [32], rationale='K=32 for shared memory tiling')
s.reorder(['M','N','K'], rationale='outer M/N for parallelism, inner K for reduction')
s.parallel(['M','N'], {'M':'blockIdx.x','N':'blockIdx.y'},
    rationale='each thread block computes one 128x128 output tile')
s.place('A_tile', 'shared', rationale='tile A in shared memory for K-loop reuse')
s.place('B_tile', 'shared', rationale='tile B in shared memory for K-loop reuse')
print(s.to_json())
"
```
