# Triton IR 与 Ascend NPU IR 算子统一分类

> 目标：将 Triton (tl.*) 和 Ascend NPU (AscendC) 的全部算子归入不超过 6 个大类，
> 为 Arke 统一抽象层提供参考。

---

## 统一分类体系（6 大类）

| 编号 | 大类 | 含义 | 关键词 |
|:---:|------|------|--------|
| **C1** | **数据搬运 (Data Movement)** | 数据在不同内存层级之间的传输 | load, store, copy, DMA, prefetch |
| **C2** | **Vector 计算 (Vector Compute)** | 逐元素 / 逐向量的算术与数学运算 | add, mul, exp, relu, elementwise |
| **C3** | **Cube 计算 (Matrix Compute)** | 矩阵 / 张量级计算（利用矩阵计算单元） | matmul, dot, MMA, Cube |
| **C4** | **归约与扫描 (Reduce & Scan)** | 沿轴聚合、前缀扫描、排序 | sum, max, reduce, cumsum, sort |
| **C5** | **张量管理 (Tensor Management)** | 张量的创建、形状变换、类型转换、索引 | reshape, cast, arange, where |
| **C6** | **核管理与同步 (Kernel Control)** | 程序标识、同步屏障、编译器提示、原子操作 | program_id, barrier, atomic, hint |

---

## Triton IR (tl.*) 算子归类

| 大类 | Triton 原始分类 | 算子 |
|:---:|-----------------|------|
| **C1 数据搬运** | Memory/Pointer Ops | `load`, `store`, `make_tensor_descriptor`, `load_tensor_descriptor`, `store_tensor_descriptor`, `make_block_ptr`, `advance` |
| **C2 Vector 计算** | Math Ops | `abs`, `ceil`, `floor`, `cos`, `sin`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `sqrt_rn`, `rsqrt`, `sigmoid`, `softmax`, `erf`, `fma`, `clamp`, `maximum`, `minimum`, `cdiv`, `div_rn`, `fdiv`, `umulhi` |
| **C3 Cube 计算** | Linear Algebra Ops | `dot`, `dot_scaled` |
| **C4 归约与扫描** | Reduction Ops | `sum`, `max`, `min`, `argmax`, `argmin`, `reduce`, `xor_sum` |
| | Scan/Sort Ops | `associative_scan`, `cumsum`, `cumprod`, `histogram`, `sort`, `topk`, `gather` |
| **C5 张量管理** | Creation Ops | `arange`, `cat`, `full`, `zeros`, `zeros_like`, `cast` |
| | Shape Manipulation Ops | `broadcast`, `broadcast_to`, `expand_dims`, `interleave`, `join`, `permute`, `ravel`, `reshape`, `split`, `trans`, `view` |
| | Indexing Ops | `flip`, `where`, `swizzle2d` |
| **C6 核管理与同步** | Programming Model | `program_id`, `num_programs` |
| | Atomic Ops | `atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_xchg`, `atomic_xor` |
| | Compiler Hint Ops | `assume`, `debug_barrier`, `max_constancy`, `max_contiguous`, `multiple_of` |
| | Random Number Generation | `rand`, `randn`, `randint`, `randint4x` |
| | Iterators | `range`, `static_range` |
| | Inline Assembly | `inline_asm_elementwise` |

---

## Ascend NPU (AscendC) 算子归类

| 大类 | AscendC 原始分类 | 算子 |
|:---:|-----------------|------|
| **C1 数据搬运** | Data Transfer (DMA) | `DataCopy` (GM→L1, L1→L0A/B, L0C→GM, etc.), `DataCopyPad`, `DataCopyEnhanced` |
| | Async DMA | `EnQue`, `DeQue`, `AsyncDataCopy`, `WaitFlag` |
| | Gather/Scatter | `GatherMask`, `ScatterMask`, `DataCopyExtParams` |
| **C2 Vector 计算** | Vec Unary | `Abs`, `Exp`, `Log`, `Sqrt`, `Rsqrt`, `Reciprocal`, `Relu`, `Sigmoid`, `Tanh`, `Ceil`, `Floor`, `Round`, `Not`, `Cast` (vec) |
| | Vec Binary | `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `And`, `Or`, `BitwiseAnd`, `BitwiseOr`, `ShiftLeft`, `ShiftRight` |
| | Vec Compare | `Compare` (EQ/NE/GT/GE/LT/LE) |
| | Vec Select | `Select`, `TopK` (vec-level) |
| | Vec Advanced | `Muls` (scalar-vec), `Adds` (scalar-vec), `Axpy`, `Gelu`, `Silu`, `Swish`, `Softmax` (vec), `LayerNorm` (vec), `DropOut` |
| **C3 Cube 计算** | Cube Ops | `Mmad` (Matrix Multiply-Accumulate), `CubeMatMul`, `BatchMatMul` |
| | Cube Config | `SetCubeFormat`, `SetCubeConfig`, `EnQueCube`, `DeQueCube` |
| **C4 归约与扫描** | Vec Reduce | `ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `ReduceCustom` |
| | WholeReduceSum | `WholeReduceSum`, `WholeReduceMax`, `WholeReduceMin` |
| | Scan | `PrefixSum` (if available in specific hardware version) |
| **C5 张量管理** | Tensor Creation | `AllocTensor`, `FreeTensor` |
| | Shape/Layout | `TransposeMatrix`, `Reshape` (logical), `SetDataFormat` |
| | Type Conversion | `Cast` (type-level), `QuantConvert`, `DeQuantConvert` |
| | Pipe Tensor | `TensorBuf`, `LocalTensor` types |
| **C6 核管理与同步** | Pipe/Sync | `PipeBarrier`, `SetFlag`, `WaitFlag`, `SyncAll` |
| | Core Management | `GetBlockIdx`, `GetBlockNum`, `GetBlockDim` |
| | Pipe Management | `TPipe` (TQue init), `InitBuffer`, `EnQue`, `DeQue` (control plane) |
| | Event | `SetEvent`, `WaitEvent`, `ProfilerStart`, `ProfilerStop` |
| | Scalar Ops | `SetAtomicAdd`, `SetAtomicMax`, `SetAtomicNone` |

---

## 对照总表

| 大类 | Triton 算子数 | Ascend 算子数 | Arke 抽象方向 |
|:---:|:---:|:---:|------|
| **C1 数据搬运** | 7 | ~15 | 声明式 `place()` + 自动搬运策略 |
| **C2 Vector 计算** | 24 | ~40 | 语义层数学表达式，策略层选指令 |
| **C3 Cube 计算** | 2 | ~5 | 语义层 matmul/reduce_sum，策略层选矩阵单元 |
| **C4 归约与扫描** | 11 | ~8 | 语义层 reduce_xxx(axis=)，策略层选实现方式 |
| **C5 张量管理** | 14 | ~8 | 类型系统 + 形状推导自动处理 |
| **C6 核管理与同步** | 17 | ~12 | 声明式 parallel() + 自动同步 |

---

## Arke 的设计启示

### 1. C1 数据搬运：Triton 显式 vs Ascend 更显式 vs Arke 声明式

```
Triton:  tl.load(ptr + offsets, mask=mask)     ← 显式指针操作
Ascend:  DataCopy(dst_local, src_gm, count)    ← 显式 DMA 调度
Arke:    place(A_tile -> shared)                ← 声明意图，工具链/LLM 决定实现
```

Arke 在语义层不出现 load/store，在策略层用 `place()` 声明数据放在哪。
具体的搬运指令由 Lowering 层根据硬件生成。

### 2. C2 vs C3：Vector 和 Cube 在语义层不区分

```
语义层：
  C[i,j] = reduce_sum(A[i,k] * B[k,j], axis=k)    ← 不关心是 Cube 还是 Vector

策略层：
  compute(matmul -> tensor_core.mma.m16n8k16)       ← NVIDIA: Cube
  compute(matmul -> cube_unit.matmul.f16)            ← Ascend: Cube
  compute(relu -> vec)                               ← 两者都是 Vector
```

在语义层，AI 描述数学；在策略层，AI 选择用什么计算单元执行。
这是 Arke 语义/策略分离的核心价值。

### 3. C6 核管理：Triton 半自动 vs Ascend 手动 vs Arke 声明式

```
Triton:  pid = tl.program_id(axis=0)            ← 半自动
Ascend:  blockIdx = GetBlockIdx()                ← 手动
         PipeBarrier(PIPE_ALL)                   ← 手动同步
Arke:    parallel(i_outer -> grid.x)             ← 声明映射
         pipeline(stages=[...], depth=2)         ← 声明流水线
         // 同步由编译器/LLM 自动插入
```

### 4. 分类对 Arke 语言设计的映射

```
C1 数据搬运  →  策略层 place() / prefetch() / double_buffer()
C2 Vector    →  语义层数学表达（+,-,*,exp,relu...）
C3 Cube      →  语义层 matmul/reduce_sum，策略层 compute()
C4 归约扫描  →  语义层 reduce_xxx(axis=)
C5 张量管理  →  类型系统自动处理（Tensor<shape,dtype>）
C6 核管理    →  策略层 parallel() / pipeline() / @constraint
```

**语义层只出现 C2+C3+C4 的数学操作。**
**策略层处理 C1+C6（搬运和调度）以及 C2/C3 的指令选择。**
**C5 由类型系统和编译器自动处理，不在语言中显式出现。**

---

*本文档配合 docs/design/e2e-design-v2.md 使用。*
