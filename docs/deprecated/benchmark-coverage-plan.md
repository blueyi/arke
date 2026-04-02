# Benchmark 场景覆盖方案

## 设计目标

全面覆盖 Vector 类（memory-bound）和 CUBE 类（compute-bound）算子的精度与性能测试，通过多维度 shape 变化暴露 kernel 在不同场景下的行为差异。

---

## 1. 算子分类

### CUBE 类（Compute-bound）
以矩阵乘法为代表，计算密度高，性能瓶颈在计算单元。

| 算子 | 公式 | 输入 | 输出 |
|:-----|:-----|:-----|:-----|
| `matmul` | C[i,j] = Σ A[i,k]·B[k,j] | A[M,K], B[K,N] | C[M,N] |
| `batch_matmul` | C[b,i,j] = Σ A[b,i,k]·B[b,k,j] | A[B,M,K], B[B,K,N] | C[B,M,N] |

### Vector 类（Memory-bound / Elementwise）
计算密度低，性能瓶颈在显存带宽。

| 算子 | 公式 | 输入 | 输出 |
|:-----|:-----|:-----|:-----|
| `relu` | Y = max(X, 0) | X[M,N] | Y[M,N] |
| `gelu` | Y = X·Φ(X) | X[M,N] | Y[M,N] |
| `add` | Y = A + B | A[M,N], B[M,N] | Y[M,N] |
| `mul` | Y = A * B | A[M,N], B[M,N] | Y[M,N] |
| `softmax` | Y[i,j] = exp(X[i,j]) / Σ exp(X[i,:]) | X[M,N] | Y[M,N] |
| `reduce_sum` | Y[i] = Σ X[i,:] | X[M,N] | Y[M] |
| `reduce_max` | Y[i] = max(X[i,:]) | X[M,N] | Y[M] |
| `transpose` | Y[j,i] = X[i,j] | X[M,N] | Y[N,M] |

---

## 2. Shape 维度设计

### 2.1 CUBE 类 — matmul Shape 覆盖

**设计原则：**
- 覆盖方阵/矩形/极端比例
- 覆盖对齐/不对齐（2 的幂 vs 非 2 的幂）
- 覆盖小/中/大规模
- 覆盖 K 维不同比例（reduction 维度对 tile 策略影响大）

| 编号 | 名称 | M | N | K | 分类 | 测试目的 |
|:-----|:-----|--:|--:|--:|:-----|:---------|
| C01 | square_small | 256 | 256 | 256 | 方阵/小 | 基础功能，block < SM 数 |
| C02 | square_medium | 1024 | 1024 | 1024 | 方阵/中 | 标准 benchmark 场景 |
| C03 | square_large | 2048 | 2048 | 2048 | 方阵/大 | 高计算密度，带宽隐藏 |
| C04 | square_xlarge | 4096 | 4096 | 4096 | 方阵/超大 | 极限计算，显存压力 |
| C05 | rect_tall | 4096 | 256 | 1024 | 矩形/高 | M >> N，行并行度高 |
| C06 | rect_wide | 256 | 4096 | 1024 | 矩形/宽 | N >> M，列并行度高 |
| C07 | rect_deep_k | 1024 | 1024 | 4096 | 矩形/深K | K很大，reduction 循环长 |
| C08 | rect_shallow_k | 1024 | 1024 | 64 | 矩形/浅K | K很小，reduction 循环短 |
| C09 | rect_transformer_qk | 1024 | 1024 | 64 | Attention Q·K^T | 典型 Attention head_dim=64 |
| C10 | rect_transformer_ff | 1024 | 4096 | 1024 | FFN 第一层 | 典型 Transformer FFN 展开 |
| C11 | rect_transformer_proj | 1024 | 1024 | 4096 | FFN 第二层 | 典型 Transformer FFN 压缩 |
| C12 | unaligned_prime | 997 | 1009 | 1013 | 非对齐/质数 | 不可整除 tile，边界处理 |
| C13 | unaligned_odd | 1000 | 1000 | 1000 | 非对齐/整数 | 常见但不是 2 的幂 |
| C14 | tiny | 16 | 16 | 16 | 极小 | 比 block 还小的矩阵 |
| C15 | extreme_ratio | 8192 | 32 | 1024 | 极端比例 | M/N 比例 256:1 |

### 2.2 CUBE 类 — batch_matmul Shape 覆盖

| 编号 | 名称 | B | M | N | K | 测试目的 |
|:-----|:-----|--:|--:|--:|--:|:---------|
| CB01 | batch_small | 8 | 512 | 512 | 512 | 小 batch |
| CB02 | batch_medium | 32 | 512 | 512 | 64 | 典型 Attention batch |
| CB03 | batch_large | 128 | 256 | 256 | 64 | 大 batch 小矩阵 |
| CB04 | batch_single | 1 | 1024 | 1024 | 1024 | B=1 退化为 matmul |

### 2.3 Vector 类 — Elementwise Shape 覆盖

**设计原则：**
- 覆盖 1D/2D shape
- 覆盖行数远大于列数（row-major 顺序访问）和反之
- 覆盖对齐/不对齐

| 编号 | 名称 | Shape | 元素数 | 测试目的 |
|:-----|:-----|:------|-------:|:---------|
| V01 | small_square | [256, 256] | 65K | 小规模基准 |
| V02 | medium_square | [1024, 1024] | 1M | 中等规模 |
| V03 | large_square | [4096, 4096] | 16M | 大规模，显存带宽测试 |
| V04 | tall_narrow | [8192, 128] | 1M | 行多列少 |
| V05 | wide_short | [128, 8192] | 1M | 列多行少 |
| V06 | single_row | [1, 65536] | 64K | 单行向量 |
| V07 | single_col | [65536, 1] | 64K | 单列向量 |
| V08 | unaligned | [1000, 1000] | 1M | 非 2 的幂 |
| V09 | prime | [997, 1009] | ~1M | 质数维度 |
| V10 | large_1d | [1, 1048576] | 1M | 纯 1D 向量 |

### 2.4 Vector 类 — Reduce 操作 Shape 覆盖

**设计原则：**
- reduce_sum/reduce_max 沿最后一个维度归约
- softmax 沿最后一个维度，需要 max + exp + sum + div
- 关键变量：归约维度长度（影响 warp/block reduction 策略）

| 编号 | 名称 | Shape | 归约维度 | 测试目的 |
|:-----|:-----|:------|:---------|:---------|
| R01 | short_reduce | [4096, 64] | 64 | 归约维度 ≤ warp_size*2 |
| R02 | medium_reduce | [4096, 1024] | 1024 | 中等归约 |
| R03 | long_reduce | [4096, 4096] | 4096 | 长归约，需多级 reduction |
| R04 | very_long_reduce | [1024, 16384] | 16384 | 超长归约 |
| R05 | single_row_reduce | [1, 65536] | 65536 | 单行全归约 |
| R06 | many_short_reduce | [16384, 32] | 32 | 很多行短归约 |
| R07 | unaligned_reduce | [1000, 1000] | 1000 | 非对齐归约 |

---

## 3. 数据类型覆盖

| dtype | 场景 | 精度要求 |
|:------|:-----|:---------|
| `f16` (float16) | 训练/推理常用 | atol=0.1, rtol=0.05 |
| `f32` (float32) | 高精度参考 | atol=1e-5, rtol=1e-4 |
| `bf16` (bfloat16) | 训练常用 | atol=0.2, rtol=0.1 |

---

## 4. Fusion 组合覆盖

测试算子融合后的精度和性能。

| 编号 | 组合 | 描述 | 关键测试点 |
|:-----|:-----|:-----|:-----------|
| F01 | matmul + relu | 基础 epilogue fusion | ReLU 零值边界 |
| F02 | matmul + gelu | 复杂 epilogue | GELU 近似精度 |
| F03 | matmul + add | 残差连接 | 广播/shape 匹配 |
| F04 | matmul + add + relu | 残差+激活 | 三算子融合 |
| F05 | matmul + mul | Scale 操作 | 元素乘融合 |
| F06 | softmax + mul | Attention score | 二阶段融合 |

---

## 5. 测试矩阵汇总

### 5.1 完整测试用例数

| 维度 | 选项 | 数量 |
|:-----|:-----|-----:|
| CUBE shapes | C01-C15 + CB01-CB04 | 19 |
| Vector shapes | V01-V10 | 10 |
| Reduce shapes | R01-R07 | 7 |
| 数据类型 | f16, f32, bf16 | 3 |
| Fusion 组合 | F01-F06 | 6 |
| **总计（全覆盖）** | | **126** |

### 5.2 推荐精简方案（核心覆盖集）

全跑 126 个太慢。推荐分层：

**Tier 1 — 核心（每次必跑，~20 个）：**
- matmul: C02, C03, C05, C12 (方阵中/大 + 矩形 + 不对齐)
- vector: V02, V03, V04 (中/大方阵 + 非方阵)
- reduce: R02, R03 (中/长归约)
- fusion: F01, F02, F03 (三种典型 epilogue)
- dtype: 全部用 f16
- → 约 **12 个 task**

**Tier 2 — 扩展（周级别跑，~40 个）：**
- 所有 matmul shapes (C01-C15)
- 所有 vector shapes (V01-V10)
- 所有 reduce shapes (R01-R07)
- f16 + f32 两种 dtype

**Tier 3 — 全覆盖（版本发布前跑）：**
- 全部 126 个组合

---

## 6. 精度验证标准

| dtype | atol | rtol | 说明 |
|:------|:-----|:-----|:-----|
| f16 | 0.1 | 0.05 | FP16 精度有限，允许较大容差 |
| f32 | 1e-5 | 1e-4 | 单精度高精度要求 |
| bf16 | 0.2 | 0.1 | BF16 尾数只有 7 位 |

**特殊场景：**
- softmax: 要求行和 ≈ 1.0（atol=0.01）
- gelu: 在 x ≈ 0 附近精度敏感
- reduce_sum: 大 N 时累积误差，放宽到 atol=0.5 for f16

---

## 7. 性能基准

| 算子类别 | Baseline | 目标 |
|:---------|:---------|:-----|
| matmul | cuBLAS `cublasGemmEx` | ≥ 90% cuBLAS |
| batch_matmul | cuBLAS batched | ≥ 85% cuBLAS |
| elementwise | PyTorch eager | ≥ 100% PyTorch（Triton 应更快） |
| softmax | PyTorch `F.softmax` | ≥ 90% PyTorch |
| reduce_sum/max | PyTorch `torch.sum/max` | ≥ 90% PyTorch |

---

## 8. 实施优先级

### Phase 1（当前 → 立即实施）

扩展现有 6 个 task 到 Tier 1 核心集：

```python
# CUBE 类新增
("matmul_small",    256, 256, 256)     # C01: 小方阵
("matmul_xlarge",   4096, 4096, 4096)  # C04: 超大方阵
("matmul_tall",     4096, 256, 1024)   # C05: 高矩形
("matmul_deep_k",   1024, 1024, 4096)  # C07: 深 K
("matmul_unaligned", 997, 1009, 1013)  # C12: 不对齐

# Vector 类新增
("relu_medium",     1024, 1024)        # V02 + relu
("add_large",       4096, 4096)        # V03 + add
("gelu_tall",       8192, 128)         # V04 + gelu

# Reduce 类新增
("softmax_short",   4096, 64)          # R01: 短 reduce
("softmax_long",    4096, 4096)        # R03: 长 reduce
("reduce_sum_medium", 4096, 1024)      # R02: reduce_sum

# Fusion 新增
("matmul_add",      1024, 1024, 1024)  # F03: 残差连接
```

### Phase 2（后续）

- 添加 batch_matmul 支持
- 添加 f32/bf16 dtype 覆盖
- 实现 Tier 2/3 自动化

---

## 9. 输出格式

每次运行归档到：
```
benchmarks/results/{phase}/{YYYY-MM-DD_HHMMSS}/
├── arke_ir/                 Arke IR 源文件 (.json)
├── triton_kernels/
│   ├── arke/                Arke 编译的 Triton kernel (.py)
│   └── direct/              LLM 直写的 Triton kernel (.py)
├── benchmark_results.csv    完整结果（shape/perf/accuracy/tokens/time）
├── task_catalog.csv         任务定义（shapes/params/ops）
└── benchmark_report.json    原始数据
```
