# Vector Tasks — Elementwise & Reduce

## Elementwise Operations

Operators: `relu`, `gelu`, `add`, `mul`

### Shape Matrix

| ID | Name | Op | Shape | Elements | Tier | Purpose |
|:---|:-----|:---|:------|:---------|:----:|:--------|
| V01 | relu_small | relu | [256, 256] | 65K | 3 | Small baseline |
| V02 | relu_medium | relu | [1024, 1024] | 1M | 1 | Standard elementwise |
| V03 | add_large | add | [4096, 4096] | 16M | 1 | Bandwidth-bound, binary op |
| V04 | gelu_tall | gelu | [8192, 128] | 1M | 2 | Tall narrow, row-major access |
| V05 | mul_wide | mul | [128, 8192] | 1M | 2 | Wide short, column access pattern |
| V06 | relu_1row | relu | [1, 65536] | 64K | 3 | Single row vector |
| V07 | gelu_1col | gelu | [65536, 1] | 64K | 3 | Single column vector |
| V08 | add_unaligned | add | [1000, 1000] | 1M | 2 | Non-power-of-2 |
| V09 | relu_prime | relu | [997, 1009] | ~1M | 3 | Prime dimensions |
| V10 | mul_1d | mul | [1, 1048576] | 1M | 3 | Pure 1D vector |
| V11 | relu_w32_n33 | relu | [1024, 33] | 33K | 1 | N%32=1, single extra element |
| V12 | add_w32_n65 | add | [1024, 65] | 66K | 1 | N%32=1, two warps + 1 |
| V13 | gelu_w32_n100 | gelu | [2048, 100] | 200K | 2 | N%32=4, non-trivial remainder |
| V14 | mul_w32_n127 | mul | [2048, 127] | 260K | 2 | N=128-1, almost full warp block |
| V15 | relu_w32_n1 | relu | [4096, 1] | 4K | 3 | N=1, degenerate single-element cols |
| V16 | add_w32_n17 | add | [4096, 17] | 68K | 3 | N=17, prime < warp_size |

### Design Rationale

**Square** (V01-V03): Scale from small to large. V03 is large enough to saturate memory bandwidth — the real bottleneck for elementwise.

**Non-square** (V04-V05): Different access patterns. Tall-narrow is row-contiguous; wide-short may cause strided access in row-major layout.

**Edge cases** (V06-V07, V10): Single-dimensional tensors. Tests vectorization and block size selection when one dimension is trivial.

**Alignment** (V08-V09): Non-power-of-2 dims need masking.

**Warp-32 不对齐** (V11-V16): 当 N 不是 32 的倍数时，最后一个 warp 需要 mask。这是 GPU kernel 常见 bug 来源：
- V11 (N=33): 最简单的不对齐，1 个 warp + 1 个元素溢出
- V12 (N=65): 2 个 warp + 1 溢出，测试多 warp 场景
- V13 (N=100): 3 个 warp + 4 溢出，非 trivial remainder
- V14 (N=127): 差 1 填满 4 个 warp，测试 mask 边界
- V15 (N=1): 极端退化，每行只有 1 个元素
- V16 (N=17): 质数且 < warp_size，不到一个完整 warp

## Reduce Operations

Operators: `softmax`, `reduce_sum`, `reduce_max`

### Shape Matrix

| ID | Name | Op | Shape | Reduce Dim | Tier | Purpose |
|:---|:-----|:---|:------|:-----------|:----:|:--------|
| R01 | softmax_short | softmax | [4096, 64] | 64 | 1 | Fits in 2 warps (aligned) |
| R02 | reduce_sum_medium | reduce_sum | [4096, 1024] | 1024 | 2 | Medium reduction (aligned) |
| R03 | softmax_4096 | softmax | [4096, 4096] | 4096 | 1 | Standard large softmax (aligned) |
| R04 | reduce_sum_long | reduce_sum | [1024, 16384] | 16384 | 2 | Very long reduction |
| R05 | reduce_max_1row | reduce_max | [1, 65536] | 65536 | 3 | Single-row full reduction |
| R06 | softmax_many_short | softmax | [16384, 32] | 32 | 3 | Many rows, exactly 1 warp |
| R07 | reduce_sum_unaligned | reduce_sum | [1000, 1000] | 1000 | 3 | Non-aligned reduction |
| R08 | softmax_w32_n33 | softmax | [4096, 33] | 33 | 1 | N%32=1, softmax mask边界 |
| R09 | reduce_sum_w32_n65 | reduce_sum | [4096, 65] | 65 | 2 | 2 warps + 1, sum mask |
| R10 | softmax_w32_n100 | softmax | [2048, 100] | 100 | 2 | N%32=4, softmax remainder |
| R11 | reduce_max_w32_n127 | reduce_max | [2048, 127] | 127 | 3 | Almost 4 warps, max mask |
| R12 | softmax_w32_n17 | softmax | [4096, 17] | 17 | 3 | Prime < warp, softmax 极短 |

### Design Rationale

**Reduction dimension length** is the critical variable:
- **Short** (R01, R06): Reduction fits in one or two warps. No cross-warp reduction needed. Tests whether kernel avoids unnecessary overhead.
- **Medium** (R02): Needs block-level reduction. Standard case.
- **Long** (R03, R04): May need multi-pass or tree reduction. Memory throughput becomes important.
- **Very long** (R05): Single-row extreme — tests full vector reduction.

**Warp-32 不对齐 reduction** (R08-R12): 归约维度不是 32 的倍数时，warp-level reduction 必须正确 mask 无效 lane：
- R08 (N=33): softmax 归约 33 元素 — 1 warp + 1，max/sum 必须排除 lane 1-31 的无效值
- R09 (N=65): reduce_sum 65 元素 — 跨 warp 归约 + mask
- R10 (N=100): softmax 100 元素 — 3 warp + 4，中等不对齐
- R11 (N=127): reduce_max 127 元素 — mask 最后 1 个无效 lane（容易 off-by-one）
- R12 (N=17): softmax 17 元素 — 不到一个完整 warp，所有运算都需要 mask

**softmax 不对齐尤其危险**：max/exp/sum/div 四步都涉及归约，任何一步 mask 错误都会导致：
- max 取到未初始化值 → exp 溢出 → NaN
- sum 多算元素 → 概率分布不归一

**softmax** is special: needs `max → exp → sum → div` — four passes over the data (or fused online softmax). Short vs long reduction dimension determines whether online softmax is beneficial.

## Performance Baselines

| Op Category | Baseline | How to Measure |
|:------------|:---------|:---------------|
| elementwise | `torch.relu(x)`, `torch.add(a,b)` etc. | `torch.cuda.Event` timing |
| softmax | `F.softmax(x, dim=-1)` | `torch.cuda.Event` timing |
| reduce_sum | `torch.sum(x, dim=-1)` | `torch.cuda.Event` timing |
| reduce_max | `torch.max(x, dim=-1)` | `torch.cuda.Event` timing |

Target: Triton kernel should be ≥ 90% of PyTorch eager for reduce ops, ≥ 100% for elementwise (Triton should beat eager due to fusion opportunity).
