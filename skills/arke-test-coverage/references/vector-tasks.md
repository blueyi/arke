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

### Design Rationale

**Square** (V01-V03): Scale from small to large. V03 is large enough to saturate memory bandwidth — the real bottleneck for elementwise.

**Non-square** (V04-V05): Different access patterns. Tall-narrow is row-contiguous; wide-short may cause strided access in row-major layout.

**Edge cases** (V06-V07, V10): Single-dimensional tensors. Tests vectorization and block size selection when one dimension is trivial.

**Alignment** (V08-V09): Same as CUBE — non-aligned dims need masking.

## Reduce Operations

Operators: `softmax`, `reduce_sum`, `reduce_max`

### Shape Matrix

| ID | Name | Op | Shape | Reduce Dim | Tier | Purpose |
|:---|:-----|:---|:------|:-----------|:----:|:--------|
| R01 | softmax_short | softmax | [4096, 64] | 64 | 1 | Fits in warp (≤2×warp_size) |
| R02 | reduce_sum_medium | reduce_sum | [4096, 1024] | 1024 | 2 | Medium reduction |
| R03 | softmax_4096 | softmax | [4096, 4096] | 4096 | 1 | Standard large softmax |
| R04 | reduce_sum_long | reduce_sum | [1024, 16384] | 16384 | 2 | Very long reduction |
| R05 | reduce_max_1row | reduce_max | [1, 65536] | 65536 | 3 | Single-row full reduction |
| R06 | softmax_many_short | softmax | [16384, 32] | 32 | 3 | Many rows, tiny reduction |
| R07 | reduce_sum_unaligned | reduce_sum | [1000, 1000] | 1000 | 3 | Non-aligned reduction |

### Design Rationale

**Reduction dimension length** is the critical variable:
- **Short** (R01, R06): Reduction fits in one or two warps. No cross-warp reduction needed. Tests whether kernel avoids unnecessary overhead.
- **Medium** (R02): Needs block-level reduction. Standard case.
- **Long** (R03, R04): May need multi-pass or tree reduction. Memory throughput becomes important.
- **Very long** (R05): Single-row extreme — tests full vector reduction.

**softmax** is special: needs `max → exp → sum → div` — four passes over the data (or fused online softmax). Short vs long reduction dimension determines whether online softmax is beneficial.

## Performance Baselines

| Op Category | Baseline | How to Measure |
|:------------|:---------|:---------------|
| elementwise | `torch.relu(x)`, `torch.add(a,b)` etc. | `torch.cuda.Event` timing |
| softmax | `F.softmax(x, dim=-1)` | `torch.cuda.Event` timing |
| reduce_sum | `torch.sum(x, dim=-1)` | `torch.cuda.Event` timing |
| reduce_max | `torch.max(x, dim=-1)` | `torch.cuda.Event` timing |

Target: Triton kernel should be ≥ 90% of PyTorch eager for reduce ops, ≥ 100% for elementwise (Triton should beat eager due to fusion opportunity).
