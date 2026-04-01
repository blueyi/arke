# Tier Assignments

Complete tier assignment for all benchmark tasks.

## Tier 1 — Core (~16 tasks)

Run every time. Quick validation of correctness and performance.

| Task | Category | Shape | Warp-32 |
|:-----|:---------|:------|:--------|
| matmul_1024 | CUBE | 1024×1024×1024 | aligned |
| matmul_2048 | CUBE | 2048×2048×2048 | aligned |
| matmul_rect | CUBE | 1024×2048×512 | aligned |
| matmul_unaligned | CUBE | 997×1009×1013 | ✗ |
| matmul_tall | CUBE | 4096×256×1024 | aligned |
| relu_medium | Vector | [1024,1024] | aligned |
| add_large | Vector | [4096,4096] | aligned |
| relu_w32_n33 | Vector | [1024,33] | **N%32=1** |
| add_w32_n65 | Vector | [1024,65] | **N%32=1** |
| softmax_short | Reduce | [4096,64] | aligned |
| softmax_4096 | Reduce | [4096,4096] | aligned |
| softmax_w32_n33 | Reduce | [4096,33] | **N%32=1** |
| fused_matmul_relu | Fusion | 1024×1024×1024 | aligned |
| fused_matmul_gelu | Fusion | 1024×2048×1024 | aligned |
| fused_matmul_add | Fusion | 1024×1024×1024 | aligned |

## Tier 2 — Extended (~30 tasks)

Includes all Tier 1 + additional shapes and warp-32 不对齐 cases.

**Added CUBE:**
matmul_small (256³), matmul_xlarge (4096³), matmul_wide (256×4096×1024), matmul_deep_k (1024²×4096), matmul_shallow_k (1024²×64)

**Added Vector:**
gelu_tall ([8192,128]), mul_wide ([128,8192]), add_unaligned ([1000,1000]), gelu_w32_n100 ([2048,100] N%32=4), mul_w32_n127 ([2048,127] N%32=31)

**Added Reduce:**
reduce_sum_medium ([4096,1024]), reduce_sum_long ([1024,16384]), reduce_sum_w32_n65 ([4096,65] N%32=1), softmax_w32_n100 ([2048,100] N%32=4)

**Added Fusion:**
fused_matmul_add_relu, fused_matmul_mul

## Tier 3 — Full (~50 tasks)

All Tier 2 + all remaining shapes, dtypes, edge cases.

**Added CUBE:** C09-C11 (transformer), C13-C15 (edge cases)

**Added Vector:** V01, V06-V07, V09-V10 (small, single-dim, prime), relu_w32_n1 ([4096,1] N=1), add_w32_n17 ([4096,17] N=17 prime)

**Added Reduce:** R05-R07 (extreme/edge), reduce_max_w32_n127 ([2048,127]), softmax_w32_n17 ([4096,17])

**Added Fusion:** F06 (softmax+mul)

## Warp-32 不对齐覆盖汇总

| Tier | Elementwise | Reduce | 覆盖的 N%32 值 |
|:-----|:------------|:-------|:---------------|
| 1 | N=33 (relu), N=65 (add) | N=33 (softmax) | 1 |
| 2 | N=100 (gelu), N=127 (mul) | N=65 (sum), N=100 (softmax) | 1, 4, 31 |
| 3 | N=1 (relu), N=17 (add) | N=127 (max), N=17 (softmax) | 1, 4, 17, 31 |
