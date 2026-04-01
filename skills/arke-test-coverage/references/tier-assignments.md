# Tier Assignments

Complete tier assignment for all benchmark tasks.

## Tier 1 — Core (~12 tasks)

Run every time. Quick validation of correctness and performance.

| Task | Category | Shape |
|:-----|:---------|:------|
| matmul_1024 | CUBE | 1024×1024×1024 |
| matmul_2048 | CUBE | 2048×2048×2048 |
| matmul_rect | CUBE | 1024×2048×512 |
| matmul_unaligned | CUBE | 997×1009×1013 |
| matmul_tall | CUBE | 4096×256×1024 |
| relu_medium | Vector | [1024,1024] |
| add_large | Vector | [4096,4096] |
| softmax_short | Reduce | [4096,64] |
| softmax_4096 | Reduce | [4096,4096] |
| fused_matmul_relu | Fusion | 1024×1024×1024 |
| fused_matmul_gelu | Fusion | 1024×2048×1024 |
| fused_matmul_add | Fusion | 1024×1024×1024 |

## Tier 2 — Extended (~40 tasks)

Includes all Tier 1 + additional shapes and ops.

**Added CUBE:**
matmul_small (256³), matmul_xlarge (4096³), matmul_wide (256×4096×1024), matmul_deep_k (1024²×4096), matmul_shallow_k (1024²×64), bmm_small (8×512³), bmm_attn (32×512²×64)

**Added Vector:**
gelu_tall ([8192,128]), mul_wide ([128,8192]), add_unaligned ([1000,1000])

**Added Reduce:**
reduce_sum_medium ([4096,1024]), reduce_sum_long ([1024,16384])

**Added Fusion:**
fused_matmul_add_relu, fused_matmul_mul

**Added dtype:**
matmul_1024_f32, matmul_2048_f32 (f32 variants of core matmuls)

## Tier 3 — Full (~126+ tasks)

All Tier 2 + all remaining shapes, dtypes, edge cases.

**Added CUBE:** C09-C11 (transformer shapes), C13-C15 (edge cases), CB03-CB04 (batch edge cases)

**Added Vector:** V01, V06-V07, V09-V10 (small, single-dim, prime)

**Added Reduce:** R05-R07 (extreme/edge cases)

**Added Fusion:** F06 (softmax+mul)

**Added dtype:** bf16 variants of all core tasks
