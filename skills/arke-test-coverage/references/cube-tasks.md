# CUBE Tasks — Matmul & Batch Matmul

## Matmul (2D)

| ID | Name | M | N | K | Tier | Purpose |
|:---|:-----|--:|--:|--:|:----:|:--------|
| C01 | matmul_small | 256 | 256 | 256 | 2 | Small square, blocks < SMs |
| C02 | matmul_1024 | 1024 | 1024 | 1024 | 1 | Standard benchmark |
| C03 | matmul_2048 | 2048 | 2048 | 2048 | 1 | High compute intensity |
| C04 | matmul_xlarge | 4096 | 4096 | 4096 | 2 | Extreme compute, memory pressure |
| C05 | matmul_tall | 4096 | 256 | 1024 | 1 | M >> N, high row parallelism |
| C06 | matmul_wide | 256 | 4096 | 1024 | 2 | N >> M, high column parallelism |
| C07 | matmul_deep_k | 1024 | 1024 | 4096 | 2 | Long reduction loop |
| C08 | matmul_shallow_k | 1024 | 1024 | 64 | 2 | Short reduction, Attention-like |
| C09 | matmul_attn_qk | 1024 | 1024 | 64 | 3 | Attention Q·K^T, head_dim=64 |
| C10 | matmul_ffn_up | 1024 | 4096 | 1024 | 3 | Transformer FFN expand |
| C11 | matmul_ffn_down | 1024 | 1024 | 4096 | 3 | Transformer FFN compress |
| C12 | matmul_unaligned | 997 | 1009 | 1013 | 1 | Prime dims, boundary handling |
| C13 | matmul_round | 1000 | 1000 | 1000 | 3 | Non-power-of-2, common in practice |
| C14 | matmul_tiny | 16 | 16 | 16 | 3 | Smaller than one block |
| C15 | matmul_extreme | 8192 | 32 | 1024 | 3 | Extreme M/N ratio 256:1 |

### Shape Design Rationale

**Square matrices** (C01-C04): Test scaling behavior from small to large. C01 has fewer blocks than SMs → underutilization. C04 stresses memory bandwidth.

**Rectangular** (C05-C08): Different M:N:K ratios stress different tiling strategies. Tall/wide test parallelism balance. Deep K tests reduction loop efficiency. Shallow K tests whether the kernel handles few iterations well.

**Transformer shapes** (C09-C11): Real-world shapes from attention and FFN layers.

**Alignment** (C12-C13): Non-power-of-2 dimensions require mask handling at tile boundaries. Prime dimensions ensure no tile size evenly divides.

**Edge cases** (C14-C15): Extreme sizes test robustness.

## Batch Matmul (3D)

| ID | Name | B | M | N | K | Tier | Purpose |
|:---|:-----|--:|--:|--:|--:|:----:|:--------|
| CB01 | bmm_small | 8 | 512 | 512 | 512 | 2 | Small batch |
| CB02 | bmm_attn | 32 | 512 | 512 | 64 | 2 | Typical attention heads |
| CB03 | bmm_large_batch | 128 | 256 | 256 | 64 | 3 | Large batch, small matrices |
| CB04 | bmm_single | 1 | 1024 | 1024 | 1024 | 3 | Degenerate B=1 |

## dtype Coverage

| dtype | Tier | Notes |
|:------|:----:|:------|
| f16 | 1 | Default for all tiers |
| f32 | 2 | Added in Tier 2 for matmul_1024, matmul_2048 |
| bf16 | 3 | Full dtype coverage |
