# G6.4 Token Efficiency Report

**Pass rate:** 8/8  **Avg ratio (.ak/Triton):** 0.321

| File | .ak lines | Triton lines | Ratio | Pass |
|------|----------:|-------------:|------:|:----:|
| 01_matmul.ak | 7 | 73 | 0.096 | ✅ |
| 02_softmax.ak | 16 | 49 | 0.327 | ✅ |
| 03_gelu.ak | 14 | 35 | 0.400 | ✅ |
| 04_layernorm.ak | 18 | 60 | 0.300 | ✅ |
| 05_matmul_gelu.ak | 28 | 74 | 0.378 | ✅ |
| 06_rmsnorm.ak | 17 | 58 | 0.293 | ✅ |
| 07_silu.ak | 14 | 35 | 0.400 | ✅ |
| 08_batch_matmul.ak | 27 | 73 | 0.370 | ✅ |

**G6.4 criterion:** `.ak` code lines < generated Triton code lines
