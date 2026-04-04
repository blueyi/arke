# Arke Benchmark Report

Generated: 2026-04-02 20:47:12

## Hardware

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU
- **GPU Memory:** 6143 MB
- **CUDA:** 12.4
- **PyTorch:** 2.6.0+cu124
- **Triton:** 3.2.0

## L1: Single Operator Results

### matmul

| Shape | cuBLAS/cuDNN | PyTorch-eager | torch.compile | Arke | FlagGems |
|:---|---:|---:|---:|---:|---:|
| tiny | 13.8 μs | 12.0 μs | 100.0 μs | 77.9 μs | 45.5 μs |
| small | 18.3 μs | 19.6 μs | 112.7 μs | 78.5 μs | 49.2 μs |
| medium | 36.6 μs | 33.8 μs | 116.1 μs | 56.5 μs | 49.1 μs |
| square-1k | 151.8 μs | 146.5 μs | 181.6 μs | 92.7 μs | 95.8 μs |
| square-2k | 892.4 μs | 872.9 μs | 974.6 μs | 801.1 μs | 1027.3 μs |
| square-4k | 6050.5 μs | 6033.0 μs | 6529.0 μs | 6355.6 μs | 6144.1 μs |
| rect-wide | 429.4 μs | 426.8 μs | 507.8 μs | 407.1 μs | 418.7 μs |
| rect-tall | 421.3 μs | 426.1 μs | 496.0 μs | 409.2 μs | 425.5 μs |
| lm-head | 675.3 μs | 667.6 μs | 1167.1 μs | 747.8 μs | 912.7 μs |
| llama-q | 6057.6 μs | 6060.7 μs | 6509.7 μs | 6306.5 μs | 6123.2 μs |
| llama-ffn | 15787.6 μs | 15789.6 μs | 16706.1 μs | 16736.4 μs | 15929.1 μs |
| seq512 | 109.6 μs | 103.6 μs | 122.3 μs | 78.8 μs | 81.4 μs |

### softmax

| Shape | cuBLAS/cuDNN | PyTorch-eager | torch.compile | Arke | FlagGems |
|:---|---:|---:|---:|---:|---:|
| attn-small | 36.1 μs | 33.6 μs | 216.4 μs | 32.5 μs | 141.6 μs |
| attn-med | 32.4 μs | 33.0 μs | 227.7 μs | 31.9 μs | 50.2 μs |
| attn-large | 33.8 μs | 31.7 μs | 235.9 μs | 31.1 μs | 40.8 μs |
| square-4k | 215.7 μs | 216.1 μs | 516.2 μs | 214.7 μs | 481.6 μs |
| wide-vocab | 32.3 μs | 30.6 μs | 237.3 μs | 1015.4 μs | 77.0 μs |


### L1 Scoring (geometric mean ratios)

| Op | Metric | Ratio |
|:---|:-------|------:|
| matmul | Arke_vs_P0 | 🔴 0.788 |
| matmul | Arke_vs_P1 | 🟢 0.945 |
| matmul | FlagGems_vs_P0 | 🟡 0.834 |
| matmul | FlagGems_vs_P1 | 🟢 1.000 |
| matmul | PyTorch-eager_vs_P0 | 🟢 1.023 |
| matmul | PyTorch-eager_vs_P1 | 🟢 1.227 |
| matmul | cuBLAS/cuDNN_vs_P0 | 🟢 1.000 |
| matmul | cuBLAS/cuDNN_vs_P1 | 🟢 1.199 |
| matmul | torch.compile_vs_P0 | 🔴 0.586 |
| matmul | torch.compile_vs_P1 | 🔴 0.703 |
| softmax | Arke_vs_P0 | 🔴 0.523 |
| softmax | Arke_vs_P1 | 🟢 1.089 |
| softmax | FlagGems_vs_P0 | 🔴 0.480 |
| softmax | FlagGems_vs_P1 | 🟢 1.000 |
| softmax | PyTorch-eager_vs_P0 | 🟢 1.035 |
| softmax | PyTorch-eager_vs_P1 | 🟢 2.153 |
| softmax | cuBLAS/cuDNN_vs_P0 | 🟢 1.000 |
| softmax | cuBLAS/cuDNN_vs_P1 | 🟢 2.081 |
| softmax | torch.compile_vs_P0 | 🔴 0.181 |
| softmax | torch.compile_vs_P1 | 🔴 0.376 |

### L1 Baseline Sources

- **cuBLAS/cuDNN** (P0): NVIDIA cuBLAS/cuDNN via PyTorch 2.6.0+cu124 (CUDA 12.4) | https://pytorch.org | License: NVIDIA EULA (proprietary)
- **FlagGems** (P1): FlagGems 5.0.0 (BAAI/FlagOS) | https://github.com/flagos-ai/FlagGems | License: Apache-2.0
- **Liger-Kernel** (P1): Liger-Kernel 0.7.0 (LinkedIn) | https://github.com/linkedin/Liger-Kernel | License: BSD-2-Clause
- **PyTorch-eager** (P3): PyTorch 2.6.0+cu124 eager mode (default dispatch) | https://pytorch.org | License: BSD-3-Clause
- **torch.compile** (P4): torch.compile (Inductor) via PyTorch 2.6.0+cu124 | https://pytorch.org | License: BSD-3-Clause
- **Arke** (P5): Arke 0.1.0-dev (KernelCache Triton codegen) | https://github.com/arke-ai/arke | License: Apache-2.0

## L2: Fused Operator Results

### matmul_gelu

| Shape | separate | torch.compile | FlagGems |
|:---|---:|---:|---:|
| tiny | 123.6 μs | 244.6 μs | 200.3 μs |
| small | 116.1 μs | 153.6 μs | 112.0 μs |
| medium | 115.8 μs | 156.5 μs | 194.6 μs |
| square-1k | 119.9 μs | 275.9 μs | 112.2 μs |
| square-2k | 1000.3 μs | 1107.0 μs | 968.0 μs |
| square-4k | 6397.8 μs | 6867.1 μs | 6412.8 μs |
| rect-wide | 495.3 μs | 481.3 μs | 488.6 μs |
| rect-tall | 489.6 μs | 490.3 μs | 495.2 μs |
| lm-head | 1025.8 μs | 966.7 μs | 1035.0 μs |
| llama-q | 6364.7 μs | 6860.4 μs | 6371.1 μs |
| llama-ffn | 16632.6 μs | 16580.4 μs | 16587.2 μs |
| seq512 | 178.3 μs | 215.4 μs | 174.9 μs |

### matmul_relu

| Shape | separate | torch.compile | FlagGems |
|:---|---:|---:|---:|
| tiny | 35.7 μs | 99.4 μs | 119.5 μs |
| small | 25.9 μs | 118.0 μs | 126.1 μs |
| medium | 49.5 μs | 115.3 μs | 117.9 μs |
| square-1k | 213.5 μs | 249.2 μs | 115.8 μs |
| square-2k | 1000.1 μs | 1076.1 μs | 1051.9 μs |
| square-4k | 6342.9 μs | 6743.1 μs | 6345.8 μs |
| rect-wide | 478.8 μs | 565.9 μs | 475.1 μs |
| rect-tall | 475.4 μs | 541.9 μs | 477.8 μs |
| lm-head | 747.4 μs | 1245.2 μs | 947.6 μs |
| llama-q | 6297.0 μs | 6702.7 μs | 6325.6 μs |
| llama-ffn | 16348.0 μs | 17213.7 μs | 16566.4 μs |
| seq512 | 118.5 μs | 133.4 μs | 114.2 μs |


## L3: E2E Model Results (GPT-2 Small)

| SeqLen | Mode | Mean (ms) | Min (ms) | Memory (MB) | Correct | Top-1 Match |
|---:|:---|---:|---:|---:|:---:|:---:|
| 128 | eager | 7.41 | 6.28 | 285.3 | ✅ | ✅ |
| 128 | torch.compile | 5.73 | 5.17 | 298.6 | ✅ | ✅ |
| 128 | arke | 11.33 | 10.99 | 383.4 | ✅ | ✅ |
| 512 | eager | 11.66 | 11.28 | 617.8 | ✅ | ✅ |
| 512 | torch.compile | 10.13 | 9.70 | 1021.6 | ✅ | ✅ |
| 512 | arke | 19.24 | 18.64 | 548.3 | ✅ | ✅ |

### L3 Scoring

| SeqLen | eager (ms) | compile (ms) | arke (ms) | arke/eager | compile/eager |
|---:|---:|---:|---:|---:|---:|
| 128 | 7.41 | 5.73 | 11.33 | 1.53x | 0.77x |
| 512 | 11.66 | 10.13 | 19.24 | 1.65x | 0.87x |

## Summary

- **L1:** 2 operators, 5 baselines, 85 measurements
- **L2:** 2 fused ops, 72 measurements
- **L3:** GPT-2 Small, seq_lens=[128, 512], 6 measurements

---
*Report generated by Arke Benchmark System*
