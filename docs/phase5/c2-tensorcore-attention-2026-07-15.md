# C2 — Tensor-Core fused attention prototype (Phase 5, week 1)

**Date:** 2026-07-15
**Author:** optimization agent (探索性多周工程第一步)
**Goal:** 验证 Tensor-Core (wmma) fp16→fp32 fused attention 路线可行性，拿到第一个真实性能数据点。**不要求**达到 SDPA parity。
**Status:** ✅ 路线可行性已验证（correctness 达标）；性能瓶颈已精确定位（occupancy-limited）。尚未达到 parity（预期，属多周工程范畴）。

---

## 1. 背景 — 为什么要走 TC 路线

现有 `arke/backend/cuda_c_attention.py` 的 `flash_attention` emitter 是 **fp32 online-softmax warp-per-row** 方案。C2 历史诊断（`docs/phase4/audit-2026-07-13.md` C2 行）已经确认：

- 该 kernel 是 **memory-bound**（算术强度 AI≈2 FLOP/byte，Ampere ridge≈38），做 online-softmax **one-key-at-a-time**。
- torch SDPA (fp32) 走 cuDNN / vendor-BLAS，把 Q·Kᵀ 当**一次大 matmul** 打到高吞吐单元。
- 5.7× / 大 seq 0.15–0.18× 的差距是**设计空间限制**，不是调参能解决的：
  - 自适应 BR/BC → 慢 17%，回退。
  - split-K → 正确但慢 17%，移除。
  - BC sweep → 仅 S=512 一个点改善 8%。
- 结论：`BR=8/BC=32` 是**该设计空间的 measured-optimal**。要缩小差距必须换设计空间 → **Tensor-Core fp16→fp32 fused attention**，把 Q·Kᵀ 和 P·V 两个 matmul 交给 TC。

本轮就是这条路线的第一步。

## 2. 硬件与工具链

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop (Ampere, **sm_86**) |
| SM 数 | 30 |
| Smem/SM | 100 KB（sm_86 opt-in 上限） |
| nvcc | CUDA 13.2 |
| torch | 2.6.0+cu124 |
| TC 支持 | wmma 16×16×16 fp16→fp32 ✅ |

## 3. Prototype 设计

文件：
- `scratch/tc_attn/tc_attn.cu` — kernel
- `scratch/tc_attn/run_tc_attn.py` — torch cpp_extension 编译 + correctness + kernel-only CUDA-event 计时

设计要点（FlashAttention-2 风格，单 head、固定 tile）：

- 固定 `HEAD_DIM = 64`。一个 block 负责一个 `(batch·head, query-tile)`。
- **Query tile `BR = 64`**，**Key tile `BC = 64`**。Block = 4 warps (128 threads)，warp `w` 全程拥有 row-tile `w`（rows 16w..16w+16）：Q·Kᵀ、softmax、P·V 都是这个划分。
- **Q·Kᵀ**：wmma，`matrix_a=Q[16,16] row_major`，`matrix_b=Kᵀ`（K 转置存进 shared，让 b 成为普通 row_major 片段），沿 D=64 分 4 个 K-step 累加到 **fp32 accumulator**。
- 中间 **fp32 online-softmax**：每 warp 处理自己的 16 行，32 lane 用 `__shfl_xor` warp-reduce 求行 max / 分母，运行态 `m/l/O` 按 FA-2 online 规则 rescale。
- **P·V**：wmma，`matrix_a=P[16,64] fp16`，`matrix_b=V[64,64] row_major`，沿 BC=64 分 4 个 K-step 累加到 fp32，再加进 running `O`。
- Epilogue：`O = O / l`，写回 fp16。
- **精度**：两个 matmul 都是 fp16 输入 → fp32 累加；softmax 全程 fp32。

Smem 布局（dynamic shared，opt-in `cudaFuncAttributeMaxDynamicSharedMemorySize`）：
Q/Kᵀ/V/P 用 fp16，S/O/Otile/m/l 用 fp32，合计 **≈ 82 KB/block**。

## 4. Correctness（对比 torch SDPA，fp16）

目标：`max_err < 1e-2`。**全部达标，且富余约 40×。**

| B×H×S×D | max_err | mean_err |
|---|---|---|
| 1×1×128×64 | 2.44e-04 | 8.4e-06 |
| 1×4×128×64 | 2.44e-04 | 9.2e-06 |
| 1×8×512×64 | 2.44e-04 | 1.3e-05 |
| 1×8×1024×64 | 2.44e-04 | 1.0e-05 |
| 1×8×2048×64 | 1.22e-04 | 7.6e-06 |
| 4×8×2048×64 | 1.22e-04 | 7.6e-06 |

→ **TC fp16→fp32 fused attention 数值正确性验证通过。** fp32 累加把误差压在 fp16 舍入量级 (~2⁻¹²)。

## 5. 性能（kernel-only，CUDA events，50 iter / 10 warmup）

| B×H×S×D | TC proto (ms) | torch SDPA fp16 (ms) | speedup vs SDPA |
|---|---|---|---|
| 1×1×128×64 | 0.057 | 0.017 | 0.30× |
| 1×4×128×64 | 0.057 | 0.017 | 0.31× |
| 1×8×512×64 | 0.604 | 0.059 | 0.10× |
| 1×8×1024×64 | 1.865 | 0.119 | 0.06× |
| 1×8×2048×64 | 6.028 | 0.371 | 0.06× |
| 4×8×2048×64 | 22.49 | 1.488 | 0.07× |

**当前 prototype 尚未跑赢 SDPA fp16。** 这符合"第一步不要求 parity"的预期——SDPA fp16 走的是 cuDNN/FlashAttention 高度优化 kernel，是极强的对比基线。

### 迭代过程（真实记录）
1. BR=16 + 单线程/行 serial softmax → S=512 0.074×，随 S 恶化。
2. BR=16 + warp-并行 softmax（`__shfl_xor` reduce）→ S=512 0.074×，S=1024 0.073×（小 S 有改善）。
3. **BR=64**（当前）+ warp-并行 softmax → K/V tile 复用从 16 行摊到 64 行，S=2048 从 0.044× → 0.061×。

## 6. 根因诊断 — 为什么还慢

`nvcc -Xptxas -v`：**55 registers，0 spill**，但 **dynamic smem ≈ 82 KB/block**。

在 sm_86 上 smem/SM = 100 KB：
- 82 KB/block → **每 SM 只能驻留 1 个 block** = 4 warps = **12.5% occupancy**（1536 threads/SM 上限只用了 128）。
- Occupancy 被 smem 卡死 → SM 上几乎没有 warp 可以隐藏 wmma / shared-memory / `expf` 的延迟。这是当前性能的**首要瓶颈**。

次要因素：
- softmax 阶段有两次 `__syncthreads` 把 TC pipeline 打断，且 softmax 是标量 fp32（非 TC）。
- 每个 K-tile 做一次 shared→shared 的 O accumulate（Otile→Osh），多一趟 smem 往返。
- Kᵀ 转置存 shared 有 bank conflict 风险（未 profile 确认，`ncu` 无 GPU counter 权限）。

## 7. 障碍与限制（如实报告）

1. **`ncu` 无权限**：`ERR_NVGPUCTRPERM`（GPU performance counters 需要 root / 驱动 flag）。无法拿到 achieved occupancy / stall reason 的直接测量，第 6 节的 occupancy 数字是**从 smem 占用 + 硬件规格推算**，非 profiler 实测。
2. **transfer 噪声**：现有 fp32 warp-per-row baseline 只能通过 arke backend 的 `run()` 测，包含 H2D/D2H，不能与 TC 的 kernel-only 数直接比。故本报告对比基线统一用 **torch SDPA fp16 kernel-only**（同一 CUDA-event 方法）。
3. **固定 D=64、single-config**：prototype 只支持 D=64、BR=BC=64、无 causal mask、无 GQA。这是**有意的最小可跑范围**。

## 8. 结论

- ✅ **TC 路线可行性验证通过**：wmma fp16→fp32 fused attention 能跑通、数值正确（max_err 2.4e-4 ≪ 1e-2）。
- ✅ **拿到第一个性能数据点**并精确定位瓶颈：**smem-limited occupancy（1 block/SM, 12.5%）**，不是 register spill，不是算法错误。
- ⏳ 未达 parity —— 属预期，是多周工程后续步骤。

## 9. 下一步（后续周）

按杠杆排序：
1. **降 smem 提 occupancy**（最高杠杆）：去掉 `Otile` 独立缓冲（累加进 fragment 或复用 `Ssh` 区）；`Ssh` fp32 可在 softmax 后立即复用为其他用途；目标把 smem 压到 ≤ 48 KB → 2 blocks/SM，占用翻倍。
2. **double-buffer K/V tile**（`cp.async` / `__pipeline`）重叠 load 与 TC 计算。
3. **减少 `__syncthreads`**：把 softmax 融进 fragment 级操作，避免 S 经 shared 往返。
4. **register-resident O accumulator**：让每 warp 的 O tile 留在 fragment / 寄存器，省掉 Otile→Osh 的 smem 往返。
5. 拿到 GPU counter 权限后用 `ncu` 实测 occupancy / stall breakdown 验证上述假设。
6. 扩展 config：任意 D（64/128）、causal mask、GQA，最后接回 arke emitter (`emit_cuda_c_flash_attention` 的 TC 变体)。

## 10. 复现

```bash
source /home/blueyi/.venvs/arke/bin/activate
export PATH=/usr/local/cuda-13.2/bin:$PATH
python scratch/tc_attn/run_tc_attn.py   # 打印 correctness + perf，写 results.json
```
