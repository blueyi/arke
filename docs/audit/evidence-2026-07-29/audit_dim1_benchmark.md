# Arke 维度一审计报告：Benchmark 覆盖度 · 性能基线对比 · 调优收益量化

- **审计对象**：`/home/blueyi/workspace/repos/arke`（AI 编译器工具链，Triton/MLIR/CUDA-C/LLVM 多后端）
- **硬件**：NVIDIA RTX 3060 Laptop GPU（sm_86），6143 MB，CUDA 12.4，PyTorch 2.6.0，Triton 3.2.0，Python 3.10.20，WSL2
- **审计日期**：数据快照最新为 2026-07-29
- **数据纪律**：本报告所有数字均直接读自仓库文件/命令输出；读不到的标注"无数据"。

---

## 一、Benchmark 覆盖度

### 1.1 算子总数与分层（实测命令）

`python -c "from benchmarks.op_registry import ALL_OPS,OT_OPS; ..."` 输出：

```
TOTAL 46
OT0 12    OT1 10    OT2 11    OT3 8    OT4 5
```

与 SSOT `docs/benchmark/benchmark-ops.md` OT Summary 表完全一致（该文件是唯一真源，`op_registry.py` 运行时解析它，`tests/test_ssot_op_registry.py` 防漂移）。

| 层 | 名称 | 数量 | 类别属性 | 代表算子 |
|:--|:--|:--:|:--|:--|
| **OT0** | Elementwise | 12 | 访存密集（memory-bound） | relu, gelu, silu, tanh, sigmoid, add, mul, where_, cast, neg, exp, rsqrt |
| **OT1** | Reduction | 10 | 访存密集（warp reduce + shared mem） | softmax, layernorm, rmsnorm, rmsnorm_residual, reduce_sum/max/mean, argmax, topk, cumsum |
| **OT2** | Data Movement & Dense | 11 | **计算密集(matmul类)** + 访存密集(movement类) 混合 | matmul, batch_matmul, grouped_matmul, transpose, concat, split, gather, scatter, embedding, permute, copy_ |
| **OT3** | Fused Compound | 8 | 融合，输出 shape ≠ 输入 | silu_and_mul, gelu_and_mul, swiglu_packed, rope, fused_linear_cross_entropy, cross_entropy, quantize_per_token, dequantize_per_channel |
| **OT4** | Attention | 5 | **计算密集**（多阶段融合 + online softmax） | flash_attention, grouped_query_attention, multi_latent_attention, cross_attention, paged_attention |

**计算密集 vs 访存密集分布**：
- 计算密集（GEMM/attention 类）：OT2 的 matmul/batch_matmul/grouped_matmul（3 个）+ OT4 全部 5 个 = **8 个**。
- 访存密集（elementwise/reduce/data-movement）：OT0 全部 12 + OT1 全部 10 + OT2 的 8 个 movement 类 = **30 个**。
- 融合类（OT3）8 个介于两者之间。
- 结论：覆盖以**访存密集为主体**（~65%），计算密集算子偏少但覆盖了 GEMM 和 5 类 attention 变体，作为 LLM 推理/训练算子集是合理的。

### 1.2 显式排除项（docs 有记录，非遗漏）

`benchmark-ops.md` §"What was intentionally NOT added" 明确排除：稀疏（SpMM/block-sparse，留 OT5 Phase3）、卷积（conv1d/conv2d，留 VLM Phase3）、通信原语（all_reduce 等，归 NCCL）、DMA/prefetch（硬件级）。**即：无 conv 覆盖是有意决策，不是审计缺口。**

### 1.3 Shape tier 覆盖（`benchmarks/shapes.py`）

三档 tier：Tier1 冒烟（秒级）、Tier2 代表性负载（分钟级）、Tier3 压力/非对齐/极端。极端边界覆盖充分：

- **超长 seq / 超宽**：softmax `extreme-wide` (1×1048576)、`extreme-tall` (65536×64)；elementwise `extreme-flat` (1×1048576)；flash_attention 至 `ds-v2-2k`(S=2048)、`ds-v2-4k`，PERF_ALL 中甚至枚举了 `ds-v2-8k/16k`、`ds-v3-32k/163k`（后者因 6GB 显存被 memory-preflight skip，见 §2）。
- **tiny / 单行**：matmul `tiny`(128³)、`extreme-1row`(1×1024×1024 GEMV)、`extreme-16`；softmax `extreme-tiny`(1×16)；norm `extreme-small`(1×768)。
- **非对齐 / off-by-one**：每个算子族都有 `non-align-*`（如 matmul 127×513×1000、1023×1025×1024、2049×2047×2050；norm 333×4097；softmax 13×513）。
- **GQA/MLA 专属**：GQA 覆盖 llama3 (32/8=4:1)、qwen2.5 (28/4=7:1)；MLA 覆盖 DeepSeek-V2 (D_c=512)、V3 (D_c=1024)。
- 另有 `benchmarks/shape_registry.py`（解析 `benchmark-shapes.md`）作为 ST4 生产 shape 的 SSOT override 层。

**覆盖度结论**：算子分层清晰、SSOT 有强约束、shape tier 含 tiny/超长/非对齐三类极端边界，覆盖度良好。唯一"缺口"（conv/稀疏）为文档记录的有意排除。

---

## 二、性能基线对比

### 2.1 标杆选择（`docs/benchmark/golden-kernel-ladder.md` + `golden_ladder.py`）

Golden Kernel 兼任正确性 oracle 与性能分母，按优先级 P0→P5 选第一个 `supports & available` 的 runner：

| P | Runner | 用于 |
|:--|:--|:--|
| P0 | cuBLAS/cuDNN | matmul, softmax, layernorm, batch_matmul, 多数 elementwise |
| P1 | FlagGems / Liger | rmsnorm(Liger), 融合类(Liger), gather/scatter/topk等(FlagGems) |
| P2 | flash-attn 2.7.4 | **flash_attention, GQA**（2026-07-27 OT4 复审锁定） |
| P3 | PyTorch-eager | rope, MLA, paged_attn, split, permute 等无生产 kernel 的 op |

**LADDER_PREFERENCES 锁定项**（偏离严格 P 序，需 Leon 签字）：rope→eager（Liger 在 odd-D 报错）；flash_attention/GQA→flash-attn（FlagGems 5.0.0 的 SDPA 是 bmm 分解式，非融合，tier-2 shape OOM 32GiB）。`audit-degraded`（无生产 kernel，只 eager 兜底）：MLA、paged_attention、swiglu_packed、dequantize_per_channel。

### 2.2 最新真实快照与"分母"陷阱（**关键，务必注意**）

`benchmarks/results/` 下非 deprecated/非 _archive 目录，按 mtime：`dynamic_shape/`(7-29) > `convergence/` > `phase5/` > `optimize/` > `phase4/live/`。全量 PERF_ALL 快照在 `phase1/stage6/trackg6/l1/`（46 算子×多 shape，1399 shape，summary 有 op geomean）。`L2/` 为 2026-04-09 旧数据（仅 fused ops，已过时）。

**⚠️ 分母陷阱**：`PERF_ALL.csv` 与 `summary.json` 中 **flash_attention 的 flash-attn(P2) golden 因 `IndexError: Dimension out of range` 未生效**（每行 reason 都记录了该报错），实际分母回退成 PyTorch-eager。因此该快照里 flash_attention 的 3.06× geomean、cross_attention 1.55×、GQA 2.31× 等**是 vs eager，不是 vs 业界标杆**——eager attention 是 O(S²) 未融合，比它快 3× 毫无含金量。

**对业界标杆(flash-attn)的真实 attention 比值只存在于 `docs/kestrel/k-att-plan.md §6`（同日 A/B，见 §四）**：FA **0.846**、GQA **0.802**、cross_attn **1.081**。这才是可信数据。

### 2.3 全量快照 op geomean（`phase1/stage6/trackg6/l1/summary.json`，overall geomean 1.1112，vs 各自 golden）

真正对**强标杆**（cuBLAS/Liger/FlagGems）且 <1 的算子（性能短板）：

| 算子 | geomean(ratio) | golden | 判读 |
|:--|:--:|:--|:--|
| **split** | **0.207** | eager(P3, view op) | 见下"回退分析"，属基准不公 |
| relu | 0.547 | cuBLAS | 访存密集小 kernel，vs cuBLAS 逊 |
| layernorm | 0.582 | cuBLAS/cuDNN | OT1 短板 |
| softmax | 0.579 | cuBLAS/cuDNN | OT1 短板 |
| silu | 0.723 | cuBLAS | OT0 短板 |
| gelu | 0.757 | cuBLAS | OT0 短板 |
| matmul | 0.808 | **cuBLAS(P0)** | GEMM vs cuBLAS 约 0.8×，核心计算密集短板 |

明显领先标杆的：rope 4.56×(vs eager)、rmsnorm_residual 3.49×(Liger)、rmsnorm 2.52×(Liger)、topk 2.45×、quantize_per_token 2.71×、embedding/gather(vs eager)。

`no_data_ops`（4 个无数据）：**multi_latent_attention、paged_attention、permute、swiglu_packed** —— 与 golden-ladder 里 audit-degraded 名单吻合，属已知空白。

### 2.4 性能回退（ratio < 0.3 的 op）——真实发现

从 `PERF_ALL.csv` 单行级：

| op@shape | Arke ratio | golden | 性质 |
|:--|:--:|:--|:--|
| **split@(整体)** | **0.207** | eager view | split 在 eager 是 O(1) 零拷贝 view，Arke 物化输出，**基准本身不公平**（同 transpose 的 lazy-view 陷阱，见 harness-perf-shape-encoding-bug.md）。非真回退。 |
| copy_@xlarge | 0.631 | eager | 大 buffer 拷贝逊于 eager；copy_@llama-long 0.659。真实弱项但非 <0.3。 |
| argmax@wide | 0.530 | eager | 单行超宽(1×50257)归约弱。 |

`dynamic_shape/2026-07-29` 快照测的是**动态 shape 首调用 vs 稳态"cliff ratio"**（编译/config 选择开销），非 vs 标杆比值：matmul cliff geomean 3.31×（首调最坏 27.9× at m16），softmax cliff geomean **40.99×**（最坏 130.7×），rmsnorm 7.22×。即：**动态 shape 首次触发时 softmax 会慢 ~41 倍（config 重选 + 编译）——这是 attention/softmax 在动态序列长下的已知性能悬崖**（K-DYN track）。

**回退结论**：无算子对强标杆出现 <0.3 的真实性能回退。<0.3 的两处（split 0.207）均为 view-op 基准不公平，文档已识别。真实弱项集中在：① matmul vs cuBLAS ~0.8×；② OT0/OT1 小 elementwise/reduce vs cuBLAS 0.5–0.76×；③ 动态 shape 首调用悬崖（softmax 最严重）。

---

## 三、调优收益量化（auto-tuning / live-agent）

### 3.1 收敛曲线（`benchmarks/results/convergence/`，KESTREL K-H5.2，2026-07-28）

回答"Arke 优化循环收敛效率"，每行 = 一次 `compile_and_profile`。实测轨迹：

| op | shape | 迭代数 | 正确 | best_so_far 演进 | 收益 |
|:--|:--|:--:|:--:|:--|:--|
| **matmul** | 512×512×512 | 2 | 2/2 | 0.3805 → **1.2786** | LLM 2 步找到超默认 config，**≈3.4× 提升**（vs cuBLAS 从 0.38× 拉到 1.28×） |
| **softmax** | 512×1024 | 4 | 4/4 | **1.0 (平)** | LLM 试 3 个替代，无一胜过默认（含一次 0.29× 回退）；收敛=确认默认最优 |
| **flash_attention** | 2×4×512×64 | 3 | **1/3** | (无)→1.0 | **2/3 正确性失败**——诚实暴露 attention 弱点，正是 K-ATT 立项动机 |

轨迹细节（matmul_512 trajectory.json）：LLM 用 sonnet-4-6，20 tool calls / 66.5s，从 `list_legal_actions` 探索 tile/unroll/parallel 动作空间，iter1 默认 template 0.38×→ 决策后 iter2 达 1.28×。**搜索效率：matmul 2 迭代收敛，属高效。**

### 3.2 Live-agent Gate 收益（`phase5/s5/gate_p5s5t.json`，P5-S5-T live gate 5/5 PASS）

23 个 gate case 加权 geomean **0.947**（阈值 ≤0.948，pass）。逐条 auto-tuning 前后（agent vs default，pair_ratio <1 = agent 更快）：

| case | 决策数 | default(μs) | agent(μs) | agent/default | 收益 |
|:--|:--:|:--:|:--:|:--:|:--|
| matmul@1024³ | 2 | 348.10 | 340.72 | 0.9788 | -2.1% |
| matmul@2048³ | 2 | 2036.67 | 1985.88 | 0.9802 | -2.0% |
| rmsnorm@1024×4096 | 1 | 118.55 | 114.55 | 0.9663 | -3.4% |
| 其余 5 例 | 0 | — | — | null | 决策空(agent=default)，确认默认已最优 |

**held-out 泛化 (C5)**：rmsnorm@256×4096 用 1024-threads 决策(带 @rationale)迁移，agent/default 0.9817；其余 held-out 决策空但不劣于默认。C4 审计：所有决策均带 `@rationale`，0 缺失。

**收益判读**：
- Live-agent 单点收益偏小（-2% ~ -3.4%），因为 default heuristic 已相当强；agent 的价值主要是**在 default 输的少数 case 上恢复到 ≤1.05×**（C2：softmax@1024×4096 从 sweep 1.0493 恢复到 1.0491）。
- 大收益体现在**动态 shape config 记忆**（K-H3.1 bucketed launch-config，避免每次重选 config）和**离线蒸馏的 strategy rule**（matmul_rule/softmax_rule/rmsnorm_rule/layernorm_rule.json）。
- RL 语料：`phase4/live/rl_corpus.jsonl`、`phase5` rationale KB **390 条**（gate_p5_final.json A_rationale_kb，阈值 200，pass，含 live_llvm_p5 72 条）。

### 3.3 搜索空间收敛效率

- 动作空间是 **Bounded Action Space**（`list_legal_actions` 返回，如 matmul 的 tile 18 候选 / block_threads / wmma_tile）。matmul 2 迭代、rmsnorm 1 决策即收敛，说明启发式先验强、搜索浅。
- 诚实短板：flash_attention 收敛轨迹 2/3 正确性失败（K-ATT 立项前），softmax 4 迭代 bounce 无增益——**说明对 attention/reduce 类，Agent 的探索价值当时未兑现**，直到 K-ATT 手动优化模板（§四）才把 FA 拉起来。

---

## 四、Attention 最新数据（`docs/kestrel/k-att-plan.md §6`，2026-07-29 同日执行且双 gate PASS）

**唯一对 flash-attn 强标杆的可信 attention 比值**（同硬件同日 A/B，flash-attn 2.7.4.post1 golden，fp16，kernel-only CUDA-events median）：

| op | baseline(7-27) | 最终 geomean | 关键杠杆 |
|:--|:--:|:--:|:--|
| flash_attention | 0.301× | **0.846×** | FA-v2 TC dtype 纪律（去掉 fp32 load，`tl.dot(fp16,fp16,out_dtype=fp32)`）是决定性杠杆，从 0.496→0.846；stage 0.35 ✅ + final 0.50 ✅ |
| grouped_query_attention | 0.172× | **0.802×** | GQA-v1 修 launcher `gqa_groups=H//Hkv` bug（此前 0.172 部分是错误 kernel）；继承 FA 的 pipeline+TC，~4× 真实提升 |
| cross_attention | ~0.5–0.6× | **1.081×** | FA-v2 副作用，非因果 Sq≠Skv 上**反超**融合 flash-attn（llava 1.25/batch4 1.08/sdxl 1.03/t5 0.98） |

**诚实 caveat（doc 自记录）**：① GQA 0.172 baseline 部分测的是有 bug 的错误 kernel；② 全部同日 A/B（laptop 时钟跨日漂 2–4×，禁比历史 PERF_ALL）；③ ds-v3-163k 类超长因 6GB 显存未测；④ 记录了一次"测量诚实事件"——首次 sweep 因默认 config 吸收 GPU spin-up 伪造出 6.4× 假加速，已用 per-shape warmup 修复，假数据从未进入 heuristic 或 commit。

---

## 五、关键结论汇总

1. **覆盖度扎实**：46 算子 / 5 层（OT0-12, OT1-10, OT2-11, OT3-8, OT4-5），SSOT 强约束防漂移；shape 含 tiny/超长(1M宽)/非对齐三类极端边界；conv/稀疏为**文档记录的有意排除**，非缺口。4 个 no-data 算子（MLA、paged_attention、permute、swiglu_packed）与 audit-degraded 名单吻合。
2. **性能基线有一个必须警惕的分母陷阱**：全量 PERF_ALL 快照里 flash_attention 的 flash-attn golden 因 IndexError 失效、回退成 eager，导致 3.06×/GQA 2.31× 等 attention 数字是 **vs eager 而非 vs 标杆**，不可信。对标杆的真实 attention 比值只在 k-att-plan §6。
3. **无对强标杆的真回退(<0.3)**：唯一 <0.3 的 split(0.207) 是 view-op 基准不公平（文档已识别）。真实弱项：matmul vs cuBLAS ~0.8×；OT0/OT1 小 kernel vs cuBLAS 0.5–0.76×（relu 0.55/softmax 0.58/layernorm 0.58/silu 0.72/gelu 0.76）；动态 shape 首调悬崖（softmax cliff geomean **41×**，最坏 131×）。
4. **调优收益真实但分化**：matmul auto-tuning 2 迭代 0.38→1.28（≈3.4×，vs cuBLAS）为最亮点；live-agent 单点收益仅 -2%~-3.4%（default heuristic 已强），价值在恢复 default 输的 case ≤1.05× + config 记忆 + rationale KB 390 条。收敛效率高（matmul 2 步、rmsnorm 1 决策）但 attention/softmax 探索当时未兑现（FA 收敛 2/3 正确性失败）。
5. **Attention 是最大且已被针对性攻克的短板**：2026-07-29 K-ATT 同日把 FA 0.301→**0.846**、GQA 0.172→**0.802**、cross_attn→**1.081**（反超），双 gate PASS。核心杠杆是 Tensor Core dtype 纪律。仍存 6GB 显存下超长 seq(163k) 未测的覆盖空白。

---
*报告基于仓库真实文件/命令输出生成；未测/无数据项已如实标注。*
