# Arke Benchmark 深度解析与性能闭环审计

**审计范围**：只读审计（未运行任何 GPU benchmark，GPU 被生产 bench 占用）。
**主数据源**：`benchmarks/results/phase1/stage6/trackg6/l1/PERF_ALL.csv`（1629 行，46 op，9 baseline）+ `stage7/track6/l2/PERF_ALL.csv`（511 行 L2 fusion）。
**审计日期**：2026-07-27 · RTX 3060 Laptop 6GB · CUDA 12.4 · PyTorch 2.6.0 · Triton 3.2.0

> ⚠️ **口径提醒**：任务背景给出的 "overall_geomean 0.9425" 与本 CSV 内 `summary.json` 记录的 `overall_geomean = 1.1188` 不同。后者含 torch.compile / PyTorch-eager 等弱基线拉高。本报告一律标注"对哪个 baseline"，避免单一数字误导。

---

## 第一部分：基准覆盖度

### 1.1 SSOT 结构（46 op，来源 `docs/benchmark/benchmark-ops.md` L14–L21）

| OT | 名称 | Count | 复杂度驱动 | 证据行 |
|---|---|---|---|---|
| OT0 | Elementwise | 12 | 访存密集 1:1 | benchmark-ops.md L14 |
| OT1 | Reduction | 10 | warp-reduce + shared mem | L15 |
| OT2 | Data Movement & Dense | 11 | tensor-core tiling / layout | L16 |
| OT3 | Fused Compound | 8 | 多算子融合 / 输出形变 | L17 |
| OT4 | Attention | 5 | 多阶段融合 / online-softmax / KV cache | L18 |

`op_registry.py` 是 SSOT 解析器（L57–L109 正则解析 markdown 表），运行时 `total_ops()` 动态返回，禁止硬编码计数（tests/test_ssot_op_registry.py 强制约束）。`shapes.py:OP_TIER` 内建静态表 L306–L326 与 SSOT 一致（12/10/11/8/5=46）。

### 1.2 计算密集 vs 访存密集覆盖

**计算密集（GEMM 类）覆盖良好**：
- `matmul`（22 shape，MATMUL_SHAPES shapes.py L43–L69）、`batch_matmul`（9 shape L208–L221）、`grouped_matmul`（4 shape，MoE L225–L233）。
- matmul shape 分层完整：tiny→square-4k（4096³）→ llama-ffn（4096×11008×4096），含 10 个 tier3 非对齐 shape（非 2 的幂、off-by-one、单行 GEMV `extreme-1row` M=1）。这是覆盖度最扎实的一块。

**conv2d：明确不实现**（benchmark-ops.md L50「What was intentionally NOT added」）。理由记录为 "decoder-only LLM 极少用 conv，视觉编码器留 Phase 3 VLM"。SSOT 里根本没有 conv1d/conv2d，PERF_ALL 也无相关行 → **不是"实现了但未测"，而是有意识地不纳入**。审计结论：对 LLM kernel 场景合理，但"通用高性能 kernel"叙事存在缺口。

**访存密集（reduce/softmax）覆盖**：
- softmax 25 shape（SOFTMAX_SHAPES L73–L102），含 `extreme-wide`（1×1048576）、`extreme-tall`（65536×64）、`wide-llama`（1×128256 词表）。
- 但 **PERF_ALL 实测 softmax 只有 10 个 shape**（`attn-gpt2-*`, `attn-llama-*`, `wide-vocab-*`, `square-4k`, `batch-large`）——**shapes.py 定义的 extreme shape 未全部进入实测 CSV**。
- reduce 族（reduce_sum/max/mean）实测仅 6 shape：`small/medium/large/wide/llama3-wide/qwen25-wide`。REDUCE_SHAPES 定义里的 `extreme-tall`(65536×64) 未出现在 PERF_ALL 中。

### 1.3 Shape tier 体系与极端边界

tier 语义（shapes.py L516–L527）：tier1=ST1 smoke，tier2=+ST2，tier3=+ST3+**ST4**（BL5 full）。

**极端边界覆盖情况（关键）**：
- **N>65536**：shapes.py 定义中存在（softmax `extreme-wide` 1M、elementwise `extreme-flat` 1M）。**这正是背景事实(3)所指 reduce 族 N>65536 静默截断 bug（commit c2df77b 修复）的高危区**——修复前 reduce 在大 N 会静默截断结果。但当前 PERF_ALL 的 reduce 实测 shape 最大 N 仅到 `wide`（50257 级）与 `qwen25-wide`，**未见 N>65536 的 reduce 实测行** → 修复后的回归覆盖存在盲区。
- **seq 163k**：`ds-v3-163k` shape 存在于 flash_attention（PERF_ALL 确认），但 **4 个 baseline 全部 `status=skipped, latency=inf`**（6GB 显存放不下）→ 定义了但无法实测，纯占位。
- **batch 边界**：paged_attention 有 `vllm-batch16 / vllm-batch32-ctx4k`（后者 memory-preflight skipped，需 4.29 GB）；attention 有 `llama2-7b-batch`（B=4）、`ds-batch-8`（B=8）。batch 边界有覆盖但受 6GB 显存钳制。

### 1.4 缺什么（dtype / 动态 shape / 实现缺失）

- **dtype 覆盖薄弱**：`cast` 算子文档声称覆盖 fp32↔fp16↔bf16↔int8/int4（benchmark-ops.md L180），但 PERF_ALL 中 cast 只有 elementwise shape，**无按 dtype-pair 拆分的测试行**。量化算子 `quantize_per_token` / `dequantize_per_channel` 存在（OT3），但**没有独立的 int8/fp8 端到端 dtype 覆盖矩阵**。fp8 完全无覆盖。
- **动态 shape**：`gate_g7.py:_check_spec_docs` L106 要求 `dynamic-shape-feasibility.md` 存在——说明动态 shape 目前是**可行性文档阶段，非实测覆盖**。所有 shape 都是静态枚举。
- **算子"声称覆盖但 Arke 未实现"**（重大发现，见 §2.5）：`swiglu_packed`、`permute`、`multi_latent_attention`、`paged_attention` 的 Arke 行全部是 `Arke.get_fn declined`（unsupported），但 summary.json 给它们打了 **score=1.0**。

---

## 第二部分：性能基线

### 2.1 status 分布（stage6/trackg6/l1/PERF_ALL.csv，1629 行）

| status | 计数 | 占比 |
|---|---|---|
| ok | 1414 | 86.8% |
| unsupported | 177 | 10.9% |
| skipped | 30 | 1.8% |
| timeout | 8 | 0.5% |

correctness 分布：ok 1357 / unsupported 201 / skipped 30 / golden_unavailable_pending_baseline 29 / timeout 8 / **error 2 / mismatch 2**。

### 2.2 各 baseline 的 status 与 unsupported 率

| baseline | total | ok | unsupported | skipped | timeout | **unsup%** |
|---|---|---|---|---|---|---|
| PyTorch-eager | 415 | 395 | 0 | 17 | 3 | 0.0% |
| Arke (SUT) | 415 | 369 | 40 | 3 | 3 | 9.6% |
| cuBLAS/cuDNN | 197 | 179 | 18 | 0 | 0 | 9.1% |
| Triton-Tutorial | 22 | 20 | 2 | 0 | 0 | 9.1% |
| flash-attn | 26 | 22 | 0 | 2 | 2 | 0.0% |
| FlagGems | 321 | 257 | 56 | 8 | 0 | **17.4%** |
| torch.compile | 85 | 67 | 18 | 0 | 0 | **21.2%** |
| **Liger-Kernel** | 72 | 29 | 43 | 0 | 0 | **59.7%** ⚠️ |
| torch.compile-max-autotune | 76 | 76 | 0 | 0 | 0 | 0.0% |

**unsupported 率最高的 baseline：Liger-Kernel 59.7%**（覆盖面极窄，只支持 rmsnorm/rope/gelu/silu/fused_ce 等少数算子；`fused_linear_cross_entropy` 在 gpt2/llama 也 declined 5 次）。**FlagGems 17.4%**（argmax/concat/copy_ 等多个算子 get_fn declined）。这两者作为"Triton 同后端 golden"时覆盖不全，是 G7.8d 分母缺失��直接原因。

### 2.3 per-baseline overall geomean（baseline_lat / Arke_lat；>1 = Arke 更快）

| baseline | n | geomean | 解读 |
|---|---|---|---|
| flash-attn | 21 | **0.2058** | ⚠️ Arke 比 flash-attn 慢 ~4.9× |
| Triton-Tutorial | 20 | 1.2975 | Arke 略快 |
| cuBLAS/cuDNN | 179 | 1.3813 | Arke 快（含小 shape wrapper 优势） |
| FlagGems | 257 | 1.7121 | Arke 快 |
| Liger-Kernel | 29 | 1.9054 | |
| PyTorch-eager | 364 | 1.9739 | Arke 明显快（弱基线）|
| torch.compile | 67 | 3.9306 | 弱基线 |
| torch.compile-max-autotune | 76 | 4.1068 | 最弱基线，严重拉高整体 |

**核心结论**：整体 geomean 被 torch.compile（3.9×）/ eager（2.0×）等弱基线严重拉高。**唯一比 Arke 快的强基线是 flash-attn（geomean 0.206 = flash-attn 比 Arke L1 attention 快约 4.9×）**，与背景事实(2)（FA 0.238 / GQA 0.169）方向一致。这是当前最大的真实性能缺口。

### 2.4 所有 Arke < 1.0 的 op（对同后端 Triton 竞品，baseline/Arke<1 = Arke 慢）

**A. Attention 族——最严重（对 flash-attn）**：
- `flash_attention`：12 个 shape 全部落后，ratio 0.106（ds-batch-8）～0.473（gpt2-sm-512）。llama2-7b-4k：Arke 45519μs vs FA 6817μs。
- `grouped_query_attention`：9 个 shape 全部落后，ratio 0.154～0.203。llama3-8b-8k：Arke 171431μs vs FA 26514μs。
- 差距根因：flash-attn 是唯一真·fused O(S) 内存 kernel；Arke L1 attention 仍走 materialized score buffer 路径。

**B. `split`——ratio 极差（背景事实(4)确认）**：
Arke 行 ratio_vs_baseline 仅 0.013～0.078（对 PyTorch-eager）。PyTorch-eager 3.2μs（zero-copy view）vs Arke `ffn-gate-up` 293μs、`llama-qkv-split` 109μs。summary.json split score **0.2072（全 46 op 最低）**。根因：eager split 是零拷贝视图，Arke 强制物化。

**C. 中等落后（对 FlagGems，小到中 shape wrapper 开销为主）**：
- `reduce_mean`：llama3-wide 0.570、medium 0.736（geomean 对 FlagGems 仅 0.977，唯一 <1 的 reduce）。
- `silu`：gpt2-ffn 0.872、gpt2-hidden 0.881、micro-tiny 0.900（小 shape ~5μs wrapper 开销）。
- `gelu`：qwen-ffn 0.886、ds-ffn-large 0.904、llama-ffn 0.914。
- `layernorm`：mixtral-ffn 0.663（异常点）；`relu` geomean 对 FlagGems 0.717/0.749（但对 Liger 1.477）。
- `matmul`：seq512 对 Triton-Tutorial 0.765 / 对 FlagGems 0.837；大 shape 基本打平（square-4k 0.999、llama-q 0.995）。
- `embedding`：对 FlagGems geomean 1.055，但 7 个 shape 微落后（batched-b4 0.860）。
- 零星：sigmoid（square-1k 0.916）、tanh（square-1k 0.945）、cumsum（sampling-probs 0.865）、rmsnorm（llama-7b 对 Liger 0.937）。

### 2.5 ⚠️ 重大发现：summary.json score 与真实实现状态不一致

以下 4 个算子 summary.json **score=1.0**，但 Arke 全部 shape `status=unsupported`（`Arke.get_fn declined`）：
- `swiglu_packed`（12 shape 全 declined）
- `permute`（9 shape 全 declined）
- `multi_latent_attention`（8 shape declined + golden_unavailable 10）
- `paged_attention`（11 declined + golden_unavailable 19 + 1 memory skip）

即 **Arke 对这 4 个算子根本没有可运行实现，score 1.0 是"无数据默认满分"占位**，不是真实性能。这会虚高覆盖叙事，是审计红线级隐患。

### 2.6 correctness 异常（2 mismatch + 2 error）

- `grouped_query_attention` @ llama3-8b-8k / qwen25-7b-8k：**Arke correctness=mismatch, max_abs_diff=0** —— diff=0 却判 mismatch，疑似 8k 长序列比对逻辑/NaN 问题（值得单独复查）。
- `flash_attention` @ ds-v2-8k（flash-attn baseline）与 `softmax` @ wide-vocab-llama3（Triton-Tutorial）：error 均为资源类（OOM 32GiB / shared-mem 超限），属 baseline 侧非 Arke 回归。

---

## 第三部分：调优收益（L1→L2→L3）

### 3.1 层定义（`docs/roadmap/plan.md`）

- **Thesis L1**（plan.md L15）单架构单后端（Phase1，Triton）：已 validated，Phase1 CLOSED 2026-06-25。
- **Thesis L2**（L24）跨架构同抽象层（Phase2 Ascend）：**dormant / 未验证**——无 Ascend 硬件，PAUSED（L31）。⚠️ 注意与"benchmark L2 fusion"是两个不同 L2 概念。
- **Thesis L3**（L33）跨抽象层（Phase3-5）：NVIDIA 单硬件 VALIDATED（L40）。
- **注意**：`arke/engine/` 目录不存在；优化引擎在 `arke/agent/`（ArkeEnv/tools/prompts）。任务描述的路径有误。

### 3.2 L1（heuristic）收益证据

L1 对弱基线有系统性收益（§2.3），但对**同后端 Triton 竞品的严格 gate（G7.8d，ε=0.03）**：
- weighted_score = **0.3006**（阈值 0.95），stage7-completion-summary.md L266。
- OT0_1 = 220/351 (0.627)；OT2 = 19/71 (0.268)；OT3 = 7/22 (0.318)；OT4 = 0/0（当时无 Triton attention 分母，L270）。
- G7 = **13/14 PASS**，仅 G7.8d 诚实 fail（L6–L7）。数学上不可达 0.95：ot4 权重 0.25 恒为 0 → 上限 0.75（L279–L287）。
- L1 heuristic 只有 `matmul.j2`/`batch_matmul.j2` 带 `@triton.autotune`，其余 20 个模板硬编码 BLOCK/warps（L154）→ **收益量化证据存在，但显示 L1 heuristic 距离同后端 Triton-best 有系统性差距**。

### 3.3 L2（fusion）收益证据 —— **数据充分且亮眼**

来源 `stage7/track6/l2/PERF_ALL.csv`（511 行，approach 列：separate/liger/arke/torch.compile/FlagGems）。Arke fusion vs 分离算子 / vs Liger 融合 golden：

| L2 op | vs separate（eager 分离）geomean | vs Liger（融合 golden）geomean | n |
|---|---|---|---|
| cross_entropy | **5.773×** | **1.981×** | 13 |
| linear_ce | **4.800×** | **2.510×** | 12 |
| gelu_and_mul | 1.765× | 2.503× | 10 |
| silu_and_mul | 1.743× | 2.448× | 10 |
| **L2 全体（arke 可评估）** | **3.236×** | **2.330×** | 45 |

**结论**：L2 fusion 收益有真实、量化的证据——Arke 融合不仅打赢 eager 分离路径（geomean 3.24×），还打赢同后端 Liger 融合 golden（geomean 2.33×）。这是全审计中 Arke 最强的一块真实性能证据。极端例：linear_ce@llama3-seq2k separate 10.9M μs → arke 105k μs（103×，含 separate 路径接近 OOM 的病态点）。

**但 L2 缺口**：`matmul_gelu`/`matmul_relu`/`qkv_fa`/`geglu`/`swiglu` **没有 arke approach 行**（只有 separate/torch.compile/FlagGems）→ 这 5 个 L2 组合 Arke 融合未实测。S7.followup.2（L231）原记录"L2 Triton fusion baseline 0 evaluable"，现已部分补齐（cross_entropy/linear_ce/gated 有 liger 分母），但 matmul-fusion 系列仍缺 Arke 侧。

### 3.4 L3（agent live）收益与搜索空间收敛

- Phase5 **P5-S5-T live-agent gate 5/5 PASS**（plan.md L576）：约束含 (1) 每 case agent 策略 ≥ default（never worse）；(2) 锁定 C2 回归点（softmax@1024×4096 1.049×）恢复到 ≤1.05×；(3) 整体 latency-weighted geomean ≤0.948×；(4) 策略由 live LLM 带 @rationale 自发现（无种子答案）；(5) held-out shape 泛化。
- G8（L110）live-LLM 闭环 gate 6/6 PASS，GPT-2 geomean 0.9517；KB @rationale 292 条（L22）。
- **搜索空间收敛效率**：AGENTS.md 定义 Bounded Action Space + budget（decision count / compile count）+ trajectory.jsonl 记录 3 个 compile→profile→adjust 周期（plan.md L337）。plan.md L153 记录 Arke token 消耗 ≤ LLM-direct 的 60%、correctness 100% vs LLM-direct 83%。
- **判定**：L3 有 gate 级"never-worse + 泛化"证据与 token 效率数字，但**"搜索空间收敛速度"缺乏专门的收敛曲线/迭代次数分布数据**——只有 budget 上限（400 LOC、3 cycle）和终态 geomean，没有"第 N 次迭代命中最优"的量化收敛证据。

---

## 突出亮点（跨维度 5 条）

1. **L2 fusion 是最硬的真实收益**：Arke 融合 geomean 3.24× vs eager 分离、**2.33× vs 同后端 Liger 融合 golden**（45 个可评估 shape），cross_entropy/linear_ce 达 2–2.5× vs Liger。这是最经得��"同后端公平"检验的性能证据。
2. **SSOT 治理严谨**：46 op 计数由 `benchmark-ops.md` 单一来源解析，`op_registry.py` + test_ssot_op_registry.py 强制全仓一致，杜绝影子目录/硬编码漂移。
3. **shape tier 体系设计完整**：tier1/2/3(含 ST4) 分层清晰，matmul 覆盖 22 shape 含 10 个非对齐/极端（单行 GEMV、off-by-one、非 2 幂）。
4. **诚实测量文化**：G7.8d 拒绝放松 ε/阈值凑分（Path 3 rejected），score 轨迹 0.30→0.32→0.30 每步向真实移动；golden ladder 因发现 FlagGems SDPA 是 bmm 分解非 fused 而 fail-loud 换 flash-attn（golden_ladder.py L82–L94）。
5. **memory-policy 证据链完整**：6GB 显存下 OOM/skip 行携带 `memory_bytes_required/budget/ratio/policy`，覆盖 OT2/OT3/OT4，keeps 覆盖率计费诚实（30 memory_pressure rows）。

## 隐患（跨维度 5 条）

1. **🔴 score=1.0 占位虚高**：`swiglu_packed`/`permute`/`multi_latent_attention`/`paged_attention` 四个算子 Arke 全部 `get_fn declined`（无实现），summary.json 却给 score 1.0 → 覆盖/性能叙事被无数据默认满分污染。需改为"unsupported 不计满分"。
2. **🔴 attention 真实性能落后 ~5×**：对唯一强 attention 基线 flash-attn，Arke L1 flash_attention/GQA geomean �� 0.21（慢约 4.9×），全部 shape 落后。这是 L1 最大真实缺口，且此前被 FlagGems bmm-分解 golden 误标为达标（S7.followup.3 前提失效）。
3. **🟠 整体 geomean 被弱基线拉高**：summary.json 1.1188 / 对 eager 1.97 / 对 torch.compile-autotune 4.1——不能代表同后端竞争力。真实同后端 gate（G7.8d）weighted 仅 0.30。报告任何单一 geomean 数字必须绑定 baseline。
4. **🟠 极端边界"定义了但没实测"**：seq 163k（ds-v3-163k）4 baseline 全 skipped(inf)；reduce 族 N>65536 无实测行（背景事实(3)的截断 bug 修复后回归覆盖有盲区）；softmax/reduce 的 extreme shape 未全部进入 PERF_ALL。
5. **🟠 覆盖缺口**：conv2d 有意不实现（LLM 场景合理但"通用 kernel"叙事有缺）；无 fp8、无按 dtype-pair 拆分的 cast/quant 矩阵；动态 shape 仍停在可行性文档；L2 的 matmul_gelu/matmul_relu/qkv_fa/geglu/swiglu 缺 Arke 融合实测行；GQA@8k 出现 max_abs_diff=0 却判 mismatch 的可疑 correctness 结果。

---

## 核心数字速查

- 46 op = OT0:12 / OT1:10 / OT2:11 / OT3:8 / OT4:5（benchmark-ops.md L14–L21）
- L1 PERF_ALL：1629 行，status ok 86.8% / unsupported 10.9% / skipped 1.8% / timeout 0.5%
- unsupported 率：Liger 59.7% > torch.compile 21.2% > FlagGems 17.4% > Arke 9.6%
- Arke overall geomean：对 flash-attn **0.206**（唯一慢）；对 eager 1.97；对 torch.compile-autotune 4.11
- 最差 op：split score 0.2072（对 eager ratio 0.013–0.078）；flash_attention/GQA 对 flash-attn 0.11–0.20
- 同后端 gate G7.8d：weighted **0.3006**（阈值 0.95，13/14 PASS，诚实 fail）
- **L2 fusion（真实亮点）：Arke vs Liger 融合 golden geomean 2.33×（45 shape）；vs eager 分离 3.24×**
- L3 live-agent gate 5/5 PASS，never-worse + 泛化；token ≤60% LLM-direct，correctness 100% vs 83%
