# Arke Stage 1 Gate Redesign — G5 → G8

> **文档目的：** 以 `benchmark-design.md` 的 BL/OT/ST/L 分层体系为唯一度量标准，
> 重新定义 Stage 1 剩余四个 Gate 的出口条件，并从 Gate 出口能力反推
> **Arke Lang / Arke IR / Arke LLM Agent / Arke 工程** 各层级必须具备的能力与开发项。
>
> **适用范围：** G5（已通过，重定义标准）、G6、G7、G8
>
> **原则：**
> - Gate 出口 = BL × L 的具体组合，可被自动化 `arke bench` 命令直接验证
> - 不引入 BL 体系外的自定义度量，保持 benchmark-design.md 为唯一 Source of Truth
> - 从 Gate 出口倒推 → 各层能力需求 → 具体开发项（可直接进任务板）

---

## Benchmark Level 回顾（度量基础）

```
BL1: OT0-2 × ST1 × L1     — 基础算子 × 微小形状 × 单算子性能
BL2: OT0-2 × ST1-2 × L1   — 基础算子 × 标准形状（含真实 LLM 形状）
BL3: OT0-2 × ST1-3 × L1   — 基础算子 × 全形状（含非对齐 stress）
BL4: OT0-4 × ST1-2 × L1+L2 — 全算子 × 标准形状 × 含融合算子
BL5: OT0-4 × ST1-4 × L1+L2 — 全算子 × 全形状 × 含融合算子
BL6: Model-Complete × L3    — 完整模型前向 × 端到端延迟
```

**Baseline Tiers（性能目标锚点）**
```
P0: cuBLAS/cuDNN (vendor)          最高性能上限
P1: FlagGems / Liger / FlashAttn-2 (expert Triton)
P3: PyTorch eager
P4: torch.compile / Inductor
P5: LLM-direct Triton              Arke 需超越的核心对比基准
```

**算子覆盖（OT0-OT4, 45 ops）**
```
OT0 (12): relu,gelu,silu,tanh,sigmoid,add,mul,where_,cast,neg,exp,rsqrt
OT1 (10): softmax,layernorm,rmsnorm,rmsnorm_residual,reduce_sum,reduce_max,
          reduce_mean,argmax,topk,cumsum
OT2 (11): matmul,batch_matmul,grouped_matmul,transpose,concat,split,
          gather,scatter,embedding,permute,copy_
OT3 (7):  swiglu,geglu,rope,fused_linear_cross_entropy,cross_entropy,
          quantize_per_token,dequantize_per_channel
OT4 (5):  flash_attention,grouped_query_attention,multi_latent_attention,
          cross_attention,paged_attention
```

---

## Gate 定义总览

| Gate | BL 出口 | L 层 | 算子覆盖 | 核心命令 | 状态 |
|:----:|:--------|:----:|:--------|:---------|:----:|
| **G5** | BL3 × L1 + BL6 × L3 | L1 + L3 | OT0-2 (33 ops) + GPT-2 E2E | `arke bench --bl 3 --ot 0-2` + `arke bench --bl 6` | ✅ 已通过（标准重定义） |
| **G6** | BL4 × L1+L2 | L1 + L2 | OT0-4 (45 ops) | `arke bench --bl 4` | ⬜ 当前目标 |
| **G7** | BL5 × L1+L2 + BL6 × L3 | L1+L2+L3 | OT0-4 全形状 + LLaMA-2/DS-V2 E2E | `arke bench --bl 5` + `arke bench --bl 6` | ⬜ |
| **G8** | BL6 × L3（扩展模型集） | L3 | 4 模型 E2E | `arke bench --bl 6 --model all` | ⬜ |

---

## G5 — BL3 × L1 + BL6/GPT-2 × L3 ✅（已通过，标准重定义）

### 出口标准（BL 语言重写）

> 原标准为 "GPT-2 Small 推理正确性 + monkey-patch 延迟"，现以 BL 体系重写。

```
L1 @ BL3（OT0-2, ST1-3）:
  ✅ 正确性: arke bench --bl 3 --ot 0-2 correctness=100%
     (atol/rtol: f16 0.1/0.05, f32 1e-5/1e-4)
  ⚠️ 性能: geomean(OT0-2, ST1-3) ≥ P3 (torch eager)   [已达到]
     NOTE: P0/P1 性能目标延后至 G6（受 dispatch 架构限制）

L3 @ BL6 / GPT-2 Small:
  ✅ 正确性: top-1 token 全 seq_len 匹配 eager
  ✅ 覆盖率: 49/48 Conv1D 替换
  ✅ 内存: ≤ 6GB VRAM
  ⚠️ 延迟: 1.71-2.20× eager  [known-fail，Stage 2 解决]
```

### G5 Known-Fail 分析（记录，非阻塞）

| 现象 | 根因 | 解决时机 |
|:-----|:-----|:---------|
| E2E 延迟 1.7-2.3× eager | monkey-patch dispatch ~60µs/call × 49 次 | G7: torch.compile Inductor backend |
| 单 matmul: Arke 76µs vs cuBLAS 44µs | L1 单算子已 OK，累积 dispatch 开销 | G6: BL4 统一度量 |

---

## G6 — BL4 × L1+L2（全算子 × 标准形状）

> **核心目标：** Arke 能为 45 个算子的标准 LLM 生产形状生成正确且有竞争力的 Triton kernel。
> 这是 Arke 从"能跑"到"能用"的分水岭。

### 出口标准

```bash
# 验证命令
arke bench --bl 4 --layer l1     # OT0-4 × ST1-2, 单算子性能
arke bench --bl 4 --layer l2     # OT0-4 × ST1-2, 融合算子性能
```

#### L1 @ BL4（OT0-4, ST1-2）

| 维度 | 要求 | 测量方式 |
|:-----|:-----|:---------|
| **正确性** | 100%（全 45 ops × ST1-2 shapes） | `bench_l1 correctness` pass rate |
| **OT0 性能** | geomean ≥ 0.90 P1（FlagGems elementwise） | `bench_l1 --ot 0 --tier 1-2` |
| **OT1 性能** | geomean ≥ 0.85 P1（FlagGems norm/softmax） | `bench_l1 --ot 1 --tier 1-2` |
| **OT2 性能** | geomean ≥ 0.90 P0（cuBLAS matmul）; 其余 ≥ P3 | `bench_l1 --ot 2 --tier 1-2` |
| **OT3 性能** | geomean ≥ 0.85 P1（Liger swiglu/geglu）; rope ≥ P3 | `bench_l1 --ot 3 --tier 1-2` |
| **OT4 性能** | geomean ≥ 0.80 P1（FlashAttn-2 / FlagGems GQA） | `bench_l1 --ot 4 --tier 1-2` |

#### L2 @ BL4（融合算子, ST1-2）

| 融合组合 | 要求 | baseline |
|:---------|:-----|:---------|
| matmul+relu, matmul+gelu | ≥ 1.05× unfused（融合收益） | P3 unfused |
| swiglu, geglu | ≥ 0.90× Liger | P1 |
| linear+cross_entropy | ≥ 1.05× unfused | P3 |
| QKV+flash_attention | ≥ 0.80× FlashAttn-2 | P1 |

#### 综合评分（G6 PASS 条件）

```
weighted_score = 0.3×score(OT0-2) + 0.3×score(OT3) + 0.4×score(OT4)
G6 PASS iff:
  - 正确性: ALL(correctness=100%)
  - weighted_score ≥ 0.85
  - L2 融合收益: ≥3/4 组合达标
```

---

### G6 能力反推

#### Arke Lang（.ak 语言层）

G6 出口要求 45 ops 全部可在 `.ak` 中表达并执行，反推语言能力：

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| 支持 4D tensor op 语法（rope, attention） | OT4 必须 expressible | `D6-L1: .ak 语法扩展 — 4D tensor, einsum 标注` |
| 支持 index/gather/scatter op 语义 | OT2 新增 7 个 data-movement ops | `D6-L2: gather/scatter 语义节点` |
| 支持量化 op（quantize_per_token 等） | OT3 量化算子 | `D6-L3: quantize 原语语法` |
| 支持 paged KV / block_table 参数 | OT4 paged_attention | `D6-L4: paged memory 语义标注（可延后至 G7）` |
| array literal + float constant | 修复 G1.4 遗留语法 gap | `D6-L5: grammar fix（已规划，前置到 G6）` |

#### Arke IR（Semantic IR + Strategy IR）

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| OT3/OT4 op 节点定义（35 新增） | BL4 覆盖 45 ops | `D6-IR1: SemanticIR op catalog 扩展至 45 ops` |
| OT4 attention 的 causal mask、KV-head 分组字段 | GQA/MHA 正确性 | `D6-IR2: AttentionSemanticIR 字段（mask_type, num_kv_heads, head_dim）` |
| rope 的 rotary 参数（theta, base） | RoPE 正确性 | `D6-IR3: RopeSemanticIR 字段` |
| 量化 op 的 scale/zero_point 类型字段 | quantize 正确性 | `D6-IR4: QuantizeSemanticIR 字段` |
| Strategy IR: 内存访问模式决策（gather/scatter pattern） | OT2 data-movement 优化 | `D6-IR5: MemoryAccessStrategy 决策类型` |

#### Arke LLM Agent

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| OT4 attn kernel 策略决策（tile_q/kv, pipeline stages） | BL4 L1 OT4 ≥ 0.80 P1 | `D6-A1: attention prompt template（causal mask、GQA group 展开）` |
| rope kernel 策略（向量化 cos/sin, half-rotate） | OT3 rope ≥ P3 | `D6-A2: rope prompt + rationale template` |
| 融合决策（L2 linear+CE, QKV+attn） | L2 融合收益 | `D6-A3: fusion opportunity detection in agent tools` |
| 量化 kernel 策略（per-token scale, 向量化） | OT3 quant ≥ P3 | `D6-A4: quantize/dequantize prompt template` |
| 多算子批量优化（BL4 45 ops 不能逐个人工） | 自动化 | `D6-A5: batch optimize pipeline（op_list × shape_list 并行 session）` |

#### Arke 工程（基础设施）

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| Triton 模板覆盖 35 个新算子 | BL4 correctness 100% | `D6-E1: Triton codegen 模板 — rope, flash_attention, GQA, MLA, gather, scatter, embedding, quantize（8 类）` |
| `bench_l1` 集成 shape_registry 路由（所有 45 ops） | BL4 可执行 | `D6-E2: bench_l1.py 路由扩展 + shape_registry 接入` |
| `bench_l2` 覆盖 OT3/OT4 融合算子 | L2 BL4 可执行 | `D6-E3: bench_l2.py OT3/OT4 融合 benchmark runner` |
| Baseline 适配：FlashAttn-2, Liger, FlagGems GQA/rope | OT4 P1 基准 | `D6-E4: operator-source-registry 实现（FA-2, Liger rope/quant, FlagGems GQA）` |
| CSV 输出按 BL/OT/L 层级组织 | benchmark 结果可追溯 | `D6-E5: PerfCSVWriter 输出目录 L1/OT{n}/perf_{op}.csv` |
| 正确性验证 harness 覆盖 45 ops | correctness 100% | `D6-E6: V1 validator 扩展（attention 数值容差、量化精度标准）` |

---

### G6 开发项汇总

```
优先级 P0（阻塞 G6 正确性）:
  D6-IR1  SemanticIR op catalog → 45 ops
  D6-IR2  AttentionSemanticIR 字段
  D6-IR3  RopeSemanticIR 字段
  D6-IR4  QuantizeSemanticIR 字段
  D6-E1   Triton 模板 8 类（rope/flash_attention/GQA/MLA/gather/scatter/embedding/quantize）
  D6-E6   V1 validator 扩展
  D6-L5   grammar fix（array literal, float constant）

优先级 P1（阻塞 G6 性能目标）:
  D6-E2   bench_l1 路由扩展
  D6-E3   bench_l2 OT3/OT4 融合 runner
  D6-E4   baseline 适配（FA-2, Liger, FlagGems GQA）
  D6-A1   attention agent prompt template
  D6-A2   rope agent prompt template
  D6-A3   fusion opportunity detection
  D6-A5   batch optimize pipeline

优先级 P2（支持 G6，不阻塞）:
  D6-L1   .ak 4D tensor 语法扩展
  D6-L2   gather/scatter 语义节点
  D6-L3   quantize 原语语法
  D6-IR5  MemoryAccessStrategy
  D6-E5   CSV 输出目录结构
  D6-A4   quantize/dequantize prompt
```

---

## G7 — BL5 × L1+L2 + BL6/LLaMA-2+DS-V2 × L3

> **核心目标：** 全算子全形状（含 ST3 非对齐 + ST4 生产规模）+ 两个大模型 E2E。
> 这是 Arke "能用于生产" 的验证 Gate。

### 出口标准

```bash
arke bench --bl 5 --layer l1       # OT0-4 × ST1-4，全形状单算子性能
arke bench --bl 5 --layer l2       # 全算子融合性能
arke bench --bl 6 --model llama2   # LLaMA-2 7B E2E
arke bench --bl 6 --model deepseek # DeepSeek-V2 E2E
```

#### L1 @ BL5（OT0-4, ST1-4）

| 维度 | 要求 | 说明 |
|:-----|:-----|:-----|
| **正确性** | 100% ST1-3；ST4 ≥ 95%（OOM 标注除外） | ST4 允许部分 OOM（标注 `⚠️ may OOM`）|
| **OT0-1 性能** | ST1-3 geomean ≥ 0.90 P1；ST4 ≥ 0.85 P1 | 包含 non-aligned shapes |
| **OT2 性能** | matmul ST4 geomean ≥ 0.92 P0；其余 ≥ P3 | LLaMA-3/Qwen2.5 production 形状 |
| **OT3 性能** | swiglu/rope ST4 geomean ≥ 0.88 P1 | DeepSeek/LLaMA-3 long-ctx 形状 |
| **OT4 性能** | flash_attention ST4 geomean ≥ 0.82 P1；GQA ST4 ≥ 0.80 P1 | 含 8K/32K context |

#### L2 @ BL5（扩展融合）

| 融合组合 | 要求 |
|:---------|:-----|
| G6 融合组合（全部） | 继承 G6 要求 |
| rope + flash_attention（decode 路径） | ≥ 0.85× separate rope + FA-2 |
| quant + dequant + matmul（INT8 推理路径） | ≥ 1.10× unfused（INT8 量化收益） |

#### L3 @ BL6（LLaMA-2 7B + DeepSeek-V2 16B）

| 模型 | 正确性 | 性能 | 内存 |
|:-----|:-------|:-----|:-----|
| LLaMA-2 7B | top-1 token 100% 匹配 eager；所有 seq_len | Arke ≤ 1.30× eager（torch.compile backend 架构） | ≤ 6GB VRAM（seq≤2048） |
| DeepSeek-V2 16B | top-1 token 100% 匹配 eager；seq∈{512,2048} | Arke ≤ 1.40× eager（MoE dispatch 开销） | ≤ 6GB VRAM（seq≤512，注意量化）|

**G7 PASS 条件：**
```
- L1 BL5: 正确性 100%(ST1-3) + ≥95%(ST4)
- L1 BL5: 各 OT 性能 geomean 达标（见上表）
- L2 BL5: ≥5/6 融合组合达标
- L3 BL6: LLaMA-2 + DS-V2 正确性 100% + 延迟 ≤ 阈值
```

---

### G7 能力反推

#### Arke Lang

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| long-context 语义标注（seq=8K~32K） | ST4 OT4 + L3 BL6 | `D7-L1: .ak @context_len 标注原语` |
| paged_attention 块表参数 | BL5 OT4 paged_attention | `D7-L2: paged memory 语义节点（block_table, page_size）` |
| MoE dispatch/combine 语义 | DS-V2 E2E（grouped_matmul + gather + scatter） | `D7-L3: moe_dispatch / moe_combine 高级原语（语法糖）` |
| multi-latent attention 参数 | BL5 OT4 MLA | `D7-L4: MLA 参数语义节点（latent_dim, kv_lora_rank）` |
| 量化 kernel 的 dtype 原语 | BL5 OT3 quant × ST4 | `D7-L5: @dtype int8/fp8 标注扩展` |

#### Arke IR

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| Strategy IR Level 2：pipeline stage 决策 | OT4 ST4 性能（长 context 需 pipelining） | `D7-IR1: PipelineStageStrategy 决策类型（prefill/decode 分离）` |
| MLA 特有字段（latent KV compress） | BL5 MLA 正确性 + 性能 | `D7-IR2: MultiLatentAttentionIR 字段（kv_lora_rank, qk_rope_head_dim）` |
| grouped_matmul 动态 expert routing | DS-V2 MoE 正确性 | `D7-IR3: GroupedMatmulSemanticIR expert_indices 字段` |
| 非对齐形状 padding 策略 | ST3 non-aligned 性能 | `D7-IR4: PaddingStrategy 决策类型（pad_to_multiple, dynamic_padding）` |

#### Arke LLM Agent

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| long-context 专属策略（chunk prefill, KV cache split） | L3 BL6 LLaMA-2/DS-V2 延迟 ≤ 阈值 | `D7-A1: long-context agent prompt（seq>4K 分支策略）` |
| 非对齐形状策略（padding vs masking 权衡） | ST3 性能 ≥ 目标 | `D7-A2: non-aligned shape agent rationale template` |
| MoE routing 感知（sparse dispatch 优化） | DS-V2 E2E | `D7-A3: MoE-aware optimization prompt（top-k sparsity, load balance）` |
| 多阶段 E2E profiling 反馈 | L3 BL6 延迟精确控制 | `D7-A4: E2E profile→kernel feedback loop（瓶颈算子识别 → re-optimize）` |
| INT8 kernel 策略（dequant fusion） | L2 BL5 quant+matmul | `D7-A5: quantized inference agent prompt（W4A8, W8A8 策略）` |

#### Arke 工程

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| torch.compile Inductor backend（替代 monkey-patch） | L3 BL6 延迟 ≤ 1.30× eager | `D7-E1: arke/integration/torch_compile_backend.py（Inductor custom op 注册）` |
| LLaMA-2 7B 集成测试 | L3 BL6 LLaMA-2 | `D7-E2: examples/llama2_arke.py + L3 bench runner` |
| DeepSeek-V2 集成测试 | L3 BL6 DS-V2 | `D7-E3: examples/deepseek_v2_arke.py + L3 bench runner（量化权重）` |
| MLA 实现（latent attention kernel） | BL5 OT4 MLA 正确性 | `D7-E4: Triton MLA template（compressed KV, lora-style project）` |
| paged_attention kernel | BL5 OT4 paged_attention | `D7-E5: Triton paged_attention template（block table scatter read）` |
| ST4 shapes 自动 OOM 检测与跳过 | BL5 ST4 OOM 处理 | `D7-E6: bench runner OOM guard（try/catch + ⚠️ 标注写入 CSV）` |
| BL6 模型完整 bench runner | L3 自动化 | `D7-E7: bench_l3.py — 模型 forward pass 多次测量 + top-1 token 比对` |

---

### G7 开发项汇总

```
优先级 P0（阻塞 L3 BL6 正确性）:
  D7-E1   torch.compile Inductor backend
  D7-E2   LLaMA-2 7B 集成
  D7-E7   bench_l3.py

优先级 P1（阻塞 BL5 性能目标）:
  D7-IR1  PipelineStageStrategy
  D7-E4   MLA Triton template
  D7-E5   paged_attention Triton template
  D7-A1   long-context agent prompt
  D7-A3   MoE-aware prompt

优先级 P2:
  D7-L1~L5  .ak 语言扩展
  D7-IR2~4  IR 字段扩展
  D7-E3   DeepSeek-V2 集成（seq≤512，内存受限）
  D7-E6   OOM guard
  D7-A2   non-aligned prompt
  D7-A4   E2E profile feedback loop
  D7-A5   量化推理 prompt
```

---

## G8 — BL6 × L3（4 模型 E2E + Arke 完整自主能力验证）

> **核心目标：** Stage 1 最终 Gate。Arke 能自主为 4 个真实 LLM 生成完整 kernel 集，
> 端到端性能全部满足生产可用阈值，并量化验证 Arke vs LLM-direct 的优势。
> 同时完成实现语言评估（Python vs 混合方案）。

### 出口标准

```bash
arke bench --bl 6 --model gpt2      # GPT-2 Small
arke bench --bl 6 --model llama2    # LLaMA-2 7B
arke bench --bl 6 --model llama3    # LLaMA-3 8B  
arke bench --bl 6 --model qwen25    # Qwen2.5 7B
```

#### L3 @ BL6（4 模型）

| 模型 | 正确性 | 性能阈值 | 内存 | seq 覆盖 |
|:-----|:-------|:---------|:-----|:---------|
| **GPT-2 Small** | top-1 100% | Arke ≤ **1.15×** eager（G5 known-fail 修复） | ≤ 4GB | 128/512/1024 |
| **LLaMA-2 7B** | top-1 100% | Arke ≤ **1.20×** eager | ≤ 6GB | 512/2048/4096 |
| **LLaMA-3 8B** | top-1 100% | Arke ≤ **1.20×** eager | ≤ 6GB | 512/2048/8192 |
| **Qwen2.5 7B** | top-1 100% | Arke ≤ **1.25×** eager（GQA 7:1 + 极宽 FFN） | ≤ 6GB | 512/2048 |

#### BL5 完整回归（继承 G7 结果，必须不退步）

```
- L1 BL5 全 45 ops：正确性不退步（≥ G7 水准）
- L1 BL5 OT0-4 性能：geomean 不低于 G7 结果
- L2 BL5 融合算子：≥ G7 覆盖
```

#### Arke vs LLM-direct 对比

| 指标 | Arke 目标 | LLM-direct 基准 | 依据 |
|:-----|:---------|:----------------|:-----|
| 正确性 | ≥ 98%（G8 全算子） | 历史值 ~83% | G4 数据 |
| 性能 geomean（BL5 L1） | ≥ 1.05× LLM-direct | — | Arke 结构化搜索优势 |
| 性能方差（stddev） | ≤ 0.5× LLM-direct | — | 确定性 IR 约束降方差 |
| Token 消耗/kernel | ≤ 0.7× LLM-direct | — | IR 约束减少探索 token |
| 端到端延迟（BL6 L3） | 每个模型 ≤ 1.25× eager | — | 用户可感知阈值 |

#### 实现语言评估（G8 同步完成）

```
G8-Lang: Python vs 混合方案数据驱动评估
  测量项：dispatch overhead（Python 路径 vs Rust/C++ 理论），
          parse latency（.ak → IR），内存占用，LLM API 集成成本
  产出：language-decision.md（结论 + 数据 + Stage 2 迁移策略）
```

**G8 PASS 条件：**
```
AND ALL:
  [1] 4 模型 L3 BL6 正确性 100%
  [2] 4 模型 E2E 延迟全部 ≤ 阈值（GPT-2≤1.15×, LLaMA-2/3≤1.20×, Qwen2.5≤1.25×）
  [3] Arke vs LLM-direct: 正确性 ≥1.15×，性能 geomean ≥1.05×，token ≤0.7×
  [4] BL5 L1 全 45 ops 无性能退步
  [5] language-decision.md 完成
```

---

### G8 能力反推

#### Arke Lang

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| Qwen2.5 7B 覆盖（GQA 7:1, FFN=18944, V=151936） | BL6 Qwen2.5 L3 | `D8-L1: .ak 示例 — qwen25_forward.ak（GQA+SwiGLU+RMSNorm 完整描述）` |
| LLaMA-3 GQA（Hq=32, Hkv=8）表达 | BL6 LLaMA-3 L3 | `D8-L2: .ak 示例 — llama3_forward.ak` |
| `.ak` I/O Spec 文档（自然语言 → .ak） | G7.1 输入类型支持 | `D8-L3: docs/spec/arke-io-spec.md` |
| 语言规范 v1.0 冻结 | Stage 1 完成标志 | `D8-L4: Language Spec v1.0（arke-lang-spec.md 更新 + 标签）` |

#### Arke IR

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| IR 规范 v1.0 冻结（含所有 45 ops + Level 2 Strategy） | Stage 1 完成标志 | `D8-IR1: IR Spec v1.0（arke-ir-spec.md 更新 + 标签）` |
| IR ↔ MLIR 映射文档 | Stage 2 准备 | `D8-IR2: docs/spec/ir-mlir-mapping.md` |
| 完整 round-trip 验证（from_json/to_json 全 45 ops） | IR Spec v1.0 | `D8-IR3: test_ir_roundtrip.py — 45 ops × JSON round-trip` |

#### Arke LLM Agent

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| 自主 I/O 合约（输入类型 a/b/c → Arke pipeline） | G7.1 三种输入 | `D8-A1: arke optimize <input> 统一入口（.ak / 自然语言 / 现有代码）` |
| 自动策略生成（无人工 strategy block） | G7.4 LLM auto-strategy | `D8-A2: kernel-only .ak → LLM 自动生成 strategy → ≥ 80% cuBLAS` |
| ≥3 轮迭代优化 | G7.5 | `D8-A3: iterative optimization loop（compile→profile→adjust 自动触发）` |
| @rationale 知识库（≥50 条 Stage 1 条目） | H3 可解释性 | `D8-A4: trajectory knowledge distillation → rationale_kb.jsonl` |
| BL5 全算子批量 agent 覆盖 | G8 BL5 不退步 | `D8-A5: batch optimize validation（全 45 ops 验证 agent 生成 kernel）` |

#### Arke 工程

| 能力需求 | 依据 | 开发项 |
|:---------|:-----|:-------|
| LLaMA-3 8B 集成（GQA, rope, RMSNorm） | BL6 LLaMA-3 L3 | `D8-E1: examples/llama3_arke.py + bench_l3 runner` |
| Qwen2.5 7B 集成（GQA 7:1, SwiGLU, wide FFN） | BL6 Qwen2.5 L3 | `D8-E2: examples/qwen25_arke.py + bench_l3 runner` |
| GPT-2 延迟修复（≤ 1.15× eager） | G5 known-fail 解决 | `D8-E3: torch.compile backend E2E 集成（依赖 D7-E1）` |
| BL5 全 45 ops 回归套件 | G8 BL5 不退步 | `D8-E4: ci/regression_bl5.py（每次 commit 跑 BL5 L1 正确性）` |
| Arke vs LLM-direct 自动对比 benchmark | G8 对比指标 | `D8-E5: benchmarks/compare_arke_vs_direct.py` |
| 语言评估 benchmark | language-decision.md | `D8-E6: dispatch overhead 测量脚本 + memory profiler` |
| Stage 1 最终评估报告 | Stage 1 完成标志 | `D8-E7: benchmarks/results/stage1/STAGE1_FINAL_REPORT.md` |

---

### G8 开发项汇总

```
优先级 P0（阻塞 4 模型 L3 BL6 正确性）:
  D8-E1   LLaMA-3 8B 集成
  D8-E2   Qwen2.5 7B 集成
  D8-E3   GPT-2 torch.compile backend（依赖 D7-E1）

优先级 P1（阻塞 G8 PASS 条件 [3][4][5]）:
  D8-A1   arke optimize 统一入口
  D8-A2   LLM auto-strategy
  D8-A3   iterative optimization loop
  D8-E4   BL5 回归套件
  D8-E5   Arke vs LLM-direct 对比
  D8-E6   语言评估
  D8-E7   Stage 1 最终报告

优先级 P2:
  D8-L1~4 .ak 示例 + 语言规范 v1.0
  D8-IR1~3 IR 规范 v1.0 + MLIR 映射 + round-trip 测试
  D8-A4   @rationale 知识库
  D8-A5   批量 agent 验证
```

---

## Gate 依赖链与开发路径

```
G5 ✅  BL3/L1 OT0-2 + BL6/GPT-2 L3
  │    正确性 ✅ | 性能 ⚠️ known-fail (monkey-patch dispatch)
  │
  ▼
G6 ⬜  BL4/L1+L2 OT0-4 × ST1-2
  │
  │  P0 前置: D6-IR1(IR catalog) + D6-E1(Triton 模板 8 类) + D6-L5(grammar fix)
  │  P1 跟进: D6-E2/E3/E4(bench runner) + D6-A1/A2/A3(agent prompt)
  │
  ▼
G7 ⬜  BL5/L1+L2 OT0-4 × ST1-4 + BL6/LLaMA-2+DS-V2 L3
  │
  │  P0 前置: D7-E1(torch.compile backend) + D7-E2(LLaMA-2) + D7-E7(bench_l3)
  │  P1 跟进: D7-E4(MLA) + D7-E5(paged_attn) + D7-A1(long-ctx agent)
  │
  ▼
G8 ⬜  BL6/4模型 L3 + BL5 不退步 + Arke vs LLM-direct 对比
       = Stage 1 最终 Gate
  │
  │  P0: D8-E1(LLaMA-3) + D8-E2(Qwen2.5) + D8-E3(GPT-2 latency fix)
  │  P1: D8-A1~A3(agent 自主能力) + D8-E4~E7(工程收尾)
  │
  ▼
Stage 2: Ascend Backend（H4 跨架构验证）
```

### 关键路径

最长开发路径（G6 → G8）：

```
D6-IR1 → D6-E1(Triton 模板) → D6-E2(bench_l1) → G6
                                    ↓
D7-E1(torch.compile backend) → D7-E2(LLaMA-2) → G7
                                    ↓
D8-E1(LLaMA-3) + D8-E2(Qwen2.5) → D8-A1~A3 → G8
```

**瓶颈：** D6-E1（8 类 Triton 模板）是 G6 的最大工作量，`flash_attention` 和 `MLA` 模板最复杂。

---

## 与 plan-v3.0.md Phase 1.9-1.11 的对应关系

| plan-v3.0 Phase | gate-redesign-v2 对应 | 关键差异 |
|:----------------|:---------------------|:---------|
| Phase 1.9 (G6: Arke Lang & IR Completeness) | **G6 = BL4 × L1+L2** | 出口改为 BL4 可执行验证；删除 G6.6 MLIR mapping（移至 G8） |
| Phase 1.10 (G7: Autonomous Kernel Gen) | **G7 = BL5 × L1+L2+L3/LLaMA-2+DS-V2** | L3 具体化为 BL6 两个模型；性能阈值明确 |
| Phase 1.11 (G8: Language Assessment) | **G8 = BL6/4模型 L3 + 对比** | 语言评估合并入 G8；Stage 1 最终 Gate |

---

## 附录：开发项全表（按 Gate 分层）

### G6 开发项（25 项）

| ID | 层 | 描述 | 优先级 | 估时 |
|:---|:---|:-----|:------:|:----:|
| D6-IR1 | IR | SemanticIR op catalog → 45 ops（含 OT3/OT4 所有字段） | P0 | M |
| D6-IR2 | IR | AttentionSemanticIR（mask_type, num_kv_heads, head_dim） | P0 | S |
| D6-IR3 | IR | RopeSemanticIR（theta, base, rotary_dim） | P0 | S |
| D6-IR4 | IR | QuantizeSemanticIR（scale_dtype, group_size, zero_point） | P0 | S |
| D6-IR5 | IR | MemoryAccessStrategy 决策类型 | P2 | S |
| D6-L1 | Lang | .ak 4D tensor 语法扩展 | P2 | M |
| D6-L2 | Lang | gather/scatter 语义节点 | P2 | S |
| D6-L3 | Lang | quantize 原语语法 | P2 | S |
| D6-L4 | Lang | paged memory 语义标注（stub） | P2 | S |
| D6-L5 | Lang | grammar fix（array literal, float constant） | P0 | S |
| D6-A1 | Agent | attention prompt template（causal mask, GQA） | P1 | M |
| D6-A2 | Agent | rope prompt + rationale template | P1 | S |
| D6-A3 | Agent | fusion opportunity detection | P1 | M |
| D6-A4 | Agent | quantize/dequantize prompt template | P2 | S |
| D6-A5 | Agent | batch optimize pipeline（45 ops 并行 session） | P1 | L |
| D6-E1 | Eng | Triton 模板 8 类：rope/FA/GQA/MLA/gather/scatter/embedding/quantize | P0 | XL |
| D6-E2 | Eng | bench_l1 路由扩展（45 ops + shape_registry 接入） | P1 | M |
| D6-E3 | Eng | bench_l2 OT3/OT4 融合 benchmark runner | P1 | M |
| D6-E4 | Eng | baseline 适配（FlashAttn-2, Liger rope/quant, FlagGems GQA） | P1 | M |
| D6-E5 | Eng | CSV 输出目录 L1/OT{n}/perf_{op}.csv | P2 | S |
| D6-E6 | Eng | V1 validator 扩展（attention 容差, 量化精度标准） | P0 | S |

### G7 开发项（20 项）

| ID | 层 | 描述 | 优先级 | 估时 |
|:---|:---|:-----|:------:|:----:|
| D7-IR1 | IR | PipelineStageStrategy（prefill/decode 分离） | P1 | M |
| D7-IR2 | IR | MultiLatentAttentionIR（kv_lora_rank, qk_rope_head_dim） | P1 | S |
| D7-IR3 | IR | GroupedMatmulSemanticIR expert_indices 字段 | P1 | S |
| D7-IR4 | IR | PaddingStrategy（pad_to_multiple, dynamic_padding） | P2 | S |
| D7-L1 | Lang | .ak @context_len 标注原语 | P2 | S |
| D7-L2 | Lang | paged memory 语义节点（block_table, page_size） | P1 | M |
| D7-L3 | Lang | moe_dispatch/combine 高级原语 | P2 | M |
| D7-L4 | Lang | MLA 参数语义节点 | P2 | S |
| D7-L5 | Lang | @dtype int8/fp8 标注扩展 | P2 | S |
| D7-A1 | Agent | long-context agent prompt（seq>4K 分支） | P1 | M |
| D7-A2 | Agent | non-aligned shape rationale template | P2 | S |
| D7-A3 | Agent | MoE-aware optimization prompt | P1 | M |
| D7-A4 | Agent | E2E profile→kernel feedback loop | P2 | L |
| D7-A5 | Agent | 量化推理 agent prompt（W4A8, W8A8） | P2 | M |
| D7-E1 | Eng | torch.compile Inductor backend | P0 | XL |
| D7-E2 | Eng | LLaMA-2 7B 集成 + bench_l3 runner | P0 | L |
| D7-E3 | Eng | DeepSeek-V2 集成（seq≤512, 量化权重） | P2 | L |
| D7-E4 | Eng | Triton MLA template（compressed KV, lora project） | P1 | L |
| D7-E5 | Eng | Triton paged_attention template | P1 | L |
| D7-E6 | Eng | bench runner OOM guard + CSV 标注 | P2 | S |
| D7-E7 | Eng | bench_l3.py（模型 forward + top-1 比对） | P0 | M |

### G8 开发项（18 项）

| ID | 层 | 描述 | 优先级 | 估时 |
|:---|:---|:-----|:------:|:----:|
| D8-IR1 | IR | IR Spec v1.0（文档 + 标签） | P1 | M |
| D8-IR2 | IR | ir-mlir-mapping.md | P1 | M |
| D8-IR3 | IR | test_ir_roundtrip.py（45 ops × JSON round-trip） | P1 | S |
| D8-L1 | Lang | qwen25_forward.ak 示例 | P0 | S |
| D8-L2 | Lang | llama3_forward.ak 示例 | P0 | S |
| D8-L3 | Lang | arke-io-spec.md | P1 | M |
| D8-L4 | Lang | Language Spec v1.0（文档 + 标签） | P1 | M |
| D8-A1 | Agent | arke optimize 统一入口（3 种输入类型） | P1 | L |
| D8-A2 | Agent | LLM auto-strategy（无 strategy block） | P1 | L |
| D8-A3 | Agent | iterative optimization loop | P1 | M |
| D8-A4 | Agent | @rationale 知识库（≥50 条） | P2 | M |
| D8-A5 | Agent | 批量 agent 验证（全 45 ops） | P1 | M |
| D8-E1 | Eng | LLaMA-3 8B 集成 + bench_l3 | P0 | L |
| D8-E2 | Eng | Qwen2.5 7B 集成 + bench_l3 | P0 | L |
| D8-E3 | Eng | GPT-2 torch.compile backend E2E（≤1.15× eager） | P0 | M |
| D8-E4 | Eng | BL5 回归套件（CI） | P1 | M |
| D8-E5 | Eng | Arke vs LLM-direct 自动对比 | P1 | M |
| D8-E6 | Eng | 语言评估 benchmark + language-decision.md | P1 | M |
| D8-E7 | Eng | Stage 1 最终评估报告 | P1 | M |

---

> **估时说明：** S≤1d, M≤3d, L≤1w, XL>1w（单人，参考值）
>
> *Created: 2026-04-05 | Author: Kitty (Arke Project)*
