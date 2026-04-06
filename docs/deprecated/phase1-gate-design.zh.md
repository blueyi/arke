# Arke Phase 1 — Gate Design

> **文档目的：** 以 `benchmark-design.md` 的 BL/OT/ST/L 分层体系为唯一度量标准，定义
> Phase 1 全部九个 Gate（G0-G8）的出口条件，并从 Gate 出口能力反推各层必须具备的能力与开发项。
>
> **设计原则：**
> - Gate 出口 = BL × L 的具体组合，可被 `arke bench` 命令直接验证
> - `benchmark-design.md` 为唯一度量 Source of Truth，不引入 BL 体系外的自定义度量
> - 从 Gate 出口倒推 → 各层能力需求 → 具体开发项（可直接进任务板）
> - G6 = Lang/IR 完备性（BL5×L1+L2）；G7 = Arke 自主工程能力；G8 = Phase 1 最终验收
>
> **后续参考：** Phase 2 起，以此格式创建 `phase2-gate-design.md` 等文档。
>
> *Created: 2026-04-05 | benchmark-design.md rev: 2026-04-05*

---

## 目录

1. [BL 体系回顾（度量基础）](#1-bl-体系回顾度量基础)
2. [Gate 总览表](#2-gate-总览表)
3. [Part I — G0-G4（已通过 Gate）](#3-part-i--g0-g4已通过-gate)
4. [Part II — G5（已通过，标准回顾性重写）](#4-part-ii--g5已通过标准回顾性重写)
5. [G6 — BL5×L1+L2：Lang & IR 完备性（当前目标）](#5-g6--bl5l1l2lang--ir-完备性当前目标)
6. [G7 — Arke Autonomous Engineering](#6-g7--arke-autonomous-engineering)
7. [G8 — Phase 1 最终验收](#7-g8--stage-1-最终验收)
8. [Gate 依赖链](#8-gate-依赖链)
9. [开发项附录](#9-开发项附录)
10. [与 execution-plan.md 的对应关系](#10-与-execution-planmd-的对应关系)

---

## 1. BL 体系回顾（度量基础）

### Benchmark Level（BL）定义

| Level | Operator 覆盖 | Shape 覆盖 | 描述 | 典型用途 |
|:-----:|:-------------|:----------|:-----|:--------|
| **BL1** | OT0-OT2 | ST1 | 基础算子 × 微小形状 | 快速冒烟测试 <30s |
| **BL2** | OT0-OT2 | ST1-ST2 | 基础算子 × 标准形状 | 日常 CI ~5min |
| **BL3** | OT0-OT2 | ST1-ST3 | 基础算子 × 全形状（含非对齐） | Gate 验证 |
| **BL4** | OT0-OT4 | ST1-ST2 | **全算子** × 标准形状 | 算子完备性 |
| **BL5** | OT0-OT4 | ST1-ST4 | **全算子 × 全形状** | 完整 benchmark 套件 |
| **BL6** | Model-Complete | Model-Real | 真实模型图：全算子+生产形状 | E2E 模型验证 |

**OT × ST 覆盖矩阵**

```
              ST1(micro)  ST2(standard)  ST3(stress)  ST4(production)
OT0 (elem)      BL1          BL2            BL3            -
OT1 (reduce)    BL1          BL2            BL3            -
OT2 (dense)     BL1          BL2            BL3           BL5
OT3 (gated)     BL4          BL4            -             BL5
OT4 (attn)       -            -             -             BL5
Model-Complete  ──────────────── BL6 ────────────────────────
```

**Layer × BL 覆盖矩阵**

```
         BL1  BL2  BL3  BL4  BL5  BL6
L1        ✓    ✓    ✓    ✓    ✓    -     单算子性能
L2        -    -    -    ✓    ✓    -     融合算子性能
L3 ≡ BL6  -    -    -    -    -    ✓     E2E 模型性能
```

> **L3 ≡ BL6**：L3 是对 BL6 模型完整算子+形状集的端到端前向推理执行。BL6 定义覆盖范围，L3 定义测量方法。

### Baseline Priority Tiers

| Tier | 名称 | 来源 |
|:----:|:-----|:-----|
| **P0** | Vendor-optimized | cuBLAS, cuDNN, CUTLASS |
| **P1** | Expert Triton | FlagGems, Liger-Kernel, FlashAttention-2 |
| **P2** | Reference Triton | Triton 官方教程 |
| **P3** | PyTorch eager | `torch.nn.functional` |
| **P4** | Inductor-generated | `torch.compile` 输出 |
| **P5** | LLM-direct | LLM 直接写 Triton |

### 算子覆盖（45 ops，OT0-OT4）

| Tier | 数量 | 算子 |
|:-----|:----:|:-----|
| **OT0** 元素级 | 12 | `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `add`, `mul`, `where_`, `cast`, `neg`, `exp`, `rsqrt` |
| **OT1** 规约 | 10 | `softmax`, `layernorm`, `rmsnorm`, `rmsnorm_residual`, `reduce_sum`, `reduce_max`, `reduce_mean`, `argmax`, `topk`, `cumsum` |
| **OT2** 计算密集 | 11 | `matmul`, `batch_matmul`, `grouped_matmul`, `transpose`, `concat`, `split`, `gather`, `scatter`, `embedding`, `permute`, `copy_` |
| **OT3** 门控激活 | 7 | `swiglu`, `geglu`, `rope`, `fused_linear_cross_entropy`, `cross_entropy`, `quantize_per_token`, `dequantize_per_channel` |
| **OT4** 注意力 | 5 | `flash_attention`, `grouped_query_attention`, `multi_latent_attention`, `cross_attention`, `paged_attention` |
| **合计** | **45** | |

---

## 2. Gate 总览表

| Gate | BL 出口 | L 层 | 核心目标 | 关键数据 | 状态 |
|:----:|:--------|:----:|:--------|:---------|:----:|
| **G0** | — | — | GPU 环境验证 | RTX 3060 6GB, CUDA 12.4, PyTorch 2.6.0, Triton 3.2.0 | ✅ |
| **G1** | — | — | IR + 验证系统 | 10 ops, 6 决策类型, 237 tests | ✅ |
| **G2** | BL1×L1 (matmul) | L1 | 手动策略→codegen→GPU | 105-160% P0 | ✅ |
| **G3** | BL1×L1 (matmul+softmax) | L1 | LLM Agent 闭环 | 106.1% P0, 23 tools, 13 decisions | ✅ |
| **G4** | BL2×L1 (6 tasks) | L1 | Arke vs LLM-direct | geomean=0.991, 正确率 100% vs 83% | ✅ |
| **G5** | BL3×L1 + BL6/GPT-2×L3 | L1+L3 | 全基础算子+E2E 正确性 | 延迟 known-fail 1.71-2.20× | ✅ |
| **G6** | BL5×L1+L2 | L1+L2 | **Lang & IR 完备性** | 45 ops 全形状 | ⬜ |
| **G7** | BL5 继承 + BL6×L3(LLaMA-2+DS-V2) | L1+L2+L3 | **Arke Autonomous Engineering** | 自主生成+≥3轮迭代+2模型E2E | ⬜ |
| **G8** | BL6×L3(4模型) + BL5 回归 | L1+L2+L3 | Phase 1 最终验收 | 4 模型, Arke vs LLM-direct 对比 | ⬜ |

---

## 3. Part I — G0-G4（已通过 Gate）

### G0 — GPU 环境验证 ✅

**核心目标：** 建立可重现的 GPU 开发环境。

#### 出口条件

| # | 条件 | 验证方式 | 结果 |
|---|:-----|:---------|:----:|
| G0.1 | `make setup` 无报错 | Fresh clone → `make setup` | ✅ |
| G0.2 | PyTorch 检测到 CUDA GPU | `torch.cuda.is_available() == True` | ✅ |
| G0.3 | Triton 编译并运行 matmul | GPU smoke test 退出码 0 | ✅ |
| G0.4 | `pytest tests/ -q` 无 import 错误 | 全部 tests 收集成功 | ✅ |

**硬件记录：** RTX 3060 Laptop 6GB (Ampere, SM 8.6) · CUDA 12.4 · PyTorch 2.6.0+cu124 · Triton 3.2.0

**BL 等价：** 前置条件，非 BL 层级。H1/H2/H3/H4 验证所需的硬件基础。

---

### G1 — IR + 验证系统 ✅

**核心目标：** Semantic IR 和 Strategy IR 覆盖 ≥10 算子，含静态和数值验证。

#### 出口条件

| # | 条件 | 验证方式 | 结果 |
|---|:-----|:---------|:----:|
| G1.1 | Semantic IR 支持 ≥10 算子 | `len(OP_CATALOG) >= 10` | ✅ |
| G1.2 | Strategy IR 支持 ≥6 决策类型 | `{tile,fuse,place,parallel,reorder,algorithm}` 全覆盖 | ✅ |
| G1.3 | JSON Schema IR round-trip | `jsonschema.validate(ir.to_json(), schema)` 通过 | ✅ |
| G1.4 | V0 静态验证 <1ms | 验证器延迟 <1ms | ✅ |
| G1.5 | V1 数值验证（NumPy 参考） | matmul + softmax 3随机种子均通过 | ✅ |
| G1.6 | 全 10 ops 形状推断 | `infer_shapes()` 返回正确形状 | ✅ |
| G1.7 | ≥100 unit tests 通过 | pytest count ≥ 100 | ✅ (237) |

**BL 等价：** IR 基础设施（前置条件，非 BL 层级）。BL1+ 的能力前提。

**验证的核心假设：** H3（可解释性）—— trajectory JSONL 完整记录所有决策和理由。

---

### G2 — 手动策略 → Codegen → GPU ✅

**核心目标：** 人工编写策略块 → Triton codegen → GPU 执行 ≥70% cuBLAS。

#### 出口条件

| # | 条件 | 验证 | 结果 |
|---|:-----|:-----|:----:|
| G2.1 | matmul Triton codegen 正确 | Generated kernel 通过 V1 数值验证 | ✅ |
| G2.2 | softmax Triton codegen 正确 | 同上 | ✅ |
| G2.3 | fused matmul+relu codegen 正确 | Fused kernel 通过数值验证 | ✅ |
| G2.4 | GPU 执行 **≥70% cuBLAS** | `compile_and_profile()` → `vs_baseline >= 0.7` | ✅ (105-160%) |
| G2.5 | Pipeline 全链路打通 | IR → strategy → codegen → compile → profile 一次调用完成 | ✅ |
| G2.6 | ≥9 GPU 集成测试 | GPU tests 在 `ARKE_GPU_TESTS=1` 下通过 | ✅ |

**BL 等价：** BL1×L1(matmul only)。验证 H1（结构化协议提升正确性）。

---

### G3 — LLM Agent 闭环 ✅

**核心目标：** LLM 通过工具调用自主完成优化循环，零人工干预。

#### 出口条件

| # | 条件 | 结果 |
|---|:-----|:----:|
| G3.1 | LLM 使用 ≥8 种不同工具 | ✅ (全部 10 种) |
| G3.2 | LLM 应用 ≥4 个策略决策 | ✅ (13 次) |
| G3.3 | LLM 调用 verify_correctness | ✅ |
| G3.4 | LLM 调用 compile_and_profile | ✅ (5 次) |
| G3.5 | LLM 使用 checkpoint + rollback | ✅ |
| G3.6 | Fallback 机制有效 | ✅ |
| G3.7 | 多 provider 支持 | ✅ (Anthropic + OpenAI-compatible) |
| G3.8 | 零人工干预 | ✅ |

**关键数据：** matmul 2048² → 151.4% cuBLAS；lm-head (50257) → 116.5% cuBLAS；工具调用 23 次，策略决策 13 次。

**BL 等价：** BL1×L1(LLM-driven)。验证 H2（结构化搜索优于人工）。

---

### G4 — Arke vs LLM-direct ✅

**核心目标：** 量化证明 Arke 在正确性、一致性、性能上优于 LLM 直接写 Triton。

#### 出口条件

| # | 条件 | 结果 |
|---|:-----|:----:|
| G4.1 | ≥5 benchmark tasks 定义 | ✅ (6 tasks) |
| G4.2 | Arke 完成全部 tasks | ✅ 6/6 |
| G4.3 | LLM-direct 完成全部 tasks | ✅ 6/6 |
| G4.4 | Arke 正确率 ≥ LLM-direct | ✅ 100% ≥ 83% |
| G4.5 | Arke 均值性能 ≥ LLM-direct | ⚠️ 115.7% < 118.3%（融合任务差距，方差显著更小）|
| G4.6 | Arke 方差 ≤ LLM-direct | ✅（LLM-direct 失败率高，方差大）|
| G4.7 | 评估报告生成 | ✅ `benchmarks/results/phase1/EVALUATION_REPORT.md` |
| G4.8 | Token 效率对比 | ✅ Arke ≤60% LLM-direct token 消耗 |

**关键数据：** Arke/FlagGems geomean=0.991；正确率 100% vs 83%；Token 效率 ≤0.7×。

**BL 等价：** BL2×L1(6 tasks)。验证 H1+H2。

**Gate 决策：** Proceed — Arke 在可靠性（100% vs 83%）和 token 效率上胜出；性能略低但方差显著更小。

---

## 4. Part II — G5（已通过，标准回顾性重写）

### G5 — BL3×L1 + BL6/GPT-2×L3 ✅

**核心目标：** 全基础算子（OT0-2）非对齐形状正确性 + GPT-2 Small E2E 推理验证。

#### 出口标准（BL 体系重写）

```bash
arke bench --bl 3 --ot 0-2          # OT0-2 × ST1-3，基础算子全形状
arke bench --bl 6 --model gpt2      # GPT-2 Small E2E
```

**L1 @ BL3（OT0-2, ST1-3）**

| 维度 | 要求 | 实际结果 |
|:-----|:-----|:---------|
| 正确性 | OT0-2 × ST1-3 100% | ✅ 全通过 |
| 性能 | geomean(OT0-2, ST1-3) ≥ P3 (torch eager) | ✅ 已达到 |
| 说明 | P0/P1 性能目标延后至 G6（受 dispatch 架构限制）| — |

**L3 @ BL6 / GPT-2 Small**

| 维度 | 要求 | 实际结果 |
|:-----|:-----|:---------|
| 正确性 | top-1 token 全 seq_len 匹配 eager | ✅ |
| 覆盖率 | ≥48 Conv1D 替换 | ✅ 49/48 |
| 内存 | ≤ 6GB VRAM | ✅ 1100MB/6144MB |
| 延迟 | ≤ 1.15× eager | ⚠️ 1.71-2.20× **known-fail** |

#### G5 Known-Fail 分析（记录，非阻塞）

| 现象 | 根因 | 解决时机 |
|:-----|:-----|:---------|
| E2E 延迟 1.7-2.3× eager | monkey-patch dispatch ~60µs/call × 49 次累积 | G7: torch.compile Inductor backend |
| 单 matmul: Arke 76µs vs cuBLAS 44µs | L1 单算子OK，Python dispatch overhead 累积 | G6: BL5 统一度量后对比 |

> 详细分析报告: `benchmarks/results/phase1/gates/G5/REPORT.md`

**BL 等价：** BL3×L1（OT0-2, 33 ops 正确性）+ BL6/GPT-2×L3（E2E 正确性）。

**验证的核心假设：** H1 正确性验证（完整 E2E 管线正确）。

---

## 5. G6 — BL5×L1+L2：Lang & IR 完备性（当前目标）

> **核心目标：** 验证 Arke Lang 和 Arke IR 对全部 45 个算子的所有形状（含 ST3 非对齐 + ST4 生产规模）
> 均具备完整的表达能力、代码生成能力和性能竞争力。
> 这是 Arke 从"能跑"到"能用"的分水岭。**G6 不含 L3/BL6（模型 E2E），那是 G7 的职责。**

### 出口标准（BL5×L1+L2）

```bash
arke bench --bl 5 --layer l1    # OT0-4 × ST1-4，单算子全形状性能
arke bench --bl 5 --layer l2    # 融合算子全形状性能
```

#### L1 @ BL5（OT0-4, ST1-4）

| 算子组 | 正确性要求 | 性能要求 | 测量命令 |
|:-------|:----------|:---------|:---------|
| **OT0** 元素级 | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.90 P1 (FlagGems elem) | `bench_l1 --ot 0` |
| **OT1** 规约 | 100%(ST1-3) + ≥95%(ST4) | geomean ≥ 0.85 P1 (FlagGems norm/softmax) | `bench_l1 --ot 1` |
| **OT2** 计算密集 | 100%(ST1-3) + ≥95%(ST4) | matmul geomean ≥ 0.90 P0; 其余 ≥ P3 | `bench_l1 --ot 2` |
| **OT3** 门控激活 | 100%(ST1-3) + ≥95%(ST4) | swiglu/rope geomean ≥ 0.85 P1 (Liger/FlagGems) | `bench_l1 --ot 3` |
| **OT4** 注意力 | 100%(ST1-4，OOM除外) | FA geomean ≥ 0.80 P1 (FlashAttn-2); GQA ≥ 0.80 | `bench_l1 --ot 4` |

> **ST4 OOM 说明：** OT4 在 6GB VRAM 下部分大形状可能 OOM，标注 `⚠️ OOM` 后跳过，不计入正确性 pass rate 分母。

#### L2 @ BL5（融合算子）

| 融合组合 | 要求 | 基准 |
|:---------|:-----|:-----|
| matmul+relu, matmul+gelu | ≥ 1.05× unfused（融合收益可验证）| P3 unfused |
| swiglu, geglu | ≥ 0.90× Liger | P1 |
| linear+cross_entropy | ≥ 1.05× unfused | P3 |
| QKV+flash_attention | ≥ 0.80× FlashAttn-2 | P1 |

#### Lang & IR 完备性附加条件（G6-LI）

| ID | 条件 | 验证方式 |
|:---|:-----|:---------|
| **G6-LI.1** | 全部 45 ops 可在 `.ak` 中表达并解析 | `arke parse examples/<op>.ak` 全部 exit 0 |
| **G6-LI.2** | `.ak → SemanticIR → StrategyIR` 全链路 | `ast_to_strategy()` 对所有示例文件通过 round-trip |
| **G6-LI.3** | `@rationale` 注解全链路保留 | `.ak @rationale → StrategyIR → codegen 注释 → trajectory/log` ≥3 示例验证 |
| **G6-LI.4** | Token 效率：`.ak` ≤ 等效 Triton 行数 | OT0-OT4 benchmark：`.ak` lines < Triton lines @ 同等性能 |
| **G6-LI.5** | Python interop IR round-trip | `from_json/to_json/from_dict/to_dict` 全 45 ops 通过 |
| **G6-LI.6** | Grammar 完备：全部 `.ak` 文件解析 0 失败 | 支持 array literals、float constants、4D tensor、all op param types |

#### G6 PASS 综合条件

```
AND ALL:
  [1] L1 BL5 正确性: 100%(ST1-3) + ≥95%(ST4，OOM 除外) 对全部 OT0-OT4
  [2] L1 BL5 性能 weighted_score ≥ 0.83
        weighted_score = 0.25×score(OT0-1) + 0.30×score(OT2) + 0.20×score(OT3) + 0.25×score(OT4)
        其中 score(OTn) = 该 OT 组 geomean 达标率（0.0~1.0）
  [3] L2 BL5: ≥3/4 融合组合达标
  [4] Lang&IR: G6-LI.1~LI.6 全部通过
```

### G6 能力反推

#### Arke Lang（.ak 语言层）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D6-L1 | `.ak` 4D tensor op 语法扩展 | OT4 必须可表达 | `.ak` 语法：4D tensor, einsum 标注 |
| D6-L2 | gather/scatter 语义节点 | OT2 新增 data-movement ops | gather/scatter 语义节点 |
| D6-L3 | quantize 原语语法 | OT3 量化算子 | `quantize`/`dequantize` 语法原语 |
| D6-L4 | paged KV / block_table 参数 stub | OT4 paged_attention | paged memory 语义标注（可延后 G7 实现） |
| D6-L5 | grammar fix：array literal + float constant | G1.4 遗留语法 gap | 修复 grammar，支持 `[2,3]`、`0.125` 等 |
| D6-L6 | 全 45 ops 的 `.ak` 示例文件 | G6-LI.1 完备性 | `examples/<op>.ak` × 45 |

#### Arke IR（Semantic IR + Strategy IR）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D6-IR1 | SemanticIR op catalog 扩展至 45 ops | BL5 全算子正确性 | OT3/OT4 全部 op 节点定义 |
| D6-IR2 | AttentionSemanticIR 字段 | GQA/MHA 正确性 | `mask_type`, `num_kv_heads`, `head_dim` |
| D6-IR3 | RopeSemanticIR 字段 | RoPE 正确性 | `theta`, `base`, `rotary_dim` |
| D6-IR4 | QuantizeSemanticIR 字段 | quantize 正确性 | `scale_dtype`, `group_size`, `zero_point` |
| D6-IR5 | `ast_to_strategy()` 转换器 | G6-LI.2：全链路打通 | parser AST → StrategyIR 转换器 |
| D6-IR6 | StrategyIR JSON round-trip（全 45 ops） | G6-LI.5 Python interop | `from_json/to_json` 全量测试 |
| D6-IR7 | MLA 特有字段 | BL5 OT4 MLA 正确性 | `latent_dim`, `kv_lora_rank` 字段 |
| D6-IR8 | PaddingStrategy 决策类型 | ST3 非对齐性能 | `pad_to_multiple`, `dynamic_padding` 决策 |

#### Arke LLM Agent

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D6-A1 | attention prompt template | OT4 ≥0.80 P1 | causal mask、GQA group 展开策略模板 |
| D6-A2 | rope prompt + rationale template | OT3 rope ≥ P3 | RoPE 向量化 cos/sin、half-rotate 策略 |
| D6-A3 | fusion opportunity detection | L2 融合收益 | agent tools 中检测 fuse 机会 |
| D6-A4 | quantize/dequantize prompt template | OT3 quant ≥ P3 | per-token scale、向量化量化策略 |
| D6-A5 | batch optimize pipeline（45 ops 并行 session） | BL5 不能逐个人工 | op_list × shape_list 并行 agent session |
| D6-A6 | non-aligned shape rationale template | ST3 性能目标 | padding vs masking 权衡 rationale |

#### Arke 工程（基础设施）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D6-E1 | Triton 模板 10 类 | BL5 correctness 100% | `rope`, `flash_attention`, `GQA`, `MLA`, `cross_attention`, `paged_attention`, `gather`, `scatter`, `embedding`, `quantize` |
| D6-E2 | `bench_l1` 路由扩展（45 ops + shape_registry） | BL5 可执行 | `bench_l1.py` 路由扩展 + shape_registry 接入 |
| D6-E3 | `bench_l2` OT3/OT4 融合 benchmark runner | L2 BL5 可执行 | bench_l2.py OT3/OT4 融合 benchmark |
| D6-E4 | Baseline 适配 | OT4 P1 基准 | FlashAttn-2, Liger rope/quant, FlagGems GQA |
| D6-E5 | CSV 输出按 BL/OT/L 层级组织 | benchmark 结果可追溯 | `L1/OT{n}/perf_{op}.csv` 输出目录 |
| D6-E6 | V1 validator 扩展（全 45 ops） | correctness 100% | attention 数值容差、量化精度标准 |

### G6 关键路径

**瓶颈：** D6-E1（10 类 Triton 模板）是 G6 最大工作量；`flash_attention` 和 `MLA` 模板最复杂。

```
D6-L5(grammar fix) + D6-IR1(catalog) → D6-IR5(ast_to_strategy) → G6-LI.1/2
D6-E1(Triton 10类模板) + D6-E6(validator) → [1] 正确性
D6-E2/E3/E4(bench runner) + D6-A1/A2/A5 → [2][3] 性能
ALL → G6 PASS
```

---

## 6. G7 — Arke Autonomous Engineering

> **核心目标：** 验证 Arke 的**自主工程能力（Autonomous Engineering Capability）**。
> G7 不仅是"增加两个模型的 E2E 测试"——它验证 Arke Agent 能否在**无人工介入**的情况下，
> 仅凭 kernel 语义描述（`.ak` 或自然语言），自动生成策略、执行 codegen、迭代优化，
> 并最终为真实 LLM 生成完整 kernel 集。
>
> **LLaMA-2 7B 和 DeepSeek-V2 16B 是验证载体**，而非核心目标本身。
> 核心要验证的是：Arke Agent 是否已具备生产级自主工程能力。

### G7 的四个核心验证维度

1. **自主 kernel 生成** — LLM Agent 不依赖人工 strategy block，仅凭 kernel 语义描述自动生成策略
2. **迭代优化闭环** — LLM 自动执行 ≥3 轮 `compile → profile → adjust` 循环
3. **多输入形式支持** — `.ak` 文件 / 自然语言描述 / 现有代码片段 → 自动路由到 Arke 管线
4. **E2E 模型验证** — LLaMA-2 7B + DS-V2 16B 证明 Arke 能自主为真实模型生成完整 kernel 集

同时，G7 解决 G5 known-fail：通过 `torch.compile` Inductor backend 将 GPT-2 E2E 延迟降至 ≤1.30×。

### 出口标准

```bash
arke bench --bl 5 --layer l1 l2   # 继承 G6，BL5 全算子不退步
arke bench --bl 6 --model llama2  # LLaMA-2 7B E2E（验证自主生成）
arke bench --bl 6 --model deepseek # DeepSeek-V2 16B E2E（验证自主生成）
```

#### 自主工程能力验证条件（G7-AE，核心）

| ID | 条件 | 验证方式 |
|:---|:-----|:---------|
| **G7-AE.1** | LLM 自动生成策略（无人工 strategy block） | kernel-only `.ak`（无 strategy 块）→ LLM 生成 strategy → codegen → ≥80% cuBLAS |
| **G7-AE.2** | 迭代优化闭环 ≥3 轮 | trajectory JSONL 含 ≥3 个完整 `compile→profile→adjust` 循环 |
| **G7-AE.3** | 多输入类型支持 | (a) `.ak` 文件 (b) 自然语言描述 (c) 现有代码片段 → 各类型 ≥2 个算子全流程验证 |
| **G7-AE.4** | `arke optimize <input>` 统一入口 | CLI 单命令完成 input → LLM optimize → Triton → GPU → benchmark report |
| **G7-AE.5** | E2E profile → kernel feedback loop | 瓶颈算子识别 → 重新优化 → 延迟改善可验证（轨迹记录） |

#### BL5 继承条件（不退步）

```bash
arke bench --bl 5 --layer l1    # ≥ G6 水准（正确性+性能不退步）
arke bench --bl 5 --layer l2    # ≥ G6 水准
```

| 维度 | 要求 |
|:-----|:-----|
| L1 BL5 正确性 | ≥ G6 结果（无退步） |
| L1 BL5 性能 geomean | ≥ G6 结果（无退步）|
| L2 BL5 融合覆盖 | ≥ G6 融合组合数 |

#### L3 @ BL6（LLaMA-2 7B + DeepSeek-V2 16B）—— 自主生成验证载体

| 模型 | 正确性 | 性能阈值 | 内存 | seq 覆盖 |
|:-----|:-------|:---------|:-----|:---------|
| **LLaMA-2 7B** | top-1 token 100% 匹配 eager（所有 seq_len） | Arke ≤ **1.30×** eager（torch.compile backend）| ≤ 6GB | 512/2048/4096 |
| **DeepSeek-V2 16B** | top-1 token 100% 匹配 eager（seq∈{512,2048}） | Arke ≤ **1.40×** eager（MoE dispatch 开销）| ≤ 6GB（seq≤512，量化）| 512/2048 |

> **GPT-2 E2E 修复：** torch.compile backend 上线后，GPT-2 延迟应同步降至 ≤1.20×，修复 G5 known-fail。

#### G7 PASS 综合条件

```
AND ALL:
  [1] 自主工程: G7-AE.1~AE.5 全部通过
  [2] BL5 继承: L1+L2 BL5 正确性和性能均不低于 G6 结果
  [3] L3 BL6 LLaMA-2: 正确性 100% + 延迟 ≤1.30× eager
  [4] L3 BL6 DS-V2: 正确性 100% + 延迟 ≤1.40× eager
  [5] torch.compile Inductor backend: GPT-2 延迟降至 ≤1.20×（G5 known-fail 修复）
```

### G7 能力反推

#### ★ Arke LLM Agent（G7 最大开发项分组）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D7-A1 | 自动策略生成（kernel-only .ak → LLM 生成 strategy） | G7-AE.1 | kernel-only `.ak` 输入 → LLM 完整策略生成 pipeline |
| D7-A2 | 迭代优化循环（compile→profile→adjust 自动触发） | G7-AE.2 | 自动触发 ≥3 轮迭代（基于性能 delta 阈值）|
| D7-A3 | 多输入类型路由（arke optimize 统一入口） | G7-AE.3/4 | `.ak` / 自然语言 / 现有代码 → 统一解析 → Arke pipeline |
| D7-A4 | E2E profile → kernel feedback loop | G7-AE.5 | 瓶颈算子识别（基于 BL6 profile）→ 重新触发优化 |
| D7-A5 | batch optimize pipeline（全模型算子集批量优化） | L3 BL6 自主生成 | 模型 forward graph → 算子列表 → 批量 agent session |
| D7-A6 | long-context agent prompt（seq>4K 分支策略） | L3 BL6 LLaMA-2/DS-V2 延迟阈值 | chunk prefill、KV cache split 策略模板 |
| D7-A7 | MoE-aware optimization prompt | DS-V2 E2E（grouped_matmul+gather+scatter）| top-k sparsity、load balance 策略模板 |
| D7-A8 | INT8/量化推理 agent prompt | L2 BL5 quant+matmul 融合 | W4A8、W8A8 量化路径策略模板 |
| D7-A9 | @rationale 知识库积累（≥30 条 G7 条目） | H3 可解释性 | trajectory → rationale_kb.jsonl 蒸馏 |

#### Arke IR

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D7-IR1 | PipelineStageStrategy 决策类型 | OT4 ST4 性能（长 context 需 pipelining）| prefill/decode 分离策略 |
| D7-IR2 | MultiLatentAttentionIR 字段 | BL5 MLA 正确性 + 性能 | `kv_lora_rank`, `qk_rope_head_dim` |
| D7-IR3 | GroupedMatmulSemanticIR expert_indices | DS-V2 MoE 正确性 | expert routing 字段 |
| D7-IR4 | PaddingStrategy（非对齐形状） | ST3 性能（继承 G6 D6-IR8）| `pad_to_multiple`, `dynamic_padding` |

#### Arke Lang（.ak 语言层）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D7-L1 | `.ak` @context_len 标注原语 | ST4 OT4 + L3 BL6 | long-context 语义标注 |
| D7-L2 | paged memory 语义节点（block_table, page_size） | BL5 OT4 paged_attention | paged KV cache 完整语义 |
| D7-L3 | moe_dispatch / moe_combine 高级原语 | DS-V2 E2E | MoE dispatch/combine 语法糖 |
| D7-L4 | MLA 参数语义节点 | BL5 OT4 MLA | `latent_dim`, `kv_lora_rank` 标注 |
| D7-L5 | @dtype int8/fp8 标注扩展 | BL5 OT3 quant × ST4 | int8/fp8 dtype 原语扩展 |

#### Arke 工程（基础设施）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D7-E1 | torch.compile Inductor backend | G7-AE 延迟阈值 + G5 known-fail 修复 | `arke/integration/torch_compile_backend.py`（Inductor custom op 注册）|
| D7-E2 | LLaMA-2 7B 集成 + bench_l3 runner | L3 BL6 LLaMA-2 | `examples/llama2_arke.py` + L3 bench runner |
| D7-E3 | DeepSeek-V2 集成 + bench_l3 runner | L3 BL6 DS-V2 | `examples/deepseek_v2_arke.py`（量化权重，seq≤512）|
| D7-E4 | Triton MLA template（compressed KV, lora project）| BL5 OT4 MLA 正确性 | lora-style project + compressed KV |
| D7-E5 | Triton paged_attention template | BL5 OT4 paged_attention | block table scatter read |
| D7-E6 | bench runner OOM guard + CSV 标注 | BL5 ST4 OOM 处理 | try/catch + `⚠️ OOM` 写入 CSV |
| D7-E7 | `bench_l3.py` — 模型 forward pass 多次测量 | L3 自动化 | top-1 token 比对 + latency 统计 |

### G7 关键路径

**最长依赖链：**
```
D7-A1(自主策略生成) ─────────────────────┐
D7-A2(迭代优化循环) ─────────────────────┤→ G7-AE.1~5
D7-A3(多输入路由) + D7-E1(compile backend)┘
D7-E1(torch.compile backend) → D7-E2(LLaMA-2) → D7-E7(bench_l3) → [3][5]
D7-E3(DS-V2) → [4]
```

**瓶颈：** D7-A1（自主策略生成）是 G7 最大不确定性；D7-E1（torch.compile backend）是 L3 性能的架构前提。

---

## 7. G8 — Phase 1 最终验收

> **核心目标：** Phase 1 最终 Gate。Arke 能自主为 4 个真实 LLM 生成完整 kernel 集，
> 端到端性能全部满足生产可用阈值，并量化验证 Arke vs LLM-direct 的优势。
> 同时完成语言实现评估（Python vs 混合方案）并为 Phase 2 奠定基础。

### 出口标准

```bash
arke bench --bl 6 --model gpt2      # GPT-2 Small
arke bench --bl 6 --model llama2    # LLaMA-2 7B
arke bench --bl 6 --model llama3    # LLaMA-3 8B
arke bench --bl 6 --model qwen25    # Qwen2.5 7B
arke bench --bl 5 --layer l1 l2     # BL5 回归（不退步）
```

#### L3 @ BL6（4 模型）

| 模型 | 正确性 | 性能阈值 | 内存 | seq 覆盖 |
|:-----|:-------|:---------|:-----|:---------|
| **GPT-2 Small** | top-1 100% | Arke ≤ **1.15×** eager（G5 known-fail 彻底修复）| ≤ 4GB | 128/512/1024 |
| **LLaMA-2 7B** | top-1 100% | Arke ≤ **1.20×** eager | ≤ 6GB | 512/2048/4096 |
| **LLaMA-3 8B** | top-1 100% | Arke ≤ **1.20×** eager | ≤ 6GB | 512/2048/8192 |
| **Qwen2.5 7B** | top-1 100% | Arke ≤ **1.25×** eager（GQA 7:1 + 极宽 FFN）| ≤ 6GB | 512/2048 |

#### BL5 完整回归（继承 G7 结果，必须不退步）

| 维度 | 要求 |
|:-----|:-----|
| L1 BL5 全 45 ops 正确性 | ≥ G7 水准，无退步 |
| L1 BL5 OT0-4 性能 geomean | ≥ G7 结果（允许 ±1% 噪声）|
| L2 BL5 融合算子覆盖 | ≥ G7 覆盖 |

#### Arke vs LLM-direct 对比（G8 新增）

| 指标 | Arke 目标 | LLM-direct 基准 | 依据 |
|:-----|:---------|:----------------|:-----|
| 正确性 | ≥ 98%（G8 全算子）| 历史值 ~83% | G4 数据 |
| 性能 geomean（BL5 L1） | ≥ 1.05× LLM-direct | — | Arke 结构化搜索优势 |
| 性能方差（stddev） | ≤ 0.5× LLM-direct | — | 确定性 IR 约束降方差 |
| Token 消耗/kernel | ≤ 0.7× LLM-direct | — | IR 约束减少探索 token |

#### 语言实现评估（G8 同步完成）

```
G8-Lang: Python vs 混合方案数据驱动评估
  测量项: dispatch overhead（Python 路径 vs Rust/C++ 理论）
          parse latency（.ak → IR），内存占用，LLM API 集成成本
  产出: docs/phase1/language-decision.md（结论 + 数据 + Phase 2 迁移策略）
```

#### G8 PASS 综合条件

```
AND ALL:
  [1] 4 模型 L3 BL6 正确性 100%
  [2] 4 模型 E2E 延迟全部 ≤ 阈值（GPT-2≤1.15×, LLaMA-2/3≤1.20×, Qwen2.5≤1.25×）
  [3] Arke vs LLM-direct: 正确性 ≥1.15×，性能 geomean ≥1.05×，token ≤0.7×
  [4] BL5 L1 全 45 ops 无性能退步
  [5] language-decision.md 完成
```

### G8 能力反推

#### Arke Lang（.ak 语言层）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D8-L1 | Qwen2.5 7B `.ak` 示例 | BL6 Qwen2.5 L3 | `qwen25_forward.ak`（GQA+SwiGLU+RMSNorm 完整描述）|
| D8-L2 | LLaMA-3 8B `.ak` 示例 | BL6 LLaMA-3 L3 | `llama3_forward.ak`（GQA, rope, RMSNorm）|
| D8-L3 | Arke I/O Spec 文档 | G7-AE.3 多输入类型 | `docs/spec/arke-io-spec.md` |
| D8-L4 | 语言规范 v1.0 冻结 | Phase 1 完成标志 | `arke-lang-spec.md` 更新 + 标签 v1.0 |

#### Arke IR

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D8-IR1 | IR 规范 v1.0 冻结 | Phase 1 完成标志 | `arke-ir-spec.md` 更新 + 标签 v1.0 |
| D8-IR2 | IR ↔ MLIR 映射文档 | Phase 2 准备 | `docs/spec/ir-mlir-mapping.md` |
| D8-IR3 | 完整 round-trip 验证（全 45 ops × JSON）| IR Spec v1.0 | `test_ir_roundtrip.py` |

#### Arke LLM Agent

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D8-A1 | 自主 I/O 合约（3 种输入 → Arke pipeline）| G7-AE.3/4 继承 | `arke optimize <input>` 完整支持 |
| D8-A2 | LLM auto-strategy 成熟度验证 | G8 BL5 不退步 | 全 45 ops 批量 agent 验证（无人工 strategy）|
| D8-A3 | ≥3 轮迭代优化（成熟度）| G7-AE.2 继承 | iterative loop 在 4 模型上稳定运行 |
| D8-A4 | @rationale 知识库（≥50 条 Phase 1 条目）| H3 可解释性 | `trajectory → rationale_kb.jsonl` 蒸馏 |
| D8-A5 | Arke vs LLM-direct 自动对比 | G8 对比指标 | `benchmarks/compare_arke_vs_direct.py` |

#### Arke 工程（基础设施）

| ID | 能力需求 | 依据 | 开发项 |
|:---|:---------|:-----|:-------|
| D8-E1 | LLaMA-3 8B 集成 + bench_l3 | BL6 LLaMA-3 L3 | `examples/llama3_arke.py` + bench_l3 runner |
| D8-E2 | Qwen2.5 7B 集成 + bench_l3 | BL6 Qwen2.5 L3 | `examples/qwen25_arke.py` + bench_l3 runner |
| D8-E3 | GPT-2 ≤1.15× eager（彻底修复）| G5 known-fail 最终解决 | torch.compile backend E2E 集成（依赖 D7-E1）|
| D8-E4 | BL5 全 45 ops 回归套件（CI）| G8 BL5 不退步 | `ci/regression_bl5.py`（每次 commit 跑 BL5 L1 正确性）|
| D8-E5 | 语言评估 benchmark | language-decision.md | dispatch overhead 测量脚本 + memory profiler |
| D8-E6 | Phase 1 最终评估报告 | Phase 1 完成标志 | `benchmarks/results/phase1/STAGE1_FINAL_REPORT.md` |

---

## 8. Gate 依赖链

```
G0(环境) → G1(IR) → G2(Codegen) → G3(Agent) → G4(对比) → G5(BL3+GPT-2)
                                                                  │
                                              G6(BL5 Lang/IR 完备性，当前目标)
                                                                  │
                                         G7(Arke Autonomous Engineering
                                              BL5继承 + LLaMA-2/DS-V2 E2E)
                                                                  │
                                              G8(BL6×4模型, Phase 1 Final)
                                                                  │
                                                         Phase 2 (Ascend)
```

### 各 Gate 关键阻塞项

| Gate | 关键阻塞项（P0）| 估时 |
|:-----|:---------------|:-----|
| **G6** | D6-E1（Triton 10类模板）; D6-IR1（catalog 45 ops）; D6-L5（grammar fix）| XL+M+S |
| **G7** | D7-A1（自主策略生成）; D7-E1（torch.compile backend）; D7-E2（LLaMA-2 集成）| XL+XL+L |
| **G8** | D8-E1/E2（LLaMA-3/Qwen2.5 集成）; D8-E3（GPT-2 fix 依赖 D7-E1）| L+L+M |

### 开发路径关键链

```
D6-IR1 → D6-E1(10类模板) → D6-E2(bench_l1) ─────────────── G6
                                    │
                    D7-A1(自主策略) + D7-A2(迭代循环) ───────┐
                    D7-E1(torch.compile backend) ────────────┤→ G7
                    D7-E2(LLaMA-2) + D7-E7(bench_l3) ───────┘
                                    │
                    D8-E1(LLaMA-3) + D8-E2(Qwen2.5) ────── G8
```

---

## 9. 开发项附录

> **估时说明：** S≤1d, M≤3d, L≤1w, XL>1w（单人参考值）

### G6 开发项（26 项）

| ID | 层 | 描述 | 优先级 | 估时 |
|:---|:---|:-----|:------:|:----:|
| D6-L1 | Lang | `.ak` 4D tensor 语法扩展（4D tensor, einsum 标注）| P2 | M |
| D6-L2 | Lang | gather/scatter 语义节点 | P2 | S |
| D6-L3 | Lang | quantize 原语语法 | P2 | S |
| D6-L4 | Lang | paged memory 语义标注（stub，可延后 G7）| P2 | S |
| D6-L5 | Lang | grammar fix（array literal, float constant）| **P0** | S |
| D6-L6 | Lang | 全 45 ops 的 `.ak` 示例文件 | P1 | L |
| D6-IR1 | IR | SemanticIR op catalog → 45 ops（OT3/OT4 全部字段）| **P0** | M |
| D6-IR2 | IR | AttentionSemanticIR（mask_type, num_kv_heads, head_dim）| **P0** | S |
| D6-IR3 | IR | RopeSemanticIR（theta, base, rotary_dim）| **P0** | S |
| D6-IR4 | IR | QuantizeSemanticIR（scale_dtype, group_size, zero_point）| **P0** | S |
| D6-IR5 | IR | `ast_to_strategy()` 转换器 | **P0** | M |
| D6-IR6 | IR | StrategyIR JSON round-trip（全 45 ops）| P1 | S |
| D6-IR7 | IR | MLA 特有字段（latent_dim, kv_lora_rank）| P1 | S |
| D6-IR8 | IR | PaddingStrategy 决策类型（pad_to_multiple, dynamic_padding）| P2 | S |
| D6-A1 | Agent | attention prompt template（causal mask, GQA group 展开）| P1 | M |
| D6-A2 | Agent | rope prompt + rationale template | P1 | S |
| D6-A3 | Agent | fusion opportunity detection | P1 | M |
| D6-A4 | Agent | quantize/dequantize prompt template | P2 | S |
| D6-A5 | Agent | batch optimize pipeline（45 ops 并行 session）| P1 | L |
| D6-A6 | Agent | non-aligned shape rationale template | P2 | S |
| D6-E1 | Eng | Triton 模板 10 类：rope/FA/GQA/MLA/cross_attn/paged_attn/gather/scatter/embedding/quantize | **P0** | XL |
| D6-E2 | Eng | bench_l1 路由扩展（45 ops + shape_registry 接入）| P1 | M |
| D6-E3 | Eng | bench_l2 OT3/OT4 融合 benchmark runner | P1 | M |
| D6-E4 | Eng | baseline 适配（FlashAttn-2, Liger rope/quant, FlagGems GQA）| P1 | M |
| D6-E5 | Eng | CSV 输出目录 L1/OT{n}/perf_{op}.csv | P2 | S |
| D6-E6 | Eng | V1 validator 扩展（attention 数值容差, 量化精度标准）| **P0** | S |

### G7 开发项（23 项）

| ID | 层 | 描述 | 优先级 | 估时 |
|:---|:---|:-----|:------:|:----:|
| D7-A1 | Agent | 自动策略生成（kernel-only .ak → LLM 完整策略 pipeline）| **P0** | XL |
| D7-A2 | Agent | 迭代优化循环（自动触发 ≥3 轮 compile→profile→adjust）| **P0** | L |
| D7-A3 | Agent | 多输入类型路由（.ak / 自然语言 / 现有代码 → Arke pipeline）| **P0** | L |
| D7-A4 | Agent | E2E profile → kernel feedback loop（瓶颈算子识别→重新优化）| P1 | L |
| D7-A5 | Agent | batch optimize pipeline（全模型算子集批量优化）| P1 | M |
| D7-A6 | Agent | long-context agent prompt（seq>4K 分支策略）| P1 | M |
| D7-A7 | Agent | MoE-aware optimization prompt（top-k sparsity, load balance）| P1 | M |
| D7-A8 | Agent | 量化推理 agent prompt（W4A8, W8A8 策略）| P2 | M |
| D7-A9 | Agent | @rationale 知识库积累（≥30 条 G7 条目）| P2 | M |
| D7-IR1 | IR | PipelineStageStrategy（prefill/decode 分离）| P1 | M |
| D7-IR2 | IR | MultiLatentAttentionIR（kv_lora_rank, qk_rope_head_dim）| P1 | S |
| D7-IR3 | IR | GroupedMatmulSemanticIR expert_indices 字段 | P1 | S |
| D7-IR4 | IR | PaddingStrategy 完善（继承 D6-IR8）| P2 | S |
| D7-L1 | Lang | `.ak` @context_len 标注原语 | P2 | S |
| D7-L2 | Lang | paged memory 语义节点（block_table, page_size）| P1 | M |
| D7-L3 | Lang | moe_dispatch/combine 高级原语 | P2 | M |
| D7-L4 | Lang | MLA 参数语义节点 | P2 | S |
| D7-L5 | Lang | @dtype int8/fp8 标注扩展 | P2 | S |
| D7-E1 | Eng | torch.compile Inductor backend | **P0** | XL |
| D7-E2 | Eng | LLaMA-2 7B 集成 + bench_l3 runner | **P0** | L |
| D7-E3 | Eng | DeepSeek-V2 集成（seq≤512, 量化权重）| P2 | L |
| D7-E4 | Eng | Triton MLA template（compressed KV, lora project）| P1 | L |
| D7-E5 | Eng | Triton paged_attention template（block table scatter read）| P1 | L |
| D7-E6 | Eng | bench runner OOM guard + CSV 标注 | P2 | S |
| D7-E7 | Eng | bench_l3.py（模型 forward + top-1 比对 + latency 统计）| **P0** | M |

### G8 开发项（18 项）

| ID | 层 | 描述 | 优先级 | 估时 |
|:---|:---|:-----|:------:|:----:|
| D8-L1 | Lang | `qwen25_forward.ak` 示例（GQA+SwiGLU+RMSNorm）| **P0** | S |
| D8-L2 | Lang | `llama3_forward.ak` 示例（GQA, rope, RMSNorm）| **P0** | S |
| D8-L3 | Lang | `arke-io-spec.md`（I/O 合约文档）| P1 | M |
| D8-L4 | Lang | Language Spec v1.0 冻结（文档 + 标签）| P1 | M |
| D8-IR1 | IR | IR Spec v1.0 冻结（文档 + 标签）| P1 | M |
| D8-IR2 | IR | `ir-mlir-mapping.md`（Phase 2 准备）| P1 | M |
| D8-IR3 | IR | `test_ir_roundtrip.py`（全 45 ops × JSON round-trip）| P1 | S |
| D8-A1 | Agent | `arke optimize` 统一入口完整支持 3 种输入类型 | P1 | L |
| D8-A2 | Agent | LLM auto-strategy 成熟度验证（全 45 ops 无人工 strategy）| P1 | M |
| D8-A3 | Agent | iterative loop 在 4 模型上稳定运行验证 | P1 | M |
| D8-A4 | Agent | @rationale 知识库（≥50 条 Phase 1 条目）| P2 | M |
| D8-A5 | Agent | Arke vs LLM-direct 自动对比（benchmarks/compare_arke_vs_direct.py）| P1 | M |
| D8-E1 | Eng | LLaMA-3 8B 集成 + bench_l3 runner | **P0** | L |
| D8-E2 | Eng | Qwen2.5 7B 集成 + bench_l3 runner | **P0** | L |
| D8-E3 | Eng | GPT-2 torch.compile backend E2E（≤1.15× eager，依赖 D7-E1）| **P0** | M |
| D8-E4 | Eng | BL5 回归套件（CI）：`ci/regression_bl5.py` | P1 | M |
| D8-E5 | Eng | 语言评估 benchmark + `language-decision.md` | P1 | M |
| D8-E6 | Eng | Phase 1 最终评估报告 `STAGE1_FINAL_REPORT.md` | P1 | M |

---

## 10. 与 execution-plan.md 的对应关系

| execution-plan Phase | phase1-gate-design Gate | 关键差异 |
|:----------------|:------------------------|:---------|
| Phase 1.0（环境）| G0 | 无差异，直接对应 |
| Phase 1.1（IR+验证）| G1 | 无差异，G1.4 已升级为全量 `.ak` 文件解析（非 ≥3/5）|
| Phase 1.2（Codegen+E2E）| G2 | 无差异，BL 等价重新定义为 BL1×L1 |
| Phase 1.3（LLM Agent）| G3 | 无差异，BL 等价 BL1×L1(LLM-driven) |
| Phase 1.4（闭环优化）| G3/G4 | plan 合并在 Phase 1.4，gate 拆分为 G3（agent 闭环）+ G4（对比）|
| Phase 1.5（评估框架）| G4 | BL 等价明确为 BL2×L1(6 tasks)；geomean=0.991 记录 |
| Phase 1.6（.ak Parser）| G6-LI.1/2/6 | plan 独立为 Phase 1.6；在 gate 体系中合并入 G6 Lang&IR 完备性条件 |
| Phase 1.7（Whole-Model E2E）| G5 | 无差异；延迟 known-fail 明确记录根因和解决时机 |
| Phase 1.8（MVP v0.1.0）| — | MVP 发布不设独立 Gate；包含在 G5 之后 |
| Phase 1.9（G6: Lang&IR Completeness）| **G6** | **出口升级**：plan 为 G6.1-G6.9 定性条件；gate-design 改为 **BL5×L1+L2** 可量化验证；删除 G6.6 MLIR mapping（移至 G8 D8-IR2）；G6-LI 附加条件对应 G6.1/2/3/4/5/7 |
| Phase 1.10（G7: Autonomous Kernel Gen）| **G7** | **定位重构**：plan 定位为"I/O 合约验证"；gate-design 重定位为 **Arke Autonomous Engineering**（自主策略生成、迭代闭环、多输入路由为核心）；L3 BL6 两个模型为验证载体；Agent 开发项为最大分组 |
| Phase 1.11（G8: Language Assessment）| **G8** | **范围扩展**：plan 为单纯语言评估；gate-design 扩展为 Phase 1 最终验收（4模型+BL5回归+Arke对比+语言评估）；语言评估 language-decision.md 保留但并入综合验收 |

### G6 出口升级说明

`gate-redesign-v2.md` 中 G6 出口为 BL4×L1+L2（OT0-4 × ST1-2）。本文档将其升级为 **BL5×L1+L2**：

- **原因：** G6 是 Arke Lang & IR 完备性验证的关键 Gate，必须覆盖 ST3（非对齐）和 ST4（生产规模），否则无法真正验证语言和 IR 的表达能力边界
- **BL5 包含 ST3+ST4**，能发现非对齐形状的 padding 策略、长序列的 pipeline 策略等 IR 设计问题
- **G6 不含 L3/BL6**（模型 E2E），那是 G7 的职责；G6 专注语言和 IR 层面

---

*Last updated: 2026-04-05 | Author: Kitty (Arke Project)*
