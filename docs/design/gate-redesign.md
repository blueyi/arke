# Stage 1 Gate Redesign v2 — 泛化驱动 + Tier 3 全量验证

## 核心修正

v1 的错误：把 SMART 做成了"挑几个 shape 达标"，这是在做 cherry-pick 而非验证泛化。

**正确的 SMART 目标：**
- ❌ "matmul square-1k ≥ 80% cuBLAS" — 过关 ≠ 泛化
- ✅ "matmul 在 Tier 3 全量 50 shapes 上，≥ X% shapes 达到 Y% cuBLAS" — 分布+泛化

---

## Tier 体系（Shape 测试集）

### Tier 1: Smoke（15 shapes）
- 快速回归，CI 每次 commit 跑
- 覆盖 tiny/medium/large 各一个代表
- 用途：开发迭代中的快速检查

### Tier 2: Standard（31 shapes）
- 日常开发完成后跑
- 覆盖方阵/矩形/LLM-typical/边界
- 用途：PR 合并前的质量门

### Tier 3: Full（50 shapes）— Gate 出口标准
- 包含 Tier 1 + Tier 2 的全部 shapes
- **额外加入：**
  - warp-32 不对齐 shapes（如 M=127, K=257, N=513）
  - 极小 shapes（M=1, M=16）
  - 极大 shapes（8192³ if memory allows）
  - 非 2 的幂（M=384, K=640, N=1536）
  - 混合长短边（M=8192, N=64, K=4096）
  - LLM 实际 shapes（GPT-2, LLaMA-7B, LLaMA-13B, Mistral 各层）
- **用途：Gate 出口验证 — 必须在 Tier 3 全量上达标**

### Tier 3 Matmul Shapes（50）

| 类别 | 数量 | 示例 shapes (M,N,K) | 设计意图 |
|:-----|:----:|:--------------------|:---------|
| 方阵-小 | 4 | 32³, 64³, 128³, 256³ | Triton launch overhead |
| 方阵-中 | 4 | 512³, 1024³, 2048³, 3072³ | 主战场 |
| 方阵-大 | 3 | 4096³, 6144³, 8192³ | 大 GEMM，内存压力 |
| 非对齐 | 6 | 127×127×127, 257×513×129, 384×640×1536, 1000×1000×1000, 1023×1025×511, 2049×2047×2050 | warp 不对齐，padding 策略 |
| LLM-GPT2 | 5 | 128×768×768, 128×2304×768, 128×50257×768, 512×2304×768, 1024×768×768 | GPT-2 实际层 |
| LLM-LLaMA7B | 5 | 4096×4096×4096, 4096×11008×4096, 1×4096×4096, 2048×4096×4096, 128×32000×4096 | LLaMA-7B 实际层 |
| LLM-LLaMA13B | 4 | 5120×5120×5120, 5120×13824×5120, 2048×5120×5120, 128×32000×5120 | LLaMA-13B |
| 矩形-宽 | 4 | 128×4096×512, 256×8192×1024, 1024×4096×1024, 64×16384×256 | FFN up projection |
| 矩形-高 | 4 | 4096×128×512, 8192×256×1024, 4096×1024×1024, 16384×64×256 | FFN down / embedding |
| 极端比例 | 4 | 1×1024×1024, 16×4096×4096, 8192×64×4096, 32768×1×1024 | 单行/极窄/超长 batch |
| Batch-seq | 4 | 32×512×768, 64×256×768, 16×1024×768, 8×2048×768 | Batch×Seq × Hidden |
| 混合实际 | 3 | 2048×7168×4096, 4096×14336×4096, 1024×8192×2048 | Mixtral/更大模型 |

### Tier 3 Softmax Shapes（25）

| 类别 | 数量 | 示例 shapes (M,N) | 设计意图 |
|:-----|:----:|:-------------------|:---------|
| Attention-小 | 4 | 12×64, 12×128, 12×256, 12×512 | GPT-2 heads |
| Attention-大 | 4 | 32×1024, 32×2048, 32×4096, 32×8192 | LLaMA/Mistral |
| 方阵 | 3 | 256×256, 1024×1024, 4096×4096 | Stress |
| 宽行 | 4 | 1×32000, 1×50257, 1×128256, 4×100000 | Vocabulary |
| 非对齐 | 4 | 12×127, 32×2049, 7×511, 15×1023 | warp 不对齐 |
| 大 batch | 3 | 128×4096, 1024×1024, 4096×512 | Memory bandwidth |
| 极端 | 3 | 1×16, 1×1048576, 65536×64 | 极小/极大/多行短 |

### Tier 3 Elementwise Shapes（15）

| 类别 | 数量 | 示例 shapes (M,N) | 设计意图 |
|:-----|:----:|:-------------------|:---------|
| 小 | 3 | 128×768, 256×768, 512×768 | GPT-2 |
| 中 | 3 | 128×3072, 1024×4096, 2048×4096 | FFN |
| 大 | 3 | 4096×4096, 8192×4096, 4096×11008 | LLaMA FFN |
| 非对齐 | 3 | 127×769, 1000×3000, 2049×4097 | 边界 |
| 极端 | 3 | 1×1048576, 65536×16, 32768×128 | 单行大/多行小 |

---

## 重新设计的 Gate 体系

### 评判原则

1. **Gate 出口 = Tier 3 全量统计指标**，不挑 shape
2. **指标是通过率 + 聚合比率**，不是单点
3. **正确性是 hard gate（100%）**，性能是 soft gate（分布指标）
4. **自动化命令产出 pass/fail + 详细 CSV**

---

### G0: Environment & Toolchain
**假设：** 硬件+软件栈可用且稳定

| # | Criterion | 验证 | 通过条件 |
|---|-----------|------|---------|
| G0.1 | CUDA 可用 | `torch.cuda.is_available()` | `True` |
| G0.2 | Triton 编译通过 | Triton matmul 128³ 编译 | exit 0 |
| G0.3 | GPU 执行正确 | Triton matmul 128³ vs NumPy | `allclose(atol=0.1)` |
| G0.4 | 测试基线 | `pytest tests/ -q` | ≥ 100 passed, 0 failed |

**出口：** `make test` CI log
**自动化：** `arke gate G0`

---

### G1: IR Expressiveness & Validation
**假设：** IR + 验证器在各种 shape/op 组合下均正确

| # | Criterion | 验证 | 通过条件 |
|---|-----------|------|---------|
| G1.1 | OP_CATALOG 覆盖 | `len(OP_CATALOG)` | ≥ 10 |
| G1.2 | Strategy 决策类型 | 枚举 kinds | ≥ 6 种 |
| G1.3 | IR 序列化 | 所有 10 ops × round-trip | 100% 一致 |
| G1.4 | V0 静态验证速度 | timer() 跨 10 ops | 100% < 1ms |
| G1.5 | V1 数值验证-泛化 | **Tier 3 全量 shapes** × matmul/softmax, 3 seeds | **100% pass** (f16: atol=0.1, rtol=0.05) |
| G1.6 | .ak 解析 | ≥ 3 kernel files | AST→IR == KernelBuilder |
| G1.7 | 测试覆盖 | `pytest tests/ -q` | ≥ 200 passed, 0 failed |

**G1.5 是关键改变：** 不是验证"几个 shape 正确"，而是"Tier 3 所有 shape 下 V1 验证都通过"。

**出口：** `arke gate G1` → 跑全量 V1 验证 + 单元测试
**产物：** `gate_results/G1/validation_matrix.csv` (shape × op × seed → pass/fail)

---

### G2: Codegen Quality — Shape 泛化
**假设：** Arke 模板生成的 kernel 在广泛 shapes 上都性能合格

| # | Criterion | 验证 | 通过条件 |
|---|-----------|------|---------|
| G2.1 | matmul 正确性 | L1, **Tier 3 全部 50 shapes** | **100% allclose** |
| G2.2 | softmax 正确性 | L1, **Tier 3 全部 25 shapes** | **100% allclose** |
| G2.3 | matmul 性能通过率 | L1 Tier 3 50 shapes, 每个 vs cuBLAS | **≥ 70% shapes 达到 ≥ 80% cuBLAS** |
| G2.4 | matmul 性能下限 | L1 Tier 3 50 shapes 中最差的 | **0% shapes 低于 20% cuBLAS**（排除 M≤32 的极端 shape） |
| G2.5 | matmul geomean vs cuBLAS | L1 Tier 3 50 shapes 的几何平均 | **geomean ≥ 85%** cuBLAS |
| G2.6 | softmax 性能通过率 | L1 Tier 3 25 shapes, 每个 vs cuDNN | **≥ 60% shapes 达到 ≥ 80% cuDNN** |
| G2.7 | softmax 性能下限 | L1 Tier 3 25 shapes 中最差的 | **0% shapes 低于 10% cuDNN**（排除 N≤32 极端） |
| G2.8 | vs FlagGems 竞争力 | L1 Tier 3 matmul geomean | **geomean ≥ 75% FlagGems** |

**关键指标解读：**
- G2.3 "70% shapes ≥ 80% cuBLAS" = 50 个 shape 中至少 35 个达到 cuBLAS 八成性能
- G2.4 "0% < 20%" = 不能有某个 shape 只跑到 cuBLAS 五分之一（排除已知差的极小 shape）
- G2.5 geomean 是整体实力的聚合指标

**出口：** `arke gate G2` → L1 全量 Tier 3 bench
**产物：**
- `gate_results/G2/matmul_tier3.csv` (50 shapes × baselines)
- `gate_results/G2/softmax_tier3.csv` (25 shapes × baselines)
- `gate_results/G2/summary.json` (通过率、geomean、最差 shape)

---

### G3: LLM Closed-Loop Optimization — 泛化能力
**假设：** LLM agent 在不同 shape 上都能自主优化到合格水平

| # | Criterion | 验证 | 通过条件 |
|---|-----------|------|---------|
| G3.1 | Agent 工具使用 | 1 session | ≥ 8 distinct tools |
| G3.2 | Agent 策略质量 | 1 session | ≥ 4 decisions |
| G3.3 | 闭环无人工 | Agent runner | 0 human steps |
| G3.4 | Agent 正确性-泛化 | Agent 对 **≥10 diverse shapes** 生成 kernel | **100% 正确** |
| G3.5 | Agent 性能-泛化 | Agent kernels 在 ≥10 shapes 上 vs cuBLAS | **geomean ≥ 70% cuBLAS** |
| G3.6 | Agent 错误恢复 | trajectory 分析 | ≥ 1 rollback→success |
| G3.7 | 轨迹完整性 | JSONL output | header + ≥6 step records |

**G3.4-G3.5 的改变：** 不是只在一个 shape 上跑 agent，而是在 10+ diverse shapes 上验证 agent 是否都能产出合格 kernel。shapes 从 Tier 3 中按类别抽样（方阵×3, 矩形×3, 非对齐×2, LLM×2）。

**出口：** `arke gate G3`
**产物：**
- `gate_results/G3/agent_trajectories/` (每个 shape 一个 JSONL)
- `gate_results/G3/agent_kernels/` (每个 shape 的生成 kernel)
- `gate_results/G3/agent_performance.csv` (shape × latency × vs_cublas)

---

### G4: Arke vs Baselines — 全面对比
**假设：** Arke 在正确性+稳定性+效率上全面优于 LLM-direct

| # | Criterion | 验证 | 通过条件 |
|---|-----------|------|---------|
| G4.1 | 正确性优势 | **Tier 3 全量** × 3 trials | Arke correct rate ≥ LLM-direct correct rate |
| G4.2 | 性能竞争 (L1) | Tier 3 matmul geomean | Arke geomean ≥ 90% LLM-direct geomean |
| G4.3 | 一致性优势 | Tier 3 × 3 trials 的 variance | Arke stddev ≤ LLM-direct stddev |
| G4.4 | Token 效率 | 端到端 token 统计 | Arke ≤ 50% LLM-direct tokens |
| G4.5 | vs P1 Expert 竞争 | L1 Tier 3 matmul geomean vs FlagGems | Arke ≥ 85% FlagGems |
| G4.6 | L2 融合泛化 | L2 **Tier 3 全量** matmul+gelu | Arke fused geomean ≥ 80% FlagGems |

**出口：** `arke gate G4`
**产物：**
- `gate_results/G4/comparison_matrix.csv` (Tier 3 × method × trial)
- `gate_results/G4/token_efficiency.json`
- `gate_results/G4/summary.json`

---

### G5: End-to-End Model — 多配置泛化
**假设：** Arke kernel 在真实模型的多种推理配置下都可接受

| # | Criterion | 验证 | 通过条件 |
|---|-----------|------|---------|
| G5.1 | 正确性 | GPT-2 logits, **3 seq_lens** (128/256/512) | **100%** top-1 match, max_diff < 5.0 |
| G5.2 | 延迟泛化 | L3, **3 seq_lens** | **≥ 2/3 seq_lens**: Arke ≤ 1.1× eager |
| G5.3 | 延迟 geomean | L3, 3 seq_lens 的 geomean ratio | **geomean(arke/eager) ≤ 1.15** |
| G5.4 | 内存 | L3, 所有 seq_lens | **100%** peak_mem ≤ 6144 MB |
| G5.5 | 替换覆盖 | patch 统计 | ≥ 48 ops 替换 |
| G5.6 | batch 泛化 | L3, batch=1/4/8 × seq=128 | **≥ 2/3 batch sizes**: 正确 + ≤ 1.15× eager |

**出口：** `arke gate G5`
**产物：**
- `gate_results/G5/e2e_results.csv` (seq × batch × mode → latency/mem/correct)
- `gate_results/G5/summary.json`

---

## 评分聚合公式

### 单算子评分（per shape）
```
shape_score = cublas_latency / arke_latency  (越高越好，>1 = 比 cuBLAS 快)
```

### Gate 聚合指标
```
geomean = exp(mean(log(shape_scores)))       # 几何平均：消除极端值影响
pass_rate = count(shape_score >= threshold) / total_shapes
worst_case = min(shape_scores)               # 最差 shape 不能太离谱
```

### Gate 判定
```python
def check_gate_G2(results: pd.DataFrame) -> GateResult:
    scores = results["cublas_us"] / results["arke_us"]
    
    # 排除极端小 shape（M ≤ 32）的性能判定
    valid = results[results["M"] > 32]
    valid_scores = valid["cublas_us"] / valid["arke_us"]
    
    return GateResult(
        correctness_rate = (results["correct"] == True).mean(),  # must be 1.0
        pass_rate_80 = (valid_scores >= 0.8).mean(),            # must be ≥ 0.70
        worst_case = valid_scores.min(),                         # must be ≥ 0.20
        geomean = gmean(valid_scores),                           # must be ≥ 0.85
    )
```

---

## 自动化 CLI 设计

```bash
# 单个 Gate
arke gate G0                    # 环境检查
arke gate G2                    # 跑 L1 Tier 3 全量 + 判定
arke gate G5                    # 跑 L3 多配置 + 判定
arke gate --all                 # 全部 Gate（耗时较长）

# 指定 Tier
arke gate G2 --tier 1           # 快速检查（15 shapes）
arke gate G2 --tier 2           # 标准检查（31 shapes）
arke gate G2 --tier 3           # Gate 出口（50 shapes）

# 输出
arke gate G2 --tier 3

  G2: Codegen Quality — Shape Generalization (Tier 3: 50 shapes)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  G2.1 matmul correctness          ✅ 50/50 (100%)
  G2.2 softmax correctness         ✅ 25/25 (100%)
  G2.3 matmul ≥80% cuBLAS rate     ✅ 38/50 (76% ≥ 70%)
  G2.4 matmul worst case           ✅ min=22% (M=32³, excluded)
                                       min valid=43% (M=127³) ≥ 20%
  G2.5 matmul geomean              ✅ 91% cuBLAS (≥ 85%)
  G2.6 softmax ≥80% cuDNN rate     ✅ 17/25 (68% ≥ 60%)
  G2.7 softmax worst case          ✅ min valid=15% (N=1048576) ≥ 10%
  G2.8 vs FlagGems geomean         ✅ 82% (≥ 75%)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  G2: PASS (8/8)

  Detailed CSV: gate_results/G2/matmul_tier3.csv
  Worst shapes:
    M=127,N=127,K=127  → 43% cuBLAS (non-aligned)
    M=64,N=64,K=64     → 31% cuBLAS (launch overhead, excluded M≤32 rule)
```

---

## 阈值设定逻辑

| 阈值 | 设定方式 | 说明 |
|:-----|:---------|:-----|
| 正确性 100% | Hard gate | 生成的 kernel 必须全部正确，不允许例外 |
| 性能通过率 70% | 基于 Pareto 分布 | 小 shape 天然慢（launch overhead），允许 30% 不达标 |
| 性能 geomean 85% | 整体实力 | geomean 比 mean 更稳健，不被极端值拉动 |
| worst case 20% | 底线 | 即使最差的 shape 也不能离谱到只有 cuBLAS 1/5 |
| E2E 1.1× | 实用阈值 | 10% overhead 对用户可接受 |
| Token 50% | 效率目标 | IR 编码比 raw code 短，应该省 token |

**排除规则：**
- 性能判定中排除 M≤32 (matmul) 或 N≤32 (softmax) 的极端 shape
- 原因：Triton ~55μs launch floor 在极小 shape 上无法避免，这是 Triton 本身的限制而非 Arke 的问题
- 这些 shape 仍要求**正确性 100%**，只是性能比率不计入通过率/geomean

---

## v1 → v2 关键变化

| 方面 | v1 (cherry-pick) | v2 (泛化驱动) |
|:-----|:-----------------|:-------------|
| Shape 覆盖 | 挑 3-4 个 shape | Tier 3 全量 50+ |
| 性能标准 | 单点 ≥ X% | 通过率 + geomean + worst case |
| 正确性 | 隐式 | 显式 hard gate 100% |
| 非对齐 | 未覆盖 | 6+ 非对齐 shapes |
| 极端 case | 未覆盖 | 极小/极大/单行/宽行 |
| E2E 泛化 | 1 个 seq_len | 3 seq_lens + 3 batch sizes |
| Agent 泛化 | 1 个 shape | 10+ diverse shapes |
| 输出 | 人工检查 | 自动 pass/fail + CSV 归档 |
