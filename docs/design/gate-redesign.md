# Stage 1 Gate Redesign — SMART 化 + Benchmark 驱动

## 当前问题诊断

### 问题 1: Gate 标准不够 SMART
| Gate | 问题 |
|:-----|:-----|
| G0 | "Triton matmul runs" — 没有量化标准，什么叫 "runs"？ |
| G1 | "Known-good strategy representable" — 主观判断，无自动验证 |
| G2 | "perf ≥ 70% cuBLAS" — 哪个 shape？哪个 dtype？单次还是多次？ |
| G3 | "matmul perf ≥ 50% cuBLAS" — 同上，且阈值太低 |
| G4 | "across ≥5 tasks" — 但 5 个 task 的具体定义不在 Gate 中 |
| G5 | "latency ≤ torch.compile" vs "≤ 1.1× eager" — 标准改过，不一致 |

### 问题 2: 验证不可自动化
- 大部分 Gate 验证靠人工跑脚本 + 目视检查
- 没有 `arke gate G2` 这样的一键验证命令
- 结果没有归档到 benchmark results

### 问题 3: 缺少 Benchmark 对齐
- G2-G5 的性能标准没有对应到 L1/L2/L3 benchmark 的具体 shape/baseline
- 无法从 benchmark CSV 自动判断 Gate 是否通过

---

## 重新设计的 Gate 体系

### 设计原则
1. **Specific** — 每个 Gate 引用具体的 benchmark shape、baseline、阈值
2. **Measurable** — 每个 criterion 有精确的数值判定，可从 CSV 自动计算
3. **Achievable** — 阈值基于实际硬件能力设定，不过高也不过低
4. **Relevant** — 每个 Gate 直接验证一个核心假设
5. **Time-bound** — 每个 Gate 有对应的 Phase，完成即过

### Gate → 假设 → Benchmark 层映射

```
Gate    验证假设                           Benchmark 层    自动化命令
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
G0      环境 + 工具链可用                   —              make setup && make test
G1      IR 表达力 + 验证正确性              单元测试        pytest tests/ -q
G2      Codegen 质量（单算子）              L1             arke gate G2
G3      LLM 闭环优化能力                    L1 + Agent     arke gate G3
G4      Arke vs LLM-direct 优势            L1 + L2        arke gate G4
G5      整模型端到端                        L3             arke gate G5
```

---

## Gate 定义（SMART 版）

### G0: Environment & Toolchain
**假设：** Triton + CUDA + PyTorch 在目标硬件上可用且稳定

| # | Criterion | 验证方法 | 通过条件 |
|---|-----------|---------|---------|
| G0.1 | CUDA 可用 | `torch.cuda.is_available()` | `True` |
| G0.2 | Triton 编译 | Triton matmul 编译无错误 | exit code 0 |
| G0.3 | GPU 执行 | Triton matmul [128,128,128] 产出正确结果 | `allclose(triton, numpy, atol=0.1)` |
| G0.4 | 测试基线 | `pytest tests/ -q` | ≥100 passed, 0 failed |

**出口产物：** `make test` 通过的 CI log

---

### G1: IR Expressiveness & Validation
**假设：** 两层 IR 能完整表达优化策略，验证器能检测非法操作

| # | Criterion | 验证方法 | 通过条件 |
|---|-----------|---------|---------|
| G1.1 | OP_CATALOG 覆盖 | `len(OP_CATALOG)` | ≥ 10 ops |
| G1.2 | Strategy 决策类型 | `len(decision_kinds)` | ≥ 6 种 (tile/fuse/place/parallel/reorder/algorithm) |
| G1.3 | IR 序列化完整 | `ir == from_json(to_json(ir))` for 所有 10 ops | 100% round-trip |
| G1.4 | V0 静态验证速度 | `timer(v0_validate(ir))` | < 1ms |
| G1.5 | V1 数值验证 | matmul+softmax, 3 random seeds | 100% pass (f16: atol=0.1, rtol=0.05) |
| G1.6 | .ak 解析 | `parse(file) → AST → IR` for ≥3 kernels | AST→IR == KernelBuilder 输出 |
| G1.7 | 测试覆盖 | `pytest tests/ -q` | ≥ 200 passed, 0 failed |

**出口产物：** `pytest tests/test_ir.py tests/test_validation.py tests/test_parser.py -v` 全绿

---

### G2: Codegen Quality (Single-Op)
**假设：** Arke 模板生成的 Triton kernel 性能可达 vendor library 水平

| # | Criterion | 验证方法 | 通过条件 |
|---|-----------|---------|---------|
| G2.1 | matmul 正确性 | L1 Arke runner, 所有 12 shapes | 100% `allclose(arke, numpy, atol=0.1)` |
| G2.2 | softmax 正确性 | L1 Arke runner, 所有 5 shapes | 100% `allclose` |
| G2.3 | matmul 性能 (中型) | L1: `square-1k` [1024³] | Arke ≥ 80% cuBLAS |
| G2.4 | matmul 性能 (大型) | L1: `square-4k` [4096³] | Arke ≥ 90% cuBLAS |
| G2.5 | matmul 性能 (LLM shape) | L1: `seq512` [512,2304,768] | Arke ≥ 80% cuBLAS |
| G2.6 | softmax 性能 | L1: `attn-large` [32,2048] | Arke ≥ 80% cuDNN |
| G2.7 | vs FlagGems (P1) | L1 matmul geomean across 12 shapes | Arke ≥ 80% FlagGems |

**出口产物：** `python -m benchmarks --layer L1 --op matmul,softmax` CSV + 自动判定脚本

**自动验证命令：** `arke gate G2`
```python
# 从 L1 CSV 读取数据，自动计算:
for shape in ["square-1k", "square-4k", "seq512"]:
    ratio = cublas_latency[shape] / arke_latency[shape]
    assert ratio >= threshold[shape], f"G2 FAIL: {shape} ratio={ratio}"
```

---

### G3: LLM Closed-Loop Optimization
**假设：** LLM 通过 tool-use 能自主优化 kernel，达到合格性能

| # | Criterion | 验证方法 | 通过条件 |
|---|-----------|---------|---------|
| G3.1 | LLM 工具使用 | 1 session trajectory | ≥ 8 distinct tools used |
| G3.2 | LLM 策略应用 | 1 session trajectory | ≥ 4 decisions applied |
| G3.3 | 闭环无人工 | Agent runner | start→finish, 0 human steps |
| G3.4 | matmul 性能 | Agent output kernel, L1 shape `square-1k` | ≥ 80% cuBLAS |
| G3.5 | 错误恢复 | Trajectory | 至少 1 次 rollback 后成功 |
| G3.6 | 轨迹完整 | JSONL output | 包含 header + ≥6 step records |

**出口产物：** 
- Agent 运行 log + trajectory JSONL
- Agent 生成的 kernel 存入 `benchmarks/results/kernels/agent/`
- L1 benchmark 对 agent kernel 的性能数据

**自动验证命令：** `arke gate G3` (运行 agent → 检查 trajectory → L1 bench agent kernel)

---

### G4: Arke vs Baselines Advantage
**假设：** Arke 在正确性+稳定性+Token 效率上优于 LLM 直接写 Triton

| # | Criterion | 验证方法 | 通过条件 |
|---|-----------|---------|---------|
| G4.1 | 正确性优势 | 5+ tasks × 3 trials | Arke correct rate ≥ LLM-direct correct rate |
| G4.2 | 性能竞争力 | L1 matmul geomean (≥6 shapes) | Arke ≥ 90% of LLM-direct perf |
| G4.3 | 一致性优势 | 5+ tasks × 3 trials | Arke variance ≤ LLM-direct variance |
| G4.4 | Token 效率 | 端到端 token 消耗 | Arke ≤ 50% of LLM-direct tokens |
| G4.5 | vs FlagGems 竞争 | L1 geomean (large shapes: ≥1K) | Arke ≥ 90% FlagGems |
| G4.6 | 融合算子竞争 | L2 matmul+gelu, shape square-1k | Arke fused ≥ 80% FlagGems fused |

**出口产物：**
- L1/L2 CSV 对比数据
- Token 消耗统计
- `benchmarks/results/comparison/arke_vs_direct.csv`

**自动验证命令：** `arke gate G4`

---

### G5: End-to-End Model Integration
**假设：** Arke kernel 集成到真实模型后，性能/正确性/内存均可接受

| # | Criterion | 验证方法 | 通过条件 |
|---|-----------|---------|---------|
| G5.1 | 正确性 | GPT-2 logits 比较 | top-1 match ✅, max_diff < 5.0 |
| G5.2 | 延迟 (seq=128) | L3 benchmark | Arke ≤ 1.1× PyTorch eager |
| G5.3 | 延迟 (seq=512) | L3 benchmark | Arke ≤ 1.1× PyTorch eager |
| G5.4 | 内存 | L3 benchmark | peak_mem ≤ 6144 MB (GPU VRAM) |
| G5.5 | 替换覆盖 | patch 统计 | ≥ 48 Conv1D/Linear 替换 |
| G5.6 | 多 seq_len 稳定 | L3 seq=128,256,512 | 3 个 seq_len 中至少 2 个 pass G5.2 标准 |

**出口产物：**
- `python -m benchmarks --layer L3 --seq-len 128,256,512` CSV
- 自动 pass/fail 判定

**自动验证命令：** `arke gate G5`

---

## Gate 自动验证 CLI

```bash
# 验证单个 Gate
arke gate G0          # 环境检查
arke gate G2          # 运行 L1 bench + 判定
arke gate G5          # 运行 L3 bench + 判定
arke gate --all       # 所有 Gate

# 输出格式
G2: Codegen Quality (Single-Op)
  G2.1 matmul correctness (12 shapes)    ✅ PASS  12/12
  G2.2 softmax correctness (5 shapes)    ✅ PASS  5/5
  G2.3 matmul perf square-1k ≥80%        ✅ PASS  164% cuBLAS
  G2.4 matmul perf square-4k ≥90%        ✅ PASS  95% cuBLAS
  G2.5 matmul perf seq512 ≥80%           ✅ PASS  139% cuBLAS
  G2.6 softmax perf attn-large ≥80%      ✅ PASS  109% cuDNN
  G2.7 vs FlagGems geomean ≥80%          ✅ PASS  94.5%
  ──────────────────────────────────────
  G2: PASS (7/7)
```

---

## 阈值设定依据

所有阈值基于 RTX 3060 Laptop 实测数据设定：

| 阈值 | 来源 | 说明 |
|:-----|:-----|:-----|
| matmul ≥80% cuBLAS (1K) | 实测 164% | 留 2× 安全余量 |
| matmul ≥90% cuBLAS (4K) | 实测 95% | 接近硬件天花板 |
| softmax ≥80% cuDNN | 实测 109% | 留余量 |
| vs FlagGems ≥80% | 实测 94.5% | FG 是 Triton 最强开源基线 |
| E2E ≤1.1× eager | 实测 1.01× | 允许 10% overhead |
| Token ≤50% LLM-direct | 实测 ~28% | IR 比 raw code 短很多 |

---

## 与 Benchmark 系统的映射

| Gate Criterion | Benchmark Command | CSV Source | Auto-check |
|:---------------|:-----------------|:-----------|:-----------|
| G2.3 matmul 1K ≥80% | `--layer L1 --op matmul` | `matmul_results.csv` shape=square-1k | `arke[latency] / cublas[latency] ≤ 1.25` |
| G2.7 vs FlagGems | `--layer L1 --op matmul` | `matmul_results.csv` | `geomean(cublas/arke) / geomean(cublas/flaggems) ≥ 0.8` |
| G4.6 fused ≥80% FG | `--layer L2` | `matmul_gelu_results.csv` | `flaggems[latency] / arke[latency] ≤ 1.25` |
| G5.2 E2E ≤1.1× | `--layer L3 --seq-len 128` | `gpt2_results.csv` | `arke[mean_ms] / eager[mean_ms] ≤ 1.1` |
