# G7.8d — 误诊纠偏 + 三选项决策文档

**HEAD**: `13d42e6` · **Gate G7**: 13/14 (92.9%) · **唯一 fail**: G7.8d weighted=0.513 / target=0.95

> **本文档第一版**（同日早些时候）提出"三层 autotune 方案"，方向错误。  
> **第二版**（本版本）记录纠偏过程与正确的决策树。

---

## 1. 错误的初始假设

第一版假设：「Arke vs torch.compile 跑输 → 给 Triton 模板加 `@triton.autotune` config 就能翻盘」。

由此推出三层方案：
- Wave-1: 给 8 个硬编码 `.j2` 模板加 autotune
- Wave-2: matmul / batch_matmul 扩 config
- Wave-3: layernorm/rmsnorm structural redesign

## 2. 纠偏触发点

实际改 `elementwise.py.j2` 后 smoke test：

```python
from arke.backend.triton_codegen import generate_triton_kernel
# ModuleNotFoundError: No module named 'arke.backend.triton_codegen'
```

`grep -rn "triton_templates\|\.j2\|jinja" arke/ benchmarks/` 全空：

```
arke/ir/ops/schema.py
arke/ir/ops/catalog.py
benchmarks/README.md
```

**没人引用 `.j2` 模板**。

## 3. 真相

### 3.1 backend 自承（`arke/backend/triton_backend.py:31`）

```python
"""
For S6, uses template-based codegen from OpRegistry.template_hint.

This is a minimal but functional implementation for:
    ...
    to PyTorch eager execution via reference_impl for correctness.

Full Triton codegen is S7 scope (template engine integration).
"""
```

### 3.2 ArkeRunner 自承（`benchmarks/baselines/arke_runner.py:6-9, 126, 134`）

```python
"""S6 bridge: routes baseline calls through ``arke.ir.ops.interpreter.INTERPRETER``
(the same ``reference_impl`` substrate that ``arke/backend/triton_backend.py``
currently executes on). When S7 replaces ``reference_impl`` with real Triton
codegen, this runner will automatically pick that up with no changes."""

# in get_fn():
INTERPRETER.execute(op, named, attrs)   # ← this is PyTorch eager equivalent
```

### 3.3 推论链

```
ArkeRunner.get_fn   →  INTERPRETER.execute   →  schema.reference_impl   →  PyTorch eager
                       (S6 placeholder)         (Python/torch)             (cuBLAS/CUDA eager)
```

**Arke "kernel" 此刻 = PyTorch eager 等价物**。

`triton_templates/*.j2` 是 **S7 codegen 的素材**，bench 此刻完全不读不渲染。

### 3.4 PERF_ALL 数据回看

| baseline 对手 | 典型 ratio | 解释 |
|---|---|---|
| PyTorch (eager) | ~1.0 | Arke 就是 eager，几乎平 |
| torch.compile | 0.17-0.40 | inductor 优化 vs eager — Arke 完败是结构性结论 |
| FlagGems / Triton-Tutorial / cuBLAS | 0.95-0.99 | 优化过的 baseline vs eager — 持续小幅落后 |
| Arke vs Arke | 0.95-1.0 抖动 | 同一 eager 实现两次跑差 5% = 测量 noise |

**G7.8d weighted=0.513 反映的是 S6 阶段 Arke 还没真正生成 kernel 的设计事实，而不是某个具体的 perf bug。**

---

## 4. 选项

### A · 提前启动 S7 codegen 接入

把 jinja 模板真正接进 `arke/backend/triton_backend.py`：

```
arke/backend/triton_backend.py
  ↓  (新加 path)
TritonCodegen.render(.j2 template) → JIT compile → KernelCache
  ↓
真正的 Triton kernel (而不是 reference_impl)
```

**优点**：
- G7.8d 的 perf 数据立刻有意义（Arke 真的在跑生成的 kernel）
- 一旦接通，模板里现有的 autotune 配置（matmul 12 个 / batch_matmul 5 个 / layernorm 15 个）立即生效
- 22 个模板都已写好，缺的只是渲染+JIT 这一层
- S7 后续工作受益（KernelCache、CodegenContext 等）

**风险 / 工作量**：
- 跨 stage 边界：S6 还没收口就跑 S7，违反 stage-by-stage 原则
- jinja2 渲染、JIT 错误处理、缓存键设计、attrs→template-var 映射等 **5-8 个独立 commit**
- 22 个模板里 19 个没 autotune（昨天发现的），接通后仍需补 autotune 才能跑出像样数字
- 需要 Leon 批：是否动 Stage 边界

### B · 调整 G7.8d 验收口径（动 Gate 定义）

承认 S6 阶段 Arke = PyTorch-eager 是**设计意图**，重新定义 weighted：

```python
# 当前 (gate_g7.py:456-570)
perf_pass = ratio >= perf_target   # default perf_target=1.0 vs ANY baseline

# 提议 (S6 期间)
perf_pass = ratio_vs_pytorch_eager >= 0.95   # 只比 PyTorch eager
# OR
perf_pass = ratio_vs_best_baseline >= 0.5    # 容忍优化型 baseline 大幅领先
```

**优点**：
- 反映 S6 的真实状态，不假装 perf 应该 ≥ torch.compile
- 改完 G7 立刻 14/14
- 工作量：~50 行 gate_g7.py 改动 + 文档同步

**风险**：
- **触 Gate 定义、threshold、exit-criteria —— 必须 Leon 明确批**（SOUL.md / AGENTS.md 硬规则）
- 等于承认 G7 的 perf gate 设计有缺陷需要打补丁

### C · 接受 G7.8d S6 阶段 fail，G7 标 13/14 完结

承认 G7.8d 的 G6→G7 跨度内本来就解不掉，留到 S7 自然解决（接通 codegen 后 perf 数字会跳一档）。

**优点**：
- 零代码改动，零 Gate 动议
- G7 13/14 在 S6 阶段已是高分，可以转去 S7 准备工作或 G6 扫尾
- 最诚实反映项目阶段

**风险**：
- 留一个 fail 没收口，roadmap 上看起来不干净
- 需要 Leon 同意 "G7=13/14 视为 S6 阶段完结"

---

## 5. 推荐与下一步

**Kitty 推荐 A**（提前启动 S7 codegen 接入）：

1. **不触 Gate 定义**（B 的硬约束规避）
2. **不留 fail 尾巴**（C 的痒点解决）
3. **战略性向前**：S7 早晚要做，提前一周开始价值密度更高
4. **工作量可控**：22 个模板已写好，缺的是 renderer + JIT + cache 这层胶水
5. **可单 commit chain**：渲染器 → matmul 单 op 接通 → 全 op 接通 → autotune 补齐 → G7 重跑

**但**：A 在 Stage 边界上有争议，故先写本文档让 Leon 决策。

---

## 6. 已知坑（保留自第一版）

1. DSv3-lmhead / long-32k @ 6GB + `--force-restart` = 9h+ 静默 OOM hang
2. `bench_l1` resume-cache hit 跳过 perf-write — rerun 后必须 `/tmp/refresh_perf.py` 重生
3. Hermes `process(action=kill)` 缺 psutil → `kill -9 <pid>`
4. Hermes terminal stdout 上限 50KB — 长 bench 用 `tail` / `grep`
5. autotune 首次 launch 跑全部 config benchmark，长 seq × 多 config 单次首跑 30s+

## 7. 回滚记录

- `arke/backend/triton_templates/elementwise.py.j2` — Kitty 改后 `git checkout` 回滚 ✓
- `arke/backend/triton_templates/layernorm.py.j2` — sibling subagent 改的 15-config autotune `git checkout` 回滚 ✓
- HEAD 仍为 `13d42e6` clean
