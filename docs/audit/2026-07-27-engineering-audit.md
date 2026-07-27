# Arke — Shape 泛化能力 + 工程落地 / 开源就绪度审计（只读）

**审计范围**：`/home/blueyi/workspace/repos/arke` @ `a0a907f`（`git log` HEAD）
**方法**：只读代码/文档 + Python 静态统计（`/tmp/stats.py`，AST import 分析）。禁止 CUDA 计算。
**日期**：2026-07-27

---

## 1. 动态 Shape 鲁棒性

### 1.1 模板层缓存机制的真实语义（关键澄清）

slim-launch pass（`8d41b68`/`6b78560`/`4ae6e31`，2026-07-26）在 reduce 族模板里引入的 **per-shape config cache 不缓存编译产物**，只缓存纯 Python 的启动配置计算。证据：

- `arke/backend/triton_templates/softmax.py.j2:251-285` — `_LAUNCH_CFG: dict = ` 键为 `N`，值为 `(is_simple, BLOCK_or_TILE, num_warps)`。注释明说：*"next_power_of_2 + the warp ladder are pure functions of N; computing them once per distinct N keeps the hot wrapper to dict-lookup + launch."*
- `reduction.py.j2:30-43` 同构：`_LAUNCH_CFG` → `(is_loop, block, warps)`。

因此**首次调用性能悬崖不来自这个 dict**。真正的 JIT/重编译由两处承担：

1. **Triton 自身的 per-shape specialization**。`arke/backend/kernel_cache.py:20-22` 明确设计：*"Triton itself handles per-shape kernel specialization internally, so we do **not** key the cache on shape"*。KernelCache 的键是 `(op_name, template_name, primary_op, extra_ctx, dtype, kernel_name)`（`kernel_cache.py:228-235`）——**完全不含 shape**。同一 wrapper 喂新 shape，交给 Triton 内部 JIT 缓存。
2. **matmul 的 `@triton.autotune(key=["M","N","K"])`**（`matmul.py.j2:45`，18 个 config）。这是**唯一按具体 M/N/K 触发重新 autotune 的路径**——每个未见过的 (M,N,K) 组合首次调用会跑完整 autotune 扫描。

### 1.2 面对动态 seq_len（Llama3 8k/32k、SD 变分辨率）的行为

- **reduce/softmax/layernorm 族**：warp ladder 是 `N` 的纯函数，新 `N` 只付一次 Triton JIT 编译（`kernel_cache.py:11-13` 量化：首个 `@triton.jit` 冷启动 ~300ms，后续 wrapper 生成 ~5-10ms）。**无 per-shape 悬崖累积**，但每个新 `N` 仍付一次 Triton 编译——对每步 seq_len 都不同的训练循环，这是逐 shape 的一次性成本。
- **matmul/bmm**：`key=["M","N","K"]` 意味着变分辨率 SD / 变 seq_len 会为**每个新 (M,N,K)** 触发 autotune。这是最真实的"首次调用悬崖"来源。文档量化过同类现象：`docs/phase1/stage8-plan.md:252` 记录 GPT-2 seq=256 因 `torch._dynamo cache_size_limit=8` 命中 **dynamic-shape recompile thrash**，ratio 0.865→修复后 1.070（bump 到 64 + `dynamic=True`）——但**该修复只作用于 torch.compile 基线路径（bench_l3），不作用于 Arke 自身模板**。

### 1.3 Shape bucketing / padding / 动态维度支持

- **IR 层有符号维度模型**：`arke/ir/semantic.py:33-72` `SymbolicDim(name, min, max, is_static, multiple_of, default)` + `ShapeConstraint`（代数约束，如 `"S % 128 == 0"`）。规范文档 792 行（`docs/spec/symbolic-dimension-spec.md`）。
- **但符号维度未贯通到 codegen**：`search_files SymbolicDim` 在 `arke/backend/` **零命中**（`ir/converters.py`、`ir/semantic.py` 内部使用，backend 模板全部按运行时具体 `int` shape 分派）。模板 wrapper 签名是 `(X: torch.Tensor)` → 读 `X.shape`（`softmax.py.j2:288-290`）。**没有 shape bucketing / padding-to-next-power / 动态维度 lowering**——`search_files` 对 `bucket|pad_to|round_up.*shape|variable.*resolution` 在 `arke/` 全部零命中（仅 benchmarks gate 里有无关的统计 "bucket"）。
- 唯一的"分档"是 warp ladder 与 N>16384 / N>65536 的**内核变体切换**（simple vs loop kernel），非 shape padding。

**���化结论**：动态 shape 支持停留在 **IR 声明层**；执行层是纯运行时具体 shape + Triton 内部 JIT + matmul autotune。无 bucketing/padding，无编译开销的显式量化 probe（`benchmarks/probes/` 4 个文件仅测 launch-floor / CPU dispatch，`launch_floor.py:25` 固定 `warmup=50`，测的是稳态 launch overhead **不是**冷启动编译成本）。JIT 编译开销仅在 `kernel_cache.py` docstring 有粗量化（~300ms 冷 / ~5-10ms 暖），无系统性 benchmark。

---

## 2. 退化控制（Fallback / 降级路径）

覆盖度较好，是本项目工程成熟度最高的一环。

| 机制 | 位置 | 触发 | 行为 | 死角评估 |
|---|---|---|---|---|
| **Golden ladder GoldenUnavailable** | `benchmarks/golden_ladder.py:69,142-170` | 无 P0-P3 runner 支持该 op / 强制 pin 的 runner 不可用 / ladder 触及 P5（被测系统本身） | **fail-loud** 抛异常 → 上层记 `golden_unavailable_pending_baseline` 审计行 | 健壮。P5 守卫（`:160`）防止 Arke 拿自己当 oracle，是很强的正确性设计 |
| **Memory preflight** | `benchmarks/memory_policy.py` + `bench_l1.py:899-923` | 预估 bytes > GPU 预算 | 记 typed skip 行（含 `bytes_required/budget/ratio`），**不静默丢点** | 健壮。含 FlagGems SDPA 分解为 bmm 会 materialize `[B*H,S,S]` 的重分类（`memory_policy.py:53-68`，实测 32-112GiB 尝试） |
| **get_fn OOM 守卫** | `bench_l1.py:932-959` | baseline get_fn/pre-warm 阶段 OOM | catch `torch.OutOfMemoryError` → `empty_cache()` → 记 `status=oom` 行 → **保活整个 run** | 健壮（`eb11e64` 修复：曾能 kill 整个 bench run） |
| **Codegen 抛异常 → interpreter fallback** | `triton_backend.py:114-181` + 测试 `tests/backend/test_triton_backend_dispatch.py:168-184` | codegen `generate_kernel` 抛任何异常 / 无 template_hint | 计入 `num_fallback` bucket，dispatcher 退回 `SemanticInterpreter`，**图仍产出正确输出**（测试断言 `allclose`） | **有测试覆盖**（合成 codegen failure）。这是 §1 死角提到的 harness 行为——已覆盖 |
| **Runtime wrapper 异常 fallback** | `triton_backend.py` + 测试 `:186` | 编译后的 wrapper 运行期抛异常 | dispatcher 重试 interpreter | 已覆盖 |
| **Watchdog 超时** | `bench_l1.py:1030-1055` + `benchmarks/watchdog.py` | 单次 measurement / correctness probe 挂起 | `WatchdogTimeout` → typed `status=timeout` 行 | 健壮（`:1013` 记录曾有 11h flash_attention hang） |
| **PlateauEarlyStop** | `arke/agent/extensions.py:228-275` | 连续 N 次 compile 无 `baseline_ratio` 改善 | 设 `should_stop`，不改 frozen runner，caller 自行 break | 设计干净（PostProfile hook seam） |
| **checkpoint/rollback** | `arke/agent/state.py:199-230` | agent 探索回退 | deep-copy 快照 + 恢复 | KeyError on unknown label（`:226`）——fail-loud |
| **V0/V1/V2 验证闸** | `README.md:77`; `arke/compiler/validator.py:26-61`（V0 结构）; `arke/agent/verification.py:12` staged 5-stage correctness gate | V0<1ms 结构合法性 → V1 数值 → V2 性能 | correctness gate performance；wrong-but-fast 得 -1（`verification.py:80`） | 设计对齐 SOTA（AutoKernel 5-stage firewall） |

### 2.1 未覆盖死角

- **matmul autotune 失败无显式 fallback 分支**：codegen 层 fallback 已覆盖，但 `@triton.autotune` 若在某 (M,N,K) 上所有 config 都编译失败（如 K 过小 / 极端非对齐），走的是 Triton 自身异常 → 会被 `triton_backend` 的 runtime except 兜住退 interpreter，但**没有针对性测试**该路径（现有测试是合成 codegen failure 与 wrapper 异常，非 autotune 编译失败）。
- **符号维度 lowering 失败无路径**：因为符号维度根本不进 codegen（§1.3），"shape 泛化失效"在执行层不会以 SymbolicDim 形式出现——退化实际由 Triton per-shape JIT 承担，**没有 Arke 自己的 shape-generalization-failure 分支**。
- **layernorm N>65536 是 `raise` 而非 fallback**（`c2df77b` commit body：*"layernorm.py.j2: N>65536 now raises loudly instead of clamping"`）——比静默截断好（fail-loud），但对合法大 hidden dim（如 14336 仍安全，但拼接场景）是硬失败而非降级。

---

## 3. 工程落地

### 3.1 代码规模（`/tmp/stats.py` 实测，AST + 行数）

```
[arke/]        files=97   loc=34,216
    backend      31  18,103   ← 最大（Triton/MLIR/CUDA-C/LLVM 4 后端 + 25 个 .j2 模板）
    agent        14   6,822
    ir           17   3,757
    compiler     14   1,918
    learn         7   1,810
    lang          5   1,079
    backends      4     235   （mlir 子包）
    integration   2     206   （torch_bridge）

[benchmarks/]  files=72   loc=20,457
    (root)       44  14,943   ← bench_l1/l2/l3 + gate_g6/g7/g8/p5 + golden_ladder + memory_policy …
    baselines    17   3,898   （cuBLAS/Liger/FlagGems/Triton-tutorial/flash-attn/vLLM runner）
    live          5     948
    probes        4     201

[tests/]       files=162  loc=25,186
    (root)      117  17,378
    backend      38   6,892

GRAND TOTAL: files=331   loc=79,859（含 25 个 .j2 模板另计）
```

### 3.2 模块解耦 / 依赖方向（AST import 分析）

```
ir           -> [lang]
backend       -> [ir]
backends      -> [ir]
compiler      -> [ir, lang]
agent         -> [backend, compiler, ir, learn]
learn         -> [agent]          ⚠️
integration   -> [backend, ir]
```

- **依赖方向总体健康**：`lang → ir → {backend, compiler} → agent` 单向分层，backend 不反向依赖 compiler/agent，符合 README 的分层图。
- **1 个循环：`agent ⇄ learn`**。证据：`learn/rl_dataset.py:40`、`learn/session_recorder.py:40` `from arke.agent.verification import robust_reward`；`learn/trajectory_schema.py:73` `from arke.agent.events import ...`；而 `agent → learn`（statser 里 agent 依赖 learn）。这是**真实的模块级环**。影响：agent 与 learn 无法独立打包/测试；不是致命（Python 允许），但违反单向分层理想，是重构债。

### 3.3 测试规模与覆盖广度

- **1,408 个 test 函数 / 162 个 test 文件 / 25,186 LOC**（`/tmp/stats.py` 计 `def test_`）。
  - 注：任务基线"~2700 pass"应为 **参数化后的用例数**（pytest `--dist loadfile -n 4`，含 `@pytest.mark.parametrize` 展开），与 1408 个 test 函数不矛盾。AGENTS.md 记 baseline 2534 passed（2026-07-15）。
- **覆盖广度**：backend 单列 38 文件 6,892 LOC / 334 test（含 codegen fallback、dispatch、多后端）——**后端测试密度最高**，与 backend 是最大代码块一致。agent/benchmark 目录测试文件极少（各 1），但 root 下 1059 test 覆盖 lang/ir/compiler/agent 混合。

---

## 4. 生态兼容 / 开源就绪度

### 4.1 已具备
- **LICENSE**：Apache-2.0（`LICENSE:1`，每个 .py 带 SPDX header）。
- **CI 公开**：`.github/workflows/ci.yml` 存在——Python 3.10/3.11/3.12 矩阵、ruff lint、mypy、pytest。（注：CI 跑 `pytest tests/ -v` 但无 GPU，CUDA 路径在 GitHub runner 上会 skip/fail，实际 GPU 门禁靠本地 `make test`。）
- **README 309 行**：清晰的架构 ASCII 图、CLI 表（`arke compile/optimize/run/mcp`）、版本语义（v0.1.0）、文档索引（分 Specs/Architecture/Benchmark/Stage 四类，~30 个 doc 链接）。
- **docs 结构完整**：`spec/`（lang/ir/symbolic-dim/pass/op-registry 5 份权威 spec）、`architecture/`（14 份，含 e2e-flow、compiler-infra、harness-handbook）、`benchmark/`、`roadmap/plan.md`、5 个 phase 目录。
- **SSOT 纪律**：op 数量单一真源（`benchmarks/op_registry.total_ops()` 实测返回 **46**；`docs/benchmark/benchmark-ops.md:717` "Full Operator Index (46 ops)" 一致）。

### 4.2 文档-代码同步度抽查（2-3 处）
1. **op 数 SSOT** ✅ 同步：runtime `total_ops()=46` == benchmark-ops.md == AGENTS.md（"46 ops"）。
2. **kernel_cache docstring 陈旧** ⚠️：`kernel_cache.py:38` 仍写 *"45-op × 4-dtype"*，落后于实际 46 op（轻微文档漂移，非功能性）。
3. **V0/V1/V2 闸** ✅ 同步：README:77 ASCII 图 `V0(<1ms)→V1→V2` == `compiler/validator.py`（V0 结构）+ `agent/verification.py`（staged gate）实现存在。

### 4.3 对 MLIR/LLVM 社区开发者的吸引点与缺口

**吸引点**：
- 4 个后端全在树（Triton `.j2`、MLIR dialect `backends/mlir` + `mlir_gpu.py`、CUDA-C、LLVM IR `llvm_*.py`），46/46 op GPU 覆盖（AGENTS.md），MLIR 1.05× cuBLAS。
- 有 nvgpu dialect 研究文档（`docs/nvgpu-dialect-research.md`）、IR↔MLIR 映射 spec（`docs/spec/ir-mlir-mapping.md`）。
- 符号维度规范 792 行，接口契约（op-registry-interface.md、pass-infrastructure-spec.md）齐全——对编译器开发者友好。

**缺口**：
- **无 CONTRIBUTING.md、无 CODE_OF_CONDUCT.md**（`ls` 确认缺失）——新贡献者无提交/评审流程入口。
- **无标准 RFC 流程**：有零散 RFC 文档（`ot4-golden-review-rfc.md`、`arke-harness-v2-rfc.md`）但**无 RFC 模板/目录/编号制度**，且需 "Leon-approved" 人工闸（`verification.py:4`、`golden_ladder.py:82`）——治理绑定单人，非社区化。
- **dialect 未上游化**：MLIR dialect 停留在树内，无 upstream LLVM/MLIR 提案痕迹。
- **CI 无 GPU / 无公开 benchmark 复现**：CI 只 lint+CPU pytest；GPU 门禁全靠本地单卡 RTX 3060。gate 结果（2534 pass、ratio 数字）不可被外部 CI 复现。
- **NVIDIA-only，无 release tag**：AGENTS.md 明记 *"v1.0.0 tag: NOT cut — DEFERRED … NVIDIA-only coverage is far from release level"*。Phase 2（Ascend）paused。
- **onboarding 路径部分存在**：harness-handbook §10 有 "Onboard a new operator ≤400 LOC" 可证伪预算，是亮点；但缺 "5 分钟跑通" 的 quickstart（README 的 CLI 示例需已装好 GPU/Triton/MLIR 环境）。

---

## 亮点（Top 5）

1. **退化控制体系成熟且 fail-loud**：GoldenUnavailable 的 P5 守卫（不拿被测系统当 oracle）、memory preflight typed-skip、codegen 异常→interpreter fallback **且有测试**、watchdog 超时——降级路径覆盖面在同类研究项目中罕见地完整。
2. **缓存设计正确**：KernelCache 不按 shape 分键、把 per-shape specialization 交给 Triton，避免了缓存爆炸；per-N config cache 只 memoize 纯函数——设计意图清晰、注释翔实。
3. **N>65536 静默截断的处理树立了 shape 泛化的正面范例**：tier-3 sweep 扫出 latent bug（`c2df77b`），修复为 loop kernel + layernorm 改 fail-loud raise，且诚实标注"非 slim-launch 回归、tier-2 从未覆盖"。这正是 shape 泛化审计要的证据链。
4. **SSOT 纪律 + 分层依赖**：op 数量单源（46，runtime==doc）、lang→ir→backend/compiler→agent 单向分层（仅 1 处环）。
5. **文档密度高**：309 行 README + 5 spec + 14 architecture + benchmark 全套 + harness-handbook 可证伪 LOC 预算，对编译器开发者的 onboarding 材料充分。

## 隐患（Top 5）

1. **动态 shape 在执行层无一等公民支持**：SymbolicDim/ShaipeConstraint 停在 IR 声明层，**零命中于 backend**；无 bucketing/padding/动态维度 lowering。训练/推理变 seq_len（Llama3 8k/32k、SD 变分辨率）下，matmul 的 `autotune(key=["M","N","K"])` 对每个新 (M,N,K) 触发完整 autotune——**真实首次调用悬崖**，且无 Arke 自己的缓解（torch.compile 的 `cache_size_limit`/`dynamic=True` 修复只作用于基线路径，不作用于 Arke 模板）。
2. **JIT/autotune 编译开销无系统量化**：`benchmarks/probes/` 只测稳态 launch overhead（warmup=50），冷启动/重编译成本仅 kernel_cache docstring 粗估（~300ms/~5-10ms），无覆盖动态 shape 序列的 recompile 成本 benchmark——这正是任务要问的量化数据缺口。
3. **agent ⇄ learn 循环依赖**：真实模块级环，阻碍独立打包/测试，是需偿还的重构债。
4. **开源治理缺失**：无 CONTRIBUTING、无 CODE_OF_CONDUCT、无制度化 RFC、治理闸绑定单人（"Leon-approved"）、CI 无 GPU 不可复现 gate 结果、dialect 未上游化——对 MLIR/LLVM 社区开发者是硬门槛。
5. **NVIDIA-only + 未发版**：所有 phase closure 是单卡 RTX 3060 上的 gate pass，非 release readiness；Ascend/AMD paused；v1.0.0 明确 DEFERRED。"跨硬件性能泛化"是 README 卖点，但当前是单硬件验证。

---
*审计脚本 `/tmp/stats.py` 保留可复现。所有数字来自 AST/行数统计与 git log，未跑 CUDA。*
