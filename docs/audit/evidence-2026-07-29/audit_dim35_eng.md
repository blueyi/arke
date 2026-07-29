# Arke 审计报告 — 维度三（动态Shape鲁棒性/退化控制）+ 维度五（代码规范/文档/开源就绪）

审计对象：`/home/blueyi/workspace/repos/arke`（AI 编译器工具链）
审计环境：venv `~/.venvs/arke`（Python 3.10.20），RTX 3060 Laptop 6GB / sm_86 / CUDA
审计原则：仅报亲眼所见事实，不编造。

---

## 一、动态 Shape 鲁棒性 + 退化控制（维度三）

### 1.1 Performance Cliff 实测数据（亲测 CSV）

数据集：`benchmarks/results/dynamic_shape/2026-07-29_191225/`（fp16，warm_reps=50）。
测量方式（读 `benchmarks/dynamic_shape.py`）：通过**生产 wrapper** `KERNEL_CACHE.get_or_build_by_op(op)`（TritonBackend 实际使用的同一对象）驱动一串 shape，逐 shape 记 `first_call_ms`（冷）/ `steady_ms`（50 次热中位数）/ `cliff_ratio = 冷/热`。用 `perf_counter`（非 CUDA event），因为 cliff 成本主要在 host 端（config 选择 + Triton JIT 编译）。

`summary.json` 实测汇总：

| op | shapes | cliff geomean | median | max | new-spec geomean | same-spec geomean |
|:--|--:|--:|--:|--:|--:|--:|
| matmul | 15 | **3.31×** | 2.33× | 27.9× | 3.66× (13) | 1.71× (2) |
| softmax | 12 | **40.99×** | 68.7× | 130.7× | 51.5× (11) | 3.34× (1) |
| rmsnorm | 11 | **7.22×** | 6.39× | 86.4× | 77.2× (2) | 4.27× (9) |

**逐行冷编译成本（CSV 实测，绝对毫秒数）：**

- **softmax**（每个新 seq-len 都付编译）：n256 冷 6.11ms/热 0.047ms（131×）；n384 4.43ms（81×）；n700 3.83ms；n1024 3.44ms；n2048 3.55ms；n4096 3.95ms。**每个新 N 类几乎都付 3.5–6ms 编译**。唯一 same-spec 行 n512 落到 blk512 已编译类 → 冷 0.18ms（3.3×）。
- **matmul**（bucket 缓解）：多数新 shape 落 2–3×（m2=1.8× m7=2.3× m32=3.0× m48=2.4×）；m64 与 m48 共享 cfg64 bucket → same-spec 2.6×；只有偶发新 bucket 首触发大值（m1=24×、m16=28×、m200=10×）。geomean 仅 3.3×。
- **rmsnorm**：cliff 集中在头两个 shape（m128=86×、m200=69×），此后落平 2–7×；因其 kernel 只按 divisibility 特化 M，`spec_key` 预测吻合。

### 1.2 对 Llama3 / Stable Diffusion 等动态 shape 模型的实际影响

- **Llama3 变长解码 = 最坏场景。** 逐 token 解码时 seq-len 单调递增，softmax（attention logits）对**每个新 N 类**付 3.5–6ms 编译。一个 prefill/decode 序列若跨越 8 个 next_pow2(N) 类，累计约 **30–50ms 纯编译墙**，全落在延迟敏感的首 token 路径上。这是 softmax geomean 41× 的直接后果——文档 `dynamic-shape-cliff.md §3` 明确指认"在 token-by-token decode loop 中这是主导的动态 shape 成本"。
- **matmul（FFN/投影）风险低。** K-H3.1 next_pow2 bucket 让相邻 M 塌缩进同一 cfg，且喂给 Triton JIT key 的是 tile constexpr 而非原始 M——大量新 shape 复用已编译 kernel（geomean 3.3× ≈ launch 噪声量级）。对 Llama 的 token-batch matmul 基本免疫。
- **rmsnorm 对 seq-len 近似不变**（除首次），对变长模型友好。
- **Stable Diffusion 空间分辨率扫掠**同理落在 softmax/attention 那条曲线上：每个新分辨率类可能触发编译。文档同时指出 attention（K-ATT）是"动态 shape 最要命的 op"，其 gate 尚未落地。

### 1.3 退化 / 缓解机制（实证）

**A) 解释器退化（真正的 fallback 安全网）** — `arke/backend/triton_backend.py`：
- `_NodePlan` 有 `use_interpreter` 位；`lower()` 三级退化链：
  1. 未知 op（KeyError）→ 标记 interpreter fallback；
  2. `op.template_hint is None`（无 Triton kernel，如 scatter）→ interpreter fallback；
  3. 真实 codegen 抛异常 → 记录 error、fall back 到 `arke.ir.ops.interpreter.INTERPRETER.execute`。
- 统计 `num_real` / `num_fallback`。**这是正确性安全网，不是性能退化**——保证任意图都能跑，代价是慢路径。

**B) Bucket cache（把 cliff 变有限而非无界）：**
- matmul `matmul.py.j2`：`_TILE_CFG_CACHE`（dict，key = `(next_pow2(M), next_pow2(N), next_pow2(K))`），冷成本降到"每 bucket 一次 Triton 编译"，config 由 bucket 维度确定性推导（`_mm_cfg`，含 2026-07-28 离线 fp16 sweep 蒸馏出的 tile 表）。带 `_cfg_override` 逃生口供 agent 覆盖。
- flash_attention `flash_attention.py.j2`：**同款模式已就位** `_FA_CFG_CACHE` + `_fa_cfg(Nb, Db, causal)`，key = `(next_pow2(N), BLOCK_D, causal)`，也带 `_cfg_override`。说明 attention 已按 matmul 的 bucket 套路做，是缓解 softmax 式 cliff 的正确方向。
- softmax **无此 bucket 缓存**（launch config 按精确 N 选，BLOCK 由 next_pow2(N) 推），故 cliff 最陡——这是数据与代码一致的根因。

**C) 编译预算/早停控制** — `arke/agent/extensions.py::PlateauEarlyStop`：
- PostProfile hook，连续 `patience`（默认3）次 `baseline_ratio` 无 `min_improvement`（默认0.01）改进 → `should_stop=True`。
- **注意其定位**：控制 agent **优化循环**的编译次数预算（避免无效反复 compile-profile），**不是运行时对 JIT cliff 的退化**。与运行时动态 shape cliff 是两个不同层面的"退化控制"。

**D) 治理定位（诚实）** — `dynamic-shape-cliff.md`：该 track 是 **D1 measure-only（2026-07-29 批准）**，只出曲线不设 pass/fail gate（`test_no_gate_threshold_in_module` 硬守卫）。即 **cliff 已被量化但尚未被门禁约束**；系统对 JIT 过大开销目前"测量 + bucket 缓解 + 解释器兜底正确性"，尚无自动切换到预编译/AOT 或降级 kernel 的运行时策略。

### 1.4 维度三结论
- ✅ 有真实、可复现的 cliff 量化基础设施 + 数据（不是空谈）。
- ✅ matmul/attention 有 bucket cache 把无界 cliff 收敛成有限"每 bucket 一次编译"。
- ✅ 有解释器 fallback 保证任意图正确性；有 PlateauEarlyStop 控编译预算。
- ⚠️ **softmax（及一般 row-scan）cliff 仍陡**（geomean 41×，max 131×），对 Llama3 变长解码首 token 路径有实质延迟冲击；缓解模式（bucket cache）已在 matmul/attention 落地但**尚未套到 softmax**。
- ⚠️ 只 measure-only，**无运行时"JIT 太贵就退化"的自动策略**（如 AOT/预编译常见 shape 集），也无 cliff gate 门禁。

---

## 二、代码规范 / 模块解耦（维度五之一）

### 2.1 模块 LOC 分布（实测）

| 模块 | LOC | py 文件数 | 备注 |
|:--|--:|--:|:--|
| arke/backend | 18,490 | 32 | **重心**，含 4 个后端 codegen |
| arke/agent | 7,017 | 15 | agent session/tools/prompts/optimize |
| arke/ir | 4,205 | 17 | SemanticIR + StrategyIR |
| arke/compiler | 1,918 | 14 | 编译管线 |
| arke/learn | 1,810 | 7 | trajectory/KB |
| arke/lang | 1,079 | 5 | .ak parser |
| arke/integration | 203 | 2 | torch_bridge |
| arke/backends（复数） | 235 | 4 | **遗留 S7 MLIR seam** |
| arke（合计） | ~35,249 | — | |
| benchmarks | 21,367 | 75 | |
| tests | 26,380 | 167 | |

### 2.2 arke/backend/ 32 文件组织评估

按前缀天然分四组，命名规范一致：
- `cuda_c_*.py`（10 个：backend/attention/rowwise/exotic/matmul_templates/final5/extra/gated/movement）
- `llvm_*.py`（9 个：attention/rowwise/dense/fused/elementwise/emitter/backend/wmma/matmul_f4）
- `mlir_*.py`（4 个：emitter/ops/gpu/backend）
- `triton_*.py`（2 个：backend/codegen）+ 模板 `triton_templates/*.j2`
- 共享层：`protocol.py`、`kernel_cache.py`、`hardware.py`、`gpu_tuning.py`、`strategy_to_transform.py`、`mock_backend.py`

**评估：解耦合理，非散乱。** 每后端一组，按算子族（attention/dense/rowwise/fused/elementwise/movement）水平切分单个后端，避免单文件膨胀。最大文件 `mlir_emitter.py` 2992 LOC 偏大（唯一接近 god object 的），但属 emitter 生成逻辑集中，非跨职责杂糅。

### 2.3 后端接口统一度（读 protocol.py）

**优秀。** `ArkeBackend` 是 4+2 方法 `Protocol`（`@runtime_checkable`）：`lower/compile/run/supports_op/capabilities`，配 `BackendArtifact`/`CompiledKernel`/`BackendCapabilities` 数据类。
- K-H2 统一了 `lower(graph, hw=None)` 签名，杜绝 per-backend 私有参数泄漏到调用点。
- `BackendCapabilities`（tensor_core/async_copy/max_pipeline_stages/supported_dtypes）供 engine 在生成期裁剪合法动作空间——接口与"AI 决策"耦合设计到位。
- `BackendRegistry` + `get_default_registry()`：Triton/MLIR-GPU/CUDA-C/LLVM 四后端按 target 字符串注册，缺失后端 import 静默跳过。**新增后端 = 实现 Protocol + register，无需核心重构**——这是明确保留的扩展 seam。

### 2.4 循环依赖 / god object（AST 实测）

包级 import 图（`/tmp/circdep.py` 亲测）：
```
arke.agent   -> [backend, compiler, ir, learn]
arke.backend -> [ir]
arke.compiler-> [ir, lang]
arke.ir      -> [lang, version]
arke.learn   -> [agent]
arke.cli     -> [agent, compiler, ir]
arke.integration -> [backend, ir]
```
- **分层基本干净**：ir/lang 是底座，backend/compiler 依赖 ir，agent 在顶层。
- **发现 1 个包级 2-循环：`arke.agent ↔ arke.learn`**（agent→learn 且 learn→agent）。属轻度耦合气味，建议抽公共接口或反转依赖方向。
- 无 god object 级单文件（最大 mlir_emitter 2992 LOC 属可接受生成器）。

### 2.5 双 backend 目录气味
存在 `arke/backend`（单数，主力，18.5K）与 `arke/backends/mlir`（复数，235 LOC，S7 遗留 MLIREmitter seam），后者仍被 `tests/test_mlir_backend.py` 与 `arke/compiler/lowering/mlir_emitter.py` 引用。**命名近似易混淆**（单复数），建议合并或明确文档标注其历史定位。此外 `AGENTS.md` 架构图列了 `arke/engine`、`arke/parser` 两个目录，**实际不存在**（parser 在 `arke/lang`，engine 逻辑并入 agent/compiler）——文档与代码有漂移。

### 2.6 测试基线（亲跑 `make test`）
**2862 passed, 1 skipped, 2 xfailed, 64 warnings — 54.06s，EXIT=0。** 与任务预期 ~2862 一致，全绿。warnings 为 Triton deprecation + FlagGems aten 劫持（已用 `--dist loadfile -n` 隔离，pyproject 有详注）。

---

## 三、文档完备度 + 开源就绪度（维度五之二）

### 3.1 docs/ 组织（78 篇 md 实测）

| 子目录 | md 数 | 内容 |
|:--|--:|:--|
| architecture | 15 | 设计理念 / e2e-flow / compiler-infra / lang&ir 设计 rationale |
| benchmark | 11 | BL/OT/ST/L 框架、ops SSOT、**dynamic-shape-cliff** |
| phase1 | 10 | Phase1 收尾 + AI-native review |
| spec | 7 | **arke-lang-spec / arke-ir-spec（canonical 契约）** |
| phase4 | 7 | CUDA-C 审计/验证 |
| phase5 | 6 | LLVM IR |
| audit | 5 | 历次工程审计 |
| plans | 3 | |
| kestrel | 3 | KESTREL 审计 / k-att-plan |
| phase3 | 2 | MLIR |
| roadmap | 1 | plan.md（gate/stage SSOT）|
| （根） | 2 | project-audit / nvgpu-dialect-research |

外加 `docs_zh/`（中文文档树）。组织清晰、SSOT 意识强（ops 目录、gate 映射均单源）。

### 3.2 开源必备文件检查（实测）

| 文件 | 状态 |
|:--|:--|
| README.md | ✅ 309 行，含 Overview/Features/Architecture/Minimal Example/Quick Start(Prereq/One-Click/Manual `pip install -e .[dev]`)/CLI/Roadmap/Documentation Guide/Project Structure/License |
| LICENSE | ✅ Apache License 2.0（完整全文）|
| pyproject.toml | ✅ 规范：name/version 0.1.0/license Apache-2.0/deps 分组(gpu/mlir-gpu/agent/dev)/ruff+mypy+pytest 配置/`[project.scripts] arke`/urls 指 github.com/arke-lang/arke |
| **CONTRIBUTING** | ❌ **缺失**（`search_files` 0 命中）|
| 独立开发者指南 / getting-started / API 文档 | ❌ 无专门文件（`find` 无 guide/api/getting 命中）；开发入门散落在 README Quick Start + AGENTS.md + docs/architecture |
| requirements.txt / requirements-benchmark.txt | ✅ 存在 |
| .github/ | ✅ 存在（CI 目录）|

### 3.3 能否吸引 LLVM/MLIR/IREE 社区开发者

**优势：**
- 接口规范扎实：`protocol.py` 4 方法 Protocol + Registry 是清晰扩展点，MLIR/LLVM 后端已作为实证存在（llvm_* 9 文件、mlir_* 4 文件），对该社区有直接吸引力。
- 设计理念文档完备：`arke-lang-spec-design.md`/`arke-ir-spec-design.md`/`e2e-flow.md`/`arke-compiler-infrastructure.md` 提供"为什么这么设计"的 rationale，非仅 API 罗列。
- Semantic/Strategy IR 分离 + `@rationale` + bounded action space 是有辨识度的设计主张，README 表达清楚。

**短板（阻碍外部贡献）：**
- ❌ **无 CONTRIBUTING.md** —— 外部开发者无从知晓 PR 流程、编码约定、测试要求、gate 治理规则（gate 锁定规则藏在 AGENTS.md，非贡献者视角）。
- ❌ **无面向用户的 API 文档 / 独立开发者指南** —— 只有 spec（契约）与内部审计文档，缺"如何写一个新算子 / 新后端"的 how-to 教程（虽 protocol.py docstring 有提示）。
- ⚠️ **强内部化痕迹**：AGENTS.md/CLAUDE.md/SOUL.md/INBOX.md/memory 等 agent 开发流程文件、phase*/audit/kestrel 大量内部里程碑文档，对外部读者是噪声，且暴露"单机 NVIDIA-only、v1.0.0 未 tag、Ascend paused"的未成熟状态。
- ⚠️ 文档与代码漂移（AGENTS.md 提到不存在的 arke/engine、arke/parser；双 backend 目录），会让新贡献者困惑。
- ⚠️ 版本 `Development Status :: 2 - Pre-Alpha`、`v1.0.0 tag: DEFERRED`——诚实但表明尚未到"社区可依赖"的发布线。

### 3.4 维度五结论
- ✅ 代码组织、后端接口统一度、测试基线（2862 全绿）达到高质量工程标准。
- ✅ 核心文档（README/spec/architecture/benchmark/LICENSE Apache-2.0/pyproject 打包）齐备且专业。
- ⚠️ 一个包级循环依赖（agent↔learn）、双 backend 目录命名歧义、AGENTS.md 架构漂移，属可控技术债。
- ❌ **开源就绪的关键缺口：无 CONTRIBUTING、无对外开发者指南/API 教程**；大量内部 agent 流程文档未剥离；处 Pre-Alpha/NVIDIA-only，未 cut release tag。**当前更像"内部研发中"而非"社区可加入"状态。**

---

## 附：关键真实数据速查
- 动态 shape cliff（实测）：softmax geomean **40.99×** / max **130.71×**（每新 seq-len 付 3.5–6ms 编译）；matmul geomean **3.31×**（bucket 缓解）；rmsnorm geomean **7.22×**。
- 测试基线（亲跑）：**2862 passed / 1 skipped / 2 xfailed / 54.06s / EXIT=0**。
- 循环依赖（AST 实测）：**arke.agent ↔ arke.learn**（唯一包级 2-cycle）。
- 开源文件：README ✅、LICENSE(Apache-2.0) ✅、pyproject ✅、CONTRIBUTING ❌、开发者指南/API doc ❌。
