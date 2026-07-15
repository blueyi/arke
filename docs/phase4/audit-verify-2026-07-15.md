# Arke Phase 4 — 独立复审报告 (audit-verify, 2026-07-15)

> **审计目标:** 独立复核 `docs/phase4/audit-2026-07-13.md` 第七节进度表中所有标记
> ✅ 的项 (A4/A5/A1/A2/A3/B1–B6/C1–C6) 是否 **真实完成、真实工作**。
> **审计原则 (承接上一轮教训):** 文档 ✅ ≠ 真实；action space 必须实测；
> 注册表声明 ≠ 实际行为；commit message 声称 ≠ diff 实际改动。所有结论附真实命令输出证据。
> **约束:** 未修改任何生产代码，仅审计 + 写报告。

---

## 0. 审计环境 (证据基线)

| 项 | 值 | 证据 |
|---|---|---|
| Python | 3.10.20 | `/home/blueyi/.venvs/arke/bin/python` |
| torch | 2.6.0+cu124, **CUDA available=True** | `torch.cuda.is_available()` |
| GPU | NVIDIA GeForce RTX 3060 Laptop (sm_86) | `get_device_name(0)` |
| triton | 3.2.0 | import |
| MLIR | `mlir-opt` @ `~/opt/mlir20/root/usr/lib/llvm-20/bin` (需 `source ~/opt/mlir20/env.sh`) | `which mlir-opt` |
| 测试收集 | **2537 collected** | `pytest --co` |
| 缺失依赖 | **fastapi / starlette / sse_starlette 未安装** | import 失败 — 影响 MCP HTTP/SSE 服务器实跑 (见 §3) |

所有第七节声称的 commit (`d3b7831 f131586 4a3996a 934634c 4fb6ec2 4df37ab f3d928c 59ffb60 1971d26 93a310d`) **均存在于 git log** —— 提交本身真实。

---

## 1. ①确认真实完成的项

| 项 | 结论 | 实测证据 |
|---|:---:|---|
| **A4** action space kind-balance | ✅ **真实** | `list_legal_actions(top_n=10)` 对 matmul/softmax/relu **均返回全部 5 种 kind** (`tile/unroll/vectorize/parallel/place`)，非旧版"只有 tile"。全空间 `top_n=1000` = 47 candidates，覆盖 i/j/k 三个 loop。round-robin `_kind_balanced_sample` 生效。 |
| **A5** state-aware 收缩 | ✅ **真实 (有限)** | 应用一个 tile 决策后，`_filter_redundant` 将该候选移除 (tile 18→17，被应用项不再出现)。**局限见 §2-G2。** |
| **A1** BackendRegistry 连线 | ✅ **真实** | `get_default_registry().list_backends()` → `['triton','mlir-gpu','cuda-c']` 三个后端全部注册成功。 |
| **A2** `arke run --backend mlir_gpu` | ✅ **真实 (需 MLIR env)** | `arke run --kernel relu --shape 1024,1024 --backend mlir_gpu` → `OK`（前提 `source ~/opt/mlir20/env.sh`，否则 `mlir-opt not found`）。matmul 同样 OK。这是环境依赖，非代码缺口。 |
| **A3** `arke run --backend cuda_c` | ✅ **真实** | 文档声称的确切命令 `arke run --kernel relu --shape 1024,1024 --backend cuda_c` → `OK: relu`。matmul 同样 `OK`。 |
| **B3** compiler/lowering/mlir_emitter 标注 | ✅ **真实** | 文件首行 `# DEPRECATED — Stage 7 early skeleton. DO NOT USE.` + 交叉引用 `arke/backend/mlir_emitter.py`。 |
| **B4** backends/mlir/ops.py 标注 | ✅ **真实** | `⚠️ HERITAGE CODE — Stage 7` + `Do NOT extend this file... use arke/backend/mlir_emitter.py`。 |
| **B6** optimize 产物 gitignore | ✅ **真实** | `.gitignore` 含 `benchmarks/results/optimize/` 与 `.../stage8/track1/optimize/`；commit `4a3996a` 删除了 5 个已提交产物。 |
| **C1** llm_soft_verify | ✅ **真实** | `llm_soft_verify(decisions, *, op_name, ..., model='gpt-4o-mini', base_url, api_key, timeout)` 签名完整；rule+LLM 两阶段 + fail-open。**7 tests pass** (mock LLM)。 |
| **C3** matmul float4 double-buffer | ✅ **真实 (perf 未独立复测)** | `cuda_c_matmul_templates.py` 含 float4 vectorized loads + double-buffered shared memory 实现 (`VEC_LOAD`, 2-stage pipeline)，N<256 自动 fallback scalar。**`test_cuda_c_backend.py` 26/26 pass**。**注:** 文档的加速数值 (+19%~+35%) 本轮**未独立重测**（缺原始 kernel-only bench 脚本），仅确认实现存在 + 正确性通过。 |
| **C4** MCP resources subscribe | ✅ **真实** | subscribe/unsubscribe + notify_resource_changed **19 tests pass**。 |
| **C5** FlagGems test isolation | ✅ **真实 (超出声称)** | **全套 `pytest tests/ --dist loadfile -n 2` → 2534 passed, 1 skipped, 2 xfailed, 0 failed** (120s)。per-file 进程隔离 confine 了 FlagGems import-time aten 劫持。实际比文档声称的 2245 passed 更多（测试量已增长）。固化进 Makefile `test` target。 |
| **C6** PlateauEarlyStop | ✅ **真实** | `PlateauEarlyStop(patience=3)` 实测：3 次连续无改善后 `should_stop` 翻 True。**6 tests pass**。PostProfile hook。 |
| **C2** FA-2 大 seq | 🟡 **诚实标注 (非 ✅)** | 文档标 🟡"技术天花板确认"而非 ✅，措辞诚实（memory-bound 设计空间限制 vs 调参）。本轮不作"完成"核验，认可其为诚实的负结论。git log 确认三次尝试+回退 (`59ffb60`, `414d4aa`, `0b71812`) 真实发生。 |

**总体测试证据:** 全套 2534 passed / 0 failed；A4/A5 30 tests、cuda_c 26 tests、C1 7、C4 19、C6 6、backend-registry 相关 150 tests 全绿。

---

## 2. ②发现的新缺口 / 回归

### 🔴 G1 — `arke run --backend triton` 完全不可用 (A1/A2/A3 隐藏缺口)

第七节 A1/A2/A3 只验证了 cuda_c（还列了确切命令），**从未验证 triton 走统一 CLI**。实测：

```
$ arke run --kernel relu   --shape 1024,1024 --backend triton  → FAILED
$ arke run --kernel matmul --shape 512,512   --backend triton  → FAILED
$ arke run --kernel softmax --shape 512,512  --backend triton  → FAILED
$ arke run --kernel add    --shape 512,512   --backend triton  → FAILED
  run error: SemanticInterpreter failed on 'relu': 'X'
```

triton 是 REGISTRY_BACKENDS 之一，但经 `_run_registry_backend()` 调用**对所有 op 都失败**。
cuda_c / mlir_gpu 正常。**根因（两层 bug，均在 `arke/agent/backends.py` 的 registry glue）：**

1. **key 大小写不匹配** (`backends.py:221`)：构建 IRNode 时 `inputs={k.lower(): ...}` 产出小写键 `x`；
   而 triton wrapper 与 interpreter 的 `ref_relu` 都取 `inputs["X"]`（大写）→ `KeyError: 'X'` / `missing argument 'X'`。
2. **张量类型不匹配** (`backends.py:242`)：即便修正键名，glue 传入 **numpy.ndarray**，triton 后端要 **torch.Tensor**
   → `empty_like(): argument 'input' must be Tensor, not numpy.ndarray`。

cuda_c/mlir_gpu 恰好容忍 numpy + 小写键，triton 不容忍。**无任何测试覆盖此 CLI 路径**
（triton codegen 本身有 6 tests 直测且通过 —— 缺口纯在 registry-run glue）。

> 影响：文档"3 个 backend 走统一 CLI"的命题**只有 2/3 成立**。triton（Phase 1 主后端）经 `arke run` 不可用。

### 🟡 G2 — A5 冗余过滤 last-write-wins，多因子同 loop 不持续收缩

`_filter_redundant` 以 `loop → factors` 建 dict，同一 loop 上应用多个 tile 时只记住最后一次。
实测连续 5 次 apply：tile 空间 `[18, 17, 17, 17, 17]` —— 第 2 次起不再收缩。
D2 原始诊断"连续 tile 不改变 action space"**部分仍在**：单因子会被去除，但同 loop 多因子历史不累积过滤。属结构性小限制，非阻塞。

### 🟢 G3 — MCP HTTP/SSE 服务器实跑本轮无法验证

`fastapi / starlette / sse_starlette` 在该环境未安装。原始审计 §一声称"34 MCP tests + 6 SSE tests"
+ "实际 HTTP 服务器启停测试" —— 本轮**无法复现 HTTP/SSE 实跑**（C4 的 19 个 subscribe 单测不依赖 HTTP，已通过）。
非 ✅ 项回归，但记录：该环境不足以复核 SSE/HTTP 通路声称。

---

## 3. ③文档与实现不一致处

### 🔴 D1 — commit `4a3996a` message 谎报 AGENTS.md 改动 (B1 + B2 **实际未做**)

commit `4a3996a` 的 message 明写：
> "B1: AGENTS.md Phase 1 stages S6-S9 updated from '⬜ ← current' to ✅/✅ CLOSED"
> "B2: AGENTS.md integration/ description → 'torch_bridge (G8 PyTorch 桥接)'"

但 **`git show --stat 4a3996a` 显示 AGENTS.md 根本不在改动文件列表中**（只改了 .gitignore、
pyproject.toml、删除 optimize 产物）。实测当前 AGENTS.md：

- **B1 未做**：无任何 stage 状态/CLOSED 表；`grep -i current/CLOSED/S6/roadmap` 无命中；
  文件末尾仍 `*Last updated: 2026-04-05*`。触碰 AGENTS.md 的最近 commit 是 `911bd9d`（op 改名），非 `4a3996a`。
- **B2 未做**：`AGENTS.md:63` 仍为 `arke/integration/ — KernelCache, PyTorch integration`
  —— 正是 B2 声称要修掉的"falsely listing nonexistent KernelCache"原文；`torch_bridge` 在 AGENTS.md 中零命中。

> **第七节把 B1、B2 标 ✅ 是虚假的** —— commit message 描述了从未落盘的编辑。这是本轮最严重的
> "文档 ✅ ≠ 真实"发现，与上一轮教训同型。

### 🟡 D2 — B5 标 ✅"已清理"不准确

第七节 B5：删除 pyproject.toml 旧注释 `# arke = "arkec.main:cli"`。
实测 `pyproject.toml:66` 仍在，仅改写为 `# arkec = "arkec.main:cli"  # historical alias...`。
**注释被改写而非删除** —— 与"已清理"不符。属低危，但仍是 doc/impl 不一致。

### 🟢 D3 — A2 依赖外部 MLIR 环境，文档未注明

A2 标 ✅ 但默认环境下 `arke run --backend mlir_gpu` 直接 `FAILED: mlir-opt not found`。
必须 `source ~/opt/mlir20/env.sh` 才 OK。文档 A2 行未记录此运行前置，易误导。

---

## 4. 复审结论汇总

| 分类 | 项 |
|---|---|
| ✅ 真实完成 | **A4, A5, A1, A3, B3, B4, B6, C1, C3(perf 未复测), C4, C5, C6** (12 项) |
| 🟡 诚实标注/有限 | **C2**(🟡 本就未标 ✅), **A2**(真实但需 MLIR env), **A5**(有 G2 限制) |
| 🔴 标 ✅ 但实际未做 | **B1, B2**（commit message 谎报 AGENTS.md 改动，文件未改） |
| 🔴 新缺口/回归 | **G1** `arke run --backend triton` 全 op 失败（A1/A2/A3 未测的 1/3 后端）|
| 🟡 次要不一致 | **D2** B5 注释未删仅改写；**D3** A2 环境前置未记；**G2** A5 多因子不累积；**G3** MCP HTTP/SSE 本轮不可复核 |

**核心判断:** 第七节 16 项中 **12 项真实**、**2 项 (B1/B2) 虚假标 ✅**、并新发现 **1 个未被任何测试
覆盖的真实功能缺口 (G1: triton CLI dispatch broken)**。上一轮的教训在本轮**再次应验**：
commit message 声称的改动可能从未落盘（B1/B2），已声称"连线"的后端可能只测了子集（G1 triton）。

**修复优先级建议（未改代码，仅建议）:**
- P0 — G1：修 `backends.py` registry glue（key 保留原大小写 + numpy→torch 转换），并补一个
  `arke run --backend triton` 的端到端测试。
- P1 — B1/B2：真正编辑 AGENTS.md；修正第七节把 B1/B2 从 ✅ 降级。
- P2 — D2/D3/G2 文档/注释同步。

---

## 附：关键命令输出摘录

```
# 全套绿基线 (Makefile test target)
$ pytest tests/ --dist loadfile -n 2 -q
  2534 passed, 1 skipped, 2 xfailed, 60 warnings in 120.87s

# A4 action space — 5 kinds present
$ list_legal_actions(matmul, top_n=10) → {'tile':2,'unroll':2,'vectorize':2,'parallel':2,'place':2}
$ list_legal_actions(matmul, top_n=1000) → 47 candidates, loops={i:12,j:12,k:12,...}

# A1 registry
$ get_default_registry().list_backends() → ['triton','mlir-gpu','cuda-c']

# A3 / A2 / G1 CLI dispatch
$ arke run --kernel relu --shape 1024,1024 --backend cuda_c   → OK
$ arke run --kernel relu --shape 1024,1024 --backend mlir_gpu → OK (after sourcing MLIR env)
$ arke run --kernel relu --shape 1024,1024 --backend triton   → FAILED: SemanticInterpreter failed on 'relu': 'X'

# D1 — commit lied about AGENTS.md
$ git show --stat 4a3996a → .gitignore, pyproject.toml, optimize/* deleted  (AGENTS.md NOT listed)
$ grep torch_bridge AGENTS.md → (no match)
$ grep "Last updated" AGENTS.md → *Last updated: 2026-04-05*
```

---

## 5. 待开发任务清单（重要性 + 完备度排序，待 Leon 确认后新 session 逐一推进）

> **排序原则:** 按 **重要性 × 完备度** 排，**不以开发周期长短作为降级理由**。
> 一个多周项目若触及 Arke 核心命题（"高性能、可泛化 kernel + bounded action space"），
> 其优先级应高于一个半小时的文档订正——因为它决定项目立论是否成立。
> 每个任务附：缺口性质、根因/现状、完成标准（DoD）、验证方式。

### 🔴 T1 — Tensor-Core fused attention 达到性能可用（核心命题缺口）

**为什么最高优先:** Arke 的立论是"生成高性能 kernel"。attention 是 LLM 时代最关键的算子，
当前 fp32 warp-per-row 在大 seq 上仅 0.15–0.18× SDPA，且已被证明是该设计空间的天花板
（`audit-2026-07-13.md` C2 行）。TC 路线是唯一能缩小差距的方向。**这不是"增强"，
是核心命题能否成立的验证。** 周期长（多周）不构成降级理由。

- **现状:** prototype 已跑通（`docs/phase5/c2-tensorcore-attention-2026-07-15.md`）——
  wmma fp16→fp32 correctness max_err 2.4e-4（富余 40×），路线可行性已验证。
  但 kernel-only 仅 0.06–0.3× SDPA，瓶颈精确定位为 **smem 82KB/block → 1 block/SM → 12.5% occupancy**。
- **DoD（分阶段，全部落地才算完成）:**
  1. 降 smem ≤ 48KB（去 Otile 独立缓冲 / 复用 Ssh 区）→ 2 blocks/SM，占用翻倍，实测提速。
  2. `cp.async` double-buffer K/V，重叠 load 与 TC 计算。
  3. register-resident O accumulator，消除 Otile→Osh smem 往返；减少 `__syncthreads`。
  4. 拿到 GPU counter 权限后用 `ncu` 实测 occupancy / stall breakdown 验证假设（当前是推算）。
  5. 扩展 config：任意 D(64/128) + causal mask + GQA。
  6. 接回 arke emitter：`emit_cuda_c_flash_attention` 的 TC 变体，走统一 backend 路径 + 测试。
- **验证:** 每阶段 kernel-only bench vs SDPA fp16，correctness max_err < 1e-2 保持；
  最终目标是在大 seq（S≥1024）上把 0.15× 提升到有意义的量级（阶段性里程碑而非一次到位）。
- **参考:** 报告第 9 节按杠杆排序的六步路线。

### 🔴 T2 — 修复 `arke run --backend triton` 全 op 失败（G1，真实功能回归 + 测试盲区）

- **现状:** triton（Phase 1 主后端）经统一 CLI 对所有 op 失败。根因两层，均在
  `arke/agent/backends.py` 的 registry glue：
  1. `backends.py:221` 键名小写化 `k.lower()` → triton wrapper/interpreter 取 `inputs["X"]`（大写）→ `KeyError`。
  2. `backends.py:242` 传 `numpy.ndarray`，triton 后端要 `torch.Tensor`。
- **DoD:** glue 保留原始键大小写 + 按后端做 numpy→torch 转换；**新增端到端测试**覆盖
  `arke run --backend triton`（此路径当前零覆盖，是缺口潜伏根因）。
- **验证:** relu/matmul/softmax/add × triton CLI 全部 `OK`；新测试跑绿；全套 `make test` 无回归。

### 🔴 T3 — 打假虚假 ✅ 并真正补做 AGENTS.md（B1+B2 / D1，诚信问题）

- **现状:** commit `4a3996a` 的 message 声称改了 AGENTS.md（B1 stage 状态、B2 torch_bridge），
  但 `git show --stat` 显示 AGENTS.md 根本未被 touch。文件仍是旧的 `KernelCache, PyTorch integration`
  + `Last updated: 2026-04-05`。**第七节把 B1/B2 标 ✅ 是虚假的。**
- **DoD:**
  1. B1：AGENTS.md Phase 1 stages S6-S9 更新为 ✅ CLOSED + roadmap 含 Phase 3/4。
  2. B2：`AGENTS.md:63` 的 `KernelCache, PyTorch integration` → `torch_bridge (G8 PyTorch 桥接)`。
  3. 更新 `Last updated`。
  4. 修正 `audit-2026-07-13.md` 第七节：B1/B2 从 ✅ 降级标注"commit message 谎报、实际未做"，附真实补做 commit。
- **验证:** `grep torch_bridge AGENTS.md` 命中；`grep KernelCache AGENTS.md` 不再命中旧描述；进度表如实。

### 🟡 T4 — A5 冗余过滤多因子累积收缩（G2，action space 完备度）

- **现状:** `_filter_redundant` 以 `loop → factors` 建 dict，last-write-wins，同一 loop 多个 tile
  只记最后一次。实测连续 5 次 apply：tile 空间 `[18,17,17,17,17]`，第 2 次起不再收缩。
  D2 原始诊断"连续 tile 不改变 action space"部分仍在。
- **DoD:** 改为累积历史过滤（同 loop 已应用的所有因子都从后续候选中移除）。
- **验证:** 连续 apply N 次 tile，action space 单调收缩；补测试断言收缩序列；A4/A5 原 30 tests 无回归。

### 🟡 T5 — B5 真正删除 pyproject 注释 + A2 环境前置补文档（D2/D3）

- **D2:** `pyproject.toml:66` 历史 alias 注释按 B5 原计划**删除**（当前只是改写为 `# arkec = ...`）。
- **D3:** 在 `audit-2026-07-13.md` A2 行注明 `arke run --backend mlir_gpu` 需先
  `source ~/opt/mlir20/env.sh`，否则 `mlir-opt not found`。
- **验证:** `grep 'arkec.main:cli' pyproject.toml` 不再命中；A2 行含环境前置说明。

### 🟢 T6 — 补 MCP HTTP/SSE 依赖并复核（G3，工程卫生）

- **现状:** `fastapi / starlette / sse_starlette` 未安装，原始审计 §一声称的"34 MCP tests + 6 SSE +
  实际 HTTP 服务器启停测试"本轮无法复核（C4 的 19 个 subscribe 单测不依赖 HTTP，已过）。
- **DoD:** 安装依赖（纳入 dev deps），实跑复核 HTTP/SSE 通路，确认声称属实或记录真实缺口。
- **验证:** SSE/HTTP 相关测试实跑通过并留输出证据。

### 🟢 T7 — C3 matmul 加速数值独立复测（完备度补证）

- **现状:** 复审确认 float4 double-buffer 实现存在 + 26 tests 通过，但文档声称的 +19%~+35%
  加速本轮未独立重测（缺原始 kernel-only bench 脚本）。
- **DoD:** 补/找回 kernel-only bench 脚本，实测复现 N=256/512/768/1024 的加速数值，坐实或修正文档数字。
- **验证:** bench 脚本可复现，数值写回文档并附 CUDA-event 原始输出。

### 🟢 T8 — 分支整理 + scratch 归置（工程卫生，收尾）

- **现状:** 复审报告 / C2 / RL 三件不相干的事 commit 混在同一分支 `phase5/rl-pipeline-deepening`；
  `scratch/`（含 TC prototype）未跟踪。
- **DoD:** 按主题理顺分支归属（审计/C2-attention/RL 各自独立），决定 scratch 内容去留（TC prototype
  应正式纳入 `scratch/tc_attn/` 或迁入正式目录并跟踪）。
- **验证:** 分支职责清晰，无孤儿 commit，scratch 明确归属。

### 优先级总览

| 序 | 任务 | 性质 | 依据 |
|:---:|---|---|---|
| **T1** | Tensor-Core attention 性能可用 | 🔴 核心命题缺口 | 决定"高性能 kernel"立论是否成立，最重要 |
| **T2** | triton CLI dispatch 修复 | 🔴 真实功能回归 | 主后端经统一 CLI 全挂 + 零测试覆盖 |
| **T3** | AGENTS.md 打假 + 补做 | 🔴 诚信问题 | commit message 谎报，虚假 ✅ 必须纠正 |
| **T4** | A5 多因子累积收缩 | 🟡 action space 完备度 | bounded action space 命题的完整性 |
| **T5** | B5 删注释 + A2 环境文档 | 🟡 doc/impl 一致 | 收尾订正 |
| **T6** | MCP HTTP/SSE 依赖复核 | 🟢 工程卫生 | 未复核声称的通路 |
| **T7** | C3 加速数值复测 | 🟢 完备度补证 | 坐实性能数字 |
| **T8** | 分支整理 + scratch 归置 | 🟢 工程卫生 | 最后收尾 |

> **推进方式:** Leon 确认后另起新 session，从 T1 开始逐一推进（T1 为多周长线，可与 T2/T3 并行起步）。

---

*Generated 2026-07-15. Independent re-audit + prioritized backlog. No production code modified.*
