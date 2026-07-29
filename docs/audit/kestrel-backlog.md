# KESTREL 审计待办 — 任务卡索引（新 session 开发入口）

**代号**: KESTREL（2026-07-27 综合审计）— 仅用于 Leon↔Kitty 交互跟踪与文档，
**不进入真实代码**（代码注释/commit message 只引用 docs/audit/ 路径，不写代号）。
**生命周期**: 全部任务卡完成后，本文件与所有 KESTREL 描述信息一并删除
（审计报告的技术内容归档保留，代号字样清除）。
**审议材料**: `docs/audit/2026-07-27-comprehensive-audit.md`（总报告）+ 同目录 3 份子审计（含文件:行号证据）
**审计 commit**: `f0d7755`；H4 修复 commit: `0bdf918`

## 任务卡状态总览

| 卡 | 内容 | 优先级 | 状态 | 预估 |
|:--|:--|:--:|:--:|:--:|
| K-H4 | 无数据 op 假满分 → score=None + no_data_ops | P0 | ✅ DONE (2026-07-27) | — |
| K-H3.1 | matmul autotune key 改 bucketed + 首调开销 probe | P1 | ✅ DONE (2026-07-28, R4) | 1-2d |
| K-H5.2 | ArkeEnv trajectory → 收敛曲线 CSV (`--emit-convergence-csv`) | P1 | ✅ DONE (2026-07-28) | 0.5d |
| K-ATT | attention flash-style 模板（online-softmax + K/V 双缓冲 + TC dot） | P1(性能主线) | ✅ **GATES PASSED** (2026-07-29)：FA 0.301→**0.846**（阶段0.35✅ 最终0.50✅），GQA 0.172→**0.802**，xattn 1.081 | 2-4w |
| K-H1 | 双 IR 统一：IRGraph `from_semantic()` 官方构造器 + 往返 golden 测试 | P2 | ✅ DONE (2026-07-29) | 1w |
| K-H2 | 显式 HardwareModel 抽象 + `lower()` 签名统一 + capabilities() | P2(Ascend 恢复前必做) | ✅ DONE (2026-07-29) | 1-2w |
| K-H5.1 | Schedule/Instruction IR 诚实降格（spec 标注 Phase-future） | P3 | ✅ DONE (2026-07-29)，H5a 已批：spec 标注实现状态，不真接降级 | 需 Leon 定方向 |
| K-DYN | dynamic-shape bench track（首调+稳态曲线） | P3 | ✅ DONE (2026-07-29)，D1 measure-only 已批：只跟踪曲线不设 gate | 3-5d |
| K-XATT | cross_attention flash-attn varlen API 评估（OT4 golden 后续） | P3 | ✅ DONE (2026-07-29)，X1 换血已批落地：golden→flash-attn，暴露真实差距 ~0.5-0.6× | 1-2d |

**K-H3.1 note (2026-07-28, 4 轮迭代收官)**: bmm/grouped_matmul 已经不用 `@triton.autotune`（launcher-side heuristic），K-H3.1 只落到 matmul.py.j2。
迭代史（全部诚实入档，证据 `benchmarks/probes/results/kh31_acceptance_2026-07-28.md`）：
(R1) kernel-arg bucket key → 3 个额外 launch args 破坏 slim-launch，tiny/gpt2 回退 59-77%，弃。
(R2) runtime do_bench sweep per bucket → <100μs kernel 上噪音选错 tile（gpt2-c_proj 三轮稳定 0.30-0.42），弃。
(R3) fp32 离线 sweep 蒸馏 heuristic → **dtype 错误**：bench harness 跑 fp16，fp32/fp16 tile 最优解截然不同，弃。
(R4 落地) fp16 离线 sweep（300-iter×3-pass 中位）蒸馏 `_mm_cfg` + bucket memo。
**验收方法修正**：跨日历史 ratio 对比在本机无效（6GB 笔记本卡 eager 基线自身跨日漂移 2-4×，tiny eager 49.9μs↔13μs），改用**同日同钟窗 A/B**（原 autotune 模板 vs R4）：tiny 2.36× 快、gpt2-c_proj 1.40× 快、square-1k 持平 1.02×——**逐 shape 无回退**。悬崖：原每新 shape 全扫 10-13s → R4 冷成本=1 次 Triton compile（零扫描）。
已知边界：小-M shape 的 Arke python wrapper 有 ~25μs dispatch 下限 vs eager C++ 路径 ~13μs，为架构级 launch-overhead gap，tile 无法弥合，另行跟踪。30 unit tests 绿。
**K-H5.2 note (2026-07-28)**: commit `bc7d7b1`. 首批 3 op 收敛曲线（matmul/softmax/flash_attention）落 `benchmarks/results/convergence/`。

**K-H1 note (2026-07-29)**: `IRGraph.from_semantic(sem, strategy=None, *, dim_bindings)` 官方构造器落 `arke/ir/graph.py`（唯一 SemanticIR→IRGraph 路径）+ dtype 词表桥 `semantic_dtype_to_graph`/`graph_dtype_to_semantic` + 便捷工厂 `IRGraph.single_node(op, shapes)` + 反向 `to_semantic()`。往返 golden `tests/test_from_semantic_roundtrip.py`（60 tests，含全 45 SSOT catalog op + 结构 + 符号维 resolve + 多输出）。散落 ad-hoc 单节点构造点收编：`agent/backends.py`、`agent/tools.py`（profile+verify）、`integration/torch_bridge.py`——各自私有 input-mapping/dtype 逻辑消除。mlir_gpu MLA/paged preprocess 是 op 语义变换（非通用转换）故保留。doc 同步 `arke-compiler-infrastructure.md §7.6`。make test 2824 pass（唯一 fail=test_benchmark_stable_across_iters perf flaky，单跑绿，与本卡无关）。

**K-H2 note (2026-07-29)**: 显式 `HardwareModel` 抽象落 `arke/backend/hardware.py`（内存层级树 MemoryLevel + 同步域 SyncDomain + 计算单元 ComputeUnit + 对齐约束 AlignmentConstraints + `nvidia_sm86()` 实例 + DEFAULT_HARDWARE），统一/取代散落的 HardwareProfile+GPUProfile+chip 字符串。`protocol.py` 加 `BackendCapabilities` + `default_capabilities()` + 统一 `lower(graph, hw=None)` 签名 + `capabilities()` 方法。4 后端（triton/mlir-gpu/cuda-c/llvm）+ mock/mlir_backend 全部实现统一签名，各自如实上报 TC/async/stages（triton TC✓4stage、cuda-c TC✓3stage、mlir-gpu TC-optin、llvm TC✗1stage）。engine 动作空间生成器 `ArkeEnv.hw_model` + `list_legal_actions` 用 `has_tensor_core()` gate wmma_tile（非 TC 硬件 0 候选，TC 硬件 52 候选）。测试 `tests/test_hardware_model.py`（12 tests）。doc 同步 `arke-compiler-infrastructure.md §7.7`。protocol.py §8 backend 扩展 seam 保持（Ascend 可插）。

## 各卡 DoD（验收标准）

### K-H3.1 — matmul autotune 悬崖缓解 [P1]
- `arke/backend/triton_templates/matmul.py.j2` 的 `@triton.autotune(key=["M","N","K"])`
  改 bucketed key（`next_pow2(M), next_pow2(N), K`），bmm 同理。
- 新增 probe `benchmarks/probes/autotune_first_call.py`：量化同 bucket 内新 shape
  首调开销（改前 vs 改后）。
- DoD: 同 bucket 新 shape 首调不再触发全量扫描（probe 数据佐证）；tier2 matmul
  中位 geomean 无回退（≥1.0 维持）；make test 全绿。

### K-H5.2 — 收敛曲线 CSV [P1，半天]
- ArkeEnv 已有 trajectory 记录；加 `--emit-convergence-csv`（iteration vs
  best-so-far ratio），跑 3 个代表 op（matmul/softmax/flash_attention）出首批曲线。
- DoD: CSV 落 `benchmarks/results/`，收敛效率首次有数据回答（审计 §一.3 的空白）。

### K-ATT — attention flash-style 模板 [P1 性能主线，最大件]
- 背景: OT4 golden 换 flash-attn 后暴露 Arke attention 0.17-0.24（慢 4-6×）。
  归因: 缺 online-softmax + K/V tile 流水（现模板物化 score 或分段 softmax）。
- 路径: sm_86 flash-style Triton 模板（online softmax + 双缓冲 K/V + tl.dot TC），
  Phase5 LLVM 软流水经验可迁移。先 FA 后 GQA（GQA 需 Hkv≠Hq 原生支持避免 K/V 扩展）。
- DoD: FA/GQA vs flash-attn golden geomean 显著收敛（阶段目标 ≥0.5，最终 ≥0.8——
  目标值属 frozen 层，正式定标需 Leon 确认）；correctness allclose 全过；不回退
  其他 op。
- 建议独立 session 跑（输出量大，一个 session 一件大事）。

### K-H1 — 双 IR 统一第一步 [P2]
- `arke/backend/` IRGraph 增加 `from_semantic(sem_ir, strategy)` 官方构造器，
  废除散落 ad-hoc 转换；dtype/stride/fusion 边信息进转换契约。
- golden 测试: SemanticIR→IRGraph→SemanticIR 往返等价。
- DoD: 所有后端入口统一走官方构造器；往返测试全绿；为后续「降格 or 升格」
  决策（需 Leon 定）留干净地基。

### K-H2 — HardwareModel 抽象 [P2，Ascend 恢复前必做]
- `arke/backend/protocol.py`: `lower(sem_ir, strategy, hw: HardwareModel)` 统一签名
  （消除三后端私参漂移）；HardwareModel = 内存层级树+同步域+计算单元描述符+对齐约束。
- StrategyIR 合法动作生成器消费 HardwareModel（tile 上限/stage 数/TC 可用性）。
- `backend.capabilities()` 能力查询，engine 动作空间生成期裁剪。
- DoD: NVIDIA HardwareModel 实例落地且 4 后端过 make test；协议文档同步
  `docs/architecture/arke-compiler-infrastructure.md`。

### K-H5.1 / K-DYN / K-XATT [P3，简述]
- K-H5.1: spec 把 Schedule/Instruction IR 标注 Phase-future（诚实降格），或让 LLVM
  后端软流水决策显式过 ScheduleIR（真接降级）——方向需 Leon 拍板后执行。
  **已批 H5a (honest downgrade, Leon 2026-07-29)**：不真接降级；在
  `docs/spec/arke-ir-spec.md §3.4` 加实现状态 note，§7/§8 头加 `[skeleton]`/
  `[Phase-future]` 逐字段标注，`pass-infrastructure-spec.md §7.2` 加交叉引用。
  结论：Layer 4/3（Semantic/Strategy）已实现驱动编译；Layer 2/1（Schedule/
  Instruction）为已填充的结构骨架，backends 各自做真实调度，ScheduleIR 字段
  当前不驱动 codegen。真接降级（LLVM 软流水/寄存器分配走 ScheduleIR）列为 future。
- K-DYN: 新 bench track，同 op 连续变 shape 测首调+稳态，产出 Performance Cliff gate。
  **测量层已落地 (2026-07-29)**：`benchmarks/dynamic_shape.py`（生产 wrapper 直测 +
  op-aware `spec_key` 预测列），25 tests，首批 3060 数据
  `benchmarks/results/dynamic_shape/2026-07-29_191225/`。核心发现：softmax 每个新
  seq-len 付 3.5-6ms 编译（cliff geomean 41×），matmul 因 K-H3.1 bucket 仅 3.3×，
  rmsnorm 7.2×。报告 `docs/benchmark/dynamic-shape-cliff.md`。
  **已批 D1 (measure-only, Leon 2026-07-29)**：本 track 只跟踪曲线、不设 pass/fail
  gate；等 K-ATT 落地 + 跨 run 方差数据后再议是否升级 D2/D3。
- K-XATT: flash_attn varlen API 支持非等长 Q/KV 后，cross_attention golden 从
  FlagGems 迁移（见 docs/benchmark/ot4-golden-review-rfc.md 尾注）。

## 硬停点提醒（frozen 层，需 Leon 确认才能动）
- K-ATT 的最终收敛目标值（≥0.8?）
- K-H5.1 的方向选择（降格 vs 真接降级）
- K-H1 第二步的 IRGraph 降格/升格决策
- 任何 Gate 阈值/评分语义改动
