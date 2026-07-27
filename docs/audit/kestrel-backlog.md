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
| K-H3.1 | matmul autotune key 改 bucketed + 首调开销 probe | P1 | ⬜ TODO | 1-2d |
| K-H5.2 | ArkeEnv trajectory → 收敛曲线 CSV (`--emit-convergence-csv`) | P1 | ⬜ TODO | 0.5d |
| K-ATT | attention flash-style 模板（online-softmax + K/V 双缓冲 + TC dot） | P1(性能主线) | ⬜ TODO | 2-4w |
| K-H1 | 双 IR 统一：IRGraph `from_semantic()` 官方构造器 + 往返 golden 测试 | P2 | ⬜ TODO | 1w |
| K-H2 | 显式 HardwareModel 抽象 + `lower()` 签名统一 + capabilities() | P2(Ascend 恢复前必做) | ⬜ TODO | 1-2w |
| K-H5.1 | Schedule/Instruction IR 诚实降格（spec 标注 Phase-future）或真接降级 | P3 | ⬜ TODO | 需 Leon 定方向 |
| K-DYN | dynamic-shape bench track（首调+稳态曲线 gate） | P3 | ⬜ TODO | 3-5d |
| K-XATT | cross_attention flash-attn varlen API 评估（OT4 golden 后续） | P3 | ⬜ TODO | 1-2d |

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
- K-DYN: 新 bench track，同 op 连续变 shape 测首调+稳态，产出 Performance Cliff gate。
- K-XATT: flash_attn varlen API 支持非等长 Q/KV 后，cross_attention golden 从
  FlagGems 迁移（见 docs/benchmark/ot4-golden-review-rfc.md 尾注）。

## 硬停点提醒（frozen 层，需 Leon 确认才能动）
- K-ATT 的最终收敛目标值（≥0.8?）
- K-H5.1 的方向选择（降格 vs 真接降级）
- K-H1 第二步的 IRGraph 降格/升格决策
- 任何 Gate 阈值/评分语义改动
