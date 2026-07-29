# Audit Follow-up Decisions (2026-07-30)

Kitty 自主决策记录 —— 承接 `2026-07-29-architecture-audit.md` 的 R1–R5 整改后，
上一 session 收官时留下的 7 项开放问题。Leon 令：「按照 Arke 能够达到的最佳效果
进行这些问题的决策，决策后继续推进」。

**决策原则**（USER.md 锁定）：以「最佳功能 + 最佳性能」为唯一目标；实现分叉选
「最有机会达成目标」而非「最低风险」；**frozen 层（Gate 阈值 / 评分语义）改动是
硬停点，必须 Leon 批准**——我只备料给出推荐，不擅改。

---

## 决策总表

| # | 问题 | 层级 | 决策 | 理由 |
|:--|:--|:--|:--|:--|
| **A** | L2 ScheduleIR 最小真实化（让 IR 真正驱动 codegen） | 架构（我定方向） | **设计先行 + 一条窄真实链**（不做全量重构） | 全量 L2→codegen 重构是多周工程，收益是「架构完整度」而非「性能/功能」。最佳效果 = 先落一条可验证的真实驱动链（1 个 op 的 tile 决策经 ScheduleIR 字段流到 codegen），证明骨架可承重，把重构成本从「一次性大爆炸」摊成「按 op 增量」。 |
| **B** | R3 运行时降级策略（JIT 过贵 → eager + async warmup） | serving 集成层 | **延后 + 落地 API 契约**（不实现完整 policy） | 真实降级 policy 依赖 serving 运行时（请求流、SLA、异步编译线程池），Arke 当前无 serving harness。最佳效果 = 现在把 `warmup_buckets()` 暴露成对外契约（已有），文档写清「serving 层如何用」，policy 本体等 serving 集成 Phase 再做。 |
| **C** | layernorm 等 row-scan warmup 跟进 | 纯实现（我定） | **立即做** | 与 rmsnorm R3 同模式，零风险高确定性 quick win。所有 row-scan reduce（softmax/rmsnorm/layernorm）应同享 bucket warmup，否则 layernorm 仍有 cliff。 |
| **D** | dynamic-shape D1→D2/D3 gate 升级 | **frozen（硬停点）** | **不动，备料给 Leon** | Gate 语义 = frozen 层。D1 measure-only 是 Leon 2026-07-29 批的。升级到 D2 soft-gate 需跨-run 方差数据，且改的是评分语义 → 必须 Leon 拍板。我给出推荐（见 §D）。 |
| **E** | GQA 正式定阈（proposed 0.30/0.45，已实测 0.802） | **frozen（硬停点）** | **不动，备料给 Leon** | Gate 阈值 = frozen 层。已实测 0.802 远超 proposed，但「锁阈值」这个动作本身是 frozen 层操作 → Leon 批。我给出推荐（见 §E）。 |
| **F** | PERF_ALL 全量刷新（attention 涨后快照过时） | 纯测量刷新 | **做（后台全量重跑）** | 快照过时是诚实性问题（旧快照 attention 数字虚高/过时）。重跑刷新是纯测量，无 frozen 改动。 |
| **G** | FA-v4 micro-opt（D=64 short-S gap 0.66） | 纯实现 | **尝试，helps 则留** | Gate 已过（0.846），这是锦上添花。按「最佳性能」原则值得一试，但不为它降低任何标准；不 helps 就诚实记录并放弃。 |
| **+** | 分母陷阱 fail-loud（审计教训） | 诚实性 bug | **审查 + 加固** | 审计发现 golden latency 分母曾静默回退 eager 致 attention 数字虚高。已确认 correctness 路径有 audit_status；需确认 **latency 分母**路径也 fail-loud。 |

---

## §D — dynamic-shape gate 推荐（备 Leon 批，frozen）

**推荐：暂维持 D1 measure-only，K-ATT 稳定后一个完整 PERF_ALL 周期再评估 D2。**

理由：D2 soft-gate（`same_spec_geomean ≤ 5×` + `n_new_spec` 上限）需要跨-run
方差基线才能设合理阈值，否则阈值是拍脑袋。当前 row-scan warmup 刚补齐
（C 项），cliff 数字还在变。等 F 项全量刷新 + 2-3 个 run 的方差数据到手，再把
D2 阈值锚在实测方差上。**这是「测量方法」层建议，target 不变——仍需 Leon 批。**

## §E — GQA 正式定阈推荐（备 Leon 批，frozen）

**实测：GQA geomean 0.802（commit 336d29b，correctness 8/8 ≤9.8e-4）。**
proposed 阈值 stage ≥0.30 / final ≥0.45 已被实测大幅超过（0.802 > 0.45）。

**推荐锁定：stage ≥0.30 / final ≥0.45（与 proposed 一致）。** 理由：GQA 的
K/V 复用结构天花板与 FA 不同，0.45 是保守但真实可持续的线（0.802 是当前实测，
留足跨-run 漂移余量）。**锁阈值 = frozen 层操作 → Leon 一句「E ok」即锁。**

---

*Decisions by Kitty, 2026-07-30. Implementation-layer items (A narrow / B contract /
C / F / G / denom) executed autonomously; frozen-layer items (D / E) await Leon.*
