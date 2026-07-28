# K-ATT — Decision Material for Leon (frozen 层拍板前必读)

**状态**: 🔴 硬停点 — 目标值属 frozen 层，必须 Leon 拍板才能开工
**日期**: 2026-07-28
**背景**: KESTREL 审计 §一.2（性能主线弱点）+ backlog K-ATT
**关联**:
- `docs/audit/kestrel-backlog.md` K-ATT 卡
- `docs/audit/2026-07-27-comprehensive-audit.md` §一.2
- `docs/benchmark/ot4-golden-review-rfc.md` 方案 A 落地历史
- `docs/phase5/c2-tensorcore-attention-2026-07-15.md` — CUDA-C v8 TC 实现（已完成，可迁移经验）
- `benchmarks/results/convergence/flash_attention_2x4x512x64.csv` — 首个 live-agent 收敛曲线（3 iter 2 fail correctness）

---

## 1. 现状（诚实数据，不含幻觉）

**OT4 golden 换血完成**（1732532, 2026-07-27）：flash-attn 2.7.4.post1 上位 FA/GQA 的 OT4 golden，
干净重跑数据固定：

- Arke Triton flash_attention vs flash-attn golden geomean = **0.301**
- Arke Triton grouped_query_attention vs flash-attn golden geomean = **0.172**

即 Arke Triton attention 全族比真 fused flash-attn 慢 **4-6×**（geomean 0.17-0.24 区间）。

**这不是回退**——是"金 baseline 被换血后差距诚实暴露"。之前的 FlagGems SDPA 用 bmm 分解物化 score，Arke 相对它似乎快，是**弱分母幻觉**。

**同期已有的 CUDA-C attention TC 后端**（Phase 5 c2, 2026-07-16 完成）：
- v8 general kernel，D∈{64,128}×{causal,非causal}，vs torch SDPA 达到 **0.35-0.42×**（S=1024-2048 kernel-only）
- 已 production 集成（`arke/backend/cuda_c_attention.py`）
- 关键工程洞察：**深化 latency hiding (cp.async 3-stage pipeline)** > 单纯提 occupancy

K-ATT 是把 CUDA-C v8 的成就**在 Triton 后端复刻**（+ 通过 Arke 编译栈让 LLM 可决策，这才是 AI-Native 命题）。

---

## 2. 必须拍板的三个 frozen 层选项

### 选项 A — K-ATT 目标 geomean 值

审计说"阶段目标 ≥0.5，最终 ≥0.8"，但**目标值是 frozen 层，需 Leon 定标**。

参考锚点：
- **0.5**（"能拿到 CUDA-C v8 一半" — 保守，Triton 上限低）
- **0.5-0.7**（"追平 CUDA-C v8 0.35-0.42×，考虑 Triton 生态复用价值") — 现实候选
- **0.8**（审计文中的"最终"目标 — 激进，需 Triton flash-style + TC 全打通）
- **等价 CUDA-C v8**（0.35-0.42 区间）— **锚定"同硬件已达成"** 的下限，最实用

推荐锚点：**Gate 阈值分两级**：
- **阶段 gate**：Triton FA/GQA vs flash-attn geomean ≥ **0.35**（=同硬件 CUDA-C v8 下限），代表"Triton 后端不再是短板"
- **最终 gate**：≥ **0.5**（1.4× 于 CUDA-C v8，让 Triton 路径成为独立可选高性能后端）

### 选项 B — K-ATT 实现路径

1. **纯 Triton flash-style 模板**（backlog 原计划）：online softmax + K/V 双缓冲 + tl.dot TC → 全在 Triton 生态内，最大兼容
2. **Triton wrapper 直调 CUDA-C v8**（复用已 production 的 TC kernel）：路径最短，但破坏"Triton 后端独立性"
3. **两条腿**：Triton flash-style 主线；如 Triton FA 一时上不去，短期 Triton→CUDA-C v8 桥接兜底

推荐：**路径 1（纯 Triton flash-style）**，理由：
- Arke AI-Native 命题的核心是"LLM 通过 StrategyIR 决策 → Arke 编译多后端"，Triton 后端必须独立能打
- Phase 5 LLVM 软流水/寄存器压力经验可迁移
- Triton 4.x 上 sm_86 TC dot (`tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)`) 已成熟

### 选项 C — GQA 处理策略

GQA 现状：Hkv < Hq（例：llama-70b Hq=64, Hkv=8）。当前 Arke Triton 实现把 K/V **扩展**到 Hq → 内存 8× 浪费。

1. **GQA 原生支持**（推荐）：Q 循环外层按 Hkv group 复用 K/V tile，从根上省 8× 内存
2. **先 FA 后 GQA**：GQA 依赖 FA 打通后再做（backlog 原计划）
3. **FA + GQA 并行**：分两个 sub-task，最终目标 geomean 分别定阈值

推荐：**方案 2（先 FA 后 GQA）**——FA online-softmax 是 GQA 的前置技能，先打通 FA 再复用；目标值 FA、GQA 分别定。

---

## 3. 建议

**Leon 只需批 3 个选项**（用紧凑代号即可）：

```
A: 阶段 gate 0.35 / 最终 0.5           （FA vs flash-attn geomean）
B: 路径 1（纯 Triton flash-style）
C: 方案 2（先 FA，后 GQA，分别定阈）
```

其余（子任务拆解、迭代次数、模板 revision、correctness allclose 阈值 0.005 等实现层）全权由我负责。

**Session 隔离**：K-ATT 是 2-4 周主线 + 输出量巨大（Triton IR/PTX dump、profiler、bench 三方对比、多轮迭代）。按 AGENTS.md「一个 session 一件大事」原则，**建议 K-ATT 拿独立 session 开工**——本 session 打完 P1 小件（K-H3.1 / K-H5.2）后收尾，K-ATT 开新 session。

---

## 4. Session 开工后我的第一步（Leon 批完 A/B/C 后）

1. 建 `docs/kestrel/k-att-plan.md`，把 Gate 阈值、路径、验收 protocol 落死
2. 起 `arke/backend/triton_templates/flash_attention_v1.py.j2` — online softmax + K/V 双缓冲雏形
3. 3060 sm_86 profile 一轮，出 baseline 数据 → 决定第一轮优化方向
4. **每个关键 revision（vN）单独 commit + push + 汇报 geomean 数据**（Leon 授权原则）

---

*等 Leon 就这份文档给 3 个字母的决定：A_ B_ C_（或让我改选项）。*
