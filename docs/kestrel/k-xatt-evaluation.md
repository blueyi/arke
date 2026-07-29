# K-XATT — cross_attention flash-attn golden 评估报告

**状态**: ✅ 评估收官（2026-07-29）— 实现（golden 换血）需 Leon 批（触及 benchmark 分母，硬停点）
**卡**: `docs/audit/kestrel-backlog.md` K-XATT（P3, 1-2d）
**关联**: `docs/benchmark/ot4-golden-review-rfc.md`（FA/GQA 换血历史）、`benchmarks/golden_ladder.py` LADDER_PREFERENCES

---

## 1. 评估背景

OT4 golden 换血（`1732532`, 2026-07-27, Leon 批 A）把 **flash_attention / grouped_query_attention** 的 golden 从 FlagGems 升级到真 fused flash-attn 2.7.4.post1。**cross_attention 当时被留在 FlagGems**，理由（`golden_ladder.py` L92-93 注释）：

> "cross_attention stays FlagGems: `flash_attn_func` requires equal Q/KV seq lens; varlen API evaluation is a separate follow-up."

K-XATT 就是这个 follow-up：评估 flash-attn 能否覆盖 cross_attention 的 Sq≠Skv（decoder Q 长度 ≠ encoder K/V 长度）。

## 2. 实测结论（2026-07-29，RTX 3060 SM 8.6, flash-attn 2.7.4.post1）

**核心发现：留在 FlagGems 的前提是错的。** `flash_attn_func` 标准 API **本身就支持 Sq≠Skv**，无需 varlen 复杂路径。

| 测试 | 输入 | 结果 |
|:--|:--|:--|
| `flash_attn_func(q,k,v)` packed [B,S,H,D] | B=2, Hq=8, **Sq=128 ≠ Skv=256**, D=64 | ✅ out (2,128,8,64)，max_abs_diff vs SDPA = **1.2e-4**（数值正确） |
| `flash_attn_varlen_func` + cu_seqlens_q/k | 同上 flatten | ✅ 可用，out (256,8,64)，但对 batch 内等长 cross-attn **不必要** |

`flash_attn_func` 的 seqlen_q 与 seqlen_k 由 q/k 的 dim=1 各自决定，二者独立 —— "requires equal Q/KV seq lens" 是对 API 的误解。varlen API 只在**同 batch 内每条序列长度不同**（真 ragged）时才需要；cross_attention benchmark 的 shape 是规则的 (B,H,Sq,D)/(B,H,Skv,D)，标准 API 完全够用。

## 3. 现有代码可行性

`benchmarks/baselines/flash_attn_runner.py`：
- `_SUPPORTED_OPS = {"flash_attention", "grouped_query_attention"}` —— **只需加 `"cross_attention"`**。
- `run_for_output` / `get_fn` 已处理 (B,H,S,D)→(B,S,H,D) transpose + causal 语义。cross_attention 是**非 causal**（encoder-decoder attention 全可见），需在这两处对 cross_attention 传 `causal=False`（现有 `kwargs.get("causal", False)` 默认已是 False，get_fn 需针对 cross_attention 分支不强制 causal=True）。
- shape 需从 runtime ctx 取 Sq(shape.S) 与 Skv(shape.Skv)，arke_runner/bench_l1 已有 Skv 通路（L599-605）。

工作量：**~0.5d 实现 + 干净重跑 cross_attention correctness/perf**（约 12 shapes）。

## 4. 需 Leon 决策（硬停点：benchmark golden 分母）

换 golden = 改 cross_attention 的性能分母（当前 vs FlagGems bmm-分解弱分母，Arke 会从"看似不慢"暴露真实差距，同 FA/GQA 换血的性质）。按硬停点边界（baseline fairness / 评分分母属 frozen 层），**是否执行 cross_attention golden 换血需 Leon 拍板**：

- **X1（推荐）**: 换血 —— cross_attention golden 从 FlagGems 升级到 flash-attn（`flash_attn_func`, causal=False），与 FA/GQA 统一血统。诚实暴露 cross_attention 真实性能差距（预期同 FA ~0.3× 量级），作为 K-ATT 的额外输入。
- **X2**: 暂不换 —— 保留 FlagGems，记录本评估结论，待 K-ATT 主线打通 Triton flash-style 后一并处理。
- **X3**: Leon 另定。

## 5. 交付物

- 本评估报告（含实测数据 + 现有代码可行性）
- 修正了 `golden_ladder.py` 中的过时错误假设（"requires equal Q/KV seq lens"）—— 见 §6 建议的注释更新
- spike 脚本逻辑存档（`flash_attn_func` Sq≠Skv 验证）

## 6. 建议的注释订正（无论 X1/X2，先纠正 golden_ladder.py 的错误陈述）

`benchmarks/golden_ladder.py` L92-93 的注释应从"`flash_attn_func` requires equal Q/KV seq lens"更正为事实："`flash_attn_func` handles Sq≠Skv natively (verified K-XATT 2026-07-29); cross_attention golden swap pending Leon decision (X1/X2)"。
