# RFC — OT4 Golden Ladder 重审（FlagGems SDPA 非 fused）

**状态**: 🔴 待 Leon 拍板（frozen 层语义变更，需 sign-off）
**日期**: 2026-07-27
**触发**: PERF_ALL tier-2 全量刷新时 FlagGems attention 在 6GB 卡上必然 OOM
**关联 commit**: `eb11e64`（harness OOM guard + FlagGems 重分类，已 push）
**关联文档**: `docs/benchmark/golden-kernel-ladder.md` §106-128（OT4 表 + ‡ 脚注）

---

## 1. 结论先行

S7.followup.3（2026-06-06）把 **FlagGems 定为 flash_attention / grouped_query_attention
/ cross_attention 的 OT4 Golden**，其论证的**两个核心前提在今天均已不成立**：

| S7.followup.3 前提（2026-06-06） | 2026-07-27 实测反证 |
|:---|:---|
| ① "FlagGems 是唯一在 sm8.6 上提供 **Triton-only fused SDPA** 的库" | ❌ flag_gems **5.0.0** 的 SDPA 走 **bmm + 显式 softmax 分解**（物化 [B*H,S,S] score），**无 flash/fmha fused kernel** |
| ② "flash-attn / FlashMLA / vLLM 在 6GB 3060 venv 里 **build failure 装不上**" | ⚠️ flash-attn PyPI 现有 **2.8.3.post1**，官方支持 Ampere sm80/86；当时的 build failure 可能已可解（**待实测**） |

**影响**：当前 OT4 golden 是一个"名为 Triton-fused、实为 bmm 分解"的 baseline。
它在小 shape 上能跑（虽非 fused），在大 shape 上因物化 score 必然 OOM。
这既是**性能分母失真**（Arke 的 fused kernel 在跟一个非 fused baseline 比），
也是**同后端公平性瑕疵的回归**（S7.followup.3 当初正是为消除"名为 cuDNN 实为
Triton"的谎言才提拔 FlagGems，如今 FlagGems 自己变成了"名为 fused 实为分解"）。

---

## 2. 证据三件套

### 2.1 代码路径 — bmm 分解（flag_gems 5.0.0）

`torch.nn.functional.scaled_dot_product_attention` 在 `flag_gems.enable()` 后
被 aten override 劫持，落到 `flag_gems/ops/bmm.py:136`：

```python
out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
```

即显式为 `Q@K^T` 的 score 矩阵分配 `(batch, M, N) = (B*H, S, S)` 缓冲区。
这是 O(S²) 内存的 naive 分解，不是 flash-style tile-wise streaming softmax。

### 2.2 Profiler 实测 — 无 fused kernel（决定性）

在 `[4,128,64]` fp16 causal SDPA 上用 `torch.profiler` 抓 kernel 分派：

```
flag_gems 5.0.0 SDPA dispatched kernels (fp16, [4,128,64] causal):
  bmm present:            True     ← Q@K^T 和 P@V 两次 bmm
  flash kernel present:   False    ← 无 flash/fmha/mha fused kernel
  softmax present:        True     ← 独立 softmax pass（说明 score 被物化）
```

`bmm + 独立 softmax + 无 flash kernel` = 教科书式的 naive attention 分解。
这从**运行时**层面坐实结论，不是靠 traceback 推断。

### 2.3 OOM 记录 — 大 shape 必崩

tier-2 全量刷新 + attention 重跑中，FlagGems 在 6GB 卡上的分配尝试：

| op | shape | 尝试分配 | memory_ratio |
|:---|:---|:---|:---|
| flash_attention | ds-v2-4k | 32 GiB | 1.36 |
| flash_attention | ds-v3-32k | — | 78.8 |
| flash_attention | ds-v3-163k | — | 1945 |
| grouped_query_attention | llama3-8b（K/V 扩展到全 query heads） | 112 GiB | 1.29 |
| grouped_query_attention | qwen25-7b-32k | — | 17.2 |

修复前：未捕获的 OOM 一路抛出，杀掉整个 bench run，PERF_ALL 缺 5 op。
修复后（`eb11e64`）：这些 shape 被 preflight 记为 typed `skipped`（带 ratio 证据），
run 全程存活，PERF_ALL 46/46 op、0 oom/error。

---

## 3. 环境事实

- flag_gems **5.0.0** (`GEMS_VENDOR=nvidia`)
- torch 2.6.0+cu124, CUDA 12.4, GPU sm_86 (RTX 3060 Laptop 6GB)
- flash-attn / flash_mla / vllm **当前均未安装**（`ModuleNotFoundError`）
- flash-attn PyPI 最新 2.8.3.post1，支持 Ampere sm80/86

---

## 4. 候选方案（待 Leon 选）

### 方案 A — flash-attn P2 上位（推荐，若能装）
- 装 flash-attn 2.8.3.post1（Dao-AILab，真正的 fused flash kernel），提为 FA/GQA
  的 OT4 Golden；cross_attention 视 flash-attn 是否支持非等长 Q/KV 决定。
- **优点**：Golden 名副其实（真 fused），性能分母正确，大 shape 不 OOM（O(S) 内存）。
- **风险/成本**：需从源码编译（sm86，耗时可能 30-60min），可能踩 CUDA/torch ABI 坑；
  需先做一次**安装可行性 spike**确认能装能跑能过 correctness，再定论。
- MLA/paged 维持 audit-degraded（flash-attn 不覆盖 MLA/paged）。

### 方案 B — 钉住旧版 flag_gems
- 找出哪个 flag_gems 版本的 SDPA 仍是 fused Triton kernel，钉版本回退。
- **优点**：不引入新依赖编译。
- **风险**：旧版可能与 torch 2.6 不兼容；且"哪个版本是 fused"需考古验证；
  即便找到，也是把 golden 绑死在一个旧版本上，长期维护负担。可能根本不存在
  "既 fused 又兼容 torch2.6" 的 flag_gems 版本。

### 方案 C — FlagGems 降级 + OT4 全部 audit-degraded（保守兜底）
- 承认 sm8.6 上 9 个审计库都没有可用的 production fused attention Triton kernel，
  把 FA/GQA/cross 也降为 audit-degraded（PyTorch-eager P3 兜底），与 MLA/paged 一致。
- **优点**：诚实、零新依赖、立即可落地。
- **缺点**：OT4 group 的 Triton-golden 覆盖归零，G7.8d 加权分会跌（当初 followup.3
  提拔 FlagGems 就是为了把 ot4 从 0/0 拉到 26/26、+83% 分数）。这是**倒退**。

---

## 5. 我的推荐

**先做方案 A 的安装可行性 spike**（半天量级，可逆，不碰 frozen 层）：
1. 在 arke venv 里试装 flash-attn 2.8.3.post1，记录 build 成败。
2. 若成功 → 写 flash_attn_runner 的真实 `get_fn` + correctness 对照，
   跑 tier-2 FA/GQA 验证 fused（profiler 确认有 flash kernel）+ 数值正确。
3. 拿到 spike 结论后，再回来向你提交**方案 A vs C 的最终二选一**（B 仅在 A 失败且你想避免 C 倒退时才考虑）。

spike 本身是实现层可逆工作，按授权我可直接做；**golden 表的实际改动**（frozen 层）
留到 spike 有结论、你 sign-off 后再动。

---

## 6. 短期现状（已生效，不阻塞本 RFC）

`eb11e64` 已让 harness 在当前 FlagGems 下**安全运行**：大 shape skipped 带证据、
不再 crash。OT4 golden 语义**暂未改动**——FlagGems 仍是 FA/GQA/cross 的 golden，
只是大 shape 变成 skipped 行。本 RFC 决定的是**是否更换 golden**，与短期修复正交。
