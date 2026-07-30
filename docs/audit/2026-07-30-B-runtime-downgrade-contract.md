# B — R3 运行时降级策略：API 契约（延后实现，先定契约）

**决策**（`docs/audit/2026-07-29-followup-decisions.md` §B）：R3 的运行时「JIT 首编译过贵
→ eager 顶住 + 异步编译」降级 policy 依赖一个 serving 运行时（请求流、SLA、异步编译线程池），
Arke 当前无 serving harness。**最佳效果 = 现在把 warmup / 降级的 API 契约钉死，policy
本体等 serving 集成 Phase 再做**，避免现在拍一个没有 serving 上下文的假 policy。

---

## 1. 问题回顾（R3 语境）

动态-shape cliff（审计 `docs/benchmark/dynamic-shape-cliff.md`）：每个新 spec bucket 的
首次调用付 Triton JIT 编译墙（softmax 冷 40.99×、rmsnorm 7.22×、layernorm 冷 ~170ms）。
R3 已用 `<kernel>_warmup_buckets()` 把编译移到**启动期**（离线预热）。但生产 serving 里
会遇到**启动期没预热过的新 shape**（长尾请求），此时首调仍付编译墙。降级 policy 要回答：
「运行时遇到冷 spec 时，是阻塞等编译，还是先用 eager 顶住 + 后台异步编译，编译好再切换？」

## 2. 已就绪的契约（本次确认，不新增代码）

Arke 侧已提供降级 policy 需要的两个原语，**契约在此钉死**：

### 2.1 预热契约（离线 / 启动期）

```python
warmed: set[int] = <kernel>_warmup_buckets(dims, device="cuda", dtype=torch.float16)
```
- **语义**：对给定的模型维度集合 `dims`（norm 的 hidden_size / softmax 的 seq-len 类），
  预编译对应 spec bucket，返回已预热的维度集合。
- **保证**：预热覆盖的 bucket 首调 compile-free（softmax/layernorm 无残留；rmsnorm 首个
  novel-M 付 ~1.4ms warm-N 残留，已诚实标注）。
- **已覆盖 op**：softmax / rmsnorm / layernorm（row-scan 家族全覆盖，R3 + C 项）；
  matmul / flash_attention 走各自的 `_TILE_CFG_CACHE` / `_FA_CFG_CACHE` bucket memo。

### 2.2 冷/暖探测契约（运行时判断是否会 cliff）

```python
# benchmarks/dynamic_shape.py 的 op-aware spec_key（K-DYN D1）
```
- **语义**：给定 (op, shape) 计算 spec_key（launcher cfg + Triton div-class），
  同 spec_key = 暖（compile-free），新 spec_key = 冷（会付编译墙）。
- **serving 层用法**：请求进来先算 spec_key，命中已编译集 → 直接跑；未命中 → 触发降级。

## 3. 降级 policy 本体（延后 — serving 集成 Phase 定义）

未实现，因为需要 serving 上下文才能设计对。留给 serving Phase 的接口形状（建议，非锁定）：

```python
class CompileDowngradePolicy:
    def on_cold_spec(self, op, shape) -> Literal["block", "eager_fallback"]:
        # SLA-aware: 若请求可容忍首调延迟 → block 等编译（一次性）
        #            若 SLA 紧 → 返回 eager_fallback，后台线程异步编译，
        #                        编译好后续请求自动切到 Triton kernel
        ...
    def submit_async_compile(self, op, shape) -> Future: ...
```
- **依赖**：一个异步编译线程池 + eager reference 路径（`benchmarks/bench_l1.py::_eval_l1_reference`
  已有 eager 实现，可复用作 fallback）+ 请求级 SLA 信号（serving 才有）。
- **验证方式**（serving Phase 定）：注入冷 spec 流，断言 p99 不被单次编译墙击穿，
  且异步编译完成后 ratio 回到暖态。

## 4. 诚实边界

- 本次**不写** policy 本体（无 serving 上下文 = 写了也是纸面假设）。
- warmup + spec_key 两个原语**已就绪且已测**，serving 层可直接消费——契约不欠债。
- 这与 A（L2 narrow driver）的处理一致：真实存在的先钉契约 + 验证，纯纸面的诚实标注延后。

*Contract by Kitty, 2026-07-30. warmup/spec_key 原语已就绪；policy 本体待 serving 集成 Phase。*
