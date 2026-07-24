# L1 性能天花板诊断 — 2026-07-24

> 数据源：benchmarks/results/**/l1/*_results.csv（2026-07-24 全量 tier-2 重跑）
> 比较口径：Same-Backend Fairness — arke_latency vs min(Triton-only refs: FlagGems/Liger/Unsloth/vLLM)
> speedup = ref/arke，>1 = Arke 更快；geomean over comparable shapes

## 低于天花板的 op（优化候选，按最差排序）

| # | op | geomean | worst | #shapes | OT |
|---|---|---|---|---|---|
| 1 | ~~embedding~~ ✅ | ~~0.678~~ → parity (d35bb62) | — | 9 | OT2 |
| 2 | batch_matmul | 0.835 | 0.708 | 6 | OT2 |
| 3 | softmax | 0.852 | 0.754 | 10 | OT1 |
| 4 | reduce_sum | 0.877 | 0.784 | 6 | OT1 |
| 5 | reduce_mean | 0.897 | 0.830 | 6 | OT1 |
| 6 | layernorm | 0.956 | 0.804 | 9 | OT1 |
| 7 | matmul | 0.974 | 0.755 | 9 | OT2（ε=0.03 内） |

> **embedding CLOSED** (2026-07-25, d35bb62): launch-overhead bound (~55µs fixed
> vs FG ~38µs, shape-independent). Fixed host-side: slim launch args 9→5,
> cache BLOCK_D, runner single-node fast-path. 0.678 → parity. See skill
> `arke-benchmark-harness/references/launch-overhead-op-optimization.md`.

## ≥1.0 的 op（18 个，健康）

cumsum 1.08 / gelu 1.20 / neg,exp,relu,rsqrt,mul,add,sigmoid,tanh,where_,silu 1.29-1.38 /
rmsnorm 1.44 / cast 1.57 / transpose 1.67 / rope 1.77 / reduce_max 2.74 / cross_attention 4.97

## 无 Triton 参考（audit-only，23 个）

argmax concat copy_ cross_entropy dequantize_per_channel flash_attention
fused_linear_cross_entropy gather geglu gelu_and_mul grouped_matmul
grouped_query_attention multi_latent_attention paged_attention permute
quantize_per_token rmsnorm_residual scatter silu_and_mul split swiglu
swiglu_packed topk
（基线只有 PyTorch-eager/cuBLAS — 按 gate 规则 audit-only，不计分）

## 优化计划（Leon 已定方向：深化 NVIDIA 路径）

顺序：embedding → batch_matmul → softmax → reduce_sum/mean（同族一起）→ layernorm → matmul worst-shape
方法：对照 FlagGems 同 op kernel（性能 ladder PRIMARY），逐 op 改 Arke Triton 模板，
单 op bench_l1 --tier 2 --no-resume --force-restart 验证，达标（geomean ≥1.0 或 ε 内）即 commit。
GPU 串行：一次只跑一个优化任务。

## 验证命令

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m benchmarks.bench_l1 --op embedding --tier 2 --no-resume --force-restart
```
