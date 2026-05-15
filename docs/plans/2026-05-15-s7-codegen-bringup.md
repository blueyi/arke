# S7 Codegen 接入 — 实施 plan

**起点 HEAD**: `13d42e6`  
**目标**: 让 `ArkeRunner` 真正跑 Triton-generated kernel（而不是 PyTorch eager `reference_impl`）  
**G7.8d 预期**: weighted 0.513 → ≥ 0.95（取决于 22 个模板的实际性能）

## 现状盘点

### 已就绪
- `arke/backend/triton_templates/*.py.j2`: **22 个模板**全部已写
- `arke/ir/ops/catalog.py`: 45 个 op 全部带 `template_hint=TemplateHint(template_name, extra_ctx)`
- `arke/ir/ops/schema.py:55` `TemplateHint(template_name, primary_op, extra_ctx)` 定义完整
- jinja2 3.1.6 + triton 3.2.0 + torch 2.6 cu124 ✓

### 缺口
- `arke/backend/triton_backend.py`: `lower()` 只生成注释，`compile()` 返 `_execute_via_reference` placeholder
- `benchmarks/baselines/arke_runner.py`: 走 `INTERPRETER.execute()`（reference_impl），不调 backend
- 模板 ctx schema 不统一：`extra_ctx={"op_variant": "relu"}` vs 模板内 `{% if activation == "relu" %}`
- 没有 KernelCache（每次 launch 都重新 jit-compile 太慢）

## 模板 ctx 需求清单（侦察结果）

| 模板 | 模板内 jinja 变量 | catalog extra_ctx | 适配策略 |
|---|---|---|---|
| elementwise | `activation` | `op_variant="relu"` | rename `op_variant` → `activation` |
| elementwise_binary | `op_variant`（应该）| `op_variant="add"` | 检查模板源 |
| matmul | `fused_activation`, `output_dtype` | (无) | 默认 `fused_activation=None`, `output_dtype=tl.float16` |
| layernorm | `norm_type` | (无) | 由 op.name 推断 |
| reduction | `reduction_op` | `op_variant="sum"` | rename + prefix `reduce_` |
| softmax | (无 jinja 分支) | (无) | 直接渲染 |
| 其余 17 个 | 待逐一检视 | 各异 | 按需适配 |

## 实施 commit chain

### C1: 渲染器骨架 + smoke test (最小可行)
**goal**: 单 op (relu) 端到端跑通：catalog → render → JIT → execute → 对齐 reference_impl 数值

**deliverables**:
- `arke/backend/triton_codegen.py`: 新模块
  - `render_kernel_source(template_name, kernel_name, ctx) -> str`
  - jinja2 `Environment(FileSystemLoader(triton_templates/))`
- `arke/backend/triton_backend.py`: `lower()` 用 `render_kernel_source`，`compile()` 用 `exec()` + `triton.jit` 拿到 callable
- `tests/backend/test_triton_codegen_smoke.py`: relu 单 op，render 成功 + JIT 成功 + 数值对齐 ref_relu

**风险**: catalog `op_variant` vs 模板 `activation` 不对齐 → 加适配函数 `_normalize_template_ctx(op_name, hint)`

### C2: 8 个 OT0/1 模板 ctx 适配 + 数值校验
**goal**: 覆盖 elementwise / elementwise_binary / softmax / reduction / cast / cumsum / topk / rmsnorm_residual

**deliverables**:
- `arke/backend/triton_codegen.py`: `_normalize_template_ctx` 完整支持 OT0/1
- `tests/backend/test_triton_codegen_ot01.py`: 21 个 op (relu/gelu/silu/.../add/mul/where/softmax/reduce_*/cumsum/argmax/...) 全部数值对齐
- 数值容差: rtol=1e-3, atol=1e-3 (fp16)

### C3: 12 个 OT2/3/4 模板 ctx 适配
**goal**: 覆盖 matmul/batch_matmul/grouped_matmul/transpose/data_movement/index_ops/quantize/gated_activation/cross_entropy/flash_attention/mla/paged_attention/rope

**deliverables**:
- 同上，数值校验全 24 个 op
- 已知问题: rope 必须用 fp32 sin/cos (Q5a 修过)，模板内是否复用?

### C4: KernelCache（避免每次 JIT 重编）
**goal**: 同一 (template, ctx, dtype, shape-bucket) 只 JIT 一次

**deliverables**:
- `arke/integration/kernel_cache.py`: in-process LRU dict keyed by `(template_name, frozenset(ctx.items()), input_dtype_signature)`
- backend `compile()` 走 cache
- pytest 验证 cache hit 后第二次 launch 不重 compile

### C5: ArkeRunner 切换到 backend 路径
**goal**: `arke_runner.py:get_fn` 不再走 `INTERPRETER.execute`，改走 `TritonBackend.lower→compile→run`

**deliverables**:
- `arke_runner.py` 重写 `get_fn` (warmup 改成 backend.compile + run)
- 留一个 `--use-eager-fallback` env flag 防止单 op 失败时崩整次 bench
- pytest: 全 45 op smoke test，所有 supports() 返 True 的 op 都能 backend 执行

### C6: bench rerun + perf 数据更新 + Gate G7
**goal**: 看真数据

**deliverables**:
- `python -m benchmarks.bench_l1 --op all --shapes all` (估时长 30-60min, autotune 首跑会慢)
- `python /tmp/refresh_perf.py` 重生 PERF_ALL
- `python -m benchmarks.gate G7 --tier 3` 出新 weighted
- 落 daily memo 数据

### C7: 文档同步
**goal**: 所有 codegen 改动反映到 docs

**deliverables**:
- `docs/architecture/arke-compiler-infrastructure.md` 加"Triton 渲染流"段
- `docs/spec/arke-ir-spec-design.md` 加 TemplateHint extra_ctx 适配规则表
- `docs/roadmap/plan.md` 标 S7 partial-start

## 估时

| Commit | 估时 | 备注 |
|---|---|---|
| C1 | 1.5h | 最关键，jinja env + JIT 错误处理 + relu 端到端 |
| C2 | 2h | 8 模板 ctx 适配 + 21 op 数值校验 |
| C3 | 3h | flash_attention/mla 复杂，rope fp32 一致性 |
| C4 | 1h | LRU dict + 测试 |
| C5 | 1h | runner 切换 + fallback |
| C6 | 1h+30min wait | 跑 bench |
| C7 | 30min | 文档 |
| **合计** | **~10h** | 可分 2-3 个 session |

## 风险

1. **22 个模板可能有 syntax bug**（写完没人跑过）— C1/C2/C3 会逐个暴露，每个独立修
2. **autotune 首跑超时**：6GB VRAM + 多 config，长 seq matmul 单次 compile 30s+，bench 全跑可能 1h+
3. **catalog `extra_ctx` 与模板变量名不一致**：估计每个模板 1-3 个 rename，C1 写 `_normalize_template_ctx` 集中处理
4. **模板假设的 input shape/dtype 与 catalog 不符**：rope 已知 fp32 sin/cos，其他可能也有
5. **flash_attention/mla/paged_attention 模板可能依赖外部 wrapper**（不止单 jinja）— 要 C3 时再判断

## 已知坑（不变）

1. DSv3-lmhead/long-32k @ 6GB + `--force-restart` = 9h+ 静默 OOM hang
2. `bench_l1` resume-cache 跳过 perf-write — rerun 后必须 `/tmp/refresh_perf.py` 重生
3. Hermes `process(action=kill)` 缺 psutil → `kill -9 <pid>`
4. **新坑（C1+）**: jinja 模板 ctx 缺字段时报 `UndefinedError`，必须在 render 前补全或用 `Environment(undefined=ChainableUndefined)` 容错

## Stage 边界声明

按 SOUL.md / AGENTS.md：本工作**提前启动 S7 实现层**，但**不修改任何 G6/G7/G8 Gate 定义**。
S6 Gate 仍按现状判定；S7 codegen 接通后 G7.8d 数据自然变化。
