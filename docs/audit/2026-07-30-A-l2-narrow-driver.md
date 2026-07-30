# A — L2 ScheduleIR 最小真实化设计（narrow real driver）

**决策**（`docs/audit/2026-07-29-followup-decisions.md` §A）：不做全量 L2→codegen 重构；
落一条**窄真实驱动链**，证明 ScheduleIR 的字段可以真实驱动一个生产 backend 的 codegen 决策，
把「4 层 IR 只 L4/L3 承重」这条审计诚实发现从「结构骨架」推进到「有一条真实承重链」。

---

## 1. 现状（审计亲验，K-H5.1 已诚实标注）

四层 IR 链：`SemanticIR (L4) → StrategyIR (L3) → ScheduleIR (L2) → InstructionIR (L1)`。

- **L4→L3→L2→L1 lowering 真实存在**：`arke/compiler/lowering.py::lower_schedule_to_instruction()`
  读 `ScheduleIR.loop_nests[].tile_factors`、`.resources`、`.placements`、`.fusion_groups`，
  emit 对应的 InstructionIR op（`loop.configure` / `resource.bind` / `memory.place` /
  `fusion.group`）。**这一段不是骨架，是真代码。**
- **断点在 L1→生产 codegen**：生产 Triton/CUDA-C backend 从 Jinja 模板渲染，launcher
  的 tile/pipeline 决策来自各 backend 自己的启发式（如 FA 的 `_fa_cfg`），**不消费
  InstructionIR 的 `loop.configure`**。所以 ScheduleIR 的 tile 决策目前「算到了 L1 就断了」。

## 2. 窄驱动链定义（本次落地范围）

**选 FA flash_attention 模板的 tile 决策作为第一条真实链**，因为：
- 它是我刚在 FA-v4 亲手调过的决策（`_fa_cfg` 产出 `BLOCK_N/BLOCK_S/num_warps/num_stages`），
  语义清晰、可量化验证。
- 它正是 AI-Native 命题里 Agent 应该拥有的 bounded StrategyIR action（tile 选择）——
  让它从 ScheduleIR 真实流到 codegen，就是命题的最小可证明实例。

**链路**：
```
ScheduleIR.loop_nests[].tile_factors  (Agent/StrategyIR 决策的落点)
   │  (lowering 已 emit loop.configure，本次不改)
   ▼
FA launcher _resolve_cfg():  若调用方传入 schedule-derived tile → 用它
                              否则回退 _fa_cfg 启发式  (行为不变，向后兼容)
   ▼
{{kernel_name}}_kernel[grid](..., BLOCK_N=cfg["BLOCK_N"], ...)  (真实 codegen)
```

## 3. 实现（最小、向后兼容、可验证）

FA 模板 launcher 已有 `_cfg_override` seam（FA-v1 sweep 用）。**复用它作为 ScheduleIR
驱动入口**：新增一个纯函数 `_cfg_from_schedule_tile(tile_factors, resources) -> dict | None`，
把 ScheduleIR 的 `LoopNest.tile_factors`（真实 IR 形状 = 位置列表 `[q_block, kv_block]`）
+ `ResourceBinding`（`{"warps":.., "num_stages":..}`）翻译成 launcher 的 cfg dict。调用方
（未来的 ScheduleIR-driven pipeline）传 `_cfg_override=_cfg_from_schedule_tile(nest.tile_factors,
sched.resources.to_dict())` 即完成真实驱动。

**验证**（本次交付的可执行证明）：一个测试构造带 `tile_factors` 的 ScheduleIR
loop_nest → 翻译 → 传给 FA launcher → 断言 kernel 实际用了该 tile（通过 `_cfg_override`
路径 + 输出 correctness），证明「ScheduleIR 字段 → 生产 codegen」这条链真实通电。

## 4. 不做什么（诚实边界）

- 不改 lowering.py（L2→L1 已真实，无需动）。
- 不让所有 backend/所有 op 都消费 ScheduleIR（那是全量重构，需 Leon 批方向 + 多周）。
- 不移除 `_fa_cfg` 启发式（它是 no-schedule 场景的默认，且 FA-v4 数据在其中）。

**增量路线**：这条 FA 链验证通过后，同模式可按 op 逐个接（matmul tile、norm block），
每接一个就多一条真实承重链，避免「一次性大爆炸」重构。

*Design by Kitty, 2026-07-30. 实现 + 验证见同批 commit。*
