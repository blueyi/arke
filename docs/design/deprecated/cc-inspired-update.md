# Arke — 借鉴 Claude Code 的工程设计更新

> 基于 Claude Code 源码分析，将其验证过的工程模式迁移到 Arke
> Date: 2026-03-31
> 前置：patch-v2.1.2.md, design-review.md, e2e-flow.md

---

## 一、核心洞察：Arke 的 LLM 优化循环就是一个 Agentic Loop

```
Claude Code:                          Arke:
  while(true) {                         while(budget_remaining) {
    API call (stream)                     LLM call (tool-use)
    parse tool_use                        parse tool call
    execute tool                          execute ArkeEnv tool
    inject tool_result                    inject tool result
    check compact                         check token budget
    if end_turn → break                   if LLM says done → break
  }                                     }
```

**两者是同构的。** Claude Code 在 51 万行代码中验证过的模式可以直接迁移。

---

## 二、设计迁移方案

### 迁移 1：AsyncGenerator 驱动的优化循环 ⭐⭐⭐

**Claude Code 的做法：**
```typescript
async function* queryLoop(): AsyncGenerator<StreamEvent | Message, Terminal>
  for await (event of queryModel()) { yield event }
  yield* runTools(toolUseBlocks)
```

**迁移到 Arke：**

```python
# arke/agent/runner.py

from typing import AsyncGenerator
from dataclasses import dataclass

@dataclass
class OptimizationEvent:
    """优化过程中的事件流"""
    type: str  # "decision" | "validation" | "compile" | "profile" | "error" | "done"
    data: dict

async def optimization_loop(
    env: ArkeEnv,
    llm: LLMProvider,
    config: OptimizationConfig,
) -> AsyncGenerator[OptimizationEvent, None]:
    """
    核心优化循环 — AsyncGenerator 模式。
    
    yield 事件给外层消费者（CLI 实时显示 / API 返回 / 日志记录）。
    消费者可随时 .athrow() 中断或 .aclose() 取消。
    """
    messages = build_initial_messages(env)
    budget = OptimizationBudget(config)
    trajectory = TrajectoryRecorder()
    best_result: CompileResult | None = None
    
    while not budget.exhausted:
        # ─── 1. LLM 调用 ───
        response = await llm.chat(messages, tools=env.get_tool_schemas())
        
        if not response.tool_calls:
            # LLM 认为优化完成
            break
        
        # ─── 2. 工具执行（带并发分区）───
        async for event in execute_tools(env, response.tool_calls, budget):
            yield event  # 流式转发给消费者
            trajectory.record(event)
            
            # 注入工具结果
            if event.type == "tool_result":
                messages.append(tool_result_message(event))
        
        # ─── 3. Token 预算检查 ───
        if should_compact(messages, config.max_context_tokens):
            messages = await compact_optimization_context(messages, llm)
            yield OptimizationEvent("compact", {"message_count": len(messages)})
        
        budget.tick()
    
    # ─── 4. 收尾：对比 fallback ───
    fallback_result = env.fallback.evaluate()
    final = best_result if (best_result and best_result > fallback_result) else fallback_result
    
    yield OptimizationEvent("done", {
        "source": "llm" if final == best_result else "fallback",
        "performance": final.performance,
        "trajectory": trajectory.export(),
    })
```

**消费者端：**

```python
# CLI 消费
async for event in optimization_loop(env, llm, config):
    match event.type:
        case "decision":
            print(f"  Step {event.data['step']}: {event.data['kind']} — {event.data['rationale'][:60]}...")
        case "validation":
            status = "✅" if event.data["pass"] else "❌"
            print(f"    V0: {status}")
        case "compile":
            print(f"    Compile #{event.data['attempt']}: {event.data['vs_cublas']:.0%} cuBLAS")
        case "compact":
            print(f"  [compact] context reduced to {event.data['message_count']} messages")
        case "done":
            print(f"\n✅ Final: {event.data['performance']['vs_cublas']:.0%} cuBLAS ({event.data['source']})")

# Python API 消费
results = [e async for e in optimization_loop(env, llm, config) if e.type == "done"]
```

**好处：**
- CLI / API / Jupyter 都能消费同一个 generator
- 实时进度反馈（不等全部完成）
- `async for` 的 `break` 自动触发 generator cleanup
- 轨迹记录和事件处理解耦

---

### 迁移 2：Tool 声明式接口 ⭐⭐⭐

**Claude Code 的做法：**
每个 Tool 自描述安全属性（`isConcurrencySafe`, `isReadOnly`, `maxResultSizeChars`），编排器据此自动决策。

**迁移到 Arke：**

```python
# arke/agent/tools/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ToolMeta:
    """工具的声明式元信息"""
    name: str
    concurrent_safe: bool      # 可与其他工具并发？
    idempotent: bool           # 幂等的？（重试安全）
    requires_compile: bool     # 需要 GPU 编译？（昂贵操作）
    mutates_strategy: bool     # 是否修改 Strategy IR？
    estimated_cost: str        # "cheap" | "medium" | "expensive"
    budget_type: str           # "decision" | "compile" | "free"

class ArkeTool(ABC):
    meta: ToolMeta
    
    @abstractmethod
    def schema(self) -> dict:
        """JSON Schema for LLM tool-use"""
    
    @abstractmethod
    async def execute(self, params: dict, env: ArkeEnv) -> ToolResult:
        """执行工具"""
    
    def validate_params(self, params: dict, env: ArkeEnv) -> ValidationResult:
        """参数静态验证（V0 的一部分）"""
        return ValidationResult(ok=True)


# 具体工具声明
class AnalyzeComputeTool(ArkeTool):
    meta = ToolMeta(
        name="analyze_compute",
        concurrent_safe=True,    # 只读分析，可并发
        idempotent=True,         # 结果确定，可缓存
        requires_compile=False,
        mutates_strategy=False,
        estimated_cost="cheap",
        budget_type="free",      # 不消耗决策预算
    )

class ApplyDecisionTool(ArkeTool):
    meta = ToolMeta(
        name="apply_decision",
        concurrent_safe=False,   # 修改 Strategy IR，必须串行
        idempotent=False,
        requires_compile=False,
        mutates_strategy=True,
        estimated_cost="cheap",
        budget_type="decision",  # 消耗决策预算
    )

class VerifyCorrectnessTool(ArkeTool):
    meta = ToolMeta(
        name="verify_correctness",
        concurrent_safe=False,   # 需要 GPU
        idempotent=True,
        requires_compile=True,   # 触发 codegen + compile
        mutates_strategy=False,
        estimated_cost="expensive",
        budget_type="compile",   # 消耗编译预算
    )

class CompileAndProfileTool(ArkeTool):
    meta = ToolMeta(
        name="compile_and_profile",
        concurrent_safe=False,
        idempotent=True,
        requires_compile=True,
        mutates_strategy=False,
        estimated_cost="expensive",
        budget_type="compile",
    )
```

**工具编排器利用声明式元信息：**

```python
# arke/agent/tools/orchestrator.py

async def execute_tools(
    env: ArkeEnv,
    tool_calls: list[ToolCall],
    budget: OptimizationBudget,
) -> AsyncGenerator[OptimizationEvent, None]:
    """
    借鉴 Claude Code 的 toolOrchestration.ts 分区策略。
    """
    for batch in partition_tool_calls(tool_calls):
        if batch.all_concurrent_safe:
            # 并发执行所有只读工具
            results = await asyncio.gather(*[
                execute_single(env, tc, budget) for tc in batch.calls
            ])
            for r in results:
                yield r
        else:
            # 串行执行
            for tc in batch.calls:
                # 预算检查
                tool = env.get_tool(tc.name)
                if tool.meta.budget_type == "decision":
                    if budget.decisions_exhausted:
                        yield OptimizationEvent("budget_warning", {
                            "type": "decisions", "remaining": 0
                        })
                        continue
                    budget.use_decision()
                elif tool.meta.budget_type == "compile":
                    if budget.compiles_exhausted:
                        yield OptimizationEvent("budget_warning", {
                            "type": "compiles", "remaining": 0
                        })
                        continue
                    budget.use_compile()
                
                async for event in execute_single(env, tc, budget):
                    yield event


def partition_tool_calls(calls: list[ToolCall]) -> list[ToolBatch]:
    """
    Claude Code 的分区算法：
    连续的 concurrent_safe 工具合并为一个并发批
    遇到非 safe 的工具断开，单独串行
    
    [analyze, get_hw_profile, apply_decision, list_legal_actions]
    → Batch([analyze, get_hw_profile], concurrent=True)
    → Batch([apply_decision], concurrent=False)
    → Batch([list_legal_actions], concurrent=True)
    """
    batches = []
    current = []
    current_safe = True
    
    for call in calls:
        tool_safe = get_tool(call.name).meta.concurrent_safe
        if tool_safe == current_safe and current:
            current.append(call)
        else:
            if current:
                batches.append(ToolBatch(current, all_concurrent_safe=current_safe))
            current = [call]
            current_safe = tool_safe
    
    if current:
        batches.append(ToolBatch(current, all_concurrent_safe=current_safe))
    return batches
```

---

### 迁移 3：分段 Prompt Cache ⭐⭐⭐

**Claude Code 的做法：** system prompt 分 4 段，各自独立缓存。

**迁移到 Arke：**

```python
# arke/agent/prompts.py

def build_system_prompt(env: ArkeEnv) -> list[dict]:
    """
    分段构建 system prompt，每段独立的缓存行为。
    
    Segment 1 (全局缓存): 角色定义 + 优化知识
    Segment 2 (硬件级缓存): HW Profile（同硬件的请求共享）
    Segment 3 (算子级缓存): 当前算子的 Semantic IR
    Segment 4 (每步变化): 当前 Strategy IR 状态 + budget
    """
    return [
        # Segment 1: 全局缓存 — 所有优化任务共享
        {
            "type": "text",
            "text": OPTIMIZATION_EXPERT_PROMPT,  # 角色定义 + GPU 优化知识
            "cache_control": {"type": "ephemeral"}
        },
        
        # Segment 2: 硬件级缓存 — 同硬件的任务共享
        {
            "type": "text",
            "text": f"## Target Hardware\n{json.dumps(env.hw_profile, indent=2)}",
            "cache_control": {"type": "ephemeral"}
        },
        
        # Segment 3: 算子级缓存 — 同算子的多次优化共享
        {
            "type": "text",
            "text": f"## Kernel Specification\n{json.dumps(env.semantic_ir, indent=2)}\n"
                    f"## Auto Analysis\n{json.dumps(env.auto_analysis, indent=2)}",
            "cache_control": {"type": "ephemeral"}
        },
        
        # Segment 4: 不缓存 — 每步都变
        {
            "type": "text",
            "text": build_dynamic_context(env),
            # 无 cache_control → 不缓存
        },
    ]

def build_dynamic_context(env: ArkeEnv) -> str:
    """每步都变的动态上下文"""
    return f"""## Current State
Decisions made: {env.decision_count} / {env.budget.max_decisions}
Compiles used: {env.compile_count} / {env.budget.max_compiles}
Best performance: {env.best_performance or 'not yet compiled'}

## Current Strategy IR (summary)
{env.strategy_ir.summary()}

## Available actions hint
{env.get_action_hint()}
"""
```

**效果分析：**
```
50 步优化 → 50 次 API 调用
  Segment 1 (优化知识): ~2000 token → 缓存命中 49 次 → 节省 ~98K token
  Segment 2 (HW Profile): ~800 token → 缓存命中 49 次 → 节省 ~39K token
  Segment 3 (Semantic IR): ~500 token → 缓存命中 49 次 → 节省 ~24K token
  Segment 4 (动态状态): ~300 token → 每次重算 → 15K token

总计: 50步约 ~176K token 节省（vs 无缓存的 ~190K → 只需 ~15K fresh）
```

---

### 迁移 4：Context Compact for 长优化过程 ⭐⭐

**Claude Code 的做法：** 对话太长 → 让 Claude 生成摘要 → 替换历史

**迁移到 Arke：**

```python
# arke/agent/compact.py

async def compact_optimization_context(
    messages: list[dict],
    llm: LLMProvider,
    env: ArkeEnv,
) -> list[dict]:
    """
    当优化过程的消息历史超过 token 限制时，压缩上下文。
    
    关键：不能丢失任何已做的决策和当前 strategy state。
    只压缩中间的观察/分析过程。
    """
    
    # 1. 构建摘要请求
    compact_prompt = f"""Summarize this GPU kernel optimization conversation.
    
MUST preserve exactly:
- All optimization decisions made (kind, params, rationale)
- Current Strategy IR state
- All compile results (correctness, performance numbers)
- Any errors encountered and how they were resolved
- The current best performance

DO NOT preserve:
- Intermediate observe() outputs (they're derived from current state)
- Detailed legal_actions listings
- Full HW profile dumps (already in system prompt)

Current Strategy IR (authoritative — do not summarize, reference as-is):
{json.dumps(env.strategy_ir.to_dict(), indent=2)}
"""
    
    # 2. 调用 LLM 生成摘要
    summary = await llm.chat([
        {"role": "system", "content": "You are summarizing a GPU optimization session."},
        {"role": "user", "content": compact_prompt + "\n\nConversation to summarize:\n" +
         format_messages(messages[:-5])},  # 保留最近 5 条
    ])
    
    # 3. 替换消息历史
    return [
        {"role": "user", "content": f"[Optimization Context Summary]\n{summary.content}"},
        *messages[-5:],  # 保留最近 5 条完整消息
    ]


async def should_compact(messages: list[dict], max_tokens: int) -> bool:
    """预估 token → 判断是否需要 compact"""
    estimated = estimate_token_count(messages)
    return estimated > max_tokens * 0.8  # 80% 阈值


async def reactive_compact(
    messages: list[dict],
    llm: LLMProvider,
    env: ArkeEnv,
    error: Exception,
) -> list[dict] | None:
    """
    借鉴 Claude Code 的 reactive compact:
    API 返回 prompt_too_long → 被动压缩 → 重试
    """
    if "prompt_too_long" in str(error) or "context_length" in str(error):
        return await compact_optimization_context(messages, llm, env)
    return None
```

---

### 迁移 5：大结果持久化（observe_diff）⭐⭐

**Claude Code 的做法：** 大 tool result → 持久化到文件 → messages 只放摘要 + 路径

**迁移到 Arke：**

```python
# arke/agent/tools/result_management.py

MAX_INLINE_RESULT_CHARS = 3000  # 超过此阈值 → 压缩

def manage_tool_result(tool_name: str, result: dict) -> dict:
    """
    对大结果做智能压缩，不丢失关键信息。
    
    核心思想：LLM 不需要每次看完整 Strategy IR。
    只需要看 delta（这步改了什么）。
    """
    result_str = json.dumps(result)
    
    if len(result_str) <= MAX_INLINE_RESULT_CHARS:
        return result  # 小结果直接返回
    
    # 大结果 → 返回摘要 + delta
    match tool_name:
        case "list_legal_actions":
            # 47 个候选太多 → 只给 top 10 + 总数
            return {
                "top_actions": result["legal_actions"][:10],
                "total_count": len(result["legal_actions"]),
                "blocked_count": len(result.get("blocked_actions", [])),
                "hint": result.get("hint", ""),
                "_note": f"Showing top 10 of {len(result['legal_actions'])} actions. "
                         f"Call with kind= filter for specific categories."
            }
        
        case "observe":
            # 完整 state 太大 → 只返回 delta
            return {
                "type": "state_delta",
                "changed_fields": result.get("delta", {}),
                "summary": result.get("summary", ""),
                "resource_usage": result.get("resource_usage", {}),
                "_note": "Full state available via observe(full=true) if needed."
            }
        
        case "verify_correctness":
            # 数值结果不大，但参考数据可能大
            return {
                "pass": result["pass"],
                "max_absolute_error": result.get("max_absolute_error"),
                "tolerance": result.get("tolerance"),
                "execution_time_ms": result.get("execution_time_ms"),
                # 去掉完整的 output tensor 数据
            }
        
        case _:
            # 通用策略：截断 + 提示
            return {
                "_truncated": True,
                "_original_size": len(result_str),
                "data": result_str[:MAX_INLINE_RESULT_CHARS],
            }
```

---

### 迁移 6：Fallback + Retry 链 ⭐⭐

**Claude Code 的做法：** withRetry + fallback model + reactive compact

**迁移到 Arke：**

```python
# arke/agent/resilience.py

class ResilientRunner:
    """
    三层容错：
    1. 工具执行级：V0 验证失败 → 自动 rollback + 给 LLM 错误信息
    2. API 调用级：rate limit / timeout → 指数退避 + provider fallback
    3. 优化循环级：LLM 搜索失败 → fallback strategy 兜底
    """
    
    async def run_with_resilience(
        self, env: ArkeEnv, config: OptimizationConfig
    ) -> AsyncGenerator[OptimizationEvent, None]:
        
        # 建立 provider fallback 链
        providers = self._build_fallback_chain(config)
        current_provider_idx = 0
        compact_attempted = False
        
        try:
            async for event in optimization_loop(env, providers[0], config):
                yield event
                
        except RateLimitError:
            # Provider 级 fallback
            current_provider_idx += 1
            if current_provider_idx < len(providers):
                yield OptimizationEvent("provider_fallback", {
                    "from": providers[current_provider_idx - 1].name,
                    "to": providers[current_provider_idx].name,
                })
                async for event in optimization_loop(
                    env, providers[current_provider_idx], config
                ):
                    yield event
            else:
                raise
                
        except ContextLengthError as e:
            # Reactive compact
            if not compact_attempted:
                compact_attempted = True
                yield OptimizationEvent("reactive_compact", {})
                # 压缩后重试
                async for event in optimization_loop(env, providers[0], config):
                    yield event
            else:
                raise
    
    async def _finalize(self, env: ArkeEnv) -> OptimizationEvent:
        """对比 LLM 结果 vs fallback"""
        llm_best = env.best_compile_result
        fallback = await env.fallback.evaluate()
        
        if llm_best and llm_best.performance > fallback.performance:
            return OptimizationEvent("done", {"source": "llm", **llm_best})
        else:
            return OptimizationEvent("done", {
                "source": "fallback",
                "note": "LLM did not improve over baseline strategy",
                **fallback
            })
```

---

### 迁移 7：Tool Result Budget（跨 compact 的结果管理）⭐

**Claude Code 的做法：** `ContentReplacementState` 跟踪哪些 tool result 被替换了，跨 compact 保持一致

**迁移到 Arke：**

```python
# arke/agent/state.py

class OptimizationState:
    """
    跨 compact 保持一致的优化状态。
    
    即使 messages 被 compact 压缩了，这些状态始终是 ground truth：
    """
    strategy_ir: StrategyIR            # 当前策略 IR（完整的，不被 compact 影响）
    compile_results: list[CompileResult]  # 所有编译结果
    checkpoints: dict[str, StrategyIR]   # 保存的检查点
    decision_log: list[Decision]         # 所有决策（含 rationale）
    best_result: CompileResult | None    # 最佳结果
    
    def to_context_for_compact(self) -> str:
        """生成 compact 时的 ground truth 上下文"""
        return f"""
## Authoritative State (do not summarize — use as-is)
Strategy IR: {json.dumps(self.strategy_ir.to_dict())}
Decisions: {len(self.decision_log)}
Best performance: {self.best_result.vs_cublas if self.best_result else 'N/A'}
Compile attempts: {len(self.compile_results)}
"""
```

---

## 三、更新后的项目结构

```
arke/
├── agent/                          # LLM Agent 系统（核心新增）
│   ├── runner.py                   # 核心优化循环 (AsyncGenerator)
│   ├── prompts.py                  # 分段 system prompt 构建
│   ├── compact.py                  # Context compact (预测式 + 反应式)
│   ├── resilience.py               # 三层容错 (tool/API/loop)
│   ├── state.py                    # 跨 compact 的 ground truth 状态
│   ├── trajectory.py               # 轨迹记录
│   ├── tools/
│   │   ├── base.py                 # ArkeTool ABC + ToolMeta 声明式接口
│   │   ├── orchestrator.py         # 工具编排 + 并发分区
│   │   ├── result_management.py    # 大结果压缩 + delta 模式
│   │   ├── create_kernel.py        # Tool: create_kernel
│   │   ├── get_hw_profile.py       # Tool: get_hw_profile
│   │   ├── analyze_compute.py      # Tool: analyze_compute
│   │   ├── list_legal_actions.py   # Tool: list_legal_actions
│   │   ├── apply_decision.py       # Tool: apply_decision
│   │   ├── verify_correctness.py   # Tool: verify_correctness
│   │   ├── compile_and_profile.py  # Tool: compile_and_profile
│   │   ├── observe.py              # Tool: observe (支持 delta 模式)
│   │   ├── checkpoint.py           # Tool: checkpoint
│   │   └── rollback.py             # Tool: rollback
│   └── providers/
│       ├── base.py                 # LLMProvider ABC
│       ├── openai_compat.py        # OpenAI-compatible
│       ├── anthropic.py            # Anthropic Claude
│       └── fallback.py             # Fallback 链
├── ir/
│   ├── semantic.py                 # Semantic IR
│   ├── strategy.py                 # Strategy IR
│   └── schema/                     # JSON Schema 定义
├── backend/
│   ├── base.py                     # ArkeBackend ABC
│   ├── triton/                     # NVIDIA Triton 后端
│   └── triton_ascend/              # Ascend 后端 (Phase 2)
├── validation/
│   ├── static.py                   # V0 静态验证
│   ├── numerical.py                # V1 数值验证
│   └── performance.py              # V2 性能验证
├── integration/
│   └── torch_ops.py                # PyTorch custom op 注册
└── cli.py                          # CLI 入口
```

---

## 四、新增/修改的任务

| 任务 | Week | 说明 | 源自 Claude Code |
|------|:----:|------|:----------------|
| AsyncGenerator 优化循环 | W3 | runner.py 核心实现 | query.ts 的 queryLoop |
| 声明式 Tool 接口 | W1 | ToolMeta + ArkeTool ABC | Tool.ts 的全维度接口 |
| 工具并发分区 | W3 | orchestrator.py | toolOrchestration.ts |
| 分段 Prompt Cache | W3 | prompts.py 4 段构建 | claude.ts 的 cache_control |
| Context Compact | W4 | compact.py 预测+反应 | compact.ts + autoCompact.ts |
| 大结果 delta 压缩 | W2 | result_management.py | toolResultStorage.ts |
| 三层容错 | W4 | resilience.py | withRetry + fallback |
| 跨 compact 状态 | W3 | state.py ground truth | ContentReplacementState |

---

## 五、验证：这些迁移解决了什么问题？

| Arke 原有问题 | Claude Code 迁移方案 | 解决效果 |
|:-------------|:-------------------|:--------|
| 50 步优化的 token 成本不可控 | 分段 prompt cache + compact | token 成本降低 80%+ |
| LLM 看太大的 legal_actions 列表 | 大结果 delta 压缩 | 只给 top 10 + 总数 |
| API rate limit 导致优化中断 | Provider fallback 链 | 自动切换备用 LLM |
| prompt_too_long 导致崩溃 | Reactive compact | 被动压缩 + 重试 |
| 工具执行无法并发 | 声明式并发分区 | analyze + get_hw 并行 |
| CLI/API/Jupyter 需要不同的输出 | AsyncGenerator 事件流 | 统一接口，消费者自行处理 |
| compact 后丢失已做决策 | Ground truth state | Strategy IR 永远完整 |

---

*版本：v1.0 | 创建日期：2026-03-31*
