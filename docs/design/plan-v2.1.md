# Arke Project — 修正执行计划 v2.1

> 核心修正：从"先做语言再接 AI"转向"先让 LLM 跑通，再补人类界面"
> 预计 MVP 周期：8 周
> Date: 2026-03-31
> Based on: e2e-design-v2.md + LLM-Native 哲学审视

---

## 一、修正理由

### v2.0 Plan 的问题

原 plan 按传统编译器思路排列：语言语法 → IR 规范 → 工具链。
但 Arke 的核心主张是 **LLM-Native**，执行顺序应该反映这一点：

```
v2.0 Plan（传统编译器思路）：
  语言语法设计 → IR 规范 → Parser → Codegen → AI 集成
  问题：AI 是最后一步，但 AI 是第一用户

v2.1 Plan（LLM-Native 思路）：
  LLM 协议 + IR → 验证系统 → Codegen → LLM 循环跑通 → 语法/人类界面
  核心：先让 AI 能用，再让人能读
```

### 关键认知

1. **LLM 不需要 .ak 语法**——JSON IR 就是 LLM 的工作界面
2. **验证系统比优化系统更重要**——LLM 不可信，验证是安全网
3. **必须有对比评估**——证明 Arke 比"直接让 LLM 写 Triton"更好
4. **Cost model 初期就是 compile_and_profile()**——LLM 自己的推理 + 实测反馈够用

---

## 二、总览

```
Week 1-2       Week 3-4       Week 5-6       Week 7-8
  │              │              │              │
  ▼              ▼              ▼              ▼
┌────────────────────────────────────────────────────────┐
│ Stream 1: LLM Agent Protocol（核心）                    │
│ [Tool Schema] → [ArkeEnv] → [LLM 联调] → [闭环优化]    │
├────────────────────────────────────────────────────────┤
│ Stream 2: IR + 验证 + Codegen（基础）                   │
│ [IR Schema] → [验证MVP] → [Triton CG] → [端到端]       │
├────────────────────────────────────────────────────────┤
│ Stream 3: 语言语法 + 人类界面（后置）                     │
│                        [Parser] → [CLI + Inspect]       │
├────────────────────────────────────────────────────────┤
│ Stream 4: 评估框架（贯穿）                               │
│              [任务定义] → [Baseline] → [对比实验]        │
└────────────────────────────────────────────────────────┘
```

---

## 三、Stream 1：LLM Agent Protocol（核心，最高优先级）

### 设计原则

LLM 与 Arke 的交互不是"写代码"，而是**通过结构化 API 逐步构建和优化计算图**。

### 3.1 Tool-Use Schema 定义（Week 1）

定义 LLM 可调用的全部 tools，遵循 OpenAI function calling / Anthropic tool_use 格式。

**Phase 1 Tool 集合（最小可用）：**

```yaml
# === 构建类 ===
create_kernel:
  desc: "创建一个新的 kernel（语义计算图）"
  params: {name, params[], return_type, computations[]}
  returns: SemanticIR (JSON)

# === 观测类 ===
get_semantic_ir:
  desc: "获取当前 kernel 的语义 IR"
  returns: SemanticIR (JSON)

get_hw_profile:
  desc: "获取目标硬件的完整参数"
  params: {target: "nvidia_ampere" | "ascend_a3"}
  returns: HWProfile (JSON)

get_current_strategy:
  desc: "获取当前的优化策略"
  returns: StrategyIR (JSON)

get_resource_usage:
  desc: "获取当前策略下的资源使用估算"
  returns: ResourceReport (JSON)

# === 分析类 ===
analyze_compute:
  desc: "分析计算特征（算术强度、瓶颈类型、融合机会）"
  returns: AnalysisReport (JSON)

list_legal_actions:
  desc: "列出当前状态下所有合法的优化动作"
  params: {kind?: "tile"|"fuse"|"parallel"|"place"|"reorder"}
  returns: Action[] with estimated_impact

# === 决策类 ===
apply_decision:
  desc: "应用一个优化决策"
  params: {kind, params, rationale}
  returns: {success, new_state, resource_delta, validation_result}

rollback:
  desc: "回滚最近 N 步决策"
  params: {steps: int}
  returns: State

checkpoint:
  desc: "保存当前状态为检查点"
  returns: {checkpoint_id}

restore:
  desc: "恢复到指定检查点"
  params: {checkpoint_id}
  returns: State

# === 验证类 ===
verify_correctness:
  desc: "编译并验证数值正确性（vs NumPy 参考）"
  returns: {pass: bool, max_error, error_locations[]}

verify_constraints:
  desc: "检查硬件约束是否满足"
  returns: {pass: bool, violations[]}

# === 编译/性能类 ===
compile_and_profile:
  desc: "编译为目标代码并实际 profiling"
  returns: {latency_us, tflops, bandwidth_util, roofline_eff, vs_baseline}
```

**交付物：**
- [ ] `arke/agent/tools_schema.py` — Tool 定义（JSON Schema 格式）
- [ ] `arke/agent/tools_schema.json` — 导出为标准 function calling schema
- [ ] `docs/agent-protocol/01-tools.md` — 协议文档

### 3.2 Session 生命周期设计（Week 1）

```
LLM 编译 Session 的完整生命周期：

1. INIT
   输入：算子规格（自然语言 or JSON） + 硬件目标
   系统：创建 SemanticIR → 初始化 ArkeEnv
   LLM 收到：initial_state + hw_profile + system_prompt

2. ANALYZE
   LLM 调用：analyze_compute() → 理解计算特征
   LLM 调用：list_legal_actions() → 了解可选优化
   LLM（推理）：制定优化计划

3. OPTIMIZE (循环)
   LLM 调用：apply_decision() → 应用优化
   自动执行：verify_constraints() → 即时反馈
   LLM（推理）：根据反馈调整
   重复直到满意或预算耗尽

4. VERIFY
   LLM 调用：verify_correctness() → 数值验证
   LLM 调用：compile_and_profile() → 性能验证
   如果失败 → 回到 OPTIMIZE

5. FINALIZE
   输出：最终 StrategyIR + 生成的代码 + rationale chain
   导出：优化轨迹（用于学习）
```

**System Prompt 模板：**

```
You are an expert GPU kernel optimizer working with the Arke toolchain.

Your task: optimize the given kernel for {target_hw} to maximize performance.

Available tools: [tool definitions]

Hardware target: {hw_profile_summary}

Workflow:
1. Call get_semantic_ir() to understand the computation
2. Call analyze_compute() to identify optimization opportunities
3. Call list_legal_actions() to see available optimizations
4. Apply decisions one at a time with apply_decision()
5. After each decision, check the validation result
6. When satisfied, call compile_and_profile() to get actual performance
7. If performance is insufficient, adjust and retry

Always provide a rationale for each decision.
Budget: {max_steps} optimization steps, {max_compiles} compile attempts.
```

**交付物：**
- [ ] `arke/agent/session.py` — Session 管理
- [ ] `arke/agent/prompts.py` — System prompt 模板
- [ ] `docs/agent-protocol/02-session.md`

### 3.3 ArkeEnv 实现（Week 2-3）

核心引擎，连接 tools → IR → 验证 → 编译。

```python
class ArkeEnv:
    """LLM Agent 的核心交互环境"""

    def __init__(self, semantic_ir: SemanticIR, hw_target: str):
        self.semantic = semantic_ir
        self.strategy = StrategyIR(target_hw=hw_target)
        self.hw = load_hw_profile(hw_target)
        self.history = []         # 决策历史
        self.checkpoints = {}     # 检查点
        self.step = 0

    # --- 每个 tool 对应一个方法 ---
    def get_semantic_ir(self) -> dict: ...
    def get_hw_profile(self) -> dict: ...
    def analyze_compute(self) -> dict: ...
    def list_legal_actions(self, kind=None) -> list[dict]: ...
    def apply_decision(self, kind, params, rationale) -> dict: ...
    def rollback(self, steps=1) -> dict: ...
    def verify_constraints(self) -> dict: ...
    def verify_correctness(self) -> dict: ...
    def compile_and_profile(self) -> dict: ...
    def export_trajectory(self) -> dict: ...
```

**交付物：**
- [ ] `arke/engine/env.py` — ArkeEnv 核心
- [ ] `arke/engine/actions.py` — 合法动作枚举
- [ ] `arke/engine/state.py` — 状态管理 + 检查点

### 3.4 LLM 联调（Week 4-5）

用真实 LLM（Claude/GPT/Qwen）跑通完整的 tool-use 优化循环。

**联调步骤：**
1. 手动构造 matmul 的 SemanticIR
2. 启动 ArkeEnv session
3. LLM 通过 tool-use 完成优化
4. 编译 → profiling → 输出性能

**交付物：**
- [ ] `arke/agent/runner.py` — LLM agent 运行器（支持多 provider）
- [ ] `examples/agent_matmul.py` — matmul 端到端 agent demo
- [ ] `examples/agent_softmax.py` — softmax agent demo

### 3.5 闭环优化（Week 6-8）

在联调基础上完善：
- 错误恢复（LLM 决策导致验证失败时的自动引导）
- 轨迹记录和导出
- 多轮优化（LLM 从自己之前的轨迹学习）

**交付物：**
- [ ] `arke/learn/trajectory.py` — 轨迹记录
- [ ] `arke/agent/recovery.py` — 错误恢复策略

---

## 四、Stream 2：IR + 验证 + Codegen（基础设施）

### 4.1 Semantic IR 完善 + JSON Schema（Week 1）

当前 `arke/ir/semantic.py` 已有骨架，需要：
- 完善 JSON Schema 定义（用于 tool 的输入输出验证）
- 补充算子库（P0: matmul, relu, softmax, add, mul, reduce_sum, reduce_max）
- 实现完整的 JSON 序列化/反序列化

**P0 算子定义**（每个包含：签名、语义公式、代数性质、融合规则）：

| 算子 | 类别 | 语义 |
|------|------|------|
| matmul | 线性代数 | C[i,j] = Σ_k A[i,k] * B[k,j] |
| batch_matmul | 线性代数 | C[b,i,j] = Σ_k A[b,i,k] * B[b,k,j] |
| relu | 逐元素 | Y = max(X, 0) |
| gelu | 逐元素 | Y = X * Φ(X) |
| add | 逐元素 | Y = A + B |
| mul | 逐元素 | Y = A * B |
| softmax | 归约 | Y[i,j] = exp(X[i,j]) / Σ_j exp(X[i,j]) |
| reduce_sum | 归约 | Y[i] = Σ_j X[i,j] |
| reduce_max | 归约 | Y[i] = max_j X[i,j] |
| transpose | 数据移动 | Y[j,i] = X[i,j] |

**交付物：**
- [ ] `arke/ir/schemas/semantic.schema.json`
- [ ] `arke/ir/schemas/strategy.schema.json`
- [ ] `arke/ir/ops/catalog.py` — 算子目录
- [ ] `arke/ir/ops/definitions.py` — 算子语义定义

### 4.2 Strategy IR 完善（Week 1-2）

当前 `arke/ir/schedule.py` 已有骨架，需要：
- 与 Semantic IR 的关联验证（strategy 引用的算子必须存在于 semantic graph）
- 约束检查（每个 decision 的 precondition 验证）
- 搜索空间枚举（给定 semantic + hw，列出所有合法 tiling/fusion/etc.）

**交付物：**
- [ ] `arke/ir/strategy.py` — 重命名并完善（schedule → strategy 对齐 v2 术语）
- [ ] `arke/engine/legal_actions.py` — 合法动作枚举引擎
- [ ] `arke/ir/targets/nvidia_ampere.json` — Ampere HW Profile

### 4.3 验证系统 MVP（Week 2，高优先级）

**V0 验证（静态，每次 apply 自动执行，<1ms）：**
- Shape 一致性检查
- 硬件约束检查（shared memory, registers, threads）
- 变换合法性（tile factor 能整除 loop bound, etc.）

**V1 验证（数值，编译后执行，~100ms）：**
- 生成 NumPy 参考实现
- 随机输入 → 对比 Arke kernel 输出 vs NumPy 输出
- 容差检查（f16: atol=1e-2, rtol=1e-2; f32: atol=1e-5, rtol=1e-5）

**V2 验证（性能，profiling，~1s）：**
- 与 cuBLAS / PyTorch 基线对比
- Roofline efficiency 计算
- 资源利用率统计

**交付物：**
- [ ] `arke/engine/validator.py` — V0 静态验证
- [ ] `arke/engine/numerical_check.py` — V1 数值验证
- [ ] `arke/engine/profiler.py` — V2 性能验证

### 4.4 Triton Codegen（Week 3-4）

两条路径并行：

**路径 A：模板翻译（确定性，保底）**
- 每种 (算子类型, strategy pattern) 组合有一套 Jinja2 模板
- matmul: Semantic + tile/parallel/place → Triton kernel 代码
- 优点：可靠；缺点：每种组合都要手写模板

**路径 B：LLM 生成（实验性，探索）**
- 输入：Semantic IR + Strategy IR（JSON）
- LLM 根据 IR 生成完整的 Triton kernel
- 通过 V1 验证检查正确性
- 优点：灵活；缺点：需要验证

```
路径 A（模板翻译）：
  SemanticIR + StrategyIR → Template Engine → Triton Python Code → compile → binary

路径 B（LLM 生成）：
  SemanticIR + StrategyIR → LLM (with prompt) → Triton Python Code → V1 verify → binary
```

**交付物：**
- [ ] `arke/backend/triton_templates/` — 模板目录
- [ ] `arke/backend/triton_template_engine.py` — 路径 A
- [ ] `arke/backend/triton_llm_gen.py` — 路径 B
- [ ] `arke/backend/compiler.py` — 编译 + 加载 + 执行

### 4.5 端到端集成（Week 5-6）

将所有组件串联：
```
SemanticIR → ArkeEnv → LLM tool-use → StrategyIR → Triton Codegen → GPU Execute → Verify
```

**交付物：**
- [ ] `arke/pipeline.py` — 端到端 pipeline
- [ ] `tests/test_e2e_matmul.py` — matmul 端到端测试
- [ ] `tests/test_e2e_softmax.py` — softmax 端到端测试

---

## 五、Stream 3：语言语法 + 人类界面（后置）

### 5.1 .ak Parser（Week 5-6）

用 Lark 实现 EBNF 语法解析，输出 AST → 转换为 Semantic IR。

**此时 Parser 是"人类输入适配器"，不是核心路径。**

**交付物：**
- [ ] `arke/lang/arke.lark` — EBNF 语法
- [ ] `arke/lang/parser.py` — Parser 实现
- [ ] `arke/lang/ast_to_ir.py` — AST → Semantic IR 转换
- [ ] 能解析 `examples/*.ak`

### 5.2 CLI + Inspect（Week 7-8）

```bash
# 从 .ak 解析
arke parse kernel.ak -o kernel.json

# 查看 IR（人类可读）
arke inspect kernel.json

# LLM 优化（启动 agent session）
arke optimize kernel.json --target ampere --budget 50

# 代码生成
arke codegen kernel.json --target triton -o kernel_triton.py

# 验证
arke verify kernel.json --ref numpy
```

**交付物：**
- [ ] `arkec/main.py` — 完善 CLI
- [ ] `arke/inspect/pretty_print.py` — IR 可视化

---

## 六、Stream 4：评估框架（贯穿，Week 4-8）

### 6.1 评估任务定义（Week 4）

| Task ID | 算子 | Shape | 硬件 | 基线 |
|---------|------|-------|------|------|
| T1 | matmul | [1024,512]@[512,2048], f16 | Ampere | cuBLAS |
| T2 | matmul | [4096,4096]@[4096,4096], f16 | Ampere | cuBLAS |
| T3 | softmax | [1024,2048], f16 | Ampere | PyTorch |
| T4 | fused_matmul_relu | [1024,512]@[512,2048], f16 | Ampere | Triton hand-tuned |
| T5 | attention (SDPA) | [1,32,1024,64], f16 | Ampere | FlashAttention |

### 6.2 对比实验设计（Week 6-8）

三组对比：

```
Group A: LLM + Arke（本项目）
  LLM 通过 Arke tool-use 协议优化 kernel
  输入：Semantic IR + HW Profile
  输出：优化后的 Triton kernel

Group B: LLM Direct Triton（基线 1）
  LLM 直接编写 Triton kernel（无 Arke）
  输入：自然语言算子描述 + HW 参数
  输出：Triton kernel

Group C: LLM Direct CUDA（基线 2）
  LLM 直接编写 CUDA kernel
  输入：同 Group B
  输出：CUDA kernel
```

**评估维度：**

| 维度 | 说明 | 权重 |
|------|------|:---:|
| 正确率 | 生成的 kernel 通过数值验证的比例 | 40% |
| 性能 | 相对于 vendor library 的比例 (0-100%) | 30% |
| 一致性 | 多次运行结果的方差 | 15% |
| Token 效率 | 完成任务消耗的总 token 数 | 15% |

**交付物：**
- [ ] `benchmarks/tasks.py` — 评估任务定义
- [ ] `benchmarks/runner.py` — 自动化评估运行器
- [ ] `benchmarks/baselines/` — 基线实现
- [ ] `benchmarks/report.py` — 结果汇总和报告生成

---

## 七、硬件环境准备（Week 1, Day 1）

### 当前环境

- GPU: NVIDIA RTX 3060 Laptop (6GB, SM 8.6, Ampere)
- CUDA: 13.1
- Python: 3.10.12
- OS: WSL2 (Linux 6.6.x)

### 需要安装

```bash
# 1. 创建虚拟环境
python3 -m venv ~/.venvs/arke
source ~/.venvs/arke/bin/activate

# 2. 安装 PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 Triton
pip install triton

# 4. 安装 Arke 开发依赖
cd ~/workspace/repos/arke
pip install -e ".[dev]"

# 5. 验证
python -c "import torch; print(torch.cuda.is_available())"
python -c "import triton; print(triton.__version__)"
```

### 验证脚本

```python
# verify_env.py — 验证 GPU 环境完整可用
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

n = 1024
x = torch.randn(n, device='cuda', dtype=torch.float16)
y = torch.randn(n, device='cuda', dtype=torch.float16)
out = torch.empty_like(x)
add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
assert torch.allclose(out, x + y, atol=1e-2)
print("✅ GPU + Triton 环境验证通过")
```

---

## 八、周度时间表

### Week 1：基础建立

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | 环境 | 安装 PyTorch + Triton + 验证 GPU | verify_env.py 通过 |
| D1 | S2 | Semantic IR JSON Schema | semantic.schema.json |
| D2 | S1 | Tool-use Schema 定义 | tools_schema.py |
| D2 | S2 | 算子目录 P0（10 个算子） | ops/catalog.py |
| D3 | S1 | Session 生命周期 + system prompt | session.py, prompts.py |
| D3 | S2 | Strategy IR 完善 + HW Profile | strategy.py, nvidia_ampere.json |
| D4 | S2 | SemanticIR 构建工具（从 Python 函数到 IR） | ir/builder.py |
| D5 | S1+S2 | 集成：手动构造 matmul IR → 用 tool schema 验证 | 集成测试 |

### Week 2：验证系统 + ArkeEnv 骨架

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S2 | V0 静态验证器（shape + 约束） | validator.py |
| D2 | S2 | V1 数值验证（NumPy 参考生成 + 对比） | numerical_check.py |
| D3 | S1 | ArkeEnv 核心框架 | engine/env.py |
| D4 | S1 | 合法动作枚举引擎 | engine/legal_actions.py |
| D5 | S1 | ArkeEnv observe/apply/rollback 完整实现 | 单元测试通过 |

### Week 3：Codegen + 端到端雏形

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S2 | Triton matmul 模板 codegen（路径 A） | triton_templates/matmul.py.j2 |
| D2 | S2 | 模板引擎 + 编译器调用 | triton_template_engine.py, compiler.py |
| D3 | S2 | 端到端：手动 IR → 手动 strategy → Triton → GPU 执行 | 集成测试 |
| D4 | S2 | V2 性能验证（profiling + vs cuBLAS） | profiler.py |
| D5 | S1+S2 | ArkeEnv 接入 codegen + verify | compile_and_profile() 可用 |

### Week 4：LLM 联调

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S1 | LLM agent runner（支持 Claude API） | agent/runner.py |
| D2 | S1 | matmul agent demo：LLM tool-use 完整循环 | examples/agent_matmul.py |
| D3 | S1 | 错误恢复：LLM 决策失败时的自动引导 | agent/recovery.py |
| D4 | S4 | 评估任务定义 + 基线实现 | benchmarks/tasks.py |
| D5 | S1 | softmax codegen 模板 + agent demo | examples/agent_softmax.py |

### Week 5：LLM Lowering 实验 + Parser

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S2 | 路径 B：LLM Triton codegen（实验） | triton_llm_gen.py |
| D2 | S2 | 路径 A vs B 对比（正确率 + 性能） | 实验报告 |
| D3 | S3 | .ak EBNF 语法 + Lark parser | arke.lark, parser.py |
| D4 | S3 | AST → Semantic IR 转换 | ast_to_ir.py |
| D5 | S3 | 解析 examples/*.ak → IR → codegen → 执行 | 集成测试 |

### Week 6：端到端完善 + 评估

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S2 | fused_matmul_relu codegen（融合模板） | 模板 + 测试 |
| D2 | ALL | 端到端 pipeline 串联 | pipeline.py |
| D3 | S4 | 基线 B 实现：LLM 直写 Triton | baselines/direct_triton.py |
| D4 | S4 | 基线 C 实现：LLM 直写 CUDA | baselines/direct_cuda.py |
| D5 | S4 | 运行 T1-T3 对比实验 | 初步数据 |

### Week 7：优化 + 评估

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S1 | 轨迹记录系统 | learn/trajectory.py |
| D2 | S1 | 多 LLM 适配（Qwen/GPT） | agent/runner.py 扩展 |
| D3 | S4 | 运行 T4 (fused) + 多 LLM 对比 | 实验数据 |
| D4 | S3 | CLI 完善（parse/inspect/optimize/codegen） | arkec/main.py |
| D5 | ALL | 整体集成测试 + bug 修复 | CI 通过 |

### Week 8：收尾 + 报告

| 天 | Stream | 任务 | 交付物 |
|:--:|:------:|------|--------|
| D1 | S4 | 完整评估报告 | benchmarks/report.md |
| D2 | ALL | 文档完善 | docs/ 更新 |
| D3 | ALL | README 更新 + 使用指南 | README.md |
| D4 | ALL | 代码清理 + 测试覆盖 | 测试通过 |
| D5 | ALL | **MVP v0.1.0 Tag** | 🎉 |

---

## 九、假设验证里程碑（Gate）

> 详见 [`design-review.md`](design-review.md)

每个 Gate 不只是"功能交付"，更是"假设验证"。如果假设不成立，必须诚实面对并决定 pivot/kill。

| Gate | 时间 | 验证什么假设 | 通过标准 | 失败意味着什么 |
|:----:|:----:|-------------|---------|---------------|
| **G0** | W1 末 | 环境可行性 | Triton matmul 在 RTX 3060 跑通 | 换硬件 |
| **G1** | W2 末 | IR 表达力 | 已知好的 strategy 能在 Arke 中表达 | IR 设计有根本问题 |
| **G2** | W3 末 | 端到端通路 | 手动 strategy → codegen → ≥ 70% cuBLAS | **Arke 的下限不行，LLM 部分全是空中楼阁** |
| **G3** | W4 末 | LLM 可行性 | LLM tool-use 50 步 → matmul ≥ 50% cuBLAS + softmax 正确 | LLM 不具备 GPU 优化推理能力 → pivot |
| **G4** | W6 末 | 对比优势 | Arke 正确率和性能 ≥ 直写 Triton | **不成立 → kill 或 pivot 为验证框架** |
| **G5** | W8 末 | 整模型端到端收益 | GPT-2 Small 推理性能 Arke ≥ torch.compile | **单算子优势无法转化为整模型收益 → 分析瓶颈** |
| G6 | Phase 2 | 多硬件可迁移 | 同一 LLM session 优化 Ascend + 抽象层不大改 | 跨硬件假设不成立 |

### Gate 4 的决策矩阵

| 实验结果 | 结论 | 下一步 |
|----------|------|--------|
| Arke 正确率高 + 性能好 | ✅ 继续 | Phase 2 |
| Arke 正确率高 + 性能差 | ⚠️ Arke 是验证框架不是优化工具 | Pivot 定位 |
| Arke ≈ 直写 Triton | ⚠️ 无优势 | 审视增量价值 |
| 两者都差 | ❌ | Kill 或根本 pivot |

---

## 十、MVP v0.1.0 成功标准

| # | 标准 | 验收方式 |
|---|------|----------|
| 1 | **LLM 端到端可用** | LLM 通过 tool-use 优化 matmul → GPU 执行 → 正确结果 |
| 2 | **验证系统可用** | 静态 + 数值 + 性能 三层验证正常工作 |
| 3 | **性能达标** | LLM 优化后 matmul ≥ 70% cuBLAS |
| 4 | **有对比数据** | Arke vs 直写 Triton vs 暴力搜索 的定量对比 |
| 5 | **多算子验证** | matmul + softmax + fused_matmul_relu 至少三个可跑 |
| 6 | **Parser 可用** | .ak → IR → codegen → 执行 |
| 7 | **轨迹可导出** | 优化过程的完整 (state, action, reward) 轨迹 |
| 8 | **Fallback 可用** | LLM 搜索不如 fallback 时自动降级 |
| 9 | **整模型端到端** | GPT-2 Small 推理中替换 Arke kernel，端到端性能 ≥ torch.compile |

---

## 十、风险与应对

| 风险 | 严重度 | 应对 |
|------|:---:|------|
| LLM tool-use 循环不收敛 | 🔴 | 限制搜索预算 + 内置 fallback 策略（预定义的 good-enough schedule） |
| Triton codegen 表达力不足 | 🟡 | 路径 B（LLM 生成）作为灵活补充；MVP 范围限制为标准模式 |
| LLM 生成的 Triton 代码不正确 | 🟡 | V1 验证系统是安全网；不正确就 rollback + retry |
| 评估显示 Arke 不比直写 Triton 好 | 🔴 | 这是关键信号——如果证实，需要重新审视项目方向或调整切入点 |
| 6GB 显存不够跑大 shape | 🟡 | MVP 用小 shape（≤2048）；大 shape 需要云 GPU |
| RTX 3060 Laptop 的 Tensor Core 行为与 A100 不同 | 🟢 | 都是 Ampere 架构，SM 8.6 vs 8.0 差异可控 |

---

## 十一、项目结构（更新）

```
arke/
├── README.md
├── LICENSE
├── pyproject.toml
├── docs/
│   ├── design/
│   │   ├── overview.md
│   │   ├── e2e-design-v1.md
│   │   ├── e2e-design-v2.md
│   │   └── plan-v2.1.md          # ← 本文档
│   ├── agent-protocol/            # ← 新增
│   │   ├── 01-tools.md
│   │   └── 02-session.md
│   ├── ir-spec/
│   └── language-ref/
├── arke/
│   ├── __init__.py
│   ├── lang/                      # 语言层（Stream 3）
│   │   ├── arke.lark
│   │   ├── parser.py
│   │   ├── ast.py
│   │   ├── ast_to_ir.py
│   │   ├── types.py
│   │   └── grammar.py
│   ├── ir/                        # IR 层（Stream 2）
│   │   ├── semantic.py
│   │   ├── strategy.py            # 原 schedule.py
│   │   ├── builder.py             # IR 构建工具
│   │   ├── schemas/
│   │   │   ├── semantic.schema.json
│   │   │   └── strategy.schema.json
│   │   ├── ops/
│   │   │   ├── catalog.py
│   │   │   └── definitions.py
│   │   └── targets/
│   │       ├── nvidia_ampere.json
│   │       └── ascend_a3.json
│   ├── engine/                    # 核心引擎（Stream 1 + 2）
│   │   ├── env.py                 # ArkeEnv
│   │   ├── legal_actions.py
│   │   ├── state.py
│   │   ├── validator.py           # V0 静态验证
│   │   ├── numerical_check.py     # V1 数值验证
│   │   └── profiler.py            # V2 性能验证
│   ├── agent/                     # LLM Agent（Stream 1）
│   │   ├── tools_schema.py
│   │   ├── tools_schema.json
│   │   ├── session.py
│   │   ├── prompts.py
│   │   ├── runner.py
│   │   └── recovery.py
│   ├── backend/                   # Codegen（Stream 2）
│   │   ├── triton_templates/
│   │   │   └── matmul.py.j2
│   │   ├── triton_template_engine.py
│   │   ├── triton_llm_gen.py
│   │   └── compiler.py
│   ├── learn/                     # 学习系统
│   │   └── trajectory.py
│   ├── pipeline.py                # 端到端 pipeline
│   └── frontend/                  # 外部导入（Phase 2+）
│       └── __init__.py
├── arkec/
│   └── main.py
├── benchmarks/                    # 评估框架（Stream 4）
│   ├── tasks.py
│   ├── runner.py
│   ├── baselines/
│   │   ├── direct_triton.py
│   │   └── direct_cuda.py
│   └── report.py
├── tests/
│   ├── test_types.py
│   ├── test_ir.py
│   ├── test_validator.py
│   ├── test_codegen.py
│   ├── test_env.py
│   ├── test_e2e_matmul.py
│   └── test_e2e_softmax.py
└── examples/
    ├── 01_matmul.ak
    ├── 02_matmul_relu_fused.ak
    ├── 03_softmax.ak
    ├── 04_attention.ak
    ├── agent_matmul.py            # LLM agent demo
    └── agent_softmax.py
```

---

## 十二、多硬件后端策略

> **v2.1.2 修正**：详见 [`patch-v2.1.2.md`](patch-v2.1.2.md) 和 [`multi-backend-design.md`](multi-backend-design.md)。

核心策略变更：
- **Ascend 后端通过 triton-ascend 对接**（Triton 代码 → triton-ascend → NPU），不再走 AscendC
- **Phase 1 纯 NVIDIA**——所有 Ascend 开发延后到 Phase 2
- **Triton codegen 是双硬件通用的**——同一份 Triton 代码可在 GPU 和 NPU 上运行
- 更深度的 Ascend 优化（Phase 3）通过 AscendNPU IR (MLIR) 实现

三层 Ascend 路径（渐进式）：
```
Phase 2: Triton → triton-ascend → NPU        （复用 Triton codegen，零额外开发）
Phase 3: Arke → AscendNPU IR HFusion → NPU   （利用 MLIR 自动调度）
Phase 4: Arke → AscendNPU IR HIVM → NPU      （极致优化，精确控制 NPU 指令）
```

### LLM API 灵活配置

详见 [`patch-v2.1.2.md`](patch-v2.1.2.md)。

- LLM Provider 抽象层：OpenAI-compatible（覆盖 GPT/Qwen/DeepSeek/本地）+ Anthropic
- YAML 配置文件：`arke.config.yaml`
- Fallback 链 + Token 追踪 + 重试机制
- CLI 支持 `--llm <provider>` 切换

### 借鉴 Claude Code 的工程模式

详见 [`cc-inspired-update.md`](cc-inspired-update.md)。

基于 Claude Code 51万行源码分析，迁移 7 个经过大规模验证的工程模式：
- **AsyncGenerator 优化循环** — 统一 CLI/API/Jupyter 消费接口
- **声明式 Tool 接口** — 工具自描述并发/安全/成本属性
- **工具并发分区** — analyze + get_hw_profile 并发；apply_decision 串行
- **分段 Prompt Cache** — 4 段独立缓存，50步优化节省 ~80% token
- **Context Compact** — 预测式 + 反应式双保险，跨 compact 保持 ground truth
- **大结果 delta 压缩** — legal_actions 只给 top 10 + 总数
- **三层容错** — 工具级 rollback + API 级 fallback + 循环级 fallback strategy

---

*计划版本：v2.1.1 | 创建日期：2026-03-31*
*核心修正：LLM 协议优先，验证前置，语法后置，必须有评估*
*v2.1.1 补充：多硬件后端抽象，优先 NVIDIA + Ascend A3*
*v2.1.3 补充：借鉴 Claude Code 的工程设计模式*