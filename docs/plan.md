# Arke Project — 详细执行计划

> 三条主线并行推进：语言语法设计、IR 形式化规范、GitHub 仓库初始化
> 预计 MVP 周期：8-10 周

---

## 总览

```
Week 1-2    Week 3-4    Week 5-6    Week 7-8    Week 9-10
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────┐
│ Stream A: 语言语法设计                                │
│ [语法草案] → [类型系统] → [语义规则] → [规范文档]      │
├─────────────────────────────────────────────────────┤
│ Stream B: IR 形式化规范                               │
│ [数据模型] → [变换规则] → [约束系统] → [规范文档]      │
├─────────────────────────────────────────────────────┤
│ Stream C: 项目仓库 + 工具链骨架                        │
│ [仓库初始化] → [Parser] → [IR核心] → [Codegen] → MVP │
└─────────────────────────────────────────────────────┘
```

---

## Stream A：Arke Language 语法设计

### A1. 语法设计原则（Week 1）

**关键决策：**

| 决策项 | 选项 | 推荐 | 理由 |
|--------|------|------|------|
| 语法风格 | Python-like / C-like / 声明式 | Python-like | AI 训练语料中 Python 最多，LLM 最熟悉 |
| 缩进 vs 括号 | 缩进 / 大括号 | 大括号 | 结构化解析更可靠，AI 不易出错 |
| 类型标注 | 可选 / 必须 | 必须 | AI First：显式优于隐式 |
| 计算表达 | 命令式 / 声明式 / 混合 | 声明式为主 | 描述"是什么"而非"怎么做" |

**交付物：**
- [ ] `docs/language-ref/01-design-principles.md`

### A2. 核心语法定义（Week 1-2）

#### A2.1 数据类型系统

```
标量类型：
  f16, f32, f64, bf16           — 浮点
  i8, i16, i32, i64             — 整数
  u8, u16, u32, u64             — 无符号整数
  bool, index                   — 布尔 / 索引

张量类型：
  Tensor<shape, dtype, layout>
  例：Tensor<[1024, 512], f16, row_major>

布局类型：
  row_major | col_major | tiled<tile_shape> | custom<affine_map>

内存层级类型：
  global | shared | local | register
```

#### A2.2 核心语法示例

```arke
// Kernel 定义（声明式计算）
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16, row_major>,
    B: Tensor<[512, 2048], f16, col_major>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

// Schedule 描述（与计算分离）
schedule fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");

    tile(loop="j", factors=[128, 8])
        @rationale("maximize memory coalescing");

    reorder(["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"]);

    parallel(loops=["i_outer", "j_outer"],
             mapping={"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"});

    place(tensor="A_tile", memory=shared)
        @rationale("broadcast along j, reuse across j iterations");

    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

#### A2.3 标志性语法：`@rationale` 注解

每个优化决策都可以携带自然语言理由，这是 Arke 的核心差异点：
- AI 生成代码时同时生成决策理由
- 人类审查时一眼看懂优化意图
- 训练数据中作为 chain-of-thought 增强学习

**交付物：**
- [ ] `docs/language-ref/02-types.md`
- [ ] `docs/language-ref/03-syntax.md`
- [ ] `docs/language-ref/04-schedule.md`
- [ ] `examples/` 目录下 5+ 个语法示例

### A3. 语义规则（Week 3-4）

- [ ] 计算语义：每个内建算子的数学定义
- [ ] 变换语义：每个 schedule 原语的前置条件、后置条件、不变量
- [ ] 等价性规则：两段 Arke 代码语义等价的条件
- [ ] 错误模型：类型错误、约束违反、非法变换的定义

**交付物：**
- [ ] `docs/language-ref/05-semantics.md`
- [ ] `docs/language-ref/06-transforms.md`

### A4. Python 互转换规范（Week 4）

```
Python DSL → Arke：
  @arke.kernel decorator → kernel 定义
  arke.matmul() → matmul 算子
  arke.schedule() context manager → schedule 块

Arke → Python DSL：
  kernel 定义 → @arke.kernel 函数
  schedule 块 → arke.schedule() 调用
  @rationale → Python 注释
```

**交付物：**
- [ ] `docs/language-ref/07-python-interop.md`

---

## Stream B：Arke IR 形式化规范

### B1. IR 数据模型（Week 1-2）

用 JSON Schema 精确定义 IR 数据结构：

```
arke/ir/schemas/
├── semantic-graph.schema.json    # Layer 1
├── schedule-tree.schema.json     # Layer 2
├── hardware-mapping.schema.json  # Layer 3
└── common-types.schema.json      # 共享类型
```

**交付物：**
- [ ] `arke/ir/schemas/` — 完整 JSON Schema
- [ ] `docs/ir-spec/01-data-model.md`

### B2. 内建算子库定义（Week 2-3）

每个算子包含：签名、语义、代数性质、融合规则、硬件提示。

**MVP 算子列表（P0）：**

| 类别 | 算子 |
|------|------|
| 线性代数 | matmul, batch_matmul |
| 逐元素 | relu, gelu, silu, tanh, sigmoid, add, mul, sub, div |
| 归约 | softmax, layer_norm, rms_norm |
| 数据移动 | transpose, reshape, cast |

**P1（Phase 2）：**
| 类别 | 算子 |
|------|------|
| 注意力 | scaled_dot_product_attention |
| 卷积 | conv2d, depthwise_conv2d |
| 池化 | max_pool, avg_pool |

**交付物：**
- [ ] `arke/ir/ops/` — 算子定义文件
- [ ] `docs/ir-spec/02-op-catalog.md`

### B3. 变换规则形式化（Week 3-5）

每种变换定义：前置条件、参数、后置条件、逆变换、搜索空间。

**MVP 变换列表（P0）：**

| 变换 | 描述 |
|------|------|
| tile | 循环切分 |
| reorder | 循环重排 |
| fuse_ops | 算子融合 |
| parallel | 并行映射 |
| place | 内存层级放置 |

**P1：** vectorize, unroll, split_k
**P2：** pipeline, prefetch

**交付物：**
- [ ] `arke/ir/transforms/` — 变换规则定义
- [ ] `docs/ir-spec/03-transforms.md`

### B4. 约束系统（Week 4-5）

定义硬件约束模型（shared memory、registers、threads 等）和合法性检查规则。

**交付物：**
- [ ] `arke/ir/targets/` — 硬件目标定义（首先支持 NVIDIA Ampere）
- [ ] `docs/ir-spec/04-constraints.md`

### B5. Cost Model 规范（Week 5-6）

性能预估模型接口：latency、bandwidth utilization、compute utilization、bottleneck 分析。

**交付物：**
- [ ] `docs/ir-spec/05-cost-model.md`

---

## Stream C：项目仓库 + 工具链骨架

### C1. GitHub 仓库初始化（Week 1, Day 1-2）

- [ ] 创建仓库
- [ ] 项目结构搭建
- [ ] README.md
- [ ] LICENSE (Apache 2.0)
- [ ] pyproject.toml
- [ ] CI (GitHub Actions)

### C2. Lexer + Parser（Week 2-4）

技术选型：Lark (EBNF) + dataclass AST

- [ ] `arke/lang/arke.lark` — EBNF 语法
- [ ] `arke/lang/parser.py` — Parser
- [ ] `arke/lang/ast.py` — AST 定义
- [ ] 能解析所有 `examples/*.ak`

### C3. IR 核心实现（Week 3-5）

- [ ] `arke/ir/semantic.py` — Layer 1
- [ ] `arke/ir/schedule.py` — Layer 2
- [ ] `arke/ir/transforms/` — 变换实现
- [ ] `arke/engine/validator.py` — 验证器

### C4. Python DSL 前端（Week 4-5）

```python
import arke

@arke.kernel
def matmul_relu(A: arke.Tensor[1024, 512, arke.f16],
                B: arke.Tensor[512, 2048, arke.f16]):
    C = arke.matmul(A, B)
    return arke.relu(C)

ir = matmul_relu.to_ir()
```

- [ ] `arke/frontend/python_dsl.py`

### C5. Triton Codegen（Week 6-7）

Arke IR → Triton Python 代码 → CUDA binary

- [ ] `arke/backend/triton_backend.py`
- [ ] 端到端：matmul 可跑通

### C6. CLI 工具（Week 7-8）

```bash
arke parse kernel.ak -o kernel.json
arke inspect kernel.json
arke codegen kernel.json --target triton
arke verify kernel.json --ref kernel.py
```

- [ ] `arkec/main.py` (基于 typer)

### C7. AI Agent 环境（Week 8-10）

```python
env = ArkeEnv.from_file("kernel.ak", target="nvidia_ampere")
state = env.observe()
actions = env.legal_actions()
new_state = env.apply(actions[0])
cost = env.estimate_cost()
```

- [ ] `arke/engine/agent_env.py`
- [ ] `arke/engine/cost_model.py`
- [ ] `arke/learn/trajectory.py`

---

## 周度时间表

### Week 1-2：基础建立
| Week | Stream | 任务 |
|------|--------|------|
| W1 | C | 创建 GitHub 仓库，初始化项目结构 |
| W1 | A | 语法设计原则 + 类型系统 |
| W1 | B | IR Layer 1/2 JSON Schema |
| W2 | A | 核心语法定义（kernel + schedule） |
| W2 | C | Lexer 初步实现 |

### Week 3-4：核心实现
| Week | Stream | 任务 |
|------|--------|------|
| W3 | B | 内建算子库定义（P0） |
| W3 | C | Parser 完整实现 |
| W3 | A | 语义规则草案 |
| W4 | B | 变换规则形式化 |
| W4 | C | IR 核心数据结构实现 |
| W4 | A | Python 互转换规范 |

### Week 5-6：集成
| Week | Stream | 任务 |
|------|--------|------|
| W5 | C | 变换引擎实现 |
| W5 | B | Cost Model 规范 |
| W5 | C | Python DSL 前端 |
| W6 | C | 验证器 + Triton codegen 初步 |

### Week 7-8：端到端
| Week | Stream | 任务 |
|------|--------|------|
| W7 | C | Triton codegen 完善（matmul 可跑通） |
| W7 | C | CLI 工具 |
| W8 | C | AI Agent 环境 |
| W8 | ALL | 端到端测试 |

### Week 9-10：完善 + 发布
| Week | Stream | 任务 |
|------|--------|------|
| W9 | ALL | 文档完善 |
| W9 | C | Cost Model 基础实现 |
| W10 | C | 轨迹记录系统 |
| W10 | ALL | **MVP v0.1.0 发布** |

---

## MVP v0.1.0 成功标准

1. ✅ 语言规范完整：能描述 matmul、softmax、attention
2. ✅ IR 规范完整：三层 IR 有 JSON Schema
3. ✅ Parser 可用：解析 `.ak` 文件为 Arke IR
4. ✅ Python DSL 可用：`@arke.kernel` 装饰器能工作
5. ✅ 基础变换可用：tile + reorder + fuse
6. ✅ Codegen 可用：Arke IR → Triton → 可执行
7. ✅ AI Agent 环境可用：observe / apply / evaluate 循环
8. ✅ 端到端 Demo：matmul 从 Arke 到 GPU 执行

---

## 需要 Leon 确认的决策点

| # | 决策 | 选项 | 截止时间 |
|---|------|------|----------|
| D1 | 仓库公开还是私有 | ✅ **私有** | 已确认 |
| D2 | 语法风格最终确认 | ✅ **方案 A+D**（Python-like计算 + 声明式Schedule + 大括号） | 已确认 |
| D3 | 首要目标硬件 | ✅ **NVIDIA Ampere + Huawei Ascend A3** | 已确认 |
| D4 | 许可证 | ✅ **Apache 2.0** | 已确认 |
| D5 | 文档语言 | ✅ **中英文支持，默认英文** | 已确认 |

---

*计划版本：v0.1 | 创建日期：2026-03-31*
