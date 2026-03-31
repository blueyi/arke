# Arke — AI-First 算子描述语言与编译工具链设计方案

> Arke：希腊神话中连接奥林匹斯与凡间的信使女神。
> 一门面向 AI Agent 优先设计的算子描述与优化语言，连接上层 AI 智能与底层硬件算力。

---

## 一、设计背景与动机

### 1.1 核心问题

当前 AI 编译器栈（TVM、MLIR、Triton 等）的设计假设是"人写代码，编译器翻译"。但随着大模型 Agent 在代码生成领域的能力飞速提升，未来 80%+ 的算子优化代码将由 AI 生成。现有工具链对 AI Agent 并不友好：

- **MLIR**：硬件表达能力强，但对 AI 和人类都不够直观
- **Triton**：Python 亲和好，但抽象层级偏低
- **TVM TensorIR**：平衡尚可，但未将 AI Agent 作为一等用户考虑

### 1.2 设计哲学：AI First

**核心主张：** 将 AI Agent 作为这层抽象的第一用户，人类可读性和 Python 兼容作为派生能力。

```
AI 原生表达（核心层）
    ├── → 人类可读视图（pretty print / 可视化工具）
    ├── → Python DSL 导出（兼容层）
    └── ← Python DSL 导入（专家经验吸收入口）
```

### 1.3 定位：语言 + IR

Arke 不是单纯的 IR，也不是通用编程语言，而是 **一门领域专用语言（DSL）+ 配套的多层 IR 体系**：

```
人 / AI Agent
    │ 写 & 读
    ▼
  Arke Language（语法、语义、类型系统）
    │ parse
    ▼
  Arke IR — High Level（语义图，算子级）
    │ optimize
    ▼
  Arke IR — Low Level（调度树，指令级）
    │ codegen
    ▼
  硬件代码（CUDA / ROCm / Metal）
```

### 1.4 三角约束模型

```
        AI 可理解 ⭐（优先）
           △
          / \
         /   \
  人类可读 ←→ Python 可转换
 （审查需求）  （兼容需求）
```

---

## 二、设计原则

### 2.1 AI First 四原则

| 原则 | 说明 |
|------|------|
| **语义显式化** | 数据依赖、并行性、内存层级约束等关键信息作为一等公民显式存在，AI 无需通过分析推导 |
| **结构化表达** | 规则明确、结构一致的 AST/JSON-like 表达，优于自由格式 DSL |
| **搜索空间可枚举** | 合法变换（tiling, fusion, reorder 等）可被系统性枚举，AI 做搜索而非创造 |
| **反馈即时化** | 支持低成本评估（性能预测、合法性检查），无需每次编译到硬件 |

### 2.2 向上向下的双向可达性

- **向上（面向 AI/Agent）**：语义清晰，意图可理解。AI 能理解"矩阵乘法+激活融合"而不纠结于内存排布细节
- **向下（面向硬件）**：保留数据布局、并行维度、内存层级等硬件映射信息，工具链能生成高效代码

### 2.3 人类参与模式

人类专家不直接编辑 AI 原生表达，而是：
1. 通过可视化工具 **审查** AI 的决策
2. 用 Python DSL **表达** 优化意图
3. 系统自动转换成 Arke 原生格式，供模型学习

---

## 三、Arke IR 设计

### 3.1 表达结构：分层图 + 约束描述

Arke IR 采用 **分层计算图 + Schedule 约束树** 的双层结构：

```
┌─────────────────────────────────┐
│  Layer 1: Semantic Graph        │  ← AI 主要操作层
│  （语义计算图，算子级）            │
├─────────────────────────────────┤
│  Layer 2: Schedule Tree         │  ← AI 优化决策层
│  （调度约束树，变换级）            │
├─────────────────────────────────┤
│  Layer 3: Hardware Mapping      │  ← 工具链自动处理层
│  （硬件映射，指令级）             │
└─────────────────────────────────┘
```

### 3.2 Layer 1：语义计算图（Semantic Graph）

描述"计算什么"，不描述"怎么计算"。

```json
{
  "graph": "fused_matmul_relu",
  "nodes": [
    {
      "id": "matmul_0",
      "op": "matmul",
      "inputs": {
        "A": {"shape": [1024, 512], "dtype": "float16", "layout": "row_major"},
        "B": {"shape": [512, 2048], "dtype": "float16", "layout": "col_major"}
      },
      "output": {"shape": [1024, 2048], "dtype": "float16"},
      "semantics": {
        "computation": "C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
        "properties": ["associative", "distributive"]
      }
    },
    {
      "id": "relu_0",
      "op": "relu",
      "inputs": {"X": "@matmul_0.output"},
      "output": {"shape": [1024, 2048], "dtype": "float16"},
      "semantics": {
        "computation": "Y[i,j] = max(X[i,j], 0)",
        "properties": ["elementwise", "monotonic"]
      }
    }
  ],
  "fusion_groups": [
    {"id": "fg_0", "nodes": ["matmul_0", "relu_0"], "type": "epilogue_fusion"}
  ],
  "data_flow": [
    {"from": "matmul_0", "to": "relu_0", "tensor": "intermediate", "lifetime": "local"}
  ]
}
```

**关键设计点：**
- 每个节点携带显式的 `semantics` 字段，AI 可直接理解计算含义
- `properties` 标注代数性质，AI 可据此判断哪些变换合法
- `fusion_groups` 显式标注融合关系，无需 AI 推导
- `data_flow` 显式标注数据生命周期

### 3.3 Layer 2：调度约束树（Schedule Tree）

描述"怎么优化"，将优化决策与计算语义解耦。

```json
{
  "schedule": "fused_matmul_relu",
  "target": "fg_0",
  "decisions": [
    {
      "type": "tiling",
      "loop": "i",
      "factors": [64, 16],
      "rationale": "L2 cache line = 64, warp size = 16"
    },
    {
      "type": "tiling",
      "loop": "j",
      "factors": [128, 8],
      "rationale": "maximize memory coalescing"
    },
    {
      "type": "tiling",
      "loop": "k",
      "factors": [32],
      "rationale": "shared memory capacity = 48KB"
    },
    {
      "type": "reorder",
      "order": ["i_outer", "j_outer", "k_outer", "i_inner", "j_inner", "k_inner"],
      "rationale": "outer loops parallel, inner loops for data reuse"
    },
    {
      "type": "parallel",
      "loops": ["i_outer", "j_outer"],
      "mapping": {"i_outer": "blockIdx.y", "j_outer": "blockIdx.x"}
    },
    {
      "type": "memory_placement",
      "tensor": "A_tile",
      "level": "shared_memory",
      "access_pattern": "broadcast_along_j"
    }
  ],
  "constraints": {
    "shared_memory_limit": 49152,
    "register_limit": 65536,
    "max_threads_per_block": 1024
  },
  "search_space": {
    "tiling_i": {"type": "power_of_2", "range": [16, 256]},
    "tiling_j": {"type": "power_of_2", "range": [16, 256]},
    "tiling_k": {"type": "power_of_2", "range": [8, 64]}
  }
}
```

**关键设计点：**
- 每个决策携带 `rationale`（决策理由），AI 可学习专家的优化思路
- `constraints` 显式列出硬件约束，AI 不需要记忆硬件参数
- `search_space` 定义合法的搜索范围，AI 做有界搜索而非自由发挥

### 3.4 Layer 3：硬件映射（Hardware Mapping）

由工具链根据 Layer 1 + Layer 2 自动生成，AI 一般不直接操作。

```json
{
  "target": {
    "arch": "nvidia_ampere",
    "compute_capability": "8.0",
    "tensor_core": true
  },
  "kernel_config": {
    "grid": [16, 16, 1],
    "block": [128, 1, 1],
    "shared_memory": 32768,
    "registers_per_thread": 64
  },
  "instruction_hints": [
    {"op": "matmul_0", "prefer": "mma.sync.aligned.m16n8k16.f16"}
  ]
}
```

---

## 四、工具链设计

### 4.1 整体架构

```
┌────────────────────────────────────────────────────┐
│                  Frontend（前端）                    │
│                                                    │
│  Python DSL ──→ ┐                                  │
│  PyTorch FX ──→ ├─→ Arke IR（语义图）               │
│  ONNX ────────→ ┘                                  │
└──────────────────────┬─────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────┐
│              Arke Engine（核心引擎）                  │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ Validator │  │ Analyzer │  │ Cost Model   │     │
│  │ 合法性检查 │  │ 依赖分析  │  │ 性能预估模型  │     │
│  └──────────┘  └──────────┘  └──────────────┘     │
│  ┌──────────────────────────────────────────┐     │
│  │         Transform Library                 │     │
│  │   tiling | fusion | reorder | vectorize   │     │
│  │   每个变换可枚举，带前置/后置条件            │     │
│  └──────────────────────────────────────────┘     │
│  ┌──────────────────────────────────────────┐     │
│  │         AI Agent Interface                │     │
│  │   observe() → 当前IR状态                   │     │
│  │   actions() → 可用变换列表                  │     │
│  │   apply(action) → 执行变换                  │     │
│  │   evaluate() → 性能预估/实测                │     │
│  │   rollback() → 回滚上一步                   │     │
│  └──────────────────────────────────────────┘     │
└──────────────────────┬─────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────┐
│               Backend（后端）                        │
│                                                    │
│  Arke IR ─────→ ┐                                  │
│                 ├─→ CUDA / ROCm / Metal             │
│  Schedule ────→ ┤   LLVM IR                         │
│  HW Mapping ──→ ┘   Triton IR（复用 Triton 后端）    │
└────────────────────────────────────────────────────┘
```

### 4.2 核心组件详细设计

#### 4.2.1 arke-parse（前端解析器）

```
输入支持：
├── Python DSL（Arke 原生语法）
├── PyTorch FX Graph（torch.compile 导出）
├── ONNX 子图
├── Triton kernel（反向提取语义）
└── NumPy 表达式

输出：Arke IR (Layer 1)
```

**Python DSL 示例：**

```python
import arke

@arke.kernel
def fused_matmul_relu(A: arke.Tensor[1024, 512, arke.f16],
                      B: arke.Tensor[512, 2048, arke.f16]) -> arke.Tensor[1024, 2048, arke.f16]:
    C = arke.matmul(A, B)
    return arke.relu(C)
```

与原始 Python 的差异极小，但背后生成的是结构化的 Arke IR。

#### 4.2.2 arke-optimize（AI 优化引擎）

这是核心差异化组件，为 AI Agent 设计的 **强化学习友好接口**：

```python
# AI Agent 的交互接口
class ArkeEnv:
    def observe(self) -> IRState:
        """返回当前 IR 的结构化表示（JSON）"""

    def legal_actions(self) -> List[Transform]:
        """枚举当前状态下所有合法变换"""

    def apply(self, action: Transform) -> IRState:
        """应用一个变换，返回新状态"""

    def estimate_cost(self) -> CostReport:
        """快速性能预估（不编译）"""

    def benchmark(self) -> BenchmarkResult:
        """实际编译并测试（慢但精确）"""

    def rollback(self, steps: int = 1):
        """回滚变换"""

    def export_trajectory(self) -> Trajectory:
        """导出优化轨迹，用于训练"""
```

#### 4.2.3 arke-verify（验证器）

```
功能：
├── 合法性检查：变换前后语义等价性验证
├── 约束检查：硬件资源约束（shared memory、registers）
├── 性能检查：检测明显的反优化（cache thrashing 等）
└── 回归检查：数值正确性自动测试
```

#### 4.2.4 arke-codegen（代码生成）

```
Arke IR + Schedule
    ├── → CUDA kernel（直接生成）
    ├── → Triton IR（复用 Triton 后端）
    ├── → LLVM IR（通用路径）
    ├── → ROCm / HIP
    └── → Metal Shading Language
```

#### 4.2.5 arke-learn（经验学习系统）

```
专家经验吸收流程：

1. 专家用 Python DSL 写出优化后的 kernel
2. arke-parse 解析为 Arke IR
3. arke-diff 对比优化前后的 IR 差异
4. 自动提取 "优化模式"（pattern）
5. 存入模式库，供 AI Agent 学习

输出：
├── 优化轨迹数据集（用于 RL 训练）
├── 优化模式库（用于规则匹配）
└── 标注数据（用于监督学习）
```

### 4.3 工具链命令行设计

```bash
# 从 Python 生成 IR
arke parse kernel.py -o kernel.ak

# 查看 IR（人类可读视图）
arke inspect kernel.ak --visual

# AI 优化（调用 Agent）
arke optimize kernel.ak --target ampere --budget 100

# 手动应用变换
arke transform kernel.ak --tile i=64,j=128 --fuse matmul+relu

# 性能预估
arke estimate kernel.ak --target ampere

# 代码生成
arke codegen kernel.ak --target cuda -o kernel.cu

# 验证正确性
arke verify kernel.ak --reference kernel.py

# 导出优化轨迹
arke trajectory kernel.ak --format jsonl -o trace.jsonl

# 与 Triton 互转
arke convert kernel.py --from triton --to arke
arke convert kernel.ak --from arke --to triton
```

---

## 五、与现有生态的关系

```
                    ┌──────────────┐
                    │  PyTorch     │
                    │  torch.compile│
                    └──────┬───────┘
                           │ FX Graph
                    ┌──────▼───────┐
                    │    Arke      │ ← 本项目
                    │ Lang + IR    │
                    │  + Toolchain │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌──────▼──┐  ┌─────▼───┐
       │ Triton  │  │  LLVM   │  │  MLIR   │
       │ Backend │  │ Backend │  │ Backend │
       └─────────┘  └─────────┘  └─────────┘
```

- **不替代 MLIR/Triton**，而是在其上层提供 AI-native 的抽象
- **复用 Triton 后端** 作为主要的 codegen 路径
- **对接 PyTorch torch.compile** 作为主要的用户入口

---

## 六、项目命名

### 名称：Arke

**含义：**
- 希腊神话中的信使女神，拥有彩虹般的翅膀
- 连接奥林匹斯（众神/AI）与凡间（硬件/物理世界）的桥梁
- 传递信息、翻译意图

**为什么适合这个项目：**
- Arke 连接 AI 智能与硬件算力，正如信使连接天与地
- 4 个字母，2 音节 /ˈɑːrki/，简短好记
- GitHub 在编译器/语言领域完全无冲突
- 工具链命名自然：`arke parse`、`arke optimize`、`arke codegen`

**命名体系：**

```
Arke Language    → 语言规范
Arke IR          → 内部表示
arkec            → 编译器（arke compiler）
.ak              → 文件后缀
arke <command>   → CLI 命令
```

---

## 七、项目结构

```
arke/
├── README.md
├── LICENSE
├── docs/
│   ├── design.md          # 本文档
│   ├── ir-spec.md         # IR 形式化规范
│   ├── language-ref.md    # 语言参考手册
│   └── tutorials/
├── arke/                  # Python 包
│   ├── lang/              # 语言定义
│   │   ├── grammar.py     # 语法定义
│   │   ├── types.py       # 类型系统
│   │   └── ast.py         # 抽象语法树
│   ├── ir/                # IR 定义与操作
│   │   ├── semantic.py    # Layer 1: 语义图
│   │   ├── schedule.py    # Layer 2: 调度树
│   │   └── hardware.py    # Layer 3: 硬件映射
│   ├── frontend/          # 前端解析
│   │   ├── python_dsl.py
│   │   ├── pytorch_fx.py
│   │   ├── onnx_import.py
│   │   └── triton_import.py
│   ├── transforms/        # 变换库
│   │   ├── tiling.py
│   │   ├── fusion.py
│   │   ├── reorder.py
│   │   └── registry.py
│   ├── engine/            # 核心引擎
│   │   ├── validator.py
│   │   ├── cost_model.py
│   │   └── agent_env.py   # AI Agent RL 环境
│   ├── backend/           # 代码生成
│   │   ├── cuda.py
│   │   ├── triton.py
│   │   └── llvm.py
│   └── learn/             # 经验学习
│       ├── trajectory.py
│       ├── pattern.py
│       └── dataset.py
├── arkec/                 # 编译器 CLI
│   └── main.py
├── tests/
└── examples/
    ├── matmul_optimize.py
    ├── attention_fuse.py
    └── conv2d_tile.py
```

---

## 八、里程碑规划

### Phase 1：语言定义 + 基础工具链（MVP）
- [ ] Arke Language 语法规范
- [ ] Arke IR 形式化规范
- [ ] Python DSL 前端
- [ ] 基本变换库（tiling, fusion）
- [ ] CUDA codegen（通过 Triton）
- [ ] CLI 工具 `arke`

### Phase 2：AI Agent 集成
- [ ] ArkeEnv 强化学习环境
- [ ] Cost Model 性能预估
- [ ] 优化轨迹记录与导出
- [ ] 基础 AI Agent（基于大模型）

### Phase 3：生态打通
- [ ] PyTorch torch.compile 集成
- [ ] ONNX 导入
- [ ] 多后端支持（ROCm, Metal）
- [ ] 专家经验学习系统

### Phase 4：规模化验证
- [ ] 主流模型算子全覆盖
- [ ] 性能对标 cuDNN / Triton 手写 kernel
- [ ] 社区共建模式库

---

*文档版本：v0.2 | 创建日期：2026-03-30 | 项目名确认：Arke*
*基于 Leon 的设计讨论整理*
