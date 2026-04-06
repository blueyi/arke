# Arke — 命名体系

> 项目全局命名规范与术语表
> Date: 2026-03-31

---

## 一、顶层命名

| 名称 | 含义 | 说明 |
|:-----|:-----|:-----|
| **Arke** | 项目名 / 品牌名 | 希腊神话彩虹信使女神 Ἄρκη，Iris 的孪生姐妹。发音 /ˈɑːrki/ |
| **Arke Language** | 语言规范 | 描述算子计算和优化策略的领域特定语言 |
| **Arke IR** | 中间表示（统称） | 包括 Semantic IR 和 Strategy IR 两层 |
| **arkec** | 编译器 CLI | **arke** + **c**ompiler，类比 `rustc`、`clang`、`gcc` |
| **.ak** | 源文件后缀 | **a**r**k**e 首尾字母，类比 `.rs`、`.py`、`.ts` |

### 命名关系图

```
Arke (项目)
├── Arke Language        语言规范（.ak 语法、类型系统、语义规则）
├── Arke IR              中间表示
│   ├── Semantic IR      "算什么"
│   └── Strategy IR      "怎么优化"
├── arkec                编译器工具链 CLI
│   ├── arke parse       .ak → Arke IR
│   ├── arke inspect     IR 可视化
│   ├── arke optimize    AI 辅助优化
│   ├── arke codegen     IR → 目标代码
│   └── arke verify      正确性验证
└── arke (Python)        Python 包（import arke）
```

---

## 二、IR 双层命名

Arke 的核心抽象是将算子描述拆成两层，各自独立、各自演化：

| 层 | 全称 | 简称 | 代码名 | 问题域 |
|:---|:-----|:-----|:-------|:-------|
| 上层 | **Semantic IR** | SIR | `SemanticGraph` | What — 计算语义（纯数学，不含任何优化） |
| 下层 | **Strategy IR** | StrIR | `StrategyIR` | How — 优化策略（tile、fuse、place 等决策序列）|

**为什么用 Strategy 而非 Schedule？**

| 候选 | 来源 | 否决理由 |
|:-----|:-----|:---------|
| Schedule | TVM / Halide | 暗示执行排程，太底层；LLM 不是在"排程"而是在"做策略决策" |
| Plan | 通用 | 太泛，和 "project plan" 混淆 |
| Recipe | OpenVINO | 偏烹饪隐喻，不够技术 |
| **Strategy** | ✅ 采用 | 强调决策序列 + 理由，和 LLM-as-decision-maker 角色完美对齐 |

**Design Rule:** Semantic IR 只描述"算什么"，Strategy IR 只描述"怎么优化"。两者通过 `kernel_id` 关联但完全解耦。同一个 Semantic IR 可以有多个 Strategy IR（不同硬件、不同优化方向）。

---

## 三、CLI 命令命名

CLI 入口为 `arke`（由 `arkec/main.py` 提供），子命令遵循 **动词** 模式：

| 命令 | 动词含义 | 输入 → 输出 | 阶段 |
|:-----|:---------|:-----------|:-----|
| `arke parse` | 解析 | `.ak` → Arke IR (JSON) | 前端 |
| `arke inspect` | 审视 | Arke IR → 人类可读视图 | 辅助 |
| `arke optimize` | 优化 | Arke IR → 优化后的 IR（带 Strategy） | 核心 |
| `arke codegen` | 生成代码 | Arke IR → Triton / CUDA / Ascend 代码 | 后端 |
| `arke verify` | 验证 | Arke IR → 验证报告（V0/V1/V2） | 验证 |

### 端到端命令流

```bash
# 完整流程
arke parse matmul.ak -o matmul.json            # .ak → IR
arke inspect matmul.json                        # 查看 IR
arke optimize matmul.json --target ampere       # AI 优化（生成 Strategy）
arke codegen matmul.json --target triton -o out.py  # IR → Triton
arke verify matmul.json --ref numpy             # 正确性验证

# 一步到位
arke optimize matmul.ak --target ampere --codegen triton -o out.py
```

### 为什么 CLI 叫 `arke` 而不是 `arkec`？

`arkec` 是包名 / 目录名（**arke** **c**ompiler），对应 `arkec/main.py`。
但安装后的命令行入口是 `arke`——用户面对的是项目名，不是实现细节。

类比：
- Rust: `rustc` 目录 → `rustc` 命令（编译器名即命令名）
- Go: `cmd/go/` 目录 → `go` 命令
- **Arke**: `arkec/` 目录 → `arke` 命令（项目名即命令名，更简洁）

---

## 四、核心组件命名

### 4.1 引擎层

| 名称 | 全称 | 含义 | 类比 |
|:-----|:-----|:-----|:-----|
| **ArkeEnv** | Arke Environment | LLM agent 的交互环境 | Gymnasium `Env`：agent 发 action，env 返 observation |
| **LegalActionsEngine** | — | 合法动作枚举器 | 棋盘游戏的合法走法生成器 |
| **StaticValidator** | — | V0 静态验证器 | 编译器的类型检查 |
| **NumericalValidator** | — | V1 数值验证器 | 单元测试的 assert |
| **PerformanceProfiler** | — | V2 性能验证器 | Benchmark harness |

### 4.2 Agent 层

| 名称 | 含义 | 来源 |
|:-----|:-----|:-----|
| **ArkeTool** | 工具基类 | 对齐 LLM tool-use 协议，每个 tool 是 ArkeEnv 的一个 API |
| **ToolMeta** | 工具元信息 | 声明式属性（并发、幂等、预算类型）。借鉴 Claude Code `Tool.ts` |
| **OptimizationEvent** | 优化事件 | AsyncGenerator yield 的事件流类型 |
| **OptimizationState** | 优化状态 | 跨 compact 的 ground truth（Strategy IR + 决策日志 + 编译结果）|
| **OptimizationBudget** | 优化预算 | 决策步数 + 编译次数的双限制 |
| **LLMProvider** | LLM 供应商 | 抽象接口，底下是 Anthropic / OpenAI-compatible / 本地 |

### 4.3 Codegen 层

| 名称 | 含义 |
|:-----|:-----|
| **ArkeBackend** | 后端抽象基类（TritonBackend, AscendBackend）|
| **ArkeCompiler** | 编译 + 加载 + 执行 |
| **TritonTemplateEngine** | 路径 A: Strategy IR → Jinja2 模板 → Triton 代码 |
| **TritonLLMGenerator** | 路径 B: Strategy IR → LLM 生成 → Triton 代码（🔬 实验性） |

---

## 五、Tool 命名规范

所有 ArkeEnv tool 遵循 `verb_noun` 模式（全小写 + 下划线）：

| Tool | 动词 | 名词 | 类别 |
|:-----|:-----|:-----|:-----|
| `create_kernel` | create | kernel | 构建 |
| `get_hw_profile` | get | hw_profile | 观测 |
| `get_semantic_ir` | get | semantic_ir | 观测 |
| `get_current_strategy` | get | current_strategy | 观测 |
| `analyze_compute` | analyze | compute | 分析 |
| `list_legal_actions` | list | legal_actions | 分析 |
| `apply_decision` | apply | decision | 决策 |
| `verify_correctness` | verify | correctness | 验证 |
| `compile_and_profile` | compile_and | profile | 验证 |
| `observe` | observe | (state) | 观测 |
| `checkpoint` | checkpoint | — | 状态管理 |
| `rollback` | rollback | — | 状态管理 |
| `restore` | restore | — | 状态管理 |

**命名规则：**
1. 观测类用 `get_` / `list_` 前缀 — 只读，不改状态
2. 决策类用 `apply_` 前缀 — 修改 Strategy IR
3. 验证类用 `verify_` / `compile_` 前缀 — 可能触发编译
4. 状态管理用裸动词 — `checkpoint`, `rollback`, `restore`
5. 全小写 + 下划线 — 对齐 OpenAI function calling 格式

---

## 六、验证层级命名

```
V0: Static Validation    静态验证    <1ms     每次 apply 自动执行
V1: Numerical Validation 数值验证    ~100ms   编译后 vs NumPy 参考
V2: Performance Profiling 性能验证   ~1-5s    vs cuBLAS/vendor 基线
```

**为什么用 V0/V1/V2？**
- 数字递增 = 成本递增 = 粒度递增 = 信心递增
- 类似软件测试的 "lint → unit test → integration test → benchmark"
- LLM 可以被告知"V0 很便宜随便用，V2 很贵节约用"

---

## 七、目录结构命名

```
arke/                  Python 包（import arke）
├── ir/                IR 层 — 编译器术语
│   ├── semantic.py    Semantic IR
│   ├── strategy.py    Strategy IR（原 schedule.py，W2-06 重命名）
│   ├── ops/           算子定义目录
│   ├── schemas/       JSON Schema
│   └── targets/       硬件 profile
├── engine/            引擎层 — 编译器术语
├── agent/             Agent 层 — AI/RL 术语
│   ├── tools/         工具实现
│   ├── providers/     LLM provider 适配
│   └── ...
├── backend/           后端层 — 编译器术语（codegen → 目标代码）
├── lang/              语言层 — 编译器术语（parser、AST）
├── learn/             学习层 — ML 术语（轨迹记录）
├── integration/       集成层（PyTorch custom op 等）
└── frontend/          前端导入层（从 PyTorch 导入算子定义）

arkec/                 编译器 CLI 入口（arkec = arke compiler）
├── main.py            Click CLI 定义
└── ...

examples/              示例
├── operators/         .ak 算子定义
├── ir/                .akir IR 示例（G6 v2 后生成）
├── pipelines/         端到端 walkthrough
└── agents/            Python Agent 示例

benchmarks/            评估框架
tests/                 测试
docs/architecture/     架构设计文档
```

**命名哲学：** 编译器领域的部分（ir, backend, lang, frontend）用编译器术语；AI 领域的部分（agent, learn）用 AI 术语。这反映了 Arke 的跨界本质。

---

## 八、文件命名规范

| 类型 | 规范 | 示例 |
|:-----|:-----|:-----|
| Python 模块 | `snake_case.py` | `semantic.py`, `legal_actions.py` |
| .ak 源文件 | `snake_case.ak` | `matmul.ak`, `fused_matmul_relu.ak` |
| Arke IR 文件 | `name.akir` | `matmul.akir` — Arke IR 文本格式（多层架构） |
| JSON Schema | `name.schema.json` | `semantic.schema.json` |
| HW Profile | `vendor_arch.json` | `nvidia_ampere.json`, `ascend_a3.json` |
| Jinja2 模板 | `pattern.py.j2` | `matmul.py.j2` |
| 设计文档 | `kebab-case.md` | `detailed-design-v2.1.md` |
| 示例文件 | `NN_name.ak/py` | `01_matmul.ak`, `agent_matmul.py` |

---

## 九、Decision 词汇表（LLM 面向）

这些是 LLM 在 tool-use 交互中使用的核心概念。System Prompt 中会定义它们：

| 术语 | 含义 | LLM 理解为 |
|:-----|:-----|:-----------|
| **Kernel** | 一个 GPU 计算核 | "要优化的函数" |
| **Decision** | 一个优化决策（tile/fuse/place/...） | "我做的一步棋" |
| **Rationale** | 决策的自然语言理由 | "我为什么这么做" |
| **Strategy** | 一系列 Decision 组成的优化方案 | "我的整体策略" |
| **Legal Action** | 当前状态下可执行的合法决策 | "我能走的棋" |
| **Checkpoint** | 保存的状态快照 | "存档点" |
| **Budget** | 剩余的决策步数和编译次数 | "我的资源限制" |
| **Fallback** | 预定义的保底策略 | "如果我搞砸了还有个兜底" |
| **Compact** | 上下文压缩（超长对话时自动触发） | "总结之前的对话继续" |

---

## 十、版本与发布命名

| 名称 | 格式 | 说明 |
|:-----|:-----|:-----|
| Python 包版本 | `0.1.0.dev0` → `0.1.0` | PEP 440，开发期用 `.devN` |
| Git tag | `v0.1.0` | `v` 前缀 |
| 设计文档版本 | `v2.1.3` | 大版本.小版本.补丁 |
| MVP 代号 | `v0.1.0` | 第一个可用版本 |

---

## 十一、缩写对照表

| 缩写 | 全称 | 首次出现 |
|:-----|:-----|:---------|
| IR | Intermediate Representation | 中间表示 |
| SIR | Semantic IR | 语义 IR |
| StrIR | Strategy IR | 策略 IR |
| HW | Hardware | 硬件 |
| CG | Code Generation | 代码生成 |
| V0/V1/V2 | Validation Level 0/1/2 | 验证层级 |
| E2E | End-to-End | 端到端 |
| CC | Claude Code | 借鉴来源标记 |
| MVP | Minimum Viable Product | 最小可用产品 |

---

*版本：v1.0 | 创建日期：2026-03-31*
