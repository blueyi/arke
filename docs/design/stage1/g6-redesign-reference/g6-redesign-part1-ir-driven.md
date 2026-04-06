# 方案一：G6 IR-Driven 架构重构

> **文档类型:** 设计方案  
> **版本:** 0.1.0-draft  
> **目标 Gate:** G6 — BL5×L1+L2 Lang & IR Completeness  
> **作者:** Arke Architecture Team  
> **创建时间:** 2026-04-06  
> **状态:** 草稿，待评审

---

## 目录

1. [G6 核心目标回顾](#1-g6-核心目标回顾)
2. [当前架构问题诊断](#2-当前架构问题诊断)
3. [IR-Driven Single Source of Truth 设计](#3-ir-driven-single-source-of-truth-设计)
4. [Arke Lang Spec v1.1 变更提案](#4-arke-lang-spec-v11-变更提案)
5. [Arke IR Spec v1.1 变更提案](#5-arke-ir-spec-v11-变更提案)
6. [编译器实现拆解](#6-编译器实现拆解)
7. [G6 Gate 验证映射](#7-g6-gate-验证映射)
8. [风险与缓解](#8-风险与缓解)

---

## 1. G6 核心目标回顾

### 1.1 G6 PASS 条件全文（9/9 必须全部通过）

```
AND ALL:
  [1] L1 BL5 correctness: 100%(ST1-3) + ≥95%(ST4, excl. OOM) for all OT0-OT4
  [2] L1 BL5 performance weighted_score ≥ 0.83
        weighted_score = 0.25×score(OT0-1) + 0.30×score(OT2) + 0.20×score(OT3) + 0.25×score(OT4)
  [3] L2 BL5: ≥3/4 fusion combinations pass
  [4] Lang&IR: G6-LI.1~LI.6 all pass
```

**各条件核心要求汇总：**

| 条件 | 操作符覆盖 | 形状覆盖 | 关键阈值 |
|:----:|:---------|:-------|:-------|
| L1 Correctness | 全部 45 ops (OT0-OT4) | ST1-ST3 (100%) + ST4 (≥95%) | — |
| L1 Performance | OT0-1: ≥0.90 P1 FlagGems<br>OT2: matmul ≥0.90 P0<br>OT3: swiglu/rope ≥0.85 P1<br>OT4: FA/GQA ≥0.80 P1 | ST1-ST4 | weighted_score ≥ 0.83 |
| L2 Fusion | matmul+relu/gelu, swiglu/geglu, linear+CE, QKV+FA | — | ≥3/4 融合组合通过 |
| G6-LI.1 | 45 ops 可解析 `.ak` | — | `arke parse` 全部 exit 0 |
| G6-LI.2 | `.ak → SemanticIR → StrategyIR` 完整 pipeline | — | round-trip 验证 |
| G6-LI.3 | `@rationale` 全链路保留 | — | ≥3 example 验证 |
| G6-LI.4 | token 效率：`.ak` ≤ Triton 行数 | OT0-OT4 | 行数比较 |
| G6-LI.5 | Python interop IR round-trip | — | from_json/to_json 45 ops |
| G6-LI.6 | Grammar 完整性：0 parse failures | — | array literal, float, 4D tensor |

### 1.2 当前实现状态分析

**已满足（继承自 G5）：**
- ✅ G6-LI.1 的 OT0-2 部分（33 ops 已有 `.ak` 文件）
- ✅ G6-LI.5 的 OT0-2 部分（JSON round-trip 通过）
- ✅ L1 BL3 Correctness（OT0-2 × ST1-3 通过）
- ✅ 基础 pipeline：`.ak → SemanticIR → StrategyIR → Triton → GPU`

**形式通过、架构不合格：**

| 问题 | 表现 | 为何"架构不合格" |
|:----|:----|:--------------|
| **分散的 op 知识** | catalog.py / shape_inference.py / numerical_check.py / triton_template_engine.py 各自维护 45 op 列表 | 添加一个新 op 需要改 6 个文件，极易遗漏；G6 的 OT3/OT4 ops 已经出现 catalog 有 op 但 numerical_check 缺对应实现的情况 |
| **numpy 手写 reference** | numerical_check.py 为每个 op 手写了独立的 `_numpy_*` 函数 | 函数签名不统一（部分用 `inputs["X"]`，部分用 `inputs.get("X", ...)` fallback），测试输入生成逻辑散落各处；无法自动组合测试 fused graph |
| **template selection if/elif 链** | triton_template_engine.py `_select_template()` 和 `_build_context()` 共 230+ 行 if/elif | 每添加一个 op 需扩展两个 if/elif 链；template 与 op 的映射关系没有显式声明，只能读代码推断 |
| **KernelCache `_build_ir()` 的 shape 硬编码** | kernel_cache.py 的 `_build_ir()` 用 if/elif 为每个 op 手动构造 SemanticIR | 没有利用 `.ak → SemanticIR` 已实现的 parser+converter 能力；添加新 op 需要在 KernelCache 里再写一遍 IR 构建逻辑 |

**尚未满足（G6 新增需求）：**
- ❌ OT3/OT4 的 12 个 op 的 `.ak` 文件缺失或不完整
- ❌ 4D 张量语法（OT4 attention ops 需要）
- ❌ `@rationale` 注解全链路保留（G6-LI.3）
- ❌ L2 BL5 fusion benchmark runner
- ❌ ST4 production shapes 的 numerical validation（部分 attention ops）

---

## 2. 当前架构问题诊断

### 2.1 六文件冗余量化分析

添加一个新 op（如 `cross_attention`）需要修改的文件和代码量：

| 文件 | 需要改动的内容 | 每个 op 的冗余代码行数 | 问题模式 |
|:----|:------------|:-----------------|:-------|
| `arke/ir/ops/catalog.py` | 添加 `OpDefinition(...)` 注册 | ~15 行 | 字段语义重复（inputs、computation、numpy_ref 都要手写） |
| `arke/ir/shape_inference.py` | 在 `_ATTENTION_OPS` 集合添加名字 + `validate_shapes()` 添加 elif | ~10 行 | op 名字在顶部集合和底部 if/elif 两处出现 |
| `arke/engine/numerical_check.py` | 添加 `_numpy_cross_attention()` + 注册到 `_OP_HANDLERS` | ~20 行 | 函数签名不统一；输入生成逻辑重复 |
| `arke/backend/triton_template_engine.py` | `_select_template()` 添加 elif + `_build_context()` 添加 elif | ~8 行 | op → template 映射隐式，无声明 |
| `arke/integration/kernel_cache.py` | `_build_ir()` 添加 if/elif + 手动构建 SemanticIR | ~20 行 | 重复 parser+converter 已实现的逻辑 |
| `arke/ir/semantic.py` | 如需新的 SemanticIR 子类（如 AttentionSemanticIR）则需修改 | ~30 行 | 每类 op 的专属 attrs 字段散落在 Node.attrs dict 中，无类型安全 |

**总计：** 添加一个 op 约需改动 6 个文件，100+ 行代码。当前 45 个 op 累计冗余代码约 **4500 行**，占 6 个文件总行数（~2500 行实现代码）的 180%——即大量代码是重复结构。

### 2.2 各文件 per-op 冗余代码模式

**catalog.py（OpDefinition 冗余）：**

```python
# 当前模式：每个 op 独立 _register() 调用
RELU = _register(OpDefinition(
    name="relu",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = max(X, 0)",
    properties=["elementwise", "monotonic"],
    can_fuse_as="epilogue",
    numpy_ref="np.maximum(X, 0)",   # ← 这里的 numpy_ref 字符串无法类型检查
))
# 问题：输入输出规格用字符串表达（"Tensor[M,K]"），无法机器验证
# 问题：numpy_ref 是字符串，运行时才求值，无法静态检测错误
```

**shape_inference.py（集合 + if/elif 双维护）：**

```python
# 当前模式：op 名字出现在两处
_ATTENTION_OPS = {"flash_attention", "grouped_query_attention",
                  "multi_latent_attention", "cross_attention", "paged_attention"}
# ... 200 行后 ...
if op_name in _ATTENTION_OPS:
    q = input_shapes["Q"]
    return list(q)   # ← 所有 attention ops 输出都是 Q 的 shape，但这是隐含知识
```

**numerical_check.py（_numpy_* 函数签名不统一）：**

```python
# 模式 A：直接 key 访问（会 KeyError）
def _numpy_matmul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.matmul(inputs["A"], inputs["B"])

# 模式 B：.get() fallback（掩盖 bug）
def _numpy_rope(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"].astype(np.float32)
    cos = inputs.get("cos", inputs.get("cos_cached", np.ones_like(...))).astype(np.float32)
    # ↑ 这个 fallback 逻辑意味着 "缺少 cos 输入时不报错，自动生成全 1"
    # 这隐藏了 IR 构建时的输入缺失 bug

# 问题：测试输入生成逻辑也散落各处，rsqrt 需要正数输入但没有声明
```

**triton_template_engine.py（隐式 op→template 映射）：**

```python
# 当前：_select_template 是 230 行的 if/elif 链
# op → template 的映射关系必须读代码才能理解
# 没有地方声明 "rope 用 rope.py.j2"——这是运行时行为

def _select_template(self, semantic: SemanticIR) -> tuple[str, str]:
    ops = [node.op for node in semantic.nodes]
    if "paged_attention" in ops:
        return "paged_attention.py.j2", "paged_attention"
    if "multi_latent_attention" in ops:
        return "mla.py.j2", "multi_latent_attention"
    attention_ops = {"flash_attention", "grouped_query_attention", "cross_attention"}
    if any(op in attention_ops for op in ops):
        ...
    # 42 个 elif 后...
```

**kernel_cache.py（_build_ir 重复 parser 逻辑）：**

```python
# 当前：手动构建 SemanticIR，等于实现了 parser+converter 已实现的功能
def _build_ir(self, op: str, **sp):
    if op in self._UNARY_ELEMENTWISE:
        n = sp.get("n_elements", sp.get("M", 1) * sp.get("N", 1))
        b = KernelBuilder(f"{op}_{n}")
        b.param("X", [n], "f16")
        node = b.op(op, X="X")
        b.returns(node, [n], "f16")
        return b.build(), strategy
    # 等于每个 op 写了一个 mini-compiler，但 .ak → IR 已经实现了这个功能
```

### 2.3 问题根本原因

**当前架构的根本缺陷：op 的知识分散在 6 个地方，没有 single source of truth。**

```
当前状态：

   .ak 文件
      ↓ parser
   Program AST
      ↓ converter
   SemanticIR ←── catalog.py 定义 op
                     ↓
               shape_inference.py
                     ↓
               numerical_check.py ←── 手写 numpy reference
                     ↓
           triton_template_engine.py ←── 手写 if/elif 链
                     ↓
               kernel_cache.py ←── 手写 IR 构建

每层都独立维护一份"我知道 45 个 op"的知识。
```

---

## 3. IR-Driven Single Source of Truth 设计

### 3.1 核心思想

**SemanticIR 及其 OpRegistry 应该是唯一的 op 知识来源。** 其他所有组件（shape inference、numerical check、template engine、kernel cache）都从 OpRegistry 派生行为，而不是各自维护自己的 op 列表。

**目标架构：**

```
OpRegistry（单一真相源）
    │
    ├── op.shape_rule      → ShapeInferenceEngine (声明式)
    ├── op.torch_ref       → SemanticInterpreter (PyTorch eager)
    ├── op.template_hint   → TemplateRouter (分类路由)
    ├── op.input_specs     → InputGenerator (测试输入生成)
    └── op.category        → 所有下游组件

   .ak 文件
      ↓ parser + converter（现有，已实现）
   SemanticIR（引用 OpRegistry 中的 op 定义）
      ↓ ShapeInferenceEngine（从 OpRegistry 派生）
   shapes resolved
      ↓ SemanticInterpreter（从 OpRegistry 的 torch_ref 派生）
   numerical reference output
      ↓ TemplateRouter（从 OpRegistry 的 category+template_hint 派生）
   Triton source
```

### 3.2 A) OpRegistry：统一 Op 注册中心

#### 3.2.1 核心数据结构

```python
# arke/ir/registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Any
from collections.abc import Sequence


# ── Shape Rule 声明式系统 ──────────────────────────────────────

@dataclass(frozen=True)
class SameAsInput:
    """输出 shape 与指定输入相同。默认取第一个输入。"""
    input_name: str = "X"


@dataclass(frozen=True)
class BroadcastShape:
    """numpy 风格广播，取所有输入的广播 shape。"""
    input_names: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class MatmulShape:
    """C[..., M, N] ← A[..., M, K] × B[..., K, N]"""
    lhs: str = "A"
    rhs: str = "B"
    batch_dims: int = 0          # 0=2D matmul, 1=batch_matmul, -1=任意前缀 batch


@dataclass(frozen=True)
class ReduceLastDim:
    """沿最后一个维度归约，输出 shape 去掉最后一维。"""
    input_name: str = "X"


@dataclass(frozen=True)
class GatedSplit:
    """gated activation（swiglu/geglu）：输入最后维度 2N → 输出 N。"""
    input_name: str = "X"


@dataclass(frozen=True)
class TransposeShape:
    """2D 转置：[M, N] → [N, M]"""
    input_name: str = "X"


@dataclass(frozen=True)
class ConcatLastDim:
    """沿最后维度拼接：[M, N1] + [M, N2] → [M, N1+N2]"""
    inputs: tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class GatherShape:
    """输出 shape = indices shape"""
    data: str = "X"
    indices: str = "idx"


@dataclass(frozen=True)
class EmbeddingShape:
    """weight[indices]：indices[B,S] + weight[V,D] → [B,S,D]"""
    indices: str = "indices"
    weight: str = "weight"


@dataclass(frozen=True)
class CustomShape:
    """复杂 op 使用自定义函数。函数注册在 OpRegistry 中。"""
    fn_name: str  # 注册的函数名，接受 dict[str, list[int]] → list[int]


# Shape rule 联合类型
ShapeRule = (
    SameAsInput | BroadcastShape | MatmulShape | ReduceLastDim |
    GatedSplit | TransposeShape | ConcatLastDim | GatherShape |
    EmbeddingShape | CustomShape
)


# ── InputSpec：输入约束声明 ────────────────────────────────────

@dataclass(frozen=True)
class TensorInputSpec:
    """一个张量输入的规格约束。"""
    name: str
    rank_min: int = 1
    rank_max: int = 8
    dtype_constraint: tuple[str, ...] = ()   # 空表示任意 dtype
    positive_only: bool = False              # rsqrt、log 等需要正数输入
    integer_only: bool = False              # indices 类输入
    shape_note: str = ""                     # 人类可读描述，如 "Tensor[B,H,S,D]"


@dataclass(frozen=True)
class ScalarInputSpec:
    """标量参数（如 eps, axis）。"""
    name: str
    dtype: str = "f32"
    default: Any = None


InputSpec = TensorInputSpec | ScalarInputSpec


# ── OpDef：单个 Op 的完整定义 ──────────────────────────────────

@dataclass(frozen=True)
class OpDef:
    """Op 的完整、统一定义。OpRegistry 中的每条记录。

    设计原则：
    - 所有字段都可以从 .ak 文件 + IR spec 中机械推导或由 op 作者注册一次
    - 下游系统（shape inference, numerical check, template engine）
      全部从这个结构派生，不再维护自己的 op 列表
    """

    # ── 基础标识 ──
    name: str                              # op 名称，全局唯一
    category: str                          # "elementwise" | "reduction" | "compute" |
                                           # "attention" | "data_movement" | "quantize"

    # ── 输入输出规格 ──
    input_specs: tuple[InputSpec, ...]     # 有序输入规格（含顺序语义）
    output_rule: ShapeRule                 # 输出 shape 计算规则
    dtype_constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # e.g. {"A": ("f16", "bf16", "f32"), "B": ("f16", "bf16", "f32")}

    # ── Numerical Reference ──
    torch_ref: Callable | None = None      # PyTorch eager 实现
    # 签名：(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor

    # ── Template 路由 ──
    template_hint: str = ""               # 对应的 .j2 模板名（不含扩展名前缀）
    # e.g. "elementwise", "reduction", "matmul", "flash_attention"
    # 为空时由 category 自动推断

    # ── 代数性质 ──
    properties: tuple[str, ...] = ()      # "commutative", "associative", etc.
    fusable_as: str | None = None         # "epilogue" | "prologue" | None

    # ── 文档 ──
    computation: str = ""                 # 数学公式，如 "C[i,j] = sum(A[i,k]*B[k,j])"
    description: str = ""                 # 人类可读描述
```

#### 3.2.2 OpRegistry：注册器与查找接口

```python
# arke/ir/registry.py（续）

class OpRegistry:
    """全局 Op 注册器。线程安全（只在模块初始化时写入）。"""

    _registry: dict[str, OpDef] = {}
    _custom_shape_fns: dict[str, Callable] = {}

    @classmethod
    def register(cls, op_def: OpDef) -> OpDef:
        """注册一个 OpDef。重复注册同名 op 会 raise。"""
        if op_def.name in cls._registry:
            raise ValueError(f"Op '{op_def.name}' already registered")
        cls._registry[op_def.name] = op_def
        return op_def

    @classmethod
    def register_shape_fn(cls, name: str, fn: Callable) -> None:
        """注册一个自定义 shape inference 函数（用于 CustomShape）。"""
        cls._custom_shape_fns[name] = fn

    @classmethod
    def get(cls, name: str) -> OpDef:
        """获取 OpDef。未找到时 raise KeyError。"""
        if name not in cls._registry:
            raise KeyError(f"Op '{name}' not registered. Available: {list(cls._registry)[:10]}...")
        return cls._registry[name]

    @classmethod
    def list_ops(cls, category: str | None = None) -> list[OpDef]:
        ops = list(cls._registry.values())
        if category:
            ops = [op for op in ops if op.category == category]
        return ops

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._registry


# ── 装饰器 API ──────────────────────────────────────────────────

def register_op(
    name: str,
    category: str,
    input_specs: Sequence[InputSpec],
    output_rule: ShapeRule,
    template_hint: str = "",
    dtype_constraints: dict[str, tuple[str, ...]] | None = None,
    properties: tuple[str, ...] = (),
    fusable_as: str | None = None,
    computation: str = "",
    description: str = "",
) -> Callable:
    """装饰器：将函数注册为 op 的 torch_ref 并完成 OpDef 注册。

    用法示例：
        @register_op(
            name="relu",
            category="elementwise",
            input_specs=[TensorInputSpec("X")],
            output_rule=SameAsInput("X"),
            template_hint="elementwise",
            fusable_as="epilogue",
            computation="Y = max(X, 0)",
        )
        def relu_ref(inputs: dict[str, Tensor], attrs: dict) -> Tensor:
            return torch.relu(inputs["X"])
    """
    def decorator(fn: Callable) -> Callable:
        op_def = OpDef(
            name=name,
            category=category,
            input_specs=tuple(input_specs),
            output_rule=output_rule,
            template_hint=template_hint or _default_template_hint(category),
            dtype_constraints=dtype_constraints or {},
            properties=properties,
            fusable_as=fusable_as,
            computation=computation,
            description=description,
            torch_ref=fn,
        )
        OpRegistry.register(op_def)
        return fn
    return decorator


def _default_template_hint(category: str) -> str:
    """从 category 推断默认 template_hint。"""
    _map = {
        "elementwise": "elementwise",
        "reduction": "reduction",
        "compute": "matmul",
        "attention": "flash_attention",
        "data_movement": "data_movement",
        "quantize": "quantize",
    }
    return _map.get(category, "elementwise")
```

#### 3.2.3 完整 Op 注册示例（与现有 catalog.py 的对比）

```python
# arke/ir/ops/registry_ops.py
# 替代 catalog.py，全部 45 个 op 在此文件注册

import torch
import torch.nn.functional as F
from arke.ir.registry import register_op, TensorInputSpec, ScalarInputSpec
from arke.ir.registry import SameAsInput, BroadcastShape, MatmulShape
from arke.ir.registry import ReduceLastDim, GatedSplit, CustomShape


# ── OT0: Elementwise ─────────────────────────────────────────

@register_op(
    name="relu",
    category="elementwise",
    input_specs=[TensorInputSpec("X")],
    output_rule=SameAsInput("X"),
    template_hint="elementwise",
    fusable_as="epilogue",
    computation="Y = max(X, 0)",
)
def relu_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    return torch.relu(inputs["X"])


@register_op(
    name="rsqrt",
    category="elementwise",
    input_specs=[TensorInputSpec("X", positive_only=True)],  # ← 声明输入约束
    output_rule=SameAsInput("X"),
    template_hint="elementwise",
    fusable_as="epilogue",
    computation="Y = 1 / sqrt(X)",
)
def rsqrt_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    return torch.rsqrt(inputs["X"])


@register_op(
    name="add",
    category="elementwise",
    input_specs=[TensorInputSpec("A"), TensorInputSpec("B")],
    output_rule=BroadcastShape(("A", "B")),   # ← v1.1 支持广播
    template_hint="elementwise_binary",
    properties=("commutative", "associative"),
    fusable_as="epilogue",
    computation="Y = A + B",
)
def add_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    return inputs["A"] + inputs["B"]


# ── OT1: Reduction ───────────────────────────────────────────

@register_op(
    name="softmax",
    category="reduction",
    input_specs=[TensorInputSpec("X", rank_min=2)],
    output_rule=SameAsInput("X"),
    template_hint="softmax",
    computation="Y[i,j] = exp(X[i,j]) / sum(exp(X[i,:]))",
)
def softmax_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    return F.softmax(inputs["X"].float(), dim=-1).to(inputs["X"].dtype)


@register_op(
    name="rmsnorm",
    category="reduction",
    input_specs=[
        TensorInputSpec("X", rank_min=2),
        TensorInputSpec("W", rank_min=1, rank_max=1, shape_note="Tensor[N]"),
        ScalarInputSpec("eps", default=1e-5),
    ],
    output_rule=SameAsInput("X"),
    template_hint="layernorm",
    computation="Y[i,j] = X[i,j] / sqrt(mean(X[i,:]^2) + eps) * W[j]",
)
def rmsnorm_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    x = inputs["X"].float()
    w = inputs["W"].float()
    eps = attrs.get("eps", 1e-5)
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rms * w).to(inputs["X"].dtype)


# ── OT2: Compute-Dense ───────────────────────────────────────

@register_op(
    name="matmul",
    category="compute",
    input_specs=[
        TensorInputSpec("A", rank_min=2, rank_max=2, shape_note="Tensor[M,K]"),
        TensorInputSpec("B", rank_min=2, rank_max=2, shape_note="Tensor[K,N]"),
    ],
    output_rule=MatmulShape(lhs="A", rhs="B", batch_dims=0),
    template_hint="matmul",
    properties=("associative", "distributive"),
    fusable_as="prologue",
    computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
)
def matmul_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    return torch.matmul(inputs["A"].float(), inputs["B"].float()).to(inputs["A"].dtype)


@register_op(
    name="batch_matmul",
    category="compute",
    input_specs=[
        TensorInputSpec("A", rank_min=3, rank_max=3, shape_note="Tensor[B,M,K]"),
        TensorInputSpec("B", rank_min=3, rank_max=3, shape_note="Tensor[B,K,N]"),
    ],
    output_rule=MatmulShape(lhs="A", rhs="B", batch_dims=1),
    template_hint="batch_matmul",
    computation="C[b,i,j] = sum(A[b,i,k] * B[b,k,j], axis=k)",
)
def batch_matmul_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    return torch.bmm(inputs["A"].float(), inputs["B"].float()).to(inputs["A"].dtype)


# ── OT4: Attention（自定义 shape function 示例）────────────────

def _attention_shape_fn(input_shapes: dict[str, list[int]]) -> list[int]:
    """所有 attention ops 的输出 shape = Q 的 shape。"""
    return list(input_shapes["Q"])

OpRegistry.register_shape_fn("attention_output", _attention_shape_fn)


@register_op(
    name="flash_attention",
    category="attention",
    input_specs=[
        TensorInputSpec("Q", rank_min=4, rank_max=4, shape_note="Tensor[B,H,S,D]"),
        TensorInputSpec("K", rank_min=4, rank_max=4, shape_note="Tensor[B,H,S,D]"),
        TensorInputSpec("V", rank_min=4, rank_max=4, shape_note="Tensor[B,H,S,D]"),
        ScalarInputSpec("causal", default=True),
        ScalarInputSpec("scale", default=None),   # None → 1/sqrt(D)
    ],
    output_rule=CustomShape(fn_name="attention_output"),
    template_hint="flash_attention",
    properties=("causal_mask_optional", "online_softmax"),
    computation="O = softmax(Q @ K^T / sqrt(D)) @ V  (tiled, online softmax)",
)
def flash_attention_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    q, k, v = inputs["Q"].float(), inputs["K"].float(), inputs["V"].float()
    causal = attrs.get("causal", True)
    scale = attrs.get("scale") or (q.shape[-1] ** -0.5)
    return F.scaled_dot_product_attention(q, k, v, scale=scale,
                                          is_causal=causal).to(inputs["Q"].dtype)


@register_op(
    name="grouped_query_attention",
    category="attention",
    input_specs=[
        TensorInputSpec("Q", rank_min=4, rank_max=4, shape_note="Tensor[B,H_q,S,D]"),
        TensorInputSpec("K", rank_min=4, rank_max=4, shape_note="Tensor[B,H_kv,S,D]"),
        TensorInputSpec("V", rank_min=4, rank_max=4, shape_note="Tensor[B,H_kv,S,D]"),
        ScalarInputSpec("causal", default=True),
        ScalarInputSpec("num_kv_heads", default=None),
    ],
    output_rule=CustomShape(fn_name="attention_output"),
    template_hint="flash_attention",
    properties=("kv_head_repeat", "online_softmax"),
    computation="GQA: Q heads grouped over fewer KV heads",
)
def gqa_ref(inputs: dict, attrs: dict) -> torch.Tensor:
    q, k, v = inputs["Q"].float(), inputs["K"].float(), inputs["V"].float()
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    group = H_q // H_kv
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    causal = attrs.get("causal", True)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).to(inputs["Q"].dtype)
---

## 4. Arke Lang Spec v1.1 变更提案

### 4.1 新增：`@meta` 注解（Op 元数据）

```ak
// v1.1 — Op 定义可以携带元数据注解（可选，用于自动推导 catalog）
kernel tanh_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16>
@meta(category="elementwise", shape_rule="same_as_input")
@meta(input_constraint="X: positive=false")
{
    let Y = tanh(X=X);
    return Y;
}
```

**设计要点：**
- `@meta` 是可选注解，不影响语义
- 用于自动生成 OpRegistry 信息（替代手写 catalog）
- 现有 v1.0 .ak 文件不需要修改（没有 `@meta` 时用默认值）

### 4.2 新增：属性参数

```ak
// v1.1 — op call 支持字面量属性（如 eps, axis, dtype）
let Y = layernorm(X=X, weight=W, bias=B, eps=1e-5);
let Z = reduce_sum(X=X, axis=-1);
let W = cast(X=X, dtype="f16");
```

v1.0 parser 已支持 string/int/float literal，但 converter 不传递到 SemanticIR。v1.1 将属性存入 `Node.attrs`。

### 4.3 语法变更总结

| 变更 | 类型 | 向后兼容 |
|------|------|---------|
| `@meta(...)` kernel 注解 | 新增 | ✅ 可选 |
| `where` clause（动态 shape，见方案三） | 新增 | ✅ 可选 |
| op call 的属性参数 → `Node.attrs` | 行为变更 | ✅ 之前被忽略，现在传递 |

---

## 5. Arke IR Spec v1.1 变更提案

### 5.1 SemanticIR 变更

| 字段 | v0.3.0 | v0.4.0 | 说明 |
|------|--------|--------|------|
| `version` | `"0.3.0"` | `"0.4.0"` | 版本升级 |
| `Node.attrs` | 不存在 | `dict[str, Any]` | op 属性（eps, axis, dtype 等） |
| `symbolic_dims` | 不存在 | `list[SymbolicDim]` | 符号维度定义（方案三） |
| `constraints` | 不存在 | `list[ShapeConstraint]` | shape 约束（方案三） |

### 5.2 OpDef 扩展（从 OpRegistry）

| 字段 | 旧 catalog | 新 OpRegistry | 说明 |
|------|-----------|--------------|------|
| `name` | ✅ | ✅ | 不变 |
| `category` | ✅ str | ✅ str | 不变 |
| `inputs` | dict[str, str] | `list[InputSpec]` | 结构化约束 |
| `output` | str | `ShapeRule` | 声明式 shape 推导 |
| `shape_rule` | ❌ | ✅ | same_as_input/broadcast/matmul 等 |
| `torch_ref` | ❌ | ✅ Callable | PyTorch reference 实现 |
| `template_hint` | ❌ | ✅ str | 路由到哪个 Triton template |
| `properties` | list[str] | tuple[str, ...] | 不变 |

---

## 6. 编译器实现拆解

| 文件 | 操作 | 工作内容 | 估时 |
|------|------|---------|------|
| `arke/ir/registry.py` | **新增** | OpRegistry + ShapeRule + InputSpec + 装饰器 API | 3d |
| `arke/ir/ops/registry_ops.py` | **新增** | 45 个 op 注册（替代 catalog.py 的数据）| 2d |
| `arke/ir/ops/catalog.py` | **改造** | 改为 OpRegistry 的兼容 wrapper（get_op → registry.get）| 0.5d |
| `arke/ir/shape_inference.py` | **重写** | 改为从 OpRegistry 的 shape_rule 派生 | 2d |
| `arke/engine/semantic_interpreter.py` | **新增** | PyTorch eager 执行 SemanticIR graph | 2d |
| `arke/engine/numerical_check.py` | **重写** | 改用 SemanticInterpreter，删除 45 个 numpy 函数 | 1d |
| `arke/backend/triton_template_engine.py` | **改造** | _select_template 改用 OpRegistry.template_hint | 1d |
| `arke/integration/kernel_cache.py` | **改造** | compile_op 走 .ak parse 路径 | 1d |
| `benchmarks/baselines/arke_runner.py` | **改造** | _build_test_inputs 从 OpRegistry InputSpec 派生 | 1d |
| `tests/test_registry.py` | **新增** | OpRegistry 单元测试 | 1d |
| `tests/test_interpreter.py` | **新增** | SemanticInterpreter 单元测试 | 1d |

**总计：~15.5 人天**

### 关键路径

```
OpRegistry (3d) → registry_ops (2d) → shape_inference 重写 (2d)
                                    → SemanticInterpreter (2d) → numerical_check 重写 (1d)
                                    → template_engine 改造 (1d)
                                    → kernel_cache 改造 (1d)
并行：catalog wrapper (0.5d), tests (2d)
总串行路径: ~8d
```

---

## 7. G6 Gate 验证映射

重构后，所有 G6 criteria 必须继续通过：

| G6 Criteria | 验证方式 | 重构影响 |
|------------|---------|---------|
| G6.7 all .ak parse | `arke parse` | ❌ 不影响（parser 不变） |
| G6.2 ast_to_strategy | `ast_to_strategy()` | ❌ 不影响 |
| G6.3 @rationale preserved | pipeline test | ❌ 不影响 |
| G6.1 .ak → GPU E2E correct | `ArkePipeline.from_ak_file()` | ✅ 改用 SemanticInterpreter 验证 |
| G6.8 45 ops correct | per-op GPU test | ✅ 改用 OpRegistry + Interpreter |
| G6.4 token efficiency | line count | ❌ 不影响 |
| G6.5 IR round-trip | `to_dict/from_dict` | ✅ 需要支持新字段（Node.attrs, symbolic_dims） |
| G6.6 IR-MLIR mapping | doc check | ❌ 不影响 |
| G6.9 Spec frozen | doc check | ⚠️ Spec 升级到 v1.1（需要标注 v1.0 冻结、v1.1 为增量） |

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| SemanticInterpreter 精度与 numpy 不一致 | G6.1/G6.8 回归 | 先跑 diff test 验证所有 45 ops 一致 |
| OpRegistry 迁移遗漏 op | 功能缺失 | 自动化检查：registry ops 集合 == catalog ops 集合 |
| template_hint 路由覆盖不全 | codegen 失败 | 保留 _select_template fallback 路径 |
| Spec v1.1 破坏 v1.0 兼容 | 旧 .ak 不能跑 | 所有新字段可选，默认值等价 v1.0 行为 |
| 重构期间 test 大面积红 | 开发效率 | 分阶段合入：先 OpRegistry，再逐步替换下游 |

---

*Created: 2026-04-06 | Author: Arke Architecture Team*
