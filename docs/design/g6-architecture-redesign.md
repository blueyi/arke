# G6 架构重设计方案

> **文档目的：** G6 实现评估与架构演进设计
> **状态：** 草案 (2026-04-06)
> **作者：** Arke 架构组
> **关联文档：**
> - `docs/design/stage1-gate-design.md` — G6/G7/G8 exit criteria
> - `docs/spec/arke-lang-spec-v1.md` — 语言规范 v1.0
> - `docs/spec/arke-ir-spec-v1.md` — IR 规范 v1.0
> - `docs/spec/ir-mlir-mapping.md` — IR-MLIR 映射

---

## 目录

1. [背景与问题陈述](#1-背景与问题陈述)
2. [方案一：G6 IR-Driven 架构重构](#2-方案一g6-ir-driven-架构重构)
3. [方案二：多级后端扩展性设计](#3-方案二多级后端扩展性设计)
4. [方案三：动态 Shape 支持设计](#4-方案三动态-shape-支持设计)
5. [方案优先级与实施路径](#5-方案优先级与实施路径)

---

## 1. 背景与问题陈述

G6 gate（BL5×L1+L2，Lang & IR Completeness）已经在技术上通过：45 个 op 的 `.ak` 文件存在，模板引擎能够生成 Triton 代码，bench 脚本可以跑通。但当前实现存在一个结构性缺陷：**45 个 op 的知识分散在 6 个独立文件中**，每个文件各自维护一份 op 列表，互相之间没有派生关系。

### 当前架构的问题清单

| 文件 | 维护的 op 知识 | 问题 |
|:-----|:--------------|:-----|
| `arke/ir/ops/catalog.py` | `OpDefinition`（`name`, `category`, `inputs`, `numpy_ref`...） | 手写每个 op 的签名、计算公式；P0 只有 ~10 个，其余 35 个 op 的 OpDef 补充程度参差不齐 |
| `arke/ir/shape_inference.py` | op 分组集合（`_ELEMENTWISE_UNARY`, `_ATTENTION_OPS`...）+ shape 推导逻辑 | 与 catalog.py 完全独立，分组逻辑重复；新增 op 必须同时在两处维护 |
| `arke/engine/numerical_check.py` | 每个 op 的 NumPy 参考实现（`_numpy_matmul`, `_numpy_softmax`...） | 45 个 `_numpy_<op>` 函数手写，与 catalog.py 的 `numpy_ref` 字段完全重复 |
| `arke/integration/kernel_cache.py` | `_build_ir()` 里每个 op 的 `KernelBuilder` 调用序列 | 大量 `if op == "matmul": ... elif op == "batch_matmul": ...` 分支 |
| `arke/backend/triton_template_engine.py` | `_select_template()` 里的 op→template 映射及优先级 | `if "paged_attention" in ops: ... elif ...` 的长链；优先级是隐式知识 |
| `arke/ir/semantic.py` | `SemanticIR.Node.attrs`（op-specific 属性） | `attrs` 是无类型 `dict`，`eps`, `axis`, `num_heads` 等参数无 schema 约束 |

**添加一个新 op 需要修改这 6 个文件。** 这不是编译器该有的架构——op 的知识应该集中定义在一处，其余组件从中派生。

### 与 G6 核心目标的深层矛盾

G6 的目标是"验证 Arke Lang 和 Arke IR 对所有 45 ops 具有完整的**表达能力**、代码生成能力和性能竞争力"。当前实现：

- **表达能力**：`.ak` 文件能写出来，parser 能 parse——但 SemanticIR 缺少 op 属性 schema，`eps`, `axis`, `theta` 等参数在 `attrs dict` 里无类型约束。parse 阶段的类型信息丢失无法在后续 pass 中恢复。
- **代码生成**：依赖 `_select_template()` 的 if/elif 链和 `_build_ir()` 的 per-op 分支——这是手工编码，不是从 IR 派生的 codegen。
- **向 Stage 2 (MLIR) 演进**：MLIR lowering 需要 OpDef 携带 `shape_rule`（shape analysis）、`dtype_constraints`（类型检查）、结构化属性信息。当前 `attrs: dict` 无法支持。

三个方案分别解决：
- **方案一**：建立 Single Source of Truth，消除 op 知识的分散
- **方案二**：为 Stage 2 (MLIR) / Stage 3 (LLVM IR) 后端扩展建立抽象层
- **方案三**：动态 Shape 端到端设计

---

## 2. 方案一：G6 IR-Driven 架构重构

### 2.1 G6 核心目标与现状评估

#### G6 PASS 条件（来自 stage1-gate-design.md §5）

```
AND ALL:
  [1] L1 BL5 correctness: 100%(ST1-3) + ≥95%(ST4, excl. OOM) for all OT0-OT4
  [2] L1 BL5 performance weighted_score ≥ 0.83
  [3] L2 BL5: ≥3/4 fusion combinations pass
  [4] Lang&IR: G6-LI.1~LI.6 all pass
```

#### G6-LI 条件现状（"形式通过"的分析）

| ID | 条件 | 当前状态 | 架构问题 |
|:---|:-----|:---------|:---------|
| G6-LI.1 | 所有 45 ops `.ak` 可解析 | ✅ 通过 | op 属性（`eps`, `axis` 等）在语法层面是 generic `arg_value`，无类型约束 |
| G6-LI.2 | `.ak → SemanticIR → StrategyIR` 全 pipeline | ✅ 通过 | op attrs 作为 `dict` 传递，无 schema 验证，parser 和 builder 生成的 attrs 可能不一致 |
| G6-LI.3 | `@rationale` 全流程保留 | ✅ 通过 | — |
| G6-LI.4 | `.ak` token 效率 | ✅ 通过 | — |
| G6-LI.5 | Python interop IR round-trip | ✅ 通过 | `attrs` round-trip 仅保证 dict 结构，不验证属性类型一致性 |
| G6-LI.6 | Grammar completeness | ✅ 通过 | — |

#### "形式通过但架构不合格"的具体诊断

**诊断 1：`SemanticIR.Node.attrs` 无类型 dict**

```python
# 当前 semantic.py
@dataclass
class Node:
    attrs: dict   # {"eps": 1e-5} 还是 {"eps": "1e-5"}？
                  # "eps" 还是 "epsilon"？无 schema，无默认值，无范围约束
```

Parser 生成的 `attrs` 和 `KernelBuilder` 手写的 `attrs` 可能键名不一致（`"eps"` vs `"epsilon"`），类型不一致（`float` vs `str`）。downstream 代码只能做 `attrs.get("eps", 1e-5)` 这种防御性取值，掩盖了潜在的 bug。

**诊断 2：`shape_inference.py` 与 `catalog.py` 平行维护**

```python
# shape_inference.py — 手工维护的 op 分组（与 catalog.py 完全独立）
_ELEMENTWISE_UNARY = {"relu", "gelu", "silu", "tanh", "sigmoid", ...}

# catalog.py — 也有 category 字段，描述同一件事
OpDefinition(name="relu",    category="elementwise", ...)
OpDefinition(name="sigmoid", category="elementwise", ...)
```

新增 op 必须同时更新两处；没有静态检查保证一致性，已经出现过 op 在 catalog 但不在 shape_inference 的 bug。

**诊断 3：`numerical_check.py` 的 45 个 `_numpy_<op>` 与 `catalog.py` 的 `numpy_ref` 重复**

```python
# catalog.py 已经有（字符串形式）
OpDefinition(name="matmul", numpy_ref="np.matmul(A, B)", ...)

# numerical_check.py 又有（可执行形式）
def _numpy_matmul(inputs): return np.matmul(inputs["A"], inputs["B"])
```

两者描述同一件事，但没有任何机制保证一致。`numpy_ref` 字段从未被执行，完全是装饰性的。

**诊断 4：`kernel_cache._build_ir()` 是 OpDef 信息的命令式重新编码**

```python
# kernel_cache.py — ~400 行 if/elif 分支
if op == "matmul":
    b.param("A", [m, k], "f16")   # OpDef.inputs 已有：{"A": "Tensor[M,K]"}
    b.param("B", [k, n], "f16")   # OpDef.inputs 已有：{"B": "Tensor[K,N]"}
    node = b.op("matmul", A="A", B="B")
    b.returns(node, [m, n], "f16")  # OpDef.output 已有：Tensor[M,N]
```

`OpDef.inputs` 已声明参数名和 shape 符号，`OpDef.output` 已声明输出 shape。`_build_ir()` 只是手工重新编码这些信息——完全可以从 OpDef 自动派生。

**诊断 5：`template_engine._select_template()` 的隐式优先级**

```python
# triton_template_engine.py — 优先级硬编码为 if/elif 顺序
if "paged_attention" in ops:          # priority=100（隐式）
    return "paged_attention.py.j2", ...
if "multi_latent_attention" in ops:   # priority=99（隐式）
    return "mla.py.j2", ...
```

模板优先级是隐式知识，没有在 OpDef 中声明。两个 op 同时出现时路由结果依赖代码顺序，没有文档说明。

---

### 2.2 Single Source of Truth 设计

核心原则：**`OpDef` 是唯一的 op 知识来源，其他组件从 `OpDef` 派生。**

#### 2.2.1 Arke IR Spec v1.1 新增数据结构

定义在 `arke/ir/ops/catalog.py`：

```python
@dataclass(frozen=True)
class AttrSpec:
    """Op 属性的规格声明（IR Spec v1.1 新增）。"""
    name: str
    dtype: type                        # int | float | str | bool | list
    default: Any                       # 默认值，None 表示必填
    description: str = ""
    valid_range: tuple | None = None   # (min, max)，用于 numeric attrs
    valid_values: list | None = None   # 枚举，用于 str 或 int attrs


@dataclass(frozen=True)
class ShapeRule:
    """声明式 shape 推导规则（IR Spec v1.1 新增）。

    用 symbolic dimension 描述 input → output 的 shape 变换。
    matmul 示例：A=[M,K], B=[K,N] → output=[M,N]，约束 A.shape[-1]==B.shape[0]。
    swiglu 示例：X=[M,N] → output=[M,N//2]（gated split）。
    """
    input_dims: dict[str, list[str]]   # {"A": ["M","K"], "B": ["K","N"]}
    output_dims: list[str]             # ["M","N"] 或 ["M","N//2"]
    constraints: list[str] = field(default_factory=list)
    # Python 表达式，编译期验证，如 "A.shape[-1] == B.shape[0]"


@dataclass(frozen=True)
class DtypeConstraint:
    """dtype 合法性约束（IR Spec v1.1 新增）。"""
    inputs: dict[str, list[str]] = field(default_factory=dict)
    # {"A": ["f16","bf16","f32"]}；空 dict 表示所有 dtype 合法
    output_follows: str | None = None
    # "A" 表示 output dtype 跟随输入 A；None 表示独立指定


@dataclass(frozen=True)
class OpDefinition:
    """Complete definition of an operator — Single Source of Truth.

    v1.1 新增字段以 [NEW] 标注。
    所有 v1.0 字段向后兼容——已有 OpDef 不需要立即补充新字段（默认为 None/空）。
    """
    name: str
    category: str        # "elementwise"|"reduce"|"compute"|"attention"|"move"|"quant"
    inputs: dict[str, str]   # {"A": "Tensor[M,K]", "B": "Tensor[K,N]"}
    output: str              # "Tensor[M,N]"
    computation: str

    # [NEW] v1.1 扩展字段
    op_class: str = ""           # 细粒度分类："unary"|"binary"|"norm"|"matmul"|"attention_mha"|...
    shape_rule: "ShapeRule | None" = None         # 声明式 shape 推导
    attr_specs: tuple = field(default_factory=tuple)  # tuple[AttrSpec, ...]
    dtype_constraints: "DtypeConstraint" = field(default_factory=DtypeConstraint)
    template_name: str | None = None       # 对应 Triton 模板文件名
    template_priority: int = 0             # fusion 时模板路由优先级（大者优先）
    numpy_fn: Any | None = None            # 可执行的参考实现 callable

    # v1.0 保留字段
    index_vars: tuple = field(default_factory=tuple)
    reduction_axes: tuple = field(default_factory=tuple)
    properties: tuple = field(default_factory=tuple)
    can_fuse_as: str | None = None
    numpy_ref: str = ""    # string 形式保留用于文档；numpy_fn 是执行形式
```

#### 2.2.2 OpDef 注册示例（重构后 catalog.py）

```python
import numpy as np

MATMUL = _register(OpDefinition(
    name="matmul",
    category="compute",
    op_class="matmul",
    inputs={"A": "Tensor[M,K]", "B": "Tensor[K,N]"},
    output="Tensor[M,N]",
    computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
    shape_rule=ShapeRule(
        input_dims={"A": ["M","K"], "B": ["K","N"]},
        output_dims=["M","N"],
        constraints=["A.shape[-1] == B.shape[0]"],
    ),
    dtype_constraints=DtypeConstraint(
        inputs={"A": ["f16","bf16","f32"], "B": ["f16","bf16","f32"]},
        output_follows="A",
    ),
    template_name="matmul.py.j2",
    template_priority=10,
    numpy_fn=lambda inp, _attrs: np.matmul(inp["A"], inp["B"]),
    can_fuse_as="prologue",
    numpy_ref="np.matmul(A, B)",
))

LAYERNORM = _register(OpDefinition(
    name="layernorm",
    category="reduce",
    op_class="norm",
    inputs={"X": "Tensor[M,N]", "W": "Tensor[N]", "B": "Tensor[N]"},
    output="Tensor[M,N]",
    computation="Y[i,j] = (X[i,j] - mean) / sqrt(var + eps) * W[j] + B[j]",
    shape_rule=ShapeRule(
        input_dims={"X": ["M","N"], "W": ["N"], "B": ["N"]},
        output_dims=["M","N"],
    ),
    attr_specs=(
        AttrSpec(name="eps", dtype=float, default=1e-5,
                 valid_range=(1e-10, 1.0), description="数值稳定性 epsilon"),
    ),
    template_name="norm.py.j2",
    template_priority=5,
    numpy_fn=lambda inp, attrs: (
        (inp["X"] - inp["X"].mean(-1, keepdims=True)) /
        np.sqrt(inp["X"].var(-1, keepdims=True) + attrs.get("eps", 1e-5)) *
        inp["W"] + inp["B"]
    ),
))

FLASH_ATTENTION = _register(OpDefinition(
    name="flash_attention",
    category="attention",
    op_class="attention_mha",
    inputs={"Q": "Tensor[B,H,S,D]", "K": "Tensor[B,H,S_k,D]", "V": "Tensor[B,H,S_k,D]"},
    output="Tensor[B,H,S,D]",
    computation="O = softmax(Q @ K^T / sqrt(D)) @ V  (online softmax, tiled)",
    shape_rule=ShapeRule(
        input_dims={"Q": ["B","H","S","D"], "K": ["B","H","S_k","D"], "V": ["B","H","S_k","D"]},
        output_dims=["B","H","S","D"],
        constraints=["Q.shape[-1] == K.shape[-1]", "K.shape[-2] == V.shape[-2]"],
    ),
    attr_specs=(
        AttrSpec(name="causal", dtype=bool, default=True),
        AttrSpec(name="scale", dtype=float, default=None, description="None → 1/sqrt(D)"),
        AttrSpec(name="dropout_p", dtype=float, default=0.0, valid_range=(0.0, 1.0)),
    ),
    dtype_constraints=DtypeConstraint(
        inputs={"Q": ["f16","bf16"], "K": ["f16","bf16"], "V": ["f16","bf16"]},
        output_follows="Q",
    ),
    template_name="flash_attention.py.j2",
    template_priority=100,    # attention 类最高优先级
    numpy_fn=_ref_flash_attention,
))

ROPE = _register(OpDefinition(
    name="rope",
    category="compute",
    op_class="positional",
    inputs={"X": "Tensor[B,S,H,D]"},
    output="Tensor[B,S,H,D]",
    computation="X_rot[..., ::2] = X * cos - X_shifted * sin; X_rot[..., 1::2] = X * sin + X_shifted * cos",
    shape_rule=ShapeRule(
        input_dims={"X": ["B","S","H","D"]},
        output_dims=["B","S","H","D"],
    ),
    attr_specs=(
        AttrSpec(name="theta", dtype=float, default=10000.0,
                 description="RoPE base frequency"),
        AttrSpec(name="rotary_dim", dtype=int, default=None,
                 description="旋转的维度数，None 表示全部"),
    ),
    template_name="rope.py.j2",
    template_priority=50,
    numpy_fn=_ref_rope,
))

SWIGLU = _register(OpDefinition(
    name="swiglu",
    category="elementwise",
    op_class="gated_activation",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M,N//2]",
    computation="gate, up = X.split(N//2, dim=-1); Y = silu(gate) * up",
    shape_rule=ShapeRule(
        input_dims={"X": ["M","N"]},
        output_dims=["M","N//2"],
        constraints=["X.shape[-1] % 2 == 0"],
    ),
    template_name="gated_activation.py.j2",
    template_priority=60,
    numpy_fn=lambda inp, _: _ref_swiglu(inp),
))
```

#### 2.2.3 shape_inference.py → 从 OpDef.shape_rule 派生

**重构后实现**（消除所有 if/elif 和 op 分组集合）：

```python
# arke/ir/shape_inference.py — v1.1 重构版

from arke.ir.ops.catalog import OP_CATALOG, ShapeRule


def infer_output_shape(op_name: str, input_shapes: dict[str, list[int]]) -> list[int]:
    """从 OpDef.shape_rule 声明式推导 output shape。

    v1.1 重构：不再有 if/elif 分支，所有 shape 逻辑来自 catalog.py 的 shape_rule。
    向后兼容：无 shape_rule 的 op fallback 到 passthrough（elementwise unary）。
    """
    op_def = OP_CATALOG.get(op_name)
    if op_def is None:
        raise ValueError(f"Unknown op: {op_name}")

    input_shapes = _normalize_inputs(input_shapes)   # 保留别名规范化逻辑

    if op_def.shape_rule is None:
        return list(next(iter(input_shapes.values())))

    return _eval_shape_rule(op_def.shape_rule, input_shapes)


def _eval_shape_rule(rule: ShapeRule, input_shapes: dict[str, list[int]]) -> list[int]:
    """Symbolic dimension binding → output shape evaluation."""
    bindings: dict[str, int] = {}

    for param, dims in rule.input_dims.items():
        if param not in input_shapes:
            continue
        shape = input_shapes[param]
        for i, sym in enumerate(dims):
            if i >= len(shape):
                continue
            if sym in bindings and bindings[sym] != shape[i]:
                raise ValueError(
                    f"Shape mismatch: dim '{sym}'={bindings[sym]} vs {shape[i]} "
                    f"(param={param})"
                )
            bindings[sym] = shape[i]

    output_shape = []
    for sym in rule.output_dims:
        if sym in bindings:
            output_shape.append(bindings[sym])
        elif "//" in sym:
            base, divisor = sym.split("//")
            output_shape.append(bindings[base.strip()] // int(divisor.strip()))
        else:
            raise ValueError(f"Unbound symbolic dimension: '{sym}'")
    return output_shape
```

**新增 shape 约束验证函数**（编译期调用）：

```python
def validate_input_shapes(op_name: str, input_shapes: dict[str, list[int]]) -> list[str]:
    """验证 input shapes 是否满足 OpDef 声明的约束。返回 error 列表，空 = OK。"""
    op_def = OP_CATALOG.get(op_name)
    if not op_def or not op_def.shape_rule:
        return []

    errors = []
    # 简单表达式求值：支持 "A.shape[-1] == B.shape[0]" 形式
    for constraint in op_def.shape_rule.constraints:
        try:
            local = {k: type("S", (), {"shape": v})() for k, v in input_shapes.items()}
            if not eval(constraint, {}, local):  # noqa: S307
                errors.append(f"Shape constraint failed: {constraint}")
        except Exception as e:
            errors.append(f"Shape constraint eval error: {constraint}: {e}")
    return errors
```

#### 2.2.4 numerical_check.py → SemanticIR Interpreter

新增文件 `arke/engine/ir_interpreter.py`，`NumericalValidator` 重构为调用 Interpreter：

```python
# arke/engine/ir_interpreter.py — 新文件

import inspect
import numpy as np
from arke.ir.semantic import SemanticIR, ParamRef, NodeRef
from arke.ir.ops.catalog import OP_CATALOG


class SemanticIRInterpreter:
    """Eager 执行 SemanticIR DAG，获得数学参考输出。

    设计目标：
    - 替代 numerical_check.py 里 45 个 _numpy_<op> 函数
    - 为 MLIR 后端验证提供 golden reference
    - 支持 fused kernel 的端到端验证（multi-node DAG）

    执行引擎选择：
    - numpy（默认）：精度高，简单算子首选
    - torch_cpu：支持 bf16、复杂 attention 算子
    """

    def __init__(self, backend: str = "numpy"):
        self.backend = backend

    def run(self, ir: SemanticIR, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """执行整个 SemanticIR DAG，返回最终 output。

        Args:
            ir: 要执行的 SemanticIR（nodes 保证拓扑序）
            inputs: param_name → numpy array

        Returns:
            最终 output numpy array
        """
        node_outputs: dict[str, np.ndarray] = {}

        for node in ir.nodes:
            op_def = OP_CATALOG.get(node.op)
            if op_def is None:
                raise ValueError(f"SemanticIR references unknown op: {node.op}")
            if op_def.numpy_fn is None:
                raise ValueError(
                    f"Op '{node.op}' has no numpy_fn in catalog. "
                    f"Add numpy_fn to OpDefinition in catalog.py."
                )

            # 解析 inputs（ParamRef → params, NodeRef → 中间结果）
            node_inputs = {}
            for port_name, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    node_inputs[port_name] = inputs[ref.name]
                elif isinstance(ref, NodeRef):
                    node_inputs[port_name] = node_outputs[ref.id]

            # 调用 numpy_fn
            # 支持两种签名：fn(inputs) 和 fn(inputs, attrs)
            try:
                sig = inspect.signature(op_def.numpy_fn)
                nparams = len(sig.parameters)
                if nparams >= 2:
                    result = op_def.numpy_fn(node_inputs, node.attrs)
                else:
                    result = op_def.numpy_fn(node_inputs)
            except Exception as e:
                raise RuntimeError(
                    f"IR Interpreter failed at node '{node.id}' (op={node.op}): {e}"
                ) from e

            node_outputs[node.id] = result

        # 返回最后一个节点的输出
        return node_outputs[ir.nodes[-1].id]
```

`NumericalValidator` 重构（`numerical_check.py` 大幅简化）：

```python
# arke/engine/numerical_check.py — v1.1 重构版

class NumericalValidator:
    """数值验证器 v1.1：通过 SemanticIRInterpreter 替代手写 numpy 函数。"""

    def __init__(self):
        self._interpreter = SemanticIRInterpreter(backend="numpy")

    def validate(self, ir: SemanticIR, gpu_output: np.ndarray,
                 inputs: dict[str, np.ndarray]) -> ValidationResult:
        """验证 GPU kernel 输出是否与 IR 语义一致。"""
        reference = self._interpreter.run(ir, inputs)
        tol = _get_tolerance(ir.output.dtype)
        passed = np.allclose(
            gpu_output.astype(np.float64),
            reference.astype(np.float64),
            **tol
        )
        max_err = float(np.max(np.abs(
            gpu_output.astype(np.float64) - reference.astype(np.float64)
        )))
        return ValidationResult(passed=passed, max_absolute_error=max_err,
                                tolerance=tol, reference_source="ir_interpreter")

    # validate_op(), validate_graph() 等公共 API 保持不变
    # 内部实现从 OP_NUMPY_FNS dict 切换到 SemanticIRInterpreter
```

#### 2.2.5 `KernelCache._build_ir()` → 从 `.ak` / OpDef 自动构建

当前 `kernel_cache.py` 的 `_build_ir()` 本质上是把 `OpDef.inputs` / `OpDef.output` 再手写一遍。重构后引入两个路径：

1. **主路径：`.ak` 驱动** — KernelCache 直接接收 `.ak` 文件或已缓存的 AST/Program，调用现有 parser + converter 生成 `SemanticIR`
2. **后备路径：OpDef 自动建 IR** — 对单 op、benchmark、runtime quick path，使用 `AutoIRBuilder` 从 `OpDef` 自动生成单节点 `SemanticIR`

新增文件 `arke/ir/auto_builder.py`：

```python
# arke/ir/auto_builder.py — 新文件

import re
from arke.ir.builder import KernelBuilder
from arke.ir.ops.catalog import OP_CATALOG
from arke.ir.strategy import StrategyIR
from arke.ir.shape_inference import infer_output_shape


class AutoIRBuilder:
    """从 OpDef + symbolic dim 绑定自动构建单-op SemanticIR。"""

    def build(self, op_name: str, shape_params: dict[str, int],
              dtype: str = "f16", attrs: dict | None = None):
        op_def = OP_CATALOG[op_name]
        kernel_id = self._make_kernel_id(op_name, shape_params)
        b = KernelBuilder(kernel_id)

        # 1. 注册输入参数
        input_shapes = {}
        for param_name, tensor_spec in op_def.inputs.items():
            shape = self._resolve_shape(tensor_spec, shape_params)
            b.param(param_name, shape, dtype)
            input_shapes[param_name] = shape

        # 2. 添加单个 op 节点
        input_refs = {name: name for name in op_def.inputs.keys()}
        node = b.op(op_name, **input_refs, **(attrs or {}))

        # 3. 推导输出 shape
        output_shape = infer_output_shape(op_name, input_shapes)
        b.returns(node, output_shape, dtype)

        return b.build(), StrategyIR(kernel_id=kernel_id)

    def _resolve_shape(self, tensor_spec: str, params: dict[str, int]) -> list[int]:
        # 解析如 "Tensor[M,K]" / "Tensor[B,H,S,D]"
        dims = re.search(r"Tensor\[(.*?)\]", tensor_spec).group(1).split(",")
        out = []
        for d in dims:
            d = d.strip()
            if "//" in d:
                base, divisor = d.split("//")
                out.append(params[base.strip()] // int(divisor.strip()))
            else:
                out.append(params[d])
        return out

    def _make_kernel_id(self, op_name: str, params: dict[str, int]) -> str:
        suffix = "_".join(str(v) for _, v in sorted(params.items()))
        return f"{op_name}_{suffix}"
```

`kernel_cache.py` 重构后的核心逻辑：

```python
class KernelCache:
    def __init__(self):
        self._backend = TritonBackend()
        self._compiler = TritonCompiler()
        self._auto_builder = AutoIRBuilder()
        self._program_cache: dict[str, SemanticIR] = {}
        self._generic_cache: dict[tuple, Callable] = {}

    def compile_op(self, op: str, **shape_params) -> Callable | None:
        """兼容旧 API：对 benchmark / quick-path 仍允许 compile_op(op, M=..., N=...)."""
        try:
            ir, strategy = self._auto_builder.build(op, shape_params)
            source = self._backend.translate(ir, strategy)
            compiled = self._compiler.compile(source)
            if not compiled.success:
                return None
            module = self._compiler._import_module(compiled.binary_path)
            return self._compiler._find_entry_function(module)
        except Exception:
            return None

    def compile_ak(self, ak_path: str, target_hw: str = "nvidia_ampere") -> Callable | None:
        """主路径：从 .ak 文件直接构建 SemanticIR，不再 per-op 分支。"""
        from arke.parser.parser import parse_file
        from arke.parser.converter import ast_to_ir
        from arke.compiler.default_strategy import DefaultStrategyGenerator

        program = parse_file(ak_path)
        ir = ast_to_ir(program)
        strategy = DefaultStrategyGenerator().generate(ir, target_hw=target_hw)
        source = self._backend.translate(ir, strategy)
        compiled = self._compiler.compile(source)
        if not compiled.success:
            return None
        module = self._compiler._import_module(compiled.binary_path)
        return self._compiler._find_entry_function(module)
```

**关键变化：** `_build_ir()` 不再有 45 个 `if op == ...` 分支。benchmark 场景走 `AutoIRBuilder`，真实编译路径走 `.ak → AST → SemanticIR`。

#### 2.2.6 `template_engine._select_template()` → 从 OpDef.category / template_name / priority 路由

当前 `_select_template()` 的 if/elif 链完全可以由 `OpDef.template_name` 和 `template_priority` 取代。

```python
# arke/backend/triton_template_engine.py — 重构后的模板选择

from arke.ir.ops.catalog import OP_CATALOG


def _select_template(self, semantic: SemanticIR) -> tuple[str, str]:
    """根据 SemanticIR 中各 node 的 OpDef.template_name / template_priority 选择模板。

    规则：
      1. 取 semantic.nodes 中 template_priority 最大的 node 作为 primary op
      2. 返回该 op_def.template_name
      3. 如果无 template_name，则按 category fallback（elementwise→elementwise.py.j2 等）
    """
    op_defs = []
    for node in semantic.nodes:
        op_def = OP_CATALOG[node.op]
        op_defs.append((node.op, op_def))

    primary_op, primary_def = max(
        op_defs,
        key=lambda item: item[1].template_priority,
    )

    if primary_def.template_name:
        return primary_def.template_name, primary_op

    # category fallback
    category_fallback = {
        "elementwise": "elementwise.py.j2",
        "reduce": "reduction.py.j2",
        "compute": "matmul.py.j2",
        "attention": "flash_attention.py.j2",
        "move": "data_movement.py.j2",
        "quant": "quantize.py.j2",
    }
    return category_fallback[primary_def.category], primary_op
```

这让模板优先级从“代码顺序里的隐式知识”变成“OpDef 上的显式声明”。

---

### 2.3 Arke Lang Spec v1.1 变更提案

目标：让 `.ak` 不只是“调用算子”，而是能**声明 op 的约束与属性 schema**，成为可生成 `OpDef` 的源语言。

#### 2.3.1 新增 `op` 声明块（可选）

当前 v1.0 的 `.ak` 只有 `kernel` 和 `strategy`。v1.1 新增可选的 `op` 声明，用于定义 / 覆盖 OpDef 元信息：

```ak
op layernorm {
    category = "reduce";
    class = "norm";

    inputs {
        X: Tensor<[M, N], f16 | bf16 | f32>;
        W: Tensor<[N], f16 | bf16 | f32>;
        B: Tensor<[N], f16 | bf16 | f32>;
    }

    attrs {
        eps: float = 1e-5 where eps > 0.0;
    }

    shape_rule {
        output = Tensor<[M, N], same_as(X)>;
    }

    template {
        name = "norm.py.j2";
        priority = 5;
    }
}
```

**设计意图：**
- 对内建 op，`op` 声明块可以省略，编译器使用内置 catalog
- 对扩展 op，`op` 声明块可直接生成新的 `OpDef`
- 对已有 op，`op` 声明块可作为 overlay，覆盖 `template_name`、`attrs.default` 等字段

#### 2.3.2 `Tensor` 类型上的 dtype union 与 shape symbol

当前 v1.0：

```ak
X: Tensor<[128, 768], f16>
```

v1.1 扩展为：

```ak
X: Tensor<[M, N], f16 | bf16 | f32>
W: Tensor<[N], same_as(X)>
```

新增语义：
- `M`, `N`, `B`, `S`, `D` 允许作为 shape symbol 出现在 type 中
- `f16 | bf16 | f32` 表示 dtype constraints
- `same_as(X)` 表示 dtype 跟随另一个输入

#### 2.3.3 kernel 中的属性声明语法

当前 v1.0：

```ak
let Y = layernorm(X=X, W=W, B=B, eps=1e-5);
```

v1.1 保持兼容，但补充属性 schema 检查：

```ak
kernel layernorm_k(
    X: Tensor<[M, N], f16>,
    W: Tensor<[N], f16>,
    B: Tensor<[N], f16>
) -> Tensor<[M, N], f16> {
    let Y = layernorm(X=X, W=W, B=B, eps=1e-5);
    return Y;
}
```

新增约束：
- `eps` 必须在 `AttrSpec` 中声明，否则 parser 发 warning / validator 报错
- `eps` 类型必须匹配 `AttrSpec.dtype=float`
- 未传的属性自动填充默认值

#### 2.3.4 新增 where 约束块

用于声明 shape / dtype / attr 约束：

```ak
kernel swiglu_k(X: Tensor<[M, N], f16>) -> Tensor<[M, N/2], f16>
where N % 2 == 0 {
    let Y = swiglu(X=X);
    return Y;
}
```

以及 attention：

```ak
kernel fa_k(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S_k, D], f16>,
    V: Tensor<[B, H, S_k, D], f16>
) -> Tensor<[B, H, S, D], f16>
where D % 16 == 0,
      K.shape[2] == V.shape[2] {
    let O = flash_attention(Q=Q, K=K, V=V, causal=true);
    return O;
}
```

#### 2.3.5 Arke Lang Spec v1.1 的 EBNF 增量

```ebnf
op_def          = "op" IDENT "{" op_body "}"
op_body         = category_decl class_decl? inputs_decl attrs_decl? shape_rule_decl? template_decl?
category_decl   = "category" "=" STRING ";"
class_decl      = "class" "=" STRING ";"
inputs_decl     = "inputs" "{" input_decl* "}"
input_decl      = IDENT ":" tensor_type ";"
attrs_decl      = "attrs" "{" attr_decl* "}"
attr_decl       = IDENT ":" scalar_type ("=" arg_value)? ("where" expr)? ";"
shape_rule_decl = "shape_rule" "{" "output" "=" tensor_type ";" "}"
template_decl   = "template" "{" "name" "=" STRING ";" "priority" "=" INT ";" "}"
where_clause    = "where" expr ("," expr)*

tensor_type     = "Tensor" "<" "[" dim_expr_list "]" "," dtype_expr ">"
dim_expr_list   = dim_expr ("," dim_expr)*
dim_expr        = IDENT | INT | IDENT "/" INT | IDENT "%" INT

dtype_expr      = scalar_type ("|" scalar_type)* | "same_as" "(" IDENT ")"
```

#### 2.3.6 向后兼容策略

- v1.0 的 `.ak` 文件**无需修改**，继续有效
- `op` 声明块是可选的；无 `op` 声明时，编译器使用内置 `OP_CATALOG`
- `where` 子句是可选的；无 `where` 子句时，不增加任何约束
- 旧 parser 读取 v1.1 文件可报“不支持 op_def / where_clause”——这是版本边界，需通过 `version` 或 CLI flag 区分

---

### 2.4 Arke IR Spec v1.1 变更提案

#### 2.4.1 `SemanticIR.Node` 增强：属性系统从 `dict` 升级为 schema-aware

当前 v1.0：

```python
@dataclass
class Node:
    id: str
    op: str
    inputs: dict[str, InputRef]
    output: TensorDesc
    semantics: Semantics
    attrs: dict
```

v1.1：

```python
@dataclass
class AttrValue:
    name: str
    value: Any
    inferred: bool = False   # True = 由默认值自动填充


@dataclass
class Node:
    id: str
    op: str
    inputs: dict[str, InputRef]
    output: TensorDesc
    semantics: Semantics
    attrs: dict[str, AttrValue]   # 从 Any→AttrValue，保留 value + inferred 元信息
    op_version: str = "1.1"
```

**兼容性：** `from_dict()` 继续接受旧格式 `attrs: dict[str, Any]`，自动提升为 `AttrValue(value=old_value, inferred=False)`。

#### 2.4.2 `SemanticIR` 顶层新增 `op_registry_version`

```python
@dataclass
class SemanticIR:
    version: str
    kernel_id: str
    params: list[Param]
    nodes: list[Node]
    edges: list[Edge]
    output: TensorDesc
    fusion_groups: list[FusionGroup]
    metadata: dict
    op_registry_version: str = "1.1"   # NEW
```

用于标记本 IR 是基于哪个版本的 `OpDef` schema 构建的。这样同一个 `.ak` 文件在不同 catalog 版本下的行为可追踪。

#### 2.4.3 `InterpreterDispatch` 接口

IR Spec v1.1 新增标准接口，供 `SemanticIRInterpreter`、未来的 `MLIRInterpreter`、甚至 hardware simulator 复用：

```python
class InterpreterDispatch(Protocol):
    def has_impl(self, op_name: str) -> bool: ...
    def run(self, op_name: str, inputs: dict[str, Any], attrs: dict[str, Any]) -> Any: ...
```

`SemanticIRInterpreter` 的调度逻辑：
- 默认 dispatch = `CatalogDispatch`，从 `OpDef.numpy_fn` 查实现
- 未来可替换为 `TorchDispatch`、`ReferenceMLIRDispatch`

#### 2.4.4 `OpDef` 正式进入 IR Spec，而不再只是 catalog.py 的实现细节

目前 IR Spec 只在 §4 粗略描述 `OpDef`：`name`, `category`, `inputs`, `output`, `description`, `properties`, `fusable_epilogues`。v1.1 需要把 `shape_rule`, `attr_specs`, `dtype_constraints`, `template_name`, `template_priority` 纳入规范正文。

建议在 `docs/spec/arke-ir-spec-v1.md` 增加新小节：

- **§4.1 OpDefinition v1.1 schema**
- **§4.2 ShapeRule**
- **§4.3 AttrSpec**
- **§4.4 DtypeConstraint**
- **§4.5 InterpreterDispatch**

---

### 2.5 编译器实现拆解

#### 2.5.1 需要修改 / 新增的文件

| 文件 | 动作 | 修改内容 | 估时 |
|:-----|:-----|:---------|:----:|
| `arke/ir/ops/catalog.py` | **重构** | `OpDefinition` 扩展 `shape_rule`, `attr_specs`, `dtype_constraints`, `template_name`, `template_priority`, `numpy_fn`；补全 45 ops | 3-4 天 |
| `arke/ir/shape_inference.py` | **重写** | 删除 op 分组集合和 if/elif 树，改为从 `OpDef.shape_rule` 派生 | 1-2 天 |
| `arke/engine/numerical_check.py` | **重构** | 删除 45 个 `_numpy_<op>`，接入 `SemanticIRInterpreter` | 2-3 天 |
| `arke/engine/ir_interpreter.py` | **新增** | 实现 `SemanticIRInterpreter` | 1-2 天 |
| `arke/ir/auto_builder.py` | **新增** | 从 OpDef 自动构建单 op `SemanticIR` | 1 天 |
| `arke/integration/kernel_cache.py` | **重构| `arke/integration/kernel_cache.py` | **重构** | 删除 `_build_ir()` 的 per-op 分支，主路径改为 `.ak → AST → SemanticIR`，quick path 走 `AutoIRBuilder` | 2-3 天 |
| `arke/backend/triton_template_engine.py` | **重构** | `_select_template()` 改为 `OpDef.template_name/priority` 路由 | 1 天 |
| `arke/ir/semantic.py` | **重构** | `Node.attrs` 升级为 schema-aware 结构；兼容旧 dict 格式 | 1-2 天 |
| `arke/parser/arke.lark` | **扩展** | 新增 `op` 声明、`where` 子句、dtype union 语法 | 2 天 |
| `arke/parser/parser.py` | **扩展** | 解析新语法，生成 op/constraint AST | 1-2 天 |
| `arke/parser/converter.py` | **扩展** | 从 AST 生成 attr-validated `SemanticIR` | 2 天 |
| `docs/spec/arke-lang-spec-v1.md` | **更新到 v1.1** | 新增 `op` 声明、where、dtype union | 1 天 |
| `docs/spec/arke-ir-spec-v1.md` | **更新到 v1.1** | 正式纳入 `OpDef v1.1`, `ShapeRule`, `AttrSpec`, `InterpreterDispatch` | 1 天 |
| `tests/test_shape_inference.py` | **重写** | 改为基于 catalog 的生成式测试 | 1 天 |
| `tests/test_ir_interpreter.py` | **新增** | 验证 `SemanticIRInterpreter` 对 45 ops 的覆盖 | 1-2 天 |
| `tests/test_catalog_consistency.py` | **新增** | 检查 `OpDef.shape_rule`, `numpy_fn`, `template_name` 完整性 | 1 天 |

**总工作量：** 约 3-4 周（单人，包含测试与文档同步）。

#### 2.5.2 实施顺序（建议）

**Phase A — schema 先行（1 周）**
1. 扩展 `OpDefinition` 数据结构
2. 为 45 ops 补齐 `shape_rule`, `attr_specs`, `template_name`
3. 新增 `tests/test_catalog_consistency.py`

**Phase B — 推导替换（1 周）**
1. 重写 `shape_inference.py`
2. 实现 `AutoIRBuilder`
3. 重构 `template_engine.py`

**Phase C — Interpreter 替换（1 周）**
1. 实现 `SemanticIRInterpreter`
2. 重构 `numerical_check.py`
3. 删除 45 个 `_numpy_<op>` 函数

**Phase D — 语言 / IR 升级（1 周）**
1. parser / converter 支持 `op` 块与 `where`
2. `semantic.py` 升级 attrs schema
3. 文档同步到 v1.1

#### 2.5.3 风险点

| 风险 | 描述 | 缓解 |
|:-----|:-----|:-----|
| `eval(constraint)` 安全性 | shape rule 约束使用 Python 表达式求值 | 仅在受信任的编译器内部使用；后续可替换为受限表达式解释器 |
| attention / quant 的 `numpy_fn` 复杂 | `numpy` 不一定适合复杂参考实现 | 对复杂 op 使用 PyTorch eager dispatch；`numpy_fn` 可接受 torch tensor |
| v1.0 attrs 兼容 | 旧 JSON round-trip 的 `attrs: dict[str, Any]` 需要兼容 | `from_dict()` 做自动提升，`to_dict()` 默认输出 v1.1，必要时提供 legacy mode |

---

### 2.6 G6 gate 验证标准保持不变

这次重构**不改变 G6 exit criteria**，只改变实现架构。验证要求：

1. **L1 BL5 correctness/performance** 不下降
2. **L2 BL5** 至少维持当前 3/4 fusion pass
3. **G6-LI.1~LI.6** 全部继续通过
4. **新增架构性验证**：
   - `shape_inference.py` 不再包含 op 分组常量（`_ELEMENTWISE_*`, `_ATTENTION_OPS`）
   - `numerical_check.py` 不再存在 45 个 `_numpy_<op>` 函数
   - `kernel_cache.py` 不再包含 45 op 的 `_build_ir` if/elif 链
   - `triton_template_engine.py` 的 `_select_template()` 不再以 op 名 if/elif 链路由
   - `tests/test_catalog_consistency.py` 保证 45 ops 的 `shape_rule`, `template_name`, `numpy_fn` 非空

换句话说，**G6 的功能标准不变，但实现标准被补齐**：
"能跑"不再等于"架构合格"，必须达到 IR-Driven。

---

## 5. 方案优先级与实施路径

### 5.1 优先级排序

基于 Stage 1 → Stage 2/3 的演进路径和 G6 gate 的紧迫性：

| 优先级 | 方案 | 理由 | 时间 |
|:---:|:-----|:-----|:----:|
| **P0** | 方案一：IR-Driven 架构重构 | G6 gate 的核心要求；为后续方案奠定基础；解决当前最严重的架构问题 | 3-4 周 |
| **P1** | 方案二：多级后端扩展性设计 | Stage 2/3 的必要准备；不阻塞 G6，但必须在 G7 前完成；与方案一可并行 | 1.5-2 周 |
| **P2** | 方案三：动态 Shape 支持设计 | 长期价值高，但不影响 G6/G7 gate；可在 Stage 2 中期启动 | 2-3 周 |

### 5.2 实施时间线

```
Week 1-4 (方案一 Phase A-D)
├─ Week 1: Phase A — OpDef schema 扩展 + 45 ops 补齐
├─ Week 2: Phase B — shape_inference / template_engine 重构
├─ Week 3: Phase C — SemanticIRInterpreter 实现
└─ Week 4: Phase D — parser/converter 升级 + 文档同步

Week 2-3 (方案二，与方案一 Phase B-C 并行)
├─ Week 2: Backend 抽象层 + TritonBackend 适配
├─ Week 3: StrategyIR Level 分层 + MLIRBackend stub

Week 5+ (方案三，G6 通过后启动)
├─ Lang/IR 动态 Shape 语法设计
├─ SemanticIRInterpreter 的 symbolic shape 支持
└─ StrategyIR 的 shape-aware 策略生成
```

### 5.3 G6 gate 验证检查清单

重构完成后，G6 gate 验证需要确认：

- [ ] 所有 45 ops 的 `OpDef` 包含 `shape_rule`, `attr_specs`, `template_name`, `numpy_fn`
- [ ] `shape_inference.py` 无 op 分组常量，所有推导来自 `OpDef.shape_rule`
- [ ] `numerical_check.py` 无 `_numpy_<op>` 函数，全部通过 `SemanticIRInterpreter`
- [ ] `kernel_cache._build_ir()` 无 per-op if/elif，使用 `AutoIRBuilder` 或 `.ak` parser
- [ ] `template_engine._select_template()` 无 op 名 if/elif 链，使用 `OpDef.template_priority`
- [ ] `tests/test_catalog_consistency.py` 通过（检查 OpDef 完整性）
- [ ] L1 BL5 correctness/performance 不下降
- [ ] L2 BL5 fusion ≥3/4 pass
- [ ] G6-LI.1~LI.6 全部通过

---

## 总结

三个方案形成递进关系：

1. **方案一** 是基础：建立 Single Source of Truth，消除 op 知识分散，为编译器架构奠定 IR-Driven 基础。
2. **方案二** 是扩展：在方案一的基础上，为 Stage 2/3 的多后端演进预留接口和抽象层。
3. **方案三** 是深化：在方案一/二的基础上，支持 AI 场景下的动态 shape，提升编译器的实用性。

**G6 gate 的成功标准不仅是"能跑"，而是"架构合格"**。这三个方案共同确保 Arke 从"形式通过"升级到"架构优雅"，为 Stage 2/3 的长期演进奠定坚实基础。

