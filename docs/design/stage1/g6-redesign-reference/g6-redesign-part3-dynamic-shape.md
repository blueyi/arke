# G6 重构方案三：动态 Shape 支持设计

> **文档目的：** 从 AI 模型优化端到端视角，为 Arke 语言、IR、编译工具链和 Agent 工程设计完整的动态 Shape 解决方案。
>
> **版本：** 1.0
> **日期：** 2026-04-06
> **关联：** arke-lang-spec-v1.md、arke-ir-spec-v1.md、stage1-gate-design.md (G7)

---

## 目录

1. [动态 Shape 的问题域分析](#1-动态-shape-的问题域分析)
2. [业界方案对比与 Arke 定位](#2-业界方案对比与-arke-定位)
3. [Arke Lang 动态 Shape 语法设计](#3-arke-lang-动态-shape-语法设计)
4. [Arke IR 动态 Shape 表示](#4-arke-ir-动态-shape-表示)
5. [StrategyIR 的动态 Shape 策略](#5-strategyir-的动态-shape-策略)
6. [编译器 Shape Analysis Pipeline](#6-编译器-shape-analysis-pipeline)
7. [Codegen 和 Runtime](#7-codegen-和-runtime)
8. [Arke Agent 的动态 Shape 工程](#8-arke-agent-的动态-shape-工程)
9. [Arke Lang Spec v1.1 变更](#9-arke-lang-spec-v11-变更)
10. [Arke IR Spec v1.1 变更](#10-arke-ir-spec-v11-变更)
11. [实现拆解与工作量](#11-实现拆解与工作量)
12. [对 G7/G8 的影响](#12-对-g7g8-的影响)

---

## 1. 动态 Shape 的问题域分析

### 1.1 哪些维度是动态的

| 维度 | 典型场景 | 动态范围 | 对编译器的影响 |
|------|---------|---------|--------------|
| **Batch (B)** | 训练 data-parallel、推理 batching | 1–256 | 并行度、tile 沿 batch 维分配 |
| **Seq length (S)** | NLP 推理（prefill vs decode）| 1–131072 | 算法选择（naive vs flash attn）、内存布局、tile size |
| **Num experts (E)** | MoE dispatch | 1–64 | gather/scatter pattern、load balancing |
| **KV cache length** | 自回归生成 | 累积增长 | paged_attention block_table 维度 |
| **Image HW** | Vision models | 可变分辨率 | conv tile 策略 |

**核心洞察：** 在 AI 算子中，**大部分维度可以在编译时确定**（D=head_dim, H=num_heads, vocab_size, hidden_dim 等），真正动态的只有 **2-3 个维度**（B, S, E）。这意味着 Arke 不需要一个通用的符号执行引擎——只需要针对这几个关键维度的参数化编译。

### 1.2 动态 Shape 对编译器各层的影响

```
Parser      : Tensor<[B, 32, S, 128], f16> — shape 不再全是 int literal
SemanticIR  : Param.shape = [Symbol("B"), 32, Symbol("S"), 128]
Shape Infer : 需要符号传播，如 matmul([B,S,D], [D,N]) → [B,S,N]
StrategyIR  : tile_size 不能硬编码 — 需要 min(128, S) 或 autotune
Codegen     : kernel 函数签名需要 shape 参数
Runtime     : 按 shape bucket 选已编译版本，或 JIT 编译新版本
Benchmark   : 一个 kernel 需要在多个 shape 下验证
```

### 1.3 与静态 Shape 的关键区别

| 方面 | 静态 Shape（当前） | 动态 Shape（目标） |
|------|-------------------|-------------------|
| Tile size | 编译时固定 `BLOCK_M=64` | 运行时选择或 autotune |
| Shared memory | 编译时精确计算 | 需要 worst-case 或参数化 |
| Launch config | 固定 grid/block | grid 依赖 input shape |
| 最优策略 | 一个 shape 一个策略 | 一个 shape range 一个策略 |
| 验证 | 跑一个 shape | 跑 shape range 的 corner cases |

---

## 2. 业界方案对比与 Arke 定位

### 2.1 业界方案

| 方案 | 代表 | 机制 | 优点 | 缺点 |
|------|------|------|------|------|
| **Guard + Recompile** | PyTorch dynamo | 记录 symbolic shape guard → shape 变化时 recompile | 对用户透明 | 编译开销大，shape 爆炸 |
| **Symbolic Shape** | TVM Relay, JAX | 符号维度 + 约束求解 | 一次编译 | 复杂约束难处理，debug 困难 |
| **Bucketing/Padding** | XLA, TensorRT | pad to 固定 bucket size | 实现简单 | 浪费计算，bucket 选择需经验 |
| **Autotune** | Triton | 多配置预编译 → 运行时选最优 | 性能好 | 编译时间长，配置空间大 |
| **Torch.compile** | Inductor | symbolic_shapes + dynamic_shapes=True | 工业级 | 不支持自定义 backend 的 symbolic |

### 2.2 Arke 的差异化方案：LLM-Guided Shape-Aware Compilation

Arke 的独特优势：**LLM Agent 理解 shape 语义**。

传统编译器做 shape 分析是机械的规则传播。Arke 的 Agent 可以：
- 理解 "S 是 seq_len，推理时 decode phase S=1 但 prefill S=2048" → **自动生成两套策略**
- 理解 "B=1 时 GPU 利用率低，需要更大的 tile 来补偿" → **shape-aware 策略调整**
- 理解 "flash_attention 在 S<64 时 naive attention 更快" → **自动插入 conditional dispatch**

**Arke 方案 = Symbolic Dims + Shape Constraints + LLM-Guided Strategy + JIT Bucketing**

不是完全的 symbolic 编译（太复杂），也不是纯 bucketing（太粗糙），而是：
1. **声明**哪些维度是动态的、范围是多少
2. **LLM Agent** 为关键 shape ranges 生成策略
3. **Compiler** 把策略编译为参数化 kernel（tile size 是 shape 的函数）
4. **Runtime** 按 shape bucket 选预编译版本，miss 时 JIT

---

## 3. Arke Lang 动态 Shape 语法设计

### 3.1 符号维度声明

```ak
// v1.0 (当前) — 纯静态
kernel matmul(
    A: Tensor<[128, 768], f16>,
    B: Tensor<[768, 3072], f16>
) -> Tensor<[128, 3072], f16> { ... }

// v1.1 (提案) — 支持符号维度
kernel matmul(
    A: Tensor<[M, K], f16>,
    B: Tensor<[K, N], f16>
) -> Tensor<[M, N], f16>
where
    M: dynamic(1..4096),
    K: static(768),
    N: static(3072)
{
    let C = matmul(A=A, B=B);
    return C;
}
```

### 3.2 维度类型

| 类型 | 语法 | 含义 |
|------|------|------|
| **静态** | `128` 或 `D: static(128)` | 编译时确定，当前行为 |
| **动态** | `S: dynamic` | 运行时确定，编译器参数化处理 |
| **有界动态** | `S: dynamic(1..8192)` | 运行时确定，范围已知 |
| **对齐动态** | `S: dynamic(1..8192, align=128)` | 运行时确定，保证 128 对齐 |

**向后兼容：** 不带 `where` clause 的纯 int literal shape → 自动视为 static。所有 v1.0 `.ak` 文件无需修改。

### 3.3 Shape 约束系统

```ak
kernel flash_attention(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S_kv, D], f16>,
    V: Tensor<[B, H, S_kv, D], f16>
) -> Tensor<[B, H, S, D], f16>
where
    B: dynamic(1..64),
    H: static(32),
    S: dynamic(1..131072, align=128),
    S_kv: dynamic(1..131072, align=128),
    D: static(128),
    // 约束
    S_kv >= S,                     // KV cache 至少和 Q 一样长
    B * H * S * D <= 67108864      // 内存预算 (~128MB fp16)
{
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}
```

约束类型：

| 约束 | 语法 | 用途 |
|------|------|------|
| **相等** | `Q.shape[2] == K.shape[2]` 或隐含同名 | 同名符号自动相等 |
| **整除** | `S % 128 == 0` 或 `align=128` | tile 对齐 |
| **范围** | `S: dynamic(1..8192)` | 编译优化范围 |
| **比较** | `S_kv >= S` | 语义约束 |
| **算术** | `B * S * D <= N` | 内存/资源预算 |

### 3.4 Shape 分组（多 shape regime）

```ak
strategy flash_attn_strategy for target("nvidia_ampere") {
    // 不同 shape range 不同策略
    shape_regime(when="S <= 64") {
        algorithm(name="naive_attention")
            @rationale("short seq: naive attention avoids flash overhead");
        launch_config(num_warps=4, num_stages=1);
    }
    shape_regime(when="S > 64") {
        algorithm(name="flash_attention_v2")
            @rationale("long seq: flash attention for O(S) memory");
        tile(loop="S", factors=[128])
            @rationale("128-token blocks for causal masking efficiency");
        launch_config(num_warps=4, num_stages=3);
    }
}
```

---

## 4. Arke IR 动态 Shape 表示

### 4.1 SymbolicDim

```python
@dataclass
class SymbolicDim:
    """A symbolic dimension that may be resolved at runtime."""
    name: str                           # "B", "S", "M"
    kind: str = "dynamic"               # "static" | "dynamic"
    value: int | None = None            # static 时有值
    range: tuple[int, int] | None = None  # dynamic 时的 (min, max)
    alignment: int | None = None        # 对齐约束（128 → S % 128 == 0）

    @property
    def is_static(self) -> bool:
        return self.kind == "static" and self.value is not None

    def to_dict(self) -> dict:
        d = {"name": self.name, "kind": self.kind}
        if self.value is not None:
            d["value"] = self.value
        if self.range is not None:
            d["range"] = list(self.range)
        if self.alignment is not None:
            d["alignment"] = self.alignment
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SymbolicDim:
        return cls(
            name=d["name"],
            kind=d.get("kind", "dynamic"),
            value=d.get("value"),
            range=tuple(d["range"]) if "range" in d else None,
            alignment=d.get("alignment"),
        )

# 统一的 Dim 类型
Dim = int | SymbolicDim
```

### 4.2 Param / TensorDesc 扩展

```python
@dataclass
class Param:
    name: str
    shape: list[Dim]           # 从 list[int] 扩展为 list[int | SymbolicDim]
    dtype: str
    layout: str = "row_major"

    @property
    def is_static(self) -> bool:
        """All dims are static ints."""
        return all(isinstance(d, int) for d in self.shape)

    @property
    def symbolic_dims(self) -> list[SymbolicDim]:
        """Return only the symbolic dimensions."""
        return [d for d in self.shape if isinstance(d, SymbolicDim)]

    def specialize(self, bindings: dict[str, int]) -> Param:
        """Bind symbolic dims to concrete values."""
        new_shape = []
        for d in self.shape:
            if isinstance(d, SymbolicDim):
                if d.name in bindings:
                    new_shape.append(bindings[d.name])
                else:
                    new_shape.append(d)
            else:
                new_shape.append(d)
        return Param(name=self.name, shape=new_shape, dtype=self.dtype, layout=self.layout)
```

### 4.3 ShapeConstraint IR

```python
@dataclass
class ShapeConstraint:
    """A constraint on symbolic dimensions."""
    kind: str           # "equal" | "divisible" | "range" | "comparison" | "budget"
    expr: str           # 人类可读表达式，如 "S % 128 == 0"
    lhs: str            # 左值，如 "S"
    op: str             # "==" | ">=" | "<=" | "%" | "*"
    rhs: str | int      # 右值，如 "128" 或 "K.shape[2]"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "expr": self.expr, "lhs": self.lhs, "op": self.op, "rhs": self.rhs}
```

### 4.4 SemanticIR 扩展

```python
@dataclass
class SemanticIR:
    version: str = "0.4.0"           # 从 0.3.0 → 0.4.0
    kernel_id: str = ""
    params: list[Param] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    output: TensorDesc = field(default_factory=TensorDesc)
    fusion_groups: list[FusionGroup] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # 新增
    symbolic_dims: list[SymbolicDim] = field(default_factory=list)  # 所有符号维度定义
    constraints: list[ShapeConstraint] = field(default_factory=list)  # shape 约束

    @property
    def is_static(self) -> bool:
        """No symbolic dimensions."""
        return len(self.symbolic_dims) == 0

    def specialize(self, bindings: dict[str, int]) -> SemanticIR:
        """Bind all symbolic dims → produce a fully static SemanticIR."""
        # 深拷贝 + 替换所有 SymbolicDim
        specialized = copy.deepcopy(self)
        specialized.params = [p.specialize(bindings) for p in specialized.params]
        specialized.symbolic_dims = []
        specialized.constraints = []
        # 更新 node output shapes
        for node in specialized.nodes:
            node.output = self._specialize_tensor_desc(node.output, bindings)
        specialized.output = self._specialize_tensor_desc(specialized.output, bindings)
        return specialized
```

### 4.5 JSON 序列化

```json
{
  "version": "0.4.0",
  "kernel_id": "flash_attention",
  "symbolic_dims": [
    {"name": "B", "kind": "dynamic", "range": [1, 64]},
    {"name": "S", "kind": "dynamic", "range": [1, 131072], "alignment": 128},
    {"name": "H", "kind": "static", "value": 32},
    {"name": "D", "kind": "static", "value": 128}
  ],
  "constraints": [
    {"kind": "comparison", "expr": "S_kv >= S", "lhs": "S_kv", "op": ">=", "rhs": "S"}
  ],
  "params": [
    {"name": "Q", "shape": [{"ref": "B"}, {"ref": "H"}, {"ref": "S"}, {"ref": "D"}], "dtype": "f16"}
  ]
}
```

向后兼容：如果 `symbolic_dims` 为空且 shape 都是 int，等价于 v0.3.0。

---

## 5. StrategyIR 的动态 Shape 策略

### 5.1 参数化 Decision

当前 Decision 的 params 是硬编码值：
```python
Decision(kind="tile", params={"loop": "M", "factors": [64]})
```

扩展为支持**符号表达式**和**条件**：

```python
@dataclass
class SymbolicExpr:
    """A symbolic expression over shape dimensions."""
    expr: str       # "min(128, next_pow2(S))"
    fallback: Any   # 符号求解失败时的默认值

Decision(kind="tile", params={
    "loop": "S",
    "factors": SymbolicExpr("min(128, next_pow2(S))", fallback=128),
})
```

### 5.2 Conditional Strategy（shape_regime）

```python
@dataclass
class ConditionalDecision(Decision):
    """条件策略 — 不同 shape range 不同 decisions."""
    kind: str = "conditional"
    params: dict = field(default_factory=lambda: {
        "condition": "S <= 64",
        "then_decisions": [...],    # list[Decision]
        "else_decisions": [...],    # list[Decision]
    })
```

**Lowering 规则：**
- Triton backend: 生成两个 kernel + runtime dispatch
- MLIR backend: `scf.if` 嵌入
- LLVM backend: branch instruction

### 5.3 Autotune Config 生成

LLM Agent 为动态 shape 生成覆盖 range 的 autotune configs：

```python
Decision(kind="autotune", params={
    "configs": [
        # 小 shape：少 warp，小 tile
        {"num_warps": 2, "num_stages": 2, "block_sizes": {"BLOCK_M": 32, "BLOCK_N": 64}},
        # 中 shape：标准配置
        {"num_warps": 4, "num_stages": 3, "block_sizes": {"BLOCK_M": 64, "BLOCK_N": 128}},
        # 大 shape：大 tile，多 pipeline stage
        {"num_warps": 8, "num_stages": 4, "block_sizes": {"BLOCK_M": 128, "BLOCK_N": 128}},
    ],
    "key": ["M", "N"],  # autotune 根据哪些维度选择
})
```

### 5.4 Strategy 对动态 Shape 的完整流程

```
.ak (with where clause)
  │
  ▼
SemanticIR (with symbolic_dims + constraints)
  │
  ├── Agent 分析 shape range → 决定策略类型
  │   ├── 单一 range → 参数化 tile + autotune
  │   └── 跨 regime → conditional strategy
  │
  ▼
StrategyIR (with conditional decisions + autotune)
  │
  ▼
Codegen → 参数化 Triton kernel（shape 是 tl.constexpr 参数）
  │
  ▼
Runtime → shape bucket → 选预编译版本
```

---

## 6. 编译器 Shape Analysis Pipeline

### 6.1 Symbolic Shape Propagation

```python
# arke/ir/shape_propagation.py (新文件)
class SymbolicShapeEngine:
    """符号 shape 传播引擎"""

    def propagate(self, semantic_ir: SemanticIR) -> SemanticIR:
        """为每个 node 推导符号 output shape。"""
        dim_env: dict[str, SymbolicDim] = {
            d.name: d for d in semantic_ir.symbolic_dims
        }

        for node in topo_sort(semantic_ir.nodes):
            input_shapes = [self._resolve_shape(ref, semantic_ir) for ref in node.inputs.values()]
            node.output.shape = self._infer_symbolic(node.op, input_shapes, dim_env)

        return semantic_ir

    def _infer_symbolic(self, op: str, input_shapes, dim_env) -> list[Dim]:
        """符号 shape 推导 — 和静态推导逻辑相同，但支持 SymbolicDim。"""
        rule = get_shape_rule(op)
        return rule.apply_symbolic(input_shapes, dim_env)
```

### 6.2 Shape Specialization

```python
class ShapeSpecializer:
    """将 symbolic SemanticIR 具体化为多个 static 版本。"""

    def specialize_for_benchmarks(self, semantic_ir: SemanticIR) -> list[tuple[dict[str, int], SemanticIR]]:
        """生成 benchmark 用的 shape 实例列表。"""
        # 从 symbolic_dims 的 range 选择代表性点
        dims = semantic_ir.symbolic_dims
        shape_points = self._select_representative_shapes(dims)
        # e.g., [({"B":1,"S":128}, ir1), ({"B":1,"S":512}, ir2), ({"B":8,"S":2048}, ir3)]
        return [(bindings, semantic_ir.specialize(bindings)) for bindings in shape_points]

    def _select_representative_shapes(self, dims: list[SymbolicDim]) -> list[dict[str, int]]:
        """选择 range 的 corner cases + 代表值。"""
        # min, max, midpoint, power-of-2 boundary points
        ...
```

### 6.3 Constraint Validation

```python
class ConstraintValidator:
    """验证 shape bindings 是否满足所有约束。"""

    def validate(self, constraints: list[ShapeConstraint], bindings: dict[str, int]) -> bool:
        for c in constraints:
            if not self._check_constraint(c, bindings):
                return False
        return True

    def _check_constraint(self, c: ShapeConstraint, bindings: dict[str, int]) -> bool:
        lhs_val = self._eval_expr(c.lhs, bindings)
        rhs_val = self._eval_expr(c.rhs, bindings) if isinstance(c.rhs, str) else c.rhs
        match c.op:
            case "==": return lhs_val == rhs_val
            case ">=": return lhs_val >= rhs_val
            case "<=": return lhs_val <= rhs_val
            case "%":  return lhs_val % rhs_val == 0
            case _:    return True
```

---

## 7. Codegen 和 Runtime

### 7.1 Triton 后端的动态 Shape

**当前（静态）:**
```python
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, ...):
    # BLOCK_M, BLOCK_N, BLOCK_K 是 tl.constexpr
    # M, N, K 在 kernel 内部硬编码或从 tl.program_id 推导
```

**目标（动态）:**
```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,                # shape 作为显式参数
    stride_am, stride_ak,   # strides 作为参数
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,  # tile size 仍然是 constexpr
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M, N, K 用于计算 grid size 和边界检查
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M  # 动态边界检查
    ...
```

**Triton Autotune:**
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],  # autotune 根据这些维度选择
)
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    ...
```

### 7.2 Template Engine 变更

`TritonTemplateEngine._build_context()` 扩展：

```python
def _build_context(self, semantic, strategy, primary_op):
    ctx = {...}

    # 新增：动态 shape 参数
    if not semantic.is_static:
        ctx["dynamic_dims"] = {
            d.name: d for d in semantic.symbolic_dims if not d.is_static
        }
        ctx["shape_params"] = [d.name for d in semantic.symbolic_dims if not d.is_static]
        ctx["autotune_keys"] = ctx["shape_params"]  # autotune 根据动态维度选择
    else:
        ctx["dynamic_dims"] = {}
        ctx["shape_params"] = []

    return ctx
```

Jinja2 template 修改（以 `matmul.py.j2` 为例）：
```python
{# 根据是否有动态 dim 决定函数签名 #}
{% if shape_params %}
@triton.autotune(configs=[...], key={{ autotune_keys }})
{% endif %}
@triton.jit
def {{ kernel_name }}(
    {% for p in params %}{{ p.name }}_ptr, {% endfor %}
    {% for dim in shape_params %}{{ dim }}, {% endfor %}
    ...
):
    {% for dim in shape_params %}
    # Dynamic bound check for {{ dim }}
    {% endfor %}
```

### 7.3 MLIR 后端的动态 Shape（Stage 2 预留）

```mlir
// 动态 shape → MLIR 的 ? 维度
func.func @flash_attention(
    %Q: tensor<?x32x?x128xf16>,   // B and S are dynamic
    %K: tensor<?x32x?x128xf16>,
    %V: tensor<?x32x?x128xf16>
) -> tensor<?x32x?x128xf16> {
    // 运行时获取维度
    %B = tensor.dim %Q, %c0 : tensor<?x32x?x128xf16>
    %S = tensor.dim %Q, %c2 : tensor<?x32x?x128xf16>
    ...
}
```

### 7.4 Runtime: DynamicKernelCache

```python
class DynamicKernelCache:
    """运行时 shape → compiled kernel 的映射。"""

    def __init__(self, op: str, compiled_versions: dict[tuple, CompiledKernel]):
        self.op = op
        self._cache: dict[tuple, CompiledKernel] = compiled_versions
        self._buckets: list[tuple] = sorted(compiled_versions.keys())  # 预编译的 shape bucket

    def get_kernel(self, **shape_bindings) -> CompiledKernel:
        key = self._make_key(shape_bindings)

        # 1. 精确匹配
        if key in self._cache:
            return self._cache[key]

        # 2. 最近的 bucket（pad 到最近的预编译 shape）
        bucket = self._find_nearest_bucket(key)
        if bucket is not None:
            return self._cache[bucket]

        # 3. JIT compile（cold path）
        kernel = self._jit_compile(shape_bindings)
        self._cache[key] = kernel
        return kernel

    def _find_nearest_bucket(self, key: tuple) -> tuple | None:
        """找到 >= key 的最小 bucket（pad 策略）。"""
        for bucket in self._buckets:
            if all(b >= k for b, k in zip(bucket, key)):
                return bucket
        return None

    def _jit_compile(self, bindings: dict[str, int]) -> CompiledKernel:
        """即时编译一个新 shape 版本。"""
        # SemanticIR.specialize(bindings) → StrategyIR → codegen → compile
        ...
```

---

## 8. Arke Agent 的动态 Shape 工程

### 8.1 Shape-Aware Prompt Design

当前 Agent prompt 假设固定 shape。扩展为 shape-aware：

```
## Kernel Information
- Op: flash_attention
- Symbolic Dims: B=dynamic(1..64), S=dynamic(1..131072, align=128)
- Static Dims: H=32, D=128
- Constraints: S_kv >= S

## Task
Generate a strategy that works well across the entire S range (1..131072).
Consider:
1. For S <= 64: naive attention may be faster (no block overhead)
2. For S in [128, 4096]: standard flash attention
3. For S > 4096: need careful memory management

Generate a conditional strategy with appropriate shape regimes.
```

### 8.2 Multi-Shape Batch Validation

Agent 优化后，不只验证一个 shape，而是验证 shape range 的代表点：

```python
class DynamicBenchmarkRunner:
    """多 shape 批量验证。"""

    def validate_kernel(
        self,
        semantic_ir: SemanticIR,  # 含 symbolic dims
        compiled_kernel: CompiledKernel,
    ) -> DynamicBenchmarkResult:
        """在 shape range 的代表点上验证 correctness + performance。"""
        specializer = ShapeSpecializer()
        shape_points = specializer.specialize_for_benchmarks(semantic_ir)

        results = []
        for bindings, static_ir in shape_points:
            inputs = generate_test_inputs(static_ir)
            # Correctness
            reference = interpreter.execute(static_ir, inputs)
            output = compiled_kernel(**inputs, **bindings)
            correct = torch.allclose(output, reference, atol=0.01, rtol=0.01)
            # Performance
            latency = benchmark_latency(compiled_kernel, inputs, bindings)
            results.append(ShapePoint(bindings=bindings, correct=correct, latency_us=latency))

        return DynamicBenchmarkResult(
            shape_points=results,
            geomean_latency=geomean([r.latency_us for r in results]),
            all_correct=all(r.correct for r in results),
        )
```

### 8.3 Shape-Performance Profile

Agent 获得 shape → latency 映射后，可以识别性能悬崖并针对性优化：

```
Shape Performance Profile for flash_attention:
  B=1, S=64:    38.0 μs  (baseline: 35.2 μs, ratio: 1.08×)  ✅
  B=1, S=128:   45.7 μs  (baseline: 42.1 μs, ratio: 1.09×)  ✅
  B=1, S=512:  152.3 μs  (baseline: 148.0 μs, ratio: 1.03×) ✅
  B=1, S=2048: 891.2 μs  (baseline: 620.0 μs, ratio: 1.44×) ⚠️ CLIFF
  B=1, S=8192: 3201.0 μs (baseline: 2800.0 μs, ratio: 1.14×) ✅

Agent action: S=2048 性能悬崖 → 调整 tile size 或换算法。
```

### 8.4 Iterative Shape-Aware Optimization

Agent 优化循环扩展为 shape-aware：

```
1. Agent 读取 kernel spec + symbolic dims
2. Agent 生成 conditional strategy（per shape regime）
3. Compiler 编译 → multi-shape benchmark
4. Agent 收到 shape-performance profile
5. Agent 识别 performance cliff → 调整该 regime 的策略
6. 重复 3-5 直到所有 shape 达标
```

这比静态 shape 的 compile→profile→adjust 循环多了一个维度：**shape 维度的搜索**。

---

## 9. Arke Lang Spec v1.1 变更

### 9.1 Grammar 扩展

```ebnf
// 新增规则（向后兼容）
kernel_def = "kernel" IDENT "(" param_list? ")" "->" tensor_type
             where_clause?       // 新增
             "{" kernel_body "}"

tensor_type  = "Tensor" "<" "[" dim_list "]" "," scalar_type ("," layout)? ">"
dim_list     = dim ("," dim)*
dim          = INT                 // 静态（v1.0 兼容）
             | IDENT               // 符号维度引用

where_clause = "where" dim_decl ("," dim_decl)* ("," constraint)*
dim_decl     = IDENT ":" dim_kind
dim_kind     = "static" "(" INT ")"
             | "dynamic"
             | "dynamic" "(" INT ".." INT ")"
             | "dynamic" "(" INT ".." INT "," "align" "=" INT ")"
constraint   = expr comp_op expr
comp_op      = "==" | ">=" | "<=" | "%"
```

### 9.2 语义规则

1. `where` clause 中声明的符号维度名必须出现在参数的 dim_list 中
2. 同名符号维度自动约束为相等（如 Q 和 K 都用 `B` → Q.shape[0] == K.shape[0]）
3. `static(N)` 等价于直接写 `N`（语法糖）
4. 不带 `where` clause 的 kernel 定义等价于所有 dim 为 static（v1.0 向后兼容）

---

## 10. Arke IR Spec v1.1 变更

### 10.1 SemanticIR 新增字段

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `symbolic_dims` | `list[SymbolicDim]` | `[]` | 所有符号维度定义 |
| `constraints` | `list[ShapeConstraint]` | `[]` | shape 约束 |

version 从 `"0.3.0"` → `"0.4.0"`。

### 10.2 Param.shape 类型变更

| 版本 | 类型 | 示例 |
|------|------|------|
| v0.3.0 | `list[int]` | `[128, 768]` |
| v0.4.0 | `list[int \| SymbolicDim]` | `[SymbolicDim("B"), 32, SymbolicDim("S"), 128]` |

JSON 序列化：int 维度保持为 JSON number，SymbolicDim 序列化为 `{"ref": "B"}`（简写）或完整 `{"name": "B", "kind": "dynamic", ...}`。

### 10.3 StrategyIR 新增 Decision kinds

| Kind | 参数 | 描述 |
|------|------|------|
| `conditional` | `condition: str`, `then_decisions: list`, `else_decisions: list` | 条件策略 |
| `shape_regime` | `when: str`, `decisions: list` | shape 分组（conditional 的语法糖）|

### 10.4 from_dict/to_dict 兼容

`from_dict()` 检查 `symbolic_dims` 字段是否存在：
- 存在 → v0.4.0 模式
- 不存在 → v0.3.0 兼容模式（所有 shape 视为 static int）

---

## 11. 实现拆解与工作量

### Phase 1: IR 层（必须先做）

| 文件 | 工作 | 估时 |
|------|------|------|
| `arke/ir/semantic.py` | SymbolicDim, ShapeConstraint dataclass; Param.shape 扩展; specialize() | 2d |
| `arke/ir/strategy.py` | ConditionalDecision, shape_regime Decision kind | 1d |
| `arke/ir/shape_propagation.py` | 新文件：符号 shape 传播引擎 | 3d |
| `arke/ir/constraint_validator.py` | 新文件：约束验证 | 1d |

### Phase 2: Parser 层

| 文件 | 工作 | 估时 |
|------|------|------|
| `arke/parser/arke.lark` | where clause 语法规则 | 1d |
| `arke/parser/parser.py` | where clause → AST 节点解析 | 1d |
| `arke/parser/converter.py` | AST → SemanticIR 的 symbolic_dims + constraints | 2d |

### Phase 3: Codegen 层

| 文件 | 工作 | 估时 |
|------|------|------|
| `arke/backend/triton_template_engine.py` | 动态 dim 上下文、autotune key、mask 生成 | 3d |
| `arke/backend/triton_templates/*.j2` | 修改所有模板支持 dynamic bound check | 3d |

### Phase 4: Runtime 层

| 文件 | 工作 | 估时 |
|------|------|------|
| `arke/integration/dynamic_kernel_cache.py` | 新文件：shape bucket + JIT compile | 2d |
| `arke/integration/kernel_cache.py` | 适配 DynamicKernelCache | 1d |

### Phase 5: Agent 层

| 文件 | 工作 | 估时 |
|------|------|------|
| `arke/agent/prompts/` | shape-aware prompt templates | 1d |
| `arke/agent/runner.py` | multi-shape benchmark 集成 | 2d |

### Phase 6: Benchmark & Test

| 文件 | 工作 | 估时 |
|------|------|------|
| `benchmarks/bench_dynamic.py` | 新文件：动态 shape benchmark runner | 2d |
| `tests/test_symbolic_shape.py` | 新文件：符号 shape 单元测试 | 2d |
| `docs/examples/*.ak` | 为 5-10 个关键 op 写动态 shape 版本 | 1d |

**总计：~27 人天**

### 优先级排序

1. **P0 (G6 重构)**: Phase 1 IR 层 — 必须先做，为后续所有工作打基础
2. **P1 (G6 重构)**: Phase 2 Parser + Phase 3 Codegen — 打通 .ak → GPU 全链路
3. **P2 (G7 前置)**: Phase 4 Runtime + Phase 5 Agent — G7 的 LLaMA/DS-V2 需要
4. **P3 (G7 验证)**: Phase 6 Benchmark — G7 的多 seq_len 验证

---

## 12. 对 G7/G8 的影响

### G7 依赖

G7 的 exit criteria 包含：
- LLaMA-2 7B E2E（seq∈{512, 2048, 4096}）
- DeepSeek-V2 16B E2E（seq∈{512, 2048}）

这意味着 **同一个 kernel 需要在多个 seq_len 下工作**。没有动态 shape 支持，每个 seq_len 需要独立编译——这不仅效率低，而且违背了 Arke "LLM 自主工程" 的理念。

动态 shape 支持使 G7 的工作模式变为：
1. Agent 读取 `flash_attention.ak`（含 `S: dynamic(1..4096)`）
2. Agent 生成覆盖 range 的策略
3. 编译一次，多 seq_len 验证
4. 性能不达标的 shape range → Agent 自动调整

### G8 依赖

G8 是 Stage 1 收尾（4 模型 E2E + 回归测试）。动态 shape 使 G8 的验证更高效——不需要为每个模型的每个 shape 单独编译。

### Stage 2 铺垫

动态 shape 的 SymbolicDim 设计直接映射到 MLIR 的 `?` 维度（`tensor<?x128xf16>`）。Stage 2 的 MLIR backend 可以直接消费 SemanticIR 的 symbolic_dims，无需额外转换。

---

*Created: 2026-04-06 | Author: Arke Architecture Team*