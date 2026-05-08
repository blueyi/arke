# Arke Op Registry Interface

> **Version:** 1.0.0  
> **Status:** Specification  
> **Date:** 2026-04-09  
> **Purpose:** Define universal operator registration, discovery, and extension mechanism

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Op Definition Specification](#3-op-definition-specification)
4. [Registry API](#4-registry-api)
5. [Hardware Variants](#5-hardware-variants)
6. [Strategy Space Definition](#6-strategy-space-definition)
7. [Constraint System](#7-constraint-system)
8. [Reference Implementation](#8-reference-implementation)
9. [Verification & Testing](#9-verification--testing)
10. [Example: matmul](#10-example-matmul)
11. [Example: attention](#11-example-attention)
12. [Ecosystem Management](#12-ecosystem-management)

---

## 1. Overview

### 1.1 Purpose

The Op Registry Interface defines how operators are registered, discovered, and extended in the Arke system. It provides a **universal, algorithm-agnostic specification** that supports:

- **Unlimited operator registration** — not limited to 45 ops, supports any operator
- **Multi-hardware variants** — same operator, different strategies per hardware
- **Dynamic discovery** — LLM can query legal actions at runtime
- **Automatic verification** — shape inference, constraint checking, correctness validation
- **Cross-hardware knowledge transfer** — adapt strategies from one hardware to another

### 1.2 Core Principle

**Operators are not hardcoded in the language or IR.** They are registered externally via the Op Registry, which acts as a **queryable operator catalog**.

```
Arke Lang (.ak)
    ↓ parse
Arke IR (Layer 4: SemanticIR)
    ↓ resolve op_name via Op Registry
Op Definition (signature, rules, strategies, constraints)
    ↓ validate & lower
Arke IR (Layer 3: StrategyIR)
    ↓ compile
GPU Binary
```

---

## 2. Design Philosophy

### 2.1 Principles

1. **Universal** — Single specification for all operators, regardless of domain
2. **Extensible** — New operators can be registered without modifying Arke Lang/IR
3. **Hardware-aware** — Operators can have hardware-specific variants
4. **LLM-friendly** — Registry provides structured data for LLM decision-making
5. **Verifiable** — Every operator definition includes verification rules and tests
6. **Composable** — Operators can be composed (fused) with clear semantics

### 2.2 Relationship to Lang & IR

```
Arke Lang Spec v0.1.0
    ├─ Defines universal syntax (kernel, strategy, where clause)
    └─ Does NOT enumerate operators
    
Arke IR Spec v0.1.0
    ├─ Defines universal data structures (SemanticIR, StrategyIR, etc.)
    └─ Does NOT enumerate operators
    
Op Registry Interface (this spec)
    ├─ Defines how to register operators
    ├─ Provides API for discovery and validation
    └─ Supports unlimited operator ecosystem
```

---

## 3. Op Definition Specification

### 3.1 Core Data Structure

```python
@dataclass
class OpSignature:
    """Operator signature: inputs, outputs, attributes."""
    name: str                           # unique operator name
    inputs: list[TensorParam]           # input tensor parameters
    outputs: list[TensorParam]          # output tensor parameters
    attrs: dict[str, AttrType]          # operator-specific attributes

@dataclass
class TensorParam:
    """Tensor parameter descriptor."""
    name: str                           # parameter name
    shape: list[int | str]              # shape (int=concrete, str=symbolic)
    dtype: str                          # data type
    optional: bool = False              # whether parameter is optional

@dataclass
class AttrType:
    """Attribute type specification."""
    name: str
    type: str                           # "int", "float", "bool", "string", "list"
    default: Any = None
    constraints: list[str] = field(default_factory=list)

@dataclass
class SemanticRule:
    """Shape inference and semantic rule."""
    rule_type: str                      # "shape_inference", "dtype_inference", "constraint"
    description: str                    # human-readable description
    implementation: str                 # Python code or formula

@dataclass
class OpDefinition:
    """Complete operator definition."""
    signature: OpSignature
    semantic_rules: list[SemanticRule]  # shape/dtype inference, constraints
    strategy_space: StrategySpace        # legal optimization decisions
    constraints: list[str]              # hardware/correctness constraints
    reference_impl: str                 # reference implementation (NumPy/PyTorch)
    verification_tests: list[str]       # test cases for V1 verification
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.2 Semantic Rules

Semantic rules define how the operator transforms shapes and types:

```python
# Example: matmul semantic rules
semantic_rules = [
    SemanticRule(
        rule_type="shape_inference",
        description="Output shape is [A.shape[0], B.shape[1]]",
        implementation="output_shape = [inputs[0].shape[0], inputs[1].shape[1]]"
    ),
    SemanticRule(
        rule_type="dtype_inference",
        description="Output dtype matches input dtype",
        implementation="output_dtype = inputs[0].dtype"
    ),
    SemanticRule(
        rule_type="constraint",
        description="Inner dimensions must match",
        implementation="assert inputs[0].shape[1] == inputs[1].shape[0]"
    )
]
```

### 3.3 Constraints

Constraints define validity conditions:

```python
constraints = [
    "inputs[0].dtype == inputs[1].dtype",  # type consistency
    "inputs[0].shape[1] == inputs[1].shape[0]",  # dimension compatibility
    "inputs[0].shape[0] > 0",  # positive dimensions
    "inputs[1].shape[1] > 0"
]
```

---

## 4. Registry API

### 4.1 Registration

```python
class OpRegistry:
    """Global operator registry."""
    
    def register(self, op_def: OpDefinition, target_hw: str | None = None) -> None:
        """Register an operator definition.
        
        Args:
            op_def: Operator definition
            target_hw: Optional hardware target (e.g., "nvidia_ampere")
                      If None, definition is hardware-agnostic
        """
        ...
    
    def unregister(self, op_name: str, target_hw: str | None = None) -> None:
        """Unregister an operator."""
        ...
```

### 4.2 Discovery

```python
    def get_op(self, op_name: str, target_hw: str | None = None) -> OpDefinition:
        """Get operator definition.
        
        Resolution order:
        1. Hardware-specific definition (if target_hw provided)
        2. Hardware-agnostic definition
        3. Raise OpNotFound
        """
        ...
    
    def list_ops(self, target_hw: str | None = None) -> list[str]:
        """List all registered operators.
        
        Args:
            target_hw: If provided, return only ops available for this hardware
        """
        ...
    
    def list_targets(self) -> list[str]:
        """List all registered hardware targets."""
        ...
```

### 4.3 Validation

```python
    def validate_call(self, op_name: str, inputs: list[TensorDesc], 
                     attrs: dict[str, Any], target_hw: str) -> ValidationResult:
        """Validate an operator call.
        
        Returns:
            ValidationResult with:
            - is_valid: bool
            - output_types: list[TensorDesc]
            - errors: list[str]
        """
        ...
    
    def get_legal_actions(self, op_name: str, target_hw: str) -> list[DecisionAction]:
        """Get legal optimization decisions for an operator.
        
        Returns list of possible decisions (tile factors, fusion candidates, etc.)
        """
        ...
```

### 4.4 Shape Inference

```python
    def infer_shape(self, op_name: str, inputs: list[TensorDesc]) -> list[TensorDesc]:
        """Infer output shapes given input shapes."""
        ...
    
    def infer_dtype(self, op_name: str, inputs: list[TensorDesc]) -> list[str]:
        """Infer output dtypes given input dtypes."""
        ...
```

---

## 5. Hardware Variants

### 5.1 Multi-Hardware Support

Same operator can have different definitions per hardware:

```python
# Register hardware-agnostic base definition
registry.register(matmul_base_def)

# Register NVIDIA Ampere variant
registry.register(matmul_ampere_def, target_hw="nvidia_ampere")

# Register Ascend 910B variant
registry.register(matmul_ascend_def, target_hw="ascend_910b")

# Query resolution
registry.get_op("matmul", target_hw="nvidia_ampere")  # → matmul_ampere_def
registry.get_op("matmul", target_hw="ascend_910b")    # → matmul_ascend_def
registry.get_op("matmul")                             # → matmul_base_def
```

### 5.2 Variant Inheritance

Hardware variants can inherit from base definition:

```python
@dataclass
class OpVariant:
    """Hardware-specific operator variant."""
    base_op: str                        # reference to base op
    target_hw: str                      # hardware target
    strategy_space_override: StrategySpace | None = None
    constraints_override: list[str] | None = None
    reference_impl_override: str | None = None
    
    # If not overridden, inherit from base
```

---

## 6. Strategy Space Definition

### 6.1 Purpose

Strategy space defines what optimization decisions are legal for an operator.

### 6.2 Data Structure

```python
@dataclass
class StrategySpace:
    """Legal optimization decisions for an operator."""
    decisions: list[DecisionType]       # types of decisions allowed
    constraints: list[str]              # constraints on decisions
    
@dataclass
class DecisionType:
    """Type of optimization decision."""
    kind: str                           # "tile", "fuse", "parallelize", etc.
    parameters: dict[str, ParameterSpec]
    constraints: list[str]              # constraints specific to this decision type

@dataclass
class ParameterSpec:
    """Parameter specification for a decision."""
    name: str
    type: str                           # "int", "list[int]", "string", etc.
    legal_values: list[Any] | None      # if None, any value satisfying constraints
    constraints: list[str]              # e.g., "value > 0", "value % 32 == 0"
```

### 6.3 Example: matmul strategy space

```python
matmul_strategy_space = StrategySpace(
    decisions=[
        DecisionType(
            kind="tile",
            parameters={
                "dim": ParameterSpec(
                    name="dim",
                    type="string",
                    legal_values=["M", "N", "K"]
                ),
                "factors": ParameterSpec(
                    name="factors",
                    type="list[int]",
                    constraints=["all(f > 0 for f in factors)", "len(factors) <= 3"]
                )
            }
        ),
        DecisionType(
            kind="fuse",
            parameters={
                "ops": ParameterSpec(
                    name="ops",
                    type="list[str]",
                    constraints=["len(ops) >= 2"]
                )
            }
        ),
        DecisionType(
            kind="compute",
            parameters={
                "num_threads": ParameterSpec(
                    name="num_threads",
                    type="int",
                    legal_values=[128, 256, 512, 1024],
                    constraints=["value <= hw_max_threads"]
                ),
                "num_stages": ParameterSpec(
                    name="num_stages",
                    type="int",
                    constraints=["value >= 1", "value <= 10"]
                )
            }
        )
    ]
)
```

---

## 7. Constraint System

### 7.1 Constraint Types

```python
class ConstraintType(Enum):
    SHAPE = "shape"                     # shape compatibility
    DTYPE = "dtype"                     # data type compatibility
    HARDWARE = "hardware"               # hardware resource limits
    CORRECTNESS = "correctness"         # mathematical correctness
    PERFORMANCE = "performance"         # performance hints
```

### 7.2 Constraint Specification

```python
@dataclass
class Constraint:
    """Single constraint."""
    type: ConstraintType
    description: str                    # human-readable
    check: str                          # Python expression to evaluate
    error_message: str                  # message if constraint violated
```

### 7.3 Example Constraints

```python
constraints = [
    Constraint(
        type=ConstraintType.SHAPE,
        description="Inner dimensions must match for matmul",
        check="inputs[0].shape[1] == inputs[1].shape[0]",
        error_message="A.shape[1] must equal B.shape[0]"
    ),
    Constraint(
        type=ConstraintType.DTYPE,
        description="Input dtypes must match",
        check="inputs[0].dtype == inputs[1].dtype",
        error_message="A and B must have same dtype"
    ),
    Constraint(
        type=ConstraintType.HARDWARE,
        description="Shared memory usage must fit",
        check="shared_memory_used <= hw_profile.shared_memory",
        error_message="Shared memory exceeded"
    )
]
```

---

## 8. Reference Implementation

### 8.1 Purpose

Reference implementation provides ground truth for V1 numerical verification.

### 8.2 Format

```python
@dataclass
class ReferenceImpl:
    """Reference implementation."""
    language: str                       # "numpy", "pytorch", "python"
    code: str                           # implementation code
    test_cases: list[TestCase]          # test cases
```

### 8.3 Example: matmul reference

```python
reference_impl = ReferenceImpl(
    language="numpy",
    code="""
def matmul_ref(A, B, transpose_a=False, transpose_b=False):
    if transpose_a:
        A = A.T
    if transpose_b:
        B = B.T
    return np.matmul(A, B)
""",
    test_cases=[
        TestCase(
            inputs={"A": np.random.randn(128, 64), "B": np.random.randn(64, 256)},
            attrs={"transpose_a": False, "transpose_b": False},
            expected_shape=(128, 256)
        ),
        # ... more test cases
    ]
)
```

---

## 9. Verification & Testing

### 9.1 Verification Levels

```python
class VerificationLevel(Enum):
    V0_STATIC = "v0_static"             # static checks (constraints, types)
    V1_NUMERICAL = "v1_numerical"       # numerical correctness (vs reference)
    V2_PERFORMANCE = "v2_performance"   # performance profiling
```

### 9.2 Test Case Structure

```python
@dataclass
class TestCase:
    """Test case for operator verification."""
    name: str
    inputs: dict[str, np.ndarray]       # input tensors
    attrs: dict[str, Any]               # operator attributes
    expected_output: np.ndarray | None  # expected output (for V1)
    expected_shape: tuple | None        # expected output shape
    expected_dtype: str | None          # expected output dtype
    tolerance: float = 1e-5             # numerical tolerance
```

### 9.3 Verification API

```python
class Verifier:
    """Operator verification."""
    
    def verify_v0(self, op_def: OpDefinition, call: OpCall) -> V0Result:
        """Static verification: constraints, types, shapes."""
        ...
    
    def verify_v1(self, op_def: OpDefinition, test_case: TestCase) -> V1Result:
        """Numerical verification: correctness vs reference."""
        ...
    
    def verify_v2(self, op_def: OpDefinition, compiled_kernel) -> V2Result:
        """Performance verification: profiling on hardware."""
        ...
```

---

## 10. Example: matmul

### 10.1 Complete Definition

```python
matmul_def = OpDefinition(
    signature=OpSignature(
        name="matmul",
        inputs=[
            TensorParam("A", ["M", "K"], "f32"),
            TensorParam("B", ["K", "N"], "f32")
        ],
        outputs=[TensorParam("C", ["M", "N"], "f32")],
        attrs={
            "transpose_a": AttrType("transpose_a", "bool", False),
            "transpose_b": AttrType("transpose_b", "bool", False)
        }
    ),
    semantic_rules=[
        SemanticRule(
            rule_type="shape_inference",
            description="Output shape is [A.shape[0], B.shape[1]]",
            implementation="output_shape = [inputs[0].shape[0], inputs[1].shape[1]]"
        ),
        SemanticRule(
            rule_type="constraint",
            description="Inner dimensions must match",
            implementation="assert inputs[0].shape[1] == inputs[1].shape[0]"
        )
    ],
    strategy_space=StrategySpace(
        decisions=[
            DecisionType(kind="tile", parameters={...}),
            DecisionType(kind="fuse", parameters={...}),
            DecisionType(kind="compute", parameters={...})
        ]
    ),
    constraints=[
        "inputs[0].dtype == inputs[1].dtype",
        "inputs[0].shape[1] == inputs[1].shape[0]",
        "M > 0 and K > 0 and N > 0"
    ],
    reference_impl="numpy.matmul",
    verification_tests=[...]
)
```

---

## 11. Example: attention

### 11.1 Multi-Output Operator

```python
attention_def = OpDefinition(
    signature=OpSignature(
        name="flash_attention",
        inputs=[
            TensorParam("Q", ["B", "H", "S", "D"], "f16"),
            TensorParam("K", ["B", "H", "S", "D"], "f16"),
            TensorParam("V", ["B", "H", "S", "D"], "f16")
        ],
        outputs=[
            TensorParam("O", ["B", "H", "S", "D"], "f16"),
            TensorParam("lse", ["B", "H", "S"], "f32", optional=True)
        ],
        attrs={
            "causal": AttrType("causal", "bool", False),
            "dropout_p": AttrType("dropout_p", "float", 0.0)
        }
    ),
    semantic_rules=[...],
    strategy_space=StrategySpace(
        decisions=[
            DecisionType(kind="tile", parameters={"dim": ["S"], "factors": [...]}),
            DecisionType(kind="block_size", parameters={"block_size": [64, 128, 256]})
        ]
    ),
    constraints=[...],
    reference_impl="torch.nn.functional.scaled_dot_product_attention",
    verification_tests=[...]
)
```

---

## 12. Ecosystem Management

### 12.1 Op Catalog Organization

```
arke/ops/
├── core/                    # fundamental ops (matmul, add, mul, etc.)
│   ├── linalg.py           # linear algebra ops
│   ├── elementwise.py       # element-wise ops
│   └── reduction.py         # reduction ops
├── nn/                      # neural network ops
│   ├── activation.py        # relu, gelu, swiglu, etc.
│   ├── normalization.py     # layernorm, batchnorm, etc.
│   └── attention.py         # attention variants
├── fusion/                  # fused operators
│   ├── matmul_relu.py
│   ├── matmul_gelu.py
│   └── rmsnorm_residual.py
└── custom/                  # user-defined operators
    └── my_custom_op.py
```

### 12.2 Op Versioning

```python
@dataclass
class OpVersion:
    """Operator version."""
    op_name: str
    version: str                        # semantic versioning (e.g., "1.0.0")
    created: str                        # ISO 8601 timestamp
    lifecycle: str = "active"              # active | experimental | retired
    successor: str | None = None           # if retired, what to use instead
```

### 12.3 Op Discovery & Loading

```python
class OpCatalog:
    """Operator catalog with discovery and loading."""
    
    def load_from_directory(self, path: str) -> None:
        """Load all operators from directory."""
        ...
    
    def load_from_package(self, package_name: str) -> None:
        """Load operators from installed package."""
        ...
    
    def search(self, query: str) -> list[OpDefinition]:
        """Search operators by name or tags."""
        ...
```

### 12.4 Op Sharing & Distribution

Operators can be shared via:
- **Git repositories** — version control, collaboration
- **PyPI packages** — distribution, dependency management
- **Op Hub** — centralized registry (future)

---

## References

- `docs/spec/arke-lang-spec.md` — Arke Language v0.1.0
- `docs/spec/arke-ir-spec.md` — Arke IR v0.1.0
- `docs/architecture/e2e-flow.md` — End-to-end flow
- `docs/architecture/agent-design.md` — Agent design

---

**End of Op Registry Interface Specification**
