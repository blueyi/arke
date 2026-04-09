# Arke Symbolic Dimension Specification

> **Version:** 1.0.0  
> **Status:** Specification  
> **Date:** 2026-04-09  
> **Purpose:** Define symbolic dimension syntax, semantics, constraint system, and propagation algorithm

---

## Table of Contents

1. [Overview](#1-overview)
2. [Symbolic Dimension Syntax](#2-symbolic-dimension-syntax)
3. [Dimension Constraints](#3-dimension-constraints)
4. [Shape Inference](#4-shape-inference)
5. [Dimension Propagation Algorithm](#5-dimension-propagation-algorithm)
6. [Constraint Solving](#6-constraint-solving)
7. [Hardware-Specific Constraints](#7-hardware-specific-constraints)
8. [Examples](#8-examples)
9. [Implementation Notes](#9-implementation-notes)

---

## 1. Overview

### 1.1 Motivation

GPU kernels often operate on tensors with **dynamic shapes** that are not known at compile time:

```python
# matmul(A, B) where A.shape = (batch, seq_len, hidden_dim)
# batch and seq_len are determined at runtime
```

Arke uses **symbolic dimensions** to represent these dynamic shapes in the IR, enabling:
- Shape inference without concrete values
- Constraint propagation across operations
- Hardware-aware memory estimation
- Correctness validation for arbitrary input shapes

### 1.2 Core Concepts

| Concept | Definition | Example |
|:--------|:-----------|:--------|
| **Concrete Dimension** | Fixed integer value | `64`, `128`, `256` |
| **Symbolic Dimension** | Named variable | `M`, `N`, `K`, `seq_len`, `batch` |
| **Shape** | List of dimensions (concrete or symbolic) | `[batch, seq_len, hidden_dim]` |
| **Constraint** | Relationship between dimensions | `M > 0`, `K == N`, `batch % 8 == 0` |
| **Dimension Space** | Set of valid dimension values | `M ∈ {1..2048}` |

### 1.3 Symbolic Dimension Hierarchy

```
Dimension
├── Concrete (int)
│   ├── Positive: 1, 2, 3, ...
│   └── Special: 0 (scalar), -1 (inferred)
└── Symbolic (str)
    ├── User-defined: M, N, K, batch, seq_len
    ├── Derived: M*2, N+1, K//8
    └── Constrained: M ∈ {1..1024}, N % 8 == 0
```

---

## 2. Symbolic Dimension Syntax

### 2.1 Dimension Literals

```
dimension ::= concrete_dim | symbolic_dim | derived_dim

concrete_dim ::= integer                    # 1, 64, 128, 256
                | "-1"                      # inferred (placeholder)
                | "0"                       # scalar

symbolic_dim ::= identifier                 # M, N, K, batch, seq_len
                | identifier "[" range "]"  # M[1..1024], batch[1..128]

derived_dim ::= symbolic_dim op symbolic_dim
              | symbolic_dim op concrete_dim
              | "(" derived_dim ")"

op ::= "+" | "-" | "*" | "//" | "%"         # arithmetic operators

range ::= concrete_dim ".." concrete_dim    # inclusive range
        | concrete_dim                      # single value
```

### 2.2 Shape Syntax

```
shape ::= "[" dimension_list "]"

dimension_list ::= dimension
                 | dimension "," dimension_list

# Examples
[64]                                        # 1D: concrete
[M, N]                                      # 2D: symbolic
[batch, seq_len, hidden_dim]                # 3D: symbolic
[batch[1..128], seq_len[1..4096], 768]      # mixed with ranges
[M, N, K//8]                                # derived dimensions
```

### 2.3 Dimension Naming Conventions

```
# Standard names (recommended)
M, N, K                                     # matrix dimensions
batch, seq_len, hidden_dim                  # sequence model dimensions
num_heads, head_dim                         # attention dimensions
in_channels, out_channels                   # convolution dimensions
height, width, depth                        # spatial dimensions

# Derived names (for intermediate dimensions)
M_tiled = M // 64                           # tiled dimension
seq_padded = (seq_len + 127) // 128         # padded dimension
```

---

## 3. Dimension Constraints

### 3.1 Constraint Types

```python
@dataclass
class Constraint:
    """Dimension constraint."""
    type: str                               # "equality", "inequality", "divisibility", "range"
    operands: List[str | int]               # dimension names or values
    operator: str                           # "==", "!=", "<", ">", "<=", ">=", "%"
    value: int | str | None                 # RHS value or dimension name

# Examples
Constraint(type="equality", operands=["K"], operator="==", value="N")
    # K == N

Constraint(type="inequality", operands=["M"], operator=">", value=0)
    # M > 0

Constraint(type="divisibility", operands=["batch"], operator="%", value=8)
    # batch % 8 == 0

Constraint(type="range", operands=["seq_len"], operator="in", value=(1, 4096))
    # 1 <= seq_len <= 4096
```

### 3.2 Constraint Syntax in Arke Lang

```arke
# In kernel definition
kernel matmul {
    inputs: [A: f32[M, K], B: f32[K, N]]
    outputs: [C: f32[M, N]]
    
    # Constraints on dimensions
    where {
        M > 0,
        N > 0,
        K > 0,
        K == A.shape[1],
        N == B.shape[1]
    }
    
    compute: C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
}

# In strategy definition
strategy matmul_opt {
    where {
        batch % 8 == 0,                     # batch must be divisible by 8
        seq_len >= 128,                     # seq_len must be at least 128
        hidden_dim % 64 == 0                # hidden_dim must be divisible by 64
    }
    
    @rationale("Tile for cache locality")
    tile(M, 64)
    tile(N, 64)
}
```

### 3.3 Constraint Propagation

```python
# Constraint propagation rules
# If A == B and B == C, then A == C (transitivity)
# If A > B and B > C, then A > C (transitivity)
# If A % 8 == 0 and B = A // 8, then B is integer (divisibility)

class ConstraintPropagator:
    def propagate(self, constraints: List[Constraint]) -> List[Constraint]:
        """Propagate constraints to derive new ones."""
        new_constraints = []
        
        # Transitivity rules
        for c1 in constraints:
            for c2 in constraints:
                if c1.operator == c2.operator == "==":
                    if c1.value == c2.operands[0]:
                        # c1: A == B, c2: B == C → A == C
                        new_constraints.append(
                            Constraint(type="equality", 
                                operands=[c1.operands[0]], 
                                operator="==", 
                                value=c2.value)
                        )
        
        return constraints + new_constraints
```

---

## 4. Shape Inference

### 4.1 Shape Inference Rules

Shape inference rules define how output shapes are computed from input shapes:

```python
@dataclass
class ShapeInferenceRule:
    """Rule for inferring output shape from input shapes."""
    op_name: str                            # operation name
    inputs: List[str]                       # input parameter names
    outputs: List[str]                      # output parameter names
    rule: str                               # Python expression or formula

# Examples
ShapeInferenceRule(
    op_name="matmul",
    inputs=["A", "B"],
    outputs=["C"],
    rule="C.shape = [A.shape[0], B.shape[1]]"
)

ShapeInferenceRule(
    op_name="transpose",
    inputs=["A"],
    outputs=["B"],
    rule="B.shape = [A.shape[1], A.shape[0]]"
)

ShapeInferenceRule(
    op_name="reshape",
    inputs=["A", "new_shape"],
    outputs=["B"],
    rule="B.shape = new_shape; assert prod(A.shape) == prod(B.shape)"
)

ShapeInferenceRule(
    op_name="attention",
    inputs=["Q", "K", "V"],
    outputs=["O"],
    rule="O.shape = [Q.shape[0], Q.shape[1], V.shape[2]]"
)
```

### 4.2 Shape Inference Algorithm

```python
class ShapeInferenceEngine:
    """Infer output shapes from input shapes and rules."""
    
    def infer_shape(self, op_name: str, input_shapes: Dict[str, List]) -> Dict[str, List]:
        """Infer output shapes for an operation.
        
        Args:
            op_name: Operation name (e.g., "matmul")
            input_shapes: Dict mapping input names to shapes
        
        Returns:
            Dict mapping output names to inferred shapes
        """
        rule = self.get_rule(op_name)
        
        # Substitute input shapes into rule
        output_shapes = {}
        for output_name in rule.outputs:
            # Execute rule to compute output shape
            output_shape = self._execute_rule(rule.rule, input_shapes)
            output_shapes[output_name] = output_shape
        
        return output_shapes
    
    def _execute_rule(self, rule_expr: str, input_shapes: Dict) -> List:
        """Execute shape inference rule expression."""
        # Create namespace with input shapes
        namespace = {
            name: shape for name, shape in input_shapes.items()
        }
        
        # Execute rule in namespace
        result = eval(rule_expr, {"__builtins__": {}}, namespace)
        return result
```

### 4.3 Shape Inference Examples

```python
# Example 1: matmul
inputs = {
    "A": ["batch", "seq_len", "hidden_dim"],
    "B": ["hidden_dim", "out_dim"]
}
rule = "C.shape = [A.shape[0], A.shape[1], B.shape[1]]"
output = infer_shape("matmul", inputs)
# output: {"C": ["batch", "seq_len", "out_dim"]}

# Example 2: attention
inputs = {
    "Q": ["batch", "num_heads", "seq_len", "head_dim"],
    "K": ["batch", "num_heads", "seq_len", "head_dim"],
    "V": ["batch", "num_heads", "seq_len", "head_dim"]
}
rule = "O.shape = [Q.shape[0], Q.shape[1], Q.shape[2], V.shape[3]]"
output = infer_shape("attention", inputs)
# output: {"O": ["batch", "num_heads", "seq_len", "head_dim"]}

# Example 3: reshape with derived dimension
inputs = {
    "A": ["batch", "seq_len", "hidden_dim"],
    "new_shape": ["-1", "hidden_dim"]  # -1 means inferred
}
rule = "B.shape = [batch * seq_len, hidden_dim]"
output = infer_shape("reshape", inputs)
# output: {"B": ["batch*seq_len", "hidden_dim"]}
```

---

## 5. Dimension Propagation Algorithm

### 5.1 Forward Propagation

Forward propagation computes output shapes from input shapes:

```python
class DimensionPropagator:
    """Propagate dimensions through computation graph."""
    
    def forward_propagate(self, graph: ComputationGraph) -> Dict[str, List]:
        """Propagate dimensions forward through graph.
        
        Args:
            graph: Computation graph with operations
        
        Returns:
            Dict mapping tensor names to inferred shapes
        """
        shapes = {}
        
        # Initialize input shapes
        for input_tensor in graph.inputs:
            shapes[input_tensor.name] = input_tensor.shape
        
        # Topologically sort operations
        ops_sorted = self._topological_sort(graph.operations)
        
        # Propagate shapes through operations
        for op in ops_sorted:
            # Get input shapes
            input_shapes = {
                name: shapes[name] for name in op.input_names
            }
            
            # Infer output shapes
            output_shapes = self.infer_shape(op.name, input_shapes)
            
            # Store output shapes
            for output_name, output_shape in output_shapes.items():
                shapes[output_name] = output_shape
        
        return shapes
    
    def _topological_sort(self, operations: List) -> List:
        """Sort operations in topological order."""
        # Implementation: DFS or Kahn's algorithm
        pass
```

### 5.2 Backward Propagation

Backward propagation infers input shapes from output shapes (useful for constraint solving):

```python
class DimensionPropagator:
    def backward_propagate(self, graph: ComputationGraph, 
                          output_shapes: Dict[str, List]) -> Dict[str, List]:
        """Propagate dimensions backward through graph.
        
        Args:
            graph: Computation graph
            output_shapes: Known output shapes
        
        Returns:
            Dict mapping tensor names to inferred shapes
        """
        shapes = {}
        
        # Initialize output shapes
        for output_tensor in graph.outputs:
            shapes[output_tensor.name] = output_shapes[output_tensor.name]
        
        # Reverse topologically sort operations
        ops_sorted = self._reverse_topological_sort(graph.operations)
        
        # Propagate shapes backward through operations
        for op in ops_sorted:
            # Get output shapes
            output_shapes_op = {
                name: shapes[name] for name in op.output_names
                if name in shapes
            }
            
            # Infer input shapes (if possible)
            input_shapes = self._infer_input_shapes(op, output_shapes_op)
            
            # Store input shapes
            for input_name, input_shape in input_shapes.items():
                shapes[input_name] = input_shape
        
        return shapes
    
    def _infer_input_shapes(self, op: Operation, 
                           output_shapes: Dict) -> Dict[str, List]:
        """Infer input shapes from output shapes."""
        # Implementation: inverse shape inference rules
        pass
```

---

## 6. Constraint Solving

### 6.1 Constraint Solver

```python
class ConstraintSolver:
    """Solve dimension constraints."""
    
    def solve(self, constraints: List[Constraint], 
              known_dims: Dict[str, int]) -> Dict[str, int]:
        """Solve constraints to find dimension values.
        
        Args:
            constraints: List of constraints
            known_dims: Known dimension values
        
        Returns:
            Dict mapping dimension names to values
        """
        # Start with known dimensions
        solution = dict(known_dims)
        
        # Iteratively solve constraints
        changed = True
        while changed:
            changed = False
            
            for constraint in constraints:
                # Try to solve constraint
                new_values = self._solve_constraint(constraint, solution)
                
                # Update solution
                for dim, value in new_values.items():
                    if dim not in solution:
                        solution[dim] = value
                        changed = True
                    elif solution[dim] != value:
                        raise ValueError(f"Conflicting values for {dim}")
        
        return solution
    
    def _solve_constraint(self, constraint: Constraint, 
                         known: Dict[str, int]) -> Dict[str, int]:
        """Solve a single constraint."""
        if constraint.type == "equality":
            # A == B: if A is known, B = A
            if constraint.operands[0] in known:
                return {constraint.value: known[constraint.operands[0]]}
        
        elif constraint.type == "divisibility":
            # A % B == 0: if A is known, check divisibility
            if constraint.operands[0] in known:
                value = known[constraint.operands[0]]
                if value % constraint.value != 0:
                    raise ValueError(f"Constraint violated: {value} % {constraint.value} != 0")
        
        return {}
```

### 6.2 Constraint Validation

```python
class ConstraintValidator:
    """Validate dimension values against constraints."""
    
    def validate(self, dims: Dict[str, int], 
                constraints: List[Constraint]) -> bool:
        """Check if dimension values satisfy all constraints.
        
        Args:
            dims: Dimension values
            constraints: List of constraints
        
        Returns:
            True if all constraints are satisfied
        """
        for constraint in constraints:
            if not self._check_constraint(constraint, dims):
                return False
        return True
    
    def _check_constraint(self, constraint: Constraint, 
                         dims: Dict[str, int]) -> bool:
        """Check a single constraint."""
        if constraint.type == "equality":
            lhs = dims.get(constraint.operands[0])
            rhs = dims.get(constraint.value) if isinstance(constraint.value, str) else constraint.value
            return lhs == rhs
        
        elif constraint.type == "inequality":
            lhs = dims.get(constraint.operands[0])
            rhs = dims.get(constraint.value) if isinstance(constraint.value, str) else constraint.value
            
            if constraint.operator == ">":
                return lhs > rhs
            elif constraint.operator == ">=":
                return lhs >= rhs
            elif constraint.operator == "<":
                return lhs < rhs
            elif constraint.operator == "<=":
                return lhs <= rhs
        
        elif constraint.type == "divisibility":
            lhs = dims.get(constraint.operands[0])
            return lhs % constraint.value == 0
        
        return True
```

---

## 7. Hardware-Specific Constraints

### 7.1 Hardware Constraint Profiles

Different hardware has different constraints on dimensions:

```python
@dataclass
class HardwareConstraintProfile:
    """Hardware-specific dimension constraints."""
    hardware_target: str                    # "nvidia_ampere", "ascend_a3"
    max_threads_per_block: int              # 1024 for NVIDIA, 1024 for Ascend
    max_shared_memory: int                  # bytes
    warp_size: int                          # 32 for NVIDIA, 32 for Ascend
    constraints: List[Constraint]           # hardware-specific constraints

# Example: NVIDIA Ampere
NVIDIA_AMPERE_CONSTRAINTS = HardwareConstraintProfile(
    hardware_target="nvidia_ampere",
    max_threads_per_block=1024,
    max_shared_memory=98304,  # 96 KB
    warp_size=32,
    constraints=[
        Constraint(type="divisibility", operands=["num_threads"], 
                  operator="%", value=32),  # threads must be multiple of warp size
        Constraint(type="inequality", operands=["num_threads"], 
                  operator="<=", value=1024),  # max threads per block
    ]
)

# Example: Ascend A3
ASCEND_A3_CONSTRAINTS = HardwareConstraintProfile(
    hardware_target="ascend_a3",
    max_threads_per_block=1024,
    max_shared_memory=131072,  # 128 KB
    warp_size=32,
    constraints=[
        Constraint(type="divisibility", operands=["num_threads"], 
                  operator="%", value=32),
        Constraint(type="inequality", operands=["num_threads"], 
                  operator="<=", value=1024),
    ]
)
```

### 7.2 Hardware-Aware Constraint Checking

```python
class HardwareAwareConstraintValidator:
    """Validate dimensions against hardware constraints."""
    
    def validate_for_hardware(self, dims: Dict[str, int], 
                             hardware_target: str) -> bool:
        """Check if dimensions are valid for target hardware."""
        profile = self.get_hardware_profile(hardware_target)
        
        # Check hardware-specific constraints
        for constraint in profile.constraints:
            if not self._check_constraint(constraint, dims):
                return False
        
        return True
    
    def get_hardware_profile(self, hardware_target: str) -> HardwareConstraintProfile:
        """Get hardware constraint profile."""
        profiles = {
            "nvidia_ampere": NVIDIA_AMPERE_CONSTRAINTS,
            "ascend_a3": ASCEND_A3_CONSTRAINTS,
        }
        return profiles.get(hardware_target)
```

---

## 8. Examples

### 8.1 Example 1: matmul with Symbolic Dimensions

```arke
kernel matmul {
    inputs: [
        A: f32[M, K],
        B: f32[K, N]
    ]
    outputs: [
        C: f32[M, N]
    ]
    
    where {
        M > 0,
        N > 0,
        K > 0
    }
    
    compute: C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
}

# Semantic IR
{
    "kernel_id": "matmul",
    "inputs": [
        {"name": "A", "shape": ["M", "K"], "dtype": "f32"},
        {"name": "B", "shape": ["K", "N"], "dtype": "f32"}
    ],
    "outputs": [
        {"name": "C", "shape": ["M", "N"], "dtype": "f32"}
    ],
    "constraints": [
        {"type": "inequality", "operands": ["M"], "operator": ">", "value": 0},
        {"type": "inequality", "operands": ["N"], "operator": ">", "value": 0},
        {"type": "inequality", "operands": ["K"], "operator": ">", "value": 0}
    ]
}
```

### 8.2 Example 2: Attention with Complex Shapes

```arke
kernel attention {
    inputs: [
        Q: f32[batch, num_heads, seq_len, head_dim],
        K: f32[batch, num_heads, seq_len, head_dim],
        V: f32[batch, num_heads, seq_len, head_dim]
    ]
    outputs: [
        O: f32[batch, num_heads, seq_len, head_dim]
    ]
    
    where {
        batch > 0,
        num_heads > 0,
        seq_len > 0,
        head_dim > 0,
        head_dim % 8 == 0
    }
    
    compute: O = softmax(Q @ K^T / sqrt(head_dim)) @ V
}

# Shape inference
inputs = {
    "Q": ["batch", "num_heads", "seq_len", "head_dim"],
    "K": ["batch", "num_heads", "seq_len", "head_dim"],
    "V": ["batch", "num_heads", "seq_len", "head_dim"]
}
output = infer_shape("attention", inputs)
# output: {"O": ["batch", "num_heads", "seq_len", "head_dim"]}
```

### 8.3 Example 3: Constraint Solving

```python
# Given constraints and known dimensions, solve for unknowns
constraints = [
    Constraint(type="equality", operands=["K"], operator="==", value="N"),
    Constraint(type="inequality", operands=["M"], operator=">", value=0),
    Constraint(type="divisibility", operands=["batch"], operator="%", value=8)
]

known_dims = {"M": 64, "N": 128}

solver = ConstraintSolver()
solution = solver.solve(constraints, known_dims)
# solution: {"M": 64, "N": 128, "K": 128, "batch": 8, 16, 24, ...}
```

---

## 9. Implementation Notes

### 9.1 Symbolic Dimension Representation

```python
from typing import Union

# Dimension can be concrete (int) or symbolic (str)
Dimension = Union[int, str]

# Shape is a list of dimensions
Shape = List[Dimension]

# Examples
shape1: Shape = [64, 128]                   # concrete
shape2: Shape = ["M", "N"]                  # symbolic
shape3: Shape = ["batch", 128, "seq_len"]   # mixed
```

### 9.2 Integration with IR

```python
# In SemanticIR
@dataclass
class SemanticIR:
    kernel_id: str
    inputs: List[TensorParam]               # shapes can be symbolic
    outputs: List[TensorParam]
    constraints: List[Constraint]           # dimension constraints
    semantic_rules: List[SemanticRule]      # shape inference rules

# In StrategyIR
@dataclass
class StrategyIR:
    kernel_id: str
    hardware_target: str
    decisions: List[Decision]               # decisions may reference symbolic dims
    constraints: List[Constraint]           # strategy-specific constraints
```

### 9.3 Testing Symbolic Dimensions

```python
# Test shape inference
def test_matmul_shape_inference():
    inputs = {
        "A": ["batch", "seq_len", "hidden_dim"],
        "B": ["hidden_dim", "out_dim"]
    }
    output = infer_shape("matmul", inputs)
    assert output["C"] == ["batch", "seq_len", "out_dim"]

# Test constraint solving
def test_constraint_solving():
    constraints = [
        Constraint(type="equality", operands=["K"], operator="==", value="N"),
    ]
    known = {"N": 128}
    solution = solver.solve(constraints, known)
    assert solution["K"] == 128

# Test hardware constraints
def test_hardware_constraints():
    dims = {"num_threads": 256}
    assert validator.validate_for_hardware(dims, "nvidia_ampere")
    assert validator.validate_for_hardware(dims, "ascend_a3")
```

---

## References

- `docs/spec/arke-ir-spec-v2.md` — IR layer definitions
- `docs/phase1/dynamic-shape-feasibility.md` — Dynamic shape feasibility analysis
- `docs/architecture/e2e-flow.md` — End-to-end flow

---

**End of Symbolic Dimension Specification**
