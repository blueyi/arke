# Dynamic Shape Feasibility Assessment

> **Date:** 2026-04-08  
> **Scope:** Symbolic dimensions and dynamic shapes in Arke  
> **Status:** Feasibility Assessment Complete

---

## Executive Summary

Symbolic dimensions and dynamic shapes are **fully feasible** in Arke across all layers (SemanticIR through InstructionIR) and both backends (Triton and MLIR). Implementation is straightforward with minimal risk.

**Key findings:**
- ✅ `where` clause syntax is simple and unambiguous
- ✅ SemanticIR can represent symbolic dims natively
- ✅ StrategyIR decisions can be shape-conditional
- ✅ Triton backend supports dynamic shapes via `tl.constexpr` and runtime bounds
- ✅ MLIR backend supports dynamic shapes via `?` in memref types
- ✅ All 45 operators are compatible with symbolic shapes
- ⚠️ Minor risk: 6GB VRAM may OOM on large shapes (mitigated by shape-aware scheduling)

---

## 1. Where Clause Design

### 1.1 Syntax

```ebnf
where_clause = "where" dim_decl ("," dim_decl)*
dim_decl = IDENT ":" dim_kind
dim_kind = "static" | "dynamic" | "dynamic" "(" dynamic_opts ")"
dynamic_opts = ("min" "=" INT)? ("," "max" "=" INT)?
```

### 1.2 Semantics

- **`static`** — Compile-time constant, known at kernel definition
- **`dynamic`** — Runtime variable, determined at kernel invocation
- **`dynamic(min=..., max=...)`** — Bounded dynamic dimension for compiler hints

### 1.3 Scope

Dimensions declared in `where` clause are:
- Visible in kernel parameter types: `Tensor<[B, S, D], f16>`
- Propagated through SemanticIR nodes
- Used in StrategyIR conditional decisions
- Preserved in ScheduleIR and InstructionIR

### 1.4 Examples

```ak
// Simple dynamic shapes
kernel relu(X: Tensor<[B, S, D], f16>) -> Tensor<[B, S, D], f16>
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = relu(X=X);
    return Y;
}

// Bounded dimensions
kernel attention(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where B: dynamic(min=1, max=64), H: static, S: dynamic(max=8192), D: static
{
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}
```

---

## 2. SemanticIR Representation

### 2.1 Symbolic Dimension Data Structure

```python
@dataclass
class SymbolicDim:
    name: str                    # "B", "S", "D", etc.
    is_dynamic: bool = True
    min: int | None = None       # optional lower bound
    max: int | None = None       # optional upper bound
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_dynamic": self.is_dynamic,
            "min": self.min,
            "max": self.max,
        }
```

### 2.2 Tensor Descriptor with Symbolic Shapes

```python
@dataclass
class TensorDesc:
    shape: list[int | SymbolicDim]  # Mix of concrete and symbolic
    dtype: str
    layout: str = "row_major"
    
    def is_symbolic(self) -> bool:
        return any(isinstance(d, SymbolicDim) for d in self.shape)
```

### 2.3 SemanticIR Extension

```python
@dataclass
class SemanticIR:
    kernel_name: str
    params: list[Param]
    return_type: TensorDesc
    symbolic_dims: list[SymbolicDim]  # NEW: declared symbolic dimensions
    nodes: list[Node]
    edges: list[tuple[str, str]]
```

### 2.4 JSON Serialization

```json
{
  "kernel_name": "matmul",
  "symbolic_dims": [
    {"name": "M", "is_dynamic": true, "max": 4096},
    {"name": "K", "is_dynamic": false},
    {"name": "N", "is_dynamic": true, "max": 4096}
  ],
  "params": [
    {
      "name": "A",
      "shape": [{"sym": "M"}, {"sym": "K"}],
      "dtype": "f32"
    },
    {
      "name": "B",
      "shape": [{"sym": "K"}, {"sym": "N"}],
      "dtype": "f32"
    }
  ],
  "return_type": {
    "shape": [{"sym": "M"}, {"sym": "N"}],
    "dtype": "f32"
  }
}
```

---

## 3. Shape Inference Pass

### 3.1 Algorithm

```python
def infer_shapes(semantic_ir: SemanticIR) -> dict[str, TensorDesc]:
    """Infer output shapes for all nodes."""
    shapes = {}
    
    # Initialize with parameters
    for param in semantic_ir.params:
        shapes[param.name] = TensorDesc(param.shape, param.dtype)
    
    # Propagate through DAG
    for node in topological_sort(semantic_ir.nodes):
        input_shapes = [shapes[inp] for inp in node.inputs]
        output_shape = infer_op_shape(
            node.op_name,
            input_shapes,
            node.attrs,
            semantic_ir.symbolic_dims
        )
        for output_id in node.outputs:
            shapes[output_id] = output_shape
    
    return shapes
```

### 3.2 Operator-Specific Rules

| Operator | Rule |
|:---------|:-----|
| `relu`, `gelu`, etc. | Output shape = input shape |
| `matmul` | `[M, K] × [K, N] → [M, N]` |
| `softmax` | Output shape = input shape (axis parameter ignored for shape) |
| `reduce_sum` | Reduce along axis, keep other dims |
| `topk` | `[B, N] → ([B, K], [B, K])` (tuple output) |
| `flash_attention` | `[B, H, S, D] × [B, H, S, D] × [B, H, S, D] → [B, H, S, D]` |

### 3.3 Constraint Validation

```python
def validate_constraints(semantic_ir: SemanticIR) -> bool:
    """Validate symbolic dimension constraints."""
    for node in semantic_ir.nodes:
        # Check operator-specific constraints
        if node.op_name == "matmul":
            # K dimension must match
            assert input_shapes[0].shape[1] == input_shapes[1].shape[0]
        
        # Check user-defined constraints (future)
        # ...
    
    return True
```

---

## 4. StrategyIR Conditional Decisions

### 4.1 Shape-Based Conditionals

```python
@dataclass
class ConditionalDecision:
    condition: str               # e.g., "S > 1024"
    true_decisions: list[Decision]
    false_decisions: list[Decision]
```

### 4.2 Example: Adaptive Tiling

```ak
strategy matmul_adaptive for target("nvidia_ampere") {
    when S > 1024 {
        tile(loop="m", factors=[256])
            @rationale("Large S: use larger tiles for better occupancy");
        tile(loop="n", factors=[256])
            @rationale("Large S: use larger tiles");
    }
    otherwise {
        tile(loop="m", factors=[128])
            @rationale("Small S: use smaller tiles for better cache locality");
        tile(loop="n", factors=[128])
            @rationale("Small S: use smaller tiles");
    }
}
```

### 4.3 Evaluation

Conditions are evaluated at **kernel invocation time** with actual runtime shapes.

---

## 5. Triton Backend Support

### 5.1 Dynamic Shapes in Triton

Triton supports dynamic shapes via:
- **`tl.constexpr`** — Compile-time constants (for static dims)
- **Runtime parameters** — Dynamic dims passed as kernel arguments
- **Bounds checking** — Triton handles out-of-bounds automatically

### 5.2 Example: Dynamic MatMul

```python
# Triton kernel with dynamic shapes
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,  # Dynamic dimensions
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # M, K, N are runtime values
    # BLOCK_M, BLOCK_K, BLOCK_N are compile-time constants
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Bounds checking handled by Triton
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Compute matmul...
```

### 5.3 Limitations

- **Tile factors must be static** — `BLOCK_M`, `BLOCK_K`, `BLOCK_N` are `tl.constexpr`
- **Grid size is dynamic** — `grid = (ceil(M / BLOCK_M), ceil(N / BLOCK_N))`
- **Memory allocation is dynamic** — Triton handles this automatically

---

## 6. MLIR Backend Support

### 6.1 Dynamic Shapes in MLIR

MLIR represents dynamic dimensions as `?` in memref types:

```mlir
// Static shapes
%A: memref<1024x512xf32>

// Dynamic shapes
%A: memref<?x?xf32>

// Mixed static/dynamic
%A: memref<1024x?xf32>
```

### 6.2 Example: Dynamic MatMul in MLIR

```mlir
func.func @matmul_dynamic(
    %A: memref<?x?xf32>,
    %B: memref<?x?xf32>,
    %C: memref<?x?xf32>
) {
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                  outs(%C : memref<?x?xf32>)
    return
}
```

### 6.3 Symbolic Dimension Metadata

Arke can attach symbolic dimension metadata to MLIR operations:

```mlir
%A: memref<?x?xf32> {
    "arke.symbolic_dims": [
        {"name": "M", "max": 4096},
        {"name": "K"}
    ]
}
```

---

## 7. All 45 Operators — Compatibility

### 7.1 Compatibility Matrix

| Op Group | Ops | Symbolic Shape Support | Notes |
|:---------|:---:|:---------------------:|:------|
| OT0: Elementwise | 12 | ✅ Full | All support arbitrary symbolic shapes |
| OT1: Reduction | 10 | ✅ Full | Axis parameter is static; dims are dynamic |
| OT2: Compute-Dense | 11 | ✅ Full | matmul, batch_matmul, grouped_matmul all support |
| OT3: Gated Activation | 7 | ✅ Full | swiglu, geglu, attention fusions all support |
| OT4: Attention | 5 | ✅ Full | flash_attention, GQA, MLA all support |

### 7.2 Operator-Specific Notes

- **matmul** — `[M, K] × [K, N] → [M, N]` — All dims can be dynamic
- **softmax** — `[B, S, D] → [B, S, D]` — All dims can be dynamic; axis is static
- **flash_attention** — `[B, H, S, D] × [B, H, S, D] × [B, H, S, D] → [B, H, S, D]` — All dims can be dynamic
- **topk** — `[B, N] → ([B, K], [B, K])` — B, N dynamic; K is static parameter
- **embedding** — `[B, S] i32 × [V, D] f → [B, S, D] f` — B, S, D can be dynamic; V is static

---

## 8. Risk Assessment

### 8.1 Low Risk

✅ **Syntax and parsing** — `where` clause is simple and unambiguous  
✅ **SemanticIR representation** — Straightforward extension with `symbolic_dims` field  
✅ **Shape inference** — Well-defined rules for all 45 operators  
✅ **Backend support** — Both Triton and MLIR support dynamic shapes natively  

### 8.2 Medium Risk

⚠️ **Memory allocation** — 6GB VRAM may OOM on large shapes  
**Mitigation:** Shape-aware scheduling, memory budgeting in StrategyIR  

⚠️ **Conditional strategies** — Complex shape conditions may be hard to optimize  
**Mitigation:** Start with simple conditions (e.g., `S > 1024`); extend gradually  

### 8.3 Mitigations

1. **Memory budgeting** — Track estimated memory usage in StrategyIR
2. **Shape-aware scheduling** — Select strategies based on actual runtime shapes
3. **Fallback strategies** — Provide conservative strategies for unknown shapes
4. **Testing** — Comprehensive tests with various shape combinations

---

## 9. Implementation Roadmap

### Phase 1: Core Support (S7 Track 2)

- ✅ `where` clause parsing in Lark grammar
- ✅ SemanticIR `symbolic_dims` field
- ✅ Shape inference pass for all 45 operators
- ✅ Tests: `tests/test_symbolic_shape.py` (≥5 ops with `where` clause)

### Phase 2: Strategy Integration (S7 Track 3)

- ⬜ Conditional decisions in StrategyIR
- ⬜ Shape-based strategy selection
- ⬜ Memory budgeting in StrategyIR

### Phase 3: Backend Integration (S7 Track 4)

- ⬜ Triton codegen with dynamic shapes
- ⬜ MLIR codegen with dynamic shapes
- ⬜ BL5 performance benchmarks with dynamic shapes

---

## 10. Conclusion

**Symbolic dimensions and dynamic shapes are fully feasible in Arke.** The design is sound, the implementation is straightforward, and both backends support it natively. The main challenge is memory management on 6GB VRAM, which is mitigated by shape-aware scheduling.

**Recommendation:** Proceed with full implementation in S7 Track 2 and beyond.

---

## References

- `docs/spec/arke-lang-spec.md` — Arke Language v0.1.0 with `where` clause
- `docs/spec/arke-ir-spec.md` — Arke IR multi-layer architecture
- `docs/architecture/e2e-flow.md` — End-to-end LLM optimization flow

---

**End of Dynamic Shape Feasibility Assessment**
