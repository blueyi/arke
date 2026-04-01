# Fusion Tasks

Fusion = combining multiple operations into a single kernel to reduce memory round-trips.

## Fusion Combinations

| ID | Name | Ops | Shape (M,N,K) | Tier | Purpose |
|:---|:-----|:----|:--------------|:----:|:--------|
| F01 | fused_matmul_relu | matmul → relu | 1024,1024,1024 | 1 | Basic epilogue fusion |
| F02 | fused_matmul_gelu | matmul → gelu | 1024,2048,1024 | 1 | Complex epilogue (GELU approx) |
| F03 | fused_matmul_add | matmul → add | 1024,1024,1024 | 1 | Residual connection (C = AB + bias) |
| F04 | fused_matmul_add_relu | matmul → add → relu | 1024,1024,1024 | 2 | Three-op fusion chain |
| F05 | fused_matmul_mul | matmul → mul | 1024,1024,1024 | 2 | Element-wise scale after matmul |
| F06 | fused_softmax_mul | softmax → mul | 4096,4096 | 3 | Attention score × V pattern |

### Fusion Type Classification

**Epilogue fusion** (F01-F05): Elementwise op applied to matmul output before writing to global memory. The fused op reads from registers, not memory — saves one full read+write.

- `relu`, `gelu`: Unary activation, no extra input tensor
- `add`, `mul`: Binary op, needs a second tensor (bias/scale) loaded alongside

**Reduce + elementwise** (F06): softmax output element-wise multiplied. Tests whether reduce and elementwise can share a kernel.

### Accuracy Concerns

| Fusion | Risk | Check |
|:-------|:-----|:------|
| matmul + gelu | GELU approximation varies (tanh vs erf) | Compare against `F.gelu(matmul(A,B))` |
| matmul + add | Overflow if matmul result is large | Test with scaled inputs |
| softmax + mul | Numerical stability of softmax | Verify row-sum ≈ 1.0 before mul |

### Building Fusion Tasks

```python
# matmul + add (residual connection)
def _build_matmul_add(name, M, N, K, dtype="f16"):
    b = KernelBuilder(name)
    b.param("A", [M, K], dtype)
    b.param("B", [K, N], dtype)
    b.param("bias", [M, N], dtype)  # residual/bias tensor
    m = b.op("matmul", A="A", B="B")
    r = b.op("add", A=m, B="bias")
    b.returns(r, [M, N], dtype)
    return b.build()

# matmul + add + relu
def _build_matmul_add_relu(name, M, N, K, dtype="f16"):
    b = KernelBuilder(name)
    b.param("A", [M, K], dtype)
    b.param("B", [K, N], dtype)
    b.param("bias", [M, N], dtype)
    m = b.op("matmul", A="A", B="B")
    a = b.op("add", A=m, B="bias")
    r = b.op("relu", X=a)
    b.returns(r, [M, N], dtype)
    return b.build()
```
