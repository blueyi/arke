# Arke Language Specification

> Version: 0.1.0-draft
> Status: 🚧 Draft — syntax may change before v0.1.0

---

## 1. Overview

Arke Language is a domain-specific language for describing tensor operator computations
and optimization strategies. It separates **what to compute** (kernel) from
**how to optimize** (strategy).

```arke
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

schedule fused_matmul_relu for target("nvidia_ampere") {
    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");
    fuse(ops=["matmul", "relu"], type=epilogue);
}
```

## 2. Lexical Structure

### 2.1 Keywords

```
kernel    schedule    for       target    let
return    import      as        if        else
```

### 2.2 Types

```
Scalar types:   f16  f32  f64  bf16  i8  i16  i32  i64  u8  u16  u32  u64  bool
Tensor type:    Tensor<[dim, ...], dtype>
Layout:         row_major  col_major
Memory:         global  shared  local  register
```

### 2.3 Literals

```
Integer:    42, 1024, 0xFF
Float:      3.14, 1e-3
String:     "hello", "nvidia_ampere"
Boolean:    true, false
Array:      [64, 16, 4]
```

### 2.4 Comments

```arke
// Single-line comment
/* Multi-line
   comment */
```

## 3. Kernel Definition

A kernel defines pure computation semantics — **what** to compute.

```
kernel_def := "kernel" IDENT "(" params ")" "->" return_type "{" body "}"
params     := param ("," param)*
param      := IDENT ":" type
type       := "Tensor" "<" shape "," scalar_type ">"
shape      := "[" INT ("," INT)* "]"
body       := (let_stmt | return_stmt)*
let_stmt   := "let" IDENT "=" op_call ";"
return_stmt := "return" IDENT ";"
op_call    := IDENT "(" args ")"
args       := expr ("," expr)*
```

### Constraints

- A kernel must have at least one `return` statement
- All input parameters must be used
- Operator calls must reference defined operators (see §6)
- Shape inference is performed at parse time

## 4. Schedule Definition

A schedule defines optimization strategy — **how** to optimize.

```
schedule_def := "schedule" IDENT "for" "target" "(" STRING ")" "{" directives "}"
directives   := (directive ";")*
directive    := kind "(" named_args ")" rationale?
kind         := "tile" | "reorder" | "fuse" | "parallel" | "place" | "vectorize" | "unroll"
named_args   := IDENT "=" expr ("," IDENT "=" expr)*
rationale    := "@rationale" "(" STRING ")"
```

### Decision Kinds

| Kind | Parameters | Semantics |
|:-----|:-----------|:----------|
| `tile` | `loop: str, factors: int[]` | Split a loop into outer/inner |
| `reorder` | `order: str[]` | Reorder nested loops |
| `fuse` | `ops: str[], type: str` | Fuse operators |
| `parallel` | `loops: str[], mapping: {}` | Map loops to GPU threads/blocks |
| `place` | `tensor: str, memory: str` | Assign tensor to memory level |
| `vectorize` | `loop: str, width: int` | Vectorize a loop |
| `unroll` | `loop: str, factor: int` | Unroll a loop |

### @rationale Annotation

Every optimization decision can carry a natural language explanation:

```arke
tile(loop="i", factors=[64, 16])
    @rationale("L2 cache line = 64, warp size = 16");
```

This is **not** a comment — it is preserved in the IR and attached to the decision.

## 5. Program Structure

```
program := (import_stmt | kernel_def | schedule_def)*
import_stmt := "import" STRING ("as" IDENT)? ";"
```

A `.ak` file contains one or more kernel and schedule definitions.

## 6. Built-in Operators

See `arke/ir/ops/catalog.py` for the complete operator catalog.

| Operator | Category | Signature |
|:---------|:---------|:----------|
| `matmul` | compute | `(Tensor[M,K], Tensor[K,N]) → Tensor[M,N]` |
| `batch_matmul` | compute | `(Tensor[B,M,K], Tensor[B,K,N]) → Tensor[B,M,N]` |
| `relu` | elementwise | `(Tensor[...]) → Tensor[...]` |
| `gelu` | elementwise | `(Tensor[...]) → Tensor[...]` |
| `add` | elementwise | `(Tensor[...], Tensor[...]) → Tensor[...]` |
| `mul` | elementwise | `(Tensor[...], Tensor[...]) → Tensor[...]` |
| `softmax` | reduce | `(Tensor[M,N]) → Tensor[M,N]` |
| `reduce_sum` | reduce | `(Tensor[M,N]) → Tensor[M]` |
| `reduce_max` | reduce | `(Tensor[M,N]) → Tensor[M]` |
| `transpose` | move | `(Tensor[M,N]) → Tensor[N,M]` |

## 7. Type System

### 7.1 Scalar Types

16 scalar types organized in 4 groups:
- **Float**: `f16`, `f32`, `f64`, `bf16`
- **Integer**: `i8`, `i16`, `i32`, `i64`
- **Unsigned**: `u8`, `u16`, `u32`, `u64`
- **Special**: `bool`, `index`

### 7.2 Tensor Types

```
Tensor<[D1, D2, ...], dtype>
```

- Shape dimensions must be positive integers (static shapes only in v0.1.0)
- Layout is optional: `Tensor<[M, N], f16, row_major>`

---

*Spec version: 0.1.0-draft | Date: 2026-03-31*
*Implementation: W5-03/04/05 (.ak parser)*
