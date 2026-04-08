# Arke Language Specification v2.0

> **Version:** 2.0.0  
> **Status:** Final Specification  
> **Date:** 2026-04-08  
> **Based on:** Arke Language Spec v1.0 (frozen) + v2.0 Design Document

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Structure](#2-file-structure)
3. [Kernel Block](#3-kernel-block)
4. [Strategy Block](#4-strategy-block)
5. [Where Clause](#5-where-clause)
6. [Type System](#6-type-system)
7. [Annotation System](#7-annotation-system)
8. [Complete EBNF Grammar](#8-complete-ebnf-grammar)
9. [All 45 Operators](#9-all-45-operators)
10. [Backward Compatibility](#10-backward-compatibility)

---

## 1. Overview

### 1.1 Purpose

Arke Language (`.ak`) is the human- and LLM-facing interface to the Arke compilation pipeline. It describes:

1. **What to compute** — kernel block encoding operator-level semantics (→ SemanticIR Layer 4)
2. **How to optimize** — strategy block encoding optimization decisions (→ StrategyIR Layer 3)

### 1.2 Design Principles

- **Operator-level abstraction** — No loops, thread indices, or memory addresses
- **LLM-Native** — Regular, simple, unambiguous grammar
- **Token efficient** — Shorter than equivalent Triton
- **Single source of truth** — `.ak` is canonical; JSON IR is serialization only
- **@rationale everywhere** — Every decision carries rationale for LLM learning
- **Backward compatible** — v2.0 is a superset of v1.0

### 1.3 v2.0 New Features

| Feature | Purpose |
|:--------|:--------|
| `where` clause | Symbolic dimensions and dynamic shapes |
| Tuple returns | Multi-output operators like `topk` |
| Type inference | Infer output shape/dtype from inputs |
| Backend-agnostic directives | `compute(...)` instead of Triton-specific `launch_config` |
| Conditional strategies | `when`/`otherwise` blocks for shape-based dispatch |
| Import system | Module imports for code reuse |

---

## 2. File Structure

```ebnf
ak_file = (import_stmt | kernel_def | strategy_def)*
```

A `.ak` file contains zero or more top-level items in any order. Typical pattern: one `kernel` + one optional `strategy`.

### 2.1 Comments

```
// single-line comment
/* multi-line
   comment */
```

---

## 3. Kernel Block

### 3.1 Syntax

```ebnf
kernel_def = "kernel" IDENT "(" param_list? ")" "->" return_type where_clause? "{" kernel_body "}"

param_list = param ("," param)*
param = IDENT ":" type_expr

return_type = type_expr | tuple_return_type | infer_type
tuple_return_type = "(" type_expr ("," type_expr)+ ")"
infer_type = "_"

kernel_body = (let_stmt | return_stmt)*
let_stmt = "let" IDENT ("," IDENT)* "=" expr ";"
return_stmt = "return" expr_list ";"
expr_list = expr ("," expr)*
```

### 3.2 Semantics

- **Kernel name** — Unique identifier for the operator
- **Parameters** — Input tensors with explicit types
- **Return type** — Output tensor type(s); `_` infers from computation
- **Where clause** — Declares symbolic dimensions (see §5)
- **Body** — Sequence of let-bindings and return statement

### 3.3 Example

```ak
kernel matmul(
    A: Tensor<[M, K], f32>,
    B: Tensor<[K, N], f32>
) -> Tensor<[M, N], f32>
where M: dynamic(max=4096), K: static, N: dynamic(max=4096)
{
    let C = matmul(A=A, B=B);
    return C;
}
```

---

## 4. Strategy Block

### 4.1 Syntax

```ebnf
strategy_def = "strategy" IDENT "for" target_spec "{" strategy_body "}"

target_spec = "target" "(" STRING ")"

strategy_body = (decision | conditional_block)*

decision = directive ("@" annotation)*
directive = tile_directive | launch_config_directive | compute_directive | ...

conditional_block = "when" condition "{" strategy_body "}" ("otherwise" "{" strategy_body "}")?
condition = shape_condition | constraint_condition
```

### 4.2 Directives

- **`tile(...)`** — Tiling factors for loops
- **`launch_config(...)`** — Triton-specific (deprecated in v2.0)
- **`compute(...)`** — Backend-agnostic compute resource specification
- **`@rationale(...)`** — Justification for the decision

### 4.3 Example

```ak
strategy matmul_strategy for target("nvidia_ampere") {
    tile(loop="m", factors=[128])
        @rationale("128 threads per block for occupancy");
    tile(loop="n", factors=[128])
        @rationale("Balanced M/N tiling");
    compute(num_threads=256, num_stages=3)
        @rationale("3-stage pipeline for memory latency hiding");
}
```

---

## 5. Where Clause

### 5.1 Syntax

```ebnf
where_clause = "where" dim_decl ("," dim_decl)*

dim_decl = IDENT ":" dim_kind
dim_kind = "static"
         | "dynamic"
         | "dynamic" "(" dynamic_opts ")"

dynamic_opts = "min" "=" INT | "max" "=" INT | "min" "=" INT "," "max" "=" INT
```

### 5.2 Semantics

- **`static`** — Dimension is compile-time constant
- **`dynamic`** — Dimension is runtime variable
- **`dynamic(min=..., max=...)`** — Bounded dynamic dimension

### 5.3 Scope

Symbolic dimensions declared in `where` clause are:
- Visible in kernel parameter types
- Propagated through SemanticIR
- Used in strategy conditions
- Preserved in StrategyIR for backend code generation

### 5.4 Examples

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

## 6. Type System

### 6.1 Scalar Types

```
f16, bf16, f32, f64
i8, i16, i32, i64
u8, u16, u32, u64
bool
```

### 6.2 Tensor Types

```ebnf
tensor_type = "Tensor" "<" "[" dim_list "]" "," dtype ">"
dim_list = dim ("," dim)*
dim = INT | IDENT
dtype = scalar_type
```

Dimensions can be:
- **Concrete integers** — `Tensor<[128, 64], f32>`
- **Symbolic names** — `Tensor<[B, S, D], f16>` (must be declared in `where` clause)

### 6.3 Tuple Types

```ebnf
tuple_type = "(" type_expr ("," type_expr)+ ")"
```

Example: `(Tensor<[N], i32>, Tensor<[N], f32>)` for `topk` returning `(indices, values)`.

### 6.4 Type Inference

When return type is `_`, the compiler infers output type from operation semantics:

```ak
kernel gelu(X: Tensor<[B, S, D], f32>) -> _
{
    let Y = gelu(X=X);
    return Y;  // Inferred: Tensor<[B, S, D], f32>
}
```

---

## 7. Annotation System

### 7.1 Syntax

```ebnf
annotation = "@" IDENT | "@" IDENT "(" STRING ")"
```

### 7.2 Standard Annotations

| Annotation | Context | Purpose |
|:-----------|:--------|:--------|
| `@rationale(...)` | Kernel, Strategy | Justification for design choice |
| `@constraint(...)` | Kernel | Shape/dtype constraints |
| `@meta(...)` | Kernel | Metadata (e.g., category, complexity) |
| `@input_gen(...)` | Kernel | Input generation strategy for benchmarking |

### 7.3 Examples

```ak
@rationale("Fused matmul+relu for LLM inference")
kernel matmul_relu(
    A: Tensor<[M, K], f32>,
    B: Tensor<[K, N], f32>,
    bias: Tensor<[N], f32>
) -> Tensor<[M, N], f32>
@constraint("M % 128 == 0 for tiling")
@meta("category=compute-dense, complexity=high")
{
    let C = matmul(A=A, B=B);
    let D = add(C=C, bias=bias);
    let E = relu(X=D);
    return E;
}
```

---

## 8. Complete EBNF Grammar

```ebnf
(* Arke Language v2.0 EBNF *)

ak_file = (import_stmt | kernel_def | strategy_def)*

(* Imports *)
import_stmt = "import" STRING ("as" IDENT)? ";"

(* Kernel Definition *)
kernel_def = annotation* "kernel" IDENT "(" param_list? ")" "->" return_type where_clause? "{" kernel_body "}"

param_list = param ("," param)*
param = IDENT ":" type_expr

return_type = type_expr | tuple_return_type | infer_type
tuple_return_type = "(" type_expr ("," type_expr)+ ")"
infer_type = "_"

where_clause = "where" dim_decl ("," dim_decl)*
dim_decl = IDENT ":" dim_kind
dim_kind = "static" | "dynamic" | "dynamic" "(" dynamic_opts ")"
dynamic_opts = ("min" "=" INT)? ("," "max" "=" INT)?

kernel_body = (let_stmt | return_stmt)*
let_stmt = "let" IDENT ("," IDENT)* "=" expr ";"
return_stmt = "return" expr_list ";"
expr_list = expr ("," expr)*

expr = IDENT | call_expr | tuple_expr
call_expr = IDENT "(" arg_list? ")"
arg_list = arg ("," arg)*
arg = IDENT "=" expr
tuple_expr = "(" expr ("," expr)+ ")"

(* Strategy Definition *)
strategy_def = "strategy" IDENT "for" target_spec "{" strategy_body "}"
target_spec = "target" "(" STRING ")"
strategy_body = (decision | conditional_block)*

decision = directive annotation*
directive = tile_directive | launch_config_directive | compute_directive
tile_directive = "tile" "(" tile_args ")"
tile_args = "loop" "=" STRING "," "factors" "=" "[" INT ("," INT)* "]"
launch_config_directive = "launch_config" "(" launch_args ")"
launch_args = "num_warps" "=" INT ("," "num_stages" "=" INT)?
compute_directive = "compute" "(" compute_args ")"
compute_args = "num_threads" "=" INT ("," "num_stages" "=" INT)?

conditional_block = "when" condition "{" strategy_body "}" ("otherwise" "{" strategy_body "}")?
condition = shape_condition | constraint_condition
shape_condition = IDENT ">" INT | IDENT "<" INT | IDENT "==" INT
constraint_condition = STRING

(* Types *)
type_expr = tensor_type | scalar_type | tuple_type
tensor_type = "Tensor" "<" "[" dim_list "]" "," dtype ">"
dim_list = dim ("," dim)*
dim = INT | IDENT
dtype = scalar_type
scalar_type = "f16" | "bf16" | "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool"
tuple_type = "(" type_expr ("," type_expr)+ ")"

(* Annotations *)
annotation = "@" IDENT | "@" IDENT "(" STRING ")"

(* Tokens *)
IDENT = [a-zA-Z_][a-zA-Z0-9_]*
INT = [0-9]+
STRING = '"' [^"]* '"'
```

---

## 9. All 45 Operators

### OT0: Elementwise (12 ops)

| Op | Signature | Example |
|:---|:----------|:--------|
| relu | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = relu(X=X);` |
| gelu | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = gelu(X=X);` |
| silu | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = silu(X=X);` |
| tanh | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = tanh(X=X);` |
| sigmoid | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = sigmoid(X=X);` |
| exp | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = exp(X=X);` |
| neg | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = neg(X=X);` |
| rsqrt | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = rsqrt(X=X);` |
| add | `Tensor<[...], f> × Tensor<[...], f> → Tensor<[...], f>` | `let Z = add(X=X, Y=Y);` |
| mul | `Tensor<[...], f> × Tensor<[...], f> → Tensor<[...], f>` | `let Z = mul(X=X, Y=Y);` |
| cast | `Tensor<[...], f1> → Tensor<[...], f2>` | `let Y = cast(X=X, dtype=f32);` |
| where_ | `Tensor<[...], bool> × Tensor<[...], f> × Tensor<[...], f> → Tensor<[...], f>` | `let Z = where_(Cond=C, X=X, Y=Y);` |

### OT1: Reduction (10 ops)

| Op | Signature | Example |
|:---|:----------|:--------|
| softmax | `Tensor<[B, S, D], f> → Tensor<[B, S, D], f>` | `let Y = softmax(X=X, axis=2);` |
| layernorm | `Tensor<[B, S, D], f> → Tensor<[B, S, D], f>` | `let Y = layernorm(X=X, weight=w, bias=b);` |
| rmsnorm | `Tensor<[B, S, D], f> → Tensor<[B, S, D], f>` | `let Y = rmsnorm(X=X, weight=w);` |
| reduce_sum | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = reduce_sum(X=X, axis=1);` |
| reduce_max | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = reduce_max(X=X, axis=1);` |
| reduce_mean | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = reduce_mean(X=X, axis=1);` |
| argmax | `Tensor<[...], f> → Tensor<[...], i32>` | `let Y = argmax(X=X, axis=1);` |
| topk | `Tensor<[B, N], f> → (Tensor<[B, K], f>, Tensor<[B, K], i32>)` | `let (vals, inds) = topk(X=X, k=10);` |
| cumsum | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = cumsum(X=X, axis=1);` |
| cross_entropy | `Tensor<[B, C], f> × Tensor<[B], i32> → Tensor<[B], f>` | `let loss = cross_entropy(logits=L, labels=Y);` |

### OT2: Compute-Dense (11 ops)

| Op | Signature | Example |
|:---|:----------|:--------|
| matmul | `Tensor<[M, K], f> × Tensor<[K, N], f> → Tensor<[M, N], f>` | `let C = matmul(A=A, B=B);` |
| batch_matmul | `Tensor<[B, M, K], f> × Tensor<[B, K, N], f> → Tensor<[B, M, N], f>` | `let C = batch_matmul(A=A, B=B);` |
| grouped_matmul | `Tensor<[G, M, K], f> × Tensor<[G, K, N], f> → Tensor<[G, M, N], f>` | `let C = grouped_matmul(A=A, B=B);` |
| quantize_per_token | `Tensor<[B, S, D], f> → (Tensor<[B, S, D], i8>, Tensor<[B, S], f>)` | `let (Q, scale) = quantize_per_token(X=X);` |
| dequantize_per_channel | `Tensor<[M, N], i8> × Tensor<[N], f> → Tensor<[M, N], f>` | `let X = dequantize_per_channel(Q=Q, scale=s);` |
| rope | `Tensor<[B, S, D], f> → Tensor<[B, S, D], f>` | `let Y = rope(X=X, theta=10000.0);` |
| embedding | `Tensor<[B, S], i32> × Tensor<[V, D], f> → Tensor<[B, S, D], f>` | `let E = embedding(ids=ids, weight=W);` |
| transpose | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = transpose(X=X, axes=[1, 0]);` |
| permute | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = permute(X=X, dims=[0, 2, 1]);` |
| concat | `Tensor<[...], f> × ... → Tensor<[...], f>` | `let Z = concat(X=X, Y=Y, axis=1);` |
| split | `Tensor<[...], f> → (Tensor<[...], f>, ...)` | `let (X1, X2) = split(X=X, axis=1, sizes=[64, 64]);` |

### OT3: Gated Activation (7 ops)

| Op | Signature | Example |
|:---|:----------|:--------|
| swiglu | `Tensor<[B, S, 2D], f> → Tensor<[B, S, D], f>` | `let Y = swiglu(X=X);` |
| geglu | `Tensor<[B, S, 2D], f> → Tensor<[B, S, D], f>` | `let Y = geglu(X=X);` |
| rmsnorm_residual | `Tensor<[B, S, D], f> × Tensor<[B, S, D], f> → Tensor<[B, S, D], f>` | `let Y = rmsnorm_residual(X=X, residual=R, weight=w);` |
| gather | `Tensor<[N, D], f> × Tensor<[M], i32> → Tensor<[M, D], f>` | `let Y = gather(X=X, indices=idx);` |
| scatter | `Tensor<[N, D], f> × Tensor<[M], i32> × Tensor<[M, D], f> → Tensor<[N, D], f>` | `let Y = scatter(X=X, indices=idx, updates=U);` |
| copy_ | `Tensor<[...], f> → Tensor<[...], f>` | `let Y = copy_(X=X);` |
| matmul_gelu | `Tensor<[M, K], f> × Tensor<[K, N], f> → Tensor<[M, N], f>` | `let C = matmul_gelu(A=A, B=B);` |

### OT4: Attention (5 ops)

| Op | Signature | Example |
|:---|:----------|:--------|
| flash_attention | `Tensor<[B, H, S, D], f> × Tensor<[B, H, S, D], f> × Tensor<[B, H, S, D], f> → Tensor<[B, H, S, D], f>` | `let O = flash_attention(Q=Q, K=K, V=V);` |
| grouped_query_attention | `Tensor<[B, H, S, D], f> × Tensor<[B, 1, S, D], f> × Tensor<[B, 1, S, D], f> → Tensor<[B, H, S, D], f>` | `let O = grouped_query_attention(Q=Q, K=K, V=V);` |
| multi_latent_attention | `Tensor<[B, H, S, D], f> × Tensor<[B, H, S, R], f> → Tensor<[B, H, S, D], f>` | `let O = multi_latent_attention(Q=Q, KV=KV);` |
| cross_attention | `Tensor<[B, H, S, D], f> × Tensor<[B, H, L, D], f> × Tensor<[B, H, L, D], f> → Tensor<[B, H, S, D], f>` | `let O = cross_attention(Q=Q, K=K, V=V);` |
| paged_attention | `Tensor<[B, H, S, D], f> × Tensor<[B, H, P, D], f> × Tensor<[B, H, P, D], f> → Tensor<[B, H, S, D], f>` | `let O = paged_attention(Q=Q, K_pages=K, V_pages=V);` |

---

## 10. Backward Compatibility

### 10.1 v1.0 Compatibility

Every valid v1.0 `.ak` file is a valid v2.0 `.ak` file. v2.0 features are optional:

- `where` clause is optional; omit it for static shapes
- Tuple returns are optional; single returns work as before
- Type inference (`_`) is optional; explicit types still work
- Backend-agnostic directives are optional; Triton-specific directives still parse

### 10.2 Migration Path

To upgrade v1.0 code to v2.0:

1. Add `where` clause for symbolic dimensions (optional)
2. Replace `launch_config(...)` with `compute(...)` (optional)
3. Use tuple destructuring for multi-output ops (optional)
4. Add `@rationale` annotations (optional)

Example v1.0 → v2.0:

```ak
// v1.0
kernel matmul(A: Tensor<[1024, 512], f32>, B: Tensor<[512, 1024], f32>) -> Tensor<[1024, 1024], f32> {
    let C = matmul(A=A, B=B);
    return C;
}

// v2.0 (backward compatible, but with symbolic shapes)
kernel matmul(A: Tensor<[M, K], f32>, B: Tensor<[K, N], f32>) -> Tensor<[M, N], f32>
where M: dynamic(max=4096), K: static, N: dynamic(max=4096)
{
    let C = matmul(A=A, B=B);
    return C;
}
```

---

## References

- `docs/spec/arke-ir-spec-v2.md` — IR multi-layer architecture
- `docs/phase1/dynamic-shape-feasibility.md` — Symbolic shape design rationale
- `docs/architecture/e2e-flow.md` — End-to-end LLM optimization flow
- `examples/operators/` — All 45 operator `.ak` examples

---

**End of Arke Language Specification v2.0**
