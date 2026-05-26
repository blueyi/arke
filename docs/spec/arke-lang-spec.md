# Arke Language Specification v0.1.0

> **Version:** 0.1.0
> **Status:** Final Specification
> **Date:** 2026-04-09
> **Scope:** Canonical v0.1.0 language surface for the active compiler pipeline
> **Philosophy:** Universal operator abstraction, LLM-native, zero algorithm-specific constructs

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [File Structure](#3-file-structure)
4. [Kernel Block](#4-kernel-block)
5. [Strategy Block](#5-strategy-block)
6. [Where Clause & Symbolic Dimensions](#6-where-clause--symbolic-dimensions)
7. [Type System](#7-type-system)
8. [Annotation System](#8-annotation-system)
9. [Decision Space](#9-decision-space)
10. [Complete EBNF Grammar](#10-complete-ebnf-grammar)
11. [Versioning](#11-versioning)

---

## 1. Overview

### 1.1 Purpose

Arke Language (`.ak`) is the human- and LLM-facing interface to the Arke compilation pipeline. It describes:

1. **What to compute** — kernel block encoding operator-level semantics (→ SemanticIR Layer 4)
2. **How to optimize** — strategy block encoding optimization decisions (→ StrategyIR Layer 3)

### 1.2 Core Principle: Universal Operator Abstraction

Arke Language is **algorithm-agnostic**. It does not enumerate specific operators (matmul, relu, attention, etc.). Instead, it defines:

- **Universal operation syntax** — how to invoke any operator
- **Universal symbolic dimension system** — how to express shape constraints
- **Universal decision space** — how to express optimization strategies
- **Universal annotation system** — how to attach rationale and metadata

Specific operators are registered via the **Op Registry** (see `op-registry-interface.md`), not hardcoded in the language.

### 1.3 Core Features

| Feature | Purpose |
|:--------|:--------|
| `where` clause | Symbolic dimensions and dynamic shape constraints |
| Tuple returns | Multi-output operators via destructuring |
| Type inference | Infer output shape/dtype from inputs |
| Backend-agnostic directives | Universal `compute(...)` instead of backend-specific configs |
| Conditional strategies | `when`/`otherwise` blocks for shape-dependent dispatch |
| Import system | Module imports for code reuse |

---

## 2. Design Philosophy

### 2.1 Principles

1. **Operator-level abstraction** — No loops, thread indices, or memory addresses. Those are compiler concerns.
2. **LLM-Native** — Regular, simple, unambiguous grammar. LLMs can read, understand, and generate `.ak` without compiler internals knowledge.
3. **Token efficient** — Shorter than equivalent Triton. Symbolic shapes and type inference reduce verbosity.
4. **Single source of truth** — `.ak` is canonical; JSON IR is serialization only.
5. **@rationale everywhere** — Every optimization decision carries rationale for LLM learning and knowledge transfer.
6. **Canonical surface** — This spec describes the current language directly as the starting contract for Arke-Lang.
7. **Algorithm-agnostic** — Language defines universal constructs; specific operators are registered externally.

### 2.2 Relationship to IR Layers

```
.ak (v0.1.0)
    │
    ├─ kernel block ──────────────────────────────────────────► Layer 4: SemanticIR
    │  (what to compute, pure math, no optimization)
    │
    └─ strategy block ─────────────────────────────────────────► Layer 3: StrategyIR
       (how to optimize, LLM decisions with @rationale)
       │
       ├─ Layer 2: ScheduleIR (thread/block/warp mapping, mostly automated)
       │
       └─ Layer 1: InstructionIR (near-LLVM IR, fully automated)
```

---

## 3. File Structure

```ebnf
ak_file = (import_stmt | kernel_def | strategy_def)*
```

A `.ak` file contains zero or more top-level items in any order. Typical pattern: one `kernel` + one optional `strategy`.

### 3.1 Comments

```
// single-line comment

/* multi-line
   comment */
```

### 3.2 Imports

```ebnf
import_stmt = "import" STRING ("as" IDENT)?
```

Example:
```ak
import "arke.ops.linalg" as linalg
import "arke.ops.attention"
```

---

## 4. Kernel Block

### 4.1 Syntax

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

### 4.2 Semantics

- **Kernel name** — Unique identifier for the operator
- **Parameters** — Input tensors with explicit types
- **Return type** — Output tensor type(s); `_` infers from computation
- **Where clause** — Declares symbolic dimensions (see §6)
- **Body** — Sequence of let-bindings and return statement

### 4.3 Operation Invocation

Operations are invoked by name with named arguments. The compiler resolves the operation name via the Op Registry.

```ebnf
expr = op_call | tensor_var | literal
op_call = IDENT "(" arg_list? ")"
arg_list = arg ("," arg)*
arg = IDENT "=" expr
```

Example:
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

### 4.4 Multi-Output Operations

Operations can return multiple values via tuple destructuring:

```ak
kernel topk_op(
    X: Tensor<[N], f32>,
    K: i32
) -> (Tensor<[K], f32>, Tensor<[K], i32>)
{
    let (values, indices) = topk(X=X, k=K);
    return values, indices;
}
```

### 4.5 Type Inference

Return type can be inferred from the operation:

```ak
kernel relu(X: Tensor<[B, S, D], f16>) -> _
{
    let Y = relu(X=X);
    return Y;
}
```

The compiler infers the return type as `Tensor<[B, S, D], f16>` based on the `relu` operation's signature.

---

## 5. Strategy Block

### 5.1 Syntax

```ebnf
strategy_def = "strategy" IDENT "for" target_spec "{" strategy_body "}"

target_spec = "target" "(" STRING ")"

strategy_body = (decision | conditional_block)*

decision = directive ("@" annotation)*
directive = tiling_directive | compute_directive | fusion_directive | memory_layout_directive | ...

conditional_block = "when" condition "{" strategy_body "}" ("otherwise" "{" strategy_body "}")?
condition = shape_condition | constraint_condition
```

### 5.2 Universal Decision Types

Strategies express optimization decisions using universal directives, not algorithm-specific ones:

#### 5.2.1 Tiling

```
tile(dim=<name>, factors=[<int>, ...])
```

Specifies tiling factors for a named dimension. The compiler maps this to loop tiling in the target backend.

Example:
```ak
tile(dim="M", factors=[128])
    @rationale("128 threads per block for occupancy");
```

#### 5.2.2 Compute Resource Specification

```
compute(warps=<int>, num_stages=<int>, shared_memory=<int>)
```

Backend-agnostic specification of compute resources. The compiler lowers this to backend-specific launch/resource configuration.

Example:
```ak
compute(warps=8, num_stages=3, shared_memory=49152)
    @rationale("3-stage pipeline for memory latency hiding");
```

#### 5.2.3 Fusion

```
fuse(ops=[<op_name>, ...], fusion_type=<fusion_kind>)
```

Specifies which operations should be fused together. `ops` may name explicit SemanticIR node ops, or a compact registered fused op's logical inner ops when the surface is represented as one canonical op (for example `silu_and_mul` as `silu` + `mul`, or `geglu` as `gelu` + `mul`).

Example:
```ak
fuse(ops=["matmul", "relu"], fusion_type="epilogue")
    @rationale("Fuse matmul+relu to reduce memory bandwidth");
```

Gated activation example:
```ak
fuse(ops=["silu", "mul"], fusion_type="epilogue")
    @rationale("Keep SwiGLU split, activation, and multiply inside one kernel");
```

#### 5.2.4 Memory Layout

```
memory_layout(tensor=<name>, layout=<layout_type>)
```

Specifies memory layout for a tensor (e.g., row-major, column-major, blocked).

Example:
```ak
memory_layout(tensor="A", layout="row_major")
    @rationale("Row-major for coalesced memory access");
```

#### 5.2.5 Compute Order

```
compute_order(ops=[<op_name>, ...])
```

Specifies the order in which operations are computed.

Example:
```ak
compute_order(ops=["load_A", "load_B", "compute", "store_C"])
    @rationale("Load both inputs before compute to hide latency");
```

### 5.3 Conditional Strategies

Strategies can be conditional based on shape properties:

```ak
strategy matmul_strategy for target("nvidia_ampere") {
    when shape(M) > 512 {
        tile(dim="M", factors=[256])
            @rationale("Larger tiles for large M");
    }
    otherwise {
        tile(dim="M", factors=[128])
            @rationale("Smaller tiles for small M");
    }
}
```

### 5.4 Example

```ak
strategy matmul_strategy for target("nvidia_ampere") {
    tile(dim="M", factors=[128])
        @rationale("128 threads per block for occupancy");
    tile(dim="N", factors=[128])
        @rationale("Balanced M/N tiling");
    compute(warps=8, num_stages=3)
        @rationale("3-stage pipeline for memory latency hiding");
    memory_layout(tensor="A", layout="row_major")
        @rationale("Coalesced memory access");
}
```

---

## 6. Where Clause & Symbolic Dimensions

### 6.1 Syntax

```ebnf
where_clause = "where" dim_decl ("," dim_decl)*

dim_decl = IDENT ":" dim_kind
dim_kind = "static"
         | "dynamic"
         | "dynamic" "(" dynamic_opts ")"

dynamic_opts = "min" "=" INT | "max" "=" INT | "min" "=" INT "," "max" "=" INT
```

### 6.2 Semantics

- **`static`** — Dimension is compile-time constant. Value is known at kernel definition time.
- **`dynamic`** — Dimension is runtime variable. Value is determined at kernel invocation.
- **`dynamic(min=..., max=...)`** — Bounded dynamic dimension. Runtime value is guaranteed to be within [min, max].

### 6.3 Scope and Propagation

Symbolic dimensions declared in `where` clause are:
- Visible in kernel parameter types
- Propagated through SemanticIR via shape inference
- Used in strategy conditions for shape-dependent dispatch
- Preserved in StrategyIR for backend code generation

### 6.4 Examples

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
    let O = attention(Q=Q, K=K, V=V);
    return O;
}

// Mixed static and dynamic
kernel batch_matmul(
    A: Tensor<[B, M, K], f32>,
    B: Tensor<[B, K, N], f32>
) -> Tensor<[B, M, N], f32>
where B: dynamic(max=32), M: static, K: static, N: static
{
    let C = batch_matmul(A=A, B=B);
    return C;
}
```

---

## 7. Type System

### 7.1 Scalar Types

```
f16, bf16, f32, f64
i8, i16, i32, i64
u8, u16, u32, u64
bool
```

### 7.2 Tensor Types

```ebnf
tensor_type = "Tensor" "<" "[" dim_list "]" "," dtype ">"
dim_list = dim ("," dim)*
dim = INT | IDENT
dtype = scalar_type
```

Dimensions can be:
- **Concrete integers** — `Tensor<[128, 64], f32>`
- **Symbolic names** — `Tensor<[B, S, D], f16>` (must be declared in `where` clause)

### 7.3 Tuple Types

```ebnf
tuple_type = "(" type_expr ("," type_expr)+ ")"
```

Example: `(Tensor<[N], i32>, Tensor<[N], f32>)`

### 7.4 Type Inference

Return type can be inferred using `_`:

```ak
kernel some_op(X: Tensor<[B, S, D], f32>) -> _
{
    let Y = some_op(X=X);
    return Y;
}
```

The compiler infers the return type based on the operation's signature.

---

## 8. Annotation System

### 8.1 Syntax

```ebnf
annotation = "@" IDENT ("(" STRING ")")?
```

### 8.2 Standard Annotations

#### 8.2.1 @rationale

Explains the reasoning behind a decision. Used in strategy blocks.

```ak
tile(dim="M", factors=[128])
    @rationale("128 threads per block maximizes occupancy on Ampere");
```

#### 8.2.2 @constraint

Specifies constraints on a dimension or operation.

```ak
kernel matmul(
    A: Tensor<[M, K], f32>,
    B: Tensor<[K, N], f32>
) -> Tensor<[M, N], f32>
where M: dynamic(max=4096) @constraint("M must be divisible by 128"),
      K: static @constraint("K must be power of 2"),
      N: dynamic(max=4096)
{
    let C = matmul(A=A, B=B);
    return C;
}
```

#### 8.2.3 @meta

Attaches metadata (e.g., performance hints, hardware requirements).

```ak
kernel matmul(...) -> _ @meta("requires_tensor_core=true")
{
    ...
}
```

#### 8.2.4 @input_gen

Specifies how to generate test inputs for this kernel.

```ak
kernel matmul(...) -> _ @input_gen("random_uniform(0, 1)")
{
    ...
}
```

### 8.3 Custom Annotations

Users can define custom annotations. The compiler ignores unknown annotations but preserves them in IR for downstream tools.

---

## 9. Decision Space

### 9.1 Universal Decision Types

The strategy block expresses optimization decisions using universal decision types. These are backend-agnostic and algorithm-agnostic:

| Decision Type | Purpose | Example |
|:---|:---|:---|
| `tile(...)` | Loop tiling | `tile(dim="M", factors=[128])` |
| `compute(...)` | Compute resource allocation | `compute(warps=8, num_stages=3)` |
| `fuse(...)` | Operation fusion | `fuse(ops=["matmul", "relu"])` |
| `memory_layout(...)` | Tensor memory layout | `memory_layout(tensor="A", layout="row_major")` |
| `compute_order(...)` | Operation execution order | `compute_order(ops=["load", "compute", "store"])` |

### 9.2 Extensibility

New decision types can be added without modifying the language grammar. They are registered in the **Decision Registry** (see `op-registry-interface.md`).

---

## 10. Complete EBNF Grammar

```ebnf
(* Top-level *)
ak_file = (import_stmt | kernel_def | strategy_def)*

import_stmt = "import" STRING ("as" IDENT)?

(* Kernel definition *)
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

(* Strategy definition *)
strategy_def = "strategy" IDENT "for" target_spec "{" strategy_body "}"
target_spec = "target" "(" STRING ")"
strategy_body = (decision | conditional_block)*
decision = directive ("@" annotation)*
directive = tiling_directive | compute_directive | fusion_directive | memory_layout_directive | compute_order_directive
tiling_directive = "tile" "(" "dim" "=" STRING "," "factors" "=" "[" INT ("," INT)* "]" ")"
compute_directive = "compute" "(" compute_args ")"
compute_args = compute_arg ("," compute_arg)*
compute_arg = IDENT "=" INT
fusion_directive = "fuse" "(" "ops" "=" "[" STRING ("," STRING)* "]" ")"
memory_layout_directive = "memory_layout" "(" "tensor" "=" STRING "," "layout" "=" STRING ")"
compute_order_directive = "compute_order" "(" "ops" "=" "[" STRING ("," STRING)* "]" ")"
conditional_block = "when" condition "{" strategy_body "}" ("otherwise" "{" strategy_body "}")?
condition = shape_condition | constraint_condition
shape_condition = "shape" "(" IDENT ")" comparison_op INT
constraint_condition = IDENT comparison_op INT
comparison_op = ">" | "<" | ">=" | "<=" | "==" | "!="

(* Where clause *)
where_clause = "where" dim_decl ("," dim_decl)*
dim_decl = IDENT ":" dim_kind
dim_kind = "static" | "dynamic" | "dynamic" "(" dynamic_opts ")"
dynamic_opts = ("min" "=" INT | "max" "=" INT | "min" "=" INT "," "max" "=" INT)

(* Type system *)
type_expr = tensor_type | tuple_type | scalar_type
tensor_type = "Tensor" "<" "[" dim_list "]" "," dtype ">"
dim_list = dim ("," dim)*
dim = INT | IDENT
dtype = scalar_type
tuple_type = "(" type_expr ("," type_expr)+ ")"
scalar_type = "f16" | "bf16" | "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool"

(* Expressions *)
expr = op_call | tensor_var | literal
op_call = IDENT "(" arg_list? ")"
arg_list = arg ("," arg)*
arg = IDENT "=" expr
tensor_var = IDENT
literal = INT | FLOAT | STRING

(* Annotations *)
annotation = "@" IDENT ("(" STRING ")")?

(* Lexical *)
IDENT = [a-zA-Z_][a-zA-Z0-9_]*
INT = [0-9]+
FLOAT = [0-9]+ "." [0-9]+
STRING = '"' [^"]* '"'
```

---

## 11. Versioning

### 11.1 Canonical Version Tag

Language-facing artifacts should identify the current schema version explicitly:

```json
{
  "version": "0.1.0",
  "schema": "arke-lang-v0.1.0"
}
```

The `version` field above is the **language schema version** and is aligned with the current Python package release line for the clean `v0.1.0` project start.

### 11.2 Relationship to Package Version

- `arke.__version__` / `pyproject.toml` track the Python package release line.
- `version: 0.1.0` and `schema: arke-lang-v0.1.0` identify the canonical `.ak` surface.
- Package release cadence and schema evolution may diverge later, but the active starting contract is aligned at `0.1.0`.
- Supporting design/spec documents should reference the same active `.ak` schema version unless they are explicitly about another subsystem.

### 11.3 Scope of This Specification

This document defines the canonical v0.1.0 language surface used by the active compiler pipeline:

- `compute(...)` is the resource directive surface
- `where` clauses define symbolic dimensions
- `when` / `otherwise` express conditional strategy branches
- non-canonical aliases are outside the language contract
- package-version history is outside the language contract

---

## References

- `docs/spec/arke-ir-spec.md` — IR multi-layer architecture
- `docs/spec/op-registry-interface.md` — Op registration and extension framework
- `docs/spec/op-catalog/semantic-ops.md` — Operator semantic definitions (reference)
- `docs/phase1/dynamic-shape-feasibility.md` — Symbolic shape design rationale
- `docs/architecture/e2e-flow.md` — End-to-end LLM optimization flow

---

**End of Arke Language Specification v0.1.0**
