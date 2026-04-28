# Arke Language Specification v2.0 — Design Document

> **Version:** active design note for spec v2.0
> **Status:** Active design reference (historical migration content removed)  
> **Author:** Kitty (Lead Engineer, Arke)  
> **Date:** 2026-04-09

---

## Table of Contents

1. [Overview & Design Philosophy](#1-overview--design-philosophy)
2. [File Structure](#2-file-structure)
3. [Kernel Block](#3-kernel-block)
4. [Strategy Block](#4-strategy-block)
5. [Where Clause](#5-where-clause)
6. [Annotation System](#6-annotation-system)
7. [Type System](#7-type-system)
8. [Complete EBNF Grammar](#8-complete-ebnf-grammar)
9. [Examples](#9-examples)
10. [Implementation Notes](#10-implementation-notes)

---

## 1. Overview & Design Philosophy

### 1.1 What Is Arke Language?

The Arke Language (`.ak`) is the top-level human- and LLM-facing interface to the Arke compilation pipeline. It is the entry point to Arke IR's LLM-Native multi-layer architecture (see `arke-ir-spec-design.md`). An `.ak` file describes:

1. **What to compute** — a `kernel` block encoding operator-level semantics (maps to SemanticIR)
2. **How to optimize** — an optional `strategy` block encoding optimization decisions (maps to StrategyIR)

Arke is intentionally **not** a loop-nest language. It operates at the operator abstraction level. A kernel is a named composition of semantic operators; the compiler translates it to efficient code for a target backend.

### 1.2 v2.0 Design Goals

This document describes the **canonical v2 language design** used by the active Stage 7 implementation. Historical compatibility and migration behavior are intentionally out of scope.

The key design goals are:

| Gap | v1.0 Limitation | v2.0 Solution |
|:----|:----------------|:--------------|
| Shape generality | Shapes hardcoded: `Tensor<[1024,1024], f16>` | Symbolic shapes with `where` clause |
| Multi-return ops | `topk` can't express `(values, indices)` | Tuple destructuring: `let (v, i) = topk(...)` |
| Type verbosity | Every tensor needs full explicit type | Type inference: infer output shape/dtype from inputs |
| Backend coupling | `launch_config(num_warps=4)` is Triton-specific | Backend-agnostic `compute(...)` directive |
| Shape-regime conditionals | No conditional strategy selection | `when`/`otherwise` blocks in strategy |
| Import system | Reserved but undefined | Defined module import syntax |

### 1.3 Design Principles

1. **Operator-level abstraction** — `.ak` never expresses loops, thread indices, or memory addresses. Those are the compiler's concern.
2. **LLM-Native** — The grammar is regular, simple, and unambiguous. LLMs can read, understand, and generate `.ak` without context about compiler internals.
3. **Token efficiency** — Shorter than equivalent Triton. New features (symbolic shapes, type inference) should *reduce* token count for most kernels.
4. **Single source of truth** — `.ak` is the canonical source format. JSON IR is the serialization format for compiler internals and Agent API, never a format humans author directly.
5. **@rationale everywhere** — Every optimization decision can carry a rationale annotation. This feeds the learning loop for the LLM Agent.
6. **Canonical surface** — Active language design should describe the current surface directly, not preserve legacy aliases or migration shims.

### 1.4 Relationship to IR Layers

```
.ak (v2.0)
    │
    ▼
Layer 4 — SemanticIR     (operator graph + symbolic shapes)     [LLM: primary author]
    │
    ▼
Layer 3 — StrategyIR     (optimization decisions, L1/L2/L3)    [LLM: decision-maker]
    │
    ▼
Layer 2 — ScheduleIR     (thread/block/warp/vector mapping)    [LLM: review only]
    │
    ▼
Layer 1 — InstructionIR  (near-LLVM IR)                        [LLM: none]
    │
    ▼
LLVM IR / MLIR standard dialects
```

The `.ak` kernel block maps directly to Layer 4 (SemanticIR). The `.ak` strategy block maps directly to Layer 3 (StrategyIR decisions). v2 symbolic shapes and `where` clauses are first-class in Layer 4.

Arke IR can lower through MLIR standard dialects (`linalg`, `transform`, `scf`, `gpu`) or directly to LLVM IR. See `arke-ir-spec-design.md` §10 for MLIR integration details.

---

## 2. File Structure

```
.ak file = (import_stmt | kernel_def | strategy_def)*
```

Unchanged from v1.0. A file may contain zero or more top-level items in any order. The typical pattern is one `kernel` + one optional `strategy`.

### 2.1 Comments

```
// single-line comment
/* multi-line
   block comment */
```

### 2.2 Identifiers

Identifiers are `[a-zA-Z_][a-zA-Z0-9_]*`. They are case-sensitive.

### 2.3 String Literals

Strings are double-quoted: `"text"`. Escape sequences: `\"`, `\\`, `\n`, `\t`.

---

## 3. Kernel Block

### 3.1 Syntax

```
kernel <name>(<param_list>) -> <return_type_or_tuple> <where_clause>? {
    <body>
}
```

The `where` clause is new in v2.0 and is optional. It follows the return type and precedes the body.

### 3.2 Parameters

```
param_list = param ("," param)*
param      = <name> : <type>
```

Types are described in full in §7. The key addition in v2.0 is that dimension sizes can be symbolic names rather than integer literals:

```
// v1.0 (still valid)
kernel relu_v1(X: Tensor<[1024, 768], f16>) -> Tensor<[1024, 768], f16> { ... }

// v2.0 with symbolic shapes
kernel relu_v2(X: Tensor<[B, S, D], f16>) -> Tensor<[B, S, D], f16>
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{ ... }
```

### 3.3 Return Type

The return type can be:

- A single tensor type: `-> Tensor<[B, S], f16>`
- An inferred single type: `-> _` (compiler infers from body)
- A tuple of types: `-> (Tensor<[B, K], f16>, Tensor<[B, K], i32>)`
- A tuple with partial inference: `-> (_, _)` or omitted for fully-inferred multi-return

**New in v2.0:** Tuple return types. If the return type uses symbolic names, those names must be declared in the `where` clause.

### 3.4 Body

The body is a sequence of `let` statements followed by a `return`:

```
body = let_stmt* return_stmt
```

**v1.0 let (still valid):**
```
let <var> = <op_call> ;
```

**v2.0 tuple destructuring (new):**
```
let (<var1>, <var2>) = <op_call> ;
let (<var1>, <var2>, <var3>) = <op_call> ;
```

**v2.0 type inference (new):**

In v1.0 the return type was always explicit. In v2.0 the return type may be `_` (a single underscore), signifying that the compiler should infer it from the body. The parser accepts `_` wherever a `tensor_type` is expected.

**Return statement:**
```
// v1.0: single variable
return <var> ;

// v2.0: tuple return (new)
return (<var1>, <var2>) ;
```

### 3.5 Operator Calls

```
op_call  = <op_name>(<arg_list>)
arg_list = named_arg ("," named_arg)*
named_arg = <name> = <value>
```

Values: variable name, integer literal, float literal, string literal, bool (`true`/`false`), array literal `[v1, v2, ...]`.

**New ops in v2.0 catalog:**

| Op | Category | Inputs | Returns | Description |
|:---|:---------|:-------|:--------|:------------|
| `topk` | E | X, k | `(values, indices)` | Top-k elements along last axis |
| `flash_attention` | A | Q, K, V | output | Fused attention kernel |
| `embedding` | A | X, weight | output | Token embedding lookup |
| `concat` | A | inputs, dim | output | Tensor concatenation |
| `slice` | A | X, dim, start, end | output | Tensor slice |
| `cast` | D | X, dtype | output | Dtype cast |
| `dropout` | D | X, p | output | Dropout |
| `sigmoid` | D | X | output | Sigmoid activation |

All v1.0 operators remain supported and unchanged.

### 3.6 Annotations on Kernel (new in v2.0)

A kernel block may carry annotations immediately before the `kernel` keyword:

```
@constraint(dtypes="f16|bf16|f32")
@meta(category="OT4", fusion_hint="epilogue")
@input_gen(dist="normal", range=[0, 1])
kernel my_kernel(...) -> ... { ... }
```

See §6 for the full annotation system.

---

## 4. Strategy Block

### 4.1 Syntax

```
strategy <name> for target("<hw_target>") {
    <strategy_body>
}
```

Unchanged structurally from v1.0. The changes are in the available directives and the addition of conditional blocks.

### 4.2 Hardware Targets

Defined target strings (case-insensitive):

| String | Hardware |
|:-------|:---------|
| `"nvidia_ampere"` | NVIDIA Ampere (SM 8.x) |
| `"nvidia_hopper"` | NVIDIA Hopper (SM 9.x) |
| `"nvidia_volta"` | NVIDIA Volta (SM 7.0) |
| `"ascend_910b"` | Huawei Ascend 910B |
| `"amd_cdna2"` | AMD CDNA2 (MI200 series) |
| `"cpu_generic"` | Generic CPU (AVX2 fallback) |

### 4.3 Strategy Directives

Active strategy directives in the Stage 7 mainline:

| Directive | Parameters | Effect |
|:----------|:-----------|:-------|
| `tile` | canonical loop/dim selector + factors | Tile a loop or semantic dimension |
| `reorder` | `order` | Reorder loop nest |
| `parallel` | `loops`, `mapping` | Map loops to HW threads/blocks |
| `fuse` | `ops`, `fusion_type` | Operator fusion |
| `vectorize` | `loop`, `width` | Vectorize a loop |
| `place` / `memory_layout` | tensor placement params | Tensor memory placement / layout hints |
| `compute` | `warps`, `num_stages`, `shared_memory` | Backend-agnostic resource directive surface |
| `unroll` | `loop`, `factor` | Loop unrolling |
| `autotune` | `configs`, `key` | Mark for autotuning |
| `algorithm` | `name` | Algorithm variant selection |

**`compute`** — canonical resource directive in the active language surface. It carries resource intent without exposing backend-specific directive names:

```
compute(warps=4, num_stages=3, shared_memory="auto")
    @rationale("4 warps with a 3-stage pipeline balance occupancy and latency hiding");
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `warps` | INT | Canonical parallel worker count for SIMT backends |
| `num_stages` | INT | Software pipeline stages for latency hiding |
| `shared_memory` | STRING/INT | Shared-memory intent or explicit budget when supported |

**`memory_layout`** — Backend-agnostic memory placement:

```
memory_layout(tensor="A", level="l1", access_pattern="sequential")
    @rationale("A accessed sequentially; prefetch into L1 for bandwidth");
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `tensor` | STRING | Tensor name |
| `level` | STRING | Memory level: `"register"`, `"l1"`, `"l2"`, `"global"` |
| `access_pattern` | STRING | `"sequential"`, `"strided"`, `"random"` |

**`precision`** — Mixed-precision control:

```
precision(accumulate="f32", output="f16")
    @rationale("accumulate in f32 for numerical stability, cast output to f16");
```

### 4.4 Conditional Strategy Blocks (new in v2.0)

A strategy can contain `when`/`otherwise` blocks to select optimization parameters based on shape regime at **compile time** (when shapes are static or known ranges):

```
strategy flash_attn_strategy for target("nvidia_ampere") {
    fuse(ops=["flash_attention"], fusion_type="triton_kernel")
        @rationale("full fused attention — one kernel from Q/K/V to output");

    when S <= 512 {
        tile(loop="S", factors=[64])
            @rationale("short seqlens: 64-tile fits query block in registers");
        compute(parallelism=32, pipeline_depth=2)
            @rationale("short seqlens saturate with fewer warps");
    }
    otherwise {
        tile(loop="S", factors=[128])
            @rationale("long seqlens: 128-tile amortizes memory overhead");
        compute(parallelism=128, pipeline_depth=3)
            @rationale("long seqlens need full pipeline to hide HBM latency");
    }
}
```

**Condition expressions:**

```
condition = IDENT ("<=" | "<" | ">=" | ">" | "==" | "!=") INT
          | condition "and" condition
          | condition "or" condition
          | "(" condition ")"
```

Conditions may only reference dimension names declared in the kernel's `where` clause. Compound conditions with `and`/`or` are supported.

**Semantics:**
- `when <cond> { ... } otherwise { ... }` — mutually exclusive branches
- `when <cond> { ... }` — optional branch with implicit no-op otherwise
- Multiple `when` blocks without `otherwise` are evaluated in order; the first matching block wins (like a match/switch)
- If a shape is `static`, the condition is resolved at parse time

---

## 5. Where Clause

### 5.1 Purpose

The `where` clause declares symbolic dimension names used in the kernel's tensor types. It replaces hardcoded integer dimensions and allows a single `.ak` file to describe a kernel for a family of shapes.

### 5.2 Syntax

```
where_clause = "where" dim_decl ("," dim_decl)*
dim_decl     = IDENT ":" dim_kind
dim_kind     = "dynamic" "(" dynamic_opts ")"
             | "static"
             | "dynamic"

dynamic_opts = dynamic_opt ("," dynamic_opt)*
dynamic_opt  = "max" "=" INT
             | "min" "=" INT
             | "multiple_of" "=" INT
             | "default" "=" INT
```

### 5.3 Dimension Kinds

| Kind | Meaning | Compiler behavior |
|:-----|:--------|:------------------|
| `static` | Fixed at compile time, value unknown to `.ak` but constant per compilation | Enable static specialization |
| `dynamic` | Value varies at runtime | Generate dynamic shapes code |
| `dynamic(max=N)` | Dynamic, bounded by N | Enable range-based optimizations |
| `dynamic(min=M, max=N)` | Dynamic, in range [M, N] | Full bounds information |
| `dynamic(multiple_of=K)` | Dynamic, always a multiple of K | Enable alignment-based vectorization |
| `dynamic(max=N, multiple_of=K)` | Combined constraints | Full information |

### 5.4 Examples

```ak
// Batch-variable transformer attention
kernel scaled_dot_product_attention(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where
    B: dynamic(max=64),
    H: static,
    S: dynamic(max=8192, multiple_of=64),
    D: static
{ ... }
```

```ak
// Fixed batch, variable sequence
kernel decode_step(
    X: Tensor<[1, S, D], f16>,
    W: Tensor<[D, D], f16>
) -> Tensor<[1, S, D], f16>
where S: dynamic(max=4096), D: static
{ ... }
```

### 5.5 Scope

Symbolic dimension names are scoped to the `kernel` block that declares them. A strategy block references them in `when` conditions using the same names. The compiler matches kernel and strategy by name convention (`<kernel_name>_strategy` or explicit `for <kernel_name>`).

### 5.6 No Where Clause → Legacy Static Shapes

If no `where` clause is present, all dimension sizes in the tensor types must be integer literals. Symbolic dimensions require an explicit `where` clause in the active language surface.

---

## 6. Annotation System

### 6.1 Overview

Annotations are `@key(...)` markers attached to kernel definitions, strategy directives, or strategy blocks. They are **non-executable metadata** — they do not change compilation semantics, but are threaded through the pipeline for tooling, testing, and LLM agent guidance.

### 6.2 Annotation Placement

```
// On kernel definition
@constraint(dtypes="f16|bf16")
@meta(category="OT4")
kernel my_kernel(...) { ... }

// On strategy directive
tile(loop="M", factors=[32])
    @rationale("M is small — 32-tile keeps register pressure low");

// Multiple annotations on one directive
fuse(ops=["matmul", "gelu"], fusion_type="epilogue")
    @rationale("save global memory roundtrip")
    @meta(perf_impact="high");
```

### 6.3 Standard Annotations

#### `@rationale`

**Purpose:** Human or LLM reasoning attached to an optimization decision.

```
@rationale("<text>")
```

- Present on strategy directives
- Preserved in all IR layers, generated code comments, and trajectory logs
- Required for any strategy generated by the LLM Agent (enforced by agent prompt)
- Not required for human-authored strategies, but strongly encouraged

#### `@constraint`

**Purpose:** Data type constraints on the kernel.

```
@constraint(dtypes="f16|bf16|f32")
@constraint(dtypes="f16|bf16", min_sm=80)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `dtypes` | STRING | Pipe-separated dtype names the kernel supports |
| `min_sm` | INT | Minimum CUDA Streaming Multiprocessor version required |
| `min_ascend` | STRING | Minimum Ascend version required |

The compiler emits an error if a constraint is violated by the build target.

#### `@meta`

**Purpose:** Metadata tags for categorization, tooling, and documentation.

```
@meta(category="OT4", fusion_hint="epilogue")
@meta(category="OT1", perf_impact="high", source="g6-redesign")
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `category` | STRING | Op category code from benchmark framework (e.g., "OT1"-"OT5") |
| `fusion_hint` | STRING | Fusion opportunity hint: `"prologue"`, `"epilogue"`, `"standalone"` |
| `perf_impact` | STRING | Expected performance impact: `"low"`, `"medium"`, `"high"` |
| `source` | STRING | Provenance identifier (e.g., which design doc or agent run) |

#### `@input_gen`

**Purpose:** Test data generation hints for the benchmark and correctness-check harness.

```
@input_gen(dist="normal", range=[0, 1])
@input_gen(dist="uniform", range=[-1, 1], seed=42)
@input_gen(dist="integer", range=[0, 50000])
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `dist` | STRING | Distribution: `"normal"`, `"uniform"`, `"integer"`, `"zeros"`, `"ones"`, `"eye"` |
| `range` | ARRAY | `[min, max]` for uniform/integer; `[mean, std]` for normal |
| `seed` | INT | Random seed for reproducibility |

This annotation is on the kernel definition and applies to all inputs unless overridden per-parameter (future feature).

#### `@deprecated`

**Purpose:** Mark a kernel or strategy as deprecated.

```
@deprecated(since="2.0", replace_with="scaled_dot_product_attention_v2")
kernel scaled_dot_product_attention_v1(...) { ... }
```

### 6.4 Custom Annotations

Any `@key(...)` not in the standard list is a **custom annotation**. The parser accepts all well-formed annotations. Unrecognized annotations are preserved as opaque metadata in the JSON IR and do not cause parse errors. This allows tooling to extend the annotation system without modifying the language.

---

## 7. Type System

### 7.1 Scalar Types

| Type | Width | Description |
|:-----|:------|:------------|
| `f16` | 16-bit | IEEE half-precision float |
| `bf16` | 16-bit | Brain float (1-sign, 8-exp, 7-mantissa) |
| `f32` | 32-bit | IEEE single-precision float |
| `f64` | 64-bit | IEEE double-precision float |
| `i8` | 8-bit | Signed integer |
| `i16` | 16-bit | Signed integer |
| `i32` | 32-bit | Signed integer |
| `i64` | 64-bit | Signed integer |
| `u8` | 8-bit | Unsigned integer |
| `u16` | 16-bit | Unsigned integer |
| `u32` | 32-bit | Unsigned integer |
| `u64` | 64-bit | Unsigned integer |
| `bool` | 1-bit | Boolean |
| `index` | arch | Platform-native index type |

### 7.2 Tensor Types

```
tensor_type = "Tensor" "<" "[" dim_list "]" "," scalar_type ("," layout)? ">"
            | "_"

dim_list = dim ("," dim)*
dim      = INT               // static integer dimension (v1.0)
         | IDENT             // symbolic dimension (v2.0, must be in where clause)

layout   = "row_major" | "col_major"   // default: row_major
```

**New in v2.0:** `dim` can be a symbolic name. The `_` type (inference hole) is also new.

### 7.3 Tuple Types

```
tuple_type = "(" tensor_type ("," tensor_type)+ ")"
```

Used in multi-return kernels. Only appears in return type position; parameters are always individual named tensors.

### 7.4 Type Inference Rules

Type inference is a v2.0 feature. When a return type or intermediate binding uses `_`, the compiler infers the type according to these rules:

| Situation | Rule |
|:----------|:-----|
| `let Y = relu(X=Z)` | Y gets same type as Z (elementwise passthrough) |
| `let Y = gelu(X=Z)` | Y gets same type as Z |
| `let Y = softmax(X=Z)` | Y gets same type as Z |
| `let Y = add(A=X, B=W)` | Y gets type of X (X and W must match) |
| `let Y = matmul(A=X, B=W)` | Y shape is `[X.dim[0], W.dim[1]]`, dtype from X |
| `let (v,i) = topk(X=Z, k=K)` | v has same dtype as Z, shape `[..., K]`; i has dtype `i32`, same shape |
| Return type `_` | Inferred from the variable returned |

Inference is **shallow** — it follows operator-level rules, not full dataflow analysis. If inference is ambiguous or unsupported for a given operator, the compiler emits an error requiring an explicit type annotation.

### 7.5 Symbolic Dimension Propagation

When a tensor's dimension is symbolic, the compiler tracks the dimension name through the operator graph:

- **Passthrough ops** (relu, gelu, softmax, etc.): output dimensions = input dimensions
- **matmul(A, B)**: output dims = `[A.dim[0], B.dim[1]]`; inner dims must match
- **topk(X, k)**: output dims = `[X.dim[0], ..., X.dim[-2], k]`
- **transpose(X)**: output dims = reverse of input dims (2D only in v1.0/v2.0)

If the compiler cannot symbolically determine output shape from inputs, a type annotation is required.

---

## 8. Complete EBNF Grammar

The following grammar sketch highlights the canonical v2 surface used by the active mainline.

```ebnf
(* ============================================================ *)
(* Arke Language v2.0 — Complete EBNF                          *)
(* ============================================================ *)

start          = top_level_item*
top_level_item = import_stmt
               | annotation* kernel_def
               | strategy_def

(* ── Import ─────────────────────────────────────────────── *)
import_stmt    = "import" STRING ("as" IDENT)? ";"

(* ── Kernel Definition ───────────────────────────────────── *)
kernel_def     = "kernel" IDENT "(" param_list? ")" "->" return_type
                 where_clause?
                 "{" kernel_body "}"

param_list     = param ("," param)*
param          = IDENT ":" tensor_type

return_type    = tensor_type
               | infer_type
               | "(" tensor_type ("," tensor_type)+ ")"
               | "(" infer_type  ("," infer_type )+  ")"

infer_type     = "_"

(* ── Where Clause ────────────────────────────────────────── *)
where_clause   = "where" dim_decl ("," dim_decl)*
dim_decl       = IDENT ":" dim_kind
dim_kind       = "static"
               | "dynamic"
               | "dynamic" "(" dynamic_opts ")"

dynamic_opts   = dynamic_opt ("," dynamic_opt)*
dynamic_opt    = "max" "=" INT
               | "min" "=" INT
               | "multiple_of" "=" INT
               | "default" "=" INT

(* ── Type System ─────────────────────────────────────────── *)
tensor_type    = "Tensor" "<" "[" dim_list "]" "," scalar_type ("," layout)? ">"

dim_list       = dim ("," dim)*
dim            = INT | IDENT

layout         = "row_major" | "col_major"

scalar_type    = "f16" | "bf16" | "f32" | "f64"
               | "i8"  | "i16"  | "i32" | "i64"
               | "u8"  | "u16"  | "u32" | "u64"
               | "bool" | "index"

(* ── Kernel Body ─────────────────────────────────────────── *)
kernel_body    = let_stmt* return_stmt

let_stmt       = "let" lhs "=" op_call ";"
lhs            = IDENT
               | "(" IDENT ("," IDENT)+ ")"

return_stmt    = "return" return_expr ";"
return_expr    = IDENT
               | "(" IDENT ("," IDENT)+ ")"

(* ── Operator Calls ──────────────────────────────────────── *)
op_call        = IDENT "(" arg_list? ")"
arg_list       = named_arg ("," named_arg)*
named_arg      = IDENT "=" arg_value

arg_value      = IDENT
               | INT
               | FLOAT
               | STRING
               | BOOL
               | "[" (arg_value ("," arg_value)*)? "]"

(* ── Strategy Definition ─────────────────────────────────── *)
strategy_def   = "strategy" IDENT "for" "target" "(" STRING ")" "{" strategy_body "}"

strategy_body  = strategy_item*
strategy_item  = strategy_stmt
               | when_block

strategy_stmt  = IDENT "(" strategy_kwargs? ")" annotation* ";"?

strategy_kwargs = strategy_kwarg ("," strategy_kwarg)*
strategy_kwarg  = IDENT "=" strategy_value

strategy_value = STRING | INT | FLOAT | BOOL | IDENT
               | "[" (strategy_value ("," strategy_value)*)? "]"
               | "{" (strategy_map_entry ("," strategy_map_entry)*)? "}"

strategy_map_entry = (STRING | IDENT) ":" strategy_value

(* ── Conditional Strategy (new in v2.0) ──────────────────── *)
when_block     = when_arm+ otherwise_arm?
when_arm       = "when" condition "{" strategy_body "}"
otherwise_arm  = "otherwise" "{" strategy_body "}"

condition      = condition "and" condition
               | condition "or" condition
               | "(" condition ")"
               | IDENT cmp_op INT

cmp_op         = "<=" | "<" | ">=" | ">" | "==" | "!="

(* ── Annotations ─────────────────────────────────────────── *)
annotation     = "@" IDENT "(" annotation_args? ")"
annotation_args = annotation_arg ("," annotation_arg)*
annotation_arg  = IDENT "=" annotation_value
                | STRING

annotation_value = STRING | INT | FLOAT | BOOL | IDENT
                 | "[" (annotation_value ("," annotation_value)*)? "]"

(* ── Lexical Tokens ──────────────────────────────────────── *)
IDENT   = /[a-zA-Z_][a-zA-Z0-9_]*/
INT     = /[0-9]+/
FLOAT   = /[0-9]+\.[0-9]*/
STRING  = /"([^"\\]|\\.)*"/
BOOL    = "true" | "false"

(* Line comments: // ...  Block comments: /* ... */ *)
```

---

## 9. Examples

### 9.1 Symbolic Shape MatMul + GeLU

Demonstrates symbolic shapes, type inference, and backend-agnostic strategy.

```ak
@constraint(dtypes="f16|bf16|f32")
@meta(category="OT1", fusion_hint="epilogue")
kernel matmul_gelu(
    X: Tensor<[M, K], f16>,
    W: Tensor<[K, N], f16>
) -> _
where M: dynamic(max=4096), K: static, N: static
{
    let Z = matmul(A=X, B=W);
    let Y = gelu(X=Z);
    return Y;
}

strategy matmul_gelu_strategy for target("nvidia_ampere") {
    fuse(ops=["matmul", "gelu"], fusion_type="epilogue")
        @rationale("apply gelu in matmul epilogue — saves global memory roundtrip");

    when M <= 128 {
        tile(loop="M", factors=[32])
            @rationale("M<=128: small M, 32-tile keeps register pressure low");
        tile(loop="N", factors=[64])
            @rationale("small regime: 64-tile N balances occupancy");
        compute(parallelism=32, pipeline_depth=2)
            @rationale("fewer warps sufficient for small M");
    }
    otherwise {
        tile(loop="M", factors=[128])
            @rationale("large M: 128-tile for better compute intensity");
        tile(loop="N", factors=[128])
            @rationale("128-tile N for large output");
        compute(parallelism=128, pipeline_depth=3)
            @rationale("3-stage pipeline hides HBM latency for large matmul");
    }

    tile(loop="K", factors=[32])
        @rationale("K-tile=32: smem footprint 32*32*2*2=4KB fits in L1");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("each block owns one (M,N) tile");
}
```

### 9.2 TopK with Multi-Return

Demonstrates tuple return, multi-variable destructuring, and type inference.

```ak
@constraint(dtypes="f16|bf16|f32")
@meta(category="OT4")
@input_gen(dist="normal", range=[0, 1])
kernel top_candidates(
    scores: Tensor<[B, V], f16>
) -> (_, _)
where B: dynamic(max=512), V: static
{
    let (values, indices) = topk(X=scores, k=50);
    return (values, indices);
}

strategy top_candidates_strategy for target("nvidia_ampere") {
    tile(loop="B", factors=[32])
        @rationale("batch over 32 rows per block");
    compute(parallelism=64, pipeline_depth=2)
        @rationale("topk is latency-bound; 2-stage pipeline adequate");
    memory_layout(tensor="scores", level="l1", access_pattern="sequential")
        @rationale("row-major access; sequential prefetch into L1");
}
```

### 9.3 Transformer Attention with Symbolic Batch/Sequence

Demonstrates multi-dimensional symbolic shapes, static dimensions, and shape-regime strategy.

```ak
@constraint(dtypes="f16|bf16", min_sm=80)
@meta(category="OT3", fusion_hint="standalone")
@input_gen(dist="normal", range=[0, 1])
kernel scaled_dot_product_attention(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where
    B: dynamic(max=64),
    H: static,
    S: dynamic(max=8192, multiple_of=64),
    D: static
{
    let attn_out = flash_attention(Q=Q, K=K, V=V);
    return attn_out;
}

strategy sdpa_strategy for target("nvidia_ampere") {
    fuse(ops=["flash_attention"], fusion_type="triton_kernel")
        @rationale("flash attention is a single fused triton kernel");

    when S <= 512 {
        tile(loop="S", factors=[64])
            @rationale("short seqlen: 64-tile, query fits in registers");
        compute(parallelism=32, pipeline_depth=2)
            @rationale("short seqlens: fewer warps, no deep pipeline needed");
    }
    when S <= 2048 {
        tile(loop="S", factors=[128])
            @rationale("medium seqlen: 128-tile balances smem usage");
        compute(parallelism=64, pipeline_depth=3)
            @rationale("medium seqlens: standard 3-stage pipeline");
    }
    otherwise {
        tile(loop="S", factors=[128])
            @rationale("long seqlen: 128-tile amortizes memory overhead");
        compute(parallelism=128, pipeline_depth=4)
            @rationale("long seqlens: full pipeline to hide HBM latency");
    }

    parallel(loops=["B", "H"], mapping={"B": "blockIdx.x", "H": "blockIdx.y"})
        @rationale("each block handles one (batch, head) pair");
}
```

### 9.4 LayerNorm with Dtype Constraint and Input Gen

Demonstrates per-kernel annotations and combined backend-agnostic strategy.

```ak
@constraint(dtypes="f16|bf16|f32")
@meta(category="OT2")
@input_gen(dist="normal", range=[0, 1])
kernel layer_norm(
    X:      Tensor<[B, S, H], f16>,
    weight: Tensor<[H], f32>,
    bias:   Tensor<[H], f32>
) -> Tensor<[B, S, H], f16>
where
    B: dynamic(max=64),
    S: dynamic(max=4096),
    H: static
{
    let Y = layernorm(X=X, weight=weight, bias=bias, eps=1e-5);
    return Y;
}

strategy layer_norm_strategy for target("nvidia_ampere") {
    tile(loop="H", factors=[256])
        @rationale("H-tile=256: process hidden dim in chunks for register reuse");
    parallel(loops=["B", "S"], mapping={"B": "blockIdx.x", "S": "blockIdx.y"})
        @rationale("each block handles one (batch, seq) row");
    precision(accumulate="f32", output="f16")
        @rationale("accumulate in f32 for numerical stability, cast back to f16");
    compute(parallelism=32, pipeline_depth=2)
        @rationale("layernorm is reduction-bound; 32 warps with 2-stage pipeline");
}
```

### 9.5 RMSNorm with Import

Demonstrates the import system and a kernel using an op from an imported module.

```ak
import "arke://ops/normalization" as norm;

@constraint(dtypes="f16|bf16|f32")
@meta(category="OT2")
kernel rms_norm(
    X:      Tensor<[B, S, D], f16>,
    weight: Tensor<[D], f32>
) -> Tensor<[B, S, D], f16>
where B: dynamic(max=64), S: dynamic(max=4096), D: static
{
    let Y = rmsnorm(X=X, weight=weight, eps=1e-6);
    return Y;
}

strategy rms_norm_strategy for target("nvidia_ampere") {
    tile(loop="D", factors=[128])
        @rationale("D-tile=128 for register reuse during RMS accumulation");
    parallel(loops=["B", "S"], mapping={"B": "blockIdx.x", "S": "blockIdx.y"})
        @rationale("each block handles one row");
    compute(parallelism=16, pipeline_depth=2)
        @rationale("RMSNorm is lightweight; 16 warps sufficient");
}
```

---

---

## 10. Implementation Notes

This design document remains useful only as an **active architectural explanation** of the canonical v2 language surface.

Implementation rules for the current mainline:

1. `compute(...)` is the only active resource directive surface in strategy blocks.
2. `launch_config(...)` and other Triton-specific directive names are not part of the active language contract.
3. `where` clauses, tuple returns, conditional strategies, and `@rationale` are first-class language features.
4. Tests and examples should be rewritten to canonical v2 syntax rather than preserved through migration shims.
5. If a historical syntax note is still needed, keep it in git history or an external archive, not in active design references.

For the normative language definition, see `docs/spec/arke-lang-spec.md`.
