# Arke Language Specification v1.0

> **Version:** 1.0  
> **Status:** Frozen (Stage 1 exit)  
> **Compatibility:** Arke compiler ≥ 0.1.0-dev  
> **Grammar file:** `arke/parser/arke.lark`

---

## 1. Overview

The **Arke Language** (`.ak` files) is a minimal, AI-first kernel description language. It describes:

1. **What** to compute — a `kernel` block (maps to SemanticIR)
2. **How** to optimize — an optional `strategy` block (maps to StrategyIR)

Design goals:
- **Minimal syntax**: fewer tokens than equivalent Triton/CUDA code (avg 32% in practice)
- **Readable by LLMs**: simple, regular grammar; no implicit behavior
- **Separable**: strategy is optional; compiler auto-generates from hardware profile
- **Annotatable**: `@rationale` attaches reasoning to every optimization decision

---

## 2. File Structure

```
.ak file = (import_stmt | kernel_def | strategy_def)*
```

An `.ak` file contains zero or more top-level definitions in any order. A typical file has one `kernel` + one optional `strategy`.

### 2.1 Comments

```
// single-line comment
/* multi-line
   block comment */
```

---

## 3. Kernel Block

### Syntax

```
kernel <name>(<param_list>) -> <return_type> {
    <body>
}
```

### Parameters

```
param_list = param ("," param)*
param      = <name> : <tensor_type>

tensor_type = Tensor<[<dims>], <dtype>>
            | Tensor<[<dims>], <dtype>, <layout>>

dims   = INT ("," INT)*
dtype  = f16 | f32 | f64 | bf16 | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | bool | index
layout = row_major | col_major  (default: row_major)
```

### Body

The body is a sequence of `let` statements followed by a `return`:

```
let <var> = <op_call> ;
return <var> ;
```

### Operator Calls

```
op_call = <op_name>(<arg_list>)
arg_list = <name>=<value> ("," <name>=<value>)*
```

Values can be: variable name, integer literal, float literal, string literal, bool (`true`/`false`), or array literal `[v1, v2, ...]`.

### Supported Operators (P0 Catalog)

| Op | Category | Inputs | Description |
|:---|:---------|:-------|:------------|
| `matmul` | A | A, B | C = A × B |
| `batch_matmul` | A | A, B | batched C = A × B |
| `softmax` | C | X | row-wise softmax along last axis |
| `layernorm` | C | X, weight, bias, eps | layer normalization |
| `rmsnorm` | C | X, weight, eps | RMS normalization |
| `relu` | D | X | max(x, 0) |
| `gelu` | D | X | GELU activation |
| `silu` | D | X | x × sigmoid(x) |
| `add` | D | A, B | elementwise add |
| `mul` | D | A, B | elementwise multiply |
| `reduce_sum` | E | X | sum along last axis |
| `reduce_max` | E | X | max along last axis |
| `transpose` | A | X | 2D transpose |

> **Note:** Semantic input names (`weight`, `bias`) are accepted as aliases for internal names (`W`, `B`) by the compiler.

### Example

```ak
kernel matmul_gelu(
    X: Tensor<[128, 768], f16>,
    W: Tensor<[768, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Z = matmul(A=X, B=W);
    let Y = gelu(X=Z);
    return Y;
}
```

---

## 4. Strategy Block (Optional)

If omitted, the Arke compiler automatically generates a hardware-aware default strategy using `DefaultStrategyGenerator`.

### Syntax

```
strategy <name> for target("<hw_target>") {
    <strategy_body>
}
```

The `<name>` convention is `<kernel_name>_strategy`. The compiler also matches `<kernel_name>` exactly or falls back to the only strategy in the file.

### Strategy Body

```
strategy_body = (strategy_action annotation? ";"?)*
strategy_action = <kind>(<kwarg_list>)
kwarg_list = <name>=<value> ("," <name>=<value>)*
annotation = @<key>("<text>")
```

Values support: `STRING`, `INT`, `FLOAT`, `BOOL`, `array = [...]`, `map = {...}`.

### Strategy Decision Kinds

| Kind | Required params | Effect |
|:-----|:----------------|:-------|
| `tile` | `loop`, `factors` | Tile a loop by given factors |
| `reorder` | `order` | Reorder loop nest |
| `parallel` | `loops`, `mapping` | Map loops to hardware threads/blocks |
| `fuse` | `ops`, `fusion_type` | Fuse operators into one kernel |
| `vectorize` | `loop`, `width` | Vectorize a loop |
| `place` | `tensor`, `memory` | Place a tensor in a memory level |
| `launch_config` | `num_warps`, `num_stages` | Set GPU launch parameters |
| `unroll` | `loop`, `factor` | Unroll a loop |
| `autotune` | `configs`, `key` | Mark for autotuning |
| `algorithm` | `name` | Select algorithm variant |

### Annotations

`@rationale("text")` attaches human-readable explanation to any decision. This is preserved through the full pipeline and appears in:
- Generated Triton code (as `# rationale: <text>`)
- Strategy IR JSON (`rationale.text`)
- Agent trajectory JSONL logs

### Example

```ak
strategy matmul_gelu_strategy for target("nvidia_ampere") {
    tile(loop="M", factors=[32])
        @rationale("M=128 small — 32 tile keeps register pressure low");
    tile(loop="N", factors=[128])
        @rationale("N=3072: 128-tile, multiple blocks cover output columns");
    tile(loop="K", factors=[32])
        @rationale("K-tile=32: A+B smem = 32*32*2*2 = 4096B, fits in L1");
    parallel(loops=["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("each block owns one (M,N) tile");
    fuse(ops=["matmul", "gelu"], fusion_type="epilogue")
        @rationale("apply gelu in matmul epilogue — saves global memory roundtrip");
    launch_config(num_warps=4, num_stages=3)
        @rationale("3 pipeline stages hide global→shared latency for A/B prefetch");
}
```

---

## 5. Import Statement

```
import "<path>" as <alias>;
```

Reserved for future multi-file composition. Not yet implemented in the pipeline.

---

## 6. Grammar Summary (EBNF)

```ebnf
start          = (import_stmt | kernel_def | strategy_def)*
import_stmt    = "import" STRING "as" IDENT ";"
kernel_def     = "kernel" IDENT "(" param_list? ")" "->" tensor_type "{" kernel_body "}"
param_list     = param ("," param)*
param          = IDENT ":" tensor_type
tensor_type    = "Tensor" "<" "[" dim_list "]" "," scalar_type ("," layout)? ">"
dim_list       = INT ("," INT)*
kernel_body    = let_stmt* return_stmt
let_stmt       = "let" IDENT "=" op_call ";"
op_call        = IDENT "(" arg_list? ")"
arg_list       = (IDENT "=" arg_value) ("," IDENT "=" arg_value)*
arg_value      = IDENT | INT | FLOAT | STRING | BOOL | "[" arg_value* "]"
return_stmt    = "return" IDENT ";"

strategy_def   = "strategy" IDENT "for" "target" "(" STRING ")" "{" strategy_body "}"
strategy_body  = strategy_stmt*
strategy_stmt  = IDENT "(" strategy_kwargs ")" annotation? ";"?
strategy_kwargs = (IDENT "=" strategy_value) ("," IDENT "=" strategy_value)*
strategy_value = STRING | INT | FLOAT | BOOL | array | map | IDENT
array          = "[" strategy_value* "]"
map            = "{" (map_key ":" strategy_value)* "}"
annotation     = "@" IDENT "(" STRING ")"
```

---

## 7. Versioning

This document describes Arke Language **v1.0** as implemented in Stage 1 of the Arke project.  
Grammar file: `arke/parser/arke.lark` (grammar version comment inside).  
Changes to the language in Stage 2+ will increment the version and be documented in `CHANGELOG.md`.
