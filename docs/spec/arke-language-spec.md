# Arke Language Specification

> Version: 0.2.0-draft
> Status: 🚧 Draft — implementation priority: Phase 6+
> Principle: **Human Interface to AI-First IR**

---

## 0. Design Philosophy

The Arke language (`.ak`) is a **human-readable view** into the same IR
that LLMs manipulate via tool-use. It is not a separate system — it compiles
to the same Semantic IR and Strategy IR defined in `arke-ir-spec.md`.

### Why a Language at All?

If LLMs work with JSON IR directly, why have a language?

1. **Human authoring**: Researchers want to define computations in readable syntax,
   not hand-write JSON
2. **Code review**: `.ak` files in git are reviewable; `diff` on JSON IR is painful
3. **Documentation by example**: `.ak` examples teach both humans and LLMs
4. **Strategy inspection**: Humans can read, modify, and commit optimization
   strategies in a form that's more natural than JSON

### The Fundamental Rule

```
.ak file → Parser → Arke IR (JSON) → same IR that LLM produces
```

There is **no** semantic difference between an IR produced from `.ak` and
one produced by LLM tool-use. If a human writes a `.ak` file and an LLM
calls `create_kernel()` with equivalent parameters, the resulting Semantic
IR must be identical (modulo auto-generated IDs).

### AI-First Implications

The language design choices are informed by the fact that LLMs are the
primary IR consumer:

- **Minimal-token representation**: A complete kernel in `.ak` is 5–10 lines vs
  50–200 lines of equivalent Triton/CUDA code. This means LLMs can generate and
  reason about kernels with 10–50× fewer tokens, reducing cost and latency while
  allowing broader search within context limits
- **No implicit semantics**: Every operator call is explicit. No overloaded `*`
  that could mean matmul or elementwise mul depending on shapes
- **Typed parameters**: Shape and dtype are always declared, never inferred
  from context (LLMs and validators need complete type info)
- **Strategy is data, not code**: The `strategy` block describes decisions,
  not imperative execution — matching how LLMs build Strategy IR step-by-step
- **@rationale is first-class**: Not a comment — it's preserved in IR and
  matters for human review and LLM learning

---

## 1. Program Structure

An `.ak` file contains one or more definitions:

```
program := (import_stmt | kernel_def | strategy_def)*
```

```arke
// Optional imports
import "nvidia_ampere" as hw;

// Computation definition
kernel fused_matmul_relu(
    A: Tensor<[1024, 512], f16>,
    B: Tensor<[512, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let C = matmul(A, B);
    let Y = relu(C);
    return Y;
}

// Optimization strategy (optional — can also be built by LLM)
strategy fused_matmul_relu for target("nvidia_ampere") {
    fuse(nodes=["matmul", "relu"], type=epilogue)
        @rationale("relu is elementwise; fusing eliminates 4MB intermediate write");

    tile(loop="i", factors=[64, 16])
        @rationale("L2 cache line = 64, warp size = 16");

    place(tensor="A_tile", memory=shared)
        @rationale("A reused 16× across j iterations");

    parallel(loops=["i_outer", "j_outer"], mapping={i_outer: "block.x", j_outer: "block.y"})
        @rationale("16×16 = 256 blocks, good SM occupancy on Ampere");
}
```

---

## 2. Lexical Structure

### 2.1 Keywords

```
kernel    strategy    for       target    let
return    import      as        if        else
```

> **Note**: The keyword is `strategy` (not `schedule`). This aligns with
> the IR terminology and the project's framing of LLM as "strategic
> decision maker". See `docs/design/naming-system.md` §II for rationale.

### 2.2 Types

```
Scalar types:   f16  f32  f64  bf16  i8  i16  i32  i64  u8  u16  u32  u64  bool  index
Tensor type:    Tensor<[dim, ...], dtype>
Layout:         row_major  col_major
Memory:         global  shared  local  register
Fusion:         epilogue  prologue  horizontal  vertical
```

### 2.3 Literals

```
Integer:    42, 1024, 0xFF
Float:      3.14, 1e-3
String:     "hello", "nvidia_ampere"
Boolean:    true, false
Array:      [64, 16, 4]
Map:        {key: "value", key2: 42}
```

### 2.4 Comments

```arke
// Single-line comment
/* Multi-line
   comment */
```

### 2.5 Annotations

Annotations are prefixed with `@` and are **semantic** — they are
preserved in the IR, not discarded like comments.

```arke
@rationale("explanation text")    // Decision rationale — preserved in Strategy IR
```

---

## 3. Kernel Definition — "What to Compute"

A `kernel` block defines computation semantics. It compiles to Semantic IR.

### 3.1 Grammar

```
kernel_def   := "kernel" IDENT "(" params ")" "->" return_type "{" body "}"
params       := param ("," param)*
param        := IDENT ":" tensor_type
tensor_type  := "Tensor" "<" "[" INT ("," INT)* "]" "," scalar_type ("," layout)? ">"
scalar_type  := "f16" | "f32" | "f64" | "bf16" | "i8" | "i16" | "i32" | "i64"
              | "u8" | "u16" | "u32" | "u64" | "bool" | "index"
layout       := "row_major" | "col_major"
body         := (let_stmt | return_stmt)+
let_stmt     := "let" IDENT "=" op_call ";"
return_stmt  := "return" IDENT ";"
op_call      := IDENT "(" args ")"
args         := IDENT ("," IDENT)*
```

### 3.2 Semantics

A kernel definition maps directly to Semantic IR:

| .ak element | Semantic IR field |
|:------------|:------------------|
| `kernel name(...)` | `kernel_id` |
| Parameters | `params[]` |
| `-> Type` | `return_type` |
| `let X = op(...)` | `nodes[]` (auto-generates ID, resolves inputs) |
| `return X` | `return_node` |
| Operator calls | `edges[]` (inferred from data flow) |

### 3.3 Constraints

- A kernel must have at least one `return` statement
- All input parameters must be used (no dead params)
- Operator calls must reference defined operators from the catalog
- Shape consistency is checked at parse time (via op catalog shape inference rules)
- Variable names are scoped to the kernel body

### 3.4 Example

```arke
kernel softmax(
    X: Tensor<[1024, 2048], f16>
) -> Tensor<[1024, 2048], f16> {
    let max_val = reduce_max(X);
    let shifted = add(X, max_val);    // broadcasting semantics TBD
    let exp_val = exp(shifted);       // requires exp in catalog
    let sum_val = reduce_sum(exp_val);
    let Y = mul(exp_val, sum_val);    // actually div, simplified here
    return Y;
}
```

> For v0.2.0, complex ops like `softmax` are better expressed as a single
> catalog op rather than decomposed. Decomposition support is a future extension.

---

## 4. Strategy Definition — "How to Optimize"

A `strategy` block defines optimization decisions. It compiles to Strategy IR.

### 4.1 Grammar

```
strategy_def := "strategy" IDENT "for" "target" "(" STRING ")" "{" directives "}"
directives   := (directive ";")*
directive    := kind "(" named_args ")" rationale?
kind         := "tile" | "reorder" | "fuse" | "parallel" | "place"
              | "vectorize" | "unroll" | "algorithm"
named_args   := IDENT "=" value ("," IDENT "=" value)*
value        := STRING | INT | FLOAT | BOOL | array | map | IDENT
array        := "[" value ("," value)* "]"
map          := "{" map_entry ("," map_entry)* "}"
map_entry    := IDENT ":" value
rationale    := "@rationale" "(" STRING ")"
```

### 4.2 Semantics

Each directive in a `strategy` block becomes one Decision in Strategy IR:

| .ak element | Strategy IR field |
|:------------|:------------------|
| `strategy name` | `kernel_id` |
| `target("...")` | `target_hw` |
| Each directive | `decisions[].kind`, `decisions[].params` |
| `@rationale(...)` | `decisions[].rationale.text` |
| Directive order | `decisions[].step` (1-indexed, sequential) |

### 4.3 Strategy vs LLM Tool-Use

A strategy block in `.ak` is **equivalent** to a sequence of `apply_decision()`
tool calls by an LLM:

```
.ak file:                              LLM tool-use:
strategy mm for target("ampere") {     apply_decision(kind="tile",
    tile(loop="i", factors=[64,16])        params={loop:"i", factors:[64,16]},
        @rationale("...");                 rationale="...")
}                                      → same Strategy IR
```

The key difference: `.ak` is static (all decisions written upfront),
while LLM tool-use is interactive (decisions made one at a time with
feedback after each step). The resulting Strategy IR is the same.

### 4.4 Human-in-the-Loop Workflows

`.ak` strategy definitions enable several human participation patterns:

**1. Human writes initial strategy, LLM refines:**
```arke
// Human provides starting point
strategy matmul_v1 for target("ampere") {
    tile(loop="i", factors=[128, 16])
        @rationale("good starting point for large matmul");
}
// → Load into ArkeEnv → LLM adds more decisions via tool-use
```

**2. LLM generates strategy, human reviews:**
```bash
arke optimize matmul.json --target ampere --llm anthropic -o matmul.strategy.json
arke export matmul.strategy.json --format ak > matmul_strategy.ak
# Human reviews .ak file, modifies, commits to git
```

**3. Human overrides specific decisions:**
```arke
strategy matmul_fixed for target("ampere") {
    // Keep LLM's fusion decision
    fuse(nodes=["matmul_0", "relu_0"], type=epilogue)
        @rationale("LLM suggested, human approved");

    // Override LLM's tile — human knows better for this shape
    tile(loop="i", factors=[128, 32])
        @rationale("HUMAN OVERRIDE: 128×32 works better for [4096,4096] on our specific GPU");
}
```

---

## 5. Import Statements

```arke
import "path/to/hw_profile.json" as hw;
import "common_kernels.ak" as common;
```

Imports make hardware profiles and other `.ak` files available.
Semantics TBD for v0.2.0 — imports are parsed but not fully resolved.

---

## 6. Built-in Operators

The `.ak` language does not define operators — it references the
**operator catalog** (`arke/ir/ops/catalog.py`). Any operator registered
in the catalog can be called from a kernel definition.

See Arke IR Spec §6 for the P0 operator table.

```arke
// All of these are valid if registered in the catalog:
let C = matmul(A, B);
let Y = relu(C);
let Z = softmax(X);
let S = reduce_sum(X);
```

To add a new operator: register it in `catalog.py` with its semantics,
shape inference rule, and NumPy reference. It becomes available in
`.ak` files, tool-use, and validation simultaneously.

---

## 7. Type System

### 7.1 Scalar Types

Same as IR Spec §7.1 — 16 types in 4 groups (float, integer, unsigned, special).

### 7.2 Tensor Types

```arke
Tensor<[1024, 512], f16>                    // shape + dtype
Tensor<[1024, 512], f16, col_major>         // + explicit layout
```

- Shape dimensions must be positive integers (static shapes in v0.2.0)
- Layout is optional, defaults to `row_major`

### 7.3 Type Checking

Types are checked at parse time:
- Parameter types are explicit (no inference)
- Operator output types are inferred from catalog shape rules
- Shape mismatches are compile errors, not runtime errors

---

## 8. Relationship to Arke IR

```
               ┌────────────────────┐
               │     .ak file       │
               │  (human authored)  │
               └────────┬───────────┘
                        │ parse
                        ▼
               ┌────────────────────┐
               │     Arke IR        │←──── LLM tool-use
               │ (JSON, canonical)  │       (AI authored)
               └────────┬───────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Validate │  │ Codegen  │  │ Inspect  │
    │ V0/V1/V2 │  │ Triton/  │  │ Human    │
    │          │  │ CUDA     │  │ readable │
    └──────────┘  └──────────┘  └──────────┘
```

The language is one of several **on-ramps** to the IR. It is never the
canonical form — the IR (JSON) is canonical. `.ak` files can always be
round-tripped:

```
.ak → parse → IR (JSON) → export → .ak'
```

Where `.ak'` is semantically identical to `.ak` (formatting may differ).

---

## 9. File Extension and Conventions

| Convention | Value |
|:-----------|:------|
| File extension | `.ak` |
| Encoding | UTF-8 |
| Naming | `snake_case.ak` |
| Kernel naming | `snake_case` |
| One kernel per file | Recommended but not required |

---

## 10. Implementation Priority

Per `plan-v3.0.md`, the language is **Phase 6** (after LLM integration proven):

```
Phase 1-4: IR + Validation + Codegen + LLM Integration (no .ak parser needed)
Phase 6:   .ak EBNF grammar (Lark) → Parser → AST → Semantic IR + CLI
```

The language exists to serve humans. The system works without it —
LLMs interact with IR directly. The parser is an adapter layer, not
a critical path component.

---

*Spec version: 0.2.0-draft | Date: 2026-04-01*
*Previous: docs/spec/deprecated/arke-language-spec.md (v0.1.0)*
*Implementation: Phase 6+ (arke/lang/parser.py, arke/lang/ast.py)*
