# Arke IR ↔ MLIR Structural Mapping

> **Version:** 1.0  
> **Status:** Phase 1 specification  
> **Purpose:** G6.6 — Map every SemanticIR/StrategyIR field to its MLIR dialect op/attribute counterpart, enabling future Phase 3 MLIR backend.

---

## 1. Design Philosophy

Arke IR is intentionally **dialect-agnostic** at the semantic level. The two IR layers map cleanly to distinct MLIR concepts:

| Arke Layer | MLIR Analog | Role |
|:-----------|:------------|:-----|
| **SemanticIR** | `linalg` / `tensor` dialect | *What* to compute (math semantics) |
| **StrategyIR** | `transform` dialect + `scf` | *How* to optimize (tiling, mapping) |

This separation mirrors MLIR's own split between computation dialects and transformation passes.

---

## 2. SemanticIR → MLIR Mapping

### 2.1 Top-level Structure

| SemanticIR field | MLIR equivalent | Notes |
|:-----------------|:----------------|:------|
| `kernel_id` | `func.func @<kernel_id>` | Function name in the module |
| `params[]` | `func.func` arguments (typed `tensor<[shape]xdtype>`) | Each param → one SSA block argument |
| `nodes[]` | Op sequence inside function body | Each Node → one MLIR op |
| `output.shape` | `func.func` return type | `tensor<[shape]xdtype>` |
| `output.dtype` | element type in tensor type | e.g. `f16`, `f32` |
| `fusion_groups[]` | `linalg.generic` with fused region | Multiple nodes in one `linalg.generic` |

### 2.2 Scalar Types

| Arke dtype | MLIR type | LLVM IR type |
|:-----------|:----------|:-------------|
| `f16` | `f16` | `half` |
| `bf16` | `bf16` | `bfloat` |
| `f32` | `f32` | `float` |
| `f64` | `f64` | `double` |
| `i8` | `i8` | `i8` |
| `i16` | `i16` | `i16` |
| `i32` | `i32` | `i32` |
| `i64` | `i64` | `i64` |
| `bool` | `i1` | `i1` |

### 2.3 Operator → MLIR Op Mapping

#### Category A: Compute

| Arke op | Primary MLIR dialect op | Alternative |
|:--------|:------------------------|:------------|
| `matmul(A, B)` | `linalg.matmul ins(%A, %B) outs(%C)` | `linalg.generic` with contraction |
| `batch_matmul(A, B)` | `linalg.batch_matmul` | `linalg.generic` with batch dim |

**SemanticIR → MLIR example (matmul):**

```
// SemanticIR (JSON)
{
  "op": "matmul",
  "inputs": {"A": "param_A", "B": "param_B"},
  "output_shape": [1024, 1024],
  "output_dtype": "f16"
}

// MLIR (linalg dialect)
%C = tensor.empty() : tensor<1024x1024xf16>
%result = linalg.matmul
    ins(%A, %B : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
    outs(%C : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
```

#### Category C: Reduction

| Arke op | MLIR equivalent |
|:--------|:----------------|
| `softmax(X, axis)` | `linalg.generic` with custom region (exp, sum, div) or `linalg.softmax` (upstream) |
| `layernorm(X, W, B, eps)` | `linalg.generic` with mean/var sub-region + scale |
| `rmsnorm(X, W, eps)` | `linalg.generic` with rms sub-region + scale |
| `reduce_sum(X)` | `linalg.reduce` with `arith.addf` combiner |
| `reduce_max(X)` | `linalg.reduce` with `arith.maximumf` combiner |

#### Category D: Elementwise

| Arke op | MLIR equivalent |
|:--------|:----------------|
| `relu(X)` | `linalg.generic` + `arith.maximumf(%x, %zero)` |
| `gelu(X)` | `linalg.generic` + `math.tanh` approximation |
| `silu(X)` | `linalg.generic` + `math.exp` + `arith.divf` |
| `add(A, B)` | `linalg.add` or `linalg.generic` + `arith.addf` |
| `mul(A, B)` | `linalg.generic` + `arith.mulf` |
| `transpose(X)` | `linalg.transpose` |

---

## 3. StrategyIR → MLIR `transform` Dialect Mapping

The `transform` dialect provides a structured way to express optimization decisions as transformations on the IR. Each StrategyIR `Decision` maps to one or more `transform` ops.

### 3.1 Decision Kind → Transform Op

| Decision `kind` | `transform` dialect op | Notes |
|:----------------|:-----------------------|:------|
| `tile` | `transform.structured.tile_using_for` | `factors` → tile sizes |
| `reorder` | `transform.structured.interchange` | `order` → loop permutation |
| `parallel` | `transform.structured.tile_using_forall` | `mapping` → thread/block dims |
| `fuse` | `transform.structured.fuse_into_containing_op` | `ops` → which ops to fuse |
| `vectorize` | `transform.structured.vectorize` | `width` → vector size |
| `place` | `transform.gpu.map_nested_forall_to_threads` + memory annotation | `memory` → `gpu.private` / `gpu.workgroup` |
| `launch_config` | `transform.gpu.launch_cluster` params | `num_warps`, `num_stages` |
| `unroll` | `transform.loop.unroll` | `factor` → unroll factor |
| `autotune` | `transform.loop.peel` + config sweep | `configs` → candidate list |

### 3.2 Mapping Example (matmul tile + parallel)

```python
# Arke StrategyIR decisions:
tile("M", factors=[64])         # @rationale: tensor-core aligned
tile("N", factors=[64])         # @rationale: tensor-core aligned
tile("K", factors=[16])         # @rationale: shared memory budget
reorder(["M", "N", "K"])
parallel(["M", "N"], mapping={"M": "blockIdx.x", "N": "blockIdx.y"})
place("A_tile", memory="shared")
launch_config(num_warps=4, num_stages=3)
```

```mlir
// Equivalent MLIR transform sequence
%func = transform.structured.match ops{["linalg.matmul"]} in %module

// Tile M, N, K
%tiled, %loops:3 = transform.structured.tile_using_for %func
    tile_sizes [64, 64, 16]

// Interchange to MNK order (already MNK, no-op here)
transform.structured.interchange %tiled iterator_interchange = [0, 1, 2]

// Map outer loops to GPU blocks
%forall, %tiled2 = transform.structured.tile_using_forall %tiled
    tile_sizes [64, 64]
    mapping [#gpu.block<x>, #gpu.block<y>]

// Map shared memory
transform.gpu.map_nested_forall_to_threads %forall
    workgroup_dims [128, 1, 1]

// Set launch config
transform.gpu.launch_cluster %forall
    num_warps = 4 num_stages = 3
```

### 3.3 `@rationale` → MLIR Comment Preservation

`@rationale` text is preserved as:
1. **Arke backend**: inline comment in generated Triton kernel (`# <rationale>`)
2. **MLIR backend**: `transform.annotate` with `"arke.rationale"` attribute on the transformed op

```mlir
transform.annotate %tiled "arke.rationale" = "K-tile=16: A+B tiles = 4096B ≤ smem/2"
```

---

## 4. FusionGroup → MLIR

| SemanticIR | MLIR equivalent |
|:-----------|:----------------|
| `FusionGroup` with `nodes=[matmul, gelu]` | Single `linalg.generic` with fused region containing matmul + gelu epilogue |
| `FusionGroup` with `kind="epilogue"` | `linalg.fuse_elementwise_ops` or `transform.structured.fuse_into_containing_op` |

---

## 5. Type Layout → MLIR Memref Attributes

| Arke layout | MLIR memref attribute |
|:------------|:----------------------|
| `row_major` | `#gpu.address_space<global>` with default strides `[N, 1]` |
| `col_major` | strides `[1, M]` |

---

## 6. Phase 3 Implementation Plan

When Phase 3 adds the MLIR backend (`arke/backend/mlir_backend.py`):

1. **SemanticIR → `linalg` dialect**: Walk `ir.nodes`, emit one `linalg.*` op per node using the mapping in §2.3
2. **StrategyIR → `transform` dialect**: Walk `strategy.decisions`, emit one `transform.*` op per decision using the mapping in §3.1
3. **`@rationale` preservation**: Emit `transform.annotate` after each transform op
4. **Fusion**: Translate `FusionGroup` → `transform.structured.fuse_into_containing_op`
5. **Lowering pipeline**: `linalg` → `scf` → `gpu` → `nvvm`/`rocdl` → LLVM IR → PTX

The existing `arke/ir/` data structures are **MLIR-ready**: no schema changes needed. Only a new backend translator is required.

---

*Generated by Arke toolchain — Phase 1, Phase 1.9 (G6.6)*
