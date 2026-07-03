# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke → MLIR text emitter (Phase 3, P3-S1).

Emits **executable** MLIR from an ``IRGraph`` — not the S7 string-skeleton.
The output uses the ``memref`` + ``linalg`` dialects and is designed to be
consumed by the CPU lowering pipeline in ``arke.backend.mlir_backend`` and
JIT-executed via ``mlir-cpu-runner``.

Design (P3-S1, matmul-first):
  * Graph inputs become ``memref<...xTY>`` function arguments.
  * Each node emits a ``linalg.*`` op writing into a freshly ``memref.alloc``'d
    output buffer (destination-passing style, as linalg on memrefs requires).
  * Output shapes for intermediates are inferred per-op (matmul: (M,K)@(K,N)→(M,N)).
  * A companion ``@main`` harness (for standalone JIT correctness runs) is emitted
    separately by the backend when it fills buffers with test data; the core
    ``emit_kernel`` here produces just the kernel ``func.func``.

Only the ops needed for the P3-S1 gate are wired up first (matmul); the
dispatch table (`_EMITTERS`) is the extension seam for P3-S2's 35-op set.
"""

from __future__ import annotations

from dataclasses import dataclass

from arke.ir.graph import IRGraph, IRNode, IRValue


# ── dtype mapping ──────────────────────────────────────────────

_DTYPE_TO_MLIR = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "float64": "f64",
    "int32": "i32",
    "int64": "i64",
    "int8": "i8",
}


def mlir_dtype(dtype: str) -> str:
    """Map an Arke dtype string to an MLIR element type."""
    if dtype not in _DTYPE_TO_MLIR:
        raise NotImplementedError(f"MLIR emitter: unsupported dtype {dtype!r}")
    return _DTYPE_TO_MLIR[dtype]


def memref_type(shape: list[int], dtype: str) -> str:
    """Render a static memref type, e.g. memref<4x8xf32>."""
    if not shape:
        return f"memref<{mlir_dtype(dtype)}>"
    dims = "x".join(str(int(d)) for d in shape)
    return f"memref<{dims}x{mlir_dtype(dtype)}>"


def tensor_type(shape: list[int], dtype: str) -> str:
    dims = "x".join(str(int(d)) for d in shape)
    return f"tensor<{dims}x{mlir_dtype(dtype)}>"


# ── shape inference (per-op) ───────────────────────────────────

def infer_output_shape(op: str, in_shapes: list[list[int]]) -> list[int]:
    """Infer the output shape of a node from its input shapes.

    Kept deliberately small for P3-S1; extended alongside _EMITTERS.
    """
    if op == "matmul":
        (m, k), (k2, n) = in_shapes[0], in_shapes[1]
        if k != k2:
            raise ValueError(f"matmul K mismatch: {in_shapes[0]} @ {in_shapes[1]}")
        return [m, n]
    if op in ("relu", "gelu", "silu", "add", "mul", "sigmoid", "exp"):
        # elementwise → same shape as first input
        return list(in_shapes[0])
    raise NotImplementedError(f"MLIR emitter: no shape rule for op {op!r}")


# ── per-op MLIR body emitters ──────────────────────────────────
# Each returns the MLIR lines that compute `out_buf` from `in_bufs`
# (all are already-allocated memref SSA names, e.g. "%arg0", "%v0").

def _emit_matmul(out_buf: str, in_bufs: list[str], out_ty: str,
                 in_tys: list[str], elem: str) -> list[str]:
    zero = "0.0" if elem.startswith("f") or elem.startswith("bf") else "0"
    cst = f"%cst_zero_{out_buf[1:]}"
    return [
        f"    {cst} = arith.constant {zero} : {elem}",
        f"    linalg.fill ins({cst} : {elem}) outs({out_buf} : {out_ty})",
        f"    linalg.matmul ins({in_bufs[0]}, {in_bufs[1]} : {in_tys[0]}, {in_tys[1]}) "
        f"outs({out_buf} : {out_ty})",
    ]


_EMITTERS = {
    "matmul": _emit_matmul,
}


SUPPORTED_OPS = frozenset(_EMITTERS.keys())


# ── transform-dialect schedule emission (P3-S1 tiling / P3-S5 L2) ───
# Ops that support linalg tiling via `transform.structured.tile_using_for`.
# Each maps to the number of loops the tiling produces (== number of tile dims
# with a non-zero size, but we always request the full rank and let MLIR fold
# zero-size dims). matmul has 3 iteration dims (M, N, K).
_TILEABLE_LOOP_COUNT = {
    "matmul": 3,
}


def emit_transform_schedule(op: str, tile_sizes: list[int]) -> str:
    """Emit a ``transform.named_sequence`` that tiles the given linalg op.

    The schedule is applied by ``mlir-opt -transform-interpreter`` as a pre-pass
    (then erased with ``-test-transform-dialect-erase-schedule``). This is the
    P3-S1 "linalg + transform dialect" path and the seam StrategyIR L2 loop-nest
    decisions lower through in P3-S5.

    Args:
        op: linalg op name (e.g. "matmul"). Must be in ``_TILEABLE_LOOP_COUNT``.
        tile_sizes: per-iteration-dim tile sizes. Length must match the op's
            iteration rank. A size of 0 means "do not tile that dim".

    Returns:
        The MLIR text of the transform named-sequence module body (the
        ``transform.named_sequence @__transform_main`` block, indented to sit
        inside a ``module attributes {transform.with_named_sequence}``).
    """
    if op not in _TILEABLE_LOOP_COUNT:
        raise NotImplementedError(
            f"transform tiling: op {op!r} not tileable "
            f"(supported: {sorted(_TILEABLE_LOOP_COUNT)})"
        )
    n_loops = _TILEABLE_LOOP_COUNT[op]
    if len(tile_sizes) != n_loops:
        raise ValueError(
            f"transform tiling: op {op!r} needs {n_loops} tile sizes, "
            f"got {len(tile_sizes)}: {tile_sizes}"
        )
    # number of scf.for loops produced == count of non-zero tile sizes
    n_nonzero = sum(1 for t in tile_sizes if t != 0)
    linalg_op = f"linalg.{op}"
    sizes = ", ".join(str(int(t)) for t in tile_sizes)
    if n_nonzero == 0:
        # degenerate: no tiling requested
        loop_results = ""
        loop_types = ""
    else:
        loop_results = ", %loops:" + str(n_nonzero)
        loop_types = ", " + ", ".join(
            "!transform.any_op" for _ in range(n_nonzero)
        )
    return "\n".join([
        "  transform.named_sequence @__transform_main("
        "%arg0: !transform.any_op {transform.readonly}) {",
        f'    %target = transform.structured.match ops{{["{linalg_op}"]}} '
        "in %arg0 : (!transform.any_op) -> !transform.any_op",
        f"    %tiled{loop_results} = transform.structured.tile_using_for "
        f"%target[{sizes}] : (!transform.any_op) -> "
        f"(!transform.any_op{loop_types})",
        "    transform.yield",
        "  }",
    ])


# ── emitter result ─────────────────────────────────────────────

@dataclass
class EmittedKernel:
    """MLIR text for a kernel plus buffer metadata for the JIT harness."""
    mlir_text: str
    kernel_name: str
    arg_names: list[str]          # graph input names, in func arg order
    arg_shapes: list[list[int]]
    arg_dtypes: list[str]
    result_name: str             # graph output value name
    result_shape: list[int]
    result_dtype: str


# ── graph → MLIR ───────────────────────────────────────────────

def _resolve_shape(graph: IRGraph, name: str, computed: dict[str, list[int]]) -> list[int]:
    if name in computed:
        return computed[name]
    v = graph.values.get(name)
    if v is not None and v.shape:
        return list(v.shape)
    raise ValueError(f"MLIR emitter: cannot resolve shape for value {name!r}")


def emit_kernel(graph: IRGraph) -> EmittedKernel:
    """Emit an executable memref-based MLIR kernel func from an IRGraph.

    Only single-output graphs with statically-shaped inputs are supported in
    P3-S1. Raises NotImplementedError for unsupported ops so the backend can
    fall back cleanly.
    """
    if len(graph.graph_outputs) != 1:
        raise NotImplementedError(
            f"MLIR emitter (P3-S1): single-output graphs only, "
            f"got outputs={graph.graph_outputs}"
        )

    # dtype: take from the first graph input (P3-S1 assumes homogeneous dtype)
    in_vals: list[IRValue] = [graph.values[n] for n in graph.graph_inputs]
    if not in_vals:
        raise ValueError("MLIR emitter: graph has no inputs")
    elem_dtype = in_vals[0].dtype
    elem = mlir_dtype(elem_dtype)

    # value name -> SSA name in MLIR (%arg0.. for inputs, %vN for node outputs)
    ssa: dict[str, str] = {}
    computed_shapes: dict[str, list[int]] = {}
    arg_types: list[str] = []
    for i, v in enumerate(in_vals):
        ssa[v.name] = f"%arg{i}"
        computed_shapes[v.name] = list(v.shape)
        arg_types.append(memref_type(v.shape, v.dtype))

    body: list[str] = []
    temp_idx = 0
    for node in graph.nodes:
        if node.op not in _EMITTERS:
            raise NotImplementedError(
                f"MLIR emitter (P3-S1): op {node.op!r} not yet supported "
                f"(supported: {sorted(SUPPORTED_OPS)})"
            )
        in_names = list(node.inputs.values())
        in_shapes = [_resolve_shape(graph, n, computed_shapes) for n in in_names]
        in_bufs = [ssa[n] for n in in_names]
        in_tys = [memref_type(s, elem_dtype) for s in in_shapes]

        out_shape = infer_output_shape(node.op, in_shapes)
        if len(node.outputs) != 1:
            raise NotImplementedError(
                f"MLIR emitter (P3-S1): single-output nodes only, node {node.id}"
            )
        out_name = node.outputs[0]
        out_buf = f"%v{temp_idx}"
        temp_idx += 1
        out_ty = memref_type(out_shape, elem_dtype)

        body.append(f"    {out_buf} = memref.alloc() : {out_ty}")
        body.extend(_EMITTERS[node.op](out_buf, in_bufs, out_ty, in_tys, elem))

        ssa[out_name] = out_buf
        computed_shapes[out_name] = out_shape

    out_val_name = graph.graph_outputs[0]
    result_buf = ssa[out_val_name]
    result_shape = computed_shapes[out_val_name]
    result_ty = memref_type(result_shape, elem_dtype)

    # Kernel signature: inputs + a caller-provided output buffer (dest-passing),
    # so the harness/bridge controls result memory. We copy the computed result
    # into the out arg and return void — the standard memref ABI for JIT.
    kernel = graph.name or "arke_kernel"
    all_arg_types = arg_types + [result_ty]
    sig_args = ", ".join(
        f"%arg{i}: {t}" for i, t in enumerate(all_arg_types)
    )
    out_arg = f"%arg{len(arg_types)}"

    lines = [
        "module {",
        f"  func.func @{kernel}({sig_args}) {{",
        *body,
        f"    memref.copy {result_buf}, {out_arg} : {result_ty} to {result_ty}",
        "    return",
        "  }",
        "}",
    ]

    return EmittedKernel(
        mlir_text="\n".join(lines),
        kernel_name=kernel,
        arg_names=list(graph.graph_inputs),
        arg_shapes=[list(v.shape) for v in in_vals],
        arg_dtypes=[v.dtype for v in in_vals],
        result_name=out_val_name,
        result_shape=result_shape,
        result_dtype=elem_dtype,
    )


# ── GPU kernel emission (P3-S1 GPU path) ───────────────────────

@dataclass
class EmittedGPUKernel:
    """A single-kernel ``gpu.module`` MLIR string + launch metadata."""
    mlir_text: str
    kernel_name: str
    arg_names: list[str]
    arg_shapes: list[list[int]]
    arg_dtypes: list[str]
    result_name: str
    result_shape: list[int]
    result_dtype: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    # order of memref args as the kernel expects them (inputs..., output)
    buffer_order: list[str]


def emit_gpu_matmul(graph: IRGraph, chip: str = "sm_86") -> EmittedGPUKernel:
    """Emit a single-kernel gpu.module matmul: thread-block (i,j) over MxN grid.

    Each block computes one C[i,j] via a K-loop accumulate. Deliberately simple
    (1 thread/block, block-per-output-element) — this is the P3-S1 GPU
    *correctness* proof, not a perf kernel; tiling/blocking come with P3-S2/S3.
    """
    if len(graph.nodes) != 1 or graph.nodes[0].op != "matmul":
        raise NotImplementedError("emit_gpu_matmul: single matmul node only (P3-S1)")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    A, B = graph.values[a_name], graph.values[b_name]
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"matmul K mismatch: {A.shape} @ {B.shape}")
    if A.dtype != "float32":
        raise NotImplementedError("emit_gpu_matmul: f32 only (P3-S1)")
    out_name = node.outputs[0]
    at = memref_type([M, K], "float32")
    bt = memref_type([K, N], "float32")
    ct = memref_type([M, N], "float32")
    kernel = graph.name or "matmul"
    text = "\n".join([
        "module attributes {gpu.container_module} {",
        f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{',
        f"    gpu.func @{kernel}(%A: {at}, %B: {bt}, %C: {ct}) kernel {{",
        "      %i = gpu.block_id x",
        "      %j = gpu.block_id y",
        "      %c0 = arith.constant 0 : index",
        "      %c1 = arith.constant 1 : index",
        f"      %cK = arith.constant {K} : index",
        "      %zero = arith.constant 0.0 : f32",
        "      %acc = scf.for %k = %c0 to %cK step %c1 "
        "iter_args(%s = %zero) -> f32 {",
        f"        %a = memref.load %A[%i, %k] : {at}",
        f"        %b = memref.load %B[%k, %j] : {bt}",
        "        %p = arith.mulf %a, %b : f32",
        "        %ns = arith.addf %s, %p : f32",
        "        scf.yield %ns : f32",
        "      }",
        f"      memref.store %acc, %C[%i, %j] : {ct}",
        "      gpu.return",
        "    }",
        "  }",
        "}",
    ])
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[a_name, b_name],
        arg_shapes=[[M, K], [K, N]],
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=[M, N],
        result_dtype="float32",
        grid=(M, N, 1),
        block=(1, 1, 1),
        buffer_order=[a_name, b_name, out_name],
    )
