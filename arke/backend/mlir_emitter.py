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
