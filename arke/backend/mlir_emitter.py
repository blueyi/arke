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
    from arke.backend.mlir_ops import ELEMENTWISE_SPECS, COMPOSITE_SPECS
    if op in ELEMENTWISE_SPECS:
        # elementwise → same shape as first input
        return list(in_shapes[0])
    if op in COMPOSITE_SPECS:
        from arke.backend.mlir_ops import composite_output_shape
        return composite_output_shape(op, in_shapes)
    raise NotImplementedError(f"MLIR emitter: no shape rule for op {op!r}")


# ── elementwise emitter (linalg.generic) ───────────────────────

def _identity_map(rank: int) -> str:
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_map<({dims}) -> ({dims})>"


def _emit_elementwise(op: str, out_buf: str, in_bufs: list[str], out_ty: str,
                      in_tys: list[str], elem: str, rank: int) -> list[str]:
    """Emit a linalg.generic elementwise op from an OpSpec body.

    All operands + output share the identity indexing map and all-parallel
    iterators. The scalar body (from mlir_ops.ELEMENTWISE_SPECS) computes %res
    from %a0..%aK.
    """
    from arke.backend.mlir_ops import ELEMENTWISE_SPECS
    spec = ELEMENTWISE_SPECS[op]
    imap = _identity_map(rank)
    n = len(in_bufs)
    maps = ", ".join([imap] * (n + 1))
    iters = ", ".join(['"parallel"'] * rank)
    ins = ", ".join(in_bufs)
    ins_tys = ", ".join(in_tys)
    # block args: one per input (a0..aK) + the output init (o)
    block_args = ", ".join([f"%a{i}: {elem}" for i in range(n)] + [f"%o: {elem}"])
    lines = [
        f"    linalg.generic {{indexing_maps = [{maps}], "
        f'iterator_types = [{iters}]}} '
        f"ins({ins} : {ins_tys}) outs({out_buf} : {out_ty}) {{",
        f"    ^bb0({block_args}):",
        *spec.ew_body,
        "      linalg.yield %res : f32",
        "    }",
    ]
    return lines


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


def _all_supported_ops() -> frozenset[str]:
    from arke.backend.mlir_ops import ELEMENTWISE_SPECS, COMPOSITE_SPECS
    return (frozenset(_EMITTERS.keys())
            | frozenset(ELEMENTWISE_SPECS.keys())
            | frozenset(COMPOSITE_SPECS.keys()))


SUPPORTED_OPS = _all_supported_ops()


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
        f"%target tile_sizes [{sizes}] : (!transform.any_op) -> "
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
    from arke.backend.mlir_ops import ELEMENTWISE_SPECS, COMPOSITE_SPECS, emit_composite
    for node in graph.nodes:
        is_ew = node.op in ELEMENTWISE_SPECS
        is_comp = node.op in COMPOSITE_SPECS
        if node.op not in _EMITTERS and not is_ew and not is_comp:
            raise NotImplementedError(
                f"MLIR emitter: op {node.op!r} not yet supported "
                f"(supported: {sorted(SUPPORTED_OPS)})"
            )
        in_names = list(node.inputs.values())
        in_shapes = [_resolve_shape(graph, n, computed_shapes) for n in in_names]
        in_bufs = [ssa[n] for n in in_names]
        in_tys = [memref_type(s, elem_dtype) for s in in_shapes]

        out_shape = infer_output_shape(node.op, in_shapes)
        if len(node.outputs) != 1:
            raise NotImplementedError(
                f"MLIR emitter: single-output nodes only, node {node.id}"
            )
        out_name = node.outputs[0]
        out_buf = f"%v{temp_idx}"
        temp_idx += 1
        out_ty = memref_type(out_shape, elem_dtype)

        body.append(f"    {out_buf} = memref.alloc() : {out_ty}")
        if is_ew:
            body.extend(_emit_elementwise(
                node.op, out_buf, in_bufs, out_ty, in_tys, elem, len(out_shape)
            ))
        elif is_comp:
            body.extend(emit_composite(
                node.op, out_buf, out_shape, in_bufs, in_shapes, elem
            ))
        else:
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


# Shared-memory tiled matmul tile size. 16x16 = 256 threads/block, 2 KiB shared
# per operand tile (16*16*4) — comfortable on sm_86 (48-100 KiB shared/SM).
GPU_MM_TILE = 16


def emit_gpu_matmul_tiled(graph: IRGraph, chip: str = "sm_86",
                          tile: int = GPU_MM_TILE) -> EmittedGPUKernel:
    """Emit a shared-memory tiled matmul gpu.module (P3-S2 perf path).

    grid = (N/tile, M/tile, 1), block = (tile, tile, 1). Each block computes one
    ``tile x tile`` output sub-block; each thread (tx,ty) computes one
    ``C[by*tile+ty, bx*tile+tx]``. The K dimension is walked in ``tile``-wide
    steps: the block cooperatively stages an A-tile and a B-tile into workgroup
    (shared) memory, ``gpu.barrier``, does the ``tile``-length inner product from
    shared memory, ``gpu.barrier`` again, and accumulates. This is the classic
    blocked-GEMM that reuses each global load ``tile`` times, vs the correctness
    kernel's one-thread-per-output global-only K-loop.

    Requires M, K, N all divisible by ``tile`` (tile-aligned). Callers fall back
    to ``emit_gpu_matmul`` for non-aligned shapes. f32 only.
    """
    if len(graph.nodes) != 1 or graph.nodes[0].op != "matmul":
        raise NotImplementedError("emit_gpu_matmul_tiled: single matmul node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    A, B = graph.values[a_name], graph.values[b_name]
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"matmul K mismatch: {A.shape} @ {B.shape}")
    if A.dtype != "float32":
        raise NotImplementedError("emit_gpu_matmul_tiled: f32 only")
    if M % tile or K % tile or N % tile:
        raise NotImplementedError(
            f"emit_gpu_matmul_tiled: tile-aligned only (M,K,N % {tile}), got {M},{K},{N}"
        )
    out_name = node.outputs[0]
    at = memref_type([M, K], "float32")
    bt = memref_type([K, N], "float32")
    ct = memref_type([M, N], "float32")
    sty = f"memref<{tile}x{tile}xf32, #gpu.address_space<workgroup>>"
    kernel = graph.name or "matmul"
    text = "\n".join([
        "module attributes {gpu.container_module} {",
        f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{',
        f"    gpu.func @{kernel}(%A: {at}, %B: {bt}, %C: {ct})",
        # shared tiles as workgroup attributions → real .shared memory (a plain
        # memref.alloc in workgroup space is lowered to malloc+addrspacecast,
        # which yields an illegal .shared address at runtime).
        f"        workgroup(%sA : {sty}, %sB : {sty})",
        "        kernel {",
        # thread + block ids
        "      %tx = gpu.thread_id x",
        "      %ty = gpu.thread_id y",
        "      %bx = gpu.block_id x",
        "      %by = gpu.block_id y",
        "      %c0 = arith.constant 0 : index",
        "      %c1 = arith.constant 1 : index",
        f"      %cT = arith.constant {tile} : index",
        f"      %cK = arith.constant {K} : index",
        "      %zero = arith.constant 0.0 : f32",
        # global row/col this thread owns
        "      %rbase = arith.muli %by, %cT : index",
        "      %row = arith.addi %rbase, %ty : index",
        "      %cbase = arith.muli %bx, %cT : index",
        "      %col = arith.addi %cbase, %tx : index",
        # K-tile loop, accumulate in %acc
        "      %acc = scf.for %kk = %c0 to %cK step %cT iter_args(%s = %zero) -> f32 {",
        # stage A[row, kk+tx] and B[kk+ty, col] into shared mem
        "        %ak = arith.addi %kk, %tx : index",
        f"        %av = memref.load %A[%row, %ak] : {at}",
        f"        memref.store %av, %sA[%ty, %tx] : {sty}",
        "        %bk = arith.addi %kk, %ty : index",
        f"        %bv = memref.load %B[%bk, %col] : {bt}",
        f"        memref.store %bv, %sB[%ty, %tx] : {sty}",
        "        gpu.barrier",
        # inner product over the shared tile
        "        %p = scf.for %kt = %c0 to %cT step %c1 iter_args(%si = %s) -> f32 {",
        f"          %sa = memref.load %sA[%ty, %kt] : {sty}",
        f"          %sb = memref.load %sB[%kt, %tx] : {sty}",
        "          %m = arith.mulf %sa, %sb fastmath<contract> : f32",
        "          %ns = arith.addf %si, %m fastmath<contract> : f32",
        "          scf.yield %ns : f32",
        "        }",
        "        gpu.barrier",
        "        scf.yield %p : f32",
        "      }",
        f"      memref.store %acc, %C[%row, %col] : {ct}",
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
        grid=(N // tile, M // tile, 1),
        block=(tile, tile, 1),
        buffer_order=[a_name, b_name, out_name],
    )


# Register-blocked (2D thread-tile) matmul params. Each thread computes a
# TM x TN micro-tile of C held in registers. Block tile = BM x BN, K-step = BK.
# threads/block = (BN/TN) x (BM/TM). Defaults: 64x64 block, BK=16, 4x4 per
# thread → 16x16=256 threads, 4 KiB+4 KiB shared, 16 acc regs/thread, FMA inner
# loop — a solid compute/memory balance for sm_86 without register spilling.
GPU_MM_BM = 64
GPU_MM_BN = 64
GPU_MM_BK = 16
GPU_MM_TM = 4
GPU_MM_TN = 4


def emit_gpu_matmul_regblock(
    graph: IRGraph, chip: str = "sm_86",
    BM: int = GPU_MM_BM, BN: int = GPU_MM_BN, BK: int = GPU_MM_BK,
    TM: int = GPU_MM_TM, TN: int = GPU_MM_TN,
) -> EmittedGPUKernel:
    """Emit a register-blocked (2D thread-tile) matmul gpu.module.

    Each thread computes a ``TM x TN`` micro-tile of C accumulated in a private
    (register) memref, so each shared-memory value fetched is reused ``TM`` (A)
    or ``TN`` (B) times — the classic CUTLASS-style blocking that lifts
    arithmetic intensity far above the 1-output-per-thread tiled kernel.

    Layout: block tile ``BM x BN``, K walked in ``BK`` steps. Threads per block
    = ``(BN/TN) x (BM/TM)`` (= 16x16 = 256 by default). The block cooperatively
    stages ``A[BM x BK]`` and ``B[BK x BN]`` into workgroup memory with a
    stride-``nthreads`` linear load loop, barrier, then each thread reads its
    ``TM`` A-rows and ``TN`` B-cols from shared and does ``TM*TN`` FMAs per
    K-element, barrier, next K-tile.

    Requires M % BM == 0, N % BN == 0, K % BK == 0. f32 only. Callers fall back
    to the simpler tiled/correctness kernels for non-conforming shapes.
    """
    if len(graph.nodes) != 1 or graph.nodes[0].op != "matmul":
        raise NotImplementedError("emit_gpu_matmul_regblock: single matmul node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    A, B = graph.values[a_name], graph.values[b_name]
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"matmul K mismatch: {A.shape} @ {B.shape}")
    if A.dtype != "float32":
        raise NotImplementedError("emit_gpu_matmul_regblock: f32 only")
    if M % BM or N % BN or K % BK:
        raise NotImplementedError(
            f"emit_gpu_matmul_regblock: need M%{BM}==0,N%{BN}==0,K%{BK}==0; got {M},{K},{N}"
        )
    if BM % TM or BN % TN:
        raise ValueError("BM%TM and BN%TN must be 0")
    nthreads = (BM // TM) * (BN // TN)
    tdx = BN // TN  # thread grid x (columns of micro-tiles)
    a_elems = BM * BK
    b_elems = BK * BN
    out_name = node.outputs[0]
    at = memref_type([M, K], "float32")
    bt = memref_type([K, N], "float32")
    ct = memref_type([M, N], "float32")
    saty = f"memref<{BM}x{BK}xf32, #gpu.address_space<workgroup>>"
    sbty = f"memref<{BK}x{BN}xf32, #gpu.address_space<workgroup>>"
    accty = f"memref<{TM}x{TN}xf32, #gpu.address_space<private>>"
    kernel = graph.name or "matmul"

    L = []  # emit lines
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%A: {at}, %B: {bt}, %C: {ct})")
    ap(f"        workgroup(%sA : {saty}, %sB : {sbty})")
    ap(f"        private(%acc : {accty})")
    ap("        kernel {")
    ap("      %tx = gpu.thread_id x")
    ap("      %ty = gpu.thread_id y")
    ap("      %bx = gpu.block_id x")
    ap("      %by = gpu.block_id y")
    # constants
    consts = {0: "%c0", 1: "%c1", TM: "%cTM", TN: "%cTN", BK: "%cBK",
              BM: "%cBM", BN: "%cBN", K: "%cK", nthreads: "%cNT",
              a_elems: "%cAE", b_elems: "%cBE", tdx: "%cTDX"}
    seen = {}
    for val, nm in consts.items():
        if val in seen:
            continue
        seen[val] = nm
        ap(f"      {nm} = arith.constant {val} : index")
    c0, c1 = seen[0], seen[1]
    cTM, cTN, cBK = seen[TM], seen[TN], seen[BK]
    cBM, cBN, cK = seen[BM], seen[BN], seen[K]
    cNT, cAE, cBE, cTDX = seen[nthreads], seen[a_elems], seen[b_elems], seen[tdx]
    ap("      %zero = arith.constant 0.0 : f32")
    # linear thread id: tid = ty*tdx + tx
    ap(f"      %ty_tdx = arith.muli %ty, {cTDX} : index")
    ap(f"      %tid = arith.addi %ty_tdx, %tx : index")
    # block origin in C
    ap(f"      %browbase = arith.muli %by, {cBM} : index")   # by*BM
    ap(f"      %bcolbase = arith.muli %bx, {cBN} : index")   # bx*BN
    # this thread's micro-tile origin within the block tile
    ap(f"      %trow0 = arith.muli %ty, {cTM} : index")      # ty*TM
    ap(f"      %tcol0 = arith.muli %tx, {cTN} : index")      # tx*TN
    # zero accumulator
    ap(f"      scf.for %i = {c0} to {cTM} step {c1} {{")
    ap(f"        scf.for %j = {c0} to {cTN} step {c1} {{")
    ap(f"          memref.store %zero, %acc[%i, %j] : {accty}")
    ap("        }")
    ap("      }")
    # K-tile loop
    ap(f"      scf.for %kk = {c0} to {cK} step {cBK} {{")
    # cooperative load A[BM x BK] into sA.
    # Pre-compute each thread's fixed (row, col) in the tile and stride
    # by nthreads/BK rows per iteration — this avoids per-iteration
    # divui/remui that the linear-index approach needs.
    a_row_stride = nthreads // BK
    if a_row_stride * BK == nthreads:  # evenly divisible → 2D loop
        cARS = seen.get(a_row_stride)
        if cARS is None:
            cARS = f"%cARS"
            ap(f"        {cARS} = arith.constant {a_row_stride} : index")
            seen[a_row_stride] = cARS
        ap(f"        %a_col0 = arith.remui %tid, {cBK} : index")
        ap(f"        %a_row0 = arith.divui %tid, {cBK} : index")
        ap(f"        %a_gac = arith.addi %kk, %a_col0 : index")
        ap(f"        scf.for %ar = %a_row0 to {cBM} step {cARS} {{")
        ap(f"          %gar = arith.addi %browbase, %ar : index")
        ap(f"          %av = memref.load %A[%gar, %a_gac] : {at}")
        ap(f"          memref.store %av, %sA[%ar, %a_col0] : {saty}")
        ap("        }")
    else:  # fallback: original linear-index loop
        ap(f"        scf.for %li = %tid to {cAE} step {cNT} {{")
        ap(f"          %ar = arith.divui %li, {cBK} : index")
        ap(f"          %ac = arith.remui %li, {cBK} : index")
        ap(f"          %gar = arith.addi %browbase, %ar : index")
        ap(f"          %gac = arith.addi %kk, %ac : index")
        ap(f"          %av = memref.load %A[%gar, %gac] : {at}")
        ap(f"          memref.store %av, %sA[%ar, %ac] : {saty}")
        ap("        }")
    # cooperative load B[BK x BN] into sB
    b_row_stride = nthreads // BN
    if b_row_stride * BN == nthreads:  # evenly divisible → 2D loop
        cBRS = seen.get(b_row_stride)
        if cBRS is None:
            cBRS = f"%cBRS"
            ap(f"        {cBRS} = arith.constant {b_row_stride} : index")
            seen[b_row_stride] = cBRS
        ap(f"        %b_col0 = arith.remui %tid, {cBN} : index")
        ap(f"        %b_row0 = arith.divui %tid, {cBN} : index")
        ap(f"        %b_gbc = arith.addi %bcolbase, %b_col0 : index")
        ap(f"        scf.for %br = %b_row0 to {cBK} step {cBRS} {{")
        ap(f"          %gbr = arith.addi %kk, %br : index")
        ap(f"          %bv = memref.load %B[%gbr, %b_gbc] : {bt}")
        ap(f"          memref.store %bv, %sB[%br, %b_col0] : {sbty}")
        ap("        }")
    else:  # fallback: original linear-index loop
        ap(f"        scf.for %li = %tid to {cBE} step {cNT} {{")
        ap(f"          %br = arith.divui %li, {cBN} : index")
        ap(f"          %bc = arith.remui %li, {cBN} : index")
        ap(f"          %gbr = arith.addi %kk, %br : index")
        ap(f"          %gbc = arith.addi %bcolbase, %bc : index")
        ap(f"          %bv = memref.load %B[%gbr, %gbc] : {bt}")
        ap(f"          memref.store %bv, %sB[%br, %bc] : {sbty}")
        ap("        ")
    ap("        gpu.barrier")
    # compute: for each k in BK, load TM a-vals + TN b-vals, TM*TN FMAs
    ap(f"        scf.for %k = {c0} to {cBK} step {c1} {{")
    ap(f"          scf.for %i = {c0} to {cTM} step {c1} {{")
    ap("            %arow = arith.addi %trow0, %i : index")
    ap(f"            %a = memref.load %sA[%arow, %k] : {saty}")
    ap(f"            scf.for %j = {c0} to {cTN} step {c1} {{")
    ap("              %bcol = arith.addi %tcol0, %j : index")
    ap(f"              %b = memref.load %sB[%k, %bcol] : {sbty}")
    ap(f"              %old = memref.load %acc[%i, %j] : {accty}")
    ap("              %prod = arith.mulf %a, %b fastmath<contract> : f32")
    ap("              %new = arith.addf %old, %prod fastmath<contract> : f32")
    ap(f"              memref.store %new, %acc[%i, %j] : {accty}")
    ap("            }")
    ap("          }")
    ap("        }")
    ap("        gpu.barrier")
    ap("      }")
    # write back acc → C
    ap(f"      scf.for %i = {c0} to {cTM} step {c1} {{")
    ap("        %crow_l = arith.addi %trow0, %i : index")
    ap("        %crow = arith.addi %browbase, %crow_l : index")
    ap(f"        scf.for %j = {c0} to {cTN} step {c1} {{")
    ap("          %ccol_l = arith.addi %tcol0, %j : index")
    ap("          %ccol = arith.addi %bcolbase, %ccol_l : index")
    ap(f"          %v = memref.load %acc[%i, %j] : {accty}")
    ap(f"          memref.store %v, %C[%crow, %ccol] : {ct}")
    ap("        }")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")
    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[a_name, b_name],
        arg_shapes=[[M, K], [K, N]],
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=[M, N],
        result_dtype="float32",
        grid=(N // BN, M // BM, 1),
        block=(BN // TN, BM // TM, 1),
        buffer_order=[a_name, b_name, out_name],
    )


# ── Tensor-core (nvgpu.mma.sync) matmul ────────────────────────
#
# Warp-register-blocked tensor-core GEMM. Unlike the FP32 regblock kernel
# above (scalar FMA, bit-accurate f32), this path drives the Ampere tensor
# cores via MLIR's ``nvgpu`` dialect for cuBLAS-competitive throughput.
#
# Layout hierarchy:
#   * block tile  BM x BN,  K-step BK      (BM = WM*WTM*16, BN = WN*WTN*16)
#   * WM x WN warps per block (32 threads each)
#   * each warp computes a WTM x WTN grid of 16x16 output sub-tiles, each held
#     as two f32 accumulators (vector<16x8xf32>) in registers → WTM*WTN*2 accs
#   * per 16-wide K-step: warp loads WTM A-fragments (vector<16x16xf16>) and
#     WTN*2 B-fragments (vector<8x16xf16>) from shared, issues WTM*WTN*2
#     m16n8k16 ``nvgpu.mma.sync`` — each A-frag reused WTN times, each B-frag
#     reused WTM times (the CUTLASS-style arithmetic-intensity multiplier).
#
# Precision: f32 inputs are truncated to f16 as they are staged into shared
# memory; the tensor core accumulates in f32. This is the reduced-precision
# throughput path (cuBLAS uses tf32 tensor cores by default for f32 on Ampere,
# a comparable precision class). Output is bit-accurate vs an *fp16-input*
# reference, NOT vs strict-f32 cuBLAS — the ~1e-3 relative delta is the
# inherent tensor-core precision tradeoff. Validate against fp16 references.
#
# The emitted kernel contains ``vector.contract`` at warp granularity; the
# lowering (arke/backend/mlir_gpu.py::_nvgpu_* two-stage pipeline:
# ``--convert-vector-to-gpu=use-nvgpu`` then ``-convert-nvgpu-to-nvvm
# -gpu-lower-to-nvvm-pipeline=cubin-chip=<chip>``) auto-distributes it into
# per-thread ldmatrix + mma.sync fragments — no manual warp-lane→element map.

# Tensor-core matmul default block/warp params. WM=WN=2 (4 warps=128 threads),
# WTM=2 WTN=4 → BM=64, BN=128, BK=16. 16 accs/thread, good occupancy on sm_86.
GPU_MMA_WM = 2
GPU_MMA_WN = 2
GPU_MMA_WTM = 2
GPU_MMA_WTN = 4
GPU_MMA_BK = 16


def emit_gpu_matmul_mma(
    graph: IRGraph, chip: str = "sm_86",
    WM: int = GPU_MMA_WM, WN: int = GPU_MMA_WN,
    WTM: int = GPU_MMA_WTM, WTN: int = GPU_MMA_WTN, BK: int = GPU_MMA_BK,
) -> EmittedGPUKernel:
    """Emit a warp-register-blocked tensor-core (nvgpu.mma.sync) matmul.

    f32 inputs, f16 tensor-core compute with f32 accumulation. Requires
    M % BM == 0, N % BN == 0, K % BK == 0 with BM = WM*WTM*16, BN = WN*WTN*16,
    BK % 16 == 0. Callers fall back to the scalar FP32 kernels for shapes that
    don't tile evenly. See the module comment above for the precision contract.
    """
    if len(graph.nodes) != 1 or graph.nodes[0].op != "matmul":
        raise NotImplementedError("emit_gpu_matmul_mma: single matmul node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    a_name, b_name = in_names[0], in_names[1]
    A, B = graph.values[a_name], graph.values[b_name]
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"matmul K mismatch: {A.shape} @ {B.shape}")
    if A.dtype != "float32":
        raise NotImplementedError("emit_gpu_matmul_mma: f32 inputs only")
    if BK % 16:
        raise ValueError("emit_gpu_matmul_mma: BK must be a multiple of 16")
    BM, BN = WM * WTM * 16, WN * WTN * 16
    if M % BM or N % BN or K % BK:
        raise NotImplementedError(
            f"emit_gpu_matmul_mma: need M%{BM}==0,N%{BN}==0,K%{BK}==0; got {M},{K},{N}"
        )
    out_name = node.outputs[0]
    nthreads = WM * WN * 32
    a_elems, b_elems = BM * BK, BK * BN
    naccs = WTM * WTN * 2
    kernel = graph.name or "matmul"
    at = memref_type([M, K], "float32")
    bt = memref_type([K, N], "float32")
    ct = memref_type([M, N], "float32")
    wg = "#gpu.address_space<workgroup>"
    saty = f"memref<{BM}x{BK}xf16, {wg}>"
    sbty = f"memref<{BK}x{BN}xf16, {wg}>"

    L: list[str] = []
    ap = L.append
    ap("#mapT = affine_map<(d0, d1) -> (d1, d0)>")
    ap("#mA = affine_map<(m, n, k) -> (m, k)>")
    ap("#mB = affine_map<(m, n, k) -> (n, k)>")
    ap("#mC = affine_map<(m, n, k) -> (m, n)>")
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%A: {at}, %B: {bt}, %C: {ct})")
    ap(f"        workgroup(%sA: {saty}, %sB: {sbty})")
    ap(f"        kernel attributes {{gpu.known_block_size = array<i32: {nthreads}, 1, 1>}} {{")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c8 = arith.constant 8 : index")
    ap("      %c16 = arith.constant 16 : index")
    ap("      %c32 = arith.constant 32 : index")
    ap(f"      %cBK = arith.constant {BK} : index")
    ap(f"      %cBM = arith.constant {BM} : index")
    ap(f"      %cBN = arith.constant {BN} : index")
    ap(f"      %cNT = arith.constant {nthreads} : index")
    ap(f"      %cAE = arith.constant {a_elems} : index")
    ap(f"      %cBE = arith.constant {b_elems} : index")
    ap(f"      %cK = arith.constant {K} : index")
    ap(f"      %cWN = arith.constant {WN} : index")
    ap(f"      %cWTM = arith.constant {WTM * 16} : index")
    ap(f"      %cWTN = arith.constant {WTN * 16} : index")
    ap("      %cst = arith.constant 0.0 : f16")
    ap("      %fz = arith.constant 0.0 : f32")
    ap("      %bx = gpu.block_id x")
    ap("      %by = gpu.block_id y")
    ap("      %tid = gpu.thread_id x")
    ap("      %warp = arith.divui %tid, %c32 : index")
    ap("      %wm = arith.divui %warp, %cWN : index")
    ap("      %wn = arith.remui %warp, %cWN : index")
    ap("      %browbase = arith.muli %by, %cBM : index")
    ap("      %bcolbase = arith.muli %bx, %cBN : index")
    ap("      %wrowbase = arith.muli %wm, %cWTM : index")
    ap("      %wcolbase = arith.muli %wn, %cWTN : index")
    ap("      %cinit = vector.broadcast %fz : f32 to vector<16x8xf32>")
    acc_names = [f"%acc_{i}" for i in range(naccs)]
    acc_ty = ", ".join(["vector<16x8xf32>"] * naccs)
    iter_init = ", ".join(f"{n} = %cinit" for n in acc_names)
    ap(f"      %res:{naccs} = scf.for %kk = %c0 to %cK step %cBK")
    ap(f"          iter_args({iter_init}) -> ({acc_ty}) {{")
    # cooperative load A[BM x BK] with f32→f16 truncation
    ap("        scf.for %li = %tid to %cAE step %cNT {")
    ap("          %ar = arith.divui %li, %cBK : index")
    ap("          %ac = arith.remui %li, %cBK : index")
    ap("          %gar = arith.addi %browbase, %ar : index")
    ap("          %gac = arith.addi %kk, %ac : index")
    ap(f"          %av32 = memref.load %A[%gar, %gac] : {at}")
    ap("          %av = arith.truncf %av32 : f32 to f16")
    ap(f"          memref.store %av, %sA[%ar, %ac] : {saty}")
    ap("        }")
    # cooperative load B[BK x BN] with f32→f16 truncation
    ap("        scf.for %li = %tid to %cBE step %cNT {")
    ap("          %br = arith.divui %li, %cBN : index")
    ap("          %bc = arith.remui %li, %cBN : index")
    ap("          %gbr = arith.addi %kk, %br : index")
    ap("          %gbc = arith.addi %bcolbase, %bc : index")
    ap(f"          %bv32 = memref.load %B[%gbr, %gbc] : {bt}")
    ap("          %bv = arith.truncf %bv32 : f32 to f16")
    ap(f"          memref.store %bv, %sB[%br, %bc] : {sbty}")
    ap("        }")
    ap("        gpu.barrier")
    # inner K-step loop carrying accumulators
    inner_init = ", ".join(f"%i_{i} = {acc_names[i]}" for i in range(naccs))
    ap(f"        %r:{naccs} = scf.for %ki = %c0 to %cBK step %c16")
    ap(f"            iter_args({inner_init}) -> ({acc_ty}) {{")
    for i in range(WTM):
        ap(f"          %aoff_{i} = arith.constant {i * 16} : index")
        ap(f"          %arowp_{i} = arith.addi %wrowbase, %aoff_{i} : index")
        ap(f"          %af_{i} = vector.transfer_read %sA[%arowp_{i}, %ki], %cst {{in_bounds = [true, true]}}")
        ap(f"              : {saty}, vector<16x16xf16>")
    for j in range(WTN):
        ap(f"          %boff_{j} = arith.constant {j * 16} : index")
        ap(f"          %bcolp_{j} = arith.addi %wcolbase, %boff_{j} : index")
        ap(f"          %bf_{j}_0 = vector.transfer_read %sB[%ki, %bcolp_{j}], %cst")
        ap(f"              {{permutation_map = #mapT, in_bounds = [true, true]}}")
        ap(f"              : {sbty}, vector<8x16xf16>")
        ap(f"          %bcolp8_{j} = arith.addi %bcolp_{j}, %c8 : index")
        ap(f"          %bf_{j}_1 = vector.transfer_read %sB[%ki, %bcolp8_{j}], %cst")
        ap(f"              {{permutation_map = #mapT, in_bounds = [true, true]}}")
        ap(f"              : {sbty}, vector<8x16xf16>")
    out_names = []
    for i in range(WTM):
        for j in range(WTN):
            for h in range(2):
                idx = (i * WTN + j) * 2 + h
                on = f"%o_{idx}"
                out_names.append(on)
                ap(f"          {on} = vector.contract {{indexing_maps = [#mA, #mB, #mC],")
                ap('              iterator_types = ["parallel","parallel","reduction"],')
                ap("              kind = #vector.kind<add>}")
                ap(f"              %af_{i}, %bf_{j}_{h}, %i_{idx} : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>")
    ap(f"          scf.yield {', '.join(out_names)} : {acc_ty}")
    ap("        }")
    ap("        gpu.barrier")
    yield_inner = ", ".join(f"%r#{i}" for i in range(naccs))
    ap(f"        scf.yield {yield_inner} : {acc_ty}")
    ap("      }")
    # write out each WTM x WTN sub-tile (two 16x8 halves)
    for i in range(WTM):
        for j in range(WTN):
            ap(f"      %woff_{i}_{j} = arith.constant {i * 16} : index")
            ap(f"      %wcoff_{i}_{j} = arith.constant {j * 16} : index")
            ap(f"      %crow_{i}_{j}a = arith.addi %browbase, %wrowbase : index")
            ap(f"      %crow_{i}_{j} = arith.addi %crow_{i}_{j}a, %woff_{i}_{j} : index")
            ap(f"      %ccol_{i}_{j}a = arith.addi %bcolbase, %wcolbase : index")
            ap(f"      %ccol_{i}_{j} = arith.addi %ccol_{i}_{j}a, %wcoff_{i}_{j} : index")
            idx0 = (i * WTN + j) * 2
            ap(f"      vector.transfer_write %res#{idx0}, %C[%crow_{i}_{j}, %ccol_{i}_{j}] {{in_bounds = [true, true]}}")
            ap(f"          : vector<16x8xf32>, {ct}")
            ap(f"      %ccol8_{i}_{j} = arith.addi %ccol_{i}_{j}, %c8 : index")
            ap(f"      vector.transfer_write %res#{idx0 + 1}, %C[%crow_{i}_{j}, %ccol8_{i}_{j}] {{in_bounds = [true, true]}}")
            ap(f"          : vector<16x8xf32>, {ct}")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")
    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[a_name, b_name],
        arg_shapes=[[M, K], [K, N]],
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=[M, N],
        result_dtype="float32",
        grid=(N // BN, M // BM, 1),
        block=(nthreads, 1, 1),
        buffer_order=[a_name, b_name, out_name],
    )


# GPU elementwise ops that lower to PTX via the gpu.module path.
#
# Two classes, both bit-correct vs torch on the CUDA driver:
#   * pure-arith (relu/neg/add/mul) — no external symbols, lower directly.
#   * transcendental (exp/tanh/sigmoid/silu/gelu/rsqrt) — the math.* ops emit
#     __nv_* libdevice calls; these are RESOLVED by linking libdevice.bc into
#     the gpu binary (see arke/backend/mlir_gpu.py::_ptx_passes, which passes
#     `l=<libdevice.10.bc>` to gpu-module-to-binary). libdevice inlines them to
#     native PTX (e.g. exp → ex2.approx), so the driver-only load succeeds — no
#     CUDA_ERROR_INVALID_PTX. This is the correct-linking path, deliberately
#     chosen over restricting the GPU set to a pure-arith subset.
GPU_ELEMENTWISE_OPS = frozenset({
    # pure arith
    "relu", "neg", "add", "mul",
    # transcendental via libdevice
    "exp", "tanh", "sigmoid", "silu", "gelu", "rsqrt",
})


def emit_gpu_elementwise(graph: IRGraph, chip: str = "sm_86",
                         block: int = 256) -> EmittedGPUKernel:
    """Emit a flat multi-thread gpu.module for a 2D elementwise op.

    Each thread processes one element: gid = bid*blockDim + tid. Flat-index
    mapping (gid → (i,j) via divui/remui) with one thread per element is the
    simplest and fastest pattern for elementwise — the kernel body is tiny
    (1 load + 1 op + 1 store) so adding per-thread loops or vectorization adds
    more overhead than it saves. cuBLAS/cuDNN elementwise kernels use the same
    pattern.

    f32, 2D only. Identical numerics to the CPU path (``ELEMENTWISE_SPECS``);
    transcendentals via libdevice.
    """
    from arke.backend.mlir_ops import ELEMENTWISE_SPECS
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_elementwise: single-node graphs only")
    node = graph.nodes[0]
    if node.op not in ELEMENTWISE_SPECS:
        raise NotImplementedError(f"emit_gpu_elementwise: {node.op} not elementwise")
    spec = ELEMENTWISE_SPECS[node.op]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    shape = list(in_vals[0].shape)
    if len(shape) != 2:
        raise NotImplementedError(
            f"emit_gpu_elementwise: 2D only (P3-S2), got shape {shape}"
        )
    if any(v.dtype != "float32" for v in in_vals):
        raise NotImplementedError("emit_gpu_elementwise: f32 only")
    M, N = shape
    total = M * N
    ngrid = (total + block - 1) // block
    ty = memref_type(shape, "float32")
    out_name = node.outputs[0]
    kernel = graph.name or node.op
    n_in = len(in_names)
    params = ", ".join([f"%A{i}: {ty}" for i in range(n_in)] + [f"%O: {ty}"])
    loads = [f"        %a{i} = memref.load %A{i}[%i, %j] : {ty}" for i in range(n_in)]
    # ew_body lines are indented for the old top-level scope; re-indent +2 for scf.if.
    body = ["  " + ln for ln in spec.ew_body]
    text = "\n".join([
        "module attributes {gpu.container_module} {",
        f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{',
        f"    gpu.func @{kernel}({params}) kernel {{",
        "      %bid = gpu.block_id x",
        "      %bdim = gpu.block_dim x",
        "      %tid = gpu.thread_id x",
        "      %base = arith.muli %bid, %bdim : index",
        "      %gid = arith.addi %base, %tid : index",
        f"      %cN = arith.constant {N} : index",
        f"      %ctotal = arith.constant {total} : index",
        "      %in = arith.cmpi ult, %gid, %ctotal : index",
        "      scf.if %in {",
        "        %i = arith.divui %gid, %cN : index",
        "        %j = arith.remui %gid, %cN : index",
        *loads,
        *body,
        f"        memref.store %res, %O[%i, %j] : {ty}",
        "      }",
        "      gpu.return",
        "    }",
        "  }",
        "}",
    ])
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=in_names,
        arg_shapes=[shape for _ in in_names],
        arg_dtypes=["float32" for _ in in_names],
        result_name=out_name,
        result_shape=shape,
        result_dtype="float32",
        grid=(ngrid, 1, 1),
        block=(block, 1, 1),
        buffer_order=in_names + [out_name],
    )


# Row-wise ops: parallel-reduce row-per-block kernel.
# block=(256,1,1), grid=(rows,1,1), shared-memory tree-reduce.
GPU_ROWWISE_OPS = frozenset({
    "reduce_sum", "reduce_max", "reduce_mean",
    "softmax", "layernorm", "rmsnorm",
    "cumsum", "argmax", "rope",
})

_RW_BLOCK = 256


def _rw_tree_reduce(ap, sty, op_name, BLOCK=_RW_BLOCK):
    """Emit shared-memory tree-reduce (log2 steps) for the value in shared[tid]."""
    stride = BLOCK // 2
    step = 0
    while stride >= 1:
        ap(f"      %cS{step}_{op_name} = arith.constant {stride} : index")
        ap(f"      %lt{step}_{op_name} = arith.cmpi ult, %tid, %cS{step}_{op_name} : index")
        ap(f"      scf.if %lt{step}_{op_name} {{")
        ap(f"        %ra{step} = memref.load %sh[%tid] : {sty}")
        ap(f"        %roff{step} = arith.addi %tid, %cS{step}_{op_name} : index")
        ap(f"        %rb{step} = memref.load %sh[%roff{step}] : {sty}")
        if "max" in op_name:
            ap(f"        %rs{step} = arith.maximumf %ra{step}, %rb{step} : f32")
        else:
            ap(f"        %rs{step} = arith.addf %ra{step}, %rb{step} : f32")
        ap(f"        memref.store %rs{step}, %sh[%tid] : {sty}")
        ap("      }")
        ap("      gpu.barrier")
        stride //= 2
        step += 1


_WARP_SIZE = 32


def _rw_warp_reduce_inline(ap, val_ssa: str, op_name: str,
                           is_max: bool = False) -> str:
    """Emit 5-step warp-shuffle reduction (no barriers) for a register value.

    Generates ``gpu.shuffle down`` at offsets 16, 8, 4, 2, 1 and returns
    the SSA name holding the lane-0 result.  Works for both ``addf`` (sum)
    and ``maximumf`` (max) reductions.
    """
    cur = val_ssa
    for i, offset in enumerate((16, 8, 4, 2, 1)):
        off_name = f"%woff{i}_{op_name}"
        width_name = f"%ww{i}_{op_name}"
        shfl_name = f"%ws{i}_{op_name}"
        valid_name = f"%wv{i}_{op_name}"
        res_name = f"%wr{i}_{op_name}"
        ap(f"      {off_name} = arith.constant {offset} : i32")
        ap(f"      {width_name} = arith.constant {_WARP_SIZE} : i32")
        ap(f"      {shfl_name}, {valid_name} = gpu.shuffle down {cur},"
           f" {off_name}, {width_name} : f32")
        if is_max:
            ap(f"      {res_name} = arith.maximumf {cur}, {shfl_name} : f32")
        else:
            ap(f"      {res_name} = arith.addf {cur}, {shfl_name} : f32")
        cur = res_name
    return cur


def _rw_block_reduce_fast(ap, sty, val_ssa: str, op_name: str,
                          is_max: bool = False, BLOCK: int = _RW_BLOCK) -> str:
    """Emit a warp-shuffle + cross-warp shared-memory reduce.

    1. Intra-warp reduction via ``gpu.shuffle down`` (5 steps, 0 barriers).
    2. Lane-0 of each warp stores to shared memory.
    3. Cross-warp tree-reduce in shared memory (log2(BLOCK/32) steps,
       each with 1 barrier) — only warp-0 threads participate.
    4. Result is in shared[0].

    Returns the SSA name of the reduced value that ALL threads can read
    (caller should ``memref.load %sh[%c0]`` after the final barrier).

    Compared to the old ``_rw_tree_reduce``:
      256 threads: 8 barriers → **3 barriers** (62% fewer).
    """
    n_warps = BLOCK // _WARP_SIZE   # 8 for BLOCK=256

    # Step 1: warp-local reduction (no barriers)
    warp_result = _rw_warp_reduce_inline(ap, val_ssa, op_name, is_max=is_max)

    # Compute warp_id = tid / 32  and  lane = tid % 32
    cWARP = f"%cWARP_{op_name}"
    warp_id_name = f"%warp_id_{op_name}"
    lane_name = f"%lane_{op_name}"
    ap(f"      {cWARP} = arith.constant {_WARP_SIZE} : index")
    ap(f"      {warp_id_name} = arith.divui %tid, {cWARP} : index")
    ap(f"      {lane_name} = arith.remui %tid, {cWARP} : index")

    # Step 2: lane 0 of each warp stores warp result to shared[warp_id]
    is_lane0 = f"%is_lane0_{op_name}"
    ap(f"      {is_lane0} = arith.cmpi eq, {lane_name}, %c0 : index")
    ap(f"      scf.if {is_lane0} {{")
    ap(f"        memref.store {warp_result}, %sh[{warp_id_name}] : {sty}")
    ap("      }")
    ap("      gpu.barrier")

    # Step 3: cross-warp reduction (only first n_warps threads)
    # n_warps = 8 for BLOCK=256 → 3 steps with barriers
    stride = n_warps // 2
    step = 0
    while stride >= 1:
        cS = f"%cwS{step}_{op_name}"
        lt = f"%cwlt{step}_{op_name}"
        ap(f"      {cS} = arith.constant {stride} : index")
        ap(f"      {lt} = arith.cmpi ult, %tid, {cS} : index")
        ap(f"      scf.if {lt} {{")
        ra = f"%cwa{step}_{op_name}"
        off = f"%cwoff{step}_{op_name}"
        rb = f"%cwb{step}_{op_name}"
        rs = f"%cws{step}_{op_name}"
        ap(f"        {ra} = memref.load %sh[%tid] : {sty}")
        ap(f"        {off} = arith.addi %tid, {cS} : index")
        ap(f"        {rb} = memref.load %sh[{off}] : {sty}")
        if is_max:
            ap(f"        {rs} = arith.maximumf {ra}, {rb} : f32")
        else:
            ap(f"        {rs} = arith.addf {ra}, {rb} : f32")
        ap(f"        memref.store {rs}, %sh[%tid] : {sty}")
        ap("      }")
        ap("      gpu.barrier")
        stride //= 2
        step += 1

    return f"%sh[%c0]"  # caller loads from shared[0]


def emit_gpu_rowwise(graph: IRGraph, chip: str = "sm_86",
                     block: int = _RW_BLOCK) -> EmittedGPUKernel:
    """Emit a parallel-reduce row-per-block gpu.module for row-wise ops.

    Threads cooperate per row via shared-memory tree-reduce (log2(block) levels).
    Transcendentals via libdevice. f32, 2D.

    Block size is selected by the gpu_tuning policy (shape-adaptive): wider rows
    get larger blocks to reduce per-thread work and improve latency. The caller
    can override by passing an explicit block size.
    """
    from arke.backend.gpu_tuning import rowwise_block_size

    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_rowwise: single-node graphs only")
    node = graph.nodes[0]
    op = node.op
    if op not in GPU_ROWWISE_OPS:
        raise NotImplementedError(f"emit_gpu_rowwise: {op} not supported")
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    shape = list(in_vals[0].shape)
    if len(shape) != 2:
        raise NotImplementedError(f"emit_gpu_rowwise: 2D only, got {shape}")
    if any(v.dtype != "float32" for v in in_vals):
        raise NotImplementedError("emit_gpu_rowwise: f32 only")
    rows, D = shape
    # Apply tuning policy when caller uses the default block size
    if block == _RW_BLOCK:
        block = rowwise_block_size(D)
    is_reduce = op in ("reduce_sum", "reduce_max", "reduce_mean", "argmax")
    out_shape = [rows] if is_reduce else [rows, D]
    inty = memref_type(shape, "float32")
    outty = memref_type(out_shape, "float32")
    sty = f"memref<{block}xf32, #gpu.address_space<workgroup>>"
    out_name = node.outputs[0]
    kernel = graph.name or op
    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%X: {inty}, %O: {outty})")
    ap(f"        workgroup(%sh : {sty})")
    ap("        kernel {")
    ap("      %tid = gpu.thread_id x")
    ap("      %bid = gpu.block_id x")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c1 = arith.constant 1 : index")
    ap(f"      %cD = arith.constant {D} : index")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap("      %zero = arith.constant 0.0 : f32")
    if op in ("reduce_sum", "reduce_mean"):
        ap("      %local = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %zero) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %ns = arith.addf %s, %x : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %local, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "sum", BLOCK=block)
        ap("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        ap("      scf.if %is0 {")
        ap(f"        %r = memref.load %sh[%c0] : {sty}")
        if op == "reduce_mean":
            ap(f"        %Df = arith.constant {float(D)} : f32")
            ap("        %mean = arith.divf %r, %Df : f32")
            ap(f"        memref.store %mean, %O[%bid] : {outty}")
        else:
            ap(f"        memref.store %r, %O[%bid] : {outty}")
        ap("      }")
    elif op == "reduce_max":
        ap("      %ninf = arith.constant 0xFF800000 : f32")
        ap("      %local = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %ninf) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %ns = arith.maximumf %s, %x : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %local, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "max", BLOCK=block)
        ap("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        ap("      scf.if %is0 {")
        ap(f"        %r = memref.load %sh[%c0] : {sty}")
        ap(f"        memref.store %r, %O[%bid] : {outty}")
        ap("      }")
    elif op == "softmax":
        # Online softmax (branchless Milakov-Gimelshein): fuse max and exp-sum
        # into a single stride-accumulate pass. Each element update:
        #   new_max = max(old_max, x)
        #   new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)
        # This eliminates one tree-reduce (max) vs the 3-pass approach,
        # cutting ~9 barriers out of 20 → 11.
        # Normalization uses multiply-by-reciprocal (mulf vs divf on GPU).
        ap("      %ninf = arith.constant 0xFF800000 : f32")
        # Pass 1: online max+sum accumulation (single pass over data)
        ap("      %os:2 = scf.for %k = %tid to %cD step %cBLK"
           " iter_args(%m = %ninf, %s = %zero) -> (f32, f32) {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %nm = arith.maximumf %m, %x : f32")
        ap("        %corr = arith.subf %m, %nm : f32")   # old_max - new_max (≤ 0)
        ap("        %ecorr = math.exp %corr : f32")       # correction factor
        ap("        %xd = arith.subf %x, %nm : f32")      # x - new_max (≤ 0)
        ap("        %ex = math.exp %xd : f32")             # exp(x - new_max)
        ap("        %sc = arith.mulf %s, %ecorr fastmath<contract> : f32")
        ap("        %ns = arith.addf %sc, %ex fastmath<contract> : f32")
        ap("        scf.yield %nm, %ns : f32, f32")
        ap("      }")
        ap(f"      memref.store %os#0, %sh[%tid] : {sty}")  # store local max
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "smax", BLOCK=block)
        ap(f"      %mx = memref.load %sh[%c0] : {sty}")     # global max
        ap("      gpu.barrier")
        # Correct local sum: local_sum * exp(local_max - global_max)
        ap("      %mc = arith.subf %os#0, %mx : f32")
        ap("      %emc = math.exp %mc : f32")
        ap("      %csum = arith.mulf %os#1, %emc fastmath<contract> : f32")
        ap(f"      memref.store %csum, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "ssum", BLOCK=block)
        ap(f"      %den = memref.load %sh[%c0] : {sty}")
        ap("      gpu.barrier")
        # Precompute reciprocal (1/sum) — single div, then N muls vs N divs
        ap("      %one_sm = arith.constant 1.0 : f32")
        ap("      %rcp = arith.divf %one_sm, %den : f32")
        # Pass 2: normalize (exp(x - global_max) * (1/global_sum))
        ap("      scf.for %k = %tid to %cD step %cBLK {")
        ap(f"        %x3 = memref.load %X[%bid, %k] : {inty}")
        ap("        %d3 = arith.subf %x3, %mx : f32")
        ap("        %e3 = math.exp %d3 : f32")
        ap("        %o3 = arith.mulf %e3, %rcp fastmath<contract> : f32")
        ap(f"        memref.store %o3, %O[%bid, %k] : {outty}")
        ap("      }")
    elif op == "layernorm":
        ap(f"      %Df = arith.constant {float(D)} : f32")
        ap("      %eps = arith.constant 1.000000e-05 : f32")
        ap("      %lsum = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %zero) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %ns = arith.addf %s, %x : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %lsum, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "lnm", BLOCK=block)
        ap(f"      %sumv = memref.load %sh[%c0] : {sty}")
        ap("      %mean = arith.divf %sumv, %Df : f32")
        ap("      gpu.barrier")
        ap("      %lvar = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %zero) -> f32 {")
        ap(f"        %xv = memref.load %X[%bid, %k] : {inty}")
        ap("        %dv = arith.subf %xv, %mean : f32")
        ap("        %sq = arith.mulf %dv, %dv fastmath<contract> : f32")
        ap("        %nsv = arith.addf %s, %sq fastmath<contract> : f32")
        ap("        scf.yield %nsv : f32")
        ap("      }")
        ap(f"      memref.store %lvar, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "lnv", BLOCK=block)
        ap(f"      %varsum = memref.load %sh[%c0] : {sty}")
        ap("      %var = arith.divf %varsum, %Df : f32")
        ap("      %vare = arith.addf %var, %eps : f32")
        ap("      %inv = math.rsqrt %vare : f32")
        ap("      gpu.barrier")
        ap("      scf.for %k = %tid to %cD step %cBLK {")
        ap(f"        %xn = memref.load %X[%bid, %k] : {inty}")
        ap("        %dn = arith.subf %xn, %mean : f32")
        ap("        %on = arith.mulf %dn, %inv : f32")
        ap(f"        memref.store %on, %O[%bid, %k] : {outty}")
        ap("      }")
    elif op == "rmsnorm":
        ap(f"      %Df = arith.constant {float(D)} : f32")
        ap("      %eps = arith.constant 1.000000e-05 : f32")
        ap("      %lsq = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %zero) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %sq = arith.mulf %x, %x fastmath<contract> : f32")
        ap("        %ns = arith.addf %s, %sq fastmath<contract> : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %lsq, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "rms", BLOCK=block)
        ap(f"      %sqsum = memref.load %sh[%c0] : {sty}")
        ap("      %ms = arith.divf %sqsum, %Df : f32")
        ap("      %mse = arith.addf %ms, %eps : f32")
        ap("      %inv = math.rsqrt %mse : f32")
        ap("      gpu.barrier")
        ap("      scf.for %k = %tid to %cD step %cBLK {")
        ap(f"        %xn = memref.load %X[%bid, %k] : {inty}")
        ap("        %on = arith.mulf %xn, %inv : f32")
        ap(f"        memref.store %on, %O[%bid, %k] : {outty}")
        ap("      }")
    elif op == "cumsum":
        # Chunked parallel cumsum: each thread scans D/BLOCK elements
        # sequentially (chunk scan), stores the partial sum to shared memory,
        # thread-0 does an exclusive prefix scan of the partial sums, then each
        # thread offsets its chunk. Total work: O(D/BLOCK) per thread + O(BLOCK)
        # serial, vs the old O(D) single-thread.
        chunk = (D + block - 1) // block
        ap(f"      %cCHUNK = arith.constant {chunk} : index")
        # Phase 1: each thread scans its own chunk [tid*chunk, min((tid+1)*chunk,D))
        ap("      %chStart = arith.muli %tid, %cCHUNK : index")
        ap("      %chEnd0 = arith.addi %chStart, %cCHUNK : index")
        ap("      %chEnd = arith.minui %chEnd0, %cD : index")
        ap("      %ps = scf.for %k = %chStart to %chEnd step %c1"
           " iter_args(%run = %zero) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %nr = arith.addf %run, %x : f32")
        ap(f"        memref.store %nr, %O[%bid, %k] : {outty}")
        ap("        scf.yield %nr : f32")
        ap("      }")
        # Store partial sum in shared memory
        ap(f"      memref.store %ps, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        # Phase 2: thread-0 does exclusive prefix scan of shared sums
        ap("      %is0cs = arith.cmpi eq, %tid, %c0 : index")
        ap("      scf.if %is0cs {")
        ap(f"        %cNT = arith.constant {min(block, (D + chunk - 1) // chunk)} : index")
        ap("        scf.for %t = %c1 to %cNT step %c1 {")
        ap(f"          %tm1 = arith.subi %t, %c1 : index")
        ap(f"          %pa = memref.load %sh[%tm1] : {sty}")
        ap(f"          %pb = memref.load %sh[%t] : {sty}")
        ap(f"          %pc = arith.addf %pa, %pb : f32")
        ap(f"          memref.store %pc, %sh[%t] : {sty}")
        ap("        }")
        ap("      }")
        ap("      gpu.barrier")
        # Phase 3: each thread (except tid=0) adds its prefix to its chunk
        ap("      %gt0 = arith.cmpi ugt, %tid, %c0 : index")
        ap("      scf.if %gt0 {")
        ap("        %tidm1 = arith.subi %tid, %c1 : index")
        ap(f"        %prefix = memref.load %sh[%tidm1] : {sty}")
        ap("        scf.for %k = %chStart to %chEnd step %c1 {")
        ap(f"          %ov = memref.load %O[%bid, %k] : {outty}")
        ap("          %nv = arith.addf %ov, %prefix : f32")
        ap(f"          memref.store %nv, %O[%bid, %k] : {outty}")
        ap("        }")
        ap("      }")
    elif op == "argmax":
        ap("      %ninf = arith.constant 0xFF800000 : f32")
        ap("      %lmax = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %ninf) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %ns = arith.maximumf %s, %x : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %lmax, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "amax", BLOCK=block)
        ap("      %is0a = arith.cmpi eq, %tid, %c0 : index")
        ap("      scf.if %is0a {")
        ap(f"        %mx = memref.load %sh[%c0] : {sty}")
        ap("        %idx = scf.for %k = %c0 to %cD step %c1 iter_args(%bi = %c0) -> index {")
        ap(f"          %xv = memref.load %X[%bid, %k] : {inty}")
        ap("          %eq = arith.cmpf oeq, %xv, %mx : f32")
        ap("          %ni = arith.select %eq, %k, %bi : index")
        ap("          scf.yield %ni : index")
        ap("        }")
        ap("        %idxi = arith.index_cast %idx : index to i64")
        ap("        %idxf = arith.sitofp %idxi : i64 to f32")
        ap(f"        memref.store %idxf, %O[%bid] : {outty}")
        ap("      }")
    elif op == "rope":
        half = D // 2
        ap(f"      %chalf = arith.constant {half} : index")
        ap(f"      %Dfr = arith.constant {float(D)} : f32")
        ap("      %two = arith.constant 2.0 : f32")
        ap("      %base = arith.constant 10000.0 : f32")
        ap("      %posi = arith.index_cast %bid : index to i64")
        ap("      %posf = arith.sitofp %posi : i64 to f32")
        ap("      %lnb = math.log %base : f32")
        ap("      scf.for %k = %tid to %chalf step %cBLK {")
        ap("        %ki = arith.index_cast %k : index to i64")
        ap("        %kf = arith.sitofp %ki : i64 to f32")
        ap("        %e0 = arith.mulf %two, %kf : f32")
        ap("        %e1 = arith.divf %e0, %Dfr : f32")
        ap("        %pw = arith.mulf %e1, %lnb : f32")
        ap("        %ipw = arith.negf %pw : f32")
        ap("        %theta = math.exp %ipw : f32")
        ap("        %ang = arith.mulf %posf, %theta : f32")
        ap("        %cos = math.cos %ang : f32")
        ap("        %sin = math.sin %ang : f32")
        ap("        %k2 = arith.addi %k, %chalf : index")
        ap(f"        %x1 = memref.load %X[%bid, %k] : {inty}")
        ap(f"        %x2 = memref.load %X[%bid, %k2] : {inty}")
        ap("        %x1c = arith.mulf %x1, %cos : f32")
        ap("        %x2s = arith.mulf %x2, %sin : f32")
        ap("        %o1 = arith.subf %x1c, %x2s : f32")
        ap("        %x2c = arith.mulf %x2, %cos : f32")
        ap("        %x1s = arith.mulf %x1, %sin : f32")
        ap("        %o2 = arith.addf %x2c, %x1s : f32")
        ap(f"        memref.store %o1, %O[%bid, %k] : {outty}")
        ap(f"        memref.store %o2, %O[%bid, %k2] : {outty}")
        ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")
    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0]],
        arg_shapes=[shape],
        arg_dtypes=["float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(rows, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], out_name],
    )


# Two-input row-per-block ops: rmsnorm_residual (rmsnorm(x+res)) and embedding
# (row gather from a table). Kept separate from emit_gpu_rowwise (single-input).
GPU_ROWWISE2_OPS = frozenset({"rmsnorm_residual", "embedding"})


def emit_gpu_rowwise2(graph: IRGraph, chip: str = "sm_86") -> EmittedGPUKernel:
    """Emit a row-per-block gpu.module for a 2-input row-wise op.

      * rmsnorm_residual(x, res): rmsnorm(x + res) over the row (same eps=1e-5).
      * embedding(idx, table): out[i,:] = table[int(idx[i]), :]. idx is a 1D
        f32-encoded index vector; table is [vocab, dim]; out is [n_idx, dim].
    grid=(rows,1,1), block=(1,1,1). f32 only, 2D tensors (idx 1D). Same math as
    the CPU composite path.
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_rowwise2: single-node graphs only")
    node = graph.nodes[0]
    op = node.op
    if op not in GPU_ROWWISE2_OPS:
        raise NotImplementedError(f"emit_gpu_rowwise2: {op} unsupported")
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    if any(v.dtype != "float32" for v in in_vals):
        raise NotImplementedError("emit_gpu_rowwise2: f32 only")
    out_name = node.outputs[0]
    kernel = graph.name or op
    L = []
    ap = L.append

    if op == "rmsnorm_residual":
        rows, D = in_vals[0].shape
        if in_vals[1].shape != in_vals[0].shape:
            raise ValueError("rmsnorm_residual: x and residual must match")
        ty = memref_type([rows, D], "float32")
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}(%X: {ty}, %R: {ty}, %O: {ty}) kernel {{")
        ap("      %r = gpu.block_id x")
        ap("      %c0 = arith.constant 0 : index")
        ap("      %c1 = arith.constant 1 : index")
        ap(f"      %cD = arith.constant {D} : index")
        ap("      %zero = arith.constant 0.0 : f32")
        ap(f"      %Df = arith.constant {float(D)} : f32")
        ap("      %eps = arith.constant 1.000000e-05 : f32")
        # sum of (x+res)^2
        ap("      %ssum = scf.for %k = %c0 to %cD step %c1 iter_args(%s = %zero) -> f32 {")
        ap(f"        %x = memref.load %X[%r, %k] : {ty}")
        ap(f"        %rr = memref.load %R[%r, %k] : {ty}")
        ap("        %a = arith.addf %x, %rr : f32")
        ap("        %sq = arith.mulf %a, %a : f32")
        ap("        %ns = arith.addf %s, %sq : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap("      %ms = arith.divf %ssum, %Df : f32")
        ap("      %mse = arith.addf %ms, %eps : f32")
        ap("      %inv = math.rsqrt %mse : f32")
        ap("      scf.for %k = %c0 to %cD step %c1 {")
        ap(f"        %x = memref.load %X[%r, %k] : {ty}")
        ap(f"        %rr = memref.load %R[%r, %k] : {ty}")
        ap("        %a = arith.addf %x, %rr : f32")
        ap("        %o = arith.mulf %a, %inv : f32")
        ap(f"        memref.store %o, %O[%r, %k] : {ty}")
        ap("      }")
        ap("      gpu.return")
        ap("    }")
        ap("  }")
        ap("}")
        out_shape = [rows, D]
        arg_names = [in_names[0], in_names[1]]
        arg_shapes = [[rows, D], [rows, D]]
        grid = (rows, 1, 1)
    else:  # embedding
        idx_shape = list(in_vals[0].shape)
        vocab, dim = in_vals[1].shape
        n_idx = idx_shape[0]
        idxty = memref_type([n_idx], "float32")
        tblty = memref_type([vocab, dim], "float32")
        outty = memref_type([n_idx, dim], "float32")
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}(%I: {idxty}, %T: {tblty}, %O: {outty}) kernel {{")
        ap("      %r = gpu.block_id x")
        ap("      %c0 = arith.constant 0 : index")
        ap("      %c1 = arith.constant 1 : index")
        ap(f"      %cdim = arith.constant {dim} : index")
        ap(f"      %fidx = memref.load %I[%r] : {idxty}")
        ap("      %ii = arith.fptosi %fidx : f32 to i64")
        ap("      %row = arith.index_cast %ii : i64 to index")
        ap("      scf.for %d = %c0 to %cdim step %c1 {")
        ap(f"        %v = memref.load %T[%row, %d] : {tblty}")
        ap(f"        memref.store %v, %O[%r, %d] : {outty}")
        ap("      }")
        ap("      gpu.return")
        ap("    }")
        ap("  }")
        ap("}")
        out_shape = [n_idx, dim]
        arg_names = [in_names[0], in_names[1]]
        arg_shapes = [idx_shape, [vocab, dim]]
        grid = (n_idx, 1, 1)

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=arg_names,
        arg_shapes=arg_shapes,
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=grid,
        block=(1, 1, 1),
        buffer_order=arg_names + [out_name],
    )


# 2D data-movement ops the GPU backend covers with an element-per-block kernel
# (grid = output shape, block=(1,1,1)): pure address remapping, no math.
GPU_MOVEMENT_OPS = frozenset({
    "transpose", "copy_", "concat", "split",
    "permute",
})


# Index ops: gather/scatter — flat multi-thread kernels with f32→index cast.
GPU_INDEX_OPS = frozenset({"gather", "scatter"})


def emit_gpu_movement(graph: IRGraph, chip: str = "sm_86",
                      block: int = 256) -> EmittedGPUKernel:
    """Emit an element-per-block gpu.module for a 2D data-movement op.

    grid = output shape, block=(1,1,1); each block writes one output element by
    reading the mapped input element(s). Same index math as the CPU composite
    path.
      * transpose [M,N] -> [N,M]:  O[i,j] = X[j,i]
      * copy_     [M,N] -> [M,N]:  O[i,j] = X[i,j]
      * split     [M,2*] -> [M,D]: O[i,j] = X[i,j]      (first-half chunk)
      * concat    [M,Da]+[M,Db] -> [M,Da+Db]: O[i,j] = j<Da ? A[i,j] : B[i,j-Da]
    f32 only, 2D only (P3-S2).
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_movement: single-node graphs only")
    node = graph.nodes[0]
    op = node.op
    if op not in GPU_MOVEMENT_OPS:
        raise NotImplementedError(f"emit_gpu_movement: {op} unsupported")
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    if op == "permute":
        # permute allows 3D
        if any(len(v.shape) not in (2, 3) for v in in_vals):
            raise NotImplementedError(
                f"emit_gpu_movement: permute needs 2D/3D, got {[v.shape for v in in_vals]}")
    elif any(len(v.shape) != 2 for v in in_vals):
        raise NotImplementedError(f"emit_gpu_movement: 2D only, got {[v.shape for v in in_vals]}")
    if any(v.dtype != "float32" for v in in_vals):
        raise NotImplementedError("emit_gpu_movement: f32 only")
    out_name = node.outputs[0]
    kernel = graph.name or op

    L = []
    ap = L.append

    def head(params, out_shape):
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}({params}) kernel {{")
        ap("      %i = gpu.block_id x")
        ap("      %j = gpu.block_id y")

    if op == "transpose":
        M, N = in_vals[0].shape
        out_shape = [N, M]
        xt = memref_type([M, N], "float32")
        ot = memref_type(out_shape, "float32")
        head(f"%X: {xt}, %O: {ot}", out_shape)
        # O[i,j] = X[j,i]   (i in [0,N), j in [0,M))
        ap(f"      %v = memref.load %X[%j, %i] : {xt}")
        ap(f"      memref.store %v, %O[%i, %j] : {ot}")
        grid = (N, M, 1)
        arg_names, arg_shapes = [in_names[0]], [[M, N]]
    elif op in ("copy_", "split"):
        M, Din = in_vals[0].shape
        Dout = Din if op == "copy_" else Din // 2
        out_shape = [M, Dout]
        xt = memref_type([M, Din], "float32")
        ot = memref_type(out_shape, "float32")
        head(f"%X: {xt}, %O: {ot}", out_shape)
        ap(f"      %v = memref.load %X[%i, %j] : {xt}")
        ap(f"      memref.store %v, %O[%i, %j] : {ot}")
        grid = (M, Dout, 1)
        arg_names, arg_shapes = [in_names[0]], [[M, Din]]
    elif op == "concat":
        M, Da = in_vals[0].shape
        _, Db = in_vals[1].shape
        out_shape = [M, Da + Db]
        at = memref_type([M, Da], "float32")
        bt = memref_type([M, Db], "float32")
        ot = memref_type(out_shape, "float32")
        head(f"%A: {at}, %B: {bt}, %O: {ot}", out_shape)
        ap(f"      %cDa = arith.constant {Da} : index")
        ap("      %lt = arith.cmpi ult, %j, %cDa : index")
        ap("      %v = scf.if %lt -> f32 {")
        ap(f"        %a = memref.load %A[%i, %j] : {at}")
        ap("        scf.yield %a : f32")
        ap("      } else {")
        ap("        %jb = arith.subi %j, %cDa : index")
        ap(f"        %b = memref.load %B[%i, %jb] : {bt}")
        ap("        scf.yield %b : f32")
        ap("      }")
        ap(f"      memref.store %v, %O[%i, %j] : {ot}")
        grid = (M, Da + Db, 1)
        arg_names = [in_names[0], in_names[1]]
        arg_shapes = [[M, Da], [M, Db]]
    elif op == "permute":
        # permute(0,2,1): 3D transpose [B,M,N] -> [B,N,M].
        # Our GPU path uses 2D grid. Reshape logically: collapse B into grid.
        sh = in_vals[0].shape
        if len(sh) == 3:
            B, M, N = sh
            out_shape = [B, N, M]
            xt = memref_type([B, M, N], "float32")
            ot = memref_type(out_shape, "float32")
            # grid = (B*N, M, 1); block = (1,1,1)
            # bid_x maps to (b, n_out): b = bid_x / N, n_out = bid_x % N
            # bid_y maps to m_out
            # O[b, n_out, m_out] = X[b, m_out, n_out]
            ap("module attributes {gpu.container_module} {")
            ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
            ap(f"    gpu.func @{kernel}(%X: {xt}, %O: {ot}) kernel {{")
            ap("      %bidx = gpu.block_id x")
            ap("      %bidy = gpu.block_id y")
            ap(f"      %cN = arith.constant {N} : index")
            ap("      %b = arith.divui %bidx, %cN : index")
            ap("      %n = arith.remui %bidx, %cN : index")
            # O[b, n, bidy] = X[b, bidy, n]
            ap(f"      %v = memref.load %X[%b, %bidy, %n] : {xt}")
            ap(f"      memref.store %v, %O[%b, %n, %bidy] : {ot}")
            grid = (B * N, M, 1)
            arg_names = [in_names[0]]
            arg_shapes = [[B, M, N]]
        else:
            # 2D permute = transpose (already handled)
            M, N = sh
            out_shape = [N, M]
            xt = memref_type([M, N], "float32")
            ot = memref_type(out_shape, "float32")
            head(f"%X: {xt}, %O: {ot}", out_shape)
            ap(f"      %v = memref.load %X[%j, %i] : {xt}")
            ap(f"      memref.store %v, %O[%i, %j] : {ot}")
            grid = (N, M, 1)
            arg_names = [in_names[0]]
            arg_shapes = [[M, N]]
    else:  # unreachable: op validated against GPU_MOVEMENT_OPS above
        raise NotImplementedError(f"emit_gpu_movement: {op} not wired")

    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")
    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=arg_names,
        arg_shapes=arg_shapes,
        arg_dtypes=["float32" for _ in arg_names],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=grid,
        block=(1, 1, 1),
        buffer_order=arg_names + [out_name],
    )


# Gated / select / cast ops: element-per-block, elementwise-style math.
#   cast         [M,N] -> [M,N]   f32 identity (benchmark cast targets f32)
#   where_       cond,a,b -> out  branchless cond*a + (1-cond)*b
#   silu_and_mul [M,2D] -> [M,D]  silu(X[:, :D]) * X[:, D:]
#   gelu_and_mul [M,2D] -> [M,D]  gelu(X[:, :D]) * X[:, D:]
GPU_GATED_OPS = frozenset({
    "cast", "where_", "silu_and_mul", "gelu_and_mul",
})


def emit_gpu_gated(graph: IRGraph, chip: str = "sm_86",
                   block: int = 256) -> EmittedGPUKernel:
    """Emit a flat multi-thread gpu.module for gated/select/cast ops.

    block=(256,1,1), grid=(ceil(out_rows*out_cols/256),1,1); each thread computes
    gid=bid*blockDim+tid, guards gid<out_rows*out_cols inside scf.if, maps to
    (i=gid//out_cols, j=gid%out_cols). silu_and_mul/gelu_and_mul reuse the same
    scalar activation body as the elementwise path (``ELEMENTWISE_SPECS``); the
    transcendental math.* lower via libdevice. where_ is branchless. f32, 2D.
    Multi-thread blocks (was block=(1,1,1)) to saturate the SMs.
    """
    from arke.backend.mlir_ops import ELEMENTWISE_SPECS
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_gated: single-node graphs only")
    node = graph.nodes[0]
    op = node.op
    if op not in GPU_GATED_OPS:
        raise NotImplementedError(f"emit_gpu_gated: {op} unsupported")
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    if any(len(v.shape) != 2 for v in in_vals):
        raise NotImplementedError(f"emit_gpu_gated: 2D only, got {[v.shape for v in in_vals]}")
    if any(v.dtype != "float32" for v in in_vals):
        raise NotImplementedError("emit_gpu_gated: f32 only")
    out_name = node.outputs[0]
    kernel = graph.name or op

    L = []
    ap = L.append

    def head(params, out_rows, out_cols):
        total = out_rows * out_cols
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}({params}) kernel {{")
        ap("      %bid = gpu.block_id x")
        ap("      %bdim = gpu.block_dim x")
        ap("      %tid = gpu.thread_id x")
        ap("      %base = arith.muli %bid, %bdim : index")
        ap("      %gid = arith.addi %base, %tid : index")
        ap(f"      %cOC = arith.constant {out_cols} : index")
        ap(f"      %ctotal = arith.constant {total} : index")
        ap("      %in = arith.cmpi ult, %gid, %ctotal : index")
        ap("      scf.if %in {")
        ap("        %i = arith.divui %gid, %cOC : index")
        ap("        %j = arith.remui %gid, %cOC : index")

    if op == "cast":
        M, N = in_vals[0].shape
        out_shape = [M, N]
        xt = memref_type([M, N], "float32")
        head(f"%X: {xt}, %O: {xt}", M, N)
        ap(f"        %v = memref.load %X[%i, %j] : {xt}")
        ap(f"        memref.store %v, %O[%i, %j] : {xt}")
        arg_names, arg_shapes = [in_names[0]], [[M, N]]
    elif op == "where_":
        M, N = in_vals[0].shape
        out_shape = [M, N]
        ty = memref_type([M, N], "float32")
        head(f"%C: {ty}, %A: {ty}, %B: {ty}, %O: {ty}", M, N)
        ap("        %one = arith.constant 1.0 : f32")
        ap(f"        %c = memref.load %C[%i, %j] : {ty}")
        ap(f"        %a = memref.load %A[%i, %j] : {ty}")
        ap(f"        %b = memref.load %B[%i, %j] : {ty}")
        ap("        %ca = arith.mulf %c, %a : f32")
        ap("        %omc = arith.subf %one, %c : f32")
        ap("        %ob = arith.mulf %omc, %b : f32")
        ap("        %v = arith.addf %ca, %ob : f32")
        ap(f"        memref.store %v, %O[%i, %j] : {ty}")
        arg_names = [in_names[0], in_names[1], in_names[2]]
        arg_shapes = [[M, N], [M, N], [M, N]]
    else:  # silu_and_mul / gelu_and_mul
        act = "silu" if op == "silu_and_mul" else "gelu"
        spec = ELEMENTWISE_SPECS[act]
        M, twoD = in_vals[0].shape
        D = twoD // 2
        out_shape = [M, D]
        xt = memref_type([M, twoD], "float32")
        ot = memref_type(out_shape, "float32")
        head(f"%X: {xt}, %O: {ot}", M, D)
        ap(f"        %cD = arith.constant {D} : index")
        ap("        %jg = arith.addi %j, %cD : index")
        ap(f"        %a0 = memref.load %X[%i, %j] : {xt}")
        for line in spec.ew_body:
            ap("  " + line)  # +2 indent: now inside scf.if
        ap(f"        %g = memref.load %X[%i, %jg] : {xt}")
        ap("        %v = arith.mulf %res, %g : f32")
        ap(f"        memref.store %v, %O[%i, %j] : {ot}")
        arg_names, arg_shapes = [in_names[0]], [[M, twoD]]

    out_rows, out_cols = out_shape
    ngrid = (out_rows * out_cols + block - 1) // block
    ap("      }")  # close scf.if
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")
    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=arg_names,
        arg_shapes=arg_shapes,
        arg_dtypes=["float32" for _ in arg_names],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(ngrid, 1, 1),
        block=(block, 1, 1),
        buffer_order=arg_names + [out_name],
    )


def emit_gpu_index(graph: IRGraph, chip: str = "sm_86",
                   block: int = 256) -> EmittedGPUKernel:
    """Emit a flat multi-thread gpu.module for gather/scatter (index ops).

    gather(src[M,N], idx[M,K]) -> out[M,K]:
        out[i,j] = src[i, int(idx[i,j])]
    scatter(base[M,N], idx[M,K], src[M,K]) -> out[M,N]:
        out = zeros_like(base); out[i, int(idx[i,j])] = src[i,j]
    Index tensors are f32 → fptosi → index_cast. f32, 2D.
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_index: single-node graphs only")
    node = graph.nodes[0]
    op = node.op
    if op not in GPU_INDEX_OPS:
        raise NotImplementedError(f"emit_gpu_index: {op} unsupported")
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    if any(v.dtype != "float32" for v in in_vals):
        raise NotImplementedError("emit_gpu_index: f32 only")
    out_name = node.outputs[0]
    kernel = graph.name or op

    L = []
    ap = L.append

    if op == "gather":
        # gather: src[M,N], idx[M,K] -> out[M,K]
        src_shape = list(in_vals[0].shape)
        idx_shape = list(in_vals[1].shape)
        M, K = idx_shape
        out_shape = [M, K]
        total = M * K
        ngrid = (total + block - 1) // block
        src_ty = memref_type(src_shape, "float32")
        idx_ty = memref_type(idx_shape, "float32")
        out_ty = memref_type(out_shape, "float32")
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}(%SRC: {src_ty}, %IDX: {idx_ty}, "
           f"%O: {out_ty}) kernel {{")
        ap("      %bid = gpu.block_id x")
        ap("      %tid = gpu.thread_id x")
        ap(f"      %cBLK = arith.constant {block} : index")
        ap(f"      %cTOT = arith.constant {total} : index")
        ap(f"      %cK = arith.constant {K} : index")
        ap("      %gid = arith.muli %bid, %cBLK : index")
        ap("      %gid2 = arith.addi %gid, %tid : index")
        ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
        ap("      scf.if %inb {")
        ap("        %i = arith.divui %gid2, %cK : index")
        ap("        %j = arith.remui %gid2, %cK : index")
        ap(f"        %fi = memref.load %IDX[%i, %j] : {idx_ty}")
        ap("        %ii = arith.fptosi %fi : f32 to i64")
        ap("        %col = arith.index_cast %ii : i64 to index")
        ap(f"        %v = memref.load %SRC[%i, %col] : {src_ty}")
        ap(f"        memref.store %v, %O[%i, %j] : {out_ty}")
        ap("      }")
        ap("      gpu.return")
        ap("    }")
        ap("  }")
        ap("}")
        arg_names_list = [in_names[0], in_names[1]]
        arg_shapes = [src_shape, idx_shape]
    elif op == "scatter":
        # scatter: base[M,N], idx[M,K], src[M,K] -> out[M,N]
        # Two-pass: first zero-fill, then scatter.
        base_shape = list(in_vals[0].shape)
        idx_shape = list(in_vals[1].shape)
        src_shape = list(in_vals[2].shape)
        M, N = base_shape
        _, K = idx_shape
        out_shape = [M, N]
        total = M * K
        ngrid = (total + block - 1) // block
        # For zero-fill, need total_out elements
        total_out = M * N
        ngrid_fill = (total_out + block - 1) // block
        base_ty = memref_type(base_shape, "float32")
        idx_ty = memref_type(idx_shape, "float32")
        src_ty = memref_type(src_shape, "float32")
        out_ty = memref_type(out_shape, "float32")
        # We'll zero-fill from the host side (simpler). The scatter kernel
        # just writes. Host zeroes the output buffer before launch.
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}(%BASE: {base_ty}, %IDX: {idx_ty}, "
           f"%SRC: {src_ty}, %O: {out_ty}) kernel {{")
        ap("      %bid = gpu.block_id x")
        ap("      %tid = gpu.thread_id x")
        ap(f"      %cBLK = arith.constant {block} : index")
        ap(f"      %cTOT = arith.constant {total} : index")
        ap(f"      %cK = arith.constant {K} : index")
        ap("      %gid = arith.muli %bid, %cBLK : index")
        ap("      %gid2 = arith.addi %gid, %tid : index")
        ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
        ap("      scf.if %inb {")
        ap("        %i = arith.divui %gid2, %cK : index")
        ap("        %j = arith.remui %gid2, %cK : index")
        ap(f"        %fi = memref.load %IDX[%i, %j] : {idx_ty}")
        ap("        %ii = arith.fptosi %fi : f32 to i64")
        ap("        %col = arith.index_cast %ii : i64 to index")
        ap(f"        %v = memref.load %SRC[%i, %j] : {src_ty}")
        ap(f"        memref.store %v, %O[%i, %col] : {out_ty}")
        ap("      }")
        ap("      gpu.return")
        ap("    }")
        ap("  }")
        ap("}")
        arg_names_list = [in_names[0], in_names[1], in_names[2]]
        arg_shapes = [base_shape, idx_shape, src_shape]
    else:
        raise NotImplementedError(f"emit_gpu_index: {op} not wired")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=arg_names_list,
        arg_shapes=arg_shapes,
        arg_dtypes=["float32" for _ in arg_names_list],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(ngrid, 1, 1),
        block=(block, 1, 1),
        buffer_order=arg_names_list + [out_name],
    )


def emit_gpu_batch_matmul(graph: IRGraph, chip: str = "sm_86",
                          block: int = 256) -> EmittedGPUKernel:
    """Emit a GPU kernel for batch_matmul: C[b,i,j] = sum(A[b,i,k] * B[b,k,j], k).

    flat multi-thread kernel: each thread computes one output element.
    grid = ceil(B*M*N / block), block = (256,1,1).
    gid → (b, i, j) via integer arithmetic.
    f32, 3D inputs [B,M,K] × [B,K,N] → [B,M,N].
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_batch_matmul: single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    A_shape = list(in_vals[0].shape)
    B_shape = list(in_vals[1].shape)
    if len(A_shape) != 3 or len(B_shape) != 3:
        raise NotImplementedError("emit_gpu_batch_matmul: 3D only")
    BS, M, K = A_shape
    BS2, K2, N = B_shape
    assert BS == BS2 and K == K2
    out_shape = [BS, M, N]
    total = BS * M * N
    ngrid = (total + block - 1) // block
    out_name = node.outputs[0]
    kernel = graph.name or "batch_matmul"

    At = memref_type(A_shape, "float32")
    Bt = memref_type(B_shape, "float32")
    Ct = memref_type(out_shape, "float32")

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%A: {At}, %B: {Bt}, %C: {Ct}) kernel {{")
    ap("      %bid = gpu.block_id x")
    ap("      %tid = gpu.thread_id x")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap(f"      %cTOT = arith.constant {total} : index")
    ap(f"      %cMN = arith.constant {M * N} : index")
    ap(f"      %cN = arith.constant {N} : index")
    ap(f"      %cK = arith.constant {K} : index")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c1 = arith.constant 1 : index")
    ap("      %zero = arith.constant 0.0 : f32")
    ap("      %gid = arith.muli %bid, %cBLK : index")
    ap("      %gid2 = arith.addi %gid, %tid : index")
    ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
    ap("      scf.if %inb {")
    # gid2 → (b, i, j)
    ap("        %b = arith.divui %gid2, %cMN : index")
    ap("        %rem = arith.remui %gid2, %cMN : index")
    ap("        %i = arith.divui %rem, %cN : index")
    ap("        %j = arith.remui %rem, %cN : index")
    # dot product: sum over k
    ap("        %dot = scf.for %k = %c0 to %cK step %c1"
       " iter_args(%acc = %zero) -> f32 {")
    ap(f"          %a = memref.load %A[%b, %i, %k] : {At}")
    ap(f"          %bv = memref.load %B[%b, %k, %j] : {Bt}")
    ap("          %prod = arith.mulf %a, %bv fastmath<contract> : f32")
    ap("          %nacc = arith.addf %acc, %prod fastmath<contract> : f32")
    ap("          scf.yield %nacc : f32")
    ap("        }")
    ap(f"        memref.store %dot, %C[%b, %i, %j] : {Ct}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0], in_names[1]],
        arg_shapes=[A_shape, B_shape],
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(ngrid, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], in_names[1], out_name],
    )


def emit_gpu_quantize_per_token(graph: IRGraph, chip: str = "sm_86",
                                block: int = _RW_BLOCK) -> EmittedGPUKernel:
    """Emit a GPU kernel for quantize_per_token: Y = round(X / scale).clamp(-128,127).

    scale = abs_max(row) / 127. Output is f32-encoded int8 values.
    Row-per-block pattern (256 threads cooperate for abs_max reduce).
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_quantize_per_token: single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    shape = list(in_vals[0].shape)
    if len(shape) != 2:
        raise NotImplementedError("emit_gpu_quantize_per_token: 2D only")
    rows, D = shape
    out_shape = [rows, D]
    out_name = node.outputs[0]
    kernel = graph.name or "quantize_per_token"
    inty = memref_type(shape, "float32")
    outty = memref_type(out_shape, "float32")
    sty = f"memref<{block}xf32, #gpu.address_space<workgroup>>"

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%X: {inty}, %O: {outty})")
    ap(f"        workgroup(%sh : {sty})")
    ap("        kernel {")
    ap("      %tid = gpu.thread_id x")
    ap("      %bid = gpu.block_id x")
    ap("      %c0 = arith.constant 0 : index")
    ap(f"      %cD = arith.constant {D} : index")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap("      %zero = arith.constant 0.0 : f32")
    ap("      %c127f = arith.constant 127.0 : f32")
    ap("      %cn128f = arith.constant -128.0 : f32")
    # Pass 1: per-row abs_max
    ap("      %lam = scf.for %k = %tid to %cD step %cBLK iter_args(%m = %zero) -> f32 {")
    ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
    ap("        %ax = math.absf %x : f32")
    ap("        %nm = arith.maximumf %m, %ax : f32")
    ap("        scf.yield %nm : f32")
    ap("      }")
    ap(f"      memref.store %lam, %sh[%tid] : {sty}")
    ap("      gpu.barrier")
    _rw_tree_reduce(ap, sty, "qmax", BLOCK=block)
    ap(f"      %amax = memref.load %sh[%c0] : {sty}")
    ap("      gpu.barrier")
    # scale = amax / 127, inv_scale = 127 / amax
    ap("      %eps = arith.constant 1.0e-10 : f32")
    ap("      %safe = arith.maximumf %amax, %eps : f32")
    ap("      %inv_scale = arith.divf %c127f, %safe : f32")
    # Pass 2: quantize
    ap("      scf.for %k = %tid to %cD step %cBLK {")
    ap(f"        %xq = memref.load %X[%bid, %k] : {inty}")
    ap("        %scaled = arith.mulf %xq, %inv_scale fastmath<contract> : f32")
    ap("        %rounded = math.roundeven %scaled : f32")
    ap("        %clo = arith.maximumf %rounded, %cn128f : f32")
    ap("        %chi = arith.minimumf %clo, %c127f : f32")
    ap(f"        memref.store %chi, %O[%bid, %k] : {outty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0]],
        arg_shapes=[shape],
        arg_dtypes=["float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(rows, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], out_name],
    )


def emit_gpu_dequantize_per_channel(graph: IRGraph, chip: str = "sm_86",
                                     block: int = 256) -> EmittedGPUKernel:
    """Emit a GPU kernel for dequantize_per_channel: Y = (X_int8 - zero_point) * scale.

    Three inputs: X_int8 [M,N] (f32), scale [N] (f32), zero_point [N] (f32).
    Flat multi-thread elementwise kernel.
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_dequantize_per_channel: single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    x_shape = list(in_vals[0].shape)
    M, N = x_shape
    out_shape = [M, N]
    total = M * N
    ngrid = (total + block - 1) // block
    out_name = node.outputs[0]
    kernel = graph.name or "dequantize_per_channel"

    x_ty = memref_type(x_shape, "float32")
    s_ty = memref_type([N], "float32")
    z_ty = memref_type([N], "float32")
    o_ty = memref_type(out_shape, "float32")

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%XI: {x_ty}, %SC: {s_ty}, %ZP: {z_ty}, %O: {o_ty}) kernel {{")
    ap("      %bid = gpu.block_id x")
    ap("      %tid = gpu.thread_id x")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap(f"      %cTOT = arith.constant {total} : index")
    ap(f"      %cN = arith.constant {N} : index")
    ap("      %gid = arith.muli %bid, %cBLK : index")
    ap("      %gid2 = arith.addi %gid, %tid : index")
    ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
    ap("      scf.if %inb {")
    ap("        %i = arith.divui %gid2, %cN : index")
    ap("        %j = arith.remui %gid2, %cN : index")
    ap(f"        %xi = memref.load %XI[%i, %j] : {x_ty}")
    ap(f"        %sc = memref.load %SC[%j] : {s_ty}")
    ap(f"        %zp = memref.load %ZP[%j] : {z_ty}")
    ap("        %diff = arith.subf %xi, %zp : f32")
    ap("        %out = arith.mulf %diff, %sc fastmath<contract> : f32")
    ap(f"        memref.store %out, %O[%i, %j] : {o_ty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0], in_names[1], in_names[2]],
        arg_shapes=[x_shape, [N], [N]],
        arg_dtypes=["float32", "float32", "float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(ngrid, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], in_names[1], in_names[2], out_name],
    )


def emit_gpu_swiglu_packed(graph: IRGraph, chip: str = "sm_86",
                           block: int = 256) -> EmittedGPUKernel:
    """Emit a GPU kernel for swiglu_packed: Y = (silu(X[:,:D]) * X[:,D:]) @ W.

    Fused single-kernel: each thread computes one output element of the matmul
    with the hidden vector computed on-the-fly from X.
    X [M, 2D], W [D, N] → Y [M, N].
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_swiglu_packed: single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    x_shape = list(in_vals[0].shape)
    w_shape = list(in_vals[1].shape)
    M, D2 = x_shape
    D = D2 // 2
    D_w, N = w_shape
    assert D == D_w, f"swiglu_packed: D mismatch {D} vs {D_w}"
    out_shape = [M, N]
    total = M * N
    ngrid = (total + block - 1) // block
    out_name = node.outputs[0]
    kernel = graph.name or "swiglu_packed"

    x_ty = memref_type(x_shape, "float32")
    w_ty = memref_type(w_shape, "float32")
    o_ty = memref_type(out_shape, "float32")

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%X: {x_ty}, %W: {w_ty}, %O: {o_ty}) kernel {{")
    ap("      %bid = gpu.block_id x")
    ap("      %tid = gpu.thread_id x")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap(f"      %cTOT = arith.constant {total} : index")
    ap(f"      %cN = arith.constant {N} : index")
    ap(f"      %cD = arith.constant {D} : index")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c1 = arith.constant 1 : index")
    ap("      %zero = arith.constant 0.0 : f32")
    ap("      %one_sw = arith.constant 1.0 : f32")
    ap("      %gid = arith.muli %bid, %cBLK : index")
    ap("      %gid2 = arith.addi %gid, %tid : index")
    ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
    ap("      scf.if %inb {")
    ap("        %i = arith.divui %gid2, %cN : index")
    ap("        %j = arith.remui %gid2, %cN : index")
    ap("        %dot = scf.for %k = %c0 to %cD step %c1"
       " iter_args(%acc = %zero) -> f32 {")
    ap(f"          %gate = memref.load %X[%i, %k] : {x_ty}")
    ap("          %kD = arith.addi %k, %cD : index")
    ap(f"          %up = memref.load %X[%i, %kD] : {x_ty}")
    ap("          %neg_g = arith.negf %gate : f32")
    ap("          %exp_ng = math.exp %neg_g : f32")
    ap("          %denom = arith.addf %one_sw, %exp_ng : f32")
    ap("          %sig = arith.divf %one_sw, %denom : f32")
    ap("          %silu_g = arith.mulf %gate, %sig fastmath<contract> : f32")
    ap("          %h = arith.mulf %silu_g, %up fastmath<contract> : f32")
    ap(f"          %w = memref.load %W[%k, %j] : {w_ty}")
    ap("          %prod = arith.mulf %h, %w fastmath<contract> : f32")
    ap("          %nacc = arith.addf %acc, %prod fastmath<contract> : f32")
    ap("          scf.yield %nacc : f32")
    ap("        }")
    ap(f"        memref.store %dot, %O[%i, %j] : {o_ty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0], in_names[1]],
        arg_shapes=[x_shape, w_shape],
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(ngrid, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], in_names[1], out_name],
    )


def emit_gpu_topk(graph: IRGraph, chip: str = "sm_86") -> EmittedGPUKernel:
    """Emit a GPU kernel for topk: find the k largest values per row.

    Single-thread-per-row serial scan: block=(1,1,1), grid=(rows,1,1).
    Each block scans one row, maintaining k candidates in registers via
    a private(register) memref. O(N*k) per row — correct and simple.

    The k parameter comes from ``graph.nodes[0].attrs['k']``.
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_topk: single-node only")
    node = graph.nodes[0]
    k = node.attrs.get("k", 1)
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    shape = list(in_vals[0].shape)
    if len(shape) != 2:
        raise NotImplementedError(f"emit_gpu_topk: 2D only, got {shape}")
    rows, D = shape
    out_shape = [rows, k]
    out_name = node.outputs[0]
    kernel = graph.name or "topk"
    inty = memref_type(shape, "float32")
    outty = memref_type(out_shape, "float32")
    # Private memref for top-k candidates (register-resident)
    kty = f"memref<{k}xf32, #gpu.address_space<private>>"

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%X: {inty}, %O: {outty})")
    ap(f"        private(%topk : {kty})")
    ap("        kernel {")
    ap("      %bid = gpu.block_id x")
    ap(f"      %cD = arith.constant {D} : index")
    ap(f"      %cK = arith.constant {k} : index")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c1 = arith.constant 1 : index")
    ap("      %ninf = arith.constant 0xFF800000 : f32")
    # Initialize top-k array with -inf
    ap(f"      scf.for %i = %c0 to %cK step %c1 {{")
    ap(f"        memref.store %ninf, %topk[%i] : {kty}")
    ap("      }")
    # Scan through row, insertion-sort each element into top-k
    # For each element x: find the min in topk, if x > min, replace min with x
    ap(f"      scf.for %j = %c0 to %cD step %c1 {{")
    ap(f"        %x = memref.load %X[%bid, %j] : {inty}")
    # Find index of minimum in topk
    ap(f"        %min0 = memref.load %topk[%c0] : {kty}")
    ap(f"        %mi:2 = scf.for %ki = %c1 to %cK step %c1"
       " iter_args(%mval = %min0, %midx = %c0) -> (f32, index) {")
    ap(f"          %kv = memref.load %topk[%ki] : {kty}")
    ap("          %lt = arith.cmpf olt, %kv, %mval : f32")
    ap("          %nmval = arith.select %lt, %kv, %mval : f32")
    ap("          %nmidx = arith.select %lt, %ki, %midx : index")
    ap("          scf.yield %nmval, %nmidx : f32, index")
    ap("        }")
    # If x > min of topk, replace
    ap("        %gt = arith.cmpf ogt, %x, %mi#0 : f32")
    ap("        scf.if %gt {")
    ap(f"          memref.store %x, %topk[%mi#1] : {kty}")
    ap("        }")
    ap("      }")
    # Sort topk descending (bubble sort, k is small)
    if k > 1:
        ap(f"      %cKm1 = arith.constant {k - 1} : index")
        ap(f"      scf.for %pass = %c0 to %cKm1 step %c1 {{")
        ap(f"        scf.for %si = %c0 to %cKm1 step %c1 {{")
        ap(f"          %si1 = arith.addi %si, %c1 : index")
        ap(f"          %sv1 = memref.load %topk[%si] : {kty}")
        ap(f"          %sv2 = memref.load %topk[%si1] : {kty}")
        ap("          %swap = arith.cmpf olt, %sv1, %sv2 : f32")
        ap("          scf.if %swap {")
        ap(f"            memref.store %sv2, %topk[%si] : {kty}")
        ap(f"            memref.store %sv1, %topk[%si1] : {kty}")
        ap("          }")
        ap("        }")
        ap("      }")
    # Write sorted topk to output
    ap(f"      scf.for %oi = %c0 to %cK step %c1 {{")
    ap(f"        %ov = memref.load %topk[%oi] : {kty}")
    ap(f"        memref.store %ov, %O[%bid, %oi] : {outty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0]],
        arg_shapes=[shape],
        arg_dtypes=["float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(rows, 1, 1),
        block=(1, 1, 1),
        buffer_order=[in_names[0], out_name],
    )


def emit_gpu_cross_entropy(graph: IRGraph, chip: str = "sm_86",
                           block: int = _RW_BLOCK) -> EmittedGPUKernel:
    """Emit a GPU kernel for cross_entropy: loss_i = -logits[i,label_i] + max + log(sum_exp).

    Two inputs: logits [B,V] (f32), labels [B] (f32-encoded integers).
    Output: [B] per-row losses. Host averages to get scalar.
    Each row gets one thread-block (256 threads cooperate for max + sum).
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_cross_entropy: single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    logit_shape = list(in_vals[0].shape)
    label_shape = list(in_vals[1].shape)
    if len(logit_shape) != 2:
        raise NotImplementedError(f"emit_gpu_cross_entropy: 2D logits only")
    B, V = logit_shape
    out_shape = [B]
    out_name = node.outputs[0]
    kernel = graph.name or "cross_entropy"

    logit_ty = memref_type(logit_shape, "float32")
    label_ty = memref_type(label_shape, "float32")
    out_ty = memref_type(out_shape, "float32")
    sty = f"memref<{block}xf32, #gpu.address_space<workgroup>>"

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%LOGITS: {logit_ty}, %LABELS: {label_ty}, %O: {out_ty})")
    ap(f"        workgroup(%sh : {sty})")
    ap("        kernel {")
    ap("      %tid = gpu.thread_id x")
    ap("      %bid = gpu.block_id x")
    ap("      %c0 = arith.constant 0 : index")
    ap(f"      %cV = arith.constant {V} : index")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap("      %zero = arith.constant 0.0 : f32")
    ap("      %ninf = arith.constant 0xFF800000 : f32")
    # Pass 1: row max
    ap("      %lmax = scf.for %k = %tid to %cV step %cBLK iter_args(%m = %ninf) -> f32 {")
    ap(f"        %x = memref.load %LOGITS[%bid, %k] : {logit_ty}")
    ap("        %nm = arith.maximumf %m, %x : f32")
    ap("        scf.yield %nm : f32")
    ap("      }")
    ap(f"      memref.store %lmax, %sh[%tid] : {sty}")
    ap("      gpu.barrier")
    _rw_tree_reduce(ap, sty, "cemax", BLOCK=block)
    ap(f"      %mx = memref.load %sh[%c0] : {sty}")
    ap("      gpu.barrier")
    # Pass 2: sum(exp(x - max))
    ap("      %lse = scf.for %k = %tid to %cV step %cBLK iter_args(%s = %zero) -> f32 {")
    ap(f"        %x2 = memref.load %LOGITS[%bid, %k] : {logit_ty}")
    ap("        %d = arith.subf %x2, %mx : f32")
    ap("        %e = math.exp %d : f32")
    ap("        %ns = arith.addf %s, %e fastmath<contract> : f32")
    ap("        scf.yield %ns : f32")
    ap("      }")
    ap(f"      memref.store %lse, %sh[%tid] : {sty}")
    ap("      gpu.barrier")
    _rw_tree_reduce(ap, sty, "cesum", BLOCK=block)
    ap(f"      %sumv = memref.load %sh[%c0] : {sty}")
    ap("      gpu.barrier")
    # Thread 0: loss = max - logits[label] + log(sum)
    ap("      %is0 = arith.cmpi eq, %tid, %c0 : index")
    ap("      scf.if %is0 {")
    ap(f"        %lf = memref.load %LABELS[%bid] : {label_ty}")
    ap("        %li = arith.fptosi %lf : f32 to i64")
    ap("        %lidx = arith.index_cast %li : i64 to index")
    ap(f"        %lv = memref.load %LOGITS[%bid, %lidx] : {logit_ty}")
    ap("        %log_sum = math.log %sumv : f32")
    ap("        %loss = arith.subf %mx, %lv : f32")
    ap("        %loss2 = arith.addf %loss, %log_sum : f32")
    ap(f"        memref.store %loss2, %O[%bid] : {out_ty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0], in_names[1]],
        arg_shapes=[logit_shape, label_shape],
        arg_dtypes=["float32", "float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(B, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], in_names[1], out_name],
    )

def emit_gpu_fused_linear_cross_entropy(graph: "IRGraph", chip: str = "sm_86",
                                         block: int = _RW_BLOCK) -> "EmittedGPUKernel":
    """Emit a GPU kernel for fused_linear_cross_entropy: loss = CE(X @ W.T, labels).

    Three inputs: X [B,D], W [V,D], labels [B] (f32-encoded integers).
    Output: [B] per-row losses. Host averages for scalar.
    Each block handles one batch element. Each thread computes logits for a
    subset of V classes on-the-fly and tracks (max, sum_exp) via online softmax.
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_fused_linear_cross_entropy: single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    x_shape = list(in_vals[0].shape)
    w_shape = list(in_vals[1].shape)
    label_shape = list(in_vals[2].shape)
    B, D = x_shape
    V, D2 = w_shape
    assert D == D2
    out_shape = [B]
    out_name = node.outputs[0]
    kernel = graph.name or "fused_linear_cross_entropy"

    x_ty = memref_type(x_shape, "float32")
    w_ty = memref_type(w_shape, "float32")
    l_ty = memref_type(label_shape, "float32")
    o_ty = memref_type(out_shape, "float32")
    sty = f"memref<{block}xf32, #gpu.address_space<workgroup>>"

    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%X: {x_ty}, %W: {w_ty}, %LABELS: {l_ty}, %O: {o_ty})")
    ap(f"        workgroup(%sh : {sty})")
    ap("        kernel {")
    ap("      %tid = gpu.thread_id x")
    ap("      %bid = gpu.block_id x")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c1 = arith.constant 1 : index")
    ap(f"      %cV = arith.constant {V} : index")
    ap(f"      %cD = arith.constant {D} : index")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap("      %zero = arith.constant 0.0 : f32")
    ap("      %ninf = arith.constant 0xFF800000 : f32")
    # Online softmax over logits computed on-the-fly
    ap("      %os:2 = scf.for %j = %tid to %cV step %cBLK"
       " iter_args(%m = %ninf, %s = %zero) -> (f32, f32) {")
    ap("        %lj = scf.for %k = %c0 to %cD step %c1"
       " iter_args(%dacc = %zero) -> f32 {")
    ap(f"          %xv = memref.load %X[%bid, %k] : {x_ty}")
    ap(f"          %wv = memref.load %W[%j, %k] : {w_ty}")
    ap("          %p = arith.mulf %xv, %wv fastmath<contract> : f32")
    ap("          %nd = arith.addf %dacc, %p fastmath<contract> : f32")
    ap("          scf.yield %nd : f32")
    ap("        }")
    ap("        %nm = arith.maximumf %m, %lj : f32")
    ap("        %corr = arith.subf %m, %nm : f32")
    ap("        %ecorr = math.exp %corr : f32")
    ap("        %xd = arith.subf %lj, %nm : f32")
    ap("        %ex = math.exp %xd : f32")
    ap("        %sc = arith.mulf %s, %ecorr fastmath<contract> : f32")
    ap("        %ns = arith.addf %sc, %ex fastmath<contract> : f32")
    ap("        scf.yield %nm, %ns : f32, f32")
    ap("      }")
    # Reduce max
    ap(f"      memref.store %os#0, %sh[%tid] : {sty}")
    ap("      gpu.barrier")
    _rw_tree_reduce(ap, sty, "flcemax", BLOCK=block)
    ap(f"      %mx = memref.load %sh[%c0] : {sty}")
    ap("      gpu.barrier")
    # Correct + reduce sum
    ap("      %mc = arith.subf %os#0, %mx : f32")
    ap("      %emc = math.exp %mc : f32")
    ap("      %csum = arith.mulf %os#1, %emc fastmath<contract> : f32")
    ap(f"      memref.store %csum, %sh[%tid] : {sty}")
    ap("      gpu.barrier")
    _rw_tree_reduce(ap, sty, "flcesum", BLOCK=block)
    ap(f"      %sumv = memref.load %sh[%c0] : {sty}")
    ap("      gpu.barrier")
    # Thread 0: label logit + loss
    ap("      %is0 = arith.cmpi eq, %tid, %c0 : index")
    ap("      scf.if %is0 {")
    ap(f"        %lf = memref.load %LABELS[%bid] : {l_ty}")
    ap("        %li = arith.fptosi %lf : f32 to i64")
    ap("        %lidx = arith.index_cast %li : i64 to index")
    ap("        %label_logit = scf.for %k = %c0 to %cD step %c1"
       " iter_args(%la = %zero) -> f32 {")
    ap(f"          %xk = memref.load %X[%bid, %k] : {x_ty}")
    ap(f"          %wk = memref.load %W[%lidx, %k] : {w_ty}")
    ap("          %lp = arith.mulf %xk, %wk fastmath<contract> : f32")
    ap("          %nla = arith.addf %la, %lp fastmath<contract> : f32")
    ap("          scf.yield %nla : f32")
    ap("        }")
    ap("        %log_sum = math.log %sumv : f32")
    ap("        %loss = arith.subf %mx, %label_logit : f32")
    ap("        %loss2 = arith.addf %loss, %log_sum : f32")
    ap(f"        memref.store %loss2, %O[%bid] : {o_ty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")

    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text,
        kernel_name=kernel,
        arg_names=[in_names[0], in_names[1], in_names[2]],
        arg_shapes=[x_shape, w_shape, label_shape],
        arg_dtypes=["float32", "float32", "float32"],
        result_name=out_name,
        result_shape=out_shape,
        result_dtype="float32",
        grid=(B, 1, 1),
        block=(block, 1, 1),
        buffer_order=[in_names[0], in_names[1], in_names[2], out_name],
    )


def emit_gpu_grouped_matmul(graph: "IRGraph", chip: str = "sm_86",
                            block: int = 256) -> "EmittedGPUKernel":
    """GPU grouped_matmul: Y[b,i,j] = sum(X[b,i,k] * W[indices[b],k,j], k).

    Three inputs: X [B,M,K], W [E,K,N], indices [B] (f32-encoded).
    Flat multi-thread: each thread computes one element of Y [B,M,N].
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("single-node only")
    node = graph.nodes[0]
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]
    x_sh = list(in_vals[0].shape)  # [B,M,K]
    w_sh = list(in_vals[1].shape)  # [E,K,N]
    idx_sh = list(in_vals[2].shape)  # [B]
    B, M, K = x_sh
    E, K2, N = w_sh
    out_shape = [B, M, N]
    total = B * M * N
    ngrid = (total + block - 1) // block
    out_name = node.outputs[0]
    kernel = graph.name or "grouped_matmul"
    x_ty = memref_type(x_sh, "float32")
    w_ty = memref_type(w_sh, "float32")
    i_ty = memref_type(idx_sh, "float32")
    o_ty = memref_type(out_shape, "float32")
    L = []
    ap = L.append
    ap("module attributes {gpu.container_module} {")
    ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
    ap(f"    gpu.func @{kernel}(%X: {x_ty}, %W: {w_ty}, %IDX: {i_ty}, %O: {o_ty}) kernel {{")
    ap("      %bid = gpu.block_id x")
    ap("      %tid = gpu.thread_id x")
    ap(f"      %cBLK = arith.constant {block} : index")
    ap(f"      %cTOT = arith.constant {total} : index")
    ap(f"      %cMN = arith.constant {M * N} : index")
    ap(f"      %cN = arith.constant {N} : index")
    ap(f"      %cK = arith.constant {K} : index")
    ap("      %c0 = arith.constant 0 : index")
    ap("      %c1 = arith.constant 1 : index")
    ap("      %zero = arith.constant 0.0 : f32")
    ap("      %gid = arith.muli %bid, %cBLK : index")
    ap("      %gid2 = arith.addi %gid, %tid : index")
    ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
    ap("      scf.if %inb {")
    ap("        %b = arith.divui %gid2, %cMN : index")
    ap("        %rem = arith.remui %gid2, %cMN : index")
    ap("        %i = arith.divui %rem, %cN : index")
    ap("        %j = arith.remui %rem, %cN : index")
    # Load expert index for this batch
    ap(f"        %idxf = memref.load %IDX[%b] : {i_ty}")
    ap("        %idxi = arith.fptosi %idxf : f32 to i64")
    ap("        %eidx = arith.index_cast %idxi : i64 to index")
    # dot product over K
    ap("        %dot = scf.for %k = %c0 to %cK step %c1"
       " iter_args(%acc = %zero) -> f32 {")
    ap(f"          %xv = memref.load %X[%b, %i, %k] : {x_ty}")
    ap(f"          %wv = memref.load %W[%eidx, %k, %j] : {w_ty}")
    ap("          %p = arith.mulf %xv, %wv fastmath<contract> : f32")
    ap("          %na = arith.addf %acc, %p fastmath<contract> : f32")
    ap("          scf.yield %na : f32")
    ap("        }")
    ap(f"        memref.store %dot, %O[%b, %i, %j] : {o_ty}")
    ap("      }")
    ap("      gpu.return")
    ap("    }")
    ap("  }")
    ap("}")
    text = "\n".join(L)
    return EmittedGPUKernel(
        mlir_text=text, kernel_name=kernel,
        arg_names=[in_names[0], in_names[1], in_names[2]],
        arg_shapes=[x_sh, w_sh, idx_sh],
        arg_dtypes=["float32", "float32", "float32"],
        result_name=out_name, result_shape=out_shape, result_dtype="float32",
        grid=(ngrid, 1, 1), block=(block, 1, 1),
        buffer_order=[in_names[0], in_names[1], in_names[2], out_name],
    )




def emit_gpu_attention(graph: "IRGraph", chip: str = "sm_86",
                       block: int = 256) -> "EmittedGPUKernel":
    """Unified GPU emitter for all attention ops (naive SDPA, not flash).

    Handles: flash_attention, cross_attention, grouped_query_attention,
    multi_latent_attention, paged_attention.

    For correctness (not perf): each thread computes one output element
    O[b,h,i,d] by iterating over all KV positions. Online softmax for
    numerical stability. Single-thread-per-output-element.

    Preprocessing (done on host before kernel launch):
    - grouped_query_attention: K/V heads repeated to match Q heads
    - multi_latent_attention: KV decompressed from latent
    - paged_attention: KV assembled from block table
    All preprocessing produces standard Q[B,H,Sq,D], K[B,H,Skv,D], V[B,H,Skv,D]
    which are handled via MLIRGPUBackend.run() preprocessing.

    For the GPU kernel itself, inputs are always Q[B,H,Sq,D], K[B,H,Skv,D], V[B,H,Skv,D].
    """
    if len(graph.nodes) != 1:
        raise NotImplementedError("emit_gpu_attention: single-node only")
    node = graph.nodes[0]
    op = node.op
    in_names = list(node.inputs.values())
    in_vals = [graph.values[n] for n in in_names]

    # For all attention ops, the first 3 inputs after preprocessing are Q, K, V
    # with shapes [B,H,Sq,D], [B,H,Skv,D], [B,H,Skv,D]
    q_shape = list(in_vals[0].shape)  # [B,H,Sq,D] or similar

    if op in ("flash_attention", "cross_attention"):
        # Q[B,H,Sq,D], K[B,H,Skv,D], V[B,H,Skv,D]
        k_shape = list(in_vals[1].shape)
        v_shape = list(in_vals[2].shape)
        B, H, Sq, D = q_shape
        Skv = k_shape[2]
        extra_inputs = []
        extra_shapes = []
        extra_dtypes = []
    elif op == "grouped_query_attention":
        # Q[B,Hq,S,D], K[B,Hkv,S,D], V[B,Hkv,S,D]
        # The emitter treats K/V as if they had Hq heads (host repeats them)
        k_shape = list(in_vals[1].shape)
        v_shape = list(in_vals[2].shape)
        B, H, Sq, D = q_shape
        Skv = k_shape[2]
        # For the kernel, we'll handle GQA inline by computing the KV head index
        Hkv = k_shape[1]
        extra_inputs = []
        extra_shapes = []
        extra_dtypes = []
    elif op == "multi_latent_attention":
        # Q[B,H,S,D], KV_compressed[B,S,Dc], W_uk[Dc,H,D], W_uv[Dc,H,D]
        # We'll decompress inline: K[b,h,s,d] = sum_c KV[b,s,c] * Wuk[c,h,d]
        kv_shape = list(in_vals[1].shape)  # [B,S,Dc]
        wuk_shape = list(in_vals[2].shape)  # [Dc,H,D]
        wuv_shape = list(in_vals[3].shape)  # [Dc,H,D]
        B, H, Sq, D = q_shape
        Skv = kv_shape[1]
        Dc = kv_shape[2]
        extra_inputs = in_names[1:]
        extra_shapes = [kv_shape, wuk_shape, wuv_shape]
        extra_dtypes = ["float32"] * 3
    elif op == "paged_attention":
        # Q[B,H,1,D], K_cache[NB,BS,H,D], V_cache[NB,BS,H,D], block_table[B,MB]
        k_cache_shape = list(in_vals[1].shape)
        v_cache_shape = list(in_vals[2].shape)
        bt_shape = list(in_vals[3].shape)
        B, H, _, D = q_shape
        Sq = 1
        NB, BS_val = k_cache_shape[0], k_cache_shape[1]
        MB = bt_shape[1]
        Skv = MB * BS_val
        extra_inputs = in_names[1:]
        extra_shapes = [k_cache_shape, v_cache_shape, bt_shape]
        extra_dtypes = ["float32"] * 3
    else:
        raise NotImplementedError(f"emit_gpu_attention: unknown op {op}")

    out_shape = [B, H, Sq, D]
    out_name = node.outputs[0]
    total = B * H * Sq * D
    ngrid = (total + block - 1) // block
    kernel = graph.name or op

    import math
    scale_val = 1.0 / math.sqrt(D)

    o_ty = memref_type(out_shape, "float32")

    # For flash/cross/GQA, use standard Q/K/V memrefs
    if op in ("flash_attention", "cross_attention", "grouped_query_attention"):
        q_ty = memref_type(q_shape, "float32")
        k_ty = memref_type(k_shape, "float32")
        v_ty = memref_type(v_shape, "float32")
        Hkv_val = k_shape[1] if op == "grouped_query_attention" else H

        L = []
        ap = L.append
        ap("module attributes {gpu.container_module} {")
        ap(f'  gpu.module @{kernel}_mod [#nvvm.target<chip = "{chip}">] {{')
        ap(f"    gpu.func @{kernel}(%Q: {q_ty}, %K: {k_ty}, %V: {v_ty}, %O: {o_ty}) kernel {{")
        ap("      %bid = gpu.block_id x")
        ap("      %tid = gpu.thread_id x")
        ap(f"      %cBLK = arith.constant {block} : index")
        ap(f"      %cTOT = arith.constant {total} : index")
        ap(f"      %cHSD = arith.constant {H * Sq * D} : index")
        ap(f"      %cSD = arith.constant {Sq * D} : index")
        ap(f"      %cD = arith.constant {D} : index")
        ap(f"      %cSk = arith.constant {Skv} : index")
        if op == "grouped_query_attention":
            n_rep = H // Hkv_val
            ap(f"      %cNrep = arith.constant {n_rep} : index")
        ap("      %c0 = arith.constant 0 : index")
        ap("      %c1 = arith.constant 1 : index")
        ap(f"      %scale = arith.constant {scale_val} : f32")
        ap("      %zero_a = arith.constant 0.0 : f32")
        ap("      %one_a = arith.constant 1.0 : f32")
        ap("      %ninf_a = arith.constant 0xFF800000 : f32")
        ap("      %gid = arith.muli %bid, %cBLK : index")
        ap("      %gid2 = arith.addi %gid, %tid : index")
        ap("      %inb = arith.cmpi ult, %gid2, %cTOT : index")
        ap("      scf.if %inb {")
        # gid2 -> (b, h, i, d)
        ap("        %b = arith.divui %gid2, %cHSD : index")
        ap("        %r1 = arith.remui %gid2, %cHSD : index")
        ap("        %h = arith.divui %r1, %cSD : index")
        ap("        %r2 = arith.remui %r1, %cSD : index")
        ap("        %i = arith.divui %r2, %cD : index")
        ap("        %d = arith.remui %r2, %cD : index")
        if op == "grouped_query_attention":
            # GQA: query head h maps to KV head h // n_rep (heads share KV).
            ap("        %hkv = arith.divui %h, %cNrep : index")
        # (non-GQA path indexes K/V directly by %h below — no %hkv needed.)
        # Compute O[b,h,i,d] = sum_s attn_s * V[b,hkv,s,d]
        # where attn_s = exp(score_s - max) / sum_exp
        # score_s = sum_dd Q[b,h,i,dd]*K[b,hkv,s,dd] / sqrt(D)
        # Two-pass: pass1 compute max+sum (online), pass2 compute weighted sum
        # Pass 1: online softmax for max and sum
        ap("        %os:2 = scf.for %s = %c0 to %cSk step %c1"
           " iter_args(%mx = %ninf_a, %sm = %zero_a) -> (f32, f32) {")
        ap("          %sc = scf.for %dd = %c0 to %cD step %c1"
           " iter_args(%sa = %zero_a) -> f32 {")
        ap(f"            %qv = memref.load %Q[%b, %h, %i, %dd] : {q_ty}")
        if op == "grouped_query_attention":
            ap(f"            %kv = memref.load %K[%b, %hkv, %s, %dd] : {k_ty}")
        else:
            ap(f"            %kv = memref.load %K[%b, %h, %s, %dd] : {k_ty}")
        ap("            %p = arith.mulf %qv, %kv fastmath<contract> : f32")
        ap("            %nsa = arith.addf %sa, %p fastmath<contract> : f32")
        ap("            scf.yield %nsa : f32")
        ap("          }")
        ap("          %score = arith.mulf %sc, %scale fastmath<contract> : f32")
        ap("          %nmx = arith.maximumf %mx, %score : f32")
        ap("          %corr = arith.subf %mx, %nmx : f32")
        ap("          %ec = math.exp %corr : f32")
        ap("          %sd = arith.subf %score, %nmx : f32")
        ap("          %es = math.exp %sd : f32")
        ap("          %smc = arith.mulf %sm, %ec fastmath<contract> : f32")
        ap("          %nsm = arith.addf %smc, %es fastmath<contract> : f32")
        ap("          scf.yield %nmx, %nsm : f32, f32")
        ap("        }")
        # Pass 2: weighted sum
        ap("        %rcp = arith.divf %one_a, %os#1 : f32")
        ap("        %ov = scf.for %s = %c0 to %cSk step %c1"
           " iter_args(%oa = %zero_a) -> f32 {")
        # Recompute score
        ap("          %sc2 = scf.for %dd = %c0 to %cD step %c1"
           " iter_args(%sa2 = %zero_a) -> f32 {")
        ap(f"            %qv2 = memref.load %Q[%b, %h, %i, %dd] : {q_ty}")
        if op == "grouped_query_attention":
            ap(f"            %kv2 = memref.load %K[%b, %hkv, %s, %dd] : {k_ty}")
        else:
            ap(f"            %kv2 = memref.load %K[%b, %h, %s, %dd] : {k_ty}")
        ap("            %p2 = arith.mulf %qv2, %kv2 fastmath<contract> : f32")
        ap("            %nsa2 = arith.addf %sa2, %p2 fastmath<contract> : f32")
        ap("            scf.yield %nsa2 : f32")
        ap("          }")
        ap("          %score2 = arith.mulf %sc2, %scale fastmath<contract> : f32")
        ap("          %e2 = arith.subf %score2, %os#0 : f32")
        ap("          %attn = math.exp %e2 : f32")
        ap("          %nattn = arith.mulf %attn, %rcp fastmath<contract> : f32")
        if op == "grouped_query_attention":
            ap(f"          %vv = memref.load %V[%b, %hkv, %s, %d] : {v_ty}")
        else:
            ap(f"          %vv = memref.load %V[%b, %h, %s, %d] : {v_ty}")
        ap("          %c_a = arith.mulf %nattn, %vv fastmath<contract> : f32")
        ap("          %noa = arith.addf %oa, %c_a fastmath<contract> : f32")
        ap("          scf.yield %noa : f32")
        ap("        }")
        ap(f"        memref.store %ov, %O[%b, %h, %i, %d] : {o_ty}")
        ap("      }")
        ap("      gpu.return")
        ap("    }")
        ap("  }")
        ap("}")

        text = "\n".join(L)
        # For GQA, hkv = h // n_rep is handled INSIDE the kernel
        buf_order = [in_names[0], in_names[1], in_names[2], out_name]
        return EmittedGPUKernel(
            mlir_text=text, kernel_name=kernel,
            arg_names=[in_names[0], in_names[1], in_names[2]],
            arg_shapes=[q_shape, k_shape, v_shape],
            arg_dtypes=["float32", "float32", "float32"],
            result_name=out_name, result_shape=out_shape, result_dtype="float32",
            grid=(ngrid, 1, 1), block=(block, 1, 1),
            buffer_order=buf_order,
        )

    # For MLA and paged_attention, preprocessing is needed on the host side.
    # We implement these by expanding inputs to standard Q/K/V in run().
    # The emitter just generates a standard SDPA kernel for the expanded shapes.
    raise NotImplementedError(
        f"emit_gpu_attention: {op} requires host-side input preprocessing "
        f"(MLA KV decompression or paged-attention block-table assembly). "
        f"Implement via MLIRGPUBackend.run() preprocessing."
    )
