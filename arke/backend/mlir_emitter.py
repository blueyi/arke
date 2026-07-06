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
        "          %m = arith.mulf %sa, %sb : f32",
        "          %ns = arith.addf %si, %m : f32",
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
    # cooperative load A[BM x BK] into sA: linear indices tid, tid+NT, ...
    ap(f"        scf.for %li = %tid to {cAE} step {cNT} {{")
    ap(f"          %ar = arith.divui %li, {cBK} : index")     # row in tile
    ap(f"          %ac = arith.remui %li, {cBK} : index")     # col in tile
    ap(f"          %gar = arith.addi %browbase, %ar : index")
    ap(f"          %gac = arith.addi %kk, %ac : index")
    ap(f"          %av = memref.load %A[%gar, %gac] : {at}")
    ap(f"          memref.store %av, %sA[%ar, %ac] : {saty}")
    ap("        }")
    # cooperative load B[BK x BN] into sB
    ap(f"        scf.for %li = %tid to {cBE} step {cNT} {{")
    ap(f"          %br = arith.divui %li, {cBN} : index")     # row in tile (k)
    ap(f"          %bc = arith.remui %li, {cBN} : index")     # col in tile (n)
    ap(f"          %gbr = arith.addi %kk, %br : index")
    ap(f"          %gbc = arith.addi %bcolbase, %bc : index")
    ap(f"          %bv = memref.load %B[%gbr, %gbc] : {bt}")
    ap(f"          memref.store %bv, %sB[%br, %bc] : {sbty}")
    ap("        }")
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

    Threads a flat ``M*N`` element space: block=(``block``,1,1),
    grid=(ceil(M*N/block),1,1). Each thread computes ``gid = bid*blockDim + tid``,
    guards ``gid < M*N``, maps to ``(i=gid//N, j=gid%N)``, and runs the SAME
    scalar body from ``ELEMENTWISE_SPECS`` as the CPU path (identical numerics;
    transcendentals via libdevice). This replaces the earlier one-thread-per-
    block correctness kernel (block=(1,1,1)), which was ~0.02-0.09x torch from
    massive under-occupancy — flat blocks of 256 threads saturate the SMs.
    f32, 2D only.
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


def emit_gpu_rowwise(graph: IRGraph, chip: str = "sm_86",
                     block: int = _RW_BLOCK) -> EmittedGPUKernel:
    """Emit a parallel-reduce row-per-block gpu.module for row-wise ops.

    block=(256,1,1), grid=(rows,1,1). 256 threads cooperate per row via
    shared-memory tree-reduce. Transcendentals via libdevice. f32, 2D.
    """
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
        _rw_tree_reduce(ap, sty, "sum")
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
        _rw_tree_reduce(ap, sty, "max")
        ap("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        ap("      scf.if %is0 {")
        ap(f"        %r = memref.load %sh[%c0] : {sty}")
        ap(f"        memref.store %r, %O[%bid] : {outty}")
        ap("      }")
    elif op == "softmax":
        ap("      %ninf = arith.constant 0xFF800000 : f32")
        ap("      %lmax = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %ninf) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %ns = arith.maximumf %s, %x : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %lmax, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "smax")
        ap(f"      %mx = memref.load %sh[%c0] : {sty}")
        ap("      gpu.barrier")
        ap("      %lsum = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %zero) -> f32 {")
        ap(f"        %x2 = memref.load %X[%bid, %k] : {inty}")
        ap("        %d = arith.subf %x2, %mx : f32")
        ap("        %e = math.exp %d : f32")
        ap("        %ns2 = arith.addf %s, %e : f32")
        ap("        scf.yield %ns2 : f32")
        ap("      }")
        ap(f"      memref.store %lsum, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "ssum")
        ap(f"      %den = memref.load %sh[%c0] : {sty}")
        ap("      gpu.barrier")
        ap("      scf.for %k = %tid to %cD step %cBLK {")
        ap(f"        %x3 = memref.load %X[%bid, %k] : {inty}")
        ap("        %d3 = arith.subf %x3, %mx : f32")
        ap("        %e3 = math.exp %d3 : f32")
        ap("        %o3 = arith.divf %e3, %den : f32")
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
        _rw_tree_reduce(ap, sty, "lnm")
        ap(f"      %sumv = memref.load %sh[%c0] : {sty}")
        ap("      %mean = arith.divf %sumv, %Df : f32")
        ap("      gpu.barrier")
        ap("      %lvar = scf.for %k = %tid to %cD step %cBLK iter_args(%s = %zero) -> f32 {")
        ap(f"        %xv = memref.load %X[%bid, %k] : {inty}")
        ap("        %dv = arith.subf %xv, %mean : f32")
        ap("        %sq = arith.mulf %dv, %dv : f32")
        ap("        %nsv = arith.addf %s, %sq : f32")
        ap("        scf.yield %nsv : f32")
        ap("      }")
        ap(f"      memref.store %lvar, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "lnv")
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
        ap("        %sq = arith.mulf %x, %x : f32")
        ap("        %ns = arith.addf %s, %sq : f32")
        ap("        scf.yield %ns : f32")
        ap("      }")
        ap(f"      memref.store %lsq, %sh[%tid] : {sty}")
        ap("      gpu.barrier")
        _rw_tree_reduce(ap, sty, "rms")
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
        ap("      %is0c = arith.cmpi eq, %tid, %c0 : index")
        ap("      scf.if %is0c {")
        ap("      %cs = scf.for %k = %c0 to %cD step %c1 iter_args(%run = %zero) -> f32 {")
        ap(f"        %x = memref.load %X[%bid, %k] : {inty}")
        ap("        %nr = arith.addf %run, %x : f32")
        ap(f"        memref.store %nr, %O[%bid, %k] : {outty}")
        ap("        scf.yield %nr : f32")
        ap("      }")
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
        _rw_tree_reduce(ap, sty, "amax")
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
})


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
    if any(len(v.shape) != 2 for v in in_vals):
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
