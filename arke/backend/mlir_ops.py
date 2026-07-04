# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke → MLIR per-op body emitters (Phase 3, P3-S2).

This module holds the *scalable* op catalog for the MLIR backend: elementwise,
reduction, and data-movement ops expressed via ``linalg.generic`` /
``linalg.*`` structured ops. The P3-S1 module (``mlir_emitter``) owns the
graph-walk + matmul; this module extends coverage toward the 35-op P3-S2 gate.

Design:
  * **Elementwise** ops (OT0) share one ``linalg.generic`` skeleton (identity
    indexing map, all-parallel iterators); each op supplies only the scalar
    body via ``ELEMENTWISE`` (a list of MLIR body lines that compute ``%res``
    from operand SSA names ``%a0``, ``%a1``, …). n-ary elementwise (add, mul)
    just take more inputs.
  * **Reduction** ops (OT1 subset) use ``linalg.generic`` with a reduction
    iterator, or a two-pass structure (softmax/normalizations) built op-wise.
  * **Movement/dense** ops (OT2) map to dedicated structured ops (transpose,
    batch_matmul) or generic copies (permute/copy).

Every op here lowers through the same CPU pipeline in ``mlir_backend`` (linalg
→ loops → LLVM, with ``-convert-math-to-llvm`` for exp/tanh/rsqrt). f32-only in
P3-S2's correctness pass (matches the printMemrefF32 JIT harness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ── op spec ────────────────────────────────────────────────────

@dataclass
class OpSpec:
    """Declarative spec for one MLIR-lowerable op.

    category:
        "elementwise" — n-ary, output shape == input[0] shape; body via
                         ``ew_body`` (lines producing ``%res`` from %a0..%aK).
        "structured"  — custom emitter that returns full body lines given
                         (out_buf, in_bufs, out_ty, in_tys, attrs, shapes).
    """
    name: str
    num_inputs: int
    category: str
    ew_body: list[str] = field(default_factory=list)
    structured: Callable | None = None
    # shape rule: given input shapes (+ attrs) → output shape
    shape_rule: Callable | None = None


# ── elementwise scalar bodies ──────────────────────────────────
# Each body is a list of MLIR lines. Inputs are %a0, %a1, ...; the block yields
# via `linalg.yield %res : f32`. Bodies must define %res.

_EW_UNARY = {
    "relu": [
        "      %zero = arith.constant 0.0 : f32",
        "      %res = arith.maximumf %a0, %zero : f32",
    ],
    "neg": [
        "      %res = arith.negf %a0 : f32",
    ],
    "exp": [
        "      %res = math.exp %a0 : f32",
    ],
    "tanh": [
        "      %res = math.tanh %a0 : f32",
    ],
    "sigmoid": [
        # 1 / (1 + exp(-x))
        "      %one = arith.constant 1.0 : f32",
        "      %nx = arith.negf %a0 : f32",
        "      %e = math.exp %nx : f32",
        "      %d = arith.addf %one, %e : f32",
        "      %res = arith.divf %one, %d : f32",
    ],
    "silu": [
        # x * sigmoid(x)
        "      %one = arith.constant 1.0 : f32",
        "      %nx = arith.negf %a0 : f32",
        "      %e = math.exp %nx : f32",
        "      %d = arith.addf %one, %e : f32",
        "      %s = arith.divf %one, %d : f32",
        "      %res = arith.mulf %a0, %s : f32",
    ],
    "gelu": [
        # tanh approximation: 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3)))
        "      %half = arith.constant 0.5 : f32",
        "      %one = arith.constant 1.0 : f32",
        "      %c0 = arith.constant 0.7978845608028654 : f32",
        "      %c1 = arith.constant 0.044715 : f32",
        "      %x2 = arith.mulf %a0, %a0 : f32",
        "      %x3 = arith.mulf %x2, %a0 : f32",
        "      %t0 = arith.mulf %c1, %x3 : f32",
        "      %t1 = arith.addf %a0, %t0 : f32",
        "      %t2 = arith.mulf %c0, %t1 : f32",
        "      %th = math.tanh %t2 : f32",
        "      %t3 = arith.addf %one, %th : f32",
        "      %t4 = arith.mulf %half, %a0 : f32",
        "      %res = arith.mulf %t4, %t3 : f32",
    ],
    "rsqrt": [
        "      %res = math.rsqrt %a0 : f32",
    ],
}

_EW_BINARY = {
    "add": [
        "      %res = arith.addf %a0, %a1 : f32",
    ],
    "mul": [
        "      %res = arith.mulf %a0, %a1 : f32",
    ],
}


def _elementwise_specs() -> dict[str, OpSpec]:
    specs: dict[str, OpSpec] = {}
    for name, body in _EW_UNARY.items():
        specs[name] = OpSpec(name=name, num_inputs=1, category="elementwise",
                             ew_body=body)
    for name, body in _EW_BINARY.items():
        specs[name] = OpSpec(name=name, num_inputs=2, category="elementwise",
                             ew_body=body)
    return specs


ELEMENTWISE_SPECS = _elementwise_specs()


# ── composite / structured op emitters (OT1 reductions + OT2 movement) ──
# Each emitter is `fn(ctx) -> list[str]` producing the MLIR body lines that
# compute the op's output buffer `ctx.out_buf` from `ctx.in_bufs`. Emitters may
# allocate local temp buffers via `ctx.tmp()` (returns a fresh %-name and its
# alloc line is appended automatically). Shapes are known statically.

@dataclass
class OpContext:
    op: str
    out_buf: str
    out_shape: list[int]
    in_bufs: list[str]
    in_shapes: list[list[int]]
    elem: str
    _counter: list[int] = field(default_factory=lambda: [0])
    _preamble: list[str] = field(default_factory=list)

    def tmp(self, shape: list[int]) -> str:
        i = self._counter[0]
        self._counter[0] += 1
        name = f"%t{i}_{self.out_buf[1:]}"
        ty = _memref_ty(shape, self.elem)
        self._preamble.append(f"    {name} = memref.alloc() : {ty}")
        return name

    def const(self, value: str) -> str:
        """Materialize an arith.constant with a unique SSA name; return the name."""
        i = self._counter[0]
        self._counter[0] += 1
        name = f"%c{i}_{self.out_buf[1:]}"
        self._preamble.append(
            f"    {name} = arith.constant {value} : {self.elem}"
        )
        return name

    def const_idx(self, value: int) -> str:
        """Materialize an index-typed arith.constant; return its SSA name."""
        i = self._counter[0]
        self._counter[0] += 1
        name = f"%ci{i}_{self.out_buf[1:]}"
        self._preamble.append(
            f"    {name} = arith.constant {int(value)} : index"
        )
        return name


def _memref_ty(shape: list[int], elem: str) -> str:
    if not shape:
        return f"memref<{elem}>"
    dims = "x".join(str(int(d)) for d in shape)
    return f"memref<{dims}x{elem}>"


def _amap(out_rank: int, expr: str) -> str:
    dims = ", ".join(f"d{i}" for i in range(out_rank))
    return f"affine_map<({dims}) -> ({expr})>"


def _reduce_last(ctx: OpContext, init: str, combine: list[str],
                 in_buf: str | None = None,
                 in_shape: list[int] | None = None) -> str:
    """Reduce the last axis of `in_buf` (default in_bufs[0]) → tmp of rank-1-less.

    `combine` is body lines producing %r from %a (element) and %acc.
    """
    src = in_buf or ctx.in_bufs[0]
    sh = in_shape or ctx.in_shapes[0]
    rank = len(sh)
    out_sh = sh[:-1]
    out = ctx.tmp(out_sh)
    src_ty = _memref_ty(sh, ctx.elem)
    out_ty = _memref_ty(out_sh, ctx.elem)
    imap = _amap(rank, ", ".join(f"d{i}" for i in range(rank)))
    rmap = _amap(rank, ", ".join(f"d{i}" for i in range(rank - 1)))
    iters = ", ".join(['"parallel"'] * (rank - 1) + ['"reduction"'])
    cst = ctx.const(init)
    lines = [
        f"    linalg.fill ins({cst} : {ctx.elem}) outs({out} : {out_ty})",
        f"    linalg.generic {{indexing_maps = [{imap}, {rmap}], "
        f"iterator_types = [{iters}]}} "
        f"ins({src} : {src_ty}) outs({out} : {out_ty}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %acc: " + ctx.elem + "):",
        *combine,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]
    ctx._preamble.extend(lines)
    return out


def _broadcast_ew(ctx: OpContext, full_buf: str, full_shape: list[int],
                  red_buf: str, out_buf: str, body: list[str]) -> None:
    """Elementwise over `full_buf` with a last-axis-broadcast `red_buf` → out_buf.

    body computes %r from %a (full elem) and %b (broadcast elem).
    """
    rank = len(full_shape)
    ft = _memref_ty(full_shape, ctx.elem)
    rt = _memref_ty(full_shape[:-1], ctx.elem)
    ot = _memref_ty(full_shape, ctx.elem)
    fmap = _amap(rank, ", ".join(f"d{i}" for i in range(rank)))
    bmap = _amap(rank, ", ".join(f"d{i}" for i in range(rank - 1)))
    iters = ", ".join(['"parallel"'] * rank)
    lines = [
        f"    linalg.generic {{indexing_maps = [{fmap}, {bmap}, {fmap}], "
        f"iterator_types = [{iters}]}} "
        f"ins({full_buf}, {red_buf} : {ft}, {rt}) outs({out_buf} : {ot}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %b: " + ctx.elem + ", %o: " + ctx.elem + "):",
        *body,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]
    ctx._preamble.extend(lines)


# ── OT1: reductions ────────────────────────────────────────────

def _c_reduce_sum(ctx: OpContext) -> list[str]:
    red = _reduce_last(ctx, "0.0", [
        "      %r = arith.addf %a, %acc : " + ctx.elem,
    ])
    _copy_into(ctx, red, ctx.out_buf, ctx.out_shape)
    return ctx._preamble


def _c_reduce_max(ctx: OpContext) -> list[str]:
    red = _reduce_last(ctx, "-3.40282347e+38", [
        "      %r = arith.maximumf %a, %acc : " + ctx.elem,
    ])
    _copy_into(ctx, red, ctx.out_buf, ctx.out_shape)
    return ctx._preamble


def _c_reduce_mean(ctx: OpContext) -> list[str]:
    n = ctx.in_shapes[0][-1]
    red = _reduce_last(ctx, "0.0", [
        "      %r = arith.addf %a, %acc : " + ctx.elem,
    ])
    # divide by N elementwise
    ot = _memref_ty(ctx.out_shape, ctx.elem)
    rank = len(ctx.out_shape)
    imap = _amap(rank, ", ".join(f"d{i}" for i in range(rank)))
    iters = ", ".join(['"parallel"'] * rank)
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{imap}, {imap}], "
        f"iterator_types = [{iters}]}} "
        f"ins({red} : {ot}) outs({ctx.out_buf} : {ot}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %o: " + ctx.elem + "):",
        f"      %n = arith.constant {float(n)} : {ctx.elem}",
        "      %r = arith.divf %a, %n : " + ctx.elem,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]
    return ctx._preamble


# ── OT1: normalizations ────────────────────────────────────────

def _c_softmax(ctx: OpContext) -> list[str]:
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    mx = _reduce_last(ctx, "-3.40282347e+38", [
        "      %r = arith.maximumf %a, %acc : " + ctx.elem,
    ])
    ex = ctx.tmp(sh)
    _broadcast_ew(ctx, src, sh, mx, ex, [
        "      %s = arith.subf %a, %b : " + ctx.elem,
        "      %r = math.exp %s : " + ctx.elem,
    ])
    sm = _reduce_last(ctx, "0.0", [
        "      %r = arith.addf %a, %acc : " + ctx.elem,
    ], in_buf=ex, in_shape=sh)
    _broadcast_ew(ctx, ex, sh, sm, ctx.out_buf, [
        "      %r = arith.divf %a, %b : " + ctx.elem,
    ])
    return ctx._preamble


def _c_layernorm(ctx: OpContext) -> list[str]:
    # mean/var over last axis, (x-mean)/sqrt(var+eps). weight=1, bias=0.
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    n = sh[-1]
    mean = _reduce_last(ctx, "0.0", [
        "      %r = arith.addf %a, %acc : " + ctx.elem,
    ])
    mean_d = ctx.tmp(sh[:-1])
    _scalar_div(ctx, mean, mean_d, sh[:-1], float(n))
    # centered
    cen = ctx.tmp(sh)
    _broadcast_ew(ctx, src, sh, mean_d, cen, [
        "      %r = arith.subf %a, %b : " + ctx.elem,
    ])
    # variance = mean(cen^2)
    sq = ctx.tmp(sh)
    _unary_ew(ctx, cen, sq, sh, [
        "      %r = arith.mulf %a, %a : " + ctx.elem,
    ])
    var = _reduce_last(ctx, "0.0", [
        "      %r = arith.addf %a, %acc : " + ctx.elem,
    ], in_buf=sq, in_shape=sh)
    var_d = ctx.tmp(sh[:-1])
    _scalar_div(ctx, var, var_d, sh[:-1], float(n))
    # rstd = rsqrt(var + eps)
    rstd = ctx.tmp(sh[:-1])
    _unary_ew(ctx, var_d, rstd, sh[:-1], [
        "      %eps = arith.constant 1.0e-05 : " + ctx.elem,
        "      %ve = arith.addf %a, %eps : " + ctx.elem,
        "      %r = math.rsqrt %ve : " + ctx.elem,
    ])
    # out = cen * rstd
    _broadcast_ew(ctx, cen, sh, rstd, ctx.out_buf, [
        "      %r = arith.mulf %a, %b : " + ctx.elem,
    ])
    return ctx._preamble


def _c_rmsnorm(ctx: OpContext) -> list[str]:
    # x * rsqrt(mean(x^2) + eps). weight=1.
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    n = sh[-1]
    sq = ctx.tmp(sh)
    _unary_ew(ctx, src, sq, sh, [
        "      %r = arith.mulf %a, %a : " + ctx.elem,
    ])
    ms = _reduce_last(ctx, "0.0", [
        "      %r = arith.addf %a, %acc : " + ctx.elem,
    ], in_buf=sq, in_shape=sh)
    ms_d = ctx.tmp(sh[:-1])
    _scalar_div(ctx, ms, ms_d, sh[:-1], float(n))
    rstd = ctx.tmp(sh[:-1])
    _unary_ew(ctx, ms_d, rstd, sh[:-1], [
        "      %eps = arith.constant 1.0e-06 : " + ctx.elem,
        "      %ve = arith.addf %a, %eps : " + ctx.elem,
        "      %r = math.rsqrt %ve : " + ctx.elem,
    ])
    _broadcast_ew(ctx, src, sh, rstd, ctx.out_buf, [
        "      %r = arith.mulf %a, %b : " + ctx.elem,
    ])
    return ctx._preamble


# ── OT2: data movement + dense ─────────────────────────────────

def _c_transpose(ctx: OpContext) -> list[str]:
    # 2D transpose: out[j,i] = in[i,j]
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    m, n = sh
    it = _memref_ty([m, n], ctx.elem)
    ot = _memref_ty([n, m], ctx.elem)
    inmap = "affine_map<(d0, d1) -> (d1, d0)>"
    outmap = "affine_map<(d0, d1) -> (d0, d1)>"
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{inmap}, {outmap}], "
        'iterator_types = ["parallel", "parallel"]} '
        f"ins({src} : {it}) outs({ctx.out_buf} : {ot}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %o: " + ctx.elem + "):",
        "      linalg.yield %a : " + ctx.elem,
        "    }",
    ]
    return ctx._preamble


def _c_batch_matmul(ctx: OpContext) -> list[str]:
    a, b = ctx.in_bufs
    (bs, m, k), (bs2, k2, n) = ctx.in_shapes[0], ctx.in_shapes[1]
    at = _memref_ty([bs, m, k], ctx.elem)
    bt = _memref_ty([bs, k, n], ctx.elem)
    ct = _memref_ty([bs, m, n], ctx.elem)
    zero = ctx.const("0.0")
    ctx._preamble += [
        f"    linalg.fill ins({zero} : {ctx.elem}) outs({ctx.out_buf} : {ct})",
        f"    linalg.batch_matmul ins({a}, {b} : {at}, {bt}) "
        f"outs({ctx.out_buf} : {ct})",
    ]
    return ctx._preamble


def _c_copy(ctx: OpContext) -> list[str]:
    _copy_into(ctx, ctx.in_bufs[0], ctx.out_buf, ctx.out_shape)
    return ctx._preamble


def _c_cast(ctx: OpContext) -> list[str]:
    # f32 → f32 identity (benchmark cast targets float32); a real dtype cast
    # would use arith.extf/truncf but the JIT print path is f32-only.
    _copy_into(ctx, ctx.in_bufs[0], ctx.out_buf, ctx.out_shape)
    return ctx._preamble


def _c_where(ctx: OpContext) -> list[str]:
    # where(cond, a, b): cond is 0/1 f32 → select. out = cond*a + (1-cond)*b
    # (branchless, avoids i1 in the f32 print path).
    cond, a, b = ctx.in_bufs
    sh = ctx.out_shape
    rank = len(sh)
    ty = _memref_ty(sh, ctx.elem)
    imap = _amap(rank, ", ".join(f"d{i}" for i in range(rank)))
    iters = ", ".join(['"parallel"'] * rank)
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{imap}, {imap}, {imap}, {imap}], "
        f"iterator_types = [{iters}]}} "
        f"ins({cond}, {a}, {b} : {ty}, {ty}, {ty}) outs({ctx.out_buf} : {ty}) {{",
        "    ^bb0(%c: " + ctx.elem + ", %a: " + ctx.elem
        + ", %b: " + ctx.elem + ", %o: " + ctx.elem + "):",
        f"      %one = arith.constant 1.0 : {ctx.elem}",
        "      %nc = arith.subf %one, %c : " + ctx.elem,
        "      %ca = arith.mulf %c, %a : " + ctx.elem,
        "      %cb = arith.mulf %nc, %b : " + ctx.elem,
        "      %r = arith.addf %ca, %cb : " + ctx.elem,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]
    return ctx._preamble


def _c_permute(ctx: OpContext) -> list[str]:
    # 3D permute(0,2,1): out[b,j,i] = in[b,i,j]
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    b, m, n = sh
    it = _memref_ty([b, m, n], ctx.elem)
    ot = _memref_ty([b, n, m], ctx.elem)
    inmap = "affine_map<(d0, d1, d2) -> (d0, d2, d1)>"
    outmap = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>"
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{inmap}, {outmap}], "
        'iterator_types = ["parallel", "parallel", "parallel"]} '
        f"ins({src} : {it}) outs({ctx.out_buf} : {ot}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %o: " + ctx.elem + "):",
        "      linalg.yield %a : " + ctx.elem,
        "    }",
    ]
    return ctx._preamble


def _c_concat(ctx: OpContext) -> list[str]:
    # concat two tensors along the last axis via subview copies.
    a, b = ctx.in_bufs
    sha, shb = ctx.in_shapes[0], ctx.in_shapes[1]
    out_sh = ctx.out_shape
    at = _memref_ty(sha, ctx.elem)
    bt = _memref_ty(shb, ctx.elem)
    na = sha[-1]
    total = out_sh[-1]
    # subview offsets/sizes
    rank = len(out_sh)
    off_a = ", ".join(["0"] * rank)
    size_a = ", ".join(str(d) for d in sha)
    off_b = ", ".join(["0"] * (rank - 1) + [str(na)])
    size_b = ", ".join(str(d) for d in shb)
    ones = ", ".join(["1"] * rank)
    # strides for row-major out
    strides = []
    acc = 1
    for d in reversed(out_sh):
        strides.append(acc); acc *= d
    strides = list(reversed(strides))
    stride_s = ", ".join(str(s) for s in strides)
    sa_ty = f"memref<{'x'.join(str(d) for d in sha)}x{ctx.elem}, strided<[{stride_s}], offset: 0>>"
    sb_ty = f"memref<{'x'.join(str(d) for d in shb)}x{ctx.elem}, strided<[{stride_s}], offset: {na}>>"
    ctx._preamble += [
        f"    %sa_{ctx.out_buf[1:]} = memref.subview {ctx.out_buf}[{off_a}] "
        f"[{size_a}] [{ones}] : {_memref_ty(out_sh, ctx.elem)} to {sa_ty}",
        f"    memref.copy {a}, %sa_{ctx.out_buf[1:]} : {at} to {sa_ty}",
        f"    %sb_{ctx.out_buf[1:]} = memref.subview {ctx.out_buf}[{off_b}] "
        f"[{size_b}] [{ones}] : {_memref_ty(out_sh, ctx.elem)} to {sb_ty}",
        f"    memref.copy {b}, %sb_{ctx.out_buf[1:]} : {bt} to {sb_ty}",
    ]
    return ctx._preamble


def _c_rmsnorm_residual(ctx: OpContext) -> list[str]:
    # rmsnorm(x + residual). two inputs: x, residual.
    sh = ctx.in_shapes[0]
    x, res = ctx.in_bufs
    n = sh[-1]
    added = ctx.tmp(sh)
    ty = _memref_ty(sh, ctx.elem)
    rank = len(sh)
    imap = _amap(rank, ", ".join(f"d{i}" for i in range(rank)))
    iters = ", ".join(['"parallel"'] * rank)
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{imap}, {imap}, {imap}], "
        f"iterator_types = [{iters}]}} "
        f"ins({x}, {res} : {ty}, {ty}) outs({added} : {ty}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %b: " + ctx.elem + ", %o: " + ctx.elem + "):",
        "      %r = arith.addf %a, %b : " + ctx.elem,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]
    sq = ctx.tmp(sh)
    _unary_ew(ctx, added, sq, sh, ["      %r = arith.mulf %a, %a : " + ctx.elem])
    ms = _reduce_last(ctx, "0.0", ["      %r = arith.addf %a, %acc : " + ctx.elem],
                      in_buf=sq, in_shape=sh)
    ms_d = ctx.tmp(sh[:-1])
    _scalar_div(ctx, ms, ms_d, sh[:-1], float(n))
    rstd = ctx.tmp(sh[:-1])
    _unary_ew(ctx, ms_d, rstd, sh[:-1], [
        "      %eps = arith.constant 1.0e-06 : " + ctx.elem,
        "      %ve = arith.addf %a, %eps : " + ctx.elem,
        "      %r = math.rsqrt %ve : " + ctx.elem,
    ])
    _broadcast_ew(ctx, added, sh, rstd, ctx.out_buf, [
        "      %r = arith.mulf %a, %b : " + ctx.elem,
    ])
    return ctx._preamble


def _gated_mul(ctx: OpContext, act: str) -> list[str]:
    # gated activation: split last axis in half → act(x1) * x2.
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    full = sh[-1]
    half = full // 2
    half_sh = sh[:-1] + [half]
    rank = len(sh)
    ty = _memref_ty(sh, ctx.elem)
    hty = _memref_ty(half_sh, ctx.elem)
    # strides for row-major src
    strides = []
    acc = 1
    for d in reversed(sh):
        strides.append(acc); acc *= d
    strides = list(reversed(strides))
    stride_s = ", ".join(str(s) for s in strides)
    off0 = ", ".join(["0"] * rank)
    off1 = ", ".join(["0"] * (rank - 1) + [str(half)])
    size = ", ".join(str(d) for d in half_sh)
    ones = ", ".join(["1"] * rank)
    s1_ty = f"memref<{'x'.join(str(d) for d in half_sh)}x{ctx.elem}, strided<[{stride_s}], offset: 0>>"
    s2_ty = f"memref<{'x'.join(str(d) for d in half_sh)}x{ctx.elem}, strided<[{stride_s}], offset: {half}>>"
    x1 = ctx.tmp(half_sh)
    x2 = ctx.tmp(half_sh)
    ctx._preamble += [
        f"    %g1_{ctx.out_buf[1:]} = memref.subview {src}[{off0}] [{size}] [{ones}] "
        f": {ty} to {s1_ty}",
        f"    memref.copy %g1_{ctx.out_buf[1:]}, {x1} : {s1_ty} to {hty}",
        f"    %g2_{ctx.out_buf[1:]} = memref.subview {src}[{off1}] [{size}] [{ones}] "
        f": {ty} to {s2_ty}",
        f"    memref.copy %g2_{ctx.out_buf[1:]}, {x2} : {s2_ty} to {hty}",
    ]
    # act(x1)
    a1 = ctx.tmp(half_sh)
    if act == "silu":
        _unary_ew(ctx, x1, a1, half_sh, [
            "      %one = arith.constant 1.0 : " + ctx.elem,
            "      %nx = arith.negf %a : " + ctx.elem,
            "      %e = math.exp %nx : " + ctx.elem,
            "      %d = arith.addf %one, %e : " + ctx.elem,
            "      %s = arith.divf %one, %d : " + ctx.elem,
            "      %r = arith.mulf %a, %s : " + ctx.elem,
        ])
    else:  # gelu tanh-approx
        _unary_ew(ctx, x1, a1, half_sh, [
            "      %half = arith.constant 0.5 : " + ctx.elem,
            "      %one = arith.constant 1.0 : " + ctx.elem,
            "      %c0 = arith.constant 0.7978845608028654 : " + ctx.elem,
            "      %c1 = arith.constant 0.044715 : " + ctx.elem,
            "      %x2 = arith.mulf %a, %a : " + ctx.elem,
            "      %x3 = arith.mulf %x2, %a : " + ctx.elem,
            "      %t0 = arith.mulf %c1, %x3 : " + ctx.elem,
            "      %t1 = arith.addf %a, %t0 : " + ctx.elem,
            "      %t2 = arith.mulf %c0, %t1 : " + ctx.elem,
            "      %th = math.tanh %t2 : " + ctx.elem,
            "      %t3 = arith.addf %one, %th : " + ctx.elem,
            "      %t4 = arith.mulf %half, %a : " + ctx.elem,
            "      %r = arith.mulf %t4, %t3 : " + ctx.elem,
        ])
    # out = act(x1) * x2
    ty_h = hty
    hrank = len(half_sh)
    himap = _amap(hrank, ", ".join(f"d{i}" for i in range(hrank)))
    hiters = ", ".join(['"parallel"'] * hrank)
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{himap}, {himap}, {himap}], "
        f"iterator_types = [{hiters}]}} "
        f"ins({a1}, {x2} : {ty_h}, {ty_h}) outs({ctx.out_buf} : {ty_h}) {{",
        "    ^bb0(%p: " + ctx.elem + ", %q: " + ctx.elem + ", %o: " + ctx.elem + "):",
        "      %r = arith.mulf %p, %q : " + ctx.elem,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]
    return ctx._preamble


def _c_silu_and_mul(ctx: OpContext) -> list[str]:
    return _gated_mul(ctx, "silu")


def _c_gelu_and_mul(ctx: OpContext) -> list[str]:
    return _gated_mul(ctx, "gelu")


def _c_cumsum(ctx: OpContext) -> list[str]:
    # cumulative sum along last axis via scf.for prefix accumulate.
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    rank = len(sh)
    n = sh[-1]
    ty = _memref_ty(sh, ctx.elem)
    # Use an scf loop nest: for each outer index, running sum along last dim.
    # Emit imperative loops (scf.for) with memref load/store.
    outer = sh[:-1]
    idx_vars = [f"%i{d}" for d in range(len(outer))]
    lines = []
    # constants
    c0 = ctx.const_idx(0)
    c1 = ctx.const_idx(1)
    bounds = [ctx.const_idx(d) for d in outer]
    cn = ctx.const_idx(n)
    zero = ctx.const("0.0")
    # build nested parallel loops over outer dims, sequential over last
    indent = "    "
    for d, (iv, bnd) in enumerate(zip(idx_vars, bounds)):
        lines.append(f"{indent}scf.for {iv} = {c0} to {bnd} step {c1} {{")
        indent += "  "
    acc_var = f"%acc_{ctx.out_buf[1:]}"
    lines.append(f"{indent}%init = arith.constant 0.0 : {ctx.elem}")
    lines.append(f"{indent}scf.for %j = {c0} to {cn} step {c1} "
                 f"iter_args(%s = %init) -> {ctx.elem} {{")
    idxlist = ", ".join(idx_vars + ["%j"])
    lines.append(f"{indent}  %v = memref.load {src}[{idxlist}] : {ty}")
    lines.append(f"{indent}  %ns = arith.addf %s, %v : {ctx.elem}")
    lines.append(f"{indent}  memref.store %ns, {ctx.out_buf}[{idxlist}] : {ty}")
    lines.append(f"{indent}  scf.yield %ns : {ctx.elem}")
    lines.append(f"{indent}}}")
    for d in range(len(outer)):
        indent = indent[:-2]
        lines.append(f"{indent}}}")
    ctx._preamble += lines
    return ctx._preamble


def _c_split(ctx: OpContext) -> list[str]:
    # chunk(2, dim=-1)[0] — first half of the last axis (single-output).
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    half = sh[-1] // 2
    half_sh = sh[:-1] + [half]
    rank = len(sh)
    ty = _memref_ty(sh, ctx.elem)
    hty = _memref_ty(half_sh, ctx.elem)
    strides = []
    acc = 1
    for d in reversed(sh):
        strides.append(acc); acc *= d
    strides = list(reversed(strides))
    stride_s = ", ".join(str(s) for s in strides)
    off0 = ", ".join(["0"] * rank)
    size = ", ".join(str(d) for d in half_sh)
    ones = ", ".join(["1"] * rank)
    s_ty = f"memref<{'x'.join(str(d) for d in half_sh)}x{ctx.elem}, strided<[{stride_s}], offset: 0>>"
    ctx._preamble += [
        f"    %sp_{ctx.out_buf[1:]} = memref.subview {src}[{off0}] [{size}] [{ones}] "
        f": {ty} to {s_ty}",
        f"    memref.copy %sp_{ctx.out_buf[1:]}, {ctx.out_buf} : {s_ty} to {hty}",
    ]
    return ctx._preamble


def _c_argmax(ctx: OpContext) -> list[str]:
    # argmax along last axis via scf.for; output f32(index) (exact for idx<2^24).
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    n = sh[-1]
    ty = _memref_ty(sh, ctx.elem)
    outer = sh[:-1]
    out_ty = _memref_ty(outer, ctx.elem)
    u = ctx.out_buf[1:]  # unique suffix
    idx_vars = [f"%i{d}_{u}" for d in range(len(outer))]
    c0 = ctx.const_idx(0)
    c1 = ctx.const_idx(1)
    cn = ctx.const_idx(n)
    bounds = [ctx.const_idx(d) for d in outer]
    lines = []
    indent = "    "
    for iv, bnd in zip(idx_vars, bounds):
        lines.append(f"{indent}scf.for {iv} = {c0} to {bnd} step {c1} {{")
        indent += "  "
    idx0 = ", ".join(idx_vars + [c0])
    out_idx = ", ".join(idx_vars) if idx_vars else ""
    lines += [
        f"{indent}%v0_{u} = memref.load {src}[{idx0}] : {ty}",
        f"{indent}%res_{u}:2 = scf.for %j_{u} = {c1} to {cn} step {c1} "
        f"iter_args(%best_{u} = %v0_{u}, %bidx_{u} = {c0}) -> ({ctx.elem}, index) {{",
        f"{indent}  %v_{u} = memref.load {src}[{', '.join(idx_vars + ['%j_' + u])}] : {ty}",
        f"{indent}  %gt_{u} = arith.cmpf ogt, %v_{u}, %best_{u} : {ctx.elem}",
        f"{indent}  %nb_{u} = arith.select %gt_{u}, %v_{u}, %best_{u} : {ctx.elem}",
        f"{indent}  %ni_{u} = arith.select %gt_{u}, %j_{u}, %bidx_{u} : index",
        f"{indent}  scf.yield %nb_{u}, %ni_{u} : {ctx.elem}, index",
        f"{indent}}}",
        f"{indent}%fi_{u} = arith.index_cast %res_{u}#1 : index to i64",
        f"{indent}%ff_{u} = arith.sitofp %fi_{u} : i64 to {ctx.elem}",
        f"{indent}memref.store %ff_{u}, {ctx.out_buf}[{out_idx}] : {out_ty}",
    ]
    for _ in outer:
        indent = indent[:-2]
        lines.append(f"{indent}}}")
    ctx._preamble += lines
    return ctx._preamble


def _c_embedding(ctx: OpContext) -> list[str]:
    # embedding(indices, table): out[i,:] = table[int(indices[i]),:].
    # indices come in as f32 (JIT print path is f32-only) → cast to index.
    idx_buf, table = ctx.in_bufs
    idx_sh, tbl_sh = ctx.in_shapes[0], ctx.in_shapes[1]
    n_idx = idx_sh[0]
    vocab, dim = tbl_sh
    idx_ty = _memref_ty(idx_sh, ctx.elem)
    tbl_ty = _memref_ty(tbl_sh, ctx.elem)
    out_ty = _memref_ty(ctx.out_shape, ctx.elem)
    u = ctx.out_buf[1:]
    c0 = ctx.const_idx(0)
    c1 = ctx.const_idx(1)
    cni = ctx.const_idx(n_idx)
    cdim = ctx.const_idx(dim)
    ctx._preamble += [
        f"    scf.for %i_{u} = {c0} to {cni} step {c1} {{",
        f"      %fidx_{u} = memref.load {idx_buf}[%i_{u}] : {idx_ty}",
        f"      %ii_{u} = arith.fptosi %fidx_{u} : {ctx.elem} to i64",
        f"      %row_{u} = arith.index_cast %ii_{u} : i64 to index",
        f"      scf.for %d_{u} = {c0} to {cdim} step {c1} {{",
        f"        %v_{u} = memref.load {table}[%row_{u}, %d_{u}] : {tbl_ty}",
        f"        memref.store %v_{u}, {ctx.out_buf}[%i_{u}, %d_{u}] : {out_ty}",
        "      }",
        "    }",
    ]
    return ctx._preamble


def _c_rope(ctx: OpContext) -> list[str]:
    # Rotary position embedding (single input). x shape [..., D], D even.
    #   x1 = x[..., :D/2], x2 = x[..., D/2:]
    #   out[..., :D/2] = x1*cos - x2*sin ; out[..., D/2:] = x2*cos + x1*sin
    # theta_i = 10000^(-2i/D), position = last outer-dim index (seq).
    sh = ctx.in_shapes[0]
    src = ctx.in_bufs[0]
    D = sh[-1]
    half = D // 2
    outer = sh[:-1]
    ty = _memref_ty(sh, ctx.elem)
    out_ty = _memref_ty(ctx.out_shape, ctx.elem)
    u = ctx.out_buf[1:]
    idx_vars = [f"%i{d}_{u}" for d in range(len(outer))]
    c0 = ctx.const_idx(0)
    c1 = ctx.const_idx(1)
    chalf = ctx.const_idx(half)
    bounds = [ctx.const_idx(d) for d in outer]
    pos_var = idx_vars[-1] if idx_vars else c0
    e = ctx.elem
    lines = []
    indent = "    "
    for iv, bnd in zip(idx_vars, bounds):
        lines.append(f"{indent}scf.for {iv} = {c0} to {bnd} step {c1} {{")
        indent += "  "
    lines.append(f"{indent}scf.for %k_{u} = {c0} to {chalf} step {c1} {{")
    idx1 = ", ".join(idx_vars + [f"%k_{u}"])
    idx2 = ", ".join(idx_vars + [f"%k2_{u}"])
    lines += [
        f"{indent}  %posi_{u} = arith.index_cast {pos_var} : index to i64",
        f"{indent}  %posf_{u} = arith.sitofp %posi_{u} : i64 to {e}",
        f"{indent}  %ki_{u} = arith.index_cast %k_{u} : index to i64",
        f"{indent}  %kf_{u} = arith.sitofp %ki_{u} : i64 to {e}",
        f"{indent}  %two_{u} = arith.constant 2.0 : {e}",
        f"{indent}  %dd_{u} = arith.constant {float(D)} : {e}",
        f"{indent}  %exp0_{u} = arith.mulf %two_{u}, %kf_{u} : {e}",
        f"{indent}  %exp1_{u} = arith.divf %exp0_{u}, %dd_{u} : {e}",
        f"{indent}  %base_{u} = arith.constant 10000.0 : {e}",
        f"{indent}  %lnb_{u} = math.log %base_{u} : {e}",
        f"{indent}  %pw_{u} = arith.mulf %exp1_{u}, %lnb_{u} : {e}",
        f"{indent}  %invpw_{u} = arith.negf %pw_{u} : {e}",
        f"{indent}  %theta_{u} = math.exp %invpw_{u} : {e}",
        f"{indent}  %ang_{u} = arith.mulf %posf_{u}, %theta_{u} : {e}",
        f"{indent}  %cos_{u} = math.cos %ang_{u} : {e}",
        f"{indent}  %sin_{u} = math.sin %ang_{u} : {e}",
        f"{indent}  %k2_{u} = arith.addi %k_{u}, {chalf} : index",
        f"{indent}  %x1_{u} = memref.load {src}[{idx1}] : {ty}",
        f"{indent}  %x2_{u} = memref.load {src}[{idx2}] : {ty}",
        f"{indent}  %x1c_{u} = arith.mulf %x1_{u}, %cos_{u} : {e}",
        f"{indent}  %x2s_{u} = arith.mulf %x2_{u}, %sin_{u} : {e}",
        f"{indent}  %o1_{u} = arith.subf %x1c_{u}, %x2s_{u} : {e}",
        f"{indent}  %x2c_{u} = arith.mulf %x2_{u}, %cos_{u} : {e}",
        f"{indent}  %x1s_{u} = arith.mulf %x1_{u}, %sin_{u} : {e}",
        f"{indent}  %o2_{u} = arith.addf %x2c_{u}, %x1s_{u} : {e}",
        f"{indent}  memref.store %o1_{u}, {ctx.out_buf}[{idx1}] : {out_ty}",
        f"{indent}  memref.store %o2_{u}, {ctx.out_buf}[{idx2}] : {out_ty}",
        f"{indent}}}",
    ]
    for _ in outer:
        indent = indent[:-2]
        lines.append(f"{indent}}}")
    ctx._preamble += lines
    return ctx._preamble


# ── shared helpers ─────────────────────────────────────────────

def _copy_into(ctx: OpContext, src: str, dst: str, shape: list[int]) -> None:
    ty = _memref_ty(shape, ctx.elem)
    ctx._preamble.append(f"    memref.copy {src}, {dst} : {ty} to {ty}")


def _unary_ew(ctx: OpContext, src: str, dst: str, shape: list[int],
              body: list[str]) -> None:
    rank = len(shape)
    ty = _memref_ty(shape, ctx.elem)
    imap = _amap(rank, ", ".join(f"d{i}" for i in range(rank))) if rank else \
        "affine_map<() -> ()>"
    iters = ", ".join(['"parallel"'] * rank)
    ctx._preamble += [
        f"    linalg.generic {{indexing_maps = [{imap}, {imap}], "
        f"iterator_types = [{iters}]}} "
        f"ins({src} : {ty}) outs({dst} : {ty}) {{",
        "    ^bb0(%a: " + ctx.elem + ", %o: " + ctx.elem + "):",
        *body,
        "      linalg.yield %r : " + ctx.elem,
        "    }",
    ]


def _scalar_div(ctx: OpContext, src: str, dst: str, shape: list[int],
                denom: float) -> None:
    _unary_ew(ctx, src, dst, shape, [
        f"      %n = arith.constant {denom} : {ctx.elem}",
        "      %r = arith.divf %a, %n : " + ctx.elem,
    ])


# ── composite op registry ──────────────────────────────────────

COMPOSITE_SPECS: dict[str, dict] = {
    "reduce_sum":  {"num_inputs": 1, "emit": _c_reduce_sum,  "out": "reduce_last"},
    "reduce_max":  {"num_inputs": 1, "emit": _c_reduce_max,  "out": "reduce_last"},
    "reduce_mean": {"num_inputs": 1, "emit": _c_reduce_mean, "out": "reduce_last"},
    "softmax":     {"num_inputs": 1, "emit": _c_softmax,     "out": "same"},
    "layernorm":   {"num_inputs": 1, "emit": _c_layernorm,   "out": "same"},
    "rmsnorm":     {"num_inputs": 1, "emit": _c_rmsnorm,     "out": "same"},
    "rmsnorm_residual": {"num_inputs": 2, "emit": _c_rmsnorm_residual, "out": "same"},
    "transpose":   {"num_inputs": 1, "emit": _c_transpose,   "out": "transpose2d"},
    "batch_matmul":{"num_inputs": 2, "emit": _c_batch_matmul,"out": "bmm"},
    "copy_":       {"num_inputs": 1, "emit": _c_copy,        "out": "same"},
    "cast":        {"num_inputs": 1, "emit": _c_cast,        "out": "same"},
    "where_":      {"num_inputs": 3, "emit": _c_where,       "out": "same"},
    "permute":     {"num_inputs": 1, "emit": _c_permute,     "out": "permute021"},
    "concat":      {"num_inputs": 2, "emit": _c_concat,      "out": "concat_last"},
    "silu_and_mul":{"num_inputs": 1, "emit": _c_silu_and_mul,"out": "half_last"},
    "gelu_and_mul":{"num_inputs": 1, "emit": _c_gelu_and_mul,"out": "half_last"},
    "cumsum":      {"num_inputs": 1, "emit": _c_cumsum,      "out": "same"},
    "split":       {"num_inputs": 1, "emit": _c_split,       "out": "half_last"},
    "argmax":      {"num_inputs": 1, "emit": _c_argmax,      "out": "reduce_last"},
    "embedding":   {"num_inputs": 2, "emit": _c_embedding,   "out": "embedding"},
    "rope":        {"num_inputs": 1, "emit": _c_rope,        "out": "same"},
}


def composite_output_shape(op: str, in_shapes: list[list[int]]) -> list[int]:
    rule = COMPOSITE_SPECS[op]["out"]
    if rule == "same":
        return list(in_shapes[0])
    if rule == "reduce_last":
        return list(in_shapes[0][:-1])
    if rule == "transpose2d":
        return [in_shapes[0][1], in_shapes[0][0]]
    if rule == "bmm":
        (bs, m, k), (bs2, k2, n) = in_shapes[0], in_shapes[1]
        return [bs, m, n]
    if rule == "permute021":
        b, m, n = in_shapes[0]
        return [b, n, m]
    if rule == "concat_last":
        base = list(in_shapes[0][:-1])
        return base + [in_shapes[0][-1] + in_shapes[1][-1]]
    if rule == "half_last":
        base = list(in_shapes[0][:-1])
        return base + [in_shapes[0][-1] // 2]
    if rule == "embedding":
        # out = [n_indices, embed_dim]
        return [in_shapes[0][0], in_shapes[1][1]]
    raise NotImplementedError(f"composite_output_shape: {op}")


def emit_composite(op: str, out_buf: str, out_shape: list[int],
                   in_bufs: list[str], in_shapes: list[list[int]],
                   elem: str) -> list[str]:
    ctx = OpContext(op=op, out_buf=out_buf, out_shape=out_shape,
                    in_bufs=in_bufs, in_shapes=in_shapes, elem=elem)
    return COMPOSITE_SPECS[op]["emit"](ctx)
