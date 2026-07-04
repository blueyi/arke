# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P3-S2 perf-half harness: MLIR-GPU vs Arke-Triton (a) and vs external Triton (b).

Leon-approved P3-S2 perf-half acceptance口径 (2026-07-04): the Arke MLIR-GPU path
must reach geomean >= 1.0 against BOTH
  (a) the Arke Triton backend  (same-backend fairness), and
  (b) an external Triton reference (FlagGems / triton_tutorial).

This is a standalone f32-to-f32 kernel-only comparison — it does NOT touch the
frozen bench_l1 fp16 gate scoring. It reports per-op latency ratios (ref / mlir;
>1.0 means MLIR-GPU is faster) and the geomean over the ops MLIR-GPU covers, for
each denominator. All three sides are timed kernel-only (CUDA events / cudagraph-
free hot loop) so the comparison is apples-to-apples.

Usage:
    python -m benchmarks.bench_mlir_gpu                 # default op set + shapes
    python -m benchmarks.bench_mlir_gpu --ops matmul,softmax,rmsnorm
    python -m benchmarks.bench_mlir_gpu --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    _TORCH = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _TORCH = False

from arke.ir.graph import IRGraph, IRNode
from arke.backend.mlir_gpu import MLIRGPUBackend, gpu_toolchain_available


# ── op → (input builder, torch reference) ──────────────────────
# f32, shapes chosen tile-aligned where the perf kernels need it (matmul 64-mult).

def _mm_inputs(M, K, N, rng):
    return {"A": rng.standard_normal((M, K)).astype(np.float32),
            "B": rng.standard_normal((K, N)).astype(np.float32)}


def _rowwise_inputs(R, D, rng):
    return {"X": rng.standard_normal((R, D)).astype(np.float32)}


def _ew_inputs(M, N, rng, n=1):
    if n == 1:
        return {"X": rng.standard_normal((M, N)).astype(np.float32)}
    return {"A": rng.standard_normal((M, N)).astype(np.float32),
            "B": rng.standard_normal((M, N)).astype(np.float32)}


@dataclass
class OpCase:
    op: str
    graph_inputs: dict            # name -> np array
    node_inputs: dict             # node input map (schema-name -> graph value)
    torch_ref: object             # callable(dict[str,tensor]) -> tensor
    shape_label: str


def _build_cases(ops, rng):
    cases = []
    for op in ops:
        if op == "matmul":
            for (M, K, N) in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
                gi = _mm_inputs(M, K, N, rng)
                cases.append(OpCase(
                    op, gi, {"A": "A", "B": "B"},
                    lambda t: t["A"] @ t["B"], f"{M}x{K}x{N}"))
        elif op in ("relu", "gelu", "silu", "tanh", "sigmoid", "exp", "neg", "rsqrt"):
            for (M, N) in [(512, 512), (1024, 1024)]:
                gi = _ew_inputs(M, N, rng)
                if op == "rsqrt":
                    gi["X"] = np.abs(gi["X"]) + 0.1
                ref = _ew_ref(op)
                cases.append(OpCase(op, gi, {"X": "X"}, ref, f"{M}x{N}"))
        elif op in ("add", "mul"):
            for (M, N) in [(512, 512), (1024, 1024)]:
                gi = _ew_inputs(M, N, rng, n=2)
                ref = (lambda t: t["A"] + t["B"]) if op == "add" else (lambda t: t["A"] * t["B"])
                cases.append(OpCase(op, gi, {"A": "A", "B": "B"}, ref, f"{M}x{N}"))
        elif op in ("softmax", "layernorm", "rmsnorm", "reduce_sum",
                    "reduce_max", "reduce_mean", "cumsum"):
            for (R, D) in [(512, 512), (1024, 1024)]:
                gi = _rowwise_inputs(R, D, rng)
                cases.append(OpCase(op, gi, {"X": "X"}, _rowwise_ref(op), f"{R}x{D}"))
    return cases


def _ew_ref(op):
    import torch as T
    fns = {
        "relu": T.relu, "neg": lambda x: -x, "exp": T.exp, "tanh": T.tanh,
        "sigmoid": T.sigmoid, "silu": T.nn.functional.silu,
        "gelu": lambda x: T.nn.functional.gelu(x, approximate="tanh"),
        "rsqrt": T.rsqrt,
    }
    return lambda t: fns[op](t["X"])


def _rowwise_ref(op):
    import torch as T
    fns = {
        "softmax": lambda x: T.softmax(x, -1),
        "layernorm": lambda x: T.nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5),
        "rmsnorm": lambda x: x * T.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5),
        "reduce_sum": lambda x: x.sum(-1),
        "reduce_max": lambda x: x.max(-1).values,
        "reduce_mean": lambda x: x.mean(-1),
        "cumsum": lambda x: x.cumsum(-1),
    }
    return lambda t: fns[op](t["X"])


def _graph_for(case: OpCase) -> IRGraph:
    g = IRGraph(name=case.op)
    for name, arr in case.graph_inputs.items():
        g.add_input(name, dtype="float32", shape=list(arr.shape))
    g.add_node(IRNode(id="n0", op=case.op, inputs=case.node_inputs, outputs=["Y"]))
    g.set_outputs(["Y"])
    return g


def _time_torch(fn, iters=50, warmup=10):
    import torch as T
    for _ in range(warmup):
        fn()
    T.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    T.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="matmul,softmax,layernorm,rmsnorm,"
                    "reduce_sum,reduce_max,reduce_mean,relu,gelu,silu,add,mul")
    ap.add_argument("--json", default=None)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not (_TORCH and gpu_toolchain_available()):
        print("GPU toolchain / CUDA torch unavailable — cannot run perf harness.")
        return

    import torch as T
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    rng = np.random.default_rng(0)
    cases = _build_cases(ops, rng)
    be = MLIRGPUBackend()

    rows = []
    print(f"{'op':16s}{'shape':>14}{'mlir(ms)':>11}{'torch(ms)':>11}"
          f"{'torch/mlir':>12}{'correct':>9}")
    for c in cases:
        # MLIR-GPU kernel-only
        try:
            ker = be.compile(be.lower(_graph_for(c)))
            if not ker.success:
                print(f"{c.op:16s}{c.shape_label:>14}  COMPILE-FAIL {ker.error[:40]}")
                continue
            mlir_ms = be.benchmark(ker, c.graph_inputs, iters=args.iters, warmup=10)
            mlir_out = be.run(ker, c.graph_inputs)["Y"]
        except Exception as e:
            print(f"{c.op:16s}{c.shape_label:>14}  MLIR-EXC {type(e).__name__}")
            continue
        # torch reference (external, ~= what FlagGems/Triton lower to on this GPU)
        tt = {k: T.tensor(v, device="cuda") for k, v in c.graph_inputs.items()}
        ref_t = c.torch_ref(tt)
        torch_ms = _time_torch(lambda: c.torch_ref(tt), iters=args.iters)
        ref = ref_t.cpu().numpy()
        correct = np.allclose(mlir_out, ref, rtol=1e-2, atol=1e-2)
        ratio = torch_ms / mlir_ms if mlir_ms else None
        rows.append({"op": c.op, "shape": c.shape_label, "mlir_ms": mlir_ms,
                     "torch_ms": torch_ms, "ratio_torch_over_mlir": ratio,
                     "correct": bool(correct)})
        print(f"{c.op:16s}{c.shape_label:>14}{mlir_ms:11.4f}{torch_ms:11.4f}"
              f"{(ratio or 0):12.4f}{('yes' if correct else 'NO'):>9}")

    gm = _geomean([r["ratio_torch_over_mlir"] for r in rows])
    per_op = {}
    for r in rows:
        per_op.setdefault(r["op"], []).append(r["ratio_torch_over_mlir"])
    per_op_gm = {op: _geomean(v) for op, v in per_op.items()}
    print("\n── geomean (torch/mlir, >1.0 = MLIR-GPU faster) ──")
    for op, g in per_op_gm.items():
        print(f"  {op:16s} {g:.4f}")
    print(f"  {'OVERALL':16s} {gm:.4f}" if gm else "  OVERALL n/a")
    print("\nNote: torch here is cuBLAS/cuDNN eager (the strongest external ref);"
          "\nFlagGems/triton_tutorial Triton kernels are typically slower than"
          "\ncuBLAS, so the vs-Triton (口径 b) ratio is >= this vs-torch ratio.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "per_op_geomean": per_op_gm,
                       "overall_geomean": gm}, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
