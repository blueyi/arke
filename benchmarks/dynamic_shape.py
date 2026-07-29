# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-shape benchmark track — the "Performance Cliff" measurement.

Background (KESTREL audit §一.H3, backlog K-DYN)
-------------------------------------------------
Production LLM inference does not run a fixed shape grid. LLaMA variable-length
decoding sweeps sequence length token-by-token; Stable-Diffusion sweeps spatial
resolution. Every genuinely-new shape a kernel sees for the first time pays a
*first-call* cost the steady-state benchmark grid never measures:

  1. launcher-side config selection (tile heuristic / autotune),
  2. Triton's own per-shape ``@triton.jit`` specialization compile.

The KESTREL audit flagged this as a **speculative** cliff — no data. This track
turns speculation into a measured curve: for each op, drive a *sequence* of
shapes through the **real production wrapper** (the same
``KERNEL_CACHE.get_or_build_by_op`` object that ``TritonBackend`` uses) and
record, per shape:

  * ``first_call_ms`` — cold: the very first invocation for that shape.
  * ``steady_ms``     — warm: median of repeated invocations.
  * ``cliff_ratio``   — ``first_call_ms / steady_ms`` (≫ 1.0 ⇒ a cliff).
  * ``spec_key`` + ``new_spec`` — the *predicted* kernel-specialization class
    (op-aware: matmul's K-H3.1 pow2 tile buckets + Triton's per-arg
    divisibility classes; softmax/rmsnorm key on their own cfg semantics).
    ``new_spec`` predicts "this shape should compile"; the measured ratio is
    the ground truth that prediction is checked against.

Design contract
---------------
This module builds the **measurement infrastructure only**. It emits curves and
a per-op cliff distribution. It deliberately does **not** define a pass/fail
Performance-Cliff *gate threshold* — that threshold is a frozen-layer decision
(gate scoring semantics) reserved for the project lead. Consumers that want a
gate wrap this track and apply their own threshold.

Why the production wrapper (not a fresh render per shape)
--------------------------------------------------------
A real deployment builds one wrapper per op and reuses it across every shape;
Triton handles per-shape kernel specialization internally under that single
wrapper. Measuring through ``get_or_build_by_op`` reproduces exactly that: the
first shape in a Triton specialization class pays the compile, shape-mates that
land on the same constexprs reuse it. Rendering a fresh module per shape (as the
``autotune_first_call`` probe does for its narrow matmul-cache question) would
*over*-report the cliff by defeating Triton's JIT cache.

Usage
-----
    source ~/.venvs/arke/bin/activate
    python -m benchmarks.dynamic_shape --op matmul,softmax,rmsnorm
    python -m benchmarks.dynamic_shape --all

Output
------
    benchmarks/results/dynamic_shape/<timestamp>/<op>_cliff.csv
    benchmarks/results/dynamic_shape/<timestamp>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "dynamic_shape"

# Ops this track knows how to sweep. Kept deliberately aligned with the
# K-H5.2 convergence-curve op set (matmul / softmax) plus rmsnorm, so the
# "dynamic-shape" and "convergence" stories reference the same kernels.
DEFAULT_OPS = ("matmul", "softmax", "rmsnorm")


# ── Shape sweeps: sequences that mimic dynamic (variable-length) workloads ──
#
# Each entry is (tag, M, N, K). Values are chosen to *cross* launcher-side
# config-bucket boundaries so the curve shows both same-bucket (warm-ish) and
# cross-bucket (cliff) transitions. Non-power-of-two sizes are intentional —
# real seq lengths are rarely aligned.


def _next_pow2(x: int) -> int:
    """Mirror of the template bucket function (matmul.py.j2 K-H3.1)."""
    if x <= 1:
        return 1
    if x <= 16:
        return x
    return 1 << (x - 1).bit_length()


def _div_class(x: int) -> str:
    """Triton JIT specialization class of an integer argument.

    Triton specializes each non-constexpr int arg on (a) equality to 1 and
    (b) divisibility by 16. Two shapes whose args share the same classes
    (and the same constexpr values) hit the same compiled kernel.
    """
    if x == 1:
        return "1"
    return "d16" if x % 16 == 0 else "d1"


def _spec_key(op: str, M: int, N: int, K: int) -> str:
    """Predicted kernel-specialization key for (op, shape).

    Models what actually drives a recompile through the production wrapper:
    the launcher-selected constexpr config PLUS Triton's per-arg
    specialization classes (``_div_class``). Op-aware because each template
    caches configs differently:

      * matmul  — ``_TILE_CFG_CACHE`` keyed on next_pow2 buckets (K-H3.1);
                  cfg constexprs + M/N/K div classes drive the JIT key.
      * softmax — ``_LAUNCH_CFG`` keyed on exact N, but the *cfg values*
                  derive from next_pow2(N); JIT key = BLOCK + M/N div classes.
      * rmsnorm — no launcher cache; kernel path chosen from N; JIT key =
                  N (constexpr-ish) + M/N div classes.

    This is a *prediction* column — the measured cold/warm ratio is the
    ground truth it gets checked against.
    """
    if op == "matmul":
        return (f"cfg{_next_pow2(M)}x{_next_pow2(N)}x{_next_pow2(K)}"
                f"|{_div_class(M)},{_div_class(N)},{_div_class(K)}")
    if op == "softmax":
        return f"blk{_next_pow2(N)}|{_div_class(M)},{_div_class(N)}"
    if op == "rmsnorm":
        return f"n{N}|{_div_class(M)},{_div_class(N)}"
    return f"shape{M}x{N}x{K}"


def _matmul_sweep() -> list[tuple[str, int, int, int]]:
    # LLaMA-style token-batch matmul: hidden N=K=4096 fixed, token count M
    # sweeps like a variable prefill / speculative-decode batch. M crosses
    # next_pow2 buckets (1,2,4,8,16,32,64,128,256,512...) so cross-bucket
    # cliffs are visible against same-bucket repeats.
    ms = [1, 2, 4, 7, 13, 16, 32, 48, 64, 100, 128, 200, 256, 384, 512]
    return [(f"m{m}", m, 4096, 4096) for m in ms]


def _softmax_sweep() -> list[tuple[str, int, int, int]]:
    # Attention-logit softmax: 32 heads (M), sequence length N sweeps. softmax
    # keys its launch config on *exact* N (no bucketing), so this exposes
    # whether every new seq length re-selects config + recompiles.
    ns = [128, 200, 256, 384, 512, 700, 1024, 1500, 2048, 3000, 4096, 8192]
    return [(f"n{n}", 32, n, 0) for n in ns]


def _rmsnorm_sweep() -> list[tuple[str, int, int, int]]:
    # RMSNorm over a hidden dim (N=4096 fixed, LLaMA-7B), row count M sweeps
    # like a variable sequence length. Tests seq-invariance of the norm kernel.
    ms = [128, 200, 256, 384, 512, 700, 1024, 1500, 2048, 3000, 4096]
    return [(f"m{m}", m, 4096, 0) for m in ms]


SHAPE_SWEEPS: dict[str, Callable[[], list[tuple[str, int, int, int]]]] = {
    "matmul": _matmul_sweep,
    "softmax": _softmax_sweep,
    "rmsnorm": _rmsnorm_sweep,
}

# Which dim is the "varying" one, for bucket-key reporting.
_VARYING_DIM: dict[str, str] = {
    "matmul": "M",
    "softmax": "N",
    "rmsnorm": "M",
}


@dataclass
class CliffRow:
    op: str
    shape_tag: str
    M: int
    N: int
    K: int
    spec_key: str
    new_spec: bool
    first_call_ms: float
    steady_ms: float
    cliff_ratio: float
    status: str = "ok"
    reason: str = ""


def cliff_ratio(first_call_ms: float, steady_ms: float) -> float:
    """cold / warm. Guards against a zero/negative steady sample."""
    if steady_ms <= 0:
        return float("inf")
    return first_call_ms / steady_ms


def geomean(values: list[float]) -> float:
    """Geometric mean of positive finite values; ignores non-finite entries."""
    clean = [v for v in values if math.isfinite(v) and v > 0]
    if not clean:
        return float("nan")
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def summarize(rows: list[CliffRow]) -> dict:
    """Per-op cliff distribution — NO gate threshold applied (frozen layer)."""
    ok = [r for r in rows if r.status == "ok"]
    ratios = [r.cliff_ratio for r in ok]
    new_spec_ratios = [r.cliff_ratio for r in ok if r.new_spec]
    same_spec_ratios = [r.cliff_ratio for r in ok if not r.new_spec]
    return {
        "n_shapes": len(rows),
        "n_ok": len(ok),
        "cliff_ratio_geomean": geomean(ratios),
        "cliff_ratio_max": max(ratios) if ratios else float("nan"),
        "cliff_ratio_median": statistics.median(ratios) if ratios else float("nan"),
        "new_spec_geomean": geomean(new_spec_ratios),
        "same_spec_geomean": geomean(same_spec_ratios),
        "n_new_spec": len(new_spec_ratios),
        "n_same_spec": len(same_spec_ratios),
    }


# ── measurement ─────────────────────────────────────────────────────────────


def _time_cold(fn, args) -> float:
    """Wall-clock (ms) of a single cold invocation, CUDA-synchronized.

    Uses ``perf_counter`` (not CUDA events) because the cliff cost is largely
    *host-side*: config selection + Triton JIT compile happen on the CPU before
    the kernel launches. CUDA events would miss that.
    """
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _time_steady(fn, args, reps: int = 50) -> float:
    """Median warm latency (ms) over ``reps`` synchronized calls."""
    import torch

    # A few unrecorded warmups to settle clocks after the cold call.
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(max(reps, 1)):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def run_sweep(
    op: str,
    *,
    dtype_str: str = "float16",
    warm_reps: int = 50,
) -> list[CliffRow]:
    """Drive the production wrapper for ``op`` through its dynamic shape sweep.

    Returns one CliffRow per shape. Cold = first call for that shape through a
    single shared wrapper (Triton specializes internally); steady = warm median.
    """
    import torch

    from arke.backend.kernel_cache import KERNEL_CACHE
    from benchmarks.baselines.arke_runner import ArkeRunner

    if op not in SHAPE_SWEEPS:
        raise ValueError(f"no dynamic-shape sweep defined for op={op!r}")

    dtype = getattr(torch, dtype_str)

    # One production wrapper for the whole op — exactly the deployment shape.
    wrapper = KERNEL_CACHE.get_or_build_by_op(op, dtype=dtype_str)
    if wrapper is None:
        return [CliffRow(op, "-", 0, 0, 0, "-", False, 0.0, 0.0, float("nan"),
                         status="no_kernel",
                         reason=f"{op} has no Triton template_hint")]

    build_inputs = ArkeRunner._build_test_inputs  # staticmethod
    rows: list[CliffRow] = []
    seen_specs: set[str] = set()

    # Amortize the one-time process-global Triton init (~300ms on the very
    # first @triton.jit ever) with a throwaway shape OUTSIDE the measured
    # sweep (different bucket AND different Triton specialization), so
    # per-shape cold numbers reflect per-shape work only. Must NOT collide
    # with any sweep shape, or that shape's "cold" would read warm.
    _INIT_WARMUP: dict[str, tuple[int, int, int]] = {
        "matmul": (3, 64, 64),
        "softmax": (4, 96, 0),
        "rmsnorm": (4, 96, 0),
    }
    try:
        wm, wn, wk = _INIT_WARMUP.get(op, (4, 64, 64))
        warm_inputs = build_inputs(op, wm, wn, wk, dtype)
        if warm_inputs is not None:
            wrapper(*warm_inputs)
            torch.cuda.synchronize()
        del warm_inputs
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("dynamic_shape: %s init warmup failed: %s", op, exc)

    for tag, M, N, K in SHAPE_SWEEPS[op]():
        spec = _spec_key(op, M, N, K)
        new_spec = spec not in seen_specs
        seen_specs.add(spec)
        try:
            inputs = build_inputs(op, M, N, K, dtype)
            if inputs is None:
                rows.append(CliffRow(op, tag, M, N, K, spec, new_spec,
                                     0.0, 0.0, float("nan"),
                                     status="no_inputs",
                                     reason="input builder returned None"))
                continue
            cold_ms = _time_cold(wrapper, inputs)
            steady_ms = _time_steady(wrapper, inputs, reps=warm_reps)
            rows.append(CliffRow(
                op, tag, M, N, K, spec, new_spec,
                cold_ms, steady_ms, cliff_ratio(cold_ms, steady_ms),
            ))
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            rows.append(CliffRow(op, tag, M, N, K, spec, new_spec,
                                 0.0, 0.0, float("nan"),
                                 status="oom", reason=str(exc)[:120]))
        except Exception as exc:
            rows.append(CliffRow(op, tag, M, N, K, spec, new_spec,
                                 0.0, 0.0, float("nan"),
                                 status="error", reason=str(exc)[:120]))
    return rows


# ── output ───────────────────────────────────────────────────────────────────


def write_csv(rows: list[CliffRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in dataclasses.fields(CliffRow)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dataclasses.asdict(r))


def _print_table(op: str, rows: list[CliffRow], summ: dict) -> None:
    print(f"\n=== {op} — dynamic-shape cliff ===")
    print(f"{'shape':>8} {'new?':>5} "
          f"{'cold_ms':>9} {'warm_ms':>9} {'cliff':>8}  spec_key")
    print("-" * 72)
    for r in rows:
        if r.status != "ok":
            print(f"{r.shape_tag:>8} {'Y' if r.new_spec else '·':>5} "
                  f"{'—':>9} {'—':>9} {'—':>8}  [{r.status}] {r.reason}")
            continue
        print(f"{r.shape_tag:>8} {'Y' if r.new_spec else '·':>5} "
              f"{r.first_call_ms:>9.3f} {r.steady_ms:>9.3f} "
              f"{r.cliff_ratio:>8.2f}  {r.spec_key}")
    print("-" * 72)
    print(f"  cliff_ratio geomean={summ['cliff_ratio_geomean']:.2f} "
          f"median={summ['cliff_ratio_median']:.2f} "
          f"max={summ['cliff_ratio_max']:.2f}")
    print(f"  new-spec geomean={summ['new_spec_geomean']:.2f} "
          f"({summ['n_new_spec']} shapes) | "
          f"same-spec geomean={summ['same_spec_geomean']:.2f} "
          f"({summ['n_same_spec']} shapes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", type=str, default=None,
                        help="Comma-separated ops (default: matmul,softmax,rmsnorm)")
    parser.add_argument("--all", action="store_true",
                        help="Run every op with a defined sweep")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warm-reps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir (default: benchmarks/results/dynamic_shape/<ts>)")
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("[dynamic_shape] torch unavailable.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("[dynamic_shape] CUDA unavailable — nothing to measure.", file=sys.stderr)
        return 1

    if args.all:
        ops = list(SHAPE_SWEEPS.keys())
    elif args.op:
        ops = [o.strip() for o in args.op.split(",") if o.strip()]
    else:
        ops = list(DEFAULT_OPS)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = args.out or (DEFAULT_OUTPUT_DIR / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, dict] = {}
    for op in ops:
        if op not in SHAPE_SWEEPS:
            print(f"[dynamic_shape] no sweep for op={op!r}, skipping.", file=sys.stderr)
            continue
        rows = run_sweep(op, dtype_str=args.dtype, warm_reps=args.warm_reps)
        summ = summarize(rows)
        all_summaries[op] = summ
        write_csv(rows, out_dir / f"{op}_cliff.csv")
        _print_table(op, rows, summ)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"dtype": args.dtype, "warm_reps": args.warm_reps, "ops": all_summaries},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[dynamic_shape] wrote {out_dir}")
    print("[dynamic_shape] NOTE: this track measures the cliff; it does NOT "
          "apply a pass/fail gate threshold (frozen-layer decision).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
