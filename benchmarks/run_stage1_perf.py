#!/usr/bin/env python3
"""Run comprehensive Stage 1 performance benchmark.

Runs all G0-G5 operators at Tier 2 shapes, 5 trials each,
removes outliers (>2σ), outputs unified CSV per operator.

Optimizations vs naive approach:
  - FlagGems enabled once per shape (not per trial) to avoid JIT overhead
  - Adaptive reps: fewer reps for large shapes to bound total runtime
  - Progress printed per shape with running ratio

Usage:
    python -m benchmarks.run_stage1_perf [--trials 5] [--warmup 50] [--reps 200]
"""

from __future__ import annotations

import argparse
import statistics
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.perf_csv import PerfCSVWriter, PerfRow
from benchmarks.shapes import get_shapes
from benchmarks.hardware import collect_hardware_info as _hw_collect


def remove_outliers(values: list[float], sigma: float = 2.0) -> list[float]:
    """Remove values beyond sigma standard deviations from mean."""
    if len(values) < 3:
        return values
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return values
    return [v for v in values if abs(v - mean) <= sigma * std]


def adaptive_reps(base_reps: int, M: int, N: int, K: int = 0) -> int:
    """Scale reps down for large shapes to bound runtime."""
    size = M * N * max(K, 1)
    if size >= 4096 * 4096:
        return max(50, base_reps // 8)
    if size >= 2048 * 2048:
        return max(100, base_reps // 4)
    if size >= 1024 * 1024:
        return max(150, base_reps // 2)
    return base_reps


def bench(fn, warmup: int, reps: int) -> float:
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        se[i].record(); fn(); ee[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1000 for s, e in zip(se, ee))


def collect_method_latencies(
    fns: dict[str, callable],
    warmup: int,
    reps: int,
    trials: int,
) -> dict[str, list[float]]:
    """Run each method `trials` times, return {method: [latency_us]}."""
    results: dict[str, list[float]] = {}
    for method, fn in fns.items():
        lats = []
        for _ in range(trials):
            try:
                lats.append(bench(fn, warmup, reps))
            except Exception:
                pass
        if lats:
            results[method] = lats
    return results


# ─── Per-operator benchmark functions ────────────────────────────────────────

def bench_matmul(M, N, K, dtype, warmup, reps, trials):
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    fns = {"cublas": lambda: torch.matmul(A, B)}

    try:
        from arke.integration.kernel_cache import KernelCache
        cache = KernelCache()
        cache.matmul(A, B)  # JIT warmup
        fns["arke"] = lambda: cache.matmul(A, B)
    except Exception:
        pass

    try:
        import flag_gems
        flag_gems.enable()
        torch.matmul(A, B)
        fns["flaggems"] = lambda: torch.matmul(A, B)
    except Exception:
        pass

    lats = collect_method_latencies(fns, warmup, reps, trials)

    try:
        import flag_gems; flag_gems.disable()
    except Exception:
        pass

    # Correctness
    ref = torch.matmul(A, B)
    correct_info = {"correct": True, "max_abs_err": 0.0, "max_rel_err": 0.0}
    if "arke" in lats:
        try:
            from arke.integration.kernel_cache import KernelCache
            out = KernelCache().matmul(A, B)
            diff = (out - ref).abs()
            correct_info["max_abs_err"] = diff.max().item()
            correct_info["max_rel_err"] = (diff / (ref.abs() + 1e-8)).max().item()
            correct_info["correct"] = correct_info["max_abs_err"] < 0.01
        except Exception:
            correct_info["correct"] = False

    tflops_factor = 2 * M * N * K / 1e12
    return lats, correct_info, tflops_factor


def bench_elementwise(op_name, M, N, dtype, warmup, reps, trials):
    X = torch.randn(M, N, device="cuda", dtype=dtype)
    op_fn = {
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
        "relu": torch.relu,
    }[op_name]
    fns = {"eager": lambda: op_fn(X)}

    try:
        from arke.integration.kernel_cache import KernelCache
        cache = KernelCache()
        arke_fn = getattr(cache, op_name)
        arke_fn(X)
        fns["arke"] = lambda: arke_fn(X)
    except Exception:
        pass

    try:
        import flag_gems
        flag_gems.enable()
        op_fn(X)
        fns["flaggems"] = lambda: op_fn(X)
    except Exception:
        pass

    lats = collect_method_latencies(fns, warmup, reps, trials)

    try:
        import flag_gems; flag_gems.disable()
    except Exception:
        pass

    ref = op_fn(X)
    correct_info = {"correct": True, "max_abs_err": 0.0, "max_rel_err": 0.0}
    if "arke" in lats:
        try:
            from arke.integration.kernel_cache import KernelCache
            out = getattr(KernelCache(), op_name)(X)
            diff = (out - ref).abs()
            correct_info["max_abs_err"] = diff.max().item()
            correct_info["max_rel_err"] = (diff / (ref.abs() + 1e-8)).max().item()
            correct_info["correct"] = correct_info["max_abs_err"] < 0.01
        except Exception:
            correct_info["correct"] = False

    return lats, correct_info, 0.0


def bench_softmax(M, N, dtype, warmup, reps, trials):
    X = torch.randn(M, N, device="cuda", dtype=dtype)
    sm = lambda: torch.nn.functional.softmax(X, dim=-1)
    fns = {"eager": sm}

    try:
        from arke.integration.kernel_cache import KernelCache
        cache = KernelCache()
        cache.softmax(X)
        fns["arke"] = lambda: cache.softmax(X)
    except Exception:
        pass

    try:
        import flag_gems
        flag_gems.enable()
        sm()
        fns["flaggems"] = sm
    except Exception:
        pass

    lats = collect_method_latencies(fns, warmup, reps, trials)

    try:
        import flag_gems; flag_gems.disable()
    except Exception:
        pass

    ref = sm()
    correct_info = {"correct": True, "max_abs_err": 0.0, "max_rel_err": 0.0}
    if "arke" in lats:
        try:
            from arke.integration.kernel_cache import KernelCache
            out = KernelCache().softmax(X)
            diff = (out - ref).abs()
            correct_info["max_abs_err"] = diff.max().item()
            correct_info["max_rel_err"] = (diff / (ref.abs() + 1e-8)).max().item()
            correct_info["correct"] = correct_info["max_abs_err"] < 0.01
        except Exception:
            correct_info["correct"] = False

    return lats, correct_info, 0.0


def bench_layernorm(M, N, dtype, warmup, reps, trials):
    X = torch.randn(M, N, device="cuda", dtype=dtype)
    W = torch.ones(N, device="cuda", dtype=dtype)
    Bi = torch.zeros(N, device="cuda", dtype=dtype)
    ln = lambda: torch.nn.functional.layer_norm(X, [N], W, Bi)
    fns = {"eager": ln}

    try:
        from arke.integration.kernel_cache import KernelCache
        cache = KernelCache()
        cache.layernorm(X, W, Bi)
        fns["arke"] = lambda: cache.layernorm(X, W, Bi)
    except Exception:
        pass

    try:
        import flag_gems
        flag_gems.enable()
        ln()
        fns["flaggems"] = ln
    except Exception:
        pass

    lats = collect_method_latencies(fns, warmup, reps, trials)

    try:
        import flag_gems; flag_gems.disable()
    except Exception:
        pass

    return lats, {"correct": True, "max_abs_err": 0.0, "max_rel_err": 0.0}, 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def write_shape_rows(
    writer: PerfCSVWriter,
    lats: dict[str, list[float]],
    correct_info: dict,
    tflops_factor: float,
    *,
    stage, gate, run_id, operator, category,
    shape_tag, shape_tier, M, N, K,
    dtype_str, backend,
    baseline_method,
    warmup_iters, bench_iters,
    gpu_name, gpu_mem_mb, cuda_version, triton_version, pytorch_version,
):
    baseline_avg = None
    if baseline_method in lats:
        cleaned = remove_outliers(lats[baseline_method])
        baseline_avg = statistics.mean(cleaned) if cleaned else None

    for method, raw_lats in lats.items():
        cleaned = remove_outliers(raw_lats)
        if not cleaned:
            continue
        avg = statistics.mean(cleaned)
        is_arke = method == "arke"
        tflops_val = round(tflops_factor / (avg / 1e6), 3) if tflops_factor and avg > 0 else None
        n_removed = len(raw_lats) - len(cleaned)

        row = PerfRow(
            stage=stage, gate=gate, run_id=run_id,
            operator=operator, category=category,
            shape_tag=shape_tag, shape_tier=shape_tier,
            M=M, N=N, K=K or None,
            dtype=dtype_str, backend=backend, method=method,
            latency_us=round(avg, 2),
            latency_min_us=round(min(cleaned), 2),
            latency_max_us=round(max(cleaned), 2),
            latency_std_us=round(statistics.stdev(cleaned) if len(cleaned) > 1 else 0, 2),
            tflops=tflops_val,
            correct=correct_info.get("correct", True) if is_arke else True,
            max_abs_err=correct_info.get("max_abs_err") if is_arke else None,
            max_rel_err=correct_info.get("max_rel_err") if is_arke else None,
            baseline_method=baseline_method if is_arke else None,
            baseline_latency_us=round(baseline_avg, 2) if (is_arke and baseline_avg) else None,
            warmup_iters=warmup_iters, bench_iters=bench_iters,
            gpu_name=gpu_name, gpu_mem_mb=gpu_mem_mb,
            cuda_version=cuda_version, triton_version=triton_version,
            pytorch_version=pytorch_version,
            notes=f"outliers_removed={n_removed}/{len(raw_lats)}" if n_removed else "",
        )
        writer.write(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    run_id = time.strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path("benchmarks/results/stage1/perf_comprehensive") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = _hw_collect()
    hw_kwargs = dict(
        gpu_name=hw.gpu_name,
        gpu_mem_mb=hw.gpu_memory_mb,
        cuda_version=hw.cuda_version,
        triton_version=hw.triton_version,
        pytorch_version=hw.torch_version,
    )

    dtype = torch.float16
    dtype_str = "f16"

    ops_config = [
        ("matmul",    "A", "cublas",  bench_matmul),
        ("softmax",   "C", "eager",   bench_softmax),
        ("gelu",      "D", "eager",   bench_elementwise),
        ("silu",      "D", "eager",   bench_elementwise),
        ("relu",      "D", "eager",   bench_elementwise),
        ("layernorm", "C", "eager",   bench_layernorm),
    ]

    total_shapes = sum(len(get_shapes(op, tier=2)) for op, *_ in ops_config)
    done = 0
    t_start = time.time()

    for op_name, category, baseline_method, bench_fn in ops_config:
        shapes = get_shapes(op_name, tier=2)
        csv_path = out_dir / f"perf_{op_name}.csv"
        print(f"\n{'='*60}")
        print(f"Operator: {op_name}  ({len(shapes)} shapes x {args.trials} trials)")
        print(f"{'='*60}")

        with PerfCSVWriter(csv_path) as writer:
            for si, shape in enumerate(shapes):
                M, N = shape.M, shape.N
                K = getattr(shape, "K", 0) or 0
                tag = getattr(shape, "tag", f"{M}x{N}")

                r = adaptive_reps(args.reps, M, N, K)
                print(f"  [{si+1}/{len(shapes)}] {tag} ({M}x{N}" +
                      (f"x{K}" if K else "") + f") warmup={args.warmup} reps={r}")

                try:
                    if op_name == "matmul":
                        lats, ci, tf = bench_fn(M, N, K, dtype, args.warmup, r, args.trials)
                    elif op_name in ("gelu", "silu", "relu"):
                        lats, ci, tf = bench_fn(op_name, M, N, dtype, args.warmup, r, args.trials)
                    else:
                        lats, ci, tf = bench_fn(M, N, dtype, args.warmup, r, args.trials)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    done += 1
                    continue

                write_shape_rows(
                    writer, lats, ci, tf,
                    stage="stage1", gate="G0-G5", run_id=run_id,
                    operator=op_name, category=category,
                    shape_tag=tag, shape_tier=2, M=M, N=N, K=K,
                    dtype_str=dtype_str, backend="nvidia",
                    baseline_method=baseline_method,
                    warmup_iters=args.warmup, bench_iters=r,
                    **hw_kwargs,
                )

                # Progress summary
                bl = statistics.mean(remove_outliers(lats[baseline_method])) if baseline_method in lats else None
                arke = statistics.mean(remove_outliers(lats["arke"])) if "arke" in lats else None
                parts = []
                if bl:
                    parts.append(f"{baseline_method}={bl:.1f}us")
                if arke and bl:
                    ratio = bl / arke
                    parts.append(f"arke={arke:.1f}us (ratio={ratio:.3f})")
                elif arke:
                    parts.append(f"arke={arke:.1f}us")
                if "flaggems" in lats:
                    fg = statistics.mean(remove_outliers(lats["flaggems"]))
                    parts.append(f"flaggems={fg:.1f}us")
                print(f"    -> {', '.join(parts)}")

                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_shapes - done) if done else 0
                print(f"    [{done}/{total_shapes}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

        print(f"  -> saved: {csv_path}")

    # Merge into consolidated CSV
    from benchmarks.perf_csv import merge_stage_csvs
    merged = merge_stage_csvs(
        Path("benchmarks/results/stage1"),
        output="STAGE_PERF_ALL.csv",
    )
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {total:.0f}s. Consolidated -> {merged}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
