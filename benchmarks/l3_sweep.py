#!/usr/bin/env python3
"""P5-S5 tightening recon: full L3 decision-space sweep over gate ops x gate shapes.

Measures, for every L3-aware gate op x gate shape (Cat C softmax/layernorm/
rmsnorm at 1024x4096 + 32x4096; Cat D matmul at 512^2 + 1024^2, plus 2048^2 as
an off-gate shape-matrix probe):

  - default config latency (strategy=None), median of MEAS_PASSES interleaved
    passes (kernel-only CUDA events via benchmark_cached)
  - every legal L3 candidate from ArkeEnv.list_legal_actions (wmma_tile full
    52-config space; block_threads {128,256,512,1024}; pipeline_stages {2,3,4}
    composed on top-3 wmma configs + default tile)
  - CUDA-C reference latency (gate denominator), same interleave discipline

and reports per-case headroom% = (default-best)/default, plus the whole-gate
latency-weighted geomean (llvm_vs_cuda_c.py aggregation) under two scenarios:
current (all defaults) vs theoretical optimum (agent tunes every L3-aware op
to its sweep best; non-L3-aware ops unchanged).

Measurement discipline (thermal laptop, RTX 3060):
  - one throwaway pass (clock ramp) then MEAS_PASSES recorded passes
  - within each pass, default + all candidates + CUDA-C ref are measured in a
    fixed order, so slow thermal drift cancels between variants
  - median across passes per variant; per-variant spread recorded as
    quality evidence
  - prepare/release per measurement keeps GPU memory bounded (2048^2 x 56
    resident modules would not fit 6 GB); module-load cost is outside the
    CUDA-event timed region

Incremental JSON is written to benchmarks/results/phase5/s5/l3_sweep.json
after every completed case; re-running skips completed cases (delete the file
or pass --fresh for a clean run).

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    export PATH=/usr/local/cuda-13.2/bin:$PATH
    python benchmarks/l3_sweep.py [--fresh] [--only KEY_SUBSTR] [--skip-non-l3]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from math import exp, log
from pathlib import Path

import numpy as np
import torch

from arke.agent.env import ArkeEnv
from arke.backend.cuda_c_backend import CudaCBackend, _ir_dtype_to_numpy
from arke.backend.llvm_backend import LLVMBackend
from arke.ir.strategy import Decision

# Gate-parity graph builders + single-pass bench helpers (Cat A path reuses
# the gate script's exact measurement functions).
from benchmarks.llvm_vs_cuda_c import (
    _bench_cuda_c,
    _bench_llvm,
    _g_binary,
    _g_layernorm,
    _g_matmul,
    _g_rmsnorm,
    _g_softmax,
    _g_unary,
    _make_inputs,
)

RESULTS = Path(__file__).parent / "results" / "phase5" / "s5" / "l3_sweep.json"
MEAS_PASSES = 3
SEED = 42


# ── aggregation (gate parity: llvm_vs_cuda_c.py weighted geomean) ──────────

def weighted_geomean(rows):
    """rows: list of (ratio, cudac_us). Weight = CUDA-C reference latency."""
    wsum = sum(w for _, w in rows)
    return exp(sum(w * log(r) for r, w in rows) / wsum)


def unweighted_geomean(ratios):
    return exp(statistics.mean([log(r) for r in ratios]))


# ── measurement primitives ─────────────────────────────────────────────────

def _iters_for(shape_label: str) -> tuple[int, int]:
    """(iters, warmup) — reduce for the slow 2048^2 sweep to bound runtime."""
    if shape_label == "2048x2048":
        return 30, 10
    return 100, 30


def bench_llvm_compiled(llvm, kern, inputs, iters, warmup):
    """Kernel-only median latency (us) for an already-compiled LLVM kernel.

    prepare -> run_fast (H2D + one validated launch) -> benchmark_cached
    (CUDA events) -> release. Module load/alloc are outside the timed region.
    """
    cached = llvm.prepare(kern)
    try:
        llvm.run_fast(cached, inputs)
        ms = llvm.benchmark_cached(cached, iters=iters, warmup=warmup)
    finally:
        llvm.release(cached)
    return ms * 1e3  # ms -> us


def run_llvm_output(llvm, kern, inputs):
    """One run_fast; returns the output numpy array (correctness evidence)."""
    cached = llvm.prepare(kern)
    try:
        out = llvm.run_fast(cached, inputs)
    finally:
        llvm.release(cached)
    return next(iter(out.values()))


def compile_llvm(llvm, graph_fn, strategy):
    g = graph_fn()
    kern = llvm.compile(llvm.lower(g, strategy=strategy))
    if not kern.success:
        raise RuntimeError(f"LLVM compile failed: {kern.error}")
    return kern


# ── candidate construction ─────────────────────────────────────────────────

def wmma_label(p):
    return f"wmma({p['WM']},{p['WN']},{p['WTM']},{p['WTN']})"


def case_candidates(op_name, env_shapes):
    """Enumerate stage-1 candidates: (label, [Decision,...]) per legal action."""
    env = ArkeEnv.from_op(op_name, env_shapes)
    out = []
    if op_name == "matmul":
        for d in env.list_legal_actions(top_n=10_000, filter_kind="wmma_tile"):
            out.append((wmma_label(d.params), [d]))
    else:
        for d in env.list_legal_actions(top_n=10_000, filter_kind="block_threads"):
            out.append((f"block_threads({d.params['n']})", [d]))
    return out


def pipeline_candidates(op_name, env_shapes, top_wmma):
    """Stage-2 (matmul only): pipeline depth on default tile + top wmma cfgs."""
    env = ArkeEnv.from_op(op_name, env_shapes)
    depths = [
        d.params["depth"]
        for d in env.list_legal_actions(top_n=10_000, filter_kind="pipeline_stages")
        if d.params["depth"] != 2  # depth=2 IS the default staging ring
    ]
    out = []
    for depth in depths:
        pd = Decision(kind="pipeline_stages", params={"depth": depth}, level=3)
        out.append((f"pipe({depth})", [pd]))  # on default tile
        for label, decisions in top_wmma:
            wd = decisions[0]
            out.append((f"{label}+pipe({depth})", [wd, pd]))
    return out


# ── per-case sweep ─────────────────────────────────────────────────────────

def measure_batch(llvm, cudac, graph_fn, variants, iters, warmup,
                  cudac_kern, cudac_inputs, log_prefix=""):
    """Interleaved median-of-N measurement of default + candidates + CUDA-C ref.

    variants: list of (label, CompiledKernel). Returns
    (per-variant {label: {"us": median, "passes": [...]}}, cudac {us, passes}).
    """
    emitted0 = variants[0][1].metadata["emitted"]
    np.random.seed(SEED)
    inputs = _make_inputs(emitted0)

    # Throwaway ramp pass (unrecorded): sustained load brings clocks up so
    # pass 1 isn't systematically slow (the 16.9% phantom-gain trap).
    for _label, kern in variants:
        bench_llvm_compiled(llvm, kern, inputs, iters=max(10, iters // 3),
                            warmup=max(5, warmup // 3))
    cudac.benchmark(cudac_kern, cudac_inputs,
                    iters=max(10, iters // 3), warmup=max(5, warmup // 3))

    per = {label: [] for label, _ in variants}
    cud = []
    for p in range(MEAS_PASSES):
        for label, kern in variants:
            per[label].append(
                bench_llvm_compiled(llvm, kern, inputs, iters, warmup))
        cud.append(cudac.benchmark(cudac_kern, cudac_inputs,
                                   iters=iters, warmup=warmup) * 1e3)
        print(f"  {log_prefix}pass {p+1}/{MEAS_PASSES} done", flush=True)
        gc.collect()

    def summarize(vals):
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med if med > 0 else 0.0
        return {"us": med, "passes": [round(v, 2) for v in vals],
                "spread": round(spread, 4)}

    return ({label: summarize(vals) for label, vals in per.items()},
            summarize(cud))


def sweep_case(llvm, cudac, op_name, shape_label, graph_fn, env_shapes,
               is_gate: bool):
    """Full L3 sweep for one op x shape. Returns the JSON record."""
    iters, warmup = _iters_for(shape_label)
    rec = {"op": op_name, "shape": shape_label, "gate": is_gate,
           "iters": iters, "warmup": warmup, "passes": MEAS_PASSES}

    # CUDA-C reference kernel (gate denominator)
    g = graph_fn()
    kern_c = cudac.compile(cudac.lower(g))
    if not kern_c.success:
        raise RuntimeError(f"CUDA-C compile failed: {kern_c.error}")
    np.random.seed(SEED)
    cudac_inputs = _make_inputs(kern_c.metadata["emitted"])

    # Stage 1: default + all first-order candidates
    candidates = case_candidates(op_name, env_shapes)
    print(f"[{op_name}@{shape_label}] {len(candidates)} stage-1 candidates; "
          f"compiling...", flush=True)
    variants = [("default", compile_llvm(llvm, graph_fn, None))]
    compile_errors = {}
    for label, decisions in candidates:
        try:
            variants.append((label, compile_llvm(llvm, graph_fn, decisions)))
        except Exception as e:  # honest failure record, keep sweeping
            compile_errors[label] = str(e)

    per, cud = measure_batch(llvm, cudac, graph_fn, variants, iters, warmup,
                             kern_c, cudac_inputs, log_prefix="s1 ")

    # Stage 2 (matmul): pipeline depth composed on top-3 wmma + default tile
    if op_name == "matmul" and len(candidates) > 0:
        ranked = sorted(
            ((label, per[label]["us"]) for label, _ in candidates
             if label in per),
            key=lambda t: t[1])
        top3_labels = {label for label, _ in ranked[:3]}
        top3 = [(label, dec) for label, dec in candidates
                if label in top3_labels]
        p2 = pipeline_candidates(op_name, env_shapes, top3)
        print(f"[{op_name}@{shape_label}] stage-2: {len(p2)} pipeline "
              f"compositions (top-3 wmma: {sorted(top3_labels)})", flush=True)
        variants2 = [("default", variants[0][1])]
        for label, decisions in p2:
            try:
                variants2.append((label, compile_llvm(llvm, graph_fn, decisions)))
            except Exception as e:
                compile_errors[label] = str(e)
        per2, cud2 = measure_batch(llvm, cudac, graph_fn, variants2, iters,
                                   warmup, kern_c, cudac_inputs,
                                   log_prefix="s2 ")
        # keep stage-1 default (same kernel measured twice = extra quality
        # evidence); record stage-2 default separately
        per["default(s2)"] = per2.pop("default")
        per.update(per2)
        cud = {"us": statistics.median([cud["us"], cud2["us"]]),
               "passes": cud["passes"] + cud2["passes"],
               "spread": max(cud["spread"], cud2["spread"])}

    default_us = per["default"]["us"]
    cand_only = {k: v for k, v in per.items()
                 if k not in ("default", "default(s2)")}
    if cand_only:
        best_label = min(cand_only, key=lambda k: cand_only[k]["us"])
        best_us = cand_only[best_label]["us"]
    else:
        best_label, best_us = "default", default_us

    # Correctness evidence for the winner vs default output
    corr = None
    if best_label != "default":
        try:
            np.random.seed(SEED)
            kd = variants[0][1]
            inputs = _make_inputs(kd.metadata["emitted"])
            out_d = run_llvm_output(llvm, kd, inputs)
            kb = None
            for lbl, k in variants:
                if lbl == best_label:
                    kb = k
            if kb is None:  # stage-2 winner
                for lbl, dec in pipeline_candidates(
                        op_name, env_shapes,
                        [(l, d) for l, d in candidates]):
                    if lbl == best_label:
                        kb = compile_llvm(llvm, graph_fn, dec)
                        break
            out_b = run_llvm_output(llvm, kb, inputs)
            denom = float(np.linalg.norm(out_d)) or 1.0
            corr = float(np.linalg.norm(out_b - out_d) / denom)
        except Exception as e:
            corr = f"check failed: {e}"

    cudac_us = cud["us"]
    rec.update({
        "cudac_us": round(cudac_us, 2),
        "cudac_spread": cud["spread"],
        "default_us": round(default_us, 2),
        "default_ratio": round(default_us / cudac_us, 4),
        "best_label": best_label,
        "best_us": round(best_us, 2),
        "best_ratio": round(best_us / cudac_us, 4),
        "headroom_pct": round((default_us - best_us) / default_us * 100, 2),
        "best_vs_default_rel_err": corr,
        "n_candidates": len(per) - (2 if "default(s2)" in per else 1),
        "compile_errors": compile_errors,
        "variants": {k: v for k, v in sorted(per.items(),
                                             key=lambda kv: kv[1]["us"])},
    })
    return rec


# ── non-L3 gate ops (Cat A) — gate-parity measurement ──────────────────────

def bench_non_l3(llvm, cudac, op, shape_label, graph_fn):
    ls, cs = [], []
    for _ in range(MEAS_PASSES):
        ls.append(_bench_llvm(llvm, graph_fn))
        cs.append(_bench_cuda_c(cudac, graph_fn))
    if any(x is None for x in ls) or any(x is None for x in cs):
        raise RuntimeError("compile fail")
    l, c = statistics.median(ls), statistics.median(cs)
    return {"op": op, "shape": shape_label, "llvm_us": round(l, 2),
            "cudac_us": round(c, 2), "ratio": round(l / c, 4),
            "llvm_passes": [round(x, 2) for x in ls],
            "cudac_passes": [round(x, 2) for x in cs]}


# ── summary ────────────────────────────────────────────────────────────────

def build_summary(data):
    gate_rows_default, gate_rows_best = [], []
    for rec in data["cases"].values():
        if not rec.get("gate"):
            continue
        gate_rows_default.append((rec["default_ratio"], rec["cudac_us"]))
        # Agent optimum: the agent may always decline to decide, so a case
        # whose best candidate is slower than default (negative headroom)
        # contributes its default ratio, not the worse candidate.
        gate_rows_best.append((min(rec["best_ratio"], rec["default_ratio"]),
                               rec["cudac_us"]))
    for rec in data["non_l3"].values():
        gate_rows_default.append((rec["ratio"], rec["cudac_us"]))
        gate_rows_best.append((rec["ratio"], rec["cudac_us"]))

    if not gate_rows_default:
        return {}
    return {
        "gate_cases": len(gate_rows_default),
        "weighted_geomean_default": round(weighted_geomean(gate_rows_default), 4),
        "weighted_geomean_all_best": round(weighted_geomean(gate_rows_best), 4),
        "unweighted_geomean_default": round(
            unweighted_geomean([r for r, _ in gate_rows_default]), 4),
        "unweighted_geomean_all_best": round(
            unweighted_geomean([r for r, _ in gate_rows_best]), 4),
        "gate_threshold": 0.952,
    }


def save(data):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2))


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing results, start clean")
    ap.add_argument("--only", default=None,
                    help="only run cases whose key contains this substring")
    ap.add_argument("--skip-non-l3", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.init()

    data = {"meta": {"gpu": torch.cuda.get_device_name(0),
                     "meas_passes": MEAS_PASSES, "seed": SEED,
                     "method": "kernel-only CUDA events, interleaved "
                               "median-of-3, throwaway ramp pass"},
            "cases": {}, "non_l3": {}, "summary": {}}
    if RESULTS.exists() and not args.fresh:
        data.update(json.loads(RESULTS.read_text()))
        print(f"resuming: {len(data['cases'])} cases + "
              f"{len(data['non_l3'])} non-L3 already done")

    llvm = LLVMBackend(chip="sm_86")
    cudac = CudaCBackend(chip="sm_86")

    # L3-aware gate cases (+ matmul 2048^2 off-gate shape probe for Tier C)
    l3_cases = []
    for M, N in [(1024, 4096), (32, 4096)]:
        l3_cases.append(("softmax", f"{M}x{N}", _g_softmax(M, N),
                         {"X": [M, N]}, True))
        l3_cases.append(("layernorm", f"{M}x{N}", _g_layernorm(M, N),
                         {"X": [M, N], "W": [1, N], "Bias": [1, N]}, True))
        l3_cases.append(("rmsnorm", f"{M}x{N}", _g_rmsnorm(M, N),
                         {"X": [M, N], "W": [1, N]}, True))
    for s, gate in [(512, True), (1024, True), (2048, False)]:
        l3_cases.append(("matmul", f"{s}x{s}", _g_matmul(s),
                         {"A": [s, s], "B": [s, s]}, gate))

    for op, shape_label, gf, env_shapes, is_gate in l3_cases:
        key = f"{op}@{shape_label}"
        if args.only and args.only not in key:
            continue
        if key in data["cases"]:
            print(f"skip {key} (done)")
            continue
        try:
            rec = sweep_case(llvm, cudac, op, shape_label, gf, env_shapes,
                             is_gate)
        except Exception as e:
            rec = {"op": op, "shape": shape_label, "gate": is_gate,
                   "error": str(e)}
            print(f"[{key}] SWEEP FAILED: {e}", file=sys.stderr)
        data["cases"][key] = rec
        save(data)
        if "error" not in rec:
            print(f"[{key}] default {rec['default_us']}us "
                  f"(ratio {rec['default_ratio']}) -> best {rec['best_us']}us "
                  f"(ratio {rec['best_ratio']}) via {rec['best_label']} | "
                  f"headroom {rec['headroom_pct']}%", flush=True)

    # Non-L3 gate ops (Cat A elementwise/fused) — needed for the whole-gate
    # weighted geomean under both scenarios.
    if not args.skip_non_l3:
        non_l3 = []
        for M, N in [(4096, 4096), (128, 4096), (32, 4096)]:
            for op in ["relu", "silu", "gelu"]:
                non_l3.append((op, f"{M}x{N}", _g_unary(op, M, N)))
            for op in ["silu_and_mul", "gelu_and_mul"]:
                non_l3.append((op, f"{M}x{N}", _g_binary(op, M, N)))
        for op, shape_label, gf in non_l3:
            key = f"{op}@{shape_label}"
            if args.only and args.only not in key:
                continue
            if key in data["non_l3"]:
                print(f"skip non-L3 {key} (done)")
                continue
            try:
                data["non_l3"][key] = bench_non_l3(llvm, cudac, op,
                                                   shape_label, gf)
                print(f"[non-L3 {key}] ratio "
                      f"{data['non_l3'][key]['ratio']}", flush=True)
            except Exception as e:
                data["non_l3"][key] = {"op": op, "shape": shape_label,
                                       "error": str(e)}
                print(f"[non-L3 {key}] FAILED: {e}", file=sys.stderr)
            save(data)

    data["summary"] = build_summary(data)
    save(data)

    # ── stdout report ──────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("L3 DECISION-SPACE SWEEP — per op x shape")
    print("=" * 88)
    print(f"{'case':<22} {'gate':<5} {'default':>9} {'best':>9} "
          f"{'d-ratio':>8} {'b-ratio':>8} {'headroom':>9}  best config")
    for key, rec in data["cases"].items():
        if "error" in rec:
            print(f"{key:<22} ERROR: {rec['error']}")
            continue
        print(f"{key:<22} {str(rec['gate']):<5} {rec['default_us']:>8.1f}u "
              f"{rec['best_us']:>8.1f}u {rec['default_ratio']:>8.3f} "
              f"{rec['best_ratio']:>8.3f} {rec['headroom_pct']:>8.2f}%  "
              f"{rec['best_label']}")
    s = data["summary"]
    if s:
        print("-" * 88)
        print(f"Gate weighted geomean (23 gate cases, vs CUDA-C, "
              f"threshold <=0.952):")
        print(f"  all-default: {s['weighted_geomean_default']}x   "
              f"all-best (theoretical agent optimum): "
              f"{s['weighted_geomean_all_best']}x")
        print(f"  (unweighted ref: {s['unweighted_geomean_default']}x -> "
              f"{s['unweighted_geomean_all_best']}x)")
    print(f"\nJSON: {RESULTS}")

    llvm.release_all()


if __name__ == "__main__":
    main()
