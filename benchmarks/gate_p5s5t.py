#!/usr/bin/env python3
# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5-S5-T tightened gate — 5 locked criteria (Leon-approved 2026-07-22).

Criteria (docs/roadmap/plan.md, P5-S5-T row — LOCKED, no relaxation):

  C1 per-case:      every explore case, agent strategy latency (us, median)
                    <= default latency. decisions=[] cases count as equal
                    (PASS by definition; measured once, feeding C3).
  C2 recovery:      every case in the LOCKED C2_CASE_KEYS list (currently
                    softmax@1024x4096, sweep 1.049) must reach agent ratio
                    (vs this-run measured CUDA-C) <= 1.05. rmsnorm@32x4096
                    was removed 2026-07-22 (Leon-approved): its sweep 1.285
                    was a same-cubin phantom (see threshold block).
  C3 overall:       latency-weighted geomean <= 0.948 over the 23 gate cases
                    = 8 L3 gate cases (this-run measured agent ratio,
                    this-run cudac_us weights; matmul@2048 gate=False stays
                    out) + 15 non_l3 cases (stored l3_sweep ratio + cudac_us
                    weights).
  C4 self-discovery: every non-empty strategy decision carries a non-empty
                    @rationale; source.extracted_from points at a real
                    state.json (existence-checked). Audit table emitted.
  C5 held-out:      rmsnorm/softmax/layernorm @ [256x4096, 64x8192] +
                    matmul @ 1536^3, decisions from
                    apply_rule(strategies/{op}_rule.json). Every case:
                    agent <= default x 1.00 (decisions=[] handled like C1).
                    Missing rule file = FAIL (not skip).

Measurement discipline is benchmarks/l3_sweep.py verbatim (imported):
throwaway ramp pass, interleaved default/agent/CUDA-C, median-of-3
kernel-only CUDA events (benchmark_cached returns MILLISECONDS -> x1e3).

Evidence JSON: benchmarks/results/phase5/s5/gate_p5s5t.json

Usage:
    cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
    export PATH=/usr/local/cuda-13.2/bin:$PATH
    python benchmarks/gate_p5s5t.py [--skip-live-measure] [--only KEY_SUBSTR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
S5_DIR = REPO_ROOT / "benchmarks" / "results" / "phase5" / "s5"
STRATEGIES_DIR = S5_DIR / "strategies"
L3_SWEEP_JSON = S5_DIR / "l3_sweep.json"
EVIDENCE_JSON = S5_DIR / "gate_p5s5t.json"

# Locked thresholds (P5-S5-T — do not relax).
# Recalibration history (Leon-approved 2026-07-22, Discord "2" = proposal 2b):
#   C3 0.940 -> 0.948. The original 0.940 derived from the recon sweep's
#   all-best 0.9347, which counted rmsnorm@32x4096 "+18.6% headroom" — later
#   proven a PHANTOM (bt(512) emits a byte-identical cubin to the default;
#   the delta was two noisy measurements of the same kernel, commit b9bfd80).
#   True realizable all-best measured 0.9456 (round5, all real headroom
#   consumed: rmsnorm@1024 -3.7%, matmul@1024 -3.3%, matmul@2048 -3.8%);
#   0.948 = measured-attainable + CoV margin. Physical reason documented,
#   thresholds remain locked at the new values.
C2_RATIO_MAX = 1.05
C3_GEOMEAN_MAX = 0.948

# C2 default-losing case list (keys into l3_sweep cases). Leon-approved
# 2026-07-22: rmsnorm@32x4096 REMOVED — its sweep default_ratio 1.285 was
# the same-cubin phantom above (cross-run ratio of the same default config:
# 1.285 / 2.234 / 2.792 / 0.918 / 0.879 / 1.097 — never a stable loss).
# softmax@1024x4096 (1.049, stable across runs) remains.
C2_CASE_KEYS = ("softmax@1024x4096",)

# Held-out generalization matrix (C5).
HELDOUT_MATRIX: list[tuple[str, list[int]]] = [
    ("rmsnorm", [256, 4096]),
    ("rmsnorm", [64, 8192]),
    ("softmax", [256, 4096]),
    ("softmax", [64, 8192]),
    ("layernorm", [256, 4096]),
    ("layernorm", [64, 8192]),
    ("matmul", [1536, 1536, 1536]),
]


# ── shape/key helpers ───────────────────────────────────────────────────────

def shape_label(dims: list[int]) -> str:
    return "x".join(str(d) for d in dims)


def l3_sweep_key(op: str, dims: list[int]) -> str:
    """Map (op, dims) to the l3_sweep.json case key.

    l3_sweep labels matmul by its square side ("1024x1024"); rowwise ops by
    MxN. Explore/strategies files label matmul by full MxKxN.
    """
    if op == "matmul":
        return f"{op}@{dims[0]}x{dims[2]}"
    return f"{op}@{shape_label(dims)}"


def dims_from_l3_case(rec: dict) -> list[int]:
    """Recover concrete dims from an l3_sweep case record."""
    parts = [int(p) for p in rec["shape"].split("x")]
    if rec["op"] == "matmul":
        s = parts[0]
        return [s, s, s]
    return parts


# ── strategies loading / Decision rebuild ───────────────────────────────────

def strategy_path(op: str, dims: list[int]) -> Path:
    return STRATEGIES_DIR / f"{op}_{shape_label(dims)}.json"


def load_strategy_record(op: str, dims: list[int]) -> dict | None:
    p = strategy_path(op, dims)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def build_decisions(dec_dicts: list[dict]) -> list[Any]:
    """Rebuild arke.ir.strategy.Decision objects from strategy-file dicts.

    Same consumption path as l3_sweep candidates: the list is handed to
    ``LLVMBackend.lower(graph, strategy=[Decision, ...])``.
    """
    from arke.ir.strategy import Decision, Rationale

    out = []
    for d in dec_dicts:
        rat = d.get("rationale")
        if isinstance(rat, dict):
            rat = rat.get("text", "")
        out.append(Decision(
            kind=d["kind"],
            params=dict(d.get("params") or {}),
            level=int(d.get("level", 3)),
            rationale=Rationale(text=str(rat)) if rat else None,
        ))
    return out


# ── pure criteria evaluators (unit-testable, no GPU) ────────────────────────

def eval_c1(explore_rows: list[dict]) -> dict:
    """explore_rows: [{key, default_us, agent_us, decisions_empty, error?,
    pair_ratio_median?}].

    decisions_empty rows PASS by definition (agent == default config).
    A row with an error (missing strategy file / compile fail) FAILs.

    Comparison aggregation (measurement quality, target unchanged):
    when the row carries pair_ratio_median (median over per-pass agent/default
    ratios from the SAME interleaved passes), the verdict is
    pair_ratio_median <= 1.0 — pass-pairing cancels slow clock/thermal drift
    that can push two independently-aggregated medians apart by ~1% on a
    ~340us kernel (observed: agent wins 7/10 paired passes at ratio 0.9955
    while a 3-pass window read +0.66% the other way). Rows without
    pair_ratio_median fall back to the plain median comparison.
    """
    cases = []
    ok = True
    for row in explore_rows:
        if row.get("error"):
            passed = False
        elif row.get("decisions_empty"):
            passed = True
        elif row.get("pair_ratio_median") is not None:
            passed = row["pair_ratio_median"] <= 1.0
        else:
            passed = row["agent_us"] <= row["default_us"]
        ok = ok and passed
        cases.append({**row, "pass": passed})
    return {"criterion": "C1 per-case agent<=default",
            "pass": ok, "cases": cases}


def eval_c2(losing_rows: list[dict], ratio_max: float = C2_RATIO_MAX) -> dict:
    """losing_rows: [{key, sweep_default_ratio, agent_us, cudac_us, error?}]."""
    cases = []
    ok = True
    for row in losing_rows:
        if row.get("error") or not row.get("cudac_us"):
            passed, ratio = False, None
        else:
            ratio = row["agent_us"] / row["cudac_us"]
            passed = ratio <= ratio_max
        ok = ok and passed
        cases.append({**row,
                      "agent_ratio": round(ratio, 4) if ratio else None,
                      "threshold": ratio_max, "pass": passed})
    return {"criterion": f"C2 default-losing cases recovered to <={ratio_max}",
            "pass": ok, "cases": cases}


def eval_c3(l3_rows: list[dict], non_l3_rows: list[dict],
            geomean_max: float = C3_GEOMEAN_MAX) -> dict:
    """Latency-weighted geomean over l3_rows + non_l3_rows.

    Each row: {key, ratio, weight_us}. Aggregation matches l3_sweep summary:
    exp(sum(w*ln r)/sum(w)), w = cudac reference latency (us).
    """
    from benchmarks.l3_sweep import weighted_geomean

    rows = [(r["ratio"], r["weight_us"]) for r in l3_rows + non_l3_rows
            if r.get("ratio") and r.get("weight_us")]
    complete = (len(rows) == len(l3_rows) + len(non_l3_rows)
                and len(l3_rows) > 0 and len(non_l3_rows) > 0)
    gm = weighted_geomean(rows) if rows else None
    ok = complete and gm is not None and gm <= geomean_max
    return {"criterion": f"C3 weighted geomean <= {geomean_max}",
            "pass": ok,
            "gate_cases": len(rows),
            "weighted_geomean": round(gm, 4) if gm is not None else None,
            "threshold": geomean_max,
            "l3_rows": l3_rows, "non_l3_rows": non_l3_rows}


def eval_c4(strategy_records: list[dict],
            exists_fn: Callable[[str], bool] | None = None) -> dict:
    """strategy_records: [{key, record}] where record is the strategies json
    (or None when the file is missing). Audit table:
      - non-empty strategy -> every decision has a non-empty rationale
      - source.extracted_from must reference an existing state.json
    """
    if exists_fn is None:
        exists_fn = lambda p: Path(p).is_file()  # noqa: E731
    audit = []
    ok = True
    for item in strategy_records:
        key, rec = item["key"], item["record"]
        if rec is None:
            audit.append({"key": key, "pass": False,
                          "reason": "strategies file missing"})
            ok = False
            continue
        decisions = rec.get("decisions") or []
        missing_rat = [i for i, d in enumerate(decisions)
                       if not str(d.get("rationale") or "").strip()]
        src = (rec.get("source") or {}).get("extracted_from") or ""
        state_path = src.split("::")[0]
        src_ok = bool(state_path) and exists_fn(state_path)
        passed = not missing_rat and src_ok
        ok = ok and passed
        audit.append({
            "key": key,
            "n_decisions": len(decisions),
            "decisions_missing_rationale": missing_rat,
            "extracted_from": src,
            "state_json_exists": src_ok,
            "pass": passed,
        })
    return {"criterion": "C4 self-discovered strategies with @rationale",
            "pass": ok, "audit": audit}


def eval_c5(heldout_rows: list[dict]) -> dict:
    """heldout_rows: [{key, default_us, agent_us, decisions_empty,
    rule_missing?, error?, pair_ratio_median?}]. Missing rule file = FAIL
    (locked). pair_ratio_median (when present) is the verdict metric, same
    drift-cancelling rationale as eval_c1."""
    cases = []
    ok = True
    for row in heldout_rows:
        if row.get("rule_missing") or row.get("error"):
            passed = False
        elif row.get("decisions_empty"):
            passed = True
        elif row.get("pair_ratio_median") is not None:
            passed = row["pair_ratio_median"] <= 1.0
        else:
            passed = row["agent_us"] <= row["default_us"] * 1.00
        ok = ok and passed
        cases.append({**row, "pass": passed})
    return {"criterion": "C5 held-out generalization agent<=default",
            "pass": ok, "cases": cases}


# ── measurement (l3_sweep primitives, GPU path) ─────────────────────────────

def _iters_for_dims(dims: list[int]) -> tuple[int, int]:
    if max(dims) >= 2048:
        return 30, 10
    return 100, 30


def _passes_iters_boost(op: str, dims: list[int],
                        iters: int, warmup: int) -> tuple[int, int, int]:
    """(passes, iters, warmup) — small-kernel anti-spike boost.

    Sub-50us rowwise kernels on WSL show intermittent multi-ms scheduling
    spikes (a 10us kernel intermittently reads 15-30us for a whole pass);
    with only 3 passes x 100 iters a single bad window flips a verdict.
    Boost to 5 passes x 300 iters so one spiked pass cannot own the median
    and each pass integrates over ~3ms. Applied SYMMETRICALLY to default,
    agent, and CUDA-C measurements — this is measurement quality, not a
    standard change (thresholds untouched).
    """
    from benchmarks.l3_sweep import MEAS_PASSES
    if op != "matmul" and len(dims) == 2 and (dims[0] * dims[1]) <= 2_000_000:
        return 5, 300, warmup
    return MEAS_PASSES, iters, warmup


def graph_fn_for(op: str, dims: list[int]):
    from benchmarks.llvm_vs_cuda_c import (
        _g_layernorm, _g_matmul, _g_rmsnorm, _g_softmax,
    )
    if op == "matmul":
        if not (dims[0] == dims[1] == dims[2]):
            raise ValueError(f"only square matmul supported here: {dims}")
        return _g_matmul(dims[0])
    m, n = dims
    return {"softmax": _g_softmax, "layernorm": _g_layernorm,
            "rmsnorm": _g_rmsnorm}[op](m, n)


def measure_case(llvm, cudac, op: str, dims: list[int],
                 decisions: list[Any]) -> dict:
    """Interleaved default/agent/CUDA-C measurement for one case.

    Mirrors l3_sweep.measure_batch discipline (throwaway ramp, interleaved
    passes, median + spread, kernel-only CUDA events) with a small-kernel
    anti-spike boost (see _passes_iters_boost): more passes/iters applied
    symmetrically to all variants. Empty decisions -> the default kernel is
    measured once and doubles as the agent config.
    """
    import statistics

    import numpy as np

    from benchmarks.l3_sweep import (
        SEED, bench_llvm_compiled, compile_llvm,
    )
    from benchmarks.llvm_vs_cuda_c import _make_inputs

    gf = graph_fn_for(op, dims)
    iters, warmup = _iters_for_dims(dims)
    passes, iters, warmup = _passes_iters_boost(op, dims, iters, warmup)

    g = gf()
    kern_c = cudac.compile(cudac.lower(g))
    if not kern_c.success:
        raise RuntimeError(f"CUDA-C compile failed: {kern_c.error}")
    np.random.seed(SEED)
    cudac_inputs = _make_inputs(kern_c.metadata["emitted"])

    variants = [("default", compile_llvm(llvm, gf, None))]
    if decisions:
        variants.append(("agent", compile_llvm(llvm, gf, decisions)))

    emitted0 = variants[0][1].metadata["emitted"]
    np.random.seed(SEED)
    inputs = _make_inputs(emitted0)

    # Throwaway ramp pass (unrecorded) — clocks up before recording.
    for _label, kern in variants:
        bench_llvm_compiled(llvm, kern, inputs, iters=max(10, iters // 3),
                            warmup=max(5, warmup // 3))
    cudac.benchmark(kern_c, cudac_inputs,
                    iters=max(10, iters // 3), warmup=max(5, warmup // 3))

    per: dict[str, list[float]] = {label: [] for label, _ in variants}
    cud: list[float] = []
    for p in range(passes):
        for label, kern in variants:
            per[label].append(
                bench_llvm_compiled(llvm, kern, inputs, iters, warmup))
        cud.append(cudac.benchmark(kern_c, cudac_inputs,
                                   iters=iters, warmup=warmup) * 1e3)
        print(f"  {op}@{shape_label(dims)} pass {p + 1}/{passes} done",
              flush=True)

    def _agg(vals: list[float]) -> dict:
        med = statistics.median(vals)
        spread = round(max(vals) / min(vals) - 1.0, 4) if min(vals) > 0 else None
        return {"us": med, "spread": spread, "passes": vals}

    default = _agg(per["default"])
    agent = _agg(per["agent"]) if "agent" in per else default
    cudac_agg = _agg(cud)
    # Per-pass pairwise agent/default ratio median: both variants are
    # measured within the same pass, so the ratio cancels slow drift.
    pair_ratio_median = None
    if "agent" in per:
        pairs = [a / d for a, d in zip(per["agent"], per["default"]) if d > 0]
        if pairs:
            pair_ratio_median = round(float(statistics.median(pairs)), 6)
    return {
        "iters": iters, "warmup": warmup, "passes_n": passes,
        "default_us": default["us"], "default_spread": default["spread"],
        "default_passes": default["passes"],
        "agent_us": agent["us"], "agent_spread": agent["spread"],
        "agent_passes": agent["passes"],
        "pair_ratio_median": pair_ratio_median,
        "cudac_us": cudac_agg["us"], "cudac_spread": cudac_agg["spread"],
        "cudac_passes": cudac_agg["passes"],
    }


# ── gate orchestration ──────────────────────────────────────────────────────

def load_l3_sweep() -> dict:
    if not L3_SWEEP_JSON.is_file():
        raise FileNotFoundError(f"missing baseline data: {L3_SWEEP_JSON}")
    return json.loads(L3_SWEEP_JSON.read_text())


def explore_matrix() -> list[tuple[str, list[int]]]:
    from benchmarks.live.run_p5s5t import EXPLORE_MATRIX
    return list(EXPLORE_MATRIX)


def gate_l3_cases(sweep: dict) -> list[dict]:
    """The 8 gate=True L3 cases from l3_sweep.json (matmul@2048 excluded)."""
    return [rec for rec in sweep["cases"].values()
            if rec.get("gate") and "error" not in rec]


def non_l3_rows(sweep: dict) -> list[dict]:
    """The 15 non_l3 rows with stored ratio + cudac_us weights."""
    rows = []
    for key, rec in sweep["non_l3"].items():
        if "error" in rec:
            continue
        rows.append({"key": key, "ratio": rec["ratio"],
                     "weight_us": rec["cudac_us"], "source": "l3_sweep.json"})
    return rows


def run_gate(skip_live_measure: bool, only: str | None = None) -> dict:
    sweep = load_l3_sweep()
    explore = explore_matrix()

    # ---- C4 (pure file checks — runs in both modes) ----
    strategy_records = []
    for op, dims in explore:
        rec = load_strategy_record(op, dims)
        strategy_records.append({"key": f"{op}@{shape_label(dims)}",
                                 "record": rec})
    c4 = eval_c4(strategy_records)

    # ---- rule-file presence (C5 precondition, both modes) ----
    rule_files = {}
    for op in ("rmsnorm", "softmax", "layernorm", "matmul"):
        p = STRATEGIES_DIR / f"{op}_rule.json"
        rule_files[op] = str(p) if p.is_file() else None

    evidence: dict[str, Any] = {
        "mode": "dry-run" if skip_live_measure else "measured",
        "l3_sweep_json": str(L3_SWEEP_JSON),
        "thresholds": {"c2_ratio_max": C2_RATIO_MAX,
                       "c3_geomean_max": C3_GEOMEAN_MAX},
        "rule_files": rule_files,
        "criteria": {},
    }

    if skip_live_measure:
        structure = {
            "strategies_present": [r["key"] for r in strategy_records
                                   if r["record"] is not None],
            "strategies_missing": [r["key"] for r in strategy_records
                                   if r["record"] is None],
            "rules_missing": [op for op, p in rule_files.items() if p is None],
        }
        evidence["structure"] = structure
        evidence["criteria"] = {
            "C1": {"criterion": "C1", "pass": None, "skipped": "dry-run"},
            "C2": {"criterion": "C2", "pass": None, "skipped": "dry-run"},
            "C3": {"criterion": "C3", "pass": None, "skipped": "dry-run"},
            "C4": c4,
            "C5": {"criterion": "C5", "pass": None, "skipped": "dry-run",
                   "rules_missing": structure["rules_missing"]},
        }
        evidence["overall"] = None
        evidence["dry_run_ok"] = (c4["pass"]
                                  and not structure["strategies_missing"]
                                  and not structure["rules_missing"])
        return evidence

    # ---- live measurement path ----
    import torch

    from arke.backend.cuda_c_backend import CudaCBackend
    from arke.backend.llvm_backend import LLVMBackend
    from benchmarks.live.generalize_p5s5t import apply_rule

    assert torch.cuda.is_available()
    torch.cuda.init()
    llvm = LLVMBackend(chip="sm_86")
    cudac = CudaCBackend(chip="sm_86")

    measurements: dict[str, dict] = {}

    def measure(op: str, dims: list[int], decisions_dicts: list[dict],
                tag: str) -> dict:
        key = f"{tag}:{op}@{shape_label(dims)}"
        if key in measurements:
            return measurements[key]
        decs = build_decisions(decisions_dicts) if decisions_dicts else []
        m = measure_case(llvm, cudac, op, dims, decs)
        measurements[key] = m
        print(f"[{key}] default {m['default_us']:.2f}us  "
              f"agent {m['agent_us']:.2f}us  cudac {m['cudac_us']:.2f}us",
              flush=True)
        return m

    # ---- C1: explore cases, agent vs default ----
    explore_rows = []
    for op, dims in explore:
        key = f"{op}@{shape_label(dims)}"
        if only and only not in key:
            continue
        rec = load_strategy_record(op, dims)
        if rec is None:
            explore_rows.append({"key": key, "error": "strategies file missing"})
            continue
        decisions = rec.get("decisions") or []
        try:
            m = measure(op, dims, decisions, "explore")
        except Exception as e:
            explore_rows.append({"key": key, "error": str(e)})
            continue
        explore_rows.append({
            "key": key,
            "decisions_empty": not decisions,
            "n_decisions": len(decisions),
            "default_us": m["default_us"], "default_spread": m["default_spread"],
            "agent_us": m["agent_us"], "agent_spread": m["agent_spread"],
            "pair_ratio_median": m.get("pair_ratio_median"),
            "cudac_us": m["cudac_us"], "cudac_spread": m["cudac_spread"],
        })
    c1 = eval_c1(explore_rows)

    # ---- C2: sweep default-losing gate cases, agent ratio vs this-run cudac ----
    # Case membership is the LOCKED C2_CASE_KEYS list (Leon-approved 2026-07-22
    # recalibration: rmsnorm@32x4096 removed as a same-cubin phantom; see the
    # threshold block at the top of this file).
    losing_rows = []
    for srec in gate_l3_cases(sweep):
        op, dims = srec["op"], dims_from_l3_case(srec)
        key = f"{op}@{shape_label(dims)}"
        if key not in C2_CASE_KEYS:
            continue
        row = next((r for r in explore_rows if r["key"] == key), None)
        if row is None or row.get("error"):
            losing_rows.append({"key": key,
                                "sweep_default_ratio": srec["default_ratio"],
                                "error": (row or {}).get("error",
                                                         "not measured")})
            continue
        losing_rows.append({
            "key": key,
            "sweep_default_ratio": srec["default_ratio"],
            "agent_us": row["agent_us"],
            "cudac_us": row["cudac_us"],
        })
    c2 = eval_c2(losing_rows)

    # ---- C3: 8 L3 gate cases (this-run agent ratio) + 15 non_l3 (stored) ----
    l3_rows = []
    for srec in gate_l3_cases(sweep):
        op, dims = srec["op"], dims_from_l3_case(srec)
        key = f"{op}@{shape_label(dims)}"
        row = next((r for r in explore_rows if r["key"] == key), None)
        if row is None or row.get("error"):
            # gate case outside the explore matrix (matmul@512): the agent
            # made no decision for it -> default config, measured this run.
            rec = load_strategy_record(op, dims)
            decisions = (rec or {}).get("decisions") or []
            try:
                m = measure(op, dims, decisions, "gate")
                row = {"key": key, "agent_us": m["agent_us"],
                       "cudac_us": m["cudac_us"],
                       "decisions_empty": not decisions}
            except Exception as e:
                l3_rows.append({"key": key, "ratio": None, "weight_us": None,
                                "error": str(e)})
                continue
        l3_rows.append({
            "key": key,
            "ratio": round(row["agent_us"] / row["cudac_us"], 4),
            "weight_us": row["cudac_us"],
            "source": "measured",
        })
    c3 = eval_c3(l3_rows, non_l3_rows(sweep))

    # ---- C5: held-out shapes via apply_rule ----
    heldout_rows = []
    for op, dims in HELDOUT_MATRIX:
        key = f"{op}@{shape_label(dims)}"
        if only and only not in key:
            continue
        rule_path = rule_files.get(op)
        if rule_path is None:
            heldout_rows.append({"key": key, "rule_missing": True})
            continue
        rule = json.loads(Path(rule_path).read_text())
        try:
            decisions = apply_rule(rule, dims)
        except Exception as e:
            heldout_rows.append({"key": key, "error": f"apply_rule: {e}"})
            continue
        try:
            m = measure(op, dims, decisions, "heldout")
        except Exception as e:
            heldout_rows.append({"key": key, "error": str(e),
                                 "n_decisions": len(decisions)})
            continue
        heldout_rows.append({
            "key": key,
            "decisions_empty": not decisions,
            "n_decisions": len(decisions),
            "decisions": decisions,
            "default_us": m["default_us"], "default_spread": m["default_spread"],
            "agent_us": m["agent_us"], "agent_spread": m["agent_spread"],
            "pair_ratio_median": m.get("pair_ratio_median"),
        })
    c5 = eval_c5(heldout_rows)

    llvm.release_all()

    evidence["criteria"] = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    evidence["overall"] = all(c["pass"] for c in (c1, c2, c3, c4, c5))
    return evidence


def print_report(evidence: dict) -> None:
    print("\n" + "=" * 78)
    print("P5-S5-T TIGHTENED GATE — 5 criteria (locked)")
    print("=" * 78)
    for name, c in evidence["criteria"].items():
        if c.get("skipped"):
            print(f"{name}: SKIPPED ({c['skipped']})")
            continue
        status = "PASS" if c["pass"] else "FAIL"
        print(f"{name}: {status} — {c.get('criterion', '')}")
        for row in c.get("cases", []) or c.get("audit", []):
            extra = ""
            if "agent_us" in row and "default_us" in row:
                extra = (f" agent {row['agent_us']:.2f}us vs "
                         f"default {row['default_us']:.2f}us")
            if "agent_ratio" in row and row["agent_ratio"]:
                extra += f" ratio {row['agent_ratio']}"
            if row.get("error"):
                extra += f" ERROR: {row['error']}"
            if row.get("rule_missing"):
                extra += " RULE FILE MISSING"
            mark = "ok " if row.get("pass") else "FAIL"
            print(f"    [{mark}] {row['key']}{extra}")
        if name == "C3" and c.get("weighted_geomean") is not None:
            print(f"    weighted geomean = {c['weighted_geomean']} "
                  f"(threshold <= {c['threshold']}, "
                  f"{c['gate_cases']} gate cases)")
    if evidence["overall"] is None:
        print(f"\nDRY-RUN structure check: "
              f"{'OK' if evidence.get('dry_run_ok') else 'PROBLEMS FOUND'}")
    else:
        print(f"\nOVERALL: {'PASS' if evidence['overall'] else 'FAIL'}")
    print(f"Evidence: {EVIDENCE_JSON}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P5-S5-T tightened gate")
    ap.add_argument("--skip-live-measure", action="store_true",
                    help="dry-run: structure/file checks only, no GPU")
    ap.add_argument("--only", default=None,
                    help="only measure cases whose key contains this substring "
                         "(debugging; gate verdict is only valid on a full run)")
    args = ap.parse_args(argv)

    evidence = run_gate(args.skip_live_measure, only=args.only)
    EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2))
    print_report(evidence)

    if evidence["overall"] is None:
        return 0 if evidence.get("dry_run_ok") else 1
    return 0 if evidence["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
