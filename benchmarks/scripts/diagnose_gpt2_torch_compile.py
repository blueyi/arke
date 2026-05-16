# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""D7-E1.1 — GPT-2 torch.compile 0.811x regression diagnostic.

Goal: produce a categorized breakdown of where the regression vs eager comes
from, so D7-E1.2..E1.5 can target the right surface.

What this captures
------------------
1.  Dynamo graph-break enumeration via ``torch._dynamo.explain(model, *inputs)``
    -> ``dynamo_explain.txt`` + a small JSON summary of break counts and reasons.
2.  Inductor codegen dump via ``TORCH_LOGS=output_code,recompiles`` saved to
    ``inductor_codegen.log`` (env-controlled, this script just records the
    intent and exposes the env var line for reproducibility).
3.  Steady-state wall-clock comparison for seq_len in {128, 256, 512}, eager vs
    ``torch.compile(mode='reduce-overhead')`` with separate warmup/measure
    buckets, ratio_vs_eager + memory.
4.  Counts of dynamo cache size and compile-times to detect recompile storms.

Outputs are written under ``OUTPUT_DIR`` (default
``benchmarks/results/phase1/stage8/track4/diagnose_<YYYY-MM-DD>/``):

  - ``dynamo_explain.txt``        - full explain() output
  - ``dynamo_explain_summary.json`` - {graph_count, break_count, break_reasons}
  - ``timings.json``              - per seq_len {eager_ms, compile_ms, ratio}
  - ``compile_metrics.json``      - {compile_times_s, cache_size, recompiles}
  - ``diagnosis.md``              - ranked root causes + recommended next step

This is intentionally a one-shot diagnostic, not a permanent harness piece.
Re-run it after each E1.x mitigation lands to track which root cause closed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
from transformers import GPT2LMHeadModel, GPT2Tokenizer


DEFAULT_OUTPUT_ROOT = Path(
    "benchmarks/results/phase1/stage8/track4"
) / f"diagnose_{datetime.now(timezone.utc):%Y-%m-%d}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_model(device: str, dtype: torch.dtype) -> tuple[torch.nn.Module, Any]:
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def _make_input(tokenizer: Any, seq_len: int, device: str) -> torch.Tensor:
    prompt = "The quick brown fox jumps over the lazy dog. " * 100
    ids = tokenizer.encode(prompt, return_tensors="pt")[:, :seq_len].to(device)
    if ids.shape[1] < seq_len:
        pad = torch.full(
            (1, seq_len - ids.shape[1]),
            tokenizer.eos_token_id or 0,
            dtype=ids.dtype,
            device=ids.device,
        )
        ids = torch.cat([ids, pad], dim=1)
    return ids


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _bench(
    fn,
    inputs: torch.Tensor,
    *,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    for _ in range(warmup):
        with torch.inference_mode():
            fn(inputs)
    _sync(device)

    samples_ms: list[float] = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            fn(inputs)
        _sync(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "stdev_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "samples": samples_ms,
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def diagnose_dynamo_explain(model, ids, output_dir: Path) -> dict[str, Any]:
    """Capture torch._dynamo.explain output verbatim and summarize."""
    # Reset so previous compile state doesn't pollute the report.
    torch._dynamo.reset()

    buf = io.StringIO()
    explanation = None
    try:
        explanation = torch._dynamo.explain(model)(ids)
    except Exception as exc:  # pragma: no cover - explain is best-effort
        buf.write(f"[explain failed] {type(exc).__name__}: {exc}\n")
    if explanation is not None:
        # explain() returns an ExplainOutput dataclass; its repr is the
        # canonical human-readable form.
        buf.write(repr(explanation))

    text = buf.getvalue()
    (output_dir / "dynamo_explain.txt").write_text(text)

    summary: dict[str, Any] = {
        "captured_at": _utc_iso(),
        "graph_count": None,
        "graph_break_count": None,
        "op_count": None,
        "break_reasons": [],
    }
    if explanation is not None:
        for attr in ("graph_count", "graph_break_count", "op_count"):
            if hasattr(explanation, attr):
                summary[attr] = getattr(explanation, attr)
        # break_reasons is a list of GraphCompileReason objects in modern torch.
        reasons = getattr(explanation, "break_reasons", []) or []
        summary["break_reasons"] = [
            {
                "reason": getattr(r, "reason", str(r)),
                "user_stack": [
                    str(frame) for frame in getattr(r, "user_stack", []) or []
                ][:5],
            }
            for r in reasons
        ]

    (output_dir / "dynamo_explain_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    return summary


def measure_compile_metrics(model, ids) -> dict[str, Any]:
    """Time the actual compile + record how many cache entries dynamo built."""
    torch._dynamo.reset()
    compiled = torch.compile(model, mode="reduce-overhead")

    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = compiled(ids)
    _sync(ids.device.type)
    first_call_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = compiled(ids)
    _sync(ids.device.type)
    second_call_s = time.perf_counter() - t0

    cache_size = 0
    try:
        # torch._dynamo.utils.counters tracks recompile counts in modern torch.
        counters = torch._dynamo.utils.counters
        recompiles = dict(counters.get("frames", {}))
        cache_size = counters.get("stats", {}).get("calls_captured", 0)
    except Exception:  # pragma: no cover - counters API may shift
        recompiles = {}

    return {
        "first_call_s": first_call_s,
        "second_call_s": second_call_s,
        "compile_overhead_s_estimate": max(0.0, first_call_s - second_call_s),
        "captured_calls": cache_size,
        "recompile_counters": recompiles,
    }


def measure_timings(
    model,
    tokenizer,
    seq_lens: list[int],
    *,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    timings: dict[str, Any] = {}

    # Fresh compile for the timing measurement; use a separate handle so we
    # don't reuse compile_metrics state.
    torch._dynamo.reset()
    compiled = torch.compile(model, mode="reduce-overhead")

    for seq_len in seq_lens:
        ids = _make_input(tokenizer, seq_len, device)

        eager_stats = _bench(model, ids, device=device, warmup=warmup, runs=runs)
        compile_stats = _bench(
            compiled, ids, device=device, warmup=warmup, runs=runs
        )

        ratio = eager_stats["mean_ms"] / compile_stats["mean_ms"]
        timings[str(seq_len)] = {
            "eager_mean_ms": eager_stats["mean_ms"],
            "eager_median_ms": eager_stats["median_ms"],
            "compile_mean_ms": compile_stats["mean_ms"],
            "compile_median_ms": compile_stats["median_ms"],
            "ratio_vs_eager": ratio,
            "eager_stdev_ms": eager_stats["stdev_ms"],
            "compile_stdev_ms": compile_stats["stdev_ms"],
            "g8_pass": ratio >= 0.95,
        }
    return timings


# --------------------------------------------------------------------------- #
# Diagnosis writer
# --------------------------------------------------------------------------- #


def write_diagnosis_md(
    output_dir: Path,
    explain_summary: dict[str, Any],
    compile_metrics: dict[str, Any],
    timings: dict[str, Any],
) -> None:
    """Synthesize ranked root causes from collected evidence."""
    lines: list[str] = [
        "# D7-E1.1 — GPT-2 torch.compile Regression Diagnosis",
        "",
        f"**Captured:** {_utc_iso()}  ",
        f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  ",
        f"**Torch:** {torch.__version__}",
        "",
        "## Timing Summary (eager vs torch.compile, mode='reduce-overhead')",
        "",
        "| seq_len | eager mean (ms) | compile mean (ms) | ratio | G8[4] ≥0.95 |",
        "|:-------:|:---------------:|:-----------------:|:-----:|:-----------:|",
    ]
    for seq_len, t in timings.items():
        lines.append(
            f"| {seq_len} "
            f"| {t['eager_mean_ms']:.2f} "
            f"| {t['compile_mean_ms']:.2f} "
            f"| {t['ratio_vs_eager']:.3f} "
            f"| {'✅' if t['g8_pass'] else '❌'} |"
        )

    lines += [
        "",
        "## Dynamo Graph Surface",
        "",
        f"- Graph count: **{explain_summary.get('graph_count', 'n/a')}**",
        f"- Graph-break count: **{explain_summary.get('graph_break_count', 'n/a')}**",
        f"- Captured op count: **{explain_summary.get('op_count', 'n/a')}**",
        f"- Distinct break reasons: **{len(explain_summary.get('break_reasons', []))}**",
        "",
    ]
    if explain_summary.get("break_reasons"):
        lines.append("### Top break reasons (first 10)")
        lines.append("")
        for i, r in enumerate(explain_summary["break_reasons"][:10], 1):
            lines.append(f"{i}. `{r['reason']}`")
    else:
        lines.append("(no graph breaks reported)")

    lines += [
        "",
        "## Compile Cost",
        "",
        f"- First-call wall time (compile + run): **{compile_metrics['first_call_s']:.3f}s**",
        f"- Second-call wall time (run only): **{compile_metrics['second_call_s']:.3f}s**",
        f"- Implied compile overhead: **{compile_metrics['compile_overhead_s_estimate']:.3f}s**",
        f"- Captured calls: {compile_metrics.get('captured_calls', 'n/a')}",
        "",
        "## Ranked Root Causes (heuristic)",
        "",
    ]

    # Rank causes from collected evidence.
    causes: list[tuple[int, str]] = []
    gbc = explain_summary.get("graph_break_count") or 0
    if gbc and gbc > 2:
        causes.append((
            100 + gbc,
            f"**Excessive graph breaks ({gbc})** — every break forces eager fallback "
            f"between fragments, adding dispatcher tax. Target in D7-E1.2.",
        ))
    elif gbc:
        causes.append((
            30,
            f"Few graph breaks ({gbc}) — likely a minor contributor; still worth "
            f"closing in D7-E1.2 for steady-state purity.",
        ))
    else:
        causes.append((
            10,
            "No graph breaks detected — regression is *not* from fallback; look "
            "to Inductor codegen quality / CUDA-Graph capture (D7-E1.3).",
        ))

    if compile_metrics["compile_overhead_s_estimate"] > 1.0:
        causes.append((
            50,
            f"Large compile overhead "
            f"(~{compile_metrics['compile_overhead_s_estimate']:.1f}s on first call) "
            f"contaminates short benches; ensure warmup ≥10 and exclude first call.",
        ))

    # Check whether ratio is consistent across seq lens (indicates fixed
    # overhead) or grows worse with longer seq (indicates kernel codegen gap).
    ratios = [t["ratio_vs_eager"] for t in timings.values()]
    if ratios and max(ratios) - min(ratios) > 0.10:
        causes.append((
            40,
            f"Ratio varies significantly across seq_len (min={min(ratios):.3f}, "
            f"max={max(ratios):.3f}) — suggests per-token dispatch tax dominates "
            f"at short seqs. CUDA Graph capture (D7-E1.3) should help.",
        ))
    elif ratios:
        causes.append((
            20,
            f"Ratio roughly constant across seq_len (~{statistics.fmean(ratios):.3f}) "
            "— points at a structural codegen gap, not per-step overhead. "
            "Inductor inspection (D7-E1.4 bridge MVP rationale) is next.",
        ))

    for score, msg in sorted(causes, key=lambda x: -x[0]):
        lines.append(f"- (score={score}) {msg}")

    lines += [
        "",
        "## Reproduction",
        "",
        "```",
        "cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate",
        "python -m benchmarks.scripts.diagnose_gpt2_torch_compile \\",
        "  --seq-lens 128,256,512 --warmup 10 --runs 20",
        "```",
        "",
        "## Artifacts",
        "",
        "- `dynamo_explain.txt` — full `torch._dynamo.explain()` output",
        "- `dynamo_explain_summary.json` — structured break-reason summary",
        "- `compile_metrics.json` — first-call / second-call / cache stats",
        "- `timings.json` — per seq_len eager vs compile timings",
        "",
    ]

    (output_dir / "diagnosis.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="D7-E1.1 diagnostic")
    parser.add_argument("--seq-lens", default="128,256,512")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Where to write artifacts (created if missing).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[FATAL] CUDA requested but unavailable.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        args.dtype
    ]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]

    print(f"[diagnose] output_dir={output_dir}")
    print(f"[diagnose] device={args.device} dtype={args.dtype} seq_lens={seq_lens}")

    model, tokenizer = _load_model(args.device, torch_dtype)
    probe_ids = _make_input(tokenizer, seq_lens[0], args.device)

    print("[diagnose] step 1/3 — torch._dynamo.explain ...")
    explain_summary = diagnose_dynamo_explain(model, probe_ids, output_dir)
    print(
        f"           graphs={explain_summary.get('graph_count')} "
        f"breaks={explain_summary.get('graph_break_count')}"
    )

    print("[diagnose] step 2/3 — compile-cost probe ...")
    compile_metrics = measure_compile_metrics(model, probe_ids)
    (output_dir / "compile_metrics.json").write_text(
        json.dumps(compile_metrics, indent=2, default=str) + "\n"
    )
    print(
        f"           first_call={compile_metrics['first_call_s']:.3f}s "
        f"second_call={compile_metrics['second_call_s']:.3f}s"
    )

    print(f"[diagnose] step 3/3 — timings (warmup={args.warmup} runs={args.runs}) ...")
    timings = measure_timings(
        model,
        tokenizer,
        seq_lens,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
    )
    (output_dir / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    for seq_len, t in timings.items():
        flag = "✅" if t["g8_pass"] else "❌"
        print(
            f"           seq_len={seq_len:>4}  "
            f"eager={t['eager_mean_ms']:.2f}ms  "
            f"compile={t['compile_mean_ms']:.2f}ms  "
            f"ratio={t['ratio_vs_eager']:.3f} {flag}"
        )

    write_diagnosis_md(output_dir, explain_summary, compile_metrics, timings)
    print(f"[diagnose] diagnosis.md written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
