# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""KESTREL-H3.1 probe — quantify the matmul autotune first-call cliff mitigation.

Measures the *first-call* wall-clock cost of the Arke Triton matmul kernel
across a sequence of shapes that share a bucketed cache key, then compares
to a control shape from a *different* bucket. The K-H3.1 contract is:

    Neighbouring shapes in the same bucket cost O(compile+launch) after the
    first shape in that bucket has warmed the launcher-side ``_TILE_CFG_CACHE``
    — NOT the multi-second per-config sweep the launcher runs on a cold
    bucket miss.

Design note: the mitigation is a **launcher-side cache** (identical pattern
to ``batch_matmul.py.j2``), not a keyed ``@triton.autotune``. Keeping the
tuning key off the kernel arg list preserves slim-launch on tiny shapes.
See ``arke/backend/triton_templates/matmul.py.j2`` module docstring for the
design rationale.

The probe prints a compact table and writes CSV. It does not gate CI (that's
what `tests/backend/test_matmul_bucketed_autotune.py` does — the pure-python
tests validate the bucket function and the cache-hit contract deterministically).
This probe exists so a human can *see* the cliff mitigation for themselves.

Usage:
    source ~/.venvs/arke/bin/activate
    python benchmarks/probes/autotune_first_call.py

Output:
    benchmarks/probes/results/autotune_first_call.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from jinja2 import Environment


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "arke" / "backend" / "triton_templates" / "matmul.py.j2"
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "probes" / "results"


def _render_matmul(kernel_name: str = "arke_mm_probe") -> str:
    env = Environment()
    tmpl = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return tmpl.render(
        kernel_name=kernel_name,
        output_dtype="tl.float32",
        fused_activation=None,
    )


def _load_fresh_module(kernel_name: str):
    """Load a freshly-rendered matmul module — each call gets a virgin
    autotune cache. Otherwise the module-level cache from prior calls
    contaminates first-call measurements.
    """
    source = _render_matmul(kernel_name)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{kernel_name}.py", delete=False, encoding="utf-8"
    )
    tmp.write(source)
    tmp.flush()
    tmp.close()
    mod_name = f"_arke_probe_{kernel_name}"
    spec = importlib.util.spec_from_file_location(mod_name, tmp.name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _time_call(fn, a, b) -> tuple[float, torch.Tensor]:
    """CUDA-synchronized wall-clock of one launch (seconds, output)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(a, b)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _make_inputs(M: int, N: int, K: int):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    return a, b


def probe_bucket_family(bucket: int, sizes: list[int], kernel_name: str) -> list[dict]:
    """Time first-call latency across a family of shapes that all bucket to
    the same power of two. First shape pays the autotune scan; subsequent
    shapes should NOT.
    """
    module = _load_fresh_module(kernel_name)
    matmul = getattr(module, kernel_name)

    rows: list[dict] = []
    for idx, size in enumerate(sizes):
        # Square matmul with all three dims in the same bucket.
        M = N = K = size
        a, b = _make_inputs(M, N, K)
        wall_s, _ = _time_call(matmul, a, b)
        rows.append({
            "bucket": bucket,
            "call_index": idx,
            "M": M, "N": N, "K": K,
            "wall_s": wall_s,
        })
        # Aggressive cleanup so the next iteration's first-call cost isn't
        # muddied by allocator or thermal state carryover.
        del a, b
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "autotune_first_call.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[probe] CUDA unavailable — nothing to measure.", file=sys.stderr)
        return 1

    # Bucket → member shape sizes (all bucket to `bucket` via next_pow2).
    # Pick two shapes that clearly share a bucket, plus repeat calls to see
    # the stable warm state.
    bucket_families = {
        1024: [513, 700, 900, 1024, 700],  # last repeat = stable warm sample
        2048: [1500, 2000, 1300, 2048],
        4096: [3000, 4096, 2500],
    }

    all_rows: list[dict] = []
    print(f"{'bucket':>6} {'call#':>5} {'M×N×K':>18} {'wall_s':>10}   note")
    print("-" * 60)
    for bucket, sizes in bucket_families.items():
        rows = probe_bucket_family(bucket, sizes, f"probe_mm_b{bucket}")
        all_rows.extend(rows)
        first_wall = rows[0]["wall_s"]
        for r in rows:
            note = ""
            if r["call_index"] == 0:
                note = "(first — pays autotune scan)"
            elif r["wall_s"] > first_wall * 0.5:
                note = "⚠ suspicious high cost"
            else:
                note = f"{r['wall_s']/first_wall:.3f}× first-call"
            print(
                f"{r['bucket']:>6} {r['call_index']:>5} "
                f"{r['M']:>5}×{r['N']:>4}×{r['K']:>4}   {r['wall_s']:>8.4f}   {note}"
            )
        # After bucket done, print quick within-bucket summary.
        warm_walls = [r["wall_s"] for r in rows[1:]]
        if warm_walls:
            ratio = statistics.median(warm_walls) / first_wall
            print(f"       bucket {bucket} median warm/cold = {ratio:.3f} (want ≪ 1.0)")

    # Cross-bucket sanity: first call to a shape in a fresh bucket after all
    # previous work should also be expensive — proves the cheap warm calls
    # above aren't just "everything is cached globally".
    print("-" * 60)
    print("[cross-bucket sanity] a fresh shape in an unseen bucket should be COLD:")
    cross_rows = probe_bucket_family(bucket=8192, sizes=[5000], kernel_name="probe_mm_cross")
    r = cross_rows[0]
    print(f"       fresh 5000×5000×5000 (bucket 8192): wall = {r['wall_s']:.4f}s")
    all_rows.extend(cross_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bucket", "call_index", "M", "N", "K", "wall_s"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
