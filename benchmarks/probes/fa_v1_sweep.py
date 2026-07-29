# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""K-ATT FA-v1: offline tile/pipeline sweep for the Triton flash-attention
template (docs/kestrel/k-att-plan.md, FA-v1).

Drives the PRODUCTION rendered wrapper via its ``_cfg_override`` seam across
a config grid of (BLOCK_N, BLOCK_S, num_warps, num_stages), on the tier-1
representative shapes, at real fp16, and reports per-(shape, config) medians
plus the per-bucket winner vs the current default (64/64/4w/2s).

Every config is correctness-checked against torch SDPA (max_abs_diff <=
5e-3) BEFORE being timed; incorrect configs are recorded and excluded — a
fast-but-wrong config must never be distilled into ``_fa_cfg``.

Usage:
    source ~/.venvs/arke/bin/activate
    python -m benchmarks.probes.fa_v1_sweep [--quick]

Output:
    benchmarks/probes/results/fa_v1_sweep_<date>.csv
    stdout summary table (per shape: default vs best config + flash-attn ref)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "benchmarks" / "probes" / "results"

# Representative shapes: cover D=64 (gpt2) and D=128 (llama/ds) classes,
# short and long S, batch>1. Subset of the benchmark tier — sweep cost must
# stay tractable on the 6GB card (ds-16k+ excluded: OOM risk under sweep).
SHAPES = [
    # tag, B, H, S, D
    ("gpt2-sm-512", 1, 12, 512, 64),
    ("gpt2-sm-1k", 1, 12, 1024, 64),
    ("gpt2-md-2k", 1, 16, 2048, 64),
    ("llama2-7b-512", 1, 32, 512, 128),
    ("llama2-7b-2k", 1, 32, 2048, 128),
    ("llama2-7b-4k", 1, 32, 4096, 128),
    ("llama2-7b-batch", 4, 32, 512, 128),
    ("ds-v2-2k", 1, 128, 2048, 128),
]

# Config grid. BLOCK_D is fixed by D. tl.dot needs both dims >= 16.
GRID = {
    "BLOCK_N": [32, 64, 128],
    "BLOCK_S": [32, 64, 128],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}

DEFAULT_CFG = {"BLOCK_N": 64, "BLOCK_S": 64, "num_warps": 4, "num_stages": 2}
CORRECTNESS_TOL = 5e-3


def _median_us(fn, reps: int = 50) -> float:
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e6)
    xs.sort()
    return xs[len(xs) // 2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="3 shapes x reduced grid (smoke)")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable", file=sys.stderr)
        return 1

    from arke.backend.kernel_cache import KERNEL_CACHE

    wrapper = KERNEL_CACHE.get_or_build_by_op("flash_attention", dtype="float16")
    assert wrapper is not None, "flash_attention wrapper build failed"

    shapes = SHAPES[:3] if args.quick else SHAPES
    grid_items = [
        dict(zip(GRID.keys(), vals))
        for vals in itertools.product(*GRID.values())
    ]
    if args.quick:
        grid_items = grid_items[::4]

    rows: list[dict] = []
    print(f"FA-v1 sweep: {len(shapes)} shapes x {len(grid_items)} configs "
          f"(+default), fp16, reps={args.reps}")

    for tag, B, H, S, D in shapes:
        try:
            q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{tag}: OOM building inputs, skip")
            continue

        # flash-attn reference latency (the golden denominator)
        flash_us = float("nan")
        try:
            from flash_attn import flash_attn_func
            qf = q.transpose(1, 2).contiguous()
            kf = k.transpose(1, 2).contiguous()
            vf = v.transpose(1, 2).contiguous()
            flash_us = _median_us(
                lambda: flash_attn_func(qf, kf, vf, causal=True), args.reps)
            del qf, kf, vf
        except Exception as exc:
            print(f"{tag}: flash-attn ref unavailable: {exc}")

        # Per-shape clock/pipe warmup so the FIRST timed config doesn't absorb
        # GPU clock spin-up + one-time Triton init (the dynamic-shape cliff).
        # Without this the default (timed first) reads cold and fabricates a
        # bogus speedup for whichever config is timed warm afterwards.
        try:
            for _ in range(20):
                wrapper(q, k, v, _cfg_override=DEFAULT_CFG)
            torch.cuda.synchronize()
        except Exception:
            pass

        results_for_shape: list[tuple[float, dict, str]] = []
        for cfg in [DEFAULT_CFG] + grid_items:
            label = (f"N{cfg['BLOCK_N']}_S{cfg['BLOCK_S']}"
                     f"_w{cfg['num_warps']}_st{cfg['num_stages']}")
            status = "ok"
            us = float("nan")
            err = float("nan")
            try:
                out = wrapper(q, k, v, _cfg_override=cfg)
                err = (out.float() - ref.float()).abs().max().item()
                if not math.isfinite(err) or err > CORRECTNESS_TOL:
                    status = "incorrect"
                else:
                    us = _median_us(
                        lambda: wrapper(q, k, v, _cfg_override=cfg), args.reps)
                del out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                status = "oom"
            except Exception as exc:
                status = f"error:{str(exc)[:60]}"
            rows.append({
                "tag": tag, "B": B, "H": H, "S": S, "D": D,
                "config": label, **cfg,
                "latency_us": us, "max_abs_err": err,
                "flash_us": flash_us, "status": status,
                "is_default": cfg == DEFAULT_CFG,
            })
            if status == "ok":
                results_for_shape.append((us, cfg, label))

        del q, k, v, ref
        torch.cuda.empty_cache()

        if results_for_shape:
            results_for_shape.sort(key=lambda t: t[0])
            best_us, best_cfg, best_label = results_for_shape[0]
            default_us = next(
                (r["latency_us"] for r in rows
                 if r["tag"] == tag and r["is_default"]
                 and r["status"] == "ok"),
                float("nan"))
            ratio_vs_flash = (flash_us / best_us
                              if math.isfinite(flash_us) else float("nan"))
            print(f"{tag:>16}  default={default_us:9.1f}us  "
                  f"best={best_us:9.1f}us ({best_label})  "
                  f"speedup={default_us / best_us:5.2f}x  "
                  f"flash={flash_us:9.1f}us  best/flash={ratio_vs_flash:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"fa_v1_sweep_{date.today().isoformat()}.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
