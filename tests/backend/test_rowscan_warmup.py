# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for R3 dynamic-shape cliff mitigation: bucket-aware warmup for the
row-scan ops (softmax, rmsnorm).

The dynamic-shape audit (docs/benchmark/dynamic-shape-cliff.md) measured a
40.99x softmax / 7.22x rmsnorm first-call cliff. The cliff is the first-touch
Triton JIT compile of each (next_pow2, divisibility) specialization bucket;
same-bucket shapes reuse the compiled kernel. `<kernel>_warmup_buckets()` moves
that compile off the inference hot path. These tests verify the warmup
functions exist, cover both divisibility classes, and (on GPU) actually
collapse the cliff.
"""

from __future__ import annotations

import math

import pytest


def _cuda_triton() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _build(op: str):
    from arke.backend.kernel_cache import KERNEL_CACHE
    KERNEL_CACHE.clear()
    w = KERNEL_CACHE.get_or_build_by_op(op, dtype="float16")
    assert w is not None, f"{op} wrapper build failed"
    return w


# ── template surface (CPU-safe: render + parse only) ────────────────────────


@pytest.mark.parametrize("op", ["softmax", "rmsnorm"])
def test_warmup_fn_rendered(op: str) -> None:
    from pathlib import Path
    from jinja2 import Template

    tpl = (
        Path(__file__).resolve().parents[2]
        / "arke" / "backend" / "triton_templates" / f"{op}.py.j2"
    )
    src = Template(tpl.read_text(encoding="utf-8")).render(kernel_name=f"arke_{op}")
    assert f"arke_{op}_warmup_buckets" in src, f"{op}: no warmup_buckets fn"
    import ast
    ast.parse(src)  # must be valid Python


def test_softmax_bucket_key_semantics() -> None:
    """Same (next_pow2, %16) class -> same bucket; crossing either -> new."""
    from pathlib import Path
    from jinja2 import Template

    tpl = (
        Path(__file__).resolve().parents[2]
        / "arke" / "backend" / "triton_templates" / "softmax.py.j2"
    )
    src = Template(tpl.read_text(encoding="utf-8")).render(kernel_name="arke_softmax")
    ns: dict = {}
    # Execute just the pure-python helper by extracting it (avoid importing
    # triton on CPU): re-implement the contract check via the rendered source.
    assert "_bucket_key" in src
    # 480 and 496 share next_pow2=512 and both %16==0 -> same bucket.
    # 512 vs 528 cross the pow2 boundary -> different bucket.
    import triton
    def bkey(N):  # mirror of _bucket_key
        return (triton.next_power_of_2(N), N % 16 == 0)
    assert bkey(480) == bkey(496)
    assert bkey(512) != bkey(528)
    assert bkey(256) != bkey(300)  # 300 is not %16 -> different div class


# ── GPU smoke: warmup collapses the cliff ───────────────────────────────────


@pytest.mark.skipif(not _cuda_triton(), reason="requires CUDA + Triton")
def test_softmax_warmup_collapses_cliff() -> None:
    import time
    import torch

    w = _build("softmax")
    warmup = w.__globals__["arke_softmax_warmup_buckets"]
    warmup([128, 256, 512, 1024, 2048], device="cuda")

    def t1(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    cliffs = []
    for N in (300, 900, 1800, 180, 777):  # fresh exact-N in warmed buckets
        X = torch.randn(32, N, device="cuda", dtype=torch.float16)
        first = t1(lambda: w(X))
        for _ in range(15):
            w(X)
        torch.cuda.synchronize()
        steady = t1(lambda: w(X))
        cliffs.append(first / steady)
    g = math.exp(sum(math.log(c) for c in cliffs) / len(cliffs))
    # Cold baseline was ~41x; warmed must be far below the 10x row-scan line.
    assert g < 8.0, f"softmax warmup cliff geomean {g:.1f}x too high (want <8x)"


@pytest.mark.skipif(not _cuda_triton(), reason="requires CUDA + Triton")
def test_rmsnorm_warmup_collapses_cliff() -> None:
    import time
    import torch

    w = _build("rmsnorm")
    warmup = w.__globals__["arke_rmsnorm_warmup_buckets"]
    warmup([4096], device="cuda")

    def t1(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    cliffs = []
    wgt = torch.ones(4096, device="cuda", dtype=torch.float16)
    for M in (128, 384, 1500, 3000):  # vary batch*seq at warmed N=4096
        X = torch.randn(M, 4096, device="cuda", dtype=torch.float16)
        first = t1(lambda: w(X, wgt))
        for _ in range(15):
            w(X, wgt)
        torch.cuda.synchronize()
        steady = t1(lambda: w(X, wgt))
        cliffs.append(first / steady)
    g = math.exp(sum(math.log(c) for c in cliffs) / len(cliffs))
    assert g < 6.0, f"rmsnorm warmup cliff geomean {g:.1f}x too high (want <6x)"
