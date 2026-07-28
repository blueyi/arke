# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KESTREL-H3.1: matmul autotune-key bucketing.

The Triton matmul kernel emitted by ``matmul.py.j2`` now uses
``key=["M_bucket", "N_bucket", "K_bucket"]`` where each ``*_bucket`` is
``next_pow2(*)``. Neighbouring shapes (e.g. M=513 and M=1024) share the same
autotune cache slot, so the *second* new-shape call in a bucket no longer
re-scans the 20-config ladder.

These tests cover:

  * ``_next_pow2`` semantics and edge cases (part of the generated template).
  * The rendered kernel launches without crashing on both the contiguous
    fast-path and the general strided path.
  * Autotune cache HIT after a warmup: two distinct shapes that share a
    bucket produce the same tuned config on their first cold call
    (validates the bucketing actually collapses the key domain, not just
    the argument list).

The kernel-level tests require CUDA + Triton and are skipped otherwise so
the file remains importable in CPU-only CI.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "arke" / "backend" / "triton_templates" / "matmul.py.j2"


# ── Bucketing helper: derived from the template source ────────────────────

def _load_template_source() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def _extract_next_pow2() -> Callable[[int], int]:
    """Extract the ``_next_pow2`` helper straight from the Jinja template
    into a live module namespace.

    We can't import the template as Python because of the ``{{ … }}``
    substitutions, so we scope the exec to the definition of the helper
    (a plain-Python def block with no Jinja markers).
    """
    src = _load_template_source()
    lines = src.splitlines(keepends=True)
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("def _next_pow2(")),
        None,
    )
    assert start is not None, "template must expose _next_pow2"
    # A top-level def ends at the next unindented non-empty line.
    end = start + 1
    while end < len(lines):
        stripped = lines[end].lstrip("\n")
        if stripped and not stripped[0].isspace():
            break
        end += 1
    fn_src = "".join(lines[start:end])
    ns: dict = {}
    exec(fn_src, ns)
    return ns["_next_pow2"]


# Extract once at import time so pytest parametrize can consume it without
# each test paying the re-parse cost. If the template ever loses _next_pow2
# this fails loudly at collection time.
_NEXT_POW2 = _extract_next_pow2()


def test_template_declares_launcher_side_bucket_cache() -> None:
    """Structural guard: launcher-side bucketed cache must be present.

    K-H3.1 mitigates the autotune cliff via a launcher-side cache
    (``_TILE_CFG_CACHE``) keyed by ``(next_pow2(M), next_pow2(N),
    next_pow2(K))``. This avoids the launch-arg regression that a keyed
    ``@triton.autotune`` would cause on tiny shapes (slim-launch preserved).
    See ``benchmarks/probes/autotune_first_call.py`` for the cold-vs-warm
    contract this guards.
    """
    src = _load_template_source()
    # The module-level cache dict must exist.
    assert "_TILE_CFG_CACHE" in src, "K-H3.1: launcher-side cache dict missing"
    # Launcher must consult it via next_pow2 for all three dims.
    assert "_next_pow2(M)" in src
    assert "_next_pow2(N)" in src
    assert "_next_pow2(K)" in src
    # Explicit anti-regression: no @triton.autotune DECORATOR must appear
    # (its required kernel-arg keying inflates launch overhead on small
    # shapes — see 2026-07-28 measurements in kestrel-backlog.md). Match
    # only the decorator syntax (`@triton.autotune(`), not descriptive
    # prose in the module docstring.
    assert not re.search(r"^@triton\.autotune\(", src, re.MULTILINE), (
        "K-H3.1: @triton.autotune decorator re-introduced — this defeats "
        "slim-launch on tiny shapes (2026-07-28 measurements)."
    )


class TestNextPow2:
    """Semantics of ``_next_pow2`` as embedded in the template."""

    fn = staticmethod(_NEXT_POW2)

    @pytest.mark.parametrize("x,expected", [
        (0, 1), (1, 1),                # collapse edge
        (2, 2), (4, 4), (8, 8), (16, 16),   # exact powers-of-two ≤ 16 stay exact
        (17, 32), (31, 32), (32, 32),
        (33, 64), (63, 64), (64, 64),
        (65, 128), (127, 128), (128, 128),
        (129, 256),
        (257, 512), (511, 512), (512, 512),
        (513, 1024), (1023, 1024), (1024, 1024),
        (4095, 4096), (4096, 4096),
    ])
    def test_bucket_values(self, x: int, expected: int) -> None:
        assert self.fn(x) == expected

    def test_neighbouring_shapes_share_bucket(self) -> None:
        """Concrete K-H3.1 contract: M=513 and M=1024 must share the bucket."""
        assert self.fn(513) == self.fn(1024) == 1024

    def test_small_dims_preserve_exact_key(self) -> None:
        """Tiny dims (≤16) stay exact to keep small-shape unit tests stable."""
        for x in range(1, 17):
            assert self.fn(x) == max(x, 1)

    def test_monotone_non_decreasing(self) -> None:
        prev = self.fn(1)
        for x in range(2, 8193):
            cur = self.fn(x)
            assert cur >= prev
            prev = cur


# ── Kernel-level: render template + compile + verify autotune cache ────────

# Rendering the Jinja template needs the same helper the codegen backend uses.

def _render_matmul(kernel_name: str = "arke_mm_test") -> str:
    """Render matmul.py.j2 to Python source, mimicking the codegen backend."""
    from jinja2 import Environment
    env = Environment()
    tmpl = env.from_string(_load_template_source())
    return tmpl.render(
        kernel_name=kernel_name,
        output_dtype="tl.float32",
        fused_activation=None,
    )


def _load_rendered_module(source: str, mod_name: str):
    """Load rendered Python source as a fresh module.

    Writes to a real temp .py file so ``inspect.getsourcelines`` — which
    ``triton.jit`` relies on to read kernel bodies — can find the source.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{mod_name}.py", delete=False, encoding="utf-8"
    )
    tmp.write(source)
    tmp.flush()
    tmp.close()
    spec = importlib.util.spec_from_file_location(mod_name, tmp.name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_HAS_CUDA_TRITON = False
try:
    import torch  # noqa: F401
    import triton  # noqa: F401
    _HAS_CUDA_TRITON = torch.cuda.is_available()
except Exception:  # torch or triton missing
    pass


@pytest.mark.skipif(not _HAS_CUDA_TRITON, reason="requires CUDA + Triton")
def test_bucketed_kernel_runs_contiguous_and_strided() -> None:
    """Sanity: rendered kernel launches on both paths and matches torch."""
    import torch
    src = _render_matmul("arke_mm_bucketed_smoke")
    mod = _load_rendered_module(src, "arke_mm_bucketed_smoke_mod")
    matmul = mod.arke_mm_bucketed_smoke

    a = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    b = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    out = matmul(a, b)
    ref = a @ b
    # fp32 matmul on Ampere with TF32 disabled still accumulates a few ULP
    # of drift over K=32 — 1% tol is well within numerical parity while
    # catching real correctness regressions.
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    # Non-contiguous path: transpose to break the contiguous fast-path.
    a2 = torch.randn(128, 64, device="cuda", dtype=torch.float32).t()  # (64, 128), non-contig
    b2 = torch.randn(128, 32, device="cuda", dtype=torch.float32)
    out2 = matmul(a2, b2)
    ref2 = a2 @ b2
    torch.testing.assert_close(out2, ref2, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _HAS_CUDA_TRITON, reason="requires CUDA + Triton")
def test_bucket_collapses_launcher_cache_key_domain() -> None:
    """Two neighbouring shapes in the same bucket share the picked config.

    With the K-H3.1 launcher-side cache (``_TILE_CFG_CACHE``), the second
    shape in a bucket must NOT trigger a fresh sweep — it looks up the
    same tuple key computed from ``(next_pow2(M), next_pow2(N),
    next_pow2(K))``. Cross-bucket shapes DO trigger a new entry.
    """
    import torch
    src = _render_matmul("arke_mm_bucketed_cache")
    mod = _load_rendered_module(src, "arke_mm_bucketed_cache_mod")
    matmul = mod.arke_mm_bucketed_cache
    cache = mod._TILE_CFG_CACHE

    cache.clear()

    # Warm-up shape #1 (M=513, N=513, K=513) → all bucket to 1024.
    a1 = torch.randn(513, 513, device="cuda", dtype=torch.float32).contiguous()
    b1 = torch.randn(513, 513, device="cuda", dtype=torch.float32).contiguous()
    _ = matmul(a1, b1)
    torch.cuda.synchronize()
    assert len(cache) == 1, f"expected 1 cache entry after first shape, got {len(cache)}"
    key_1 = next(iter(cache))
    assert key_1 == (1024, 1024, 1024)

    # Neighbouring shape (M=700, N=800, K=900) — all still bucket to 1024.
    a2 = torch.randn(700, 900, device="cuda", dtype=torch.float32).contiguous()
    b2 = torch.randn(900, 800, device="cuda", dtype=torch.float32).contiguous()
    _ = matmul(a2, b2)
    torch.cuda.synchronize()
    assert len(cache) == 1, (
        f"launcher cache grew from 1 → {len(cache)} for a neighbouring shape — "
        "bucketing is not effective."
    )

    # Cross-bucket sanity: M=2000 buckets to 2048 → NEW cache entry.
    a3 = torch.randn(2000, 513, device="cuda", dtype=torch.float32).contiguous()
    b3 = torch.randn(513, 513, device="cuda", dtype=torch.float32).contiguous()
    _ = matmul(a3, b3)
    torch.cuda.synchronize()
    assert len(cache) == 2, (
        f"cross-bucket shape did not add a new entry (cache size={len(cache)})"
    )
    assert (2048, 1024, 1024) in cache
