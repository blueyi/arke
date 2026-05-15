# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — KernelCache (S7 Track C5).

Process-global LRU cache for Triton wrappers produced by
``arke.backend.triton_codegen.generate_kernel``.

Why this exists
---------------
Generating a kernel costs ~5–10 ms of Jinja render + Python ``exec`` +
``@triton.jit`` registration (the *first* @triton.jit call costs ~300 ms
on cold start, but every subsequent call is cheap). For a benchmark
sweep that lowers thousands of IRGraph nodes — most of which share the
same op + template + dtype — that adds up. More importantly, every
distinct wrapper *object* triggers a fresh Triton autotune pass the
first time it sees a new shape; sharing one wrapper across every IRNode
that wants the same kernel collapses that cost.

Triton itself handles per-shape kernel specialization internally, so we
do **not** key the cache on shape — same wrapper, called with new
shapes, just lets Triton's own JIT cache do its job. Our cache key is:

    (op_name, template_name, dtype)

Plus an explicit ``kernel_name`` so callers can request a uniquely-named
wrapper when they really need debug-distinguishable kernels (rare).

Thread-safety
-------------
A ``threading.Lock`` guards the cache because Triton's ``@triton.jit``
registration is *not* re-entrant under threading. In practice Arke is
single-threaded today; the lock is a cheap insurance policy for the
``arke.agent`` async paths that will land in Stage 8.

Sizing
------
Default ``maxsize=256`` is comfortable for the 45-op × 4-dtype × ~5
template-variant space (≈180 distinct keys at full saturation).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Callable

from arke.backend.triton_codegen import generate_kernel
from arke.ir.ops.registry import REGISTRY
from arke.ir.ops.schema import TemplateHint

logger = logging.getLogger(__name__)


class KernelCache:
    """LRU cache for compiled Triton wrappers.

    Usage::

        cache = KernelCache(maxsize=256)
        wrapper = cache.get_or_build("matmul", template_hint, dtype="float16")
        y = wrapper(A, B)
    """

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError(f"KernelCache maxsize must be positive, got {maxsize}")
        self._maxsize = maxsize
        self._cache: OrderedDict[tuple, Callable[..., Any]] = OrderedDict()
        self._lock = threading.Lock()
        # Stats (read-only from outside; rebuild assertions in tests)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._build_failures = 0

    # ── public api ──────────────────────────────────────────────

    def get_or_build(
        self,
        op_name: str,
        template_hint: TemplateHint,
        *,
        dtype: str = "float16",
        kernel_name: str | None = None,
    ) -> Callable[..., Any]:
        """Return a cached or freshly-built Triton wrapper callable.

        Args:
            op_name: operator name (e.g. ``"matmul"``)
            template_hint: from ``OpSchema.template_hint``
            dtype: input dtype hint (affects codegen output_dtype, accum)
            kernel_name: optional unique kernel name; when set, the
                wrapper is keyed on it (forces cache miss for that name).

        Returns:
            Compiled wrapper callable: positional ``torch.Tensor`` inputs,
            tensor (or tuple) output.

        Raises:
            Whatever ``generate_kernel`` raises on the cold path. Build
            failures are *not* cached — the next call retries.
        """
        key = self._make_key(op_name, template_hint, dtype, kernel_name)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                # LRU touch
                self._cache.move_to_end(key)
                self._hits += 1
                return cached
            self._misses += 1

        # Build outside the lock so concurrent cold misses for *different*
        # keys can run in parallel (Triton itself has internal serialization).
        try:
            wrapper = generate_kernel(
                op_name, template_hint, dtype=dtype, kernel_name=kernel_name,
            )
        except Exception:
            with self._lock:
                self._build_failures += 1
            raise

        with self._lock:
            # Re-check under lock: another thread may have built the same key
            # while we were generating. If so, keep the existing one (avoids
            # holding two compiled wrappers for the same logical kernel).
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing

            self._cache[key] = wrapper
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # evict oldest
                self._evictions += 1
            return wrapper

    def get_or_build_by_op(
        self,
        op_name: str,
        *,
        dtype: str = "float16",
        kernel_name: str | None = None,
    ) -> Callable[..., Any] | None:
        """Convenience: look up template_hint via REGISTRY then call get_or_build.

        Returns None if the op has no template_hint (caller should fall
        back to the interpreter path).
        """
        try:
            schema = REGISTRY.get(op_name)
        except KeyError:
            return None
        if schema.template_hint is None:
            return None
        return self.get_or_build(
            op_name, schema.template_hint, dtype=dtype, kernel_name=kernel_name,
        )

    # ── introspection ───────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._cache

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "build_failures": self._build_failures,
            }

    def clear(self) -> None:
        """Drop every cached wrapper. Counters survive."""
        with self._lock:
            self._cache.clear()

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._build_failures = 0

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _make_key(
        op_name: str,
        template_hint: TemplateHint,
        dtype: str,
        kernel_name: str | None,
    ) -> tuple:
        """Canonical cache key.

        Includes ``extra_ctx`` items (sorted, hashable) because they
        actually parameterize codegen — e.g. ``binary_op`` /
        ``reduction_op`` / ``gate_activation`` switch the rendered
        kernel body.
        """
        extra = tuple(sorted(template_hint.extra_ctx.items())) if template_hint.extra_ctx else ()
        return (
            op_name,
            template_hint.template_name,
            template_hint.primary_op,
            extra,
            dtype,
            kernel_name,
        )


# Module-level singleton — matches the INTERPRETER convention.
KERNEL_CACHE = KernelCache(maxsize=256)
