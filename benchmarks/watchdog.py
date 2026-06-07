"""Per-measurement watchdog for bench_l1 / bench_l2.

Background
----------
``bench_l1.run_op`` runs ``bench_fn(fn, warmup=200, reps=500)`` inside a
``for runner in ...`` loop. A pathological op/shape (e.g. GQA at
``llama3-8b-8k`` taking 859 s for one Arke measurement, or a FlagGems
post-measurement that silently hangs for 11 h) can stall the entire run with
no external visibility and no upper bound.

This module provides a simple POSIX watchdog using ``signal.SIGALRM`` on the
main thread, with a graceful thread-based fallback for non-main-thread
callers. It is intentionally minimal — we only need:

* a hard upper bound per measurement, and
* a hard upper bound on post-measurement cleanup (``merge_perf_all`` etc.).

Both budgets are configurable via CLI flags and environment variables.

Usage
-----

    from benchmarks.watchdog import with_watchdog, WatchdogTimeout

    try:
        result = with_watchdog(300, slow_fn, args, kwargs)
    except WatchdogTimeout as exc:
        # ``result`` row gets ``status="timeout"``;
        # caller logs ``exc.elapsed_s`` for the audit trail.
        ...

The function always runs the wrapped call **synchronously** on the caller's
thread; the watchdog only delivers a timeout signal (or raises via the
thread-based fallback). No daemon threads, no Pool, no asyncio. Keep it
boring — the bench loop is already complex enough.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# Env-var defaults — CLI flags in bench_l1 / bench_l2 override these. Set to
# 0 (or any non-positive number) to disable the watchdog entirely.
DEFAULT_PER_MEASUREMENT_TIMEOUT_S = int(
    os.environ.get("ARKE_BENCH_PER_MEASUREMENT_TIMEOUT", "900")
)
DEFAULT_POST_MEASUREMENT_TIMEOUT_S = int(
    os.environ.get("ARKE_BENCH_POST_MEASUREMENT_TIMEOUT", "300")
)


class WatchdogTimeout(Exception):
    """Raised when a wrapped call exceeds its timeout budget."""

    def __init__(self, label: str, timeout_s: float, elapsed_s: float):
        super().__init__(
            f"watchdog timeout: {label} exceeded {timeout_s:.1f}s "
            f"(actual={elapsed_s:.1f}s)"
        )
        self.label = label
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


@contextmanager
def watchdog(timeout_s: float, label: str = "watchdog") -> Iterator[None]:
    """Context-manager watchdog. Raises ``WatchdogTimeout`` if the body
    exceeds ``timeout_s`` seconds.

    On POSIX main thread: uses ``signal.SIGALRM`` (the body is interrupted
    promptly with a ``WatchdogTimeout`` raised from the signal handler).

    Off main thread: falls back to a polling daemon thread that sets a flag;
    we re-check on exit, so the body is **not** interrupted mid-call, but the
    elapsed time is still surfaced. This is fine for our use — the bench
    main loop is always on the main thread, so SIGALRM is the hot path.

    ``timeout_s <= 0`` disables the watchdog (the body runs without any
    deadline check). This is the documented kill-switch for ``--no-watchdog``
    runs.
    """
    if timeout_s <= 0:
        yield
        return

    start = time.monotonic()

    if _is_main_thread() and hasattr(signal, "SIGALRM"):
        # POSIX hard-stop path.
        def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            elapsed = time.monotonic() - start
            raise WatchdogTimeout(label, timeout_s, elapsed)

        prev = signal.signal(signal.SIGALRM, _handler)
        # Use float alarm via setitimer for sub-second accuracy when wanted.
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
    else:
        # Off-main-thread fallback: best-effort post-hoc check.
        # We can't safely interrupt arbitrary Python code from another thread
        # without ``ctypes.pythonapi.PyThreadState_SetAsyncExc``, which can
        # corrupt C-extension state (notably CUDA). Defer the raise to after
        # the body returns; the operator still sees a timeout entry, even
        # though it could not be killed in-flight.
        logger.warning(
            "watchdog(%s) on non-main thread — using post-hoc check (cannot interrupt)",
            label,
        )
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise WatchdogTimeout(label, timeout_s, elapsed)


def with_watchdog(
    timeout_s: float,
    fn: Callable[..., Any],
    *args: Any,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` under a watchdog. Convenience wrapper
    around :func:`watchdog`.
    """
    lbl = label or getattr(fn, "__name__", "anonymous")
    with watchdog(timeout_s, lbl):
        return fn(*args, **kwargs)
