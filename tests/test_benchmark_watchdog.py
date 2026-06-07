"""Tests for ``benchmarks.watchdog`` and ``ProgressTracker.phase``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmarks import progress as p
from benchmarks.watchdog import (
    WatchdogTimeout,
    watchdog,
    with_watchdog,
)


# --------------------------------------------------------------------------- #
# Watchdog primitives
# --------------------------------------------------------------------------- #


def test_watchdog_passes_through_on_no_timeout():
    """A fast body must return normally with no exception."""
    with watchdog(5.0, label="fast"):
        result = 1 + 1
    assert result == 2


def test_watchdog_raises_on_timeout():
    """A body that runs longer than ``timeout_s`` must raise
    :class:`WatchdogTimeout` with elapsed > timeout."""
    with pytest.raises(WatchdogTimeout) as excinfo:
        with watchdog(0.05, label="slow"):
            # Use time.sleep so SIGALRM can preempt cleanly (sleep is a
            # syscall, EINTR-friendly).
            time.sleep(0.5)
    assert excinfo.value.label == "slow"
    assert excinfo.value.elapsed_s >= 0.05
    assert excinfo.value.timeout_s == pytest.approx(0.05)


def test_watchdog_disabled_when_timeout_nonpositive():
    """``timeout_s <= 0`` must disable the watchdog entirely (the body runs
    without any deadline). This is the documented kill-switch for
    ``--no-watchdog``-style runs."""
    with watchdog(0, label="disabled"):
        time.sleep(0.1)
    with watchdog(-1, label="negative"):
        time.sleep(0.1)


def test_with_watchdog_returns_function_result():
    def add(a: int, b: int) -> int:
        return a + b

    result = with_watchdog(5.0, add, 3, 4)
    assert result == 7


def test_with_watchdog_uses_function_name_as_default_label():
    def expensive_op() -> None:
        time.sleep(0.5)

    with pytest.raises(WatchdogTimeout) as excinfo:
        with_watchdog(0.05, expensive_op)
    assert excinfo.value.label == "expensive_op"


def test_with_watchdog_respects_explicit_label():
    def f() -> None:
        time.sleep(0.5)

    with pytest.raises(WatchdogTimeout) as excinfo:
        with_watchdog(0.05, f, label="custom_label")
    assert excinfo.value.label == "custom_label"


# --------------------------------------------------------------------------- #
# ProgressTracker.phase context manager
# --------------------------------------------------------------------------- #


def test_phase_context_emits_start_and_end(tmp_path: Path):
    tr = p.ProgressTracker(base_dir=tmp_path, layer="l1", config_fingerprint="fp")
    with tr.phase("per_op_measurement", op="matmul"):
        pass

    lines = (tmp_path / p.PROGRESS_LOG_NAME).read_text().splitlines()
    events = [json.loads(l) for l in lines]
    assert [e["event"] for e in events] == ["phase_start", "phase_end"]
    assert all(e["phase"] == "per_op_measurement" for e in events)
    assert all(e["op"] == "matmul" for e in events)
    assert "elapsed_s" in events[1]
    assert events[1]["elapsed_s"] >= 0


def test_phase_end_fires_even_when_body_raises(tmp_path: Path):
    """If the wrapped body raises, ``phase_end`` must STILL fire so a hang
    or crash leaves a tail event in ``progress.jsonl``."""
    tr = p.ProgressTracker(base_dir=tmp_path, layer="l1", config_fingerprint="fp")
    with pytest.raises(RuntimeError, match="boom"):
        with tr.phase("merge_perf_all"):
            raise RuntimeError("boom")

    lines = (tmp_path / p.PROGRESS_LOG_NAME).read_text().splitlines()
    events = [json.loads(l) for l in lines]
    assert events[-1]["event"] == "phase_end"
    assert events[-1]["phase"] == "merge_perf_all"


def test_heartbeat_emits_event(tmp_path: Path):
    tr = p.ProgressTracker(base_dir=tmp_path, layer="l1", config_fingerprint="fp")
    tr.heartbeat(op="gqa", shape_tag="llama3-8b-8k", elapsed_s=120.5)

    lines = (tmp_path / p.PROGRESS_LOG_NAME).read_text().splitlines()
    events = [json.loads(l) for l in lines]
    assert len(events) == 1
    assert events[0]["event"] == "heartbeat"
    assert events[0]["op"] == "gqa"
    assert events[0]["elapsed_s"] == 120.5
