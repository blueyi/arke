# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark progress / resume infrastructure.

Goals:

* Persist every measurement immediately so a crash never loses more than the
  in-flight test point.
* Allow ``benchmarks.bench_l1`` / ``benchmarks.bench_l2`` to be re-launched
  against the same output directory and skip already-completed work.
* Surface progress via a ``progress.jsonl`` event log + a single ``status.json``
  snapshot suitable for CLI inspection.
* Detect concurrent runs via a PID-bearing ``.lock`` file.
* Validate the run configuration fingerprint so unintended config drift forces
  the user to start a new run instead of silently mixing data.

The module is intentionally lightweight and has no torch / CUDA dependency so
it can be imported by status tooling and unit tests without side effects.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #

#: Statuses considered terminal-success — never re-run on resume.
SUCCESS_STATUSES = {"ok"}

#: Statuses that represent a known limitation already recorded — skip on resume.
#: ``oom`` and ``skipped`` come from memory preflight; ``unsupported`` /
#: ``incompatible`` come from baseline compatibility checks.
PERMANENT_FAILURE_STATUSES = {"oom", "skipped", "unsupported", "incompatible"}

#: Statuses that are retryable on resume by default — they usually represent
#: transient failures (driver hiccup, partial CUDA error, killed by OOM-killer).
RETRYABLE_FAILURE_STATUSES = {"error", "timeout"}

#: Retry policies for resume.
RETRY_POLICY_AUTO = "auto"      # skip success+permanent, retry retryable
RETRY_POLICY_NONE = "none"      # skip everything that has a row
RETRY_POLICY_ALL = "all"        # retry every non-success row
RETRY_POLICIES = (RETRY_POLICY_AUTO, RETRY_POLICY_NONE, RETRY_POLICY_ALL)


def should_skip(existing_row: dict[str, str], policy: str = RETRY_POLICY_AUTO) -> bool:
    """Return True if a recorded row should be considered already-done."""
    status = (existing_row.get("status") or "").strip().lower()
    if status in SUCCESS_STATUSES:
        return True
    if policy == RETRY_POLICY_NONE:
        return True
    if policy == RETRY_POLICY_ALL:
        return False
    # auto
    return status in PERMANENT_FAILURE_STATUSES


# --------------------------------------------------------------------------- #
# Config fingerprint
# --------------------------------------------------------------------------- #

#: Keys included in the fingerprint. Anything that materially changes which
#: test points are produced must be listed here.
FINGERPRINT_KEYS = (
    "ops",
    "shape_tags",
    "tier",
    "warmup",
    "reps",
    "phase",
    "stage",
    "track",
    "layer",
)


def compute_fingerprint(config: dict[str, Any]) -> str:
    """Stable hash over fingerprint-relevant config fields."""
    payload: dict[str, Any] = {}
    for key in FINGERPRINT_KEYS:
        if key in config:
            value = config[key]
            if isinstance(value, list):
                value = sorted(str(v) for v in value)
            payload[key] = value
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class ConfigCheck:
    """Result of comparing a stored config against the current one."""

    compatible: bool
    reason: str = ""
    stored_fingerprint: str = ""
    current_fingerprint: str = ""


def validate_config(
    base_dir: Path,
    current_config: dict[str, Any],
    *,
    force: bool = False,
) -> ConfigCheck:
    """Compare current config against the stored ``config.json`` if any."""
    config_path = base_dir / "config.json"
    current_fp = compute_fingerprint(current_config)
    if not config_path.exists():
        return ConfigCheck(
            compatible=True,
            stored_fingerprint="",
            current_fingerprint=current_fp,
        )
    try:
        stored = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if force:
            return ConfigCheck(
                compatible=True,
                reason=f"unreadable stored config (forced): {exc}",
                current_fingerprint=current_fp,
            )
        return ConfigCheck(
            compatible=False,
            reason=f"stored config unreadable: {exc}",
            current_fingerprint=current_fp,
        )
    stored_fp = compute_fingerprint(stored)
    if stored_fp == current_fp:
        return ConfigCheck(
            compatible=True,
            stored_fingerprint=stored_fp,
            current_fingerprint=current_fp,
        )
    if force:
        return ConfigCheck(
            compatible=True,
            reason="fingerprint mismatch (forced)",
            stored_fingerprint=stored_fp,
            current_fingerprint=current_fp,
        )
    return ConfigCheck(
        compatible=False,
        reason="config fingerprint changed; resume aborted (use --force-restart)",
        stored_fingerprint=stored_fp,
        current_fingerprint=current_fp,
    )


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #

LOCK_NAME = ".bench.lock"


@dataclass
class LockInfo:
    pid: int
    started_at: float
    host: str = ""
    layer: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "pid": self.pid,
                "started_at": self.started_at,
                "host": self.host,
                "layer": self.layer,
            },
            sort_keys=True,
        )

    @classmethod
    def from_path(cls, path: Path) -> "LockInfo | None":
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return cls(
            pid=int(data.get("pid", 0)),
            started_at=float(data.get("started_at", 0.0)),
            host=str(data.get("host", "")),
            layer=str(data.get("layer", "")),
        )


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def acquire_lock(base_dir: Path, layer: str = "", *, force: bool = False) -> Path:
    """Acquire a directory-scoped PID lock.

    Raises ``RuntimeError`` if another live process holds the lock and
    ``force`` is False.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / LOCK_NAME
    if path.exists():
        existing = LockInfo.from_path(path)
        if existing and _process_alive(existing.pid) and existing.pid != os.getpid():
            if not force:
                raise RuntimeError(
                    f"benchmark output dir already locked by PID {existing.pid} "
                    f"(started {existing.started_at}, host={existing.host}); "
                    "stop it or rerun with --force-restart"
                )
            logger.warning(
                "Forcing acquisition of lock held by live PID %s", existing.pid
            )
    info = LockInfo(
        pid=os.getpid(),
        started_at=time.time(),
        host=os.uname().nodename if hasattr(os, "uname") else "",
        layer=layer,
    )
    path.write_text(info.to_json())
    return path


def release_lock(base_dir: Path) -> None:
    path = base_dir / LOCK_NAME
    try:
        existing = LockInfo.from_path(path)
        if existing and existing.pid != os.getpid():
            return
        path.unlink(missing_ok=True)
    except OSError:
        pass


def lock_status(base_dir: Path) -> dict[str, Any] | None:
    """Return lock metadata + liveness, or None if no lock present."""
    path = base_dir / LOCK_NAME
    info = LockInfo.from_path(path) if path.exists() else None
    if info is None:
        return None
    return {
        "pid": info.pid,
        "started_at": info.started_at,
        "host": info.host,
        "layer": info.layer,
        "alive": _process_alive(info.pid),
    }


# --------------------------------------------------------------------------- #
# CSV append + load
# --------------------------------------------------------------------------- #


def load_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read an existing per-op CSV. Returns empty list if missing/unreadable."""
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except OSError as exc:
        logger.warning("could not read %s: %s", csv_path, exc)
        return []


def index_rows(
    rows: Iterable[dict[str, str]],
    key_fields: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, str]]:
    """Index rows by a key tuple. Last write wins on duplicates."""
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        out[key] = row
    return out


def ensure_header(csv_path: Path, fieldnames: list[str]) -> None:
    """Create the CSV with a header row if it does not exist."""
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_row(csv_path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    """Atomically append a single result row to the CSV.

    Each call flushes + fsyncs so a SIGKILL or power loss leaves the CSV in a
    well-formed state.
    """
    ensure_header(csv_path, fieldnames)
    # Use DictWriter to honour field order + skip extras.
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Progress event log
# --------------------------------------------------------------------------- #


PROGRESS_LOG_NAME = "progress.jsonl"
STATUS_SNAPSHOT_NAME = "status.json"


@dataclass
class ProgressTracker:
    """Lightweight progress recorder."""

    base_dir: Path
    layer: str
    config_fingerprint: str
    start_time: float = field(default_factory=time.time)

    def _log_path(self) -> Path:
        return self.base_dir / PROGRESS_LOG_NAME

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            "layer": self.layer,
            "fingerprint": self.config_fingerprint,
            **payload,
        }
        path = self._log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
        except OSError as exc:
            logger.debug("failed to write progress event: %s", exc)

    def snapshot(self, summary: dict[str, Any]) -> None:
        path = self.base_dir / STATUS_SNAPSHOT_NAME
        payload = {
            "ts": time.time(),
            "layer": self.layer,
            "fingerprint": self.config_fingerprint,
            "elapsed_s": time.time() - self.start_time,
            **summary,
        }
        try:
            path.write_text(json.dumps(payload, indent=2, default=str))
        except OSError as exc:
            logger.debug("failed to write status snapshot: %s", exc)


# --------------------------------------------------------------------------- #
# Output dir hygiene
# --------------------------------------------------------------------------- #


def normalize_output_root(
    raw_output: str | os.PathLike[str],
    *,
    phase: int,
    stage: int,
    track: int | str,
    layer: str,
) -> Path:
    """Return the canonical ``<root>`` such that the layer dir is
    ``<root>/phase{phase}/stage{stage}/track{track}/{layer}``.

    Strips a trailing ``phase{phase}/stage{stage}/track{track}`` (and an
    optional ``{layer}``) from ``raw_output`` so callers that already pass the
    full path do not produce nested duplicates such as
    ``track6/phase1/stage7/track6/l2/``.
    """
    suffix_track = f"track{track}"
    suffix_stage = f"stage{stage}"
    suffix_phase = f"phase{phase}"

    parts = list(Path(raw_output).parts)
    # Drop trailing layer segment if present.
    if parts and parts[-1] == layer:
        parts.pop()
    # Drop trailing trackN/stageN/phaseN repetition.
    if parts and parts[-1] == suffix_track:
        parts.pop()
        if parts and parts[-1] == suffix_stage:
            parts.pop()
            if parts and parts[-1] == suffix_phase:
                parts.pop()
    if not parts:
        return Path(".")
    return Path(*parts)


# --------------------------------------------------------------------------- #
# Plan + summary helpers
# --------------------------------------------------------------------------- #


@dataclass
class OpPlan:
    """Planned (shape, baseline) keys for one op, used for resume bookkeeping."""

    op: str
    csv_path: Path
    fieldnames: list[str]
    key_fields: tuple[str, ...]
    planned_keys: list[tuple[str, ...]] = field(default_factory=list)

    def planned_count(self) -> int:
        return len(self.planned_keys)


def summarize_csv(
    csv_path: Path,
    key_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Compute a quick status summary for one CSV file."""
    rows = load_existing_rows(csv_path)
    indexed = index_rows(rows, key_fields)
    ok = sum(1 for r in indexed.values() if (r.get("status") or "").lower() in SUCCESS_STATUSES)
    permanent = sum(
        1 for r in indexed.values()
        if (r.get("status") or "").lower() in PERMANENT_FAILURE_STATUSES
    )
    retryable = sum(
        1 for r in indexed.values()
        if (r.get("status") or "").lower() in RETRYABLE_FAILURE_STATUSES
    )
    other = len(indexed) - ok - permanent - retryable
    return {
        "rows": len(indexed),
        "ok": ok,
        "permanent_failure": permanent,
        "retryable_failure": retryable,
        "other": other,
    }
