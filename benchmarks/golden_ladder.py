# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Golden Kernel ladder — single source of truth for (op → designated golden).

The Golden Kernel for an op simultaneously plays two roles:

  1. **Correctness oracle** — its output on a given input is the expected
     value that all other implementations are compared against.
  2. **Perf denominator** — its latency on the same (op, shape) is the
     baseline against which ``ratio_vs_baseline`` is computed.

Selection rule
--------------
For each op, iterate registered runners sorted by priority ascending (P0
first), return the first one where ``runner.supports(op) and
runner.available``. If none qualifies, raise :class:`GoldenUnavailable` so
the caller can emit a ``golden_unavailable_pending_baseline`` audit row.

The complete locked ladder lives in
``docs/benchmark/golden-kernel-ladder.md`` (added in commit 5). Code-side,
the only mechanism is *priority ordering + supports()*; per-op preferences
are encoded by tweaking individual runners' ``supports()`` sets so that the
desired golden naturally wins the ladder.

Overrides
---------
Callers (e.g. ``bench_l1 --golden op=name``) can pass a mapping
``{op: runner_name}`` to ``golden_runner_for`` that pins a specific runner
regardless of priority. The pinned runner must still be available; if it
isn't, :class:`GoldenUnavailable` is raised with a descriptive reason.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.baselines.base import BaselineRunner

logger = logging.getLogger(__name__)


class GoldenUnavailable(Exception):
    """No runner in the ladder can serve as Golden Kernel for this op."""

    def __init__(self, op: str, reason: str = "") -> None:
        self.op = op
        self.reason = reason or f"no available runner supports op={op!r}"
        super().__init__(self.reason)


def golden_runner_for(
    op: str,
    *,
    overrides: dict[str, str] | None = None,
) -> "BaselineRunner":
    """Pick the designated Golden Kernel for ``op``.

    Iterates :func:`benchmarks.baselines.base.get_all_runners` in priority
    order (P0..P5) and returns the first ``runner`` such that
    ``runner.supports(op) and runner.available``.

    Parameters
    ----------
    op : str
        Catalog operator name.
    overrides : dict[str, str], optional
        Map ``{op: runner_name}`` pinning a specific runner. The pinned
        runner must still be available; otherwise :class:`GoldenUnavailable`
        fires (no silent fall-through).

    Raises
    ------
    GoldenUnavailable
        If no runner qualifies (or the override target is unavailable).
    """
    # Local import to avoid a top-level cycle: base imports `torch`, which
    # is fine, but the ladder lives next to bench_l1 so we keep imports
    # lazy for cleaner unit-test isolation.
    from benchmarks.baselines.base import get_all_runners

    runners = get_all_runners()  # already sorted by priority ascending

    if overrides and op in overrides:
        pinned = overrides[op]
        for r in runners:
            if r.name == pinned:
                if not r.supports(op):
                    raise GoldenUnavailable(
                        op,
                        f"override pinned runner {pinned!r} but it does not "
                        f"declare supports({op!r})",
                    )
                return r
        raise GoldenUnavailable(
            op,
            f"override pinned runner {pinned!r} not registered or unavailable",
        )

    for r in runners:
        if r.supports(op):
            return r

    raise GoldenUnavailable(op)


def parse_overrides_file(path: str | None) -> dict[str, str]:
    """Parse a YAML mapping ``{op: runner_name}`` from ``--golden-file``.

    Returns an empty dict if ``path`` is None. Accepts both real YAML and a
    plain ``key: value`` per-line format so the file can be hand-edited
    without pulling pyyaml when it's unavailable.
    """
    if not path:
        return {}
    text = open(path).read()
    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            data[k.strip()] = v.strip().strip("'\"")
    if not isinstance(data, dict):
        raise ValueError(
            f"--golden-file {path!r} must contain a mapping of op→runner_name"
        )
    return {str(k): str(v) for k, v in data.items()}


def parse_inline_overrides(spec: list[str] | None) -> dict[str, str]:
    """Parse ``--golden op=name`` CLI specifications.

    Each item is a single ``op=runner_name`` token. Returns the combined
    overrides dict (later items override earlier ones for the same op).
    """
    out: dict[str, str] = {}
    if not spec:
        return out
    for item in spec:
        if "=" not in item:
            raise ValueError(
                f"--golden value must be 'op=runner_name', got {item!r}"
            )
        op, _, name = item.partition("=")
        out[op.strip()] = name.strip()
    return out
