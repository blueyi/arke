# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convergence curve extraction from an agent trajectory.

Purpose (KESTREL K-H5.2, 2026-07-28):
    Answer the audit gap "how efficient is the optimization loop's
    convergence?" (2026-07-27 comprehensive audit §一.3). Given the ordered
    event trajectory produced by :class:`arke.agent.runner.LLMRunner`, project
    the ``compile_and_profile`` events into a convergence curve:

    ``(iteration, tool, backend, success, correct, latency_ms,
       baseline_ratio, vs_default, best_so_far_ratio)``

    where ``best_so_far_ratio`` tracks the running maximum of a *comparable*
    performance metric across iterations (prefers ``vs_default`` when
    available — that's the gate criterion — else falls back to
    ``baseline_ratio``). Only correct successful measurements are eligible
    to advance ``best_so_far_ratio``; incorrect / failed events are still
    emitted as CSV rows (audit trail) but do not update the running best.

Design contract:
    * **Input**: the ``trajectory`` list produced by ``LLMRunner.optimize``
      (see ``OptimizeResult.trajectory``) — a sequence of dicts with keys
      ``{type, step, tool, params, result}``. ``result`` mirrors the
      ``ToolResult`` JSON form: ``{success, data, error?, warnings?}``.
    * **Filter**: only events with ``tool == 'compile_and_profile'`` contribute
      rows. Other events (``list_legal_actions``, ``apply_decision``, etc.)
      are ignored — this is a *performance* curve, not a decision log.
    * **Metric**: for the running best we prefer ``vs_default`` because
      it is the gate criterion (< 1.0 means the agent's kernel beats the
      strategy=None default on the same shapes). We invert to ratio-space
      (default / agent) so higher is better and the curve is monotone
      non-decreasing. Fallback: ``baseline_ratio`` (baseline / arke, higher
      is better) when ``vs_default`` is missing.
    * **Failure handling**: rows with ``success=False`` or ``correct=False``
      are still emitted with the current ``best_so_far_ratio`` (unchanged)
      so the CSV shows honest per-iteration attempt state.

The extraction is a pure function of the trajectory — no re-execution, no
GPU touch. Safe to call after any ``arke run --backend builtin`` completes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# ── Column names for the CSV output (stable public contract) ──────────────

CONVERGENCE_COLUMNS: tuple[str, ...] = (
    "iteration",          # 1-based; N-th compile_and_profile call
    "step",               # global trajectory step index (all tools)
    "tool",               # always 'compile_and_profile' (kept for join clarity)
    "backend",            # 'triton' | 'cuda_c' | 'llvm' | 'mock' | ...
    "success",            # tool-level success
    "correct",            # V1 numeric correctness (None if not measured)
    "max_diff",           # V1 max abs delta vs reference (None if not measured)
    "latency_ms",         # V2 wall-clock (None on mock backend)
    "baseline_ratio",     # baseline / arke (higher is better)
    "vs_default",         # arke / same-backend-default (< 1.0 beats default)
    "meas_spread",        # max/min - 1 across measurement passes
    "current_ratio",      # chosen metric this iteration (higher is better)
    "best_so_far_ratio",  # running max of eligible current_ratio values
)


def _extract_metric(data: dict[str, Any]) -> float | None:
    """Prefer vs_default (gate criterion), fall back to baseline_ratio.

    ``vs_default`` is agent / default — lower means faster. Invert to
    default / agent so higher is better across metrics, keeping the curve
    monotonically non-decreasing on improvement.

    Returns ``None`` if neither metric is measurable (e.g. mock backend).
    """
    vs_default = data.get("vs_default")
    if vs_default is not None and vs_default > 0:
        return 1.0 / float(vs_default)
    baseline_ratio = data.get("baseline_ratio")
    if baseline_ratio is not None:
        return float(baseline_ratio)
    return None


def build_convergence_rows(
    trajectory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract the convergence curve from an LLMRunner trajectory.

    See module docstring for contract and metric selection rationale.

    Args:
        trajectory: ordered list of trajectory events (from
            ``OptimizeResult.trajectory``). Non-``compile_and_profile``
            events are skipped.

    Returns:
        List of row dicts, one per ``compile_and_profile`` event, in
        iteration order. Column set matches :data:`CONVERGENCE_COLUMNS`.
    """
    rows: list[dict[str, Any]] = []
    best_so_far: float | None = None
    iteration = 0

    for event in trajectory:
        if not isinstance(event, dict):
            continue
        if event.get("tool") != "compile_and_profile":
            continue
        iteration += 1

        result = event.get("result", {}) or {}
        # ToolResult serialization keeps warnings/error at top level and
        # tool payload under "data" (see ArkeTool.to_json).
        data = result.get("data", {}) or {}

        success = bool(result.get("success", False))
        correct = data.get("correct")
        current_ratio = _extract_metric(data)

        # Advance best_so_far only on a correct successful measurement.
        # `correct is None` (verify skipped) does NOT advance — we want
        # the running best to reflect only trustworthy measurements.
        if success and correct is True and current_ratio is not None:
            if best_so_far is None or current_ratio > best_so_far:
                best_so_far = current_ratio

        rows.append({
            "iteration": iteration,
            "step": event.get("step"),
            "tool": "compile_and_profile",
            "backend": data.get("backend"),
            "success": success,
            "correct": correct,
            "max_diff": data.get("max_diff"),
            "latency_ms": data.get("latency_ms"),
            "baseline_ratio": data.get("baseline_ratio"),
            "vs_default": data.get("vs_default"),
            "meas_spread": data.get("meas_spread"),
            "current_ratio": current_ratio,
            "best_so_far_ratio": best_so_far,
        })

    return rows


def emit_convergence_csv(
    trajectory: list[dict[str, Any]],
    path: str | Path,
) -> int:
    """Write the convergence curve to a CSV file.

    Args:
        trajectory: LLMRunner trajectory (see :func:`build_convergence_rows`).
        path: output CSV path. Parent dirs are created if missing.

    Returns:
        Number of rows written (excluding header). Zero rows still emits a
        header-only file — callers can distinguish "ran but did not profile"
        from "did not run" by file existence + row count.
    """
    rows = build_convergence_rows(trajectory)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CONVERGENCE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


__all__ = [
    "CONVERGENCE_COLUMNS",
    "build_convergence_rows",
    "emit_convergence_csv",
]
