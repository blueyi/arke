# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runner — runs Arke and LLM-direct-Triton on benchmark tasks.

Two modes:
- Arke mode: LLM optimizes via tool-use (structured IR + validation)
- Direct mode: LLM writes Triton kernel code directly (no IR, no validation)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a single benchmark trial."""

    task_name: str
    method: str  # "arke" or "direct"
    trial: int
    correct: bool
    vs_baseline: float | None = None
    latency_us: float | None = None
    tflops: float | None = None
    error: str | None = None
    decisions: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "method": self.method,
            "trial": self.trial,
            "correct": self.correct,
            "vs_baseline": self.vs_baseline,
            "latency_us": self.latency_us,
            "tflops": self.tflops,
            "error": self.error,
            "decisions": self.decisions,
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_s": self.duration_s,
        }


@dataclass
class TaskSummary:
    """Aggregated results for one task × one method."""

    task_name: str
    method: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.correct) / len(self.trials)

    @property
    def mean_perf(self) -> float | None:
        perfs = [t.vs_baseline for t in self.trials if t.vs_baseline is not None]
        return float(np.mean(perfs)) if perfs else None

    @property
    def std_perf(self) -> float | None:
        perfs = [t.vs_baseline for t in self.trials if t.vs_baseline is not None]
        return float(np.std(perfs)) if len(perfs) > 1 else None

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens_in + t.tokens_out for t in self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "method": self.method,
            "correct_rate": self.correct_rate,
            "mean_perf": self.mean_perf,
            "std_perf": self.std_perf,
            "total_tokens": self.total_tokens,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    arke_results: dict[str, TaskSummary] = field(default_factory=dict)
    direct_results: dict[str, TaskSummary] = field(default_factory=dict)
    timestamp: str = ""

    def gate_g4_pass(self) -> tuple[bool, list[str]]:
        """Check if Gate G4 criteria are met.

        G4: Arke correctness AND perf >= LLM-direct-Triton.
        """
        reasons = []
        all_pass = True

        # Correctness comparison
        arke_correct = np.mean([
            s.correct_rate for s in self.arke_results.values()
        ]) if self.arke_results else 0.0
        direct_correct = np.mean([
            s.correct_rate for s in self.direct_results.values()
        ]) if self.direct_results else 0.0

        if arke_correct >= direct_correct:
            reasons.append(
                f"✅ Correctness: Arke {arke_correct:.1%}"
                f" >= Direct {direct_correct:.1%}"
            )
        else:
            reasons.append(
                f"❌ Correctness: Arke {arke_correct:.1%}"
                f" < Direct {direct_correct:.1%}"
            )
            all_pass = False

        # Performance comparison
        arke_perfs = [
            s.mean_perf for s in self.arke_results.values()
            if s.mean_perf is not None
        ]
        direct_perfs = [
            s.mean_perf for s in self.direct_results.values()
            if s.mean_perf is not None
        ]

        if arke_perfs and direct_perfs:
            arke_mean = float(np.mean(arke_perfs))
            direct_mean = float(np.mean(direct_perfs))
            if arke_mean >= direct_mean:
                reasons.append(
                    f"✅ Performance: Arke {arke_mean:.3f}"
                    f" >= Direct {direct_mean:.3f}"
                )
            else:
                reasons.append(
                    f"❌ Performance: Arke {arke_mean:.3f}"
                    f" < Direct {direct_mean:.3f}"
                )
                all_pass = False

        # Variance comparison
        arke_vars = [
            s.std_perf for s in self.arke_results.values()
            if s.std_perf is not None
        ]
        direct_vars = [
            s.std_perf for s in self.direct_results.values()
            if s.std_perf is not None
        ]

        if arke_vars and direct_vars:
            arke_var = float(np.mean(arke_vars))
            direct_var = float(np.mean(direct_vars))
            if arke_var <= direct_var:
                reasons.append(
                    f"✅ Consistency: Arke σ={arke_var:.4f}"
                    f" <= Direct σ={direct_var:.4f}"
                )
            else:
                reasons.append(
                    f"⚠️ Consistency: Arke σ={arke_var:.4f}"
                    f" > Direct σ={direct_var:.4f}"
                )

        return all_pass, reasons

    def to_dict(self) -> dict[str, Any]:
        passed, reasons = self.gate_g4_pass()
        return {
            "timestamp": self.timestamp,
            "gate_g4": {
                "passed": passed,
                "reasons": reasons,
            },
            "arke": {
                k: v.to_dict()
                for k, v in self.arke_results.items()
            },
            "direct": {
                k: v.to_dict()
                for k, v in self.direct_results.items()
            },
        }

    def save(self, path: str) -> None:
        """Save report as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {path}")
