# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — Accuracy Benchmark Framework.

Comprehensive element-wise numerical accuracy comparison between
different implementations (NumPy CPU, GPU kernel, Ascend, etc.).

Supports pluggable reference sources and configurable thresholds.

Usage:
    benchmark = AccuracyBenchmark()
    result = benchmark.compare(
        test_output=gpu_tensor,
        ref_output=numpy_array,
        config=CompareConfig(precision_test="f16", precision_ref="f32"),
    )
    print(result.verdict)  # "accept" / "review" / "reject"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ============================================================
# Verdict
# ============================================================

class Verdict(str, Enum):
    """Accuracy comparison verdict."""
    ACCEPT = "accept"    # Within normal tolerance
    REVIEW = "review"    # Suspicious, needs human review
    REJECT = "reject"    # Unacceptable deviation


# ============================================================
# Metrics
# ============================================================

@dataclass
class AccuracyMetrics:
    """Complete accuracy metrics for an element-wise comparison."""

    # Absolute error
    abs_max: float = 0.0
    abs_mean: float = 0.0
    abs_median: float = 0.0
    abs_std: float = 0.0

    # Relative error (computed only where |ref| > epsilon)
    rel_max: float = 0.0
    rel_mean: float = 0.0
    rel_median: float = 0.0
    rel_p90: float = 0.0
    rel_p99: float = 0.0

    # ULP error
    ulp_mean: float = 0.0
    ulp_max: float = 0.0
    ulp_p99: float = 0.0

    # Anomaly counts
    nan_count: int = 0
    inf_count: int = 0
    sign_mismatch_rate: float = 0.0

    # Similarity
    cosine_similarity: float = 1.0
    zero_diff_rate: float = 1.0  # fraction of exactly equal elements

    # Shape info
    total_elements: int = 0
    nontrivial_elements: int = 0  # |ref| > epsilon

    def to_dict(self) -> dict[str, Any]:
        return {
            "abs_max": round(self.abs_max, 8),
            "abs_mean": round(self.abs_mean, 8),
            "abs_median": round(self.abs_median, 8),
            "abs_std": round(self.abs_std, 8),
            "rel_max": round(self.rel_max, 8),
            "rel_mean": round(self.rel_mean, 8),
            "rel_median": round(self.rel_median, 8),
            "rel_p90": round(self.rel_p90, 8),
            "rel_p99": round(self.rel_p99, 8),
            "ulp_mean": round(self.ulp_mean, 4),
            "ulp_max": round(self.ulp_max, 4),
            "ulp_p99": round(self.ulp_p99, 4),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "sign_mismatch_rate": round(self.sign_mismatch_rate, 8),
            "cosine_similarity": round(self.cosine_similarity, 8),
            "zero_diff_rate": round(self.zero_diff_rate, 6),
            "total_elements": self.total_elements,
            "nontrivial_elements": self.nontrivial_elements,
        }


# ============================================================
# Configuration
# ============================================================

@dataclass
class CompareConfig:
    """Configuration for an accuracy comparison.

    Design principle: reference uses THE SAME dtype as test by default.
    This measures implementation correctness (is my kernel right?),
    not precision loss (is f16 good enough?).

    To measure precision loss instead, set precision_ref to a higher dtype.
    """

    # Precision labels — same dtype by default
    precision_test: str = "f16"   # e.g. "f16", "f32", "int8", "f8"
    precision_ref: str = "f16"    # defaults to SAME as test

    # Threshold for "nontrivial" elements (skip near-zero for relative error)
    epsilon: float = 1e-6

    # Verdict thresholds
    accept_rel_mean: float = 1e-4
    accept_rel_p99: float = 1e-3
    review_rel_mean: float = 1e-2
    review_rel_p99: float = 1e-1
    reject_sign_mismatch: float = 1e-3  # 0.1%
    accept_ulp_mean: float = 4.0
    review_ulp_p99: float = 16.0

    # Input generation
    seed: int = 42
    num_trials: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_test": self.precision_test,
            "precision_ref": self.precision_ref,
            "epsilon": self.epsilon,
            "accept_rel_mean": self.accept_rel_mean,
            "accept_rel_p99": self.accept_rel_p99,
            "review_rel_mean": self.review_rel_mean,
            "review_rel_p99": self.review_rel_p99,
            "seed": self.seed,
            "num_trials": self.num_trials,
        }


# Per-dtype default configs — reference uses SAME dtype as test
# These thresholds account for non-determinism in GPU implementations
# (different reduction order, fused multiply-add, etc.)
DTYPE_CONFIGS: dict[str, CompareConfig] = {
    "f16": CompareConfig(
        precision_test="f16", precision_ref="f16",
        accept_rel_mean=1e-3, accept_rel_p99=5e-2,
        review_rel_mean=5e-2, review_rel_p99=2e-1,
        accept_ulp_mean=8.0, review_ulp_p99=64.0,
    ),
    "bf16": CompareConfig(
        precision_test="bf16", precision_ref="bf16",
        accept_rel_mean=1e-3, accept_rel_p99=5e-2,
        review_rel_mean=5e-2, review_rel_p99=2e-1,
        accept_ulp_mean=8.0, review_ulp_p99=64.0,
    ),
    "f32": CompareConfig(
        precision_test="f32", precision_ref="f32",
        accept_rel_mean=1e-5, accept_rel_p99=1e-4,
        review_rel_mean=1e-3, review_rel_p99=1e-2,
    ),
    "int8": CompareConfig(
        precision_test="int8", precision_ref="int8",
        accept_rel_mean=5e-2, accept_rel_p99=2e-1,
        review_rel_mean=1e-1, review_rel_p99=5e-1,
    ),
}

# Cross-precision configs — for measuring precision loss (optional)
CROSS_DTYPE_CONFIGS: dict[str, CompareConfig] = {
    "f16_vs_f32": CompareConfig(
        precision_test="f16", precision_ref="f32",
        accept_rel_mean=1e-3, accept_rel_p99=5e-2,
        review_rel_mean=5e-2, review_rel_p99=2e-1,
    ),
    "f16_vs_f64": CompareConfig(
        precision_test="f16", precision_ref="f64",
        accept_rel_mean=1e-3, accept_rel_p99=5e-2,
        review_rel_mean=5e-2, review_rel_p99=2e-1,
    ),
    "f32_vs_f64": CompareConfig(
        precision_test="f32", precision_ref="f64",
        accept_rel_mean=1e-5, accept_rel_p99=1e-4,
        review_rel_mean=1e-3, review_rel_p99=1e-2,
    ),
}


# ============================================================
# Comparison result
# ============================================================

@dataclass
class CompareResult:
    """Result of a single accuracy comparison."""
    op_name: str
    input_shape: list[int]
    config: CompareConfig
    metrics: AccuracyMetrics
    verdict: Verdict
    verdict_reasons: list[str] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    top_errors: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "input_shape": self.input_shape,
            "precision_test": self.config.precision_test,
            "precision_ref": self.config.precision_ref,
            "seed": self.config.seed,
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict.value,
            "verdict_reasons": self.verdict_reasons,
            "top_errors": self.top_errors,
            "anomalies": self.anomalies,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkResult:
    """Aggregated result across multiple trials."""
    op_name: str
    input_shape: list[int]
    config: CompareConfig
    trials: list[CompareResult] = field(default_factory=list)
    aggregate_metrics: AccuracyMetrics | None = None
    final_verdict: Verdict = Verdict.ACCEPT
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "input_shape": self.input_shape,
            "config": self.config.to_dict(),
            "num_trials": len(self.trials),
            "aggregate_metrics": self.aggregate_metrics.to_dict() if self.aggregate_metrics else None,
            "final_verdict": self.final_verdict.value,
            "trials": [t.to_dict() for t in self.trials],
            "environment": self.environment,
        }


# ============================================================
# Core comparison engine
# ============================================================

class AccuracyBenchmark:
    """Element-wise accuracy comparison engine.

    Supports pluggable reference sources:
    - NumPy CPU (default)
    - PyTorch GPU (torch.matmul etc.)
    - Custom reference tensors
    """

    def compute_metrics(
        self,
        test: np.ndarray,
        ref: np.ndarray,
        config: CompareConfig | None = None,
    ) -> AccuracyMetrics:
        """Compute comprehensive accuracy metrics between test and reference.

        Both inputs should be float arrays (upcast if needed).
        """
        if config is None:
            config = CompareConfig()

        # Ensure float64 for computation
        t = test.astype(np.float64).ravel()
        r = ref.astype(np.float64).ravel()

        assert t.shape == r.shape, f"Shape mismatch: {t.shape} vs {r.shape}"

        total = t.size
        metrics = AccuracyMetrics(total_elements=total)

        # NaN / Inf checks
        metrics.nan_count = int(np.sum(np.isnan(t)))
        metrics.inf_count = int(np.sum(np.isinf(t)))

        # Replace NaN/Inf for safe computation
        valid_mask = np.isfinite(t) & np.isfinite(r)
        t_safe = np.where(valid_mask, t, 0.0)
        r_safe = np.where(valid_mask, r, 0.0)

        # Absolute error
        abs_err = np.abs(t_safe - r_safe)
        if np.any(valid_mask):
            valid_abs = abs_err[valid_mask]
            metrics.abs_max = float(np.max(valid_abs))
            metrics.abs_mean = float(np.mean(valid_abs))
            metrics.abs_median = float(np.median(valid_abs))
            metrics.abs_std = float(np.std(valid_abs))

        # Relative error (only where |ref| > epsilon)
        nontrivial = valid_mask & (np.abs(r) > config.epsilon)
        metrics.nontrivial_elements = int(np.sum(nontrivial))

        if metrics.nontrivial_elements > 0:
            rel_err = abs_err[nontrivial] / np.abs(r[nontrivial])
            metrics.rel_max = float(np.max(rel_err))
            metrics.rel_mean = float(np.mean(rel_err))
            metrics.rel_median = float(np.median(rel_err))
            metrics.rel_p90 = float(np.percentile(rel_err, 90))
            metrics.rel_p99 = float(np.percentile(rel_err, 99))

        # ULP error
        ulp_errors = self._compute_ulp(t_safe, r_safe, valid_mask)
        if ulp_errors is not None and len(ulp_errors) > 0:
            metrics.ulp_mean = float(np.mean(ulp_errors))
            metrics.ulp_max = float(np.max(ulp_errors))
            metrics.ulp_p99 = float(np.percentile(ulp_errors, 99))

        # Sign mismatch
        sign_mask = valid_mask & (np.abs(r) > config.epsilon) & (np.abs(t) > config.epsilon)
        if np.any(sign_mask):
            sign_mismatch = np.sign(t[sign_mask]) != np.sign(r[sign_mask])
            metrics.sign_mismatch_rate = float(np.mean(sign_mismatch))

        # Cosine similarity
        if np.any(valid_mask):
            t_v = t_safe[valid_mask]
            r_v = r_safe[valid_mask]
            denom = np.linalg.norm(t_v) * np.linalg.norm(r_v)
            if denom > 0:
                metrics.cosine_similarity = float(np.dot(t_v, r_v) / denom)

        # Zero diff rate
        metrics.zero_diff_rate = float(np.mean(t_safe == r_safe))

        return metrics

    def judge(
        self, metrics: AccuracyMetrics, config: CompareConfig
    ) -> tuple[Verdict, list[str]]:
        """Determine verdict based on metrics and thresholds."""
        reasons: list[str] = []

        # Reject conditions
        if metrics.nan_count > 0:
            reasons.append(f"NaN detected: {metrics.nan_count} elements")
            return Verdict.REJECT, reasons

        if metrics.inf_count > 0:
            reasons.append(f"Inf detected: {metrics.inf_count} elements")
            return Verdict.REJECT, reasons

        if metrics.sign_mismatch_rate > config.reject_sign_mismatch:
            reasons.append(
                f"Sign mismatch rate {metrics.sign_mismatch_rate:.4%} > "
                f"threshold {config.reject_sign_mismatch:.4%}"
            )
            return Verdict.REJECT, reasons

        if metrics.rel_mean > config.review_rel_mean:
            reasons.append(f"rel_mean {metrics.rel_mean:.2e} > {config.review_rel_mean:.2e}")
            return Verdict.REJECT, reasons

        if metrics.rel_p99 > config.review_rel_p99:
            reasons.append(f"rel_p99 {metrics.rel_p99:.2e} > {config.review_rel_p99:.2e}")
            return Verdict.REJECT, reasons

        # Review conditions
        if metrics.rel_mean > config.accept_rel_mean:
            reasons.append(f"rel_mean {metrics.rel_mean:.2e} > accept threshold {config.accept_rel_mean:.2e}")

        if metrics.rel_p99 > config.accept_rel_p99:
            reasons.append(f"rel_p99 {metrics.rel_p99:.2e} > accept threshold {config.accept_rel_p99:.2e}")

        if metrics.ulp_p99 > config.review_ulp_p99:
            reasons.append(f"ulp_p99 {metrics.ulp_p99:.1f} > {config.review_ulp_p99:.1f}")

        if reasons:
            return Verdict.REVIEW, reasons

        return Verdict.ACCEPT, ["All metrics within acceptable range"]

    def compare(
        self,
        test: np.ndarray,
        ref: np.ndarray,
        op_name: str = "unknown",
        config: CompareConfig | None = None,
    ) -> CompareResult:
        """Run a single comparison and return result with verdict."""
        if config is None:
            config = CompareConfig()

        metrics = self.compute_metrics(test, ref, config)
        verdict, reasons = self.judge(metrics, config)

        # Find top-k error elements
        top_errors = self._find_top_errors(test, ref, k=5)

        return CompareResult(
            op_name=op_name,
            input_shape=list(ref.shape),
            config=config,
            metrics=metrics,
            verdict=verdict,
            verdict_reasons=reasons,
            top_errors=top_errors,
        )

    def benchmark(
        self,
        test_fn,
        ref_fn,
        input_gen_fn,
        op_name: str = "unknown",
        config: CompareConfig | None = None,
    ) -> BenchmarkResult:
        """Run multiple trials and aggregate results.

        Args:
            test_fn: callable(inputs) -> np.ndarray (test implementation)
            ref_fn: callable(inputs) -> np.ndarray (reference implementation)
            input_gen_fn: callable(seed) -> dict[str, np.ndarray]
            op_name: operator name
            config: comparison config
        """
        if config is None:
            config = CompareConfig()

        result = BenchmarkResult(
            op_name=op_name,
            input_shape=[],
            config=config,
            environment=self._get_environment(),
        )

        all_metrics: list[AccuracyMetrics] = []

        for trial in range(config.num_trials):
            seed = config.seed + trial
            inputs = input_gen_fn(seed)

            ref_output = ref_fn(inputs)
            test_output = test_fn(inputs)

            if trial == 0:
                result.input_shape = list(ref_output.shape)

            trial_config = CompareConfig(**{**config.__dict__, "seed": seed})
            trial_result = self.compare(
                test_output, ref_output,
                op_name=op_name, config=trial_config,
            )
            result.trials.append(trial_result)
            all_metrics.append(trial_result.metrics)

        # Aggregate: use median of each metric across trials
        if all_metrics:
            result.aggregate_metrics = self._aggregate_metrics(all_metrics)
            result.final_verdict, _ = self.judge(result.aggregate_metrics, config)

        return result

    # ─── Internal helpers ───

    def _compute_ulp(
        self, test: np.ndarray, ref: np.ndarray, mask: np.ndarray
    ) -> np.ndarray | None:
        """Compute ULP (Unit in Last Place) error.

        ULP = |test - ref| / machine_epsilon_at_ref
        """
        valid = mask & (ref != 0)
        if not np.any(valid):
            return None

        # For f32: eps ≈ 1.19e-7, for f64: eps ≈ 2.22e-16
        # ULP = |test - ref| / (|ref| * eps)
        eps = np.finfo(np.float64).eps
        ulp = np.abs(test[valid] - ref[valid]) / (np.abs(ref[valid]) * eps)
        return ulp

    def _find_top_errors(
        self, test: np.ndarray, ref: np.ndarray, k: int = 5
    ) -> list[dict[str, Any]]:
        """Find the k elements with largest absolute error."""
        t = test.astype(np.float64).ravel()
        r = ref.astype(np.float64).ravel()
        abs_err = np.abs(t - r)

        # Top-k indices
        if len(abs_err) <= k:
            top_idx = np.argsort(abs_err)[::-1]
        else:
            top_idx = np.argpartition(abs_err, -k)[-k:]
            top_idx = top_idx[np.argsort(abs_err[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            if abs_err[idx] == 0:
                continue
            # Convert flat index to multi-dim
            multi_idx = np.unravel_index(idx, test.shape) if test.ndim > 1 else (int(idx),)
            rel = abs_err[idx] / max(abs(r[idx]), 1e-12)
            results.append({
                "index": [int(i) for i in multi_idx],
                "test_value": float(t[idx]),
                "ref_value": float(r[idx]),
                "abs_error": float(abs_err[idx]),
                "rel_error": float(rel),
            })

        return results

    def _aggregate_metrics(
        self, metrics_list: list[AccuracyMetrics]
    ) -> AccuracyMetrics:
        """Aggregate metrics across trials using median."""
        def med(attr: str) -> float:
            return float(np.median([getattr(m, attr) for m in metrics_list]))

        def max_(attr: str) -> float:
            return float(np.max([getattr(m, attr) for m in metrics_list]))

        def sum_(attr: str) -> int:
            return int(np.sum([getattr(m, attr) for m in metrics_list]))

        return AccuracyMetrics(
            abs_max=max_("abs_max"),
            abs_mean=med("abs_mean"),
            abs_median=med("abs_median"),
            abs_std=med("abs_std"),
            rel_max=max_("rel_max"),
            rel_mean=med("rel_mean"),
            rel_median=med("rel_median"),
            rel_p90=med("rel_p90"),
            rel_p99=med("rel_p99"),
            ulp_mean=med("ulp_mean"),
            ulp_max=max_("ulp_max"),
            ulp_p99=med("ulp_p99"),
            nan_count=sum_("nan_count"),
            inf_count=sum_("inf_count"),
            sign_mismatch_rate=med("sign_mismatch_rate"),
            cosine_similarity=med("cosine_similarity"),
            zero_diff_rate=med("zero_diff_rate"),
            total_elements=metrics_list[0].total_elements,
            nontrivial_elements=metrics_list[0].nontrivial_elements,
        )

    @staticmethod
    def _get_environment() -> dict[str, Any]:
        """Capture execution environment info."""
        import platform
        env: dict[str, Any] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        try:
            import torch
            env["torch"] = torch.__version__
            if torch.cuda.is_available():
                env["cuda"] = torch.version.cuda or "N/A"
                env["gpu"] = torch.cuda.get_device_name(0)
                env["cudnn"] = str(torch.backends.cudnn.version())
                env["deterministic"] = torch.are_deterministic_algorithms_enabled()
        except ImportError:
            pass
        return env
