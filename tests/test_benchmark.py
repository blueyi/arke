# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark framework (tasks, runner, report)."""

import json
import tempfile

import pytest

from benchmarks.runner import (
    BenchmarkReport,
    TaskSummary,
    TrialResult,
)
from benchmarks.tasks import (
    BENCHMARK_TASKS,
    get_task,
    get_tasks,
)

# ============================================================
# Task Registry Tests
# ============================================================

class TestBenchmarkTasks:

    def test_at_least_5_tasks(self):
        """Phase 1.5.1: ≥5 benchmark tasks defined."""
        assert len(BENCHMARK_TASKS) >= 5

    def test_all_tasks_have_ir(self):
        """Each task has a valid SemanticIR."""
        for task in BENCHMARK_TASKS:
            assert task.semantic_ir is not None
            assert len(task.semantic_ir.params) > 0
            assert task.semantic_ir.return_node is not None

    def test_task_names_unique(self):
        names = [t.name for t in BENCHMARK_TASKS]
        assert len(names) == len(set(names))

    def test_get_task_by_name(self):
        task = get_task("matmul_1024")
        assert task.name == "matmul_1024"
        assert "1024" in task.description

    def test_get_task_invalid(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")

    def test_filter_by_tag(self):
        matmul_tasks = get_tasks(tags=["matmul"])
        assert len(matmul_tasks) >= 3
        for t in matmul_tasks:
            assert "matmul" in t.tags

    def test_filter_by_fusion_tag(self):
        fusion_tasks = get_tasks(tags=["fusion"])
        assert len(fusion_tasks) >= 2

    def test_task_repr(self):
        task = get_task("matmul_1024")
        r = repr(task)
        assert "matmul_1024" in r
        assert "A:" in r

    def test_softmax_task(self):
        task = get_task("softmax_4096")
        assert "softmax" in task.tags
        assert task.semantic_ir.params[0].shape == [4096, 4096]

    def test_task_dtypes(self):
        for task in BENCHMARK_TASKS:
            assert task.dtype in ("f16", "f32", "bf16")


# ============================================================
# Trial Result Tests
# ============================================================

class TestTrialResult:

    def test_trial_to_dict(self):
        trial = TrialResult(
            task_name="matmul_1024",
            method="arke",
            trial=0,
            correct=True,
            vs_baseline=1.06,
            latency_us=160.0,
            tflops=13.4,
            decisions=4,
            tool_calls=23,
            tokens_in=5000,
            tokens_out=3000,
            duration_s=45.0,
        )
        d = trial.to_dict()
        assert d["correct"] is True
        assert d["vs_baseline"] == 1.06
        assert d["method"] == "arke"

    def test_failed_trial(self):
        trial = TrialResult(
            task_name="test",
            method="direct",
            trial=0,
            correct=False,
            error="Compilation failed",
        )
        d = trial.to_dict()
        assert d["correct"] is False
        assert "Compilation" in d["error"]


# ============================================================
# Task Summary Tests
# ============================================================

class TestTaskSummary:

    def _make_summary(
        self, method: str, correct_trials: int, total: int, perfs: list
    ) -> TaskSummary:
        summary = TaskSummary(task_name="test", method=method)
        for i in range(total):
            summary.trials.append(TrialResult(
                task_name="test",
                method=method,
                trial=i,
                correct=i < correct_trials,
                vs_baseline=perfs[i] if i < len(perfs) else None,
                tokens_in=1000,
                tokens_out=500,
            ))
        return summary

    def test_correct_rate(self):
        s = self._make_summary("arke", 2, 3, [1.0, 1.1, None])
        assert abs(s.correct_rate - 2 / 3) < 0.01

    def test_mean_perf(self):
        s = self._make_summary("arke", 3, 3, [1.0, 1.1, 0.9])
        assert abs(s.mean_perf - 1.0) < 0.01

    def test_std_perf(self):
        s = self._make_summary("arke", 3, 3, [1.0, 1.0, 1.0])
        assert s.std_perf == 0.0

    def test_total_tokens(self):
        s = self._make_summary("arke", 3, 3, [1.0, 1.0, 1.0])
        assert s.total_tokens == 3 * 1500  # 1000 + 500 per trial

    def test_empty_summary(self):
        s = TaskSummary(task_name="empty", method="arke")
        assert s.correct_rate == 0.0
        assert s.mean_perf is None
        assert s.std_perf is None

    def test_to_dict(self):
        s = self._make_summary("arke", 2, 2, [1.0, 1.1])
        d = s.to_dict()
        assert d["correct_rate"] == 1.0
        assert len(d["trials"]) == 2


# ============================================================
# Benchmark Report Tests
# ============================================================

class TestBenchmarkReport:

    def _make_report(
        self,
        arke_correct: float = 1.0,
        arke_perf: float = 1.0,
        direct_correct: float = 0.8,
        direct_perf: float = 0.9,
    ) -> BenchmarkReport:
        report = BenchmarkReport(timestamp="2026-04-01T23:00:00")

        # Arke results
        arke_summary = TaskSummary(task_name="test", method="arke")
        for i in range(3):
            arke_summary.trials.append(TrialResult(
                task_name="test",
                method="arke",
                trial=i,
                correct=i < int(arke_correct * 3),
                vs_baseline=arke_perf + (i - 1) * 0.01,
            ))
        report.arke_results["test"] = arke_summary

        # Direct results
        direct_summary = TaskSummary(
            task_name="test", method="direct"
        )
        for i in range(3):
            direct_summary.trials.append(TrialResult(
                task_name="test",
                method="direct",
                trial=i,
                correct=i < int(direct_correct * 3),
                vs_baseline=direct_perf + (i - 1) * 0.01,
            ))
        report.direct_results["test"] = direct_summary

        return report

    def test_gate_g4_pass(self):
        """Arke better → G4 passes."""
        report = self._make_report(
            arke_correct=1.0,
            arke_perf=1.05,
            direct_correct=0.67,
            direct_perf=0.9,
        )
        passed, reasons = report.gate_g4_pass()
        assert passed is True
        assert any("✅" in r for r in reasons)

    def test_gate_g4_fail_correctness(self):
        """Arke worse correctness → G4 fails."""
        report = self._make_report(
            arke_correct=0.33,
            arke_perf=1.05,
            direct_correct=1.0,
            direct_perf=0.9,
        )
        passed, reasons = report.gate_g4_pass()
        assert passed is False
        assert any("Correctness" in r and "❌" in r for r in reasons)

    def test_gate_g4_fail_performance(self):
        """Arke worse performance → G4 fails."""
        report = self._make_report(
            arke_correct=1.0,
            arke_perf=0.5,
            direct_correct=1.0,
            direct_perf=0.9,
        )
        passed, reasons = report.gate_g4_pass()
        assert passed is False
        assert any("Performance" in r and "❌" in r for r in reasons)

    def test_report_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        assert "gate_g4" in d
        assert "arke" in d
        assert "direct" in d
        assert d["timestamp"] == "2026-04-01T23:00:00"

    def test_report_json_serializable(self):
        report = self._make_report()
        json_str = json.dumps(report.to_dict())
        assert "gate_g4" in json_str

    def test_report_save(self):
        report = self._make_report()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            report.save(f.name)
            f.seek(0)
            loaded = json.load(open(f.name))
            assert loaded["gate_g4"]["passed"] is True
