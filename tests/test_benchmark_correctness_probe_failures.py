# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from benchmarks.bench_l1 import _measure_l1_correctness


class _DummyRunner:
    def __init__(self, output, name: str = "DummyRunner"):
        self.output = output
        self.name = name
        self.available = True

    def run_with_inputs(self, op, *inputs):
        return self.output


def test_l1_correctness_probe_reports_mismatch_for_shape_difference():
    runner = _DummyRunner(torch.zeros((127, 128), dtype=torch.float16, device="cpu"))

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "mismatch"
    assert result["allclose"] is False
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert "shape mismatch" in result["correctness_reason"]


def test_l1_correctness_probe_reports_unsupported_for_non_tensor_output():
    runner = _DummyRunner({"not": "a tensor"})

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "unsupported"
    assert result["allclose"] is None
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert "non-tensor correctness output" in result["correctness_reason"]


def test_l1_correctness_probe_reports_error_when_runner_raises():
    class _RaisingRunner:
        name = "RaisingRunner"
        available = True

        def run_with_inputs(self, op, *inputs):
            raise RuntimeError("boom")

    result = _measure_l1_correctness(_RaisingRunner(), "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "error"
    assert result["allclose"] is None
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert result["correctness_reason"] == "boom"


def test_l1_correctness_probe_reports_tuple_length_mismatch(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: (torch.zeros((128, 128), dtype=torch.float16, device="cpu"),),
    )
    runner = _DummyRunner(
        (
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
        )
    )

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "mismatch"
    assert result["allclose"] is False
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert "tuple length mismatch" in result["correctness_reason"]


def test_l1_correctness_probe_reports_tuple_shape_mismatch(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: (
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
        ),
    )
    runner = _DummyRunner(
        (
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
            torch.zeros((127, 128), dtype=torch.float16, device="cpu"),
        )
    )

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "mismatch"
    assert result["allclose"] is False
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert "shape mismatch in tuple element" in result["correctness_reason"]


def test_l1_correctness_probe_reports_tuple_non_tensor_output_as_unsupported(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: (
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
        ),
    )
    runner = _DummyRunner((torch.zeros((128, 128), dtype=torch.float16, device="cpu"), "bad"))

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "unsupported"
    assert result["allclose"] is None
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert "non-tensor tuple correctness output" in result["correctness_reason"]


def test_l1_correctness_probe_reports_value_mismatch_without_exception(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: torch.zeros((128, 128), dtype=torch.float32, device="cpu"),
    )
    runner = _DummyRunner(torch.ones((128, 128), dtype=torch.float32, device="cpu"))

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float32)

    assert result["correctness_status"] == "mismatch"
    assert result["allclose"] is False
    assert result["max_abs_diff"] is not None
    assert result["mean_abs_diff"] is not None
    assert result["correctness_reason"] == ""


def test_l1_correctness_probe_reports_tuple_value_mismatch_without_exception(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: (
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
            torch.zeros((128, 128), dtype=torch.float16, device="cpu"),
        ),
    )
    runner = _DummyRunner(
        (
            torch.ones((128, 128), dtype=torch.float16, device="cpu"),
            torch.ones((128, 128), dtype=torch.float16, device="cpu"),
        )
    )

    result = _measure_l1_correctness(runner, "relu", 128, 128, 0, dtype=torch.float16)

    assert result["correctness_status"] == "mismatch"
    assert result["allclose"] is False
    assert result["max_abs_diff"] is not None
    assert result["mean_abs_diff"] is not None
    assert result["correctness_reason"] == ""


def test_l1_correctness_probe_reports_unsupported_when_reference_is_missing(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._make_l1_correctness_inputs",
        lambda op, M, N, K, dtype: (torch.zeros((1, 1), dtype=dtype, device="cpu"),),
    )
    runner = _DummyRunner(torch.zeros((1, 1), dtype=torch.float16, device="cpu"))

    result = _measure_l1_correctness(runner, "unknown_op", 1, 1, 0, dtype=torch.float16)

    assert result["correctness_status"] == "unsupported"
    assert result["allclose"] is None
    assert result["max_abs_diff"] is None
    assert result["mean_abs_diff"] is None
    assert result["correctness_reason"] == "No correctness reference for L1 op: unknown_op"


def test_l1_correctness_probe_handles_empty_tensor_outputs(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._make_l1_correctness_inputs",
        lambda op, M, N, K, dtype: (torch.zeros((0, 4), dtype=dtype, device="cpu"),),
    )
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: torch.zeros((0, 4), dtype=torch.float32, device="cpu"),
    )
    runner = _DummyRunner(torch.zeros((0, 4), dtype=torch.float32, device="cpu"))

    result = _measure_l1_correctness(runner, "relu", 0, 4, 0, dtype=torch.float32)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["mean_abs_diff"] == 0.0



def test_l1_correctness_probe_handles_empty_tuple_outputs(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._make_l1_correctness_inputs",
        lambda op, M, N, K, dtype: (torch.zeros((0, 4), dtype=dtype, device="cpu"),),
    )
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: (
            torch.zeros((0, 4), dtype=torch.float32, device="cpu"),
            torch.zeros((0,), dtype=torch.float32, device="cpu"),
        ),
    )
    runner = _DummyRunner(
        (
            torch.zeros((0, 4), dtype=torch.float32, device="cpu"),
            torch.zeros((0,), dtype=torch.float32, device="cpu"),
        )
    )

    result = _measure_l1_correctness(runner, "relu", 0, 4, 0, dtype=torch.float32)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["mean_abs_diff"] == 0.0


def test_l1_correctness_probe_accepts_values_within_atol(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: torch.zeros((2, 2), dtype=torch.float16, device="cpu"),
    )
    runner = _DummyRunner(torch.full((2, 2), 5e-3, dtype=torch.float16, device="cpu"))

    result = _measure_l1_correctness(runner, "matmul", 2, 2, 2, dtype=torch.float16)

    assert result["correctness_status"] == "ok"
    assert result["allclose"] is True
    assert result["rtol"] == 1e-2
    assert result["atol"] == 1e-2


def test_l1_correctness_probe_rejects_values_outside_atol(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.bench_l1._eval_l1_reference",
        lambda op, inputs: torch.zeros((2, 2), dtype=torch.float16, device="cpu"),
    )
    runner = _DummyRunner(torch.full((2, 2), 2e-2, dtype=torch.float16, device="cpu"))

    result = _measure_l1_correctness(runner, "matmul", 2, 2, 2, dtype=torch.float16)

    assert result["correctness_status"] == "mismatch"
    assert result["allclose"] is False
    assert result["rtol"] == 1e-2
    assert result["atol"] == 1e-2
