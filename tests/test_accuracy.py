# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for accuracy benchmark framework."""

import json
import os

import numpy as np
import pytest

from arke.engine.accuracy import (
    CROSS_DTYPE_CONFIGS,
    DTYPE_CONFIGS,
    AccuracyBenchmark,
    AccuracyMetrics,
    CompareConfig,
    CompareResult,
    Verdict,
)
from arke.engine.reference_sources import (
    NumPyCPUSource,
    TorchGPUSource,
    get_reference_source,
)
from arke.ir.builder import KernelBuilder

# ============================================================
# AccuracyMetrics tests
# ============================================================

class TestAccuracyMetrics:

    def test_identical_arrays(self):
        bench = AccuracyBenchmark()
        a = np.random.randn(100, 100).astype(np.float32)
        m = bench.compute_metrics(a, a)
        assert m.abs_max == 0.0
        assert m.abs_mean == 0.0
        assert m.zero_diff_rate == 1.0
        assert m.cosine_similarity == pytest.approx(1.0, abs=1e-6)
        assert m.nan_count == 0
        assert m.inf_count == 0

    def test_small_perturbation(self):
        bench = AccuracyBenchmark()
        ref = np.random.randn(1000).astype(np.float64) + 1.0  # avoid near-zero
        noise = np.random.randn(1000).astype(np.float64) * 1e-6
        test = (ref + noise).astype(np.float64)
        m = bench.compute_metrics(test, ref)
        assert m.abs_max < 1e-4
        assert m.rel_mean < 1e-5
        assert m.cosine_similarity > 0.9999

    def test_nan_detection(self):
        bench = AccuracyBenchmark()
        ref = np.array([1.0, 2.0, 3.0])
        test = np.array([1.0, float("nan"), 3.0])
        m = bench.compute_metrics(test, ref)
        assert m.nan_count == 1

    def test_inf_detection(self):
        bench = AccuracyBenchmark()
        ref = np.array([1.0, 2.0, 3.0])
        test = np.array([1.0, float("inf"), 3.0])
        m = bench.compute_metrics(test, ref)
        assert m.inf_count == 1

    def test_sign_mismatch(self):
        bench = AccuracyBenchmark()
        ref = np.array([1.0, -2.0, 3.0, -4.0])
        test = np.array([1.0, 2.0, 3.0, -4.0])  # one sign flip
        m = bench.compute_metrics(test, ref)
        assert m.sign_mismatch_rate == pytest.approx(0.25, abs=0.01)

    def test_f16_simulation(self):
        """Simulate f16 precision loss."""
        bench = AccuracyBenchmark()
        ref = np.random.randn(512, 512).astype(np.float32)
        # Simulate f16 by round-tripping
        test = ref.astype(np.float16).astype(np.float32)
        m = bench.compute_metrics(test, ref)
        # f16 should have some error but be reasonable
        assert m.abs_max < 1.0
        assert m.rel_mean < 0.01

    def test_metrics_to_dict(self):
        m = AccuracyMetrics(abs_max=0.1, rel_mean=0.001, total_elements=100)
        d = m.to_dict()
        assert "abs_max" in d
        assert "rel_mean" in d
        assert d["total_elements"] == 100


# ============================================================
# Verdict tests
# ============================================================

class TestVerdict:

    def test_accept_verdict(self):
        bench = AccuracyBenchmark()
        ref = np.random.randn(1000).astype(np.float64) + 1.0
        test = ref.copy()  # Identical = always accept
        m = bench.compute_metrics(test, ref)
        verdict, reasons = bench.judge(m, CompareConfig())
        assert verdict == Verdict.ACCEPT

    def test_reject_nan(self):
        bench = AccuracyBenchmark()
        m = AccuracyMetrics(nan_count=1)
        verdict, reasons = bench.judge(m, CompareConfig())
        assert verdict == Verdict.REJECT
        assert "NaN" in reasons[0]

    def test_reject_inf(self):
        bench = AccuracyBenchmark()
        m = AccuracyMetrics(inf_count=5)
        verdict, reasons = bench.judge(m, CompareConfig())
        assert verdict == Verdict.REJECT

    def test_reject_high_rel_error(self):
        bench = AccuracyBenchmark()
        m = AccuracyMetrics(rel_mean=0.05, rel_p99=0.2)
        verdict, _ = bench.judge(m, CompareConfig())
        assert verdict == Verdict.REJECT

    def test_review_moderate_error(self):
        bench = AccuracyBenchmark()
        m = AccuracyMetrics(rel_mean=5e-4, rel_p99=5e-3)
        verdict, _ = bench.judge(m, CompareConfig())
        assert verdict == Verdict.REVIEW

    def test_dtype_specific_config(self):
        """Default configs use same dtype for test and ref."""
        f16_config = DTYPE_CONFIGS["f16"]
        assert f16_config.precision_test == "f16"
        assert f16_config.precision_ref == "f16"  # same dtype

        f32_config = DTYPE_CONFIGS["f32"]
        assert f32_config.precision_ref == "f32"  # same dtype
        assert f32_config.accept_rel_mean < f16_config.accept_rel_mean  # f32 tighter

    def test_cross_dtype_config(self):
        """Cross-dtype configs for precision loss measurement."""
        config = CROSS_DTYPE_CONFIGS["f16_vs_f32"]
        assert config.precision_test == "f16"
        assert config.precision_ref == "f32"


# ============================================================
# Compare and benchmark tests
# ============================================================

class TestCompare:

    def test_compare_returns_result(self):
        bench = AccuracyBenchmark()
        ref = np.random.randn(100, 100).astype(np.float32)
        test = ref.copy()
        result = bench.compare(test, ref, op_name="test_op")
        assert isinstance(result, CompareResult)
        assert result.op_name == "test_op"
        assert result.verdict == Verdict.ACCEPT

    def test_compare_top_errors(self):
        bench = AccuracyBenchmark()
        ref = np.ones((10, 10), dtype=np.float32)
        test = ref.copy()
        test[5, 7] = 100.0  # Inject large error
        result = bench.compare(test, ref, op_name="test")
        assert len(result.top_errors) > 0
        assert result.top_errors[0]["index"] == [5, 7]

    def test_compare_serialization(self):
        bench = AccuracyBenchmark()
        ref = np.random.randn(50).astype(np.float32)
        result = bench.compare(ref, ref, op_name="serialize_test")
        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert "serialize_test" in json_str

    def test_benchmark_multi_trial(self):
        bench = AccuracyBenchmark()
        config = CompareConfig(num_trials=3, seed=42)

        def ref_fn(inputs):
            return np.matmul(inputs["A"], inputs["B"])

        def test_fn(inputs):
            # Simulate f16 precision
            a = inputs["A"].astype(np.float16).astype(np.float32)
            b = inputs["B"].astype(np.float16).astype(np.float32)
            return np.matmul(a, b)

        def input_gen(seed):
            rng = np.random.RandomState(seed)
            return {
                "A": rng.randn(64, 32).astype(np.float32),
                "B": rng.randn(32, 64).astype(np.float32),
            }

        result = bench.benchmark(test_fn, ref_fn, input_gen, "matmul_f16", config)
        assert len(result.trials) == 3
        assert result.aggregate_metrics is not None
        assert result.final_verdict in (Verdict.ACCEPT, Verdict.REVIEW)


# ============================================================
# Reference source tests
# ============================================================

class TestReferenceSources:

    def _build_matmul(self):
        b = KernelBuilder("test_mm")
        b.param("A", [64, 32], "f32")
        b.param("B", [32, 128], "f32")
        m = b.op("matmul", A="A", B="B")
        b.returns(m, [64, 128], "f32")
        return b.build()

    def test_numpy_cpu_source(self):
        ir = self._build_matmul()
        # Default: same-dtype reference (no upcast)
        source = NumPyCPUSource()
        inputs = source.generate_inputs(ir, seed=42)
        assert "A" in inputs and "B" in inputs
        assert inputs["A"].shape == (64, 32)

        ref = source.generate_reference(ir, inputs)
        assert ref.shape == (64, 128)

    def test_numpy_cpu_source_upcast(self):
        """Cross-precision mode: upcast to f64."""
        ir = self._build_matmul()
        source = NumPyCPUSource(compute_dtype="f64")
        inputs = source.generate_inputs(ir, seed=42)
        ref = source.generate_reference(ir, inputs)
        assert ref.shape == (64, 128)

        # f64 reference should be accurate
        expected = np.matmul(inputs["A"].astype(np.float64), inputs["B"].astype(np.float64))
        np.testing.assert_allclose(ref, expected, rtol=1e-10)

    def test_numpy_cpu_input_types(self):
        ir = self._build_matmul()
        source = NumPyCPUSource()

        for input_type in ["normal", "uniform", "edge"]:
            inputs = source.generate_inputs(ir, seed=42, input_type=input_type)
            assert "A" in inputs
            assert inputs["A"].shape == (64, 32)

    def test_get_reference_source(self):
        source = get_reference_source("numpy_cpu")
        assert source.name == "numpy_cpu"

        with pytest.raises(ValueError):
            get_reference_source("nonexistent")

    @pytest.mark.skipif(
        not os.environ.get("ARKE_GPU_TESTS"),
        reason="GPU tests require ARKE_GPU_TESTS=1",
    )
    def test_torch_gpu_source(self):
        ir = self._build_matmul()
        source = TorchGPUSource()

        inputs = source.generate_inputs(ir, seed=42)
        ref = source.generate_reference(ir, inputs)
        assert ref.shape == (64, 128)

        # Compare with NumPy
        numpy_ref = NumPyCPUSource().generate_reference(ir, inputs)
        np.testing.assert_allclose(ref, numpy_ref.astype(np.float32), rtol=1e-5, atol=1e-5)


# ============================================================
# Integration: Accuracy + Arke kernel
# ============================================================

class TestAccuracyIntegration:

    @pytest.mark.skipif(
        not os.environ.get("ARKE_GPU_TESTS"),
        reason="GPU tests require ARKE_GPU_TESTS=1",
    )
    def test_arke_matmul_accuracy_vs_numpy(self):
        """Compare Arke Triton matmul kernel against NumPy reference."""
        import torch

        from arke.backend.triton_backend import TritonBackend
        from arke.engine.env import ArkeEnv

        # Build kernel
        b = KernelBuilder("accuracy_mm")
        b.param("A", [256, 128], "f16")
        b.param("B", [128, 512], "f16")
        m = b.op("matmul", A="A", B="B")
        b.returns(m, [256, 512], "f16")
        ir = b.build()

        # Apply strategy
        env = ArkeEnv(ir, "nvidia_ampere")
        env.apply_decision("tile", {"loop": "i", "factors": [64, 16]}, "t")
        env.apply_decision("tile", {"loop": "j", "factors": [128, 16]}, "t")
        env.apply_decision("tile", {"loop": "k", "factors": [32, 16]}, "t")

        # Generate same-dtype reference (f16 kernel → f16 NumPy reference)
        ref_source = NumPyCPUSource()  # default: same dtype
        inputs_np = ref_source.generate_inputs(ir, seed=42)
        # Compute reference at f16 precision (same as kernel)
        f16_inputs = {k: v.astype(np.float16) for k, v in inputs_np.items()}
        ref_output = ref_source.generate_reference(ir, f16_inputs).astype(np.float32)

        # Run Arke kernel
        backend = TritonBackend()
        source = backend.translate(ir, env.strategy)
        compiled = backend.compile(source)
        assert compiled.success

        gpu_inputs = {}
        for p in ir.params:
            gpu_inputs[p.name] = torch.from_numpy(
                inputs_np[p.name].astype(np.float32)
            ).to(dtype=torch.float16, device="cuda")

        gpu_output = backend.run(compiled, gpu_inputs)
        if isinstance(gpu_output, dict):
            gpu_output = gpu_output["output"]
        test_output = gpu_output.cpu().float().numpy()

        # Accuracy comparison
        bench = AccuracyBenchmark()
        config = DTYPE_CONFIGS["f16"]
        result = bench.compare(test_output, ref_output, "matmul_f16", config)

        # Should be acceptable for f16
        assert result.verdict in (Verdict.ACCEPT, Verdict.REVIEW)
        assert result.metrics.nan_count == 0
        assert result.metrics.inf_count == 0
        assert result.metrics.cosine_similarity > 0.99
