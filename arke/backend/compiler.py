# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Backend — Triton Compiler + Profiler.

Compiles generated Triton source code, executes kernels, and profiles performance.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from arke.backend.base import CompileResult, ProfileResult


class TritonCompiler:
    """Compiles Triton source and runs on GPU."""

    def compile(self, source: str) -> CompileResult:
        """Write source to a temp file and verify it can be imported.

        Returns a CompileResult with the temp module path.
        """
        try:
            # Write source to a temp .py file
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix="arke_triton_",
                delete=False, dir=tempfile.gettempdir(),
            )
            tmp.write(source)
            tmp.flush()
            tmp.close()

            # Attempt to import (validates syntax + imports)
            self._import_module(tmp.name)
            return CompileResult(
                success=True,
                code=source,
                binary_path=tmp.name,
            )
        except Exception as e:
            return CompileResult(
                success=False,
                code=source,
                error=str(e),
            )

    def compile_and_run(
        self,
        source: str,
        input_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compile Triton source, execute, return output tensor."""
        result = self.compile(source)
        if not result.success:
            raise RuntimeError(f"Compilation failed: {result.error}")
        return self.run(result, input_tensors)

    def run(
        self,
        compiled: CompileResult,
        input_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Execute compiled kernel with given inputs.

        The module is expected to have a top-level function whose name
        matches the kernel (not ending with '_kernel'). We detect it
        by scanning module attributes.
        """
        assert compiled.binary_path is not None
        module = self._import_module(compiled.binary_path)
        func = self._find_entry_function(module)

        # Build ordered positional args from input_tensors
        # Convention: matmul expects (A, B), softmax expects (X,)
        tensors = list(input_tensors.values())
        return func(*tensors)

    def profile(
        self,
        source: str,
        input_tensors: dict[str, torch.Tensor],
        warmup: int = 5,
        runs: int = 20,
    ) -> ProfileResult:
        """Benchmark the kernel and return performance metrics."""
        compiled = self.compile(source)
        if not compiled.success:
            raise RuntimeError(f"Compilation failed: {compiled.error}")

        assert compiled.binary_path is not None
        module = self._import_module(compiled.binary_path)
        func = self._find_entry_function(module)
        tensors = list(input_tensors.values())

        # Warmup
        for _ in range(warmup):
            func(*tensors)
        torch.cuda.synchronize()

        # Timed runs
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

        for i in range(runs):
            start_events[i].record()
            func(*tensors)
            end_events[i].record()

        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_ms = sum(times_ms) / len(times_ms)
        latency_us = avg_ms * 1000.0

        # Compute TFLOPS for matmul (2*M*N*K)
        tflops = 0.0
        vs_baseline = None
        shapes = [t.shape for t in tensors]
        if len(tensors) == 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
            M, K = shapes[0]
            _, N = shapes[1]
            flops = 2.0 * M * N * K
            tflops = flops / (avg_ms * 1e-3) / 1e12

            # cuBLAS baseline comparison
            try:
                baseline_ms = self._cublas_baseline(tensors[0], tensors[1], warmup, runs)
                if baseline_ms > 0:
                    vs_baseline = baseline_ms / avg_ms  # >1 means we're faster
            except Exception:
                pass

        return ProfileResult(
            latency_us=latency_us,
            tflops=tflops,
            vs_baseline=vs_baseline,
        )

    # ─── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _import_module(path: str) -> Any:
        """Dynamically import a Python module from a file path."""
        module_name = Path(path).stem
        # Remove from sys.modules to allow re-import of same-name temp files
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _cublas_baseline(
        a: torch.Tensor, b: torch.Tensor, warmup: int, runs: int
    ) -> float:
        """Run cuBLAS matmul baseline and return average time in ms."""
        # Warmup
        for _ in range(warmup):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        # Timed runs
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        for i in range(runs):
            start_events[i].record()
            torch.matmul(a, b)
            end_events[i].record()
        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return sum(times_ms) / len(times_ms)

    @staticmethod
    def _find_entry_function(module: Any) -> Any:
        """Find the entry-point function in a generated Triton module.

        The entry function is a callable whose name does NOT end with '_kernel'
        and is not a private/magic attribute.
        """
        candidates = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and not name.endswith("_kernel"):
                # Skip imported modules/classes
                if not isinstance(obj, type) and not hasattr(obj, "__module__"):
                    candidates.append((name, obj))
                elif hasattr(obj, "__module__") and obj.__module__ == module.__name__:
                    candidates.append((name, obj))
                elif callable(obj) and not isinstance(obj, type):
                    candidates.append((name, obj))

        # Prefer functions defined in this module
        for name, func in candidates:
            if hasattr(func, "__code__"):
                return func

        if candidates:
            return candidates[0][1]

        raise RuntimeError(
            f"No entry function found in module. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )
