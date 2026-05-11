# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for BaselineRunner.run_for_output (Golden Kernel hook).

The hook is a non-breaking addition: every registered runner inherits a
default that delegates to ``run_with_inputs``. Tests below assert the API
is uniformly exposed and behaves contract-wise.
"""

from __future__ import annotations

import pytest

# Trigger registration of all known runners.
import benchmarks.baselines.arke_runner  # noqa: F401
import benchmarks.baselines.cublas  # noqa: F401
import benchmarks.baselines.inductor  # noqa: F401
import benchmarks.baselines.liger  # noqa: F401
import benchmarks.baselines.llm_direct  # noqa: F401
import benchmarks.baselines.pytorch_eager  # noqa: F401
import benchmarks.baselines.triton_tutorial  # noqa: F401

try:  # FlagGems is optional
    import benchmarks.baselines.flaggems  # noqa: F401
except Exception:
    pass

from benchmarks.baselines.base import BaselineRunner, _REGISTRY


def test_run_for_output_method_exists_on_every_registered_runner():
    """API guard: every runner class exposes ``run_for_output``."""
    assert _REGISTRY, "no baseline runners registered"
    for cls in _REGISTRY:
        assert hasattr(cls, "run_for_output"), (
            f"{cls.__name__} missing run_for_output (Golden Kernel hook)"
        )
        # Must be a real method, not a forgotten attribute placeholder.
        assert callable(getattr(cls, "run_for_output"))


def test_run_for_output_default_delegates_to_run_with_inputs():
    """The default implementation defers to ``run_with_inputs``.

    Verified without GPU: a stub subclass overriding ``run_with_inputs``
    must observe its overridden return value flowing through
    ``run_for_output`` unchanged.
    """
    sentinel = object()

    class StubRunner(BaselineRunner):
        name = "stub"  # type: ignore[assignment]
        priority = 9  # type: ignore[assignment]
        source = "test"  # type: ignore[assignment]
        available = True  # type: ignore[assignment]

        def supports(self, op: str) -> bool:
            return op == "noop"

        def get_fn(self, op, M, N, K=0, dtype=None):  # type: ignore[override]
            return None

        def run_with_inputs(self, op, *inputs, **kwargs):  # type: ignore[override]
            return sentinel  # type: ignore[return-value]

    runner = StubRunner()
    assert runner.run_for_output("noop") is sentinel


def test_run_for_output_default_returns_none_when_not_implemented():
    """Runners that never override ``run_with_inputs`` get ``None`` (= unsupported)."""

    class BareRunner(BaselineRunner):
        name = "bare"  # type: ignore[assignment]
        priority = 9  # type: ignore[assignment]
        source = "test"  # type: ignore[assignment]
        available = True  # type: ignore[assignment]

        def supports(self, op: str) -> bool:
            return False

        def get_fn(self, op, M, N, K=0, dtype=None):  # type: ignore[override]
            return None

    assert BareRunner().run_for_output("nope") is None
