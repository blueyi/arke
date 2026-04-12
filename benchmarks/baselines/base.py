# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""BaselineRunner ABC and registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


class BaselineRunner(ABC):
    """Base class for all baseline implementations.

    Each runner wraps one source of GPU kernels (cuBLAS, FlagGems, etc.)
    and knows which operators it supports.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g. 'cuBLAS', 'FlagGems')."""

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority tier: P0=0 (vendor) .. P5=5 (LLM-direct)."""

    @property
    @abstractmethod
    def source(self) -> str:
        """Provenance: package name, version, and origin URL."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this baseline's dependencies are installed."""

    @abstractmethod
    def supports(self, op: str) -> bool:
        """Whether this runner supports the given op."""

    @abstractmethod
    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        """Return a zero-arg callable that runs the kernel.

        The callable must:
        1. Use pre-allocated inputs on GPU
        2. Return the output tensor
        3. Be safe to call repeatedly (no side effects)

        Returns None if the op/shape is not supported.

        Args:
            op: Operator name (matmul, softmax, layernorm, etc.)
            M, N: Primary dimensions
            K: Secondary dimension (for matmul: inner dim)
            dtype: Data type
        """

    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Optional correctness-oriented execution hook.

        This hook lets benchmarks feed fixed inputs into a runner so
        correctness can be measured independently of the zero-arg timing API.

        Default behavior is opt-out: return ``None`` to indicate unsupported.
        Existing performance benchmarks continue using ``get_fn()`` unchanged.
        """
        return None

    def __repr__(self) -> str:
        return f"{self.name}(P{self.priority})"


# ── Registry ────────────────────────────────────────────────

_REGISTRY: list[type[BaselineRunner]] = []


def register_baseline(cls: type[BaselineRunner]) -> type[BaselineRunner]:
    """Decorator to register a baseline runner class."""
    _REGISTRY.append(cls)
    return cls


def get_all_runners() -> list[BaselineRunner]:
    """Instantiate all registered runners (available ones only)."""
    runners = []
    for cls in _REGISTRY:
        try:
            runner = cls()
            if runner.available:
                runners.append(runner)
            else:
                logger.info(f"Baseline {runner.name} not available (deps missing)")
        except Exception as e:
            logger.warning(f"Failed to instantiate {cls.__name__}: {e}")
    runners.sort(key=lambda r: r.priority)
    return runners


def get_runners_for_op(op: str) -> list[BaselineRunner]:
    """Get all available runners that support a given op."""
    return [r for r in get_all_runners() if r.supports(op)]
