# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P2: FlashMLA baseline (DeepSeek-AI).

Covers ``multi_latent_attention``. FlashMLA targets sm_90+ (Hopper); on
older devices it either fails to import or produces incorrect results, so
``available`` enforces both pip-importability AND a CC>=9.0 check. When
unavailable the Golden Kernel ladder falls through to PyTorch-eager and
``bench_l1`` emits an ``mla_golden_degraded=true`` audit row.

Upstream: https://github.com/deepseek-ai/FlashMLA
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
try:
    import flashmla  # type: ignore[import-not-found]  # noqa: F401

    _AVAILABLE = True
except Exception:
    pass


_SUPPORTED_OPS = frozenset({"multi_latent_attention"})


def _has_hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cc = torch.cuda.get_device_capability()
        return cc >= (9, 0)
    except Exception:
        return False


@register_baseline
class FlashMLARunner(BaselineRunner):
    """P2: FlashMLA — DeepSeek's MLA kernel for Hopper GPUs."""

    @property
    def name(self) -> str:
        return "FlashMLA"

    @property
    def priority(self) -> int:
        return 2

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            from importlib.metadata import version
            v = version("flashmla")
        except Exception:
            pass
        return (
            f"FlashMLA {v} (DeepSeek-AI) | "
            "https://github.com/deepseek-ai/FlashMLA | License: MIT"
        )

    @property
    def available(self) -> bool:
        # Both conditions must hold: package installed AND Hopper+ device.
        # On sm<9.0 (e.g. RTX 3060 sm_86) we deliberately report False so the
        # ladder falls through; bench_l1 marks the row mla_golden_degraded.
        return _AVAILABLE and _has_hopper_or_newer()

    def supports(self, op: str) -> bool:
        return op in _SUPPORTED_OPS

    def run_for_output(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        # Defensive: if hardware can't run FlashMLA, return None so the
        # ladder falls through. The audit annotation happens in bench_l1.
        if not self.available or not self.supports(op):
            return None

        from flashmla import flash_mla_with_kvcache  # type: ignore[import-not-found]

        if op == "multi_latent_attention" and len(inputs) >= 2:
            q = inputs[0]
            kv_c = inputs[1]
            try:
                return flash_mla_with_kvcache(q, kv_c, **kwargs)
            except Exception as e:
                logger.warning("FlashMLA invocation failed: %s", e)
                return None
        return None

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        # Timing API not implemented for L1 perf — Golden Kernel hook is the
        # canonical entry point. Return None on sm<9.0 or when uninstalled.
        return None
