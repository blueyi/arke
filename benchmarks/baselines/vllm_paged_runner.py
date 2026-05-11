# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P2: vLLM paged_attention baseline.

Covers ``paged_attention``. vLLM is a heavy dependency (~5GB); ``available``
flips False on ImportError so the ladder falls through to PyTorch-eager.

Upstream: https://github.com/vllm-project/vllm
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
_PAGED_ATTN_MOD = None
try:
    import vllm.attention.ops.paged_attn as _paged  # type: ignore[import-not-found]

    _PAGED_ATTN_MOD = _paged
    _AVAILABLE = True
except Exception:
    pass


_SUPPORTED_OPS = frozenset({"paged_attention"})


@register_baseline
class VLLMPagedRunner(BaselineRunner):
    """P2: vLLM — paged-attention kernel."""

    @property
    def name(self) -> str:
        return "vLLM"

    @property
    def priority(self) -> int:
        return 2

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            from importlib.metadata import version
            v = version("vllm")
        except Exception:
            pass
        return (
            f"vLLM {v} (vllm-project) | "
            "https://github.com/vllm-project/vllm | License: Apache-2.0"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in _SUPPORTED_OPS

    def run_for_output(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if not self.available or not self.supports(op):
            return None
        if _PAGED_ATTN_MOD is None:
            return None

        # vLLM's paged_attention has a complex KV-cache layout (block tables,
        # block_size, head_mapping). Callers must pass the full kwarg set;
        # we forward and let vLLM validate. If the caller didn't supply the
        # required structure, return None so the ladder falls through.
        op_fn = getattr(_PAGED_ATTN_MOD, "PagedAttention", None) or getattr(
            _PAGED_ATTN_MOD, "paged_attention_v1", None
        )
        if op_fn is None:
            return None
        try:
            return op_fn(*inputs, **kwargs)
        except Exception as e:
            logger.warning("vLLM paged_attention invocation failed: %s", e)
            return None

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        # paged_attention requires KV-cache structure that doesn't fit the
        # zero-arg get_fn API; the Golden Kernel hook is the canonical entry.
        return None
