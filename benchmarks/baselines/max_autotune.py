# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Demo B (D8-X2) — new baseline runner onboarding.

`MaxAutotuneRunner` wraps ``torch.compile(mode="max-autotune")``, a baseline
source distinct from the existing P4 ``InductorRunner`` (which uses
``mode="reduce-overhead"``). ``max-autotune`` enables Inductor's autotuning
Triton template search + CUDA-graph capture, producing different (often faster
for compute-bound ops) kernels than the reduce-overhead path. It is therefore a
legitimate *alternate Triton-implementation source* under the Same-Backend
Triton Fairness rule, and a clean falsifiable demonstration that a new
``BaselineRunner`` subclass onboards within the ≤200 LOC Tier-1 budget without
touching the harness core.

This is the Stage 8 Tier-1 [HARNESS-3] Extensibility Demo B artifact. See
`docs/architecture/arke-harness.md` §"Onboarding a new BaselineRunner" and
`docs/phase1/stage8-plan.md` "Demo B — New baseline runner onboarding".
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

# Ops this runner wraps. Kept to a focused, well-tested set so the demo stays
# inside the LOC budget while still spanning OT0 (elementwise), OT1 (norm),
# and OT2 (dense matmul) — enough for BL1 + BL3 cross-coverage.
_SUPPORTED: frozenset[str] = frozenset(
    {"matmul", "softmax", "layernorm", "rmsnorm", "relu", "gelu", "silu"}
)


def _compiled(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Compile a fn with Inductor max-autotune (autotuned Triton + CUDA graphs)."""
    return torch.compile(fn, mode="max-autotune")


def _build(op: str) -> Callable[..., torch.Tensor] | None:
    """Return a compiled callable for ``op`` (inputs passed at call time)."""
    if op == "matmul":
        return _compiled(lambda a, b: torch.matmul(a, b))
    if op == "softmax":
        return _compiled(lambda x: torch.nn.functional.softmax(x, dim=-1))
    if op == "layernorm":
        return _compiled(
            lambda x: torch.nn.functional.layer_norm(
                x,
                [x.shape[-1]],
                torch.ones(x.shape[-1], device=x.device, dtype=x.dtype),
                torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype),
            )
        )
    if op == "rmsnorm":
        return _compiled(
            lambda x: (
                x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            )
            * torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        )
    if op == "relu":
        return _compiled(lambda x: torch.nn.functional.relu(x))
    if op == "gelu":
        return _compiled(lambda x: torch.nn.functional.gelu(x))
    if op == "silu":
        return _compiled(lambda x: torch.nn.functional.silu(x))
    return None


@register_baseline
class MaxAutotuneRunner(BaselineRunner):
    """P4: torch.compile(mode='max-autotune') — autotuned Inductor/Triton."""

    @property
    def name(self) -> str:
        return "torch.compile-max-autotune"

    @property
    def priority(self) -> int:
        # Same P4 tier as InductorRunner (both are torch.compile sources);
        # ordered after the reduce-overhead variant in the ladder.
        return 4

    @property
    def source(self) -> str:
        return (
            f"torch.compile (Inductor, mode=max-autotune) via PyTorch "
            f"{torch.__version__} | https://pytorch.org | License: BSD-3-Clause"
        )

    @property
    def available(self) -> bool:
        return torch.cuda.is_available() and hasattr(torch, "compile")

    def supports(self, op: str) -> bool:
        return op in _SUPPORTED

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        compiled = _build(op)
        if compiled is None:
            return None
        if op == "matmul":
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)
            compiled(a, b)  # warm autotune + graph capture
            torch.cuda.synchronize()
            return lambda: compiled(a, b)
        # all other supported ops are unary M×N
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        compiled(x)
        torch.cuda.synchronize()
        return lambda: compiled(x)

    def run_for_output(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | None:
        """Golden-hook: compute the output on the given inputs (correctness)."""
        compiled = _build(op)
        if compiled is None:
            return None
        if op == "matmul" and len(inputs) == 2:
            out = compiled(inputs[0], inputs[1])
        elif op in _SUPPORTED and len(inputs) == 1:
            out = compiled(inputs[0])
        else:
            return None
        torch.cuda.synchronize()
        return out
