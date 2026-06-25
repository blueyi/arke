# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke → torch.compile bridge (D7-E1.4, G8[4b]).

Registers Arke-generated Triton kernels as ``torch.library.custom_op`` so that
``torch.compile``'d models (e.g. transformers GPT-2) can dispatch into Arke
kernels as atomic graph ops, with a ``register_fake`` abstract impl so Dynamo
can trace through them without graph breaks.

⚠️ SCOPE GUARDRAILS (locked — see docs/architecture/arke-harness.md §3.0.3
and docs/phase1/stage8-plan.md "D7-E1.4 scope guardrails"):

  | Dimension       | Constraint                                              |
  |-----------------|---------------------------------------------------------|
  | File location   | THIS single file. Forbidden to leak into lang/ir/       |
  |                 | backend/compiler/agent.                                 |
  | Op count        | ≤3 ops for Phase 1 (currently 2: rmsnorm, matmul).      |
  | Autograd        | Inference-only. register_fake only. NO backward.        |
  | API exposure    | NOT exported from arke.__init__. NOT a .ak feature. NOT |
  |                 | an Agent tool. Explicit opt-in via register_arke_ops(). |
  | Façade status   | NOT part of the Façade contract. Opaque op-handles.     |
  | Lifecycle       | Frozen after G8 PASSes. Phase-boundary: delete/replace. |

This bridge is a **transient Substrate artifact**, not a product capability.
It exists purely to produce BL6 G8[4b] end-to-end evidence (≥1 Arke kernel on
GPT-2's critical path). It routes through the stable ``arke.backend`` Substrate
API (``TritonBackend.lower/compile/run``) — it never imports ``arke.compiler.*``
directly.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Guard so register_arke_ops() is idempotent (custom_op re-registration raises).
_REGISTERED = False

# Cache compiled Arke kernels per (op, shape-key, dtype) so repeated forwards
# don't re-lower/re-compile. Keyed inside the op impls.
_KERNEL_CACHE: dict[tuple, Any] = {}


def _ir_dtype(dt: torch.dtype) -> str:
    return {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }.get(dt, "float16")


def _get_backend():
    """Lazily build a CUDA TritonBackend; None if unavailable (CPU/CI)."""
    if not torch.cuda.is_available():
        return None
    try:
        from arke.backend.triton_backend import TritonBackend
    except Exception as exc:  # pragma: no cover
        logger.debug("torch_bridge: TritonBackend import failed: %s", exc)
        return None
    try:
        return TritonBackend(device="cuda")
    except Exception as exc:  # pragma: no cover
        logger.debug("torch_bridge: TritonBackend init failed: %s", exc)
        return None


def _compile_arke_kernel(op: str, named: dict[str, torch.Tensor]):
    """Lower+compile a single-node Arke IRGraph for ``op``; cache + return kernel.

    Returns None if no real Triton kernel could be produced (caller then falls
    back to the eager reference, so correctness is never compromised).
    """
    from arke.backend.triton_backend import TritonBackend  # noqa: F401  (type)
    from arke.ir.graph import IRGraph, IRNode

    backend = _get_backend()
    if backend is None:
        return None

    first = next(iter(named.values()))
    dtype_str = _ir_dtype(first.dtype)
    shape_key = tuple((k, tuple(v.shape)) for k, v in named.items())
    cache_key = (op, shape_key, dtype_str)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    graph = IRGraph(name=f"bridge_{op}")
    for input_name, tensor in named.items():
        graph.add_input(input_name, dtype=dtype_str, shape=list(tensor.shape))
    graph.add_node(IRNode(
        id="n0", op=op,
        inputs={k: k for k in named.keys()}, outputs=["out"],
        attrs={},
    ))
    graph.set_outputs(["out"])

    try:
        artifact = backend.lower(graph)
        if artifact.metadata.get("num_real_kernels", 0) != 1:
            return None
        kernel = backend.compile(artifact)
        if not kernel.success:
            return None
    except Exception as exc:
        logger.debug("torch_bridge: compile(%s) failed: %s", op, exc)
        return None

    _KERNEL_CACHE[cache_key] = (backend, kernel)
    return _KERNEL_CACHE[cache_key]


def _run_arke(op: str, named: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Run the Arke Triton kernel for ``op`` on ``named`` inputs; None on miss."""
    bk = _compile_arke_kernel(op, named)
    if bk is None:
        return None
    backend, kernel = bk
    try:
        out = backend.run(kernel, named)
    except Exception as exc:
        logger.debug("torch_bridge: run(%s) failed: %s", op, exc)
        return None
    if isinstance(out, dict):
        out = out.get("out")
    return out if isinstance(out, torch.Tensor) else None


# ── eager references (correctness fallback + fake-impl shape source) ──────────

def _eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).to(dt)


def register_arke_ops() -> bool:
    """Register the Arke custom ops into the ``arke::`` torch library.

    Idempotent. Returns True if ops are registered and a CUDA backend is
    available (so torch.compile can dispatch into real Arke kernels), False
    otherwise (CPU/CI — ops still register but always use the eager fallback).

    Explicit opt-in ONLY — never auto-runs on import (scope guardrail).
    """
    global _REGISTERED
    if _REGISTERED:
        return torch.cuda.is_available()

    # ---- arke::rmsnorm(x, weight, eps) -> Tensor --------------------------
    @torch.library.custom_op("arke::rmsnorm", mutates_args=())
    def arke_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        out = _run_arke("rmsnorm", {"X": x.contiguous(), "W": weight.contiguous()})
        if out is None:
            return _eager_rmsnorm(x, weight, eps)
        return out

    @arke_rmsnorm.register_fake
    def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # Abstract impl for Dynamo trace: rmsnorm is shape-preserving on x.
        return torch.empty_like(x)

    # ---- arke::matmul(a, b) -> Tensor -------------------------------------
    @torch.library.custom_op("arke::matmul", mutates_args=())
    def arke_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = _run_arke("matmul", {"A": a.contiguous(), "B": b.contiguous()})
        if out is None:
            return torch.matmul(a, b)
        return out

    @arke_matmul.register_fake
    def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Abstract impl: (M,K) @ (K,N) -> (M,N).
        return a.new_empty((a.shape[0], b.shape[1]))

    _REGISTERED = True
    logger.info("torch_bridge: registered arke::rmsnorm, arke::matmul")
    return torch.cuda.is_available()


# Public op handles (after registration, callable as torch.ops.arke.*)
def arke_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Call the registered arke::rmsnorm op (registers on first use)."""
    register_arke_ops()
    return torch.ops.arke.rmsnorm(x, weight, eps)


def arke_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Call the registered arke::matmul op (registers on first use)."""
    register_arke_ops()
    return torch.ops.arke.matmul(a, b)
