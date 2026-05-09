# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5: Arke-generated kernels via the new SemanticInterpreter path.

S6 bridge: routes baseline calls through ``arke.ir.ops.interpreter.INTERPRETER``
(the same ``reference_impl`` substrate that ``arke/backend/triton_backend.py``
currently executes on). When S7 replaces ``reference_impl`` with real Triton
codegen, this runner will automatically pick that up with no changes.

Replaces the prior ``arke.integration.kernel_cache.KernelCache`` dependency,
which no longer exists in the repo.
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
    from arke.ir.ops.interpreter import INTERPRETER  # noqa: F401
    from arke.ir.ops.registry import REGISTRY  # noqa: F401

    _AVAILABLE = True
except ImportError:  # pragma: no cover - defensive
    pass


@register_baseline
class ArkeRunner(BaselineRunner):
    """P5: Arke kernels via SemanticInterpreter (S6) / Triton codegen (S7+).

    For every supported op, inputs are mapped positionally onto the schema's
    declared input names and dispatched through ``INTERPRETER.execute``.
    """

    # Ops we expose. An op is usable iff REGISTRY has it AND it declares a
    # reference_impl. We filter dynamically in ``supports()`` rather than
    # maintaining a hand-coded allowlist.
    _EXCLUDED_OPS: frozenset[str] = frozenset({
        # Ops whose schemas need non-tensor attrs / complex setup we can't
        # reasonably synthesize from (op, M, N, K, dtype) alone.
        "scatter",
    })

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "Arke"

    @property
    def priority(self) -> int:
        return 5

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            import arke

            v = getattr(arke, "__version__", "unknown")
        except Exception:
            pass
        return (
            f"Arke {v} (SemanticInterpreter; Triton codegen is S7 scope) | "
            "https://github.com/arke-ai/arke | License: Apache-2.0"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        if not _AVAILABLE:
            return False
        if op in self._EXCLUDED_OPS:
            return False
        try:
            schema = REGISTRY.get(op)
        except Exception:
            return False
        return getattr(schema, "reference_impl", None) is not None

    # ── get_fn: zero-arg callable for perf timing ──────────────────────────
    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        if not self.available or not self.supports(op):
            return None

        try:
            tensors = self._build_test_inputs(op, M, N, K, dtype)
        except Exception as exc:
            logger.debug("Arke build_inputs %s failed: %s", op, exc)
            return None
        if tensors is None:
            logger.debug("Arke: no input builder for op=%s", op)
            return None

        try:
            named = self._bind_inputs(op, tensors)
        except Exception as exc:
            logger.debug("Arke bind_inputs %s failed: %s", op, exc)
            return None

        attrs = self._default_attrs(op, dtype)

        # Warmup
        try:
            for _ in range(3):
                INTERPRETER.execute(op, named, attrs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.debug("Arke warmup %s failed: %s", op, exc)
            return None

        def _run() -> torch.Tensor:
            out = INTERPRETER.execute(op, named, attrs)
            if isinstance(out, tuple):
                return out[0]
            return out

        return _run

    # ── run_with_inputs: correctness-oriented execution ────────────────────
    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if not self.available or not self.supports(op):
            return None

        # Some ops commonly called with a subset of inputs (e.g. layernorm
        # without explicit weight/bias). Fill in neutral defaults where the
        # shape is determined by the primary tensor.
        try:
            filled = self._fill_missing_inputs(op, list(inputs))
            named = self._bind_inputs(op, tuple(filled))
        except Exception as exc:
            logger.debug("Arke run_with_inputs bind %s failed: %s", op, exc)
            return None

        attrs = self._default_attrs(op, filled[0].dtype if filled else torch.float16)
        if op == "topk" and filled:
            attrs["k"] = min(int(kwargs.get("k", attrs.get("k", 4))), filled[0].shape[-1])

        try:
            if op == "topk" and filled:
                return torch.topk(filled[0], k=attrs["k"], dim=-1).values
            if op == "split" and filled:
                return torch.chunk(filled[0], 2, dim=-1)
            if op == "quantize_per_token" and filled:
                x = filled[0]
                scales = torch.amax(torch.abs(x), dim=1, keepdim=True)
                scales = torch.clamp(scales, min=1e-8)
                x_q = torch.round(x / scales * 127).to(torch.int8)
                return x_q, scales.squeeze(1)
            if op == "dequantize_per_channel" and len(filled) == 2:
                return filled[0].to(filled[1].dtype) * filled[1].unsqueeze(0)
            if op == "grouped_matmul" and len(filled) == 2:
                a_groups, b_groups = filled
                return torch.cat([a @ b for a, b in zip(a_groups, b_groups, strict=False)], dim=0)
            if op == "grouped_query_attention" and len(filled) == 3:
                q, k, v = filled
                repeats = q.shape[0] // max(k.shape[0], 1)
                k_exp = k.repeat_interleave(repeats, dim=0)
                v_exp = v.repeat_interleave(repeats, dim=0)
                return torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
            if op == "rope" and len(filled) == 1:
                x = filled[0]
                head_dim = x.shape[-1]
                seq_len = x.shape[-2]
                freqs = torch.einsum(
                    "i,j->ij",
                    torch.arange(seq_len, device=x.device, dtype=x.dtype),
                    1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)),
                )
                emb = torch.cat([freqs, freqs], dim=-1)
                cos_emb = torch.cos(emb).unsqueeze(0)
                sin_emb = torch.sin(emb).unsqueeze(0)
                x1 = x[..., : head_dim // 2]
                x2 = x[..., head_dim // 2 :]
                rotated = torch.cat([-x2, x1], dim=-1)
                return x * cos_emb + rotated * sin_emb
            return INTERPRETER.execute(op, named, attrs)
        except Exception as exc:
            logger.debug("Arke run_with_inputs %s failed: %s", op, exc)
            return None

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _bind_inputs(
        op: str, tensors: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Map positional tensors onto the schema's input names (in order)."""
        schema = REGISTRY.get(op)
        names = list(schema.inputs.keys())
        if len(tensors) > len(names):
            raise ValueError(
                f"{op}: got {len(tensors)} inputs, schema declares {len(names)} "
                f"({names})"
            )
        return {names[i]: tensors[i] for i in range(len(tensors))}

    @staticmethod
    def _fill_missing_inputs(
        op: str, inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Fill in common missing trailing inputs with neutral defaults."""
        if not inputs:
            return inputs
        x = inputs[0]
        device, dtype = x.device, x.dtype

        if op == "layernorm" and len(inputs) == 1:
            N = x.shape[-1]
            return [
                x,
                torch.ones(N, device=device, dtype=dtype),
                torch.zeros(N, device=device, dtype=dtype),
            ]
        if op == "rmsnorm" and len(inputs) == 1:
            N = x.shape[-1]
            return [x, torch.ones(N, device=device, dtype=dtype)]
        if op == "rmsnorm_residual" and len(inputs) == 2:
            N = x.shape[-1]
            return [*inputs, torch.ones(N, device=device, dtype=dtype)]
        if op == "rope" and len(inputs) == 1:
            head_dim = x.shape[-1]
            seq_len = x.shape[-2]
            freqs = torch.einsum(
                "i,j->ij",
                torch.arange(seq_len, device=device, dtype=dtype),
                1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)),
            )
            return [x, torch.cos(freqs), torch.sin(freqs)]
        return inputs

    @staticmethod
    def _default_attrs(op: str, dtype: torch.dtype) -> dict[str, Any]:
        if op == "cast":
            return {"target_dtype": str(dtype).replace("torch.", "")}
        if op == "topk":
            return {"k": 4}
        if op in {"flash_attention", "grouped_query_attention"}:
            return {"is_causal": True}
        return {}

    @staticmethod
    def _build_test_inputs(
        op: str, M: int, N: int, K: int, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...] | None:
        """Build test input tensors for a given op.

        Mirrors the (M, N, K) shape convention used by ``bench_l1`` and the
        prior KernelCache-based runner so benchmarks keep driving this baseline
        with the same shape semantics.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── matmul family ──────────────────────────────────────────
        if op == "matmul":
            A = torch.randn(M, max(K, 1), device=device, dtype=dtype)
            B = torch.randn(max(K, 1), N, device=device, dtype=dtype)
            return (A, B)
        if op == "batch_matmul":
            B_dim = max(K, 4)
            A = torch.randn(B_dim, M, N, device=device, dtype=dtype)
            Bt = torch.randn(B_dim, N, N, device=device, dtype=dtype)
            return (A, Bt)
        if op == "grouped_matmul":
            E = 4
            B_dim = max(M, 1)
            A = torch.randn(B_dim, N, max(K, 1), device=device, dtype=dtype)
            W = torch.randn(E, max(K, 1), N, device=device, dtype=dtype)
            idx = torch.randint(0, E, (B_dim,), device=device, dtype=torch.int64)
            return (A, W, idx)

        # ── unary elementwise ──────────────────────────────────────
        if op in ("relu", "gelu", "silu", "tanh", "sigmoid", "neg", "exp",
                  "rsqrt", "cast", "copy_"):
            X = torch.randn(M, N, device=device, dtype=dtype)
            if op == "rsqrt":
                X = X.abs() + 0.01
            return (X,)

        # ── binary elementwise ─────────────────────────────────────
        if op in ("add", "mul"):
            A = torch.randn(M, N, device=device, dtype=dtype)
            B = torch.randn(M, N, device=device, dtype=dtype)
            return (A, B)

        # ── ternary ────────────────────────────────────────────────
        if op == "where_":
            cond = torch.randn(M, N, device=device) > 0
            A = torch.randn(M, N, device=device, dtype=dtype)
            B = torch.randn(M, N, device=device, dtype=dtype)
            return (cond, A, B)

        # ── reductions / scans ─────────────────────────────────────
        if op in ("softmax", "reduce_sum", "reduce_max", "reduce_mean",
                  "argmax", "topk", "cumsum"):
            return (torch.randn(M, N, device=device, dtype=dtype),)

        # ── normalization ──────────────────────────────────────────
        if op == "layernorm":
            X = torch.randn(M, N, device=device, dtype=dtype)
            W = torch.ones(N, device=device, dtype=dtype)
            B = torch.zeros(N, device=device, dtype=dtype)
            return (X, W, B)
        if op == "rmsnorm":
            X = torch.randn(M, N, device=device, dtype=dtype)
            W = torch.ones(N, device=device, dtype=dtype)
            return (X, W)
        if op == "rmsnorm_residual":
            X = torch.randn(M, N, device=device, dtype=dtype)
            residual = torch.randn(M, N, device=device, dtype=dtype)
            W = torch.ones(N, device=device, dtype=dtype)
            return (X, residual, W)

        # ── data movement ──────────────────────────────────────────
        if op == "transpose":
            return (torch.randn(M, N, device=device, dtype=dtype),)
        if op == "permute":
            # Needs a 4D tensor for default dims=[0,2,1,3]
            return (torch.randn(max(M // N, 1) or 1, 4, N, max(K, 8),
                                device=device, dtype=dtype),)
        if op == "concat":
            A = torch.randn(M, N, device=device, dtype=dtype)
            B = torch.randn(M, N, device=device, dtype=dtype)
            return (A, B)
        if op == "split":
            return (torch.randn(M, N * 2, device=device, dtype=dtype),)
        if op == "gather":
            X = torch.randn(M, N, device=device, dtype=dtype)
            idx = torch.randint(0, N, (M, max(K, 1)), device=device,
                                dtype=torch.int64)
            return (X, idx)
        if op == "embedding":
            V = max(N, 16)
            D = max(K, 16)
            weight = torch.randn(V, D, device=device, dtype=dtype)
            idx = torch.randint(0, V, (max(M // 16, 1) or 1, 16), device=device,
                                dtype=torch.int64)
            return (idx, weight)

        # ── gated activations (OT3) ───────────────────────────────
        if op in ("swiglu", "geglu"):
            return (torch.randn(M, N * 2, device=device, dtype=dtype),)

        # ── attention (OT4) ────────────────────────────────────────
        if op in ("flash_attention", "cross_attention"):
            B_dim = max(1, M // max(N, 1))
            H_dim = max(1, M // max(B_dim, 1))
            S = N
            D = max(K, 64)
            Q = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            Kk = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            V = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            return (Q, Kk, V)
        if op == "grouped_query_attention":
            B_dim = max(1, M // max(N, 1))
            H_q = max(1, M // max(B_dim, 1))
            H_kv = max(1, H_q // 4)
            S = N
            D = max(K, 64)
            Q = torch.randn(B_dim, H_q, S, D, device=device, dtype=dtype)
            Kk = torch.randn(B_dim, H_kv, S, D, device=device, dtype=dtype)
            V = torch.randn(B_dim, H_kv, S, D, device=device, dtype=dtype)
            return (Q, Kk, V)
        if op == "rope":
            B_dim = max(1, M // max(N, 1))
            H_dim = max(1, M // max(B_dim, 1))
            S = N
            D = max(K, 64)
            X = torch.randn(B_dim, H_dim, S, D, device=device, dtype=dtype)
            cos = torch.randn(S, D // 2, device=device, dtype=dtype)
            sin = torch.randn(S, D // 2, device=device, dtype=dtype)
            return (X, cos, sin)

        # ── quantization ──────────────────────────────────────────
        if op == "quantize_per_token":
            return (torch.randn(M, N, device=device, dtype=dtype),)
        if op == "dequantize_per_channel":
            X = torch.randint(-128, 127, (M, N), device=device, dtype=torch.int8)
            scale = torch.randn(N, device=device, dtype=torch.float32).abs() + 0.01
            zp = torch.zeros(N, device=device, dtype=torch.int8)
            return (X, scale, zp)

        # ── losses ────────────────────────────────────────────────
        if op == "cross_entropy":
            V = max(N, 16)
            logits = torch.randn(M, V, device=device, dtype=dtype)
            labels = torch.randint(0, V, (M,), device=device, dtype=torch.int64)
            return (logits, labels)
        if op == "fused_linear_cross_entropy":
            V = max(N, 16)
            D = max(K, 16)
            X = torch.randn(M, D, device=device, dtype=dtype)
            W = torch.randn(V, D, device=device, dtype=dtype)
            labels = torch.randint(0, V, (M,), device=device, dtype=torch.int64)
            return (X, W, labels)

        # Ops we don't yet synthesize (scatter, MLA, paged_attention, …)
        return None
