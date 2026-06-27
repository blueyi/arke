# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P1: FlagGems baselines — 200+ Triton operators."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

_AVAILABLE = False
_ENABLED = False
try:
    import flag_gems  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


def _ensure_enabled() -> None:
    """Enable FlagGems globally (once). Persistent, no cleanup needed."""
    global _ENABLED
    if _ENABLED:
        return
    import flag_gems

    flag_gems.enable()
    _ENABLED = True


@register_baseline
class FlagGemsRunner(BaselineRunner):
    """P1: FlagGems Triton operator library."""

    @property
    def name(self) -> str:
        return "FlagGems"

    @property
    def priority(self) -> int:
        return 1

    @property
    def source(self) -> str:
        v = "unknown"
        try:
            import flag_gems
            v = getattr(flag_gems, "__version__", "unknown")
        except Exception:
            pass
        return (
            f"FlagGems {v} (BAAI/FlagOS) | "
            "https://github.com/flagos-ai/FlagGems | License: Apache-2.0"
        )

    @property
    def available(self) -> bool:
        return _AVAILABLE and torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        # FlagGems overrides ATen dispatch — supports most standard ops
        return op in (
            "matmul", "softmax", "layernorm", "rmsnorm",
            "relu", "gelu", "silu", "dropout",
            # OT0 elementwise (via ATen override)
            "tanh", "sigmoid", "add", "mul", "neg", "exp", "rsqrt",
            "where_", "cast",
            # OT1 reduction
            "reduce_sum", "reduce_max", "reduce_mean", "cumsum",
            "argmax", "topk",
            # OT2 data movement & dense
            "batch_matmul", "transpose", "embedding",
            "grouped_matmul", "concat", "gather", "scatter", "copy_",
            # OT3 quantization
            "quantize_per_token",
            # OT4 attention (S7.followup.3 2026-06-06)
            # FlagGems ships Triton-only SDPA + flash_attention kernels.
            # When enabled, F.scaled_dot_product_attention dispatches
            # through FlagGems Triton — making FlagGems the same-backend
            # Triton golden for these 3 ops. MLA + paged_attention stay
            # audit-degraded (no production Triton kernel in 9 audited
            # community libraries; PyTorch-eager remains P3 fallback).
            "flash_attention", "grouped_query_attention", "cross_attention",
        )

    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        _ensure_enabled()
        if op == "matmul" and len(inputs) == 2:
            return torch.matmul(inputs[0], inputs[1])
        if op == "softmax" and len(inputs) == 1:
            return torch.nn.functional.softmax(inputs[0], dim=-1)
        if op == "layernorm" and len(inputs) == 1:
            x = inputs[0]
            weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            bias = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
            return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias)
        if op == "rmsnorm" and len(inputs) == 1:
            x = inputs[0]
            weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            eps = 1e-6
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        if op == "relu" and len(inputs) == 1:
            return torch.relu(inputs[0])
        if op == "gelu" and len(inputs) == 1:
            return torch.nn.functional.gelu(inputs[0])
        if op == "silu" and len(inputs) == 1:
            return torch.nn.functional.silu(inputs[0])
        if op == "tanh" and len(inputs) == 1:
            return torch.tanh(inputs[0])
        if op == "sigmoid" and len(inputs) == 1:
            return torch.sigmoid(inputs[0])
        if op == "add" and len(inputs) == 2:
            return inputs[0] + inputs[1]
        if op == "mul" and len(inputs) == 2:
            return inputs[0] * inputs[1]
        if op == "neg" and len(inputs) == 1:
            return -inputs[0]
        if op == "exp" and len(inputs) == 1:
            return torch.exp(inputs[0])
        if op == "rsqrt" and len(inputs) == 1:
            return torch.rsqrt(inputs[0])
        if op == "where_" and len(inputs) == 3:
            return torch.where(inputs[0].bool(), inputs[1], inputs[2])
        if op == "cast" and len(inputs) == 1:
            return inputs[0].to(torch.float32)
        if op == "reduce_sum" and len(inputs) == 1:
            return inputs[0].sum(dim=-1)
        if op == "reduce_max" and len(inputs) == 1:
            return inputs[0].max(dim=-1).values
        if op == "reduce_mean" and len(inputs) == 1:
            return inputs[0].mean(dim=-1)
        if op == "cumsum" and len(inputs) == 1:
            return torch.cumsum(inputs[0], dim=-1)
        if op == "batch_matmul" and len(inputs) == 2:
            return torch.bmm(inputs[0], inputs[1])
        if op == "transpose" and len(inputs) == 1:
            return inputs[0].T.contiguous()
        if op == "embedding" and len(inputs) == 2:
            return torch.nn.functional.embedding(inputs[0].long(), inputs[1])
        # OT4 attention via FlagGems Triton SDPA (hijacked F.scaled_dot_product_attention).
        # FlagGems supplies a Triton-only implementation; when flag_gems.enable()
        # has run, calls to torch.nn.functional.scaled_dot_product_attention
        # dispatch through it. This makes FlagGems the same-backend Triton
        # golden for flash_attention / GQA / cross_attention (S7.followup.3).
        if op == "flash_attention" and len(inputs) == 3:
            return torch.nn.functional.scaled_dot_product_attention(
                inputs[0], inputs[1], inputs[2], is_causal=True,
            )
        if op == "cross_attention" and len(inputs) == 3:
            return torch.nn.functional.scaled_dot_product_attention(
                inputs[0], inputs[1], inputs[2],
            )
        if op == "grouped_query_attention" and len(inputs) == 3:
            q, k, v = inputs
            # GQA: Q has more heads than K/V; broadcast K/V to match
            # via PyTorch's native enable_gqa flag (SDPA-supported since
            # 2.5) so FlagGems sees the canonical SDPA call shape.
            repeats = q.shape[0] // max(k.shape[0], 1)
            if repeats > 1:
                k = k.repeat_interleave(repeats, dim=0)
                v = v.repeat_interleave(repeats, dim=0)
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True,
            )
        return None

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        # FlagGems registers as ATen backend globally. Once enabled,
        # all torch ops dispatch through FlagGems Triton kernels.
        _ensure_enabled()

        if op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)
            # Pre-warm to trigger Triton compilation
            torch.matmul(A, B)
            torch.cuda.synchronize()
            return lambda: torch.matmul(A, B)

        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.softmax(X, dim=-1)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.softmax(X, dim=-1)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.ones(N, device="cuda", dtype=dtype)
            bias = torch.zeros(N, device="cuda", dtype=dtype)
            torch.nn.functional.layer_norm(X, [N], weight, bias)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.layer_norm(X, [N], weight, bias)

        elif op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.relu(X)
            torch.cuda.synchronize()
            return lambda: torch.relu(X)

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.gelu(X)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.gelu(X)

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.silu(X)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.silu(X)

        elif op == "dropout":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.nn.functional.dropout(X, p=0.1, training=True)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.dropout(X, p=0.1, training=True)

        elif op == "rmsnorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            # FlagGems intercepts the underlying ops
            eps = 1e-6
            def _rmsnorm():
                rms = torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + eps)
                return X * rms * w
            _rmsnorm(); torch.cuda.synchronize()
            return _rmsnorm

        elif op == "tanh":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.tanh(X); torch.cuda.synchronize()
            return lambda: torch.tanh(X)
        elif op == "sigmoid":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.sigmoid(X); torch.cuda.synchronize()
            return lambda: torch.sigmoid(X)
        elif op == "add":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            (A + B); torch.cuda.synchronize()
            return lambda: A + B
        elif op == "mul":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            (A * B); torch.cuda.synchronize()
            return lambda: A * B
        elif op == "neg":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            (-X); torch.cuda.synchronize()
            return lambda: -X
        elif op == "exp":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.exp(X); torch.cuda.synchronize()
            return lambda: torch.exp(X)
        elif op == "rsqrt":
            X = torch.randn(M, N, device="cuda", dtype=dtype).abs() + 1e-6
            torch.rsqrt(X); torch.cuda.synchronize()
            return lambda: torch.rsqrt(X)
        elif op == "where_":
            cond = torch.randn(M, N, device="cuda") > 0
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.where(cond, A, B); torch.cuda.synchronize()
            return lambda: torch.where(cond, A, B)
        elif op == "cast":
            X = torch.randn(M, N, device="cuda", dtype=torch.float32)
            X.to(dtype); torch.cuda.synchronize()
            return lambda: X.to(dtype)
        elif op == "reduce_sum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            X.sum(dim=-1); torch.cuda.synchronize()
            return lambda: X.sum(dim=-1)
        elif op == "reduce_max":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            X.max(dim=-1); torch.cuda.synchronize()
            return lambda: X.max(dim=-1).values
        elif op == "reduce_mean":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            X.mean(dim=-1); torch.cuda.synchronize()
            return lambda: X.mean(dim=-1)
        elif op == "cumsum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            torch.cumsum(X, dim=-1); torch.cuda.synchronize()
            return lambda: torch.cumsum(X, dim=-1)
        elif op == "batch_matmul":
            from benchmarks.baselines._shared_inputs import build_batch_matmul_inputs
            A, Bm = build_batch_matmul_inputs(M, N, K, dtype)
            torch.bmm(A, Bm); torch.cuda.synchronize()
            return lambda: torch.bmm(A, Bm)
        elif op == "transpose":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            X.T.contiguous(); torch.cuda.synchronize()
            return lambda: X.T.contiguous()
        elif op == "embedding":
            weight = torch.randn(M, N, device="cuda", dtype=dtype)
            indices = torch.randint(0, M, (min(K or 128, M),), device="cuda")
            torch.nn.functional.embedding(indices, weight); torch.cuda.synchronize()
            return lambda: torch.nn.functional.embedding(indices, weight)

        # ── OT4 Attention (S7.followup.3 2026-06-06) ─────────────────
        # FlagGems hijacks aten::_flash_attention_forward + ATen SDPA
        # paths; once enable() runs, F.scaled_dot_product_attention
        # dispatches into FlagGems Triton kernels. We mirror
        # pytorch_eager's input construction so shape contracts stay
        # identical across runners — the only delta is the underlying
        # dispatcher (Triton vs eager). Pre-warm via one call before
        # returning the lambda so Triton autotune doesn't pollute the
        # measured latency.
        elif op == "flash_attention":
            # M = batch*heads, N = seq_len, K = head_dim
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            Q = torch.randn(batch_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            torch.nn.functional.scaled_dot_product_attention(Q, K_, V, is_causal=True)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.scaled_dot_product_attention(
                Q, K_, V, is_causal=True,
            )

        elif op == "grouped_query_attention":
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            num_kv_groups = max(batch_heads // 4, 1)
            Q = torch.randn(batch_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            K_ = torch.randn(num_kv_groups, seq_len, head_dim, device="cuda", dtype=dtype)
            V = torch.randn(num_kv_groups, seq_len, head_dim, device="cuda", dtype=dtype)
            repeats = batch_heads // num_kv_groups
            K_exp = K_.repeat_interleave(repeats, dim=0)
            V_exp = V.repeat_interleave(repeats, dim=0)
            torch.nn.functional.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=True)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.scaled_dot_product_attention(
                Q, K_exp, V_exp, is_causal=True,
            )

        elif op == "cross_attention":
            # Cross-attention: Q from decoder, K/V from encoder (Sq != Skv).
            from benchmarks.baselines._runtime_ctx import get_current_shape
            shape = get_current_shape()
            if shape is not None and getattr(shape, "Skv", None) is not None:
                batch_heads = shape.B * shape.H
                q_len = shape.S
                kv_len = shape.Skv
                head_dim = shape.D
            else:
                batch_heads = M
                q_len = max(N // 2, 1)
                kv_len = N
                head_dim = max(K, 64)
            Q = torch.randn(batch_heads, q_len, head_dim, device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, kv_len, head_dim, device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, kv_len, head_dim, device="cuda", dtype=dtype)
            torch.nn.functional.scaled_dot_product_attention(Q, K_, V)
            torch.cuda.synchronize()
            return lambda: torch.nn.functional.scaled_dot_product_attention(Q, K_, V)

        return None
