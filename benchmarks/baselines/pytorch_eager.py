# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P3: PyTorch eager baselines.

Identical to cuBLAS runner but explicitly named for clarity
in benchmark reports when comparing Arke vs "PyTorch default".
This runner represents what a user gets out of the box.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from benchmarks.baselines.base import BaselineRunner, register_baseline

# All ops with a PyTorch eager implementation in get_fn()
_SUPPORTED_OPS = frozenset({
    # OT0 Elementwise
    "relu", "gelu", "silu", "tanh", "sigmoid",
    "add", "mul", "where_", "cast", "neg", "exp", "rsqrt",
    # OT1 Reduction
    "softmax", "layernorm", "rmsnorm", "rmsnorm_residual",
    "reduce_sum", "reduce_max", "reduce_mean", "argmax", "topk", "cumsum",
    # OT2 Data Movement & Dense
    "matmul", "batch_matmul", "transpose", "concat", "split",
    "gather", "scatter", "embedding", "permute", "copy_",
    # OT3 Fused Compound
    "swiglu", "geglu", "cross_entropy", "fused_linear_cross_entropy",
    # OT4 Attention
    "flash_attention", "grouped_query_attention", "cross_attention",
    # OT3 Quantization (new)
    "quantize_per_token", "dequantize_per_channel",
    # OT2 Special (new)
    "rope", "grouped_matmul",
})


@register_baseline
class PyTorchEagerRunner(BaselineRunner):
    """P3: PyTorch eager mode (user's default)."""

    @property
    def name(self) -> str:
        return "PyTorch-eager"

    @property
    def priority(self) -> int:
        return 3

    @property
    def source(self) -> str:
        v = torch.__version__
        return (
            f"PyTorch {v} eager mode (default dispatch) | "
            "https://pytorch.org | License: BSD-3-Clause"
        )

    @property
    def available(self) -> bool:
        return torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in _SUPPORTED_OPS

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        # ── OT0 Elementwise ─────────────────────────────────────────
        if op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.relu(X)

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.gelu(X)

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.silu(X)

        elif op == "tanh":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.tanh(X)

        elif op == "sigmoid":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.sigmoid(X)

        elif op == "add":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: A + B

        elif op == "mul":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: A * B

        elif op == "where_":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            cond = torch.randn(M, N, device="cuda") > 0
            return lambda: torch.where(cond, A, B)

        elif op == "cast":
            # Cast float32 → target dtype
            X = torch.randn(M, N, device="cuda", dtype=torch.float32)
            return lambda: X.to(dtype)

        elif op == "neg":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: -X

        elif op == "exp":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.exp(X)

        elif op == "rsqrt":
            X = torch.randn(M, N, device="cuda", dtype=dtype).abs() + 1e-6
            return lambda: torch.rsqrt(X)

        # ── OT1 Reduction ───────────────────────────────────────────
        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.softmax(X, dim=-1)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            b = torch.zeros(N, device="cuda", dtype=dtype)
            return lambda: F.layer_norm(X, [N], w, b)

        elif op == "rmsnorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            eps = 1e-6
            return lambda: (X * torch.rsqrt(
                X.pow(2).mean(-1, keepdim=True) + eps
            )) * w

        elif op == "rmsnorm_residual":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            residual = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            eps = 1e-6
            return lambda: (
                (X + residual) * torch.rsqrt(
                    (X + residual).pow(2).mean(-1, keepdim=True) + eps
                )
            ) * w

        elif op == "reduce_sum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.sum(dim=-1)

        elif op == "reduce_max":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.max(dim=-1).values

        elif op == "reduce_mean":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.mean(dim=-1)

        elif op == "argmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.argmax(dim=-1)

        elif op == "topk":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            k = min(50, N)
            return lambda: torch.topk(X, k=k, dim=-1).values

        elif op == "cumsum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.cumsum(X, dim=-1)

        # ── OT2 Data Movement & Dense ───────────────────────────────
        elif op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)
            return lambda: torch.matmul(A, B)

        elif op == "batch_matmul":
            batch = max(K, 4)  # use K as batch dim, floor at 4
            A = torch.randn(batch, M, N, device="cuda", dtype=dtype)
            B = torch.randn(batch, N, M, device="cuda", dtype=dtype)
            return lambda: torch.bmm(A, B)

        elif op == "transpose":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.T

        elif op == "concat":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.cat([A, B], dim=-1)

        elif op == "split":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            split_size = max(N // 2, 1)
            return lambda: torch.split(X, split_size, dim=-1)

        elif op == "gather":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            idx = torch.randint(0, N, (M, N), device="cuda")
            return lambda: torch.gather(X, 1, idx)

        elif op == "scatter":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            idx = torch.randint(0, N, (M, N), device="cuda")
            src = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.zeros_like(X).scatter_(1, idx, src)

        elif op == "embedding":
            vocab_size = M
            seq_len = N
            weight = torch.randn(vocab_size, max(K, 128),
                                 device="cuda", dtype=dtype)
            indices = torch.randint(0, vocab_size, (seq_len,),
                                    device="cuda")
            return lambda: F.embedding(indices, weight)

        elif op == "permute":
            # 3D tensor: (M, N, K) → (M, K, N)
            dim2 = max(K, 64)
            X = torch.randn(M, N, dim2, device="cuda", dtype=dtype)
            return lambda: X.permute(0, 2, 1)

        elif op == "copy_":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.clone()

        # ── OT3 Fused Compound ──────────────────────────────────────
        elif op == "swiglu":
            # Input width = 2*N so chunk gives two N-wide halves
            X = torch.randn(M, 2 * N, device="cuda", dtype=dtype)
            x1, x2 = X.chunk(2, dim=-1)
            return lambda: F.silu(x1) * x2

        elif op == "geglu":
            X = torch.randn(M, 2 * N, device="cuda", dtype=dtype)
            x1, x2 = X.chunk(2, dim=-1)
            return lambda: F.gelu(x1) * x2

        elif op == "cross_entropy":
            num_classes = N
            logits = torch.randn(M, num_classes, device="cuda",
                                 dtype=torch.float32)
            labels = torch.randint(0, num_classes, (M,), device="cuda")
            return lambda: F.cross_entropy(logits, labels)

        elif op == "fused_linear_cross_entropy":
            # X[M, K] @ W[N, K].T → logits[M, N] → cross_entropy
            hidden = max(K, 128)
            num_classes = N
            X = torch.randn(M, hidden, device="cuda", dtype=dtype)
            W = torch.randn(num_classes, hidden, device="cuda", dtype=dtype)
            labels = torch.randint(0, num_classes, (M,), device="cuda")
            return lambda: F.cross_entropy(
                X.to(torch.float32) @ W.to(torch.float32).T, labels
            )

        # ── OT4 Attention ───────────────────────────────────────────
        elif op == "flash_attention":
            # M = batch*heads, N = seq_len, K = head_dim
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            Q = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, seq_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            return lambda: F.scaled_dot_product_attention(
                Q, K_, V, is_causal=True,
            )

        elif op == "grouped_query_attention":
            # GQA: Q has more heads than K/V; repeat K/V to match
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            num_kv_groups = max(batch_heads // 4, 1)
            Q = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(num_kv_groups, seq_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(num_kv_groups, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            repeats = batch_heads // num_kv_groups
            K_exp = K_.repeat_interleave(repeats, dim=0)
            V_exp = V.repeat_interleave(repeats, dim=0)
            return lambda: F.scaled_dot_product_attention(
                Q, K_exp, V_exp, is_causal=True,
            )

        elif op == "cross_attention":
            # Cross-attention: Q from decoder, K/V from encoder
            batch_heads = M
            q_len = max(N // 2, 1)
            kv_len = N
            head_dim = max(K, 64)
            Q = torch.randn(batch_heads, q_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, kv_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, kv_len, head_dim,
                            device="cuda", dtype=dtype)
            return lambda: F.scaled_dot_product_attention(Q, K_, V)

        # ── OT3 Quantization ────────────────────────────────────────
        elif op == "quantize_per_token":
            # Quantize: X (M, N) -> (M, N) int8 + scale (M,)
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            def quantize_per_token():
                # Per-token quantization: scale = max(abs(X)) / 127
                scales = torch.amax(torch.abs(X), dim=1, keepdim=True)
                scales = torch.clamp(scales, min=1e-8)
                X_q = torch.round(X / scales * 127).to(torch.int8)
                return X_q, scales.squeeze(1)
            return quantize_per_token

        elif op == "dequantize_per_channel":
            # Dequantize: X_q (M, N) int8 + scale (N,) -> (M, N) fp
            X_q = torch.randint(-128, 127, (M, N), device="cuda", dtype=torch.int8)
            scale = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.01
            def dequantize_per_channel():
                return X_q.to(dtype) * scale.unsqueeze(0)
            return dequantize_per_channel

        elif op == "rope":
            # RoPE: apply rotary position embeddings
            batch_size = M
            seq_len = N
            head_dim = max(K, 64)
            X = torch.randn(batch_size, seq_len, head_dim, device="cuda", dtype=dtype)
            def rope():
                inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=dtype) / head_dim))
                t = torch.arange(seq_len, device="cuda", dtype=dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat([freqs, freqs], dim=-1)
                cos_emb = torch.cos(emb).unsqueeze(0)
                sin_emb = torch.sin(emb).unsqueeze(0)
                x1 = X[..., :head_dim // 2]
                x2 = X[..., head_dim // 2:]
                rotated = torch.cat([-x2, x1], dim=-1)
                return X * cos_emb + rotated * sin_emb
            return rope

        elif op == "grouped_matmul":
            # Grouped matmul: multiple independent matmuls
            # Simulate as: for each group, do a small matmul
            num_groups = max(M // 4, 1)
            group_size = M // num_groups
            inner_dim = N
            out_dim = K
            As = [torch.randn(group_size, inner_dim, device="cuda", dtype=dtype) for _ in range(num_groups)]
            Bs = [torch.randn(inner_dim, out_dim, device="cuda", dtype=dtype) for _ in range(num_groups)]
            def grouped_matmul():
                return torch.cat([A @ B for A, B in zip(As, Bs)], dim=0)
            return grouped_matmul

        # ── Unsupported ─────────────────────────────────────────────
        return None
