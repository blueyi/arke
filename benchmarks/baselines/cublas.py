# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P0: cuBLAS / cuDNN baselines via PyTorch native ops."""

from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline


@register_baseline
class CuBLASRunner(BaselineRunner):
    """P0: cuBLAS for matmul, cuDNN for softmax/layernorm via PyTorch."""

    @property
    def name(self) -> str:
        return "cuBLAS/cuDNN"

    @property
    def priority(self) -> int:
        return 0

    @property
    def source(self) -> str:
        v = torch.__version__
        cuda = torch.version.cuda or "unknown"
        return (
            f"NVIDIA cuBLAS/cuDNN via PyTorch {v} (CUDA {cuda}) | "
            "https://pytorch.org | License: NVIDIA EULA (proprietary)"
        )

    @property
    def available(self) -> bool:
        return torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in (
            "matmul", "batch_matmul", "softmax", "layernorm",
            "relu", "gelu", "silu", "dropout",
            # OT0 extensions (cuDNN dispatch)
            "tanh", "sigmoid", "add", "mul", "neg", "exp", "rsqrt",
            # OT1 extensions
            "reduce_sum", "reduce_max", "reduce_mean",
            # OT2 extensions
            "transpose",
            # OT4 (cuDNN flash attention)
            "flash_attention",
        )

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        if op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)
            return lambda: torch.matmul(A, B)

        elif op == "batch_matmul":
            # M = batch, N = seq, K = hidden
            A = torch.randn(M, N, K, device="cuda", dtype=dtype)
            B = torch.randn(M, K, N, device="cuda", dtype=dtype)
            return lambda: torch.bmm(A, B)

        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.softmax(X, dim=-1)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.ones(N, device="cuda", dtype=dtype)
            bias = torch.zeros(N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.layer_norm(
                X, [N], weight, bias
            )

        elif op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.relu(X)

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.gelu(X)

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.silu(X)

        elif op == "dropout":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.dropout(X, p=0.1, training=True)

        # ── OT0 Elementwise extensions ──
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
        elif op == "neg":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: -X
        elif op == "exp":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.exp(X)
        elif op == "rsqrt":
            X = torch.randn(M, N, device="cuda", dtype=dtype).abs() + 1e-6
            return lambda: torch.rsqrt(X)

        # ── OT1 Reduction extensions ──
        elif op == "reduce_sum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.sum(dim=-1)
        elif op == "reduce_max":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.max(dim=-1).values
        elif op == "reduce_mean":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.mean(dim=-1)

        # ── OT2 extensions ──
        elif op == "transpose":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.T.contiguous()

        # ── OT4 cuDNN flash attention ──
        elif op == "flash_attention":
            # M=batch*heads, N=seq_len, K=head_dim
            B_size = max(1, M // 8)  # assume 8 heads
            H = 8
            S = N
            D = max(K, 64)
            Q = torch.randn(B_size, H, S, D, device="cuda", dtype=dtype)
            Kk = torch.randn(B_size, H, S, D, device="cuda", dtype=dtype)
            V = torch.randn(B_size, H, S, D, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.scaled_dot_product_attention(
                Q, Kk, V, is_causal=True
            )

        return None
