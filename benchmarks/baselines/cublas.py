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
            "rmsnorm", "rmsnorm_residual",
            # OT0 elementwise (CUDA built-in, true cublas/cudnn lineage)
            "relu", "gelu", "silu", "dropout",
            "tanh", "sigmoid", "add", "mul", "neg", "exp", "rsqrt",
            "reduce_sum", "reduce_max", "reduce_mean",
            "transpose",
            # NOTE (S7.followup.3 2026-06-06): `flash_attention` REMOVED.
            # PyTorch F.scaled_dot_product_attention dispatches via aten:: —
            # once any Triton vendor (FlagGems) calls flag_gems.enable(),
            # the SDPA path is globally hijacked. Claiming `cuBLAS/cuDNN`
            # serves the OT4 attention golden while the dispatcher is
            # actually Triton creates a same-backend-fairness lie. The
            # honest Triton golden is now FlagGems (P1); cuBLAS/cuDNN
            # stays out of OT4 to keep the runner name semantically
            # truthful about its backend.
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
            from benchmarks.baselines._shared_inputs import build_batch_matmul_inputs
            A, B = build_batch_matmul_inputs(M, N, K, dtype)
            return lambda: torch.bmm(A, B)

        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.softmax(X, dim=-1)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.ones(N, device="cuda", dtype=dtype)
            bias = torch.zeros(N, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.layer_norm(X, [N], weight, bias)

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

        elif op == "reduce_sum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.sum(dim=-1)
        elif op == "reduce_max":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.max(dim=-1).values
        elif op == "reduce_mean":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.mean(dim=-1)

        elif op == "transpose":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.T.contiguous()

        elif op == "flash_attention":
            B_size = max(1, M // 8)
            H = 8
            S = N
            D = max(K, 64)
            Q = torch.randn(B_size, H, S, D, device="cuda", dtype=dtype)
            Kk = torch.randn(B_size, H, S, D, device="cuda", dtype=dtype)
            V = torch.randn(B_size, H, S, D, device="cuda", dtype=dtype)
            return lambda: torch.nn.functional.scaled_dot_product_attention(Q, Kk, V, is_causal=True)

        return None

    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if op == "relu" and len(inputs) == 1:
            return torch.nn.functional.relu(inputs[0])
        if op == "gelu" and len(inputs) == 1:
            return torch.nn.functional.gelu(inputs[0])
        if op == "silu" and len(inputs) == 1:
            return torch.nn.functional.silu(inputs[0])
        if op == "tanh" and len(inputs) == 1:
            return torch.tanh(inputs[0])
        if op == "sigmoid" and len(inputs) == 1:
            return torch.sigmoid(inputs[0])
        if op == "neg" and len(inputs) == 1:
            return -inputs[0]
        if op == "exp" and len(inputs) == 1:
            return torch.exp(inputs[0])
        if op == "rsqrt" and len(inputs) == 1:
            return torch.rsqrt(inputs[0])
        if op == "softmax" and len(inputs) == 1:
            return torch.nn.functional.softmax(inputs[0], dim=-1)
        if op == "layernorm" and len(inputs) == 1:
            x = inputs[0]
            w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            b = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
            return torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b)
        if op == "rmsnorm" and len(inputs) == 1:
            x = inputs[0]
            w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            eps = 1e-6
            return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
        if op == "rmsnorm_residual" and len(inputs) == 2:
            x, residual = inputs
            w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            eps = 1e-6
            y = x + residual
            return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)) * w
        if op == "add" and len(inputs) == 2:
            return inputs[0] + inputs[1]
        if op == "mul" and len(inputs) == 2:
            return inputs[0] * inputs[1]
        if op == "reduce_sum" and len(inputs) == 1:
            return torch.sum(inputs[0], dim=-1)
        if op == "reduce_max" and len(inputs) == 1:
            return torch.max(inputs[0], dim=-1).values
        if op == "reduce_mean" and len(inputs) == 1:
            return torch.mean(inputs[0], dim=-1)
        if op == "matmul" and len(inputs) == 2:
            return torch.matmul(inputs[0], inputs[1])
        if op == "batch_matmul" and len(inputs) == 2:
            return torch.bmm(inputs[0], inputs[1])
        if op == "transpose" and len(inputs) == 1:
            return inputs[0].T
        if op == "flash_attention" and len(inputs) == 3:
            return torch.nn.functional.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2], is_causal=True)
        return None
