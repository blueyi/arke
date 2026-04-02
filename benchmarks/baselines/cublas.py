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

        return None
