# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shape matrix for benchmark operators.

Each shape set covers small/medium/large and square/rectangular
to exercise different GPU efficiency regimes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatmulShape:
    tag: str
    M: int
    N: int
    K: int
    notes: str = ""


@dataclass(frozen=True)
class Shape2D:
    """Generic 2D shape for softmax, layernorm, elementwise."""

    tag: str
    M: int
    N: int
    notes: str = ""


# ── Matmul shapes ──────────────────────────────────────────

MATMUL_SHAPES: list[MatmulShape] = [
    MatmulShape("tiny", 128, 128, 128, "Launch overhead test"),
    MatmulShape("small", 128, 768, 768, "GPT-2 c_proj"),
    MatmulShape("medium", 128, 2304, 768, "GPT-2 c_attn QKV"),
    MatmulShape("square-1k", 1024, 1024, 1024, "Classic GEMM"),
    MatmulShape("square-2k", 2048, 2048, 2048, "Compute-bound"),
    MatmulShape("square-4k", 4096, 4096, 4096, "Large GEMM"),
    MatmulShape("rect-wide", 1024, 4096, 1024, "LLM FFN up"),
    MatmulShape("rect-tall", 4096, 1024, 1024, "LLM FFN down"),
    MatmulShape("lm-head", 128, 50257, 768, "Vocabulary projection"),
    MatmulShape("llama-q", 4096, 4096, 4096, "LLaMA-7B attention"),
    MatmulShape("llama-ffn", 4096, 11008, 4096, "LLaMA-7B FFN"),
    MatmulShape("seq512", 512, 2304, 768, "GPT-2 seq=512"),
]

# ── Softmax shapes ─────────────────────────────────────────

SOFTMAX_SHAPES: list[Shape2D] = [
    Shape2D("attn-small", 12, 128, "GPT-2 12-head seq=128"),
    Shape2D("attn-med", 12, 512, "GPT-2 12-head seq=512"),
    Shape2D("attn-large", 32, 2048, "LLaMA 32-head seq=2048"),
    Shape2D("square-4k", 4096, 4096, "Stress test"),
    Shape2D("wide-vocab", 1, 50257, "Vocabulary softmax"),
]

# ── LayerNorm / RMSNorm shapes ─────────────────────────────

NORM_SHAPES: list[Shape2D] = [
    Shape2D("gpt2", 128, 768, "GPT-2"),
    Shape2D("llama", 128, 4096, "LLaMA-7B"),
    Shape2D("large", 2048, 4096, "Long sequence"),
]

# ── Elementwise shapes (relu, gelu, silu) ──────────────────

ELEMENTWISE_SHAPES: list[Shape2D] = [
    Shape2D("small", 128, 768, "GPT-2 hidden"),
    Shape2D("medium", 128, 3072, "GPT-2 FFN"),
    Shape2D("large", 4096, 4096, "Stress test"),
]


def get_shapes(op: str) -> list[MatmulShape] | list[Shape2D]:
    """Get shapes for an operator."""
    op = op.lower()
    if op in ("matmul", "batch_matmul", "mm", "gemm"):
        return MATMUL_SHAPES
    elif op in ("softmax",):
        return SOFTMAX_SHAPES
    elif op in ("layernorm", "rmsnorm"):
        return NORM_SHAPES
    elif op in ("relu", "gelu", "silu", "dropout"):
        return ELEMENTWISE_SHAPES
    else:
        raise ValueError(f"No shape set for op '{op}'")
