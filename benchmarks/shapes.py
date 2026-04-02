# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shape matrix for benchmark operators.

Each shape set covers small/medium/large and square/rectangular
to exercise different GPU efficiency regimes.  Shapes are organised
into three tiers:

* **Tier 1** – fast smoke-test set (seconds).
* **Tier 2** – representative workloads (minutes).
* **Tier 3** – stress / non-aligned / extreme (thorough).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MatmulShape:
    tag: str
    M: int
    N: int
    K: int
    notes: str = ""
    tier: int = field(default=1)


@dataclass(frozen=True)
class Shape2D:
    """Generic 2D shape for softmax, layernorm, elementwise."""

    tag: str
    M: int
    N: int
    notes: str = ""
    tier: int = field(default=1)


# ── Matmul shapes (Tier 1: 6, Tier 2: 6, Tier 3: 10 = 22) ─

MATMUL_SHAPES: list[MatmulShape] = [
    # Tier 1
    MatmulShape("tiny", 128, 128, 128, "Launch overhead test", tier=1),
    MatmulShape("small", 128, 768, 768, "GPT-2 c_proj", tier=1),
    MatmulShape("medium", 128, 2304, 768, "GPT-2 c_attn QKV", tier=1),
    MatmulShape("square-1k", 1024, 1024, 1024, "Classic GEMM", tier=1),
    MatmulShape("square-2k", 2048, 2048, 2048, "Compute-bound", tier=1),
    MatmulShape("square-4k", 4096, 4096, 4096, "Large GEMM", tier=1),
    # Tier 2
    MatmulShape("rect-wide", 1024, 4096, 1024, "LLM FFN up", tier=2),
    MatmulShape("rect-tall", 4096, 1024, 1024, "LLM FFN down", tier=2),
    MatmulShape("lm-head", 128, 50257, 768, "Vocabulary projection", tier=2),
    MatmulShape("llama-q", 4096, 4096, 4096, "LLaMA-7B attention", tier=2),
    MatmulShape("llama-ffn", 4096, 11008, 4096, "LLaMA-7B FFN", tier=2),
    MatmulShape("seq512", 512, 2304, 768, "GPT-2 seq=512", tier=2),
    # Tier 3
    MatmulShape("non-align-1", 127, 513, 1000, "Non-aligned dims", tier=3),
    MatmulShape("non-align-2", 333, 777, 555, "Non-aligned dims", tier=3),
    MatmulShape("non-align-3", 1023, 1025, 1024, "Off-by-one", tier=3),
    MatmulShape("non-align-4", 1000, 1000, 1000, "Round non-power-of-2", tier=3),
    MatmulShape("non-align-5", 384, 640, 1536, "Mixed alignment", tier=3),
    MatmulShape("non-align-6", 2049, 2047, 2050, "Large non-aligned", tier=3),
    MatmulShape("non-align-7", 513, 2305, 769, "Odd QKV-like", tier=3),
    MatmulShape("extreme-1row", 1, 1024, 1024, "Single-row GEMV", tier=3),
    MatmulShape("extreme-16", 16, 4096, 4096, "Tiny batch GEMM", tier=3),
    MatmulShape("extreme-long", 8192, 64, 4096, "Long-seq narrow", tier=3),
]

# ── Softmax shapes (Tier 1: 3, Tier 2: 10, Tier 3: 12 = 25) ─

SOFTMAX_SHAPES: list[Shape2D] = [
    # Tier 1
    Shape2D("attn-small", 12, 128, "GPT-2 12-head seq=128", tier=1),
    Shape2D("attn-med", 12, 512, "GPT-2 12-head seq=512", tier=1),
    Shape2D("attn-256", 12, 256, "GPT-2 12-head seq=256", tier=1),
    # Tier 2
    Shape2D("attn-large", 32, 2048, "LLaMA 32-head seq=2048", tier=2),
    Shape2D("attn-64", 12, 64, "Short attention", tier=2),
    Shape2D("attn-4k", 32, 4096, "LLaMA 32-head seq=4096", tier=2),
    Shape2D("attn-8k", 32, 8192, "Long context 8k", tier=2),
    Shape2D("square-1k", 1024, 1024, "Square stress 1k", tier=2),
    Shape2D("square-4k", 4096, 4096, "Square stress 4k", tier=2),
    Shape2D("wide-vocab", 1, 50257, "Vocabulary softmax", tier=2),
    Shape2D("wide-llama", 1, 128256, "LLaMA vocabulary softmax", tier=2),
    Shape2D("batch-large", 128, 4096, "Batched attention", tier=2),
    Shape2D("batch-xlarge", 1024, 1024, "Large batch attention", tier=2),
    # Tier 3
    Shape2D("non-align-1", 13, 513, "Non-aligned heads/seq", tier=3),
    Shape2D("non-align-2", 7, 511, "Non-aligned heads/seq", tier=3),
    Shape2D("non-align-3", 15, 1023, "Non-aligned heads/seq", tier=3),
    Shape2D("non-align-4", 32, 2049, "Off-by-one seq", tier=3),
    Shape2D("non-align-5", 11, 127, "Non-aligned small", tier=3),
    Shape2D("non-align-6", 1, 50261, "Non-aligned vocab", tier=3),
    Shape2D("non-align-7", 33, 1000, "Odd heads round N", tier=3),
    Shape2D("extreme-tiny", 1, 16, "Minimal softmax", tier=3),
    Shape2D("extreme-wide", 1, 1048576, "1M-wide softmax", tier=3),
    Shape2D("extreme-tall", 65536, 64, "64k-row softmax", tier=3),
    Shape2D("extreme-batch", 4096, 512, "Large batch short seq", tier=3),
    Shape2D("mixed-1", 100, 3000, "Mixed dimensions", tier=3),
]

# ── LayerNorm / RMSNorm shapes (Tier 1: 2, Tier 2: 5, Tier 3: 8 = 15) ─

NORM_SHAPES: list[Shape2D] = [
    # Tier 1
    Shape2D("gpt2", 128, 768, "GPT-2", tier=1),
    Shape2D("gpt2-ffn", 128, 3072, "GPT-2 FFN", tier=1),
    # Tier 2
    Shape2D("llama", 128, 4096, "LLaMA-7B", tier=2),
    Shape2D("llama-13b", 128, 5120, "LLaMA-13B", tier=2),
    Shape2D("large", 2048, 4096, "Long sequence", tier=2),
    Shape2D("seq1k", 1024, 768, "Seq=1024 GPT-2 hidden", tier=2),
    Shape2D("batch-large", 4096, 4096, "Large batch large hidden", tier=2),
    # Tier 3
    Shape2D("non-align-1", 127, 769, "Non-aligned batch/hidden", tier=3),
    Shape2D("non-align-2", 1000, 3000, "Round non-power-of-2", tier=3),
    Shape2D("non-align-3", 333, 4097, "Off-by-one hidden", tier=3),
    Shape2D("non-align-4", 2049, 4095, "Large non-aligned", tier=3),
    Shape2D("non-align-5", 100, 5121, "Non-aligned 13B-like", tier=3),
    Shape2D("extreme-small", 1, 768, "Single-token norm", tier=3),
    Shape2D("extreme-large", 8192, 4096, "8k-seq norm", tier=3),
    Shape2D("extreme-hidden", 128, 14336, "Large hidden dim", tier=3),
]

# ── Elementwise shapes (relu, gelu, silu) (Tier 1: 2, Tier 2: 4, Tier 3: 9 = 15) ─

ELEMENTWISE_SHAPES: list[Shape2D] = [
    # Tier 1
    Shape2D("small", 128, 768, "GPT-2 hidden", tier=1),
    Shape2D("medium", 128, 3072, "GPT-2 FFN", tier=1),
    # Tier 2
    Shape2D("large", 4096, 4096, "Stress test", tier=2),
    Shape2D("llama-ffn", 4096, 11008, "LLaMA-7B FFN", tier=2),
    Shape2D("xlarge", 8192, 4096, "8k-seq large", tier=2),
    Shape2D("seq1k", 1024, 768, "Seq=1024 GPT-2 hidden", tier=2),
    # Tier 3
    Shape2D("non-align-1", 127, 769, "Non-aligned", tier=3),
    Shape2D("non-align-2", 1000, 3000, "Round non-power-of-2", tier=3),
    Shape2D("non-align-3", 2049, 4097, "Off-by-one", tier=3),
    Shape2D("non-align-4", 333, 11009, "Non-aligned FFN-like", tier=3),
    Shape2D("non-align-5", 513, 769, "Odd QKV-like", tier=3),
    Shape2D("extreme-flat", 1, 1048576, "1M-element flat", tier=3),
    Shape2D("extreme-tall", 65536, 16, "64k-row narrow", tier=3),
    Shape2D("extreme-wide", 32768, 128, "32k-row moderate", tier=3),
    Shape2D("mixed-1", 100, 14336, "Mixed dimensions", tier=3),
]


def get_shapes(
    op: str, *, tier: int | None = None
) -> list[MatmulShape] | list[Shape2D]:
    """Get shapes for an operator, optionally filtered by tier.

    Parameters
    ----------
    op : str
        Operator name (e.g. ``"matmul"``, ``"softmax"``).
    tier : int, optional
        If given, return shapes with ``shape.tier <= tier``.
        ``tier=1`` → Tier 1 only, ``tier=2`` → Tier 1+2, ``tier=3`` → all.
        If *None*, return all shapes regardless of tier.
    """
    op = op.lower()
    if op in ("matmul", "batch_matmul", "mm", "gemm"):
        shapes = MATMUL_SHAPES
    elif op in ("softmax",):
        shapes = SOFTMAX_SHAPES
    elif op in ("layernorm", "rmsnorm"):
        shapes = NORM_SHAPES
    elif op in ("relu", "gelu", "silu", "dropout"):
        shapes = ELEMENTWISE_SHAPES
    else:
        raise ValueError(f"No shape set for op '{op}'")

    if tier is not None:
        shapes = [s for s in shapes if s.tier <= tier]
    return shapes
