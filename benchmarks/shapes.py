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





# ── New dataclasses for expanded operator coverage ──────────────────────

@dataclass(frozen=True)
class BatchMatmulShape:
    """Shape for batch_matmul: [B, M, K] × [B, K, N] → [B, M, N]."""
    tag: str
    B: int
    M: int
    K: int
    N: int
    notes: str = ""
    tier: int = field(default=1)


@dataclass(frozen=True)
class GroupedMatmulShape:
    """Shape for grouped_matmul: [B, M, K] × [E, K, N] × indices[B] → [B, M, N]."""
    tag: str
    B: int
    E: int
    M: int
    K: int
    N: int
    notes: str = ""
    tier: int = field(default=1)


@dataclass(frozen=True)
class AttentionShape:
    """Shape for attention ops: Q/K/V [B, H, S, D]."""
    tag: str
    B: int
    H: int
    S: int
    D: int
    Hkv: int | None = None    # GQA: KV heads (< H)
    D_c: int | None = None    # MLA: KV compressed dim
    notes: str = ""
    tier: int = field(default=4)


@dataclass(frozen=True)
class GatedShape:
    """Shape for gated activations (swiglu, geglu): [seq, ffn×2] → [seq, ffn]."""
    tag: str
    seq: int
    ffn_x2: int  # Input dim = 2 × FFN dim
    notes: str = ""
    tier: int = field(default=1)


# ── Batch matmul shapes ─────────────────────────────────────────────────

BATCH_MATMUL_SHAPES: list[BatchMatmulShape] = [
    # Tier 1
    BatchMatmulShape("gpt2-attn-128", 8, 128, 64, 128, "GPT-2 H=8 seq=128", tier=1),
    BatchMatmulShape("gpt2-attn-512", 12, 512, 64, 512, "GPT-2 H=12 seq=512", tier=1),
    # Tier 2
    BatchMatmulShape("llama-attn-512", 32, 512, 128, 512, "LLaMA-2 7B seq=512", tier=2),
    BatchMatmulShape("llama-attn-2k", 32, 2048, 128, 2048, "LLaMA-2 7B seq=2k", tier=2),
    BatchMatmulShape("batched-8", 8, 64, 64, 64, "Batched inference", tier=2),
    BatchMatmulShape("batched-32", 32, 128, 128, 128, "Batched inference", tier=2),
    # Tier 3
    BatchMatmulShape("non-align-1", 7, 127, 65, 129, "Non-aligned all", tier=3),
    BatchMatmulShape("non-align-2", 15, 513, 129, 513, "Off-by-one", tier=3),
    BatchMatmulShape("extreme-bsz", 64, 32, 64, 32, "Large batch small seq", tier=3),
]

# ── Grouped matmul shapes ───────────────────────────────────────────────

GROUPED_MATMUL_SHAPES: list[GroupedMatmulShape] = [
    # Tier 1
    GroupedMatmulShape("moe-tiny", 4, 8, 32, 64, 128, "Minimal MoE", tier=1),
    GroupedMatmulShape("moe-small", 8, 8, 64, 256, 256, "Small MoE", tier=1),
    # Tier 2
    GroupedMatmulShape("moe-medium", 16, 8, 128, 768, 3072, "GPT-2 scale MoE", tier=2),
    # Tier 3
    GroupedMatmulShape("non-align-1", 7, 8, 127, 769, 769, "Non-aligned dims", tier=3),
]

# ── Transpose shapes ────────────────────────────────────────────────────

TRANSPOSE_SHAPES: list[Shape2D] = [
    Shape2D("small", 128, 512, "Typical", tier=1),
    Shape2D("square-1k", 1024, 1024, "Square", tier=1),
    Shape2D("llama-kv", 512, 4096, "KV head reshape", tier=2),
    Shape2D("wide", 64, 8192, "Wide matrix", tier=2),
    Shape2D("non-align-1", 127, 513, "Off-by-one", tier=3),
    Shape2D("extreme-tall", 65536, 64, "Extreme aspect ratio", tier=3),
]

# ── Gated activation shapes (swiglu, geglu) ─────────────────────────────

GATED_SHAPES: list[GatedShape] = [
    # Tier 1
    GatedShape("gpt2-sm", 128, 6144, "GPT-2 Small GeGLU", tier=1),
    # Tier 2
    GatedShape("llama-7b-512", 512, 22016, "LLaMA-2 7B SwiGLU", tier=2),
    GatedShape("llama-7b-2k", 2048, 22016, "LLaMA-2 7B SwiGLU long", tier=2),
    GatedShape("llama3-8b", 512, 28672, "LLaMA-3 8B SwiGLU", tier=2),
    # Tier 3
    GatedShape("non-align-1", 127, 6145, "Off-by-one all", tier=3),
    GatedShape("non-align-2", 333, 22017, "LLaMA FFN+1", tier=3),
]

# ── Reduce shapes (reduce_sum, reduce_max) ──────────────────────────────

REDUCE_SHAPES: list[Shape2D] = [
    Shape2D("small", 128, 768, "GPT-2 hidden", tier=1),
    Shape2D("medium", 128, 4096, "LLaMA hidden", tier=1),
    Shape2D("large", 1024, 4096, "Stress test", tier=2),
    Shape2D("wide", 1, 50257, "Vocabulary", tier=2),
    Shape2D("non-align-1", 127, 769, "Off-by-one", tier=3),
    Shape2D("non-align-2", 333, 4097, "Off-by-one hidden", tier=3),
    Shape2D("extreme-tall", 65536, 64, "Many short rows", tier=3),
]

# ── Attention shapes (flash_attention, GQA, MLA) ────────────────────────

FLASH_ATTENTION_SHAPES: list[AttentionShape] = [
    AttentionShape("gpt2-sm-128", 1, 12, 128, 64, notes="GPT-2 Small short", tier=4),
    AttentionShape("gpt2-sm-512", 1, 12, 512, 64, notes="GPT-2 Small", tier=4),
    AttentionShape("gpt2-sm-1k", 1, 12, 1024, 64, notes="GPT-2 Small max", tier=4),
    AttentionShape("llama2-7b-512", 1, 32, 512, 128, notes="LLaMA-2 7B typical", tier=4),
    AttentionShape("llama2-7b-2k", 1, 32, 2048, 128, notes="LLaMA-2 7B max", tier=4),
    AttentionShape("llama2-7b-4k", 1, 32, 4096, 128, notes="LLaMA-2 7B extended", tier=4),
    AttentionShape("llama2-7b-batch", 4, 32, 512, 128, notes="LLaMA-2 7B batched", tier=4),
    AttentionShape("ds-v2-512", 1, 128, 512, 128, notes="DeepSeek-V2 many-head", tier=4),
    AttentionShape("ds-v2-2k", 1, 128, 2048, 128, notes="DeepSeek-V2", tier=4),
]

GQA_SHAPES: list[AttentionShape] = [
    AttentionShape("llama3-8b-512", 1, 32, 512, 128, Hkv=8, notes="GQA 4:1", tier=4),
    AttentionShape("llama3-8b-2k", 1, 32, 2048, 128, Hkv=8, notes="GQA 4:1", tier=4),
    AttentionShape("llama3-8b-8k", 1, 32, 8192, 128, Hkv=8, notes="GQA 4:1 max", tier=4),
    AttentionShape("qwen25-7b-512", 1, 28, 512, 128, Hkv=4, notes="GQA 7:1", tier=4),
    AttentionShape("qwen25-7b-2k", 1, 28, 2048, 128, Hkv=4, notes="GQA 7:1", tier=4),
]

MLA_SHAPES: list[AttentionShape] = [
    AttentionShape("ds-v2-mla-512", 1, 128, 512, 128, D_c=512, notes="DeepSeek-V2", tier=4),
    AttentionShape("ds-v2-mla-2k", 1, 128, 2048, 128, D_c=512, notes="DeepSeek-V2", tier=4),
    AttentionShape("ds-v2-mla-4k", 1, 128, 4096, 128, D_c=512, notes="DeepSeek-V2", tier=4),
    AttentionShape("ds-v3-mla-512", 1, 128, 512, 128, D_c=1024, notes="DeepSeek-V3", tier=4),
    AttentionShape("ds-v3-mla-2k", 1, 128, 2048, 128, D_c=1024, notes="DeepSeek-V3", tier=4),
]


# ── Operator Tier mapping ───────────────────────────────────────────────

# OT0: Elementwise, OT1: Reduction, OT2: Dense, OT3: Gated, OT4: Attention
OP_TIER: dict[str, int] = {
    # OT0 — Elementwise (12)
    "relu": 0, "gelu": 0, "silu": 0, "tanh": 0, "sigmoid": 0,
    "add": 0, "mul": 0, "where_": 0, "cast": 0,
    "neg": 0, "exp": 0, "rsqrt": 0,
    # OT1 — Reduction (10)
    "softmax": 1, "layernorm": 1, "rmsnorm": 1, "rmsnorm_residual": 1,
    "reduce_sum": 1, "reduce_max": 1, "reduce_mean": 1,
    "argmax": 1, "topk": 1, "cumsum": 1,
    # OT2 — Data Movement & Dense (11)
    "matmul": 2, "batch_matmul": 2, "grouped_matmul": 2, "transpose": 2,
    "concat": 2, "split": 2, "gather": 2, "scatter": 2,
    "embedding": 2, "permute": 2, "copy_": 2,
    # OT3 — Fused Compound (7)
    "swiglu": 3, "geglu": 3, "rope": 3,
    "fused_linear_cross_entropy": 3, "cross_entropy": 3,
    "quantize_per_token": 3, "dequantize_per_channel": 3,
    # OT4 — Attention (5)
    "flash_attention": 4, "grouped_query_attention": 4, "multi_latent_attention": 4,
    "cross_attention": 4, "paged_attention": 4,
}


# ── Extended get_shapes ─────────────────────────────────────────────────

# Alias map for backward compatibility and flexible naming
_SHAPE_MAP: dict[str, str] = {
    "mm": "matmul", "gemm": "matmul",
    "bmm": "batch_matmul",
    "dropout": "relu",  # same elementwise shape
    "gqa": "grouped_query_attention",
    "mla": "multi_latent_attention",
    # OT0 aliases
    "where": "where_", "to": "cast",
    # OT1 aliases
    "mean": "reduce_mean",
    # OT2 aliases
    "cat": "concat", "index_select": "gather",
    "scatter_add": "scatter", "contiguous": "copy_",
    # OT3 aliases
    "fused_ce": "fused_linear_cross_entropy", "ce": "cross_entropy",
    "quantize": "quantize_per_token", "dequantize": "dequantize_per_channel",
    # OT4 aliases
    "paged_attn": "paged_attention", "cross_attn": "cross_attention",
}

def get_shapes(  # noqa: F811 — intentional override of the original above
    op: str, *, tier: int | None = None
) -> list:
    """Get shapes for an operator, optionally filtered by tier.

    Supports all 45 OP_CATALOG operators (benchmark-ops.md).

    Parameters
    ----------
    op : str
        Operator name (e.g. ``"matmul"``, ``"flash_attention"``).
    tier : int, optional
        If given, return shapes with ``shape.tier <= tier``.
        ``tier=1`` → Tier 1 only, ``tier=2`` → Tier 1+2, etc.
        If *None*, return all shapes regardless of tier.
    """
    op = _SHAPE_MAP.get(op.lower(), op.lower())

    shapes: list
    if op in ("matmul",):
        shapes = MATMUL_SHAPES
    elif op == "batch_matmul":
        shapes = BATCH_MATMUL_SHAPES
    elif op == "grouped_matmul":
        shapes = GROUPED_MATMUL_SHAPES
    elif op == "softmax":
        shapes = SOFTMAX_SHAPES
    elif op in ("layernorm", "rmsnorm", "rmsnorm_residual"):
        shapes = NORM_SHAPES
    elif op in ("relu", "gelu", "silu", "add", "mul",
                "tanh", "sigmoid", "where_", "cast", "neg", "exp", "rsqrt"):
        shapes = ELEMENTWISE_SHAPES
    elif op in ("reduce_sum", "reduce_max", "reduce_mean", "argmax",
                "topk", "cumsum"):
        shapes = REDUCE_SHAPES
    elif op == "transpose":
        shapes = TRANSPOSE_SHAPES
    elif op in ("swiglu", "geglu"):
        shapes = GATED_SHAPES
    elif op == "flash_attention":
        shapes = FLASH_ATTENTION_SHAPES
    elif op == "grouped_query_attention":
        shapes = GQA_SHAPES
    elif op == "multi_latent_attention":
        shapes = MLA_SHAPES
    # --- OT2 data movement ops ---
    elif op in ("concat", "split", "copy_", "permute"):
        shapes = ELEMENTWISE_SHAPES
    elif op in ("gather", "scatter"):
        shapes = REDUCE_SHAPES
    elif op == "embedding":
        shapes = MATMUL_SHAPES
    # --- OT3 fused compound ops ---
    elif op == "rope":
        shapes = FLASH_ATTENTION_SHAPES
    elif op in ("cross_entropy", "fused_linear_cross_entropy"):
        shapes = MATMUL_SHAPES
    elif op in ("quantize_per_token", "dequantize_per_channel"):
        shapes = ELEMENTWISE_SHAPES
    # --- OT4 attention variants ---
    elif op == "cross_attention":
        shapes = FLASH_ATTENTION_SHAPES
    elif op == "paged_attention":
        shapes = GQA_SHAPES
    else:
        raise ValueError(f"No shape set for op '{op}'")

    if tier is not None:
        shapes = [s for s in shapes if s.tier <= tier]
    return shapes
