# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""C3c: Numerical correctness tests for OT4 attention-family ops via Triton codegen.

Covers 6 attention/positional ops across 4 templates:
- flash_attention        (template: flash_attention, causal MHA)
- grouped_query_attention (template: flash_attention, gqa_groups=4)
- cross_attention        (template: flash_attention, non-causal)
- multi_latent_attention (template: mla)
- paged_attention        (template: paged_attention, decode-phase)
- rope                   (template: rope)

Each test renders the op's Jinja template via
`arke.backend.triton_codegen.generate_kernel`, JIT-compiles, runs on GPU
with random inputs, and checks max-abs-diff against a PyTorch reference
(`F.scaled_dot_product_attention` for attention; explicit rotation for RoPE).

Tolerance rationale
-------------------
Attention kernels accumulate Q·Kᵀ followed by a streaming fp16 softmax and
weighted sum over V. In small shapes (B=1, H≤4, N=S≤64, D≤64) the fp16
softmax-accumulation error vs. an fp32 PyTorch SDPA reference typically
lands in the 1e-3 – 2e-2 range. We pick `atol=2e-2, rtol=2e-2` which is
loose enough to absorb fp16 rounding but tight enough to catch real
algorithmic bugs (kernels with broken causal masks, GQA indexing, or block
boundary handling routinely diff by 1e-1 or more).

Shapes are kept very small to fit a 6 GB GPU with up to 3 concurrent
subagents; tests free the CUDA cache after running.
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

pytestmark = [
    pytest.mark.skipif(not HAS_TRITON, reason="triton not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _gen(op_name: str):
    """Render + JIT-compile a kernel for ``op_name`` at fp16."""
    from arke.ir.ops.registry import REGISTRY
    from arke.backend.triton_codegen import generate_kernel
    op = REGISTRY.get(op_name)
    return generate_kernel(op_name, op.template_hint, dtype="float16")


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 2e-2, rtol: float = 2e-2) -> bool:
    """fp16 attention tolerance — see module docstring."""
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    return torch.allclose(a, b, atol=atol, rtol=rtol)


@pytest.fixture(autouse=True)
def _free_cuda_cache():
    """Release the per-test cache so 3 concurrent subagents fit in 6 GB."""
    yield
    torch.cuda.empty_cache()


# ── rope ────────────────────────────────────────────────────────────────────

def test_rope():
    """Standard RoPE half-rotate, spec layout X[B,H,S,D], cos/sin[S,D/2].

    Tensor-only signature (X, cos, sin) — the kernel infers all dims from
    shapes and uses one program per (b,h,s) row (Liger-style). cos/sin
    broadcast over (B, H): row (b,h,s) uses cos[s], sin[s].
    """
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 8, 16
    half = D // 2

    X = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    cos = torch.randn(S, half, device="cuda", dtype=torch.float16)
    sin = torch.randn(S, half, device="cuda", dtype=torch.float16)

    k = _gen("rope")
    Y = k(X, cos, sin)

    # Reference (ref_rope): x1,x2 = X[...,:D/2], X[...,D/2:]; cos/sin broadcast.
    x1 = X[..., :half].float()
    x2 = X[..., half:].float()
    c = cos.float()  # [S, half] broadcasts over (B, H)
    s = sin.float()
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    Y_ref = torch.cat([y1, y2], dim=-1).to(torch.float16)

    assert _close(Y, Y_ref), f"max diff = {(Y.float() - Y_ref.float()).abs().max().item():.4e}"


# ── flash_attention (causal MHA) ────────────────────────────────────────────

def test_flash_attention():
    """Causal multi-head attention; reference = SDPA(is_causal=True)."""
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 64, 64
    S = 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    k = _gen("flash_attention")
    O = k(Q, K, V)

    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert _close(O, O_ref), f"max diff = {(O.float() - O_ref.float()).abs().max().item():.4e}"


# ── grouped_query_attention (causal, gqa_groups=4) ──────────────────────────

def test_grouped_query_attention():
    """GQA with 4 Q heads per KV head; reference = SDPA on tiled KV."""
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 32, 64
    S = 32
    Hkv = 1  # ctx-builder hardcodes gqa_groups=4, so H/4 = 1
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)

    k = _gen("grouped_query_attention")
    O = k(Q, K, V)

    K_rep = K.repeat_interleave(H // Hkv, dim=1)
    V_rep = V.repeat_interleave(H // Hkv, dim=1)
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K_rep, V_rep, is_causal=True)
    assert _close(O, O_ref), f"max diff = {(O.float() - O_ref.float()).abs().max().item():.4e}"


# ── cross_attention (non-causal, N != S) ────────────────────────────────────

def test_cross_attention():
    """Cross-attention with N=16 queries against S=32 keys; non-causal."""
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 16, 64
    S = 32
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    k = _gen("cross_attention")
    O = k(Q, K, V)

    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    assert _close(O, O_ref), f"max diff = {(O.float() - O_ref.float()).abs().max().item():.4e}"


# ── multi_latent_attention (DeepSeek style) ─────────────────────────────────

def test_multi_latent_attention():
    """MLA: decompress KV via up-projections, then standard attention.

    Reference reconstructs K = einsum('bsc,chd->bhsd', kv_compressed, W_uk),
    likewise V, then SDPA(non-causal).
    """
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 16, 32
    S, Dc = 16, 16
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    kvc = torch.randn(B, S, Dc, device="cuda", dtype=torch.float16)
    # Scale weights down so the decompressed K/V are in a sane range.
    W_uk = torch.randn(Dc, H, D, device="cuda", dtype=torch.float16) * 0.1
    W_uv = torch.randn(Dc, H, D, device="cuda", dtype=torch.float16) * 0.1

    k = _gen("multi_latent_attention")
    O = k(Q, kvc, W_uk, W_uv)

    K = torch.einsum("bsc,chd->bhsd", kvc.float(), W_uk.float()).to(torch.float16)
    V = torch.einsum("bsc,chd->bhsd", kvc.float(), W_uv.float()).to(torch.float16)
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    assert _close(O, O_ref), f"max diff = {(O.float() - O_ref.float()).abs().max().item():.4e}"


# ── paged_attention (vLLM-style decode) ─────────────────────────────────────

def test_paged_attention():
    """Single-token decode against a paged KV cache.

    Reference materializes K, V for ``context_lens[b]`` positions by
    indexing through ``block_tables`` into ``kv_cache``, then runs SDPA.
    """
    torch.manual_seed(0)
    B, H, D = 1, 2, 32
    Hkv = 2
    num_blocks, block_size = 4, 16
    context_len = 24  # spans two blocks

    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    kv_cache = torch.randn(num_blocks, 2, Hkv, block_size, D, device="cuda", dtype=torch.float16)
    # Use non-contiguous block ids to exercise the block_tables indirection.
    block_tables = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([context_len], device="cuda", dtype=torch.int32)

    k = _gen("paged_attention")
    O = k(Q, kv_cache, block_tables, context_lens)

    # Reference: gather K, V from blocks for the live context.
    Ks, Vs = [], []
    cl = context_lens[0].item()
    num_ctx_blocks = (cl + block_size - 1) // block_size
    for blk in range(num_ctx_blocks):
        phys = block_tables[0, blk].item()
        start = blk * block_size
        end = min(start + block_size, cl)
        n_in_blk = end - start
        Ks.append(kv_cache[phys, 0, :, :n_in_blk, :])  # [Hkv, n, D]
        Vs.append(kv_cache[phys, 1, :, :n_in_blk, :])
    K_full = torch.cat(Ks, dim=1).unsqueeze(0)  # [1, Hkv, cl, D]
    V_full = torch.cat(Vs, dim=1).unsqueeze(0)

    gqa_groups = H // Hkv
    K_rep = K_full.repeat_interleave(gqa_groups, dim=1)
    V_rep = V_full.repeat_interleave(gqa_groups, dim=1)

    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(2), K_rep, V_rep, is_causal=False
    ).squeeze(2)

    # Sanity check: catch silent zero outputs.
    assert O.abs().max().item() > 0.0
    assert _close(O, O_ref), f"max diff = {(O.float() - O_ref.float()).abs().max().item():.4e}"
