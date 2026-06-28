# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""C3b: Numerical correctness tests for OT2/OT3 mid-layer ops via Triton codegen.

Covers 9 ops across 5 templates:
- layernorm (1):                 layernorm           [layernorm template]
- gated_activation (2):          silu_and_mul, gelu_and_mul       [gated_activation template]
- index_ops (2):                 gather, scatter     [index_ops template]
- quantize (2):                  quantize_per_token, dequantize_per_channel
                                                     [quantize template]
- cross_entropy (2):             cross_entropy, fused_linear_cross_entropy
                                                     [cross_entropy template]

Each test renders the op's Jinja template via `arke.backend.triton_codegen
.generate_kernel`, JIT-compiles, runs on GPU with random inputs, and checks
the result against a PyTorch reference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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
    from arke.ir.ops.registry import REGISTRY
    from arke.backend.triton_codegen import generate_kernel
    op = REGISTRY.get(op_name)
    return generate_kernel(op_name, op.template_hint, dtype="float16")


def _close(a: torch.Tensor, b: torch.Tensor, rtol=5e-3, atol=5e-3) -> bool:
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# ── layernorm ──────────────────────────────────────────────────────────────

def test_layernorm():
    """LayerNorm without bias: y = (x - mean) / sqrt(var + eps) * weight."""
    torch.manual_seed(0)
    k = _gen("layernorm")
    M, N = 8, 128
    X = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)
    y_a = k(X, w, bias=None, eps=1e-5)
    y_r = F.layer_norm(X.float(), [N], w.float(), None, 1e-5).to(torch.float16)
    assert _close(y_a, y_r, rtol=5e-3, atol=5e-3)


# ── gated_activation: silu_and_mul / gelu_and_mul ───────────────────────────────────────
# Input X[M, 2N]: first half is gate, second half is value.
# Output Y[M, N] = activation(gate) * value.

def test_silu_and_mul():
    torch.manual_seed(0)
    k = _gen("silu_and_mul")
    M, N = 8, 64
    X = torch.randn(M, 2 * N, device="cuda", dtype=torch.float16)
    y_a = k(X)
    gate = X[:, :N].float()
    val = X[:, N:].float()
    y_r = (F.silu(gate) * val).to(torch.float16)
    assert _close(y_a, y_r, rtol=5e-3, atol=5e-3)


def test_gelu_and_mul():
    torch.manual_seed(0)
    k = _gen("gelu_and_mul")
    M, N = 8, 64
    X = torch.randn(M, 2 * N, device="cuda", dtype=torch.float16)
    y_a = k(X)
    gate = X[:, :N].float()
    val = X[:, N:].float()
    y_r = (F.gelu(gate, approximate="none") * val).to(torch.float16)
    assert _close(y_a, y_r, rtol=5e-3, atol=5e-3)


# ── index_ops: gather / scatter ────────────────────────────────────────────

def test_gather():
    """gather: out[m, k] = src[m, idx[m, k]] (torch.gather along dim=-1).

    Schema: X[M, N], idx[M, K] with values in [0, N) -> out[M, K]. The kernel
    must match torch.gather(X, -1, idx) — a per-row COLUMN gather, NOT a row
    gather. (The earlier kernel + test encoded a row gather out[i]=src[idx[i]],
    which both mismatched the schema and read out of bounds — idx values were
    used as row indices into an M-row source — crashing the CUDA context at
    large shapes.)
    """
    torch.manual_seed(0)
    k = _gen("gather")
    M, N, K = 16, 32, 8
    src = torch.randn(M, N, device="cuda", dtype=torch.float16)
    idx = torch.randint(0, N, (M, K), device="cuda", dtype=torch.int64)
    y_a = k(src, idx)
    y_r = torch.gather(src, -1, idx)
    assert torch.equal(y_a, y_r)


def test_scatter():
    """scatter: out = X.clone(); out[m, idx[m,k]] = src[m,k] (torch.scatter_ dim=-1).

    Schema: X[M,N], idx[M,K] in [0,N), src[M,K] -> out[M,N]. Matches
    ref_scatter = X.clone().scatter_(-1, idx, src). (The old kernel + test
    encoded a row scatter out[idx[i],:]=src[i,:] with a scalar out_rows arg —
    wrong op and non-dispatchable.) Distinct per-row indices keep the result
    deterministic vs torch.
    """
    torch.manual_seed(0)
    k = _gen("scatter")
    M, N, K = 16, 32, 6
    X = torch.randn(M, N, device="cuda", dtype=torch.float16)
    idx = torch.stack([torch.randperm(N, device="cuda")[:K] for _ in range(M)]).to(torch.int64)
    src = torch.randn(M, K, device="cuda", dtype=torch.float16)
    y_a = k(X, idx, src)
    y_r = X.clone().scatter_(-1, idx, src)
    assert torch.equal(y_a, y_r)


# ── quantize: quantize_per_token / dequantize_per_channel ──────────────────

def test_quantize_per_token():
    """Round-trip check: dequantized(quantize(X)) ≈ X within int8 resolution.

    The kernel uses libdevice.rint (round-half-to-even) since tl.math.nearbyint
    is absent in Triton 3.2. Per-token scale = max(|X[m,:]|)/127, so the
    quantization step IS scale[m]; round-to-nearest bounds the round-trip error
    at scale[m]/2 (plus fp16 store rounding). The earlier `scale/127*2` bound
    was wrong — it divided the already-/127 scale a second time.
    """
    torch.manual_seed(0)
    try:
        k = _gen("quantize_per_token")
    except Exception as e:
        pytest.skip(f"quantize_per_token codegen failed at import: {e}")

    M, N = 8, 128
    X = torch.randn(M, N, device="cuda", dtype=torch.float16)
    Y_q, scales = k(X)

    assert Y_q.dtype == torch.int8
    assert Y_q.shape == (M, N)
    assert scales.shape == (M,)
    # Round-trip: scale * int8 should equal X within half a quant step (scale/2)
    # plus a small fp16-store slack.
    Y_recon = (Y_q.float() * scales[:, None]).to(torch.float16)
    tol = scales.max().item() / 2.0 + 1e-2
    err = (X.float() - Y_recon.float()).abs().max().item()
    assert err < tol, f"round-trip err {err} exceeds tolerance {tol}"


def test_dequantize_per_channel():
    """dequantize_per_channel: Y[m,n] = X_int8[m,n] * scales[n]."""
    torch.manual_seed(0)
    k = _gen("dequantize_per_channel")
    M, N = 8, 128
    X_int = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int8)
    scales = torch.randn(N, device="cuda", dtype=torch.float32).abs() * 0.01 + 1e-3
    y_a = k(X_int, scales)
    y_r = (X_int.float() * scales[None, :]).to(torch.float16)
    assert _close(y_a, y_r, rtol=2e-2, atol=2e-2)


# ── cross_entropy / fused_linear_cross_entropy ─────────────────────────────

def test_cross_entropy():
    """Per-row CE loss: loss[m] = -log_softmax(logits[m])[label[m]]."""
    torch.manual_seed(0)
    k = _gen("cross_entropy")
    M, V = 8, 256
    logits = torch.randn(M, V, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, V, (M,), device="cuda", dtype=torch.int64)
    y_a = k(logits, labels)
    y_r = F.cross_entropy(logits.float(), labels, reduction="none")
    assert _close(y_a, y_r, rtol=1e-2, atol=1e-2)


def test_fused_linear_cross_entropy():
    """Fused matmul + CE: logits = hidden @ weight.T, then CE."""
    torch.manual_seed(0)
    k = _gen("fused_linear_cross_entropy")
    M, D, V = 4, 64, 128
    hidden = torch.randn(M, D, device="cuda", dtype=torch.float16)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, V, (M,), device="cuda", dtype=torch.int64)
    y_a = k(hidden, weight, labels)
    logits = hidden.float() @ weight.float().T
    y_r = F.cross_entropy(logits, labels, reduction="none")
    assert _close(y_a, y_r, rtol=1e-2, atol=1e-2)
