# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""C2: Numerical correctness tests for OT0/OT1 ops via Triton codegen.

Covers 21 ops across 8 templates:
- elementwise (8): relu, gelu, silu, tanh, sigmoid, neg, exp, rsqrt
- elementwise_binary (3): add, mul, where_
- reduction (4): reduce_sum, reduce_max, reduce_mean, argmax
- softmax (1): softmax
- cast (1): cast
- cumsum (1): cumsum
- topk (1): topk
- rmsnorm_residual (2): rmsnorm, rmsnorm_residual

Each test renders the op's Jinja template, JIT-compiles, runs on GPU
with random inputs, and checks max-abs-diff against a PyTorch reference.
"""

from __future__ import annotations

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
    from arke.ir.ops.registry import REGISTRY
    from arke.backend.triton_codegen import generate_kernel
    op = REGISTRY.get(op_name)
    return generate_kernel(op_name, op.template_hint, dtype="float16")


def _close(a: torch.Tensor, b: torch.Tensor, rtol=3e-3, atol=3e-3) -> bool:
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# ── elementwise (8) ────────────────────────────────────────────────────────

@pytest.mark.parametrize("op_name,ref_fn", [
    ("relu",    lambda x: torch.relu(x)),
    ("gelu",    lambda x: torch.nn.functional.gelu(x.float(), approximate="none").to(x.dtype)),
    ("silu",    lambda x: torch.nn.functional.silu(x.float()).to(x.dtype)),
    ("tanh",    lambda x: torch.tanh(x.float()).to(x.dtype)),
    ("sigmoid", lambda x: torch.sigmoid(x.float()).to(x.dtype)),
    ("neg",     lambda x: -x),
    ("exp",     lambda x: torch.exp(x.float()).to(x.dtype)),
])
def test_elementwise_unary(op_name, ref_fn):
    torch.manual_seed(0)
    k = _gen(op_name)
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    assert _close(k(x), ref_fn(x))


def test_elementwise_rsqrt():
    """rsqrt template adds +1e-6 internally; use positive input."""
    torch.manual_seed(0)
    k = _gen("rsqrt")
    x = torch.randn(1024, device="cuda", dtype=torch.float16).abs() + 0.1
    ref = torch.rsqrt(x.float().abs() + 1e-6).to(x.dtype)
    assert _close(k(x), ref)


# ── elementwise_binary (3) ─────────────────────────────────────────────────

@pytest.mark.parametrize("op_name,ref_fn", [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
])
def test_elementwise_binary(op_name, ref_fn):
    torch.manual_seed(0)
    k = _gen(op_name)
    a = torch.randn(1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, device="cuda", dtype=torch.float16)
    assert _close(k(a, b), ref_fn(a, b))


def test_where_ternary():
    torch.manual_seed(0)
    k = _gen("where_")
    cond = torch.randint(0, 2, (1024,), device="cuda").to(torch.float16)
    a = torch.randn(1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, device="cuda", dtype=torch.float16)
    ref = torch.where(cond != 0, a, b)
    assert _close(k(cond, a, b), ref)


# ── reduction (4) + softmax (1) ────────────────────────────────────────────

@pytest.mark.parametrize("op_name,ref_fn", [
    ("softmax",     lambda x: torch.softmax(x.float(), dim=-1).to(x.dtype)),
    ("reduce_sum",  lambda x: x.float().sum(dim=-1).to(x.dtype)),
    ("reduce_max",  lambda x: x.float().max(dim=-1).values.to(x.dtype)),
    ("reduce_mean", lambda x: x.float().mean(dim=-1).to(x.dtype)),
])
def test_reduction_float(op_name, ref_fn):
    torch.manual_seed(0)
    k = _gen(op_name)
    x = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    assert _close(k(x), ref_fn(x))


def test_reduction_argmax():
    torch.manual_seed(0)
    k = _gen("argmax")
    x = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    y_a = k(x)
    y_r = x.argmax(dim=-1)
    assert torch.equal(y_a.long().flatten(), y_r.long().flatten())


# ── cast / cumsum / topk ───────────────────────────────────────────────────

def test_cast_fp32_to_fp16():
    torch.manual_seed(0)
    k = _gen("cast")
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y_a = k(x, target_dtype=torch.float16)
    y_r = x.to(torch.float16)
    assert _close(y_a, y_r, rtol=1e-3, atol=1e-3)


def test_cumsum():
    torch.manual_seed(0)
    k = _gen("cumsum")
    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    y_a = k(x)
    y_r = x.float().cumsum(dim=-1).to(torch.float16)
    assert _close(y_a, y_r, rtol=5e-3, atol=5e-3)


def test_topk():
    torch.manual_seed(0)
    k = _gen("topk")
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    out = k(x, k=4)
    v_a = out[0] if isinstance(out, tuple) else out
    v_r, _ = torch.topk(x, k=4, dim=-1)
    assert _close(v_a, v_r, rtol=2e-3, atol=2e-3)


# ── rmsnorm / rmsnorm_residual ─────────────────────────────────────────────

def test_rmsnorm():
    """rmsnorm uses rmsnorm_residual template; codegen injects zero residual."""
    torch.manual_seed(0)
    k = _gen("rmsnorm")
    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    w = torch.ones(128, device="cuda", dtype=torch.float16)
    y_a = k(x, w)
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y_r = (x.float() * torch.rsqrt(var + 1e-6)).to(torch.float16) * w
    assert _close(y_a, y_r, rtol=5e-3, atol=5e-3)


def test_rmsnorm_residual():
    torch.manual_seed(0)
    k = _gen("rmsnorm_residual")
    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    res = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    w = torch.ones(128, device="cuda", dtype=torch.float16)
    out = k(x, res, w)
    y_a = out[0] if isinstance(out, tuple) else out
    s = x.float() + res.float()
    var = s.pow(2).mean(dim=-1, keepdim=True)
    y_r = (s * torch.rsqrt(var + 1e-6)).to(torch.float16) * w
    assert _close(y_a, y_r, rtol=5e-3, atol=5e-3)
