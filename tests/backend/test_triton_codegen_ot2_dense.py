# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""C3a: Numerical correctness for 6 OT2 dense + data-movement ops via Triton codegen.

Covers:
- matmul family (3): matmul, batch_matmul, grouped_matmul
- data movement (2): concat (axis=1), split (axis=1)
- transpose (1): 2D transpose

Renders Jinja template → JIT compiles → executes on CUDA → checks vs PyTorch.

Tolerances:
  - 5e-3 for matmul/batch_matmul (typical fp16 GEMM accumulation envelope)
  - 1e-2 for grouped_matmul (longer accumulation chain when working — see xfail note)
  - exact (atol=0) for concat/split/transpose (data movement, no FP arithmetic)
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


def _close(a: torch.Tensor, b: torch.Tensor, rtol=5e-3, atol=5e-3) -> bool:
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# ── matmul family ──────────────────────────────────────────────────────────

def test_matmul():
    torch.manual_seed(0)
    k = _gen("matmul")
    a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    y_a = k(a, b)
    y_r = (a.float() @ b.float()).to(torch.float16)
    assert _close(y_a, y_r), f"max_diff={(y_a.float()-y_r.float()).abs().max().item()}"


def test_batch_matmul():
    torch.manual_seed(0)
    k = _gen("batch_matmul")
    a = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
    y_a = k(a, b)
    y_r = (a.float() @ b.float()).to(torch.float16)
    assert _close(y_a, y_r), f"max_diff={(y_a.float()-y_r.float()).abs().max().item()}"


@pytest.mark.xfail(
    reason="grouped_matmul.py.j2 uses Python `break` inside the device kernel "
           "(group-id binary-search loop), which Triton does not currently support. "
           "Template needs rewrite to use bounded `tl.static_range` + masking. "
           "Follow-up tracked outside C3.",
    strict=True,
)
def test_grouped_matmul():
    torch.manual_seed(0)
    k = _gen("grouped_matmul")  # raises at JIT-compile (UnsupportedLanguageConstruct)
    a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    group_offsets = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    y_a = k(a, b, group_offsets)
    y_r = (a.float() @ b.float()).to(torch.float16)
    assert _close(y_a, y_r, rtol=1e-2, atol=1e-2)


# ── transpose ──────────────────────────────────────────────────────────────

def test_transpose_2d():
    """Default transpose_op != 'copy_' → produces Y[N, M] = X.T."""
    torch.manual_seed(0)
    k = _gen("transpose")
    x = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    y_a = k(x)
    y_r = x.t().contiguous()
    assert y_a.shape == y_r.shape, f"shape {y_a.shape} != {y_r.shape}"
    assert torch.equal(y_a, y_r), f"max_diff={(y_a.float()-y_r.float()).abs().max().item()}"


# ── data movement ──────────────────────────────────────────────────────────

def test_concat_axis1():
    """Template concatenates along axis=1 (not 0)."""
    torch.manual_seed(0)
    k = _gen("concat")
    a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(32, 96, device="cuda", dtype=torch.float16)
    y_a = k(a, b)
    y_r = torch.cat([a, b], dim=1)
    assert y_a.shape == y_r.shape
    assert torch.equal(y_a, y_r)


def test_split_axis1():
    """Tensor-only launcher: splits X into two halves along axis=1.

    The split point is inferred from the tensor shape as ceil(N/2), matching
    torch.chunk(X, 2, dim=-1) — the backend dispatches wrapper(*tensors) and
    cannot supply a scalar split point.
    """
    torch.manual_seed(0)
    k = _gen("split")
    x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    out = k(x)
    assert isinstance(out, tuple) and len(out) == 2
    left, right = out
    assert torch.equal(left, x[:, :64])
    assert torch.equal(right, x[:, 64:])
