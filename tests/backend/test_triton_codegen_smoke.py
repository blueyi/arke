# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for arke.backend.triton_codegen — C1 (S7 codegen bring-up).

Validates the minimal end-to-end path:
  catalog template_hint → render → JIT compile → execute → numerical match
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton kernels", allow_module_level=True)


from arke.backend.triton_codegen import (
    build_template_ctx,
    compile_kernel_source,
    generate_kernel,
    render_kernel_source,
)
from arke.ir.ops.registry import REGISTRY


# ── Unit: ctx adaptation ────────────────────────────────────────────────────

def test_build_ctx_elementwise_renames_op_variant_to_activation():
    op = REGISTRY.get("relu")
    ctx = build_template_ctx("relu", op.template_hint, dtype="float16")
    assert ctx == {"activation": "relu"}


def test_build_ctx_matmul_includes_output_dtype():
    op = REGISTRY.get("matmul")
    ctx = build_template_ctx("matmul", op.template_hint, dtype="float16")
    assert ctx["output_dtype"] == "tl.float16"
    assert ctx["fused_activation"] == "none"


def test_build_ctx_layernorm_norm_type_from_op_name():
    op = REGISTRY.get("layernorm")
    ctx = build_template_ctx("layernorm", op.template_hint, dtype="float16")
    assert ctx["norm_type"] == "layernorm"


def test_build_ctx_reduction_prefixes_reduce():
    op = REGISTRY.get("reduce_sum")
    ctx = build_template_ctx("reduce_sum", op.template_hint, dtype="float16")
    assert ctx["reduction_op"] == "reduce_sum"


# ── Unit: render ────────────────────────────────────────────────────────────

def test_render_relu_contains_kernel_name_and_decorator():
    src = render_kernel_source(
        "elementwise", "test_relu_render", {"activation": "relu"}
    )
    assert "test_relu_render_kernel" in src
    assert "test_relu_render" in src
    assert "@triton.jit" in src
    assert "tl.where(x > 0, x, 0.0)" in src  # relu branch picked


def test_render_strict_undefined_raises_on_missing_var():
    import jinja2
    with pytest.raises(jinja2.exceptions.UndefinedError):
        # elementwise template needs `activation` — omit it
        render_kernel_source("elementwise", "broken", {})


# ── Unit: compile ───────────────────────────────────────────────────────────

def test_compile_returns_callable():
    src = render_kernel_source(
        "elementwise", "test_compile_relu", {"activation": "relu"}
    )
    fn = compile_kernel_source(src, "test_compile_relu")
    assert callable(fn)


# ── E2E: relu numerical correctness across sizes ────────────────────────────

@pytest.mark.parametrize("size", [256, 4096, 65536, 524288, 1048576])
def test_e2e_relu_matches_torch_reference(size):
    op = REGISTRY.get("relu")
    kernel = generate_kernel("relu", op.template_hint, dtype="float16")

    torch.manual_seed(0)
    x = torch.randn(size, device="cuda", dtype=torch.float16)
    y_arke = kernel(x)
    y_ref = torch.relu(x)

    assert torch.allclose(y_arke, y_ref, rtol=1e-3, atol=1e-3), (
        f"relu mismatch at size={size}: "
        f"max_diff={(y_arke - y_ref).abs().max().item()}"
    )
