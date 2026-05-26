# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Triton codegen — Jinja2 template rendering + JIT compile bridge.

This is the S7 codegen substrate that replaces the S6 reference_impl
fallback in arke/backend/triton_backend.py.

Pipeline:
  OpSchema.template_hint
    → render_kernel_source(template_name, kernel_name, ctx)
    → compile_kernel_source(source, wrapper_name)
    → callable(*tensors) → torch.Tensor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import jinja2

from arke.ir.ops.schema import TemplateHint

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "triton_templates"

_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    # Be strict about missing variables — better to fail at render-time than
    # generate silently broken Triton code.
    undefined=jinja2.StrictUndefined,
)


# ── ctx adaptation ───────────────────────────────────────────────────────────
#
# OpSchema.template_hint.extra_ctx uses the convention `op_variant=<name>`,
# but Jinja templates use template-specific variable names (`activation`,
# `reduction_op`, `norm_type`, `output_dtype`, …). This mapping bridges them.
#
# Each template_name is mapped to a function (op_name, hint, dtype) -> dict
# that returns the full Jinja context (excluding kernel_name, which is
# injected separately).

def _ctx_elementwise(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    return {"activation": hint.extra_ctx.get("op_variant", op_name)}


def _ctx_elementwise_binary(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # Template variable is `binary_op` (not `op_variant`).
    # Special case: `where_` op has catalog op_variant="where" but the
    # template's where branch matches the literal string "where_".
    variant = hint.extra_ctx.get("op_variant", op_name)
    if op_name == "where_" or variant == "where":
        variant = "where_"
    return {"binary_op": variant}


def _ctx_matmul(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    return {
        "fused_activation": hint.extra_ctx.get("fused_activation", "none"),
        "output_dtype": _triton_dtype(dtype),
    }


def _ctx_layernorm(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # Same template handles both layernorm and rmsnorm (norm_type branch)
    return {"norm_type": op_name}


def _ctx_reduction(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # extra_ctx says e.g. op_variant="sum"; template wants reduction_op="reduce_sum"
    variant = hint.extra_ctx.get("op_variant", "sum")
    if variant == "argmax":
        return {"reduction_op": "argmax"}
    return {"reduction_op": f"reduce_{variant}"}


def _ctx_default(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    return dict(hint.extra_ctx)


def _ctx_data_movement(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # Template variable is `data_op`. Catalog gives op_variant.
    # copy_ op has variant="copy" (no underscore) which the template branch matches.
    return {"data_op": hint.extra_ctx.get("op_variant", op_name)}


def _ctx_gated_activation(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # Template variable is `gate_activation` and branches on "silu"/"gelu".
    # Catalog gives op_variant="silu_and_mul"/"geglu" — translate the gate function.
    variant = hint.extra_ctx.get("op_variant", op_name)
    gate = {"silu_and_mul": "silu", "geglu": "gelu"}.get(variant, variant)
    return {"gate_activation": gate}


def _ctx_index_ops(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    return {"index_op": hint.extra_ctx.get("op_variant", op_name)}


def _ctx_quantize(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # quantize_per_token → "quantize", dequantize_per_channel → "dequantize"
    if "dequantize" in op_name:
        variant = "dequantize"
    else:
        variant = "quantize"
    return {"quant_op": variant}


def _ctx_transpose(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # Default: 2D transpose. Template branches on transpose_op ("2d" vs "permute")
    return {"transpose_op": hint.extra_ctx.get("op_variant", "2d")}


def _ctx_cross_entropy(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # fused_linear=True for `fused_linear_cross_entropy`, False for plain `cross_entropy`.
    fused_linear = (op_name == "fused_linear_cross_entropy")
    return {"fused_linear": fused_linear}


def _ctx_flash_attention(op_name: str, hint: TemplateHint, dtype: str) -> dict[str, Any]:
    # flash_attention (MHA): causal=True, gqa_groups=1
    # grouped_query_attention: causal=True, gqa_groups>1 (default 4 for smoke; runtime picks real value)
    # cross_attention: causal=False, gqa_groups=1
    variant = hint.extra_ctx.get("op_variant", op_name)
    if variant == "cross":
        return {"causal": False, "gqa_groups": 1}
    if variant == "gqa":
        return {"causal": True, "gqa_groups": 4}
    return {"causal": True, "gqa_groups": 1}


_CTX_BUILDERS: dict[str, Callable[[str, TemplateHint, str], dict[str, Any]]] = {
    "elementwise": _ctx_elementwise,
    "elementwise_binary": _ctx_elementwise_binary,
    "matmul": _ctx_matmul,
    "batch_matmul": _ctx_matmul,
    "grouped_matmul": _ctx_matmul,
    "layernorm": _ctx_layernorm,
    "reduction": _ctx_reduction,
    "data_movement": _ctx_data_movement,
    "gated_activation": _ctx_gated_activation,
    "index_ops": _ctx_index_ops,
    "quantize": _ctx_quantize,
    "transpose": _ctx_transpose,
    "cross_entropy": _ctx_cross_entropy,
    "flash_attention": _ctx_flash_attention,
}


def _triton_dtype(dtype: str) -> str:
    """Map torch dtype string to triton.language dtype expression."""
    table = {
        "float16": "tl.float16",
        "fp16": "tl.float16",
        "bfloat16": "tl.bfloat16",
        "bf16": "tl.bfloat16",
        "float32": "tl.float32",
        "fp32": "tl.float32",
    }
    key = dtype.replace("torch.", "")
    return table.get(key, "tl.float16")


def build_template_ctx(
    op_name: str,
    template_hint: TemplateHint,
    dtype: str = "float16",
) -> dict[str, Any]:
    """Build the complete Jinja2 context for an op, given its template_hint."""
    builder = _CTX_BUILDERS.get(template_hint.template_name, _ctx_default)
    return builder(op_name, template_hint, dtype)


# ── render + compile ─────────────────────────────────────────────────────────

def render_kernel_source(
    template_name: str,
    kernel_name: str,
    ctx: dict[str, Any],
) -> str:
    """Render a Triton kernel source from a Jinja2 template.

    Args:
        template_name: filename without .py.j2 extension (e.g. "elementwise")
        kernel_name: unique identifier for the generated kernel function
        ctx: Jinja2 context (template-specific variables)

    Returns:
        Complete Python source code for the kernel module.

    Raises:
        jinja2.exceptions.TemplateError: if rendering fails.
    """
    template = _jinja_env.get_template(f"{template_name}.py.j2")
    full_ctx = {"kernel_name": kernel_name, **ctx}
    return template.render(**full_ctx)


def compile_kernel_source(
    source: str,
    wrapper_name: str,
) -> Callable[..., Any]:
    """Compile rendered Triton source and return the wrapper callable.

    The rendered source contains:
      - `import torch`, `import triton`, `import triton.language as tl`
      - `@triton.jit def <kernel_name>_kernel(...)` (the device kernel)
      - `def <kernel_name>(*tensors) -> torch.Tensor` (the host wrapper)

    We exec() it into a fresh namespace and return the wrapper function.

    Args:
        source: complete Python source from render_kernel_source()
        wrapper_name: name of the host wrapper function (matches kernel_name)

    Returns:
        Callable that accepts torch.Tensor inputs and returns torch.Tensor output.

    Raises:
        SyntaxError: rendered source is malformed.
        KeyError: wrapper function not found in namespace.
    """
    # Triton's @triton.jit decorator calls inspect.getsourcelines() on the
    # kernel function, which requires the source to be retrievable via
    # linecache. exec()'d code with a synthetic filename has no entry there
    # by default — register it explicitly.
    import linecache
    filename = f"<arke_kernel:{wrapper_name}>"
    source_lines = [line + "\n" for line in source.split("\n")]
    linecache.cache[filename] = (
        len(source), None, source_lines, filename,
    )

    namespace: dict[str, Any] = {"__file__": filename, "__name__": f"arke_gen_{wrapper_name}"}
    try:
        exec(compile(source, filename, "exec"), namespace)
    except Exception as exc:
        logger.error(
            "Failed to exec rendered Triton kernel %s: %s\n--- source ---\n%s",
            wrapper_name, exc, source,
        )
        raise
    if wrapper_name not in namespace:
        raise KeyError(
            f"Rendered kernel module does not define wrapper {wrapper_name!r}. "
            f"Available names: {sorted(k for k in namespace if not k.startswith('_'))}"
        )
    return namespace[wrapper_name]


def generate_kernel(
    op_name: str,
    template_hint: TemplateHint,
    *,
    dtype: str = "float16",
    kernel_name: str | None = None,
) -> Callable[..., Any]:
    """End-to-end: build ctx → render → compile → return callable.

    This is the high-level entry point used by TritonBackend.

    Args:
        op_name: operator name (used to derive ctx and default kernel_name)
        template_hint: from OpSchema.template_hint
        dtype: input dtype hint (affects matmul output_dtype, etc.)
        kernel_name: override; defaults to f"arke_{op_name}"

    Returns:
        A Python callable that takes torch.Tensor inputs and returns the output.
    """
    if kernel_name is None:
        kernel_name = f"arke_{op_name}"
    ctx = build_template_ctx(op_name, template_hint, dtype)
    source = render_kernel_source(template_hint.template_name, kernel_name, ctx)
    raw_callable = compile_kernel_source(source, kernel_name)

    # Adapter shims for op→template signature mismatches.
    #
    # (None currently needed — `rmsnorm` now has its own dedicated template.)

    return raw_callable
