# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Register Arke kernels as torch custom ops via torch.library.

This allows torch.compile to fuse Arke kernels with surrounding ops,
eliminating Python dispatch overhead.
"""

from __future__ import annotations

import torch

from arke.integration.kernel_cache import KernelCache

# Global kernel cache
_cache = KernelCache()


# Register the Arke matmul as a custom op
@torch.library.custom_op("arke::matmul", mutates_args=())
def arke_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Arke-optimized matmul: C = A @ B."""
    return _cache.matmul(a, b)


@arke_matmul.register_fake
def arke_matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FakeTensor impl for torch.compile tracing."""
    m = a.shape[:-1]
    n = b.shape[-1]
    return a.new_empty((*m, n))


# Register the Arke softmax as a custom op
@torch.library.custom_op("arke::softmax", mutates_args=())
def arke_softmax(x: torch.Tensor) -> torch.Tensor:
    """Arke-optimized softmax over last dim."""
    return _cache.softmax(x)


@arke_softmax.register_fake
def arke_softmax_fake(x: torch.Tensor) -> torch.Tensor:
    """FakeTensor impl for torch.compile tracing."""
    return torch.empty_like(x)


def precompile_for_gpt2(seq_len: int = 128) -> None:
    """Pre-compile all Arke kernels needed for GPT-2 Small."""
    from arke.integration.gpt2_e2e import get_gpt2_shapes

    shapes = get_gpt2_shapes(seq_len)
    _cache.precompile_matmul(shapes["matmul"])
    _cache.precompile_softmax(shapes["softmax"])


def patch_gpt2_custom_op(model, seq_len: int = 128):
    """Patch GPT-2 using torch custom ops (compatible with torch.compile)."""
    precompile_for_gpt2(seq_len)

    patched_linear = 0

    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None

    for _name, module in model.named_modules():
        if Conv1D is not None and isinstance(module, Conv1D):
            def make_conv1d_fwd(mod):
                """Create a forward function that uses Arke matmul for Conv1D."""
                def forward(x):
                    out = torch.ops.arke.matmul(x, mod.weight)
                    if mod.bias is not None:
                        out = out + mod.bias
                    return out
                return forward

            module.forward = make_conv1d_fwd(module)
            patched_linear += 1

        elif isinstance(module, torch.nn.Linear):
            def make_linear_fwd(mod):
                """Create a forward function that uses Arke matmul for Linear."""
                def forward(x):
                    out = torch.ops.arke.matmul(x, mod.weight.t().contiguous())
                    if mod.bias is not None:
                        out = out + mod.bias
                    return out
                return forward

            module.forward = make_linear_fwd(module)
            patched_linear += 1

    # Patch attention softmax
    model.config._attn_implementation = "eager"
    import transformers.models.gpt2.modeling_gpt2 as gpt2_module

    def arke_eager_attention_forward(
        module, query, key, value, attention_mask,
        scaling=None, dropout=0.0, **kwargs
    ):
        """Arke-patched eager attention using custom softmax op."""
        if scaling is None:
            scaling = query.size(-1) ** -0.5

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.ops.arke.softmax(attn_weights)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    gpt2_module.eager_attention_forward = arke_eager_attention_forward

    return patched_linear, 12
