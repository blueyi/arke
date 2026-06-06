# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — PyTorch reference implementations for the kernel catalog.

One ``ref_<kernel>`` function per kernel in the SSOT
(``docs/benchmark/benchmark-ops.md``, parsed by
``benchmarks.op_registry``). Each is wired into the kernel-schema view
(``arke/ir/ops/catalog.py``) via ``OpSchema.reference_impl`` and called
by ``SemanticInterpreter`` for numerical validation.

Each function has signature:
    fn(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor

Layer boundary: these are **kernel-level** reference implementations
(matmul, flash_attention, rmsnorm, …) — semantically aligned with the
benchmark catalog, not with future low-level IR primitives. When a new
kernel is added to the SSOT, also add a ``ref_<name>`` here and wire it
in ``catalog.py``; ``test_ir_ops_schema_covers_kernel_catalog`` will
guard the coverage relationship.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ============================================================
# OT0: Elementwise
# ============================================================

def ref_relu(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.relu(inputs["X"])

def ref_gelu(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.gelu(inputs["X"])

def ref_silu(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.silu(inputs["X"])

def ref_add(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["A"] + inputs["B"]

def ref_mul(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["A"] * inputs["B"]

def ref_tanh(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.tanh(inputs["X"])

def ref_sigmoid(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.sigmoid(inputs["X"])

def ref_neg(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return -inputs["X"]

def ref_exp(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.exp(inputs["X"])

def ref_rsqrt(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.rsqrt(inputs["X"])

def ref_where(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.where(inputs["cond"].bool(), inputs["A"], inputs["B"])

def ref_cast(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "int32": torch.int32, "int64": torch.int64, "int8": torch.int8}
    target = attrs.get("target_dtype", "float32")
    return inputs["X"].to(dtype_map.get(target, torch.float32))


# ============================================================
# OT1: Reduction
# ============================================================

def ref_layernorm(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"].float()
    w = inputs.get("W")
    b = inputs.get("B")
    eps = attrs.get("eps", 1e-5)
    return F.layer_norm(x, [x.shape[-1]], w.float() if w is not None else None, b.float() if b is not None else None, eps).to(inputs["X"].dtype)

def ref_rmsnorm(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"].float()
    w = inputs["W"].float()
    eps = attrs.get("eps", 1e-6)
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x / rms * w).to(inputs["X"].dtype)

def ref_softmax(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.softmax(inputs["X"], dim=-1)

def ref_reduce_sum(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["X"].sum(dim=attrs.get("axis", -1))

def ref_reduce_max(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["X"].max(dim=attrs.get("axis", -1)).values

def ref_reduce_mean(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["X"].mean(dim=attrs.get("axis", -1))

def ref_argmax(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["X"].argmax(dim=attrs.get("axis", -1))

def ref_topk(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.topk(inputs["X"], attrs.get("k", 1), dim=-1).values

def ref_cumsum(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.cumsum(inputs["X"], dim=attrs.get("axis", -1))


# ============================================================
# OT2: Compute-Dense
# ============================================================

def ref_matmul(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.matmul(inputs["A"], inputs["B"])

def ref_batch_matmul(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.matmul(inputs["A"], inputs["B"])

def ref_grouped_matmul(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    X, W, idx = inputs["X"], inputs["W"], inputs["indices"].long()
    return torch.stack([X[b] @ W[idx[b]] for b in range(X.shape[0])])

def ref_transpose(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    return x.transpose(-2, -1).contiguous()


# ============================================================
# OT2: Data Movement
# ============================================================

def ref_concat(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.cat([inputs["A"], inputs["B"]], dim=attrs.get("axis", -1))

def ref_split(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    # Return first half (reference impl returns single tensor)
    chunks = torch.chunk(inputs["X"], 2, dim=attrs.get("axis", -1))
    return chunks[0]

def ref_gather(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return torch.gather(inputs["X"], -1, inputs["idx"].long())

def ref_scatter(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    out = inputs["X"].clone()
    return out.scatter_(-1, inputs["idx"].long(), inputs["src"])

def ref_embedding(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.embedding(inputs["indices"].long(), inputs["weight"])

def ref_permute(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    dims = attrs.get("dims", list(range(inputs["X"].ndim - 1, -1, -1)))
    return inputs["X"].permute(*dims).contiguous()

def ref_copy(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return inputs["X"].clone()


# ============================================================
# OT3: Gated Activations & Fused Ops
# ============================================================

def ref_silu_and_mul(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    return F.silu(gate) * up

def ref_swiglu_packed(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    """True SwiGLU packed FFN projection: split → silu×mul → matmul."""
    x = inputs["X"]
    w = inputs["W"]
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    hidden = F.silu(gate) * up
    return hidden @ w

def ref_gelu_and_mul(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    return F.gelu(gate) * up

def ref_rmsnorm_residual(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"].float()
    res = inputs["residual"].float()
    w = inputs["W"].float()
    eps = attrs.get("eps", 1e-6)
    h = x + res
    rms = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h / rms * w).to(inputs["X"].dtype)

def ref_rope(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    cos = inputs["cos"]
    sin = inputs["sin"]
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    # Broadcast cos/sin to match x shape
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)

def ref_cross_entropy(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.cross_entropy(inputs["logits"].float(), inputs["labels"].long())

def ref_fused_linear_cross_entropy(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    logits = inputs["X"].float() @ inputs["W"].float().T
    return F.cross_entropy(logits, inputs["labels"].long())

def ref_quantize_per_token(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X"]
    scale = x.abs().amax(dim=-1) / 127.0
    return (x / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

def ref_dequantize_per_channel(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    x = inputs["X_int8"].float()
    scale = inputs["scale"]
    zp = inputs["zero_point"].float()
    return (x - zp) * scale


# ============================================================
# OT4: Attention
# ============================================================

def ref_flash_attention(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    Q, K, V = inputs["Q"], inputs["K"], inputs["V"]
    is_causal = attrs.get("is_causal", False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def ref_grouped_query_attention(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    Q, K, V = inputs["Q"], inputs["K"], inputs["V"]
    h_q, h_kv = Q.shape[1], K.shape[1]
    n_rep = h_q // h_kv
    if n_rep > 1:
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)
    is_causal = attrs.get("is_causal", False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def ref_multi_latent_attention(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    Q = inputs["Q"]
    kv_c = inputs["KV_compressed"]
    w_uk = inputs["W_uk"]
    w_uv = inputs["W_uv"]
    K = torch.einsum("bsd,dhn->bhsn", kv_c.float(), w_uk.float()).to(Q.dtype)
    V = torch.einsum("bsd,dhn->bhsn", kv_c.float(), w_uv.float()).to(Q.dtype)
    is_causal = attrs.get("is_causal", False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def ref_cross_attention(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    return F.scaled_dot_product_attention(inputs["Q"], inputs["K"], inputs["V"], is_causal=False)

def ref_paged_attention(inputs: dict[str, torch.Tensor], attrs: dict) -> torch.Tensor:
    Q = inputs["Q"]  # [B,H,1,D]
    K_cache = inputs["K_cache"]  # [num_blocks,block_size,H,D]
    V_cache = inputs["V_cache"]
    block_table = inputs["block_table"].long()  # [B,max_blocks]
    B, H, _, D = Q.shape
    outputs = []
    for b in range(B):
        blocks = block_table[b]
        k_blocks = K_cache[blocks]  # [max_blocks, block_size, H, D]
        v_blocks = V_cache[blocks]
        k = k_blocks.reshape(-1, H, D).permute(1, 0, 2).unsqueeze(0)  # [1,H,S,D]
        v = v_blocks.reshape(-1, H, D).permute(1, 0, 2).unsqueeze(0)
        out = F.scaled_dot_product_attention(Q[b:b+1], k, v, is_causal=False)
        outputs.append(out)
    return torch.cat(outputs, dim=0)
