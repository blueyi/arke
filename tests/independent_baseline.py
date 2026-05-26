# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent PyTorch baseline implementations — completely separate from Arke.

These functions are written independently to serve as ground truth for
validating the Arke compiler pipeline. They do NOT import or use any
Arke modules (no reference_impls, no SemanticInterpreter, no OpRegistry).

Each function signature: fn(inputs: dict[str, Tensor], attrs: dict) -> Tensor
"""

import torch
import torch.nn.functional as F


# ============================================================
# OT0: Elementwise
# ============================================================

def baseline_relu(inputs, attrs):
    return F.relu(inputs["X"])

def baseline_gelu(inputs, attrs):
    return F.gelu(inputs["X"])

def baseline_silu(inputs, attrs):
    return F.silu(inputs["X"])

def baseline_tanh(inputs, attrs):
    return torch.tanh(inputs["X"])

def baseline_sigmoid(inputs, attrs):
    return torch.sigmoid(inputs["X"])

def baseline_neg(inputs, attrs):
    return -inputs["X"]

def baseline_exp(inputs, attrs):
    return torch.exp(inputs["X"])

def baseline_rsqrt(inputs, attrs):
    return torch.rsqrt(inputs["X"])

def baseline_add(inputs, attrs):
    return inputs["A"] + inputs["B"]

def baseline_mul(inputs, attrs):
    return inputs["A"] * inputs["B"]

def baseline_where_(inputs, attrs):
    return torch.where(inputs["cond"].bool(), inputs["A"], inputs["B"])

def baseline_cast(inputs, attrs):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int8": torch.int8,
    }
    target = attrs.get("target_dtype", "float32")
    return inputs["X"].to(dtype_map.get(target, torch.float32))


# ============================================================
# OT1: Reduction
# ============================================================

def baseline_softmax(inputs, attrs):
    dim = attrs.get("dim", -1)
    return F.softmax(inputs["X"], dim=dim)

def baseline_layernorm(inputs, attrs):
    X = inputs["X"]
    W = inputs["W"]
    B = inputs["B"]
    eps = attrs.get("eps", 1e-5)
    normalized_shape = [W.shape[0]]
    return F.layer_norm(X, normalized_shape, W, B, eps)

def baseline_rmsnorm(inputs, attrs):
    X = inputs["X"]
    W = inputs["W"]
    eps = attrs.get("eps", 1e-6)
    variance = X.pow(2).mean(-1, keepdim=True)
    return X * torch.rsqrt(variance + eps) * W

def baseline_rmsnorm_residual(inputs, attrs):
    X = inputs["X"]
    residual = inputs["residual"]
    W = inputs["W"]
    eps = attrs.get("eps", 1e-6)
    X_res = X + residual
    variance = X_res.pow(2).mean(-1, keepdim=True)
    return X_res * torch.rsqrt(variance + eps) * W

def baseline_reduce_sum(inputs, attrs):
    dim = attrs.get("dim", -1)
    return inputs["X"].sum(dim=dim)

def baseline_reduce_max(inputs, attrs):
    dim = attrs.get("dim", -1)
    return inputs["X"].max(dim=dim).values

def baseline_reduce_mean(inputs, attrs):
    dim = attrs.get("dim", -1)
    return inputs["X"].mean(dim=dim)

def baseline_argmax(inputs, attrs):
    dim = attrs.get("dim", -1)
    return inputs["X"].argmax(dim=dim)

def baseline_topk(inputs, attrs):
    k = attrs.get("k", 5)
    dim = attrs.get("dim", -1)
    return torch.topk(inputs["X"], k=k, dim=dim).values

def baseline_cumsum(inputs, attrs):
    dim = attrs.get("dim", -1)
    return inputs["X"].cumsum(dim=dim)


# ============================================================
# OT2: Data Movement & Dense
# ============================================================

def baseline_matmul(inputs, attrs):
    return torch.matmul(inputs["A"], inputs["B"])

def baseline_batch_matmul(inputs, attrs):
    return torch.bmm(inputs["A"], inputs["B"])

def baseline_grouped_matmul(inputs, attrs):
    X = inputs["X"]  # [B, S, D]
    W = inputs["W"]  # [G, D, D_out]
    indices = inputs["indices"].long()  # [B]
    B, S, D = X.shape
    outputs = []
    for b in range(B):
        g = indices[b].item()
        out = torch.matmul(X[b], W[g])  # [S, D_out]
        outputs.append(out)
    return torch.stack(outputs, dim=0)

def baseline_transpose(inputs, attrs):
    dim0 = attrs.get("dim0", 0)
    dim1 = attrs.get("dim1", 1)
    return inputs["X"].transpose(dim0, dim1)

def baseline_concat(inputs, attrs):
    dim = attrs.get("dim", -1)
    return torch.cat([inputs["A"], inputs["B"]], dim=dim)

def baseline_split(inputs, attrs):
    split_size = attrs.get("split_size", None)
    dim = attrs.get("dim", -1)
    if split_size is None:
        split_size = inputs["X"].shape[dim] // 2
    return torch.split(inputs["X"], split_size, dim=dim)[0]

def baseline_gather(inputs, attrs):
    dim = attrs.get("dim", -1)
    return torch.gather(inputs["X"], dim, inputs["idx"].long())

def baseline_scatter(inputs, attrs):
    dim = attrs.get("dim", -1)
    return inputs["X"].scatter(dim, inputs["idx"].long(), inputs["src"])

def baseline_embedding(inputs, attrs):
    return F.embedding(inputs["indices"].long(), inputs["weight"])

def baseline_permute(inputs, attrs):
    dims = attrs.get("dims", [1, 0])
    return inputs["X"].permute(*dims)

def baseline_copy_(inputs, attrs):
    return inputs["X"].clone()


# ============================================================
# OT3: Fused Compound
# ============================================================

def baseline_silu_and_mul(inputs, attrs):
    X = inputs["X"]
    x1, x2 = X.chunk(2, dim=-1)
    return F.silu(x1) * x2

def baseline_geglu(inputs, attrs):
    X = inputs["X"]
    x1, x2 = X.chunk(2, dim=-1)
    return F.gelu(x1) * x2

def baseline_rope(inputs, attrs):
    X = inputs["X"]  # [B, H, S, D]
    cos = inputs["cos"]  # [S, D/2]
    sin = inputs["sin"]  # [S, D/2]
    x1, x2 = X.chunk(2, dim=-1)
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)

def baseline_cross_entropy(inputs, attrs):
    logits = inputs["logits"]
    labels = inputs["labels"].long()
    return F.cross_entropy(logits, labels, reduction='mean')

def baseline_fused_linear_cross_entropy(inputs, attrs):
    X = inputs["X"]
    W = inputs["W"]
    labels = inputs["labels"].long()
    logits = torch.matmul(X, W.t())
    return F.cross_entropy(logits, labels, reduction='mean')

def baseline_quantize_per_token(inputs, attrs):
    X = inputs["X"]
    scale = X.abs().max(dim=-1, keepdim=True).values / 127.0
    scale = scale.clamp(min=1e-8)
    return (X / scale).round().clamp(-128, 127).to(torch.int8)

def baseline_dequantize_per_channel(inputs, attrs):
    X_int8 = inputs["X_int8"]
    scale = inputs["scale"]
    zero_point = inputs["zero_point"]
    return (X_int8.float() - zero_point) * scale


# ============================================================
# OT4: Attention
# ============================================================

def baseline_flash_attention(inputs, attrs):
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    is_causal = attrs.get("is_causal", False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def baseline_grouped_query_attention(inputs, attrs):
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    h_q = Q.shape[1]
    h_kv = K.shape[1]
    n_rep = h_q // h_kv
    if n_rep > 1:
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)
    is_causal = attrs.get("is_causal", False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def baseline_multi_latent_attention(inputs, attrs):
    Q = inputs["Q"]
    kv_c = inputs["KV_compressed"]
    w_uk = inputs["W_uk"]
    w_uv = inputs["W_uv"]
    K = torch.einsum("bsd,dhn->bhsn", kv_c.float(), w_uk.float()).to(Q.dtype)
    V = torch.einsum("bsd,dhn->bhsn", kv_c.float(), w_uv.float()).to(Q.dtype)
    is_causal = attrs.get("is_causal", False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def baseline_cross_attention(inputs, attrs):
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    return F.scaled_dot_product_attention(Q, K, V, is_causal=False)

def baseline_paged_attention(inputs, attrs):
    Q = inputs["Q"]  # [B, H, 1, D]
    K_cache = inputs["K_cache"]  # [num_blocks, block_size, H, D]
    V_cache = inputs["V_cache"]
    block_table = inputs["block_table"].long()  # [B, max_blocks]
    B, H, _, D = Q.shape
    outputs = []
    for b in range(B):
        blocks = block_table[b]
        k_blocks = K_cache[blocks]
        v_blocks = V_cache[blocks]
        k = k_blocks.reshape(-1, H, D).permute(1, 0, 2).unsqueeze(0)
        v = v_blocks.reshape(-1, H, D).permute(1, 0, 2).unsqueeze(0)
        out = F.scaled_dot_product_attention(Q[b:b+1], k, v, is_causal=False)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


# ============================================================
# Registry
# ============================================================

BASELINE_REGISTRY = {
    "relu": baseline_relu,
    "gelu": baseline_gelu,
    "silu": baseline_silu,
    "tanh": baseline_tanh,
    "sigmoid": baseline_sigmoid,
    "neg": baseline_neg,
    "exp": baseline_exp,
    "rsqrt": baseline_rsqrt,
    "add": baseline_add,
    "mul": baseline_mul,
    "where_": baseline_where_,
    "cast": baseline_cast,
    "softmax": baseline_softmax,
    "layernorm": baseline_layernorm,
    "rmsnorm": baseline_rmsnorm,
    "rmsnorm_residual": baseline_rmsnorm_residual,
    "reduce_sum": baseline_reduce_sum,
    "reduce_max": baseline_reduce_max,
    "reduce_mean": baseline_reduce_mean,
    "argmax": baseline_argmax,
    "topk": baseline_topk,
    "cumsum": baseline_cumsum,
    "matmul": baseline_matmul,
    "batch_matmul": baseline_batch_matmul,
    "grouped_matmul": baseline_grouped_matmul,
    "transpose": baseline_transpose,
    "concat": baseline_concat,
    "split": baseline_split,
    "gather": baseline_gather,
    "scatter": baseline_scatter,
    "embedding": baseline_embedding,
    "permute": baseline_permute,
    "copy_": baseline_copy_,
    "silu_and_mul": baseline_silu_and_mul,
    "geglu": baseline_geglu,
    "rope": baseline_rope,
    "cross_entropy": baseline_cross_entropy,
    "fused_linear_cross_entropy": baseline_fused_linear_cross_entropy,
    "quantize_per_token": baseline_quantize_per_token,
    "dequantize_per_channel": baseline_dequantize_per_channel,
    "flash_attention": baseline_flash_attention,
    "grouped_query_attention": baseline_grouped_query_attention,
    "multi_latent_attention": baseline_multi_latent_attention,
    "cross_attention": baseline_cross_attention,
    "paged_attention": baseline_paged_attention,
}
