# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Operator Catalog (P0 operators).

Each operator definition includes:
- Signature (inputs, output)
- Semantic formula
- Algebraic properties
- Fusion rules
- NumPy reference for V1 validation
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpDefinition:
    """Complete definition of an operator."""
    name: str
    category: str                       # "compute" | "elementwise" | "reduce" | "move"
    inputs: dict[str, str]              # {"A": "Tensor[M,K]", "B": "Tensor[K,N]"}
    output: str                         # "Tensor[M,N]"
    computation: str                    # "C[i,j] = sum(A[i,k] * B[k,j], axis=k)"
    index_vars: list[str] = field(default_factory=list)
    reduction_axes: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    can_fuse_as: str | None = None   # "epilogue" | "prologue" | None
    numpy_ref: str = ""                 # "np.matmul(A, B)"


# ============================================================
# P0 Operator Catalog (10 operators)
# ============================================================

OP_CATALOG: dict[str, OpDefinition] = {}


def _register(op: OpDefinition) -> OpDefinition:
    OP_CATALOG[op.name] = op
    return op


# --- Compute-bound ---

MATMUL = _register(OpDefinition(
    name="matmul",
    category="compute",
    inputs={"A": "Tensor[M,K]", "B": "Tensor[K,N]"},
    output="Tensor[M,N]",
    computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
    index_vars=["i", "j", "k"],
    reduction_axes=["k"],
    properties=["associative", "distributive"],
    can_fuse_as="prologue",
    numpy_ref="np.matmul(A, B)",
))

BATCH_MATMUL = _register(OpDefinition(
    name="batch_matmul",
    category="compute",
    inputs={"A": "Tensor[B,M,K]", "B": "Tensor[B,K,N]"},
    output="Tensor[B,M,N]",
    computation="C[b,i,j] = sum(A[b,i,k] * B[b,k,j], axis=k)",
    index_vars=["b", "i", "j", "k"],
    reduction_axes=["k"],
    properties=["associative", "distributive"],
    can_fuse_as="prologue",
    numpy_ref="np.matmul(A, B)",
))

# --- Elementwise ---

RELU = _register(OpDefinition(
    name="relu",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = max(X, 0)",
    properties=["elementwise", "monotonic"],
    can_fuse_as="epilogue",
    numpy_ref="np.maximum(X, 0)",
))

GELU = _register(OpDefinition(
    name="gelu",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = X * Phi(X)",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="0.5 * X * (1 + scipy.special.erf(X / math.sqrt(2)))",
))

SILU = _register(OpDefinition(
    name="silu",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = X * sigmoid(X)",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="X / (1 + np.exp(-X))",
))

ADD = _register(OpDefinition(
    name="add",
    category="elementwise",
    inputs={"A": "Tensor[...]", "B": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = A + B",
    properties=["elementwise", "commutative", "associative"],
    can_fuse_as="epilogue",
    numpy_ref="A + B",
))

MUL = _register(OpDefinition(
    name="mul",
    category="elementwise",
    inputs={"A": "Tensor[...]", "B": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = A * B",
    properties=["elementwise", "commutative", "associative"],
    can_fuse_as="epilogue",
    numpy_ref="A * B",
))

# --- Reduce ---

LAYERNORM = _register(OpDefinition(
    name="layernorm",
    category="reduce",
    inputs={"X": "Tensor[M,N]", "W": "Tensor[N]", "B": "Tensor[N]"},
    output="Tensor[M,N]",
    computation=(
        "Y[i,j] = (X[i,j] - mean(X[i,:], axis=j)) "
        "/ sqrt(var(X[i,:], axis=j) + eps) * W[j] + B[j]"
    ),
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["row-wise"],
    can_fuse_as=None,
    numpy_ref="(X - X.mean(-1, keepdims=True)) / np.sqrt(X.var(-1, keepdims=True) + eps) * W + B",
))

RMSNORM = _register(OpDefinition(
    name="rmsnorm",
    category="reduce",
    inputs={"X": "Tensor[M,N]", "W": "Tensor[N]"},
    output="Tensor[M,N]",
    computation="Y[i,j] = X[i,j] / sqrt(mean(X[i,:]^2, axis=j) + eps) * W[j]",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["row-wise"],
    can_fuse_as=None,
    numpy_ref="X / np.sqrt(np.mean(X**2, axis=-1, keepdims=True) + eps) * W",
))

SOFTMAX = _register(OpDefinition(
    name="softmax",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M,N]",
    computation="Y[i,j] = exp(X[i,j]) / sum(exp(X[i,:]), axis=j)",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["row-wise"],
    can_fuse_as=None,
    numpy_ref="scipy.special.softmax(X, axis=-1)",
))

REDUCE_SUM = _register(OpDefinition(
    name="reduce_sum",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M]",
    computation="Y[i] = sum(X[i,:], axis=j)",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["associative", "commutative"],
    can_fuse_as=None,
    numpy_ref="np.sum(X, axis=-1)",
))

REDUCE_MAX = _register(OpDefinition(
    name="reduce_max",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M]",
    computation="Y[i] = max(X[i,:], axis=j)",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["associative", "commutative"],
    can_fuse_as=None,
    numpy_ref="np.max(X, axis=-1)",
))

# --- Data Movement ---

TRANSPOSE = _register(OpDefinition(
    name="transpose",
    category="move",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[N,M]",
    computation="Y[j,i] = X[i,j]",
    index_vars=["i", "j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="X.T",
))


# --- Cat A: Dense Linear (additional) ---

GROUPED_MATMUL = _register(OpDefinition(
    name="grouped_matmul",
    category="compute",
    inputs={"X": "Tensor[B,M,K]", "W": "Tensor[E,K,N]", "indices": "Tensor[B]"},
    output="Tensor[B,M,N]",
    computation="Y[b,i,j] = sum(X[b,i,k] * W[indices[b],k,j], axis=k)",
    index_vars=["b", "i", "j", "k"],
    reduction_axes=["k"],
    properties=["associative"],
    can_fuse_as="prologue",
    numpy_ref="np.stack([X[b] @ W[idx[b]] for b in range(B)])",
))

# --- Cat B: Attention ---

FLASH_ATTENTION = _register(OpDefinition(
    name="flash_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H,S,D]", "K": "Tensor[B,H,S,D]", "V": "Tensor[B,H,S,D]"},
    output="Tensor[B,H,S,D]",
    computation="O = softmax(Q @ K^T / sqrt(D)) @ V  (tiled, online softmax)",
    index_vars=["b", "h", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["causal_mask_optional", "online_softmax"],
    can_fuse_as=None,
    numpy_ref="softmax(Q @ K.T / sqrt(D)) @ V",
))

GROUPED_QUERY_ATTENTION = _register(OpDefinition(
    name="grouped_query_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H_q,S,D]", "K": "Tensor[B,H_kv,S,D]", "V": "Tensor[B,H_kv,S,D]"},
    output="Tensor[B,H_q,S,D]",
    computation="GQA: Q heads grouped over fewer KV heads; O = softmax(Q @ K^T / sqrt(D)) @ V",
    index_vars=["b", "h_q", "h_kv", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["causal_mask_optional", "online_softmax", "kv_head_repeat"],
    can_fuse_as=None,
    numpy_ref="GQA with head repeat",
))

MULTI_LATENT_ATTENTION = _register(OpDefinition(
    name="multi_latent_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H,S,D]", "KV_compressed": "Tensor[B,S,D_c]", "W_uk": "Tensor[D_c,H,D]", "W_uv": "Tensor[D_c,H,D]"},
    output="Tensor[B,H,S,D]",
    computation="MLA: decompress KV from low-rank latent, then standard attention",
    index_vars=["b", "h", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["causal_mask_optional", "online_softmax", "latent_decompress"],
    can_fuse_as=None,
    numpy_ref="MLA: K=KV_c@W_uk, V=KV_c@W_uv, then softmax(Q@K^T/sqrt(D))@V",
))

# --- Cat C: Normalization (additional) ---

RMSNORM_RESIDUAL = _register(OpDefinition(
    name="rmsnorm_residual",
    category="reduce",
    inputs={"X": "Tensor[M,N]", "residual": "Tensor[M,N]", "W": "Tensor[N]"},
    output="Tensor[M,N]",
    computation="H = X + residual; Y[i,j] = H[i,j] / sqrt(mean(H[i,:]^2) + eps) * W[j]",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["row-wise", "fused_residual"],
    can_fuse_as=None,
    numpy_ref="H = X + residual; H / np.sqrt(np.mean(H**2, axis=-1, keepdims=True) + eps) * W",
))

# --- Cat D: Activation (additional) ---

SWIGLU = _register(OpDefinition(
    name="swiglu",
    category="elementwise",
    inputs={"X": "Tensor[...,2N]"},
    output="Tensor[...,N]",
    computation="x1, x2 = split(X); Y = silu(x1) * x2",
    properties=["elementwise", "gated"],
    can_fuse_as="epilogue",
    numpy_ref="x1, x2 = np.split(X, 2, axis=-1); x1 / (1 + np.exp(-x1)) * x2",
))

GEGLU = _register(OpDefinition(
    name="geglu",
    category="elementwise",
    inputs={"X": "Tensor[...,2N]"},
    output="Tensor[...,N]",
    computation="x1, x2 = split(X); Y = gelu(x1) * x2",
    properties=["elementwise", "gated"],
    can_fuse_as="epilogue",
    numpy_ref="x1, x2 = np.split(X, 2, axis=-1); 0.5*x1*(1+erf(x1/sqrt(2))) * x2",
))


# ============================================================
# OT0: Additional Elementwise ops
# ============================================================

TANH = _register(OpDefinition(
    name="tanh",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = tanh(X)",
    properties=["elementwise", "monotonic"],
    can_fuse_as="epilogue",
    numpy_ref="np.tanh(X)",
))

SIGMOID = _register(OpDefinition(
    name="sigmoid",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = 1 / (1 + exp(-X))",
    properties=["elementwise", "monotonic"],
    can_fuse_as="epilogue",
    numpy_ref="1 / (1 + np.exp(-X))",
))

NEG = _register(OpDefinition(
    name="neg",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = -X",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="-X",
))

EXP = _register(OpDefinition(
    name="exp",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = exp(X)",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="np.exp(X)",
))

RSQRT = _register(OpDefinition(
    name="rsqrt",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = 1 / sqrt(X)",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="1 / np.sqrt(X)",
))

WHERE = _register(OpDefinition(
    name="where_",
    category="elementwise",
    inputs={"cond": "Tensor[...]", "A": "Tensor[...]", "B": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = A if cond else B",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="np.where(cond, A, B)",
))

CAST = _register(OpDefinition(
    name="cast",
    category="elementwise",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = cast(X, target_dtype)",
    properties=["elementwise"],
    can_fuse_as="epilogue",
    numpy_ref="X.astype(target_dtype)",
))

# ============================================================
# OT1: Additional Reduction ops
# ============================================================

REDUCE_MEAN = _register(OpDefinition(
    name="reduce_mean",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M]",
    computation="Y[i] = mean(X[i,:], axis=j)",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["associative"],
    can_fuse_as=None,
    numpy_ref="np.mean(X, axis=-1)",
))

ARGMAX = _register(OpDefinition(
    name="argmax",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M]",
    computation="Y[i] = argmax(X[i,:], axis=j)",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.argmax(X, axis=-1)",
))

TOPK = _register(OpDefinition(
    name="topk",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M,K]",
    computation="values, indices = topk(X[i,:], k, axis=j)",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.partition(X, -k, axis=-1)[..., -k:]",
))

CUMSUM = _register(OpDefinition(
    name="cumsum",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tensor[M,N]",
    computation="Y[i,j] = sum(X[i,0:j+1])",
    index_vars=["i", "j"],
    reduction_axes=[],
    properties=["scan"],
    can_fuse_as=None,
    numpy_ref="np.cumsum(X, axis=-1)",
))

# ============================================================
# OT2: Additional Data Movement ops
# ============================================================

CONCAT = _register(OpDefinition(
    name="concat",
    category="move",
    inputs={"A": "Tensor[M,N1]", "B": "Tensor[M,N2]"},
    output="Tensor[M,N1+N2]",
    computation="Y = concat(A, B, axis=-1)",
    index_vars=["i", "j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.concatenate([A, B], axis=-1)",
))

SPLIT = _register(OpDefinition(
    name="split",
    category="move",
    inputs={"X": "Tensor[M,N]"},
    output="Tuple[Tensor[M,N/2], Tensor[M,N/2]]",
    computation="A, B = split(X, 2, axis=-1)",
    index_vars=["i", "j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.split(X, 2, axis=-1)",
))

GATHER = _register(OpDefinition(
    name="gather",
    category="move",
    inputs={"X": "Tensor[M,N]", "idx": "Tensor[M,K]"},
    output="Tensor[M,K]",
    computation="Y[i,j] = X[i, idx[i,j]]",
    index_vars=["i", "j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.take_along_axis(X, idx, axis=-1)",
))

SCATTER = _register(OpDefinition(
    name="scatter",
    category="move",
    inputs={"X": "Tensor[M,N]", "idx": "Tensor[M,K]", "src": "Tensor[M,K]"},
    output="Tensor[M,N]",
    computation="Y = X.copy(); Y[i, idx[i,j]] = src[i,j]",
    index_vars=["i", "j"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.put_along_axis(X.copy(), idx, src, axis=-1)",
))

EMBEDDING = _register(OpDefinition(
    name="embedding",
    category="move",
    inputs={"indices": "Tensor[B,S]", "weight": "Tensor[V,D]"},
    output="Tensor[B,S,D]",
    computation="Y[b,s,:] = weight[indices[b,s], :]",
    index_vars=["b", "s", "d"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="weight[indices]",
))

PERMUTE = _register(OpDefinition(
    name="permute",
    category="move",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = permute(X, dims)",
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.transpose(X, axes=dims)",
))

COPY = _register(OpDefinition(
    name="copy_",
    category="move",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = X.clone()",
    properties=["elementwise"],
    can_fuse_as=None,
    numpy_ref="X.copy()",
))

# ============================================================
# OT3: Additional Fused ops
# ============================================================

ROPE = _register(OpDefinition(
    name="rope",
    category="elementwise",
    inputs={"X": "Tensor[B,H,S,D]", "cos": "Tensor[S,D/2]", "sin": "Tensor[S,D/2]"},
    output="Tensor[B,H,S,D]",
    computation="Y = X * cos + rotate_half(X) * sin",
    properties=["elementwise", "position_encoding"],
    can_fuse_as="epilogue",
    numpy_ref="x * cos + rotate_half(x) * sin",
))

CROSS_ENTROPY = _register(OpDefinition(
    name="cross_entropy",
    category="reduce",
    inputs={"logits": "Tensor[B,V]", "labels": "Tensor[B]"},
    output="Tensor[]",
    computation="loss = -mean(log_softmax(logits)[i, labels[i]])",
    index_vars=["i", "j"],
    reduction_axes=["i", "j"],
    properties=["loss_function"],
    can_fuse_as=None,
    numpy_ref="-np.mean(np.log(softmax(logits))[np.arange(B), labels])",
))

FUSED_LINEAR_CROSS_ENTROPY = _register(OpDefinition(
    name="fused_linear_cross_entropy",
    category="compute",
    inputs={"X": "Tensor[B,D]", "W": "Tensor[V,D]", "labels": "Tensor[B]"},
    output="Tensor[]",
    computation="loss = cross_entropy(X @ W^T, labels)",
    index_vars=["i", "j", "k"],
    reduction_axes=["i", "j", "k"],
    properties=["fused", "loss_function"],
    can_fuse_as=None,
    numpy_ref="cross_entropy(X @ W.T, labels)",
))

QUANTIZE_PER_TOKEN = _register(OpDefinition(
    name="quantize_per_token",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tuple[Tensor[M,N](int8), Tensor[M](f32)]",
    computation="scale[i] = max(abs(X[i,:]))/127; Q[i,j] = round(X[i,j]/scale[i])",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["quantization"],
    can_fuse_as=None,
    numpy_ref="scale = X.abs().max(-1).values/127; (X/scale[...,None]).round().to(int8)",
))

DEQUANTIZE_PER_CHANNEL = _register(OpDefinition(
    name="dequantize_per_channel",
    category="elementwise",
    inputs={"X_int8": "Tensor[M,N](int8)", "scale": "Tensor[N](f32)", "zero_point": "Tensor[N](int8)"},
    output="Tensor[M,N]",
    computation="Y[i,j] = (X_int8[i,j] - zero_point[j]) * scale[j]",
    properties=["elementwise", "dequantization"],
    can_fuse_as="prologue",
    numpy_ref="(X_int8.float() - zero_point) * scale",
))

# ============================================================
# OT4: Additional Attention ops
# ============================================================

CROSS_ATTENTION = _register(OpDefinition(
    name="cross_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H,Sq,D]", "K": "Tensor[B,H,Skv,D]", "V": "Tensor[B,H,Skv,D]"},
    output="Tensor[B,H,Sq,D]",
    computation="O = softmax(Q @ K^T / sqrt(D)) @ V  (no causal mask)",
    index_vars=["b", "h", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["online_softmax"],
    can_fuse_as=None,
    numpy_ref="softmax(Q @ K.T / sqrt(D)) @ V",
))

PAGED_ATTENTION = _register(OpDefinition(
    name="paged_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H,1,D]", "K_cache": "Tensor[num_blocks,block_size,H,D]", "V_cache": "Tensor[num_blocks,block_size,H,D]", "block_table": "Tensor[B,max_blocks]"},
    output="Tensor[B,H,1,D]",
    computation="Paged KV-cache attention with block_table indirection",
    index_vars=["b", "h", "i", "j"],
    reduction_axes=["j"],
    properties=["decode_only", "paged_kv"],
    can_fuse_as=None,
    numpy_ref="paged attention with block indirection",
))

# ============================================================
# Lookup utilities
# ============================================================

def get_op(name: str) -> OpDefinition:
    """Get operator definition by name. Raises KeyError if not found."""
    return OP_CATALOG[name]


def list_ops(category: str | None = None) -> list[OpDefinition]:
    """List all operators, optionally filtered by category."""
    ops = list(OP_CATALOG.values())
    if category:
        ops = [op for op in ops if op.category == category]
    return ops


def is_fusable_epilogue(name: str) -> bool:
    """Check if an operator can be fused as epilogue."""
    op = OP_CATALOG.get(name)
    return op is not None and op.can_fuse_as == "epilogue"
