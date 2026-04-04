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
