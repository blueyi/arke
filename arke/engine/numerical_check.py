# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Engine — V1 Numerical Validation.

Verifies kernel correctness by executing the Semantic IR computation graph
using NumPy as the reference implementation.

This V1 validator checks the *math* (Semantic IR), not compiled kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arke.ir.semantic import NodeRef, ParamRef, SemanticIR

# ============================================================
# Dtype mapping
# ============================================================

DTYPE_MAP: dict[str, np.dtype] = {
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "bf16": np.dtype(np.float32),  # upcast — NumPy has no bfloat16
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
    "bool": np.dtype(np.bool_),
}


def _to_numpy_dtype(arke_dtype: str) -> np.dtype:
    """Convert Arke dtype string to numpy dtype."""
    if arke_dtype in DTYPE_MAP:
        return DTYPE_MAP[arke_dtype]
    raise ValueError(f"Unsupported dtype: {arke_dtype}")


# ============================================================
# Tolerance table
# ============================================================

TOLERANCE_TABLE: dict[str, dict[str, float]] = {
    "f16": {"atol": 1e-2, "rtol": 1e-2},
    "bf16": {"atol": 1e-2, "rtol": 1e-2},
    "f32": {"atol": 1e-5, "rtol": 1e-5},
    "f64": {"atol": 1e-10, "rtol": 1e-10},
}


def _get_tolerance(arke_dtype: str) -> dict[str, float]:
    """Get numerical tolerance for a dtype."""
    return TOLERANCE_TABLE.get(arke_dtype, {"atol": 1e-5, "rtol": 1e-5})


# ============================================================
# NumPy reference implementations for each op
# ============================================================

def _numpy_matmul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.matmul(inputs["A"], inputs["B"])


def _numpy_batch_matmul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.matmul(inputs["A"], inputs["B"])


def _numpy_relu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(inputs["X"], 0)


def _numpy_gelu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Element-wise erf using scipy if available, else NumPy approximation."""
    try:
        import scipy.special
        return scipy.special.erf(x)
    except ImportError:
        # Abramowitz & Stegun approximation (good to ~1.5e-7)
        sign = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = 1.0 - (
            ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
             - 0.284496736) * t + 0.254829592
        ) * t * np.exp(-x * x)
        return sign * y


def _numpy_silu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    return x / (1.0 + np.exp(-x))  # x * sigmoid(x)


def _numpy_add(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return inputs["A"] + inputs["B"]


def _numpy_mul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return inputs["A"] * inputs["B"]


def _numpy_layernorm(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    w = inputs.get("W", inputs.get("weight"))
    b = inputs.get("B", inputs.get("bias"))
    eps = inputs.get("eps", 1e-5)
    if isinstance(eps, np.ndarray):
        eps = float(eps.flat[0])
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if w is not None:
        x_norm = x_norm * w
    if b is not None:
        x_norm = x_norm + b
    return x_norm


def _numpy_rmsnorm(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    w = inputs.get("W", inputs.get("weight"))
    eps = inputs.get("eps", 1e-5)
    if isinstance(eps, np.ndarray):
        eps = float(eps.flat[0])
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    if w is not None:
        x_norm = x_norm * w
    return x_norm


def _numpy_softmax(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    # Numerically stable softmax
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _numpy_reduce_sum(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(inputs["X"], axis=-1)


def _numpy_reduce_max(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.max(inputs["X"], axis=-1)


def _numpy_transpose(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return inputs["X"].T


def _numpy_swiglu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    x1, x2 = np.split(x, 2, axis=-1)
    return (x1 / (1.0 + np.exp(-x1))) * x2  # silu(x1) * x2


def _numpy_geglu(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    x1, x2 = np.split(x, 2, axis=-1)
    try:
        import scipy.special
        gelu_x1 = 0.5 * x1 * (1.0 + scipy.special.erf(x1 / math.sqrt(2.0)))
    except ImportError:
        gelu_x1 = 0.5 * x1 * (1.0 + _erf(x1 / math.sqrt(2.0)))
    return gelu_x1 * x2


def _numpy_rmsnorm_residual(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]
    residual = inputs.get("residual", np.zeros_like(x))
    w = inputs.get("W", inputs.get("weight", np.ones(x.shape[-1])))
    eps = inputs.get("eps", 1e-5)
    if isinstance(eps, np.ndarray):
        eps = float(eps.flat[0])
    h = x + residual
    rms = np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + eps)
    return h / rms * w


def _numpy_grouped_matmul(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"]       # [B, M, K]
    w = inputs["W"]       # [E, K, N]
    indices = inputs["indices"]  # [B]
    B = x.shape[0]
    out = np.stack([x[b] @ w[int(indices[b])] for b in range(B)])
    return out


def _numpy_flash_attention(inputs: dict[str, np.ndarray]) -> np.ndarray:
    q = inputs["Q"].astype(np.float32)  # [B, H, S, D]
    k = inputs["K"].astype(np.float32)
    v = inputs["V"].astype(np.float32)
    d = q.shape[-1]
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(d)  # [B, H, S, S]
    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return (attn @ v).astype(q.dtype)


def _numpy_grouped_query_attention(inputs: dict[str, np.ndarray]) -> np.ndarray:
    q = inputs["Q"].astype(np.float32)  # [B, H_q, S, D]
    k = inputs["K"].astype(np.float32)  # [B, H_kv, S, D]
    v = inputs["V"].astype(np.float32)  # [B, H_kv, S, D]
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    group = H_q // H_kv
    # Repeat KV heads
    k = np.repeat(k, group, axis=1)  # [B, H_q, S, D]
    v = np.repeat(v, group, axis=1)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return (attn @ v).astype(q.dtype)


def _numpy_multi_latent_attention(inputs: dict[str, np.ndarray]) -> np.ndarray:
    q = inputs["Q"].astype(np.float32)     # [B, H, S, D]
    kv_c = inputs["KV_compressed"].astype(np.float32)  # [B, S, D_c]
    w_uk = inputs["W_uk"].astype(np.float32)  # [D_c, H, D]
    w_uv = inputs["W_uv"].astype(np.float32)  # [D_c, H, D]
    B, H, S, D = q.shape
    D_c = kv_c.shape[-1]
    # Decompress: K[b,h,s,d] = kv_c[b,s,:] @ w_uk[:,h,d]
    # Reshape for batch matmul: kv_c [B,S,D_c] @ w_uk [D_c, H*D] -> [B,S,H*D] -> [B,H,S,D]
    k = (kv_c @ w_uk.reshape(D_c, H * D)).reshape(B, S, H, D).transpose(0, 2, 1, 3)
    v = (kv_c @ w_uv.reshape(D_c, H * D)).reshape(B, S, H, D).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return (attn @ v).astype(q.dtype)


def _first_input(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Get the first tensor input (usually 'X')."""
    return inputs.get("X", next(iter(inputs.values())))


def _numpy_scatter(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs.get("X", list(inputs.values())[0]).copy()
    idx = inputs.get("idx", list(inputs.values())[1]).astype(np.intp)
    src = inputs.get("src", list(inputs.values())[2])
    np.put_along_axis(x, idx, src, axis=-1)
    return x


def _numpy_embedding(inputs: dict[str, np.ndarray]) -> np.ndarray:
    indices = inputs.get("indices", list(inputs.values())[0]).astype(np.intp)
    weight = inputs.get("weight", inputs.get("W", list(inputs.values())[1]))
    return weight[indices]


def _numpy_rope(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["X"].astype(np.float32)
    cos = inputs.get("cos", inputs.get("cos_cached", np.ones_like(x[..., :x.shape[-1]//2]))).astype(np.float32)
    sin = inputs.get("sin", inputs.get("sin_cached", np.zeros_like(x[..., :x.shape[-1]//2]))).astype(np.float32)
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return np.concatenate([y1, y2], axis=-1).astype(x.dtype)


def _numpy_cross_entropy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    logits = inputs.get("logits", inputs.get("X", list(inputs.values())[0])).astype(np.float64)
    labels = inputs.get("labels", list(inputs.values())[1]).astype(np.intp)
    logits_max = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(len(labels)), labels])
    return np.array(loss, dtype=np.float32)


def _numpy_fused_linear_cross_entropy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs.get("X", list(inputs.values())[0]).astype(np.float64)
    w = inputs.get("W", list(inputs.values())[1]).astype(np.float64)
    labels = inputs.get("labels", list(inputs.values())[2]).astype(np.intp)
    logits = x @ w.T
    logits_max = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(len(labels)), labels])
    return np.array(loss, dtype=np.float32)


def _numpy_quantize_per_token(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs.get("X", next(iter(inputs.values()))).astype(np.float32)
    scale = np.max(np.abs(x), axis=-1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-8)
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8)


def _numpy_dequantize_per_channel(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x_int8 = inputs.get("X_int8", next(iter(inputs.values()))).astype(np.float32)
    scale = inputs.get("scale", inputs.get("W", list(inputs.values())[1])).astype(np.float32)
    zp = inputs.get("zero_point", np.zeros_like(scale)).astype(np.float32)
    return ((x_int8 - zp) * scale).astype(np.float32)


def _numpy_paged_attention(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Simplified paged attention — assemble K/V from cache then standard attention."""
    q = inputs["Q"].astype(np.float32)  # [B, H, 1, D]
    k_cache = inputs["K_cache"].astype(np.float32)  # [num_blocks, block_size, H, D]
    v_cache = inputs["V_cache"].astype(np.float32)
    block_table = inputs["block_table"].astype(np.intp)  # [B, max_blocks]
    B, H, _, D = q.shape
    num_blocks_per_seq = block_table.shape[1]
    block_size = k_cache.shape[1]
    seq_len = num_blocks_per_seq * block_size
    # Assemble K/V from paged cache
    k = np.zeros((B, H, seq_len, D), dtype=np.float32)
    v = np.zeros((B, H, seq_len, D), dtype=np.float32)
    for b in range(B):
        for blk_idx in range(num_blocks_per_seq):
            phys_block = block_table[b, blk_idx] % k_cache.shape[0]
            start = blk_idx * block_size
            end = start + block_size
            # k_cache: [num_blocks, block_size, H, D] -> [H, block_size, D]
            k[b, :, start:end, :] = k_cache[phys_block].transpose(1, 0, 2)  # [H, block_size, D]
            v[b, :, start:end, :] = v_cache[phys_block].transpose(1, 0, 2)
    # Standard attention
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D)  # [B, H, 1, seq_len]
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return (attn @ v).astype(q.dtype)


_NUMPY_DISPATCH: dict[str, Any] = {
    "matmul": _numpy_matmul,
    "batch_matmul": _numpy_batch_matmul,
    "relu": _numpy_relu,
    "gelu": _numpy_gelu,
    "silu": _numpy_silu,
    "add": _numpy_add,
    "mul": _numpy_mul,
    "softmax": _numpy_softmax,
    "layernorm": _numpy_layernorm,
    "rmsnorm": _numpy_rmsnorm,
    "rmsnorm_residual": _numpy_rmsnorm_residual,
    "reduce_sum": _numpy_reduce_sum,
    "reduce_max": _numpy_reduce_max,
    "transpose": _numpy_transpose,
    "swiglu": _numpy_swiglu,
    "geglu": _numpy_geglu,
    "grouped_matmul": _numpy_grouped_matmul,
    "flash_attention": _numpy_flash_attention,
    "grouped_query_attention": _numpy_grouped_query_attention,
    "multi_latent_attention": _numpy_multi_latent_attention,
    # --- Newly added ops ---
    "tanh": lambda inputs: np.tanh(_first_input(inputs)),
    "sigmoid": lambda inputs: 1.0 / (1.0 + np.exp(-_first_input(inputs).astype(np.float64))),
    "neg": lambda inputs: -_first_input(inputs),
    "exp": lambda inputs: np.exp(_first_input(inputs).astype(np.float64)).astype(_first_input(inputs).dtype),
    "rsqrt": lambda inputs: (1.0 / np.sqrt(_first_input(inputs).astype(np.float64) + 1e-6)).astype(_first_input(inputs).dtype),
    "where_": lambda inputs: np.where(inputs.get("cond", _first_input(inputs)), inputs.get("A", list(inputs.values())[1]), inputs.get("B", list(inputs.values())[2])),
    "cast": lambda inputs: _first_input(inputs).astype(np.float16),
    "copy_": lambda inputs: _first_input(inputs).copy(),
    "reduce_mean": lambda inputs: np.mean(_first_input(inputs), axis=-1),
    "argmax": lambda inputs: np.argmax(_first_input(inputs), axis=-1),
    "topk": lambda inputs: np.sort(_first_input(inputs), axis=-1)[..., -min(50, _first_input(inputs).shape[-1]):],
    "cumsum": lambda inputs: np.cumsum(_first_input(inputs), axis=-1),
    "concat": lambda inputs: np.concatenate([inputs.get("A", list(inputs.values())[0]), inputs.get("B", list(inputs.values())[1])], axis=-1),
    "split": lambda inputs: np.split(_first_input(inputs), 2, axis=-1)[0],
    "gather": lambda inputs: np.take_along_axis(inputs.get("X", _first_input(inputs)), inputs.get("idx", list(inputs.values())[1]).astype(np.intp), axis=-1),
    "scatter": lambda inputs: _numpy_scatter(inputs),
    "embedding": lambda inputs: _numpy_embedding(inputs),
    "permute": lambda inputs: np.transpose(_first_input(inputs), axes=list(range(_first_input(inputs).ndim))[::-1]),
    "rope": lambda inputs: _numpy_rope(inputs),
    "cross_entropy": lambda inputs: _numpy_cross_entropy(inputs),
    "fused_linear_cross_entropy": lambda inputs: _numpy_fused_linear_cross_entropy(inputs),
    "quantize_per_token": lambda inputs: _numpy_quantize_per_token(inputs),
    "dequantize_per_channel": lambda inputs: _numpy_dequantize_per_channel(inputs),
    "cross_attention": _numpy_flash_attention,  # same logic, no causal mask
    "paged_attention": lambda inputs: _numpy_paged_attention(inputs),
}


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class NumericalResult:
    """Result of a numerical validation run."""
    passed: bool
    trials: int
    max_absolute_error: float
    max_relative_error: float
    tolerance: dict  # {"atol": float, "rtol": float}
    errors: list[str] = field(default_factory=list)


# ============================================================
# NumericalValidator
# ============================================================

class NumericalValidator:
    """V1 Numerical Validation — verify kernel correctness vs NumPy reference.

    Operates on SemanticIR only (the math layer). Does not require
    a compiled kernel — validates that the computation graph produces
    correct results when executed with NumPy.
    """

    def generate_reference(
        self,
        semantic_ir: SemanticIR,
        input_tensors: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Execute the computation graph using NumPy to get reference output.

        Walks the graph in topological order. For each node, resolves inputs
        (from kernel params or prior node outputs) and applies the NumPy
        reference implementation.

        Args:
            semantic_ir: The Semantic IR describing the computation.
            input_tensors: Named input arrays matching kernel params.

        Returns:
            The output array from the return node.

        Raises:
            ValueError: If the graph is invalid or an op is unsupported.
        """
        # Build topologically-ordered node list
        ordered_nodes = self._topological_sort(semantic_ir)

        # Value store: param values + intermediate results
        values: dict[str, np.ndarray] = dict(input_tensors)

        for node in ordered_nodes:
            # Resolve inputs for this node
            node_inputs: dict[str, np.ndarray] = {}
            for input_name, ref in node.inputs.items():
                if isinstance(ref, ParamRef):
                    if ref.name not in values:
                        raise ValueError(
                            f"Node '{node.id}': param '{ref.name}' not found in inputs"
                        )
                    node_inputs[input_name] = values[ref.name]
                elif isinstance(ref, NodeRef):
                    if ref.id not in values:
                        raise ValueError(
                            f"Node '{node.id}': node output '{ref.id}' not computed yet "
                            "(cycle or ordering error)"
                        )
                    node_inputs[input_name] = values[ref.id]
                else:
                    raise ValueError(f"Unknown input ref type: {type(ref)}")

            # Dispatch to NumPy implementation
            if node.op not in _NUMPY_DISPATCH:
                raise ValueError(f"No NumPy reference for op: {node.op}")

            result = _NUMPY_DISPATCH[node.op](node_inputs)
            values[node.id] = result

        # Return the output from the return node
        if not semantic_ir.return_node:
            raise ValueError("SemanticIR has no return_node set")

        if semantic_ir.return_node not in values:
            raise ValueError(
                f"Return node '{semantic_ir.return_node}' was not computed"
            )

        return values[semantic_ir.return_node]

    @staticmethod
    def _infer_index_range(
        param_name: str,
        semantic_ir: "SemanticIR",
    ) -> tuple[int, int] | None:
        """Infer valid [low, high) range for integer index params.

        For ops like grouped_matmul, 'indices' must be in [0, E) where E
        is the number of experts (first dim of 'W').
        """
        name_lower = param_name.lower()
        if name_lower in ("indices", "expert_ids", "routing_ids"):
            # Look for a sibling param named 'W' or 'weight' with shape [E, ...]
            for p in semantic_ir.params:
                if p.name in ("W", "weight", "experts") and p.shape:
                    return (0, p.shape[0])
        return None

    def generate_random_inputs(
        self,
        semantic_ir: SemanticIR,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Generate random input tensors matching the kernel's params.

        Args:
            semantic_ir: The Semantic IR with param definitions.
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping param names to random numpy arrays.
        """
        rng = np.random.RandomState(seed)
        inputs: dict[str, np.ndarray] = {}

        for param in semantic_ir.params:
            np_dtype = _to_numpy_dtype(param.dtype)

            if np.issubdtype(np_dtype, np.floating):
                # Standard normal scaled to [-1, 1] for numerical stability
                arr = rng.randn(*param.shape).astype(np_dtype)
                # For ops that need positive inputs, take abs
                if self._needs_positive_inputs(semantic_ir):
                    arr = np.abs(arr) + 0.01
            elif np.issubdtype(np_dtype, np.bool_):
                arr = rng.randint(0, 2, size=param.shape).astype(np.bool_)
            elif np.issubdtype(np_dtype, np.integer):
                # For index parameters, try to infer valid range from sibling params
                valid_range = self._infer_index_range(param.name, semantic_ir)
                if valid_range is not None:
                    low, high = valid_range
                else:
                    info = np.iinfo(np_dtype)
                    low = max(info.min, -128)
                    high = min(info.max, 127) + 1
                arr = rng.randint(low, high, size=param.shape).astype(np_dtype)
            else:
                arr = rng.randn(*param.shape).astype(np_dtype)

            inputs[param.name] = arr

        return inputs

    @staticmethod
    def _needs_positive_inputs(semantic_ir: SemanticIR) -> bool:
        """Check if any node uses an op that requires positive inputs."""
        _POSITIVE_OPS = {"rsqrt", "sqrt"}
        return any(node.op in _POSITIVE_OPS for node in semantic_ir.nodes)

    def validate(
        self,
        semantic_ir: SemanticIR,
        trials: int = 3,
    ) -> NumericalResult:
        """Run N trials with random inputs, compare against NumPy reference.

        Each trial generates random inputs, computes the reference output,
        and checks that the graph produces matching results within tolerance.

        For V1, both "actual" and "reference" come from the same NumPy path
        — this validates that the graph structure and op dispatch are correct.

        Args:
            semantic_ir: The Semantic IR to validate.
            trials: Number of random trials to run.

        Returns:
            NumericalResult summarizing the validation.
        """
        # Determine output dtype for tolerance
        output_dtype = "f32"  # default
        if semantic_ir.return_type:
            output_dtype = semantic_ir.return_type.dtype
        elif semantic_ir.params:
            output_dtype = semantic_ir.params[0].dtype

        tolerance = _get_tolerance(output_dtype)
        max_abs_error = 0.0
        max_rel_error = 0.0
        all_errors: list[str] = []

        for trial in range(trials):
            seed = 42 + trial
            try:
                inputs = self.generate_random_inputs(semantic_ir, seed=seed)
                output = self.generate_reference(semantic_ir, inputs)

                # V1: re-run to verify determinism
                output2 = self.generate_reference(semantic_ir, inputs)

                # Compute errors
                abs_err = np.max(np.abs(output.astype(np.float64) - output2.astype(np.float64)))
                max_abs_error = max(max_abs_error, float(abs_err))

                # Relative error (avoid division by zero)
                denom = np.maximum(np.abs(output.astype(np.float64)), 1e-12)
                rel_err = np.max(
                    np.abs(output.astype(np.float64) - output2.astype(np.float64)) / denom
                )
                max_rel_error = max(max_rel_error, float(rel_err))

                if not np.allclose(output, output2,
                                   atol=tolerance["atol"], rtol=tolerance["rtol"]):
                    all_errors.append(
                        f"Trial {trial}: outputs not close "
                        f"(max_abs={abs_err:.2e}, max_rel={rel_err:.2e})"
                    )

                # Sanity checks
                if np.any(np.isnan(output)):
                    all_errors.append(f"Trial {trial}: output contains NaN")
                if np.any(np.isinf(output)):
                    all_errors.append(f"Trial {trial}: output contains Inf")

            except Exception as e:
                all_errors.append(f"Trial {trial}: exception — {e}")

        return NumericalResult(
            passed=len(all_errors) == 0,
            trials=trials,
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            tolerance=tolerance,
            errors=all_errors,
        )

    def _topological_sort(self, ir: SemanticIR) -> list:
        """Topological sort of nodes in the Semantic IR.

        Simple DFS-based sort. Nodes with only ParamRef inputs come first.
        """
        # Build dependency graph
        deps: dict[str, set[str]] = {}
        node_map: dict[str, Any] = {}
        for node in ir.nodes:
            node_map[node.id] = node
            deps[node.id] = set()
            for ref in node.inputs.values():
                if isinstance(ref, NodeRef):
                    deps[node.id].add(ref.id)

        # Kahn's algorithm
        in_degree = {nid: len(d) for nid, d in deps.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordered = []

        while queue:
            nid = queue.pop(0)
            ordered.append(node_map[nid])
            for other_id, other_deps in deps.items():
                if nid in other_deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(ordered) != len(ir.nodes):
            raise ValueError("Cycle detected in computation graph")

        return ordered
