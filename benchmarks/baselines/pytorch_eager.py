# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P3: PyTorch eager baselines.

Identical to cuBLAS runner but explicitly named for clarity
in benchmark reports when comparing Arke vs "PyTorch default".
This runner represents what a user gets out of the box.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from benchmarks.baselines.base import BaselineRunner, register_baseline

# All ops with a PyTorch eager implementation in get_fn()
_SUPPORTED_OPS = frozenset({
    # OT0 Elementwise
    "relu", "gelu", "silu", "tanh", "sigmoid",
    "add", "mul", "where_", "cast", "neg", "exp", "rsqrt",
    # OT1 Reduction
    "softmax", "layernorm", "rmsnorm", "rmsnorm_residual",
    "reduce_sum", "reduce_max", "reduce_mean", "argmax", "topk", "cumsum",
    # OT2 Data Movement & Dense
    "matmul", "batch_matmul", "transpose", "concat", "split",
    "gather", "scatter", "embedding", "permute", "copy_",
    # OT3 Fused Compound
    "silu_and_mul", "gelu_and_mul", "swiglu_packed", "cross_entropy", "fused_linear_cross_entropy",
    # OT4 Attention
    "flash_attention", "grouped_query_attention", "cross_attention",
    "multi_latent_attention", "paged_attention",
    # OT3 Quantization (new)
    "quantize_per_token", "dequantize_per_channel",
    # OT2 Special (new)
    "rope", "grouped_matmul",
})


@register_baseline
class PyTorchEagerRunner(BaselineRunner):
    """P3: PyTorch eager mode (user's default)."""

    @property
    def name(self) -> str:
        return "PyTorch-eager"

    @property
    def priority(self) -> int:
        return 3

    @property
    def source(self) -> str:
        v = torch.__version__
        return (
            f"PyTorch {v} eager mode (default dispatch) | "
            "https://pytorch.org | License: BSD-3-Clause"
        )

    @property
    def available(self) -> bool:
        return torch.cuda.is_available()

    def supports(self, op: str) -> bool:
        return op in _SUPPORTED_OPS

    def run_with_inputs(
        self,
        op: str,
        *inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if op == "relu" and len(inputs) == 1:
            return F.relu(inputs[0])
        if op == "gelu" and len(inputs) == 1:
            return F.gelu(inputs[0])
        if op == "silu" and len(inputs) == 1:
            return F.silu(inputs[0])
        if op == "tanh" and len(inputs) == 1:
            return torch.tanh(inputs[0])
        if op == "sigmoid" and len(inputs) == 1:
            return torch.sigmoid(inputs[0])
        if op == "neg" and len(inputs) == 1:
            return -inputs[0]
        if op == "exp" and len(inputs) == 1:
            return torch.exp(inputs[0])
        if op == "rsqrt" and len(inputs) == 1:
            return torch.rsqrt(inputs[0])
        if op == "softmax" and len(inputs) == 1:
            return F.softmax(inputs[0], dim=-1)
        if op == "layernorm" and len(inputs) == 1:
            x = inputs[0]
            w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            b = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
            return F.layer_norm(x, [x.shape[-1]], w, b)
        if op == "rmsnorm" and len(inputs) == 1:
            x = inputs[0]
            w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            eps = 1e-6
            return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
        if op == "rmsnorm_residual" and len(inputs) == 2:
            x, residual = inputs
            w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            eps = 1e-6
            y = x + residual
            return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)) * w
        if op == "add" and len(inputs) == 2:
            return inputs[0] + inputs[1]
        if op == "mul" and len(inputs) == 2:
            return inputs[0] * inputs[1]
        if op == "where_" and len(inputs) == 3:
            return torch.where(inputs[0].bool(), inputs[1], inputs[2])
        if op == "cast" and len(inputs) == 1:
            return inputs[0].to(torch.float32)
        if op == "reduce_sum" and len(inputs) == 1:
            return torch.sum(inputs[0], dim=-1)
        if op == "reduce_max" and len(inputs) == 1:
            return torch.max(inputs[0], dim=-1).values
        if op == "reduce_mean" and len(inputs) == 1:
            return torch.mean(inputs[0], dim=-1)
        if op == "argmax" and len(inputs) == 1:
            return torch.argmax(inputs[0], dim=-1)
        if op == "cumsum" and len(inputs) == 1:
            return torch.cumsum(inputs[0], dim=-1)
        if op == "topk" and len(inputs) == 1:
            k = min(kwargs.get("k", 4), inputs[0].shape[-1])
            return torch.topk(inputs[0], k=k, dim=-1).values
        if op == "matmul" and len(inputs) == 2:
            return torch.matmul(inputs[0], inputs[1])
        if op == "batch_matmul" and len(inputs) == 2:
            return torch.bmm(inputs[0], inputs[1])
        if op == "transpose" and len(inputs) == 1:
            return inputs[0].T
        if op == "concat" and len(inputs) == 2:
            return torch.cat([inputs[0], inputs[1]], dim=-1)
        if op == "split" and len(inputs) == 1:
            return torch.chunk(inputs[0], 2, dim=-1)
        if op == "gather" and len(inputs) == 2:
            return torch.gather(inputs[0], 1, inputs[1].long())
        if op == "scatter" and len(inputs) == 3:
            return torch.zeros_like(inputs[0]).scatter_(1, inputs[1].long(), inputs[2])
        if op == "embedding" and len(inputs) == 2:
            return F.embedding(inputs[0].long(), inputs[1])
        if op == "permute" and len(inputs) == 1:
            return inputs[0].permute(0, 2, 1)
        if op == "copy_" and len(inputs) == 1:
            return inputs[0].clone()
        if op == "silu_and_mul" and len(inputs) == 1:
            x1, x2 = inputs[0].chunk(2, dim=-1)
            return F.silu(x1) * x2
        if op == "swiglu_packed" and len(inputs) == 2:
            x1, x2 = inputs[0].chunk(2, dim=-1)
            return (F.silu(x1) * x2) @ inputs[1]
        if op == "gelu_and_mul" and len(inputs) == 1:
            x1, x2 = inputs[0].chunk(2, dim=-1)
            return F.gelu(x1) * x2
        if op == "cross_entropy" and len(inputs) == 2:
            return F.cross_entropy(inputs[0].to(torch.float32), inputs[1].long())
        if op == "fused_linear_cross_entropy" and len(inputs) == 3:
            x, w, labels = inputs
            logits = x.to(torch.float32) @ w.to(torch.float32).T
            return F.cross_entropy(logits, labels.long())
        if op == "quantize_per_token" and len(inputs) == 1:
            x = inputs[0]
            scales = torch.amax(torch.abs(x), dim=1, keepdim=True)
            scales = torch.clamp(scales, min=1e-8)
            x_q = torch.round(x / scales * 127).to(torch.int8)
            return x_q, scales.squeeze(1)
        if op == "dequantize_per_channel" and len(inputs) == 2:
            return inputs[0].to(inputs[1].dtype) * inputs[1].unsqueeze(0)
        if op == "grouped_matmul" and len(inputs) == 2:
            a_groups, b_groups = inputs
            return torch.cat([a @ b for a, b in zip(a_groups, b_groups, strict=False)], dim=0)
        if op == "rope" and len(inputs) == 1:
            x = inputs[0]
            head_dim = x.shape[-1]
            if head_dim % 2 != 0:
                # RoPE rotates pairs of channels — head_dim must be even.
                # Odd-D shapes (e.g. non-align-1 D=65, non-align-2 D=127)
                # exist in the catalog to stress allocation/tiling logic
                # for *other* ops; for RoPE itself they're mathematically
                # ill-defined. Return None so the harness records this
                # row as 'unsupported' with a typed reason rather than
                # crashing in cat([-x2, x1]) shape mismatch.
                return None
            seq_len = x.shape[1]
            # Numerical-stability fix (2026-05-14, Q5a): compute sin/cos in
            # fp32 then cast back. fp16 torch.arange(seq) loses integer
            # precision at seq>=2048 and overflows past fp16_max≈65504, so at
            # extreme-long (seq=65536) freqs decays to NaN/inf and the rope
            # output is all NaN. Same fix mirrored in bench_l1._torch_reference
            # and arke_runner (run_with_inputs + _fill_missing_inputs). Matches
            # HF Transformers / Liger-Kernel / FlashInfer rotary convention
            # (fp32 trig, fp16 hadamard).
            freqs = torch.einsum(
                "i,j->ij",
                torch.arange(seq_len, device=x.device, dtype=torch.float32),
                1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)),
            )
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_emb = torch.cos(emb).unsqueeze(0).to(x.dtype)
            sin_emb = torch.sin(emb).unsqueeze(0).to(x.dtype)
            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]
            rotated = torch.cat([-x2, x1], dim=-1)
            return x * cos_emb + rotated * sin_emb
        if op == "flash_attention" and len(inputs) == 3:
            return F.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2], is_causal=True)
        if op == "cross_attention" and len(inputs) == 3:
            return F.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2])
        if op == "grouped_query_attention" and len(inputs) == 3:
            q, k, v = inputs
            repeats = q.shape[0] // max(k.shape[0], 1)
            k_exp = k.repeat_interleave(repeats, dim=0)
            v_exp = v.repeat_interleave(repeats, dim=0)
            return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        if op == "multi_latent_attention" and len(inputs) == 3:
            # P3 degraded golden for MLA: treat (Q,K,V) as plain SDPA.
            # The "latent" projection collapse is encoded in the input
            # shapes upstream — here we just measure SDPA behavior on
            # whatever rank-3 tensors arrive. Audit emits
            # mla_golden_degraded=true at the bench_l1 layer.
            return F.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2])
        if op == "paged_attention" and len(inputs) == 3:
            # P3 degraded golden for paged_attention: treat the KV cache
            # as already gathered into contiguous (B,S,D) tensors and run
            # SDPA. Real paged_attention does block-table indirection,
            # which only vLLM/FlashInfer model exactly.
            return F.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2])
        return None

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        # ── OT0 Elementwise ─────────────────────────────────────────
        if op == "relu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.relu(X)

        elif op == "gelu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.gelu(X)

        elif op == "silu":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.silu(X)

        elif op == "tanh":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.tanh(X)

        elif op == "sigmoid":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.sigmoid(X)

        elif op == "add":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: A + B

        elif op == "mul":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: A * B

        elif op == "where_":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            cond = torch.randn(M, N, device="cuda") > 0
            return lambda: torch.where(cond, A, B)

        elif op == "cast":
            # Cast float32 → target dtype
            X = torch.randn(M, N, device="cuda", dtype=torch.float32)
            return lambda: X.to(dtype)

        elif op == "neg":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: -X

        elif op == "exp":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.exp(X)

        elif op == "rsqrt":
            X = torch.randn(M, N, device="cuda", dtype=dtype).abs() + 1e-6
            return lambda: torch.rsqrt(X)

        # ── OT1 Reduction ───────────────────────────────────────────
        elif op == "softmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: F.softmax(X, dim=-1)

        elif op == "layernorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            b = torch.zeros(N, device="cuda", dtype=dtype)
            return lambda: F.layer_norm(X, [N], w, b)

        elif op == "rmsnorm":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            eps = 1e-6
            return lambda: (X * torch.rsqrt(
                X.pow(2).mean(-1, keepdim=True) + eps
            )) * w

        elif op == "rmsnorm_residual":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            residual = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            eps = 1e-6
            return lambda: (
                (X + residual) * torch.rsqrt(
                    (X + residual).pow(2).mean(-1, keepdim=True) + eps
                )
            ) * w

        elif op == "reduce_sum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.sum(dim=-1)

        elif op == "reduce_max":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.max(dim=-1).values

        elif op == "reduce_mean":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.mean(dim=-1)

        elif op == "argmax":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.argmax(dim=-1)

        elif op == "topk":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            k = min(50, N)
            return lambda: torch.topk(X, k=k, dim=-1).values

        elif op == "cumsum":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.cumsum(X, dim=-1)

        # ── OT2 Data Movement & Dense ───────────────────────────────
        elif op == "matmul":
            A = torch.randn(M, K, device="cuda", dtype=dtype)
            B = torch.randn(K, N, device="cuda", dtype=dtype)
            return lambda: torch.matmul(A, B)

        elif op == "batch_matmul":
            from benchmarks.baselines._shared_inputs import build_batch_matmul_inputs
            A, B = build_batch_matmul_inputs(M, N, K, dtype)
            return lambda: torch.bmm(A, B)

        elif op == "transpose":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            # Materialize (.contiguous()): Arke's transpose kernel writes a real
            # transposed tensor, so the eager baseline must also materialize for
            # an apples-to-apples ratio. A bare X.T is a lazy O(1) view (~1us, no
            # data movement) — comparing Arke's materializing kernel against it
            # produced the bogus ~0.01x ratio. cublas/flaggems already use
            # .contiguous(). See docs/benchmark/harness-perf-shape-encoding-bug.md
            # and the transpose audit-only note in benchmark-protocol.md.
            return lambda: X.T.contiguous()

        elif op == "concat":
            A = torch.randn(M, N, device="cuda", dtype=dtype)
            B = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.cat([A, B], dim=-1)

        elif op == "split":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            split_size = max(N // 2, 1)
            return lambda: torch.split(X, split_size, dim=-1)

        elif op == "gather":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            idx = torch.randint(0, N, (M, N), device="cuda")
            return lambda: torch.gather(X, 1, idx)

        elif op == "scatter":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            idx = torch.randint(0, N, (M, N), device="cuda")
            src = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: torch.zeros_like(X).scatter_(1, idx, src)

        elif op == "embedding":
            vocab_size = M
            seq_len = N
            weight = torch.randn(vocab_size, max(K, 128),
                                 device="cuda", dtype=dtype)
            indices = torch.randint(0, vocab_size, (seq_len,),
                                    device="cuda")
            return lambda: F.embedding(indices, weight)

        elif op == "permute":
            # 3D tensor: (M, N, K) → (M, K, N)
            dim2 = max(K, 64)
            X = torch.randn(M, N, dim2, device="cuda", dtype=dtype)
            # Materialize: like transpose, a bare permute is a lazy view. Arke
            # writes a real permuted tensor, so materialize for a fair ratio.
            return lambda: X.permute(0, 2, 1).contiguous()

        elif op == "copy_":
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            return lambda: X.clone()

        # ── OT3 Fused Compound ──────────────────────────────────────
        elif op == "silu_and_mul":
            from benchmarks.baselines._shared_inputs import build_gated_perf_inputs
            X = build_gated_perf_inputs(M, N, dtype)
            x1, x2 = X.chunk(2, dim=-1)
            return lambda: F.silu(x1) * x2

        elif op == "swiglu_packed":
            # X[M, 2K] → split to hidden K, then project by W[K, N].
            K_eff = max(K, 1)
            X = torch.randn(M, 2 * K_eff, device="cuda", dtype=dtype)
            W = torch.randn(K_eff, N, device="cuda", dtype=dtype)
            return lambda: (F.silu(X[:, :K_eff]) * X[:, K_eff:]) @ W

        elif op == "gelu_and_mul":
            from benchmarks.baselines._shared_inputs import build_gated_perf_inputs
            X = build_gated_perf_inputs(M, N, dtype)
            x1, x2 = X.chunk(2, dim=-1)
            return lambda: F.gelu(x1) * x2

        elif op == "cross_entropy":
            num_classes = N
            logits = torch.randn(M, num_classes, device="cuda",
                                 dtype=torch.float32)
            labels = torch.randint(0, num_classes, (M,), device="cuda")
            return lambda: F.cross_entropy(logits, labels)

        elif op == "fused_linear_cross_entropy":
            # X[M, K] @ W[N, K].T → logits[M, N] → cross_entropy
            hidden = max(K, 128)
            num_classes = N
            X = torch.randn(M, hidden, device="cuda", dtype=dtype)
            W = torch.randn(num_classes, hidden, device="cuda", dtype=dtype)
            labels = torch.randint(0, num_classes, (M,), device="cuda")
            return lambda: F.cross_entropy(
                X.to(torch.float32) @ W.to(torch.float32).T, labels
            )

        # ── OT4 Attention ───────────────────────────────────────────
        elif op == "flash_attention":
            # M = batch*heads, N = seq_len, K = head_dim
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            Q = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, seq_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            return lambda: F.scaled_dot_product_attention(
                Q, K_, V, is_causal=True,
            )

        elif op == "grouped_query_attention":
            # GQA: Q has more heads than K/V; repeat K/V to match
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            num_kv_groups = max(batch_heads // 4, 1)
            Q = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(num_kv_groups, seq_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(num_kv_groups, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            repeats = batch_heads // num_kv_groups
            K_exp = K_.repeat_interleave(repeats, dim=0)
            V_exp = V.repeat_interleave(repeats, dim=0)
            return lambda: F.scaled_dot_product_attention(
                Q, K_exp, V_exp, is_causal=True,
            )

        elif op == "cross_attention":
            # Cross-attention: Q from decoder, K/V from encoder (Sq != Skv).
            from benchmarks.baselines._runtime_ctx import get_current_shape
            shape = get_current_shape()
            if shape is not None and getattr(shape, "Skv", None) is not None:
                batch_heads = shape.B * shape.H
                q_len = shape.S
                kv_len = shape.Skv
                head_dim = shape.D
            else:
                batch_heads = M
                q_len = max(N // 2, 1)
                kv_len = N
                head_dim = max(K, 64)
            Q = torch.randn(batch_heads, q_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, kv_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, kv_len, head_dim,
                            device="cuda", dtype=dtype)
            return lambda: F.scaled_dot_product_attention(Q, K_, V)

        elif op == "multi_latent_attention":
            # P3 degraded golden — see run_with_inputs comment. Plain SDPA
            # on (Q,K,V); the latent compression is encoded by the caller's
            # shape choice (smaller K/V heads or compressed head_dim).
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            Q = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, seq_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            return lambda: F.scaled_dot_product_attention(
                Q, K_, V, is_causal=True,
            )

        elif op == "paged_attention":
            # P3 degraded golden — SDPA on already-gathered KV (no block
            # table indirection). Real vLLM paged_attention does indexed
            # gather; this gives the perf/correctness lower bound.
            batch_heads = M
            seq_len = N
            head_dim = max(K, 64)
            Q = torch.randn(batch_heads, 1, head_dim,
                            device="cuda", dtype=dtype)
            K_ = torch.randn(batch_heads, seq_len, head_dim,
                             device="cuda", dtype=dtype)
            V = torch.randn(batch_heads, seq_len, head_dim,
                            device="cuda", dtype=dtype)
            return lambda: F.scaled_dot_product_attention(Q, K_, V)

        # ── OT3 Quantization ────────────────────────────────────────
        elif op == "quantize_per_token":
            # Quantize: X (M, N) -> (M, N) int8 + scale (M,)
            X = torch.randn(M, N, device="cuda", dtype=dtype)
            def quantize_per_token():
                # Per-token quantization: scale = max(abs(X)) / 127
                scales = torch.amax(torch.abs(X), dim=1, keepdim=True)
                scales = torch.clamp(scales, min=1e-8)
                X_q = torch.round(X / scales * 127).to(torch.int8)
                return X_q, scales.squeeze(1)
            return quantize_per_token

        elif op == "dequantize_per_channel":
            # Dequantize: X_q (M, N) int8 + scale (N,) -> (M, N) fp
            X_q = torch.randint(-128, 127, (M, N), device="cuda", dtype=torch.int8)
            scale = torch.randn(N, device="cuda", dtype=dtype).abs() + 0.01
            def dequantize_per_channel():
                return X_q.to(dtype) * scale.unsqueeze(0)
            return dequantize_per_channel

        elif op == "rope":
            # RoPE: apply rotary position embeddings
            batch_size = M
            seq_len = N
            head_dim = max(K, 64)
            if head_dim % 2 != 0:
                # RoPE rotates pairs of channels — head_dim must be even.
                # Catalog shapes non-align-1 (D=65) / non-align-2 (D=127)
                # exist to stress alignment for other ops; for rope itself
                # they're mathematically ill-defined. Decline so the
                # harness records this row as 'unsupported' rather than
                # crashing in cat([-x2, x1]) shape mismatch.
                return None
            X = torch.randn(batch_size, seq_len, head_dim, device="cuda", dtype=dtype)
            def rope():
                # Numerical-stability fix (2026-05-14, Q5a): compute sin/cos in
                # fp32 then cast back. fp16 arange(seq) loses integer precision
                # at seq>=2048 and overflows past fp16_max≈65504, so at
                # extreme-long (seq=65536) the entire rope output is NaN. Same
                # fix mirrored in bench_l1._torch_reference and arke_runner.
                inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
                t = torch.arange(seq_len, device="cuda", dtype=torch.float32)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat([freqs, freqs], dim=-1)
                cos_emb = torch.cos(emb).unsqueeze(0).to(dtype)
                sin_emb = torch.sin(emb).unsqueeze(0).to(dtype)
                x1 = X[..., :head_dim // 2]
                x2 = X[..., head_dim // 2:]
                rotated = torch.cat([-x2, x1], dim=-1)
                return X * cos_emb + rotated * sin_emb
            return rope

        elif op == "grouped_matmul":
            from benchmarks.baselines._shared_inputs import build_grouped_matmul_inputs
            X, W, idx = build_grouped_matmul_inputs(M, N, K, dtype)
            # Per-group reference: group b uses expert weight W[idx[b]].
            ng = min(X.shape[0], W.shape[0])
            idx_l = idx.tolist()
            def grouped_matmul():
                return torch.stack([X[b] @ W[idx_l[b]] for b in range(ng)], dim=0)
            return grouped_matmul

        # ── Unsupported ─────────────────────────────────────────────
        return None
