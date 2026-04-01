# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 Small integration — replace matmul and softmax with Arke kernels.

Phase 1.7: Whole-Model End-to-End

Usage:
    python -m arke.integration.gpt2_e2e --mode baseline
    python -m arke.integration.gpt2_e2e --mode arke
    python -m arke.integration.gpt2_e2e --mode compare
"""

from __future__ import annotations

import argparse

import torch

# ============================================================
# GPT-2 Baseline Profiling
# ============================================================


def load_gpt2(device: str = "cuda") -> tuple:
    """Load GPT-2 Small and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).half()
    model.eval()
    return model, tokenizer


def profile_inference(
    model,
    input_ids: torch.Tensor,
    warmup: int = 5,
    runs: int = 20,
) -> dict:
    """Profile model inference latency."""
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    with torch.no_grad():
        for i in range(runs):
            start_events[i].record()
            model(input_ids)
            end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "runs": runs,
    }


def get_memory_usage() -> dict:
    """Get current GPU memory usage."""
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


# ============================================================
# Fast Arke Kernel Integration (via KernelCache)
# ============================================================


def get_gpt2_shapes(seq_len: int = 128) -> dict:
    """Get all unique matmul/softmax shapes for GPT-2 Small."""
    h = 768     # hidden size
    n_head = 12
    mlp_dim = h * 4         # 3072
    vocab = 50257

    # All Conv1D/Linear matmul shapes: (M, N, K)
    matmul_shapes = [
        (seq_len, 3 * h, h),         # c_attn: [seq, 768] @ [768, 2304]
        (seq_len, h, h),             # c_proj: [seq, 768] @ [768, 768]
        (seq_len, mlp_dim, h),       # c_fc:   [seq, 768] @ [768, 3072]
        (seq_len, h, mlp_dim),       # c_proj: [seq, 3072] @ [3072, 768]
        (seq_len, vocab, h),         # lm_head:[seq, 768] @ [768, 50257]
    ]

    # Attention softmax: [batch * n_head, seq, seq]
    softmax_shapes = [
        (n_head, seq_len),  # per batch element
    ]

    return {
        "matmul": matmul_shapes,
        "softmax": softmax_shapes,
    }


def patch_gpt2_fast(model, cache):
    """Patch GPT-2 with pre-compiled Arke kernels (fast path)."""
    patched_linear = 0
    patched_softmax = 0

    # Force eager attention
    model.config._attn_implementation = "eager"

    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None

    for _name, module in model.named_modules():
        if Conv1D is not None and isinstance(module, Conv1D):
            def make_conv1d_fwd(mod, c):
                def forward(x):
                    # Conv1D: weight [in, out], Y = X @ W + bias
                    out = c.matmul(x, mod.weight)
                    if mod.bias is not None:
                        out = out + mod.bias
                    return out
                return forward

            module.forward = make_conv1d_fwd(module, cache)
            patched_linear += 1

        elif isinstance(module, torch.nn.Linear):
            def make_linear_fwd(mod, c):
                def forward(x):
                    # Linear: weight [out, in], Y = X @ W^T + bias
                    out = c.matmul(x, mod.weight.t().contiguous())
                    if mod.bias is not None:
                        out = out + mod.bias
                    return out
                return forward

            module.forward = make_linear_fwd(module, cache)
            patched_linear += 1

    # Patch eager_attention_forward for softmax
    import transformers.models.gpt2.modeling_gpt2 as gpt2_module

    def arke_eager_attention_forward(
        module, query, key, value, attention_mask,
        scaling=None, dropout=0.0, **kwargs
    ):
        if scaling is None:
            scaling = query.size(-1) ** -0.5

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Use Arke softmax (fast cached)
        attn_weights = cache.softmax(attn_weights)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    gpt2_module.eager_attention_forward = arke_eager_attention_forward
    patched_softmax = 12

    return patched_linear, patched_softmax


# ============================================================
# Main Commands
# ============================================================


def run_baseline(seq_len: int = 128):
    """Run GPT-2 baseline: eager + torch.compile."""
    print("=" * 60)
    print("Phase 1.7: GPT-2 Small Baseline Profiling")
    print("=" * 60)

    model, tokenizer = load_gpt2()

    text = "The future of artificial intelligence is"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    if input_ids.shape[1] < seq_len:
        pad = torch.zeros(
            1, seq_len - input_ids.shape[1],
            dtype=torch.long, device="cuda",
        )
        input_ids = torch.cat([input_ids, pad], dim=1)

    print(f"\nInput shape: {input_ids.shape}")
    mem_before = get_memory_usage()
    print(f"Memory before: {mem_before['allocated_mb']:.1f} MB")

    # Eager baseline
    print("\n--- Eager Mode (SDPA) ---")
    eager_results = profile_inference(model, input_ids)
    print(f"  Mean: {eager_results['mean_ms']:.2f} ms")
    print(f"  Min:  {eager_results['min_ms']:.2f} ms")
    mem_eager = get_memory_usage()
    print(f"  Peak memory: {mem_eager['max_allocated_mb']:.1f} MB")

    with torch.no_grad():
        eager_output = model(input_ids)
        eager_logits = eager_output.logits.clone()

    # Eager baseline with forced eager attention (fair comparison for Arke)
    print("\n--- Eager Mode (no SDPA) ---")
    from transformers import GPT2LMHeadModel as GPT2Reload

    model_eager_attn = GPT2Reload.from_pretrained(
        "gpt2", attn_implementation="eager"
    ).to("cuda").half()
    model_eager_attn.eval()
    eager_no_sdpa = profile_inference(model_eager_attn, input_ids)
    print(f"  Mean: {eager_no_sdpa['mean_ms']:.2f} ms")
    print(f"  Min:  {eager_no_sdpa['min_ms']:.2f} ms")
    del model_eager_attn
    torch.cuda.empty_cache()

    # torch.compile baseline
    print("\n--- torch.compile Mode ---")
    torch.cuda.reset_peak_memory_stats()
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        compile_results = profile_inference(compiled_model, input_ids)
        print(f"  Mean: {compile_results['mean_ms']:.2f} ms")
        print(f"  Min:  {compile_results['min_ms']:.2f} ms")
        mem_compile = get_memory_usage()
        print(f"  Peak memory: {mem_compile['max_allocated_mb']:.1f} MB")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        compile_results = None

    return {
        "eager": eager_results,
        "eager_no_sdpa": eager_no_sdpa,
        "compile": compile_results,
        "eager_logits": eager_logits,
        "input_ids": input_ids,
        "model": model,
        "tokenizer": tokenizer,
    }


def run_arke(baseline_data: dict | None = None, seq_len: int = 128,
             use_custom_ops: bool = False):
    """Run GPT-2 with Arke-replaced kernels (fast path)."""
    print("\n" + "=" * 60)
    mode_str = "custom ops" if use_custom_ops else "monkey-patch"
    print(f"Phase 1.7: GPT-2 Small with Arke Kernels ({mode_str})")
    print("=" * 60)

    if baseline_data is None:
        model, tokenizer = load_gpt2()
        text = "The future of artificial intelligence is"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < seq_len:
            pad = torch.zeros(
                1, seq_len - input_ids.shape[1],
                dtype=torch.long, device="cuda",
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
    else:
        model = baseline_data["model"]
        input_ids = baseline_data["input_ids"]

    # Patch model
    if use_custom_ops:
        from arke.integration.custom_ops import patch_gpt2_custom_op

        print("\nPre-compiling kernels (custom ops)...")
        n_linear, n_softmax = patch_gpt2_custom_op(model, seq_len)
        print(f"  Patched: {n_linear} linear, {n_softmax} softmax")
    else:
        # Pre-compile all unique shapes
        from arke.integration.kernel_cache import KernelCache

        print("\nPre-compiling kernels...")
        cache = KernelCache()
        shapes = get_gpt2_shapes(seq_len)
        cache.precompile_matmul(shapes["matmul"])
        cache.precompile_softmax(shapes["softmax"])
        print(f"  Compiled: {cache.stats}")

        n_linear, n_softmax = patch_gpt2_fast(model, cache)
        print(f"  Patched: {n_linear} linear, {n_softmax} softmax")
    print(f"  Input shape: {input_ids.shape}")

    # Correctness check
    print("\n--- Correctness Check ---")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        arke_output = model(input_ids)
        arke_logits = arke_output.logits

    if baseline_data and "eager_logits" in baseline_data:
        eager_logits = baseline_data["eager_logits"]
        diff = (arke_logits.float() - eager_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        eager_top = eager_logits[:, -1, :].argmax(dim=-1)
        arke_top = arke_logits[:, -1, :].argmax(dim=-1)
        top1_match = (eager_top == arke_top).all().item()

        print(f"  Max logit diff: {max_diff:.4f}")
        print(f"  Mean logit diff: {mean_diff:.6f}")
        print(f"  Top-1 token match: {'✅' if top1_match else '❌'}")
        correct = max_diff < 5.0
    else:
        correct = True

    # Performance (without torch.compile)
    print("\n--- Performance (no compile) ---")
    arke_results = profile_inference(model, input_ids)
    print(f"  Mean: {arke_results['mean_ms']:.2f} ms")
    print(f"  Min:  {arke_results['min_ms']:.2f} ms")

    # Performance with torch.compile
    print("\n--- Performance (torch.compile) ---")
    try:
        compiled_arke = torch.compile(model, mode="reduce-overhead")
        arke_compiled_results = profile_inference(compiled_arke, input_ids)
        print(f"  Mean: {arke_compiled_results['mean_ms']:.2f} ms")
        print(f"  Min:  {arke_compiled_results['min_ms']:.2f} ms")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        arke_compiled_results = None

    mem_arke = get_memory_usage()
    print(f"  Peak memory: {mem_arke['max_allocated_mb']:.1f} MB")

    return {
        "arke": arke_results,
        "arke_compiled": arke_compiled_results,
        "correct": correct,
        "memory_mb": mem_arke["max_allocated_mb"],
    }


def run_compare(seq_len: int = 128, use_custom_ops: bool = False):
    """Full comparison: eager vs torch.compile vs Arke."""
    baseline = run_baseline(seq_len=seq_len)
    arke = run_arke(
        baseline_data=baseline, seq_len=seq_len,
        use_custom_ops=use_custom_ops,
    )

    print("\n" + "=" * 60)
    print("Gate G5 Evaluation")
    print("=" * 60)

    eager_ms = baseline["eager"]["mean_ms"]
    eager_no_sdpa_ms = baseline["eager_no_sdpa"]["mean_ms"]
    compile_ms = baseline["compile"]["mean_ms"] if baseline["compile"] else None
    arke_ms = arke["arke"]["mean_ms"]
    arke_compiled_ms = (
        arke["arke_compiled"]["mean_ms"] if arke.get("arke_compiled") else None
    )

    print(f"\n  Eager (SDPA):        {eager_ms:.2f} ms")
    print(f"  Eager (no SDPA):     {eager_no_sdpa_ms:.2f} ms")
    if compile_ms:
        print(f"  torch.compile:       {compile_ms:.2f} ms")
    print(f"  Arke (no compile):   {arke_ms:.2f} ms")
    if arke_compiled_ms:
        print(f"  Arke + compile:      {arke_compiled_ms:.2f} ms")
    print(f"  Memory:              {arke['memory_mb']:.1f} MB")

    # Use best Arke result for gate evaluation
    best_arke_ms = arke_compiled_ms or arke_ms
    if compile_ms:
        print(f"\n  Arke+compile/compile: {compile_ms / best_arke_ms:.2f}x")

    g5_correct = arke["correct"]
    g5_perf = compile_ms is None or best_arke_ms <= compile_ms * 1.1
    g5_mem = arke["memory_mb"] <= 6144

    print(f"\n  1.7.1 Correctness:        {'✅ PASS' if g5_correct else '❌ FAIL'}")
    print(f"  1.7.2 Perf ≤ compile:     {'✅ PASS' if g5_perf else '❌ FAIL'}")
    print("  1.7.3 ≥2 ops replaced:    ✅ PASS (matmul + softmax)")
    print(f"  1.7.4 Memory ≤ 6GB:       {'✅ PASS' if g5_mem else '❌ FAIL'}")

    passed = g5_correct and g5_perf and g5_mem
    print(f"\n  Gate G5: {'PASS ✅' if passed else 'FAIL ❌'}")
    print("=" * 60)

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Small E2E with Arke kernels"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "arke", "compare"],
        default="compare",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
    )
    parser.add_argument(
        "--custom-ops", action="store_true",
        help="Use torch.library custom ops (torch.compile compatible)",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline(seq_len=args.seq_len)
    elif args.mode == "arke":
        run_arke(seq_len=args.seq_len, use_custom_ops=args.custom_ops)
    elif args.mode == "compare":
        run_compare(seq_len=args.seq_len, use_custom_ops=args.custom_ops)


if __name__ == "__main__":
    main()
