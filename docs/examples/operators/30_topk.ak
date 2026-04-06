// Example 30: Top-K (Category E — reduction, multi-value)
//
// Natural Language:
//   "按行取 Top-K：X:[128, 32000] → Values:[128, 50], Indices:[128, 50]"
//
// Semantics: values[i,:], indices[i,:] = topk(X[i,:], k=50)
// Category: E (reduction — topk selection)
// Note: returns two tensors; we model the fused values output here

kernel topk_kernel(
    X: Tensor<[128, 32000], f16>
) -> Tensor<[128, 50], f16> {
    let Y = topk(X=X, k=50);
    return Y;
}

strategy topk_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("1 row per CTA — topk over 32000 vocab needs full CTA cooperation");
    launch_config(num_warps=8, num_stages=1)
        @rationale("8 warps: large reduction dim needs more parallelism within CTA");
}
