// Example 07: SiLU Activation (elementwise, LLaMA FFN)
//
// Natural Language:
//   "对 LLaMA FFN gate output (batch=128, hidden=11008) 做 SiLU 激活"
//
// Semantics: out[i,j] = x[i,j] * sigmoid(x[i,j])
// Category: D (elementwise)

kernel silu(
    X: Tensor<[128, 11008], f16>
) -> Tensor<[128, 11008], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = silu(X=X);
    return Y;
}

strategy silu_strategy for target("nvidia_ampere") {
    tile(loop="flat", factors=[4096])
        @rationale("flat 1D tiling — elementwise has no data dependency between rows");
    vectorize(loop="flat", width=8)
        @rationale("vec8 for f16: 128-bit load/store matches L2 cache line width");
    launch_config(num_warps=4, num_stages=1)
        @rationale("silu is memory-bound; 4 warps sufficient for bandwidth saturation");
}
