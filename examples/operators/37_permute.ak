// Example 37: Permute (Category A — memory/layout, dimension reorder)
//
// Natural Language:
//   "维度置换：X:[2, 12, 128, 64] → Y:[2, 128, 12, 64] (perm=[0,2,1,3])"
//
// Semantics: out[b, s, h, d] = X[b, h, s, d]
// Category: A (memory — layout transform)

kernel permute_kernel(
    X: Tensor<[2, 12, 128, 64], f16>
) -> Tensor<[2, 128, 12, 64], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = permute(X=X, dims=[0, 2, 1, 3]);
    return Y;
}

strategy permute_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[32])
        @rationale("32-row tile: coalesced read + shared memory staging");
    place(tensor="X_tile", memory="shared")
        @rationale("stage through shared mem to coalesce global writes after permute");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: bandwidth-bound layout transform");
}
