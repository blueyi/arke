// Example 31: Cumulative Sum (Category E — scan/prefix-sum)
//
// Natural Language:
//   "按行累积求和：X:[128, 768] → Y:[128, 768]"
//
// Semantics: out[i,j] = sum_{k<=j}(X[i,k])
// Category: E (reduction — prefix sum / scan)

kernel cumsum_kernel(
    X: Tensor<[128, 768], f16>
) -> Tensor<[128, 768], f16> {
    let Y = cumsum(X=X, dim=1);
    return Y;
}

strategy cumsum_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per CTA — each warp does Blelloch scan on one row");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: scan is inherently sequential within a row");
}
