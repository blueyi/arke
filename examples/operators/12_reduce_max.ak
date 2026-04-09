// Example 12: Reduce Max (Category E — reduction)
//
// Natural Language:
//   "按行取最大值：X:[128, 768] → Y:[128]"
//
// Semantics: out[i] = max_j(X[i,j])
// Category: E (reduction)

kernel reduce_max_kernel(
    X: Tensor<[128, 768], f16>
) -> Tensor<[128], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = reduce_max(X=X);
    return Y;
}

strategy reduce_max_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per CTA — each warp scans one row for max");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: reduction is memory-bound");
}
