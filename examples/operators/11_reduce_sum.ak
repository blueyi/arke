// Example 11: Reduce Sum (Category E — reduction)
//
// Natural Language:
//   "按行求和：X:[128, 768] → Y:[128]"
//
// Semantics: out[i] = sum_j(X[i,j])
// Category: E (reduction)

kernel reduce_sum_kernel(
    X: Tensor<[128, 768], f16>
) -> Tensor<[128], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = reduce_sum(X=X);
    return Y;
}

strategy reduce_sum_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per CTA — each warp reduces one row");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: reduction is compute+memory balanced");
}
