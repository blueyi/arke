// Example 28: Reduce Mean (Category E — reduction)
//
// Natural Language:
//   "按行求均值：X:[128, 768] → Y:[128]"
//
// Semantics: out[i] = mean_j(X[i,j]) = sum_j(X[i,j]) / 768
// Category: E (reduction — mean)

kernel reduce_mean_kernel(
    X: Tensor<[128, 768], f16>
) -> Tensor<[128], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = reduce_mean(X=X);
    return Y;
}

strategy reduce_mean_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per CTA — each warp reduces one row then divides by N");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: reduction is compute+memory balanced");
}
