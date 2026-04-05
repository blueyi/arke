// Example 29: Argmax (Category E — reduction, index-producing)
//
// Natural Language:
//   "按行求最大值索引：X:[128, 768] → Y:[128]"
//
// Semantics: out[i] = argmax_j(X[i,j])
// Category: E (reduction — argmax)

kernel argmax_kernel(
    X: Tensor<[128, 768], f16>
) -> Tensor<[128], i64> {
    let Y = argmax(X=X, dim=1);
    return Y;
}

strategy argmax_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per CTA — each warp tracks max+index per row");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: reduction with index tracking, memory-bound");
}
