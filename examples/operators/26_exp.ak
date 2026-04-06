// Example 26: Exp (Category D — elementwise transcendental)
//
// Natural Language:
//   "指数函数：exp(X)，X:[128, 3072]"
//
// Semantics: out[i,j] = exp(X[i,j])
// Category: D (elementwise — transcendental)

kernel exp_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = exp(X=X);
    return Y;
}

strategy exp_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise transcendental, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: memory-bound exponential");
}
