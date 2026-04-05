// Example 25: Neg (Category D — elementwise unary)
//
// Natural Language:
//   "取负：-X，X:[128, 3072]"
//
// Semantics: out[i,j] = -X[i,j]
// Category: D (elementwise — unary)

kernel neg_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = neg(X=X);
    return Y;
}

strategy neg_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — trivial elementwise, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: pure memory-bound negation");
}
