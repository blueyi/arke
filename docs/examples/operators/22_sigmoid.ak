// Example 22: Sigmoid (Category D — elementwise activation)
//
// Natural Language:
//   "Sigmoid激活：σ(X) = 1/(1+exp(-X))，X:[128, 3072]"
//
// Semantics: out[i,j] = 1 / (1 + exp(-X[i,j]))
// Category: D (elementwise — activation)

kernel sigmoid_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = sigmoid(X=X);
    return Y;
}

strategy sigmoid_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: pure elementwise, memory-bound");
}
