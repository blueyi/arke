// Example 00: ReLU (Category D — elementwise)
//
// Natural Language:
//   "ReLU激活函数：max(X, 0)，X:[128, 3072]"
//
// Semantics: out[i,j] = max(X[i,j], 0)
// Category: D (elementwise)

kernel relu_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = relu(X=X);
    return Y;
}

strategy relu_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — fully memory-bound, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: pure elementwise, no arithmetic pipeline benefit");
}
