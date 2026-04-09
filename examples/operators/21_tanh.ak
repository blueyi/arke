// Example 21: Tanh (Category D — elementwise activation)
//
// Natural Language:
//   "双曲正切激活：tanh(X)，X:[128, 3072]"
//
// Semantics: out[i,j] = tanh(X[i,j])
// Category: D (elementwise — activation)

kernel tanh_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = tanh(X=X);
    return Y;
}

strategy tanh_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise, maximize occupancy");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: pure elementwise, memory-bound");
}
