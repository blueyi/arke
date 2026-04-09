// Example 10: Elementwise Mul (Category D — elementwise)
//
// Natural Language:
//   "逐元素乘法：A:[128, 768] × B:[128, 768]"
//
// Semantics: out[i,j] = A[i,j] * B[i,j]
// Category: D (elementwise)

kernel mul_kernel(
    A: Tensor<[128, 768], f16>,
    B: Tensor<[128, 768], f16>
) -> Tensor<[128, 768], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = mul(A=A, B=B);
    return Y;
}

strategy mul_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — simple elementwise, maximize occupancy");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: pure memory-bound, no reuse to pipeline");
}
