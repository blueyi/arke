// Example 24: Cast (Category D — elementwise type conversion)
//
// Natural Language:
//   "类型转换：f32→f16，X:[128, 3072]"
//
// Semantics: out[i,j] = (f16) X[i,j]
// Category: D (elementwise — type cast)

kernel cast_kernel(
    X: Tensor<[128, 3072], f32>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = cast(X=X);
    return Y;
}

strategy cast_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise cast, maximize occupancy");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: pure memory-bound type conversion");
}
