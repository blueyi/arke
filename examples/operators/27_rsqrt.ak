// Example 27: Rsqrt (Category D — elementwise transcendental)
//
// Natural Language:
//   "逆平方根：1/√X，X:[128, 3072]"
//
// Semantics: out[i,j] = 1 / sqrt(X[i,j])
// Category: D (elementwise — transcendental)

kernel rsqrt_kernel(
    X: Tensor<[128, 3072], f16>
) -> Tensor<[128, 3072], f16> {
    let Y = rsqrt(X=X);
    return Y;
}

strategy rsqrt_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — fast HW rsqrt on Ampere, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: memory-bound, rsqrt is single SFU instruction");
}
