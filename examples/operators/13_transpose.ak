// Example 13: Transpose (Category A — memory/layout)
//
// Natural Language:
//   "转置矩阵：X:[512, 128] → Y:[128, 512]"
//
// Semantics: out[j,i] = X[i,j]
// Category: A (compute — memory layout transform)

kernel transpose_kernel(
    X: Tensor<[512, 128], f16>
) -> Tensor<[128, 512], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = transpose(X=X);
    return Y;
}

strategy transpose_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[32])
        @rationale("32-row tile: coalesced read + shared memory staging avoids write conflict");
    tile(loop="col", factors=[32])
        @rationale("32-col tile: 32×32 smem block fits in 2KB, zero bank conflicts");
    place(tensor="X_tile", memory="shared")
        @rationale("stage through shared mem to coalesce global writes");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: bandwidth-bound, no arithmetic to pipeline");
}
