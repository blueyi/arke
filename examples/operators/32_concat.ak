// Example 32: Concat (Category A — memory/layout, tensor concatenation)
//
// Natural Language:
//   "张量拼接：A:[128, 768] + B:[128, 256] → Y:[128, 1024] (dim=1)"
//
// Semantics: Y = concat(A, B, dim=1) → shape [128, 1024]
// Category: A (memory — layout/reshape)

kernel concat_kernel(
    A: Tensor<[128, 768], f16>,
    B: Tensor<[128, 256], f16>
) -> Tensor<[128, 1024], f16> {
    let Y = concat(A=A, B=B, dim=1);
    return Y;
}

strategy concat_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — copy-based, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: pure memory copy, no compute to pipeline");
}
