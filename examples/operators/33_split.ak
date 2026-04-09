// Example 33: Split (Category A — memory/layout, tensor split)
//
// Natural Language:
//   "张量分割：X:[128, 1024] → Y:[128, 512] (取前半, dim=1)"
//
// Semantics: Y = split(X, sections=2, dim=1)[0] → shape [128, 512]
// Category: A (memory — layout/reshape)
// Note: single-output modeling; split returns a slice view

kernel split_kernel(
    X: Tensor<[128, 1024], f16>
) -> Tensor<[128, 512], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = split(X=X, sections=2, dim=1);
    return Y;
}

strategy split_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — slice copy, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: pure memory operation, no compute");
}
