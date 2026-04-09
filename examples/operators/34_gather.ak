// Example 34: Gather (Category A — memory/layout, indexed read)
//
// Natural Language:
//   "聚集操作：X:[128, 768] 按 idx:[128, 64] 在 dim=1 聚集 → Y:[128, 64]"
//
// Semantics: out[i,j] = X[i, idx[i,j]]
// Category: A (memory — gather/scatter)

kernel gather_kernel(
    X: Tensor<[128, 768], f16>,
    idx: Tensor<[128, 64], i64>
) -> Tensor<[128, 64], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = gather(X=X, idx=idx);
    return Y;
}

strategy gather_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — random access pattern, keep blocks small");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: irregular memory access, no pipeline benefit");
}
