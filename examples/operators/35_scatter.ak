// Example 35: Scatter (Category A — memory/layout, indexed write)
//
// Natural Language:
//   "散布操作：src:[128, 64] 按 idx:[128, 64] 写入 X:[128, 768] (dim=1)"
//
// Semantics: Y[i, idx[i,j]] = src[i,j]
// Category: A (memory — gather/scatter)

kernel scatter_kernel(
    X: Tensor<[128, 768], f16>,
    idx: Tensor<[128, 64], i64>,
    src: Tensor<[128, 64], f16>
) -> Tensor<[128, 768], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = scatter(X=X, idx=idx, src=src);
    return Y;
}

strategy scatter_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — random write pattern, small blocks avoid conflicts");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: irregular memory write, no pipeline benefit");
}
