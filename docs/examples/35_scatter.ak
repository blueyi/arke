// Example 35: Scatter (Category A — memory/layout, indexed write)
//
// Natural Language:
//   "散布操作：Src:[128, 64] 按 Index:[128, 64] 写入 Y:[128, 768] (dim=1)"
//
// Semantics: Y[i, Index[i,j]] = Src[i,j]
// Category: A (memory — gather/scatter)

kernel scatter_kernel(
    Src: Tensor<[128, 64], f16>,
    Index: Tensor<[128, 64], i64>,
    Dst: Tensor<[128, 768], f16>
) -> Tensor<[128, 768], f16> {
    let Y = scatter(Src=Src, Index=Index, Dst=Dst, dim=1);
    return Y;
}

strategy scatter_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — random write pattern, small blocks avoid conflicts");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: irregular memory write, no pipeline benefit");
}
