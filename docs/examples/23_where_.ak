// Example 23: Where (Category D — elementwise conditional selection)
//
// Natural Language:
//   "条件选择：where(cond, X, Y)，cond:[128, 768] bool, X/Y:[128, 768]"
//
// Semantics: out[i,j] = cond[i,j] ? X[i,j] : Y[i,j]
// Category: D (elementwise — conditional)

kernel where_kernel(
    Cond: Tensor<[128, 768], bool>,
    X: Tensor<[128, 768], f16>,
    Y: Tensor<[128, 768], f16>
) -> Tensor<[128, 768], f16> {
    let Z = where_(Cond=Cond, X=X, Y=Y);
    return Z;
}

strategy where_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise ternary, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: pure elementwise, memory-bound");
}
