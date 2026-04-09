// Example 23: Where (Category D — elementwise conditional selection)
//
// Natural Language:
//   "条件选择：where(cond, X, Y)，cond:[128, 768] bool, X/Y:[128, 768]"
//
// Semantics: out[i,j] = cond[i,j] ? X[i,j] : Y[i,j]
// Category: D (elementwise — conditional)

kernel where_kernel(
    Cond: Tensor<[B, S], bool>,
    X: Tensor<[B, S], f16>,
    Y: Tensor<[B, S], f16>
) -> Tensor<[B, S], f16>
where B: dynamic(max=64), S: dynamic(max=8192)
{
    let Z = where_(Cond=Cond, X=X, Y=Y);
    return Z;
}

strategy where_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise ternary, maximize occupancy");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: pure elementwise, memory-bound");
}
