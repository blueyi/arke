// Example 14: Grouped MatMul (Category A — compute, MoE expert dispatch)
//
// Natural Language:
//   "MoE expert matmul: 每个 batch 样本选一个 expert weight，X:[4,128,768] × W[experts[b]]:[8,768,3072]"
//
// Semantics: Y[b,i,j] = sum_k(X[b,i,k] * W[indices[b],k,j])
// Category: A (compute — grouped matmul)

kernel grouped_matmul_kernel(
    X: Tensor<[4, 128, 768], f16>,
    W: Tensor<[8, 768, 3072], f16>,
    indices: Tensor<[4], i32>
) -> Tensor<[4, 128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = grouped_matmul(X=X, W=W, indices=indices);
    return Y;
}

strategy grouped_matmul_kernel_strategy for target("nvidia_ampere") {
    tile(loop="M", factors=[64])
        @rationale("64-row tile: good tensor-core occupancy per expert");
    tile(loop="N", factors=[128])
        @rationale("128-col tile: covers 3072/128=24 blocks across output");
    tile(loop="K", factors=[32])
        @rationale("K-tile=32: A+W smem fits comfortably");
    compute(warps=4, num_stages=3)
        @rationale("3 stages: hide global→shared latency for weight fetch");
}
