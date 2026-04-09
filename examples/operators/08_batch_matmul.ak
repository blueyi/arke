// Example 08: Batch MatMul (Category A — compute, attention QK^T)
//
// Natural Language:
//   "计算 GPT-2 多头注意力的 QK^T: Q:[8,128,64] × K_T:[8,64,128]"
//
// Semantics: out[b,i,j] = sum_k(Q[b,i,k] * K_T[b,k,j])
//   Note: K is provided already transposed as K_T: [batch, head_dim, seq]
//   This matches the standard attention pattern: QK^T where K^T is pre-transposed.
// Category: A (compute — batch matmul)

kernel batch_matmul(
    Q:   Tensor<[B, S, D], f16>,
    K_T: Tensor<[B, S, D], f16>
) -> Tensor<[B, S, S], f16> {
    let S = batch_matmul(A=Q, B=K_T);
    return S;
}

strategy batch_matmul_strategy for target("nvidia_ampere") {
    tile(loop="batch", factors=[1])
        @rationale("each block handles one (batch, M, N) tile — batch is outer loop");
    tile(loop="M", factors=[32])
        @rationale("M=128, tile=32: 4 tiles per batch, fits tensor core 16×8 MMA shape");
    tile(loop="N", factors=[32])
        @rationale("N=128, tile=32: symmetric with M tile");
    tile(loop="K", factors=[16])
        @rationale("K=64, tile=16: smem = 16*32*2*2 = 2048B well within budget");
    reorder(order=["batch", "M", "N", "K"])
        @rationale("batch outermost for block mapping, K innermost for reduction");
    parallel(loops=["batch", "M", "N"], mapping={"batch": "blockIdx.z", "M": "blockIdx.x", "N": "blockIdx.y"})
        @rationale("3D grid: z=batch, x=M-tile, y=N-tile");
    place(tensor="Q_tile", memory="shared")
        @rationale("Q tile in smem for K-reduction reuse");
    place(tensor="K_T_tile", memory="shared")
        @rationale("K_T tile in smem for K-reduction reuse");
    launch_config(num_warps=4, num_stages=2)
        @rationale("2 pipeline stages for Q/K_T prefetch during computation");
}
