// Example 04: Scaled Dot-Product Attention
// Demonstrates complex multi-operator kernel with FlashAttention-style schedule

kernel attention(
    Q: Tensor<[1, 32, 1024, 64], f16>,
    K: Tensor<[1, 32, 1024, 64], f16>,
    V: Tensor<[1, 32, 1024, 64], f16>
) -> Tensor<[1, 32, 1024, 64], f16> {
    let K_t = transpose(K, dims=[2, 3]);
    let scores = batch_matmul(Q, K_t);
    let scale = 0.125;  // 1/sqrt(64)
    let scaled = mul(scores, scale);
    let weights = softmax(scaled, axis=-1);
    let O = batch_matmul(weights, V);
    return O;
}

schedule attention for target("nvidia_ampere") {
    // FlashAttention-style tiling
    tile(loop="seq_q", factors=[128])
        @rationale("query block size = 128, fits in shared memory");

    tile(loop="seq_k", factors=[64])
        @rationale("key block size = 64, online softmax over K blocks");

    parallel(loops=["batch", "head"],
             mapping={"batch": "blockIdx.z", "head": "blockIdx.y"})
        @rationale("batch and head dimensions are independent");

    place(tensor="Q_block", memory=shared)
        @rationale("query block stays in shared memory for K-block iteration");

    place(tensor="K_block", memory=shared)
        @rationale("key block loaded per iteration");

    place(tensor="V_block", memory=shared)
        @rationale("value block loaded per iteration");

    fuse(ops=["batch_matmul", "mul", "softmax", "batch_matmul"], type=vertical)
        @rationale("FlashAttention: fuse QK^T, scale, softmax, V multiply to avoid materializing attention matrix");
}
