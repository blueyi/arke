// Example 44: Cross Attention (Category B — attention, encoder-decoder)
//
// Natural Language:
//   "交叉注意力: Q:[2,12,128,64] from decoder, K/V:[2,12,256,64] from encoder → O:[2,12,128,64]"
//   "Decoder queries attend to encoder key-value pairs"
//
// Semantics: O = softmax(Q @ K^T / sqrt(D)) @ V  (Q from decoder, K/V from encoder)
// Category: B (attention — cross attention)

kernel cross_attention_kernel(
    Q: Tensor<[2, 12, 128, 64], f16>,
    K: Tensor<[2, 12, 256, 64], f16>,
    V: Tensor<[2, 12, 256, 64], f16>
) -> Tensor<[2, 12, 128, 64], f16> {
    let O = cross_attention(Q=Q, K=K, V=V);
    return O;
}

strategy cross_attention_kernel_strategy for target("nvidia_ampere") {
    tile(loop="Br", factors=[64])
        @rationale("Br=64: query tile — 64 decoder positions per block");
    tile(loop="Bc", factors=[64])
        @rationale("Bc=64: key/value tile — iterate over encoder KV blocks");
    place(tensor="Q_tile", memory="register")
        @rationale("Q stays in registers; K/V stream from smem");
    place(tensor="K_tile", memory="shared")
        @rationale("encoder K block in shared memory for dot product");
    place(tensor="V_tile", memory="shared")
        @rationale("encoder V block in smem, used after softmax");
    launch_config(num_warps=4, num_stages=2)
        @rationale("2-stage: prefetch next encoder KV block while computing current");
}
