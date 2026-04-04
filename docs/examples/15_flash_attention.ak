// Example 15: Flash Attention (Category B — attention, standard MHA)
//
// Natural Language:
//   "多头注意力: Q,K,V:[2,12,128,64] → O:[2,12,128,64]，FlashAttention tiled online softmax"
//
// Semantics: O = softmax(Q @ K^T / sqrt(D)) @ V  (tiled, online softmax)
// Category: B (attention — flash attention)

kernel flash_attention_kernel(
    Q: Tensor<[2, 12, 128, 64], f16>,
    K: Tensor<[2, 12, 128, 64], f16>,
    V: Tensor<[2, 12, 128, 64], f16>
) -> Tensor<[2, 12, 128, 64], f16> {
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy flash_attention_kernel_strategy for target("nvidia_ampere") {
    tile(loop="Br", factors=[64])
        @rationale("Br=64: query tile — 64 rows fit in register file");
    tile(loop="Bc", factors=[64])
        @rationale("Bc=64: key/value tile — online softmax over KV blocks");
    place(tensor="Q_tile", memory="register")
        @rationale("Q stays in registers; K/V stream from smem");
    place(tensor="K_tile", memory="shared")
        @rationale("K block loaded to shared memory for softmax compute");
    place(tensor="V_tile", memory="shared")
        @rationale("V block in smem, used after softmax to accumulate output");
    launch_config(num_warps=4, num_stages=2)
        @rationale("2-stage pipeline: prefetch next KV block while computing current");
}
