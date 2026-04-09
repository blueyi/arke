// Stage 7 Track 2 production-shape example: Flash Attention ST4 family
// Source family: benchmark-shapes.md ST4 attention (e.g. llama3-attn-8k / ds-v2-attn-16k)

kernel flash_attention_st4(
    Q: Tensor<[B, H, S, D], f16>,
    K: Tensor<[B, H, S, D], f16>,
    V: Tensor<[B, H, S, D], f16>
) -> Tensor<[B, H, S, D], f16>
where
    B: dynamic(min=1, max=4, default=1),
    H: static,
    S: dynamic(min=512, max=32768, multiple_of=128, default=8192),
    D: static
{
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy flash_attention_st4_strategy for target("nvidia_ampere") {
    tile(loop="Br", factors=[128])
        @rationale("ST4 long-context query tiles should align with 128-row online-softmax blocks");
    tile(loop="Bc", factors=[128])
        @rationale("ST4 KV traversal should stream in 128-row blocks for long-context attention");
    place(tensor="Q_tile", memory="register")
        @rationale("query tile remains register-resident");
    place(tensor="K_tile", memory="shared")
        @rationale("KV tiles stream through shared memory");
    place(tensor="V_tile", memory="shared")
        @rationale("value tiles follow K tiles for fused online accumulation");
    compute(num_warps=8, num_stages=2, shared_memory=65536)
        @rationale("long-context ST4 attention needs larger tiles and smem budget than ST2 examples");
}
