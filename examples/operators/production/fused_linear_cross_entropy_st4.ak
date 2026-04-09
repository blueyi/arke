// Stage 7 Track 2 production-shape example: Fused Linear Cross Entropy ST4 family
// Source family: benchmark-shapes.md ST4 fused_linear_cross_entropy (LLaMA-3 / Qwen / DeepSeek vocab-scale CE)

kernel fused_linear_cross_entropy_st4(
    X: Tensor<[T, H], f16>,
    W: Tensor<[V, H], f16>,
    labels: Tensor<[T], i64>
) -> Tensor<[T], f32>
where
    T: dynamic(min=512, max=8192, multiple_of=128, default=512),
    H: static,
    V: static
{
    let loss = fused_linear_cross_entropy(X=X, W=W, labels=labels);
    return loss;
}

strategy fused_linear_cross_entropy_st4_strategy for target("nvidia_ampere") {
    tile(loop="token", factors=[128])
        @rationale("token tiles should align with production decode/train minibatch blocks");
    tile(loop="vocab_chunk", factors=[4096])
        @rationale("stream vocab chunks to avoid materializing full logits for ST4 vocab sizes");
    compute(num_warps=8, num_stages=2, shared_memory=65536)
        @rationale("streaming CE over large-vocab ST4 shapes needs explicit chunked compute resource planning");
}
