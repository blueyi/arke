// Stage 7 L2 surface example: QKV projection + flash attention fusion

kernel qkv_fa_kernel(
    X: Tensor<[B, S, D], f16>,
    Wq: Tensor<[D, H], f16>,
    Wk: Tensor<[D, H], f16>,
    Wv: Tensor<[D, H], f16>
) -> _
where B: dynamic(max=64), S: dynamic(max=32768), D: static, H: static
{
    let Q = matmul(A=X, B=Wq);
    let K = matmul(A=X, B=Wk);
    let V = matmul(A=X, B=Wv);
    let O = flash_attention(Q=Q, K=K, V=V);
    return O;
}

strategy qkv_fa_kernel for target("nvidia_ampere") {
    fuse(ops=["matmul", "matmul", "matmul", "flash_attention"], fusion_type="producer_consumer")
        @rationale("Stage 7 L2 fusion slot: qkv_fa");
    compute(warps=4, num_stages=2, shared_memory=65536)
        @rationale("shared-memory-heavy staging keeps QKV tiles close to flash-attention consumption");
}
