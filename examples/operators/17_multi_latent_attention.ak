// Example 17: Multi-Latent Attention (Category B — attention, DeepSeek MLA)
//
// Natural Language:
//   "MLA: Q:[2,16,128,64], KV_compressed:[2,128,32], 解压矩阵 W_uk/W_uv:[32,16,64]"
//   "先从低秩压缩向量解压 K 和 V，再做标准注意力"
//
// Semantics: K=KV_c @ W_uk, V=KV_c @ W_uv, O = softmax(Q @ K^T / sqrt(D)) @ V
// Category: B (attention — multi-latent attention, DeepSeek-V2/V3)

kernel mla_kernel(
    Q: Tensor<[2, 16, 128, 64], f16>,
    KV_compressed: Tensor<[2, 128, 32], f16>,
    W_uk: Tensor<[32, 16, 64], f16>,
    W_uv: Tensor<[32, 16, 64], f16>
) -> Tensor<[2, 16, 128, 64], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let O = multi_latent_attention(Q=Q, KV_compressed=KV_compressed, W_uk=W_uk, W_uv=W_uv);
    return O;
}

strategy mla_kernel_strategy for target("nvidia_ampere") {
    tile(loop="Br", factors=[64])
        @rationale("Br=64: query block for tiled attention");
    tile(loop="Bc", factors=[64])
        @rationale("Bc=64: KV block — decompress + attend in one pass");
    algorithm(name="fused_decompress_attend")
        @rationale("fuse KV decompression with attention to avoid materializing full K/V");
    place(tensor="KV_compressed_tile", memory="shared")
        @rationale("compressed KV in smem: D_c=32 << D*H, saves bandwidth");
    launch_config(num_warps=4, num_stages=2)
        @rationale("2-stage pipeline: prefetch next compressed KV block");
}
