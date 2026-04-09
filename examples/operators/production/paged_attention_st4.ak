// Stage 7 Track 2 production-shape example: Paged Attention ST4 family
// Source family: benchmark-shapes.md ST4 long-context attention / paged KV-cache serving

kernel paged_attention_st4(
    Q: Tensor<[B, H, 1, D], f16>,
    K_cache: Tensor<[P, BLOCK, H, D], f16>,
    V_cache: Tensor<[P, BLOCK, H, D], f16>,
    block_table: Tensor<[B, MAX_BLOCKS], i32>
) -> Tensor<[B, H, 1, D], f16>
where
    B: dynamic(min=1, max=16, default=1),
    H: static,
    D: static,
    P: dynamic(min=64, max=163840, multiple_of=16, default=1024),
    BLOCK: static,
    MAX_BLOCKS: dynamic(min=64, max=8192, multiple_of=16, default=512)
{
    let O = paged_attention(Q=Q, K_cache=K_cache, V_cache=V_cache, block_table=block_table);
    return O;
}

strategy paged_attention_st4_strategy for target("nvidia_ampere") {
    tile(loop="query", factors=[1])
        @rationale("decode-mode paged attention processes one query token at a time");
    tile(loop="page", factors=[16])
        @rationale("KV cache paging is naturally aligned to block-sized traversal");
    place(tensor="Q_tile", memory="register")
        @rationale("single-token query stays in registers");
    place(tensor="K_page", memory="shared")
        @rationale("paged KV blocks stream into shared memory through block table indirection");
    compute(warps=4, num_stages=2, shared_memory=65536)
        @rationale("paged attention needs explicit smem-aware execution planning for long-context serving");
}
