// Example 36: Embedding (Category A — memory/layout, table lookup)
//
// Natural Language:
//   "嵌入查表：Token:[128] 从 Weight:[32000, 4096] 查表 → Y:[128, 4096]"
//
// Semantics: out[i,:] = Weight[Token[i], :]
// Category: A (memory — embedding lookup)

kernel embedding_kernel(
    Token: Tensor<[128], i64>,
    Weight: Tensor<[32000, 4096], f16>
) -> Tensor<[128, 4096], f16> {
    let Y = embedding(Token=Token, Weight=Weight);
    return Y;
}

strategy embedding_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 tokens per block — each row is an independent lookup");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: random read from embedding table, no reuse");
}
