// Example 16: Grouped Query Attention (Category B — attention, GQA)
//
// Natural Language:
//   "GQA: Q:[2,32,128,64] with K,V:[2,8,128,64] — 4 query heads per KV head"
//
// Semantics: repeat KV heads, then O = softmax(Q @ K^T / sqrt(D)) @ V
// Category: B (attention — grouped query attention)

kernel gqa_kernel(
    Q: Tensor<[2, 32, 128, 64], f16>,
    K: Tensor<[2, 8, 128, 64], f16>,
    V: Tensor<[2, 8, 128, 64], f16>
) -> Tensor<[2, 32, 128, 64], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let O = grouped_query_attention(Q=Q, K=K, V=V);
    return O;
}

strategy gqa_kernel_strategy for target("nvidia_ampere") {
    tile(loop="Br", factors=[64])
        @rationale("Br=64: query block rows");
    tile(loop="Bc", factors=[64])
        @rationale("Bc=64: key/value block columns for online softmax");
    algorithm(name="kv_head_repeat_implicit")
        @rationale("implicit repeat: 4 Q heads index same KV head, no copy");
    compute(warps=4, num_stages=2)
        @rationale("2-stage pipeline for KV prefetch");
}
