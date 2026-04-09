// Example 45: Paged Attention (Category B — attention, vLLM-style paged KV cache)
//
// Natural Language:
//   "分页注意力: Q:[1, 12, 1, 64] + KV cache in paged blocks → O:[1, 12, 1, 64]"
//   "vLLM 风格：KV cache 以 block_size=16 分页存储，通过 block_table 索引"
//
// Semantics: O = paged_softmax(Q @ K_pages^T / sqrt(D)) @ V_pages
//            K_pages, V_pages gathered via block_table indirection
// Category: B (attention — paged KV cache)

kernel paged_attention_kernel(
    Q: Tensor<[1, 12, 1, 64], f16>,
    K_cache: Tensor<[64, 16, 12, 64], f16>,
    V_cache: Tensor<[64, 16, 12, 64], f16>,
    block_table: Tensor<[1, 64], i32>
) -> Tensor<[1, 12, 1, 64], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let O = paged_attention(Q=Q, K_cache=K_cache, V_cache=V_cache, block_table=block_table);
    return O;
}

strategy paged_attention_kernel_strategy for target("nvidia_ampere") {
    tile(loop="seq", factors=[1])
        @rationale("1 query per CTA — decode phase processes one token at a time");
    tile(loop="kv_block", factors=[16])
        @rationale("16 KV entries per page — iterate over block_table entries");
    place(tensor="Q_tile", memory="register")
        @rationale("single query vector stays in registers");
    place(tensor="K_page", memory="shared")
        @rationale("load KV page to shared memory via block_table indirection");
    compute(warps=4, num_stages=2)
        @rationale("2-stage: prefetch next KV page while computing current attention");
}
