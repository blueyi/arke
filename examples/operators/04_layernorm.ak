// Example 04: LayerNorm
//
// Natural Language:
//   "对 GPT-2 hidden state (batch=128, hidden=768) 做 LayerNorm"
//
// Semantics: out[i,j] = (x[i,j] - mean(x[i,:])) / sqrt(var(x[i,:]) + eps) * w[j] + b[j]
// Category: C (reduction + elementwise)

kernel layernorm(
    X:      Tensor<[128, 768], f16>,
    weight: Tensor<[768], f16>,
    bias:   Tensor<[768], f16>
) -> Tensor<[128, 768], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = layernorm(X=X, weight=weight, bias=bias, eps=0.000010);
    return Y;
}

strategy layernorm_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("one block per row — mean/var reduction is row-local");
    vectorize(loop="col", width=8)
        @rationale("vec8 for f16 to saturate load bandwidth");
    place(tensor="X_row", memory="shared")
        @rationale("cache row in shared mem to avoid 2x global reads (forward + variance pass)");
    compute(warps=4, num_stages=1)
        @rationale("768 elements / (128 threads * 8) = ~0.75 iterations/thread — good fit");
}
