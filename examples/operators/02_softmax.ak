// Example 02: Softmax
//
// Natural Language:
//   "对一个 2048×1024 的半精度矩阵沿最后一个轴做 softmax"
//
// Semantics: out[i,j] = exp(x[i,j]) / sum(exp(x[i,:]))
// Category: C (reduction)

kernel softmax(
    X: Tensor<[2048, 1024], f16>
) -> Tensor<[2048, 1024], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = softmax(X=X, axis=-1);
    return Y;
}

strategy softmax_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("one block per row — each row is independent");
    vectorize(loop="col", width=8)
        @rationale("vec8 for f16 — 128-bit load matches memory bus");
    place(tensor="X_row", memory="register")
        @rationale("entire row fits in registers for 1024-width");
    compute(warps=4, num_stages=1)
        @rationale("4 warps * 32 threads = 128 threads/block, good occupancy");
}
