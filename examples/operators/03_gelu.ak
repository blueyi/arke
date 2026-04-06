// Example 03: GELU Activation (elementwise)
//
// Natural Language:
//   "对一个 1024×4096 的半精度矩阵做逐元素 GELU 激活"
//
// Semantics: out[i,j] = x[i,j] * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// Category: D (elementwise)

kernel gelu(
    X: Tensor<[1024, 4096], f16>
) -> Tensor<[1024, 4096], f16> {
    let Y = gelu(X=X);
    return Y;
}

strategy gelu_strategy for target("nvidia_ampere") {
    tile(loop="flat", factors=[4096])
        @rationale("1D tiling — treat as flat array of n_elements");
    vectorize(loop="flat", width=8)
        @rationale("vec8 for f16 — maximise memory bandwidth utilization");
    launch_config(num_warps=4, num_stages=1)
        @rationale("elementwise: memory-bound, 4 warps sufficient");
}
