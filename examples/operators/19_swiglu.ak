// Example 19: SwiGLU (Category D — activation, LLaMA/DeepSeek FFN gate)
//
// Natural Language:
//   "SwiGLU: split X 后半做 gate，X:[128, 6144] → Y:[128, 3072]"
//   "FFN 输出 = silu(x1) * x2，x1 和 x2 是 gate_proj 输出的两半"
//
// Semantics: x1, x2 = split(X, dim=-1); Y = silu(x1) * x2
// Category: D (activation — gated, SwiGLU)

kernel swiglu_kernel(
    X: Tensor<[128, 6144], f16>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = swiglu(X=X);
    return Y;
}

strategy swiglu_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise gated op, maximize occupancy");
    tile(loop="col", factors=[256])
        @rationale("256-col tile: covers gate pairs together; half goes to silu, half to gate");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: memory-bound, no compute to pipeline");
}
