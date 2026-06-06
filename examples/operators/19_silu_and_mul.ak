// Example 19: SiLU-and-Mul (Category D — gated activation)
//
// Natural Language:
//   "SiLU-and-Mul: split X into two halves, apply silu to x1, multiply by x2; X:[128, 6144] → Y:[128, 3072]"
//   "FFN gate payload = silu(x1) * x2; x1 and x2 are two halves of gate_proj output"
//
// Semantics: x1, x2 = split(X, dim=-1); Y = silu(x1) * x2
// Category: D (activation — gated, SiLU-and-Mul)

kernel silu_and_mul_kernel(
    X: Tensor<[128, 6144], f16>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = silu_and_mul(X=X);
    return Y;
}

strategy silu_and_mul_kernel_strategy for target("nvidia_ampere") {
    fuse(ops=["silu", "mul"], fusion_type="epilogue")
        @rationale("Stage 7 L2 gated activation fusion: keep split, SiLU, and multiply in one epilogue-style kernel");
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise gated op, maximize occupancy");
    tile(loop="col", factors=[256])
        @rationale("256-col tile: covers gate pairs together; half goes to silu, half to gate");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: memory-bound, no compute to pipeline");
}
