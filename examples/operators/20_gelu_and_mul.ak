// Example 20: GELU-and-Mul (Category D — gated activation)
//
// Natural Language:
//   "GELU-and-Mul: split X into two halves, apply gelu to x1, multiply by x2; X:[128, 6144] → Y:[128, 3072]"
//   "Y = gelu(x1) * x2; x1 and x2 are two halves of the split input"
//
// Semantics: x1, x2 = split(X, dim=-1); Y = gelu(x1) * x2
// Category: D (activation — gated, GELU-and-Mul)

kernel gelu_and_mul_kernel(
    X: Tensor<[128, 6144], f16>
) -> Tensor<[128, 3072], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = gelu_and_mul(X=X);
    return Y;
}

strategy gelu_and_mul_kernel_strategy for target("nvidia_ampere") {
    fuse(ops=["gelu", "mul"], fusion_type="epilogue")
        @rationale("Stage 7 L2 gated activation fusion: keep split, GELU, and multiply in one epilogue-style kernel");
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise gated, maximize occupancy");
    tile(loop="col", factors=[256])
        @rationale("256-col tile: GELU more expensive than SiLU, smaller tile helps register");
    compute(warps=4, num_stages=1)
        @rationale("single-stage: compute-light, memory-bound");
}
