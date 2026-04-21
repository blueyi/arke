// Stage 7 L2 surface example: streamed linear + cross entropy fusion

kernel linear_ce_kernel(
    Hidden: Tensor<[B, D], f16>,
    Weight: Tensor<[V, D], f16>,
    Labels: Tensor<[B], i64>
) -> Tensor<[B], f32>
where B: dynamic(max=4096), D: static, V: dynamic(max=262144)
{
    let Loss = fused_linear_cross_entropy(Hidden=Hidden, Weight=Weight, Labels=Labels);
    return Loss;
}

strategy linear_ce_kernel for target("nvidia_ampere") {
    tile(loop="row", factors=[1])
        @rationale("one sequence row per CTA keeps softmax statistics local");
    tile(loop="col", factors=[4096])
        @rationale("stream vocab tiles instead of materializing full logits");
    fuse(ops=["matmul", "cross_entropy"], fusion_type="producer_consumer")
        @rationale("Stage 7 L2 fusion slot: linear_ce");
    compute(warps=8, num_stages=2)
        @rationale("overlap weight-tile fetch with streamed loss accumulation");
}
