// Example 43: Dequantize Per-Channel (Category D — dequantization)
//
// Natural Language:
//   "Per-channel 反量化：Q:[128, 4096] i8 + Scale:[4096] f16 → Y:[128, 4096] f16"
//   "每列使用独立 scale 反量化回 f16"
//
// Semantics: Y[i,j] = Q[i,j] * Scale[j]
// Category: D (dequantization — per-channel static)

kernel dequantize_per_channel_kernel(
    Q: Tensor<[128, 4096], i8>,
    Scale: Tensor<[4096], f16>
) -> Tensor<[128, 4096], f16> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Y = dequantize_per_channel(Q=Q, Scale=Scale);
    return Y;
}

strategy dequantize_per_channel_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per block — elementwise scale multiply, maximize occupancy");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: memory-bound dequant, scale broadcast along rows");
}
