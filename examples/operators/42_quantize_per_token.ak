// Example 42: Quantize Per-Token (Category D — quantization)
//
// Natural Language:
//   "Per-token 动态量化：X:[128, 4096] f16 → Q:[128, 4096] i8"
//   "每行独立计算 scale，动态量化到 int8"
//
// Semantics: scale[i] = max(abs(X[i,:])) / 127; Q[i,j] = round(X[i,j] / scale[i])
// Category: D (quantization — per-token dynamic)

kernel quantize_per_token_kernel(
    X: Tensor<[128, 4096], f16>
) -> Tensor<[128, 4096], i8> 
where B: dynamic(max=64), S: dynamic(max=8192), D: static
{
    let Q = quantize_per_token(X=X);
    return Q;
}

strategy quantize_per_token_kernel_strategy for target("nvidia_ampere") {
    tile(loop="row", factors=[4])
        @rationale("4 rows per CTA — each row: reduce absmax + scale + quantize");
    launch_config(num_warps=4, num_stages=1)
        @rationale("single-stage: reduction + elementwise in one pass per row");
}
