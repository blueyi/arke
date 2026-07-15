"""Correctness + perf harness for TC flash-attention v2 (low-smem variant).

Compiles tc_attn_v2.cu via torch cpp_extension, compares against SDPA,
and reports kernel-only timing with CUDA events.
"""
import json
import os
import sys
import torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
CU_V2 = os.path.join(HERE, "tc_attn_v2.cu")

D = 64
BR = 64
BC = 64

with open(CU_V2) as f:
    kernel_src = f.read()

cpp_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_tc_flash_attn_v2(const half*, const half*, const half*, half*,
                             int, int, int, float, cudaStream_t);

torch::Tensor tc_attn_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf, "Q must be cuda fp16");
    int B = Q.size(0), H = Q.size(1), S = Q.size(2), Dd = Q.size(3);
    TORCH_CHECK(Dd == 64, "prototype fixed D=64");
    auto O = torch::empty_like(Q);
    launch_tc_flash_attn_v2(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, S, (float)scale, c10::cuda::getCurrentCUDAStream());
    return O;
}
"""

launcher = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ void tc_flash_attn_v2(const half*, const half*, const half*,
                                            half*, int, int, int, float);
static int g_smem_v2 = 0;
void launch_tc_flash_attn_v2(const half* Q, const half* K, const half* V, half* O,
                             int B, int H, int S, float scale, cudaStream_t stream) {
    const int br=64, bc=64, dd=64;
    // Smem: Qsh(8K) + KSsh(16K) + Psh(8K) + Osh(16K) + m/l(512B) = 49664
    int smem = 8192 + 16384 + 8192 + 16384 + 256 + 256;  // = 49664
    if (smem != g_smem_v2) {
        cudaFuncSetAttribute(tc_flash_attn_v2,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem_v2 = smem;
    }
    dim3 grid((S + br - 1) / br, B * H, 1);
    dim3 block(128, 1, 1);
    tc_flash_attn_v2<<<grid, block, smem, stream>>>(Q, K, V, O, B, H, S, scale);
}
"""

full_cuda = kernel_src + "\n" + launcher

print("Compiling tc_attn_v2...", flush=True)
mod = load_inline(
    name="tc_attn_v2_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[full_cuda],
    functions=["tc_attn_v2"],
    extra_cuda_cflags=["-O3", "-arch=sm_86", "--use_fast_math"],
    verbose=False,
)
print("Compilation OK.", flush=True)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(B, H, S):
    torch.manual_seed(0)
    dev = "cuda"
    Q = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
    K = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
    V = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    out = mod.tc_attn_v2(Q, K, V, scale)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    t_v2 = bench(lambda: mod.tc_attn_v2(Q, K, V, scale))
    t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V))

    return {
        "B": B, "H": H, "S": S, "D": D,
        "max_err": max_err, "mean_err": mean_err,
        "v2_ms": t_v2, "sdpa_ms": t_sdpa,
        "speedup_vs_sdpa": t_sdpa / t_v2,
    }


if __name__ == "__main__":
    print(f"# GPU: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print(f"# torch {torch.__version__}")
    print(f"# smem per block: 49664 B = {49664/1024:.1f} KB")
    print(f"# Expected occupancy: 2 blocks/SM (49664 * 2 = 99328 < 100KB)")
    print()

    cases = [(1, 1, 128), (1, 4, 128), (1, 8, 512), (1, 8, 1024), (1, 8, 2048), (4, 8, 2048)]
    results = []
    for (B, H, S) in cases:
        r = run_case(B, H, S)
        results.append(r)
        status = "✓" if r["max_err"] < 1e-2 else "✗"
        print(f"[{B}x{H}x{S}x{D}] {status} max_err={r['max_err']:.4e} mean_err={r['mean_err']:.2e} "
              f"v2={r['v2_ms']:.4f}ms sdpa={r['sdpa_ms']:.4f}ms speedup={r['speedup_vs_sdpa']:.3f}x")

    with open(os.path.join(HERE, "results_v2.json"), "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "version": "v2-low-smem", "results": results}, f, indent=2)
    print("\nResults saved to results_v2.json")
