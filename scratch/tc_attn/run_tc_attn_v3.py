"""Correctness + perf harness for TC flash-attention v3 (register-resident O+S)."""
import json
import os
import torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
CU_V3 = os.path.join(HERE, "tc_attn_v3.cu")

D = 64

with open(CU_V3) as f:
    kernel_src = f.read()

cpp_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_tc_flash_attn_v3(const half*, const half*, const half*, half*,
                             int, int, int, float, cudaStream_t);

torch::Tensor tc_attn_v3(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf, "Q must be cuda fp16");
    int B = Q.size(0), H = Q.size(1), S = Q.size(2), Dd = Q.size(3);
    TORCH_CHECK(Dd == 64, "prototype fixed D=64");
    auto O = torch::empty_like(Q);
    launch_tc_flash_attn_v3(
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
extern "C" __global__ void tc_flash_attn_v3(const half*, const half*, const half*,
                                            half*, int, int, int, float);
static int g_smem_v3 = 0;
void launch_tc_flash_attn_v3(const half* Q, const half* K, const half* V, half* O,
                             int B, int H, int S, float scale, cudaStream_t stream) {
    // Smem: Qsh(8K) + KVsh(8K) + Psh(8K) = 24576
    int smem = 24576;
    if (smem != g_smem_v3) {
        cudaFuncSetAttribute(tc_flash_attn_v3,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem_v3 = smem;
    }
    dim3 grid((S + 63) / 64, B * H, 1);
    dim3 block(128, 1, 1);
    tc_flash_attn_v3<<<grid, block, smem, stream>>>(Q, K, V, O, B, H, S, scale);
}
"""

full_cuda = kernel_src + "\n" + launcher

print("Compiling tc_attn_v3 (register-resident)...", flush=True)
mod = load_inline(
    name="tc_attn_v3_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[full_cuda],
    functions=["tc_attn_v3"],
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

    out = mod.tc_attn_v3(Q, K, V, scale)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    t_v3 = bench(lambda: mod.tc_attn_v3(Q, K, V, scale))
    t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V))

    return {
        "B": B, "H": H, "S": S, "D": D,
        "max_err": max_err, "mean_err": mean_err,
        "v3_ms": t_v3, "sdpa_ms": t_sdpa,
        "speedup_vs_sdpa": t_sdpa / t_v3,
    }


if __name__ == "__main__":
    print(f"# GPU: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print(f"# torch {torch.__version__}")
    print(f"# smem per block: 24576 B = 24.0 KB")
    print(f"# Expected occupancy: 4 blocks/SM (24576 * 4 = 98304 < 102400)")
    print()

    cases = [(1, 1, 128), (1, 4, 128), (1, 8, 512), (1, 8, 1024), (1, 8, 2048), (4, 8, 2048)]
    results = []
    for (B, H, S) in cases:
        r = run_case(B, H, S)
        results.append(r)
        status = "✓" if r["max_err"] < 1e-2 else "✗"
        print(f"[{B}x{H}x{S}x{D}] {status} max_err={r['max_err']:.4e} mean_err={r['mean_err']:.2e} "
              f"v3={r['v3_ms']:.4f}ms sdpa={r['sdpa_ms']:.4f}ms speedup={r['speedup_vs_sdpa']:.3f}x")

    with open(os.path.join(HERE, "results_v3.json"), "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "version": "v3-reg-resident", "results": results}, f, indent=2)
    print("\nResults saved to results_v3.json")
