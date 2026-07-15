"""Harness for v4b_opt_softmax (pre-scaled softmax, eliminates redundant scale multiplications)."""
import json, os, torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
D = 64

with open(os.path.join(HERE, "tc_attn_v4b_opt_softmax.cu")) as f:
    kernel_src = f.read()

# Also load v4b for comparison
with open(os.path.join(HERE, "tc_attn_v4b.cu")) as f:
    kernel_v4b_src = f.read()

cpp_src_opt = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_v4b_opt(const half*, const half*, const half*, half*, int, int, int, float, cudaStream_t);
torch::Tensor tc_attn_v4b_opt_softmax(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf);
    int B=Q.size(0), H=Q.size(1), S=Q.size(2);
    auto O = torch::empty_like(Q);
    launch_v4b_opt(reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
               reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
               reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
               reinterpret_cast<half*>(O.data_ptr<at::Half>()),
               B, H, S, (float)scale, c10::cuda::getCurrentCUDAStream());
    return O;
}
"""

launcher_opt = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ void tc_flash_attn_v4b_opt_softmax(const half*, const half*, const half*, half*, int, int, int, float);
static int g_smem_opt = 0;
void launch_v4b_opt(const half* Q, const half* K, const half* V, half* O,
                int B, int H, int S, float scale, cudaStream_t stream) {
    int smem = 40960;  // 40 KB
    if (smem != g_smem_opt) {
        cudaFuncSetAttribute(tc_flash_attn_v4b_opt_softmax, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem_opt = smem;
    }
    dim3 grid((S + 63) / 64, B * H);
    tc_flash_attn_v4b_opt_softmax<<<grid, 128, smem, stream>>>(Q, K, V, O, B, H, S, scale);
}
"""

cpp_src_v4b = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_v4b(const half*, const half*, const half*, half*, int, int, int, float, cudaStream_t);
torch::Tensor tc_attn_v4b(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf);
    int B=Q.size(0), H=Q.size(1), S=Q.size(2);
    auto O = torch::empty_like(Q);
    launch_v4b(reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
               reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
               reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
               reinterpret_cast<half*>(O.data_ptr<at::Half>()),
               B, H, S, (float)scale, c10::cuda::getCurrentCUDAStream());
    return O;
}
"""

launcher_v4b = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ void tc_flash_attn_v4b(const half*, const half*, const half*, half*, int, int, int, float);
static int g_smem = 0;
void launch_v4b(const half* Q, const half* K, const half* V, half* O,
                int B, int H, int S, float scale, cudaStream_t stream) {
    int smem = 40960;  // 40 KB
    if (smem != g_smem) {
        cudaFuncSetAttribute(tc_flash_attn_v4b, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem = smem;
    }
    dim3 grid((S + 63) / 64, B * H);
    tc_flash_attn_v4b<<<grid, 128, smem, stream>>>(Q, K, V, O, B, H, S, scale);
}
"""

print("Compiling v4b_opt_softmax (pre-scaled softmax)...", flush=True)
mod_opt = load_inline(name="tc_v4b_opt", cpp_sources=[cpp_src_opt], cuda_sources=[kernel_src+"\n"+launcher_opt],
                  functions=["tc_attn_v4b_opt_softmax"], extra_cuda_cflags=["-O3","-arch=sm_86","--use_fast_math"], verbose=False)
print("OK.", flush=True)

print("Compiling v4b (baseline)...", flush=True)
mod_v4b = load_inline(name="tc_v4b_base", cpp_sources=[cpp_src_v4b], cuda_sources=[kernel_v4b_src+"\n"+launcher_v4b],
                  functions=["tc_attn_v4b"], extra_cuda_cflags=["-O3","-arch=sm_86","--use_fast_math"], verbose=False)
print("OK.", flush=True)

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

cases = [(1,1,128),(1,4,128),(1,8,512),(1,8,1024),(1,8,2048),(4,8,2048)]
print(f"# GPU: {torch.cuda.get_device_name(0)}")
print(f"# Optimization: pre-scale s_frag ONCE, eliminate 32 redundant fmul per tile")
print(f"# smem: 40 KB")
print()
print(f"{'Case':<20} {'Correct':>8} {'max_err':>10} {'v4b_opt':>9} {'v4b':>9} {'sdpa':>9} {'opt/v4b':>8} {'opt/sdpa':>9}")
print("-" * 90)

for B,H,S in cases:
    torch.manual_seed(0)
    Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    K=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    V=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    scale=1.0/(D**0.5)
    
    out_opt = mod_opt.tc_attn_v4b_opt_softmax(Q,K,V,scale)
    out_v4b = mod_v4b.tc_attn_v4b(Q,K,V,scale)
    ref = torch.nn.functional.scaled_dot_product_attention(Q,K,V)
    
    diff_opt = (out_opt.float()-ref.float()).abs()
    me_opt = diff_opt.max().item()
    
    # Also check opt vs v4b match
    diff_v4b = (out_opt.float()-out_v4b.float()).abs()
    me_v4b = diff_v4b.max().item()
    
    t_opt = bench(lambda:mod_opt.tc_attn_v4b_opt_softmax(Q,K,V,scale))
    t_v4b = bench(lambda:mod_v4b.tc_attn_v4b(Q,K,V,scale))
    t_sdpa = bench(lambda:torch.nn.functional.scaled_dot_product_attention(Q,K,V))
    
    st = "✓" if me_opt < 1e-2 else "✗"
    ratio_v4b = t_opt / t_v4b
    ratio_sdpa = t_sdpa / t_opt
    
    print(f"[{B}x{H}x{S}x{D}] {st:>8} {me_opt:>10.4e} {t_opt:>8.4f}ms {t_v4b:>8.4f}ms {t_sdpa:>8.4f}ms {ratio_v4b:>7.3f}x {ratio_sdpa:>8.3f}x")
    if me_v4b > 1e-4:
        print(f"  ⚠ opt vs v4b diff: {me_v4b:.4e}")

print()
print("# ratio opt/v4b < 1.0 means opt is faster; > 1.0 means opt is slower")
print("# ratio opt/sdpa > 1.0 means opt is faster than PyTorch SDPA")
