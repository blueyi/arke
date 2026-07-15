"""Harness for v3b (K+V coloaded, separate buffers, 32KB smem)."""
import json, os, torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
D = 64

with open(os.path.join(HERE, "tc_attn_v3b.cu")) as f:
    kernel_src = f.read()

cpp_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_v3b(const half*, const half*, const half*, half*, int, int, int, float, cudaStream_t);
torch::Tensor tc_attn_v3b(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kHalf);
    int B=Q.size(0), H=Q.size(1), S=Q.size(2);
    auto O = torch::empty_like(Q);
    launch_v3b(reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
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
extern "C" __global__ void tc_flash_attn_v3b(const half*, const half*, const half*, half*, int, int, int, float);
static int g_smem = 0;
void launch_v3b(const half* Q, const half* K, const half* V, half* O,
                int B, int H, int S, float scale, cudaStream_t stream) {
    int smem = 32768;  // 32 KB
    if (smem != g_smem) {
        cudaFuncSetAttribute(tc_flash_attn_v3b, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_smem = smem;
    }
    dim3 grid((S + 63) / 64, B * H);
    tc_flash_attn_v3b<<<grid, 128, smem, stream>>>(Q, K, V, O, B, H, S, scale);
}
"""

print("Compiling v3b...", flush=True)
mod = load_inline(name="tc_v3b", cpp_sources=[cpp_src], cuda_sources=[kernel_src+"\n"+launcher],
                  functions=["tc_attn_v3b"], extra_cuda_cflags=["-O3","-arch=sm_86","--use_fast_math"], verbose=False)
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
print(f"# smem: 32 KB → 3 blocks/SM (32*3=96K < 100K)")
print()

results = []
for B,H,S in cases:
    torch.manual_seed(0)
    Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    K=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    V=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    scale=1.0/(D**0.5)
    out=mod.tc_attn_v3b(Q,K,V,scale)
    ref=torch.nn.functional.scaled_dot_product_attention(Q,K,V)
    diff=(out.float()-ref.float()).abs()
    me=diff.max().item()
    tv=bench(lambda:mod.tc_attn_v3b(Q,K,V,scale))
    ts=bench(lambda:torch.nn.functional.scaled_dot_product_attention(Q,K,V))
    st="✓" if me<1e-2 else "✗"
    print(f"[{B}x{H}x{S}x{D}] {st} max_err={me:.4e} v3b={tv:.4f}ms sdpa={ts:.4f}ms speed={ts/tv:.3f}x")
    results.append({"B":B,"H":H,"S":S,"D":D,"max_err":me,"v3b_ms":tv,"sdpa_ms":ts,"speedup":ts/tv})

with open(os.path.join(HERE,"results_v3b.json"),"w") as f:
    json.dump({"version":"v3b","results":results},f,indent=2)
