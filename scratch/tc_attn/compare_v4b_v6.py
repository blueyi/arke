"""Head-to-head v4b vs v6 (BC=32), stable timing, single process."""
import os, torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
D = 64

def build(name, cu, fn, smem):
    with open(os.path.join(HERE, cu)) as f:
        src = f.read()
    cpp = f"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_{name}(const half*, const half*, const half*, half*, int, int, int, float, cudaStream_t);
torch::Tensor {name}(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {{
    int B=Q.size(0), H=Q.size(1), S=Q.size(2);
    auto O = torch::empty_like(Q);
    launch_{name}(reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
              reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
              reinterpret_cast<half*>(O.data_ptr<at::Half>()),
              B, H, S, (float)scale, c10::cuda::getCurrentCUDAStream());
    return O;
}}
"""
    launcher = f"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ void {fn}(const half*, const half*, const half*, half*, int, int, int, float);
static int g_smem = 0;
void launch_{name}(const half* Q, const half* K, const half* V, half* O,
               int B, int H, int S, float scale, cudaStream_t stream) {{
    int smem = {smem};
    if (smem != g_smem) {{ cudaFuncSetAttribute({fn}, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); g_smem = smem; }}
    dim3 grid((S + 63) / 64, B * H);
    {fn}<<<grid, 128, smem, stream>>>(Q, K, V, O, B, H, S, scale);
}}
"""
    return load_inline(name=name, cpp_sources=[cpp], cuda_sources=[src+"\n"+launcher],
                       functions=[name], extra_cuda_cflags=["-O3","-arch=sm_86","--use_fast_math"], verbose=False)

print("compiling...", flush=True)
m4 = build("tc_attn_v4b", "tc_attn_v4b.cu", "tc_flash_attn_v4b", 40960)
m6 = build("tc_attn_v6", "tc_attn_v6_bc32.cu", "tc_flash_attn_v6", 24576)
print("OK", flush=True)

def bench(fn, iters=200, warmup=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    import statistics
    samples=[]
    for _ in range(5):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        samples.append(s.elapsed_time(e)/iters)
    return statistics.median(samples)

cases=[(1,8,512),(1,8,1024),(1,8,2048),(4,8,2048)]
print(f"# {torch.cuda.get_device_name(0)}  (median of 5x200 iters)\n")
print(f"{'case':16} {'v4b ms':>9} {'v6 ms':>9} {'sdpa ms':>9} {'v4b/sdpa':>9} {'v6/sdpa':>9} {'v6/v4b':>8}")
for B,H,S in cases:
    torch.manual_seed(0)
    Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    K=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    V=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    sc=1.0/(D**0.5)
    o4=m4.tc_attn_v4b(Q,K,V,sc); o6=m6.tc_attn_v6(Q,K,V,sc)
    ref=torch.nn.functional.scaled_dot_product_attention(Q,K,V)
    e4=(o4.float()-ref.float()).abs().max().item()
    e6=(o6.float()-ref.float()).abs().max().item()
    t4=bench(lambda:m4.tc_attn_v4b(Q,K,V,sc))
    t6=bench(lambda:m6.tc_attn_v6(Q,K,V,sc))
    ts=bench(lambda:torch.nn.functional.scaled_dot_product_attention(Q,K,V))
    print(f"{B}x{H}x{S}x{D:<6} {t4:9.4f} {t6:9.4f} {ts:9.4f} {ts/t4:9.3f} {ts/t6:9.3f} {t4/t6:8.3f}  err4={e4:.1e} err6={e6:.1e}")
