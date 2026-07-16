"""Head-to-head v4b vs v6 (BC=32) vs v7 (3-stage), stable timing, single process."""
import os, torch, statistics
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
m7 = build("tc_attn_v7", "tc_attn_v7_3stage.cu", "tc_flash_attn_v7_3stage", 32768)
print("OK", flush=True)

def bench(fn, iters=200, warmup=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
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
print(f"{'case':14} {'v4b':>8} {'v6':>8} {'v7':>8} {'v7/sdpa':>8} {'v7/v6':>7} {'v7/v4b':>7}")
for B,H,S in cases:
    torch.manual_seed(0)
    Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    K=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    V=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16)
    sc=1.0/(D**0.5)
    o4=m4.tc_attn_v4b(Q,K,V,sc); o6=m6.tc_attn_v6(Q,K,V,sc); o7=m7.tc_attn_v7(Q,K,V,sc)
    ref=torch.nn.functional.scaled_dot_product_attention(Q,K,V)
    e7=(o7.float()-ref.float()).abs().max().item()
    t4=bench(lambda:m4.tc_attn_v4b(Q,K,V,sc))
    t6=bench(lambda:m6.tc_attn_v6(Q,K,V,sc))
    t7=bench(lambda:m7.tc_attn_v7(Q,K,V,sc))
    ts=bench(lambda:torch.nn.functional.scaled_dot_product_attention(Q,K,V))
    print(f"{B}x{H}x{S:<6} {t4:8.4f} {t6:8.4f} {t7:8.4f} {ts/t7:8.3f} {t6/t7:7.3f} {t4/t7:7.3f}  err7={e7:.1e}")
