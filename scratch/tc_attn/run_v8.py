"""Verify v8 generalized TC attention: D in {64,128} x {non-causal, causal}."""
import os, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "tc_attn_v8_general.cu")) as f:
    BASE = f.read()

def build(tag, HEAD_D, CAUSAL):
    fn = f"tc_v8_{tag}"
    src = BASE.replace("KERNEL_NAME", fn)
    smem = (64*HEAD_D*2) + 3*(2*32*HEAD_D*2)  # Qsh + 3*(K+V); =32K(D64)/64K(D128)
    cpp = f"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
void launch_{fn}(const half*,const half*,const half*,half*,int,int,int,float,cudaStream_t);
torch::Tensor {fn}(torch::Tensor Q,torch::Tensor K,torch::Tensor V,double scale){{
    int B=Q.size(0),H=Q.size(1),S=Q.size(2);
    auto O=torch::empty_like(Q);
    launch_{fn}(reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B,H,S,(float)scale,c10::cuda::getCurrentCUDAStream());
    return O;
}}
"""
    launcher = f"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ void {fn}(const half*,const half*,const half*,half*,int,int,int,float);
static int g=0;
void launch_{fn}(const half* Q,const half* K,const half* V,half* O,
        int B,int H,int S,float scale,cudaStream_t stream){{
    int smem={smem};
    if(smem!=g){{cudaFuncSetAttribute({fn},cudaFuncAttributeMaxDynamicSharedMemorySize,smem);g=smem;}}
    dim3 grid((S+63)/64,B*H);
    {fn}<<<grid,128,smem,stream>>>(Q,K,V,O,B,H,S,scale);
}}
"""
    mod = load_inline(name=fn, cpp_sources=[cpp], cuda_sources=[src+"\n"+launcher],
        functions=[fn],
        extra_cuda_cflags=["-O3","-arch=sm_86","--use_fast_math",
                           f"-DHEAD_D={HEAD_D}",f"-DCAUSAL={CAUSAL}"], verbose=False)
    return getattr(mod, fn)

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

configs = [
    ("d64",   64, 0, [(1,4,128),(1,8,512),(1,8,1024)]),
    ("d128", 128, 0, [(1,4,128),(1,8,512),(1,8,1024)]),
    ("d64c",  64, 1, [(1,4,128),(1,8,512)]),
    ("d128c",128, 1, [(1,4,128),(1,8,512)]),
]
print(f"# {torch.cuda.get_device_name(0)}\n")
allpass=True
for tag,HD,CA,shapes in configs:
    k=build(tag,HD,CA)
    for B,H,S in shapes:
        torch.manual_seed(0)
        Q=torch.randn(B,H,S,HD,device='cuda',dtype=torch.float16)
        K=torch.randn(B,H,S,HD,device='cuda',dtype=torch.float16)
        V=torch.randn(B,H,S,HD,device='cuda',dtype=torch.float16)
        sc=1.0/(HD**0.5)
        out=k(Q,K,V,sc)
        ref=F.scaled_dot_product_attention(Q,K,V,is_causal=bool(CA))
        me=(out.float()-ref.float()).abs().max().item()
        ok=me<1e-2; allpass &= ok
        extra=""
        if tag=="d64" and S==1024:
            tv=bench(lambda:k(Q,K,V,sc)); ts=bench(lambda:F.scaled_dot_product_attention(Q,K,V))
            extra=f"  speed={ts/tv:.3f}x"
        print(f"[D={HD} causal={CA}] {B}x{H}x{S}x{HD}  max_err={me:.3e}  {'PASS' if ok else 'FAIL'}{extra}")
print(f"\n{'ALL PASS' if allpass else 'SOME FAILED'}")
