"""Verify the TC emitter end-to-end through the real CudaCBackend path."""
import numpy as np, torch
import torch.nn.functional as F
from arke.backend.cuda_c_backend import CudaCBackend, cuda_c_toolchain_available
from arke.ir.graph import IRGraph, IRNode

assert cuda_c_toolchain_available(), "toolchain unavailable"
be = CudaCBackend(chip="sm_86")

def run_case(B,H,S,D,causal):
    g = IRGraph(name="fa")
    g.add_input("Q", dtype="float16", shape=[B,H,S,D])
    g.add_input("K", dtype="float16", shape=[B,H,S,D])
    g.add_input("V", dtype="float16", shape=[B,H,S,D])
    g.add_node(IRNode(id="n0", op="flash_attention",
                      inputs={"Q":"Q","K":"K","V":"V"}, outputs=["out"],
                      attrs={"causal": causal}))
    g.set_outputs(["out"])
    art = be.lower(g)
    ker = be.compile(art)
    assert ker.success, ker.error
    emitted = ker.metadata["emitted"]
    assert emitted.kernel_name.startswith("arke_tc_flash_attn"), emitted.kernel_name
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((B,H,S,D)).astype(np.float16)
    K = rng.standard_normal((B,H,S,D)).astype(np.float16)
    V = rng.standard_normal((B,H,S,D)).astype(np.float16)
    out = be.run(ker, {"Q":Q,"K":K,"V":V})["out"]
    ref = F.scaled_dot_product_attention(
        torch.tensor(Q).cuda(), torch.tensor(K).cuda(), torch.tensor(V).cuda(),
        is_causal=causal).cpu().numpy()
    me = np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32)))
    ok = me < 1e-2
    print(f"[D={D} causal={causal}] {B}x{H}x{S}x{D} smem={emitted.shared_mem} max_err={me:.3e} {'PASS' if ok else 'FAIL'}")
    return ok

allpass = True
for D in (64, 128):
    for causal in (False, True):
        for S in (128, 512):
            allpass &= run_case(1, 4, S, D, causal)
# fp32 should NOT route to TC (fallback path)
g = IRGraph(name="fa32")
for n in ("Q","K","V"): g.add_input(n, dtype="float32", shape=[1,4,128,64])
g.add_node(IRNode(id="n0", op="flash_attention", inputs={"Q":"Q","K":"K","V":"V"}, outputs=["out"]))
g.set_outputs(["out"])
em = be.lower(g).metadata["emitted"]
fallback_ok = em.kernel_name.startswith("arke_flash_attn") and not em.kernel_name.startswith("arke_tc")
print(f"[fp32 fallback] kernel={em.kernel_name} {'PASS' if fallback_ok else 'FAIL'}")
allpass &= fallback_ok
print("\n" + ("ALL PASS" if allpass else "SOME FAILED"))
