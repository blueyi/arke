# Scratch harness: correctness + kernel-only benchmark for flash_attention CUDA-C.
import sys, math
import numpy as np
import torch
import torch.nn.functional as F

from arke.ir.graph import IRGraph, IRNode
from arke.backend.cuda_c_backend import CudaCBackend
from arke.backend import cuda_c_attention as attn


def make_graph(B, H, S, D, dtype="float32"):
    g = IRGraph(name="fa")
    g.add_input("Q", dtype=dtype, shape=[B, H, S, D])
    g.add_input("K", dtype=dtype, shape=[B, H, S, D])
    g.add_input("V", dtype=dtype, shape=[B, H, S, D])
    g.add_node(IRNode(id="n0", op="flash_attention",
                      inputs={"Q": "Q", "K": "K", "V": "V"}, outputs=["O"]))
    g.set_outputs(["O"])
    return g


def torch_ref(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def run_case(emitter_name, B, H, S, D, iters=50, warmup=10, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    g = make_graph(B, H, S, D)
    be = CudaCBackend(chip="sm_86")
    be._init_emitters()
    # swap emitter
    be._EMITTERS["flash_attention"] = getattr(attn, emitter_name)
    art = be.lower(g)
    comp = be.compile(art)
    if not comp.success:
        print(f"COMPILE FAIL [{emitter_name}]:", comp.error)
        return None
    inputs = {"Q": q, "K": k, "V": v}
    out = be.run(comp, inputs)["O"]
    ref = torch_ref(q, k, v).numpy()
    rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-8)
    ms = be.benchmark(comp, inputs, iters=iters, warmup=warmup)
    return rel, ms * 1000.0  # us


if __name__ == "__main__":
    emitter = sys.argv[1] if len(sys.argv) > 1 else "emit_cuda_c_flash_attention"
    B, H, D = 1, 8, 64
    print(f"emitter={emitter}  B={B} H={H} D={D}")
    print(f"{'S':>6} {'rel_err':>12} {'us':>12}")
    for S in [128, 256, 512, 1024, 2048]:
        r = run_case(emitter, B, H, S, D)
        if r is None:
            continue
        rel, us = r
        print(f"{S:>6} {rel:>12.3e} {us:>12.2f}")
