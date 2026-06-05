# `swiglu_packed` Baseline Audit — Route B (Exhaustive Community Survey)

**Date:** 2026-05-30
**Op spec (Arke benchmark catalog):**

```
swiglu_packed:
  X: Tensor[M, 2K]            # packed gate/up activation
  W: Tensor[K, N]             # down-projection weight
  computation: gate, up = split(X, 2, dim=-1); H = silu(gate) * up; Y = H @ W
  output: Tensor[M, N]
```

This is the **true fused SwiGLU FFN down-projection**: silu·gate·multiply **AND**
the down-projection matmul in a single GPU kernel launch. It is **architecturally
distinct** from `silu_and_mul`, which only performs the elementwise payload
(`split → silu*mul`) without the trailing matmul.

## Audit Scope

Route B (per Leon 2026-05-30 `1yb`): exhaustive survey of all priority-band
community kernel libraries listed in `docs/benchmark/golden-kernel-ladder.md`,
plus extended candidates (xformers, TransformerEngine, Megatron-LM, FlashInfer,
NVIDIA Apex, DeepSpeed).

For each repo: clone shallow @ HEAD, grep for SwiGLU / silu_and_mul /
gated_mlp / fused_ffn variants, read every kernel source hit, classify as:

| Verdict | Meaning |
|:--------|:--------|
| **PACKED-FUSED** | `(silu(X[:,:K]) * X[:,K:]) @ W` in a single kernel — match |
| **PAYLOAD-ONLY** | Elementwise `silu*mul` only; no matmul fused in |
| **DECOMPOSED**   | Multiple kernels orchestrated at Python module level |
| **NONE**         | No SwiGLU code path at all |

## Results

### Priority band P0 (cuBLAS/cuDNN)

| Repo | Latest commit | Verdict |
|:-----|:--------------|:--------|
| PyTorch vendor backends | n/a | **NONE** — `torch.matmul` and `F.silu` are separate kernels; no vendor-fused gated-activation+matmul kernel exists |

### Priority band P1

#### FlagGems — https://github.com/FlagOpen/FlagGems

- **Latest commit:** `9f836360` (2026-05-30T16:36:13+08:00, master)
- **Verdict:** **PAYLOAD-ONLY**
- **Files inspected:**
  - `src/flag_gems/fused/swiglu.py` — `swiglu_kernel`, `dswiglu_kernel`
  - `src/flag_gems/fused/silu_and_mul.py`
  - `src/flag_gems/fused/silu_and_mul_with_clamp.py`
  - `src/flag_gems/fused_moe_mxq.py` — `fused_moe_kernel_fp16_swiglu` (MoE-specific, wrong topology)
- **Evidence:**
  ```python
  # src/flag_gems/fused/swiglu.py — swiglu_kernel body
  x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
  x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)
  silu_x_a = x_a * sigmoid(x_a)
  out = silu_x_a * x_b
  tl.store(output_ptr, out.to(x_a.dtype), mask=mask)
  ```
  Single packed `[M, 2H]` input → output `[M, H]`. No `tl.dot`, no weight tensor.
- **Near-miss:** `fused_moe_kernel_fp16_swiglu` does fuse `silu*mul` with `@ W` but
  for the full MoE FFN `W2 @ (silu(W1·x) * (W3·x))` — its inputs are raw hidden
  states + three weight matrices + expert routing arrays. Architecturally different
  from packed-activation post-gate/up→down.

#### Liger-Kernel — https://github.com/linkedin/Liger-Kernel

- **Latest commit:** `9497a29b` (2026-05-28T00:46:22Z, main)
- **Verdict:** **PAYLOAD-ONLY**
- **Files inspected:**
  - `src/liger_kernel/ops/swiglu.py` — `_swiglu_forward_kernel`, `_swiglu_backward_kernel`, `LigerSiLUMulFunction`
  - `src/liger_kernel/ops/tiled_mlp.py` — `LigerTiledMLPFunction` (pure Python, no Triton)
  - `src/liger_kernel/ops/backends/_ascend/ops/swiglu.py`
- **Evidence:**
  ```python
  # _swiglu_forward_kernel body
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row).cast(b_row.dtype) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
  ```
  Signature `(a_ptr, b_ptr, c_ptr, stride, gate_multiplier, n_cols, BLOCK_SIZE)`
  is purely elementwise. `down_multiplier` in `LigerSiLUMulFunction` is a **scalar**,
  not a `W` matmul. `grep tl.dot` in this file returns 0 hits.
- **Near-miss:** `tiled_mlp.py` is a `torch.autograd.Function` that chunks the sequence
  dim and re-invokes user `fn(mlp_module, x_shard)` — no Triton kernel, no fused matmul.

### Priority band P2

#### vLLM — https://github.com/vllm-project/vllm

- **Latest commit:** `3becc5db` (2026-05-30T10:13:18Z)
- **Verdict:** **PAYLOAD-ONLY / DECOMPOSED**
- **Files inspected:**
  - `csrc/quantization/activation_kernels.cu` — `silu_and_mul`, `silu_and_mul_quant`, `silu_mul_fp8_quant_deep_gemm_kernel`
  - `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`
  - `csrc/ops.h:49–70`
  - `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:354–380`
- **Evidence:**
  ```cpp
  // csrc/ops.h
  void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
  void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit);
  void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input, torch::Tensor& scale);
  ```
  No `weight`/`W` parameter on any silu+mul op. The MoE forward calls
  `ops.silu_and_mul_per_block_quant(...)` and then a **separate**
  `invoke_fused_moe_triton_kernel(qintermediate_cache2, w2, ...)` — two distinct
  kernel launches.

#### flash-attn — https://github.com/Dao-AILab/flash-attention

- **Verdict:** **NONE** (out of scope — attention library, no FFN kernels)

#### FlashMLA — https://github.com/deepseek-ai/FlashMLA

- **Verdict:** **NONE** (out of scope — MLA attention only, sm_90+)

#### xformers — https://github.com/facebookresearch/xformers

- **Latest commit:** `c04f47b6` (2026-05-21T15:42:07Z)
- **Verdict:** **DECOMPOSED**
- **Files inspected:**
  - `xformers/ops/swiglu_op.py` — only registered op is `SwiGLUEagerOp`
  - `xformers/ops/__init__.py:42`
  - No `*.cu` / `*.h` contain `swiglu` (grep ⇒ 0 hits)
- **Evidence:**
  ```python
  # xformers/ops/swiglu_op.py:111-123
  def _eager_functional_swiglu(x, w1, b1, w2, b2, w3, b3):
      x1 = F.linear(x, w1, b1)
      x2 = F.linear(x, w2, b2)
      hidden = F.silu(x1) * x2
      return F.linear(hidden, w3, b3)
  ```
  The historical `_SwiGLUDecomposedFunc` docstring explicitly states:
  *"This implementation is worse than pytorch, because pytorch is able to fuse
  some operations (eg the linear forward …) that are decomposed here."* —
  xformers offloads fusion to PyTorch/inductor and ships no first-party fused
  down-projection kernel.

#### NVIDIA TransformerEngine — https://github.com/NVIDIA/TransformerEngine

- **Latest commit:** `79821e2b` (2026-05-29T22:38:00Z)
- **Verdict:** **DECOMPOSED for the down-projection (FC2)**
- **Files inspected:**
  - `transformer_engine/common/activation/swiglu.cu` — `nvte_swiglu`
  - `transformer_engine/pytorch/module/layernorm_mlp.py:572,603,616,675`
  - `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py:399,438`
- **Evidence:**
  ```cpp
  // common/activation/swiglu.cu
  void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
    using namespace transformer_engine;
    Empty e = {};
    gated_act_fn<fp32, Empty, silu<fp32, fp32>>(input, output, e, stream);
  }
  ```
  ```python
  # pytorch/module/layernorm_mlp.py
  # ACTIVATION - sometimes activation is fused with the GEMM above.
  ...
  act_out = activation_func(fc1_out, None, **act_params)
  act_out = fc2_input_quantizer(act_out)
  ...
  gemm_out, *_, reduce_scatter_out = general_gemm(fc2_weight_final, ...)
  ```
- **Architectural note:** TE *does* fuse activation, but on the **wrong side** —
  it fuses into the **preceding** GEMM (FC1 epilogue, esp. sm100 cuDNN path),
  not into the **following** GEMM (FC2 down-projection). The Arke
  `swiglu_packed` op corresponds to the post-activation + FC2 fusion, which TE
  does **not** ship.

### Extended candidates

#### Megatron-LM — https://github.com/NVIDIA/Megatron-LM

- **Latest commit:** `52d1d681` (2026-05-30T07:51:22Z)
- **Verdict:** **DECOMPOSED (payload-only fusion)**
- **Files inspected:**
  - `megatron/core/fusions/fused_bias_swiglu.py`
  - `megatron/core/transformer/mlp.py`
  - `tests/unit_tests/fusions/test_swiglu_fusion.py`
- **Evidence:**
  ```python
  @jit_fuser
  def swiglu(y):
      y_1, y_2 = torch.chunk(y, 2, -1)
      return F.silu(y_1) * y_2
  ```
  TorchScript JIT-fused **elementwise only**. `mlp.py` pipeline is
  `linear_fc1 → activation_func → linear_fc2` — `linear_fc2` is a separate
  `ColumnParallelLinear/RowParallelLinear` call. Unit test confirms input
  `[16, 64]` → output `[16, 32]` (last dim halved, no `@ W`).

#### FlashInfer — https://github.com/flashinfer-ai/flashinfer

- **Latest commit:** `fc12ef21` (2026-05-30T04:43:58Z)
- **Verdict:** **PAYLOAD-ONLY** + unrelated FC1-side MoE GEMM+SwiGLU fusion
- **Files inspected:**
  - `flashinfer/activation.py`
  - `flashinfer/triton/kernels/activation.py`, `flashinfer/triton/activation.py`
  - `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py`
  - `docs/tutorials/jax_tvm_ffi/gemma3_flashinfer_jax.py`
- **Evidence:**
  ```python
  # flashinfer/activation.py
  assert input.shape[-1] == 2 * output.shape[-1]
  ```
  ```python
  # blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py:332
  # C[row] = up * silu(gate), where [gate, up] = alpha[expert] * (A[token_id] @ B[expert])
  ```
  The CuteDSL MoE kernel fuses `A @ B → split → silu*mul` — **up-side / FC1**
  fusion, opposite topology from `swiglu_packed`.

#### NVIDIA Apex — https://github.com/NVIDIA/apex

- **Latest commit:** `becbb77c` (2026-05-27T15:17:36Z)
- **Verdict:** **NONE**
- **Files inspected:** `apex/mlp/mlp.py`, `csrc/mlp_cuda.cu`, `apex/fused_dense/fused_dense.py`, `csrc/fused_dense.cpp`
- **Evidence:** Full repo grep for `swiglu|silu_and_mul|gated|silu` ⇒ zero meaningful hits.
  `mlp_cuda.cu` only supports ReLU/Sigmoid epilogues via cuBLASLt
  (`CUBLASLT_EPILOGUE_RELU_BIAS`). `fused_dense.py` provides
  `linear_gelu_linear_forward` (GeLU, non-gated, non-SiLU). Apex has **no
  SwiGLU code path** at all.

#### DeepSpeed — https://github.com/microsoft/DeepSpeed

- **Latest commit:** `819af0e5` (2026-05-30T06:29:29Z)
- **Verdict:** **PAYLOAD-ONLY + DECOMPOSED**
- **Files inspected:**
  - `deepspeed/inference/v2/kernels/core_ops/gated_activations/gated_activation_kernels_cuda.cu`
  - `deepspeed/inference/v2/kernels/core_ops/gated_activations/gated_activation.py`
  - `deepspeed/inference/v2/model_implementations/inference_transformer_base.py`
  - `deepspeed/ops/transformer/inference/op_binding/gated_activation.py`
- **Evidence:**
  ```cuda
  // gated_activation_kernels_cuda.cu:37-40
  template <> DS_D_INLINE float gated_act_fn<ActivationType::SiGLU>(float x, float y) {
      return y * (x / (1.0f + expf(-x)));
  }
  // line 58:
  T* output_row = output + row * cols / 2;
  ```
  Output stride `cols/2` confirms elementwise payload only. The v2 inference
  engine pipeline is `mlp_1 (linear up) → gated_activation (kernel) → mlp_2
  (linear down)` — three separate kernels at the module level.

## Summary Table

| Repo | Latest commit (UTC) | Verdict |
|:-----|:--------------------|:--------|
| PyTorch (P0 cuBLAS/cuDNN) | n/a | NONE |
| FlagGems (P1) | `9f836360` (2026-05-30) | PAYLOAD-ONLY |
| Liger-Kernel (P1) | `9497a29b` (2026-05-28) | PAYLOAD-ONLY |
| vLLM (P2) | `3becc5db` (2026-05-30) | PAYLOAD-ONLY / DECOMPOSED |
| flash-attn (P2) | n/a | NONE (out of scope) |
| FlashMLA (P2) | n/a | NONE (out of scope) |
| xformers (ext.) | `c04f47b6` (2026-05-21) | DECOMPOSED |
| TransformerEngine (ext.) | `79821e2b` (2026-05-29) | DECOMPOSED (FC2; FC1 fusion exists) |
| Megatron-LM (ext.) | `52d1d681` (2026-05-30) | DECOMPOSED (payload-only JIT fusion) |
| FlashInfer (ext.) | `fc12ef21` (2026-05-30) | PAYLOAD-ONLY + FC1-side MoE fusion |
| NVIDIA Apex (ext.) | `becbb77c` (2026-05-27) | NONE |
| DeepSpeed (ext.) | `819af0e5` (2026-05-30) | PAYLOAD-ONLY + DECOMPOSED |

## Conclusion

**No community kernel library** at HEAD as of 2026-05-30 ships a single fused
kernel computing `Y = (silu(X[:, :K]) * X[:, K:]) @ W`. Three independent
patterns emerge:

1. **Payload-only fusion** (FlagGems, Liger, Megatron, DeepSpeed, FlashInfer):
   ship a Triton/CUDA kernel that fuses `split → silu × mul` but leaves the
   down-projection to a separate cuBLAS/Marlin/Triton GEMM call.
2. **Up-side (FC1) GEMM+activation fusion** (TransformerEngine sm100 cuDNN,
   FlashInfer CuteDSL MoE): fuse the gated activation into the **preceding**
   matmul as an epilogue. Architecturally opposite to `swiglu_packed`.
3. **No SwiGLU support** (Apex, flash-attn, FlashMLA): out of scope.

`swiglu_packed` is genuinely a fusion gap relative to all surveyed P0/P1/P2
community kernels.

## Recommended Golden Kernel Assignment

Following the precedent set by `dequantize_per_channel` and `cast` in
`docs/benchmark/golden-kernel-ladder.md`:

| Op             | Golden                     | Fallback | Notes |
|:---------------|:---------------------------|:---------|:------|
| swiglu_packed  | PyTorch-eager (P3)         | —        | D8-X1 demo op; no production single-kernel baseline in any of 9 audited community libraries; audit-degraded |

PyTorch-eager golden = `(F.silu(X[:, :K]) * X[:, K:]) @ W` (two-kernel
sequence: gated activation + cuBLAS GEMM). This becomes the correctness
oracle **and** the perf denominator. Arke's job is to beat this two-kernel
sequence with a single fused Triton kernel (which is precisely the D8-X1 demo
target).

## Audit Trail

| Repo | Audit date | Commit SHA | Investigator | Method |
|:-----|:-----------|:-----------|:-------------|:-------|
| FlagGems | 2026-05-30 | `9f836360` | Kitty subagent A | shallow clone + grep + read kernel source |
| Liger-Kernel | 2026-05-30 | `9497a29b` | Kitty subagent A | shallow clone + grep + read kernel source |
| vLLM | 2026-05-30 | `3becc5db` | Kitty subagent B | shallow clone + grep + read csrc/ + Python orchestration |
| xformers | 2026-05-30 | `c04f47b6` | Kitty subagent B | shallow clone + grep + read swiglu_op.py |
| TransformerEngine | 2026-05-30 | `79821e2b` | Kitty subagent B | shallow clone + grep + read layernorm_mlp.py |
| Megatron-LM | 2026-05-30 | `52d1d681` | Kitty subagent C | shallow clone + grep + read fused_bias_swiglu.py + mlp.py |
| FlashInfer | 2026-05-30 | `fc12ef21` | Kitty subagent C | shallow clone + grep + read activation.py + MoE fusion |
| NVIDIA Apex | 2026-05-30 | `becbb77c` | Kitty subagent C | shallow clone + grep (no SwiGLU hits) |
| DeepSpeed | 2026-05-30 | `819af0e5` | Kitty subagent C | shallow clone + grep + read gated_activation_kernels_cuda.cu |

---

*Audit author: Kitty (Hermes) on behalf of Leon, lead engineer of Arke.*
*Methodology: shallow clone @ HEAD; grep for `swiglu|silu_and_mul|silu_mul|gated_mlp|gated_linear|fused.*ffn|mlp.*fused|swiglu.*fused`; read every kernel source hit; classify by signature analysis.*
