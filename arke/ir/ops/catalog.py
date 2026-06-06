# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke IR — Kernel Schema View (S6 OpSchema).

This file enriches each *kernel name* from the benchmark SSOT
(``docs/benchmark/benchmark-ops.md``, parsed by
``benchmarks.op_registry``) with the metadata that the compiler /
interpreter / codegen need: ``shape_rule``, ``template_hint``,
``reference_impl``, ``input_gen``.

╔══════════════════════════════════════════════════════════════════════╗
║  Layer boundary — read before extending this file                    ║
║                                                                      ║
║  This catalog is a **kernel-schema view**, not the kernel SSOT, and  ║
║  not the future IR-dialect primitive registry.                       ║
║                                                                      ║
║    • Kernel SSOT          : ``docs/benchmark/benchmark-ops.md``      ║
║      Authoritative parser : ``benchmarks/op_registry.py``            ║
║      Layer                : *high-level kernels* (matmul,            ║
║                             flash_attention, rmsnorm, rope, …)       ║
║      Use ``total_ops() / ALL_OPS / OT_OPS`` to enumerate.            ║
║                                                                      ║
║    • Kernel schema view   : THIS FILE + ``arke/ir/ops/registry.py``  ║
║      Purpose              : attach compiler/runtime metadata to each ║
║                             SSOT kernel name. Derives from SSOT,     ║
║                             must not invent new kernel names. Tests  ║
║                             ``test_ir_ops_schema_covers_kernel_catalog`` ║
║                             and ``test_ir_ops_schema_no_shadow_kernels`` ║
║                             enforce this in both directions.         ║
║                                                                      ║
║    • IR dialect primitives: *not built yet* (Stage 8 / 9 + later     ║
║      MLIR dialect). Will live under a separate registry — e.g.       ║
║      ``arke/ir/dialects/...`` — and enumerate *low-level* ops        ║
║      (load, store, arith.*, scf.*, …). IR primitives will *lower    ║
║      to* the kernel schemas in this file; the two layers stay        ║
║      decoupled by design.                                            ║
║                                                                      ║
║  Adding a new kernel: (1) edit the SSOT markdown, (2) register an    ║
║  OpSchema here, (3) add a ``ref_*`` function in reference_impls.py.  ║
║  Never hardcode the catalog size in this file or its consumers —    ║
║  call ``benchmarks.op_registry.total_ops()`` instead.                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from arke.ir.ops.schema import InputGen, OpSchema, ReferenceImpl, ShapeRule, TemplateHint
from arke.ir.ops.reference_impls import (
    ref_add, ref_argmax, ref_batch_matmul, ref_cast, ref_concat, ref_copy,
    ref_cross_attention, ref_cross_entropy, ref_cumsum,
    ref_dequantize_per_channel, ref_embedding, ref_exp,
    ref_flash_attention, ref_fused_linear_cross_entropy,
    ref_gather, ref_gelu_and_mul, ref_gelu, ref_grouped_matmul,
    ref_grouped_query_attention, ref_layernorm, ref_matmul,
    ref_multi_latent_attention, ref_mul, ref_neg, ref_paged_attention,
    ref_permute, ref_quantize_per_token, ref_reduce_max, ref_reduce_mean,
    ref_reduce_sum, ref_relu, ref_rmsnorm, ref_rmsnorm_residual, ref_rope,
    ref_rsqrt, ref_scatter, ref_sigmoid, ref_silu, ref_softmax, ref_split,
    ref_silu_and_mul, ref_swiglu_packed, ref_tanh, ref_topk, ref_transpose, ref_where,
)

OP_CATALOG: dict[str, OpSchema] = {}

def _register(op: OpSchema) -> OpSchema:
    OP_CATALOG[op.name] = op
    return op

# OT2: Compute-Dense
MATMUL = _register(OpSchema(name="matmul", category="compute",
    inputs={"A": "Tensor[M,K]", "B": "Tensor[K,N]"}, output="Tensor[M,N]",
    computation="C[i,j] = sum(A[i,k] * B[k,j], axis=k)",
    index_vars=["i","j","k"], reduction_axes=["k"],
    properties=["associative","distributive"], can_fuse_as="prologue",
    numpy_ref="np.matmul(A, B)",
    shape_rule=ShapeRule(kind="matmul_rule"),
    template_hint=TemplateHint(template_name="matmul"),
    reference_impl=ReferenceImpl(fn=ref_matmul, dtype_map={"bf16":"f32"}),
    input_gen=InputGen(distributions={"A":"normal","B":"normal"}, constraints=["A.shape[-1]==B.shape[-2]"]),
))
BATCH_MATMUL = _register(OpSchema(name="batch_matmul", category="compute",
    inputs={"A": "Tensor[B,M,K]", "B": "Tensor[B,K,N]"}, output="Tensor[B,M,N]",
    computation="C[b,i,j] = sum(A[b,i,k] * B[b,k,j], axis=k)",
    index_vars=["b","i","j","k"], reduction_axes=["k"],
    properties=["associative","distributive"], can_fuse_as="prologue",
    numpy_ref="np.matmul(A, B)",
    shape_rule=ShapeRule(kind="batch_matmul_rule"),
    template_hint=TemplateHint(template_name="batch_matmul"),
    reference_impl=ReferenceImpl(fn=ref_batch_matmul, dtype_map={"bf16":"f32"}),
    input_gen=InputGen(distributions={"A":"normal","B":"normal"}, constraints=["A.shape[-1]==B.shape[-2]"]),
))
GROUPED_MATMUL = _register(OpSchema(name="grouped_matmul", category="compute",
    inputs={"X":"Tensor[B,M,K]","W":"Tensor[E,K,N]","indices":"Tensor[B]"}, output="Tensor[B,M,N]",
    computation="Y[b,i,j] = sum(X[b,i,k] * W[indices[b],k,j], axis=k)",
    index_vars=["b","i","j","k"], reduction_axes=["k"],
    properties=["associative"], can_fuse_as="prologue",
    numpy_ref="np.stack([X[b] @ W[idx[b]] for b in range(B)])",
    shape_rule=ShapeRule(kind="batch_matmul_rule", input_key="X"),
    template_hint=TemplateHint(template_name="grouped_matmul"),
    reference_impl=ReferenceImpl(fn=ref_grouped_matmul),
    input_gen=InputGen(distributions={"X":"normal","W":"normal","indices":"randint"}, ranges={"indices":(0,1)}),
))

# OT0: Elementwise unary
RELU = _register(OpSchema(name="relu", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = max(X, 0)", properties=["elementwise","monotonic"],
    can_fuse_as="epilogue", numpy_ref="np.maximum(X, 0)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"relu"}),
    reference_impl=ReferenceImpl(fn=ref_relu),
    input_gen=InputGen(distributions={"X":"normal"}),
))
GELU = _register(OpSchema(name="gelu", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = X * Phi(X)", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="0.5*X*(1+erf(X/sqrt(2)))",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"gelu"}),
    reference_impl=ReferenceImpl(fn=ref_gelu),
    input_gen=InputGen(distributions={"X":"normal"}),
))
SILU = _register(OpSchema(name="silu", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = X * sigmoid(X)", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="X/(1+np.exp(-X))",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"silu"}),
    reference_impl=ReferenceImpl(fn=ref_silu),
    input_gen=InputGen(distributions={"X":"normal"}),
))
TANH = _register(OpSchema(name="tanh", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = tanh(X)", properties=["elementwise","monotonic"],
    can_fuse_as="epilogue", numpy_ref="np.tanh(X)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"tanh"}),
    reference_impl=ReferenceImpl(fn=ref_tanh),
    input_gen=InputGen(distributions={"X":"normal"}),
))
SIGMOID = _register(OpSchema(name="sigmoid", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = 1/(1+exp(-X))", properties=["elementwise","monotonic"],
    can_fuse_as="epilogue", numpy_ref="1/(1+np.exp(-X))",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"sigmoid"}),
    reference_impl=ReferenceImpl(fn=ref_sigmoid),
    input_gen=InputGen(distributions={"X":"normal"}),
))
NEG = _register(OpSchema(name="neg", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = -X", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="-X",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"neg"}),
    reference_impl=ReferenceImpl(fn=ref_neg),
    input_gen=InputGen(distributions={"X":"normal"}),
))
EXP = _register(OpSchema(name="exp", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = exp(X)", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="np.exp(X)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"exp"}),
    reference_impl=ReferenceImpl(fn=ref_exp),
    input_gen=InputGen(distributions={"X":"uniform"}, ranges={"X":(-5.0,5.0)}),
))
RSQRT = _register(OpSchema(name="rsqrt", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = 1/sqrt(X)", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="1/np.sqrt(X)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="elementwise", extra_ctx={"op_variant":"rsqrt"}),
    reference_impl=ReferenceImpl(fn=ref_rsqrt),
    input_gen=InputGen(distributions={"X":"uniform"}, ranges={"X":(0.1,10.0)}, constraints=["X>0"]),
))
CAST = _register(OpSchema(name="cast", category="elementwise",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = cast(X, target_dtype)", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="X.astype(target_dtype)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="cast"),
    reference_impl=ReferenceImpl(fn=ref_cast),
    input_gen=InputGen(distributions={"X":"normal"}),
    attrs={"target_dtype":"float16"},
))

# OT0: Elementwise binary
ADD = _register(OpSchema(name="add", category="elementwise",
    inputs={"A":"Tensor[...]","B":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = A + B", properties=["elementwise","commutative","associative"],
    can_fuse_as="epilogue", numpy_ref="A + B",
    shape_rule=ShapeRule(kind="same_as_input", input_key="A"),
    template_hint=TemplateHint(template_name="elementwise_binary", extra_ctx={"op_variant":"add"}),
    reference_impl=ReferenceImpl(fn=ref_add),
    input_gen=InputGen(distributions={"A":"normal","B":"normal"}),
))
MUL = _register(OpSchema(name="mul", category="elementwise",
    inputs={"A":"Tensor[...]","B":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = A * B", properties=["elementwise","commutative","associative"],
    can_fuse_as="epilogue", numpy_ref="A * B",
    shape_rule=ShapeRule(kind="same_as_input", input_key="A"),
    template_hint=TemplateHint(template_name="elementwise_binary", extra_ctx={"op_variant":"mul"}),
    reference_impl=ReferenceImpl(fn=ref_mul),
    input_gen=InputGen(distributions={"A":"normal","B":"normal"}),
))
WHERE = _register(OpSchema(name="where_", category="elementwise",
    inputs={"cond":"Tensor[...]","A":"Tensor[...]","B":"Tensor[...]"}, output="Tensor[...]",
    computation="Y = A if cond else B", properties=["elementwise"],
    can_fuse_as="epilogue", numpy_ref="np.where(cond, A, B)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="A"),
    template_hint=TemplateHint(template_name="elementwise_binary", extra_ctx={"op_variant":"where"}),
    reference_impl=ReferenceImpl(fn=ref_where),
    input_gen=InputGen(distributions={"cond":"bool_mask","A":"normal","B":"normal"}),
))

# OT1: Reduction
LAYERNORM = _register(OpSchema(name="layernorm", category="reduce",
    inputs={"X":"Tensor[M,N]","W":"Tensor[N]","B":"Tensor[N]"}, output="Tensor[M,N]",
    computation="Y[i,j]=(X[i,j]-mean(X[i,:]))/sqrt(var(X[i,:])+eps)*W[j]+B[j]",
    index_vars=["i","j"], reduction_axes=["j"], properties=["row-wise"],
    numpy_ref="(X-X.mean(-1,keepdims=True))/np.sqrt(X.var(-1,keepdims=True)+eps)*W+B",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="layernorm"),
    reference_impl=ReferenceImpl(fn=ref_layernorm, dtype_map={"bf16":"f32","f16":"f32"}),
    input_gen=InputGen(distributions={"X":"normal","W":"ones","B":"ones"}),
    attrs={"eps":1e-5},
))
RMSNORM = _register(OpSchema(name="rmsnorm", category="reduce",
    inputs={"X":"Tensor[M,N]","W":"Tensor[N]"}, output="Tensor[M,N]",
    computation="Y[i,j]=X[i,j]/sqrt(mean(X[i,:]**2)+eps)*W[j]",
    index_vars=["i","j"], reduction_axes=["j"], properties=["row-wise"],
    numpy_ref="X/np.sqrt(np.mean(X**2,axis=-1,keepdims=True)+eps)*W",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="rmsnorm"),
    reference_impl=ReferenceImpl(fn=ref_rmsnorm, dtype_map={"bf16":"f32","f16":"f32"}),
    input_gen=InputGen(distributions={"X":"normal","W":"ones"}),
    attrs={"eps":1e-6},
))
SOFTMAX = _register(OpSchema(name="softmax", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M,N]",
    computation="Y[i,j]=exp(X[i,j])/sum(exp(X[i,:]),axis=j)",
    index_vars=["i","j"], reduction_axes=["j"], properties=["row-wise"],
    numpy_ref="scipy.special.softmax(X, axis=-1)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="softmax"),
    reference_impl=ReferenceImpl(fn=ref_softmax),
    input_gen=InputGen(distributions={"X":"normal"}),
))
REDUCE_SUM = _register(OpSchema(name="reduce_sum", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M]",
    computation="Y[i]=sum(X[i,:],axis=j)", index_vars=["i","j"], reduction_axes=["j"],
    properties=["associative","commutative"], numpy_ref="np.sum(X, axis=-1)",
    shape_rule=ShapeRule(kind="reduce_rule", axes=[-1]),
    template_hint=TemplateHint(template_name="reduction", extra_ctx={"op_variant":"sum"}),
    reference_impl=ReferenceImpl(fn=ref_reduce_sum),
    input_gen=InputGen(distributions={"X":"normal"}),
))
REDUCE_MAX = _register(OpSchema(name="reduce_max", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M]",
    computation="Y[i]=max(X[i,:],axis=j)", index_vars=["i","j"], reduction_axes=["j"],
    properties=["associative","commutative"], numpy_ref="np.max(X, axis=-1)",
    shape_rule=ShapeRule(kind="reduce_rule", axes=[-1]),
    template_hint=TemplateHint(template_name="reduction", extra_ctx={"op_variant":"max"}),
    reference_impl=ReferenceImpl(fn=ref_reduce_max),
    input_gen=InputGen(distributions={"X":"normal"}),
))
REDUCE_MEAN = _register(OpSchema(name="reduce_mean", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M]",
    computation="Y[i]=mean(X[i,:],axis=j)", index_vars=["i","j"], reduction_axes=["j"],
    properties=["associative"], numpy_ref="np.mean(X, axis=-1)",
    shape_rule=ShapeRule(kind="reduce_rule", axes=[-1]),
    template_hint=TemplateHint(template_name="reduction", extra_ctx={"op_variant":"mean"}),
    reference_impl=ReferenceImpl(fn=ref_reduce_mean),
    input_gen=InputGen(distributions={"X":"normal"}),
))
ARGMAX = _register(OpSchema(name="argmax", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M]",
    computation="Y[i]=argmax(X[i,:],axis=j)", index_vars=["i","j"], reduction_axes=["j"],
    numpy_ref="np.argmax(X, axis=-1)",
    shape_rule=ShapeRule(kind="reduce_rule", axes=[-1]),
    template_hint=TemplateHint(template_name="reduction", extra_ctx={"op_variant":"argmax"}),
    reference_impl=ReferenceImpl(fn=ref_argmax),
    input_gen=InputGen(distributions={"X":"normal"}),
))
TOPK = _register(OpSchema(name="topk", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M,K]",
    computation="values,indices=topk(X[i,:],k,axis=j)", index_vars=["i","j"], reduction_axes=["j"],
    numpy_ref="np.partition(X,-k,axis=-1)[...,-k:]",
    shape_rule=ShapeRule(kind="topk_rule"),
    template_hint=TemplateHint(template_name="topk"),
    reference_impl=ReferenceImpl(fn=ref_topk),
    input_gen=InputGen(distributions={"X":"normal"}),
    attrs={"k":1},
))
CUMSUM = _register(OpSchema(name="cumsum", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[M,N]",
    computation="Y[i,j]=sum(X[i,0:j+1])", index_vars=["i","j"],
    numpy_ref="np.cumsum(X, axis=-1)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="cumsum"),
    reference_impl=ReferenceImpl(fn=ref_cumsum),
    input_gen=InputGen(distributions={"X":"normal"}),
))

# OT2: Data Movement
TRANSPOSE = _register(OpSchema(name="transpose", category="move",
    inputs={"X":"Tensor[M,N]"}, output="Tensor[N,M]",
    computation="Y[j,i]=X[i,j]", index_vars=["i","j"], numpy_ref="X.T",
    shape_rule=ShapeRule(kind="custom",
        fn=lambda shapes,attrs: shapes["X"][:-2]+list(reversed(shapes["X"][-2:]))),
    template_hint=TemplateHint(template_name="transpose"),
    reference_impl=ReferenceImpl(fn=ref_transpose),
    input_gen=InputGen(distributions={"X":"normal"}),
))
CONCAT = _register(OpSchema(name="concat", category="move",
    inputs={"A":"Tensor[M,N1]","B":"Tensor[M,N2]"}, output="Tensor[M,N1+N2]",
    computation="Y=concat(A,B,axis=-1)", index_vars=["i","j"],
    numpy_ref="np.concatenate([A, B], axis=-1)",
    shape_rule=ShapeRule(kind="concat_rule", axis_attr="axis"),
    template_hint=TemplateHint(template_name="data_movement", extra_ctx={"op_variant":"concat"}),
    reference_impl=ReferenceImpl(fn=ref_concat),
    input_gen=InputGen(distributions={"A":"normal","B":"normal"}),
))
SPLIT = _register(OpSchema(name="split", category="move",
    inputs={"X":"Tensor[M,N]"}, output="Tuple[Tensor[M,N/2],Tensor[M,N/2]]",
    computation="A,B=split(X,2,axis=-1)", index_vars=["i","j"],
    numpy_ref="np.split(X, 2, axis=-1)",
    shape_rule=ShapeRule(kind="split_rule", axis_attr="axis"),
    template_hint=TemplateHint(template_name="data_movement", extra_ctx={"op_variant":"split"}),
    reference_impl=ReferenceImpl(fn=ref_split),
    input_gen=InputGen(distributions={"X":"normal"}, constraints=["X.shape[-1]%2==0"]),
))
GATHER = _register(OpSchema(name="gather", category="move",
    inputs={"X":"Tensor[M,N]","idx":"Tensor[M,K]"}, output="Tensor[M,K]",
    computation="Y[i,j]=X[i,idx[i,j]]", index_vars=["i","j"],
    numpy_ref="np.take_along_axis(X, idx, axis=-1)",
    shape_rule=ShapeRule(kind="gather_rule", input_key="idx"),
    template_hint=TemplateHint(template_name="index_ops", extra_ctx={"op_variant":"gather"}),
    reference_impl=ReferenceImpl(fn=ref_gather),
    input_gen=InputGen(distributions={"X":"normal","idx":"randint"}, ranges={"idx":(0,1)},
                       constraints=["idx values < X.shape[-1]"]),
))
SCATTER = _register(OpSchema(name="scatter", category="move",
    inputs={"X":"Tensor[M,N]","idx":"Tensor[M,K]","src":"Tensor[M,K]"}, output="Tensor[M,N]",
    computation="Y=X.copy();Y[i,idx[i,j]]=src[i,j]", index_vars=["i","j"],
    numpy_ref="np.put_along_axis(X.copy(), idx, src, axis=-1)",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="index_ops", extra_ctx={"op_variant":"scatter"}),
    reference_impl=ReferenceImpl(fn=ref_scatter),
    input_gen=InputGen(distributions={"X":"normal","idx":"randint","src":"normal"},
                       ranges={"idx":(0,1)}, constraints=["idx values < X.shape[-1]"]),
))
EMBEDDING = _register(OpSchema(name="embedding", category="move",
    inputs={"indices":"Tensor[B,S]","weight":"Tensor[V,D]"}, output="Tensor[B,S,D]",
    computation="Y[b,s,:]=weight[indices[b,s],:]", index_vars=["b","s","d"],
    numpy_ref="weight[indices]",
    shape_rule=ShapeRule(kind="embedding_rule"),
    template_hint=TemplateHint(template_name="data_movement", extra_ctx={"op_variant":"embedding"}),
    reference_impl=ReferenceImpl(fn=ref_embedding),
    input_gen=InputGen(distributions={"indices":"randint","weight":"normal"},
                       ranges={"indices":(0,1)}, constraints=["indices values < weight.shape[0]"]),
))
PERMUTE = _register(OpSchema(name="permute", category="move",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y=permute(X,dims)", numpy_ref="np.transpose(X,axes=dims)",
    shape_rule=ShapeRule(kind="permute_rule"),
    template_hint=TemplateHint(template_name="data_movement", extra_ctx={"op_variant":"permute"}),
    reference_impl=ReferenceImpl(fn=ref_permute),
    input_gen=InputGen(distributions={"X":"normal"}),
    attrs={"dims":[0,2,1,3]},
))
COPY = _register(OpSchema(name="copy_", category="move",
    inputs={"X":"Tensor[...]"}, output="Tensor[...]",
    computation="Y=X.clone()", properties=["elementwise"], numpy_ref="X.copy()",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="data_movement", extra_ctx={"op_variant":"copy"}),
    reference_impl=ReferenceImpl(fn=ref_copy),
    input_gen=InputGen(distributions={"X":"normal"}),
))

# OT3: Gated + Fused Norms + Rope + Quant + Loss
SWIGLU = _register(OpSchema(name="silu_and_mul", category="elementwise",
    inputs={"X":"Tensor[...,2N]"}, output="Tensor[...,N]",
    computation="x1,x2=split(X);Y=silu(x1)*x2",
    properties=["elementwise","gated"], can_fuse_as="epilogue",
    numpy_ref="x1,x2=np.split(X,2,axis=-1);x1/(1+np.exp(-x1))*x2",
    shape_rule=ShapeRule(kind="gated_halve_rule", input_key="X"),
    template_hint=TemplateHint(template_name="gated_activation", extra_ctx={"op_variant":"silu_and_mul"}),
    reference_impl=ReferenceImpl(fn=ref_silu_and_mul),
    input_gen=InputGen(distributions={"X":"normal"}, constraints=["X.shape[-1]%2==0"]),
))
SWIGLU_PACKED = _register(OpSchema(name="swiglu_packed", category="gated",
    inputs={"X":"Tensor[M,2K]", "W":"Tensor[K,N]"}, output="Tensor[M,N]",
    computation="gate,up=split(X);H=silu(gate)*up;Y=H@W",
    index_vars=["i", "j"], reduction_axes=["k"],
    properties=["gated", "matmul", "fused_projection"], can_fuse_as="compound",
    numpy_ref="gate,up=np.split(X,2,axis=-1);(gate/(1+np.exp(-gate))*up)@W",
    shape_rule=ShapeRule(kind="matmul_rule", input_key="X"),
    template_hint=TemplateHint(template_name="gated_activation", extra_ctx={"op_variant":"swiglu_packed"}),
    reference_impl=ReferenceImpl(fn=ref_swiglu_packed),
    input_gen=InputGen(distributions={"X":"normal", "W":"normal"}, constraints=["X.shape[-1]%2==0", "W.shape[0]==X.shape[-1]/2"]),
))
GEGLU = _register(OpSchema(name="gelu_and_mul", category="elementwise",
    inputs={"X":"Tensor[...,2N]"}, output="Tensor[...,N]",
    computation="x1,x2=split(X);Y=gelu(x1)*x2",
    properties=["elementwise","gated"], can_fuse_as="epilogue",
    numpy_ref="x1,x2=np.split(X,2,axis=-1);0.5*x1*(1+erf(x1/sqrt(2)))*x2",
    shape_rule=ShapeRule(kind="gated_halve_rule", input_key="X"),
    template_hint=TemplateHint(template_name="gated_activation", extra_ctx={"op_variant":"gelu_and_mul"}),
    reference_impl=ReferenceImpl(fn=ref_gelu_and_mul),
    input_gen=InputGen(distributions={"X":"normal"}, constraints=["X.shape[-1]%2==0"]),
))
RMSNORM_RESIDUAL = _register(OpSchema(name="rmsnorm_residual", category="reduce",
    inputs={"X":"Tensor[M,N]","residual":"Tensor[M,N]","W":"Tensor[N]"}, output="Tensor[M,N]",
    computation="H=X+residual;Y[i,j]=H[i,j]/sqrt(mean(H[i,:]**2)+eps)*W[j]",
    index_vars=["i","j"], reduction_axes=["j"], properties=["row-wise","fused_residual"],
    numpy_ref="H=X+residual;H/np.sqrt(np.mean(H**2,axis=-1,keepdims=True)+eps)*W",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="rmsnorm_residual"),
    reference_impl=ReferenceImpl(fn=ref_rmsnorm_residual, dtype_map={"bf16":"f32","f16":"f32"}),
    input_gen=InputGen(distributions={"X":"normal","residual":"normal","W":"ones"}),
    attrs={"eps":1e-6},
))
ROPE = _register(OpSchema(name="rope", category="elementwise",
    inputs={"X":"Tensor[B,H,S,D]","cos":"Tensor[S,D/2]","sin":"Tensor[S,D/2]"},
    output="Tensor[B,H,S,D]",
    computation="Y=X*cos+rotate_half(X)*sin", properties=["elementwise","position_encoding"],
    can_fuse_as="epilogue", numpy_ref="x*cos+rotate_half(x)*sin",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="rope"),
    reference_impl=ReferenceImpl(fn=ref_rope),
    input_gen=InputGen(distributions={"X":"normal","cos":"uniform","sin":"uniform"},
                       ranges={"cos":(-1.0,1.0),"sin":(-1.0,1.0)}),
))
CROSS_ENTROPY = _register(OpSchema(name="cross_entropy", category="reduce",
    inputs={"logits":"Tensor[B,V]","labels":"Tensor[B]"}, output="Tensor[]",
    computation="loss=-mean(log_softmax(logits)[i,labels[i]])",
    index_vars=["i","j"], reduction_axes=["i","j"], properties=["loss_function"],
    numpy_ref="-np.mean(np.log(softmax(logits))[np.arange(B),labels])",
    shape_rule=ShapeRule(kind="custom", fn=lambda shapes,attrs:[]),
    template_hint=TemplateHint(template_name="cross_entropy"),
    reference_impl=ReferenceImpl(fn=ref_cross_entropy),
    input_gen=InputGen(distributions={"logits":"normal","labels":"randint"},
                       ranges={"labels":(0,1)}, constraints=["labels values < logits.shape[-1]"]),
))
FUSED_LINEAR_CROSS_ENTROPY = _register(OpSchema(name="fused_linear_cross_entropy", category="compute",
    inputs={"X":"Tensor[B,D]","W":"Tensor[V,D]","labels":"Tensor[B]"}, output="Tensor[]",
    computation="loss=cross_entropy(X@W^T,labels)",
    index_vars=["i","j","k"], reduction_axes=["i","j","k"], properties=["fused","loss_function"],
    numpy_ref="cross_entropy(X@W.T, labels)",
    shape_rule=ShapeRule(kind="custom", fn=lambda shapes,attrs:[]),
    template_hint=TemplateHint(template_name="cross_entropy"),
    reference_impl=ReferenceImpl(fn=ref_fused_linear_cross_entropy),
    input_gen=InputGen(distributions={"X":"normal","W":"normal","labels":"randint"},
                       ranges={"labels":(0,1)}, constraints=["labels values < W.shape[0]"]),
))
QUANTIZE_PER_TOKEN = _register(OpSchema(name="quantize_per_token", category="reduce",
    inputs={"X":"Tensor[M,N]"}, output="Tuple[Tensor[M,N](int8),Tensor[M](f32)]",
    computation="scale[i]=max(abs(X[i,:]))/127;Q[i,j]=round(X[i,j]/scale[i])",
    index_vars=["i","j"], reduction_axes=["j"], properties=["quantization"],
    numpy_ref="scale=X.abs().max(-1).values/127;(X/scale[...,None]).round().to(int8)",
    shape_rule=ShapeRule(kind="custom", fn=lambda shapes,attrs: shapes["X"]),
    template_hint=TemplateHint(template_name="quantize"),
    reference_impl=ReferenceImpl(fn=ref_quantize_per_token),
    input_gen=InputGen(distributions={"X":"normal"}),
))
DEQUANTIZE_PER_CHANNEL = _register(OpSchema(name="dequantize_per_channel", category="elementwise",
    inputs={"X_int8":"Tensor[M,N](int8)","scale":"Tensor[N](f32)","zero_point":"Tensor[N](int8)"},
    output="Tensor[M,N]",
    computation="Y[i,j]=(X_int8[i,j]-zero_point[j])*scale[j]",
    properties=["elementwise","dequantization"], can_fuse_as="prologue",
    numpy_ref="(X_int8.float()-zero_point)*scale",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X_int8"),
    template_hint=TemplateHint(template_name="quantize"),
    reference_impl=ReferenceImpl(fn=ref_dequantize_per_channel),
    input_gen=InputGen(distributions={"X_int8":"randint","scale":"uniform","zero_point":"randint"},
                       ranges={"X_int8":(-127,127),"scale":(0.01,1.0),"zero_point":(-10,10)}),
))

# OT4: Attention
FLASH_ATTENTION = _register(OpSchema(name="flash_attention", category="attention",
    inputs={"Q":"Tensor[B,H,S,D]","K":"Tensor[B,H,S,D]","V":"Tensor[B,H,S,D]"},
    output="Tensor[B,H,S,D]",
    computation="O=softmax(Q@K^T/sqrt(D))@V (tiled, online softmax)",
    index_vars=["b","h","i","j","k"], reduction_axes=["j","k"],
    properties=["causal_mask_optional","online_softmax"],
    numpy_ref="softmax(Q@K.T/sqrt(D))@V",
    shape_rule=ShapeRule(kind="attention_rule"),
    template_hint=TemplateHint(template_name="flash_attention"),
    reference_impl=ReferenceImpl(fn=ref_flash_attention),
    input_gen=InputGen(distributions={"Q":"normal","K":"normal","V":"normal"},
                       constraints=["Q.shape==K.shape==V.shape"]),
))
GROUPED_QUERY_ATTENTION = _register(OpSchema(name="grouped_query_attention", category="attention",
    inputs={"Q":"Tensor[B,H_q,S,D]","K":"Tensor[B,H_kv,S,D]","V":"Tensor[B,H_kv,S,D]"},
    output="Tensor[B,H_q,S,D]",
    computation="GQA: Q heads grouped over fewer KV heads; O=softmax(Q@K^T/sqrt(D))@V",
    index_vars=["b","h_q","h_kv","i","j","k"], reduction_axes=["j","k"],
    properties=["causal_mask_optional","online_softmax","kv_head_repeat"],
    numpy_ref="GQA with head repeat",
    shape_rule=ShapeRule(kind="attention_rule"),
    template_hint=TemplateHint(template_name="flash_attention", extra_ctx={"op_variant":"gqa"}),
    reference_impl=ReferenceImpl(fn=ref_grouped_query_attention),
    input_gen=InputGen(distributions={"Q":"normal","K":"normal","V":"normal"},
                       constraints=["Q.shape[1] % K.shape[1] == 0"]),
))
MULTI_LATENT_ATTENTION = _register(OpSchema(name="multi_latent_attention", category="attention",
    inputs={"Q":"Tensor[B,H,S,D]","KV_compressed":"Tensor[B,S,D_c]",
            "W_uk":"Tensor[D_c,H,D]","W_uv":"Tensor[D_c,H,D]"},
    output="Tensor[B,H,S,D]",
    computation="MLA: decompress KV from low-rank latent, then standard attention",
    index_vars=["b","h","i","j","k"], reduction_axes=["j","k"],
    properties=["causal_mask_optional","online_softmax","latent_decompress"],
    numpy_ref="MLA: K=KV_c@W_uk, V=KV_c@W_uv, then softmax(Q@K^T/sqrt(D))@V",
    shape_rule=ShapeRule(kind="attention_rule"),
    template_hint=TemplateHint(template_name="mla"),
    reference_impl=ReferenceImpl(fn=ref_multi_latent_attention),
    input_gen=InputGen(distributions={"Q":"normal","KV_compressed":"normal",
                                      "W_uk":"normal","W_uv":"normal"}),
))
CROSS_ATTENTION = _register(OpSchema(name="cross_attention", category="attention",
    inputs={"Q":"Tensor[B,H,Sq,D]","K":"Tensor[B,H,Skv,D]","V":"Tensor[B,H,Skv,D]"},
    output="Tensor[B,H,Sq,D]",
    computation="O=softmax(Q@K^T/sqrt(D))@V (no causal mask)",
    index_vars=["b","h","i","j","k"], reduction_axes=["j","k"],
    properties=["online_softmax"],
    numpy_ref="softmax(Q@K.T/sqrt(D))@V",
    shape_rule=ShapeRule(kind="attention_rule"),
    template_hint=TemplateHint(template_name="flash_attention", extra_ctx={"op_variant":"cross"}),
    reference_impl=ReferenceImpl(fn=ref_cross_attention),
    input_gen=InputGen(distributions={"Q":"normal","K":"normal","V":"normal"}),
))
PAGED_ATTENTION = _register(OpSchema(name="paged_attention", category="attention",
    inputs={"Q":"Tensor[B,H,1,D]","K_cache":"Tensor[num_blocks,block_size,H,D]",
            "V_cache":"Tensor[num_blocks,block_size,H,D]","block_table":"Tensor[B,max_blocks]"},
    output="Tensor[B,H,1,D]",
    computation="Paged KV-cache attention with block_table indirection",
    index_vars=["b","h","i","j"], reduction_axes=["j"],
    properties=["decode_only","paged_kv"],
    numpy_ref="paged attention with block indirection",
    shape_rule=ShapeRule(kind="same_as_input", input_key="Q"),
    template_hint=TemplateHint(template_name="paged_attention"),
    reference_impl=ReferenceImpl(fn=ref_paged_attention),
    input_gen=InputGen(distributions={"Q":"normal","K_cache":"normal","V_cache":"normal",
                                      "block_table":"randint"},
                       ranges={"block_table":(0,1)},
                       constraints=["block_table values < K_cache.shape[0]"]),
))

# ============================================================
# Lookup utilities
# ============================================================

def get_op(name: str) -> OpSchema:
    """Get operator definition by name. Raises KeyError if not found."""
    return OP_CATALOG[name]


def list_ops(category: str | None = None) -> list[OpSchema]:
    """List all operators, optionally filtered by category."""
    ops = list(OP_CATALOG.values())
    if category:
        ops = [op for op in ops if op.category == category]
    return ops


def is_fusable_epilogue(name: str) -> bool:
    """Check if an operator can be fused as epilogue."""
    op = OP_CATALOG.get(name)
    return op is not None and op.can_fuse_as == "epilogue"
