#!/usr/bin/env python3
"""Append the remaining ops to catalog.py"""

TAIL = """

EMBEDDING = _register(OpSchema(
    name="embedding",
    category="move",
    inputs={"indices": "Tensor[B,S]", "weight": "Tensor[V,D]"},
    output="Tensor[B,S,D]",
    computation="Y[b,s,:] = weight[indices[b,s], :]",
    index_vars=["b", "s", "d"],
    properties=[],
    can_fuse_as=None,
    numpy_ref="weight[indices]",
    shape_rule=ShapeRule(kind="embedding_rule", input_key="indices"),
    template_hint=TemplateHint(template_name="data_movement"),
    reference_impl=ReferenceImpl(fn=ref_embedding),
    input_gen=InputGen(
        distributions={"indices": "randint", "weight": "normal"},
        ranges={"indices": (0, 32)},
    ),
))

PERMUTE = _register(OpSchema(
    name="permute",
    category="move",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = permute(X, dims)",
    properties=[],
    can_fuse_as=None,
    numpy_ref="np.transpose(X, axes=dims)",
    shape_rule=ShapeRule(kind="permute_rule", input_key="X", dims_attr="dims"),
    template_hint=TemplateHint(template_name="data_movement"),
    reference_impl=ReferenceImpl(fn=ref_permute),
    input_gen=InputGen(distributions={"X": "normal"}),
))

COPY = _register(OpSchema(
    name="copy_",
    category="move",
    inputs={"X": "Tensor[...]"},
    output="Tensor[...]",
    computation="Y = X.clone()",
    properties=["elementwise"],
    can_fuse_as=None,
    numpy_ref="X.copy()",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="data_movement"),
    reference_impl=ReferenceImpl(fn=ref_copy),
    input_gen=InputGen(distributions={"X": "normal"}),
))

ROPE = _register(OpSchema(
    name="rope",
    category="elementwise",
    inputs={"X": "Tensor[B,H,S,D]", "cos": "Tensor[S,D/2]", "sin": "Tensor[S,D/2]"},
    output="Tensor[B,H,S,D]",
    computation="Y = X * cos + rotate_half(X) * sin",
    properties=["elementwise", "position_encoding"],
    can_fuse_as="epilogue",
    numpy_ref="x * cos + rotate_half(x) * sin",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X"),
    template_hint=TemplateHint(template_name="rope"),
    reference_impl=ReferenceImpl(fn=ref_rope),
    input_gen=InputGen(
        distributions={"X": "normal", "cos": "uniform", "sin": "uniform"},
        ranges={"cos": (-1.0, 1.0), "sin": (-1.0, 1.0)},
    ),
))

CROSS_ENTROPY = _register(OpSchema(
    name="cross_entropy",
    category="reduce",
    inputs={"logits": "Tensor[B,V]", "labels": "Tensor[B]"},
    output="Tensor[]",
    computation="loss = -mean(log_softmax(logits)[i, labels[i]])",
    index_vars=["i", "j"],
    reduction_axes=["i", "j"],
    properties=["loss_function"],
    can_fuse_as=None,
    numpy_ref="-np.mean(np.log(softmax(logits))[np.arange(B), labels])",
    shape_rule=ShapeRule(kind="custom", input_key="logits"),
    template_hint=TemplateHint(template_name="cross_entropy"),
    reference_impl=ReferenceImpl(fn=ref_cross_entropy),
    input_gen=InputGen(
        distributions={"logits": "normal", "labels": "randint"},
        ranges={"labels": (0, 64)},
    ),
))

FUSED_LINEAR_CROSS_ENTROPY = _register(OpSchema(
    name="fused_linear_cross_entropy",
    category="compute",
    inputs={"X": "Tensor[B,D]", "W": "Tensor[V,D]", "labels": "Tensor[B]"},
    output="Tensor[]",
    computation="loss = cross_entropy(X @ W^T, labels)",
    index_vars=["i", "j", "k"],
    reduction_axes=["i", "j", "k"],
    properties=["fused", "loss_function"],
    can_fuse_as=None,
    numpy_ref="cross_entropy(X @ W.T, labels)",
    shape_rule=ShapeRule(kind="custom", input_key="X"),
    template_hint=TemplateHint(template_name="cross_entropy"),
    reference_impl=ReferenceImpl(fn=ref_fused_linear_cross_entropy),
    input_gen=InputGen(
        distributions={"X": "normal", "W": "normal", "labels": "randint"},
        ranges={"labels": (0, 64)},
    ),
))

QUANTIZE_PER_TOKEN = _register(OpSchema(
    name="quantize_per_token",
    category="reduce",
    inputs={"X": "Tensor[M,N]"},
    output="Tuple[Tensor[M,N](int8), Tensor[M](f32)]",
    computation="scale[i] = max(abs(X[i,:]))/127; Q[i,j] = round(X[i,j]/scale[i])",
    index_vars=["i", "j"],
    reduction_axes=["j"],
    properties=["quantization"],
    can_fuse_as=None,
    numpy_ref="scale = X.abs().max(-1).values/127; (X/scale[...,None]).round().to(int8)",
    shape_rule=ShapeRule(kind="custom", input_key="X"),
    template_hint=TemplateHint(template_name="quantize"),
    reference_impl=ReferenceImpl(fn=ref_quantize_per_token),
    input_gen=InputGen(distributions={"X": "normal"}),
))

DEQUANTIZE_PER_CHANNEL = _register(OpSchema(
    name="dequantize_per_channel",
    category="elementwise",
    inputs={
        "X_int8": "Tensor[M,N](int8)",
        "scale": "Tensor[N](f32)",
        "zero_point": "Tensor[N](int8)",
    },
    output="Tensor[M,N]",
    computation="Y[i,j] = (X_int8[i,j] - zero_point[j]) * scale[j]",
    properties=["elementwise", "dequantization"],
    can_fuse_as="prologue",
    numpy_ref="(X_int8.float() - zero_point) * scale",
    shape_rule=ShapeRule(kind="same_as_input", input_key="X_int8"),
    template_hint=TemplateHint(template_name="quantize"),
    reference_impl=ReferenceImpl(fn=ref_dequantize_per_channel),
    input_gen=InputGen(
        distributions={"X_int8": "randint", "scale": "uniform", "zero_point": "randint"},
        ranges={"X_int8": (-127, 127), "scale": (0.001, 1.0), "zero_point": (-10, 10)},
    ),
))

FLASH_ATTENTION = _register(OpSchema(
    name="flash_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H,S,D]", "K": "Tensor[B,H,S,D]", "V": "Tensor[B,H,S,D]"},
    output="Tensor[B,H,S,D]",
    computation="O = softmax(Q @ K^T / sqrt(D)) @ V  (tiled, online softmax)",
    index_vars=["b", "h", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["causal_mask_optional", "online_softmax"],
    can_fuse_as=None,
    numpy_ref="softmax(Q @ K.T / sqrt(D)) @ V",
    shape_rule=ShapeRule(kind="attention_rule", input_key="Q"),
    template_hint=TemplateHint(template_name="flash_attention"),
    reference_impl=ReferenceImpl(fn=ref_flash_attention),
    input_gen=InputGen(distributions={"Q": "normal", "K": "normal", "V": "normal"}),
))

GROUPED_QUERY_ATTENTION = _register(OpSchema(
    name="grouped_query_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H_q,S,D]", "K": "Tensor[B,H_kv,S,D]", "V": "Tensor[B,H_kv,S,D]"},
    output="Tensor[B,H_q,S,D]",
    computation="GQA: Q heads grouped over fewer KV heads",
    index_vars=["b", "h_q", "h_kv", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["causal_mask_optional", "online_softmax", "kv_head_repeat"],
    can_fuse_as=None,
    numpy_ref="GQA with head repeat",
    shape_rule=ShapeRule(kind="attention_rule", input_key="Q"),
    template_hint=TemplateHint(template_name="flash_attention"),
    reference_impl=ReferenceImpl(fn=ref_grouped_query_attention),
    input_gen=InputGen(
        distributions={"Q": "normal", "K": "normal", "V": "normal"},
        constraints=["Q.shape[1] % K.shape[1] == 0"],
    ),
))

MULTI_LATENT_ATTENTION = _register(OpSchema(
    name="multi_latent_attention",
    category="attention",
    inputs={
        "Q": "Tensor[B,H,S,D]",
        "KV_compressed": "Tensor[B,S,D_c]",
        "W_uk": "Tensor[D_c,H,D]",
        "W_uv": "Tensor[D_c,H,D]",
    },
    output="Tensor[B,H,S,D]",
    computation="MLA: decompress KV from low-rank latent, then standard attention",
    index_vars=["b", "h", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["causal_mask_optional", "online_softmax", "latent_decompress"],
    can_fuse_as=None,
    numpy_ref="MLA: K=KV_c@W_uk, V=KV_c@W_uv, then sdpa",
    shape_rule=ShapeRule(kind="attention_rule", input_key="Q"),
    template_hint=TemplateHint(template_name="mla"),
    reference_impl=ReferenceImpl(fn=ref_multi_latent_attention),
    input_gen=InputGen(
        distributions={
            "Q": "normal",
            "KV_compressed": "normal",
            "W_uk": "normal",
            "W_uv": "normal",
        },
    ),
))

CROSS_ATTENTION = _register(OpSchema(
    name="cross_attention",
    category="attention",
    inputs={"Q": "Tensor[B,H,Sq,D]", "K": "Tensor[B,H,Skv,D]", "V": "Tensor[B,H,Skv,D]"},
    output="Tensor[B,H,Sq,D]",
    computation="O = softmax(Q @ K^T / sqrt(D)) @ V  (no causal mask)",
    index_vars=["b", "h", "i", "j", "k"],
    reduction_axes=["j", "k"],
    properties=["online_softmax"],
    can_fuse_as=None,
    numpy_ref="softmax(Q @ K.T / sqrt(D)) @ V",
    shape_rule=ShapeRule(kind="attention_rule", input_key="Q"),
    template_hint=TemplateHint(template_name="flash_attention"),
    reference_impl=ReferenceImpl(fn=ref_cross_attention),
    input_gen=InputGen(distributions={"Q": "normal", "K": "normal", "V": "normal"}),
))

PAGED_ATTENTION = _register(OpSchema(
    name="paged_attention",
    category="attention",
    inputs={
        "Q": "Tensor[B,H,1,D]",
        "K_cache": "Tensor[num_blocks,block_size,H,D]",
        "V_cache": "Tensor[num_blocks,block_size,H,D]",
        "block_table": "Tensor[B,max_blocks]",
    },
    output="Tensor[B,H,1,D]",
    computation="Paged KV-cache attention with block_table indirection",
    index_vars=["b", "h", "i", "j"],
    reduction_axes=["j"],
    properties=["decode_only", "paged_kv"],
    can_fuse_as=None,
    numpy_ref="paged attention with block indirection",
    shape_rule=ShapeRule(kind="same_as_input", input_key="Q"),
    template_hint=TemplateHint(template_name="paged_attention"),
    reference_impl=ReferenceImpl(fn=ref_paged_attention),
    input_gen=InputGen(
        distributions={
            "Q": "normal",
            "K_cache": "normal",
            "V_cache": "normal",
            "block_table": "randint",
        },
        ranges={"block_table": (0, 4)},
    ),
))


# ============================================================
# Lookup utilities (backward compat)
# ============================================================

def get_op(name: str) -> "OpSchema":
    return OP_CATALOG[name]


def list_ops(category=None):
    ops = list(OP_CATALOG.values())
    if category:
        ops = [op for op in ops if op.category == category]
    return ops


def is_fusable_epilogue(name: str) -> bool:
    op = OP_CATALOG.get(name)
    return op is not None and op.can_fuse_as == "epilogue"
"""

with open("/home/blueyi/workspace/repos/arke/arke/ir/ops/catalog.py", "a") as f:
    f.write(TAIL)

print(f"Appended {len(TAIL)} chars.")
