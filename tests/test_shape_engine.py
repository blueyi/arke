# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ShapeInferenceEngine (S6 Track 1, Task C1.3).

Validates shape inference for all rule kinds with representative ops.
"""

import pytest

from arke.ir.ops.shape_engine import SHAPE_ENGINE


class TestShapeInference:
    """Shape inference engine tests."""

    # same_as_input
    def test_relu_shape(self):
        assert SHAPE_ENGINE.infer("relu", {"X": [4, 8]}) == [4, 8]

    def test_relu_3d(self):
        assert SHAPE_ENGINE.infer("relu", {"X": [2, 4, 8]}) == [2, 4, 8]

    def test_add_shape(self):
        assert SHAPE_ENGINE.infer("add", {"A": [4, 8], "B": [4, 8]}) == [4, 8]

    def test_layernorm_shape(self):
        assert SHAPE_ENGINE.infer("layernorm", {"X": [32, 768], "W": [768], "B": [768]}) == [32, 768]

    def test_softmax_shape(self):
        assert SHAPE_ENGINE.infer("softmax", {"X": [4, 1024]}) == [4, 1024]

    def test_copy_shape(self):
        assert SHAPE_ENGINE.infer("copy_", {"X": [10, 20]}) == [10, 20]

    # matmul_rule
    def test_matmul_shape(self):
        assert SHAPE_ENGINE.infer("matmul", {"A": [1024, 512], "B": [512, 2048]}) == [1024, 2048]

    def test_matmul_3d(self):
        assert SHAPE_ENGINE.infer("matmul", {"A": [4, 1024, 512], "B": [4, 512, 2048]}) == [4, 1024, 2048]

    # batch_matmul_rule
    def test_batch_matmul_shape(self):
        assert SHAPE_ENGINE.infer("batch_matmul", {"A": [8, 64, 32], "B": [8, 32, 128]}) == [8, 64, 128]

    def test_grouped_matmul_shape(self):
        assert SHAPE_ENGINE.infer("grouped_matmul", {"X": [4, 64, 128], "W": [8, 128, 256]}) == [4, 64, 256]

    # reduce_rule
    def test_reduce_sum_shape(self):
        assert SHAPE_ENGINE.infer("reduce_sum", {"X": [4, 8]}) == [4]

    def test_reduce_max_shape(self):
        assert SHAPE_ENGINE.infer("reduce_max", {"X": [16, 32]}) == [16]

    def test_argmax_shape(self):
        assert SHAPE_ENGINE.infer("argmax", {"X": [4, 1024]}) == [4]

    # topk_rule
    def test_topk_shape(self):
        assert SHAPE_ENGINE.infer("topk", {"X": [4, 100]}, {"k": 5}) == [4, 5]

    def test_topk_default_k(self):
        assert SHAPE_ENGINE.infer("topk", {"X": [4, 100]}) == [4, 1]  # default k=1

    # concat_rule
    def test_concat_shape(self):
        assert SHAPE_ENGINE.infer("concat", {"A": [4, 8], "B": [4, 12]}) == [4, 20]

    # split_rule
    def test_split_shape(self):
        assert SHAPE_ENGINE.infer("split", {"X": [4, 16]}) == [4, 8]

    # gather_rule
    def test_gather_shape(self):
        assert SHAPE_ENGINE.infer("gather", {"X": [4, 100], "idx": [4, 10]}) == [4, 10]

    # embedding_rule
    def test_embedding_shape(self):
        assert SHAPE_ENGINE.infer("embedding", {"indices": [8, 128], "weight": [50000, 768]}) == [8, 128, 768]

    # permute_rule
    def test_permute_shape(self):
        assert SHAPE_ENGINE.infer("permute", {"X": [2, 4, 8, 16]}, {"dims": [0, 2, 1, 3]}) == [2, 8, 4, 16]

    # gated_halve_rule
    def test_silu_and_mul_shape(self):
        assert SHAPE_ENGINE.infer("silu_and_mul", {"X": [4, 256]}) == [4, 128]

    def test_geglu_shape(self):
        assert SHAPE_ENGINE.infer("geglu", {"X": [4, 512]}) == [4, 256]

    # attention_rule
    def test_flash_attention_shape(self):
        assert SHAPE_ENGINE.infer("flash_attention", {"Q": [2, 8, 128, 64], "K": [2, 8, 128, 64], "V": [2, 8, 128, 64]}) == [2, 8, 128, 64]

    def test_gqa_shape(self):
        assert SHAPE_ENGINE.infer("grouped_query_attention", {"Q": [2, 32, 128, 64], "K": [2, 4, 128, 64], "V": [2, 4, 128, 64]}) == [2, 32, 128, 64]

    # custom
    def test_transpose_shape(self):
        assert SHAPE_ENGINE.infer("transpose", {"X": [4, 8]}) == [8, 4]

    def test_transpose_3d(self):
        assert SHAPE_ENGINE.infer("transpose", {"X": [2, 4, 8]}) == [2, 8, 4]

    def test_cross_entropy_shape(self):
        assert SHAPE_ENGINE.infer("cross_entropy", {"logits": [32, 50000], "labels": [32]}) == []

    # Error cases
    def test_unknown_op(self):
        with pytest.raises(KeyError):
            SHAPE_ENGINE.infer("nonexistent_op", {"X": [4, 8]})

    def test_missing_input(self):
        with pytest.raises(ValueError, match="not found"):
            SHAPE_ENGINE.infer("relu", {"Y": [4, 8]})  # wrong input name
